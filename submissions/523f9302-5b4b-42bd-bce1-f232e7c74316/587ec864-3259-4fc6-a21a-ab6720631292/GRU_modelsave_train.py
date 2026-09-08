# -*- coding: utf-8 -*-
import os
import gc
import json
import time
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader, random_split
import structlog

logger = structlog.get_logger()

MODEL_PATH = "stock_model.json"

# ---------- 配置 ----------
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"
SEQ_LEN = 64
EPOCHS = 80
BATCH = 256
LR = 1e-3
GRAD_ACCUM_STEPS = 2
MAX_TRAIN_INSTRUMENTS = 200
DAI_CHUNK_SIZE = 20  # 每次 dai.query 最多查 20 只股票, 防止内存溢出
SEED = 42

# 验证集: 训练集最后 6 个月
VAL_SPLIT_DATE = "2023-07-01"

# 多任务预测 horizon (分钟)
HORIZONS = [5, 15, 30, 60]
N_TASKS = len(HORIZONS)
# 多任务权重: 30min 主任务权重最高, 5/60 仅提供微弱信号
TASK_WEIGHTS = [0.1, 0.3, 0.5, 0.1]

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS = ["volume", "amount", "bid_volume1", "ask_volume1"]
BASE_FEATURE_COLS = PRICE_COLS + VOL_COLS
N_BASE_FEAT = len(BASE_FEATURE_COLS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# 滚动标准化
# ======================================================================

def rolling_normalize(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    cumsum = np.cumsum(arr, axis=0)
    cumsum2 = np.cumsum(arr ** 2, axis=0)
    n = np.arange(1, len(arr) + 1, dtype=np.float64).reshape(-1, 1)
    m = cumsum / n
    s = np.sqrt(np.maximum(cumsum2 / n - m ** 2, 1e-6))
    out = (arr - m) / s
    return out


# ======================================================================
# 增强技术指标 (原始 5 个 + 新增 9 个 = 14 个)
# ======================================================================

def compute_rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(period) / period, mode="valid")
    avg_loss = np.convolve(loss, np.ones(period) / period, mode="valid")
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    pad = np.full(period, 50.0, dtype=np.float32)
    return np.concatenate([pad, rsi])


def add_technical_features(sub: pd.DataFrame) -> pd.DataFrame:
    close = sub["close"].to_numpy(np.float64)
    volume = sub["volume"].to_numpy(np.float64)
    high = sub["high"].to_numpy(np.float64)
    low = sub["low"].to_numpy(np.float64)
    ask = sub["ask_price1"].to_numpy(np.float64)
    bid = sub["bid_price1"].to_numpy(np.float64)
    amount = sub["amount"].to_numpy(np.float64)
    vwap = amount / np.maximum(volume, 1e-10)

    # ---- 原始 5 个技术指标 ----
    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().to_numpy()
    sub["norm_close"] = (close / np.maximum(sma20, 1e-10)).astype(np.float32)
    sub["rsi_14"] = compute_rsi(close, 14).astype(np.float32)

    vol_sma20 = pd.Series(volume).rolling(20, min_periods=1).mean().to_numpy()
    sub["vol_ratio"] = (volume / np.maximum(vol_sma20, 1e-10)).astype(np.float32)
    sub["range_ratio"] = ((high - low) / np.maximum(close, 1e-10)).astype(np.float32)
    spread = (ask - bid) / np.maximum(close, 1e-10)
    sub["spread"] = np.clip(spread, 0.0, 0.1).astype(np.float32)

    # ---- 新增技术指标 ----
    # (1) MACD
    ema12 = pd.Series(close).ewm(span=12).mean().to_numpy()
    ema26 = pd.Series(close).ewm(span=26).mean().to_numpy()
    sub["macd"] = (ema12 - ema26).astype(np.float32)
    # (2) MACD Signal
    sub["macd_signal"] = pd.Series(sub["macd"]).ewm(span=9).mean().to_numpy()

    # (3) 布林带位置
    ma20 = pd.Series(close).rolling(20).mean().to_numpy()
    std20 = pd.Series(close).rolling(20).std(ddof=0).to_numpy()
    sub["bb_position"] = ((close - ma20) / np.maximum(2.0 * std20, 1e-10)).clip(-2, 2).astype(np.float32)

    # (4) VWAP 偏离度
    vwap_ma20 = pd.Series(vwap).rolling(20).mean().to_numpy()
    sub["vwap_dev"] = ((vwap - vwap_ma20) / np.maximum(vwap_ma20, 1e-10)).clip(-0.1, 0.1).astype(np.float32)

    # (5) 日内价格位置 (20 日滚动)
    day_high = pd.Series(high).rolling(20).max().to_numpy()
    day_low = pd.Series(low).rolling(20).min().to_numpy()
    sub["price_position"] = ((close - day_low) / np.maximum(day_high - day_low, 1e-10)).astype(np.float32)

    # (6)-(9) 历史收益: 前 1/3/5/10 根 K 线收益率 (动量信号)
    for period, label in [(1, "ret_1"), (3, "ret_3"), (5, "ret_5"), (10, "ret_10")]:
        ret = np.full_like(close, 0.0, dtype=np.float32)
        if len(close) > period:
            ret[period:] = (close[period:] / close[:-period] - 1.0).astype(np.float32)
        sub[label] = ret

    return sub


TECH_FEATURE_COLS = [
    "norm_close", "rsi_14", "vol_ratio", "range_ratio", "spread",       # 原始 5
    "macd", "macd_signal", "bb_position", "vwap_dev", "price_position", # 新增 5
    "ret_1", "ret_3", "ret_5", "ret_10",                                 # 新增 4
]
FEATURE_COLS = BASE_FEATURE_COLS + TECH_FEATURE_COLS   # 10 + 14 = 24
N_FEAT = len(FEATURE_COLS)

# 横截面 Rank 特征: 对上述 24 个特征做截面排名, 拼接到原始特征后
N_FEAT_TOTAL = N_FEAT * 2   # 48


# ======================================================================
# 未来 VWAP 收益 (多 horizon)
# ======================================================================

def compute_vwap_future_returns(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, horizons: list = None,
) -> np.ndarray:
    """同时计算多个 horizon 的未来 VWAP 收益。返回 (n, len(horizons))。"""
    if horizons is None:
        horizons = HORIZONS
    typical = (high + low + close) / 3.0
    pv = typical * volume
    n = len(close)
    rets = np.full((n, len(horizons)), np.nan, dtype=np.float64)
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    for j, horizon in enumerate(horizons):
        for i in range(n - horizon):
            total_pv = cum_pv[i + horizon] - cum_pv[i]
            total_vol = cum_vol[i + horizon] - cum_vol[i]
            if total_vol > 1e-10 and close[i] > 1e-10:
                future_vwap = total_pv / total_vol
                rets[i, j] = future_vwap / close[i] - 1.0
    return rets


# ======================================================================
# 混合损失: RankIC Loss + Huber
# ======================================================================

class MixedLoss(nn.Module):
    """RankIC Loss (直接优化排序) + Huber Loss (数值稳定) 混合。

    RankIC 使模型关注跨截面的排序正确性, Huber 防止训练初期发散。
    alpha 控制混合比例: alpha=0 → 纯 Huber, alpha=1 → 纯 RankIC。
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = pred.size(0)
        if n < 2:
            return self.huber(pred, target)

        # --- RankIC 部分: 最大化 batch 内 Spearman 秩相关 ---
        pred_rank = pred.argsort().float() / n
        tgt_rank = target.argsort().float() / n
        pred_centered = pred_rank - pred_rank.mean()
        tgt_centered = tgt_rank - tgt_rank.mean()
        pred_norm = pred_centered / (pred_centered.norm() + 1e-8)
        tgt_norm = tgt_centered / (tgt_centered.norm() + 1e-8)
        rankic = (pred_norm * tgt_norm).sum()
        rankic_loss = 1.0 - rankic

        # --- Huber 部分 ---
        huber_loss = self.huber(pred, target)

        return self.alpha * rankic_loss + (1.0 - self.alpha) * huber_loss


# ======================================================================
# 日期分组 Batch 工具 — 确保同日期股票在同一 batch
# ======================================================================

class DateGroupDataset(torch.utils.data.Dataset):
    """在 X, y 外额外返回 date_groups 的 Dataset。"""
    def __init__(self, X: np.ndarray, y: np.ndarray, date_groups: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.date_groups = torch.from_numpy(date_groups)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.date_groups[idx]


class DateGroupBatchSampler(torch.utils.data.Sampler):
    """按日期组取 batch: 同一 batch 内只包含同一日期的股票, 不同日期不会混入。

    这样 CS-Attention 在同一 batch 内做横截面交互时, 同日期股票恰好在一个组里。
    训练时日期组之间的顺序随机打乱, 增加训练多样性。
    """
    def __init__(self, date_groups: np.ndarray, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = defaultdict(list)
        for idx, g in enumerate(date_groups):
            self.groups[g].append(idx)

    def __iter__(self):
        batches = []
        for g in sorted(self.groups.keys()):
            indices = list(self.groups[g])
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


# ======================================================================
# 模型
# ======================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = 10_000 ** (torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class MultiQueryAttentionPooling(nn.Module):
    """多查询注意力池化: 用多个可学习 query 从序列中提取不同维度的信息。"""
    def __init__(self, d_model: int, n_queries: int = 2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5
        self.out_proj = nn.Linear(d_model * n_queries, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.proj(self.queries).expand(h.size(0), -1, -1)
        attn = torch.bmm(q, h.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        pooled = torch.bmm(attn, h)  # (B, n_q, d_model)
        return self.out_proj(pooled.reshape(h.size(0), -1))


class CrossSectionalAttention(nn.Module):
    """横截面注意力: 同一日期组内的股票之间做交互注意力。

    输入: x=(B, d_model), groups=(B,) 整数组 ID (已排序且连续)
    输出: (B, d_model) — 每组内股票互相看到对方隐藏状态后重新校准的表示

    按日期分组循环, 每组内 stocks 互做 MultiheadAttention (key=value=query)。
    对同组仅 1 只股票和 <2 只股票的组直接跳过 (信息量不足)。
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        # x: (B, d), groups: (B,) — 已按组排序且连续
        attn_outputs = []
        group_ids = groups.unique_consecutive()
        for g in group_ids:
            mask = groups == g
            n = mask.sum().item()
            if n < 2:
                # 单只股票或空组: 不需要做 CS-Attention
                attn_outputs.append(x[mask])
                continue
            g_x = x[mask].unsqueeze(0)  # (1, n, d)
            g_out = self.attn(g_x, g_x, g_x, need_weights=False)[0].squeeze(0)
            attn_outputs.append(g_out)

        attn_out = torch.cat(attn_outputs, dim=0)
        return self.norm(x + attn_out)  # 残差连接


class StockGRU(nn.Module):
    """GRU 时序编码 + CS-Attention 横截面交互 + 多任务预测。"""
    def __init__(
        self,
        n_feat: int,
        d_model: int = 128,
        dropout: float = 0.2,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.proj = nn.Sequential(
            nn.LayerNorm(n_feat),
            nn.Linear(n_feat, d_model),
        )

        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len)

        self.gru = nn.GRU(
            d_model, d_model, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=False,
        )

        self.pool = MultiQueryAttentionPooling(d_model, n_queries=2)

        self.cs_attn = CrossSectionalAttention(d_model, nhead=4, dropout=dropout)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            for _ in range(N_TASKS)
        ])

    def forward(self, x: torch.Tensor, date_groups: Optional[torch.Tensor] = None,
                task: Optional[int] = None) -> torch.Tensor:
        h = self.proj(x) + self.pos_encoder(x)
        h, _ = self.gru(h)
        h = self.pool(h)
        if date_groups is not None:
            h = self.cs_attn(h, date_groups)
        if task is not None:
            return self.heads[task](h).squeeze(-1)
        return torch.cat([head(h) for head in self.heads], dim=-1)


MODEL_CFG = dict(
    n_feat=N_FEAT_TOTAL, d_model=128, dropout=0.2, seq_len=SEQ_LEN,
)


# ======================================================================
# 数据构建 (含横截面 Rank)
# ======================================================================

def build_dataset(
    table: str, sd: str, ed: str, mode: str,
    instruments: List[str],
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    # note: mode='train' 额外返回第5项 date_groups (np.ndarray); mode='infer' 仅4项
    """构建带横截面 Rank 特征的数据集。

    两阶段流程:
      Phase 1: 对每只股票单独处理, 计算原始特征 + 标签, 按日期暂存。
      Phase 2: 按日期聚合, 对每个特征的最后值做横截面 Rank,
               Rank 值重复 SEQ_LEN 次拼接到特征矩阵后。
    训练模式下标签也做截面 Rank 化 (混合原始收益率保持稳定)。
    """
    t0 = time.time()
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    sql_tpl = (
        f"SELECT date, instrument, {', '.join(BASE_FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)

    # ---- Phase 1: 按股票分块读取, 防止 dai 内存溢出 ----
    all_samples = []
    n_stocks_used = 0

    for chunk_start in range(0, len(instruments), DAI_CHUNK_SIZE):
        chunk_inst = instruments[chunk_start:chunk_start + DAI_CHUNK_SIZE]
        df = dai.query(
            sql_tpl,
            filters={"date": [buf, ed], "instrument": chunk_inst},
        ).df()

        # 处理 NaN: 价格类填充, 成交量类填 0
        for c in PRICE_COLS:
            df[c] = df.groupby("instrument")[c].transform(lambda g: g.ffill().bfill())
        for c in VOL_COLS:
            df[c] = df.groupby("instrument")[c].transform(lambda g: g.ffill().bfill().fillna(0.0))
        df = df.dropna(subset=PRICE_COLS)
        for c in VOL_COLS:
            df[c] = np.log1p(df[c].clip(lower=0))

        for ins, sub in df.groupby("instrument", sort=False):
            if len(sub) <= SEQ_LEN + 30:
                continue
            n_stocks_used += 1
            sub = sub.sort_values("date").reset_index(drop=True)
            sub = add_technical_features(sub)

            close_arr = sub["close"].to_numpy(np.float64)
            high_arr = sub["high"].to_numpy(np.float64)
            low_arr = sub["low"].to_numpy(np.float64)
            vol_arr = sub["volume"].to_numpy(np.float64)
            future_rets = compute_vwap_future_returns(
                close_arr, high_arr, low_arr, vol_arr, horizons=HORIZONS,
            )

            feats = sub[FEATURE_COLS].to_numpy(np.float32)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feats = rolling_normalize(feats)

            day = sub["date"].dt.normalize().to_numpy()
            close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
            dates = day[close_pos]

            for k, p in enumerate(close_pos):
                d = pd.Timestamp(dates[k])
                if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                    continue
                label_y = future_rets[p] if mode == "train" else None
                if mode == "train" and (not np.all(np.isfinite(label_y))):
                    continue
                all_samples.append({
                    "date": d,
                    "instrument": ins,
                    "feat": feats[p - SEQ_LEN + 1: p + 1],
                    "label": label_y,
                })

    if not all_samples:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")

    # ---- Phase 2: 横截面 Rank ----
    by_date = defaultdict(list)
    for s in all_samples:
        by_date[s["date"]].append(s)

    wins, ys, keys, date_groups = [], [], [], []
    date_group_id = 0

    for date in sorted(by_date.keys()):
        samples = by_date[date]
        n = len(samples)

        # 提取每个样本最后一个 bar 的特征值
        last_vals = np.array([s["feat"][-1] for s in samples], dtype=np.float32)  # (n, 24)

        if mode == "train" and n >= 3:
            # 对每个特征做横截面排序分位数
            rank_order = np.argsort(np.argsort(last_vals, axis=0), axis=0)
            feat_ranks = rank_order.astype(np.float32) / (n - 1)  # [0, 1]

            # 标签 (n, 4): 对每个 horizon 分别做截面 Rank + 归一化
            labels_arr = np.array([s["label"] for s in samples], dtype=np.float32)
            label_order = np.argsort(np.argsort(labels_arr, axis=0), axis=0)
            label_ranks = label_order.astype(np.float32) / (n - 1)
            targets = np.zeros_like(labels_arr)
            for j in range(N_TASKS):
                lo, hi = np.percentile(labels_arr[:, j], [1, 99])
                clipped = np.clip(labels_arr[:, j], lo, hi)
                norm = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-10)
                targets[:, j] = 0.7 * label_ranks[:, j] + 0.3 * norm
        elif n >= 3:
            rank_order = np.argsort(np.argsort(last_vals, axis=0), axis=0)
            feat_ranks = rank_order.astype(np.float32) / max(n - 1, 1)
            targets = None
        else:
            # 当日股票太少 (<3), 无法做有意义的截面排名
            feat_ranks = np.zeros((n, N_FEAT), dtype=np.float32)
            targets = np.full(n, 0.5, dtype=np.float32) if mode == "train" else None

        # 组装最终样本: 原始特征 (64,24) + 横截面 Rank (64,24) = (64,48)
        for i, s in enumerate(samples):
            rank_feat = np.tile(feat_ranks[i:i+1], (SEQ_LEN, 1))  # (64, 24)
            feat_full = np.concatenate([s["feat"], rank_feat], axis=1)  # (64, 48)
            wins.append(feat_full)
            keys.append((s["date"], s["instrument"]))
            date_groups.append(date_group_id)
            if mode == "train":
                ys.append(targets[i] if targets is not None else s["label"])
        date_group_id += 1

    X = np.stack(wins).astype(np.float32)

    # ---- 全局标准化 (在横截面特征拼接后进行) ----
    if stats is None:
        flat = X.reshape(-1, N_FEAT_TOTAL)
        stats = (
            flat.mean(0).astype(np.float32),
            flat.std(0).astype(np.float32) + 1e-6,
        )
    m, s = stats
    X = ((X - m) / s).astype(np.float32)

    logger.info(
        f"{mode} 集构建完成",
        samples=len(keys),
        n_feat=N_FEAT_TOTAL,
        n_stocks=n_stocks_used,
        n_dates=len(by_date),
        elapsed=round(time.time() - t0, 2),
    )
    if mode == "train":
        return X, np.array(ys, np.float32), None, stats, np.array(date_groups, np.int32)
    return X, None, pd.DataFrame(keys, columns=["date", "instrument"]), stats


def pool(sd: str, ed: str) -> List[str]:
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return df["instrument"].tolist()


# ======================================================================
# 模型存/读: JSON 文本文件
# ======================================================================

def save_model(ckpt: dict, model_path: str = MODEL_PATH) -> str:
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),
        }
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info("模型已保存", path=model_path)
    return model_path


def load_model(model_path=MODEL_PATH, device="cpu"):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(device)
    cfg = payload["model_cfg"]
    model = StockGRU(**cfg).to(device)
    model.load_state_dict(sd)
    mean = torch.tensor(payload["mean"], dtype=torch.float32).to(device)
    std = torch.tensor(payload["std"], dtype=torch.float32).to(device)
    return model, mean, std


# ======================================================================
# 训练 + 验证 + 早停 + 持久化
# ======================================================================

def train_and_save(datasources: dict, model_path: str = MODEL_PATH) -> str:
    table = datasources["bar1m"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(dev), table=table)

    # ---- 构建全部训练数据 (含日期组) ----
    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    X_all, y_all, _, stats, date_groups_all = build_dataset(
        table, TRAIN_START, TRAIN_END, "train",
        pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS],
    )

    # ---- 按时间分割: 后 20% 做验证 ----
    n_val = max(1, len(y_all) // 5)
    n_train = len(y_all) - n_val

    # 训练模式下 y_all / date_groups_all 不可能是 None
    assert y_all is not None and date_groups_all is not None

    Xtr, Xval = X_all[:n_train], X_all[n_train:]
    ytr, yval = y_all[:n_train], y_all[n_train:]
    dgtr, dgval = date_groups_all[:n_train], date_groups_all[n_train:]

    logger.info(
        "数据集分割",
        train=len(ytr), val=len(yval),
        val_pct=round(100 * n_val / len(y_all), 1),
    )

    # ---- 构建模型 ----
    model = StockGRU(**MODEL_CFG).to(dev)
    logger.info(
        "模型参数量",
        total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ---- DataLoaders (同日期股票在同一 batch) ----
    train_dataset = DateGroupDataset(Xtr, ytr, dgtr)
    train_sampler = DateGroupBatchSampler(dgtr, BATCH, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=(dev.type == "cuda"),
    )

    val_dataset = DateGroupDataset(Xval, yval, dgval)
    val_sampler = DateGroupBatchSampler(dgval, BATCH * 2, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    # ---- 优化器 + 调度器 + 损失 ----
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # CosineAnnealingWarmRestarts: 每 15 epoch 一个完整周期, 与早停完美兼容
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=15, T_mult=1, eta_min=1e-6,
    )
    loss_fn = MixedLoss(alpha=0.7)

    def multi_task_loss(pred, target):
        """加权多任务损失: 30min 主任务占 0.5, 辅助任务权重低。"""
        total = 0.0
        for j in range(N_TASKS):
            total = total + TASK_WEIGHTS[j] * loss_fn(pred[:, j], target[:, j])
        return total

    # ---- NaN 检测 ----
    if np.any(np.isnan(Xtr)):
        logger.error("训练数据 Xtr 含 NaN!"); raise ValueError("Xtr has NaN")
    if np.any(np.isnan(ytr)):
        logger.error("训练数据 ytr 含 NaN!"); raise ValueError("ytr has NaN")
    if np.any(np.isnan(Xval)):
        logger.error("验证数据 Xval 含 NaN!"); raise ValueError("Xval has NaN")
    if np.any(np.isnan(yval)):
        logger.error("验证数据 yval 含 NaN!"); raise ValueError("yval has NaN")

    # ---- 训练循环 ----
    best_val_loss = float("inf")
    best_epoch = -1
    patience = 12
    stale = 0
    best_sd = None

    for ep in range(EPOCHS):
        # ----- 训练 -----
        model.train()
        t_start = time.time()
        train_loss = 0.0
        n_batches = 0

        for xb, yb, dg in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            dg = dg.to(dev, non_blocking=True)

            pred = model(xb, date_groups=dg)  # (B, 4)
            pred = torch.clamp(pred, -10.0, 10.0)
            loss = multi_task_loss(pred, yb) / GRAD_ACCUM_STEPS
            if torch.isnan(loss):
                logger.error("Loss 为 NaN!", pred=pred, yb=yb)
                raise RuntimeError("Loss is NaN — 请检查数据或降低 LR")
            loss.backward()

            n_batches += 1
            train_loss += loss.item() * GRAD_ACCUM_STEPS

            if n_batches % GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        train_loss /= max(n_batches, 1)

        # ----- 验证 -----
        model.eval()
        val_loss = 0.0
        n_vb = 0
        with torch.no_grad():
            for xb, yb, dg in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                dg = dg.to(dev)
                vpred = torch.clamp(model(xb, date_groups=dg), -10.0, 10.0)
                val_loss += multi_task_loss(vpred, yb).item()
                n_vb += 1
        val_loss /= max(n_vb, 1)
        scheduler.step()

        elapsed = time.time() - t_start
        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {ep+1:3d}/{EPOCHS}",
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            lr=f"{lr_now:.2e}",
            time=f"{elapsed:.1f}s",
        )

        # ----- 早停 -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep + 1
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info(
                    f"早停触发 (已 {patience} 轮未改善)",
                    best_epoch=best_epoch,
                    best_val_loss=f"{best_val_loss:.6f}",
                )
                break

    # ---- 恢复到最佳 checkpoint ----
    if best_sd is not None:
        model.load_state_dict(best_sd)
        logger.info("已恢复最佳权重", epoch=best_epoch)

    # ---- SWA 权重平均 ----
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_epochs = 3
    swa_lr = LR * 0.02
    for param_group in opt.param_groups:
        param_group["lr"] = swa_lr
    logger.info("开始 SWA 阶段", epochs=swa_epochs, lr=swa_lr)

    for ep in range(swa_epochs):
        model.train()
        for xb, yb, dg in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            dg = dg.to(dev, non_blocking=True)
            loss = multi_task_loss(model(xb, date_groups=dg), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        swa_model.update_parameters(model)
        logger.info(f"SWA Epoch {ep+1}/{swa_epochs}")

    model.load_state_dict(swa_model.module.state_dict())
    logger.info("SWA 完成, 已加载平均权重")

    # ---- 持久化 ----
    mean, std = stats
    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
        "n_feat_total": N_FEAT_TOTAL,
    }, model_path)
    logger.info("训练完成, 模型已保存", path=model_path)
    return model_path


ENSEMBLE_SEEDS = [42, 123, 456, 789, 111]


def merge_ensemble_to_single(out_path: str = MODEL_PATH):
    """将多个种子模型的权重逐元素平均, 合并为一个模型文件。"""
    import glob as _glob
    seed_paths = sorted(_glob.glob("stock_model_seed*.json"))
    if len(seed_paths) < 2:
        logger.info("无需合并 (仅一个种子模型)")
        if seed_paths:
            import shutil
            shutil.copy(seed_paths[0], out_path)
        return out_path

    logger.info(f"合并 {len(seed_paths)} 个种子模型 -> {out_path}")
    payloads = []
    for path in seed_paths:
        with open(path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))

    merged = payloads[0].copy()
    tensor_keys = list(payloads[0]["state_dict"].keys())

    for key in tensor_keys:
        dtype = getattr(torch, payloads[0]["state_dict"][key]["dtype"])
        mean_data = torch.tensor(payloads[0]["state_dict"][key]["data"], dtype=dtype).clone()
        for p in payloads[1:]:
            t = torch.tensor(p["state_dict"][key]["data"], dtype=dtype)
            mean_data += t
        mean_data /= len(payloads)
        merged["state_dict"][key]["data"] = mean_data.reshape(-1).tolist()

    for stat_key in ["mean", "std"]:
        if stat_key in payloads[0]:
            merged[stat_key] = list(np.mean([p[stat_key] for p in payloads], axis=0))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    logger.info(f"已保存合并模型: {out_path}")
    return out_path


if __name__ == "__main__":
    datasources = {"bar1m": "bigalpha_2026_stock_bar1m"}
    import glob as _glob
    # 如果有现成的集成模型跳过训练
    if os.path.exists(MODEL_PATH) and _glob.glob("stock_model_seed*.json"):
        logger.info("已存在 stock_model.json, 跳过训练。如需重训请删除该文件。")
    elif _glob.glob("stock_model_seed*.json"):
        merge_ensemble_to_single(MODEL_PATH)
    else:
        logger.info("训练 5 个种子模型 (集成)")
        for seed in ENSEMBLE_SEEDS:
            seed_path = f"stock_model_seed{seed}.json"
            if os.path.exists(seed_path):
                logger.info(f"跳过 (已存在)  seed={seed}")
                continue
            logger.info(f"===== 训练种子 {seed} =====")
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_and_save(datasources, model_path=seed_path)
        merge_ensemble_to_single(MODEL_PATH)
        logger.info("集成训练完成, 已生成 stock_model.json")

