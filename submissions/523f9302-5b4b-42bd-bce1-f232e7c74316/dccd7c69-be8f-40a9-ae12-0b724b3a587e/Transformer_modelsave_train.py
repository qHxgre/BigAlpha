 #-*- coding: utf-8 -*-
"""Transformer 训练侧脚本 (与 fixed_stock_transformer.py 模型定义一致)。

本文件承担两件事:
  1. 沉淀训练与推理共用的定义 (配置 / 模型 / 数据), 与推理侧脚本
     fixed_stock_transformer.py 中的 StockTransformer 完全一致。
  2. 提供 `train_and_save(...)`: 在写死的训练区间上从零训练, 把
     权重 + 标准化统计 + 结构超参 一并保存到 transformer_model.json (纯文本)。

模型存为文本类文件 (JSON), 不使用 .pt 等二进制。
"""
import os
import json
import time
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

MODEL_PATH = "transformer_model.json"

# ---------- 配置 (与 fixed_stock_transformer.py 同步) ----------
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"
SEQ_LEN = 80
EPOCHS, BATCH, LR, SEED = 50, 256, 1e-3, 42
GRAD_ACCUM_STEPS = 2
MAX_TRAIN_INSTRUMENTS = 500

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS   = ["volume", "amount", "bid_volume1", "ask_volume1"]
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
# 技术指标
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

    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().to_numpy()
    sub["norm_close"] = (close / np.maximum(sma20, 1e-10)).astype(np.float32)

    sub["rsi_14"] = compute_rsi(close, 14).astype(np.float32)

    sub["rsi_7"] = compute_rsi(close, 7).astype(np.float32)

    vol_sma20 = pd.Series(volume).rolling(20, min_periods=1).mean().to_numpy()
    sub["vol_ratio"] = (volume / np.maximum(vol_sma20, 1e-10)).astype(np.float32)

    sub["range_ratio"] = ((high - low) / np.maximum(close, 1e-10)).astype(np.float32)

    spread = (ask - bid) / np.maximum(close, 1e-10)
    sub["spread"] = np.clip(spread, 0.0, 0.1).astype(np.float32)

    # roc_10: 10 日收益率
    roc = np.full_like(close, 0.0, dtype=np.float64)
    roc[10:] = (close[10:] - close[:-10]) / (close[:-10] + 1e-10)
    sub["roc_10"] = roc.astype(np.float32)

    return sub


TECH_FEATURE_COLS = ["norm_close", "rsi_14", "rsi_7", "vol_ratio", "range_ratio", "spread", "roc_10"]
FEATURE_COLS = BASE_FEATURE_COLS + TECH_FEATURE_COLS
N_FEAT = len(FEATURE_COLS)


# ======================================================================
# 模型 (与 fixed_stock_transformer.py 完全一致)
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


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.proj(self.query).expand(h.size(0), -1, -1)
        attn = torch.bmm(q, h.transpose(1, 2)) * self.scale
        if mask is not None:
            attn = attn + mask[:, -1:, :]
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, h).squeeze(1)


class StockTransformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        d_model: int = 96,
        nhead: int = 6,
        nlayers: int = 3,
        dim_ff: int = 192,
        seq_len: int = SEQ_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.proj = nn.Sequential(
            nn.LayerNorm(n_feat),
            nn.Linear(n_feat, d_model),
        )

        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len)

        layer = TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = TransformerEncoder(layer, nlayers)

        self.pool = AttentionPooling(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos_encoder(x)
        h = self.encoder(h, mask=self.causal_mask)
        h = self.pool(h, mask=self.causal_mask.unsqueeze(0))
        return self.head(h).squeeze(-1)


MODEL_CFG = dict(
    n_feat=N_FEAT, d_model=128, nhead=8, nlayers=4,
    dim_ff=256, seq_len=SEQ_LEN,
)


# ======================================================================
# 未来 30 bar VWAP 收益
# ======================================================================

def compute_vwap_future_return(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, horizon: int = 30,
) -> np.ndarray:
    typical = (high + low + close) / 3.0
    pv = typical * volume
    n = len(close)
    ret = np.full(n, np.nan, dtype=np.float64)
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    for i in range(n - horizon):
        total_pv = cum_pv[i + horizon] - cum_pv[i]
        total_vol = cum_vol[i + horizon] - cum_vol[i]
        if total_vol > 1e-10 and close[i] > 1e-10:
            future_vwap = total_pv / total_vol
            ret[i] = future_vwap / close[i] - 1.0
    return ret


# ======================================================================
# 数据构建 (与 fixed_stock_transformer.py 一致)
# ======================================================================

def build_dataset(
    table: str, sd: str, ed: str, mode: str,
    instruments: List[str],
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    t0 = time.time()
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    sql = (
        f"SELECT date, instrument, {', '.join(BASE_FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": instruments}).df()

    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))

    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    wins, ys, keys = [], [], []

    for ins, sub in df.groupby("instrument", sort=False):
        if len(sub) <= SEQ_LEN + 30:
            continue
        sub = sub.sort_values("date").reset_index(drop=True)
        sub = add_technical_features(sub)

        close_arr = sub["close"].to_numpy(np.float64)
        high_arr = sub["high"].to_numpy(np.float64)
        low_arr = sub["low"].to_numpy(np.float64)
        vol_arr = sub["volume"].to_numpy(np.float64)
        future_ret = compute_vwap_future_return(
            close_arr, high_arr, low_arr, vol_arr, horizon=30,
        )

        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        feats = rolling_normalize(feats)

        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        dates = day[close_pos]

        if mode == "train":
            close_px = sub["close"].to_numpy(np.float64)[close_pos]
            for k, p in enumerate(close_pos):
                d = pd.Timestamp(dates[k])
                if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                    continue
                label_y = future_ret[p]
                if not np.isfinite(label_y):
                    continue
                wins.append(feats[p - SEQ_LEN + 1: p + 1])
                ys.append(np.float32(label_y))
                keys.append((d, ins))
        else:
            for k, p in enumerate(close_pos):
                d = pd.Timestamp(dates[k])
                if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                    continue
                wins.append(feats[p - SEQ_LEN + 1: p + 1])
                ys.append(None)
                keys.append((d, ins))

    if not keys:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")

    X = np.stack(wins).astype(np.float32)
    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        stats = (
            flat.mean(0).astype(np.float32),
            flat.std(0).astype(np.float32) + 1e-6,
        )
    m, s = stats
    X = ((X - m) / s).astype(np.float32)

    logger.info(
        f"{mode} 集构建完成", samples=len(keys),
        n_feat=N_FEAT, elapsed=round(time.time() - t0, 2),
    )
    if mode == "train":
        return X, np.array(ys, np.float32), None, stats
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
    model = StockTransformer(**cfg).to(device)
    model.load_state_dict(sd)
    mean = torch.tensor(payload["mean"], dtype=torch.float32).to(device)
    std = torch.tensor(payload["std"], dtype=torch.float32).to(device)
    return model, mean, std


# ======================================================================
# 训练并持久化 (梯度累积)
# ======================================================================

def train_and_save(datasources: dict, model_path: str = MODEL_PATH) -> str:
    table = datasources["bar1m"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(dev), table=table)

    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    Xtr, ytr, _, stats = build_dataset(
        table, TRAIN_START, TRAIN_END, "train",
        pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS],
    )

    lo, hi = np.percentile(ytr, [1, 99])
    ytr = np.clip(ytr, lo, hi)

    model = StockTransformer(**MODEL_CFG).to(dev)
    logger.info("可训练参数量", n_params=sum(p.numel() for p in model.parameters()))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=BATCH, shuffle=True,
        pin_memory=(dev.type == "cuda"),
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    model.train()
    opt.zero_grad()
    global_step = 0

    for ep in range(EPOCHS):
        t, tot, nb = time.time(), 0.0, 0
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            loss = loss_fn(model(xb), yb)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            tot += loss.item() * GRAD_ACCUM_STEPS
            nb += 1
            global_step += 1

            if global_step % GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        scheduler.step()
        logger.info(
            "epoch 完成", epoch=ep + 1,
            mse=round(tot / max(nb, 1), 8),
            elapsed=round(time.time() - t, 2),
        )

    mean, std = stats
    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
    }, model_path)
    logger.info("训练完成, 模型已保存", path=model_path)
    return model_path


if __name__ == "__main__":
    datasources = {"bar1m": "bigalpha_2026_stock_bar1m"}
    train_and_save(datasources)
#
