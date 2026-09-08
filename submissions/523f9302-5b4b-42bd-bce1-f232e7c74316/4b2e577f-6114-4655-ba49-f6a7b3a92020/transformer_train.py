# -*- coding: utf-8 -*-
"""
多频率端到端 Transformer —— 训练侧脚本(共享定义 + 从零训练并持久化)

在官方极简 demo (单频 1m K 线 + 普通 Transformer) 基础上做三处升级:

  1. TimeMixer-lite: 同时接入 1m/5m/15m/30m 四个频率的原始 K 线+盘口 level1 数据,
     每个频率各自过一个小分支编码器, 再融合成一个综合表征, 对应赛题给出的
     多频率数据, 而不是只用 1m。

  2. iTransformer-lite: 每个频率分支内部除了做"时间维度"的 patch+attention,
     还并联一路"字段维度"的 attention(把每个原始字段的整段窗口当一个 token),
     让模型自己去发现字段间的联动关系(例如买卖量失衡这类规律),
     替代赛制不允许手工计算的衍生因子。

  3. PatchTST-lite: 时间维度上用不重叠 patch 池化降噪、压缩序列长度,
     控制 3 小时训练预算内的算力开销。


    本脚本每个频率的 SEQ_LEN 都保守设置在数十的量级, 回看周期不超过1-2天（需要修改）

字段预算: 当前设计 4 个频率 x 10 个原始字段 = 40 个字段, 远低于 100 上限,
后续如果要加入买卖 2~5 档, 每个频率还能再加约 32 个字段(仍在预算内)。

训练区间/防泄漏/文本化存盘等约定与官方 demo 保持一致, 不做改动:
  - TRAIN_START/TRAIN_END 写死, 不使用平台注入的 start_date/end_date 训练
  - 模型只存 JSON(不用二进制), 权重 + 结构超参一并保存
  - 私榜阶段平台会调用 train_and_save 在隔离环境从零重训, 训练逻辑需要在
    3 小时 GPU 预算内跑完、且可复现(固定随机种子)

标准化说明(与旧版的区别):
  - 不再使用"训练集算全局 mean/std, 存进 stats, 推理时复用"的方式。
  - 改为按天做横截面标准化(daily_cross_section_standardize), train/infer
    都是各自用当天的横截面自己算 mean/std, 不依赖任何预存的统计量。
  - 因此 build_dataset 返回的第 4 项统计量位不再使用, checkpoint 里也不再
    保存 stats 字段。

单日有效股票数过滤说明:
  - MIN_DAILY_STOCKS 统一在 windows_from_grouped 里对 train/infer 都生效,
    避免某天可用股票数过少(接近上市首日、大规模停牌等)导致横截面标准化
    的 mean/std 不稳定、进而污染当天的分数或损失。
"""
import os
import json
import time

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_multifreq_model.json")

# ---------- 训练区间(写死, 不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2019-01-01", "2023-12-31 23:59:59"
EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42
MAX_TRAIN_INSTRUMENTS = 502  # 使用全量成分股, 提升样本覆盖
MIN_DAILY_STOCKS = 5  # 单日横截面样本数下限, train/infer 统一生效(标准化+损失都按天计算, 样本太少会不稳定)

# ---------- 数据源配置: 4 个频率, 需按平台实际表名核实 ----------
FREQ_NAMES = ["bar1m", "bar5m", "bar15m", "bar30m"]
DEV_TABLES = {
    "bar1m":  "bigalpha_2026_stock_bar1m",
    "bar5m":  "bigalpha_2026_stock_bar5m",
    "bar15m": "bigalpha_2026_stock_bar15m",
    "bar30m": "bigalpha_2026_stock_bar30m",
}
# 各频率的原始字段(与官方 demo 保持一致的 10 个字段; VOL_COLS 训练前先 log1p)
PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS   = ["volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

# 各频率回看步数(可调, 越大越吃显存/时间; 4 个频率各自远低于 240)
SEQ_LEN = {"bar1m": 60, "bar5m": 48, "bar15m": 32, "bar30m": 16}
PATCH_LEN = {"bar1m": 4, "bar5m": 4, "bar15m": 4, "bar30m": 4}  # 需整除 SEQ_LEN

# ---------- 模型结构超参(训练/推理必须一致, 存入权重文件供推理端重建) ----------
D_MODEL = 48
NHEAD = 4
MODEL_CFG = dict(
    freq_names=FREQ_NAMES,
    n_feat=N_FEAT,
    seq_len=SEQ_LEN,
    patch_len=PATCH_LEN,
    d_model=D_MODEL,
    nhead=NHEAD,
    fusion_dim=96,
)


# ==================== 模型定义 ====================
class VariateAttention(nn.Module):
    """iTransformer-lite: 把每个原始字段的整段窗口当作一个 token, 做字段间 attention。
    用于在"禁止手工衍生特征"的约束下, 让模型自己学出买卖量失衡这类跨字段规律,
    而不需要你先手算 order imbalance / microprice 再喂进去。"""
    def __init__(self, seq_len, d_model, nhead=4):
        super().__init__()
        self.token_proj = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, 0.1,
                                            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):              # x: (B, L, N_FEAT)
        tok = self.token_proj(x.transpose(1, 2))   # (B, N_FEAT, d_model), 每个字段->1个token
        h = self.encoder(tok)                       # 字段间 attention
        return h.mean(dim=1)                         # (B, d_model)


class FreqBranch(nn.Module):
    """单频率分支: PatchTST-lite(不重叠 patch 池化降噪+压缩) 时间编码,
    并联 VariateAttention 字段编码, 两路拼接输出。"""
    def __init__(self, n_feat, seq_len, patch_len, d_model, nhead=4):
        super().__init__()
        assert seq_len % patch_len == 0, "seq_len 必须能被 patch_len 整除"
        self.patch_len = patch_len
        self.n_patch = seq_len // patch_len
        self.patch_proj = nn.Linear(n_feat * patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, 0.1,
                                            batch_first=True, activation="gelu")
        self.time_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.variate_attn = VariateAttention(seq_len, d_model, nhead=nhead)
        self.out_dim = d_model * 2

    def forward(self, x):              # x: (B, L, N_FEAT)
        B, L, F = x.shape
        patches = x.reshape(B, self.n_patch, self.patch_len * F)  # 不重叠 patch 化
        tok = self.patch_proj(patches) + self.pos
        h_time = self.time_encoder(tok).mean(dim=1)
        h_var = self.variate_attn(x)
        return torch.cat([h_time, h_var], dim=-1)


class MultiFreqStockModel(nn.Module):
    """TimeMixer-lite: 各频率分支独立编码后融合成一个综合表征, 接回归头输出打分。"""
    def __init__(self, freq_names, n_feat, seq_len, patch_len, d_model, nhead, fusion_dim):
        super().__init__()
        self.branches = nn.ModuleDict({
            name: FreqBranch(n_feat, seq_len[name], patch_len[name], d_model, nhead)
            for name in freq_names
        })
        total_dim = sum(b.out_dim for b in self.branches.values())
        self.fuse = nn.Sequential(
            nn.LayerNorm(total_dim), nn.Linear(total_dim, fusion_dim), nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, 1))

    def forward(self, x_dict):         # x_dict: {freq_name: (B, L_freq, N_FEAT)}
        embs = [self.branches[name](x_dict[name]) for name in self.branches]
        h = self.fuse(torch.cat(embs, dim=-1))
        return self.head(h).squeeze(-1)


# ==================== 数据构建 ====================
def pool(sd, ed):
    """区间内中证 1000 成分股代码。"""
    df = dai.query("SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
                    filters={"date": [sd, ed]}).df()
    return df["instrument"].tolist()


def _load_freq_raw(table, sd, ed, instruments, buf_days):
    """读取单个频率表的原始字段, 量纲大的字段先 log1p。"""
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=buf_days)).strftime("%Y-%m-%d")
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": instruments}).df()
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))
    return df


def load_raw_grouped(datasources, sd, ed, instruments):
    """预处理里最重的一步: 对 4 个频率表各发一次 SQL 查询, 拉取原始字段并按
    股票分组、按时间排好序。这一步涉及数据库网络IO, 是"预处理慢"的主要来源。

    单独拆出这个函数, 是为了让调用方(比如 CV 多折验证)能只调用一次, 把加载
    结果缓存在内存里给多个日期区间复用, 而不是每次都重新查一遍数据库
    (原来 build_dataset 每次调用都会重新查, 这也是 CV 脚本变慢的主因)。

    返回: {freq_name: {instrument: (day_array_升序, feat_array_对应顺序)}}
    """
    grouped = {}
    for name in FREQ_NAMES:
        table = datasources[name]
        # 缓冲天数按频率估一个够用的量, 1m 数据密集缓冲短些, 30m 数据稀疏缓冲长些
        buf_days = {"bar1m": 10, "bar5m": 15, "bar15m": 30, "bar30m": 45}[name]
        raw = _load_freq_raw(table, sd, ed, instruments, buf_days)
        per_ins = {}
        for ins, sub in raw.groupby("instrument", sort=False):
            per_ins[ins] = (sub["date"].dt.normalize().to_numpy(),
                             sub[FEATURE_COLS].to_numpy(np.float32))
        grouped[name] = per_ins
    return grouped


def windows_from_grouped(grouped, sd, ed, mode, instruments=None):
    """预处理里轻量的一步: 用已经加载好的 grouped 原始数据, 在 [sd, ed] 范围内
    逐日切窗口构造样本(不做标准化, 标准化交给上层处理, 方便同一份 grouped
    数据配合不同区间复用)。这一步是纯内存操作, 不涉及网络IO, 很快。

    instruments: 可选, 传入则只用这些股票(用于 CV 里按每折实际成分股过滤,
    而 grouped 本身可以是更大范围的并集缓存)。

    单日有效股票数过滤: 切完窗口后, 会统计每个交易日的样本数(即当天可用
    股票数), 剔除样本数 < MIN_DAILY_STOCKS 的整个交易日。train/infer 都会
    执行这一步, 保持一致(train 阶段以前只在训练循环里过滤, 现在统一提前
    到这里, infer 阶段以前完全没有过滤, 现在也一并补上)。

    返回: (X_dict, y_arr, idx_df)
    """
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    common_ins = set(grouped["bar1m"].keys())
    for name in FREQ_NAMES:
        common_ins &= set(grouped[name].keys())
    if instruments is not None:
        common_ins &= set(instruments)

    close_idx = FEATURE_COLS.index("close")
    wins = {name: [] for name in FREQ_NAMES}
    ys, keys = [], []

    for ins in common_ins:
        day1m, feat1m = grouped["bar1m"][ins]
        close_pos = np.flatnonzero(np.append(day1m[1:] != day1m[:-1], True))
        close_px = feat1m[close_pos, close_idx].astype(np.float64)
        dates = day1m[close_pos]

        for k, p in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if d < sd_ts or d > ed_ts:
                continue
            label = None
            if k + 1 < len(close_pos) and close_px[k] > 0:
                r = close_px[k + 1] / close_px[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue

            ok = True
            freq_windows = {}
            d64 = np.datetime64(d)
            for name in FREQ_NAMES:
                sub_day, sub_feats = grouped[name][ins]
                # searchsorted 找到第一个 > d 的位置, 用于截取 "<=d 的最后 L 根"
                cutoff = np.searchsorted(sub_day, d64, side="right")
                L = SEQ_LEN[name]
                if cutoff < L:
                    ok = False
                    break
                freq_windows[name] = sub_feats[cutoff - L: cutoff]
            if not ok:
                continue

            for name in FREQ_NAMES:
                wins[name].append(freq_windows[name])
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))

    if not keys:
        raise RuntimeError(f"windows_from_grouped 无样本 (mode={mode}, {sd}~{ed})")

    X_dict = {name: np.stack(wins[name]).astype(np.float32) for name in FREQ_NAMES}
    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])
    y_arr = np.array(ys, np.float32)

    # ---- 统一的"单日有效股票数过少"过滤: train/infer 都生效 ----
    day_counts = idx_df.groupby("date")["instrument"].transform("count")
    keep_mask = (day_counts >= MIN_DAILY_STOCKS).to_numpy()
    n_dropped_days = idx_df.loc[~keep_mask, "date"].nunique()
    if n_dropped_days > 0:
        logger.warning(
            "以下交易日样本数低于 MIN_DAILY_STOCKS, 已剔除",
            mode=mode, n_dropped_days=n_dropped_days, min_daily_stocks=MIN_DAILY_STOCKS,
        )
    if not keep_mask.any():
        raise RuntimeError(
            f"过滤 MIN_DAILY_STOCKS={MIN_DAILY_STOCKS} 后无剩余样本 (mode={mode}, {sd}~{ed})")

    X_dict = {name: X_dict[name][keep_mask] for name in FREQ_NAMES}
    y_arr = y_arr[keep_mask]
    idx_df = idx_df.loc[keep_mask].reset_index(drop=True)

    return X_dict, y_arr, idx_df


def daily_cross_section_standardize(X_dict, y_arr, idx_df):
    """按日期做横截面标准化。

    对每个交易日、每个频率、每个字段分别在当天横截面上算 mean/std。
    y_arr 也按当天横截面做 z-score；如果 y_arr 为 None，则只处理特征。

    注: 调用方(build_dataset)已经通过 windows_from_grouped 里的
    MIN_DAILY_STOCKS 过滤保证了传入这里的每一天样本数都够, 这里不再重复判断。
    """
    X_out = {name: np.empty_like(X_dict[name], dtype=np.float32) for name in FREQ_NAMES}
    y_out = None if y_arr is None else np.empty_like(y_arr, dtype=np.float32)

    for _, day_idx in idx_df.groupby("date", sort=False).indices.items():
        day_idx = np.asarray(day_idx, dtype=np.int64)
        for name in FREQ_NAMES:
            block = X_dict[name][day_idx]
            flat = block.reshape(-1, N_FEAT)
            mean = flat.mean(0, keepdims=True).astype(np.float32)
            std = flat.std(0, keepdims=True).astype(np.float32) + 1e-6
            X_out[name][day_idx] = ((block - mean) / std).astype(np.float32)
        if y_out is not None:
            day_y = y_arr[day_idx].astype(np.float32)
            mean_y = np.float32(day_y.mean())
            std_y = np.float32(day_y.std() + 1e-6)
            y_out[day_idx] = ((day_y - mean_y) / std_y).astype(np.float32)

    return X_out, y_out


def daily_rank_ic_loss(pred, target):
    """日度 RankIC 的可导代理：同一截面内最大化相关性。"""
    pred = pred - pred.mean()
    target = target - target.mean()
    pred = pred / (pred.std(unbiased=False) + 1e-6)
    target = target / (target.std(unbiased=False) + 1e-6)
    return 1.0 - (pred * target).mean()


def build_dataset(datasources, sd, ed, mode, instruments):
    """单次训练/推理的便捷入口(内部调用 load_raw_grouped + windows_from_grouped
    + 按天截面标准化)。train_and_save / predict 的 main() 只调用一次, 直接用这个即可。

    !! CV 多折场景不要直接反复调用这个函数, 会导致每折都重新查一遍数据库 !!
    应该改用 load_raw_grouped 只查一次、windows_from_grouped 对每折切片
    (见 transformer_cv.py 的用法)。

    返回:
      mode='train': (X_dict, y, idx_df)
      mode='infer': (X_dict, None, idx_df)
            X_dict: {freq_name: (N, SEQ_LEN[freq], N_FEAT)}

    注: 不再返回/使用全局 stats, 标准化统一按天在内部做完(daily_cross_section_standardize)。
    """
    t0 = time.time()
    grouped = load_raw_grouped(datasources, sd, ed, instruments)
    X_dict, y_arr, idx_df = windows_from_grouped(grouped, sd, ed, mode)

    X_dict, y_arr = daily_cross_section_standardize(X_dict, y_arr, idx_df)

    logger.info(f"{mode} 集构建完成", samples=len(idx_df), days=idx_df["date"].nunique(),
                elapsed=round(time.time() - t0, 2))
    if mode == "train":
        return X_dict, y_arr, idx_df
    return X_dict, None, idx_df


# ==================== 模型存 / 读(JSON 文本, 与官方 demo 约定一致) ====================
def save_model(ckpt, model_path=MODEL_PATH):
    """把 checkpoint 存成 JSON 文本文件, 张量转成 {dtype, shape, data(扁平list)}。"""
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {"dtype": str(t.dtype).replace("torch.", ""),
                      "shape": list(t.shape), "data": t.reshape(-1).tolist()}
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    """读取 save_model 写出的 JSON, 还原为 state_dict。"""
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ==================== 训练并持久化 ====================
def train_and_save(datasources, model_path=MODEL_PATH):
    """在写死的训练区间上从零训练, 把 权重 + 结构超参 + 字段列表 一并存盘。

    公榜阶段平台不会重训, 直接加载该文件做推理; 私榜阶段平台用本函数从零重训,
    故训练逻辑需保持可复现(固定随机种子)且能在 3 小时 GPU 预算内跑完。"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), datasources=datasources)

    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    Xtr, ytr, idx_df = build_dataset(
        datasources, TRAIN_START, TRAIN_END, "train",
        pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS])

    model = MultiFreqStockModel(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("可训练参数量", n_params=n_params)
    if not (1e5 <= n_params <= 1e8):
        logger.warning("参数量超出赛制允许区间[1e5,1e8], 请调整 D_MODEL/fusion_dim 后重跑",
                        n_params=n_params)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # windows_from_grouped 已经统一过滤过 MIN_DAILY_STOCKS, 这里理论上不会再
    # 剔除任何一天; 保留这个条件只是双重保险, 避免未来有人绕开 build_dataset
    # 直接拼数据时漏掉过滤。
    day_groups = [np.asarray(idx, dtype=np.int64)
                  for idx in idx_df.groupby("date", sort=False).indices.values()
                  if len(idx) >= MIN_DAILY_STOCKS]

    model.train()
    for ep in range(EPOCHS):
        t, tot, nb = time.time(), 0.0, 0
        for gidx in np.random.permutation(len(day_groups)):
            day_idx = day_groups[gidx]
            xb_dict = {name: torch.from_numpy(Xtr[name][day_idx]).to(device, non_blocking=True)
                       for name in FREQ_NAMES}
            yb = torch.from_numpy(ytr[day_idx]).to(device, non_blocking=True)
            opt.zero_grad()
            loss = daily_rank_ic_loss(model(xb_dict), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        logger.info("epoch 完成", epoch=ep + 1, loss=round(tot / max(nb, 1), 8),
                    elapsed=round(time.time() - t, 2))

    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
    }, model_path)
    logger.info("模型已保存, 请随 notebook 一并上传", path=model_path)
    return model_path


if __name__ == "__main__":
    # 本地训练入口: 在写死的训练区间上从零训练并保存 transformer_multifreq_model.json
    datasources = {name: DEV_TABLES[name] for name in FREQ_NAMES}
    train_and_save(datasources)
