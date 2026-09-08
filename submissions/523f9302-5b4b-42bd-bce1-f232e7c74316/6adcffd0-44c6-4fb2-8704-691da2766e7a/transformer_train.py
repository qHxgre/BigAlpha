# -*- coding: utf-8 -*-
"""Transformer 端到端 demo —— 训练侧脚本 (共享定义 + 从零训练并持久化)。

本文件承担两件事:
  1. 沉淀 **训练与推理共用** 的定义 (配置 / 模型结构 / 数据构建), 作为单一事实来源;
     配套 notebook 在推理时直接 `from transformer_train import ...` 复用, 避免两边漂移。
  2. 提供 `train_and_save(...)`: 在写死的训练区间上从零训练, 把
     **权重 + 标准化统计 + 结构超参** 一并保存到 `transformer_model.json` (纯文本)。

用法:
    python transformer_train.py
    from transformer_train import train_and_save
    train_and_save({"bar30m": "bigalpha_2026_stock_bar30m"})

默认 30 分钟 K 线。标签为「隔夜 + 次日开盘」收益 (close_t -> open_{t+1})。
训练含截面 z-score、GRU+Transformer、末 bar 拼接、2023H2 验证 early stopping。
"""
import os
import json
import time
import gc
import logging
import sys
import copy

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
import structlog

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")
LOG_PATH = os.path.join(_HERE, "transformer_train.log")


def setup_logging(log_path=LOG_PATH):
    """Stdout + 文件双写; 文件即时 flush, 方便 `tail -f` 看训练进度。"""
    root = logging.getLogger()
    if getattr(setup_logging, "_done", False):
        return log_path
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    class _FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    fh = _FlushFileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    setup_logging._done = True
    return log_path


setup_logging()
logger = structlog.get_logger()


def memory_mb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


# ---------- 配置 ----------
FIT_START, FIT_END = "2019-01-01", "2023-06-30 23:59:59"
VAL_START, VAL_END = "2023-07-01", "2023-12-31 23:59:59"
# 兼容旧名
TRAIN_START, TRAIN_END = FIT_START, VAL_END

BAR_FREQ = "30m"
BAR_TABLE = "bigalpha_2026_stock_bar30m"
SEQ_LEN = 40
EPOCHS, BATCH, LR, SEED = 15, 128, 2e-4, 42
PATIENCE = 3
INFER_INSTRUMENT_CHUNK = 10_000  # 截面zscore要求按日完整股票池; 不再按股票切碎
INFER_CALENDAR_DAY_CHUNK = 31    # 30m 下按月分块控内存即可
MAX_TRAIN_INSTRUMENTS = 450
TRAIN_PERIOD_CHUNKS = 10
MIN_CS_SIZE = 20
STATS_CHUNKS = 3


PRICE_COLS = ["pre_close", "open", "high", "low", "close"] + [
    f"{side}_price{level}" for side in ("bid", "ask") for level in range(1, 6)
]
VOL_COLS = ["volume", "amount", "deal_number"] + [
    f"{side}_{kind}{level}"
    for side in ("bid", "ask")
    for kind in ("volume", "num_orders")
    for level in range(1, 6)
]
RAW_COLS = PRICE_COLS + VOL_COLS
DERIVED_COLS = [
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16",
    "hl_spread", "mid_spread", "vol_share", "vwap_dev", "open_dev", "overnight_gap",
    "rv_4", "rv_16", "vol_z_16",
    "book_imbalance1", "book_imbalance5",
    "order_imbalance5", "trade_intensity", "range_pos_16",
]
FEATURE_COLS = RAW_COLS + DERIVED_COLS
N_FEAT = len(FEATURE_COLS)

MODEL_CFG = dict(
    n_feat=N_FEAT, d_model=160, nhead=8, nlayers=4, dim_ff=512,
    seq_len=SEQ_LEN, gru_layers=1,
)


def resolve_table(datasources):
    if isinstance(datasources, str):
        return datasources
    for key in ("bar30m", "bar15m", "bar1m"):
        if key in datasources and datasources[key]:
            return datasources[key]
    return BAR_TABLE


def table_freq(table):
    name = str(table).lower()
    if "bar30m" in name or name.endswith("30m"):
        return "30m"
    if "bar15m" in name or name.endswith("15m"):
        return "15m"
    if "bar1m" in name or name.endswith("1m"):
        return "1m"
    return BAR_FREQ


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h):
        w = torch.softmax(self.score(h), dim=1)
        return (h * w).sum(dim=1)


# ---------- 模型: stem -> GRU -> Transformer -> (attn pool || last) -> head ----------
class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=160, nhead=8, nlayers=4, dim_ff=512,
                 seq_len=SEQ_LEN, gru_layers=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            d_model, d_model, num_layers=gru_layers,
            batch_first=True, dropout=0.0,
        )
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, 0.12,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        h, _ = self.gru(h)
        h = self.encoder(h + self.pos)
        pooled = self.pool(h)
        last = h[:, -1, :]
        return self.head(torch.cat([pooled, last], dim=-1)).squeeze(-1)


def pool(sd, ed):
    t0 = time.time()
    logger.info("读取股票池", start=sd, end=ed)
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments ORDER BY instrument",
        filters={"date": [sd, ed]}, compression=True,
    ).df()
    logger.info("股票池读取完成", instruments=len(df), elapsed=round(time.time() - t0, 2),
                memory_mb=memory_mb())
    return df["instrument"].tolist()


def select_train_instruments(instruments, limit, seed):
    if limit is None or len(instruments) <= limit:
        return instruments
    chosen = np.random.RandomState(seed).choice(instruments, limit, replace=False)
    return sorted(chosen.tolist())


def universe_membership(sd, ed, instruments=None):
    filters = {"date": [sd, ed]}
    if instruments is not None:
        filters["instrument"] = instruments
    t0 = time.time()
    logger.info("读取历史成分关系", start=sd, end=ed,
                requested_instruments=None if instruments is None else len(instruments))
    u = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters=filters, compression=True,
    ).df()
    u["trade_date"] = pd.to_datetime(u["date"]).dt.normalize()
    u = u[["trade_date", "instrument"]].drop_duplicates()
    logger.info("历史成分关系读取完成", rows=len(u), days=u["trade_date"].nunique(),
                elapsed=round(time.time() - t0, 2), memory_mb=memory_mb())
    return u


def _add_derived_features(df):
    close_safe = df["close"].replace(0, np.nan)
    close_group = df.groupby("instrument", sort=False)["close"]
    for lag in (1, 2, 4, 8, 16):
        df[f"ret_{lag}"] = close_group.pct_change(lag).fillna(0.0).clip(-0.2, 0.2)
    df["hl_spread"] = ((df["high"] - df["low"]) / close_safe).fillna(0.0).clip(0.0, 1.0)
    df["mid_spread"] = (
        (df["ask_price1"] - df["bid_price1"]) / close_safe
    ).fillna(0.0).clip(-0.1, 0.1)
    day_group = df.groupby(["instrument", "trade_date"], sort=False)
    cum_vol = day_group["volume"].cumsum()
    cum_amount = day_group["amount"].cumsum()
    df["vol_share"] = (df["volume"] / cum_vol.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    cum_vwap = cum_amount / cum_vol.replace(0, np.nan)
    df["vwap_dev"] = ((df["close"] - cum_vwap) / close_safe).fillna(0.0).clip(-0.1, 0.1)
    day_open = day_group["open"].transform("first").replace(0, np.nan)
    df["day_open"] = day_open
    df["open_dev"] = (df["close"] / day_open - 1.0).fillna(0.0).clip(-0.2, 0.2)
    pre_close = df["pre_close"].replace(0, np.nan)
    df["overnight_gap"] = (day_open / pre_close - 1.0).fillna(0.0).clip(-0.2, 0.2)

    ret1 = df["ret_1"]
    for window in (4, 16):
        df[f"rv_{window}"] = (
            ret1.groupby(df["instrument"], sort=False)
            .rolling(window, min_periods=3).std()
            .reset_index(level=0, drop=True)
            .fillna(0.0).clip(0.0, 0.2)
        )
    vol = df["volume"].astype(np.float64)
    vol_mean = (
        vol.groupby(df["instrument"], sort=False)
        .rolling(16, min_periods=4).mean().reset_index(level=0, drop=True)
    )
    vol_std = (
        vol.groupby(df["instrument"], sort=False)
        .rolling(16, min_periods=4).std().reset_index(level=0, drop=True)
    )
    df["vol_z_16"] = ((vol - vol_mean) / (vol_std + 1e-6)).fillna(0.0).clip(-5.0, 5.0)

    bid1, ask1 = df["bid_volume1"], df["ask_volume1"]
    df["book_imbalance1"] = ((bid1 - ask1) / (bid1 + ask1).replace(0, np.nan)).fillna(0.0)
    bid5 = sum(df[f"bid_volume{i}"] for i in range(1, 6))
    ask5 = sum(df[f"ask_volume{i}"] for i in range(1, 6))
    df["book_imbalance5"] = ((bid5 - ask5) / (bid5 + ask5).replace(0, np.nan)).fillna(0.0)
    bid_orders = sum(df[f"bid_num_orders{i}"] for i in range(1, 6))
    ask_orders = sum(df[f"ask_num_orders{i}"] for i in range(1, 6))
    df["order_imbalance5"] = (
        (bid_orders - ask_orders) / (bid_orders + ask_orders).replace(0, np.nan)
    ).fillna(0.0)
    df["trade_intensity"] = (
        df["deal_number"] / df["volume"].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 10.0)
    roll_high = (
        df["high"].groupby(df["instrument"], sort=False)
        .rolling(16, min_periods=4).max().reset_index(level=0, drop=True)
    )
    roll_low = (
        df["low"].groupby(df["instrument"], sort=False)
        .rolling(16, min_periods=4).min().reset_index(level=0, drop=True)
    )
    df["range_pos_16"] = (
        (df["close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan) - 0.5
    ).fillna(0.0).clip(-0.5, 0.5)
    return df


def _stationarize_raw_features(df):
    close_safe = df["close"].replace(0, np.nan)
    close_ret = (df["close"] / df["pre_close"].replace(0, np.nan) - 1.0)
    for c in PRICE_COLS:
        if c == "close":
            continue
        df[c] = (df[c] / close_safe - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[c] = df[c].clip(-0.3, 0.3)
    df["close"] = close_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.2, 0.2)
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))
    return df


def to_cs_rank(dates, ys):
    df = pd.DataFrame({"date": dates, "y": ys})
    df["y"] = df.groupby("date")["y"].rank(pct=True, method="average")
    return df["y"].to_numpy(np.float32)


def apply_cs_zscore(X, dates):
    """Per-day cross-sectional z-score over instruments for each (time, feature).

    要求 dates 对应的样本在当日尽量覆盖完整截面; 若只传入部分股票,
    z-score 会与全市场不一致 (训练/推理漂移)。
    """
    dates = np.asarray(pd.to_datetime(dates))
    # 用整数日代码分组, 避免 Timestamp 比较的隐患
    day_codes = dates.astype("datetime64[D]").astype(np.int64)
    for code in np.unique(day_codes):
        mask = day_codes == code
        n = int(mask.sum())
        if n < 2:
            continue
        block = X[mask]
        mean = block.mean(axis=0, keepdims=True)
        std = block.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        X[mask] = ((block - mean) / std).astype(np.float32)
    np.clip(X, -8.0, 8.0, out=X)
    return X


def soft_rank(x):
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    return torch.sigmoid(diff * 8.0).sum(dim=1)


def ic_loss(pred, target, huber_fn, alpha=0.15):
    if pred.numel() < MIN_CS_SIZE:
        return huber_fn(pred, target)
    pr = soft_rank(pred)
    tr = soft_rank(target)
    px = pr - pr.mean()
    ty = tr - tr.mean()
    corr = (px * ty).mean() / (px.std() * ty.std() + 1e-8)
    return (1.0 - corr) + alpha * huber_fn(pred, target)


def daily_rank_ic(pred, target):
    """Non-differentiable Spearman IC for validation."""
    if pred.size < MIN_CS_SIZE:
        return np.nan
    pr = pd.Series(pred).rank(method="average").to_numpy()
    tr = pd.Series(target).rank(method="average").to_numpy()
    if pr.std() < 1e-12 or tr.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(pr, tr)[0, 1])


def split_train_periods(start, end, n_chunks=TRAIN_PERIOD_CHUNKS):
    sd = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    if n_chunks <= 1 or (ed - sd).days < 120:
        return [(start, end if " " in str(end) else f"{end} 23:59:59")]
    edges = np.linspace(sd.value, ed.value, n_chunks + 1, dtype=np.int64)
    periods = []
    for i in range(n_chunks):
        ps = pd.Timestamp(edges[i]).normalize()
        pe = pd.Timestamp(edges[i + 1])
        if i + 1 < n_chunks:
            pe = pe - pd.Timedelta(seconds=1)
        if ps > ed:
            continue
        pe = min(pe, ed)
        periods.append((ps.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d %H:%M:%S")))
    return periods


def train_one_chunk(model, opt, huber_fn, X, y, idx_df, device, epoch, chunk_no,
                    n_chunks, seed, log_first=False):
    lo, hi = np.percentile(y, [1, 99])
    y = to_cs_rank(idx_df["date"].values, np.clip(y, lo, hi))
    dates = pd.to_datetime(idx_df["date"]).values
    unique_dates = pd.unique(dates)
    order = np.random.RandomState(seed + epoch * 997 + chunk_no).permutation(len(unique_dates))
    n_days = len(unique_dates)
    t, tot, nb, skipped = time.time(), 0.0, 0, 0
    last_progress_log = 0.0
    for batch_no, di in enumerate(order, 1):
        mask = dates == unique_dates[di]
        if int(mask.sum()) < MIN_CS_SIZE:
            skipped += 1
            continue
        xb = torch.from_numpy(np.ascontiguousarray(X[mask])).to(device)
        yb = torch.from_numpy(y[mask]).to(device)
        if log_first and nb == 0:
            logger.info("分块训练首批", epoch=epoch + 1, chunk=f"{chunk_no}/{n_chunks}",
                        x_shape=list(xb.shape), memory_mb=memory_mb())
        opt.zero_grad(set_to_none=True)
        loss = ic_loss(model(xb), yb, huber_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
        nb += 1
        now = time.time()
        if now - last_progress_log >= 2.0 or batch_no == n_days:
            logger.info(
                "分块训练进度",
                epoch=f"{epoch + 1}/{EPOCHS}",
                chunk=f"{chunk_no}/{n_chunks}",
                batch=f"{batch_no}/{n_days}",
                used_days=nb, skipped_days=skipped,
                avg_loss=round(tot / max(nb, 1), 8),
                memory_mb=memory_mb(),
            )
            last_progress_log = now
    del X, y, idx_df
    gc.collect()
    return tot / max(nb, 1), round(time.time() - t, 2)


@torch.no_grad()
def eval_rank_ic(model, X, y, idx_df, device):
    """Mean daily Rank-IC on a held-out chunk."""
    model.eval()
    lo, hi = np.percentile(y, [1, 99])
    y = np.clip(y, lo, hi)
    dates = pd.to_datetime(idx_df["date"]).values
    ics = []
    for d in pd.unique(dates):
        mask = dates == d
        if int(mask.sum()) < MIN_CS_SIZE:
            continue
        xb = torch.from_numpy(np.ascontiguousarray(X[mask])).to(device)
        pred = model(xb).cpu().numpy()
        ic = daily_rank_ic(pred, y[mask])
        if np.isfinite(ic):
            ics.append(ic)
    model.train()
    if not ics:
        return float("nan"), 0
    return float(np.mean(ics)), len(ics)


def _merge_stats(stats_list):
    means = np.stack([s[0] for s in stats_list], axis=0)
    stds = np.stack([s[1] for s in stats_list], axis=0)
    return means.mean(0).astype(np.float32), stds.mean(0).astype(np.float32) + 1e-6


def _iter_window_samples(sub, ins, sd_ts, ed_ts, mode):
    """Yield (window, label, trade_date). Label = open_{t+1}/close_t - 1 (隔夜+次日开盘)."""
    if len(sub) <= SEQ_LEN:
        return
    feats = sub[FEATURE_COLS].to_numpy(np.float32)
    day = sub["trade_date"].to_numpy()
    close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
    close_arr = np.asarray(sub["label_close"], dtype=np.float64).reshape(-1)
    day_open_arr = np.asarray(sub["label_day_open"], dtype=np.float64).reshape(-1)
    close_px = close_arr[close_pos]
    next_open = day_open_arr[close_pos]
    dates = day[close_pos]
    for k, p in enumerate(close_pos):
        d = pd.Timestamp(dates[k])
        if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
            continue
        label = None
        # 隔夜 + 次日开盘: close_t -> open_{t+1}
        if k + 1 < len(close_pos) and close_px[k] > 0 and next_open[k + 1] > 0:
            r = next_open[k + 1] / close_px[k] - 1.0
            if np.isfinite(r):
                label = np.float32(r)
        if mode == "train" and label is None:
            continue
        yield feats[p - SEQ_LEN + 1: p + 1], (
            label if label is not None else np.float32(0.0)
        ), d, ins


def _count_samples(df, sd_ts, ed_ts, mode):
    n = 0
    grouped = df.groupby("instrument", sort=False)
    total = df["instrument"].nunique()
    for ins_no, (ins, sub) in enumerate(grouped, 1):
        n += sum(1 for _ in _iter_window_samples(sub, ins, sd_ts, ed_ts, mode))
        if ins_no % 50 == 0 or ins_no == total:
            logger.info("样本计数进度", mode=mode, completed=ins_no, total=total,
                        samples=n, memory_mb=memory_mb())
    return n


def _fill_samples(df, sd_ts, ed_ts, mode, X, ys):
    keys_date, keys_ins = [], []
    grouped = df.groupby("instrument", sort=False)
    total = df["instrument"].nunique()
    idx = 0
    for ins_no, (ins, sub) in enumerate(grouped, 1):
        for win, label, d, ins_code in _iter_window_samples(sub, ins, sd_ts, ed_ts, mode):
            X[idx] = win
            ys[idx] = label
            keys_date.append(d)
            keys_ins.append(ins_code)
            idx += 1
        if ins_no % 50 == 0 or ins_no == total:
            logger.info("时序窗口进度", mode=mode, completed=ins_no, total=total,
                        samples=idx, memory_mb=memory_mb())
    return keys_date, keys_ins


def build_dataset(table, sd, ed, mode, instruments, stats=None, cs_zscore=True):
    """切窗口并标准化。mode='train'/'infer'。
    标准化: 全局 mean/std -> 当日截面 cs-zscore。

    注意: cs_zscore=True 时, instruments 在每个交易日应尽量是完整截面;
    推理不要按股票子集分块后再 z-score, 否则与训练分布不一致。
    """
    t0 = time.time()
    buf_start = (pd.to_datetime(sd) - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    # 向后多取几天, 保证区间末日仍有次日开盘价做标签 (train); infer 无害
    buf_end = (pd.to_datetime(ed) + pd.Timedelta(days=10)).strftime("%Y-%m-%d 23:59:59")
    sql = f"SELECT date, instrument, {', '.join(RAW_COLS)} FROM {table} ORDER BY instrument, date"
    filters = {"date": [buf_start, buf_end]}
    if instruments is not None:
        filters["instrument"] = instruments
    logger.info("开始查询K线", mode=mode, table=table, start=buf_start, end=buf_end,
                instruments=None if instruments is None else len(instruments),
                memory_mb=memory_mb())
    tq = time.time()
    df = dai.query(sql, filters=filters, compression=True).df()
    logger.info("K线查询完成", mode=mode, rows=len(df),
                elapsed=round(time.time() - tq, 2), memory_mb=memory_mb())
    df["date"] = pd.to_datetime(df["date"])
    df["trade_date"] = df["date"].dt.normalize()
    members = universe_membership(buf_start, buf_end, instruments)
    df = df.merge(members, on=["trade_date", "instrument"], how="inner")
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    logger.info("历史成分过滤完成", mode=mode, rows=len(df),
                instruments=df["instrument"].nunique(), memory_mb=memory_mb())
    for c in RAW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[RAW_COLS] = df.groupby("instrument")[RAW_COLS].ffill().fillna(0.0)
    df["label_close"] = df["close"].astype(np.float64)
    df = _add_derived_features(df)
    df["label_day_open"] = df["day_open"].astype(np.float64)
    df = _stationarize_raw_features(df)
    keep_cols = ["instrument", "trade_date", "label_close", "label_day_open"] + FEATURE_COLS
    df = df[keep_cols]

    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    logger.info("开始统计样本数", mode=mode, instruments=df["instrument"].nunique(),
                seq_len=SEQ_LEN)
    n_samples = _count_samples(df, sd_ts, ed_ts, mode)
    if n_samples == 0:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")

    est_mb = round(n_samples * SEQ_LEN * N_FEAT * 4 / 1024 / 1024, 1)
    logger.info("预分配样本数组", mode=mode, samples=n_samples, est_mb=est_mb,
                memory_mb=memory_mb())
    X = np.empty((n_samples, SEQ_LEN, N_FEAT), dtype=np.float32)
    ys = np.empty(n_samples, dtype=np.float32)
    keys_date, keys_ins = _fill_samples(df, sd_ts, ed_ts, mode, X, ys)
    if len(keys_date) != n_samples:
        raise RuntimeError(
            f"样本数不一致: count={n_samples}, filled={len(keys_date)}"
        )
    del df
    gc.collect()
    logger.info("原始DataFrame已释放", mode=mode, memory_mb=memory_mb())

    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        stats = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32) + 1e-6)
    m, s = stats
    X -= m
    X /= s
    np.clip(X, -8.0, 8.0, out=X)
    idx_df = pd.DataFrame({"date": keys_date, "instrument": keys_ins})
    if cs_zscore:
        apply_cs_zscore(X, idx_df["date"].values)
        logger.info("截面zscore完成", mode=mode, shape=list(X.shape),
                    days=idx_df["date"].nunique(), memory_mb=memory_mb())
    logger.info(f"{mode} 集构建完成", samples=n_samples, elapsed=round(time.time() - t0, 2))
    if mode == "train":
        return X, ys, idx_df, stats
    return X, None, idx_df, stats


def save_model(ckpt, model_path=MODEL_PATH):
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
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


def train_and_save(datasources, model_path=MODEL_PATH):
    t0 = time.time()
    table = resolve_table(datasources)
    freq = table_freq(table)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cpu_threads = max(1, min(8, os.cpu_count() or 1))
    torch.set_num_threads(cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("日志写入文件", path=LOG_PATH)
    logger.info("训练设备", device=str(device), table=table, bar_frequency=freq,
                cpu_threads=cpu_threads, training_profile="ic_30m_es",
                epochs=EPOCHS, seq_len=SEQ_LEN, features=N_FEAT,
                instruments=MAX_TRAIN_INSTRUMENTS, patience=PATIENCE,
                label="close_t->open_t+1", fit_end=FIT_END, val=f"{VAL_START}~{VAL_END}")

    instruments = select_train_instruments(
        pool(FIT_START, VAL_END), MAX_TRAIN_INSTRUMENTS, SEED)
    fit_periods = split_train_periods(FIT_START, FIT_END, TRAIN_PERIOD_CHUNKS)
    logger.info("训练股票与分块就绪", selected=len(instruments),
                fit_chunks=len(fit_periods), first=fit_periods[0], last=fit_periods[-1])

    model = StockTransformer(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(f"trainable parameter count out of range: {n_params}")
    logger.info("可训练参数量", n_params=n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LR * 0.05)
    huber_fn = nn.HuberLoss(delta=0.05)
    model.train()

    # ---- 标准化统计 ----
    stats_buf = []
    for pi in range(min(STATS_CHUNKS, len(fit_periods))):
        ps, pe = fit_periods[pi]
        logger.info("累计标准化统计", chunk=f"{pi + 1}/{STATS_CHUNKS}",
                    period_start=ps, period_end=pe, memory_mb=memory_mb())
        Xtmp, ytmp, _, chunk_stats = build_dataset(
            table, ps, pe, "train", instruments, stats=None, cs_zscore=False)
        stats_buf.append(chunk_stats)
        del Xtmp, ytmp
        gc.collect()
    global_stats = _merge_stats(stats_buf)
    logger.info("标准化统计已确定", features=N_FEAT, from_chunks=len(stats_buf),
                memory_mb=memory_mb())

    # ---- 验证集 (常驻, 30m 半年约可控) ----
    logger.info("构建验证集", start=VAL_START, end=VAL_END)
    Xva, yva, idx_va, _ = build_dataset(
        table, VAL_START, VAL_END, "train", instruments, stats=global_stats, cs_zscore=True)
    logger.info("验证集就绪", samples=len(yva), days=idx_va["date"].nunique(),
                memory_mb=memory_mb())

    best_ic = -1e9
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for ep in range(EPOCHS):
        ep_t = time.time()
        ep_loss, ep_steps = 0.0, 0
        chunk_order = np.random.RandomState(SEED + ep).permutation(len(fit_periods))
        logger.info("训练轮次开始", epoch=f"{ep + 1}/{EPOCHS}", chunks=len(fit_periods),
                    memory_mb=memory_mb())
        for ci, pi in enumerate(chunk_order, 1):
            ps, pe = fit_periods[pi]
            logger.info("构建分块数据", epoch=ep + 1, chunk=f"{ci}/{len(fit_periods)}",
                        period_start=ps, period_end=pe, memory_mb=memory_mb())
            Xtr, ytr, idx_df, _ = build_dataset(
                table, ps, pe, "train", instruments, stats=global_stats, cs_zscore=True)
            chunk_loss, chunk_elapsed = train_one_chunk(
                model, opt, huber_fn, Xtr, ytr, idx_df, device, ep, ci, len(fit_periods),
                SEED, log_first=(ep == 0 and ci == 1))
            ep_loss += chunk_loss
            ep_steps += 1
            logger.info("分块训练完成", epoch=ep + 1, chunk=f"{ci}/{len(fit_periods)}",
                        chunk_loss=round(chunk_loss, 8), elapsed=chunk_elapsed,
                        memory_mb=memory_mb())

        val_ic, val_days = eval_rank_ic(model, Xva, yva, idx_va, device)
        logger.info(
            "epoch 完成",
            epoch=ep + 1,
            train_loss=round(ep_loss / max(ep_steps, 1), 8),
            val_rank_ic=round(val_ic, 6) if np.isfinite(val_ic) else None,
            val_days=val_days,
            elapsed=round(time.time() - ep_t, 2),
            memory_mb=memory_mb(),
        )
        scheduler.step()
        logger.info("学习率更新", epoch=ep + 1, lr=opt.param_groups[0]["lr"])

        if np.isfinite(val_ic) and val_ic > best_ic:
            best_ic = val_ic
            best_epoch = ep + 1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            logger.info("刷新最佳验证IC", best_epoch=best_epoch, best_val_ic=round(best_ic, 6))
        else:
            bad_epochs += 1
            logger.info("验证未提升", bad_epochs=bad_epochs, patience=PATIENCE,
                        best_val_ic=round(best_ic, 6))
            if bad_epochs >= PATIENCE:
                logger.info("Early stopping", stopped_epoch=ep + 1, best_epoch=best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("已恢复最佳权重", best_epoch=best_epoch, best_val_ic=round(best_ic, 6))
    else:
        logger.warning("无有效验证IC, 使用最后一轮权重")

    del Xva, yva, idx_va
    gc.collect()

    mean, std = global_stats
    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "data_source": table,
        "bar_frequency": freq,
        "label": "close_to_next_open",
        "cs_zscore": True,
        "best_epoch": best_epoch,
        "best_val_ic": float(best_ic) if np.isfinite(best_ic) else None,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
    }, model_path)
    logger.info("模型已保存, 请随 notebook 一并上传", path=model_path,
                size_mb=round(os.path.getsize(model_path) / 1024 / 1024, 2),
                total_elapsed=round(time.time() - t0, 2), memory_mb=memory_mb())
    return model_path


if __name__ == "__main__":
    datasources = {"bar30m": "bigalpha_2026_stock_bar30m"}
    train_and_save(datasources)
