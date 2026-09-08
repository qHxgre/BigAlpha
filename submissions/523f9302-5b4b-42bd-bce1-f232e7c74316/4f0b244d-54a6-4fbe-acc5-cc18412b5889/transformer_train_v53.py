"""端到端 Tiny Transformer v53 —— v49 全日 5m + Conv-tokenize→TF（CTTS 风格）。

相对 v49（单因子：局部 CNN tokenize 再注意力）：
  - X / SEQ48 / CE / D-stress / e20 不变；复用 `.cache_v49`
  - Conv1d(k=5,stride=2) 把 T 根 bar 压成 token，再浅 TF（≠ v45 替换 Linear proj）
  - 纵向对 v49

用法:
    python transformer_train_v53.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import dai
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = structlog.get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model_v53.json")
CKPT_PATH = os.path.join(_HERE, "transformer_model_v53.ckpt.pt")

TRAIN_START, TRAIN_END = "2019-01-01", "2023-12-31 23:59:59"
LOCAL_TEST_START, LOCAL_TEST_END = "2024-01-01 00:00:00", "2024-12-31 23:59:59"
LOCAL_EVAL_START, LOCAL_EVAL_END = "2024-01-01", "2024-12-31"

_HALF_DAY = os.environ.get("V8C_5M_HALF", "0") != "0"
SEQ_DAILY = 20
SEQ_INTRA = 24 if _HALF_DAY else 48
EPOCHS, BASE_BATCH, LR, SEED = 20, 512, 3e-4, 42
DROPOUT = 0.1
D_MODEL, NHEAD, N_LAYERS, DIM_FF = 128, 4, 2, 256
CONV_KERNEL, CONV_STRIDE = 5, 2
NUM_CLASSES = 10
MAX_TRAIN_INSTRUMENTS = None
DEFAULT_BAR1M = "bigalpha_2026_stock_bar1m"
DEFAULT_BAR5M = "bigalpha_2026_stock_bar5m"

STRESS_UP_Q, STRESS_DN_Q = 0.90, 0.10
STRESS_UP_WEIGHT, STRESS_DN_WEIGHT = 1.75, 1.40

RAW_DAILY_COLS = [
    "open", "high", "low", "close", "pre_close",
    "volume_log", "amount_log",
]
DAILY_FEATS = [
    "open_log", "high_log", "low_log", "close_log", "pre_close_log",
    "volume_log", "amount_log",
]
RAW_INTRA_COLS = [
    "close", "pre_close", "volume_log", "ask_volume1", "bid_volume1",
]
INTRA_FEATS = [
    "close_log", "pre_close_log", "volume_log", "ask_volume_log", "bid_volume_log",
]
N_DAILY, N_INTRA = len(DAILY_FEATS), len(INTRA_FEATS)

CLASS_WEIGHTS = [12.0, 3.0, 1.5, 1.2, 0.8, 0.8, 1.2, 1.5, 3.0, 12.0]
SCORE_WEIGHTS = [-12.0, -3.0, -1.5, -1.2, -0.8, 0.8, 1.2, 1.5, 3.0, 12.0]

TOWER_NAME = f"5m_{'half24' if _HALF_DAY else 'full48'}_convtok_tf_d128_e20"
VERSION = "v53"


def _n_tokens(seq_len: int) -> int:
    # Conv1d length: floor((L + 2*pad - k) / stride) + 1；pad = k//2
    pad = CONV_KERNEL // 2
    return (seq_len + 2 * pad - CONV_KERNEL) // CONV_STRIDE + 1


MODEL_CFG = dict(
    n_feat=N_INTRA,
    d_model=D_MODEL,
    nhead=NHEAD,
    nlayers=N_LAYERS,
    dim_ff=DIM_FF,
    seq_len=SEQ_INTRA,
    dropout=DROPOUT,
    num_classes=NUM_CLASSES,
)

BATCH = BASE_BATCH


def _resolve_tables(datasources):
    """平台可能只注入 bar1m；日内表固定官方 bar5m（勿回落 bar15m）。"""
    ds = datasources or {}
    return (
        ds.get("bar1m", DEFAULT_BAR1M),
        ds.get("bar5m", DEFAULT_BAR5M),
    )


def _cuda_usable():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(8, device="cuda")
        y = x * 2
        torch.cuda.synchronize()
        del x, y
        return True
    except Exception as e:
        logger.warning("CUDA 探针失败，回退 CPU", err=str(e)[:160])
        return False


def configure_runtime():
    n_cpu = max(1, os.cpu_count() or 4)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cpu))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_cpu))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n_cpu))

    use_cuda = _cuda_usable()
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_name, mem_gb = None, None
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        batch = 4096
        try:
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024 ** 3)
            if mem_gb < 6:
                batch = 1024
            elif mem_gb < 10:
                batch = 2048
            elif mem_gb < 16:
                batch = 4096
            elif mem_gb < 24:
                batch = 6144  # SEQ16 浅塔，中档卡抬 batch
            else:
                batch = 8192
            gpu_name = props.name
        except Exception:
            gpu_name = "cuda"
        seq_scale = max(1, int(round(SEQ_INTRA / 16.0)))
        batch = max(256, int(batch // seq_scale))
        num_workers = 0
        if "TRANSFORMER_NUM_WORKERS" in os.environ:
            num_workers = int(os.environ["TRANSFORMER_NUM_WORKERS"])
        use_amp, pin_memory = True, True
    else:
        torch.set_num_threads(n_cpu)
        if n_cpu >= 32:
            batch = 2048
        elif n_cpu >= 16:
            batch = 1024
        elif n_cpu >= 8:
            batch = 768
        else:
            batch = BASE_BATCH
        seq_scale = max(1, int(round(SEQ_INTRA / 16.0)))
        batch = max(128, int(batch // seq_scale))
        num_workers, use_amp, pin_memory = 0, False, False

    if "TRANSFORMER_BATCH" in os.environ:
        batch = int(os.environ["TRANSFORMER_BATCH"])

    cfg = dict(
        device=device,
        batch=int(batch),
        num_workers=int(num_workers),
        use_amp=bool(use_amp),
        pin_memory=bool(pin_memory),
        n_cpu=int(n_cpu),
        gpu_name=gpu_name,
        gpu_mem_gb=None if mem_gb is None else round(float(mem_gb), 1),
        compile=os.environ.get("V8C_COMPILE", "0") != "0",
    )
    logger.info("运行时配置", **{k: (str(v) if k == "device" else v) for k, v in cfg.items()})
    return cfg


class _ConvTokenTower(nn.Module):
    """CTTS：Conv1d 局部窗→token，再 Transformer 长程依赖。"""

    def __init__(self, n_feat, seq_len, d_model, nhead, nlayers, dim_ff, dropout):
        super().__init__()
        pad = CONV_KERNEL // 2
        self.tokenize = nn.Conv1d(
            n_feat, d_model,
            kernel_size=CONV_KERNEL, stride=CONV_STRIDE, padding=pad,
        )
        n_tok = _n_tokens(seq_len)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)

    def forward(self, x):
        # x: (B, T, F) → Conv over time → (B, n_tok, D)
        h = self.drop(self.act(self.tokenize(x.transpose(1, 2)))).transpose(1, 2)
        t = h.size(1)
        return self.encoder(h + self.pos[:, :t]).mean(dim=1)


class EndToEndTransformerV53(nn.Module):
    """v49 全日/半日 5m + Conv-tokenize→TF；d128；epochs=20。"""

    def __init__(
        self,
        n_feat,
        d_model=128,
        nhead=4,
        nlayers=2,
        dim_ff=256,
        seq_len=SEQ_INTRA,
        dropout=0.1,
        num_classes=10,
    ):
        super().__init__()
        self.tower = _ConvTokenTower(
            n_feat, seq_len, d_model, nhead, nlayers, dim_ff, dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_intra):
        return self.head(self.tower(x_intra))


def pool(sd, ed):
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return df["instrument"].tolist()


def _month_slices(start_dt, end_dt):
    start_dt = pd.to_datetime(start_dt).normalize()
    end_dt = pd.to_datetime(end_dt)
    cur = start_dt.replace(day=1)
    last = end_dt.normalize()
    out = []
    while cur <= last:
        nxt = cur + pd.offsets.MonthBegin(1)
        m_end = min(nxt - pd.Timedelta(seconds=1), end_dt)
        out.append((cur, m_end))
        cur = nxt
    return out


def _intra_time_slices(start_dt, end_dt, chunk=None):
    """5m 拉数分块：默认按季（提高并行度）；V8C_INTRA_CHUNK=year|quarter|month。"""
    chunk = (chunk or os.environ.get("V8C_INTRA_CHUNK", "quarter")).lower().strip()
    start_dt = pd.to_datetime(start_dt).normalize()
    end_dt = pd.to_datetime(end_dt)
    if chunk == "month":
        return _month_slices(start_dt, end_dt)
    if chunk == "year":
        cur = start_dt.replace(month=1, day=1)
        last = end_dt.normalize()
        out = []
        while cur <= last:
            nxt = cur.replace(year=cur.year + 1)
            m_end = min(nxt - pd.Timedelta(seconds=1), end_dt)
            if m_end >= start_dt:
                out.append((max(cur, start_dt), m_end))
            cur = nxt
        return out
    # quarter（默认）
    cur = start_dt.replace(day=1)
    q0 = ((cur.month - 1) // 3) * 3 + 1
    cur = cur.replace(month=q0, day=1)
    last = end_dt.normalize()
    out = []
    while cur <= last:
        nxt = cur + pd.offsets.MonthBegin(3)
        m_end = min(nxt - pd.Timedelta(seconds=1), end_dt)
        if m_end >= start_dt:
            out.append((max(cur, start_dt), m_end))
        cur = nxt
    return out


def _cache_dir():
    return os.path.join(_HERE, ".cache_v49")


def _cache_key(table_1m, table_15m, start_dt, end_dt, n_ins, with_label):
    raw = "|".join(
        [
            f"v49_5m_{'half24' if _HALF_DAY else 'full48'}_v1",
            str(table_1m),
            str(table_15m),
            str(pd.to_datetime(start_dt).date()),
            str(pd.to_datetime(end_dt).date()),
            str(n_ins),
            str(int(bool(with_label))),
            str(SEQ_DAILY),
            str(SEQ_INTRA),
            ",".join(DAILY_FEATS),
            ",".join(INTRA_FEATS),
            f"upq{STRESS_UP_Q}",
            f"dnq{STRESS_DN_Q}",
            f"upw{STRESS_UP_WEIGHT}",
            f"dnw{STRESS_DN_WEIGHT}",
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _intra_parallel_workers():
    """并行拉 5m 线程数；默认随 CPU 抬高，V8C_INTRA_PARALLEL 可覆盖（=1 关闭并行）。"""
    if "V8C_INTRA_PARALLEL" in os.environ:
        return max(1, int(os.environ["V8C_INTRA_PARALLEL"]))
    n_cpu = max(1, os.cpu_count() or 4)
    return int(min(8, max(4, n_cpu // 2)))


def _dai_filters(start_dt, end_dt, instruments=None):
    s = pd.to_datetime(start_dt).strftime("%Y-%m-%d 00:00:00")
    e = pd.to_datetime(end_dt).strftime("%Y-%m-%d 23:59:59")
    filters = {"date": [s, e]}
    if instruments is not None:
        filters["instrument"] = list(instruments)
    return filters


def query_daily_panel(table, start_dt, end_dt, instruments=None):
    """bar1m → 日频（仅日频塔需要的原始列）；instrument 下推到 DAI。"""
    start_dt, end_dt = pd.to_datetime(start_dt), pd.to_datetime(end_dt)
    sql = f"""
        SELECT
            date::DATE::DATETIME AS date,
            instrument,
            first(open ORDER BY date) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close ORDER BY date) AS close,
            first(pre_close ORDER BY date) AS pre_close,
            ln(sum(volume) + 1) AS volume_log,
            ln(sum(amount) + 1) AS amount_log
        FROM {table}
        WHERE ask_price1 > 0 AND bid_price1 > 0
        GROUP BY instrument, date::DATE
        ORDER BY instrument, date
    """
    df = dai.query(
        sql, filters=_dai_filters(start_dt, end_dt, instruments), compression=True,
    ).df()
    if df is None or len(df) == 0:
        raise RuntimeError(f"empty daily: {start_dt} ~ {end_dt}")
    df = df.sort_values(["instrument", "date"]).drop_duplicates(
        ["instrument", "date"], keep="last"
    )
    df = df[["instrument", "date"] + RAW_DAILY_COLS].copy()
    for c in RAW_DAILY_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


_AFTERNOON_SQL_OK = True  # 进程内：午后谓词一旦失败则后续块直接全日，避免双倍 query


def query_bar5m(table, start_dt, end_dt, instruments=None):
    """5m 路径。全日默认不截午后；半日（V8C_5M_HALF）默认 date≥13:00。

    V8C_5M_AFTERNOON 可显式覆盖（1/0）。
    """
    global _AFTERNOON_SQL_OK
    start_dt, end_dt = pd.to_datetime(start_dt), pd.to_datetime(end_dt)
    default_pm = "1" if _HALF_DAY else "0"
    want_pm = os.environ.get("V8C_5M_AFTERNOON", default_pm) != "0"
    filters = _dai_filters(start_dt, end_dt, instruments)

    def _pull(time_pred: str):
        sql = f"""
            SELECT
                date, instrument, close, pre_close,
                ln(volume + 1) AS volume_log,
                ask_volume1, bid_volume1
            FROM {table}
            WHERE ask_price1 > 0 AND bid_price1 > 0
              {time_pred}
            ORDER BY instrument, date
        """
        return dai.query(sql, filters=filters, compression=True).df()

    time_pred = "AND date::TIME >= TIME '13:00:00'" if (want_pm and _AFTERNOON_SQL_OK) else ""
    try:
        df = _pull(time_pred)
    except Exception as e:
        if time_pred:
            _AFTERNOON_SQL_OK = False
            logger.warning("5m 午后 SQL 失败，本进程改全日拉取", err=str(e)[:160])
            df = _pull("")
        else:
            raise
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["instrument", "date"] + RAW_INTRA_COLS)
    df["date"] = pd.to_datetime(df["date"])
    if instruments is not None:
        df = df[df["instrument"].isin(instruments)]
    df = df.sort_values(["instrument", "date"]).drop_duplicates(
        ["instrument", "date"], keep="last"
    )
    df = df[["instrument", "date"] + RAW_INTRA_COLS].copy()
    for c in RAW_INTRA_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def featurize_daily(df):
    """仅单字段 log；不做 x/pre_close 等跨字段变换。"""
    out = df.copy()
    for src, dst in [
        ("open", "open_log"),
        ("high", "high_log"),
        ("low", "low_log"),
        ("close", "close_log"),
        ("pre_close", "pre_close_log"),
    ]:
        out[dst] = np.log(out[src].clip(lower=1e-6))
    out[DAILY_FEATS] = out[DAILY_FEATS].replace([np.inf, -np.inf], np.nan)
    return out


def featurize_intra(df):
    """仅单字段 log / log1p。"""
    out = df.copy()
    out["close_log"] = np.log(out["close"].clip(lower=1e-6))
    out["pre_close_log"] = np.log(out["pre_close"].clip(lower=1e-6))
    out["ask_volume_log"] = np.log1p(out["ask_volume1"].clip(lower=0.0))
    out["bid_volume_log"] = np.log1p(out["bid_volume1"].clip(lower=0.0))
    out[INTRA_FEATS] = out[INTRA_FEATS].replace([np.inf, -np.inf], np.nan)
    return out


def attach_day_labels(daily_df):
    """日频面板：次日收益十分位 label + 当日 day_ret（压力日识别，PIT）。"""
    out = daily_df.copy()
    out["cal_date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["day_ret"] = out["close"] / out["pre_close"] - 1.0
    out["next_return"] = (
        out.groupby("instrument", group_keys=False)["close"].shift(-1) / out["close"] - 1.0
    )
    valid = out["next_return"].notna()
    out.loc[valid, "label"] = (
        out.loc[valid]
        .groupby("cal_date")["next_return"]
        .rank(method="first", pct=True)
        .mul(NUM_CLASSES)
        .clip(upper=NUM_CLASSES - 1e-6)
    )
    return out


def fit_day_stress_weights(daily_df, start_n, end_n):
    """用训练窗内「当日截面均值 day_ret」分位识别齐涨/齐跌日 → 日权重表。"""
    d = daily_df.copy()
    d["cal_date"] = pd.to_datetime(d["cal_date"]).dt.normalize()
    mkt = d.groupby("cal_date")["day_ret"].mean()
    mkt = mkt[(mkt.index >= start_n) & (mkt.index <= end_n)].dropna()
    if len(mkt) < 20:
        raise RuntimeError(f"too few days for stress fit: {len(mkt)}")
    up_th = float(mkt.quantile(STRESS_UP_Q))
    dn_th = float(mkt.quantile(STRESS_DN_Q))
    wmap = {}
    n_up = n_dn = 0
    for dt, v in mkt.items():
        k = pd.Timestamp(dt).normalize()
        if v >= up_th:
            wmap[k] = float(STRESS_UP_WEIGHT)
            n_up += 1
        elif v <= dn_th:
            wmap[k] = float(STRESS_DN_WEIGHT)
            n_dn += 1
        else:
            wmap[k] = 1.0
    logger.info(
        "压力日权已拟合",
        n_days=len(mkt),
        n_up=n_up,
        n_dn=n_dn,
        up_th=round(up_th, 6),
        dn_th=round(dn_th, 6),
        up_w=STRESS_UP_WEIGHT,
        dn_w=STRESS_DN_WEIGHT,
    )
    return wmap, up_th, dn_th


def _scale_arr(X, stats=None):
    if stats is None:
        flat = X.reshape(-1, X.shape[-1])
        mu = flat.mean(axis=0).astype(np.float64)
        sigma = flat.std(axis=0).astype(np.float64)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        stats = (mu.astype(np.float32), sigma.astype(np.float32))
    mu, sigma = stats
    X = (X - mu.reshape(1, 1, -1)) / sigma.reshape(1, 1, -1)
    return np.nan_to_num(X.astype(np.float32), nan=0.0), stats


def _daily_windows_for_days(daily_df, target_days, seq_len, require_label=True):
    """对指定日历日，构造日频窗 (seq_len, F)。"""
    feats = daily_df[DAILY_FEATS].to_numpy(dtype=np.float32, copy=False)
    cals = pd.to_datetime(daily_df["cal_date"]).to_numpy()
    ins = daily_df["instrument"].to_numpy()
    labels = daily_df["label"].to_numpy(dtype=np.float64) if "label" in daily_df.columns else None
    target_set = set(pd.Timestamp(d).normalize() for d in target_days)

    out = {}
    pos = 0
    for _, g in daily_df.groupby("instrument", sort=False):
        n = len(g)
        for i in range(seq_len - 1, n):
            end = pos + i
            d = pd.Timestamp(cals[end]).normalize()
            if d not in target_set:
                continue
            if require_label and labels is not None and not np.isfinite(labels[end]):
                continue
            start = end - seq_len + 1
            key = (d, ins[end])
            lab = int(labels[end]) if (labels is not None and np.isfinite(labels[end])) else None
            out[key] = (feats[start : end + 1].copy(), lab)
        pos += n
    return out


def _intra_windows(intra_df, seq_len, with_label_keys=None, allow_pad=False):
    """当日日内窗：取该日历日最后 seq_len 根（全日≈48 / 半日≈24）。

    allow_pad=True（推理）：不足 seq_len 时左侧用首根 bar 填充，避免硬丢样本导致覆盖度挂掉。
    训练仍要求 length >= seq_len。
    """
    dates = pd.to_datetime(intra_df["date"])
    cals = dates.dt.normalize().to_numpy()
    feats = intra_df[INTRA_FEATS].to_numpy(dtype=np.float32, copy=False)
    instruments = intra_df["instrument"].to_numpy()
    out = {}
    n = len(intra_df)
    if n == 0:
        return out
    i = 0
    while i < n:
        ins_i = instruments[i]
        cal_i = cals[i]
        j = i + 1
        while j < n and instruments[j] == ins_i and cals[j] == cal_i:
            j += 1
        length = j - i
        key = (pd.Timestamp(cal_i).normalize(), ins_i)
        if with_label_keys is not None and key not in with_label_keys:
            i = j
            continue
        if length >= seq_len:
            out[key] = feats[j - seq_len : j].copy()
        elif allow_pad and length > 0:
            chunk = feats[i:j]
            pad_n = seq_len - length
            out[key] = np.concatenate(
                [np.repeat(chunk[:1], pad_n, axis=0), chunk], axis=0
            ).astype(np.float32, copy=False)
        i = j
    return out


def collect_aligned_month_tensors(
    table_1m,
    table_15m,
    start_dt,
    end_dt,
    instruments,
    daily_stats=None,
    intra_stats=None,
    with_label=True,
):
    """日频一次拉齐 + 5m 按年（可配）分块；日窗只建一次；标签缓冲只加在日频上。"""
    return collect_aligned_tensors(
        table_1m,
        table_15m,
        start_dt,
        end_dt,
        instruments,
        daily_stats=daily_stats,
        intra_stats=intra_stats,
        with_label=with_label,
    )


def collect_aligned_tensors(
    table_1m,
    table_15m,
    start_dt,
    end_dt,
    instruments,
    daily_stats=None,
    intra_stats=None,
    with_label=True,
):
    """对齐建样本（加速版）。"""
    start_n = pd.to_datetime(start_dt).normalize()
    end_n = pd.to_datetime(end_dt).normalize()
    use_cache = os.environ.get("V8C_CACHE", "1") != "0"
    # 有外部传入 stats 时不走整包缓存（续训场景由上层管）
    cache_ok = use_cache and daily_stats is None and intra_stats is None
    cache_path = None
    if cache_ok:
        os.makedirs(_cache_dir(), exist_ok=True)
        ck = _cache_key(table_1m, table_15m, start_n, end_n, len(instruments), with_label)
        cache_path = os.path.join(_cache_dir(), f"{ck}.npz")
        if os.path.exists(cache_path):
            logger.info("命中本地对齐缓存", path=cache_path)
            blob = np.load(cache_path, allow_pickle=True)
            Xd = blob["Xd"]
            Xi = blob["Xi"]
            y = blob["y"] if with_label and "y" in blob.files else None
            w = blob["w"].astype(np.float32) if with_label and "w" in blob.files else None
            metas = blob["metas"].tolist()
            d_stats = (blob["d_mu"].astype(np.float32), blob["d_std"].astype(np.float32))
            i_stats = (blob["i_mu"].astype(np.float32), blob["i_std"].astype(np.float32))
            stress_meta = None
            if with_label and "stress_up_th" in blob.files:
                stress_meta = (
                    float(blob["stress_up_th"]),
                    float(blob["stress_dn_th"]),
                )
            return Xd, Xi, y, metas, d_stats, i_stats, w, stress_meta

    Xd_all, Xi_all, y_all, w_all, metas = [], [], [], [], []
    d_stats, i_stats = daily_stats, intra_stats
    stress_meta = None
    stress_wmap = {}

    slices = _intra_time_slices(start_dt, end_dt)
    n_par = _intra_parallel_workers()
    logger.info(
        "5m 分块计划",
        n_chunks=len(slices),
        chunk=os.environ.get("V8C_INTRA_CHUNK", "quarter"),
        parallel=n_par,
        afternoon_only=os.environ.get(
            "V8C_5M_AFTERNOON", "1" if _HALF_DAY else "0"
        ) != "0",
        half_day=_HALF_DAY,
        seq_intra=SEQ_INTRA,
    )

    # 先提交 5m 并行拉取，与日频构建重叠，缩短首次 wall time
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        futs = [
            ex.submit(query_bar5m, table_15m, ms, me, instruments)
            for ms, me in slices
        ]

        daily_buf = start_n - pd.Timedelta(days=2 * SEQ_DAILY + 40)
        daily_pull_end = end_n + pd.Timedelta(days=12 if with_label else 0)
        t_daily0 = time.time()
        logger.info(
            "拉日频面板",
            start=str(daily_buf.date()),
            end=str(daily_pull_end.date()),
            table=table_1m,
            n_ins=len(instruments),
        )
        daily = query_daily_panel(table_1m, daily_buf, daily_pull_end, instruments)
        daily = featurize_daily(daily).dropna(subset=DAILY_FEATS)
        daily = attach_day_labels(daily)
        daily["cal_date"] = pd.to_datetime(daily["date"]).dt.normalize()
        logger.info("日频就绪", rows=len(daily), elapsed=round(time.time() - t_daily0, 1))

        if with_label:
            stress_wmap, up_th, dn_th = fit_day_stress_weights(daily, start_n, end_n)
            stress_meta = (up_th, dn_th)

        t_win0 = time.time()
        target_days = pd.date_range(start_n, end_n, freq="D")
        daily_map = _daily_windows_for_days(
            daily, target_days, SEQ_DAILY, require_label=with_label
        )
        if with_label:
            daily_map = {k: v for k, v in daily_map.items() if v[1] is not None}
        daily_map = {k: v for k, v in daily_map.items() if start_n <= k[0] <= end_n}
        del daily
        logger.info("日窗一次构建完成", n=len(daily_map), elapsed=round(time.time() - t_win0, 1))
        if not daily_map:
            raise RuntimeError(f"empty daily windows: {start_dt} ~ {end_dt}")

        daily_keys = set(daily_map.keys())

        # 按时间序消费 future（保证 scaler 统计量与串行版一致）
        for (ms, me), fut in zip(slices, futs):
            t0 = time.time()
            logger.info("等待/对齐5m块", start=str(ms.date()), end=str(me.date()), table=table_15m)
            try:
                intra = fut.result()
            except Exception as e:
                logger.warning("5m 块失败，跳过", start=str(ms.date()), err=str(e)[:160])
                continue
            if intra is None or len(intra) == 0:
                continue
            intra = featurize_intra(intra).dropna(subset=INTRA_FEATS)
            t0b, t1b = ms.normalize(), min(end_n, pd.to_datetime(me).normalize())
            chunk_keys = {k for k in daily_keys if t0b <= k[0] <= t1b}
            # 推理：短序列可 pad；无日内样本交给 predict 中位填
            intra_map = _intra_windows(
                intra,
                SEQ_INTRA,
                with_label_keys=chunk_keys,
                allow_pad=not with_label,
            )
            del intra

            keys = sorted(chunk_keys & set(intra_map.keys()))
            if not keys:
                continue

            Xd = np.stack([daily_map[k][0] for k in keys], axis=0)
            Xi = np.stack([intra_map[k] for k in keys], axis=0)
            y = (
                np.asarray([daily_map[k][1] for k in keys], dtype=np.int64)
                if with_label
                else None
            )

            if d_stats is None:
                Xd, d_stats = _scale_arr(Xd, None)
            else:
                Xd, _ = _scale_arr(Xd, d_stats)
            if i_stats is None:
                Xi, i_stats = _scale_arr(Xi, None)
            else:
                Xi, _ = _scale_arr(Xi, i_stats)

            Xd_all.append(Xd)
            Xi_all.append(Xi)
            if with_label:
                y_all.append(y)
                ww = np.asarray(
                    [float(stress_wmap.get(pd.Timestamp(k[0]).normalize(), 1.0)) for k in keys],
                    dtype=np.float32,
                )
                w_all.append(ww)
            metas.extend(keys)
            logger.info(
                "块对齐完成",
                n=len(keys),
                total=sum(a.shape[0] for a in Xd_all),
                elapsed=round(time.time() - t0, 1),
            )

    if not Xd_all:
        raise RuntimeError(f"empty aligned dataset: {start_dt} ~ {end_dt}")
    Xd = np.concatenate(Xd_all, axis=0)
    Xi = np.concatenate(Xi_all, axis=0)
    y = np.concatenate(y_all, axis=0) if with_label else None
    w = np.concatenate(w_all, axis=0) if with_label else None

    if cache_ok and cache_path is not None:
        payload = dict(
            Xd=Xd,
            Xi=Xi,
            metas=np.array(metas, dtype=object),
            d_mu=d_stats[0],
            d_std=d_stats[1],
            i_mu=i_stats[0],
            i_std=i_stats[1],
        )
        if with_label:
            payload["y"] = y
            payload["w"] = w
            if stress_meta is not None:
                payload["stress_up_th"] = np.float32(stress_meta[0])
                payload["stress_dn_th"] = np.float32(stress_meta[1])
        tmp = cache_path + ".tmp.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, cache_path)
        logger.info("已写本地对齐缓存", path=cache_path)

    return Xd, Xi, y, metas, d_stats, i_stats, w, stress_meta


def build_infer_tensors(
    table_1m, table_15m, start_date, end_date, daily_stats, intra_stats, instruments=None,
):
    """返回 (Xi, idx_df)；日频仅参与对齐/标签管线，不进模型。"""
    if instruments is None:
        instruments = pool(start_date, end_date)
    _, Xi, _, meta, _, _, _, _ = collect_aligned_month_tensors(
        table_1m,
        table_15m,
        start_date,
        end_date,
        instruments=instruments,
        daily_stats=daily_stats,
        intra_stats=intra_stats,
        with_label=False,
    )
    idx = pd.DataFrame(meta, columns=["date", "instrument"])
    return Xi, idx


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


def save_checkpoint(path, *, epoch, model, opt, scaler, daily_stats, intra_stats, model_cfg, train_meta):
    d_mu, d_std = daily_stats
    i_mu, i_std = intra_stats
    payload = {
        "version": VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "model_cfg": model_cfg,
        "daily_mean": np.asarray(d_mu, np.float32),
        "daily_std": np.asarray(d_std, np.float32),
        "intra_mean": np.asarray(i_mu, np.float32),
        "intra_std": np.asarray(i_std, np.float32),
        "train_meta": train_meta,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def train_and_save(datasources, model_path=MODEL_PATH, resume=True, ckpt_path=CKPT_PATH):
    """训练 → transformer_model_v53.json（v49 + Conv-tokenize→TF）。"""
    global BATCH
    table_1m, table_15m = _resolve_tables(datasources)
    rt = configure_runtime()
    device = rt["device"]
    BATCH = rt["batch"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    start_epoch = 0
    daily_stats = intra_stats = None
    resumed = False

    if resume and os.path.exists(ckpt_path):
        logger.info("发现断点，准备续跑", path=ckpt_path)
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        if ckpt.get("version") != VERSION or ckpt.get("model_cfg") != MODEL_CFG:
            logger.warning("断点与当前 MODEL_CFG/version 不一致，忽略断点从头训练")
        else:
            start_epoch = int(ckpt["epoch"])
            daily_stats = (
                np.asarray(ckpt["daily_mean"], np.float32),
                np.asarray(ckpt["daily_std"], np.float32),
            )
            intra_stats = (
                np.asarray(ckpt["intra_mean"], np.float32),
                np.asarray(ckpt["intra_std"], np.float32),
            )
            resumed = True
            try:
                torch.set_rng_state(ckpt["torch_rng"])
                np.random.set_state(ckpt["numpy_rng"])
                if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
                    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            except Exception as e:
                logger.warning("RNG 恢复失败", err=str(e)[:120])
            logger.info("断点已加载", finished_epoch=start_epoch, next_epoch=start_epoch + 1)

    if start_epoch >= EPOCHS and os.path.exists(model_path):
        logger.info("训练已完成且 JSON 已存在，跳过", path=model_path)
        return model_path

    logger.info(
        "窗口协议",
        train=f"{TRAIN_START} ~ {TRAIN_END}",
        tower=TOWER_NAME,
        stress_up_q=STRESS_UP_Q,
        stress_dn_q=STRESS_DN_Q,
        stress_up_w=STRESS_UP_WEIGHT,
        stress_dn_w=STRESS_DN_WEIGHT,
        intra_feats=INTRA_FEATS,
        seq_intra=SEQ_INTRA,
        d_model=D_MODEL,
        table_1m=table_1m,
        table_15m=table_15m,
    )
    instruments = pool(TRAIN_START, TRAIN_END)
    if MAX_TRAIN_INSTRUMENTS:
        instruments = instruments[:MAX_TRAIN_INSTRUMENTS]
    logger.info("构建对齐训练集", n_ins=len(instruments))

    # 日频：标签 + 压力日权；模型只用 Xi
    _Xd, Xi, ytr, _, daily_stats, intra_stats, wtr, stress_meta = collect_aligned_month_tensors(
        table_1m,
        table_15m,
        TRAIN_START,
        TRAIN_END,
        instruments=instruments,
        daily_stats=daily_stats,
        intra_stats=intra_stats,
        with_label=True,
    )
    del _Xd
    gc.collect()
    _, cnts = np.unique(ytr, return_counts=True)
    logger.info(
        "训练样本",
        n=int(len(ytr)),
        xi=list(Xi.shape),
        label_hist=cnts.tolist(),
        w_mean=round(float(np.mean(wtr)), 4),
        w_max=round(float(np.max(wtr)), 4),
        n_up_w=int(np.sum(np.isclose(wtr, STRESS_UP_WEIGHT))),
        n_dn_w=int(np.sum(np.isclose(wtr, STRESS_DN_WEIGHT))),
    )

    model = EndToEndTransformerV53(**MODEL_CFG)
    try:
        model = model.to(device)
    except RuntimeError as e:
        logger.warning("model.to 失败，回退 CPU", err=str(e)[:160])
        device = torch.device("cpu")
        rt = {**rt, "device": device, "use_amp": False, "pin_memory": False, "num_workers": 0}
        BATCH = 512
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 100_000 <= n_params <= 100_000_000, f"param count out of range: {n_params}"
    logger.info("可训练参数量", n_params=n_params, device=str(device), batch=BATCH)

    if rt.get("compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("已启用 torch.compile", mode="reduce-overhead")
        except Exception as e:
            logger.warning("torch.compile 失败，继续 eager", err=str(e)[:160])

    loader_kw = dict(
        batch_size=BATCH,
        shuffle=True,
        pin_memory=rt["pin_memory"],
        num_workers=rt["num_workers"],
    )
    if rt["num_workers"] > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xi),
            torch.from_numpy(ytr),
            torch.from_numpy(np.asarray(wtr, dtype=np.float32)),
        ),
        **loader_kw,
    )
    try:
        opt = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=1e-4, fused=(device.type == "cuda"),
        )
    except TypeError:
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w, reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=rt["use_amp"])

    if resumed:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        try:
            opt.load_state_dict(ckpt["opt_state"])
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        except Exception as e:
            logger.warning("optimizer 加载失败", err=str(e)[:120])
        if rt["use_amp"] and ckpt.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                logger.warning("AMP scaler 加载失败", err=str(e)[:120])

    train_meta = {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "seq_intra": SEQ_INTRA,
        "intra_freq": "5m",
        "tower": TOWER_NAME,
        "stress_up_q": STRESS_UP_Q,
        "stress_dn_q": STRESS_DN_Q,
        "stress_up_weight": STRESS_UP_WEIGHT,
        "stress_dn_weight": STRESS_DN_WEIGHT,
        "stress_up_th": None if stress_meta is None else float(stress_meta[0]),
        "stress_dn_th": None if stress_meta is None else float(stress_meta[1]),
        "n_samples": int(len(ytr)),
        "n_instruments": len(instruments),
    }

    if start_epoch < EPOCHS:
        model.train()
        for ep in range(start_epoch, EPOCHS):
            t0, tot, nb = time.time(), 0.0, 0
            for xi, yb, wb in loader:
                xi = xi.to(device, non_blocking=rt["pin_memory"])
                yb = yb.to(device, non_blocking=rt["pin_memory"])
                wb = wb.to(device, non_blocking=rt["pin_memory"])
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=rt["use_amp"]):
                    per = loss_fn(model(xi), yb)
                    loss = (per * wb).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += float(loss.item())
                nb += 1
            finished = ep + 1
            logger.info(
                "epoch 完成",
                epoch=finished,
                ce=round(tot / max(nb, 1), 8),
                elapsed=round(time.time() - t0, 2),
            )
            if finished % 2 == 0 or finished >= EPOCHS:
                save_checkpoint(
                    ckpt_path,
                    epoch=finished,
                    model=model,
                    opt=opt,
                    scaler=scaler if rt["use_amp"] else None,
                    daily_stats=daily_stats,
                    intra_stats=intra_stats,
                    model_cfg=MODEL_CFG,
                    train_meta=train_meta,
                )

    d_mu, d_std = daily_stats
    i_mu, i_std = intra_stats
    save_model(
        {
            "state_dict": model.state_dict(),
            "model_cfg": MODEL_CFG,
            "intra_feats": INTRA_FEATS,
            "seq_intra": SEQ_INTRA,
            "intra_freq": "5m",
            "tower": TOWER_NAME,
            "mean": np.asarray(i_mu, np.float32).tolist(),
            "std": np.asarray(i_std, np.float32).tolist(),
            "daily_mean": np.asarray(d_mu, np.float32).tolist(),
            "daily_std": np.asarray(d_std, np.float32).tolist(),
            "intra_mean": np.asarray(i_mu, np.float32).tolist(),
            "intra_std": np.asarray(i_std, np.float32).tolist(),
            "feature_cols": [f"5m::{c}" for c in INTRA_FEATS],
            "seq_len": SEQ_INTRA,
            "version": VERSION,
            "objective": "weighted_ce_decile_tail_dstress",
            "stress_up_q": STRESS_UP_Q,
            "stress_dn_q": STRESS_DN_Q,
            "stress_up_weight": STRESS_UP_WEIGHT,
            "stress_dn_weight": STRESS_DN_WEIGHT,
            "stress_up_th": None if stress_meta is None else float(stress_meta[0]),
            "stress_dn_th": None if stress_meta is None else float(stress_meta[1]),
            "class_weights": CLASS_WEIGHTS,
            "score_weights": SCORE_WEIGHTS,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "local_test_start": LOCAL_TEST_START,
            "local_test_end": LOCAL_TEST_END,
        },
        model_path,
    )
    logger.info("模型已保存", path=model_path)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Transformer v53 5m Conv-tokenize→TF")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--reset-ckpt", action="store_true")
    parser.add_argument(
        "--half-day", action="store_true",
        help="半日窗 SEQ=24（等同 V8C_5M_HALF=1；须在启动时设环境变量更稳）",
    )
    args = parser.parse_args()
    if args.half_day and not _HALF_DAY:
        logger.warning(
            "请用环境变量启动半日窗: V8C_5M_HALF=1 python transformer_train_v53.py"
        )
    if args.reset_ckpt and os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
        logger.info("已删除断点", path=CKPT_PATH)
    train_and_save(
        {"bar1m": DEFAULT_BAR1M, "bar5m": DEFAULT_BAR5M},
        resume=not args.no_resume,
    )
