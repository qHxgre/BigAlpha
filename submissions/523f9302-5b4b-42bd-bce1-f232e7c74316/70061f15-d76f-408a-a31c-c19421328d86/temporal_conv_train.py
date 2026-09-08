# -*- coding: utf-8 -*-
"""M3 pack=full2 TemporalConv(maxpool)+full readout+GRU — public submission (4-seed).

Public: predict.ipynb loads model_seed{42,7,123,2024}.json and averages scores.
Private: train_and_save(...) retrains all 4 seeds on cloud and writes the JSONs.

Config: pack_mode=full2 post_pack + temporal_conv_maxpool_readout_full_gru + soft_rankic;
        FEATURES=25 (LOB3+Kline+deal_number+num_orders);
        云端表列名直接用 deal_number（非 num_trades）；
        LR=2e-4 + ReduceLROnPlateau; patience=20;
        train 2023-01-01~2024-09-30, val 2024-10-01~2024-12-31 (purge=5).
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Cloud IO helpers
# ---------------------------------------------------------------------------
_BOOK_PREFIXES = (
    "ask_price",
    "bid_price",
    "ask_volume",
    "bid_volume",
    "ask_num_orders",
    "bid_num_orders",
)


def to_canonical(df, *, is_local: bool = False, id_to_code=None):
    out = df.copy()
    if is_local:
        raise RuntimeError("submission bundle supports cloud canonicalization only")
    # 兼容：若偶发出现 num_trades，统一成 deal_number（与本地/JSON 一致）
    if "deal_number" not in out.columns and "num_trades" in out.columns:
        out = out.rename(columns={"num_trades": "deal_number"})
    drop_cols = [
        c
        for c in out.columns
        if any(
            c.startswith(p) and len(c) > len(p) and c[len(p) :].isdigit()
            and int(c[len(p) :]) >= 4
            for p in _BOOK_PREFIXES
        )
    ]
    return out.drop(columns=sorted(set(drop_cols)), errors="ignore")


def get_dai():
    try:
        import dai
    except ImportError:
        from bigquant import dai
    return dai


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLOUD_TABLE = "bigalpha_2026_stock_bar1m"
_HERE = Path(__file__).resolve().parent
SEEDS = (42, 7, 123, 2024)


def model_path_for_seed(seed: int, root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else _HERE
    return base / f"model_seed{int(seed)}.json"


def list_model_paths(root: Path | str | None = None) -> list[Path]:
    return [model_path_for_seed(s, root) for s in SEEDS]


# 兼容旧单文件命名（一般不用）
MODEL_PATH = str(model_path_for_seed(SEEDS[0]))

SUBMIT_TRAIN_START = "2023-01-01"
SUBMIT_TRAIN_END = "2024-09-30"
SUBMIT_VAL_START = "2024-10-01"
SUBMIT_VAL_END = "2024-12-31"
PURGE_DAYS = 5

PACK_MODE = "full2"
PACK_SLICES = ((0, 0, 240), (1, 0, 240))
FEAT_NORM = "post_pack"
T_DAY = 240  # raw intraday bars per calendar day
N_DAY = 2  # lookback days for packing
T_INTRADAY = 480  # packed length fed to model (n_day_model=1)
T_COMPRESSED = 240
CONV_OUT = 128
LABEL_OPEN_START_LAG = 1
LABEL_OPEN_END_LAG = 2
MAD_THRESH = 5.0
MISSING_BAR_FRAC = 0.10

LOB_COLS = [
    "ask_price1", "ask_volume1", "bid_price1", "bid_volume1",
    "ask_price2", "ask_volume2", "bid_price2", "bid_volume2",
    "ask_price3", "ask_volume3", "bid_price3", "bid_volume3",
]
KLINE_COLS = ["open", "high", "low", "close", "volume", "amount"]
EXTRA_COLS = [
    "deal_number",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
FEATURE_COLS = LOB_COLS + KLINE_COLS + EXTRA_COLS
N_FEAT = len(FEATURE_COLS)  # 25
VOL_COLS = [
    "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "deal_number",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]

CONV_MID = 64
CONV_KERNEL = 5
GRU_HIDDEN = 192
GRU_LAYERS = 3
GRU_DROPOUT = 0.10
HEAD_DROPOUT = 0.10
HEAD_HIDDEN = 64
DOWNSAMPLE_MODE = "maxpool"
READOUT_KIND = "last_mean_max_attn"
MODEL_TYPE = "temporal_conv_maxpool_readout_full_gru"
MODEL_CFG = dict(
    n_feat=N_FEAT,
    n_day=1,
    t_intraday=T_INTRADAY,
    t_compressed=T_COMPRESSED,
    conv_out=CONV_OUT,
    conv_mid=CONV_MID,
    conv_kernel=CONV_KERNEL,
    encoder_kind="plain",
    downsample_mode=DOWNSAMPLE_MODE,
    readout_kind=READOUT_KIND,
    bidirectional=False,
    gru_hidden=GRU_HIDDEN,
    gru_layers=GRU_LAYERS,
    gru_dropout=GRU_DROPOUT,
    head_dropout=HEAD_DROPOUT,
    head_hidden=HEAD_HIDDEN,
    head_kind="shallow",
)

LOSS_TYPE = "soft_rankic"
VAL_METRIC = "rankic"
SOFT_RANK_TAU = 1.0
EPOCHS = 100
EARLY_STOP_PATIENCE = 20
LR = 2e-4
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 5
PLATEAU_MIN_LR = 1e-6
GRAD_CLIP = 1.0
SEED = 42
USE_AMP = True
MIN_N = 5


# ---------------------------------------------------------------------------
# Model: TemporalConv (maxpool 240→120) + GRU + full readout
# ---------------------------------------------------------------------------
class TemporalConvEncoder(nn.Module):
    """单日时间卷积：Conv s=1 + MaxPool → 120 → CONV_OUT（M3）。"""

    def __init__(
        self,
        n_feat: int = N_FEAT,
        out_dim: int = CONV_OUT,
        t_compressed: int = T_COMPRESSED,
        t_intraday: int = T_INTRADAY,
        kernel_size: int = CONV_KERNEL,
        mid_channels: int = CONV_MID,
        downsample_mode: str = DOWNSAMPLE_MODE,
    ) -> None:
        super().__init__()
        del downsample_mode  # 提交包固定 maxpool
        self.t_intraday = int(t_intraday)
        self.t_compressed = int(t_compressed)
        k = int(kernel_size)
        pad = k // 2
        mid = int(mid_channels)
        self.out_dim = out_dim
        self.stem = nn.Sequential(
            nn.Conv1d(n_feat, mid, kernel_size=k, stride=1, padding=pad),
            nn.GELU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj = nn.Sequential(
            nn.Conv1d(mid, out_dim, kernel_size=k, stride=1, padding=pad),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] → [B, T', D]
        h = self.stem(x.transpose(1, 2))
        h = self.pool(h)
        h = self.proj(h)
        return h.transpose(1, 2)


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h), dim=1)
        return (h * w).sum(dim=1)


class TemporalConvGRUModel(nn.Module):
    """x: [N, n_day*240, F] → maxpool TemporalConv → GRU → full readout → score [N]."""

    def __init__(
        self,
        n_feat: int = N_FEAT,
        n_day: int = 1,
        t_intraday: int = T_INTRADAY,
        t_compressed: int = T_COMPRESSED,
        conv_out: int = CONV_OUT,
        conv_mid: int = CONV_MID,
        conv_kernel: int = CONV_KERNEL,
        encoder_kind: str = "plain",
        downsample_mode: str = DOWNSAMPLE_MODE,
        readout_kind: str = READOUT_KIND,
        bidirectional: bool = False,
        gru_hidden: int = GRU_HIDDEN,
        gru_layers: int = GRU_LAYERS,
        gru_dropout: float = GRU_DROPOUT,
        head_dropout: float = HEAD_DROPOUT,
        head_hidden: int = HEAD_HIDDEN,
        head_kind: str = "shallow",
    ) -> None:
        super().__init__()
        del encoder_kind, head_kind  # 提交包固定 plain + shallow
        self.n_day = max(1, int(n_day))
        self.t_intraday = int(t_intraday)
        self.t_compressed = int(t_compressed)
        self.readout_kind = (readout_kind or READOUT_KIND).lower()
        self.bidirectional = bool(bidirectional)
        self.conv = TemporalConvEncoder(
            n_feat=n_feat,
            out_dim=conv_out,
            t_compressed=self.t_compressed,
            t_intraday=self.t_intraday,
            kernel_size=conv_kernel,
            mid_channels=conv_mid,
            downsample_mode=downsample_mode,
        )
        self.gru = nn.GRU(
            input_size=conv_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        h_out = int(gru_hidden) * (2 if self.bidirectional else 1)
        self.time_attn = TemporalAttentionPool(h_out)
        # full readout: last + mean + max + attn → 4H
        head_in = h_out * 4
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def _encode_days(self, x: torch.Tensor) -> torch.Tensor:
        n, t_total, f = x.shape
        expect = self.n_day * self.t_intraday
        if t_total != expect:
            raise ValueError(f"期望 T={expect} (n_day={self.n_day})，收到 T={t_total}")
        flat = x.reshape(n * self.n_day, self.t_intraday, f)
        enc = self.conv(flat)
        if enc.shape[1] != self.t_compressed:
            raise ValueError(f"卷积输出长度 {enc.shape[1]} != {self.t_compressed}")
        return enc.reshape(n, self.n_day * self.t_compressed, -1)

    def _readout(self, out: torch.Tensor) -> torch.Tensor:
        parts = [
            out[:, -1, :],
            out.mean(dim=1),
            out.max(dim=1).values,
            self.time_attn(out),
        ]
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, industry: torch.Tensor | None = None) -> torch.Tensor:
        del industry
        h = self._encode_days(x.float())
        out, _ = self.gru(h)
        return self.head(self._readout(out)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum()) + eps
    return (a * b).sum() / denom


def _hard_rank(y: torch.Tensor) -> torch.Tensor:
    y = y.detach()
    order = torch.argsort(y, stable=True)
    ranks = torch.empty(y.shape[0], device=y.device, dtype=torch.float32)
    ranks[order] = torch.arange(1, y.shape[0] + 1, device=y.device, dtype=torch.float32)
    return ranks


def soft_rank(scores: torch.Tensor, tau: float) -> torch.Tensor:
    diff = (scores.unsqueeze(0) - scores.unsqueeze(1)) / max(float(tau), 1e-8)
    return torch.sigmoid(diff).sum(dim=0) + 0.5


def soft_rankic_loss(pred: torch.Tensor, y: torch.Tensor, *, tau: float = 1.0) -> torch.Tensor:
    p = pred.float().reshape(-1)
    t = y.float().reshape(-1)
    return -_pearson_corr(soft_rank(p, tau), _hard_rank(t))


def compute_train_loss(pred: torch.Tensor, y: torch.Tensor, loss_type: str = LOSS_TYPE) -> torch.Tensor:
    lt = loss_type.lower()
    if lt == "mse":
        return F.mse_loss(pred.float(), y.float())
    if lt == "soft_rankic":
        return soft_rankic_loss(pred, y, tau=SOFT_RANK_TAU)
    if lt == "ic":
        return -_pearson_corr(pred.float().reshape(-1), y.float().reshape(-1))
    raise ValueError(f"unsupported loss_type={loss_type}")


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
def mad_clip(x: np.ndarray, thresh: float = MAD_THRESH) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64, copy=False)
    med = np.nanmedian(flat, axis=0)
    mad = np.nanmedian(np.abs(flat - med), axis=0)
    mad = np.where(mad < 1e-12, 1.0, mad)
    out = np.clip(flat, med - thresh * mad, med + thresh * mad)
    return out.reshape(x.shape).astype(np.float32, copy=False)


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64, copy=False)
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0) + eps
    return ((flat - mean) / std).reshape(x.shape).astype(np.float32, copy=False)


def mad_clip_1d(y: np.ndarray, thresh: float = MAD_THRESH) -> np.ndarray:
    y = y.astype(np.float64, copy=False)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if mad < 1e-12:
        mad = 1.0
    return np.clip(y, med - thresh * mad, med + thresh * mad).astype(np.float32)


def zscore_1d(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = y.astype(np.float64, copy=False)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y)) + eps
    return ((y - mean) / std).astype(np.float32)


def ffill_then_zero(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    n, t, f = out.shape
    flat = out.transpose(0, 2, 1).reshape(n * f, t)
    bad = ~np.isfinite(flat)
    if bad.any():
        flat = flat.astype(np.float64, copy=True)
        flat[bad] = np.nan
        # copy=True：pandas 3 的 to_numpy 会返回只读视图，下一行原地赋值会报错
        flat = pd.DataFrame(flat).ffill(axis=1).to_numpy(dtype=np.float32, copy=True)
        flat[~np.isfinite(flat)] = 0.0
        out = flat.reshape(n, f, t).transpose(0, 2, 1)
    return out.astype(np.float32, copy=False)


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in VOL_COLS:
        if col in out.columns:
            out[col] = np.log1p(out[col].clip(lower=0).astype("float64"))
    return out


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------
class TradingCalendar:
    def __init__(self, days: list[str]) -> None:
        self.days = list(days)
        self._idx = {d: i for i, d in enumerate(self.days)}

    @classmethod
    def from_cloud(cls, table: str, start: str, end: str) -> "TradingCalendar":
        dai = get_dai()
        df = dai.query(
            f"SELECT DISTINCT date::DATE AS date FROM {table} ORDER BY date",
            filters={"date": [start, end]},
        ).df()
        days = sorted(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist())
        return cls(days)

    def index_of(self, day: str) -> int | None:
        return self._idx.get(day)

    def shift(self, day: str, k: int) -> str | None:
        i = self._idx.get(day)
        if i is None:
            return None
        j = i + k
        if j < 0 or j >= len(self.days):
            return None
        return self.days[j]

    def window(self, day: str, n_back: int) -> list[str] | None:
        i = self._idx.get(day)
        if i is None or i - n_back + 1 < 0:
            return None
        return self.days[i - n_back + 1 : i + 1]

    def days_between(self, start: str, end: str) -> list[str]:
        return [d for d in self.days if start <= d <= end]


def valid_sample_days(
    calendar: TradingCalendar,
    start: str,
    end: str,
    *,
    n_day: int = N_DAY,
    label_end_lag: int = LABEL_OPEN_END_LAG,
    mode: str = "train",
) -> list[str]:
    out: list[str] = []
    for day in calendar.days_between(start, end):
        if calendar.window(day, n_day) is None:
            continue
        if mode != "infer" and calendar.shift(day, label_end_lag) is None:
            continue
        out.append(day)
    return out


def apply_boundary_purge(
    days: list[str],
    next_block_first: str | None,
    calendar: TradingCalendar,
    purge: int = PURGE_DAYS,
) -> list[str]:
    if next_block_first is None or purge <= 0:
        return list(days)
    idx = calendar.index_of(next_block_first)
    if idx is None:
        return list(days)
    banned = set(calendar.days[max(0, idx - purge) : idx])
    return [d for d in days if d not in banned]


def build_submit_split_days(
    calendar: TradingCalendar,
    *,
    n_day: int = N_DAY,
    purge: int = PURGE_DAYS,
) -> tuple[list[str], list[str]]:
    train = valid_sample_days(calendar, SUBMIT_TRAIN_START, SUBMIT_TRAIN_END, n_day=n_day)
    val = valid_sample_days(calendar, SUBMIT_VAL_START, SUBMIT_VAL_END, n_day=n_day)
    train = apply_boundary_purge(train, val[0] if val else None, calendar, purge)
    return train, val


# ---------------------------------------------------------------------------
# Day panel + cloud load
# ---------------------------------------------------------------------------
@dataclass
class DayPanel:
    instruments: np.ndarray
    x: np.ndarray


def _pivot_day_features(df: pd.DataFrame, require_bars: int = T_DAY) -> tuple[np.ndarray, np.ndarray]:
    ins_col = "instrument"
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(feat_cols) != N_FEAT:
        missing = set(FEATURE_COLS) - set(feat_cols)
        raise ValueError(f"missing feature cols: {missing}")
    sub = df.sort_values([ins_col, "date"], kind="mergesort")
    counts = sub.groupby(ins_col, sort=False).size()
    valid_ids = counts[counts == require_bars].index
    sub = sub[sub[ins_col].isin(valid_ids)]
    if sub.empty:
        return np.zeros(0, dtype=object), np.zeros((0, require_bars, N_FEAT), dtype=np.float32)
    vals = sub[feat_cols].to_numpy(np.float32, copy=False)
    n = len(valid_ids)
    stacked = vals.reshape(n, require_bars, N_FEAT)
    first = sub[ins_col].to_numpy()[::require_bars]
    return np.asarray(first, dtype=object), stacked


def build_raw_day_panel(df: pd.DataFrame) -> DayPanel | None:
    """log1p → pivot → 缺失过滤 → ffill，不做任何归一化。

    与 mae_train.build_raw_day_panel 逐字等价，推理时可只建一次给两族共用。
    """
    if df is None or df.empty:
        return None
    feat = apply_log1p(df)
    instruments, feats = _pivot_day_features(feat)
    if len(instruments) == 0:
        return None
    bar_missing = ~np.isfinite(feats).all(axis=-1)
    keep = bar_missing.mean(axis=1) <= MISSING_BAR_FRAC
    if not keep.any():
        return None
    instruments = instruments[keep]
    feats = feats[keep]
    feats = ffill_then_zero(feats)
    return DayPanel(instruments=instruments, x=feats.astype(np.float32, copy=False))


def build_day_panel_from_frame(df: pd.DataFrame) -> DayPanel | None:
    # prenorm for post_pack: no per-day MAD/zscore
    return build_raw_day_panel(df)


def _cloud_select_sql_cols() -> str:
    """云端 stock_bar1m 实际列名为 deal_number（不是竞赛文档里的 num_trades）。"""
    cols = dict.fromkeys(
        ["date", "instrument", "adjust_factor", "open", "close"] + list(FEATURE_COLS)
    )
    return ", ".join(cols)


def load_cloud_single_day(
    table: str,
    day: str,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    dai = get_dai()
    cols = _cloud_select_sql_cols()
    sql = f"SELECT {cols} FROM {table} ORDER BY instrument, date"
    filters: dict = {"date": [f"{day} 00:00:00", f"{day} 23:59:59"]}
    if instruments:
        filters["instrument"] = instruments
    df = dai.query(sql, filters=filters).df()
    return to_canonical(df, is_local=False)


def pool_instruments(start: str, end: str) -> list[str]:
    dai = get_dai()
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [start, end]},
    ).df()
    return df["instrument"].astype(str).tolist()


def build_adj_open_from_bars(df: pd.DataFrame) -> pd.DataFrame:
    """From 1m bars: first-bar open * adjust_factor → adj_open panel."""
    if df.empty:
        return pd.DataFrame(columns=["date", "instrument", "adj_open"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["trading_day"] = out["date"].dt.normalize()
    out = out.sort_values(["instrument", "date"], kind="mergesort")
    first = out.groupby(["trading_day", "instrument"], sort=False).first().reset_index()
    op = first["open"].astype("float64")
    af = first["adjust_factor"].astype("float64") if "adjust_factor" in first.columns else 1.0
    valid = np.isfinite(op) & (op > 0) & np.isfinite(af)
    return pd.DataFrame(
        {
            "date": first["trading_day"],
            "instrument": first["instrument"].astype(str),
            "adj_open": np.where(valid, op * af, np.nan),
        }
    )


def load_cloud_range_bars(
    table: str,
    start: str,
    end: str,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """Pull [start, end] bar1m by month chunks."""
    dai = get_dai()
    cols = _cloud_select_sql_cols()
    sql = f"SELECT {cols} FROM {table} ORDER BY instrument, date"
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    months = pd.date_range(start_ts.replace(day=1), end_ts, freq="MS")
    if len(months) == 0:
        months = pd.DatetimeIndex([start_ts.replace(day=1)])
    frames: list[pd.DataFrame] = []
    for month_start in months:
        chunk_start = max(start_ts, month_start)
        chunk_end = min(end_ts, (month_start + pd.offsets.MonthEnd(0)))
        filters: dict = {
            "date": [
                f"{chunk_start.strftime('%Y-%m-%d')} 00:00:00",
                f"{chunk_end.strftime('%Y-%m-%d')} 23:59:59",
            ]
        }
        if instruments:
            filters["instrument"] = instruments
        t0 = time.time()
        part = dai.query(sql, filters=filters).df()
        canon = to_canonical(part, is_local=False) if not part.empty else part
        print(
            f"[timing] cloud chunk {chunk_start.date()}~{chunk_end.date()} "
            f"rows={len(canon):,} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if canon is not None and not canon.empty:
            frames.append(canon)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------

def apply_pack_slices(day_xs, slices=PACK_SLICES):
    """day_xs: list of [N,T_day,F] → [N,T_pack,F]."""
    parts = []
    for day_idx, a, b in slices:
        parts.append(day_xs[int(day_idx)][:, int(a) : int(b), :])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


# Dataset
# ---------------------------------------------------------------------------
class CloudDailyDataset(Dataset):
    """One sample = one trading-day cross-section."""

    def __init__(
        self,
        calendar: TradingCalendar,
        sample_days: Sequence[str],
        *,
        mode: str = "train",
        open_panel: pd.DataFrame | None = None,
        n_day: int = N_DAY,
        cloud_table: str = CLOUD_TABLE,
        cloud_instruments: list[str] | None = None,
        panel_cache: dict[str, DayPanel] | None = None,
        max_cached_panels: int = 64,
    ) -> None:
        self.calendar = calendar
        self.sample_days = list(sample_days)
        self.mode = mode
        self.n_day = n_day
        self.cloud_table = cloud_table
        self.cloud_instruments = cloud_instruments
        self.max_cached_panels = max(1, int(max_cached_panels))
        self._cache: OrderedDict[str, DayPanel] = OrderedDict()
        if panel_cache:
            for d, p in panel_cache.items():
                self._cache[d] = p
            self.max_cached_panels = max(self.max_cached_panels, len(self._cache))
        self._open_lookup: pd.Series | None = None
        if open_panel is not None and not open_panel.empty:
            op = open_panel.copy()
            op["date"] = pd.to_datetime(op["date"]).dt.normalize()
            op["instrument"] = op["instrument"].astype(str)
            self._open_lookup = op.set_index(["date", "instrument"])["adj_open"]

    def __len__(self) -> int:
        return len(self.sample_days)

    def _get_panel(self, day: str) -> DayPanel | None:
        if day in self._cache:
            self._cache.move_to_end(day)
            return self._cache[day]
        raw = load_cloud_single_day(self.cloud_table, day, self.cloud_instruments)
        panel = build_day_panel_from_frame(raw)
        if panel is None:
            return None
        self._cache[day] = panel
        self._cache.move_to_end(day)
        while len(self._cache) > self.max_cached_panels:
            self._cache.popitem(last=False)
        return panel

    def set_panels(
        self,
        panels: dict[str, DayPanel],
        *,
        replace: bool = True,
    ) -> "CloudDailyDataset":
        """注入已建好的日面板（按块推理用）；上限自动放宽到不会淘汰注入项。"""
        if replace:
            self._cache.clear()
        for day, panel in panels.items():
            if panel is None:
                continue
            self._cache[day] = panel
            self._cache.move_to_end(day)
        self.max_cached_panels = max(self.max_cached_panels, len(self._cache))
        return self

    def preload(self, bars: pd.DataFrame | None = None, *, verbose: bool = True) -> "CloudDailyDataset":
        needed: list[str] = []
        seen: set[str] = set()
        for day in self.sample_days:
            win = self.calendar.window(day, self.n_day)
            if win is None:
                continue
            for d in win:
                if d not in seen:
                    seen.add(d)
                    needed.append(d)
        if bars is not None and not bars.empty:
            bars = bars.copy()
            bars["trading_day"] = pd.to_datetime(bars["date"]).dt.normalize().dt.strftime("%Y-%m-%d")
            for d, sub in bars.groupby("trading_day", sort=False):
                if d in needed and d not in self._cache:
                    panel = build_day_panel_from_frame(sub.drop(columns=["trading_day"], errors="ignore"))
                    if panel is not None:
                        self._cache[d] = panel
        else:
            for d in needed:
                if d not in self._cache:
                    self._get_panel(d)
        if verbose:
            print(f"[preload] cached {len(self._cache)}/{len(needed)} day panels", flush=True)
        return self

    def __getitem__(self, i: int):
        day = self.sample_days[i]
        window = self.calendar.window(day, self.n_day)
        if window is None:
            raise RuntimeError(f"invalid window day={day}")
        panels = [self._get_panel(d) for d in window]
        if any(p is None for p in panels):
            raise RuntimeError(f"missing day panel day={day} window={window}")

        common = panels[0].instruments
        for p in panels[1:]:
            common = np.intersect1d(common, p.instruments)
        if len(common) == 0:
            raise RuntimeError(f"no common instruments day={day}")

        xs = []
        for p in panels:
            pos = {ins: j for j, ins in enumerate(p.instruments)}
            idx = np.array([pos[ins] for ins in common], dtype=np.int64)
            xs.append(p.x[idx])
        # pack then one MAD+zscore on [N,T_pack,F]
        x = apply_pack_slices(xs, PACK_SLICES)
        x = mad_clip(x, MAD_THRESH)
        x = zscore(x)
        instruments = common

        if self.mode == "infer":
            y = np.zeros(len(instruments), dtype=np.float32)
            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                day,
                instruments,
                torch.zeros(len(instruments), dtype=torch.int64),
            )

        if self._open_lookup is None:
            raise RuntimeError("open_gap label needs adj_open panel")
        d_start = self.calendar.shift(day, LABEL_OPEN_START_LAG)
        d_end = self.calendar.shift(day, LABEL_OPEN_END_LAG)
        if d_start is None or d_end is None:
            raise RuntimeError(f"missing label days for {day}")
        try:
            p0 = self._open_lookup.xs(pd.Timestamp(d_start), level="date").reindex(instruments)
            p1 = self._open_lookup.xs(pd.Timestamp(d_end), level="date").reindex(instruments)
        except KeyError as exc:
            raise RuntimeError(f"open panel missing rows day={day}") from exc

        y_raw = p1.to_numpy(dtype=np.float64) / p0.to_numpy(dtype=np.float64) - 1.0
        p0_arr = p0.to_numpy(dtype=np.float64)
        valid = np.isfinite(y_raw) & np.isfinite(p0_arr) & (p0_arr > 0)
        if not valid.any():
            raise RuntimeError(f"no valid labels day={day}")
        x = x[valid]
        y = mad_clip_1d(y_raw[valid].astype(np.float32), MAD_THRESH)
        y = zscore_1d(y)
        instruments = instruments[valid]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            day,
            instruments,
            torch.zeros(len(instruments), dtype=torch.int64),
        )


# ---------------------------------------------------------------------------
# Train / eval / checkpoint
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_context(device: torch.device):
    if not USE_AMP or device.type != "cuda":
        return nullcontext, None
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = None if bf16 else torch.amp.GradScaler("cuda")

    def factory():
        return torch.amp.autocast("cuda", dtype=dtype)

    return factory, scaler


def _encode_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v)
        out[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),
        }
    return out


def _decode_state_dict(payload_sd: dict, map_location: str = "cpu") -> dict:
    sd = {}
    for k, meta in payload_sd.items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    return sd


def save_model(ckpt: dict, model_path: Path | str = MODEL_PATH) -> str:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = _encode_state_dict(ckpt["state_dict"])
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return str(model_path)


def load_model(model_path: Path | str = MODEL_PATH, map_location: str = "cpu") -> dict:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    size = model_path.stat().st_size
    if size == 0:
        raise ValueError(
            f"model file is empty (0 bytes): {model_path}; "
            "re-upload trained model_seed*.json with the notebook"
        )
    with open(model_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError(
            f"model file has no JSON content: {model_path} ({size} bytes); "
            "re-upload trained model_seed*.json with the notebook"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:80].replace("\n", "\\n")
        raise ValueError(
            f"invalid model JSON at {model_path} ({size} bytes), "
            f"starts with {preview!r}: {e}"
        ) from e
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = _decode_state_dict(payload["state_dict"], map_location)
    return ckpt


def load_ensemble_models(
    root: Path | str | None = None,
    map_location: str = "cpu",
    device: torch.device | None = None,
) -> tuple[list[TemporalConvGRUModel], dict]:
    """加载 4 个 seed JSON，返回 (models, 共享 meta；取自第一个 ckpt)。"""
    paths = list_model_paths(root)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing seed weight files:\n  " + "\n  ".join(missing)
        )
    models: list[TemporalConvGRUModel] = []
    meta: dict | None = None
    target = device if device is not None else torch.device(map_location)
    for path in paths:
        ckpt = load_model(path, map_location="cpu")
        if meta is None:
            meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        cfg = dict(ckpt["model_cfg"])
        m = TemporalConvGRUModel(**cfg).to(target)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
    assert meta is not None
    meta["n_models"] = len(models)
    meta["seeds"] = list(SEEDS)
    return models, meta


def daily_pearson_ic(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) < MIN_N or np.std(scores) < 1e-12 or np.std(labels) < 1e-12:
        return float("nan")
    a = scores - scores.mean()
    b = labels - labels.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def daily_spearman_ic(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) < MIN_N:
        return float("nan")
    rs = pd.Series(scores).rank().to_numpy()
    rl = pd.Series(labels).rank().to_numpy()
    return daily_pearson_ic(rs, rl)


@torch.no_grad()
def eval_dataset(model: nn.Module, dataset: Dataset, device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    ics, rank_ics = [], []
    total_mse, n_days = 0.0, 0
    mse_fn = nn.MSELoss()
    for batch in loader:
        x, y, _day, _ins, industry = batch
        pred = model(x.to(device), industry.to(device))
        total_mse += mse_fn(pred, y.to(device)).item()
        n_days += 1
        s = pred.detach().cpu().numpy()
        t = y.numpy()
        pic = daily_pearson_ic(s, t)
        ric = daily_spearman_ic(s, t)
        if np.isfinite(pic):
            ics.append(pic)
        if np.isfinite(ric):
            rank_ics.append(ric)
    return {
        "mse": total_mse / max(n_days, 1),
        "mean_ic": float(np.nanmean(ics)) if ics else float("nan"),
        "mean_rank_ic": float(np.nanmean(rank_ics)) if rank_ics else float("nan"),
        "n_days": n_days,
    }


def train_epochs(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    device: torch.device,
    *,
    epochs: int = EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    lr: float = LR,
) -> tuple[nn.Module, dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        min_lr=PLATEAU_MIN_LR,
    )
    amp_factory, scaler = _amp_context(device)
    best_metric = -np.inf
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    history: list[dict] = []
    loader = DataLoader(train_ds, batch_size=None, shuffle=False, num_workers=0)

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        total_loss, n_steps = 0.0, 0
        for batch in loader:
            x, y, _day, _ins, industry = batch
            x = x.to(device)
            y = y.to(device)
            industry = industry.to(device)
            opt.zero_grad(set_to_none=True)
            with amp_factory():
                pred = model(x, industry)
                loss = compute_train_loss(pred, y, LOSS_TYPE)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            total_loss += float(loss.item())
            n_steps += 1

        train_loss = total_loss / max(n_steps, 1)
        val_metrics = eval_dataset(model, val_ds, device)
        sel = val_metrics["mean_rank_ic"]
        improved = np.isfinite(sel) and sel > best_metric
        if improved:
            best_metric = float(sel)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if np.isfinite(sel):
            scheduler.step(sel)
        cur_lr = float(opt.param_groups[0]["lr"])
        row = {
            "epoch": ep + 1,
            "train_loss": train_loss,
            "train_soft_rankic": -train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": cur_lr,
            "elapsed": round(time.time() - t0, 2),
        }
        history.append(row)
        print(
            f"epoch {ep + 1}/{epochs}  soft_rankic={-train_loss:.6f}  "
            f"val_ic={val_metrics['mean_ic']:.4f}  "
            f"val_rank_ic={val_metrics['mean_rank_ic']:.4f}  "
            f"lr={cur_lr:.2e}  "
            f"no_improve={epochs_no_improve}/{patience}  "
            f"elapsed={row['elapsed']:.1f}s",
            flush=True,
        )
        if epochs_no_improve >= patience:
            print(f"early stop @ epoch {ep + 1}: best rankic={best_metric:.4f} @ {best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "history": history,
        "best_val_rank_ic": best_metric,
        "best_epoch": best_epoch,
        "stopped_early": epochs_no_improve >= patience,
        "total_epochs": len(history),
    }
    return model, info


def _checkpoint_payload(
    model: nn.Module,
    info: dict,
    *,
    seed: int,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    n_day: int = N_DAY,
) -> dict:
    return {
        "format": "temporal_conv_seed_single",
        "model_type": MODEL_TYPE,
        "seed": int(seed),
        "model_cfg": dict(MODEL_CFG),
        "state_dict": model.state_dict(),
        "loss_type": LOSS_TYPE,
        "val_metric": VAL_METRIC,
        "pred_cs_zscore": False,
        "patience": EARLY_STOP_PATIENCE,
        "lr": LR,
        "scheduler": "plateau",
        "n_day": n_day,
        "label_type": "open_gap",
        "label_open_start_lag": LABEL_OPEN_START_LAG,
        "label_open_end_lag": LABEL_OPEN_END_LAG,
        "normalization": "post_pack_cross_sectional_mad_zscore",
        "pack_mode": PACK_MODE,
        "feat_norm": FEAT_NORM,
        "t_pack": T_INTRADAY,
        "pack_slices": [list(s) for s in PACK_SLICES],
        "feature_cols": list(FEATURE_COLS),
        "lob_cols": list(LOB_COLS),
        "kline_cols": list(KLINE_COLS),
        "extra_cols": list(EXTRA_COLS),
        "vol_cols": list(VOL_COLS),
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "best_metric": info["best_val_rank_ic"],
        "best_epoch": info["best_epoch"],
        "history": info["history"],
    }


def train_one_seed(
    train_ds: Dataset,
    val_ds: Dataset,
    device: torch.device,
    *,
    seed: int,
    epochs: int = EPOCHS,
    out_path: Path | str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    val_start: str | None = None,
    val_end: str | None = None,
) -> str:
    """用给定 seed 训练单个 TemporalConvGRU，写入 model_seed{seed}.json。"""
    set_seed(seed)
    model = TemporalConvGRUModel(**MODEL_CFG).to(device)
    print(f"[seed={seed}] params={count_params(model):,}", flush=True)
    model, info = train_epochs(model, train_ds, val_ds, device, epochs=epochs)
    path = Path(out_path) if out_path is not None else model_path_for_seed(seed)
    ts = train_start or (train_ds.sample_days[0] if train_ds.sample_days else SUBMIT_TRAIN_START)
    te = train_end or (train_ds.sample_days[-1] if train_ds.sample_days else SUBMIT_TRAIN_END)
    vs = val_start or (val_ds.sample_days[0] if val_ds.sample_days else SUBMIT_VAL_START)
    ve = val_end or (val_ds.sample_days[-1] if val_ds.sample_days else SUBMIT_VAL_END)
    saved = save_model(
        _checkpoint_payload(
            model,
            info,
            seed=seed,
            train_start=ts,
            train_end=te,
            val_start=vs,
            val_end=ve,
        ),
        path,
    )
    print(
        f"[seed={seed}] saved → {saved}  best_rankic={info['best_val_rank_ic']:.4f}"
        f"@epoch{info['best_epoch']}",
        flush=True,
    )
    return saved


def train_and_save(
    datasources: dict,
    model_path: Path | str | None = None,
    *,
    train_start: str = SUBMIT_TRAIN_START,
    train_end: str = SUBMIT_TRAIN_END,
    val_start: str = SUBMIT_VAL_START,
    val_end: str = SUBMIT_VAL_END,
    epochs: int = EPOCHS,
    seeds: Sequence[int] = SEEDS,
) -> list[str]:
    """Private-board entry: 串行训练 4 个 seed，各自保存 JSON。"""
    del model_path  # 固定按 seed 命名
    table = datasources.get("bar1m", CLOUD_TABLE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device}  loss={LOSS_TYPE}  val={VAL_METRIC}  seeds={list(seeds)}",
        flush=True,
    )

    buffer_start = (pd.Timestamp(train_start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    buffer_end = (pd.Timestamp(val_end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    instruments = pool_instruments(buffer_start, buffer_end)
    calendar = TradingCalendar.from_cloud(table, buffer_start, buffer_end)
    train_days, val_days = build_submit_split_days(calendar)
    train_days = [d for d in train_days if train_start <= d <= train_end]
    val_days = [d for d in val_days if val_start <= d <= val_end]
    print(f"train_days={len(train_days)} val_days={len(val_days)}", flush=True)

    print("loading cloud bars for train+val ...", flush=True)
    all_start = min(train_days[0], val_days[0])
    all_end = max(
        calendar.shift(val_days[-1], LABEL_OPEN_END_LAG) or val_days[-1],
        val_days[-1],
    )
    bars = load_cloud_range_bars(table, all_start, all_end, instruments)
    open_panel = build_adj_open_from_bars(bars)
    print(f"open_panel rows={len(open_panel):,}", flush=True)

    train_ds = CloudDailyDataset(
        calendar, train_days, mode="train", open_panel=open_panel,
        cloud_table=table, cloud_instruments=instruments,
    ).preload(bars)
    val_ds = CloudDailyDataset(
        calendar, val_days, mode="val", open_panel=open_panel,
        cloud_table=table, cloud_instruments=instruments,
        panel_cache=dict(train_ds._cache),
    )
    val_ds.preload(bars)

    paths: list[str] = []
    for seed in seeds:
        paths.append(
            train_one_seed(
                train_ds,
                val_ds,
                device,
                seed=int(seed),
                epochs=epochs,
                train_start=train_days[0],
                train_end=train_days[-1],
                val_start=val_days[0],
                val_end=val_days[-1],
            )
        )
    return paths


def _forward_day(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        pred = model(x.to(device), None)
    return pred.detach().cpu().numpy()
