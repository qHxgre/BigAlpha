"""Shared training and prediction code for BigAlpha 2026 E2E submissions."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import duckdb
except Exception:  # Cloud inference does not need DuckDB.
    duckdb = None  # type: ignore

try:
    import dai  # type: ignore
except Exception:  # pragma: no cover - local SDK uses bigquant.dai
    try:
        from bigquant import dai  # type: ignore
    except Exception:
        dai = None  # type: ignore

try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)

    class _KeywordLogger:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def _write(self, level: int, event: str, **fields) -> None:
            suffix = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
            self._logger.log(level, f"{event} {suffix}".rstrip())

        def info(self, event: str, **fields) -> None:
            self._write(logging.INFO, event, **fields)

        def warning(self, event: str, **fields) -> None:
            self._write(logging.WARNING, event, **fields)

    logger = _KeywordLogger("bigalpha_e2e")


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_model.json")

FREQS = ("1m", "5m", "15m", "30m")
EXPECTED_BARS = {"1m": 240, "5m": 48, "15m": 16, "30m": 8}
LOCAL_TABLES = {
    "1m": "bigalpha_2026_e2e_bar1m",
    "5m": "bigalpha_2026_e2e_bar5m",
    "15m": "bigalpha_2026_e2e_bar15m",
    "30m": "bigalpha_2026_e2e_bar30m",
}
PLATFORM_TABLES = {
    "1m": "bigalpha_2026_stock_bar1m",
    "5m": "bigalpha_2026_stock_bar5m",
    "15m": "bigalpha_2026_stock_bar15m",
    "30m": "bigalpha_2026_stock_bar30m",
}
DATASOURCE_KEYS = {"1m": "bar1m", "5m": "bar5m", "15m": "bar15m", "30m": "bar30m"}
INSTRUMENTS_TABLE = "bigalpha_2026_instruments"

PRICE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "ask_price1",
    "ask_price2",
    "ask_price3",
    "bid_price1",
    "bid_price2",
    "bid_price3",
]
FLOW_COLS = [
    "deal_number",
    "volume",
    "amount",
    "ask_volume1",
    "ask_volume2",
    "ask_volume3",
    "bid_volume1",
    "bid_volume2",
    "bid_volume3",
    "ask_num_orders1",
    "ask_num_orders2",
    "ask_num_orders3",
    "bid_num_orders1",
    "bid_num_orders2",
    "bid_num_orders3",
]
FEATURE_COLS = ["adjust_factor"] + PRICE_COLS + FLOW_COLS
BAR_COLUMNS = ["date", "instrument", "instrument_id"] + FEATURE_COLS
MODEL_CFG = {
    "n_feat": len(FEATURE_COLS),
    "d_model": 96,
    "nhead": 4,
    "nlayers": 2,
    "dim_ff": 192,
    "dropout": 0.10,
}

MODEL_CFG_V2 = {
    "n_feat": len(FEATURE_COLS),
    "d_model": 128,
    "nhead": 8,
    "nlayers": 3,
    "dim_ff": 384,
    "cross_layers": 2,
    "dropout": 0.10,
}

PATCH_SIZE = {"1m": 4, "5m": 2, "15m": 1, "30m": 1}


@dataclass
class DayTensor:
    keys: np.ndarray
    adj_close: np.ndarray
    features: dict[str, np.ndarray]


class FreqEncoder(nn.Module):
    def __init__(self, n_feat: int, bars: int, d_model: int, nhead: int, nlayers: int, dim_ff: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, bars, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(x) + self.pos)
        return self.norm(h.mean(dim=1))


class MultiFreqTransformer(nn.Module):
    def __init__(self, n_feat: int, d_model: int, nhead: int, nlayers: int, dim_ff: int, dropout: float):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                freq: FreqEncoder(n_feat, EXPECTED_BARS[freq], d_model, nhead, nlayers, dim_ff, dropout)
                for freq in FREQS
            }
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * len(FREQS)),
            nn.Linear(d_model * len(FREQS), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.encoders[freq](batch[freq]) for freq in FREQS]
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


class PatchFreqEncoder(nn.Module):
    """Learned temporal patching followed by a per-frequency Transformer."""

    def __init__(
        self,
        n_feat: int,
        bars: int,
        patch_size: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        dropout: float,
    ):
        super().__init__()
        if bars % patch_size:
            raise ValueError(f"bars={bars} must be divisible by patch_size={patch_size}")
        self.patch = nn.Conv1d(n_feat, d_model, kernel_size=patch_size, stride=patch_size)
        n_tokens = bars // patch_size
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch(x.transpose(1, 2)).transpose(1, 2)
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos
        return self.norm(self.encoder(h)[:, 0])


class MultiFreqPatchTransformer(nn.Module):
    """Multi-frequency raw-sequence model with learned temporal and frequency fusion."""

    def __init__(
        self,
        n_feat: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        cross_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                freq: PatchFreqEncoder(
                    n_feat=n_feat,
                    bars=EXPECTED_BARS[freq],
                    patch_size=PATCH_SIZE[freq],
                    d_model=d_model,
                    nhead=nhead,
                    nlayers=nlayers,
                    dim_ff=dim_ff,
                    dropout=dropout,
                )
                for freq in FREQS
            }
        )
        self.freq_pos = nn.Parameter(torch.zeros(1, len(FREQS), d_model))
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(cross_layer, cross_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.freq_pos, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = torch.stack([self.encoders[freq](batch[freq]) for freq in FREQS], dim=1)
        h = self.cross_encoder(h + self.freq_pos).mean(dim=1)
        return self.head(h).squeeze(-1)


class MultiFreqEnsemble(nn.Module):
    """One end-to-end model with independently trained complementary branches."""

    def __init__(self, v1_cfg: dict, v2_cfg: dict, alpha_v1: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha_v1 <= 1.0:
            raise ValueError("alpha_v1 must be between zero and one")
        self.v1 = MultiFreqTransformer(**v1_cfg)
        self.v2 = MultiFreqPatchTransformer(**v2_cfg)
        self.alpha_v1 = float(alpha_v1)

    @staticmethod
    def cross_sectional_zscore(score: torch.Tensor) -> torch.Tensor:
        if score.numel() < 2:
            return score
        return (score - score.mean()) / (score.std(unbiased=False) + 1e-6)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        v1_score = self.cross_sectional_zscore(self.v1(batch))
        v2_score = self.cross_sectional_zscore(self.v2(batch))
        return self.alpha_v1 * v1_score + (1.0 - self.alpha_v1) * v2_score


def build_model(arch: str, cfg: dict) -> nn.Module:
    if arch == "v1":
        return MultiFreqTransformer(**cfg)
    if arch == "v2":
        return MultiFreqPatchTransformer(**cfg)
    if arch == "ensemble":
        return MultiFreqEnsemble(**cfg)
    raise ValueError(f"unknown architecture: {arch}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def table_for_freq(datasources: dict | None, freq: str, local: bool) -> str:
    if datasources:
        key = DATASOURCE_KEYS[freq]
        if key in datasources:
            return datasources[key]
    return LOCAL_TABLES[freq] if local else PLATFORM_TABLES[freq]


def instruments_table(datasources: dict | None) -> str:
    if datasources and "instruments" in datasources:
        return datasources["instruments"]
    return INSTRUMENTS_TABLE


def inclusive_end_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp == timestamp.normalize():
        timestamp += pd.Timedelta(hours=23, minutes=59, seconds=59)
    return timestamp


def infer_source(df: pd.DataFrame) -> str:
    return "stock" if "instrument" in df.columns and df["instrument"].notna().any() else "e2e"


def canonical_features(df: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray]:
    work = df.copy()
    if source == "e2e":
        for col in PRICE_COLS:
            work[col] = work[col].astype("float64") / 100.0
        work["amount"] = work["amount"].astype("float64") / 100.0

    for col in PRICE_COLS:
        x = work[col].astype("float64")
        x = x.where(x > 0, np.nan)
        work[col] = np.log(x)

    adj = work["adjust_factor"].astype("float64").where(work["adjust_factor"].astype("float64") > 0, np.nan)
    work["adjust_factor"] = np.log(adj)

    for col in FLOW_COLS:
        x = work[col].astype("float64")
        x = x.where(x >= 0, np.nan)
        work[col] = np.log1p(x)

    close = df["close"].astype("float64")
    if source == "e2e":
        close = close / 100.0
    af = df["adjust_factor"].astype("float64")
    adj_close = (close.where(close > 0, np.nan) * af.where(af > 0, np.nan)).to_numpy(np.float64)
    return work[FEATURE_COLS].to_numpy(np.float32), adj_close


def tensorize_day(df: pd.DataFrame, freq: str, key_col: str, source: str) -> tuple[dict, dict]:
    bars = EXPECTED_BARS[freq]
    if df.empty:
        return {}, {}
    df = df.sort_values([key_col, "date"], kind="stable").reset_index(drop=True)
    vals, adj_close = canonical_features(df, source)
    codes, unique_keys = pd.factorize(df[key_col], sort=True)
    n_keys = len(unique_keys)
    counts = np.bincount(codes, minlength=n_keys)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    within = np.arange(len(df), dtype=np.int64) - starts[codes]
    first_kept = np.maximum(counts - bars, 0)
    keep = within >= first_kept[codes]
    slots = within[keep] - first_kept[codes[keep]] + np.maximum(bars - counts[codes[keep]], 0)
    packed = np.full((n_keys, bars, vals.shape[1]), np.nan, dtype=np.float32)
    packed[codes[keep], slots] = vals[keep]

    close_series = pd.Series(adj_close).where(np.isfinite(adj_close))
    last_close = close_series.groupby(codes, sort=False).last().reindex(range(n_keys)).to_numpy(np.float64)
    tensors = {key: packed[i] for i, key in enumerate(unique_keys)}
    closes = {key: float(last_close[i]) for i, key in enumerate(unique_keys)}
    return tensors, closes


def make_day_tensor(freq_frames: dict[str, pd.DataFrame], key_col: str | None = None) -> DayTensor:
    if key_col is None:
        any_df = next((df for df in freq_frames.values() if not df.empty), pd.DataFrame())
        key_col = "instrument" if "instrument" in any_df.columns and any_df["instrument"].notna().any() else "instrument_id"

    per_freq: dict[str, dict] = {}
    close_map: dict = {}
    key_sets = []
    for freq, df in freq_frames.items():
        source = infer_source(df)
        tensors, closes = tensorize_day(df, freq, key_col, source)
        per_freq[freq] = tensors
        key_sets.append(set(tensors))
        if freq == "1m":
            close_map = closes

    if not key_sets:
        raise RuntimeError("no frames passed to make_day_tensor")
    keys = sorted(set.intersection(*key_sets) & set(close_map))
    if not keys:
        raise RuntimeError("no aligned instruments for day tensor")

    features = {
        freq: np.stack([per_freq[freq][key] for key in keys]).astype(np.float32, copy=False)
        for freq in FREQS
    }
    adj_close = np.asarray([close_map[key] for key in keys], dtype=np.float64)
    return DayTensor(keys=np.asarray(keys), adj_close=adj_close, features=features)


def parquet_glob(cache_dir: Path, freq: str) -> str:
    table = LOCAL_TABLES[freq]
    candidates = [
        cache_dir / table / "*.parquet",
        cache_dir / freq / "*.parquet",
    ]
    for path in candidates:
        if list(path.parent.glob(path.name)):
            return str(path)
    return str(candidates[0])


def list_cache_days(cache_dir: Path, start: str, end: str) -> list[pd.Timestamp]:
    if duckdb is None:
        raise RuntimeError("duckdb is required to read the local parquet cache")
    con = duckdb.connect()
    glob = parquet_glob(cache_dir, "30m")
    start_ts = str(pd.Timestamp(start))
    end_ts = str(inclusive_end_timestamp(end))
    sql = """
        SELECT DISTINCT CAST(date AS DATE) AS d
        FROM read_parquet(?)
        WHERE date >= ? AND date <= ?
        ORDER BY d
    """
    df = con.execute(sql, [glob, start_ts, end_ts]).df()
    con.close()
    return [pd.Timestamp(x) for x in df["d"]]


def read_cache_day(cache_dir: Path, freq: str, day: pd.Timestamp) -> pd.DataFrame:
    if duckdb is None:
        raise RuntimeError("duckdb is required to read the local parquet cache")
    con = duckdb.connect()
    glob = parquet_glob(cache_dir, freq)
    start = f"{day:%Y-%m-%d} 00:00:00"
    end = f"{day:%Y-%m-%d} 23:59:59"
    cols = ["date", "instrument_id"] + FEATURE_COLS
    sql = f"""
        SELECT {", ".join(cols)}
        FROM read_parquet(?)
        WHERE date >= ? AND date <= ?
        ORDER BY instrument_id, date
    """
    df = con.execute(sql, [glob, start, end]).df()
    con.close()
    df = df.drop_duplicates(["date", "instrument_id"], keep="last")
    return df


def build_tensor_cache(cache_dir: str | Path, tensor_dir: str | Path, start: str, end: str, overwrite: bool = False) -> None:
    cache_dir = Path(cache_dir)
    tensor_dir = Path(tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    days = list_cache_days(cache_dir, start, end)
    logger.info("building tensor cache", days=len(days), tensor_dir=str(tensor_dir))
    for day in days:
        out = tensor_dir / f"{day:%Y%m%d}.npz"
        if out.exists() and not overwrite:
            continue
        frames = {freq: read_cache_day(cache_dir, freq, day) for freq in FREQS}
        try:
            day_tensor = make_day_tensor(frames, key_col="instrument_id")
        except Exception as exc:
            logger.warning("skip tensor day", day=str(day.date()), error=str(exc))
            continue
        tmp = tensor_dir / f"{day:%Y%m%d}.tmp.npz"
        np.savez_compressed(
            tmp,
            instrument_id=day_tensor.keys.astype(np.int32),
            adj_close=day_tensor.adj_close.astype(np.float64),
            **{f"x_{freq}": day_tensor.features[freq].astype(np.float32) for freq in FREQS},
        )
        os.replace(tmp, out)
        logger.info("tensor day written", day=str(day.date()), n=len(day_tensor.keys))


def build_tensor_cache_from_datasources(
    datasources: dict,
    tensor_dir: str | Path,
    start: str,
    end: str,
    overwrite: bool = False,
    chunk_days: int = 31,
) -> None:
    """Build the same tensor cache from platform-injected raw stock tables.

    This is the private-leaderboard reproduction path.  It deliberately shares
    canonical_features/make_day_tensor with local compressed-data training.
    """
    if dai is None:
        raise RuntimeError("dai SDK is unavailable for datasource training")
    tensor_dir = Path(tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    end_query = str(inclusive_end_timestamp(end))
    for chunk_start, chunk_end in month_chunks(start, end_query, max_days=chunk_days):
        frames = query_platform_frames(datasources, chunk_start, chunk_end)
        by_day = split_frames_by_day(frames)
        for day in sorted(by_day):
            out = tensor_dir / f"{day:%Y%m%d}.npz"
            if out.exists() and not overwrite:
                continue
            day_frames = by_day[day]
            if not all(freq in day_frames and not day_frames[freq].empty for freq in FREQS):
                logger.warning("skip incomplete datasource day", day=str(day.date()))
                continue
            try:
                day_tensor = make_day_tensor(day_frames, key_col="instrument")
            except Exception as exc:
                logger.warning("skip datasource tensor day", day=str(day.date()), error=str(exc))
                continue
            tmp = tensor_dir / f"{day:%Y%m%d}.tmp.npz"
            np.savez_compressed(
                tmp,
                instrument_id=day_tensor.keys.astype(str),
                adj_close=day_tensor.adj_close.astype(np.float64),
                **{f"x_{freq}": day_tensor.features[freq].astype(np.float32) for freq in FREQS},
            )
            os.replace(tmp, out)
            logger.info("datasource tensor day written", day=str(day.date()), n=len(day_tensor.keys))
        del frames, by_day


def tensor_files(tensor_dir: str | Path, start: str, end: str) -> list[Path]:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    files = []
    for path in sorted(Path(tensor_dir).glob("*.npz")):
        try:
            day = pd.Timestamp(datetime.strptime(path.stem, "%Y%m%d"))
        except ValueError:
            continue
        if start_day <= day <= end_day:
            files.append(path)
    return files


def compute_stats(files: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sums = {freq: np.zeros(len(FEATURE_COLS), dtype=np.float64) for freq in FREQS}
    sums2 = {freq: np.zeros(len(FEATURE_COLS), dtype=np.float64) for freq in FREQS}
    counts = {freq: np.zeros(len(FEATURE_COLS), dtype=np.float64) for freq in FREQS}
    for path in files:
        with np.load(path) as data:
            for freq in FREQS:
                x = data[f"x_{freq}"].astype(np.float64)
                mask = np.isfinite(x)
                vals = np.where(mask, x, 0.0)
                sums[freq] += vals.sum(axis=(0, 1))
                sums2[freq] += (vals * vals).sum(axis=(0, 1))
                counts[freq] += mask.sum(axis=(0, 1))
    stats = {}
    for freq in FREQS:
        cnt = np.maximum(counts[freq], 1.0)
        mean = sums[freq] / cnt
        var = np.maximum(sums2[freq] / cnt - mean * mean, 1e-6)
        stats[freq] = (mean.astype(np.float32), np.sqrt(var).astype(np.float32))
    return stats


def compute_stats_from_days(days: list[DayTensor]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sums = {freq: np.zeros(len(FEATURE_COLS), dtype=np.float64) for freq in FREQS}
    sums2 = {freq: np.zeros(len(FEATURE_COLS), dtype=np.float64) for freq in FREQS}
    counts = {freq: np.zeros(len(FEATURE_COLS), dtype=np.float64) for freq in FREQS}
    for day in days:
        for freq in FREQS:
            x = day.features[freq].astype(np.float64)
            mask = np.isfinite(x)
            vals = np.where(mask, x, 0.0)
            sums[freq] += vals.sum(axis=(0, 1))
            sums2[freq] += (vals * vals).sum(axis=(0, 1))
            counts[freq] += mask.sum(axis=(0, 1))
    stats = {}
    for freq in FREQS:
        cnt = np.maximum(counts[freq], 1.0)
        mean = sums[freq] / cnt
        var = np.maximum(sums2[freq] / cnt - mean * mean, 1e-6)
        stats[freq] = (mean.astype(np.float32), np.sqrt(var).astype(np.float32))
    return stats


def standardize(features: dict[str, np.ndarray], stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for freq, x in features.items():
        mean, std = stats[freq]
        z = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        out[freq] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return out


def align_day_pair_full(
    cur: DayTensor, nxt: DayTensor
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cur_pos = {k: i for i, k in enumerate(cur.keys.tolist())}
    nxt_pos = {k: i for i, k in enumerate(nxt.keys.tolist())}
    keys = [k for k in cur.keys.tolist() if k in nxt_pos]
    if not keys:
        raise RuntimeError("no overlap between adjacent tensor days")
    ci = np.asarray([cur_pos[k] for k in keys], dtype=np.int64)
    ni = np.asarray([nxt_pos[k] for k in keys], dtype=np.int64)
    raw_y = nxt.adj_close[ni] / cur.adj_close[ci] - 1.0
    ok = np.isfinite(raw_y)
    if ok.sum() < 20:
        raise RuntimeError("too few finite labels")
    lo, hi = np.nanpercentile(raw_y[ok], [1, 99])
    y = np.clip(raw_y, lo, hi)
    mu, sd = np.nanmean(y[ok]), np.nanstd(y[ok]) + 1e-6
    y = ((y - mu) / sd).astype(np.float32)
    ok &= np.isfinite(y) & np.isfinite(raw_y)
    ci = ci[ok]
    keys_arr = np.asarray(keys)[ok]
    raw_y = raw_y[ok].astype(np.float64)
    y = y[ok]
    x = {freq: cur.features[freq][ci] for freq in FREQS}
    return keys_arr, x, y, raw_y


def align_day_pair(cur: DayTensor, nxt: DayTensor) -> tuple[dict[str, np.ndarray], np.ndarray]:
    _, x, y, _ = align_day_pair_full(cur, nxt)
    return x, y


def load_npz_day(path: Path) -> DayTensor:
    with np.load(path) as data:
        return DayTensor(
            keys=data["instrument_id"].copy(),
            adj_close=data["adj_close"].copy(),
            features={freq: data[f"x_{freq}"].copy() for freq in FREQS},
        )


def preload_tensor_files(files: list[Path], workers: int) -> list[DayTensor]:
    if workers <= 0:
        raise ValueError("preload workers must be positive")
    logger.info("preloading tensor files", files=len(files), workers=workers)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        days = list(pool.map(load_npz_day, files))
    logger.info("tensor preload complete", files=len(days), elapsed=round(time.time() - t0, 2))
    return days


def batch_iter(x: dict[str, np.ndarray], y: np.ndarray, batch_size: int, shuffle: bool = True):
    idx = np.arange(len(y))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        take = idx[start : start + batch_size]
        yield {freq: torch.from_numpy(x[freq][take]) for freq in FREQS}, torch.from_numpy(y[take])


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = torch.sqrt((pred.square().sum() + 1e-6) * (target.square().sum() + 1e-6))
    return 1.0 - (pred * target).sum() / denom


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, max_pairs: int = 2048) -> torch.Tensor:
    n = pred.numel()
    if n < 2:
        return pred.new_zeros(())
    pairs = min(max_pairs, n * 4)
    left = torch.randint(0, n, (pairs,), device=pred.device)
    right = torch.randint(0, n, (pairs,), device=pred.device)
    direction = torch.sign(target[left] - target[right])
    keep = direction != 0
    if not torch.any(keep):
        return pred.new_zeros(())
    margin = (pred[left[keep]] - pred[right[keep]]) * direction[keep]
    return F.softplus(-margin).mean()


def training_loss(pred: torch.Tensor, target: torch.Tensor, objective: str) -> torch.Tensor:
    huber = F.smooth_l1_loss(pred, target, beta=0.5)
    if objective == "huber":
        return huber
    if objective == "ic":
        return 0.45 * huber + 0.45 * correlation_loss(pred, target) + 0.10 * pairwise_rank_loss(pred, target)
    raise ValueError(f"unknown objective: {objective}")


def save_model(payload: dict, model_path: str = MODEL_PATH) -> str:
    """Save exact tensor bytes in a compact, text-only JSON representation."""
    payload = dict(payload)
    sd = payload.pop("state_dict")
    tensors = {}
    for key, value in sd.items():
        t = value.detach().cpu().contiguous()
        raw = t.view(torch.uint8).numpy().tobytes()
        tensors[key] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "encoding": "base64",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data_b64": base64.b64encode(raw).decode("ascii"),
        }
    payload["state_dict"] = tensors
    payload["state_dict_format"] = "base64-v1"
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_file.with_name(model_file.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))
    os.replace(tmp_path, model_file)
    return model_path


def load_model(model_path: str = MODEL_PATH, map_location: str | torch.device = "cpu") -> dict:
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for key, meta in payload["state_dict"].items():
        dtype = getattr(torch, meta["dtype"])
        if "data_b64" in meta:
            raw = base64.b64decode(meta["data_b64"], validate=True)
            expected_hash = meta.get("sha256")
            if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError(f"weight checksum mismatch: {key}")
            expected_bytes = math.prod(meta["shape"]) * torch.empty((), dtype=dtype).element_size()
            if len(raw) != expected_bytes:
                raise ValueError(f"weight byte count mismatch: {key}")
            t = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        else:  # Backward compatibility with the original float-list JSON files.
            t = torch.tensor(meta["data"], dtype=dtype)
        sd[key] = t.reshape(meta["shape"]).to(map_location)
    payload["state_dict"] = sd
    return payload


def branch_training_metadata(checkpoint: dict) -> dict:
    return {
        "arch": checkpoint.get("arch", "v1"),
        "epochs": checkpoint.get("epochs"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_loss": checkpoint.get("best_loss"),
        "best_score": checkpoint.get("best_score"),
        "batch_size": checkpoint.get("batch_size"),
        "learning_rate": checkpoint.get("learning_rate"),
        "weight_decay": checkpoint.get("weight_decay", 1e-4),
        "objective": checkpoint.get("objective"),
        "scheduler": checkpoint.get("scheduler", "constant"),
        "preload_workers": checkpoint.get("preload_workers"),
        "seed": checkpoint.get("seed"),
        "trainable_parameters": checkpoint.get("trainable_parameters"),
        "created_at": checkpoint.get("created_at"),
    }


def combine_models(
    v1_path: str,
    v2_path: str,
    model_path: str = MODEL_PATH,
    alpha_v1: float = 0.5,
) -> str:
    """Combine two competition-trained branches into one auditable model."""
    v1 = load_model(v1_path, map_location="cpu")
    v2 = load_model(v2_path, map_location="cpu")
    if v1.get("arch", "v1") != "v1" or v2.get("arch") != "v2":
        raise ValueError("combine_models expects one v1 checkpoint and one v2 checkpoint")
    if v1["feature_cols"] != v2["feature_cols"] or v1["freqs"] != v2["freqs"]:
        raise ValueError("branch feature definitions do not match")
    if (v1.get("train_start"), v1.get("train_end")) != (v2.get("train_start"), v2.get("train_end")):
        raise ValueError("branch training ranges do not match")

    stats = {}
    for freq in FREQS:
        for key in ("mean", "std"):
            a = np.asarray(v1["stats"][freq][key], dtype=np.float32)
            b = np.asarray(v2["stats"][freq][key], dtype=np.float32)
            if not np.array_equal(a, b):
                raise ValueError(f"branch preprocessing statistics differ: {freq}.{key}")
        stats[freq] = v1["stats"][freq]

    model_cfg = {"v1_cfg": v1["model_cfg"], "v2_cfg": v2["model_cfg"], "alpha_v1": alpha_v1}
    model = build_model("ensemble", model_cfg)
    state = {f"v1.{key}": value for key, value in v1["state_dict"].items()}
    state.update({f"v2.{key}": value for key, value in v2["state_dict"].items()})
    model.load_state_dict(state, strict=True)
    n_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(f"trainable parameter count is outside competition bounds: {n_params}")

    payload = {
        "state_dict": model.state_dict(),
        "arch": "ensemble",
        "model_cfg": model_cfg,
        "feature_cols": v1["feature_cols"],
        "freqs": v1["freqs"],
        "expected_bars": EXPECTED_BARS,
        "patch_size": PATCH_SIZE,
        "stats": stats,
        "train_start": v1.get("train_start"),
        "train_end": v1.get("train_end"),
        "alpha_v1": alpha_v1,
        "branch_training": {
            "v1": branch_training_metadata(v1),
            "v2": branch_training_metadata(v2),
        },
        "branch_history": {"v1": v1.get("history", []), "v2": v2.get("history", [])},
        "seed": v1.get("seed"),
        "trainable_parameters": n_params,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_model(payload, model_path)
    logger.info("ensemble saved", path=model_path, alpha_v1=alpha_v1, parameters=n_params)
    return model_path


def evaluate_tensor_files(
    model: nn.Module,
    files: list[Path] | list[DayTensor],
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    rows = []
    model.eval()
    for idx in range(len(files) - 1):
        try:
            cur_item, nxt_item = files[idx], files[idx + 1]
            cur = cur_item if isinstance(cur_item, DayTensor) else load_npz_day(cur_item)
            nxt = nxt_item if isinstance(nxt_item, DayTensor) else load_npz_day(nxt_item)
            keys, x_np, _, raw_y = align_day_pair_full(cur, nxt)
            day = DayTensor(keys=keys, adj_close=np.ones(len(keys)), features=x_np)
            pred = predict_day(model, day, stats, device, batch_size)
        except Exception as exc:
            logger.warning("skip validation day", file=str(idx), error=str(exc))
            continue
        ok = np.isfinite(pred) & np.isfinite(raw_y)
        pred, raw_y = pred[ok], raw_y[ok]
        if len(pred) < 100 or np.std(pred) < 1e-12 or np.std(raw_y) < 1e-12:
            continue
        pearson = float(np.corrcoef(pred, raw_y)[0, 1])
        rank_pred = pd.Series(pred).rank(method="average").to_numpy()
        rank_y = pd.Series(raw_y).rank(method="average").to_numpy()
        rank_ic = float(np.corrcoef(rank_pred, rank_y)[0, 1])
        order = np.argsort(pred)
        width = max(1, len(order) // 10)
        long_short = float(raw_y[order[-width:]].mean() - raw_y[order[:width]].mean())
        rows.append((pearson, rank_ic, long_short))
    if not rows:
        return {
            "days": 0.0,
            "pearson_mean": float("nan"),
            "rank_ic_mean": float("nan"),
            "long_short_mean": float("nan"),
            "long_short_sharpe": float("nan"),
        }
    values = np.asarray(rows, dtype=np.float64)
    ls_std = values[:, 2].std(ddof=1) + 1e-12
    return {
        "days": float(len(values)),
        "pearson_mean": float(values[:, 0].mean()),
        "pearson_ir": float(values[:, 0].mean() / (values[:, 0].std(ddof=1) + 1e-12)),
        "rank_ic_mean": float(values[:, 1].mean()),
        "rank_ic_ir": float(values[:, 1].mean() / (values[:, 1].std(ddof=1) + 1e-12)),
        "long_short_mean": float(values[:, 2].mean()),
        "long_short_sharpe": float(values[:, 2].mean() / ls_std * math.sqrt(242.0)),
    }


def train_and_save(
    datasources: dict | None = None,
    model_path: str = MODEL_PATH,
    cache_dir: str = "data/e2e_cache",
    tensor_dir: str = "data/tensor_cache",
    train_start: str = "2019-01-01",
    train_end: str = "2024-12-31",
    validation_start: str | None = None,
    validation_end: str | None = None,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    arch: str = "v2",
    objective: str = "ic",
    scheduler_name: str = "cosine",
    preload_workers: int = 0,
) -> str:
    # A prebuilt tensor cache is the fast local path.  Datasources provide the
    # isolated private-retraining path when no cache is present.
    set_seed(seed)
    files = tensor_files(tensor_dir, train_start, train_end)
    if len(files) < 2:
        if datasources:
            build_tensor_cache_from_datasources(datasources, tensor_dir, train_start, train_end)
        else:
            build_tensor_cache(cache_dir, tensor_dir, train_start, train_end)
        files = tensor_files(tensor_dir, train_start, train_end)
    if len(files) < 2:
        raise RuntimeError(f"not enough tensor cache files in {tensor_dir}")

    validation_files: list[Path] = []
    if validation_start and validation_end:
        validation_files = tensor_files(tensor_dir, validation_start, validation_end)
        if len(validation_files) < 2:
            raise RuntimeError("validation range has fewer than two tensor days")

    train_source: list[Path] | list[DayTensor] = files
    validation_source: list[Path] | list[DayTensor] = validation_files
    if preload_workers > 0:
        train_source = preload_tensor_files(files, preload_workers)
        stats = compute_stats_from_days(train_source[:-1])
        if validation_files:
            validation_source = preload_tensor_files(validation_files, preload_workers)
    else:
        stats = compute_stats(files[:-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = dict(MODEL_CFG_V2 if arch == "v2" else MODEL_CFG)
    model = build_model(arch, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(f"trainable parameter count is outside competition bounds: {n_params}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.1)
    elif scheduler_name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"unknown scheduler: {scheduler_name}")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    logger.info(
        "training",
        arch=arch,
        objective=objective,
        scheduler=scheduler_name,
        parameters=n_params,
        files=len(files),
        validation_files=len(validation_files),
        preload_workers=preload_workers,
        device=str(device),
        gpus=torch.cuda.device_count(),
        epochs=epochs,
        amp=str(amp_dtype) if use_amp else "off",
    )

    best_loss = math.inf
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        day_order = list(range(len(train_source) - 1))
        random.shuffle(day_order)
        total, batches = 0.0, 0
        for idx in day_order:
            try:
                cur_item, nxt_item = train_source[idx], train_source[idx + 1]
                cur = cur_item if isinstance(cur_item, DayTensor) else load_npz_day(cur_item)
                nxt = nxt_item if isinstance(nxt_item, DayTensor) else load_npz_day(nxt_item)
                x_np, y_np = align_day_pair(cur, nxt)
                x_np = standardize(x_np, stats)
            except Exception as exc:
                logger.warning("skip train day", file=str(idx), error=str(exc))
                continue
            for xb, yb in batch_iter(x_np, y_np, batch_size=batch_size, shuffle=True):
                xb = {freq: val.to(device, non_blocking=True) for freq, val in xb.items()}
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred = model(xb)
                    loss = training_loss(pred.float(), yb.float(), objective)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                total += float(loss.detach().cpu())
                batches += 1
        avg = total / max(batches, 1)
        best_loss = min(best_loss, avg)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        metrics = (
            evaluate_tensor_files(raw_model, validation_source, stats, device, batch_size=max(batch_size, 1024))
            if validation_source
            else {}
        )
        score = (
            metrics.get("rank_ic_mean", 0.0)
            + metrics.get("pearson_mean", 0.0)
            + 0.01 * metrics.get("long_short_sharpe", 0.0)
            if metrics
            else -avg
        )
        record = {"epoch": epoch, "loss": avg, "score": score, **metrics}
        history.append(record)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
        logger.info(
            "epoch",
            epoch=epoch,
            loss=round(avg, 6),
            lr=round(opt.param_groups[0]["lr"], 8),
            validation={k: round(v, 6) for k, v in metrics.items()},
            best_epoch=best_epoch,
            elapsed=round(time.time() - t0, 2),
        )
        scheduler.step()

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    if best_state is not None:
        raw_model.load_state_dict(best_state)
    payload = {
        "state_dict": raw_model.state_dict(),
        "arch": arch,
        "model_cfg": model_cfg,
        "feature_cols": FEATURE_COLS,
        "freqs": list(FREQS),
        "expected_bars": EXPECTED_BARS,
        "patch_size": PATCH_SIZE if arch == "v2" else None,
        "stats": {freq: {"mean": stats[freq][0].tolist(), "std": stats[freq][1].tolist()} for freq in FREQS},
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_score": best_score,
        "history": history,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "objective": objective,
        "scheduler": scheduler_name,
        "preload_workers": preload_workers,
        "seed": seed,
        "trainable_parameters": n_params,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_model(payload, model_path)
    logger.info("model saved", path=model_path, best_epoch=best_epoch, best_score=best_score)
    return model_path


def train_ensemble_and_save(
    branch_configs: dict,
    datasources: dict | None = None,
    model_path: str = MODEL_PATH,
    cache_dir: str = "data/e2e_cache",
    tensor_dir: str = "data/tensor_cache",
    train_start: str = "2019-01-01",
    train_end: str = "2024-12-31",
    alpha_v1: float = 0.5,
    preload_workers: int = 0,
) -> str:
    """Train both branches from scratch and serialize them as one model."""
    if set(branch_configs) != {"v1", "v2"}:
        raise ValueError("branch_configs must contain exactly v1 and v2")
    model_file = Path(model_path)
    v1_path = model_file.with_name(model_file.name + ".branch_v1.tmp.json")
    v2_path = model_file.with_name(model_file.name + ".branch_v2.tmp.json")
    try:
        for arch, branch_path in (("v1", v1_path), ("v2", v2_path)):
            cfg = branch_configs[arch]
            train_and_save(
                datasources=datasources,
                model_path=str(branch_path),
                cache_dir=cache_dir,
                tensor_dir=tensor_dir,
                train_start=train_start,
                train_end=train_end,
                epochs=int(cfg["epochs"]),
                batch_size=int(cfg["batch_size"]),
                lr=float(cfg["learning_rate"]),
                weight_decay=float(cfg.get("weight_decay", 1e-4)),
                seed=int(cfg["seed"]),
                arch=arch,
                objective=cfg["objective"],
                scheduler_name=cfg["scheduler"],
                preload_workers=preload_workers,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return combine_models(str(v1_path), str(v2_path), model_path=model_path, alpha_v1=alpha_v1)
    finally:
        v1_path.unlink(missing_ok=True)
        v2_path.unlink(missing_ok=True)


def month_chunks(start: str, end: str, max_days: int = 15) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur <= stop:
        nxt = min(cur + pd.Timedelta(days=max_days) - pd.Timedelta(seconds=1), stop)
        yield cur, nxt
        cur = nxt + pd.Timedelta(seconds=1)


def query_platform_frames(datasources: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    if dai is None:
        raise RuntimeError("dai SDK is not available")
    frames = {}
    for freq in FREQS:
        table = table_for_freq(datasources, freq, local=False)
        cols = [c for c in BAR_COLUMNS if c != "instrument_id"]
        sql = f"SELECT {', '.join(cols)} FROM {table} ORDER BY instrument, date"
        df = dai.query(sql, filters={"date": [str(start), str(end)]}, compression=True).df()
        df = df.drop_duplicates(["date", "instrument"], keep="last")
        frames[freq] = df
    pool = dai.query(
        f"SELECT date::DATE::DATETIME AS date, instrument FROM {instruments_table(datasources)}",
        filters={"date": [str(start), str(end)]},
    ).df()
    pool["__day"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool = pool[["__day", "instrument"]].drop_duplicates()
    if pool.empty:
        raise RuntimeError(f"competition instrument pool is empty for {start} through {end}")
    for freq, frame in frames.items():
        frame = frame.copy()
        frame["__day"] = pd.to_datetime(frame["date"]).dt.normalize()
        frames[freq] = frame.merge(pool, on=["__day", "instrument"], how="inner").drop(columns="__day")
    return frames


def split_frames_by_day(frames: dict[str, pd.DataFrame]) -> dict[pd.Timestamp, dict[str, pd.DataFrame]]:
    by_day: dict[pd.Timestamp, dict[str, pd.DataFrame]] = {}
    for freq, df in frames.items():
        if df.empty:
            continue
        df = df.copy()
        df["__day"] = pd.to_datetime(df["date"]).dt.normalize()
        for day, sub in df.groupby("__day", sort=True):
            by_day.setdefault(pd.Timestamp(day), {})[freq] = sub.drop(columns=["__day"])
    return by_day


def predict_day(model: nn.Module, day_tensor: DayTensor, stats: dict[str, tuple[np.ndarray, np.ndarray]], device, batch_size: int):
    feats = standardize(day_tensor.features, stats)
    n = len(day_tensor.keys)
    if isinstance(model, MultiFreqEnsemble):
        batch_size = n  # Fusion normalization must see the complete daily cross-section.
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = {
                freq: torch.from_numpy(feats[freq][start : start + batch_size]).to(device)
                for freq in FREQS
            }
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).astype(np.float64)


def predict_with_saved_model(
    datasources: dict,
    start_date,
    end_date,
    model_path: str = MODEL_PATH,
    batch_size: int = 2048,
) -> pd.DataFrame:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"missing model file: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_model(model_path, map_location=device)
    stats = {
        freq: (
            np.asarray(ckpt["stats"][freq]["mean"], dtype=np.float32),
            np.asarray(ckpt["stats"][freq]["std"], dtype=np.float32),
        )
        for freq in FREQS
    }
    model = build_model(ckpt.get("arch", "v1"), ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    query_start = pd.Timestamp(start_date)
    query_end = inclusive_end_timestamp(end_date)
    rows = []
    for cs, ce in month_chunks(str(query_start), str(query_end), max_days=15):
        frames = query_platform_frames(datasources, cs, ce)
        for day, day_frames in split_frames_by_day(frames).items():
            if not all(freq in day_frames and not day_frames[freq].empty for freq in FREQS):
                continue
            try:
                tensor = make_day_tensor(day_frames, key_col="instrument")
                score = predict_day(model, tensor, stats, device, batch_size)
            except Exception as exc:
                logger.warning("skip predict day", day=str(day.date()), error=str(exc))
                continue
            rows.append(pd.DataFrame({"date": day, "instrument": tensor.keys.astype(str), "score": score}))
            logger.info("predicted day", day=str(day.date()), rows=len(score))
    if not rows:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    result = pd.concat(rows, ignore_index=True)
    result = (
        result.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])
        [["date", "instrument", "score"]]
        .reset_index(drop=True)
    )
    try:
        pool = dai.query(
            f"SELECT date::DATE::DATETIME AS date, instrument FROM {instruments_table(datasources)}",
            filters={"date": [str(query_start), str(query_end)]},
        ).df()
        pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        result = result.merge(
            pool.drop_duplicates(["date", "instrument"]),
            on=["date", "instrument"],
            how="inner",
        )[["date", "instrument", "score"]].reset_index(drop=True)
        expected = pool.drop_duplicates(["date", "instrument"]).groupby("date").size()
        actual = result.groupby("date").size().reindex(expected.index, fill_value=0)
        missing_days = actual.index[actual.eq(0)].strftime("%Y-%m-%d").tolist()
        low_coverage = (actual / expected).loc[lambda values: values < 0.60]
        if missing_days:
            raise RuntimeError(f"prediction is missing trading days: {missing_days[:10]}")
        if len(low_coverage):
            details = {str(day.date()): round(float(value), 4) for day, value in low_coverage.items()}
            raise RuntimeError(f"prediction coverage is below 60%: {details}")
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        logger.warning("pool intersection unavailable; using competition bar universe", error=str(exc))
    return result


def smoke_predict_from_cache(model_path: str, cache_dir: str, tensor_dir: str, start: str, end: str) -> pd.DataFrame:
    if len(tensor_files(tensor_dir, start, end)) == 0:
        build_tensor_cache(cache_dir, tensor_dir, start, end)
    ckpt = load_model(model_path, map_location="cpu")
    stats = {
        freq: (
            np.asarray(ckpt["stats"][freq]["mean"], dtype=np.float32),
            np.asarray(ckpt["stats"][freq]["std"], dtype=np.float32),
        )
        for freq in FREQS
    }
    model = build_model(ckpt.get("arch", "v1"), ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    out = []
    for path in tensor_files(tensor_dir, start, end):
        day = pd.Timestamp(datetime.strptime(path.stem, "%Y%m%d"))
        tensor = load_npz_day(path)
        score = predict_day(model, tensor, stats, torch.device("cpu"), batch_size=2048)
        out.append(pd.DataFrame({"date": day, "instrument_id": tensor.keys, "score": score}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-tensors")
    b.add_argument("--cache-dir", default="data/e2e_cache")
    b.add_argument("--tensor-dir", default="data/tensor_cache")
    b.add_argument("--start", default="2019-01-01")
    b.add_argument("--end", default="2024-12-31")
    b.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("train")
    t.add_argument("--cache-dir", default="data/e2e_cache")
    t.add_argument("--tensor-dir", default="data/tensor_cache")
    t.add_argument("--model-path", default=MODEL_PATH)
    t.add_argument("--train-start", default="2019-01-01")
    t.add_argument("--train-end", default="2024-12-31")
    t.add_argument("--validation-start")
    t.add_argument("--validation-end")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--arch", choices=["v1", "v2"], default="v2")
    t.add_argument("--objective", choices=["huber", "ic"], default="ic")
    t.add_argument("--scheduler", choices=["cosine", "constant"], default="cosine")
    t.add_argument("--preload-workers", type=int, default=0)

    v = sub.add_parser("validate")
    v.add_argument("--model-path", default=MODEL_PATH)
    v.add_argument("--tensor-dir", default="data/tensor_cache")
    v.add_argument("--start", default="2024-01-01")
    v.add_argument("--end", default="2024-12-31")
    v.add_argument("--batch-size", type=int, default=1024)

    s = sub.add_parser("smoke-predict")
    s.add_argument("--model-path", default=MODEL_PATH)
    s.add_argument("--cache-dir", default="data/e2e_cache")
    s.add_argument("--tensor-dir", default="data/tensor_cache")
    s.add_argument("--start", default="2024-01-02")
    s.add_argument("--end", default="2024-01-05")
    s.add_argument("--out", default="outputs/e2e_smoke_predict.parquet")

    args = ap.parse_args()
    if args.cmd == "build-tensors":
        build_tensor_cache(args.cache_dir, args.tensor_dir, args.start, args.end, overwrite=args.overwrite)
    elif args.cmd == "train":
        train_and_save(
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            tensor_dir=args.tensor_dir,
            train_start=args.train_start,
            train_end=args.train_end,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            arch=args.arch,
            objective=args.objective,
            scheduler_name=args.scheduler,
            preload_workers=args.preload_workers,
        )
    elif args.cmd == "validate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_model(args.model_path, map_location=device)
        stats = {
            freq: (
                np.asarray(ckpt["stats"][freq]["mean"], dtype=np.float32),
                np.asarray(ckpt["stats"][freq]["std"], dtype=np.float32),
            )
            for freq in FREQS
        }
        model = build_model(ckpt.get("arch", "v1"), ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = evaluate_tensor_files(
            model,
            tensor_files(args.tensor_dir, args.start, args.end),
            stats,
            device,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.cmd == "smoke-predict":
        df = smoke_predict_from_cache(args.model_path, args.cache_dir, args.tensor_dir, args.start, args.end)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(df.head())
        print({"rows": len(df), "days": df["date"].nunique() if len(df) else 0})


if __name__ == "__main__":
    main_cli()
