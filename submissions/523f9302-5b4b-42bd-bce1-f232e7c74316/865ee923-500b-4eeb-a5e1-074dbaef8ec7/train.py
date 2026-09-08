# -*- coding: utf-8 -*-
"""Raw-sequence neural models for the BigAlpha 2026 end-to-end track.

The model consumes raw one-minute fields only.  The only preprocessing is a
fixed per-field log transform followed by training-set mean/std scaling, both
explicitly allowed by the competition rules.  Cross-field interaction happens
inside the neural network; no handcrafted factor or rolling feature is built.

Public-board workflow:
    prepare_local.py -> run_local.py -> submission.ipynb + train.py + model JSON

Private-board workflow:
    the platform may call train_and_save(datasources) to retrain from scratch.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


log = logging.getLogger("e2e.raw_sequence")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

HERE = Path(__file__).resolve().parent
MODEL_PATH = str(HERE / "raw_e2e_model.json")
SCHEMA_VERSION = "raw_e2e_1m_v3"
PARAM_MIN = 100_000
PARAM_MAX = 100_000_000
JSON_SIZE_CAP = 49_000_000
GPU_CACHE_RESERVE_BYTES = 8 * 1024**3

# Twenty-one raw columns shared by the downloaded 2019-2024 Feather files and
# the cloud bar1m table.  The local package only retains three quote-size/order
# levels, while the official 2025-2026 export exposes quote prices at levels 1
# and 5.  Restricting the model to the semantic intersection avoids a silent
# local/cloud feature mismatch.
RAW_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "deal_number",
    "ask_price1",
    "bid_price1",
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
)

# Historical D9D artifacts used levels 1/5.  Keep those names loadable so an
# archived model JSON can still be inspected, while new caches use RAW_FIELDS.
SUPPORTED_RAW_FIELDS = frozenset(
    {
        *RAW_FIELDS,
        "ask_price5",
        "bid_price5",
        "ask_volume5",
        "bid_volume5",
        "ask_num_orders5",
        "bid_num_orders5",
    }
)

PRICE_FIELDS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "ask_price1",
        "ask_price5",
        "bid_price1",
        "bid_price5",
    }
)


@dataclass(frozen=True)
class Config:
    architecture: str = "hierarchical_mixer"
    fields: tuple[str, ...] = RAW_FIELDS
    bars_per_day: int = 240
    min_bars: int = 200
    min_xsec: int = 60
    target: str = "adj_c2c"
    history_days: int = 20
    patch: int = 12
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.0
    cs_blocks: int = 1
    head_width: int = 1024
    head_blocks: int = 3
    mixer_width: int = 1280
    mixer_hidden: int = 1280
    mixer_blocks: int = 2
    mixer_heads: int = 8
    mixer_cs_attention: bool = False
    intraday_width: int = 96
    intraday_blocks: int = 3
    daily_width: int = 192
    daily_blocks: int = 4
    daily_kernel: int = 3
    cs_rank: int = 32
    cs_weight: float = 0.15
    epochs: int = 24
    eval_every: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    mse_weight: float = 0.05
    rank_weight: float = 0.10
    huber_delta: float = 1.0
    training_mode: str = "date_stratified"
    loss: str = "huber_rank"
    batch_size: int = 384
    dates_per_batch: int = 6
    val_tail_days: int = 252
    val_tail_fraction: float = 0.15
    val_probe_days: int = 84
    purge_days: int = 2
    early_stop_patience: int = 6
    weight_ema_decay: float = 0.999
    grad_clip: float = 1.0
    seed: int = 1
    cloud_lookback_days: int = 2600


FROZEN_CONFIG = Config()
PRIVATE_ENSEMBLE_SEEDS: tuple[int, ...] = (1, 2)


def validate_config(cfg: Config) -> None:
    if cfg.architecture not in {
        "patchtst",
        "raw_swiglu_mixer",
        "hierarchical_mixer",
    }:
        raise ValueError(f"unsupported architecture: {cfg.architecture}")
    if len(cfg.fields) > 100:
        raise ValueError(f"raw field count {len(cfg.fields)} exceeds 100")
    if len(cfg.fields) != len(set(cfg.fields)):
        raise ValueError("raw fields contain duplicates")
    unknown = set(cfg.fields) - SUPPORTED_RAW_FIELDS
    if unknown:
        raise ValueError(f"unknown raw fields: {sorted(unknown)}")
    if cfg.bars_per_day % cfg.patch:
        raise ValueError("bars_per_day must be divisible by patch")
    if cfg.target not in {"adj_c2c", "raw_c2c", "open_to_open"}:
        raise ValueError(f"unsupported target: {cfg.target}")
    if cfg.history_days <= 0:
        raise ValueError("history_days must be positive")
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if cfg.cs_blocks < 0:
        raise ValueError("cs_blocks must be non-negative")
    if cfg.head_width <= 0 or cfg.head_blocks < 0:
        raise ValueError("head width/blocks must be positive/non-negative")
    if cfg.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if cfg.mixer_width <= 0 or cfg.mixer_hidden <= 0 or cfg.mixer_blocks < 0:
        raise ValueError("mixer dimensions/blocks must be positive/non-negative")
    if cfg.mixer_width % cfg.mixer_heads:
        raise ValueError("mixer_width must be divisible by mixer_heads")
    if cfg.intraday_width <= 0 or cfg.intraday_blocks <= 0:
        raise ValueError("intraday dimensions must be positive")
    if cfg.daily_width <= 0 or cfg.daily_blocks <= 0:
        raise ValueError("daily dimensions must be positive")
    if cfg.daily_kernel <= 0 or cfg.daily_kernel % 2 == 0:
        raise ValueError("daily_kernel must be a positive odd integer")
    if cfg.cs_rank <= 0:
        raise ValueError("cs_rank must be positive")
    if cfg.training_mode not in {
        "random_samples",
        "full_dates",
        "date_stratified",
    }:
        raise ValueError(f"unsupported training mode: {cfg.training_mode}")
    if cfg.loss not in {"mse", "pearson", "huber_rank"}:
        raise ValueError(f"unsupported loss: {cfg.loss}")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.dates_per_batch <= 0:
        raise ValueError("dates_per_batch must be positive")
    if (
        cfg.val_tail_days <= 0
        or cfg.val_probe_days <= 0
        or not (0.0 < cfg.val_tail_fraction < 0.5)
    ):
        raise ValueError("validation tail settings are invalid")
    if cfg.purge_days < 0 or cfg.early_stop_patience <= 0:
        raise ValueError("purge/patience settings are invalid")
    if not (0.0 <= cfg.weight_ema_decay < 1.0):
        raise ValueError("weight_ema_decay must be in [0, 1)")
    if cfg.warmup_steps < 0 or cfg.min_learning_rate < 0:
        raise ValueError("learning-rate schedule values must be non-negative")


def config_hash(cfg: Config) -> str:
    payload = json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def data_config_hash(cfg: Config) -> str:
    """Hash only preprocessing choices that determine the tensor cache."""
    payload = {
        "fields": list(cfg.fields),
        "bars_per_day": cfg.bars_per_day,
        "min_bars": cfg.min_bars,
        "price_fields": sorted(PRICE_FIELDS.intersection(cfg.fields)),
        "schema_version": SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def transform_field(field: str, values: np.ndarray) -> np.ndarray:
    """Allowed fixed per-field transform; never mixes two source columns."""
    values = np.asarray(values, dtype=np.float32)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if field in PRICE_FIELDS:
        valid = values > 0.0
        out[valid] = np.log(values[valid])
    else:
        valid = values >= 0.0
        out[valid] = np.log1p(values[valid])
    return out


def minute_slots(values: pd.Series) -> np.ndarray:
    """Map Chinese A-share one-minute bar end-times to fixed slots 0..239."""
    dt = pd.DatetimeIndex(pd.to_datetime(values))
    minute = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
    morning = minute - (9 * 60 + 31)
    afternoon = 120 + minute - (13 * 60 + 1)
    return np.where(minute <= 11 * 60 + 30, morning, afternoon).astype(np.int16)


def canonicalize_local_frame(
    frame: pd.DataFrame,
    cfg: Config = FROZEN_CONFIG,
) -> pd.DataFrame:
    """Convert the downloaded Feather schema to the cloud bar1m schema.

    Local prices and ``amount`` are stored as integer cents, OHLC uses ``-1``
    as a missing sentinel, and stock identity is ``instrument_id``.  Quote
    sizes, volumes, deal counts, and order counts are already counts and must
    not be divided by 100.  ``adjust_factor`` is retained only for labels.
    """
    validate_config(cfg)
    required = {"date", "instrument_id", "adjust_factor", *cfg.fields}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"local raw frame missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"]),
            "instrument": frame["instrument_id"].astype(str),
        }
    )
    for field in cfg.fields:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(
            np.float64, copy=True
        )
        if field in PRICE_FIELDS:
            values[values <= 0.0] = np.nan
            values *= 0.01
        elif field == "amount":
            values[values < 0.0] = np.nan
            values *= 0.01
        else:
            values[values < 0.0] = np.nan
        out[field] = values

    adjust = pd.to_numeric(
        frame["adjust_factor"], errors="coerce"
    ).to_numpy(np.float64, copy=True)
    adjust[adjust <= 0.0] = np.nan
    out["adjust_factor"] = adjust
    return out


def validate_cloud_frame(
    frame: pd.DataFrame,
    cfg: Config = FROZEN_CONFIG,
    require_pre_close: bool = False,
) -> pd.DataFrame:
    """Validate/canonicalize a cloud-style frame without changing its units."""
    required = {"date", "instrument", *cfg.fields}
    if require_pre_close:
        required.add("pre_close")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cloud raw frame missing columns: {sorted(missing)}")
    columns = ["date", "instrument", *cfg.fields]
    if "pre_close" in frame.columns:
        columns.append("pre_close")
    out = frame[columns].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype(str)
    for field in cfg.fields:
        values = pd.to_numeric(out[field], errors="coerce").to_numpy(
            np.float64, copy=True
        )
        if field in PRICE_FIELDS:
            values[values <= 0.0] = np.nan
        else:
            values[values < 0.0] = np.nan
        out[field] = values
    if "pre_close" in out.columns:
        values = pd.to_numeric(out["pre_close"], errors="coerce").to_numpy(
            np.float64, copy=True
        )
        values[values <= 0.0] = np.nan
        out["pre_close"] = values
    return out


def frame_to_tensor(
    frame: pd.DataFrame,
    cfg: Config = FROZEN_CONFIG,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw long bars -> stock-day tensor without creating engineered features."""
    validate_config(cfg)
    required = {"date", "instrument", "open", "close", *cfg.fields}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"raw frame missing columns: {sorted(missing)}")

    columns = list(
        dict.fromkeys(["date", "instrument", "open", "close", *cfg.fields])
    )
    columns.extend(
        field
        for field in ("adjust_factor", "pre_close")
        if field in frame.columns
    )
    work = frame[columns].copy()
    work["date"] = pd.to_datetime(work["date"])
    pos = minute_slots(work["date"])
    slot_ok = (pos >= 0) & (pos < cfg.bars_per_day)
    if not bool(slot_ok.all()):
        work = work.loc[slot_ok].reset_index(drop=True)
        pos = pos[slot_ok]
    if work.empty:
        return (
            pd.DataFrame(
                columns=[
                    "instrument",
                    "date",
                    "row",
                    "n_bars",
                    "day_open",
                    "day_close",
                ]
            ),
            np.empty((0, cfg.bars_per_day, len(cfg.fields)), dtype=np.float16),
        )

    days_raw = work["date"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    inst_codes, inst_values = pd.factorize(
        work["instrument"].astype(str), sort=True
    )
    day_codes, day_values = pd.factorize(days_raw, sort=True)
    combo_raw = inst_codes.astype(np.int64) * len(day_values) + day_codes
    combo, combo_values = pd.factorize(combo_raw, sort=True)
    n_samples = len(combo_values)

    pair_code = combo.astype(np.int64) * cfg.bars_per_day + pos.astype(np.int64)
    if len(np.unique(pair_code)) != len(pair_code):
        raise ValueError("duplicate (instrument, day, minute-slot) rows")

    x = np.full(
        (n_samples, cfg.bars_per_day, len(cfg.fields)),
        np.nan,
        dtype=np.float16,
    )
    for column_idx, field in enumerate(cfg.fields):
        transformed = transform_field(field, work[field].to_numpy())
        x[combo, pos, column_idx] = transformed.astype(np.float16)

    def reduce_daily(column: str, first: bool) -> np.ndarray:
        panel = np.full(
            (n_samples, cfg.bars_per_day), np.nan, dtype=np.float32
        )
        raw = pd.to_numeric(work[column], errors="coerce").to_numpy(
            np.float32, copy=True
        )
        raw[raw <= 0.0] = np.nan
        panel[combo, pos] = raw
        result = np.full(n_samples, np.nan, dtype=np.float32)
        slots = (
            range(cfg.bars_per_day - 1, -1, -1)
            if first
            else range(cfg.bars_per_day)
        )
        for slot in slots:
            current = panel[:, slot]
            result = np.where(np.isfinite(current), current, result)
        return result

    day_open = reduce_daily("open", first=True)
    day_close = reduce_daily("close", first=False)

    counts = np.bincount(combo, minlength=n_samples)
    inst_index = (np.asarray(combo_values) // len(day_values)).astype(np.int64)
    day_index = (np.asarray(combo_values) % len(day_values)).astype(np.int64)
    meta = pd.DataFrame(
        {
            "instrument": np.asarray(inst_values, dtype=object)[inst_index],
            "date": pd.to_datetime(np.asarray(day_values)[day_index]),
            "n_bars": counts.astype(np.int16),
            "day_open": day_open,
            "day_close": day_close,
        }
    )
    for label_column in ("adjust_factor", "pre_close"):
        if label_column in work.columns:
            meta[label_column] = reduce_daily(label_column, first=False)
    keep = (
        (counts >= cfg.min_bars)
        & np.isfinite(day_open)
        & np.isfinite(day_close)
    )
    x = x[keep]
    meta = meta.loc[keep].reset_index(drop=True)
    meta["row"] = np.arange(len(meta), dtype=np.int32)
    output_columns = [
        "instrument",
        "date",
        "row",
        "n_bars",
        "day_open",
        "day_close",
    ]
    output_columns.extend(
        column
        for column in ("adjust_factor", "pre_close")
        if column in meta.columns
    )
    return meta[output_columns], x


def chunk_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Streaming moments for transformed fields, ignoring missing values."""
    n_fields = x.shape[-1]
    count = np.zeros(n_fields, dtype=np.int64)
    total = np.zeros(n_fields, dtype=np.float64)
    total2 = np.zeros(n_fields, dtype=np.float64)
    for idx in range(n_fields):
        values = np.asarray(x[..., idx], dtype=np.float32)
        finite = np.isfinite(values)
        selected = values[finite].astype(np.float64, copy=False)
        count[idx] = len(selected)
        total[idx] = selected.sum(dtype=np.float64)
        total2[idx] = np.square(selected).sum(dtype=np.float64)
    return count, total, total2


def moments_to_stats(
    count: np.ndarray,
    total: np.ndarray,
    total2: np.ndarray,
) -> dict[str, list[float]]:
    if (count <= 0).any():
        raise ValueError("at least one raw field has no finite observations")
    mean = total / count
    variance = np.maximum(total2 / count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return {"mean": mean.tolist(), "std": std.tolist()}


def standardize_tensor(x: np.ndarray, stats: Mapping[str, list[float]]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    z = (np.asarray(x, dtype=np.float32) - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32, copy=False
    )


def fit_stats_from_metadata(
    chunks: Mapping[str, np.ndarray],
    meta: pd.DataFrame,
    train_end_date: pd.Timestamp,
    batch_rows: int = 4096,
) -> dict[str, list[float]]:
    """Fit field moments only on stock-days available to the training split."""
    required = {"date", "chunk", "row"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing stats columns: {sorted(missing)}")
    selected = meta.loc[
        pd.to_datetime(meta["date"]).dt.normalize()
        <= pd.Timestamp(train_end_date).normalize()
    ]
    if selected.empty:
        raise ValueError("no metadata rows in the training statistics window")
    n_fields = next(iter(chunks.values())).shape[-1]
    total_count = np.zeros(n_fields, dtype=np.int64)
    total_sum = np.zeros(n_fields, dtype=np.float64)
    total_sum2 = np.zeros(n_fields, dtype=np.float64)
    for chunk, group in selected.groupby("chunk", sort=True):
        name = str(chunk)
        if name not in chunks:
            raise KeyError(f"metadata references missing chunk {name}")
        rows = group["row"].to_numpy(np.int64, copy=True)
        for start in range(0, len(rows), batch_rows):
            values = np.asarray(chunks[name][rows[start : start + batch_rows]])
            count, total, total2 = chunk_moments(values)
            total_count += count
            total_sum += total
            total_sum2 += total2
    return moments_to_stats(total_count, total_sum, total_sum2)


def _normal_scores(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average").to_numpy(np.float64)
    q = (ranks - 0.5) / len(ranks)
    qt = torch.from_numpy(q).clamp(1e-6, 1.0 - 1e-6)
    z = (math.sqrt(2.0) * torch.erfinv(2.0 * qt - 1.0)).numpy()
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float32)


def build_open_to_open_targets(
    meta: pd.DataFrame,
    cfg: Config = FROZEN_CONFIG,
) -> pd.DataFrame:
    """Score date t target = open[t+2] / open[t+1] - 1.

    Label construction is unrestricted by the competition.  This function uses
    no future value in model inputs; it only constructs the supervised target.
    """
    required = {
        "instrument",
        "date",
        "day_open",
        "chunk",
        "row",
    }
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    out = meta.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    all_days = np.sort(out["date"].unique())
    day_number = {pd.Timestamp(day): idx for idx, day in enumerate(all_days)}
    out["day_number"] = out["date"].map(day_number).astype(np.int32)
    label_date = np.full(
        len(out), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    label_position = out["day_number"].to_numpy(np.int64) + 2
    within_calendar = label_position < len(all_days)
    label_date[within_calendar] = np.asarray(all_days)[label_position[within_calendar]]
    out["label_date"] = pd.to_datetime(label_date)

    open_map = pd.Series(
        out["day_open"].to_numpy(np.float64),
        index=pd.MultiIndex.from_arrays(
            [out["instrument"].astype(str), out["day_number"]]
        ),
    )
    idx1 = pd.MultiIndex.from_arrays(
        [out["instrument"].astype(str), out["day_number"] + 1]
    )
    idx2 = pd.MultiIndex.from_arrays(
        [out["instrument"].astype(str), out["day_number"] + 2]
    )
    open1 = open_map.reindex(idx1).to_numpy(np.float64)
    open2 = open_map.reindex(idx2).to_numpy(np.float64)
    raw = open2 / open1 - 1.0
    valid = (
        np.isfinite(raw)
        & np.isfinite(open1)
        & np.isfinite(open2)
        & (open1 > 0.0)
        & (open2 > 0.0)
        & (np.abs(raw) <= 0.35)
    )
    out["target_raw"] = raw
    out = out.loc[valid].copy()
    if out.empty:
        raise ValueError("no valid open-to-open targets")
    out["target"] = np.nan
    for _, index in out.groupby("date", sort=True).groups.items():
        if len(index) < cfg.min_xsec:
            continue
        out.loc[index, "target"] = _normal_scores(out.loc[index, "target_raw"])
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(np.float32)
    return out[
        [
            "date",
            "label_date",
            "instrument",
            "chunk",
            "row",
            "n_bars",
            "target_raw",
            "target",
        ]
    ]


def build_close_to_close_targets(
    meta: pd.DataFrame,
    cfg: Config = FROZEN_CONFIG,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Build score-date ``t`` labels from close[t] to close[t+1].

    The downloaded data uses ``close * adjust_factor``.  On the cloud table,
    where ``adjust_factor`` is unavailable, the equivalent one-day adjusted
    return is ``close[t+1] / pre_close[t+1] - 1``.
    """
    required = {
        "instrument",
        "date",
        "day_close",
        "chunk",
        "row",
    }
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    if adjusted and not ({"adjust_factor", "pre_close"} & set(meta.columns)):
        raise ValueError(
            "adjusted close target requires adjust_factor (local) or pre_close (cloud)"
        )

    out = meta.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    all_days = np.sort(out["date"].unique())
    day_number = {pd.Timestamp(day): idx for idx, day in enumerate(all_days)}
    out["day_number"] = out["date"].map(day_number).astype(np.int32)
    label_date = np.full(
        len(out), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    label_position = out["day_number"].to_numpy(np.int64) + 1
    within_calendar = label_position < len(all_days)
    label_date[within_calendar] = np.asarray(all_days)[label_position[within_calendar]]
    out["label_date"] = pd.to_datetime(label_date)
    identity = out["instrument"].astype(str)
    lookup_index = pd.MultiIndex.from_arrays([identity, out["day_number"]])
    next_index = pd.MultiIndex.from_arrays([identity, out["day_number"] + 1])

    close = pd.to_numeric(out["day_close"], errors="coerce").to_numpy(
        np.float64, copy=True
    )
    close_map = pd.Series(close, index=lookup_index)
    next_close = close_map.reindex(next_index).to_numpy(np.float64)
    if not adjusted:
        raw = next_close / close - 1.0
    else:
        raw = np.full(len(out), np.nan, dtype=np.float64)
        if "adjust_factor" in out.columns:
            factor = pd.to_numeric(
                out["adjust_factor"], errors="coerce"
            ).to_numpy(np.float64, copy=True)
            adjusted_close = close * factor
            adjusted_map = pd.Series(adjusted_close, index=lookup_index)
            next_adjusted = adjusted_map.reindex(next_index).to_numpy(np.float64)
            local_raw = next_adjusted / adjusted_close - 1.0
            local_valid = (
                np.isfinite(local_raw)
                & np.isfinite(adjusted_close)
                & (adjusted_close > 0.0)
            )
            raw[local_valid] = local_raw[local_valid]
        if "pre_close" in out.columns:
            pre_close = pd.to_numeric(
                out["pre_close"], errors="coerce"
            ).to_numpy(np.float64, copy=True)
            pre_close_map = pd.Series(pre_close, index=lookup_index)
            next_pre_close = pre_close_map.reindex(next_index).to_numpy(np.float64)
            cloud_raw = next_close / next_pre_close - 1.0
            cloud_valid = np.isfinite(cloud_raw) & (next_pre_close > 0.0)
            raw[~np.isfinite(raw) & cloud_valid] = cloud_raw[
                ~np.isfinite(raw) & cloud_valid
            ]

    valid = (
        np.isfinite(raw)
        & np.isfinite(close)
        & np.isfinite(next_close)
        & (close > 0.0)
        & (next_close > 0.0)
        & (np.abs(raw) <= 0.35)
    )
    out["target_raw"] = raw
    out = out.loc[valid].copy()
    if out.empty:
        raise ValueError("no valid close-to-close targets")
    out["target"] = np.nan
    for _, index in out.groupby("date", sort=True).groups.items():
        if len(index) < cfg.min_xsec:
            continue
        out.loc[index, "target"] = _normal_scores(out.loc[index, "target_raw"])
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(np.float32)
    return out[
        [
            "date",
            "label_date",
            "day_number",
            "instrument",
            "chunk",
            "row",
            "n_bars",
            "target_raw",
            "target",
        ]
    ]


def build_targets(
    meta: pd.DataFrame,
    cfg: Config = FROZEN_CONFIG,
    target: Optional[str] = None,
) -> pd.DataFrame:
    target_name = target or cfg.target
    if target_name == "adj_c2c":
        return build_close_to_close_targets(meta, cfg, adjusted=True)
    if target_name == "raw_c2c":
        return build_close_to_close_targets(meta, cfg, adjusted=False)
    if target_name == "open_to_open":
        return build_open_to_open_targets(meta, cfg)
    raise ValueError(f"unsupported target: {target_name}")


@dataclass(frozen=True)
class HistoryIndex:
    """Compact zero-copy references from samples to monthly memmap rows."""

    chunk_names: tuple[str, ...]
    chunk_codes: np.ndarray  # [sample, history_day], int16
    rows: np.ndarray  # [sample, history_day], int32/int64

    def __post_init__(self) -> None:
        if self.chunk_codes.shape != self.rows.shape:
            raise ValueError("history chunk_codes/rows shape mismatch")
        if self.chunk_codes.ndim != 2:
            raise ValueError("history arrays must be [sample, history_day]")

    def take(self, indices: np.ndarray) -> "HistoryIndex":
        return HistoryIndex(
            self.chunk_names,
            self.chunk_codes[indices],
            self.rows[indices],
        )


@dataclass(frozen=True)
class ResidentChunkStore:
    """Raw monthly tensors concatenated on one device for indexed gathers."""

    chunk_names: tuple[str, ...]
    values: torch.Tensor
    row_offsets: torch.Tensor
    total_bytes: int

    @property
    def device(self) -> torch.device:
        return self.values.device


def build_resident_chunk_store(
    chunks: Mapping[str, np.ndarray],
    chunk_names: tuple[str, ...],
    device: torch.device,
    reserve_bytes: int = GPU_CACHE_RESERVE_BYTES,
) -> Optional[ResidentChunkStore]:
    """Copy the immutable float16 cache to a device when memory permits.

    The original CPU/memmap path remains the fallback for smaller official
    GPUs.  Keeping the six-year raw tensor resident avoids rebuilding every
    20-day history batch through random CPU memmap gathers.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    names = tuple(str(name) for name in chunk_names)
    if not names:
        raise ValueError("resident chunk store requires at least one chunk")
    arrays = []
    tail_shape: Optional[tuple[int, ...]] = None
    total_rows = 0
    total_bytes = 0
    offsets = []
    for name in names:
        if name not in chunks:
            raise KeyError(f"resident cache references missing chunk {name}")
        array = np.asarray(chunks[name])
        if array.dtype != np.float16:
            raise ValueError(f"resident chunk {name} must be float16")
        if tail_shape is None:
            tail_shape = tuple(int(value) for value in array.shape[1:])
        elif tuple(array.shape[1:]) != tail_shape:
            raise ValueError("resident chunks have inconsistent tensor shapes")
        offsets.append(total_rows)
        total_rows += int(len(array))
        total_bytes += int(array.nbytes)
        arrays.append(array)
    assert tail_shape is not None

    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        required = total_bytes + max(0, int(reserve_bytes))
        if required > free_bytes:
            log.info(
                "resident GPU cache disabled: raw=%.2fGiB reserve=%.2fGiB "
                "free=%.2fGiB",
                total_bytes / 2**30,
                reserve_bytes / 2**30,
                free_bytes / 2**30,
            )
            return None

    started = time.time()
    try:
        values = torch.empty(
            (total_rows, *tail_shape),
            dtype=torch.float16,
            device=device,
        )
        for offset, array in zip(offsets, arrays):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given NumPy array is not writable",
                )
                source = torch.from_numpy(array)
            values[offset : offset + len(array)].copy_(source)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log.warning("resident GPU cache allocation failed; using CPU gathers")
        return None
    row_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    log.info(
        "resident raw cache ready: device=%s rows=%d bytes=%.2fGiB elapsed=%.1fs",
        device,
        total_rows,
        total_bytes / 2**30,
        time.time() - started,
    )
    return ResidentChunkStore(names, values, row_offsets, total_bytes)


def load_resident_history_batch(
    store: ResidentChunkStore,
    history: HistoryIndex,
    sample_indices: np.ndarray,
) -> torch.Tensor:
    """Gather ``[B,D,240,F]`` directly from a resident device tensor."""
    if tuple(history.chunk_names) != store.chunk_names:
        raise ValueError("resident store/history chunk names do not match")
    indices = np.asarray(sample_indices, dtype=np.int64)
    selected_codes = np.asarray(history.chunk_codes[indices], dtype=np.int64)
    selected_rows = np.asarray(history.rows[indices], dtype=np.int64)
    if (selected_codes < 0).any() or (selected_rows < 0).any():
        raise ValueError("history index contains a missing row")
    codes = torch.as_tensor(
        selected_codes,
        dtype=torch.long,
        device=store.device,
    )
    rows = torch.as_tensor(
        selected_rows,
        dtype=torch.long,
        device=store.device,
    )
    return store.values[rows + store.row_offsets[codes]]


def build_history_index(
    meta: pd.DataFrame,
    anchors: pd.DataFrame,
    history_days: int,
) -> tuple[pd.DataFrame, HistoryIndex]:
    """Link every anchor to consecutive stock-days without copying tensors.

    Trading-day ordinals are global across the supplied metadata, so a sample
    is retained only when the same instrument has all ``history_days`` rows.
    This prevents new listings and long suspensions from becoming mostly
    imputed histories while keeping the monthly tensors memory-mapped.
    """
    if history_days <= 0:
        raise ValueError("history_days must be positive")
    required_meta = {"date", "instrument", "chunk", "row"}
    missing = required_meta - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing history columns: {sorted(missing)}")
    required_anchor = {"date", "instrument"}
    missing = required_anchor - set(anchors.columns)
    if missing:
        raise ValueError(f"anchors missing history columns: {sorted(missing)}")

    metadata = meta[["date", "instrument", "chunk", "row"]].copy()
    metadata["date"] = pd.to_datetime(metadata["date"]).dt.normalize()
    metadata["instrument"] = metadata["instrument"].astype(str)
    if metadata.duplicated(["date", "instrument"]).any():
        raise ValueError("duplicate (date, instrument) metadata rows")
    all_days = np.sort(metadata["date"].unique())
    day_map = {pd.Timestamp(day): idx for idx, day in enumerate(all_days)}
    metadata["day_number"] = metadata["date"].map(day_map).astype(np.int32)
    lookup = pd.MultiIndex.from_arrays(
        [metadata["instrument"], metadata["day_number"]]
    )
    chunk_codes, chunk_names = pd.factorize(
        metadata["chunk"].astype(str), sort=True
    )
    meta_rows = metadata["row"].to_numpy(np.int64, copy=True)

    selected = anchors.copy().reset_index(drop=True)
    selected["date"] = pd.to_datetime(selected["date"]).dt.normalize()
    selected["instrument"] = selected["instrument"].astype(str)
    anchor_day = selected["date"].map(day_map)
    known_day = anchor_day.notna().to_numpy()
    anchor_day_values = anchor_day.fillna(-1).to_numpy(np.int32)
    identity = selected["instrument"].to_numpy(dtype=object)
    codes = np.full((len(selected), history_days), -1, dtype=np.int16)
    rows = np.full((len(selected), history_days), -1, dtype=np.int64)
    valid = known_day.copy()

    for column, lag in enumerate(range(history_days - 1, -1, -1)):
        requested = pd.MultiIndex.from_arrays(
            [identity, anchor_day_values - lag]
        )
        positions = lookup.get_indexer(requested)
        present = positions >= 0
        valid &= present
        safe = np.where(present, positions, 0)
        codes[:, column] = np.where(
            present, chunk_codes[safe], -1
        ).astype(np.int16)
        rows[:, column] = np.where(present, meta_rows[safe], -1)

    selected = selected.loc[valid].reset_index(drop=True)
    index = HistoryIndex(
        tuple(str(value) for value in chunk_names),
        codes[valid],
        rows[valid].astype(np.int32, copy=False),
    )
    return selected, index


def load_history_batch(
    chunks: Mapping[str, np.ndarray],
    history: HistoryIndex,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Gather ``[B,D,240,F]`` float16 inputs from monthly memmaps."""
    indices = np.asarray(sample_indices, dtype=np.int64)
    selected_codes = history.chunk_codes[indices]
    selected_rows = history.rows[indices]
    first = next(iter(chunks.values()))
    batch = np.empty(
        (len(indices), selected_codes.shape[1], *first.shape[1:]),
        dtype=np.float16,
    )
    flat_codes = selected_codes.reshape(-1)
    flat_rows = selected_rows.reshape(-1)
    flat_batch = batch.reshape(-1, *first.shape[1:])
    for code in np.unique(flat_codes):
        if code < 0:
            raise ValueError("history index contains a missing row")
        positions = np.flatnonzero(flat_codes == code)
        flat_batch[positions] = np.asarray(
            chunks[history.chunk_names[int(code)]][flat_rows[positions]],
            dtype=np.float16,
        )
    return batch


class ResidualStockBlock(nn.Module):
    """High-capacity model-internal stock representation block."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        # Start every residual branch as an exact identity.  This keeps the
        # large memorization head numerically stable while its branches learn.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = nn.functional.gelu(self.fc1(h))
        return x + self.dropout(self.fc2(h))


class SwiGLUStockBlock(nn.Module):
    """Zero-initialized gated residual block inspired by modern MLP mixers."""

    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gate_up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(self.norm(x)).chunk(2, dim=-1)
        update = nn.functional.silu(gate) * value
        return x + self.dropout(self.down(update))


class RawPatchTST(nn.Module):
    """Channel-independent PatchTST plus permutation-equivariant stock attention."""

    def __init__(self, cfg: Config = FROZEN_CONFIG):
        super().__init__()
        validate_config(cfg)
        self.cfg = cfg
        self.n_patches = cfg.bars_per_day // cfg.patch
        self.patch_embed = nn.Linear(cfg.patch, cfg.d_model)
        self.patch_pos = nn.Parameter(
            torch.zeros(1, self.n_patches, cfg.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, cfg.n_layers)
        self.field_mix = nn.Linear(len(cfg.fields) * cfg.d_model, cfg.d_model)
        self.field_norm = nn.LayerNorm(cfg.d_model)
        self.cs_attention = nn.ModuleList()
        self.cs_norm = nn.ModuleList()
        for _ in range(cfg.cs_blocks):
            self.cs_attention.append(
                nn.MultiheadAttention(
                    cfg.d_model,
                    cfg.n_heads,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
            )
            self.cs_norm.append(nn.LayerNorm(cfg.d_model))
        self.stock_expand = nn.Linear(cfg.d_model, cfg.head_width)
        self.stock_blocks = nn.ModuleList(
            ResidualStockBlock(cfg.head_width, cfg.dropout)
            for _ in range(cfg.head_blocks)
        )
        self.head_norm = nn.LayerNorm(cfg.head_width)
        self.head = nn.Linear(cfg.head_width, 1)

    def encode_stocks(self, x: torch.Tensor) -> torch.Tensor:
        n_stocks, n_bars, n_fields = x.shape
        if n_bars != self.cfg.bars_per_day or n_fields != len(self.cfg.fields):
            raise ValueError(
                f"expected [N,{self.cfg.bars_per_day},{len(self.cfg.fields)}], "
                f"got {tuple(x.shape)}"
            )
        h = x.permute(0, 2, 1).reshape(
            n_stocks * n_fields, self.n_patches, self.cfg.patch
        )
        h = self.temporal(self.patch_embed(h) + self.patch_pos).mean(dim=1)
        h = h.reshape(n_stocks, n_fields * self.cfg.d_model)
        return self.field_norm(self.field_mix(h))

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.encode_stocks(x)
        if self.cs_attention:
            hb = h.unsqueeze(0)
            mask = None if pad_mask is None else pad_mask.unsqueeze(0)
            for attention, norm in zip(self.cs_attention, self.cs_norm):
                update, _ = attention(
                    hb,
                    hb,
                    hb,
                    key_padding_mask=mask,
                    need_weights=False,
                )
                hb = norm(hb + update)
            h = hb.squeeze(0)
        h = nn.functional.gelu(self.stock_expand(h))
        for block in self.stock_blocks:
            h = block(h)
        score = self.head(self.head_norm(h)).squeeze(-1)
        if pad_mask is not None:
            score = score.masked_fill(pad_mask, 0.0)
        return score


class RawSwiGLUMixer(nn.Module):
    """All-MLP raw sequence mixer with optional stock-axis attention.

    The first learned projection sees the exact ordered 240 x F raw tensor.
    It is a model layer, not preprocessing: no return, spread, rolling value,
    identifier, or calendar feature is constructed outside the network.
    """

    def __init__(self, cfg: Config = FROZEN_CONFIG):
        super().__init__()
        validate_config(cfg)
        self.cfg = cfg
        input_width = cfg.bars_per_day * len(cfg.fields)
        self.input_proj = nn.Linear(input_width, cfg.mixer_width)
        self.blocks = nn.ModuleList(
            SwiGLUStockBlock(
                cfg.mixer_width,
                cfg.mixer_hidden,
                cfg.dropout,
            )
            for _ in range(cfg.mixer_blocks)
        )
        if cfg.mixer_cs_attention:
            self.cs_attention = nn.MultiheadAttention(
                cfg.mixer_width,
                cfg.mixer_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.cs_norm = nn.LayerNorm(cfg.mixer_width)
        else:
            self.cs_attention = None
            self.cs_norm = None
        self.head_norm = nn.LayerNorm(cfg.mixer_width)
        self.head = nn.Linear(cfg.mixer_width, 1)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_stocks, n_bars, n_fields = x.shape
        if n_bars != self.cfg.bars_per_day or n_fields != len(self.cfg.fields):
            raise ValueError(
                f"expected [N,{self.cfg.bars_per_day},{len(self.cfg.fields)}], "
                f"got {tuple(x.shape)}"
            )
        h = nn.functional.gelu(self.input_proj(x.reshape(n_stocks, -1)))
        for block in self.blocks:
            h = block(h)
        if self.cs_attention is not None:
            hb = h.unsqueeze(0)
            mask = None if pad_mask is None else pad_mask.unsqueeze(0)
            update, _ = self.cs_attention(
                hb,
                hb,
                hb,
                key_padding_mask=mask,
                need_weights=False,
            )
            h = self.cs_norm(h + update.squeeze(0))
        score = self.head(self.head_norm(h)).squeeze(-1)
        if pad_mask is not None:
            score = score.masked_fill(pad_mask, 0.0)
        return score


class TemporalMixerBlock(nn.Module):
    """Linear-complexity temporal block with depthwise convolution + SwiGLU."""

    def __init__(
        self,
        width: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.temporal_norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation,
            groups=width,
        )
        self.temporal_out = nn.Linear(width, width)
        self.channel_norm = nn.LayerNorm(width)
        self.gate_up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(2 * width, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.temporal_out.weight)
        nn.init.zeros_(self.temporal_out.bias)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_norm(x).transpose(1, 2)
        temporal = self.depthwise(temporal).transpose(1, 2)
        x = x + self.dropout(self.temporal_out(nn.functional.silu(temporal)))
        gate, value = self.gate_up(self.channel_norm(x)).chunk(2, dim=-1)
        update = nn.functional.silu(gate) * value
        return x + self.dropout(self.down(update))


class HierarchicalMinuteMixer(nn.Module):
    """Patch one-minute bars, mix across days, then aggregate the cross-section.

    Complexity is linear in bars, history length, and stock count.  The
    low-rank date-group mean is permutation equivariant and avoids the
    quadratic ``N_stock²`` memory cost of full cross-sectional attention.
    """

    def __init__(self, cfg: Config = FROZEN_CONFIG):
        super().__init__()
        validate_config(cfg)
        self.cfg = cfg
        self.n_patches = cfg.bars_per_day // cfg.patch
        patch_width = cfg.patch * len(cfg.fields)
        self.patch_projection = nn.Linear(patch_width, cfg.intraday_width)
        self.patch_position = nn.Parameter(
            torch.zeros(1, self.n_patches, cfg.intraday_width)
        )
        self.intraday = nn.ModuleList(
            TemporalMixerBlock(
                cfg.intraday_width,
                kernel=3,
                dilation=2 ** (index % 3),
                dropout=cfg.dropout,
            )
            for index in range(cfg.intraday_blocks)
        )
        self.day_projection = nn.Linear(
            2 * cfg.intraday_width, cfg.daily_width
        )
        self.day_position = nn.Parameter(
            torch.zeros(1, cfg.history_days, cfg.daily_width)
        )
        self.daily = nn.ModuleList(
            TemporalMixerBlock(
                cfg.daily_width,
                kernel=cfg.daily_kernel,
                dilation=2 ** (index % 4),
                dropout=cfg.dropout,
            )
            for index in range(cfg.daily_blocks)
        )
        self.final_norm = nn.LayerNorm(cfg.daily_width)
        self.cross_in = nn.Linear(cfg.daily_width, cfg.cs_rank, bias=False)
        self.cross_out = nn.Linear(cfg.cs_rank, cfg.daily_width, bias=False)
        self.head = nn.Sequential(
            nn.Linear(cfg.daily_width, cfg.daily_width),
            nn.SiLU(),
            nn.Linear(cfg.daily_width, 1),
        )

    def encode_stocks(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [N,D,T,F], got {tuple(x.shape)}")
        n_stocks, n_days, n_bars, n_fields = x.shape
        if (
            n_days != self.cfg.history_days
            or n_bars != self.cfg.bars_per_day
            or n_fields != len(self.cfg.fields)
        ):
            raise ValueError(
                "expected "
                f"[N,{self.cfg.history_days},{self.cfg.bars_per_day},"
                f"{len(self.cfg.fields)}], got {tuple(x.shape)}"
            )
        patches = x.reshape(
            n_stocks,
            n_days,
            self.n_patches,
            self.cfg.patch,
            n_fields,
        ).reshape(
            n_stocks * n_days,
            self.n_patches,
            self.cfg.patch * n_fields,
        )
        h = self.patch_projection(patches) + self.patch_position
        for block in self.intraday:
            h = block(h)
        day = torch.cat([h.mean(dim=1), h[:, -1]], dim=-1)
        day = self.day_projection(day).reshape(
            n_stocks, n_days, self.cfg.daily_width
        )
        day = day + self.day_position
        for block in self.daily:
            day = block(day)
        return self.final_norm(day[:, -1])

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.encode_stocks(x)
        if group_ids is None:
            group_ids = torch.zeros(len(h), dtype=torch.long, device=h.device)
        else:
            group_ids = group_ids.to(device=h.device, dtype=torch.long)
        if group_ids.numel() != len(h):
            raise ValueError("group_ids length does not match stock batch")
        n_groups = int(group_ids.max().item()) + 1 if len(group_ids) else 0
        low_rank = torch.tanh(self.cross_in(h))
        summary = torch.zeros(
            n_groups,
            self.cfg.cs_rank,
            device=h.device,
            dtype=low_rank.dtype,
        )
        summary.index_add_(0, group_ids, low_rank)
        counts = torch.bincount(group_ids, minlength=n_groups).clamp_min(1)
        summary = summary / counts.to(summary.dtype).unsqueeze(-1)
        h = h + self.cfg.cs_weight * self.cross_out(summary[group_ids])
        score = self.head(h).squeeze(-1)
        if pad_mask is not None:
            score = score.masked_fill(pad_mask, 0.0)
        return score


def build_model(cfg: Config = FROZEN_CONFIG) -> nn.Module:
    if cfg.architecture == "patchtst":
        return RawPatchTST(cfg)
    if cfg.architecture == "raw_swiglu_mixer":
        return RawSwiGLUMixer(cfg)
    if cfg.architecture == "hierarchical_mixer":
        return HierarchicalMinuteMixer(cfg)
    raise ValueError(f"unsupported architecture: {cfg.architecture}")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def assert_parameter_count(model: nn.Module) -> int:
    count = parameter_count(model)
    if not (PARAM_MIN <= count < PARAM_MAX):
        raise ValueError(
            f"trainable parameter count {count} outside [{PARAM_MIN}, {PARAM_MAX})"
        )
    return count


def pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_centered = prediction.float() - prediction.float().mean()
    target_centered = target.float() - target.float().mean()
    denominator = torch.sqrt(
        pred_centered.square().sum() * target_centered.square().sum()
    ).clamp_min(1e-12)
    corr = (pred_centered * target_centered).sum() / denominator
    pred_z = pred_centered / prediction.float().std(unbiased=False).clamp_min(1e-6)
    loss = -corr + mse_weight * nn.functional.mse_loss(pred_z, target.float())
    return loss, corr.detach()


def rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 3:
        return float("nan")
    # pandas 3 copy-on-write may expose rank results as read-only arrays.
    # Evaluation centers both arrays in place, so request explicit writable copies.
    pred_rank = pd.Series(prediction).rank(method="average").to_numpy(
        np.float64, copy=True
    )
    target_rank = pd.Series(target).rank(method="average").to_numpy(
        np.float64, copy=True
    )
    pred_rank -= pred_rank.mean()
    target_rank -= target_rank.mean()
    denominator = np.sqrt(
        np.square(pred_rank).sum() * np.square(target_rank).sum()
    )
    return (
        float((pred_rank * target_rank).sum() / denominator)
        if denominator > 0
        else float("nan")
    )


def _day_groups(targets: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    work = targets.reset_index(drop=True).copy()
    work["_sample_index"] = np.arange(len(work), dtype=np.int64)
    groups = []
    for day, group in work.groupby("date", sort=True):
        if len(group) >= FROZEN_CONFIG.min_xsec:
            groups.append((pd.Timestamp(day), group.reset_index(drop=True)))
    return groups


def _load_day_tensor(
    chunks: Mapping[str, np.ndarray],
    group: pd.DataFrame,
    stats: Mapping[str, list[float]],
) -> np.ndarray:
    chunk_names = group["chunk"].astype(str).unique()
    if len(chunk_names) != 1:
        raise ValueError(f"one trading day spans chunks: {chunk_names.tolist()}")
    rows = group["row"].to_numpy(np.int64)
    x = np.asarray(chunks[chunk_names[0]][rows])
    return standardize_tensor(x, stats)


def _random_sample_index(
    targets: pd.DataFrame,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Compact references for random-sample training without copying the cache."""
    chunk_codes, chunk_names = pd.factorize(
        targets["chunk"].astype(str), sort=True
    )
    return (
        tuple(str(value) for value in chunk_names),
        chunk_codes.astype(np.int16, copy=False),
        targets["row"].to_numpy(np.int64, copy=True),
        targets["target"].to_numpy(np.float32, copy=True),
    )


def _load_random_batch(
    chunks: Mapping[str, np.ndarray],
    chunk_names: tuple[str, ...],
    chunk_codes: np.ndarray,
    rows: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Gather one shuffled batch as float16; standardization stays on the GPU."""
    selected_codes = chunk_codes[sample_indices]
    selected_rows = rows[sample_indices]
    first = next(iter(chunks.values()))
    batch = np.empty((len(sample_indices), *first.shape[1:]), dtype=np.float16)
    for code in np.unique(selected_codes):
        positions = np.flatnonzero(selected_codes == code)
        batch[positions] = np.asarray(
            chunks[chunk_names[int(code)]][selected_rows[positions]],
            dtype=np.float16,
        )
    return batch


def evaluate_model(
    model: nn.Module,
    chunks: Mapping[str, np.ndarray],
    targets: pd.DataFrame,
    stats: Mapping[str, list[float]],
    device: torch.device,
    history: Optional[HistoryIndex] = None,
    resident_store: Optional[ResidentChunkStore] = None,
) -> dict[str, object]:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    model.eval()
    daily_ic: list[float] = []
    daily_dates: list[pd.Timestamp] = []
    if resident_store is not None:
        if history is None:
            raise ValueError("resident evaluation requires a HistoryIndex")
        if resident_store.device != device:
            raise ValueError("resident store and model devices do not match")
        field_mean = torch.tensor(
            stats["mean"], dtype=torch.float32, device=device
        ).view(1, 1, 1, -1)
        field_std = torch.tensor(
            stats["std"], dtype=torch.float32, device=device
        ).view(1, 1, 1, -1)
    with torch.no_grad():
        for day, group in _day_groups(targets):
            if resident_store is not None:
                x = load_resident_history_batch(
                    resident_store,
                    history,
                    group["_sample_index"].to_numpy(np.int64),
                )
                x = torch.nan_to_num(
                    (x.float() - field_mean) / field_std,
                    nan=0.0,
                    posinf=8.0,
                    neginf=-8.0,
                )
            elif history is None:
                x_np = _load_day_tensor(chunks, group, stats)
                x = torch.from_numpy(x_np).to(device)
            else:
                x_np = standardize_tensor(
                    load_history_batch(
                        chunks,
                        history,
                        group["_sample_index"].to_numpy(np.int64),
                    ),
                    stats,
                )
                x = torch.from_numpy(x_np).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                if history is None:
                    output = model(x)
                else:
                    group_ids = torch.zeros(
                        len(x), dtype=torch.long, device=device
                    )
                    output = model(x, group_ids=group_ids)
                prediction = output.float().cpu().numpy()
            ic = rank_ic(prediction, group["target_raw"].to_numpy(np.float64))
            if np.isfinite(ic):
                daily_ic.append(ic)
                daily_dates.append(day)
    values = np.asarray(daily_ic, dtype=np.float64)
    mean = float(values.mean()) if len(values) else float("nan")
    std = float(values.std()) if len(values) else float("nan")
    block_means = [
        float(block.mean())
        for block in np.array_split(values, min(4, len(values)))
        if len(block)
    ]
    min_block = min(block_means) if block_means else float("nan")
    tariff_mask = np.array(
        [
            pd.Timestamp("2025-04-01") <= day <= pd.Timestamp("2025-04-30")
            for day in daily_dates
        ],
        dtype=bool,
    )
    return {
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_ic_ir": mean / std if std > 0 else float("nan"),
        "rank_ic_min": float(values.min()) if len(values) else float("nan"),
        "rank_ic_blocks": block_means,
        "rank_ic_min_block": min_block,
        "selection_score": min_block,
        "tariff_rank_ic": (
            float(values[tariff_mask].mean()) if tariff_mask.any() else None
        ),
        "n_days": int(len(values)),
    }


def _split_train_validation(
    targets: pd.DataFrame,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    dates = np.sort(pd.to_datetime(targets["date"]).dt.normalize().unique())
    if len(dates) < 12:
        raise ValueError("at least 12 target days are required for a purged split")
    n_val = min(
        cfg.val_tail_days,
        max(5, int(round(len(dates) * cfg.val_tail_fraction))),
    )
    n_val = min(n_val, len(dates) - cfg.purge_days - 5)
    val_dates = dates[-n_val:]
    train_end = len(dates) - n_val - cfg.purge_days
    train_dates = dates[:train_end]
    normalized = pd.to_datetime(targets["date"]).dt.normalize().to_numpy()
    train_indices = np.flatnonzero(np.isin(normalized, train_dates))
    val_indices = np.flatnonzero(np.isin(normalized, val_dates))
    if not len(train_indices) or not len(val_indices):
        raise ValueError("empty train/validation split")
    return train_indices, val_indices


def _date_stratified_batches(
    targets: pd.DataFrame,
    rng: np.random.Generator,
    batch_size: int,
    dates_per_batch: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    per_date = max(8, batch_size // dates_per_batch)
    pieces: list[np.ndarray] = []
    for _, group in targets.groupby("date", sort=False):
        indices = group.index.to_numpy(np.int64, copy=True)
        rng.shuffle(indices)
        pieces.extend(
            indices[start : start + per_date]
            for start in range(0, len(indices), per_date)
        )
    rng.shuffle(pieces)
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    date_values = pd.to_datetime(targets["date"]).dt.normalize().to_numpy()
    for start in range(0, len(pieces), dates_per_batch):
        selected_pieces = pieces[start : start + dates_per_batch]
        sample_indices = np.concatenate(selected_pieces)
        _, group_ids = np.unique(
            date_values[sample_indices], return_inverse=True
        )
        batches.append(
            (
                sample_indices.astype(np.int64, copy=False),
                group_ids.astype(np.int64, copy=False),
            )
        )
    return batches


def _probe_subset(
    targets: pd.DataFrame,
    history: HistoryIndex,
    max_days: int,
) -> tuple[pd.DataFrame, HistoryIndex]:
    dates = np.sort(pd.to_datetime(targets["date"]).dt.normalize().unique())
    if len(dates) <= max_days:
        indices = np.arange(len(targets), dtype=np.int64)
    else:
        selected_dates = dates[
            np.unique(np.linspace(0, len(dates) - 1, max_days).round().astype(int))
        ]
        normalized = pd.to_datetime(targets["date"]).dt.normalize().to_numpy()
        indices = np.flatnonzero(np.isin(normalized, selected_dates))
    return (
        targets.iloc[indices].reset_index(drop=True),
        history.take(indices),
    )


def _group_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for group in torch.unique(group_ids):
        mask = group_ids == group
        if int(mask.sum()) < 3:
            continue
        pred = prediction[mask].float()
        truth = target[mask].float()
        pred = pred - pred.mean()
        truth = truth - truth.mean()
        denominator = torch.sqrt(
            pred.square().sum() * truth.square().sum()
        ).clamp_min(1e-12)
        losses.append(1.0 - (pred * truth).sum() / denominator)
    if not losses:
        return prediction.float().sum() * 0.0
    return torch.stack(losses).mean()


def _train_hierarchical_model(
    chunks: Mapping[str, np.ndarray],
    targets: pd.DataFrame,
    stats: Mapping[str, list[float]],
    cfg: Config,
    history_index: HistoryIndex,
    metadata: Optional[pd.DataFrame],
    resident_store: Optional[ResidentChunkStore],
    use_resident_cache: bool,
) -> tuple[nn.Module, list[dict[str, object]], dict[str, object]]:
    train_indices, val_indices = _split_train_validation(targets, cfg)
    train_targets = targets.iloc[train_indices].reset_index(drop=True)
    val_targets = targets.iloc[val_indices].reset_index(drop=True)
    train_history = history_index.take(train_indices)
    val_history = history_index.take(val_indices)
    if metadata is not None:
        stats = fit_stats_from_metadata(
            chunks,
            metadata,
            pd.Timestamp(train_targets["date"].max()),
        )
    probe_targets, probe_history = _probe_subset(
        val_targets,
        val_history,
        cfg.val_probe_days,
    )

    set_seed(cfg.seed)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if resident_store is None and use_resident_cache:
            resident_store = build_resident_chunk_store(
                chunks,
                history_index.chunk_names,
                device,
            )
    elif resident_store is not None:
        raise ValueError("resident CUDA store cannot be used with a CPU model")
    if resident_store is not None and resident_store.device != device:
        raise ValueError("resident store and model devices do not match")
    model = build_model(cfg).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    n_params = assert_parameter_count(model)
    optimizer_kwargs = {
        "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    field_mean = torch.tensor(
        stats["mean"], dtype=torch.float32, device=device
    ).view(1, 1, 1, -1)
    field_std = torch.tensor(
        stats["std"], dtype=torch.float32, device=device
    ).view(1, 1, 1, -1)
    rng = np.random.default_rng(cfg.seed)
    estimated_steps_per_epoch = max(1, math.ceil(len(train_targets) / cfg.batch_size))
    total_steps = max(1, cfg.epochs * estimated_steps_per_epoch)
    warmup_steps = min(cfg.warmup_steps, max(0, total_steps - 1))
    best_state = copy.deepcopy(ema_model.state_dict())
    best_metric = -np.inf
    best_epoch = 0
    stale_evals = 0
    global_step = 0
    history_rows: list[dict[str, object]] = []
    started = time.time()
    log.info(
        "hierarchical training: device=%s params=%d train_days=%d val_days=%d "
        "train_samples=%d val_samples=%d history=%d fields=%d",
        device,
        n_params,
        train_targets["date"].nunique(),
        val_targets["date"].nunique(),
        len(train_targets),
        len(val_targets),
        cfg.history_days,
        len(cfg.fields),
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batches = _date_stratified_batches(
            train_targets,
            rng,
            cfg.batch_size,
            cfg.dates_per_batch,
        )
        epoch_loss = 0.0
        epoch_huber = 0.0
        epoch_rank = 0.0
        seen = 0
        for batch_number, (sample_indices, group_ids_np) in enumerate(
            batches, start=1
        ):
            global_step += 1
            if global_step <= warmup_steps and warmup_steps:
                fraction = global_step / warmup_steps
                lr = cfg.learning_rate * fraction
            else:
                progress = (global_step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = cfg.min_learning_rate + (
                    cfg.learning_rate - cfg.min_learning_rate
                ) * cosine
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if resident_store is None:
                raw = load_history_batch(chunks, train_history, sample_indices)
                x = torch.from_numpy(raw).to(device)
            else:
                x = load_resident_history_batch(
                    resident_store,
                    train_history,
                    sample_indices,
                )
            x = torch.nan_to_num(
                (x.float() - field_mean) / field_std,
                nan=0.0,
                posinf=8.0,
                neginf=-8.0,
            )
            y = torch.from_numpy(
                train_targets.iloc[sample_indices]["target"].to_numpy(
                    np.float32, copy=True
                )
            ).to(device)
            group_ids = torch.from_numpy(group_ids_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(x, group_ids=group_ids)
                huber = nn.functional.huber_loss(
                    prediction.float(),
                    y.float(),
                    delta=cfg.huber_delta,
                )
                rank_term = _group_correlation_loss(
                    prediction,
                    y,
                    group_ids,
                )
                loss = huber + cfg.rank_weight * rank_term
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch} batch={batch_number}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            with torch.no_grad():
                decay = cfg.weight_ema_decay
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters()
                ):
                    ema_parameter.mul_(decay).add_(
                        parameter.detach(), alpha=1.0 - decay
                    )
                for ema_buffer, buffer in zip(
                    ema_model.buffers(), model.buffers()
                ):
                    ema_buffer.copy_(buffer)

            batch_n = len(sample_indices)
            epoch_loss += float(loss.detach()) * batch_n
            epoch_huber += float(huber.detach()) * batch_n
            epoch_rank += float(rank_term.detach()) * batch_n
            seen += batch_n
            if batch_number % 100 == 0 or batch_number == len(batches):
                elapsed = time.time() - started
                completed = (epoch - 1) * len(batches) + batch_number
                total_batches = cfg.epochs * len(batches)
                eta = elapsed / completed * (total_batches - completed)
                print(
                    f"\repoch {epoch}/{cfg.epochs} batch {batch_number}/{len(batches)} "
                    f"loss={epoch_loss / seen:.4f} lr={lr:.2e} "
                    f"ETA={eta / 60:.1f}m",
                    end="",
                    flush=True,
                )
        print()
        row: dict[str, object] = {
            "epoch": epoch,
            "loss": epoch_loss / max(seen, 1),
            "huber": epoch_huber / max(seen, 1),
            "rank_loss": epoch_rank / max(seen, 1),
            "updates": len(batches),
            "global_step": global_step,
            "learning_rate": lr,
        }
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            metrics = evaluate_model(
                ema_model,
                chunks,
                probe_targets,
                stats,
                device,
                history=probe_history,
                resident_store=resident_store,
            )
            row.update({f"probe_{key}": value for key, value in metrics.items()})
            metric = float(metrics["selection_score"])
            if np.isfinite(metric) and metric > best_metric + 1e-12:
                best_metric = metric
                best_epoch = epoch
                stale_evals = 0
                best_state = copy.deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in ema_model.state_dict().items()
                    }
                )
            else:
                stale_evals += 1
            log.info("epoch %d probe metrics: %s", epoch, metrics)
        history_rows.append(row)
        if stale_evals >= cfg.early_stop_patience:
            log.info(
                "early stop at epoch %d; best epoch=%d metric=%.6f",
                epoch,
                best_epoch,
                best_metric,
            )
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate_model(
        model,
        chunks,
        val_targets,
        stats,
        device,
        history=val_history,
        resident_store=resident_store,
    )
    final_metrics.update(
        {
            "n_params": n_params,
            "best_probe_selection_score": best_metric,
            "best_epoch": best_epoch,
            "n_train_days": int(train_targets["date"].nunique()),
            "n_val_days": int(val_targets["date"].nunique()),
            "n_train_samples": int(len(train_targets)),
            "n_val_samples": int(len(val_targets)),
            "feature_stats": dict(stats),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "resident_cache_bytes": (
                resident_store.total_bytes if resident_store is not None else 0
            ),
        }
    )
    return model, history_rows, final_metrics


def train_model(
    chunks: Mapping[str, np.ndarray],
    targets: pd.DataFrame,
    stats: Mapping[str, list[float]],
    cfg: Config = FROZEN_CONFIG,
    history_index: Optional[HistoryIndex] = None,
    metadata: Optional[pd.DataFrame] = None,
    resident_store: Optional[ResidentChunkStore] = None,
    use_resident_cache: bool = True,
) -> tuple[nn.Module, list[dict[str, object]], dict[str, object]]:
    validate_config(cfg)
    if cfg.architecture == "hierarchical_mixer":
        if history_index is None:
            raise ValueError("hierarchical_mixer requires a HistoryIndex")
        return _train_hierarchical_model(
            chunks,
            targets,
            stats,
            cfg,
            history_index,
            metadata,
            resident_store,
            use_resident_cache,
        )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    n_params = assert_parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    groups = _day_groups(targets)
    if not groups:
        raise ValueError("no training days")
    sample_index = None
    if cfg.training_mode == "random_samples":
        sample_index = _random_sample_index(targets)
        if not len(sample_index[1]):
            raise ValueError("no training samples")
        field_mean = torch.tensor(
            stats["mean"], dtype=torch.float32, device=device
        ).view(1, 1, -1)
        field_std = torch.tensor(
            stats["std"], dtype=torch.float32, device=device
        ).view(1, 1, -1)
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, object]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -np.inf
    started = time.time()
    log.info(
        "training %s: device=%s params=%d days=%d samples=%d mode=%s loss=%s fields=%d",
        cfg.architecture,
        device,
        n_params,
        len(groups),
        len(targets),
        cfg.training_mode,
        cfg.loss,
        len(cfg.fields),
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        unit_count = len(targets) if sample_index is not None else len(groups)
        order = rng.permutation(unit_count)
        epoch_loss = 0.0
        epoch_corr = 0.0
        seen = 0
        updates = 0
        batch_orders = (
            [order[start : start + cfg.batch_size] for start in range(0, unit_count, cfg.batch_size)]
            if sample_index is not None
            else [np.asarray([idx], dtype=np.int64) for idx in order]
        )
        for batch_order in batch_orders:
            if sample_index is not None:
                chunk_names, chunk_codes, rows, sample_targets = sample_index
                x_raw = _load_random_batch(
                    chunks,
                    chunk_names,
                    chunk_codes,
                    rows,
                    batch_order,
                )
                x = torch.from_numpy(x_raw).to(device)
                x = torch.nan_to_num(
                    (x.float() - field_mean) / field_std,
                    nan=0.0,
                    posinf=8.0,
                    neginf=-8.0,
                )
                y = torch.from_numpy(sample_targets[batch_order]).to(device)
            else:
                _, group = groups[int(batch_order[0])]
                x_np = _load_day_tensor(chunks, group, stats)
                x = torch.from_numpy(x_np).to(device)
                y = torch.from_numpy(
                    group["target"].to_numpy(np.float32, copy=True)
                ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(x)
                if cfg.loss == "mse":
                    loss = nn.functional.mse_loss(
                        prediction.float(), y.float()
                    )
                    _, corr = pearson_loss(prediction, y, 0.0)
                else:
                    loss, corr = pearson_loss(
                        prediction, y, cfg.mse_weight
                    )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch}, day={group['date'].iloc[0]}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            batch_n = len(batch_order) if sample_index is not None else 1
            epoch_loss += float(loss.detach()) * batch_n
            epoch_corr += float(corr) * batch_n
            seen += batch_n
            updates += 1
            report_every = max(1, 50 if sample_index is None else 50 * cfg.batch_size)
            if seen % report_every < batch_n or seen == unit_count:
                elapsed = time.time() - started
                completed = (epoch - 1) * unit_count + seen
                total = cfg.epochs * unit_count
                eta = elapsed / completed * (total - completed)
                print(
                    f"\repoch {epoch}/{cfg.epochs} "
                    f"{'sample' if sample_index is not None else 'day'} "
                    f"{seen}/{unit_count} "
                    f"loss={epoch_loss / seen:.4f} "
                    f"corr={epoch_corr / seen:.4f} ETA={eta / 60:.1f}m",
                    end="",
                    flush=True,
                )
        print()
        row: dict[str, object] = {
            "epoch": epoch,
            "loss": epoch_loss / unit_count,
            "batch_pearson": epoch_corr / unit_count,
            "updates": updates,
        }
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            metrics = evaluate_model(model, chunks, targets, stats, device)
            row.update(metrics)
            metric = float(metrics["rank_ic_mean"])
            if np.isfinite(metric) and metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
            log.info("epoch %d metrics: %s", epoch, metrics)
        history.append(row)

    model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, chunks, targets, stats, device)
    final_metrics["n_params"] = n_params
    final_metrics["best_train_rank_ic"] = best_metric
    return model, history, final_metrics


def _encode_tensor(tensor: torch.Tensor) -> dict[str, object]:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f2", copy=False)
    return {
        "dtype": "float16",
        "shape": list(array.shape),
        "b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _decode_tensor(meta: Mapping[str, object]) -> torch.Tensor:
    raw = base64.b64decode(str(meta["b64"]))
    dtype_name = str(meta.get("dtype", "float32"))
    dtypes = {"float16": "<f2", "float32": "<f4"}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported serialized tensor dtype: {dtype_name}")
    array = np.frombuffer(raw, dtype=dtypes[dtype_name]).copy().reshape(meta["shape"])
    return torch.from_numpy(array)


def save_model(
    model: nn.Module,
    stats: Mapping[str, list[float]],
    cfg: Config,
    training: Mapping[str, object],
    model_path: str = MODEL_PATH,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "config_hash": config_hash(cfg),
        "fields": list(cfg.fields),
        "feature_stats": dict(stats),
        "n_params": assert_parameter_count(model),
        "weight_storage_dtype": "float16",
        "training": dict(training),
        "state_dict": {
            key: _encode_tensor(value)
            for key, value in model.state_dict().items()
        },
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    if temporary.stat().st_size > JSON_SIZE_CAP:
        temporary.unlink(missing_ok=True)
        raise ValueError("model JSON exceeds the 50MB submission limit")
    os.replace(temporary, path)
    loaded = load_model(str(path))
    for key, value in model.state_dict().items():
        expected = value.detach().cpu().to(torch.float16)
        if not torch.equal(expected, loaded["state_dict"][key]):
            raise RuntimeError(f"model JSON round-trip failed for {key}")
    return str(path)


def save_ensemble_model(
    models: list[nn.Module],
    stats: Mapping[str, list[float]],
    configs: list[Config],
    training: Mapping[str, object],
    model_path: str = MODEL_PATH,
) -> str:
    if len(models) < 2 or len(models) != len(configs):
        raise ValueError("ensemble requires matching model/config lists of size >= 2")
    reference = asdict(configs[0])
    reference.pop("seed")
    for cfg in configs:
        comparable = asdict(cfg)
        comparable.pop("seed")
        if comparable != reference:
            raise ValueError("ensemble configs may differ only by seed")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ensemble": True,
        "configs": [asdict(cfg) for cfg in configs],
        "config_hashes": [config_hash(cfg) for cfg in configs],
        "fields": list(configs[0].fields),
        "feature_stats": dict(stats),
        "n_params": sum(assert_parameter_count(model) for model in models),
        "weight_storage_dtype": "float16",
        "training": dict(training),
        "state_dicts": [
            {
                key: _encode_tensor(value)
                for key, value in model.state_dict().items()
            }
            for model in models
        ],
    }
    if payload["n_params"] > PARAM_MAX:
        raise ValueError("ensemble trainable parameter count exceeds competition cap")
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    if temporary.stat().st_size > JSON_SIZE_CAP:
        temporary.unlink(missing_ok=True)
        raise ValueError("ensemble model JSON exceeds the 50MB submission limit")
    os.replace(temporary, path)
    loaded = load_model(str(path))
    for model_index, model in enumerate(models):
        for key, value in model.state_dict().items():
            expected = value.detach().cpu().to(torch.float16)
            if not torch.equal(
                expected,
                loaded["state_dicts"][model_index][key],
            ):
                raise RuntimeError(
                    f"ensemble JSON round-trip failed for model {model_index} {key}"
                )
    return str(path)


def load_model(model_path: str = MODEL_PATH) -> dict[str, object]:
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if payload.get("ensemble") is True:
        if not (
            len(payload.get("configs", ()))
            == len(payload.get("config_hashes", ()))
            == len(payload.get("state_dicts", ()))
        ):
            raise ValueError("ensemble serialized list lengths are inconsistent")
        payload["state_dicts"] = [
            {
                key: _decode_tensor(value)
                for key, value in state_dict.items()
            }
            for state_dict in payload["state_dicts"]
        ]
        configs = []
        for raw, expected_hash in zip(
            payload["configs"],
            payload["config_hashes"],
        ):
            cfg_raw = dict(raw)
            cfg_raw["fields"] = tuple(cfg_raw["fields"])
            cfg = Config(**cfg_raw)
            if expected_hash != config_hash(cfg):
                raise ValueError("ensemble model/config hash mismatch")
            configs.append(cfg)
        if len(configs) < 2 or len(configs) != len(payload["state_dicts"]):
            raise ValueError("ensemble model/config list lengths are invalid")
        if tuple(payload.get("fields", ())) != configs[0].fields:
            raise ValueError("ensemble model field manifest mismatch")
        reference = asdict(configs[0])
        reference.pop("seed")
        for cfg in configs[1:]:
            comparable = asdict(cfg)
            comparable.pop("seed")
            if comparable != reference:
                raise ValueError("ensemble configs differ by more than seed")
        payload["cfgs"] = configs
        payload["cfg"] = configs[0]
        return payload
    payload["state_dict"] = {
        key: _decode_tensor(value)
        for key, value in payload["state_dict"].items()
    }
    cfg_raw = dict(payload["config"])
    cfg_raw["fields"] = tuple(cfg_raw["fields"])
    cfg = Config(**cfg_raw)
    if payload.get("config_hash") != config_hash(cfg):
        raise ValueError("model/config hash mismatch")
    if tuple(payload.get("fields", ())) != cfg.fields:
        raise ValueError("model field manifest mismatch")
    payload["cfg"] = cfg
    return payload


def month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    result: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        month_end = min(cursor + pd.offsets.MonthEnd(0), end)
        if month_end < cursor:
            month_end = min(cursor + pd.offsets.MonthEnd(1), end)
        result.append((str(cursor.date()), str(month_end.date())))
        cursor = month_end + pd.Timedelta(days=1)
    return result


def query_cloud_frame(
    table: str,
    start_date: str,
    end_date: str,
    cfg: Config = FROZEN_CONFIG,
    include_labels: bool = False,
    exact_membership: bool = True,
) -> pd.DataFrame:
    """Read raw bars for the exact daily competition pool.

    DAI currently returns zero rows for SQL joins between the minute table and
    ``bigalpha_2026_instruments``, including CTE, direct JOIN, DATE_TRUNC, and
    EXISTS variants.  Query the small pool first, push its monthly instrument
    union into the minute-table filter, then enforce exact daily membership in
    pandas.  This keeps the storage scan narrow without relying on the broken
    cross-table join path.
    """
    import dai

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    start_text = str(start.date())
    end_text = str(end.date())
    date_filter = {"date": [start_text, end_text + " 23:59:59"]}
    output_columns = ["date", "instrument", *cfg.fields]
    if include_labels:
        output_columns.append("pre_close")

    pool = dai.query(
        "SELECT date, instrument "
        "FROM bigalpha_2026_instruments "
        "ORDER BY date, instrument",
        filters=date_filter,
        compression=False,
    ).df()
    if pool.empty:
        pool_keys = pd.DataFrame(columns=["pool_date", "instrument"])
        instruments = None
        log.warning(
            "membership table has no rows for %s..%s; querying the injected "
            "bar1m universe without an instrument filter",
            start_text,
            end_text,
        )
    else:
        pool["pool_date"] = pd.to_datetime(pool["date"]).dt.normalize()
        pool["instrument"] = pool["instrument"].astype(str)
        pool_keys = pool[["pool_date", "instrument"]].drop_duplicates()
        instruments = sorted(pool_keys["instrument"].unique().tolist())

    query_fields = [*cfg.fields]
    if include_labels:
        query_fields.append("pre_close")
    columns = ", ".join(query_fields)
    raw_filters = dict(date_filter)
    if instruments is not None:
        raw_filters["instrument"] = instruments
    raw = dai.query(
        f"SELECT date, instrument, {columns} "
        f"FROM {table} ORDER BY date, instrument",
        filters=raw_filters,
        compression=True,
    ).df()
    if raw.empty:
        return pd.DataFrame(columns=output_columns)

    raw["date"] = pd.to_datetime(raw["date"])
    raw["instrument"] = raw["instrument"].astype(str)
    pool_index = pd.MultiIndex.from_frame(pool_keys)
    if exact_membership and not pool_keys.empty:
        raw_index = pd.MultiIndex.from_arrays(
            [raw["date"].dt.normalize(), raw["instrument"]],
            names=["pool_date", "instrument"],
        )
        keep = raw_index.isin(pool_index)
    else:
        keep = np.ones(len(raw), dtype=bool)
    selected = (
        raw.loc[keep, output_columns]
        .sort_values(["date", "instrument"], kind="mergesort")
        .reset_index(drop=True)
    )
    return validate_cloud_frame(
        selected,
        cfg,
        require_pre_close=include_labels,
    )


def _discover_cloud_range(table: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    import dai

    frame = dai.query(
        f"SELECT MIN(date) AS lo, MAX(date) AS hi FROM {table}",
        filters={"date": ["2000-01-01", "2100-01-01"]},
        compression=False,
    ).df()
    return pd.Timestamp(frame["lo"].iloc[0]), pd.Timestamp(frame["hi"].iloc[0])


def build_cloud_training_data(
    table: str,
    cfg: Config = FROZEN_CONFIG,
) -> tuple[
    dict[str, np.ndarray],
    pd.DataFrame,
    dict[str, list[float]],
    dict[str, object],
    HistoryIndex,
    pd.DataFrame,
]:
    lo, hi = _discover_cloud_range(table)
    start = max(lo.normalize(), hi.normalize() - pd.Timedelta(days=cfg.cloud_lookback_days))
    disk_cache = Path(tempfile.mkdtemp(prefix="bigalpha_d14_train_"))
    chunks: dict[str, np.ndarray] = {}
    metadata: list[pd.DataFrame] = []
    count = np.zeros(len(cfg.fields), dtype=np.int64)
    total = np.zeros(len(cfg.fields), dtype=np.float64)
    total2 = np.zeros(len(cfg.fields), dtype=np.float64)
    for month_start, month_end in month_ranges(str(start.date()), str(hi.date())):
        frame = query_cloud_frame(
            table,
            month_start,
            month_end,
            cfg,
            include_labels=cfg.target == "adj_c2c",
        )
        if frame.empty:
            continue
        meta, x = frame_to_tensor(frame, cfg)
        del frame
        chunk = month_start[:7].replace("-", "")
        meta["chunk"] = chunk
        c, s1, s2 = chunk_moments(x)
        x_path = disk_cache / f"x_{chunk}.npy"
        temporary = x_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, x, allow_pickle=False)
        os.replace(temporary, x_path)
        del x
        chunks[chunk] = np.load(x_path, mmap_mode="r", allow_pickle=False)
        count += c
        total += s1
        total2 += s2
        metadata.append(meta)
        log.info("cloud chunk %s: %d stock-days", chunk, len(meta))
    if not metadata:
        raise RuntimeError("cloud training query returned no stock-days")
    meta_all = pd.concat(metadata, ignore_index=True)
    targets = build_targets(meta_all, cfg)
    targets, history_index = build_history_index(
        meta_all,
        targets,
        cfg.history_days,
    )
    stats = moments_to_stats(count, total, total2)
    training_meta = {
        "source": table,
        "train_start": str(start.date()),
        "train_end": str(hi.date()),
        "selection": "purged_tail_maximin_rank_ic_with_weight_ema",
        "target": cfg.target,
        "raw_fields_only": True,
        "disk_mmap_cache": str(disk_cache),
        "disk_mmap_bytes": sum(path.stat().st_size for path in disk_cache.iterdir()),
    }
    return chunks, targets, stats, training_meta, history_index, meta_all


def train_and_save(
    datasources,
    model_path: str = MODEL_PATH,
) -> str:
    """Private retraining entry point; training never uses the injected test table."""
    if isinstance(datasources, Mapping):
        if "bar1m" not in datasources:
            raise KeyError("private retraining requires datasources['bar1m']")
        table = str(datasources["bar1m"])
    else:
        table = str(datasources)
    if not table.strip():
        raise ValueError("private bar1m training table is empty")
    chunks, targets, stats, training_meta, history_index, metadata = build_cloud_training_data(
        table, FROZEN_CONFIG
    )
    if PRIVATE_ENSEMBLE_SEEDS:
        seeds = tuple(dict.fromkeys(int(seed) for seed in PRIVATE_ENSEMBLE_SEEDS))
        if len(seeds) < 2:
            raise ValueError("private ensemble requires at least two unique seeds")
        models: list[nn.Module] = []
        configs: list[Config] = []
        records: list[dict[str, object]] = []
        effective_stats: Mapping[str, list[float]] | None = None
        for seed in seeds:
            cfg = replace(FROZEN_CONFIG, seed=seed)
            model, history, metrics = train_model(
                chunks,
                targets,
                stats,
                cfg,
                history_index=history_index,
                metadata=metadata,
            )
            current_stats = metrics.pop("feature_stats", stats)
            if effective_stats is None:
                effective_stats = current_stats
            else:
                for key in ("mean", "std"):
                    if not np.array_equal(
                        np.asarray(effective_stats[key]),
                        np.asarray(current_stats[key]),
                    ):
                        raise RuntimeError(
                            "private ensemble seeds produced different feature stats"
                        )
            model.cpu()
            models.append(model)
            configs.append(cfg)
            records.append(
                {
                    "seed": seed,
                    "config_hash": config_hash(cfg),
                    "history": history,
                    "metrics": metrics,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        training_meta["ensemble_seeds"] = list(seeds)
        training_meta["ensemble_members"] = records
        return save_ensemble_model(
            models,
            effective_stats or stats,
            configs,
            training_meta,
            model_path=model_path,
        )
    model, history, metrics = train_model(
        chunks,
        targets,
        stats,
        FROZEN_CONFIG,
        history_index=history_index,
        metadata=metadata,
    )
    effective_stats = metrics.pop("feature_stats", stats)
    training_meta["history"] = history
    training_meta["metrics"] = metrics
    return save_model(
        model,
        effective_stats,
        FROZEN_CONFIG,
        training_meta,
        model_path=model_path,
    )


def _mean_model_score(
    models: list[nn.Module],
    x: torch.Tensor,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if not models:
        raise ValueError("prediction requires at least one model")
    predictions = [
        (
            model(x)
            if group_ids is None
            else model(x, group_ids=group_ids)
        ).float()
        for model in models
    ]
    return torch.stack(predictions, dim=0).mean(dim=0)


def _predict_chunk(
    models: list[nn.Module],
    x: np.ndarray,
    meta: pd.DataFrame,
    stats: Mapping[str, list[float]],
    device: torch.device,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for day, group in meta.groupby("date", sort=True):
            rows = group["row"].to_numpy(np.int64)
            x_day = torch.from_numpy(
                standardize_tensor(np.asarray(x[rows]), stats)
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                score = _mean_model_score(models, x_day).cpu().numpy()
            outputs.append(
                pd.DataFrame(
                    {
                        "date": pd.Timestamp(day).normalize(),
                        "instrument": group["instrument"].astype(str).to_numpy(),
                        "score": score.astype(np.float64),
                    }
                )
            )
    if not outputs:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    return pd.concat(outputs, ignore_index=True)


def _predict_history(
    models: list[nn.Module],
    chunks: Mapping[str, np.ndarray],
    anchors: pd.DataFrame,
    history: HistoryIndex,
    stats: Mapping[str, list[float]],
    device: torch.device,
) -> pd.DataFrame:
    work = anchors.reset_index(drop=True).copy()
    work["_sample_index"] = np.arange(len(work), dtype=np.int64)
    outputs: list[pd.DataFrame] = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for day, group in work.groupby("date", sort=True):
            sample_indices = group["_sample_index"].to_numpy(np.int64)
            x_np = standardize_tensor(
                load_history_batch(chunks, history, sample_indices),
                stats,
            )
            x = torch.from_numpy(x_np).to(device)
            group_ids = torch.zeros(len(x), dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                score = _mean_model_score(
                    models,
                    x,
                    group_ids=group_ids,
                ).cpu().numpy()
            outputs.append(
                pd.DataFrame(
                    {
                        "date": pd.Timestamp(day).normalize(),
                        "instrument": group["instrument"].astype(str).to_numpy(),
                        "score": score.astype(np.float64),
                    }
                )
            )
    if not outputs:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    return pd.concat(outputs, ignore_index=True)


def main(datasources, start_date: str, end_date: str) -> pd.DataFrame:
    """Competition inference entry point."""
    payload = load_model(MODEL_PATH)
    cfg: Config = payload["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if payload.get("ensemble") is True:
        configs = payload["cfgs"]
        state_dicts = payload["state_dicts"]
    else:
        configs = [cfg]
        state_dicts = [payload["state_dict"]]
    models = []
    for model_cfg, state_dict in zip(configs, state_dicts):
        model = build_model(model_cfg).to(device)
        model.load_state_dict(
            {key: value.to(device) for key, value in state_dict.items()}
        )
        models.append(model)
    table = (
        datasources["bar1m"]
        if isinstance(datasources, dict)
        else str(datasources)
    )
    if cfg.architecture == "hierarchical_mixer":
        buffer_start = (
            pd.Timestamp(start_date)
            - pd.Timedelta(days=cfg.history_days * 2 + 15)
        )
        ranges = month_ranges(str(buffer_start.date()), str(end_date)[:10])
        chunks: dict[str, np.ndarray] = {}
        metadata: list[pd.DataFrame] = []
        for idx, (month_start, month_end) in enumerate(ranges, start=1):
            raw = query_cloud_frame(
                table,
                month_start,
                month_end,
                cfg,
                exact_membership=False,
            )
            if raw.empty:
                continue
            meta, x = frame_to_tensor(raw, cfg)
            chunk = month_start[:7].replace("-", "")
            meta["chunk"] = chunk
            chunks[chunk] = x
            metadata.append(meta)
            log.info(
                "inference history chunk %d/%d %s: %d stock-days",
                idx,
                len(ranges),
                chunk,
                len(meta),
            )
        if metadata:
            meta_all = pd.concat(metadata, ignore_index=True)
            dates = pd.to_datetime(meta_all["date"]).dt.normalize()
            anchors = meta_all.loc[
                (dates >= pd.Timestamp(start_date).normalize())
                & (dates <= pd.Timestamp(end_date).normalize())
            ].copy()
            anchors, history_index = build_history_index(
                meta_all,
                anchors,
                cfg.history_days,
            )
            predicted = _predict_history(
                models,
                chunks,
                anchors,
                history_index,
                payload["feature_stats"],
                device,
            )
        else:
            predicted = pd.DataFrame(
                columns=["date", "instrument", "score"]
            )
    else:
        frames: list[pd.DataFrame] = []
        ranges = month_ranges(str(start_date)[:10], str(end_date)[:10])
        for idx, (month_start, month_end) in enumerate(ranges, start=1):
            raw = query_cloud_frame(table, month_start, month_end, cfg)
            if raw.empty:
                continue
            meta, x = frame_to_tensor(raw, cfg)
            frames.append(
                _predict_chunk(
                    models,
                    x,
                    meta,
                    payload["feature_stats"],
                    device,
                )
            )
            log.info(
                "inference chunk %d/%d %s..%s: %d stock-days",
                idx,
                len(ranges),
                month_start,
                month_end,
                len(meta),
            )
        predicted = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["date", "instrument", "score"])
        )
    import dai

    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    result = (
        pool[["date", "instrument"]]
        .drop_duplicates()
        .merge(predicted, on=["date", "instrument"], how="left")
    )
    result["score"] = (
        result["score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float64)
    )
    return result.sort_values(
        ["date", "instrument"], kind="mergesort"
    ).reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(
        "Use prepare_local.py + run_local.py locally, or train_and_save() on BigQuant."
    )
