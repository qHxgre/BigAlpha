# -*- coding: utf-8 -*-
"""Shared local-training/cloud-inference code for the BigAlpha E2E track.

The important contract in this module is that local compressed Feather rows and
cloud ``stock`` rows are converted to exactly the same canonical representation
before any model preprocessing is applied.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PRICE_SCALE = 100.0
OHLC_COLS = ("open", "high", "low", "close")
PRICE_COLS = (
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
)
SCALE_COLS = PRICE_COLS + ("amount",)
LOG1P_COLS = (
    "deal_number", "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
)

# ``adjust_factor`` is deliberately excluded: it exists in the downloaded
# Feather files but is absent from the published cloud-table example.  Keeping
# only columns known to exist on both sides prevents train/inference drift.
FEATURE_COLS = PRICE_COLS + (
    "deal_number", "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
)


@dataclass(frozen=True)
class ModelConfig:
    n_feat: int = len(FEATURE_COLS)
    d_model: int = 96
    nhead: int = 4
    nlayers: int = 3
    dim_ff: int = 192
    seq_len: int = 240
    patch_size: int = 1
    dropout: float = 0.10


DEFAULT_MODEL_CONFIG = ModelConfig()


class StockTransformer(nn.Module):
    """A per-stock temporal encoder producing one score per stock-day."""

    def __init__(
        self,
        n_feat: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        seq_len: int,
        patch_size: int = 1,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        if self.patch_size < 1 or self.seq_len % self.patch_size != 0:
            raise ValueError("patch_size must be positive and divide seq_len exactly")
        if self.patch_size == 1:
            self.proj = nn.Linear(n_feat, d_model)
        else:
            self.proj = nn.Conv1d(
                in_channels=n_feat,
                out_channels=d_model,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        token_len = self.seq_len // self.patch_size
        self.pos = nn.Parameter(torch.zeros(1, token_len, d_model))
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
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            tokens = self.proj(x)
        else:
            tokens = self.proj(x.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(tokens + self.pos[:, : tokens.shape[1]])
        return self.head(h.mean(dim=1)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_canonical(df: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    """Convert downloaded E2E rows or cloud stock rows to one representation."""
    required = {"date", *FEATURE_COLS}
    key_col = "instrument_id" if is_local else "instrument"
    missing = sorted(required.difference(df.columns))
    if key_col not in df.columns:
        missing.append(key_col)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df.loc[:, ["date", key_col, *FEATURE_COLS]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if is_local:
        for col in OHLC_COLS:
            out.loc[out[col] == -1, col] = np.nan
        for col in SCALE_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64") / PRICE_SCALE
    else:
        for col in SCALE_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    for col in FEATURE_COLS:
        if col not in SCALE_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in LOG1P_COLS:
        values = out[col].clip(lower=0)
        out[col] = np.log1p(values)

    out["key"] = out[key_col]
    out = out.drop(columns=[key_col]).dropna(subset=["date", "key"])
    return out


def _month_number(ts: pd.Timestamp) -> int:
    return int(ts.strftime("%Y%m"))


def local_month_files(
    table_dir: str,
    start_date,
    end_date,
    *,
    lookback_days: int = 10,
    lookahead_days: int = 10,
) -> list[str]:
    start = pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=lookahead_days)
    lo, hi = _month_number(start), _month_number(end)
    paths = []
    for path in sorted(glob.glob(os.path.join(table_dir, "*.feather"))):
        stem = os.path.basename(path).split(".", 1)[0]
        if stem.isdigit() and lo <= int(stem) <= hi:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no monthly Feather files found in {table_dir} for {lo}..{hi}")
    return paths


def load_local_frame(
    table_dir: str,
    start_date,
    end_date,
    *,
    max_instruments: Optional[int] = None,
    instrument_ids: Optional[Sequence[int]] = None,
    lookback_days: int = 10,
    lookahead_days: int = 10,
) -> tuple[pd.DataFrame, list[int]]:
    """Read only required columns and only months overlapping the requested range."""
    paths = local_month_files(
        table_dir,
        start_date,
        end_date,
        lookback_days=lookback_days,
        lookahead_days=lookahead_days,
    )
    columns = ["date", "instrument_id", *FEATURE_COLS]
    chosen = list(instrument_ids) if instrument_ids is not None else None
    parts = []
    lower = pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days)
    upper = pd.Timestamp(end_date) + pd.Timedelta(days=lookahead_days)

    for path in paths:
        part = pd.read_feather(path, columns=columns)
        part["date"] = pd.to_datetime(part["date"], errors="coerce")
        part = part[(part["date"] >= lower) & (part["date"] <= upper)]
        if chosen is None and max_instruments:
            chosen = sorted(part["instrument_id"].dropna().unique().tolist())[:max_instruments]
        if chosen is not None:
            part = part[part["instrument_id"].isin(chosen)]
        if not part.empty:
            parts.append(part)

    if not parts:
        raise RuntimeError("no rows remain after local date/instrument filtering")
    raw = pd.concat(parts, ignore_index=True)
    if chosen is None:
        chosen = sorted(raw["instrument_id"].dropna().unique().tolist())
    return to_canonical(raw, is_local=True), chosen


def build_windows(
    canonical: pd.DataFrame,
    start_date,
    end_date,
    *,
    seq_len: int,
    mode: str,
) -> tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """Build one end-of-day sample per stock and an exact next-trading-day label.

    The global trading-day calendar is used to prevent ``shift(-1)`` from
    accidentally labeling a constituent with its return after a long absence.
    """
    if mode not in {"train", "infer"}:
        raise ValueError("mode must be 'train' or 'infer'")
    sd = pd.Timestamp(start_date).normalize()
    ed = pd.Timestamp(end_date).normalize()
    work = canonical.sort_values(["key", "date"], kind="mergesort").reset_index(drop=True)
    all_days = np.sort(work["date"].dt.normalize().unique())
    next_day = {pd.Timestamp(all_days[i]): pd.Timestamp(all_days[i + 1]) for i in range(len(all_days) - 1)}

    windows: list[np.ndarray] = []
    labels: list[np.float32] = []
    keys: list[tuple[pd.Timestamp, object]] = []

    for key, sub in work.groupby("key", sort=False):
        if len(sub) < seq_len:
            continue
        ts = sub["date"].to_numpy()
        days = sub["date"].dt.normalize().to_numpy()
        feats = sub.loc[:, FEATURE_COLS].to_numpy(dtype="float32")
        close = sub["close"].to_numpy(dtype="float64")
        eod = np.flatnonzero(np.append(days[1:] != days[:-1], True))
        close_by_day = {pd.Timestamp(days[p]): close[p] for p in eod}

        for p in eod:
            day = pd.Timestamp(days[p])
            if day < sd or day > ed or p + 1 < seq_len:
                continue
            window = feats[p - seq_len + 1 : p + 1]
            # Enforce a genuine chronological window with no duplicate timestamps.
            if len(np.unique(ts[p - seq_len + 1 : p + 1])) != seq_len:
                continue
            if mode == "train":
                nd = next_day.get(day)
                current_close = close_by_day.get(day)
                future_close = close_by_day.get(nd) if nd is not None else None
                if (
                    future_close is None
                    or not np.isfinite(current_close)
                    or not np.isfinite(future_close)
                    or current_close <= 0
                ):
                    continue
                target = future_close / current_close - 1.0
                if not np.isfinite(target):
                    continue
                labels.append(np.float32(target))
            windows.append(window)
            keys.append((day, key))

    if not windows:
        raise RuntimeError(f"no usable {mode} windows for {sd.date()}..{ed.date()}")
    x = np.stack(windows).astype("float32")
    index = pd.DataFrame(keys, columns=["date", "key"])
    y = np.asarray(labels, dtype="float32") if mode == "train" else None
    return x, y, index


def fit_feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1]).astype("float64")
    mean = np.nanmean(flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(flat), flat, mean)
    std = filled.std(axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype("float32"), std.astype("float32")


def normalize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype="float32")
    std = np.asarray(std, dtype="float32")
    filled = np.where(np.isfinite(x), x, mean.reshape(1, 1, -1))
    return ((filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype("float32")


def save_model_json(
    model: nn.Module,
    path: str,
    *,
    model_config: ModelConfig,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: Optional[dict] = None,
) -> str:
    tensors = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    payload = {
        "schema_version": "bigalpha_e2e_transformer_v1",
        "state_dict": tensors,
        "model_config": asdict(model_config),
        "feature_cols": list(FEATURE_COLS),
        "preprocessing": {
            "price_scale_local": PRICE_SCALE,
            "ohlc_missing_local": -1,
            "log1p_cols": list(LOG1P_COLS),
            "mean": np.asarray(mean, dtype="float32").tolist(),
            "std": np.asarray(std, dtype="float32").tolist(),
        },
        "metadata": metadata or {},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return path


def load_model_json(path: str, device: Optional[torch.device] = None):
    device = device or torch.device("cpu")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("feature_cols") != list(FEATURE_COLS):
        raise ValueError("checkpoint feature columns do not match this code")
    config = ModelConfig(**payload["model_config"])
    model = StockTransformer(**asdict(config)).to(device)
    state = {}
    for name, spec in payload["state_dict"].items():
        dtype = getattr(torch, spec["dtype"])
        state[name] = torch.tensor(spec["data"], dtype=dtype).reshape(spec["shape"])
    model.load_state_dict(state)
    prep = payload["preprocessing"]
    mean = np.asarray(prep["mean"], dtype="float32")
    std = np.asarray(prep["std"], dtype="float32")
    return model, config, mean, std, payload


def predict_array(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs).astype("float64")


def cloud_predict_scores(
    datasources: dict,
    start_date,
    end_date,
    *,
    model_path: str,
    instrument_chunk_size: int = 40,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Cloud-only DAI reader plus memory-bounded inference."""
    import dai

    table = datasources["bar1m"]
    device = choose_device()
    model, config, mean, std, _ = load_model_json(model_path, device=device)
    model.eval()

    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.dropna(subset=["date", "instrument"]).drop_duplicates(["date", "instrument"])
    instruments = pool["instrument"].drop_duplicates().tolist()

    buffer_days = max(10, int(math.ceil(config.seq_len / 240.0) * 3))
    query_start = (pd.Timestamp(start_date) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    pieces = []

    for offset in range(0, len(instruments), instrument_chunk_size):
        chunk = instruments[offset : offset + instrument_chunk_size]
        raw = dai.query(
            sql,
            filters={"date": [query_start, end_date], "instrument": chunk},
            compression=True,
        ).df()
        if raw.empty:
            continue
        canonical = to_canonical(raw, is_local=False)
        try:
            x, _, index = build_windows(
                canonical,
                start_date,
                end_date,
                seq_len=config.seq_len,
                mode="infer",
            )
        except RuntimeError:
            continue
        x = normalize_features(x, mean, std)
        index["score"] = predict_array(model, x, device=device, batch_size=batch_size)
        pieces.append(index.rename(columns={"key": "instrument"}))

    if pieces:
        predicted = pd.concat(pieces, ignore_index=True)
        predicted["instrument"] = predicted["instrument"].astype(str)
        predicted = predicted.drop_duplicates(["date", "instrument"], keep="last")
    else:
        predicted = pd.DataFrame(columns=["date", "instrument", "score"])

    result = pool.merge(predicted, on=["date", "instrument"], how="left")
    result["score"] = (
        pd.to_numeric(result["score"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float64")
    )
    return result.sort_values(["date", "instrument"])[["date", "instrument", "score"]].reset_index(drop=True)


# Local training CLI and reproducibility entry point.
import argparse
import json
import os
import time
from dataclasses import asdict, replace
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--table", default="bigalpha_2026_e2e_bar1m")
    parser.add_argument("--train-start", default="2024-01-02")
    parser.add_argument("--train-end", default="2024-01-31 23:59:59")
    parser.add_argument("--valid-start", default="2024-02-01")
    parser.add_argument("--valid-end", default="2024-02-29 23:59:59")
    parser.add_argument("--max-instruments", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1,
        help="learned temporal patch size inside the model; must divide seq-len",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--loss-mode",
        choices=("mse", "daily_corr_neutral"),
        default="mse",
        help="daily_corr_neutral trains on one complete daily cross-section",
    )
    parser.add_argument("--neutral-penalty", type=float, default=1.0)
    parser.add_argument(
        "--label-mode",
        choices=("raw", "cross_sectional_zscore", "proxy_residual_zscore"),
        default="proxy_residual_zscore",
        help="training-target transform; validation RankIC always uses raw returns",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-model", help="optional compatible JSON checkpoint used to initialize weights")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="process one calendar month at a time; required for multi-year/full-pool training",
    )
    parser.add_argument(
        "--final-train",
        action="store_true",
        help="train on the full requested interval and save without running a validation split",
    )
    parser.add_argument("--model-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_model.json"))
    return parser.parse_args()


LABEL_PROXY_COLS = ("close", "volume", "amount", "deal_number", "bid_volume1", "ask_volume1")


def transform_training_labels(
    y: np.ndarray,
    index: pd.DataFrame,
    mode: str,
    x_raw: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(y, dtype="float64")
    if mode == "raw":
        return values.astype("float32")
    frame = pd.DataFrame({"date": pd.to_datetime(index["date"]).to_numpy(), "target": values})
    transformed = np.empty(len(values), dtype="float64")
    if mode == "proxy_residual_zscore" and x_raw is None:
        raise ValueError("proxy_residual_zscore requires raw input windows")
    proxy_positions = [FEATURE_COLS.index(column) for column in LABEL_PROXY_COLS]

    for _, positions in frame.groupby("date", sort=False).indices.items():
        positions = np.asarray(positions, dtype="int64")
        target = values[positions]
        target_std = np.nanstd(target)
        target_z = (target - np.nanmean(target)) / (target_std if target_std > 1e-12 else 1.0)
        if mode == "cross_sectional_zscore":
            transformed[positions] = target_z
            continue

        exposure = np.asarray(x_raw[positions, -1, :][:, proxy_positions], dtype="float64")
        exposure_mean = np.nanmean(exposure, axis=0)
        exposure = np.where(np.isfinite(exposure), exposure, exposure_mean)
        exposure_std = exposure.std(axis=0)
        exposure = (exposure - exposure_mean) / np.where(exposure_std > 1e-12, exposure_std, 1.0)
        design = np.column_stack([np.ones(len(positions)), exposure])
        residual = target_z - design @ np.linalg.lstsq(design, target_z, rcond=None)[0]
        residual_std = residual.std()
        transformed[positions] = (residual - residual.mean()) / (residual_std if residual_std > 1e-12 else 1.0)
    return transformed.astype("float32")


def validation_metrics(index: pd.DataFrame) -> tuple[pd.Series, dict]:
    daily_ic = index.groupby("date", sort=True)[["score", "target"]].apply(
        lambda part: part["score"].corr(part["target"], method="spearman")
    )
    ic_mean = float(daily_ic.mean())
    ic_std = float(daily_ic.std(ddof=1))
    summary = {
        "rank_ic_mean": ic_mean,
        "rank_ic_std": ic_std,
        "rank_ic_ir": float(ic_mean / ic_std) if ic_std > 0 else float("nan"),
        "rank_ic_median": float(daily_ic.median()),
        "positive_ic_rate": float((daily_ic > 0).mean()),
        "validation_days": int(daily_ic.notna().sum()),
        "validation_samples": int(len(index)),
        "score_std": float(index["score"].std(ddof=0)),
    }
    return daily_ic, summary


def save_metrics(model_path: str, summary: dict, daily_ic: pd.Series) -> str:
    path = model_path + ".metrics.json"
    payload = dict(summary)
    payload["daily_rank_ic"] = {
        pd.Timestamp(day).strftime("%Y-%m-%d"): (None if not np.isfinite(value) else float(value))
        for day, value in daily_ic.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def daily_corr_neutral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    exposure: torch.Tensor,
    *,
    neutral_penalty: float,
) -> torch.Tensor:
    eps = 1e-6
    pred_z = (prediction - prediction.mean()) / (prediction.std(unbiased=False) + eps)
    target_z = (target - target.mean()) / (target.std(unbiased=False) + eps)
    exposure_z = (exposure - exposure.mean(dim=0, keepdim=True)) / (
        exposure.std(dim=0, unbiased=False, keepdim=True) + eps
    )
    prediction_correlation = (pred_z * target_z).mean()
    exposure_correlation = (pred_z[:, None] * exposure_z).mean(dim=0)
    mse = torch.mean((pred_z - target_z) ** 2)
    return (
        1.0 - prediction_correlation
        + float(neutral_penalty) * torch.mean(exposure_correlation ** 2)
        + 0.05 * mse
    )


def month_ranges(start_date, end_date) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for period in pd.period_range(start=start, end=end, freq="M"):
        yield max(start, period.start_time), min(end, period.end_time)


def _load_month_windows(table_dir, start, end, config, chosen_ids, max_instruments):
    frame, discovered = load_local_frame(
        table_dir,
        start,
        end,
        max_instruments=max_instruments if chosen_ids is None else None,
        instrument_ids=chosen_ids,
    )
    x, y, index = build_windows(frame, start, end, seq_len=config.seq_len, mode="train")
    return x, y, index, discovered


def train_streaming(args: argparse.Namespace) -> dict:
    """Two-pass, month-bounded training suitable for the full six-year table."""
    set_seed(args.seed)
    device = choose_device()
    table_dir = os.path.join(os.path.abspath(args.data_root), args.table)
    config = replace(DEFAULT_MODEL_CONFIG, seq_len=args.seq_len, patch_size=args.patch_size)
    print("device:", device)
    print("table:", table_dir)
    print("mode: monthly streaming")
    started = time.time()

    # Pass 1: compute exact training-window field statistics and label percentiles.
    sums = np.zeros(config.n_feat, dtype="float64")
    sums_sq = np.zeros(config.n_feat, dtype="float64")
    counts = np.zeros(config.n_feat, dtype="int64")
    label_parts = []
    chosen_ids = None
    train_samples = 0
    for month_start, month_end in month_ranges(args.train_start, args.train_end):
        try:
            x, y, index, discovered = _load_month_windows(
                table_dir, month_start, month_end, config, chosen_ids, args.max_instruments
            )
        except RuntimeError as exc:
            print("skip stats month:", month_start.strftime("%Y-%m"), exc)
            continue
        if chosen_ids is None and args.max_instruments:
            chosen_ids = discovered
        flat = x.reshape(-1, config.n_feat).astype("float64")
        finite = np.isfinite(flat)
        safe = np.where(finite, flat, 0.0)
        sums += safe.sum(axis=0)
        sums_sq += np.square(safe).sum(axis=0)
        counts += finite.sum(axis=0)
        y = transform_training_labels(y, index, args.label_mode, x)
        label_parts.append(y)
        train_samples += len(y)
        print("stats month:", month_start.strftime("%Y-%m"), "samples:", len(y))
        del x, y, flat, finite, safe

    if not label_parts or np.any(counts == 0):
        raise RuntimeError("streaming statistics pass produced no usable training data")
    mean64 = sums / counts
    variance = np.maximum(sums_sq / counts - np.square(mean64), 0.0)
    std64 = np.sqrt(variance)
    std64 = np.where(np.isfinite(std64) & (std64 > 1e-6), std64, 1.0)
    mean, std = mean64.astype("float32"), std64.astype("float32")
    all_labels = np.concatenate(label_parts)
    label_lo, label_hi = np.percentile(all_labels, [1, 99])
    del label_parts, all_labels

    model = StockTransformer(**asdict(config)).to(device)
    if args.init_model:
        initial_model, initial_config, _, _, _ = load_model_json(args.init_model, device=device)
        if initial_config != config:
            raise ValueError(f"init-model config {initial_config} does not match requested config {config}")
        model.load_state_dict(initial_model.state_dict())
        print("initialized from:", args.init_model)
    n_params = parameter_count(model)
    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(f"model parameter count {n_params:,} violates competition limits")
    print(f"parameters: {n_params:,}; training samples: {train_samples:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Pass 2+: re-read one month at a time for each epoch and discard it afterwards.
    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        total, batches = 0.0, 0
        epoch_started = time.time()
        for month_start, month_end in month_ranges(args.train_start, args.train_end):
            try:
                x, y, index, _ = _load_month_windows(
                    table_dir, month_start, month_end, config, chosen_ids, args.max_instruments
                )
            except RuntimeError:
                continue
            y = transform_training_labels(y, index, args.label_mode, x)
            x = normalize_features(x, mean, std)
            y = np.clip(y, label_lo, label_hi).astype("float32")
            if args.loss_mode == "mse":
                loader = DataLoader(
                    TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                    batch_size=args.batch_size,
                    shuffle=True,
                )
                iterator = ((xb, yb, None) for xb, yb in loader)
            else:
                proxy_positions = [FEATURE_COLS.index(column) for column in LABEL_PROXY_COLS]
                day_positions = list(index.groupby("date", sort=False).indices.values())
                np.random.shuffle(day_positions)
                iterator = (
                    (
                        torch.from_numpy(x[positions]),
                        torch.from_numpy(y[positions]),
                        torch.from_numpy(x[positions, -1, :][:, proxy_positions]),
                    )
                    for positions in day_positions
                    if len(positions) >= 20
                )
            for xb, yb, exposure in iterator:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb)
                if exposure is None:
                    loss = loss_fn(prediction, yb)
                else:
                    loss = daily_corr_neutral_loss(
                        prediction,
                        yb,
                        exposure.to(device),
                        neutral_penalty=args.neutral_penalty,
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach().cpu())
                batches += 1
            del x, y, iterator
        epoch_loss = total / max(batches, 1)
        epoch_losses.append(float(epoch_loss))
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.8f} "
            f"elapsed={time.time() - epoch_started:.1f}s"
        )

    common_metadata = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "valid_start": None if args.final_train else args.valid_start,
        "valid_end": None if args.final_train else args.valid_end,
        "max_instruments": args.max_instruments,
        "seed": args.seed,
        "epochs": args.epochs,
        "epoch_losses": epoch_losses,
        "streaming": True,
        "final_train": args.final_train,
        "label_mode": args.label_mode,
        "loss_mode": args.loss_mode,
        "neutral_penalty": args.neutral_penalty,
        "init_model": args.init_model,
        "label": "next_trading_day_close_to_close_raw_return",
    }
    if args.final_train:
        save_model_json(
            model,
            args.model_path,
            model_config=config,
            mean=mean,
            std=std,
            metadata=common_metadata,
        )
        print("final model saved:", args.model_path)
        print(f"total elapsed: {time.time() - started:.1f}s")
        return {"model_path": args.model_path, "training_samples": train_samples}

    # Validation also stays month-bounded.
    validation_parts = []
    for month_start, month_end in month_ranges(args.valid_start, args.valid_end):
        try:
            x, y, index, _ = _load_month_windows(
                table_dir, month_start, month_end, config, chosen_ids, args.max_instruments
            )
        except RuntimeError:
            continue
        x = normalize_features(x, mean, std)
        index = index.copy()
        index["target"] = y
        index["score"] = predict_array(model, x, device=device, batch_size=args.batch_size)
        validation_parts.append(index)
        del x, y
    if not validation_parts:
        raise RuntimeError("streaming validation produced no usable samples")
    valid_index = pd.concat(validation_parts, ignore_index=True)
    daily_ic, metrics = validation_metrics(valid_index)
    ic_mean, ic_ir = metrics["rank_ic_mean"], metrics["rank_ic_ir"]
    print(
        f"validation rankIC mean={ic_mean:.6f}; ICIR={ic_ir:.6f}; "
        f"positive_rate={metrics['positive_ic_rate']:.3f}; days={metrics['validation_days']}"
    )

    save_model_json(
        model,
        args.model_path,
        model_config=config,
        mean=mean,
        std=std,
        metadata=common_metadata,
    )
    metrics["label_mode"] = args.label_mode
    metrics_path = save_metrics(args.model_path, metrics, daily_ic)
    print("model saved:", args.model_path)
    print("metrics saved:", metrics_path)
    print(f"total elapsed: {time.time() - started:.1f}s")
    return {"rank_ic_mean": ic_mean, "ic_ir": ic_ir, "model_path": args.model_path}


def train(args: argparse.Namespace) -> dict:
    if args.loss_mode != "mse":
        raise ValueError("daily_corr_neutral requires --streaming")
    set_seed(args.seed)
    device = choose_device()
    table_dir = os.path.join(os.path.abspath(args.data_root), args.table)
    config = replace(DEFAULT_MODEL_CONFIG, seq_len=args.seq_len, patch_size=args.patch_size)
    print("device:", device)
    print("table:", table_dir)

    started = time.time()
    train_frame, chosen_ids = load_local_frame(
        table_dir,
        args.train_start,
        args.train_end,
        max_instruments=args.max_instruments,
    )
    x_train, y_train, train_index = build_windows(
        train_frame,
        args.train_start,
        args.train_end,
        seq_len=config.seq_len,
        mode="train",
    )
    mean, std = fit_feature_stats(x_train)
    y_train = transform_training_labels(y_train, train_index, args.label_mode, x_train)
    x_train = normalize_features(x_train, mean, std)
    lo, hi = np.percentile(y_train, [1, 99])
    y_train = np.clip(y_train, lo, hi).astype("float32")

    valid_frame, _ = load_local_frame(
        table_dir,
        args.valid_start,
        args.valid_end,
        instrument_ids=chosen_ids,
    )
    x_valid, y_valid, valid_index = build_windows(
        valid_frame,
        args.valid_start,
        args.valid_end,
        seq_len=config.seq_len,
        mode="train",
    )
    x_valid = normalize_features(x_valid, mean, std)

    model = StockTransformer(**asdict(config)).to(device)
    if args.init_model:
        initial_model, initial_config, _, _, _ = load_model_json(args.init_model, device=device)
        if initial_config != config:
            raise ValueError(f"init-model config {initial_config} does not match requested config {config}")
        model.load_state_dict(initial_model.state_dict())
        print("initialized from:", args.init_model)
    n_params = parameter_count(model)
    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(f"model parameter count {n_params:,} violates competition limits")
    print(f"parameters: {n_params:,}")
    print(f"train samples: {len(y_train):,}; validation samples: {len(y_valid):,}")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        batches = 0
        epoch_started = time.time()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={total / max(batches, 1):.8f} "
            f"elapsed={time.time() - epoch_started:.1f}s"
        )

    predictions = predict_array(model, x_valid, device=device, batch_size=args.batch_size)
    valid_index = valid_index.copy()
    valid_index["target"] = y_valid
    valid_index["score"] = predictions
    daily_ic, metrics = validation_metrics(valid_index)
    ic_mean, ic_ir = metrics["rank_ic_mean"], metrics["rank_ic_ir"]
    print(
        f"validation rankIC mean={ic_mean:.6f}; ICIR={ic_ir:.6f}; "
        f"positive_rate={metrics['positive_ic_rate']:.3f}; days={metrics['validation_days']}"
    )

    save_model_json(
        model,
        args.model_path,
        model_config=config,
        mean=mean,
        std=std,
        metadata={
            "train_start": args.train_start,
            "train_end": args.train_end,
            "valid_start": args.valid_start,
            "valid_end": args.valid_end,
            "max_instruments": args.max_instruments,
            "seed": args.seed,
            "epochs": args.epochs,
            "label_mode": args.label_mode,
            "loss_mode": args.loss_mode,
            "neutral_penalty": args.neutral_penalty,
            "init_model": args.init_model,
            "label": "next_trading_day_close_to_close_raw_return",
        },
    )
    metrics["label_mode"] = args.label_mode
    metrics_path = save_metrics(args.model_path, metrics, daily_ic)
    print("model saved:", args.model_path)
    print("metrics saved:", metrics_path)
    print(f"total elapsed: {time.time() - started:.1f}s")
    return {"rank_ic_mean": ic_mean, "ic_ir": ic_ir, "model_path": args.model_path}


def train_and_save(args: argparse.Namespace) -> dict:
    """Reproducible public entry point for local or isolated retraining."""
    return train_streaming(args) if args.streaming else train(args)


if __name__ == "__main__":
    train_and_save(parse_args())
