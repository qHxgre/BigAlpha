# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import base64
import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

try:
    import dai
except Exception:  # pragma: no cover - only available on BigQuant
    dai = None


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "transformer_model.json"
BATCH = 32
_LAST_LOADED_MEDIAN: np.ndarray | None = None
# This checkpoint was trained after replacing standardized OHLC with zero.
# Zero is the training-mean representation after fieldwise normalization.
MASK_OHLC = True

FEATURE_COLS = [
    "open", "high", "low", "close", "deal_number", "volume", "amount",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
LOG1P_COLS = [
    "deal_number", "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
QUOTE_PRICE_COLS = [
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
STYLE_COLS = [
    "SIZE", "BETA", "BTOP", "GROWTH", "RESVOL",
    "SIZENL", "EARNYILD", "LEVERAGE", "LIQUIDTY", "MOMENTUM",
]
INDUSTRY_COLS = [
    "AGRIFOREST", "MINING", "CHEM", "IRONSTEEL", "NONFERMETAL",
    "ELECTRONICS", "AUTO", "HOUSEAPP", "FOODBEVER", "TEXTILE",
    "LIGHTINDUS", "HEALTH", "UTILITIES", "TRANSPORTATION", "REALESTATE",
    "COMMETRADE", "LEISERVICE", "BANK", "NONBANKFINAN", "CONGLOMERATES",
    "CONMAT", "BUILDDECO", "ELECEQP", "AERODEF", "COMPUTER", "MEDIA",
    "TELECOM", "COAL", "PETRO", "ENVP", "BEAUTY",
]
EXPOSURE_COLS = STYLE_COLS + INDUSTRY_COLS

SEQ_LEN = 240
MODEL_CFG = {'dropout': 0.15, 'hidden': 104, 'gru_layers': 2, 'kernel_size': 5, 't_model': 'conv_gru'}
TRAIN_START = "2019-01-01"
TRAIN_END = "2024-12-31"
SEED = 114514
EPOCHS = 35
T2_SETTINGS = {'lr': 0.0002, 'weight_decay': 0.0001, 'dropout': 0.15, 'temperature': 0.2, 'pearson_weight': 0.4, 'rank_weight': 0.5, 'huber_weight': 0.1, 'eta_min': 2e-05, 'warmup_steps': 1000, 'hold_steps': 7000, 'decay_end_steps': 15000, 'epochs': 35}


T2_HIDDEN = 104
T2_GRU_LAYERS = 2
T2_KERNEL_SIZE = 5
T2_EXPECTED_PARAMETERS = 202593


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(value, (self.left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


class Head(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class ConvGRUCore(nn.Module):
    def __init__(self, dropout: float, hidden: int, gru_layers: int, kernel_size: int) -> None:
        super().__init__()
        self.local = CausalConv1d(25, hidden, kernel_size=kernel_size)
        self.local_norm = ChannelLayerNorm(hidden)
        self.position = nn.Parameter(torch.zeros(1, 240, hidden))
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=gru_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = Head(3 * hidden, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.local(x.transpose(1, 2))
        value = F.gelu(self.local_norm(value)).transpose(1, 2)
        value = value + self.position
        sequence, _ = self.gru(value)
        pooled = torch.cat(
            (sequence[:, -1], sequence.mean(dim=1), sequence.amax(dim=1)),
            dim=-1,
        )
        return self.head(pooled)


class StockTransformer(nn.Module):
    """Standalone T2 CNN-GRU with state names identical to the audited trainer."""

    def __init__(
        self,
        dropout: float = 0.15,
        hidden: int = T2_HIDDEN,
        gru_layers: int = T2_GRU_LAYERS,
        kernel_size: int = T2_KERNEL_SIZE,
        t_model: str = "conv_gru",
    ) -> None:
        super().__init__()
        if (
            hidden != T2_HIDDEN
            or gru_layers != T2_GRU_LAYERS
            or kernel_size != T2_KERNEL_SIZE
            or t_model != "conv_gru"
        ):
            raise ValueError("T2 requires the fixed hidden/layer/kernel ConvGRU contract")
        self.core = ConvGRUCore(dropout, hidden, gru_layers, kernel_size)
        parameters = sum(parameter.numel() for parameter in self.parameters())
        if parameters != T2_EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"T2 parameter contract changed: {parameters} != {T2_EXPECTED_PARAMETERS}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (240, 25):
            raise ValueError(f"T2 requires [batch, 240, 25], got {tuple(x.shape)}")
        return self.core(x)


def _table_name(datasources: Any) -> str:
    if isinstance(datasources, str):
        return datasources
    if isinstance(datasources, dict):
        return datasources.get("bar1m") or datasources.get("data") or next(iter(datasources.values()))
    raise TypeError("datasources must be a table name or a dict containing bar1m")


def pool(start_date: str, end_date: str) -> list[Any]:
    if dai is None:
        raise RuntimeError("dai is required on the competition platform")
    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    return frame["instrument"].tolist()


def _query_bar1m(
    table: str,
    start_date: str,
    end_date: str,
    instruments: list[Any] | None = None,
) -> pd.DataFrame:
    if dai is None:
        raise RuntimeError("dai is required on the competition platform")
    sql = f"""
    SELECT date, instrument, {", ".join(FEATURE_COLS)}
    FROM {table}
    ORDER BY instrument, date
    """
    filters: dict[str, Any] = {"date": [start_date, end_date]}
    if instruments is not None:
        filters["instrument"] = instruments
    return dai.query(sql, filters=filters).df()


def _prepare_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # The query result is disposable; copying it doubles peak host memory.
    output = frame
    if "instrument_id" in output.columns and "instrument" not in output.columns:
        output = output.rename(columns={"instrument_id": "instrument"})
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["trade_date"] = output["date"].dt.normalize()
    output = output.dropna(subset=["date", "trade_date", "instrument"])
    output["instrument"] = output["instrument"].astype(str)
    output = output.sort_values(["instrument", "date"], kind="mergesort")
    for column in FEATURE_COLS:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    for column in QUOTE_PRICE_COLS:
        values = output[column].to_numpy(np.float32, copy=True)
        values[values == 0.0] = np.nan
        output[column] = values
    for column in LOG1P_COLS:
        values = output[column].to_numpy(np.float32, copy=True)
        values = np.where(np.isfinite(values) & (values >= 0.0), values, np.nan)
        output[column] = np.log1p(values).astype("float32")
    return output


def _stats_to_arrays(
    stats: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if stats is None:
        return None, None, None
    if isinstance(stats, tuple):
        mean = np.asarray(stats[0], np.float32)
        std = np.asarray(stats[1], np.float32)
        median = _LAST_LOADED_MEDIAN
        if median is None:
            median = np.zeros_like(mean)
        return mean, std, np.asarray(median, np.float32)
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    median = np.asarray(stats.get("median", np.zeros_like(mean)), np.float32)
    return mean, std, median


def _make_samples(
    frame: pd.DataFrame,
    include_labels: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    next_one: dict[pd.Timestamp, pd.Timestamp] = {}
    next_two: dict[pd.Timestamp, pd.Timestamp] = {}
    open_lookup: dict[tuple[str, pd.Timestamp], float] = {}
    if include_labels:
        calendar = [
            pd.Timestamp(value)
            for value in sorted(frame["trade_date"].dropna().unique())
        ]
        next_one = {
            calendar[i]: calendar[i + 1]
            for i in range(max(0, len(calendar) - 1))
        }
        next_two = {
            calendar[i]: calendar[i + 2]
            for i in range(max(0, len(calendar) - 2))
        }
        first_rows = frame.groupby(
            ["instrument", "trade_date"], sort=True, as_index=False
        ).head(1)
        open_lookup = {
            (str(row.instrument), pd.Timestamp(row.trade_date)): float(row.open)
            for row in first_rows[["instrument", "trade_date", "open"]].itertuples(index=False)
            if np.isfinite(row.open)
        }

    windows: list[np.ndarray] = []
    keys: list[tuple[pd.Timestamp, str]] = []
    labels: list[float] = []
    for (instrument, trade_date), group in frame.groupby(["instrument", "trade_date"], sort=True):
        group = group.sort_values("date", kind="mergesort")
        if len(group) < SEQ_LEN:
            continue
        day = pd.Timestamp(trade_date)
        code = str(instrument)
        windows.append(group.tail(SEQ_LEN)[FEATURE_COLS].to_numpy(np.float32, copy=True))
        keys.append((day, code))
        if include_labels:
            target_start = next_one.get(day)
            target_end = next_two.get(day)
            open_t1 = (
                open_lookup.get((code, pd.Timestamp(target_start)))
                if target_start is not None
                else None
            )
            open_t2 = (
                open_lookup.get((code, pd.Timestamp(target_end)))
                if target_end is not None
                else None
            )
            labels.append(
                float(open_t2 / open_t1 - 1.0)
                if open_t1 is not None and open_t2 is not None and open_t1 != 0.0
                else np.nan
            )
    if not windows:
        raise RuntimeError("build_dataset produced no samples")
    index = pd.DataFrame(keys, columns=["date", "instrument"])
    return np.stack(windows).astype(np.float32), index, np.asarray(labels, np.float32)


def _labels_from_adjusted_bar1d(index: pd.DataFrame, bars: pd.DataFrame) -> np.ndarray:
    """Align official adjusted open[T+2] / open[T+1] returns on the market calendar."""
    daily = bars[["date", "instrument", "open"]].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["instrument"] = daily["instrument"].astype(str)
    daily["open"] = pd.to_numeric(daily["open"], errors="coerce")
    daily = daily.dropna(subset=["date", "instrument"])
    daily = daily.drop_duplicates(["date", "instrument"], keep="last")
    calendar = pd.DatetimeIndex(sorted(daily["date"].unique()))
    mapping = pd.DataFrame({
        "date": calendar[:-2],
        "target_start_date": calendar[1:-1],
        "target_end_date": calendar[2:],
    })
    aligned = index[["date", "instrument"]].copy()
    aligned["date"] = pd.to_datetime(aligned["date"], errors="coerce").dt.normalize()
    aligned["instrument"] = aligned["instrument"].astype(str)
    aligned["_row"] = np.arange(len(aligned), dtype=np.int64)
    aligned = aligned.merge(mapping, on="date", how="left", validate="many_to_one")
    open_t1 = daily.rename(columns={"date": "target_start_date", "open": "open_t1"})[
        ["target_start_date", "instrument", "open_t1"]
    ]
    open_t2 = daily.rename(columns={"date": "target_end_date", "open": "open_t2"})[
        ["target_end_date", "instrument", "open_t2"]
    ]
    aligned = aligned.merge(
        open_t1, on=["target_start_date", "instrument"], how="left", validate="many_to_one"
    )
    aligned = aligned.merge(
        open_t2, on=["target_end_date", "instrument"], how="left", validate="many_to_one"
    ).sort_values("_row", kind="mergesort")
    labels = aligned["open_t2"].to_numpy(np.float64) / aligned["open_t1"].to_numpy(np.float64) - 1.0
    labels[~np.isfinite(labels)] = np.nan
    return labels.astype(np.float32)


def _load_adjusted_open_labels(
    index: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> np.ndarray:
    if dai is None:
        raise RuntimeError("dai is required to load official adjusted bar1d labels")
    bars = dai.query(
        "SELECT date, instrument, open FROM bigalpha_2026_bar1d",
        filters={"date": [start_date, end_date]},
    ).df()
    return _labels_from_adjusted_bar1d(index, bars)


def build_dataset(
    datasources: Any,
    start_date: str,
    end_date: str,
    mode: str = "infer",
    instruments: list[Any] | None = None,
    stats: Any = None,
):
    table = _table_name(datasources)
    if stats is None and instruments is not None and not isinstance(instruments, list):
        stats = instruments
        instruments = None
    frame = _prepare_raw_frame(_query_bar1m(table, start_date, end_date, instruments))
    x, index, _ = _make_samples(frame, include_labels=False)
    labels = (
        _load_adjusted_open_labels(index, start_date, end_date)
        if mode == "train"
        else np.full(len(index), np.nan, np.float32)
    )
    mean, std, median = _stats_to_arrays(stats)
    if mean is None or std is None:
        flat = x.reshape(-1, x.shape[-1])
        median = np.nanmedian(flat, axis=0).astype(np.float32)
        filled = np.where(np.isfinite(x), x, median.reshape(1, 1, -1))
        mean = filled.reshape(-1, x.shape[-1]).mean(axis=0).astype(np.float32)
        std = filled.reshape(-1, x.shape[-1]).std(axis=0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)
        stats = {"mean": mean.tolist(), "std": std.tolist(), "median": median.tolist()}
    else:
        filled = np.where(np.isfinite(x), x, median.reshape(1, 1, -1))
    x = ((filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    if MASK_OHLC:
        x[..., :4] = 0.0
    if mode == "train":
        valid = np.isfinite(labels)
        return x[valid], labels[valid], index.loc[valid].reset_index(drop=True), stats
    return x, None, index, stats


def _load_exposures(
    index: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[np.ndarray, np.ndarray]:
    if dai is None:
        raise RuntimeError("dai is required to load style exposures")
    exposure = dai.query(
        f"SELECT date, instrument, {', '.join(EXPOSURE_COLS)} FROM bigalpha_2026_exposure",
        filters={"date": [start_date, end_date]},
    ).df()
    exposure["date"] = pd.to_datetime(exposure["date"], errors="coerce").dt.normalize()
    exposure["instrument"] = exposure["instrument"].astype(str)
    exposure = exposure.drop_duplicates(["date", "instrument"], keep="last")
    aligned = index.copy()
    aligned["date"] = pd.to_datetime(aligned["date"], errors="coerce").dt.normalize()
    aligned["instrument"] = aligned["instrument"].astype(str)
    aligned["_row"] = np.arange(len(aligned))
    aligned = aligned.merge(
        exposure[["date", "instrument", *EXPOSURE_COLS]],
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).sort_values("_row", kind="mergesort")
    exposures = aligned[EXPOSURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(exposures).all(axis=1)
    return exposures, valid


def _daily_targets(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(raw, np.float64)
    std = max(float(values.std(ddof=1)), 1e-12)
    y_z = ((values - values.mean()) / std).astype(np.float32)
    low, high = np.quantile(values, [0.01, 0.99])
    clipped = np.clip(values, low, high)
    clipped_std = max(float(clipped.std(ddof=1)), 1e-12)
    y_huber = ((clipped - clipped.mean()) / clipped_std).astype(np.float32)
    rank = pd.Series(values).rank(method="average").to_numpy(np.float64)
    y_rank = ((rank - 1.0) / max(1.0, len(rank) - 1.0)).astype(np.float32)
    return y_z, y_huber, y_rank


def _svd_basis(exposures: np.ndarray) -> np.ndarray:
    values = np.asarray(exposures, np.float64)
    design = np.column_stack([np.ones(len(values), np.float64), values])
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        return np.empty((len(values), 0), np.float32)
    rank = int((singular > singular[0] * 1e-15).sum())
    return u[:, :rank].astype(np.float32)


def residualize_prediction(score: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    value = score.float()
    std = value.std(unbiased=True).clamp_min(1e-6)
    mean = value.mean()
    value = torch.clamp(value, mean - 3.0 * std, mean + 3.0 * std)
    value = (value - value.mean()) / value.std(unbiased=True).clamp_min(1e-6)
    if q.numel():
        value = value - q @ (q.transpose(0, 1) @ value)
    return value


def pearson_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = score.float() - score.float().mean()
    y = target.float() - target.float().mean()
    corr = (x * y).mean() / (
        x.square().mean().sqrt() * y.square().mean().sqrt()
    ).clamp_min(1e-8)
    return 1.0 - corr


def weighted_pairwise(
    score: torch.Tensor,
    rank: torch.Tensor,
    min_gap: float = 0.10,
    cap: float = 2.0,
) -> torch.Tensor:
    left, right = torch.triu_indices(len(score), len(score), offset=1, device=score.device)
    difference = rank[left] - rank[right]
    keep = difference.abs() >= min_gap
    if not keep.any():
        return score.sum() * 0.0
    gap = difference[keep].abs()
    weight = (gap / min_gap).clamp(max=cap)
    loss = F.softplus(
        -torch.sign(difference[keep]) * (score[left[keep]] - score[right[keep]])
    )
    return (weight * loss).sum() / weight.sum().clamp_min(1e-8)


def _soft_spearman(score, rank, temperature):
    if len(score) < 2:
        return score.sum() * 0.0
    smooth_rank = torch.sigmoid(
        (score[:, None] - score[None, :]) / max(float(temperature), 1e-4)
    ).sum(dim=1)
    return pearson_loss(smooth_rank, rank)


def _scheduled_lr(step):
    ratio = T2_SETTINGS["eta_min"] / T2_SETTINGS["lr"]
    if step < T2_SETTINGS["warmup_steps"]:
        multiplier = ratio + (1.0 - ratio) * (step + 1) / T2_SETTINGS["warmup_steps"]
    elif step < T2_SETTINGS["hold_steps"]:
        multiplier = 1.0
    elif step < T2_SETTINGS["decay_end_steps"]:
        progress = (step - T2_SETTINGS["hold_steps"]) / (
            T2_SETTINGS["decay_end_steps"] - T2_SETTINGS["hold_steps"]
        )
        multiplier = ratio + 0.5 * (1.0 - ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
    else:
        multiplier = ratio
    return T2_SETTINGS["lr"] * multiplier


def save_model(
    checkpoint: dict[str, Any],
    model_path: str | os.PathLike[str] = MODEL_PATH,
) -> str:
    tensors: dict[str, Any] = {}
    for key, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        tensors[key] = {
            "encoding": "base64",
            "dtype": str(array.dtype),
            "shape": list(tensor.shape),
            "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    payload = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return str(model_path)


def load_model(
    model_path: str | os.PathLike[str] = MODEL_PATH,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    global _LAST_LOADED_MEDIAN
    with open(model_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dict = {}
    for key, metadata in payload["state_dict"].items():
        if metadata.get("encoding") == "base64":
            array = np.frombuffer(
                base64.b64decode(metadata["data"]),
                dtype=np.dtype(metadata["dtype"]),
            ).copy()
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.tensor(metadata["data"], dtype=getattr(torch, metadata["dtype"]))
        state_dict[key] = tensor.reshape(metadata["shape"]).to(map_location)
    payload["state_dict"] = state_dict
    _LAST_LOADED_MEDIAN = np.asarray(payload.get("median", []), np.float32)
    return payload


def train_and_save(
    datasources: Any,
    model_path: str | os.PathLike[str] = MODEL_PATH,
) -> str:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = _table_name(datasources)
    embedded = load_model(model_path, map_location="cpu")
    frozen_stats = {
        "mean": embedded["mean"],
        "std": embedded["std"],
        "median": embedded["median"],
    }
    x, raw, index, stats = build_dataset(
        table,
        TRAIN_START,
        TRAIN_END,
        "train",
        pool(TRAIN_START, TRAIN_END),
        frozen_stats,
    )
    exposures, exposure_valid = _load_exposures(index, TRAIN_START, TRAIN_END)
    x, raw, exposures = x[exposure_valid], raw[exposure_valid], exposures[exposure_valid]
    index = index.loc[exposure_valid].reset_index(drop=True)

    model = StockTransformer(**MODEL_CFG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=T2_SETTINGS['lr'], weight_decay=T2_SETTINGS['weight_decay'])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    indexed = index.copy()
    indexed["row"] = np.arange(len(indexed))
    day_groups = [
        group["row"].to_numpy(np.int64)
        for _, group in indexed.groupby("date", sort=True)
    ]

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        order = list(day_groups)
        date_rng = np.random.default_rng(SEED + epoch)
        date_rng.shuffle(order)
        model.train()
        for day_index, rows in enumerate(order):
            if len(rows) > 1000:
                sample_rng = np.random.default_rng(SEED + epoch * 100000 + day_index)
                rows = np.sort(sample_rng.choice(rows, size=1000, replace=False))
            y_z, y_huber, y_rank = _daily_targets(raw[rows])
            xb = torch.from_numpy(x[rows]).to(device, non_blocking=True)
            q_batch = torch.from_numpy(_svd_basis(exposures[rows])).to(device, non_blocking=True)
            z_batch = torch.from_numpy(y_z).to(device, non_blocking=True)
            huber_batch = torch.from_numpy(y_huber).to(device, non_blocking=True)
            rank_batch = torch.from_numpy(y_rank).to(device, non_blocking=True)
            for group in optimizer.param_groups:
                group['lr'] = _scheduled_lr(global_step)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                prediction = model(xb)
            with torch.autocast(device_type=device.type, enabled=False):
                residual = residualize_prediction(prediction.float(), q_batch.float())
                corr = pearson_loss(residual, z_batch.float())
                ranking = _soft_spearman(residual, rank_batch.float(), T2_SETTINGS['temperature'])
                huber = F.smooth_l1_loss(residual, huber_batch.float(), beta=0.01)
                loss = (T2_SETTINGS['pearson_weight'] * corr
                        + T2_SETTINGS['rank_weight'] * ranking
                        + T2_SETTINGS['huber_weight'] * huber)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

    return save_model(
        {
            "state_dict": model.state_dict(),
            "model_cfg": MODEL_CFG,
            "feature_cols": FEATURE_COLS,
            "log1p_cols": LOG1P_COLS,
            "quote_price_zero_as_missing_cols": QUOTE_PRICE_COLS,
            "mean": stats["mean"],
            "std": stats["std"],
            "median": stats.get("median", [0.0] * len(FEATURE_COLS)),
            "label_definition": "open[T+2] / open[T+1] - 1 on platform-adjusted prices",
            "seed": SEED,
            "epochs": EPOCHS,
            "training_contract": T2_SETTINGS,
            "lr_schedule": "update_hold_cosine",
        },
        model_path,
    )


if __name__ == "__main__":
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})
