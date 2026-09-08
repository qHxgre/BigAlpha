from __future__ import annotations

import gc
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import dai  # BigQuant AIStudio
except ImportError:
    from bigquant import dai  # BigQuant SDK fallback


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class Config:
    bar_table: str = "bigalpha_2026_stock_bar5m"
    exposure_table: str = "bigalpha_2026_exposure"
    instruments_table: str = "bigalpha_2026_instruments"

    # First runnable baseline. Expand after this version runs end-to-end.
    train_start: str = "2023-01-01"
    train_end: str = "2023-12-31"
    valid_start: str = "2024-01-01"
    valid_end: str = "2024-03-31"

    # About five 5-minute trading days (roughly 48 bars/day).
    window_bars: int = 240
    query_chunk_days: int = 21
    lookback_calendar_days: int = 45
    label_horizon_days: int = 1
    label_scale: float = 100.0

    hidden_size: int = 128
    dropout: float = 0.20

    epochs: int = 8
    batch_size: int = 256
    min_date_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    ic_loss_weight: float = 0.10
    grad_clip: float = 1.0
    patience: int = 3
    num_workers: int = 0
    seed: int = 2026

    checkpoint_path: str = "deep_lob_bigalpha.pt"
    metrics_path: str = "deep_lob_training_metrics.csv"


CFG = Config()

# 25 original fields: OHLC / transaction flow + first three LOB levels.
FEATURE_COLS: List[str] = [
    "open", "high", "low", "close",
    "volume", "amount", "deal_number",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]

STYLE_COLS: List[str] = [
    "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL",
    "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE",
]


# ============================================================
# 2. UTILITIES
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def int_to_date(values: Sequence[int]) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(values).astype(str), format="%Y%m%d")


def iter_date_chunks(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    chunk_days: int,
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    current = pd.Timestamp(start).normalize()
    final = pd.Timestamp(end).normalize()
    while current <= final:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), final)
        yield current, chunk_end
        current = chunk_end + pd.Timedelta(days=1)


# ============================================================
# 3. BIGQUANT DATA ACCESS
# ============================================================

def _query_df(sql: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    filters = {
        "date": [
            f"{start:%Y-%m-%d} 00:00:00",
            f"{end:%Y-%m-%d} 23:59:59",
        ]
    }
    return dai.query(sql, filters=filters, compression=True).df()


def load_bars_chunked(
    table: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    feature_cols: Sequence[str],
    chunk_days: int = 21,
) -> pd.DataFrame:
    columns = ["date", "instrument", *feature_cols]
    sql = f"""
    SELECT {", ".join(columns)}
    FROM {table}
    ORDER BY instrument, date
    """
    parts: List[pd.DataFrame] = []
    for chunk_start, chunk_end in iter_date_chunks(start, end, chunk_days):
        part = _query_df(sql, chunk_start, chunk_end)
        print(
            f"[bars] {chunk_start.date()} -> {chunk_end.date()}: "
            f"{len(part):,} rows"
        )
        if not part.empty:
            parts.append(part)

    if not parts:
        raise RuntimeError(
            f"No rows returned from {table} between {start} and {end}."
        )

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    df = (
        df.drop_duplicates(["date", "instrument"])
        .sort_values(["instrument", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    df["trading_date"] = df["date"].dt.normalize()
    return df


def load_exposure_chunked(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    table: str = "bigalpha_2026_exposure",
    chunk_days: int = 180,
) -> pd.DataFrame:
    columns = [
        "date", "instrument", "ret", "weights", "industry_level1_code",
        *STYLE_COLS,
    ]
    sql = f"""
    SELECT {", ".join(columns)}
    FROM {table}
    ORDER BY instrument, date
    """
    parts: List[pd.DataFrame] = []
    for chunk_start, chunk_end in iter_date_chunks(start, end, chunk_days):
        part = _query_df(sql, chunk_start, chunk_end)
        print(
            f"[exposure] {chunk_start.date()} -> {chunk_end.date()}: "
            f"{len(part):,} rows"
        )
        if not part.empty:
            parts.append(part)

    if not parts:
        raise RuntimeError(
            f"No rows returned from {table} between {start} and {end}."
        )

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["instrument"] = df["instrument"].astype(str)
    return (
        df.drop_duplicates(["date", "instrument"])
        .sort_values(["instrument", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def load_instrument_pool(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    table: str = "bigalpha_2026_instruments",
) -> pd.DataFrame:
    sql = f"""
    SELECT date, instrument
    FROM {table}
    ORDER BY date, instrument
    """
    df = _query_df(sql, pd.Timestamp(start), pd.Timestamp(end))
    if df.empty:
        return pd.DataFrame(columns=["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["instrument"] = df["instrument"].astype(str)
    return df[["date", "instrument"]].drop_duplicates()


# ============================================================
# 4. NEXT-DAY BARRA-RESIDUAL LABEL
# ============================================================

def _residualize_one_date(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    required = ["future_ret", *STYLE_COLS]
    valid = group[required].notna().all(axis=1)

    if valid.sum() < 50:
        group["label"] = np.nan
        return group[["date", "instrument", "label"]]

    g = group.loc[valid].copy()
    y = g["future_ret"].to_numpy(dtype=np.float64)

    style_x = g[STYLE_COLS].to_numpy(dtype=np.float64)
    style_x = np.nan_to_num(style_x, nan=0.0, posinf=0.0, neginf=0.0)

    industry = (
        g["industry_level1_code"]
        .astype("string")
        .fillna("UNKNOWN")
    )
    industry_x = pd.get_dummies(
        industry,
        drop_first=True,
    ).to_numpy(dtype=np.float64)

    x = np.column_stack([
        np.ones(len(g), dtype=np.float64),
        style_x,
        industry_x,
    ])

    weights = g["weights"].to_numpy(dtype=np.float64)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    sqrt_weights = np.sqrt(np.clip(weights, 1e-8, None))

    beta, *_ = np.linalg.lstsq(
        x * sqrt_weights[:, None],
        y * sqrt_weights,
        rcond=1e-6,
    )
    residual = y - x @ beta

    output = group[["date", "instrument"]].copy()
    output["label"] = np.nan
    output.loc[valid, "label"] = residual
    return output


def build_residual_labels(
    exposure: pd.DataFrame,
    horizon_days: int = 1,
    label_scale: float = 100.0,
) -> pd.DataFrame:
    exposure = exposure.copy().sort_values(
        ["instrument", "date"],
        kind="mergesort",
    )
    exposure["future_ret"] = (
        exposure.groupby("instrument", sort=False)["ret"]
        .shift(-horizon_days)
    )

    pieces: List[pd.DataFrame] = []
    for i, (_, group) in enumerate(exposure.groupby("date", sort=True), start=1):
        pieces.append(_residualize_one_date(group))
        if i % 100 == 0:
            print(f"[label] residualized {i} trading dates")

    labels = pd.concat(pieces, ignore_index=True)
    labels["label"] = labels["label"].astype(np.float32) * np.float32(label_scale)
    return labels.dropna(subset=["label"]).reset_index(drop=True)


# ============================================================
# 5. ALLOWED PREPROCESSING
# ============================================================

@dataclass
class PreprocessorState:
    feature_cols: List[str]
    fill_values: List[float]
    means: List[float]
    stds: List[float]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict) -> "PreprocessorState":
        return cls(**obj)


def _raw_to_logged_array(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    arr = df[list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    # Per-field log1p transform; all selected cloud-table fields are non-negative.
    arr = np.log1p(np.maximum(arr, 0.0))
    return arr


def fit_preprocessor(
    bars: pd.DataFrame,
    feature_cols: Sequence[str],
    train_end: str | pd.Timestamp,
) -> PreprocessorState:
    train_mask = bars["trading_date"] <= pd.Timestamp(train_end).normalize()
    arr = _raw_to_logged_array(bars.loc[train_mask], feature_cols)

    fill_values = np.nanmedian(arr, axis=0)
    fill_values = np.nan_to_num(fill_values, nan=0.0)
    missing = ~np.isfinite(arr)
    if missing.any():
        arr[missing] = np.take(fill_values, np.where(missing)[1])

    means = arr.mean(axis=0, dtype=np.float64)
    stds = arr.std(axis=0, dtype=np.float64)
    stds = np.where(stds < 1e-6, 1.0, stds)

    return PreprocessorState(
        feature_cols=list(feature_cols),
        fill_values=fill_values.astype(float).tolist(),
        means=means.astype(float).tolist(),
        stds=stds.astype(float).tolist(),
    )


def transform_bars(
    bars: pd.DataFrame,
    state: PreprocessorState,
) -> np.ndarray:
    arr = _raw_to_logged_array(bars, state.feature_cols)
    fill_values = np.asarray(state.fill_values, dtype=np.float32)
    means = np.asarray(state.means, dtype=np.float32)
    stds = np.asarray(state.stds, dtype=np.float32)

    missing = ~np.isfinite(arr)
    if missing.any():
        arr[missing] = np.take(fill_values, np.where(missing)[1])

    arr = (arr - means) / stds
    return np.ascontiguousarray(arr, dtype=np.float32)


# ============================================================
# 6. LAZY STOCK-DAY WINDOW DATASET
# ============================================================

# (group_start, end_index, date_int, instrument, label)
SampleRef = Tuple[int, int, int, str, float]


class StockDaySequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        samples: Sequence[SampleRef],
        window_bars: int,
    ):
        self.features = features
        self.samples = list(samples)
        self.window_bars = int(window_bars)
        self.date_to_indices: Dict[int, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            self.date_to_indices.setdefault(sample[2], []).append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        group_start, end_idx, date_int_value, instrument, label = self.samples[idx]
        start_idx = max(group_start, end_idx - self.window_bars + 1)
        segment = self.features[start_idx:end_idx + 1]
        length = len(segment)

        x = np.zeros(
            (self.window_bars, self.features.shape[1]),
            dtype=np.float32,
        )
        mask = np.zeros(self.window_bars, dtype=np.bool_)
        x[:length] = segment
        mask[:length] = True

        return (
            torch.from_numpy(x),
            torch.tensor(label, dtype=torch.float32),
            torch.from_numpy(mask),
            int(date_int_value),
            instrument,
        )


def collate_stock_day(batch):
    x, y, mask, dates, instruments = zip(*batch)
    return (
        torch.stack(x, dim=0),
        torch.stack(y, dim=0),
        torch.stack(mask, dim=0),
        torch.tensor(dates, dtype=torch.int32),
        list(instruments),
    )


def make_samples(
    bars: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    target_start: str | pd.Timestamp,
    target_end: str | pd.Timestamp,
) -> List[SampleRef]:
    target_start = pd.Timestamp(target_start).normalize()
    target_end = pd.Timestamp(target_end).normalize()

    label_map: Optional[Dict[Tuple[pd.Timestamp, str], float]]
    if labels is None:
        label_map = None
    else:
        label_map = {
            (pd.Timestamp(row.date).normalize(), str(row.instrument)): float(row.label)
            for row in labels.itertuples(index=False)
        }

    samples: List[SampleRef] = []
    group_start = 0

    for instrument, group in bars.groupby("instrument", sort=False):
        group_end = group_start + len(group)
        dates = group["trading_date"].to_numpy()
        is_last = np.r_[dates[1:] != dates[:-1], True]
        relative_ends = np.flatnonzero(is_last)

        for rel_end in relative_ends:
            absolute_end = group_start + int(rel_end)
            dt = pd.Timestamp(dates[rel_end]).normalize()
            if dt < target_start or dt > target_end:
                continue

            if label_map is None:
                label = float("nan")
            else:
                key = (dt, str(instrument))
                if key not in label_map:
                    continue
                label = label_map[key]

            samples.append((
                group_start,
                absolute_end,
                int(dt.year * 10000 + dt.month * 100 + dt.day),
                str(instrument),
                label,
            ))

        group_start = group_end

    return samples


class DateChunkBatchSampler(Sampler[List[int]]):
    """Every mini-batch contains stocks from a single trading date."""

    def __init__(
        self,
        dataset: StockDaySequenceDataset,
        batch_size: int,
        shuffle_dates: bool,
        min_batch_size: int = 1,
        seed: int = 2026,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_dates = bool(shuffle_dates)
        self.min_batch_size = int(min_batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        dates = list(self.dataset.date_to_indices.keys())
        if self.shuffle_dates:
            rng.shuffle(dates)
        else:
            dates.sort()

        for dt in dates:
            indices = list(self.dataset.date_to_indices[dt])
            if self.shuffle_dates:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                if len(chunk) >= self.min_batch_size:
                    yield chunk

    def __len__(self) -> int:
        total = 0
        for indices in self.dataset.date_to_indices.values():
            full, remainder = divmod(len(indices), self.batch_size)
            total += full
            if remainder >= self.min_batch_size:
                total += 1
        return total


# ============================================================
# 7. MODEL
# ============================================================

class DeepLOB(nn.Module):
    def __init__(self, n_feat: int, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.local_encoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.branch_1 = nn.Conv1d(64, 32, kernel_size=1)
        self.branch_3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.branch_5 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.branch_15 = nn.Conv1d(64, 32, kernel_size=15, padding=7)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_projection(x)  # [B, T, 64]
        x = x.transpose(1, 2)         # [B, 64, T]
        x = self.local_encoder(x)
        x = torch.cat(
            [
                torch.relu(self.branch_1(x)),
                torch.relu(self.branch_3(x)),
                torch.relu(self.branch_5(x)),
                torch.relu(self.branch_15(x)),
            ],
            dim=1,
        )
        x = x.transpose(1, 2)         # [B, T, 128]
        x, _ = self.lstm(x)

        logits = self.attention(x).squeeze(-1)
        if mask is not None:
            mask = mask.to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(
                ~mask,
                torch.finfo(logits.dtype).min,
            )
        weights = torch.softmax(
            logits.float(),
            dim=1,
        ).to(dtype=x.dtype).unsqueeze(-1)
        pooled = torch.sum(weights * x, dim=1)
        return self.head(pooled).squeeze(-1)


# ============================================================
# 8. LOSS / METRICS
# ============================================================

def pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denominator = torch.sqrt(
        torch.sum(pred.square()) * torch.sum(target.square()) + 1e-12
    )
    return torch.sum(pred * target) / denominator


def training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ic_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    huber = F.smooth_l1_loss(pred, target)
    corr = pearson_corr(pred, target)
    total = huber + ic_weight * (1.0 - corr)
    return total, huber.detach(), corr.detach()


def calculate_validation_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    daily_ic: List[float] = []
    daily_ls: List[float] = []

    for _, group in predictions.groupby("date", sort=True):
        group = group.dropna(subset=["score", "label"])
        if len(group) < 20:
            continue

        ic = group["score"].rank().corr(group["label"].rank())
        if pd.notna(ic):
            daily_ic.append(float(ic))

        try:
            buckets = pd.qcut(
                group["score"].rank(method="first"),
                q=10,
                labels=False,
            )
            top = group.loc[buckets == 9, "label"].mean()
            bottom = group.loc[buckets == 0, "label"].mean()
            if pd.notna(top) and pd.notna(bottom):
                daily_ls.append(float(top - bottom))
        except ValueError:
            pass

    ic_array = np.asarray(daily_ic, dtype=np.float64)
    ls_array = np.asarray(daily_ls, dtype=np.float64)

    ic_mean = float(np.nanmean(ic_array)) if len(ic_array) else float("nan")
    ic_std = float(np.nanstd(ic_array, ddof=1)) if len(ic_array) > 1 else float("nan")
    ic_ir = (
        ic_mean / ic_std
        if np.isfinite(ic_std) and ic_std > 1e-12
        else float("nan")
    )
    ls_mean = float(np.nanmean(ls_array)) if len(ls_array) else float("nan")
    ls_std = float(np.nanstd(ls_array, ddof=1)) if len(ls_array) > 1 else float("nan")
    ls_sharpe = (
        ls_mean / ls_std * math.sqrt(252.0)
        if np.isfinite(ls_std) and ls_std > 1e-12
        else float("nan")
    )

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "long_short_sharpe": ls_sharpe,
        "n_dates": float(len(ic_array)),
    }


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: List[pd.DataFrame] = []

    for x, y, mask, dates, instruments in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            score = model(x, mask)

        rows.append(pd.DataFrame({
            "date_int": dates.cpu().numpy(),
            "instrument": instruments,
            "score": score.float().cpu().numpy(),
            "label": y.cpu().numpy(),
        }))

    if not rows:
        return pd.DataFrame(columns=["date", "instrument", "score", "label"])

    result = pd.concat(rows, ignore_index=True)
    result["date"] = int_to_date(result.pop("date_int"))
    return result[["date", "instrument", "score", "label"]]


# ============================================================
# 9. TRAINING ENTRY POINT
# ============================================================

def train_model(config: Config = CFG) -> Dict:
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    train_start = pd.Timestamp(config.train_start)
    train_end = pd.Timestamp(config.train_end)
    valid_start = pd.Timestamp(config.valid_start)
    valid_end = pd.Timestamp(config.valid_end)

    bars_start = train_start - pd.Timedelta(days=config.lookback_calendar_days)
    bars = load_bars_chunked(
        table=config.bar_table,
        start=bars_start,
        end=valid_end,
        feature_cols=FEATURE_COLS,
        chunk_days=config.query_chunk_days,
    )

    exposure_end = valid_end + pd.Timedelta(
        days=15 + config.label_horizon_days * 3
    )
    exposure = load_exposure_chunked(
        start=train_start,
        end=exposure_end,
        table=config.exposure_table,
    )
    labels = build_residual_labels(
        exposure,
        horizon_days=config.label_horizon_days,
        label_scale=config.label_scale,
    )
    del exposure

    preprocessor = fit_preprocessor(
        bars,
        feature_cols=FEATURE_COLS,
        train_end=train_end,
    )
    features = transform_bars(bars, preprocessor)

    train_samples = make_samples(
        bars,
        labels,
        target_start=train_start,
        target_end=train_end,
    )
    valid_samples = make_samples(
        bars,
        labels,
        target_start=valid_start,
        target_end=valid_end,
    )

    print(f"Train samples: {len(train_samples):,}")
    print(f"Valid samples: {len(valid_samples):,}")
    if not train_samples or not valid_samples:
        raise RuntimeError(
            "No train/validation samples were constructed. "
            "Check dates and table permissions."
        )

    train_dataset = StockDaySequenceDataset(
        features,
        train_samples,
        config.window_bars,
    )
    valid_dataset = StockDaySequenceDataset(
        features,
        valid_samples,
        config.window_bars,
    )

    train_sampler = DateChunkBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle_dates=True,
        min_batch_size=config.min_date_batch_size,
        seed=config.seed,
    )
    valid_sampler = DateChunkBatchSampler(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle_dates=False,
        min_batch_size=1,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stock_day,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stock_day,
    )

    model = DeepLOB(
        n_feat=len(FEATURE_COLS),
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")
    if not (100_000 <= n_params <= 100_000_000):
        raise ValueError(
            f"Parameter count {n_params:,} violates competition limits."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_selection_score = -np.inf
    epochs_without_improvement = 0
    history: List[Dict] = []

    for epoch in range(1, config.epochs + 1):
        print(f"\n===== Epoch {epoch}/{config.epochs} =====", flush=True)
        train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = huber_sum = corr_sum = 0.0
        n_batches = 0

        for x, y, mask, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x, mask)
                loss, huber, corr = training_loss(
                    pred,
                    y,
                    ic_weight=config.ic_loss_weight,
                )

            if not torch.isfinite(loss):
                warnings.warn("Skipped non-finite training loss.")
                continue

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            loss_sum += float(loss.detach().cpu())
            huber_sum += float(huber.cpu())
            corr_sum += float(corr.cpu())
            n_batches += 1

        scheduler.step()

        valid_predictions = predict_dataset(model, valid_loader, device)
        metrics = calculate_validation_metrics(valid_predictions)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_batches, 1),
            "train_huber": huber_sum / max(n_batches, 1),
            "train_batch_corr": corr_sum / max(n_batches, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

        ic_mean = metrics["ic_mean"]
        ic_ir = metrics["ic_ir"]
        selection_score = (
            (ic_mean if np.isfinite(ic_mean) else -1.0)
            + 0.10 * (ic_ir if np.isfinite(ic_ir) else -1.0)
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            epochs_without_improvement = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_kwargs": {
                    "n_feat": len(FEATURE_COLS),
                    "hidden_size": config.hidden_size,
                    "dropout": config.dropout,
                },
                "feature_cols": FEATURE_COLS,
                "preprocessor": preprocessor.to_dict(),
                "window_bars": config.window_bars,
                "config": asdict(config),
                "best_epoch": epoch,
                "best_metrics": metrics,
                "parameter_count": n_params,
            }
            torch.save(checkpoint, config.checkpoint_path)
            print(f"Saved best checkpoint to {config.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv(config.metrics_path, index=False)
    print(f"Saved training history to {config.metrics_path}")
    return _torch_load(config.checkpoint_path, map_location="cpu")


# ============================================================
# 10. COMPETITION INFERENCE main()
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "deep_lob_model.json"


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _json_safe(value):
    """Convert nested checkpoint metadata to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON metadata type: {type(value)!r}")


def save_model_json(checkpoint: Dict, model_path: str | Path = MODEL_PATH) -> str:
    """Save checkpoint as a text-only JSON file compatible with submission."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for name, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu().contiguous()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }

    payload = {
        key: _json_safe(value)
        for key, value in checkpoint.items()
        if key != "state_dict"
    }
    payload["state_dict"] = tensors
    model_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return str(model_path)


def load_model_json(
    model_path: str | Path = MODEL_PATH,
    map_location: str | torch.device = "cpu",
) -> Dict:
    """Load a text JSON checkpoint and restore its tensor state_dict."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model JSON not found: {model_path}")

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    tensor_meta = payload.pop("state_dict")
    state_dict = {}
    for name, meta in tensor_meta.items():
        dtype_name = str(meta["dtype"])
        if not hasattr(torch, dtype_name):
            raise TypeError(f"Unsupported torch dtype in JSON: {dtype_name}")
        dtype = getattr(torch, dtype_name)
        tensor = torch.tensor(meta["data"], dtype=dtype)
        tensor = tensor.reshape(meta["shape"]).to(map_location)
        state_dict[name] = tensor
    payload["state_dict"] = state_dict
    return payload


def convert_pt_checkpoint_to_json(
    pt_path: str | Path = "deep_lob_bigalpha.pt",
    json_path: str | Path = MODEL_PATH,
) -> str:
    """Convert an already-trained public-board .pt checkpoint to text JSON."""
    checkpoint = _torch_load(pt_path, map_location="cpu")
    return save_model_json(checkpoint, json_path)


def _load_checkpoint(path: str | Path) -> Dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_model_json(path, map_location="cpu")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return _torch_load(path, map_location="cpu")


def _resolve_bar_table(datasources) -> str:
    if isinstance(datasources, dict):
        for key in ("bar5m", "stock_bar5m", "bars"):
            if key in datasources:
                return str(datasources[key])
        if len(datasources) == 1:
            return str(next(iter(datasources.values())))
        raise KeyError(
            "datasources must contain the key 'bar5m', e.g. "
            "{'bar5m': 'bigalpha_2026_stock_bar5m'}."
        )
    if isinstance(datasources, str):
        # Backward-compatible local testing only.
        return datasources
    raise TypeError("datasources must be a dict injected by the platform.")


def main(
    datasources,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Official BigAlpha inference entry: main(datasources, start_date, end_date)."""
    checkpoint = load_model_json(MODEL_PATH, map_location="cpu")

    feature_cols = list(checkpoint["feature_cols"])
    preprocessor = PreprocessorState.from_dict(checkpoint["preprocessor"])
    window_bars = int(checkpoint["window_bars"])
    saved_config = checkpoint.get("config", {})
    lookback_days = int(saved_config.get("lookback_calendar_days", 45))
    chunk_days = int(saved_config.get("query_chunk_days", 10))

    target_start = pd.Timestamp(start_date).normalize()
    target_end = pd.Timestamp(end_date).normalize()
    query_start = target_start - pd.Timedelta(days=lookback_days)
    table = _resolve_bar_table(datasources)

    bars = load_bars_chunked(
        table=table,
        start=query_start,
        end=target_end,
        feature_cols=feature_cols,
        chunk_days=chunk_days,
    )
    features = transform_bars(bars, preprocessor)
    samples = make_samples(
        bars,
        labels=None,
        target_start=target_start,
        target_end=target_end,
    )
    if not samples:
        raise RuntimeError("No inference stock-day samples were constructed.")

    dataset = StockDaySequenceDataset(features, samples, window_bars)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_stock_day,
    )

    device = get_device()
    model = DeepLOB(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    prediction = predict_dataset(model, loader, device)[
        ["date", "instrument", "score"]
    ]

    pool = load_instrument_pool(
        start=target_start,
        end=target_end,
        table=saved_config.get("instruments_table", "bigalpha_2026_instruments"),
    )
    if not pool.empty:
        output = pool.merge(prediction, on=["date", "instrument"], how="left")
        output["score"] = output["score"].fillna(0.0)
    else:
        output = prediction

    output = (
        output[["date", "instrument", "score"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    output["score"] = output["score"].astype(np.float64)

    if list(output.columns) != ["date", "instrument", "score"]:
        raise AssertionError("Output columns are invalid.")
    if output["score"].isna().any():
        raise AssertionError("Output contains NaN scores.")
    return output


def smoke_test_inference(
    start_date: str,
    end_date: str,
    table: str = "bigalpha_2026_stock_bar5m",
) -> pd.DataFrame:
    output = main({"bar5m": table}, start_date, end_date)
    print(output.head())
    print(output.groupby("date").size().describe())
    return output


# ============================================================
# 11. YEAR-WISE FINAL TRAINING (2019-2024)
# ============================================================

@dataclass
class FinalTrainConfig:
    """Configuration for the final all-data fit.

    Hyperparameters are frozen from the 2022 -> 2023 validation experiment.
    No validation/early stopping is used in this final refit.
    """

    years: Tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024)
    bar_table: str = "bigalpha_2026_stock_bar5m"
    exposure_table: str = "bigalpha_2026_exposure"
    instruments_table: str = "bigalpha_2026_instruments"

    window_bars: int = 480
    lookback_calendar_days: int = 45
    label_horizon_days: int = 1
    label_scale: float = 100.0
    query_chunk_days: int = 10

    hidden_size: int = 128
    dropout: float = 0.20

    epochs: int = 10
    batch_size: int = 256
    min_date_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    ic_loss_weight: float = 0.10
    grad_clip: float = 1.0
    seed: int = 2026

    # BigQuant notebook containers are more stable with a single-process loader.
    num_workers: int = 0
    prefetch_factor: int = 2

    # One-time disk cache. It prevents querying/processing six years again
    # in every epoch and keeps RAM bounded to roughly one year at a time.
    cache_dir: str = "deeplob_year_cache"
    scaler_sample_rows_per_year: int = 100_000
    rebuild_cache: bool = False

    # Resume file contains optimizer/scheduler state. Submission file is lean.
    resume_path: str = "deep_lob_final_resume.pt"
    checkpoint_path: str = "deep_lob_bigalpha.pt"
    metrics_path: str = "deep_lob_final_training_metrics.csv"
    save_resume: bool = True


class CachedStockDaySequenceDataset(Dataset):
    """Dataset backed by a memory-mapped yearly feature array."""

    def __init__(
        self,
        features: np.ndarray,
        group_starts: np.ndarray,
        end_indices: np.ndarray,
        date_ints: np.ndarray,
        instruments: np.ndarray,
        labels: np.ndarray,
        window_bars: int,
    ):
        self.features = features
        self.group_starts = group_starts
        self.end_indices = end_indices
        self.date_ints = date_ints
        self.instruments = instruments
        self.labels = labels
        self.window_bars = int(window_bars)

        self.date_to_indices: Dict[int, List[int]] = {}
        for idx, dt in enumerate(self.date_ints.tolist()):
            self.date_to_indices.setdefault(int(dt), []).append(idx)

    def __len__(self) -> int:
        return int(len(self.end_indices))

    def __getitem__(self, idx: int):
        group_start = int(self.group_starts[idx])
        end_idx = int(self.end_indices[idx])
        start_idx = max(group_start, end_idx - self.window_bars + 1)
        segment = np.asarray(self.features[start_idx:end_idx + 1], dtype=np.float32)
        length = int(len(segment))

        x = np.zeros(
            (self.window_bars, self.features.shape[1]),
            dtype=np.float32,
        )
        mask = np.zeros(self.window_bars, dtype=np.bool_)
        x[:length] = segment
        mask[:length] = True

        return (
            torch.from_numpy(x),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            torch.from_numpy(mask),
            int(self.date_ints[idx]),
            str(self.instruments[idx]),
        )


def _final_cache_paths(cache_dir: Path, year: int) -> Dict[str, Path]:
    return {
        "features": cache_dir / f"features_{year}.npy",
        "samples": cache_dir / f"samples_{year}.npz",
        "meta": cache_dir / f"meta_{year}.json",
    }


def _fit_preprocessor_yearwise(config: FinalTrainConfig) -> PreprocessorState:
    """Fit one global train-only scaler without holding all years in RAM.

    Medians are estimated from a deterministic row sample. Means/stds are
    then calculated exactly from all finite values plus the sampled medians
    used as the missing-value fill.
    """

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "preprocessor.json"

    if state_path.exists() and not config.rebuild_cache:
        print(f"Loading cached preprocessor: {state_path}")
        return PreprocessorState.from_dict(
            json.loads(state_path.read_text(encoding="utf-8"))
        )

    n_feat = len(FEATURE_COLS)
    finite_count = np.zeros(n_feat, dtype=np.int64)
    finite_sum = np.zeros(n_feat, dtype=np.float64)
    finite_sumsq = np.zeros(n_feat, dtype=np.float64)
    total_rows = 0
    median_samples: List[np.ndarray] = []
    rng = np.random.default_rng(config.seed)

    print("\n===== Pass 1/2: fit global preprocessor =====")
    for year in config.years:
        bars = load_bars_chunked(
            table=config.bar_table,
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            feature_cols=FEATURE_COLS,
            chunk_days=config.query_chunk_days,
        )
        arr = _raw_to_logged_array(bars, FEATURE_COLS)
        finite = np.isfinite(arr)

        finite_count += finite.sum(axis=0, dtype=np.int64)
        safe = np.where(finite, arr, 0.0).astype(np.float64, copy=False)
        finite_sum += safe.sum(axis=0, dtype=np.float64)
        finite_sumsq += np.square(safe).sum(axis=0, dtype=np.float64)
        total_rows += len(arr)

        sample_n = min(config.scaler_sample_rows_per_year, len(arr))
        if sample_n > 0:
            sample_idx = rng.choice(len(arr), size=sample_n, replace=False)
            median_samples.append(arr[sample_idx].copy())

        print(f"[scaler] {year}: {len(arr):,} rows")
        del bars, arr, finite, safe
        gc.collect()

    if not median_samples:
        raise RuntimeError("Could not collect rows for preprocessing.")

    sample_matrix = np.concatenate(median_samples, axis=0)
    fill_values = np.nanmedian(sample_matrix, axis=0)
    fill_values = np.nan_to_num(fill_values, nan=0.0).astype(np.float64)
    del sample_matrix, median_samples

    missing_count = total_rows - finite_count
    filled_sum = finite_sum + missing_count * fill_values
    filled_sumsq = finite_sumsq + missing_count * np.square(fill_values)
    means = filled_sum / max(total_rows, 1)
    variances = filled_sumsq / max(total_rows, 1) - np.square(means)
    stds = np.sqrt(np.maximum(variances, 1e-12))
    stds = np.where(stds < 1e-6, 1.0, stds)

    state = PreprocessorState(
        feature_cols=list(FEATURE_COLS),
        fill_values=fill_values.astype(float).tolist(),
        means=means.astype(float).tolist(),
        stds=stds.astype(float).tolist(),
    )
    state_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved global preprocessor: {state_path}")
    return state


def _build_one_year_cache(
    year: int,
    preprocessor: PreprocessorState,
    config: FinalTrainConfig,
) -> None:
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _final_cache_paths(cache_dir, year)

    complete = all(path.exists() for path in paths.values())
    if complete and not config.rebuild_cache:
        print(f"[cache] {year}: already exists, skipping")
        return

    print(f"\n===== Pass 2/2: build cache for {year} =====")
    target_start = pd.Timestamp(f"{year}-01-01")
    target_end = pd.Timestamp(f"{year}-12-31")
    bars_start = target_start - pd.Timedelta(
        days=config.lookback_calendar_days
    )

    bars = load_bars_chunked(
        table=config.bar_table,
        start=bars_start,
        end=target_end,
        feature_cols=FEATURE_COLS,
        chunk_days=config.query_chunk_days,
    )

    # Deliberately stop exposure at year-end. The final trading date may have
    # no next-day label and will simply be dropped; no next-year return leaks in.
    exposure = load_exposure_chunked(
        start=target_start,
        end=target_end,
        table=config.exposure_table,
    )
    labels = build_residual_labels(
        exposure,
        horizon_days=config.label_horizon_days,
        label_scale=config.label_scale,
    )
    del exposure

    features = transform_bars(bars, preprocessor)
    samples = make_samples(
        bars,
        labels,
        target_start=target_start,
        target_end=target_end,
    )
    if not samples:
        raise RuntimeError(f"No training samples were built for {year}.")

    np.save(paths["features"], features, allow_pickle=False)
    np.savez_compressed(
        paths["samples"],
        group_starts=np.asarray([s[0] for s in samples], dtype=np.int64),
        end_indices=np.asarray([s[1] for s in samples], dtype=np.int64),
        date_ints=np.asarray([s[2] for s in samples], dtype=np.int32),
        instruments=np.asarray([s[3] for s in samples], dtype="U16"),
        labels=np.asarray([s[4] for s in samples], dtype=np.float32),
    )
    meta = {
        "year": year,
        "n_bars": int(len(features)),
        "n_samples": int(len(samples)),
        "feature_cols": FEATURE_COLS,
        "window_bars": config.window_bars,
    }
    paths["meta"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[cache] {year}: {len(features):,} bars, "
        f"{len(samples):,} samples"
    )

    del bars, labels, features, samples
    gc.collect()


def build_final_year_caches(config: FinalTrainConfig) -> PreprocessorState:
    preprocessor = _fit_preprocessor_yearwise(config)
    for year in config.years:
        _build_one_year_cache(year, preprocessor, config)
    return preprocessor


def _load_cached_year(
    year: int,
    config: FinalTrainConfig,
) -> CachedStockDaySequenceDataset:
    paths = _final_cache_paths(Path(config.cache_dir), year)
    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError(
            f"Year cache for {year} is incomplete. Run build_final_year_caches()."
        )

    features = np.load(paths["features"], mmap_mode="r")
    sample_data = np.load(paths["samples"], allow_pickle=False)
    return CachedStockDaySequenceDataset(
        features=features,
        group_starts=sample_data["group_starts"],
        end_indices=sample_data["end_indices"],
        date_ints=sample_data["date_ints"],
        instruments=sample_data["instruments"],
        labels=sample_data["labels"],
        window_bars=config.window_bars,
    )


def _final_loader(
    dataset: CachedStockDaySequenceDataset,
    sampler: DateChunkBatchSampler,
    config: FinalTrainConfig,
    device: torch.device,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_stock_day,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def _final_submission_checkpoint(
    model: nn.Module,
    preprocessor: PreprocessorState,
    config: FinalTrainConfig,
    epoch: int,
    parameter_count: int,
) -> Dict:
    inference_config = {
        "bar_table": config.bar_table,
        "exposure_table": config.exposure_table,
        "instruments_table": config.instruments_table,
        "lookback_calendar_days": config.lookback_calendar_days,
        "query_chunk_days": config.query_chunk_days,
        "label_horizon_days": config.label_horizon_days,
        "label_scale": config.label_scale,
        "training_years": list(config.years),
        "final_refit": True,
    }
    return {
        "state_dict": model.state_dict(),
        "model_kwargs": {
            "n_feat": len(FEATURE_COLS),
            "hidden_size": config.hidden_size,
            "dropout": config.dropout,
        },
        "feature_cols": FEATURE_COLS,
        "preprocessor": preprocessor.to_dict(),
        "window_bars": config.window_bars,
        "config": inference_config,
        "best_epoch": epoch,
        "best_metrics": {
            "note": "Final refit on all training years; no validation used."
        },
        "parameter_count": parameter_count,
    }


def train_final_yearwise(
    config: FinalTrainConfig = FinalTrainConfig(),
    resume: bool = True,
) -> Dict:
    """Final refit on all years, one year in RAM at a time.

    Every epoch visits 2019 -> ... -> 2024. Hyperparameters and epoch count
    are already frozen from the 2022 -> 2023 validation experiment.
    """

    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    preprocessor = build_final_year_caches(config)

    model = DeepLOB(
        n_feat=len(FEATURE_COLS),
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(device)
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")
    if not (100_000 <= n_params <= 100_000_000):
        raise ValueError(
            f"Parameter count {n_params:,} violates competition limits."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    history: List[Dict] = []
    resume_path = Path(config.resume_path)
    if resume and resume_path.exists():
        saved = _torch_load(resume_path, map_location="cpu")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved["completed_epoch"]) + 1
        history = list(saved.get("history", []))
        print(f"Resuming from completed epoch {start_epoch - 1}")

    if start_epoch > config.epochs:
        print("All requested epochs are already complete.")
        return _load_checkpoint(config.checkpoint_path)

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
        print(f"\n===== Final Epoch {epoch}/{config.epochs} =====", flush=True)
        model.train()

        total_loss = 0.0
        total_huber = 0.0
        total_corr = 0.0
        total_batches = 0
        year_records: List[Dict] = []

        # Chronological order is intentional: every epoch finishes on the
        # latest regime (2024) while still seeing every historical year.
        for year in config.years:
            year_start = time.time()
            dataset = _load_cached_year(year, config)
            sampler = DateChunkBatchSampler(
                dataset,
                batch_size=config.batch_size,
                shuffle_dates=True,
                min_batch_size=config.min_date_batch_size,
                seed=config.seed,
            )
            sampler.set_epoch(epoch * 10_000 + year)
            loader = _final_loader(dataset, sampler, config, device)

            year_loss = 0.0
            year_huber = 0.0
            year_corr = 0.0
            year_batches = 0

            for x, y, mask, _, _ in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    pred = model(x, mask)
                    loss, huber, corr = training_loss(
                        pred,
                        y,
                        ic_weight=config.ic_loss_weight,
                    )

                if not torch.isfinite(loss):
                    warnings.warn("Skipped non-finite training loss.")
                    continue

                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
                amp_scaler.step(optimizer)
                amp_scaler.update()

                loss_value = float(loss.detach().cpu())
                huber_value = float(huber.cpu())
                corr_value = float(corr.cpu())
                year_loss += loss_value
                year_huber += huber_value
                year_corr += corr_value
                year_batches += 1

            if year_batches == 0:
                raise RuntimeError(f"No usable batches were produced for {year}.")

            year_record = {
                "year": year,
                "batches": year_batches,
                "loss": year_loss / year_batches,
                "huber": year_huber / year_batches,
                "batch_corr": year_corr / year_batches,
                "minutes": (time.time() - year_start) / 60.0,
            }
            year_records.append(year_record)
            print(json.dumps(year_record, ensure_ascii=False, indent=2), flush=True)

            total_loss += year_loss
            total_huber += year_huber
            total_corr += year_corr
            total_batches += year_batches

            del loader, sampler, dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_batches, 1),
            "train_huber": total_huber / max(total_batches, 1),
            "train_batch_corr": total_corr / max(total_batches, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batches": total_batches,
            "minutes": (time.time() - epoch_start) / 60.0,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False, indent=2), flush=True)
        pd.DataFrame(history).to_csv(config.metrics_path, index=False)

        submission_ckpt = _final_submission_checkpoint(
            model=model,
            preprocessor=preprocessor,
            config=config,
            epoch=epoch,
            parameter_count=n_params,
        )
        checkpoint_path = Path(config.checkpoint_path)
        if checkpoint_path.suffix.lower() == ".json":
            save_model_json(submission_ckpt, checkpoint_path)
        else:
            torch.save(submission_ckpt, checkpoint_path)

        if config.save_resume:
            resume_ckpt = {
                **submission_ckpt,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "completed_epoch": epoch,
                "history": history,
                "year_records_last_epoch": year_records,
            }
            torch.save(resume_ckpt, config.resume_path)
            print(
                f"Saved epoch {epoch}: {config.checkpoint_path} "
                f"and resume state {config.resume_path}",
                flush=True,
            )
        else:
            print(f"Saved epoch {epoch}: {config.checkpoint_path}", flush=True)

    print("\nFinal all-data training completed.")
    return _load_checkpoint(config.checkpoint_path)


def train_and_save(datasources, model_path: str | Path = MODEL_PATH) -> str:
    """Official private-board training entry: zero-init, fixed train range, JSON output."""
    table = _resolve_bar_table(datasources)
    model_path = Path(model_path)
    runtime_dir = model_path.parent

    config = FinalTrainConfig(
        years=(2019, 2020, 2021, 2022, 2023, 2024),
        bar_table=table,
        window_bars=480,
        epochs=10,
        batch_size=256,
        num_workers=0,
        prefetch_factor=2,
        query_chunk_days=10,
        seed=2026,
        cache_dir=str(runtime_dir / "deeplob_private_cache"),
        resume_path=str(runtime_dir / "deeplob_private_resume.pt"),
        checkpoint_path=str(model_path),
        metrics_path=str(runtime_dir / "deeplob_private_metrics.csv"),
        rebuild_cache=False,
        save_resume=False,
    )

    # train_final_yearwise creates a fresh model every call; resume is disabled.
    train_final_yearwise(config, resume=False)
    if not model_path.exists():
        raise RuntimeError(f"Training finished but model JSON was not created: {model_path}")
    return str(model_path)


if __name__ == "__main__":
    train_and_save({"bar5m": "bigalpha_2026_stock_bar5m"})
