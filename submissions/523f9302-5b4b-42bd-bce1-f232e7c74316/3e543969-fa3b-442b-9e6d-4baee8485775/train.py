"""BigAlpha final submission: exact-PIT three-seed Stage-A model.

Public inference loads ``weights.json``. Private retraining calls
``train_and_save`` and trains every parameter from random initialization on all
available labelled competition dates. Feature scaling and target clipping use
the sealed 2019-2022 exact-PIT reference period; later dates are transformed
with those fixed statistics.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import dai
except ImportError:  # Local static/synthetic tests.
    dai = None

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache_submission"
OUTPUT_ROOT = ROOT / ".models_submission"
MODEL_PATH = str(ROOT / "weights.json")
NORMALIZATION_END = np.datetime64("2022-12-31", "D")

SEQ_LEN_1M = 64
SEQ_LEN_5M = 48
CONTEXT_TOKENS = 12
CONTEXT_SCALE = 0.20
DATES_PER_STEP = 4
SOFT_RANK_TEMPERATURE = 0.5
CHAMPION_EPOCHS = 15
ADAPTER_EPOCHS = 8
CHAMPION_LR = 5e-4
ADAPTER_LR = 5e-4
SEEDS = (11, 29, 47)
QUERY_CHUNK_SIZE = 40
INFERENCE_BATCH = 1024
PRECOMPUTE_BATCH = 2048
MIN_VALID_FRACTION = 0.95

FEATURE_COLS_3 = [
    "adjust_factor",
    "high",
    "open",
    "low",
    "close",
    "deal_number",
    "volume",
    "amount",
    "ask_price1",
    "ask_price2",
    "ask_price3",
    "bid_price1",
    "bid_price2",
    "bid_price3",
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

FEATURE_COLS_5 = FEATURE_COLS_3 + [
    "ask_price4",
    "ask_price5",
    "bid_price4",
    "bid_price5",
    "ask_volume4",
    "ask_volume5",
    "bid_volume4",
    "bid_volume5",
    "ask_num_orders4",
    "ask_num_orders5",
    "bid_num_orders4",
    "bid_num_orders5",
]

LOG1P_BASE = {
    "deal_number",
    "volume",
    "amount",
    *[f"ask_volume{i}" for i in range(1, 6)],
    *[f"bid_volume{i}" for i in range(1, 6)],
    *[f"ask_num_orders{i}" for i in range(1, 6)],
    *[f"bid_num_orders{i}" for i in range(1, 6)],
}


@dataclass(frozen=True)
class DataSpec:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    universe_policy: str  # legacy600 | pit_full | pit_limited
    max_instruments: Optional[int]
    ffill_mode: str  # legacy | intraday
    depth_5m: int  # 3 or 5
    query_chunk_size: int = QUERY_CHUNK_SIZE
    min_valid_fraction: float = MIN_VALID_FRACTION

    @property
    def feature_cols_1m(self) -> List[str]:
        return list(FEATURE_COLS_3)

    @property
    def feature_cols_5m(self) -> List[str]:
        return list(FEATURE_COLS_3 if self.depth_5m == 3 else FEATURE_COLS_5)

    @property
    def exact_pit(self) -> bool:
        return self.universe_policy.startswith("pit")


@dataclass
class DiskDataset:
    cache_dir: Path
    x1: np.ndarray
    x5: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    has5: np.ndarray
    dates: np.ndarray
    instruments: np.ndarray
    mean1: np.ndarray
    std1: np.ndarray
    mean5: np.ndarray
    std5: np.ndarray
    metadata: dict


@dataclass
class TrainedEnsemble:
    spec: DataSpec
    seeds: Tuple[int, ...]
    models: List["StageADualFrequencyModel"]
    mean1: np.ndarray
    std1: np.ndarray
    mean5: np.ndarray
    std5: np.ndarray


def log_event(event: str, **fields) -> None:
    record = {"time": pd.Timestamp.utcnow().isoformat(), "event": event, **fields}
    print("PITLAB=" + json.dumps(record, ensure_ascii=False), flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, padding=0
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.pad(
            hidden.transpose(1, 2), (self.left_padding, 0)
        )
        return self.depthwise(hidden).transpose(1, 2)


class CausalMultiScaleTransformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        d_model: int = 80,
        nhead: int = 4,
        nlayers: int = 2,
        dim_ff: int = 320,
        seq_len: int = SEQ_LEN_1M,
        pooling: str = "last",
    ):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(n_feat, d_model)
        self.local_branches = nn.ModuleList(
            CausalDepthwiseConv(d_model, kernel) for kernel in (3, 7, 15)
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.local_gate = nn.Parameter(torch.zeros(d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        local = self.local_fusion(
            torch.cat([branch(projected) for branch in self.local_branches], dim=-1)
        )
        hidden = (
            projected
            + torch.sigmoid(self.local_gate) * local
            + self.pos[:, : features.shape[1]]
        )
        return self.encoder(
            hidden, mask=causal_attention_mask(hidden.shape[1], hidden.device)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        pooled = hidden[:, -1] if self.pooling == "last" else hidden.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class CausalFiveMinuteEncoder(nn.Module):
    def __init__(
        self,
        n_feat: int,
        d_model: int = 32,
        nhead: int = 4,
        nlayers: int = 1,
        dim_ff: int = 128,
        seq_len: int = SEQ_LEN_5M,
    ):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.local_branches = nn.ModuleList(
            CausalDepthwiseConv(d_model, kernel) for kernel in (3, 7, 15)
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.local_gate = nn.Parameter(torch.zeros(d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        local = self.local_fusion(
            torch.cat([branch(projected) for branch in self.local_branches], dim=-1)
        )
        hidden = (
            projected
            + torch.sigmoid(self.local_gate) * local
            + self.pos[:, : features.shape[1]]
        )
        return self.encoder(
            hidden, mask=causal_attention_mask(hidden.shape[1], hidden.device)
        )


class StageADualFrequencyModel(nn.Module):
    def __init__(
        self,
        close_branch: CausalMultiScaleTransformer,
        n_feat_5m: int,
        context_scale: float = CONTEXT_SCALE,
        freeze_close: bool = True,
    ):
        super().__init__()
        self.close_branch = close_branch
        self.context_branch = CausalFiveMinuteEncoder(n_feat=n_feat_5m)
        d_close = close_branch.proj.out_features
        self.query_norm = nn.LayerNorm(d_close)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_close,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
            kdim=32,
            vdim=32,
        )
        self.context_dropout = nn.Dropout(0.1)
        self.context_scale = float(context_scale)
        if freeze_close:
            self.freeze_close_branch()

    def freeze_close_branch(self) -> None:
        for parameter in self.close_branch.parameters():
            parameter.requires_grad_(False)
        self.close_branch.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.close_branch.eval()
        return self

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("close_branch.") and parameter.requires_grad:
                yield parameter

    def encode_context(
        self,
        close_last: torch.Tensor,
        features_5m: torch.Tensor,
        has5: torch.Tensor,
    ) -> torch.Tensor:
        hidden5 = self.context_branch(features_5m)
        key_value = hidden5[:, -CONTEXT_TOKENS:]
        query = self.query_norm(close_last.detach()).unsqueeze(1)
        context, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        context = self.context_dropout(context[:, 0])
        return has5.to(close_last.dtype).unsqueeze(1) * context

    def forward_from_close_hidden(
        self,
        close_last: torch.Tensor,
        features_5m: torch.Tensor,
        has5: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_context(close_last, features_5m, has5)
        return self.close_branch.head(
            close_last + self.context_scale * context
        ).squeeze(-1)

    def forward(
        self,
        features_1m: torch.Tensor,
        features_5m: torch.Tensor,
        has5: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            close_last = self.close_branch.encode(features_1m)[:, -1]
        return self.forward_from_close_hidden(close_last, features_5m, has5)


def _pearson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = prediction - prediction.mean()
    py = target - target.mean()
    return (px * py).sum() / (px.norm() * py.norm() + 1e-8)


def _soft_rank(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean()
    scale = centered.pow(2).mean().sqrt().clamp_min(1e-6)
    normalized = centered / scale
    differences = normalized[:, None] - normalized[None, :]
    return 1.0 + torch.sigmoid(differences / SOFT_RANK_TEMPERATURE).sum(dim=1)


class SoftSpearmanLoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        daily = []
        for group in torch.unique(groups, sorted=True):
            mask = groups == group
            if int(mask.sum()) < 3:
                continue
            prediction_rank = _soft_rank(prediction[mask])
            target_rank = torch.argsort(torch.argsort(target[mask])).to(
                prediction.dtype
            )
            daily.append(1.0 - _pearson(prediction_rank, target_rank))
        if not daily:
            return prediction.sum() * 0.0
        return torch.stack(daily).mean()


class MultiDateBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        groups: np.ndarray,
        dates_per_step: int,
        seed: int,
        shuffle: bool = True,
    ):
        groups = np.asarray(groups, dtype=np.int64)
        order = np.argsort(groups, kind="stable")
        sorted_groups = groups[order]
        boundaries = np.flatnonzero(
            np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True]
        )
        self.daily_indices = [
            order[boundaries[i] : boundaries[i + 1]].tolist()
            for i in range(len(boundaries) - 1)
        ]
        self.dates_per_step = int(dates_per_step)
        self.rng = random.Random(seed)
        self.shuffle = bool(shuffle)

    def __iter__(self) -> Iterator[List[int]]:
        order = list(range(len(self.daily_indices)))
        if self.shuffle:
            self.rng.shuffle(order)
        for begin in range(0, len(order), self.dates_per_step):
            yield [
                index
                for day in order[begin : begin + self.dates_per_step]
                for index in self.daily_indices[day]
            ]

    def __len__(self) -> int:
        return math.ceil(len(self.daily_indices) / self.dates_per_step)


class FeatureRunningStats:
    """Per-feature finite-value moments, allowing NaNs before mean imputation."""

    def __init__(self, n_feat: int):
        self.count = np.zeros(n_feat, np.int64)
        self.total = np.zeros(n_feat, np.float64)
        self.square_total = np.zeros(n_feat, np.float64)

    def update(self, values: np.ndarray) -> None:
        matrix = np.asarray(values, np.float64).reshape(-1, values.shape[-1])
        finite = np.isfinite(matrix)
        safe = np.where(finite, matrix, 0.0)
        self.count += finite.sum(axis=0)
        self.total += safe.sum(axis=0)
        self.square_total += (safe * safe).sum(axis=0)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if np.any(self.count <= 0):
            missing = np.flatnonzero(self.count <= 0).tolist()
            raise RuntimeError(f"no finite observations for feature indices {missing}")
        mean = self.total / self.count
        variance = self.square_total / self.count - mean**2
        std = np.sqrt(np.clip(variance, 0.0, None)) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)


def _chunks(values: Sequence[str], size: int) -> Iterator[List[str]]:
    values = list(values)
    for begin in range(0, len(values), size):
        yield values[begin : begin + size]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _hash_jsonable(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def universe_instruments(start_date: str, end_date: str) -> List[str]:
    if dai is None:
        raise RuntimeError("DAI is available only inside BigQuant AIStudio")
    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    return sorted(frame["instrument"].dropna().astype(str).unique().tolist())


def membership_keys(
    start_date: str,
    end_date: str,
    instruments: Optional[Sequence[str]] = None,
) -> set:
    if dai is None:
        raise RuntimeError("DAI is available only inside BigQuant AIStudio")
    filters: Dict[str, object] = {"date": [start_date, end_date]}
    if instruments is not None:
        filters["instrument"] = list(instruments)
    frame = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters=filters,
    ).df()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return set(zip(frame["date"], frame["instrument"].astype(str)))


def select_instruments(spec: DataSpec) -> List[str]:
    instruments = universe_instruments(spec.train_start, spec.train_end)
    if spec.universe_policy == "legacy600":
        limit = 600 if spec.max_instruments is None else int(spec.max_instruments)
        return instruments[:limit]
    if spec.max_instruments is not None:
        return instruments[: int(spec.max_instruments)]
    return instruments


def _query_chunk(
    table: str,
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    if dai is None:
        raise RuntimeError("DAI is available only inside BigQuant AIStudio")
    sql = (
        f"SELECT date, instrument, {', '.join(feature_cols)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    frame = dai.query(
        sql,
        filters={
            "date": [start_date, end_date],
            "instrument": list(instruments),
        },
    ).df()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    for column in feature_cols:
        if column in LOG1P_BASE:
            frame[column] = np.log1p(
                pd.to_numeric(frame[column], errors="coerce").clip(lower=0)
            )
    return frame


def audit_five_level_schema(
    table: str,
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
) -> dict:
    sample = list(instruments)[: min(20, len(instruments))]
    try:
        frame = _query_chunk(
            table,
            start_date,
            end_date,
            sample,
            FEATURE_COLS_5,
        )
    except Exception as error:  # AIStudio schema errors are backend-specific.
        return {"available": False, "error": repr(error)}
    if frame.empty:
        return {"available": False, "error": "query returned no rows"}
    added = [column for column in FEATURE_COLS_5 if column not in FEATURE_COLS_3]
    result = {"available": True, "rows": int(len(frame)), "columns": {}}
    for column in added:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        result["columns"][column] = {
            "finite_rate": float(np.isfinite(numeric.to_numpy()).mean()),
            "zero_rate": float((numeric.fillna(np.nan) == 0).mean()),
        }
    return result


def _prepare_features(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    ffill_mode: str,
) -> np.ndarray:
    local = frame[list(feature_cols)].copy()
    if ffill_mode == "legacy":
        local = local.ffill()
    elif ffill_mode == "intraday":
        day = frame["date"].dt.normalize()
        local = local.groupby(day, sort=False).ffill()
    else:
        raise ValueError(f"unknown ffill mode: {ffill_mode}")
    return local.to_numpy(np.float32, copy=True)


def _one_minute_samples(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    ffill_mode: str,
    need_label: bool,
    min_valid_fraction: float,
) -> Tuple[List[np.ndarray], List[np.float32], List[Tuple[pd.Timestamp, str]]]:
    windows: List[np.ndarray] = []
    targets: List[np.float32] = []
    keys: List[Tuple[pd.Timestamp, str]] = []
    if frame.empty:
        return windows, targets, keys
    instrument = str(frame["instrument"].iloc[0])
    frame = frame.sort_values("date").reset_index(drop=True)
    features = _prepare_features(frame, feature_cols, ffill_mode)
    normalized_dates = frame["date"].dt.normalize()
    day_positions = list(frame.groupby(normalized_dates, sort=False).indices.items())
    day_closes = []
    for _, positions in day_positions:
        positions = np.asarray(positions, dtype=np.int64)
        close_value = pd.to_numeric(
            frame.loc[int(positions[-1]), "close"], errors="coerce"
        )
        day_closes.append(float(close_value) if np.isfinite(close_value) else np.nan)

    for day_index, (day, positions) in enumerate(day_positions):
        date = pd.Timestamp(day)
        if date < start_ts or date > end_ts:
            continue
        positions = np.asarray(positions, dtype=np.int64)
        if ffill_mode == "legacy":
            end_position = int(positions[-1])
            if end_position + 1 < SEQ_LEN_1M:
                continue
            selected = np.arange(
                end_position - SEQ_LEN_1M + 1, end_position + 1, dtype=np.int64
            )
        else:
            if len(positions) < SEQ_LEN_1M:
                continue
            selected = positions[-SEQ_LEN_1M:]
        window = features[selected]
        valid_fraction = float(np.isfinite(window).mean())
        if ffill_mode == "legacy" and valid_fraction < 1.0:
            continue
        if ffill_mode == "intraday" and valid_fraction < min_valid_fraction:
            continue
        target: Optional[np.float32] = None
        if day_index + 1 < len(day_closes):
            current_close = day_closes[day_index]
            next_close = day_closes[day_index + 1]
            if current_close > 0 and np.isfinite(next_close):
                value = next_close / current_close - 1.0
                if np.isfinite(value):
                    target = np.float32(value)
        if need_label and target is None:
            continue
        windows.append(window)
        targets.append(target if target is not None else np.float32(np.nan))
        keys.append((date.normalize(), instrument))
    return windows, targets, keys


def _five_minute_map(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    ffill_mode: str,
    min_valid_fraction: float,
) -> Dict[Tuple[pd.Timestamp, str], np.ndarray]:
    output: Dict[Tuple[pd.Timestamp, str], np.ndarray] = {}
    if frame.empty:
        return output
    instrument = str(frame["instrument"].iloc[0])
    frame = frame.sort_values("date").reset_index(drop=True)
    features = _prepare_features(frame, feature_cols, ffill_mode)
    normalized_dates = frame["date"].dt.normalize()
    for day, positions in frame.groupby(normalized_dates, sort=False).indices.items():
        date = pd.Timestamp(day)
        if date < start_ts or date > end_ts:
            continue
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) < SEQ_LEN_5M:
            continue
        window = features[positions[-SEQ_LEN_5M:]]
        valid_fraction = float(np.isfinite(window).mean())
        if ffill_mode == "legacy" and valid_fraction < 1.0:
            continue
        if ffill_mode == "intraday" and valid_fraction < min_valid_fraction:
            continue
        output[(date.normalize(), instrument)] = window
    return output


def _cache_signature(
    spec: DataSpec,
    datasources: Mapping[str, str],
    instruments: Sequence[str],
) -> dict:
    return {
        "version": "final_submission_fixedstats_v1",
        "normalization_end": str(NORMALIZATION_END),
        "spec": asdict(spec),
        "datasources": dict(datasources),
        "feature_cols_1m": spec.feature_cols_1m,
        "feature_cols_5m": spec.feature_cols_5m,
        "instrument_hash": hashlib.sha256(
            "\n".join(instruments).encode("utf-8")
        ).hexdigest(),
    }


def _required_cache_files() -> List[str]:
    return [
        "x1.npy",
        "x5.npy",
        "targets.npy",
        "groups.npy",
        "has5.npy",
        "dates.npy",
        "instruments.npy",
        "stats.npz",
        "metadata.json",
    ]


def load_disk_dataset(cache_dir: Path, mmap_mode: Optional[str] = "r+") -> DiskDataset:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    stats = np.load(cache_dir / "stats.npz", allow_pickle=False)
    return DiskDataset(
        cache_dir=cache_dir,
        x1=np.load(cache_dir / "x1.npy", mmap_mode=mmap_mode),
        x5=np.load(cache_dir / "x5.npy", mmap_mode=mmap_mode),
        targets=np.load(cache_dir / "targets.npy", mmap_mode=mmap_mode),
        groups=np.load(cache_dir / "groups.npy", mmap_mode=mmap_mode),
        has5=np.load(cache_dir / "has5.npy", mmap_mode=mmap_mode),
        dates=np.load(cache_dir / "dates.npy", mmap_mode=mmap_mode),
        instruments=np.load(cache_dir / "instruments.npy", mmap_mode=mmap_mode),
        mean1=stats["mean1"].astype(np.float32),
        std1=stats["std1"].astype(np.float32),
        mean5=stats["mean5"].astype(np.float32),
        std5=stats["std5"].astype(np.float32),
        metadata=metadata,
    )


def build_training_cache(
    spec: DataSpec,
    datasources: Mapping[str, str],
    cache_root: Path = CACHE_ROOT,
    force: bool = False,
) -> DiskDataset:
    instruments = select_instruments(spec)
    signature = _cache_signature(spec, datasources, instruments)
    cache_dir = cache_root / spec.name
    metadata_path = cache_dir / "metadata.json"
    if not force and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            valid = metadata.get("signature_hash") == _hash_jsonable(signature)
        except Exception:
            valid = False
        if valid and all((cache_dir / name).exists() for name in _required_cache_files()):
            log_event("cache_reused", spec=spec.name, path=str(cache_dir))
            return load_disk_dataset(cache_dir)

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    exact_membership = None
    if spec.exact_pit:
        exact_membership = membership_keys(
            spec.train_start,
            spec.train_end,
            instruments,
        )
    start_ts = pd.Timestamp(spec.train_start)
    end_ts = pd.Timestamp(spec.train_end)
    buffer_start = (start_ts - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    stats1 = FeatureRunningStats(len(spec.feature_cols_1m))
    stats5 = FeatureRunningStats(len(spec.feature_cols_5m))
    shard_records: List[dict] = []
    total_samples = 0
    five_count = 0

    for chunk_index, chunk in enumerate(
        _chunks(instruments, spec.query_chunk_size), 1
    ):
        frame1 = _query_chunk(
            datasources["bar1m"],
            buffer_start,
            spec.train_end,
            chunk,
            spec.feature_cols_1m,
        )
        local_windows: List[np.ndarray] = []
        local_targets: List[np.float32] = []
        local_keys: List[Tuple[pd.Timestamp, str]] = []
        for _, instrument_frame in frame1.groupby("instrument", sort=False):
            windows, targets, keys = _one_minute_samples(
                instrument_frame,
                spec.feature_cols_1m,
                start_ts,
                end_ts,
                spec.ffill_mode,
                True,
                spec.min_valid_fraction,
            )
            for window, target, key in zip(windows, targets, keys):
                if exact_membership is not None and key not in exact_membership:
                    continue
                local_windows.append(window)
                local_targets.append(target)
                local_keys.append(key)
        del frame1
        gc.collect()
        if not local_windows:
            continue

        frame5 = _query_chunk(
            datasources["bar5m"],
            spec.train_start,
            spec.train_end,
            chunk,
            spec.feature_cols_5m,
        )
        five_map: Dict[Tuple[pd.Timestamp, str], np.ndarray] = {}
        for _, instrument_frame in frame5.groupby("instrument", sort=False):
            five_map.update(
                _five_minute_map(
                    instrument_frame,
                    spec.feature_cols_5m,
                    start_ts,
                    end_ts,
                    spec.ffill_mode,
                    spec.min_valid_fraction,
                )
            )
        del frame5
        gc.collect()

        x1 = np.stack(local_windows).astype(np.float32, copy=False)
        x5 = np.full(
            (len(local_keys), SEQ_LEN_5M, len(spec.feature_cols_5m)),
            np.nan,
            dtype=np.float32,
        )
        has5 = np.zeros(len(local_keys), np.float32)
        for index, key in enumerate(local_keys):
            window5 = five_map.get(key)
            if window5 is not None:
                x5[index] = window5
                has5[index] = 1.0
        dates = np.asarray(
            [np.datetime64(key[0].date(), "D") for key in local_keys],
            dtype="datetime64[D]",
        )
        reference_mask = dates <= NORMALIZATION_END
        if np.any(reference_mask):
            stats1.update(x1[reference_mask])
            reference_five = reference_mask & (has5 > 0)
            if np.any(reference_five):
                stats5.update(x5[reference_five])
        max_len = max(len(key[1]) for key in local_keys)
        codes = np.asarray([key[1] for key in local_keys], dtype=f"U{max_len}")
        targets = np.asarray(local_targets, np.float32)
        shard_path = shard_dir / f"shard_{chunk_index:04d}.npz"
        np.savez(
            shard_path,
            x1=x1,
            x5=x5,
            targets=targets,
            has5=has5,
            dates=dates,
            instruments=codes,
        )
        shard_records.append(
            {"path": shard_path.name, "samples": int(len(targets))}
        )
        total_samples += len(targets)
        five_count += int((has5 > 0).sum())
        log_event(
            "cache_shard_complete",
            spec=spec.name,
            chunk=chunk_index,
            samples=total_samples,
            five_minute_coverage=float(five_count / max(total_samples, 1)),
        )
        del x1, x5, targets, has5, dates, codes, local_windows, local_targets
        gc.collect()

    if total_samples <= 0:
        raise RuntimeError(f"{spec.name}: no training samples were built")
    mean1, std1 = stats1.finalize()
    mean5, std5 = stats5.finalize()

    target_values = []
    reference_target_values = []
    max_code_len = 1
    for record in shard_records:
        shard = np.load(shard_dir / record["path"], allow_pickle=False)
        shard_targets = shard["targets"].astype(np.float32)
        target_values.append(shard_targets)
        reference_mask = shard["dates"] <= NORMALIZATION_END
        if np.any(reference_mask):
            reference_target_values.append(shard_targets[reference_mask])
        max_code_len = max(max_code_len, int(shard["instruments"].dtype.itemsize // 4))
    all_targets = np.concatenate(target_values)
    clipping_reference = (
        np.concatenate(reference_target_values)
        if reference_target_values
        else all_targets
    )
    lower, upper = np.percentile(clipping_reference, [1, 99])
    np.clip(all_targets, lower, upper, out=all_targets)

    partials = {}
    partials["x1"] = open_memmap(
        cache_dir / "x1.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, SEQ_LEN_1M, len(spec.feature_cols_1m)),
    )
    partials["x5"] = open_memmap(
        cache_dir / "x5.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, SEQ_LEN_5M, len(spec.feature_cols_5m)),
    )
    partials["targets"] = open_memmap(
        cache_dir / "targets.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples,),
    )
    partials["has5"] = open_memmap(
        cache_dir / "has5.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples,),
    )
    partials["dates"] = open_memmap(
        cache_dir / "dates.partial.npy",
        mode="w+",
        dtype="datetime64[D]",
        shape=(total_samples,),
    )
    partials["instruments"] = open_memmap(
        cache_dir / "instruments.partial.npy",
        mode="w+",
        dtype=f"U{max_code_len}",
        shape=(total_samples,),
    )

    cursor = 0
    target_cursor = 0
    for record in shard_records:
        shard = np.load(shard_dir / record["path"], allow_pickle=False)
        count = int(record["samples"])
        x1 = shard["x1"].astype(np.float32, copy=False)
        x5 = shard["x5"].astype(np.float32, copy=False)
        normalized1 = (x1 - mean1) / std1
        normalized1[~np.isfinite(normalized1)] = 0.0
        normalized5 = (x5 - mean5) / std5
        normalized5[~np.isfinite(normalized5)] = 0.0
        has5 = shard["has5"].astype(np.float32)
        normalized5[has5 <= 0] = 0.0
        destination = slice(cursor, cursor + count)
        partials["x1"][destination] = normalized1
        partials["x5"][destination] = normalized5
        partials["targets"][destination] = all_targets[
            target_cursor : target_cursor + count
        ]
        partials["has5"][destination] = has5
        partials["dates"][destination] = shard["dates"]
        partials["instruments"][destination] = shard["instruments"]
        cursor += count
        target_cursor += count
    for array in partials.values():
        array.flush()
    del partials
    gc.collect()

    dates = np.load(cache_dir / "dates.partial.npy", mmap_mode="r")
    groups, _ = pd.factorize(pd.to_datetime(np.asarray(dates)), sort=True)
    group_array = open_memmap(
        cache_dir / "groups.partial.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_samples,),
    )
    group_array[:] = groups.astype(np.int64)
    group_array.flush()
    del group_array, dates

    for name in ["x1", "x5", "targets", "has5", "dates", "instruments", "groups"]:
        os.replace(cache_dir / f"{name}.partial.npy", cache_dir / f"{name}.npy")
    np.savez(
        cache_dir / "stats.npz",
        mean1=mean1,
        std1=std1,
        mean5=mean5,
        std5=std5,
    )
    metadata = {
        **signature,
        "signature_hash": _hash_jsonable(signature),
        "samples": int(total_samples),
        "days": int(len(np.unique(groups))),
        "instruments_in_union": int(len(instruments)),
        "five_minute_coverage": float(five_count / total_samples),
        "target_clip": [float(lower), float(upper)],
        "preprocessing_reference_end": str(NORMALIZATION_END),
        "estimated_bytes": int(
            total_samples
            * 4
            * (
                SEQ_LEN_1M * len(spec.feature_cols_1m)
                + SEQ_LEN_5M * len(spec.feature_cols_5m)
                + 3
            )
        ),
    }
    _atomic_json(metadata_path, metadata)
    shutil.rmtree(shard_dir, ignore_errors=True)
    log_event(
        "cache_complete",
        spec_name=spec.name,
        samples=metadata["samples"],
        days=metadata["days"],
        instruments_in_union=metadata["instruments_in_union"],
        five_minute_coverage=metadata["five_minute_coverage"],
        estimated_bytes=metadata["estimated_bytes"],
    )
    return load_disk_dataset(cache_dir)


def cleanup_large_cache(cache_dir: Path) -> None:
    """Remove only large arrays after models/results are safely persisted."""
    for name in ["x1.npy", "x5.npy", "targets.npy", "groups.npy", "has5.npy"]:
        path = cache_dir / name
        if path.exists():
            path.unlink()
    for path in cache_dir.glob("close_hidden_seed*.npy"):
        path.unlink()
    log_event("large_cache_removed", path=str(cache_dir))


def champion_config(n_feat: int) -> dict:
    return {
        "n_feat": int(n_feat),
        "d_model": 80,
        "nhead": 4,
        "nlayers": 2,
        "dim_ff": 320,
        "seq_len": SEQ_LEN_1M,
        "pooling": "last",
    }


def train_champion(
    dataset: DiskDataset,
    seed: int,
    device: torch.device,
    epochs: int = CHAMPION_EPOCHS,
) -> CausalMultiScaleTransformer:
    set_seed(seed)
    model = CausalMultiScaleTransformer(**champion_config(dataset.x1.shape[-1])).to(device)
    x1_tensor = torch.from_numpy(np.asarray(dataset.x1))
    target_tensor = torch.from_numpy(np.asarray(dataset.targets))
    group_tensor = torch.from_numpy(np.asarray(dataset.groups))
    loader = DataLoader(
        TensorDataset(x1_tensor, target_tensor, group_tensor),
        batch_sampler=MultiDateBatchSampler(dataset.groups, DATES_PER_STEP, seed),
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CHAMPION_LR)
    objective = SoftSpearmanLoss().to(device)
    for epoch in range(1, epochs + 1):
        started = time.time()
        losses = []
        model.train()
        for batch_x1, batch_target, batch_group in loader:
            batch_x1 = batch_x1.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_group = batch_group.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x1)
            loss = objective(prediction, batch_target, batch_group)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite champion loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        log_event(
            "champion_epoch",
            seed=int(seed),
            epoch=epoch,
            loss=float(np.mean(losses)),
            elapsed_seconds=round(time.time() - started, 2),
        )
    model.eval()
    return model


def precompute_close_hidden(
    champion: CausalMultiScaleTransformer,
    dataset: DiskDataset,
    seed: int,
    device: torch.device,
    force: bool = False,
) -> np.ndarray:
    path = dataset.cache_dir / f"close_hidden_seed{seed}.npy"
    expected_shape = (len(dataset.targets), champion.proj.out_features)
    if path.exists() and not force:
        existing = np.load(path, mmap_mode="r")
        if existing.shape == expected_shape:
            return existing
        path.unlink()
    temporary = dataset.cache_dir / f"close_hidden_seed{seed}.partial.npy"
    output = open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    champion = champion.to(device)
    champion.eval()
    started = time.time()
    with torch.inference_mode():
        for begin in range(0, len(dataset.x1), PRECOMPUTE_BATCH):
            batch = torch.from_numpy(
                np.asarray(dataset.x1[begin : begin + PRECOMPUTE_BATCH])
            ).to(device, non_blocking=True)
            hidden = champion.encode(batch)[:, -1]
            output[begin : begin + len(batch)] = hidden.float().cpu().numpy()
    output.flush()
    del output
    os.replace(temporary, path)
    champion.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_event(
        "close_hidden_complete",
        seed=int(seed),
        shape=list(expected_shape),
        elapsed_seconds=round(time.time() - started, 2),
    )
    return np.load(path, mmap_mode="r+")


def train_adapter(
    champion: CausalMultiScaleTransformer,
    close_hidden: np.ndarray,
    dataset: DiskDataset,
    seed: int,
    device: torch.device,
    epochs: int = ADAPTER_EPOCHS,
) -> StageADualFrequencyModel:
    set_seed(seed)
    model = StageADualFrequencyModel(
        close_branch=champion,
        n_feat_5m=int(dataset.x5.shape[-1]),
        freeze_close=True,
    ).to(device)
    hidden_tensor = torch.from_numpy(np.asarray(close_hidden))
    x5_tensor = torch.from_numpy(np.asarray(dataset.x5))
    target_tensor = torch.from_numpy(np.asarray(dataset.targets))
    group_tensor = torch.from_numpy(np.asarray(dataset.groups))
    has5_tensor = torch.from_numpy(np.asarray(dataset.has5))
    loader = DataLoader(
        TensorDataset(
            hidden_tensor,
            x5_tensor,
            target_tensor,
            group_tensor,
            has5_tensor,
        ),
        batch_sampler=MultiDateBatchSampler(dataset.groups, DATES_PER_STEP, seed),
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    parameters = list(model.adapter_parameters())
    optimizer = torch.optim.Adam(parameters, lr=ADAPTER_LR)
    objective = SoftSpearmanLoss().to(device)
    for epoch in range(1, epochs + 1):
        started = time.time()
        losses = []
        model.train()
        for hidden, x5, target, group, has5 in loader:
            hidden = hidden.to(device, non_blocking=True)
            x5 = x5.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            group = group.to(device, non_blocking=True)
            has5 = has5.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_from_close_hidden(hidden, x5, has5)
            loss = objective(prediction, target, group)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite adapter loss")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        log_event(
            "adapter_epoch",
            seed=int(seed),
            epoch=epoch,
            loss=float(np.mean(losses)),
            elapsed_seconds=round(time.time() - started, 2),
        )
    model.eval()
    return model


def save_model_checkpoint(
    model: StageADualFrequencyModel,
    dataset: DiskDataset,
    spec: DataSpec,
    seed: int,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": asdict(spec),
        "seed": int(seed),
        "mean1": dataset.mean1,
        "std1": dataset.std1,
        "mean5": dataset.mean5,
        "std5": dataset.std5,
        "state_dict": model.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_model_checkpoint(path: Path, map_location: str | torch.device = "cpu"):
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    spec = DataSpec(**payload["spec"])
    champion = CausalMultiScaleTransformer(**champion_config(len(spec.feature_cols_1m)))
    model = StageADualFrequencyModel(
        close_branch=champion,
        n_feat_5m=len(spec.feature_cols_5m),
        freeze_close=True,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload, model


def train_seed_for_spec(
    spec: DataSpec,
    datasources: Mapping[str, str],
    seed: int,
    model_dir: Path,
    device: torch.device,
    champion_epochs: int = CHAMPION_EPOCHS,
    adapter_epochs: int = ADAPTER_EPOCHS,
    force_cache: bool = False,
    force_model: bool = False,
) -> Path:
    model_path = model_dir / spec.name / f"seed_{seed}.pt"
    if model_path.exists() and not force_model:
        log_event("model_reused", spec=spec.name, seed=int(seed), path=str(model_path))
        return model_path
    dataset = build_training_cache(spec, datasources, force=force_cache)
    champion = train_champion(dataset, seed, device, epochs=champion_epochs)
    close_hidden = precompute_close_hidden(champion, dataset, seed, device)
    model = train_adapter(
        champion,
        close_hidden,
        dataset,
        seed,
        device,
        epochs=adapter_epochs,
    )
    save_model_checkpoint(model.to("cpu"), dataset, spec, seed, model_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_path


def load_ensemble(paths: Sequence[Path]) -> TrainedEnsemble:
    payloads = []
    models = []
    for path in paths:
        payload, model = load_model_checkpoint(path, map_location="cpu")
        payloads.append(payload)
        models.append(model)
    specs = [DataSpec(**payload["spec"]) for payload in payloads]
    if len({json.dumps(asdict(spec), sort_keys=True) for spec in specs}) != 1:
        raise ValueError("ensemble checkpoints use different specs")
    first = payloads[0]
    return TrainedEnsemble(
        spec=specs[0],
        seeds=tuple(int(payload["seed"]) for payload in payloads),
        models=models,
        mean1=np.asarray(first["mean1"], np.float32),
        std1=np.asarray(first["std1"], np.float32),
        mean5=np.asarray(first["mean5"], np.float32),
        std5=np.asarray(first["std5"], np.float32),
    )


def _build_inference_chunk(
    frame1: pd.DataFrame,
    frame5: pd.DataFrame,
    spec: DataSpec,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    membership: set,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[pd.Timestamp, str]]]:
    local_x1: List[np.ndarray] = []
    local_keys: List[Tuple[pd.Timestamp, str]] = []
    for _, instrument_frame in frame1.groupby("instrument", sort=False):
        windows, _, keys = _one_minute_samples(
            instrument_frame,
            spec.feature_cols_1m,
            start_ts,
            end_ts,
            spec.ffill_mode,
            False,
            spec.min_valid_fraction,
        )
        for window, key in zip(windows, keys):
            if key in membership:
                local_x1.append(window)
                local_keys.append(key)
    if not local_x1:
        return (
            np.empty((0, SEQ_LEN_1M, len(spec.feature_cols_1m)), np.float32),
            np.empty((0, SEQ_LEN_5M, len(spec.feature_cols_5m)), np.float32),
            np.empty((0,), np.float32),
            [],
        )
    five_map: Dict[Tuple[pd.Timestamp, str], np.ndarray] = {}
    for _, instrument_frame in frame5.groupby("instrument", sort=False):
        five_map.update(
            _five_minute_map(
                instrument_frame,
                spec.feature_cols_5m,
                start_ts,
                end_ts,
                spec.ffill_mode,
                spec.min_valid_fraction,
            )
        )
    x1 = np.stack(local_x1).astype(np.float32)
    x5 = np.full(
        (len(local_keys), SEQ_LEN_5M, len(spec.feature_cols_5m)),
        np.nan,
        np.float32,
    )
    has5 = np.zeros(len(local_keys), np.float32)
    for index, key in enumerate(local_keys):
        value = five_map.get(key)
        if value is not None:
            x5[index] = value
            has5[index] = 1.0
    return x1, x5, has5, local_keys


def predict_ensemble(
    ensemble: TrainedEnsemble,
    datasources: Mapping[str, str],
    start_date: str,
    end_date: str,
    device: torch.device,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    batch_size: int = INFERENCE_BATCH,
) -> Tuple[pd.DataFrame, dict]:
    spec = ensemble.spec
    instruments = universe_instruments(start_date, end_date)
    membership = membership_keys(start_date, end_date, instruments)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    buffer_start = (start_ts - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    raw_parts = []
    coverage_rows = []
    for chunk_index, chunk in enumerate(_chunks(instruments, query_chunk_size), 1):
        frame1 = _query_chunk(
            datasources["bar1m"],
            buffer_start,
            end_date,
            chunk,
            spec.feature_cols_1m,
        )
        frame5 = _query_chunk(
            datasources["bar5m"],
            start_date,
            end_date,
            chunk,
            spec.feature_cols_5m,
        )
        x1, x5, has5, keys = _build_inference_chunk(
            frame1, frame5, spec, start_ts, end_ts, membership
        )
        del frame1, frame5
        gc.collect()
        if not keys:
            continue
        x1 = (x1 - ensemble.mean1) / ensemble.std1
        x1[~np.isfinite(x1)] = 0.0
        x5 = (x5 - ensemble.mean5) / ensemble.std5
        x5[~np.isfinite(x5)] = 0.0
        x5[has5 <= 0] = 0.0
        output = pd.DataFrame(keys, columns=["date", "instrument"])
        for model_index, model in enumerate(ensemble.models):
            model = model.to(device)
            model.eval()
            champion_parts = []
            stage_parts = []
            with torch.inference_mode():
                for begin in range(0, len(x1), batch_size):
                    one = torch.from_numpy(x1[begin : begin + batch_size]).to(
                        device, non_blocking=True
                    )
                    five = torch.from_numpy(x5[begin : begin + batch_size]).to(
                        device, non_blocking=True
                    )
                    mask = torch.from_numpy(has5[begin : begin + batch_size]).to(
                        device, non_blocking=True
                    )
                    close_last = model.close_branch.encode(one)[:, -1]
                    champion_score = model.close_branch.head(close_last).squeeze(-1)
                    stage_score = model.forward_from_close_hidden(
                        close_last, five, mask
                    )
                    champion_parts.append(champion_score.float().cpu().numpy())
                    stage_parts.append(stage_score.float().cpu().numpy())
            output[f"champion_raw_{model_index}"] = np.concatenate(champion_parts)
            output[f"stage_raw_{model_index}"] = np.concatenate(stage_parts)
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raw_parts.append(output)
        coverage_rows.append(
            {
                "chunk": chunk_index,
                "rows": len(output),
                "five_minute_coverage": float(has5.mean()),
            }
        )
        log_event(
            "prediction_chunk_complete",
            spec=spec.name,
            chunk=chunk_index,
            rows=int(len(output)),
            five_minute_coverage=float(has5.mean()),
        )
    if not raw_parts:
        raise RuntimeError(f"{spec.name}: prediction generated no rows")
    raw = pd.concat(raw_parts, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    champion_ranks = []
    stage_ranks = []
    for model_index in range(len(ensemble.models)):
        champion_rank = raw.groupby("date")[f"champion_raw_{model_index}"].rank(
            pct=True, method="average"
        )
        stage_rank = raw.groupby("date")[f"stage_raw_{model_index}"].rank(
            pct=True, method="average"
        )
        champion_ranks.append(champion_rank.to_numpy(np.float64))
        stage_ranks.append(stage_rank.to_numpy(np.float64))
    result = raw[["date", "instrument"]].copy()
    result["champion"] = np.mean(champion_ranks, axis=0)
    result["stage_a"] = np.mean(stage_ranks, axis=0)
    for beta in (0.10, 0.15, 0.20):
        result[f"blend_b{int(beta * 100):02d}"] = (
            (1.0 - beta) * result["champion"] + beta * result["stage_a"]
        )
    result = result.drop_duplicates(["date", "instrument"])
    daily_counts = result.groupby("date").size()
    diagnostics = {
        "rows": int(len(result)),
        "days": int(result["date"].nunique()),
        "coverage_min": int(daily_counts.min()),
        "coverage_median": float(daily_counts.median()),
        "coverage_max": int(daily_counts.max()),
        "five_minute_coverage": float(
            np.average(
                [row["five_minute_coverage"] for row in coverage_rows],
                weights=[row["rows"] for row in coverage_rows],
            )
        ),
    }
    return result.sort_values(["date", "instrument"]).reset_index(drop=True), diagnostics


def serialize_state(state_dict: Mapping[str, torch.Tensor]) -> dict:
    output = {}
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu()
        output[name] = {
            "dtype": str(value.dtype).replace("torch.", ""),
            "shape": list(value.shape),
            "data": value.reshape(-1).tolist(),
        }
    return output


def save_json_ensemble(
    checkpoint_paths: Sequence[Path],
    candidate: str,
    beta: float,
    path: Path,
    training_note: str,
) -> Path:
    ensemble = load_ensemble(checkpoint_paths)
    payload = {
        "name": f"long_history_{ensemble.spec.name}_{candidate}",
        "model_kind": "dual_frequency_rank_blend",
        "submission_candidate": candidate,
        "output_blend_beta": float(beta),
        "spec": asdict(ensemble.spec),
        "feature_cols_1m": ensemble.spec.feature_cols_1m,
        "feature_cols_5m": ensemble.spec.feature_cols_5m,
        "mean_1m": ensemble.mean1.tolist(),
        "std_1m": ensemble.std1.tolist(),
        "mean_5m": ensemble.mean5.tolist(),
        "std_5m": ensemble.std5.tolist(),
        "seeds": list(ensemble.seeds),
        "context_scale": CONTEXT_SCALE,
        "champion_epochs": CHAMPION_EPOCHS,
        "adapter_epochs": ADAPTER_EPOCHS,
        "training_note": training_note,
        "states": [
            {
                "seed": int(seed),
                "state_dict": serialize_state(model.state_dict()),
            }
            for seed, model in zip(ensemble.seeds, ensemble.models)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


# ============================================================================
# Competition submission entrypoints
# ============================================================================

def _deserialize_state(
    tensors: Mapping[str, Mapping[str, object]],
) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    for name, item in tensors.items():
        dtype_name = str(item["dtype"])
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported tensor dtype: {dtype_name}")
        tensor = torch.tensor(item["data"], dtype=dtype_map[dtype_name])
        output[name] = tensor.reshape(tuple(int(x) for x in item["shape"]))
    return output


def load_json_ensemble(
    path: str | Path = MODEL_PATH,
) -> Tuple[dict, TrainedEnsemble]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("submission_candidate") != "stage_a":
        raise ValueError("final submission requires submission_candidate='stage_a'")
    if abs(float(payload.get("output_blend_beta", -1.0)) - 1.0) > 1e-12:
        raise ValueError("final submission requires output_blend_beta=1.0")
    spec = DataSpec(**payload["spec"])
    models: List[StageADualFrequencyModel] = []
    seeds: List[int] = []
    for item in payload["states"]:
        champion = CausalMultiScaleTransformer(
            **champion_config(len(payload["feature_cols_1m"]))
        )
        model = StageADualFrequencyModel(
            close_branch=champion,
            n_feat_5m=len(payload["feature_cols_5m"]),
            freeze_close=True,
        )
        model.load_state_dict(_deserialize_state(item["state_dict"]), strict=True)
        model.eval()
        models.append(model)
        seeds.append(int(item["seed"]))
    expected = tuple(int(x) for x in payload["seeds"])
    if tuple(seeds) != expected:
        raise ValueError(f"state seed order mismatch: {seeds} != {expected}")
    if expected != SEEDS:
        raise ValueError(f"unexpected seed set: {expected}")
    ensemble = TrainedEnsemble(
        spec=spec,
        seeds=tuple(seeds),
        models=models,
        mean1=np.asarray(payload["mean_1m"], np.float32),
        std1=np.asarray(payload["std_1m"], np.float32),
        mean5=np.asarray(payload["mean_5m"], np.float32),
        std5=np.asarray(payload["std_5m"], np.float32),
    )
    for values, name in [
        (ensemble.std1, "std_1m"),
        (ensemble.std5, "std_5m"),
    ]:
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"invalid {name}")
    return payload, ensemble


def _available_training_range() -> Tuple[str, str]:
    """Use every date exposed by the official PIT universe table."""
    if dai is None:
        raise RuntimeError("DAI is available only inside BigQuant AIStudio")
    frame = dai.query(
        "SELECT DISTINCT date FROM bigalpha_2026_instruments ORDER BY date"
    ).df()
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values()
    if dates.empty:
        raise RuntimeError("official PIT universe contains no dates")
    start = dates.iloc[0].normalize().strftime("%Y-%m-%d")
    end = dates.iloc[-1].normalize().strftime("%Y-%m-%d 23:59:59")
    return start, end


def _submission_spec(start_date: str, end_date: str) -> DataSpec:
    return DataSpec(
        name="final_stage_a_exact_pit_all_labelled_dates",
        train_start=str(start_date),
        train_end=str(end_date),
        validation_start=str(start_date),
        validation_end=str(end_date),
        universe_policy="pit_full",
        max_instruments=None,
        ffill_mode="legacy",
        depth_5m=3,
        query_chunk_size=QUERY_CHUNK_SIZE,
        min_valid_fraction=MIN_VALID_FRACTION,
    )


def train_and_save(
    datasources: Mapping[str, str],
    model_path: str = MODEL_PATH,
) -> str:
    """Private-training entrypoint. Every parameter is trained from scratch."""
    missing = {"bar1m", "bar5m"}.difference(datasources)
    if missing:
        raise KeyError(f"missing required datasource keys: {sorted(missing)}")
    start_date, end_date = _available_training_range()
    spec = _submission_spec(start_date, end_date)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths: List[Path] = []
    for seed in SEEDS:
        model_paths.append(
            train_seed_for_spec(
                spec=spec,
                datasources=datasources,
                seed=int(seed),
                model_dir=OUTPUT_ROOT,
                device=device,
                champion_epochs=CHAMPION_EPOCHS,
                adapter_epochs=ADAPTER_EPOCHS,
                force_cache=False,
                force_model=False,
            )
        )
    saved = save_json_ensemble(
        checkpoint_paths=model_paths,
        candidate="stage_a",
        beta=1.0,
        path=Path(model_path),
        training_note=(
            "Private retraining from random initialization on all labelled exact "
            "daily PIT CSI1000 samples. Feature scaling and target clipping use "
            "the sealed 2019-2022 reference period. Final output is pure Stage A."
        ),
    )
    log_event(
        "submission_training_complete",
        model_path=str(saved),
        train_start=start_date,
        train_end=end_date,
        seeds=list(SEEDS),
    )
    return str(saved)


def predict_submission_scores(
    datasources: Mapping[str, str],
    start_date: str,
    end_date: str,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """Public/private inference entrypoint returning the pure Stage-A ensemble."""
    missing = {"bar1m", "bar5m"}.difference(datasources)
    if missing:
        raise KeyError(f"missing required datasource keys: {sorted(missing)}")
    _, ensemble = load_json_ensemble(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored, diagnostics = predict_ensemble(
        ensemble=ensemble,
        datasources=datasources,
        start_date=str(start_date),
        end_date=str(end_date),
        device=device,
        query_chunk_size=QUERY_CHUNK_SIZE,
        batch_size=INFERENCE_BATCH,
    )
    output = scored[["date", "instrument", "stage_a"]].rename(
        columns={"stage_a": "score"}
    )
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output = output.drop_duplicates(["date", "instrument"])
    if not np.isfinite(output["score"].to_numpy(np.float64)).all():
        raise FloatingPointError("prediction contains non-finite scores")
    output = output.sort_values(["date", "instrument"]).reset_index(drop=True)
    log_event("submission_prediction_complete", **diagnostics)
    return output[["date", "instrument", "score"]]
