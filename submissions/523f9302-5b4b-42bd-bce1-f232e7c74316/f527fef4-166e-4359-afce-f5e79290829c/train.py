"""BigAlpha Stage-A A5-Lite DeepLOB-style adapter for AIStudio.

Design goals
------------
1. Preserve the submitted Tier-1 one-minute champion exactly.
2. Freeze the champion branch and train a DeepLOB-style 5-minute context
   encoder plus one last-token cross-attention adapter.
3. Feed only official raw fields to learned layers. The order-book tensor is a
   pure reshape of the six price/volume/order-count groups across three levels;
   no spread, imbalance, OFI, rolling statistic, or other derived input exists.
4. Cache aligned 1m/5m tensors and the frozen champion close hidden states so
   repeated seeds/experiments do not repeat DAI queries or the 1m encoder.
5. Keep public inference self-contained in ``weights.json`` after training.

The development workflow needs ``champion_weights.json`` beside this file.
``train_and_save`` creates the Stage-A ``weights.json``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Required by torch deterministic algorithms for CUDA/cuBLAS matmul.  Set it
# before the first CUDA operation so GPU smoke/full runs remain reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import dai
except ImportError:  # Local syntax/model tests do not have AIStudio DAI.
    dai = None

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


HERE = os.path.dirname(os.path.abspath(__file__))
CHAMPION_MODEL_PATH = os.path.join(HERE, "champion_weights.json")
MODEL_PATH = os.path.join(HERE, "weights.json")
CACHE_DIR = os.path.join(HERE, ".cache_dualfreq_stageA")

TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31 23:59:59"
SEQ_LEN_1M = 64
SEQ_LEN_5M = 48
CONTEXT_TOKENS = 12
D_MODEL_5M = 64
NHEAD_5M = 4
DIM_FF_5M = 256
NLAYERS_5M = 1
LOB_CHANNELS = 32
LOB_TEMPORAL_DILATIONS = (1, 2, 4)
CONTEXT_SCALE = 0.20
ADAPTER_EPOCHS = 8
LR_ADAPTER = 5e-4
DATES_PER_STEP = 4
MAX_TRAIN_INSTRUMENTS = 600
QUERY_CHUNK_SIZE = 100
INFERENCE_BATCH = 512
PRECOMPUTE_BATCH = 1024
SOFT_RANK_TEMPERATURE = 0.5
CACHE_VERSION = "dualfreq_stageA_A5Lite_deeplob_3L_U600_v1"
MIN_TRAINABLE_PARAMETERS = 100_000
USE_AMP = True

FEATURE_COLS = [
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
LOG1P_COLS = [
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

# These groups only describe how official raw columns are presented to learned
# convolutions. No arithmetic or fixed cross-field operator is applied.
BAR_FEATURE_NAMES = FEATURE_COLS[:8]
LOB_GROUP_NAMES = [
    ["ask_price1", "ask_price2", "ask_price3"],
    ["bid_price1", "bid_price2", "bid_price3"],
    ["ask_volume1", "ask_volume2", "ask_volume3"],
    ["bid_volume1", "bid_volume2", "bid_volume3"],
    ["ask_num_orders1", "ask_num_orders2", "ask_num_orders3"],
    ["bid_num_orders1", "bid_num_orders2", "bid_num_orders3"],
]
BAR_INPUT_INDICES = [FEATURE_COLS.index(name) for name in BAR_FEATURE_NAMES]
LOB_INPUT_INDICES = [
    [FEATURE_COLS.index(name) for name in group] for group in LOB_GROUP_NAMES
]

FIVE_MINUTE_CFG = {
    "n_feat": len(FEATURE_COLS),
    "d_model": D_MODEL_5M,
    "nhead": NHEAD_5M,
    "nlayers": NLAYERS_5M,
    "dim_ff": DIM_FF_5M,
    "seq_len": SEQ_LEN_5M,
    "bar_indices": BAR_INPUT_INDICES,
    "lob_indices": LOB_INPUT_INDICES,
    "lob_channels": LOB_CHANNELS,
    "temporal_dilations": list(LOB_TEMPORAL_DILATIONS),
}


@dataclass
class CachedDataset:
    x1: np.ndarray
    x5: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    has5: np.ndarray
    dates: np.ndarray
    instruments: np.ndarray
    mean5: np.ndarray
    std5: np.ndarray
    metadata: dict


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
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        # Stable adapter gradients are more important than flash-attention speed.
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
    """The original Tier-1 one-minute champion architecture."""

    def __init__(
        self,
        n_feat: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        seq_len: int,
        pooling: str,
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
            d_model,
            nhead,
            dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        local = self.local_fusion(
            torch.cat(
                [branch(projected) for branch in self.local_branches], dim=-1
            )
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


class CausalResidualTemporalBlock(nn.Module):
    """Learned causal temporal interaction with a gated residual path."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.norm = nn.LayerNorm(channels)
        self.temporal = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            padding=0,
        )
        self.output = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        local = self.norm(hidden).transpose(1, 2)
        local = torch.nn.functional.pad(local, (self.left_padding, 0))
        local = torch.nn.functional.glu(self.temporal(local), dim=1)
        local = self.output(local).transpose(1, 2)
        return residual + self.dropout(local)


class DeepLOBCausalFiveMinuteEncoder(nn.Module):
    """Raw three-level book encoder with learned depth/time interactions.

    The six-by-three order-book tensor is formed only by indexing and reshaping
    official raw columns. Every cross-field and cross-level operation below has
    trainable parameters.
    """

    def __init__(
        self,
        n_feat: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        seq_len: int,
        bar_indices: Sequence[int],
        lob_indices: Sequence[Sequence[int]],
        lob_channels: int,
        temporal_dilations: Sequence[int],
    ):
        super().__init__()
        flat_lob_indices = [
            int(index) for group in lob_indices for index in group
        ]
        if len(bar_indices) + len(flat_lob_indices) != n_feat:
            raise ValueError("DeepLOB raw-field layout must cover every input field")
        if any(len(group) != 3 for group in lob_indices):
            raise ValueError("DeepLOB expects exactly three raw levels per group")
        self.lob_groups = len(lob_indices)
        self.lob_levels = 3
        self.register_buffer(
            "_bar_indices",
            torch.tensor(list(bar_indices), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_lob_indices",
            torch.tensor(flat_lob_indices, dtype=torch.long),
            persistent=False,
        )

        # Kernel width three learns the complete three-level shape for every
        # time step. It does not encode any fixed spread/imbalance formula.
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(
                self.lob_groups,
                lob_channels,
                kernel_size=(1, self.lob_levels),
                padding=0,
            ),
            nn.GELU(),
        )
        self.depth_norm = nn.LayerNorm(lob_channels)
        self.bar_encoder = nn.Sequential(
            nn.Linear(len(bar_indices), lob_channels),
            nn.GELU(),
            nn.LayerNorm(lob_channels),
        )
        self.input_fusion = nn.Sequential(
            nn.Linear(2 * lob_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.temporal_blocks = nn.ModuleList(
            CausalResidualTemporalBlock(
                d_model,
                kernel_size=3,
                dilation=int(dilation),
            )
            for dilation in temporal_dilations
        )
        self.post_temporal_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = features.shape
        bar_raw = torch.index_select(features, -1, self._bar_indices)
        lob_raw = torch.index_select(features, -1, self._lob_indices)
        lob_raw = lob_raw.reshape(
            batch, steps, self.lob_groups, self.lob_levels
        ).permute(0, 2, 1, 3)
        book_hidden = self.depth_encoder(lob_raw).squeeze(-1).transpose(1, 2)
        book_hidden = self.depth_norm(book_hidden)
        bar_hidden = self.bar_encoder(bar_raw)
        hidden = self.input_fusion(
            torch.cat([bar_hidden, book_hidden], dim=-1)
        )
        for block in self.temporal_blocks:
            hidden = block(hidden)
        hidden = hidden + self.post_temporal_ffn(hidden)
        hidden = hidden + self.pos[:, :steps]
        return self.encoder(
            hidden, mask=causal_attention_mask(hidden.shape[1], hidden.device)
        )


class StageADualFrequencyModel(nn.Module):
    """Frozen champion plus a trainable 5m residual adapter."""

    def __init__(
        self,
        close_branch: CausalMultiScaleTransformer,
        five_minute_cfg: dict = FIVE_MINUTE_CFG,
        context_tokens: int = CONTEXT_TOKENS,
        context_scale: float = CONTEXT_SCALE,
    ):
        super().__init__()
        self.close_branch = close_branch
        self.context_branch = DeepLOBCausalFiveMinuteEncoder(**five_minute_cfg)
        d_close = close_branch.proj.out_features
        d_context = five_minute_cfg["d_model"]
        self.query_norm = nn.LayerNorm(d_close)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_close,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
            kdim=d_context,
            vdim=d_context,
        )
        self.context_dropout = nn.Dropout(0.1)
        self.context_tokens = int(context_tokens)
        self.context_scale = float(context_scale)
        self.freeze_close_branch()

    def freeze_close_branch(self) -> None:
        for parameter in self.close_branch.parameters():
            parameter.requires_grad_(False)
        self.close_branch.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen champion must never re-enable dropout during adapter training.
        self.close_branch.eval()
        return self

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("close_branch.") and parameter.requires_grad:
                yield parameter

    def parameter_counts(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        frozen = total - trainable
        return {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(frozen),
        }

    def forward_from_close_hidden(
        self,
        close_last: torch.Tensor,
        features_5m: torch.Tensor,
        has5: torch.Tensor,
    ) -> torch.Tensor:
        hidden_5m = self.context_branch(features_5m)
        key_value = hidden_5m[:, -self.context_tokens :]
        # The 1m hidden state conditions attention, but receives no gradient.
        query = self.query_norm(close_last.detach()).unsqueeze(1)
        context, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        context = self.context_dropout(context[:, 0])
        fused = close_last + (
            has5.to(close_last.dtype).unsqueeze(1)
            * self.context_scale
            * context
        )
        return self.close_branch.head(fused).squeeze(-1)

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


def _soft_rank(
    values: torch.Tensor, temperature: float = SOFT_RANK_TEMPERATURE
) -> torch.Tensor:
    centered = values - values.mean()
    scale = centered.pow(2).mean().sqrt().clamp_min(1e-6)
    normalized = centered / scale
    differences = normalized[:, None] - normalized[None, :]
    return 1.0 + torch.sigmoid(differences / temperature).sum(dim=1)


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
            prediction_rank = _soft_rank(prediction[mask])
            target_rank = torch.argsort(torch.argsort(target[mask])).to(
                prediction.dtype
            )
            daily.append(1.0 - _pearson(prediction_rank, target_rank))
        return torch.stack(daily).mean()


class MultiDateBatchSampler(Sampler):
    def __init__(
        self,
        groups: np.ndarray,
        dates_per_step: int,
        seed: int,
        shuffle: bool = True,
    ):
        groups = np.asarray(groups, dtype=np.int64)
        self.daily_indices = [
            np.flatnonzero(groups == group).tolist() for group in np.unique(groups)
        ]
        self.dates_per_step = dates_per_step
        self.rng = random.Random(seed)
        self.shuffle = shuffle

    def __iter__(self):
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


class RunningStats:
    def __init__(self, n_feat: int):
        self.count = 0
        self.total = np.zeros(n_feat, np.float64)
        self.square_total = np.zeros(n_feat, np.float64)

    def update(self, values: np.ndarray) -> None:
        values64 = values.astype(np.float64, copy=False)
        self.count += len(values64)
        self.total += values64.sum(axis=0)
        self.square_total += (values64**2).sum(axis=0)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise RuntimeError("5m statistics contain no valid observations")
        mean = self.total / self.count
        variance = self.square_total / self.count - mean**2
        std = np.sqrt(np.clip(variance, 0, None)) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)


def pool(start_date: str, end_date: str) -> List[str]:
    if dai is None:
        raise RuntimeError("dai is available only in BigQuant AIStudio")
    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    return frame["instrument"].tolist()


def point_in_time_membership(
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
) -> set:
    """Return exact daily competition-universe membership keys."""
    if dai is None:
        raise RuntimeError("dai is available only in BigQuant AIStudio")
    frame = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={
            "date": [start_date, end_date],
            "instrument": list(instruments),
        },
    ).df()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return set(zip(frame["date"], frame["instrument"]))


def _chunks(values: Sequence[str], size: int):
    for begin in range(0, len(values), size):
        yield values[begin : begin + size]


def _query_chunk(
    table: str,
    buffer_start: str,
    end_date: str,
    instruments: Sequence[str],
) -> pd.DataFrame:
    if dai is None:
        raise RuntimeError("dai is available only in BigQuant AIStudio")
    sql = (
        f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    frame = dai.query(
        sql,
        filters={
            "date": [buffer_start, end_date],
            "instrument": list(instruments),
        },
    ).df()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    for column in LOG1P_COLS:
        frame[column] = np.log1p(frame[column].clip(lower=0))
    return frame


def _one_minute_windows_from_one(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    need_label: bool,
) -> Tuple[List[np.ndarray], List[np.float32], List[Tuple[pd.Timestamp, str]]]:
    windows: List[np.ndarray] = []
    targets: List[np.float32] = []
    keys: List[Tuple[pd.Timestamp, str]] = []
    if len(frame) <= SEQ_LEN_1M:
        return windows, targets, keys
    instrument = frame["instrument"].iloc[0]
    features = frame[FEATURE_COLS].ffill().to_numpy(np.float32, copy=True)
    dates = frame["date"].dt.normalize().to_numpy()
    close_positions = np.flatnonzero(np.append(dates[1:] != dates[:-1], True))
    close = frame["close"].to_numpy(np.float64)[close_positions]
    for day_index, position in enumerate(close_positions):
        date = pd.Timestamp(dates[position])
        if position + 1 < SEQ_LEN_1M or date < start_date or date > end_date:
            continue
        target = None
        if day_index + 1 < len(close_positions) and close[day_index] > 0:
            value = close[day_index + 1] / close[day_index] - 1.0
            if np.isfinite(value):
                target = np.float32(value)
        if need_label and target is None:
            continue
        window = features[position - SEQ_LEN_1M + 1 : position + 1]
        if not np.isfinite(window).all():
            continue
        windows.append(window)
        targets.append(target if target is not None else np.float32(np.nan))
        keys.append((date, instrument))
    return windows, targets, keys


def _five_minute_map_from_one(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Dict[Tuple[pd.Timestamp, str], np.ndarray]:
    output: Dict[Tuple[pd.Timestamp, str], np.ndarray] = {}
    if frame.empty:
        return output
    instrument = frame["instrument"].iloc[0]
    features = frame[FEATURE_COLS].ffill().to_numpy(np.float32, copy=True)
    normalized_dates = frame["date"].dt.normalize()
    for day, positions in frame.groupby(normalized_dates, sort=False).indices.items():
        date = pd.Timestamp(day)
        if date < start_date or date > end_date:
            continue
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) < SEQ_LEN_5M:
            continue
        # Official daily 5m data should contain 48 bars. Taking the final 48
        # is defensive against duplicated or extra records without crossing days.
        window = features[positions[-SEQ_LEN_5M:]]
        if window.shape == (SEQ_LEN_5M, len(FEATURE_COLS)) and np.isfinite(
            window
        ).all():
            output[(date, instrument)] = window
    return output


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_metadata(
    datasources: dict,
    instruments: Sequence[str],
    start_date: str,
    end_date: str,
    max_instruments: int,
) -> dict:
    instrument_hash = hashlib.sha256(
        "\n".join(instruments).encode("utf-8")
    ).hexdigest()
    return {
        "cache_version": CACHE_VERSION,
        "bar1m_table": datasources["bar1m"],
        "bar5m_table": datasources["bar5m"],
        "train_start": start_date,
        "train_end": end_date,
        "seq_len_1m": SEQ_LEN_1M,
        "seq_len_5m": SEQ_LEN_5M,
        "feature_cols": FEATURE_COLS,
        "log1p_cols": LOG1P_COLS,
        "max_instruments": int(max_instruments),
        "instrument_hash": instrument_hash,
        "champion_sha256": _sha256_file(CHAMPION_MODEL_PATH),
        "missing_policy": "per-instrument ffill; missing 5m uses zero+has5 mask",
        "normalization": "champion 1m stats; separate train-only 5m stats",
        "universe_policy": "exact point-in-time bigalpha_2026_instruments membership",
    }


def _metadata_matches(path: str, expected: dict) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as stream:
            actual = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    keys = [
        "cache_version",
        "bar1m_table",
        "bar5m_table",
        "train_start",
        "train_end",
        "seq_len_1m",
        "seq_len_5m",
        "feature_cols",
        "log1p_cols",
        "max_instruments",
        "instrument_hash",
        "champion_sha256",
        "missing_policy",
        "normalization",
        "universe_policy",
    ]
    return all(actual.get(key) == expected.get(key) for key in keys)


def _save_array(cache_dir: str, name: str, values: np.ndarray) -> None:
    path = os.path.join(cache_dir, name)
    np.save(path, values, allow_pickle=False)


def load_stage_a_cache(
    cache_dir: str = CACHE_DIR,
    mmap_mode: Optional[str] = None,
) -> CachedDataset:
    with open(os.path.join(cache_dir, "metadata.json"), "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    stats = np.load(os.path.join(cache_dir, "stats5.npz"), allow_pickle=False)
    return CachedDataset(
        x1=np.load(os.path.join(cache_dir, "x1.npy"), mmap_mode=mmap_mode),
        x5=np.load(os.path.join(cache_dir, "x5.npy"), mmap_mode=mmap_mode),
        targets=np.load(os.path.join(cache_dir, "targets.npy"), mmap_mode=mmap_mode),
        groups=np.load(os.path.join(cache_dir, "groups.npy"), mmap_mode=mmap_mode),
        has5=np.load(os.path.join(cache_dir, "has5.npy"), mmap_mode=mmap_mode),
        dates=np.load(os.path.join(cache_dir, "dates.npy"), mmap_mode=mmap_mode),
        instruments=np.load(
            os.path.join(cache_dir, "instruments.npy"), mmap_mode=mmap_mode
        ),
        mean5=stats["mean"].astype(np.float32),
        std5=stats["std"].astype(np.float32),
        metadata=metadata,
    )


def prepare_stage_a_cache(
    datasources: dict,
    cache_dir: str = CACHE_DIR,
    force: bool = False,
    start_date: str = TRAIN_START,
    end_date: str = TRAIN_END,
    max_instruments: int = MAX_TRAIN_INSTRUMENTS,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    mmap_mode: Optional[str] = None,
) -> CachedDataset:
    """Query once and cache aligned normalized tensors.

    The 1m normalization comes directly from the submitted champion checkpoint.
    The 5m normalization is fitted only on matched training windows.
    """
    if not os.path.exists(CHAMPION_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {CHAMPION_MODEL_PATH}; upload champion_weights.json first."
        )
    champion_payload, _ = load_champion_ensemble(
        CHAMPION_MODEL_PATH, map_location="cpu"
    )
    mean1 = np.asarray(champion_payload["mean"], np.float32)
    std1 = np.asarray(champion_payload["std"], np.float32)

    instruments = sorted(pool(start_date, end_date))[:max_instruments]
    expected_metadata = _cache_metadata(
        datasources, instruments, start_date, end_date, max_instruments
    )
    metadata_path = os.path.join(cache_dir, "metadata.json")
    required = [
        "x1.npy",
        "x5.npy",
        "targets.npy",
        "groups.npy",
        "has5.npy",
        "dates.npy",
        "instruments.npy",
        "stats5.npz",
    ]
    if (
        not force
        and _metadata_matches(metadata_path, expected_metadata)
        and all(os.path.exists(os.path.join(cache_dir, item)) for item in required)
    ):
        logger.info("loading valid dual-frequency cache", cache_dir=cache_dir)
        return load_stage_a_cache(cache_dir, mmap_mode=mmap_mode)

    os.makedirs(cache_dir, exist_ok=True)
    # Rebuilding tensors invalidates any previously precomputed champion hidden.
    for filename in os.listdir(cache_dir):
        if filename.startswith("close_hidden_seed") and filename.endswith(".npy"):
            os.remove(os.path.join(cache_dir, filename))
    buffer_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=20)
    ).strftime("%Y-%m-%d")
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    membership = point_in_time_membership(
        start_date, end_date, instruments
    )

    x1_parts: List[np.ndarray] = []
    x5_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    has5_parts: List[np.ndarray] = []
    all_dates: List[np.datetime64] = []
    all_instruments: List[str] = []
    stats5 = RunningStats(len(FEATURE_COLS))

    for chunk_number, chunk in enumerate(
        _chunks(instruments, query_chunk_size), 1
    ):
        started = time.time()
        frame1 = _query_chunk(
            datasources["bar1m"], buffer_start, end_date, chunk
        )
        local_x1: List[np.ndarray] = []
        local_targets: List[np.float32] = []
        local_keys: List[Tuple[pd.Timestamp, str]] = []
        for _, instrument_frame in frame1.groupby("instrument", sort=False):
            windows, targets, keys = _one_minute_windows_from_one(
                instrument_frame, start_ts, end_ts, need_label=True
            )
            for window, target, key in zip(windows, targets, keys):
                if (pd.Timestamp(key[0]).normalize(), key[1]) in membership:
                    local_x1.append(window)
                    local_targets.append(target)
                    local_keys.append(key)
        del frame1
        gc.collect()

        frame5 = _query_chunk(
            datasources["bar5m"], start_date, end_date, chunk
        )
        five_minute_map: Dict[Tuple[pd.Timestamp, str], np.ndarray] = {}
        for _, instrument_frame in frame5.groupby("instrument", sort=False):
            five_minute_map.update(
                _five_minute_map_from_one(instrument_frame, start_ts, end_ts)
            )
        del frame5
        gc.collect()

        if not local_x1:
            logger.info("empty training chunk", chunk=chunk_number)
            continue

        x1_chunk = np.stack(local_x1).astype(np.float32, copy=False)
        x1_chunk = ((x1_chunk - mean1) / std1).astype(np.float32)
        x5_chunk = np.zeros(
            (len(local_keys), SEQ_LEN_5M, len(FEATURE_COLS)), np.float32
        )
        has5_chunk = np.zeros(len(local_keys), np.float32)
        for index, key in enumerate(local_keys):
            window5 = five_minute_map.get(key)
            if window5 is not None:
                x5_chunk[index] = window5
                has5_chunk[index] = 1.0
                stats5.update(window5)

        x1_parts.append(x1_chunk)
        x5_parts.append(x5_chunk)
        target_parts.append(np.asarray(local_targets, np.float32))
        has5_parts.append(has5_chunk)
        all_dates.extend(np.datetime64(key[0].date(), "D") for key in local_keys)
        all_instruments.extend(key[1] for key in local_keys)

        del local_x1, local_targets, local_keys, five_minute_map
        gc.collect()
        logger.info(
            "dual-frequency cache chunk complete",
            chunk=chunk_number,
            samples=sum(len(part) for part in target_parts),
            five_minute_coverage=float(has5_chunk.mean()),
            elapsed_seconds=round(time.time() - started, 2),
        )

    if not x1_parts:
        raise RuntimeError("training dataset contains no samples")

    x1 = np.concatenate(x1_parts, axis=0)
    x5 = np.concatenate(x5_parts, axis=0)
    targets = np.concatenate(target_parts).astype(np.float32, copy=False)
    has5 = np.concatenate(has5_parts).astype(np.float32, copy=False)
    mean5, std5 = stats5.finalize()
    present = has5 > 0
    x5[present] = ((x5[present] - mean5) / std5).astype(np.float32)
    x5[~present] = 0.0

    lower, upper = np.percentile(targets, [1, 99])
    np.clip(targets, lower, upper, out=targets)
    dates = np.asarray(all_dates, dtype="datetime64[D]")
    groups, _ = pd.factorize(pd.to_datetime(dates), sort=True)
    groups = groups.astype(np.int64)
    max_length = max(len(item) for item in all_instruments)
    instrument_array = np.asarray(all_instruments, dtype=f"U{max_length}")

    _save_array(cache_dir, "x1.npy", x1)
    _save_array(cache_dir, "x5.npy", x5)
    _save_array(cache_dir, "targets.npy", targets)
    _save_array(cache_dir, "groups.npy", groups)
    _save_array(cache_dir, "has5.npy", has5)
    _save_array(cache_dir, "dates.npy", dates)
    _save_array(cache_dir, "instruments.npy", instrument_array)
    np.savez(
        os.path.join(cache_dir, "stats5.npz"), mean=mean5, std=std5
    )

    expected_metadata.update(
        {
            "samples": int(len(targets)),
            "days": int(len(np.unique(groups))),
            "five_minute_coverage": float(has5.mean()),
            "x1_shape": list(x1.shape),
            "x5_shape": list(x5.shape),
            "target_winsor_lower": float(lower),
            "target_winsor_upper": float(upper),
        }
    )
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(expected_metadata, stream, ensure_ascii=False, indent=2)

    logger.info(
        "dual-frequency cache saved",
        cache_dir=cache_dir,
        samples=len(targets),
        days=len(np.unique(groups)),
        five_minute_coverage=float(has5.mean()),
    )
    return load_stage_a_cache(cache_dir, mmap_mode=mmap_mode)


def _serialize_state(state_dict: dict) -> dict:
    tensors = {}
    for name, value in state_dict.items():
        tensor = value.detach().cpu()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    return tensors


def _deserialize_state(tensors: dict, map_location) -> dict:
    state = {}
    for name, meta in tensors.items():
        tensor = torch.tensor(
            meta["data"], dtype=getattr(torch, meta["dtype"])
        ).reshape(meta["shape"])
        state[name] = tensor.to(map_location)
    return state


def load_champion_ensemble(
    model_path: str = CHAMPION_MODEL_PATH,
    map_location="cpu",
):
    with open(model_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    models = []
    for item in payload["states"]:
        model = CausalMultiScaleTransformer(**payload["model_cfg"]).to(
            map_location
        )
        model.load_state_dict(
            _deserialize_state(item["state_dict"], map_location)
        )
        model.eval()
        models.append(model)
    return payload, models


def _hidden_cache_path(cache_dir: str, seed: int) -> str:
    return os.path.join(cache_dir, f"close_hidden_seed{seed}.npy")


def precompute_close_hidden(
    champion_models: Sequence[CausalMultiScaleTransformer],
    champion_seeds: Sequence[int],
    dataset: CachedDataset,
    device: torch.device,
    cache_dir: str = CACHE_DIR,
    force: bool = False,
    batch_size: int = PRECOMPUTE_BATCH,
) -> Dict[int, np.ndarray]:
    """Cache frozen 1m close hidden states once per champion seed."""
    output: Dict[int, np.ndarray] = {}
    x1_tensor = torch.from_numpy(np.asarray(dataset.x1))
    for seed, model in zip(champion_seeds, champion_models):
        path = _hidden_cache_path(cache_dir, int(seed))
        if not force and os.path.exists(path):
            values = np.load(path, mmap_mode=None)
            if values.shape == (len(dataset.targets), model.proj.out_features):
                output[int(seed)] = values
                logger.info(
                    "loaded cached champion hidden", seed=int(seed), path=path
                )
                continue

        model = model.to(device)
        model.eval()
        parts = []
        started = time.time()
        with torch.inference_mode():
            for begin in range(0, len(x1_tensor), batch_size):
                batch = x1_tensor[begin : begin + batch_size].to(
                    device, non_blocking=True
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=USE_AMP and device.type == "cuda",
                ):
                    hidden = model.encode(batch)[:, -1]
                parts.append(hidden.float().cpu().numpy())
        values = np.concatenate(parts).astype(np.float32, copy=False)
        np.save(path, values, allow_pickle=False)
        output[int(seed)] = values
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "champion hidden cached",
            seed=int(seed),
            shape=list(values.shape),
            elapsed_seconds=round(time.time() - started, 2),
        )
    return output


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).cpu())


def train_adapter_one(
    close_branch: CausalMultiScaleTransformer,
    close_hidden: np.ndarray,
    dataset: CachedDataset,
    seed: int,
    device: torch.device,
    epochs: int = ADAPTER_EPOCHS,
    lr: float = LR_ADAPTER,
) -> StageADualFrequencyModel:
    set_seed(seed)
    model = StageADualFrequencyModel(close_branch).to(device)
    adapter_parameters = list(model.adapter_parameters())
    if not adapter_parameters:
        raise RuntimeError("adapter contains no trainable parameters")
    counts = model.parameter_counts()
    logger.info("stage A parameter audit", **counts)
    if counts["trainable"] < MIN_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "A5 trainable-parameter floor failed: "
            f"{counts['trainable']} < {MIN_TRAINABLE_PARAMETERS}"
        )

    sampler = MultiDateBatchSampler(
        dataset.groups, DATES_PER_STEP, seed, shuffle=True
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(close_hidden)),
            torch.from_numpy(np.asarray(dataset.x5)),
            torch.from_numpy(np.asarray(dataset.targets)),
            torch.from_numpy(np.asarray(dataset.groups)),
            torch.from_numpy(np.asarray(dataset.has5)),
        ),
        batch_sampler=sampler,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    optimizer = torch.optim.Adam(adapter_parameters, lr=lr)
    objective = SoftSpearmanLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP and device.type == "cuda"
    )

    for epoch in range(1, epochs + 1):
        started = time.time()
        losses = []
        grad_norms = []
        model.train()
        for (
            batch_hidden,
            batch_x5,
            batch_targets,
            batch_groups,
            batch_has5,
        ) in loader:
            batch_hidden = batch_hidden.to(device, non_blocking=True)
            batch_x5 = batch_x5.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_groups = batch_groups.to(device, non_blocking=True)
            batch_has5 = batch_has5.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=USE_AMP and device.type == "cuda",
            ):
                prediction = model.forward_from_close_hidden(
                    batch_hidden, batch_x5, batch_has5
                )
            loss = objective(
                prediction.float(), batch_targets.float(), batch_groups
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norms.append(_gradient_norm(adapter_parameters))
            nn.utils.clip_grad_norm_(adapter_parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        logger.info(
            "stage A adapter epoch complete",
            seed=seed,
            epoch=epoch,
            loss=float(np.mean(losses)),
            adapter_grad_norm=float(np.mean(grad_norms)),
            elapsed_seconds=round(time.time() - started, 2),
        )
    model.eval()
    return model


def save_stage_a_ensemble(
    models: Sequence[StageADualFrequencyModel],
    seeds: Sequence[int],
    champion_payload: dict,
    mean5: np.ndarray,
    std5: np.ndarray,
    model_path: str = MODEL_PATH,
    epochs: int = ADAPTER_EPOCHS,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    max_instruments: int = MAX_TRAIN_INSTRUMENTS,
) -> str:
    payload = {
        "name": "A5Lite_deeplob_3L_U600_frozen_champion_tail12_adapter",
        "model_kind": "stageA_A5Lite_deeplob_3L_U600_frozen_adapter",
        "champion_model_cfg": champion_payload["model_cfg"],
        "five_minute_cfg": FIVE_MINUTE_CFG,
        "feature_cols_1m": FEATURE_COLS,
        "feature_cols_5m": FEATURE_COLS,
        "seq_len_1m": SEQ_LEN_1M,
        "seq_len_5m": SEQ_LEN_5M,
        "context_tokens": CONTEXT_TOKENS,
        "context_scale": CONTEXT_SCALE,
        "mean_1m": champion_payload["mean"],
        "std_1m": champion_payload["std"],
        "mean_5m": np.asarray(mean5, np.float32).tolist(),
        "std_5m": np.asarray(std5, np.float32).tolist(),
        "seeds": [int(seed) for seed in seeds],
        "adapter_epochs": int(epochs),
        "dates_per_step": DATES_PER_STEP,
        "soft_rank_temperature": SOFT_RANK_TEMPERATURE,
        "train_start": train_start,
        "train_end": train_end,
        "max_train_instruments": int(max_instruments),
        "base_champion_name": champion_payload.get("name"),
        "base_champion_sha256": _sha256_file(CHAMPION_MODEL_PATH),
        "training_mode": (
            "champion branch/head frozen; precomputed close hidden; "
            "raw-field DeepLOB 5m encoder and cross-attention only"
        ),
        "input_compliance": (
            "official raw fields only; learned depth/time interactions; "
            "no derived inputs or fixed cross-field operators"
        ),
        "parameter_counts": (
            models[0].parameter_counts() if models else {}
        ),
        "point_in_time_universe": True,
        "amp_enabled": bool(USE_AMP),
        "blend": "equal mean of within-date percentile ranks",
        "states": [
            {
                "seed": int(seed),
                "state_dict": _serialize_state(model.state_dict()),
            }
            for seed, model in zip(seeds, models)
        ],
    }
    with open(model_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    logger.info("stage A ensemble saved", path=model_path, models=len(models))
    return model_path


def load_stage_a_ensemble(
    model_path: str = MODEL_PATH,
    map_location="cpu",
):
    with open(model_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    models = []
    for item in payload["states"]:
        close_branch = CausalMultiScaleTransformer(
            **payload["champion_model_cfg"]
        )
        model = StageADualFrequencyModel(
            close_branch=close_branch,
            five_minute_cfg=payload["five_minute_cfg"],
            context_tokens=payload["context_tokens"],
            context_scale=payload["context_scale"],
        ).to(map_location)
        model.load_state_dict(
            _deserialize_state(item["state_dict"], map_location)
        )
        model.freeze_close_branch()
        model.eval()
        models.append(model)
    return payload, models


def train_and_save_stage_a(
    datasources: dict,
    model_path: str = MODEL_PATH,
    cache_dir: str = CACHE_DIR,
    force_cache: bool = False,
    force_hidden: bool = False,
    seeds: Optional[Sequence[int]] = None,
    epochs: int = ADAPTER_EPOCHS,
    max_instruments: int = MAX_TRAIN_INSTRUMENTS,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    start_date: str = TRAIN_START,
    end_date: str = TRAIN_END,
) -> str:
    """Train Stage A on top of the submitted champion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    champion_payload, champion_models = load_champion_ensemble(
        CHAMPION_MODEL_PATH, map_location="cpu"
    )
    champion_seeds = [int(item["seed"]) for item in champion_payload["states"]]
    selected_seeds = list(champion_seeds if seeds is None else seeds)
    missing = sorted(set(selected_seeds) - set(champion_seeds))
    if missing:
        raise ValueError(f"requested seeds are absent from champion: {missing}")

    dataset = prepare_stage_a_cache(
        datasources,
        cache_dir=cache_dir,
        force=force_cache,
        start_date=start_date,
        end_date=end_date,
        max_instruments=max_instruments,
        query_chunk_size=query_chunk_size,
        mmap_mode=None,
    )
    model_by_seed = {
        int(seed): model for seed, model in zip(champion_seeds, champion_models)
    }
    selected_models = [model_by_seed[int(seed)] for seed in selected_seeds]
    hidden_by_seed = precompute_close_hidden(
        selected_models,
        selected_seeds,
        dataset,
        device,
        cache_dir=cache_dir,
        force=force_hidden,
    )

    trained_models = []
    for seed in selected_seeds:
        trained = train_adapter_one(
            close_branch=model_by_seed[int(seed)],
            close_hidden=hidden_by_seed[int(seed)],
            dataset=dataset,
            seed=int(seed),
            device=device,
            epochs=epochs,
        )
        trained_models.append(trained.to("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return save_stage_a_ensemble(
        trained_models,
        selected_seeds,
        champion_payload,
        dataset.mean5,
        dataset.std5,
        model_path=model_path,
        epochs=epochs,
        train_start=start_date,
        train_end=end_date,
        max_instruments=max_instruments,
    )


# The competition/private-training entrypoint name remains unchanged.
def train_and_save(datasources: dict, model_path: str = MODEL_PATH) -> str:
    return train_and_save_stage_a(datasources, model_path=model_path)


def _build_inference_chunk(
    table1m: str,
    table5m: str,
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
    mean1: np.ndarray,
    std1: np.ndarray,
    mean5: np.ndarray,
    std5: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[pd.Timestamp, str]],
    np.ndarray,
]:
    buffer_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=20)
    ).strftime("%Y-%m-%d")
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)

    frame1 = _query_chunk(table1m, buffer_start, end_date, instruments)
    windows1: List[np.ndarray] = []
    keys: List[Tuple[pd.Timestamp, str]] = []
    forward_returns: List[np.float32] = []
    for _, instrument_frame in frame1.groupby("instrument", sort=False):
        local_windows, local_returns, local_keys = _one_minute_windows_from_one(
            instrument_frame, start_ts, end_ts, need_label=False
        )
        windows1.extend(local_windows)
        keys.extend(local_keys)
        forward_returns.extend(local_returns)
    del frame1
    gc.collect()
    if not windows1:
        return (
            np.empty((0, SEQ_LEN_1M, len(FEATURE_COLS)), np.float32),
            np.empty((0, SEQ_LEN_5M, len(FEATURE_COLS)), np.float32),
            np.empty((0,), np.float32),
            [],
            np.empty((0,), np.float32),
        )

    frame5 = _query_chunk(table5m, start_date, end_date, instruments)
    five_minute_map: Dict[Tuple[pd.Timestamp, str], np.ndarray] = {}
    for _, instrument_frame in frame5.groupby("instrument", sort=False):
        five_minute_map.update(
            _five_minute_map_from_one(instrument_frame, start_ts, end_ts)
        )
    del frame5
    gc.collect()

    x1 = np.stack(windows1).astype(np.float32, copy=False)
    x1 = ((x1 - mean1) / std1).astype(np.float32)
    x5 = np.zeros((len(keys), SEQ_LEN_5M, len(FEATURE_COLS)), np.float32)
    has5 = np.zeros(len(keys), np.float32)
    for index, key in enumerate(keys):
        window5 = five_minute_map.get(key)
        if window5 is not None:
            x5[index] = ((window5 - mean5) / std5).astype(np.float32)
            has5[index] = 1.0
    return (
        x1,
        x5,
        has5,
        keys,
        np.asarray(forward_returns, np.float32),
    )


def _rank_ensemble(raw_output: pd.DataFrame, model_count: int) -> pd.DataFrame:
    rank_columns = []
    for model_index in range(model_count):
        column = f"rank_{model_index}"
        raw_output[column] = raw_output.groupby("date")[
            f"model_{model_index}"
        ].rank(method="average", pct=True)
        rank_columns.append(column)
    raw_output["score"] = raw_output[rank_columns].mean(axis=1)
    return raw_output[["date", "instrument", "score"]]


def predict_champion_scores(
    models: Sequence[CausalMultiScaleTransformer],
    table1m: str,
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
    stats1: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    batch_size: int = INFERENCE_BATCH,
) -> pd.DataFrame:
    mean1, std1 = stats1
    buffer_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=20)
    ).strftime("%Y-%m-%d")
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    outputs = []
    for chunk_number, chunk in enumerate(
        _chunks(instruments, query_chunk_size), 1
    ):
        frame = _query_chunk(table1m, buffer_start, end_date, chunk)
        windows, keys = [], []
        for _, instrument_frame in frame.groupby("instrument", sort=False):
            local_windows, _, local_keys = _one_minute_windows_from_one(
                instrument_frame, start_ts, end_ts, need_label=False
            )
            windows.extend(local_windows)
            keys.extend(local_keys)
        del frame
        gc.collect()
        if not windows:
            continue
        features = np.stack(windows).astype(np.float32)
        features = ((features - mean1) / std1).astype(np.float32)
        tensor = torch.from_numpy(features)
        chunk_output = pd.DataFrame(keys, columns=["date", "instrument"])
        for model_index, model in enumerate(models):
            model = model.to(device).eval()
            values = []
            with torch.inference_mode():
                for begin in range(0, len(tensor), batch_size):
                    values.append(
                        model(tensor[begin : begin + batch_size].to(device))
                        .cpu()
                        .numpy()
                    )
            chunk_output[f"model_{model_index}"] = np.concatenate(values)
        outputs.append(chunk_output)
        logger.info("champion inference chunk complete", chunk=chunk_number)
    if not outputs:
        raise RuntimeError("champion inference dataset contains no samples")
    return _rank_ensemble(pd.concat(outputs, ignore_index=True), len(models))


def predict_stage_a_scores(
    models: Sequence[StageADualFrequencyModel],
    table1m: str,
    table5m: str,
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
    stats1: Tuple[np.ndarray, np.ndarray],
    stats5: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    batch_size: int = INFERENCE_BATCH,
) -> pd.DataFrame:
    mean1, std1 = stats1
    mean5, std5 = stats5
    outputs = []
    for chunk_number, chunk in enumerate(
        _chunks(instruments, query_chunk_size), 1
    ):
        x1, x5, has5, keys, _ = _build_inference_chunk(
            table1m,
            table5m,
            start_date,
            end_date,
            chunk,
            mean1,
            std1,
            mean5,
            std5,
        )
        if not keys:
            continue
        tensor1 = torch.from_numpy(x1)
        tensor5 = torch.from_numpy(x5)
        tensor_has5 = torch.from_numpy(has5)
        chunk_output = pd.DataFrame(keys, columns=["date", "instrument"])
        for model_index, model in enumerate(models):
            model = model.to(device).eval()
            values = []
            with torch.inference_mode():
                for begin in range(0, len(tensor1), batch_size):
                    values.append(
                        model(
                            tensor1[begin : begin + batch_size].to(
                                device, non_blocking=True
                            ),
                            tensor5[begin : begin + batch_size].to(
                                device, non_blocking=True
                            ),
                            tensor_has5[begin : begin + batch_size].to(
                                device, non_blocking=True
                            ),
                        )
                        .cpu()
                        .numpy()
                    )
            chunk_output[f"model_{model_index}"] = np.concatenate(values)
        outputs.append(chunk_output)
        logger.info(
            "stage A inference chunk complete",
            chunk=chunk_number,
            five_minute_coverage=float(has5.mean()),
        )
    if not outputs:
        raise RuntimeError("stage A inference dataset contains no samples")
    return _rank_ensemble(pd.concat(outputs, ignore_index=True), len(models))


def predict_champion_and_stage_a_scores(
    models: Sequence[StageADualFrequencyModel],
    table1m: str,
    table5m: str,
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
    stats1: Tuple[np.ndarray, np.ndarray],
    stats5: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    batch_size: int = INFERENCE_BATCH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare champion and Stage A with one DAI pass and one 1m encoding.

    The frozen close branch embedded in every Stage-A state is the exact
    champion seed. Reusing its hidden state removes duplicate DAI queries and
    duplicate one-minute Transformer work during model comparison.
    """
    mean1, std1 = stats1
    mean5, std5 = stats5
    champion_outputs = []
    stage_outputs = []
    label_outputs = []
    device_models = [model.to(device).eval() for model in models]

    for chunk_number, chunk in enumerate(
        _chunks(instruments, query_chunk_size), 1
    ):
        x1, x5, has5, keys, forward_returns = _build_inference_chunk(
            table1m,
            table5m,
            start_date,
            end_date,
            chunk,
            mean1,
            std1,
            mean5,
            std5,
        )
        if not keys:
            continue
        tensor1 = torch.from_numpy(x1)
        tensor5 = torch.from_numpy(x5)
        tensor_has5 = torch.from_numpy(has5)
        champion_chunk = pd.DataFrame(keys, columns=["date", "instrument"])
        stage_chunk = pd.DataFrame(keys, columns=["date", "instrument"])

        for model_index, model in enumerate(device_models):
            champion_values = []
            stage_values = []
            with torch.inference_mode():
                for begin in range(0, len(tensor1), batch_size):
                    batch1 = tensor1[begin : begin + batch_size].to(
                        device, non_blocking=True
                    )
                    batch5 = tensor5[begin : begin + batch_size].to(
                        device, non_blocking=True
                    )
                    batch_has5 = tensor_has5[begin : begin + batch_size].to(
                        device, non_blocking=True
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=USE_AMP and device.type == "cuda",
                    ):
                        close_last = model.close_branch.encode(batch1)[:, -1]
                        champion_value = model.close_branch.head(
                            close_last
                        ).squeeze(-1)
                        stage_value = model.forward_from_close_hidden(
                            close_last, batch5, batch_has5
                        )
                    champion_values.append(
                        champion_value.float().cpu().numpy()
                    )
                    stage_values.append(stage_value.float().cpu().numpy())

            champion_chunk[f"model_{model_index}"] = np.concatenate(
                champion_values
            )
            stage_chunk[f"model_{model_index}"] = np.concatenate(stage_values)

        champion_outputs.append(champion_chunk)
        stage_outputs.append(stage_chunk)
        label_chunk = pd.DataFrame(keys, columns=["date", "instrument"])
        label_chunk["forward_return"] = forward_returns
        label_outputs.append(label_chunk)
        logger.info(
            "joint champion/stage A inference chunk complete",
            chunk=chunk_number,
            five_minute_coverage=float(has5.mean()),
        )

    if not champion_outputs:
        raise RuntimeError("joint inference dataset contains no samples")
    champion = _rank_ensemble(
        pd.concat(champion_outputs, ignore_index=True), len(models)
    )
    stage_a = _rank_ensemble(
        pd.concat(stage_outputs, ignore_index=True), len(models)
    )
    labels = pd.concat(label_outputs, ignore_index=True)
    return champion, stage_a, labels


def prediction_diagnostics(
    champion: pd.DataFrame,
    stage_a: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    merged = pd.merge(
        champion,
        stage_a,
        on=["date", "instrument"],
        how="inner",
        suffixes=("_champion", "_stage_a"),
    )
    daily = (
        merged.groupby("date")
        .apply(
            lambda frame: pd.Series(
                {
                    "rows": len(frame),
                    "score_corr": frame["score_champion"].corr(
                        frame["score_stage_a"], method="spearman"
                    ),
                    "mean_abs_change": (
                        frame["score_stage_a"] - frame["score_champion"]
                    ).abs().mean(),
                }
            )
        )
        .reset_index()
    )
    summary = {
        "matched_rows": int(len(merged)),
        "days": int(merged["date"].nunique()),
        "mean_daily_score_corr": float(daily["score_corr"].mean()),
        "mean_daily_abs_change": float(daily["mean_abs_change"].mean()),
    }
    return daily, summary


if __name__ == "__main__":
    train_and_save(
        {
            "bar1m": "bigalpha_2026_stock_bar1m",
            "bar5m": "bigalpha_2026_stock_bar5m",
        }
    )
