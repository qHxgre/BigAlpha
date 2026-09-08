# -*- coding: utf-8 -*-
"""BigAlpha 2026 dual-scale TimesFM-style model.

This module follows the baseline package contract:
  - local parquet data first, BigQuant DAI fallback on platform;
  - `train_and_save(datasources)` trains from random initialization;
  - `main(datasources, start_date, end_date)` returns date/instrument/score.

This structure has its own experiment folder. It uses raw bar5m and raw bar1m
inputs without handcrafted features: bar5m is the medium-horizon trunk and
bar1m is a recent intraday microstructure branch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import copy
import base64
import ctypes
import gc
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Sampler, TensorDataset

try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover - platform usually has structlog
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# ============ Configuration ============
BAR5M_KEY = "bar5m"
BAR1M_KEY = "bar1m"
TRAIN_START = os.environ.get("BQ_TRAIN_START", "2019-01-01")
TRAIN_END = os.environ.get("BQ_TRAIN_END", "2023-12-31 23:59:59")
VAL_START = os.environ.get("BQ_VAL_START", "2024-01-01")
VAL_END = os.environ.get("BQ_VAL_END", "2024-12-31 23:59:59")
VAL_RAW_RET = os.environ.get("BQ_VAL_RAW_RET", "1") == "1"
SAVE_LAST = os.environ.get("BQ_SAVE_LAST", "0") == "1"
SAVE_EPOCH_STATES = os.environ.get("BQ_SAVE_EPOCH_STATES", "0") == "1"
LABEL_MODE = os.environ.get("BQ_LABEL_MODE", "cs_gaussian_rank")
LOSS_MODE = os.environ.get("BQ_LOSS_MODE", "smooth_l1")
CHANNEL_MASK_PROB = float(os.environ.get("BQ_CHANNEL_MASK_PROB", "0.0"))
if not 0.0 <= CHANNEL_MASK_PROB < 1.0:
    raise ValueError("BQ_CHANNEL_MASK_PROB must be in [0, 1)")
TARGET_CURRICULUM = os.environ.get("BQ_TARGET_CURRICULUM", "none")
CURRICULUM_SWITCH_EPOCH = int(os.environ.get("BQ_CURRICULUM_SWITCH_EPOCH", "0"))
CURRICULUM_RAW_LR_SCALE = float(os.environ.get("BQ_CURRICULUM_RAW_LR_SCALE", "0.25"))
CURRICULUM_BLEND_EPOCHS = int(os.environ.get("BQ_CURRICULUM_BLEND_EPOCHS", "0"))
CURRICULUM_SELECT_POST_ONLY = os.environ.get("BQ_CURRICULUM_SELECT_POST_ONLY", "0") == "1"
CURRICULUM_FREEZE_TRUNKS = os.environ.get("BQ_CURRICULUM_FREEZE_TRUNKS", "0") == "1"
SELECT_METRIC = "val_loss_provisional"
SELECT_MIN_IC = float(os.environ.get("BQ_SELECT_MIN_IC", "0.08"))
SELECT_WARMUP = int(os.environ.get("BQ_SELECT_WARMUP", "4"))
EARLY_STOP_PATIENCE = int(os.environ.get("BQ_EARLY_STOP_PATIENCE", "0"))
VAL_LOSS_MIN_DELTA = float(os.environ.get("BQ_VAL_LOSS_MIN_DELTA", "1e-6"))
METRIC_MIN_DELTA = float(os.environ.get("BQ_METRIC_MIN_DELTA", "1e-5"))

BAR5M_LOOKBACK_DAYS = int(os.environ.get("BQ_BAR5M_LOOKBACK_DAYS", "5"))
BAR1M_LOOKBACK_DAYS = int(os.environ.get("BQ_BAR1M_LOOKBACK_DAYS", "1"))
BAR5M_SEQ_LEN = BAR5M_LOOKBACK_DAYS * 48
BAR1M_SEQ_LEN = BAR1M_LOOKBACK_DAYS * 240
BAR5M_FROM_1M_STRIDE = int(os.environ.get("BQ_BAR5M_FROM_1M_STRIDE", "5"))
BAR5M_FROM_1M_SOURCE_LEN = BAR5M_SEQ_LEN * BAR5M_FROM_1M_STRIDE
BAR5M_PATCH_LEN = int(os.environ.get("BQ_BAR5M_PATCH_LEN", "16"))
BAR1M_PATCH_LEN = int(os.environ.get("BQ_BAR1M_PATCH_LEN", "30"))
BAR1M_FINE_PATCH_LEN = int(os.environ.get("BQ_BAR1M_FINE_PATCH_LEN", "15"))
if BAR5M_SEQ_LEN % BAR5M_PATCH_LEN != 0:
    raise ValueError("BAR5M_SEQ_LEN must be divisible by BAR5M_PATCH_LEN")
if BAR1M_SEQ_LEN % BAR1M_PATCH_LEN != 0:
    raise ValueError("BAR1M_SEQ_LEN must be divisible by BAR1M_PATCH_LEN")
if BAR1M_SEQ_LEN % BAR1M_FINE_PATCH_LEN != 0:
    raise ValueError("BAR1M_SEQ_LEN must be divisible by BAR1M_FINE_PATCH_LEN")

EPOCHS = int(os.environ.get("BQ_EPOCHS", "40"))
# The notebook imports BATCH for public inference. Keep it inside the runtime
# envelope of the platform-confirmed Anchor; training has its own batch knob.
BATCH = int(os.environ.get("BQ_BATCH", "192"))
SUBMISSION_TORCH_THREADS = int(os.environ.get("BQ_SUBMISSION_TORCH_THREADS", "4"))
SUBMISSION_DATALOADER_DAYS = int(os.environ.get("BQ_SUBMISSION_DATALOADER_DAYS", "10"))
if SUBMISSION_DATALOADER_DAYS < 1:
    raise ValueError("BQ_SUBMISSION_DATALOADER_DAYS must be positive")
TRAIN_BATCH = int(os.environ.get("BQ_TRAIN_BATCH", "1024"))
LR = float(os.environ.get("BQ_LR", "8e-5"))
LR_MIN = float(os.environ.get("BQ_LR_MIN", "5e-6"))
SCHED_MODE = os.environ.get("BQ_SCHED_MODE", "epoch_cosine")
WARMUP_FRAC = float(os.environ.get("BQ_WARMUP_FRAC", "0.05"))
WEIGHT_DECAY = float(os.environ.get("BQ_WEIGHT_DECAY", "1e-4"))
USE_EMA = os.environ.get("BQ_USE_EMA", "0") == "1"
EMA_DECAY = float(os.environ.get("BQ_EMA_DECAY", "0.999"))
AUX_HORIZON = int(os.environ.get("BQ_AUX_HORIZON", "3"))
AUX_WEIGHT = float(os.environ.get("BQ_AUX_WEIGHT", "0.1"))
AUX_ANNEAL_LAST_FRAC = float(os.environ.get("BQ_AUX_ANNEAL_LAST_FRAC", "0.0"))
SEED = int(os.environ.get("BQ_SEED", "20260820"))
MAX_TRAIN_INSTRUMENTS = int(os.environ.get("BQ_MAX_TRAIN_INSTRUMENTS", "999999"))
BUILD_WORKERS = int(os.environ.get("BQ_BUILD_WORKERS", "0"))
READ_WORKERS = int(os.environ.get("BQ_READ_WORKERS", "0"))
SHARD_CACHE_DIR = os.environ.get("BQ_SHARD_CACHE_DIR", "cache_shards")
SHARD_WORKERS = int(os.environ.get("BQ_SHARD_WORKERS", "16"))
SHARD_INSTRUMENTS_PER_TASK = int(os.environ.get("BQ_SHARD_INSTRUMENTS_PER_TASK", "25"))
USE_SHARDED_TRAIN_BUILDER = os.environ.get("BQ_USE_SHARDED_TRAIN_BUILDER", "0") == "1"
LISTWISE_WEIGHT = float(os.environ.get("BQ_LISTWISE_WEIGHT", "0"))
LISTWISE_TEMPERATURE = float(os.environ.get("BQ_LISTWISE_TEMPERATURE", "1"))
RANK_LOSS_KIND = os.environ.get("BQ_RANK_LOSS_KIND", "listmle")


class DayBatchSampler(Sampler[list[int]]):
    """Build fixed-size, single-day batches without changing the sample set."""

    def __init__(self, dates: np.ndarray, batch_size: int, seed: int):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for index, day in enumerate(dates.astype("datetime64[D]").astype(np.int64)):
            groups.setdefault(int(day), []).append(index)
        self.groups = list(groups.values())
        self.n_batches = sum(max(1, math.ceil(len(group) / batch_size)) for group in self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for group in self.groups:
            shuffled = np.asarray(group, dtype=np.int64)
            rng.shuffle(shuffled)
            n_chunks = max(1, math.ceil(len(shuffled) / self.batch_size))
            batches.extend(chunk.tolist() for chunk in np.array_split(shuffled, n_chunks))
        rng.shuffle(batches)
        yield from batches


def two_sided_listmle(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    """Plackett-Luce likelihood for both the long and short rankings."""
    if pred.ndim != 1 or target.ndim != 1 or pred.numel() != target.numel():
        raise ValueError("two_sided_listmle expects equal one-dimensional tensors")
    if pred.numel() < 2:
        return pred.sum() * 0.0

    def one_side(scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        ordered = scores[order] / temperature
        log_denominator = torch.logcumsumexp(ordered.flip(0), dim=0).flip(0)
        return (log_denominator - ordered).mean()

    long_order = torch.argsort(target, descending=True)
    short_order = torch.argsort(target, descending=False)
    return 0.5 * (
        one_side(pred, long_order) + one_side(-pred, short_order)
    )


def task_aligned_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    temperature: float,
) -> torch.Tensor:
    """Single-day, label-only objective variants for controlled ablation."""
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    if kind == "listmle":
        return two_sided_listmle(pred, target, temperature)
    if kind == "pearson":
        denom = pred_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()
        return 1.0 - (pred_centered * target_centered).sum() / denom.clamp_min(1e-6)
    if kind == "soft_spearman":
        soft_rank = torch.sigmoid(
            (pred.unsqueeze(1) - pred.unsqueeze(0)) / temperature
        ).sum(dim=1)
        target_rank = torch.argsort(torch.argsort(target)).to(pred.dtype)
        soft_rank = soft_rank - soft_rank.mean()
        target_rank = target_rank - target_rank.mean()
        denom = soft_rank.square().sum().sqrt() * target_rank.square().sum().sqrt()
        return 1.0 - (soft_rank * target_rank).sum() / denom.clamp_min(1e-6)
    if kind == "pairwise":
        losses = []
        for offset in (1, 7, 31):
            if pred.numel() <= offset:
                continue
            target_diff = target[offset:] - target[:-offset]
            valid = target_diff != 0
            if valid.any():
                pred_diff = pred[offset:] - pred[:-offset]
                losses.append(F.softplus(-target_diff[valid].sign() * pred_diff[valid]).mean())
        return torch.stack(losses).mean() if losses else pred.sum() * 0.0
    if kind == "longshort_margin":
        count = max(pred.numel() // 10, 1)
        order = torch.argsort(target)
        spread = pred[order[-count:]].mean() - pred[order[:count]].mean()
        return F.softplus(0.1 - spread)
    raise ValueError(f"unsupported BQ_RANK_LOSS_KIND={kind!r}")


def _apply_training_channel_mask(x: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0.0:
        return x
    keep = torch.rand(
        (x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device
    ) >= probability
    return x * keep


BASE_COLS = ["pre_close", "open", "high", "low", "close", "deal_number", "volume", "amount"]
BOOK_PRICE_COLS = [f"ask_price{i}" for i in range(1, 6)] + [f"bid_price{i}" for i in range(1, 6)]
BOOK_VOLUME_COLS = [f"ask_volume{i}" for i in range(1, 6)] + [f"bid_volume{i}" for i in range(1, 6)]
ORDER_COLS = [f"ask_num_orders{i}" for i in range(1, 6)] + [f"bid_num_orders{i}" for i in range(1, 6)]
FEATURE_COLS = BASE_COLS + BOOK_PRICE_COLS + BOOK_VOLUME_COLS + ORDER_COLS
N_FEAT = len(FEATURE_COLS)
LOCAL_ONLY_BOOK_COLS = (
    [f"ask_price{i}" for i in (4, 5)]
    + [f"bid_price{i}" for i in (4, 5)]
    + [f"ask_volume{i}" for i in (4, 5)]
    + [f"bid_volume{i}" for i in (4, 5)]
    + [f"ask_num_orders{i}" for i in (4, 5)]
    + [f"bid_num_orders{i}" for i in (4, 5)]
)

STYLE_COLS = [
    "SIZE",
    "BETA",
    "MOMENTUM",
    "RESVOL",
    "LIQUIDTY",
    "BTOP",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
    "SIZENL",
]
INDUSTRY_COL = "industry_level1_code"

MODEL_CFG = dict(
    n_feat=N_FEAT,
    seq_len=BAR1M_SEQ_LEN,
    n_tokens=int(os.environ.get("BQ_ADAPTIVE_TOKENS", "48")),
    tokenizer_dim=int(os.environ.get("BQ_TOKENIZER_DIM", "128")),
    d_model=int(os.environ.get("BQ_D_MODEL", "424")),
    nlayers=int(os.environ.get("BQ_MIXER_LAYERS", "11")),
    expansion=int(os.environ.get("BQ_MIXER_EXPANSION", "4")),
    kernel_size=int(os.environ.get("BQ_MIXER_KERNEL", "5")),
    pool_gate_cap=float(os.environ.get("BQ_POOL_GATE_CAP", "0.25")),
    dropout=float(os.environ.get("BQ_DROPOUT", "0.1")),
)

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent
PROJECT_ROOT = HERE.parents[2]
MODEL_PATH = HERE / "transformer_model.json"
ASYNC_OFFICIAL_EVAL = os.environ.get("BQ_ASYNC_OFFICIAL_EVAL", "0") == "1"
OFFICIAL_EVAL_PYTHON = Path(os.environ.get("BQ_OFFICIAL_EVAL_PYTHON", sys.executable))
OFFICIAL_EVAL_SCRIPT = Path(
    os.environ.get(
        "BQ_OFFICIAL_EVAL_SCRIPT",
        str(PROJECT_ROOT / "tools" / "evaluate_official_v4_local.py"),
    )
)
OFFICIAL_EVAL_RUNTIME = Path(
    os.environ.get(
        "BQ_OFFICIAL_EVAL_RUNTIME",
        str(PROJECT_ROOT / "tools" / "official_bigalpha_eval_runtime" / "bigalpha_eval"),
    )
)
OFFICIAL_EVAL_EXPOSURE = Path(
    os.environ.get(
        "BQ_OFFICIAL_EVAL_EXPOSURE",
        str(PROJECT_ROOT / "data" / "data_export_2024_canonical" / "exposure_2024.parquet"),
    )
)


# ============ Utilities ============
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _log_info(event: str, **kwargs) -> None:
    if hasattr(logger, "info"):
        logger.info(event, **kwargs)


def _launch_async_official_eval(idx_df: pd.DataFrame, pred: np.ndarray, epoch: int) -> int | None:
    """Persist one epoch's 2024 scores and launch official v4 without blocking training."""
    if not ASYNC_OFFICIAL_EVAL:
        return None

    required = (OFFICIAL_EVAL_SCRIPT, OFFICIAL_EVAL_RUNTIME, OFFICIAL_EVAL_EXPOSURE)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        _log_info("async official v4 skipped", epoch=epoch, missing=missing)
        return None

    task_dir = HERE / "async_official_eval" / f"epoch_{epoch:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    score_path = task_dir / "score_2024.parquet"
    result_path = task_dir / "official_v4.json"
    log_path = task_dir / "official_v4.log"
    task_path = task_dir / "task.json"

    score = idx_df[["date", "instrument"]].copy()
    score["score"] = np.asarray(pred, dtype=np.float64)
    score = (
        score.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])
        .reset_index(drop=True)
    )
    score.to_parquet(score_path, index=False)

    cmd = [
        str(OFFICIAL_EVAL_PYTHON),
        str(OFFICIAL_EVAL_SCRIPT),
        str(score_path),
        "--exposure",
        str(OFFICIAL_EVAL_EXPOSURE),
        "--runtime",
        str(OFFICIAL_EVAL_RUNTIME),
        "--output",
        str(result_path),
    ]
    with log_path.open("ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    task_path.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "pid": proc.pid,
                "status": "running",
                "score_path": str(score_path),
                "result_path": str(result_path),
                "log_path": str(log_path),
                "command": cmd,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n"
    )
    _log_info("async official v4 launched", epoch=epoch, pid=proc.pid, rows=len(score))
    return proc.pid


def _local_data_dir() -> str | None:
    candidates = [
        os.environ.get("BQ_DATA_DIR"),
        str(PROJ_ROOT / "data" / "data_export"),
        str(Path.cwd() / "data" / "data_export"),
        "./data/data_export",
        "data/data_export",
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            return d
    return None


def _ym_range(start: str, end: str) -> list[str]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    cur = s.replace(day=1)
    out = []
    while cur <= e:
        out.append(cur.strftime("%Y-%m"))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def _as_str_list(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [str(x) for x in values]


def table_file_prefix(table: str) -> str:
    for key in ("bar1m", "bar5m", "bar15m", "bar30m"):
        if key in str(table):
            return key
    raise ValueError(f"unsupported bar table: {table}")


# ============ TimesFM-style model ============
class RevIN(nn.Module):
    """Per-sample reversible normalization over the time axis."""

    def __init__(self, n_feat: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, n_feat))
        self.beta = nn.Parameter(torch.zeros(1, 1, n_feat))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        clean = torch.where(valid > 0, x, torch.zeros_like(x))
        mean = clean.sum(dim=1, keepdim=True) / denom
        var = ((torch.where(valid > 0, x - mean, torch.zeros_like(x))) ** 2).sum(dim=1, keepdim=True) / denom
        z = (torch.where(valid > 0, x, mean) - mean) / torch.sqrt(var + self.eps)
        return z * self.gamma + self.beta


class PatchEncoder(nn.Module):
    """Raw sequence -> causal patch tokens and final patch embedding."""

    def __init__(
        self,
        n_feat: int,
        seq_len: int,
        patch_len: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")
        self.n_feat = n_feat
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len

        token_in = patch_len * n_feat * 2
        self.revin = RevIN(n_feat)
        self.tokenizer = nn.Sequential(
            nn.Linear(token_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[2] != self.n_feat:
            raise ValueError(f"expected (batch, {self.seq_len}, {self.n_feat}), got {tuple(x.shape)}")
        valid = torch.isfinite(x).to(x.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z = self.revin(x, valid)
        z = torch.cat([z, valid], dim=-1)
        b = z.shape[0]
        patches = z.reshape(b, self.n_patches, self.patch_len * self.n_feat * 2)
        h = self.tokenizer(patches) + self.pos
        causal_mask = torch.triu(
            torch.ones(self.n_patches, self.n_patches, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        return self.encoder(h, mask=causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)[:, -1]


class CausalConvBlock(nn.Module):
    """Depthwise temporal residual block with strictly left-only context."""

    def __init__(self, d_model: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = 2 * dilation
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            dilation=dilation,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        y = h.transpose(1, 2)
        y = self.depthwise(F.pad(y, (self.left_pad, 0)))
        y = self.pointwise(F.gelu(y)).transpose(1, 2)
        return self.norm(residual + self.dropout(y))


class RawConvEncoder(nn.Module):
    """Encode the unpatched recent 1m sequence with a causal local receptive field."""

    def __init__(self, n_feat: int, d_model: int, dropout: float):
        super().__init__()
        self.revin = RevIN(n_feat)
        self.input_proj = nn.Sequential(
            nn.Linear(2 * n_feat, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList(
            CausalConvBlock(d_model, dilation, dropout)
            for dilation in (1, 2, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x).to(x.dtype)
        clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.input_proj(torch.cat([self.revin(clean, valid), valid], dim=-1))
        for block in self.blocks:
            h = block(h)
        return h[:, -1]


class DualScaleTimesFM(nn.Module):
    """Fuse slow, coarse-1m and fine-1m context through a residual gate."""

    def __init__(
        self,
        n_feat: int,
        bar5m_seq_len: int,
        bar1m_seq_len: int,
        bar5m_patch_len: int,
        bar1m_patch_len: int,
        bar1m_fine_patch_len: int,
        d_model: int,
        nhead: int,
        bar5m_layers: int,
        bar1m_layers: int,
        bar1m_fine_layers: int,
        fusion_layers: int,
        dim_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bar5m = PatchEncoder(n_feat, bar5m_seq_len, bar5m_patch_len, d_model, nhead, bar5m_layers, dim_ff, dropout)
        self.bar1m = PatchEncoder(n_feat, bar1m_seq_len, bar1m_patch_len, d_model, nhead, bar1m_layers, dim_ff, dropout)
        self.bar1m_fine = PatchEncoder(
            n_feat,
            bar1m_seq_len,
            bar1m_fine_patch_len,
            d_model,
            nhead,
            bar1m_fine_layers,
            dim_ff,
            dropout,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.scale_embed = nn.Parameter(torch.zeros(1, 3, d_model))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, fusion_layers)
        branch_dim = d_model * 3
        self.direct = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        gate_dim = d_model * 2
        self.mix_gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, d_model),
            nn.Sigmoid(),
        )
        self.raw1m = RawConvEncoder(n_feat, d_model, dropout)
        self.local_proj = nn.Linear(d_model, d_model, bias=False)
        self.local_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.scale_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.mix_gate[1].weight)
        nn.init.zeros_(self.mix_gate[1].bias)
        nn.init.normal_(self.local_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.local_gate[1].weight)
        nn.init.zeros_(self.local_gate[1].bias)
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final.bias)
        # Construct the auxiliary head only after every baseline parameter has
        # been initialized so adding it cannot perturb the controlled seed.
        rng_state = torch.get_rng_state()
        self.aux_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        aux_final = self.aux_head[-1]
        if isinstance(aux_final, nn.Linear):
            nn.init.normal_(aux_final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(aux_final.bias)
        torch.set_rng_state(rng_state)

    def forward(self, x5m: torch.Tensor, x1m: torch.Tensor, return_aux: bool = False):
        h5 = self.bar5m.forward_tokens(x5m) + self.scale_embed[:, 0:1]
        h1 = self.bar1m.forward_tokens(x1m) + self.scale_embed[:, 1:2]
        h1_fine = self.bar1m_fine.forward_tokens(x1m) + self.scale_embed[:, 2:3]
        cls = self.cls_token.expand(x5m.shape[0], -1, -1)
        fused = self.fusion(torch.cat([cls, h5, h1, h1_fine], dim=1))
        global_repr = fused[:, 0]
        direct_repr = self.direct(
            torch.cat([h5[:, -1], h1[:, -1], h1_fine[:, -1]], dim=-1)
        )
        gate = self.mix_gate(torch.cat([global_repr, direct_repr], dim=-1))
        mixed = gate * global_repr + (1.0 - gate) * direct_repr
        local = self.local_proj(self.raw1m(x1m))
        local_gate = self.local_gate(torch.cat([mixed, local], dim=-1))
        mixed = mixed + 2.0 * local_gate * local
        primary = self.head(mixed).squeeze(-1)
        if return_aux:
            return primary, self.aux_head(mixed).squeeze(-1)
        return primary


class StockTransformer(DualScaleTimesFM):
    """Official notebook adapter.

    The competition sample calls `model(xb)` with one tensor. This wrapper keeps
    that public interface while using the trained dual-scale 5m+1m structure:
    `xb[:, 0]` is the 5-minute window and `xb[:, 1]` is the 1-minute window.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"expected packed input (batch, 2, seq_len, n_feat), got {tuple(x.shape)}")
        return super().forward(x[:, 0], x[:, 1])


# ============ Pure 1m efficient model ============
class CausalMixerBlock(nn.Module):
    """Causal token mixing plus a wide channel MLP."""

    def __init__(
        self,
        d_model: int,
        expansion: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.left_pad = dilation * (kernel_size - 1)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=d_model,
        )
        self.temporal_proj = nn.Linear(d_model, d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        hidden = d_model * expansion
        self.channel = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.temporal_scale = nn.Parameter(torch.full((d_model,), 1e-3))
        self.channel_scale = nn.Parameter(torch.full((d_model,), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal_norm(x).transpose(1, 2)
        h = torch.nn.functional.pad(h, (self.left_pad, 0))
        h = self.temporal(h).transpose(1, 2)
        x = x + self.temporal_scale * self.temporal_proj(torch.nn.functional.gelu(h))
        return x + self.channel_scale * self.channel(self.channel_norm(x))


class AdaptiveFrequencyTokenizer(nn.Module):
    """Learn a monotone, content-dependent clock before temporal compression."""

    def __init__(
        self,
        n_feat: int,
        seq_len: int,
        n_tokens: int,
        tokenizer_dim: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        if not (1 < n_tokens <= seq_len):
            raise ValueError("n_tokens must be in [2, seq_len]")
        self.n_feat = n_feat
        self.seq_len = seq_len
        self.n_tokens = n_tokens
        self.revin = RevIN(n_feat)
        self.input_proj = nn.Sequential(
            nn.Linear(2 * n_feat, tokenizer_dim),
            nn.GELU(),
            nn.LayerNorm(tokenizer_dim),
        )
        self.context = nn.ModuleList(
            CausalConvBlock(tokenizer_dim, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        self.rate_norm = nn.LayerNorm(tokenizer_dim)
        self.rate_head = nn.Linear(tokenizer_dim, 1)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.constant_(self.rate_head.bias, 0.541324854612918)
        self.log_temperature = nn.Parameter(torch.tensor(7.600902459542082))
        self.output_proj = nn.Sequential(
            nn.Linear(tokenizer_dim + 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        self.register_buffer(
            "token_centers",
            (torch.arange(n_tokens, dtype=torch.float32) + 0.5) / n_tokens,
            persistent=False,
        )
        self.register_buffer(
            "minute_positions",
            (torch.arange(seq_len, dtype=torch.float32) + 0.5) / seq_len,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_feat):
            raise ValueError(
                f"expected (batch, {self.seq_len}, {self.n_feat}), got {tuple(x.shape)}"
            )
        valid = torch.isfinite(x).to(x.dtype)
        clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.input_proj(torch.cat([self.revin(clean, valid), valid], dim=-1))
        for block in self.context:
            h = block(h)

        # A positive learned rate defines a monotone sample-specific clock.
        # High rates allocate more output tokens to that part of the day.
        rate = F.softplus(self.rate_head(self.rate_norm(h))).squeeze(-1) + 1e-4
        clock = torch.cumsum(rate, dim=1) - 0.5 * rate
        clock = clock / rate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        temperature = self.log_temperature.exp().clamp(100.0, 20000.0)
        distance = clock.unsqueeze(-1) - self.token_centers.view(1, 1, -1)
        assignment = torch.softmax(-temperature * distance.square(), dim=-1)

        mass = assignment.sum(dim=1).clamp_min(1e-6)
        pooled = torch.einsum("btk,btd->bkd", assignment, h) / mass.unsqueeze(-1)
        mean_position = torch.einsum(
            "btk,t->bk", assignment, self.minute_positions
        ) / mass
        relative_duration = mass * (self.n_tokens / self.seq_len)
        metadata = torch.stack((relative_duration, 2.0 * mean_position - 1.0), dim=-1)
        return self.output_proj(torch.cat((pooled, metadata), dim=-1)) + self.pos


class AdaptiveFrequencyMixer(nn.Module):
    """Raw 1m model whose effective temporal frequency is learned end to end."""

    def __init__(
        self,
        n_feat: int,
        seq_len: int,
        n_tokens: int,
        tokenizer_dim: int,
        d_model: int,
        nlayers: int,
        expansion: int,
        kernel_size: int,
        pool_gate_cap: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.seq_len = seq_len
        self.tokenizer = AdaptiveFrequencyTokenizer(
            n_feat=n_feat,
            seq_len=seq_len,
            n_tokens=n_tokens,
            tokenizer_dim=tokenizer_dim,
            d_model=d_model,
            dropout=dropout,
        )
        dilations = (1, 2, 4, 8)
        self.blocks = nn.ModuleList(
            CausalMixerBlock(
                d_model,
                expansion,
                kernel_size,
                dilations[i % len(dilations)],
                dropout,
            )
            for i in range(nlayers)
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.aux_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        for head in (self.head, self.aux_head):
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head[-1].bias)
        self.pool_score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.pool_gate = nn.Linear(2 * d_model, 1)
        if not 0.11920292202211755 < pool_gate_cap <= 1.0:
            raise ValueError("pool_gate_cap must exceed the common initial global mix")
        self.pool_gate_cap = float(pool_gate_cap)
        nn.init.zeros_(self.pool_score[-1].weight)
        nn.init.zeros_(self.pool_score[-1].bias)
        nn.init.zeros_(self.pool_gate.weight)
        initial_ratio = 0.11920292202211755 / self.pool_gate_cap
        nn.init.constant_(self.pool_gate.bias, float(np.log(initial_ratio / (1.0 - initial_ratio))))

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_feat):
            raise ValueError(
                f"expected (batch, {self.seq_len}, {self.n_feat}), got {tuple(x.shape)}"
            )
        h = self.tokenizer(x)
        for block in self.blocks:
            h = block(h)
        last = h[:, -1]
        pool_weight = torch.softmax(self.pool_score(h).squeeze(-1), dim=1)
        global_repr = torch.sum(pool_weight.unsqueeze(-1) * h, dim=1)
        gate = self.pool_gate_cap * torch.sigmoid(
            self.pool_gate(torch.cat((last, global_repr), dim=-1))
        )
        final = last + gate * (global_repr - last)
        primary = self.head(final).squeeze(-1)
        if return_aux:
            return primary, self.aux_head(final).squeeze(-1)
        return primary


# Keep the official sample's class names while replacing the dual-scale model.
DualScaleTimesFM = AdaptiveFrequencyMixer


class StockTransformer(AdaptiveFrequencyMixer):
    pass


# ============ Label neutralization ============
def barra_neutralize(
    fwd_ret_df: pd.DataFrame,
    expo_df: pd.DataFrame,
    style_cols: list[str] = STYLE_COLS,
    industry_col: str = INDUSTRY_COL,
) -> pd.DataFrame:
    fwd_ret_df = fwd_ret_df.copy()
    expo_df = expo_df.copy()
    fwd_ret_df["date"] = pd.to_datetime(fwd_ret_df["date"]).dt.normalize()
    expo_df["date"] = pd.to_datetime(expo_df["date"]).dt.normalize()

    df = fwd_ret_df.merge(expo_df, on=["date", "instrument"], how="inner")
    use_styles = [c for c in style_cols if c in df.columns]
    if not use_styles:
        raise RuntimeError(f"no expected BARRA style columns found in exposure: {style_cols}")
    have_industry = industry_col in df.columns

    out = []
    for date, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["fwd_ret"] + use_styles).copy()
        if len(g) < len(use_styles) + 5:
            continue
        y = g["fwd_ret"].to_numpy(np.float64)
        x_style = g[use_styles].to_numpy(np.float64)
        x_style = (x_style - x_style.mean(axis=0)) / (x_style.std(axis=0) + 1e-8)
        if have_industry:
            x_ind = pd.get_dummies(g[industry_col].astype(str), drop_first=True).to_numpy(np.float64)
            x = np.hstack([x_style, x_ind])
        else:
            x = x_style
        x = np.column_stack([np.ones(len(y)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        sub = g[["date", "instrument"]].copy()
        sub["label"] = (y - x @ beta).astype(np.float32)
        out.append(sub)
    if not out:
        raise RuntimeError("barra_neutralize produced no valid cross sections")
    _log_info("BARRA neutralization complete", days=len(out), styles=use_styles, industry=have_industry)
    return pd.concat(out, ignore_index=True)


def transform_train_labels(labels: pd.DataFrame, mode: str = LABEL_MODE) -> pd.DataFrame:
    """Optionally transform training labels within each date cross-section.

    This changes the supervised target only. It does not add hand-crafted input
    features and is not used for raw-return 2024 validation labels.
    """
    if mode == "raw":
        return labels
    if "label" not in labels.columns:
        raise ValueError("transform_train_labels expects a label column")

    out = labels.copy()
    raw_values = out["label"].to_numpy(np.float64, copy=True)
    date_col = "_day" if "_day" in out.columns else "date"
    grouped = out.groupby(date_col, sort=False)["label"]
    if mode == "cs_zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        out["label"] = ((out["label"] - mean) / (std + 1e-8)).fillna(0.0).astype(np.float32)
    elif mode == "cs_gaussian_rank":
        rank = grouped.rank(method="average")
        count = grouped.transform("size").astype(np.float64)
        pct = np.clip(((rank - 0.5) / count).to_numpy(np.float64), 1e-6, 1.0 - 1e-6)
        values = torch.erfinv(torch.from_numpy(2.0 * pct - 1.0)).numpy() * np.sqrt(2.0)
        out["label"] = values.astype(np.float32)
    elif mode == "cs_rank":
        pct = grouped.rank(method="average", pct=True)
        out["label"] = ((pct - 0.5) * 2.0).astype(np.float32)
    else:
        raise ValueError(
            f"unsupported BQ_LABEL_MODE={mode!r}; use raw, cs_zscore, cs_gaussian_rank, or cs_rank"
        )
    if mode in {"cs_zscore", "cs_gaussian_rank"}:
        transformed = out["label"].to_numpy(np.float64)
        raw_lo, raw_hi = np.percentile(raw_values, [1, 99])
        dst_lo, dst_hi = np.percentile(transformed, [1, 99])
        raw_scale = np.std(np.clip(raw_values, raw_lo, raw_hi))
        dst_scale = np.std(np.clip(transformed, dst_lo, dst_hi))
        if raw_scale > 0 and dst_scale > 0:
            out["label"] = (out["label"] * (raw_scale / dst_scale)).astype(np.float32)
    return out


# ============ Local validation metrics ============
def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    ar = a.rank(method="average").to_numpy(np.float64)
    br = b.rank(method="average").to_numpy(np.float64)
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = np.sqrt(np.sum(ar * ar) * np.sum(br * br)) + 1e-12
    return float(np.sum(ar * br) / denom)


def smooth_l1_numpy(pred: np.ndarray, target: np.ndarray, beta: float = 0.01) -> float:
    diff = np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64))
    loss = np.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return float(np.mean(loss))


def evaluate_scores(df: pd.DataFrame, min_cross_section: int = 20, groups: int = 10) -> dict:
    """Compute local holdout metrics from columns date/instrument/score/label.

    This is a local approximation of the platform view:
      - IC uses daily cross-sectional Spearman rank correlation.
      - IC_IR is mean(IC) / std(IC).
      - SR is annualized daily long-short return from top-vs-bottom score groups.
      - Stress is the worst monthly IC mean with at least five valid IC days.
    """
    required = {"date", "instrument", "score", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"evaluate_scores missing columns: {sorted(missing)}")

    work = df[list(required)].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])

    ic_rows, ls_rows = [], []
    for date, g in work.groupby("date", sort=True):
        if len(g) < min_cross_section or g["score"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        ic = _rank_corr(g["score"], g["label"])
        ic_rows.append((date, ic))

        q = min(groups, max(2, len(g) // 2))
        ranks = g["score"].rank(method="first")
        bucket = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
        tmp = g.assign(_bucket=bucket)
        lo = tmp[tmp["_bucket"] == tmp["_bucket"].min()]["label"].mean()
        hi = tmp[tmp["_bucket"] == tmp["_bucket"].max()]["label"].mean()
        if np.isfinite(hi) and np.isfinite(lo):
            ls_rows.append((date, float(hi - lo)))

    if not ic_rows:
        return {
            "days": 0,
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "long_short_sr": float("nan"),
            "stress_ic_mean": float("nan"),
        }

    ic_df = pd.DataFrame(ic_rows, columns=["date", "ic"])
    ic_values = ic_df["ic"].to_numpy(np.float64)
    ic_mean = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    ic_ir = float(ic_mean / (ic_std + 1e-12))

    if ls_rows:
        ls = np.array([x[1] for x in ls_rows], dtype=np.float64)
        ls_std = float(np.std(ls, ddof=1)) if len(ls) > 1 else 0.0
        long_short_sr = float(np.mean(ls) / (ls_std + 1e-12) * np.sqrt(252.0))
    else:
        long_short_sr = float("nan")

    ic_df["month"] = ic_df["date"].dt.to_period("M")
    monthly = ic_df.groupby("month").filter(lambda x: len(x) >= 5).groupby("month")["ic"].mean()
    stress_ic_mean = float(monthly.min()) if len(monthly) else float(np.min(ic_values))

    return {
        "days": int(len(ic_values)),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "long_short_sr": long_short_sr,
        "stress_ic_mean": stress_ic_mean,
    }


def _format_metrics(metrics: dict) -> str:
    if metrics.get("days", 0) == 0:
        return "days=0 IC_mean=nan IC_IR=nan SR=nan Stress=nan"
    return (
        f"days={metrics['days']} "
        f"IC_mean={metrics['ic_mean']:+.5f} "
        f"IC_IR={metrics['ic_ir']:+.5f} "
        f"SR={metrics['long_short_sr']:+.5f} "
        f"Stress={metrics['stress_ic_mean']:+.5f}"
    )


def proxy_composite_score(history: list[dict]) -> float:
    """Approximate the competition's equal-weight rank score within one run."""
    if not history:
        return float("-inf")
    keys = ["ic_mean", "ic_ir", "long_short_sr", "stress_ic_mean"]
    cur = history[-1]
    ranks = []
    for k in keys:
        vals = [m.get(k, float("nan")) for m in history]
        finite = [v for v in vals if np.isfinite(v)]
        cv = cur.get(k, float("nan"))
        if not finite or not np.isfinite(cv):
            ranks.append(0.0)
            continue
        ranks.append(sum(v <= cv for v in finite) / len(finite))
    return float(np.mean(ranks))


def baseline_geomean_score(metrics: dict) -> float:
    references = {
        "ic_mean": 0.08746,
        "ic_ir": 0.66147,
        "long_short_sr": 9.22817,
        "stress_ic_mean": 0.02871,
    }
    ratios = np.asarray([metrics.get(k, np.nan) / v for k, v in references.items()], np.float64)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        return float("-inf")
    return float(np.exp(np.mean(np.log(ratios))))


def pareto_geomean_score(metrics: dict) -> float:
    """Reject candidates dominated by a known strong model, then rank by geomean."""
    keys = ("ic_mean", "ic_ir", "long_short_sr", "stress_ic_mean")
    values = np.asarray([metrics.get(key, np.nan) for key in keys], np.float64)
    references = (
        np.asarray([0.08746082848609603, 0.6614692274282783, 9.22816701726943, 0.02871583486358266]),
        np.asarray([0.09040542393273701, 0.6255975057758728, 9.46828370526795, 0.0326832740279514]),
        np.asarray([0.07972186496927368, 0.8243440455332257, 11.37648706704926, 0.044285332366270566]),
    )
    if not np.all(np.isfinite(values)):
        return float("-inf")
    for reference in references:
        if np.all(values <= reference) and np.any(values < reference):
            return float("-inf")
    return baseline_geomean_score(metrics)


def _predict_array(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    xt = torch.from_numpy(np.ascontiguousarray(x))
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            preds.append(model(xt[i : i + batch_size].to(device, non_blocking=True)).detach().cpu().numpy())
    model.train()
    return np.concatenate(preds).astype(np.float64)


def _predict_multiscale_array(model: nn.Module, x5: np.ndarray, x1: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    x5t = torch.from_numpy(np.ascontiguousarray(x5))
    x1t = torch.from_numpy(np.ascontiguousarray(x1))
    with torch.no_grad():
        for i in range(0, len(x5), batch_size):
            preds.append(
                model(
                    x5t[i : i + batch_size].to(device, non_blocking=True),
                    x1t[i : i + batch_size].to(device, non_blocking=True),
                ).detach().cpu().numpy()
            )
    model.train()
    return np.concatenate(preds).astype(np.float64)


# ============ Data access ============
def _align_submission_bar_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the model input aligned between compressed e2e data and cloud stock data.

    The local e2e tables only contain 3 book levels. Cloud prediction tables carry
    5 levels, so levels 4/5 must be ignored instead of becoming unseen live inputs.
    The columns remain present to keep checkpoint shapes backward compatible.
    """
    df = df.copy()
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    for c in LOCAL_ONLY_BOOK_COLS:
        if c in df.columns:
            df[c] = 0
    return df


def _submission_time_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts == end_ts.normalize():
        end_exclusive = end_ts + pd.Timedelta(days=1)
    else:
        end_exclusive = end_ts + pd.Timedelta(nanoseconds=1)
    return start_ts, end_exclusive


def read_bar(table: str, start: str, end: str, instruments: Iterable[str] | None = None) -> pd.DataFrame:
    cols = ["date", "instrument"] + FEATURE_COLS
    prefix = table_file_prefix(table)
    inst_list = _as_str_list(instruments)
    dd = _local_data_dir()
    if dd:
        import pyarrow.parquet as pq

        start_ts, end_exclusive = _submission_time_bounds(start, end)
        filters = [("date", ">=", start_ts), ("date", "<", end_exclusive)]
        if inst_list is not None:
            filters.append(("instrument", "in", inst_list))
        t0 = time.time()
        files = [os.path.join(dd, f"{prefix}_{ym}.parquet") for ym in _ym_range(start, end)]
        files = [f for f in files if os.path.exists(f)]
        read_workers = max(0, int(os.environ.get("BQ_READ_WORKERS", str(READ_WORKERS))))

        def _read_one(path: str) -> pd.DataFrame:
            return pq.read_table(path, columns=cols, filters=filters, use_threads=True).to_pandas()

        if read_workers > 1 and len(files) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(read_workers, len(files))) as ex:
                parts = list(ex.map(_read_one, files))
        else:
            parts = [_read_one(f) for f in files]
        if not parts:
            raise FileNotFoundError(f"no local {prefix} files cover [{start}, {end}] in {dd}")
        df = pd.concat(parts, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df["instrument"] = df["instrument"].astype(str)
        df = df[(df["date"] >= start_ts) & (df["date"] < end_exclusive)]
        if inst_list is not None:
            df = df[df["instrument"].isin(inst_list)]
        df = _align_submission_bar_schema(df)
        _log_info(
            "local bar read complete",
            table=prefix,
            files=len(files),
            rows=len(df),
            read_workers=read_workers,
            elapsed=round(time.time() - t0, 2),
        )
        return df

    import dai

    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table}"
    df = dai.query(sql, filters={"date": [start, end], "instrument": inst_list}, compression=True).df()
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    return _align_submission_bar_schema(df)


def read_bar5m(start: str, end: str, instruments: Iterable[str] | None = None) -> pd.DataFrame:
    return read_bar("bar5m", start, end, instruments)


def read_exposure(start: str, end: str) -> pd.DataFrame:
    cols = ["date", "instrument"] + STYLE_COLS + [INDUSTRY_COL]
    dd = _local_data_dir()
    if dd:
        import pyarrow.parquet as pq

        f = os.path.join(dd, "exposure.parquet")
        try:
            df = pq.read_table(
                f,
                columns=cols,
                filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))],
            ).to_pandas()
        except Exception:
            df = pq.read_table(f, columns=cols).to_pandas()
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        df["date"] = pd.to_datetime(df["date"])
        df["instrument"] = df["instrument"].astype(str)
        return df

    import dai

    return dai.query("SELECT * FROM bigalpha_2026_exposure", filters={"date": [start, end]}, compression=True).df()


def read_instruments(start: str, end: str) -> pd.DataFrame:
    dd = _local_data_dir()
    if dd:
        import pyarrow.parquet as pq

        f = os.path.join(dd, "instruments.parquet")
        try:
            df = pq.read_table(
                f,
                columns=["date", "instrument"],
                filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))],
            ).to_pandas()
        except Exception:
            df = pq.read_table(f, columns=["date", "instrument"]).to_pandas()
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        df["date"] = pd.to_datetime(df["date"])
        df["instrument"] = df["instrument"].astype(str)
        return df[["date", "instrument"]]

    import dai

    return dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start, end]},
        compression=True,
    ).df()


def pool(start: str, end: str) -> list[str]:
    return read_instruments(start, end)["instrument"].astype(str).unique().tolist()


# ============ Sample building ============
def _daily_fwd_ret(df_bar: pd.DataFrame) -> pd.DataFrame:
    close = df_bar.groupby(["instrument", "_day"], sort=True)["close"].last().reset_index()
    close = close.sort_values(["instrument", "_day"])
    close["fwd_ret"] = close.groupby("instrument")["close"].shift(-1) / close["close"] - 1.0
    return close


def _preprocess_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    df = df.copy()
    arr = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    np.maximum(arr, 0.0, out=arr)
    np.log1p(arr, out=arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    feature_df = pd.DataFrame(arr, index=df.index, columns=FEATURE_COLS)
    df = pd.concat([df.drop(columns=FEATURE_COLS), feature_df], axis=1)
    _log_info("bar feature preprocess complete", rows=len(df), cols=len(FEATURE_COLS), elapsed=round(time.time() - t0, 2))
    return df


def _preprocess_feature_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.log1p(np.clip(x, 0.0, None))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _aggregate_1m_to_5m_like_array(raw: np.ndarray, stride: int = 5) -> np.ndarray:
    """Aggregate raw 1m rows into 5m-like bars, then apply feature preprocessing."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE_COLS):
        raise ValueError(f"expected (n, {len(FEATURE_COLS)}), got {raw.shape}")
    if len(raw) % stride != 0:
        raise ValueError("raw length must be divisible by stride")
    groups = raw.reshape(len(raw) // stride, stride, len(FEATURE_COLS))
    agg = groups[:, -1, :].copy()
    ci = {c: i for i, c in enumerate(FEATURE_COLS)}

    for c in ("pre_close", "open"):
        agg[:, ci[c]] = groups[:, 0, ci[c]]
    agg[:, ci["high"]] = groups[:, :, ci["high"]].max(axis=1)
    agg[:, ci["low"]] = groups[:, :, ci["low"]].min(axis=1)
    agg[:, ci["close"]] = groups[:, -1, ci["close"]]
    for c in ("deal_number", "volume", "amount"):
        agg[:, ci[c]] = groups[:, :, ci[c]].sum(axis=1)
    return _preprocess_feature_array(agg)


_PAR_WINDOW_DF = None
_PAR_WINDOW_GROUPS = None
_PAR_WINDOW_SEQ_LEN = None
_PAR_WINDOW_SD_TS = None
_PAR_WINDOW_ED_TS = None
_PAR_WINDOW_KEYS_FILTER = None
_PAR_WINDOW_KIND = None


def _parallel_worker_by_instrument(ins: str) -> list[tuple[tuple[pd.Timestamp, str], np.ndarray]]:
    idx = _PAR_WINDOW_GROUPS[ins]
    sub = _PAR_WINDOW_DF.take(idx).sort_values("date")
    days = sub["_day"].to_numpy()
    feats = sub[FEATURE_COLS].to_numpy(np.float32)
    day_last = np.flatnonzero(np.append(days[1:] != days[:-1], True))
    out = []
    for p in day_last:
        d = pd.Timestamp(days[p])
        key = (d, ins)
        if p + 1 < _PAR_WINDOW_SEQ_LEN or d < _PAR_WINDOW_SD_TS or d > _PAR_WINDOW_ED_TS:
            continue
        if _PAR_WINDOW_KEYS_FILTER is not None and key not in _PAR_WINDOW_KEYS_FILTER:
            continue
        if _PAR_WINDOW_KIND == "resampled5m":
            raw_window = feats[p - _PAR_WINDOW_SEQ_LEN + 1 : p + 1]
            arr = _aggregate_1m_to_5m_like_array(raw_window, BAR5M_FROM_1M_STRIDE)
        else:
            arr = feats[p - _PAR_WINDOW_SEQ_LEN + 1 : p + 1].copy()
        out.append((key, arr))
    return out


def _parallel_windows_by_key(
    df: pd.DataFrame,
    seq_len: int,
    sd: str,
    ed: str,
    keys_filter: set[tuple[pd.Timestamp, str]] | None = None,
    kind: str = "plain",
) -> tuple[dict[tuple[pd.Timestamp, str], np.ndarray], pd.DataFrame]:
    workers = max(0, int(os.environ.get("BQ_BUILD_WORKERS", str(BUILD_WORKERS))))
    if workers <= 1:
        if kind == "resampled5m":
            return _make_resampled5m_windows_by_key_serial(df, seq_len, sd, ed, keys_filter)
        return _make_windows_by_key_serial(df, seq_len, sd, ed, keys_filter)
    import multiprocessing as mp

    if not hasattr(mp, "get_context"):
        if kind == "resampled5m":
            return _make_resampled5m_windows_by_key_serial(df, seq_len, sd, ed, keys_filter)
        return _make_windows_by_key_serial(df, seq_len, sd, ed, keys_filter)

    global _PAR_WINDOW_DF, _PAR_WINDOW_GROUPS, _PAR_WINDOW_SEQ_LEN, _PAR_WINDOW_SD_TS
    global _PAR_WINDOW_ED_TS, _PAR_WINDOW_KEYS_FILTER, _PAR_WINDOW_KIND
    _PAR_WINDOW_DF = df
    _PAR_WINDOW_GROUPS = df.groupby("instrument", sort=False).indices
    _PAR_WINDOW_SEQ_LEN = seq_len
    _PAR_WINDOW_SD_TS = pd.Timestamp(sd)
    _PAR_WINDOW_ED_TS = pd.Timestamp(ed)
    _PAR_WINDOW_KEYS_FILTER = keys_filter
    _PAR_WINDOW_KIND = kind

    instruments = list(_PAR_WINDOW_GROUPS.keys())
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(workers, len(instruments))) as pool:
        parts = pool.map(_parallel_worker_by_instrument, instruments, chunksize=1)
    out = {}
    rows = []
    for part in parts:
        for key, arr in part:
            out[key] = arr
            rows.append(key)
    _log_info(
        "parallel windows built",
        kind=kind,
        workers=workers,
        instruments=len(instruments),
        windows=len(rows),
        elapsed=round(time.time() - t0, 2),
    )
    return out, pd.DataFrame(rows, columns=["date", "instrument"])


def _make_windows_by_key(
    df: pd.DataFrame,
    seq_len: int,
    sd: str,
    ed: str,
    keys_filter: set[tuple[pd.Timestamp, str]] | None = None,
) -> tuple[dict[tuple[pd.Timestamp, str], np.ndarray], pd.DataFrame]:
    return _parallel_windows_by_key(df, seq_len, sd, ed, keys_filter, "plain")


def _make_windows_by_key_serial(
    df: pd.DataFrame,
    seq_len: int,
    sd: str,
    ed: str,
    keys_filter: set[tuple[pd.Timestamp, str]] | None = None,
) -> tuple[dict[tuple[pd.Timestamp, str], np.ndarray], pd.DataFrame]:
    sd_ts = pd.Timestamp(sd)
    ed_ts = pd.Timestamp(ed)
    out = {}
    keys = []
    for ins, sub in df.groupby("instrument", sort=False):
        sub = sub.sort_values("date")
        days = sub["_day"].to_numpy()
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day_last = np.flatnonzero(np.append(days[1:] != days[:-1], True))
        for p in day_last:
            d = pd.Timestamp(days[p])
            key = (d, ins)
            if p + 1 < seq_len or d < sd_ts or d > ed_ts:
                continue
            if keys_filter is not None and key not in keys_filter:
                continue
            out[key] = feats[p - seq_len + 1 : p + 1]
            keys.append(key)
    return out, pd.DataFrame(keys, columns=["date", "instrument"])


def _make_resampled5m_windows_by_key_serial(
    df_raw: pd.DataFrame,
    source_len: int,
    sd: str,
    ed: str,
    keys_filter: set[tuple[pd.Timestamp, str]] | None = None,
) -> tuple[dict[tuple[pd.Timestamp, str], np.ndarray], pd.DataFrame]:
    sd_ts = pd.Timestamp(sd)
    ed_ts = pd.Timestamp(ed)
    out = {}
    keys = []
    for ins, sub in df_raw.groupby("instrument", sort=False):
        sub = sub.sort_values("date")
        days = sub["_day"].to_numpy()
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day_last = np.flatnonzero(np.append(days[1:] != days[:-1], True))
        for p in day_last:
            d = pd.Timestamp(days[p])
            key = (d, ins)
            if p + 1 < source_len or d < sd_ts or d > ed_ts:
                continue
            if keys_filter is not None and key not in keys_filter:
                continue
            raw_window = feats[p - source_len + 1 : p + 1]
            out[key] = _aggregate_1m_to_5m_like_array(raw_window, BAR5M_FROM_1M_STRIDE)
            keys.append(key)
    return out, pd.DataFrame(keys, columns=["date", "instrument"])


def _make_sampled_windows_by_key(
    df: pd.DataFrame,
    source_len: int,
    stride: int,
    sd: str,
    ed: str,
    keys_filter: set[tuple[pd.Timestamp, str]] | None = None,
) -> tuple[dict[tuple[pd.Timestamp, str], np.ndarray], pd.DataFrame]:
    sd_ts = pd.Timestamp(sd)
    ed_ts = pd.Timestamp(ed)
    out = {}
    keys = []
    for ins, sub in df.groupby("instrument", sort=False):
        sub = sub.sort_values("date")
        days = sub["_day"].to_numpy()
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day_last = np.flatnonzero(np.append(days[1:] != days[:-1], True))
        for p in day_last:
            d = pd.Timestamp(days[p])
            key = (d, ins)
            if p + 1 < source_len or d < sd_ts or d > ed_ts:
                continue
            if keys_filter is not None and key not in keys_filter:
                continue
            # Slow branch is sampled from raw 1m bars, taking one snapshot every
            # five rows. This avoids depending on a separate bar5m datasource.
            out[key] = feats[p - source_len + 1 : p + 1][stride - 1 :: stride]
            keys.append(key)
    return out, pd.DataFrame(keys, columns=["date", "instrument"])


def build_multiscale_samples(
    table5m: str,
    table1m: str,
    sd: str,
    ed: str,
    mode: str,
    instruments: Iterable[str],
    expo_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    t0 = time.time()
    buf = (pd.Timestamp(sd) - pd.Timedelta(days=max(BAR5M_LOOKBACK_DAYS, BAR1M_LOOKBACK_DAYS) + 5)).strftime("%Y-%m-%d")
    df5 = _preprocess_bar_features(read_bar(table5m, buf, ed, instruments))
    df1 = _preprocess_bar_features(read_bar(table1m, buf, ed, instruments))
    for df in (df5, df1):
        df["date"] = pd.to_datetime(df["date"])
        df["_day"] = df["date"].dt.normalize()

    labels = None
    if mode == "train":
        if expo_df is None:
            raise ValueError("expo_df is required for train mode")
        fwd = _daily_fwd_ret(df5).rename(columns={"_day": "date"})
        labels = barra_neutralize(fwd[["date", "instrument", "fwd_ret"]], expo_df).rename(columns={"date": "_day"})
        labels = transform_train_labels(labels)

    win5, idx5 = _make_windows_by_key(df5, BAR5M_SEQ_LEN, sd, ed)
    common_keys = set(win5.keys())
    if mode == "train" and labels is not None:
        label_map = {
            (pd.Timestamp(d), str(ins)): float(label)
            for d, ins, label in zip(labels["_day"], labels["instrument"], labels["label"])
            if np.isfinite(float(label))
        }
        common_keys &= set(label_map.keys())
    else:
        label_map = {}
    win1, _ = _make_windows_by_key(df1, BAR1M_SEQ_LEN, sd, ed, common_keys)
    common_keys &= set(win1.keys())
    keys = sorted(common_keys)

    if not keys:
        raise RuntimeError(f"build_multiscale_samples produced no samples for mode={mode}, range={sd}~{ed}")
    x5 = np.stack([win5[k] for k in keys]).astype(np.float32)
    x1 = np.stack([win1[k] for k in keys]).astype(np.float32)
    y = np.asarray([label_map[k] for k in keys], np.float32) if mode == "train" else None
    idx = pd.DataFrame(keys, columns=["date", "instrument"])
    _log_info("multiscale samples built", mode=mode, n=len(idx), elapsed=round(time.time() - t0, 2))
    return x5, x1, y, idx


def build_multiscale_samples_from_1m(
    table1m: str,
    sd: str,
    ed: str,
    mode: str,
    instruments: Iterable[str],
    expo_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    t0 = time.time()
    buf = (pd.Timestamp(sd) - pd.Timedelta(days=BAR5M_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")
    df1 = _preprocess_bar_features(read_bar(table1m, buf, ed, instruments))
    df1["date"] = pd.to_datetime(df1["date"])
    df1["_day"] = df1["date"].dt.normalize()

    labels = None
    if mode == "train":
        if expo_df is None:
            raise ValueError("expo_df is required for train mode")
        fwd = _daily_fwd_ret(df1).rename(columns={"_day": "date"})
        labels = barra_neutralize(fwd[["date", "instrument", "fwd_ret"]], expo_df).rename(columns={"date": "_day"})
        labels = transform_train_labels(labels)

    win1, _ = _make_windows_by_key(df1, BAR1M_SEQ_LEN, sd, ed)
    common_keys = set(win1.keys())
    if mode == "train" and labels is not None:
        label_map = {
            (pd.Timestamp(d), str(ins)): float(label)
            for d, ins, label in zip(labels["_day"], labels["instrument"], labels["label"])
            if np.isfinite(float(label))
        }
        common_keys &= set(label_map.keys())
    else:
        label_map = {}
    win5, _ = _make_sampled_windows_by_key(
        df1,
        BAR5M_FROM_1M_SOURCE_LEN,
        BAR5M_FROM_1M_STRIDE,
        sd,
        ed,
        common_keys,
    )
    common_keys &= set(win5.keys())
    keys = sorted(common_keys)

    if not keys:
        raise RuntimeError(f"build_multiscale_samples_from_1m produced no samples for mode={mode}, range={sd}~{ed}")
    x5 = np.stack([win5[k] for k in keys]).astype(np.float32)
    x1 = np.stack([win1[k] for k in keys]).astype(np.float32)
    y = np.asarray([label_map[k] for k in keys], np.float32) if mode == "train" else None
    idx = pd.DataFrame(keys, columns=["date", "instrument"])
    _log_info("1m-sampled multiscale samples built", mode=mode, n=len(idx), elapsed=round(time.time() - t0, 2))
    return x5, x1, y, idx


def build_multiscale_samples_from_1m_resampled5m(
    table1m: str,
    sd: str,
    ed: str,
    mode: str,
    instruments: Iterable[str],
    expo_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    t0 = time.time()
    buf = (pd.Timestamp(sd) - pd.Timedelta(days=BAR5M_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")
    df_raw = read_bar(table1m, buf, ed, instruments)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["_day"] = df_raw["date"].dt.normalize()
    df1 = _preprocess_bar_features(df_raw)

    labels = None
    if mode == "train":
        if expo_df is None:
            raise ValueError("expo_df is required for train mode")
        fwd = _daily_fwd_ret(df_raw).rename(columns={"_day": "date"})
        labels = barra_neutralize(fwd[["date", "instrument", "fwd_ret"]], expo_df).rename(columns={"date": "_day"})
        labels = transform_train_labels(labels)

    win1, _ = _make_windows_by_key(df1, BAR1M_SEQ_LEN, sd, ed)
    common_keys = set(win1.keys())
    if mode == "train" and labels is not None:
        label_map = {
            (pd.Timestamp(d), str(ins)): float(label)
            for d, ins, label in zip(labels["_day"], labels["instrument"], labels["label"])
            if np.isfinite(float(label))
        }
        common_keys &= set(label_map.keys())
    else:
        label_map = {}

    win5, _ = _parallel_windows_by_key(
        df_raw,
        BAR5M_FROM_1M_SOURCE_LEN,
        sd,
        ed,
        common_keys,
        "resampled5m",
    )

    common_keys &= set(win5.keys())
    keys = sorted(common_keys)
    if not keys:
        raise RuntimeError(f"build_multiscale_samples_from_1m_resampled5m produced no samples for mode={mode}, range={sd}~{ed}")
    x5 = np.stack([win5[k] for k in keys]).astype(np.float32)
    x1 = np.stack([win1[k] for k in keys]).astype(np.float32)
    y = np.asarray([label_map[k] for k in keys], np.float32) if mode == "train" else None
    idx = pd.DataFrame(keys, columns=["date", "instrument"])
    _log_info("1m-resampled5m multiscale samples built", mode=mode, n=len(idx), elapsed=round(time.time() - t0, 2))
    return x5, x1, y, idx


def _shard_manifest_path(cache_dir: str | os.PathLike) -> Path:
    return Path(cache_dir) / "manifest.json"


def _instrument_chunks(instruments: Iterable[str], chunk_size: int) -> list[list[str]]:
    values = [str(x) for x in instruments]
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def read_close_bar(table: str, start: str, end: str, instruments: Iterable[str] | None = None) -> pd.DataFrame:
    cols = ["date", "instrument", "close"]
    prefix = table_file_prefix(table)
    inst_list = _as_str_list(instruments)
    dd = _local_data_dir()
    if dd:
        import pyarrow.parquet as pq

        start_ts, end_exclusive = _submission_time_bounds(start, end)
        filters = [("date", ">=", start_ts), ("date", "<", end_exclusive)]
        if inst_list is not None:
            filters.append(("instrument", "in", inst_list))
        t0 = time.time()
        files = [os.path.join(dd, f"{prefix}_{ym}.parquet") for ym in _ym_range(start, end)]
        files = [f for f in files if os.path.exists(f)]

        def _read_one(path: str) -> pd.DataFrame:
            return pq.read_table(path, columns=cols, filters=filters, use_threads=True).to_pandas()

        read_workers = max(0, int(os.environ.get("BQ_READ_WORKERS", str(READ_WORKERS))))
        if read_workers > 1 and len(files) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(read_workers, len(files))) as ex:
                parts = list(ex.map(_read_one, files))
        else:
            parts = [_read_one(f) for f in files]
        if not parts:
            raise FileNotFoundError(f"no local {prefix} files cover [{start}, {end}] in {dd}")
        df = pd.concat(parts, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df["instrument"] = df["instrument"].astype(str)
        df = df[(df["date"] >= start_ts) & (df["date"] < end_exclusive)]
        if inst_list is not None:
            df = df[df["instrument"].isin(inst_list)]
        _log_info(
            "local close read complete",
            table=prefix,
            files=len(files),
            rows=len(df),
            read_workers=read_workers,
            elapsed=round(time.time() - t0, 2),
        )
        return df

    import dai

    sql = f"SELECT date, instrument, close FROM {table}"
    df = dai.query(sql, filters={"date": [start, end], "instrument": inst_list}, compression=True).df()
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    return df


def build_or_load_train_labels(table1m: str, train_start: str, train_end: str, instruments: Iterable[str], cache_dir: Path) -> Path:
    label_path = cache_dir / "train_labels.parquet"
    if label_path.exists():
        return label_path
    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_close = read_close_bar(table1m, train_start, train_end, instruments)
    df_close["_day"] = pd.to_datetime(df_close["date"]).dt.normalize()
    fwd = _daily_fwd_ret(df_close).rename(columns={"_day": "date"})
    expo = read_exposure(train_start, train_end)
    labels = barra_neutralize(fwd[["date", "instrument", "fwd_ret"]], expo).rename(columns={"date": "_day"})
    labels = transform_train_labels(labels)
    labels = labels[["_day", "instrument", "label"]].dropna(subset=["label"]).copy()
    labels["_day"] = pd.to_datetime(labels["_day"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels.to_parquet(label_path, index=False)
    _log_info("train label cache built", rows=len(labels), path=str(label_path), elapsed=round(time.time() - t0, 2))
    return label_path


def _build_train_shard_worker(args: tuple[int, list[str], str, str, str, str, str]) -> dict:
    shard_id, instruments, table1m, train_start, train_end, cache_dir, label_path = args
    t0 = time.time()
    old_build_workers = os.environ.get("BQ_BUILD_WORKERS")
    os.environ["BQ_BUILD_WORKERS"] = "0"
    try:
        x5, x1, _, idx = build_multiscale_samples_from_1m_resampled5m(
            table1m,
            train_start,
            train_end,
            "infer",
            instruments,
        )
    finally:
        if old_build_workers is None:
            os.environ.pop("BQ_BUILD_WORKERS", None)
        else:
            os.environ["BQ_BUILD_WORKERS"] = old_build_workers

    import pyarrow.parquet as pq

    labels = pq.read_table(label_path, filters=[("instrument", "in", [str(x) for x in instruments])]).to_pandas()
    labels["_day"] = pd.to_datetime(labels["_day"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    label_map = {
        (pd.Timestamp(d), str(ins)): float(label)
        for d, ins, label in zip(labels["_day"], labels["instrument"], labels["label"])
        if np.isfinite(float(label))
    }
    keep = []
    y = []
    for i, (d, ins) in enumerate(zip(idx["date"], idx["instrument"])):
        key = (pd.Timestamp(d), str(ins))
        label = label_map.get(key)
        if label is not None and np.isfinite(label):
            keep.append(i)
            y.append(label)
    if not keep:
        raise RuntimeError(f"empty train shard {shard_id} for {len(instruments)} instruments")

    keep_arr = np.asarray(keep, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.float32)
    idx = idx.iloc[keep_arr].reset_index(drop=True)
    x5 = x5[keep_arr].astype(np.float32, copy=False)
    x1 = x1[keep_arr].astype(np.float32, copy=False)

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    prefix = cache / f"train_shard_{shard_id:04d}"
    np.save(str(prefix) + "_x5.npy", x5)
    np.save(str(prefix) + "_x1.npy", x1)
    np.save(str(prefix) + "_y.npy", y_arr)
    idx.to_parquet(str(prefix) + "_idx.parquet", index=False)
    return {
        "shard_id": int(shard_id),
        "n": int(len(idx)),
        "instruments": [str(x) for x in instruments],
        "x5": str(prefix) + "_x5.npy",
        "x1": str(prefix) + "_x1.npy",
        "y": str(prefix) + "_y.npy",
        "idx": str(prefix) + "_idx.parquet",
        "elapsed": round(time.time() - t0, 2),
    }


def build_or_load_train_shards(
    table1m: str,
    train_start: str,
    train_end: str,
    instruments: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(SHARD_CACHE_DIR)
    manifest_path = _shard_manifest_path(cache_dir)
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        _log_info("train shard manifest loaded", path=str(manifest_path), shards=len(manifest.get("shards", [])))
    else:
        t0 = time.time()
        cache_dir.mkdir(parents=True, exist_ok=True)
        instruments = [str(x) for x in instruments]
        label_path = build_or_load_train_labels(table1m, train_start, train_end, instruments, cache_dir)
        chunks = _instrument_chunks(instruments, SHARD_INSTRUMENTS_PER_TASK)
        args = [(i, chunk, table1m, train_start, train_end, str(cache_dir), str(label_path)) for i, chunk in enumerate(chunks)]
        workers = min(max(1, SHARD_WORKERS), len(args))
        if workers == 1:
            shards = [_build_train_shard_worker(arg) for arg in args]
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                shards = pool.map(_build_train_shard_worker, args, chunksize=1)
        manifest = {
            "table1m": table1m,
            "train_start": train_start,
            "train_end": train_end,
            "feature_cols": FEATURE_COLS,
            "label_mode": LABEL_MODE,
            "shard_workers": workers,
            "shard_instruments_per_task": SHARD_INSTRUMENTS_PER_TASK,
            "shards": shards,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        _log_info(
            "train shards built",
            shards=len(shards),
            rows=sum(s["n"] for s in shards),
            workers=workers,
            elapsed=round(time.time() - t0, 2),
        )

    shards = manifest.get("shards", [])
    if not shards:
        raise RuntimeError(f"no train shards found in {manifest_path}")
    t0 = time.time()
    x1 = np.concatenate([np.load(s["x1"], mmap_mode=None) for s in shards], axis=0).astype(np.float32, copy=False)
    y = np.concatenate([np.load(s["y"], mmap_mode=None) for s in shards], axis=0).astype(np.float32, copy=False)
    cached_label_mode = manifest.get("label_mode", "raw")
    if LABEL_MODE != cached_label_mode:
        if cached_label_mode != "raw" or LABEL_MODE not in {"cs_zscore", "cs_gaussian_rank", "cs_rank"}:
            raise ValueError(
                f"cannot transform cached labels from {cached_label_mode!r} to {LABEL_MODE!r}"
            )
        idx = pd.concat(
            [pd.read_parquet(s["idx"], columns=["date"]) for s in shards],
            ignore_index=True,
        )
        if len(idx) != len(y):
            raise RuntimeError(f"cached index/label mismatch: {len(idx)} != {len(y)}")
        labels = idx.rename(columns={"date": "_day"})
        labels["label"] = y
        y = transform_train_labels(labels, LABEL_MODE)["label"].to_numpy(np.float32)
        _log_info(
            "cached labels transformed by complete date cross-section",
            source_mode=cached_label_mode,
            target_mode=LABEL_MODE,
            rows=len(y),
            days=int(pd.to_datetime(labels["_day"]).nunique()),
        )
    _log_info("train shards assembled", rows=len(y), shards=len(shards), elapsed=round(time.time() - t0, 2))
    return x1, y


def load_cached_training_dates(expected_rows: int) -> np.ndarray:
    manifest_path = _shard_manifest_path(Path(SHARD_CACHE_DIR))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    idx = pd.concat(
        [pd.read_parquet(s["idx"], columns=["date"]) for s in manifest.get("shards", [])],
        ignore_index=True,
    )
    if len(idx) != expected_rows:
        raise RuntimeError(f"cached date/sample mismatch: {len(idx)} != {expected_rows}")
    return pd.to_datetime(idx["date"]).to_numpy(dtype="datetime64[ns]")


def transform_cached_labels(y: np.ndarray, mode: str) -> np.ndarray:
    """Transform assembled raw labels using the exact cached sample-date order."""
    manifest_path = _shard_manifest_path(Path(SHARD_CACHE_DIR))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    shards = manifest.get("shards", [])
    idx = pd.concat(
        [pd.read_parquet(s["idx"], columns=["date"]) for s in shards],
        ignore_index=True,
    )
    if len(idx) != len(y):
        raise RuntimeError(f"cached index/label mismatch: {len(idx)} != {len(y)}")
    labels = idx.rename(columns={"date": "_day"})
    labels["label"] = np.asarray(y, dtype=np.float32)
    transformed = transform_train_labels(labels, mode)["label"].to_numpy(np.float32)
    _log_info(
        "curriculum labels transformed by complete date cross-section",
        source_mode="raw",
        target_mode=mode,
        rows=len(transformed),
        days=int(pd.to_datetime(labels["_day"]).nunique()),
    )
    return transformed


def build_future_mean_aux_labels(y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a future-only multi-day supervision target in cached sample order."""
    if horizon < 2:
        raise ValueError("AUX_HORIZON must be at least 2")
    manifest_path = _shard_manifest_path(Path(SHARD_CACHE_DIR))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    shards = manifest.get("shards", [])
    idx = pd.concat(
        [pd.read_parquet(s["idx"], columns=["date", "instrument"]) for s in shards],
        ignore_index=True,
    )
    if len(idx) != len(y):
        raise RuntimeError(f"cached index/label mismatch: {len(idx)} != {len(y)}")
    frame = idx.copy()
    frame["_row"] = np.arange(len(frame), dtype=np.int64)
    frame["_label"] = np.asarray(y, dtype=np.float32)
    frame = frame.sort_values(["instrument", "date"], kind="mergesort")
    grouped = frame.groupby("instrument", sort=False)["_label"]
    future = [frame["_label"].to_numpy(np.float32)]
    for step in range(1, horizon):
        future.append(grouped.shift(-step).to_numpy(np.float32))
    stacked = np.stack(future, axis=1)
    valid = np.isfinite(stacked).all(axis=1)
    aux_sorted = np.zeros(len(frame), dtype=np.float32)
    aux_sorted[valid] = stacked[valid].mean(axis=1)
    main_std = float(np.std(stacked[valid, 0], dtype=np.float64))
    aux_std = float(np.std(aux_sorted[valid], dtype=np.float64))
    if not np.isfinite(aux_std) or aux_std <= 0:
        raise RuntimeError(f"invalid auxiliary-label std: {aux_std}")
    scale = main_std / aux_std
    aux_sorted[valid] *= scale
    rows = frame["_row"].to_numpy(np.int64)
    aux = np.zeros(len(frame), dtype=np.float32)
    mask = np.zeros(len(frame), dtype=np.float32)
    aux[rows] = aux_sorted
    mask[rows] = valid.astype(np.float32)
    _log_info(
        "future-mean auxiliary labels built",
        horizon=horizon,
        rows=len(aux),
        valid_rows=int(mask.sum()),
        scale=scale,
        main_std=main_std,
        aux_std_before_scale=aux_std,
    )
    return aux, mask


def build_1m_samples(
    table1m: str,
    sd: str,
    ed: str,
    mode: str,
    instruments: Iterable[str],
    expo_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Build one raw 1m window per stock-day without a derived slow branch."""
    t0 = time.time()
    buf = (pd.Timestamp(sd) - pd.Timedelta(days=BAR1M_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")
    df_raw = read_bar(table1m, buf, ed, instruments)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["_day"] = df_raw["date"].dt.normalize()
    df1 = _preprocess_bar_features(df_raw)

    label_map: dict[tuple[pd.Timestamp, str], float] = {}
    if mode == "train":
        if expo_df is None:
            raise ValueError("expo_df is required for train mode")
        fwd = _daily_fwd_ret(df_raw).rename(columns={"_day": "date"})
        labels = barra_neutralize(
            fwd[["date", "instrument", "fwd_ret"]], expo_df
        ).rename(columns={"date": "_day"})
        label_map = {
            (pd.Timestamp(d), str(ins)): float(label)
            for d, ins, label in zip(labels["_day"], labels["instrument"], labels["label"])
            if np.isfinite(float(label))
        }

    windows, _ = _make_windows_by_key(df1, BAR1M_SEQ_LEN, sd, ed)
    keys = set(windows)
    if mode == "train":
        keys &= set(label_map)
    ordered = sorted(keys)
    if not ordered:
        raise RuntimeError(f"build_1m_samples produced no samples for mode={mode}, range={sd}~{ed}")
    x1 = np.stack([windows[k] for k in ordered]).astype(np.float32)
    y = np.asarray([label_map[k] for k in ordered], np.float32) if mode == "train" else None
    idx = pd.DataFrame(ordered, columns=["date", "instrument"])
    _log_info("pure 1m samples built", mode=mode, n=len(idx), elapsed=round(time.time() - t0, 2))
    return x1, y, idx


def build_dataset(
    table: str,
    sd: str,
    ed: str,
    mode: str,
    instruments: Iterable[str],
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame, tuple[np.ndarray, np.ndarray] | None]:
    """Official adapter returning only the raw 1m model input."""
    x1, y, idx = build_1m_samples(table, sd, ed, mode, instruments)
    return x1, y, idx, stats


class InferenceBatchDataset(IterableDataset):
    """Yield model-ready batches while keeping only a few trading days live."""

    def __init__(self, table, sd, ed, instruments, trading_dates, batch_size):
        super().__init__()
        self.table = table
        self.sd = str(sd)
        self.ed = str(ed)
        self.instruments = tuple(str(x) for x in instruments)
        dates = pd.to_datetime(pd.Series(list(trading_dates))).dt.normalize()
        start = pd.Timestamp(self.sd).normalize()
        finish = pd.Timestamp(self.ed).normalize()
        self.trading_dates = tuple(
            pd.Timestamp(x)
            for x in sorted(dates[(dates >= start) & (dates <= finish)].unique())
        )
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not self.trading_dates:
            raise RuntimeError(f"no effective trading dates for {self.sd}~{self.ed}")

    def __iter__(self):
        start = pd.Timestamp(self.sd)
        finish = pd.Timestamp(self.ed)
        for offset in range(0, len(self.trading_dates), SUBMISSION_DATALOADER_DAYS):
            dates = self.trading_dates[offset : offset + SUBMISSION_DATALOADER_DAYS]
            chunk_start = max(start, dates[0])
            chunk_end = min(
                finish,
                dates[-1] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            )
            x_chunk, _, idx_chunk = build_1m_samples(
                self.table,
                chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                "infer",
                self.instruments,
            )
            x_tensor = torch.from_numpy(x_chunk)
            date_ns = pd.to_datetime(idx_chunk["date"]).astype("int64").to_numpy(copy=True)
            names = idx_chunk["instrument"].astype(str).tolist()
            for batch_offset in range(0, len(idx_chunk), self.batch_size):
                stop = min(batch_offset + self.batch_size, len(idx_chunk))
                yield (
                    x_tensor[batch_offset:stop],
                    torch.from_numpy(date_ns[batch_offset:stop]),
                    names[batch_offset:stop],
                )
            del x_tensor, x_chunk, date_ns, names, idx_chunk
            _release_inference_memory()


def _identity_collate(batch):
    return batch


def _release_inference_memory():
    gc.collect()
    try:
        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def build_inference_dataloader(table, sd, ed, instruments, trading_dates, batch_size=BATCH):
    dataset = InferenceBatchDataset(table, sd, ed, instruments, trading_dates, batch_size)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
        collate_fn=_identity_collate,
    )


def build_raw_return_holdout_1m(
    table1m: str,
    sd: str,
    ed: str,
    instruments: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build inference windows and raw next-day return labels for public-val proxy.

    Official scoring neutralizes against BARRA exposures, but local 2024 exposure
    is not available in the current SDK account. This function is only for local
    model selection on 2024 and must not be treated as the official metric.
    """
    x1, _, idx = build_1m_samples(table1m, sd, ed, "infer", instruments)
    df_bar = read_bar(table1m, sd, ed, instruments)
    df_bar["date"] = pd.to_datetime(df_bar["date"])
    df_bar["_day"] = df_bar["date"].dt.normalize()
    fwd = _daily_fwd_ret(df_bar)
    label_map = {
        (pd.Timestamp(d), str(ins)): float(ret)
        for ins, d, ret in zip(fwd["instrument"], fwd["_day"], fwd["fwd_ret"])
        if np.isfinite(float(ret))
    }
    keep = np.asarray(
        [(pd.Timestamp(d), str(ins)) in label_map for d, ins in zip(idx["date"], idx["instrument"])],
        dtype=bool,
    )
    if not keep.any():
        raise RuntimeError(f"no raw-return validation labels for {sd}~{ed}")
    idx2 = idx.loc[keep].reset_index(drop=True)
    y = np.asarray(
        [label_map[(pd.Timestamp(d), str(ins))] for d, ins in zip(idx2["date"], idx2["instrument"])],
        dtype=np.float32,
    )
    return x1[keep], y, idx2


# ============ Checkpoint ============
def save_model(ckpt: dict, path: str | os.PathLike = MODEL_PATH) -> str:
    tensors = {}
    for k, v in ckpt["state_dict"].items():
        t = v.detach().cpu()
        tensors[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data_b64": base64.b64encode(t.numpy().tobytes()).decode("ascii"),
        }
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict_b64"] = tensors
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return str(path)


def load_model(path: str | os.PathLike = MODEL_PATH, map_location: str | torch.device = "cpu") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    state_dict = {}
    if "state_dict_b64" in payload:
        for k, meta in payload["state_dict_b64"].items():
            dtype = getattr(torch, meta["dtype"])
            np_dtype = getattr(np, meta["dtype"])
            arr = np.frombuffer(base64.b64decode(meta["data_b64"].encode("ascii")), dtype=np_dtype).copy()
            tensor = torch.from_numpy(arr).to(dtype=dtype)
            state_dict[k] = tensor.reshape(meta["shape"]).to(map_location)
    else:
        for k, meta in payload["state_dict"].items():
            tensor = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
            state_dict[k] = tensor.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k not in ("state_dict", "state_dict_b64")}
    ckpt["state_dict"] = state_dict
    return ckpt


# ============ Train / predict ============
def train_and_save(datasources: dict, model_path: str | os.PathLike = MODEL_PATH) -> str:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log_info(
        "training start",
        device=str(device),
        tables=f"{datasources[BAR1M_KEY]} pure raw 1m branch",
        train=f"{TRAIN_START}~{TRAIN_END}",
        val=f"{VAL_START}~{VAL_END}",
        data_dir=_local_data_dir() or "(dai)",
    )

    train_instruments = pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS]
    table1m = datasources[BAR1M_KEY]
    if USE_SHARDED_TRAIN_BUILDER:
        x1tr, ytr = build_or_load_train_shards(
            table1m,
            TRAIN_START,
            TRAIN_END,
            train_instruments,
        )
    else:
        expo = read_exposure(TRAIN_START, TRAIN_END)
        x1tr, ytr, _ = build_1m_samples(
            table1m,
            TRAIN_START,
            TRAIN_END,
            "train",
            train_instruments,
            expo_df=expo,
        )
    if TARGET_CURRICULUM in {"gaussian_to_raw", "gaussian_to_raw_linear"}:
        if LABEL_MODE != "raw":
            raise ValueError("gaussian_to_raw curriculum requires BQ_LABEL_MODE=raw")
        if not (0 < CURRICULUM_SWITCH_EPOCH < EPOCHS):
            raise ValueError("BQ_CURRICULUM_SWITCH_EPOCH must be between 1 and EPOCHS-1")
        if TARGET_CURRICULUM == "gaussian_to_raw_linear" and not (
            0 < CURRICULUM_BLEND_EPOCHS < EPOCHS - CURRICULUM_SWITCH_EPOCH
        ):
            raise ValueError("BQ_CURRICULUM_BLEND_EPOCHS must leave at least one raw epoch")
        ytr_secondary = transform_cached_labels(ytr, "cs_gaussian_rank")
    elif TARGET_CURRICULUM == "none":
        ytr_secondary = ytr.copy()
    else:
        raise ValueError(f"unsupported BQ_TARGET_CURRICULUM={TARGET_CURRICULUM!r}")
    if TARGET_CURRICULUM != "none":
        raise ValueError("future-mean auxiliary training requires BQ_TARGET_CURRICULUM=none")
    ytr_aux, ytr_aux_mask = build_future_mean_aux_labels(ytr, AUX_HORIZON)
    lo, hi = np.percentile(ytr, [1, 99])
    ytr = np.clip(ytr, lo, hi).astype(np.float32)
    secondary_lo, secondary_hi = np.percentile(ytr_secondary, [1, 99])
    ytr_secondary = np.clip(ytr_secondary, secondary_lo, secondary_hi).astype(np.float32)
    aux_lo, aux_hi = np.percentile(ytr_aux[ytr_aux_mask > 0], [1, 99])
    ytr_aux = np.clip(ytr_aux, aux_lo, aux_hi).astype(np.float32)

    x1val, yval, idx_val = None, None, None
    if VAL_START and VAL_END:
        val_instruments = pool(VAL_START, VAL_END)[:MAX_TRAIN_INSTRUMENTS]
        if VAL_RAW_RET:
            x1val, yval, idx_val = build_raw_return_holdout_1m(
                table1m,
                VAL_START,
                VAL_END,
                val_instruments,
            )
        else:
            expo_val = read_exposure(VAL_START, VAL_END)
            x1val, yval, idx_val = build_1m_samples(
                table1m,
                VAL_START,
                VAL_END,
                "train",
                val_instruments,
                expo_df=expo_val,
            )

    base_model = DualScaleTimesFM(**MODEL_CFG).to(device)
    n_params = count_parameters(base_model)
    if not (100_000 <= n_params <= 100_000_000):
        raise RuntimeError(f"parameter budget violation: {n_params}")
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model)
        _log_info("model parameter count", n_params=n_params, gpu_count=torch.cuda.device_count())
    else:
        model = base_model
        _log_info("model parameter count", n_params=n_params, gpu_count=torch.cuda.device_count() if device.type == "cuda" else 0)

    train_dataset = TensorDataset(
        torch.from_numpy(x1tr),
        torch.from_numpy(ytr),
        torch.from_numpy(ytr_secondary),
        torch.from_numpy(ytr_aux),
        torch.from_numpy(ytr_aux_mask),
    )
    day_batch_sampler = None
    if LISTWISE_WEIGHT > 0:
        if not USE_SHARDED_TRAIN_BUILDER:
            raise ValueError("daily ListMLE requires BQ_USE_SHARDED_TRAIN_BUILDER=1")
        train_dates = load_cached_training_dates(len(ytr))
        day_batch_sampler = DayBatchSampler(train_dates, TRAIN_BATCH, SEED)
        loader = DataLoader(
            train_dataset,
            batch_sampler=day_batch_sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        _log_info(
            "day-stratified listwise batches configured",
            days=int(np.unique(train_dates.astype("datetime64[D]")).size),
            batches=len(day_batch_sampler),
            listwise_weight=LISTWISE_WEIGHT,
            temperature=LISTWISE_TEMPERATURE,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if LOSS_MODE == "smooth_l1":
        loss_fn = nn.SmoothL1Loss(beta=0.01)
    elif LOSS_MODE == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"unsupported BQ_LOSS_MODE={LOSS_MODE!r}")
    scheduler_interval = "epoch"
    if SCHED_MODE == "epoch_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(EPOCHS, 1),
            eta_min=min(LR_MIN, LR),
        )
    elif SCHED_MODE in {"step_warmup_cosine", "step_warmup_linear", "step_wsd"}:
        scheduler_interval = "step"
        total_steps = max(EPOCHS * len(loader), 1)
        warmup_steps = max(int(round(total_steps * WARMUP_FRAC)), 1)
        min_ratio = min(LR_MIN, LR) / LR

        def lr_multiplier(step):
            if step < warmup_steps:
                return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
            if SCHED_MODE == "step_wsd":
                stable_end = int(round(total_steps * 0.80))
                if step < stable_end:
                    return 1.0
                cooldown = (step - stable_end) / max(total_steps - stable_end, 1)
                return max(1.0 - cooldown, 0.0)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            decay = (
                0.5 * (1.0 + math.cos(math.pi * progress))
                if SCHED_MODE == "step_warmup_cosine"
                else 1.0 - progress
            )
            return min_ratio + (1.0 - min_ratio) * decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_multiplier)
        _log_info(
            "step warmup schedule configured",
            schedule=SCHED_MODE,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_frac=WARMUP_FRAC,
        )
    else:
        raise ValueError(f"unsupported BQ_SCHED_MODE={SCHED_MODE!r}")
    _log_info("regression loss configured", loss_mode=LOSS_MODE)
    model.train()
    ema_state = None
    if USE_EMA:
        ema_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        _log_info("ema enabled", decay=EMA_DECAY)
    best_epoch = 0
    best_state_dict = None
    best_val_loss = float("inf")
    last_progress_epoch = 0
    completed_epochs = 0
    for ep in range(EPOCHS):
        if day_batch_sampler is not None:
            day_batch_sampler.set_epoch(ep)
        effective_aux_weight = AUX_WEIGHT
        if AUX_ANNEAL_LAST_FRAC > 0:
            anneal_epochs = max(int(round(EPOCHS * AUX_ANNEAL_LAST_FRAC)), 1)
            anneal_start = EPOCHS - anneal_epochs
            if ep >= anneal_start:
                remaining = EPOCHS - ep - 1
                effective_aux_weight = AUX_WEIGHT * remaining / max(anneal_epochs - 1, 1)
        curriculum_phase = "single"
        curriculum_alpha = 1.0
        if TARGET_CURRICULUM in {"gaussian_to_raw", "gaussian_to_raw_linear"}:
            if ep < CURRICULUM_SWITCH_EPOCH:
                curriculum_phase = "gaussian"
                curriculum_alpha = 0.0
            elif TARGET_CURRICULUM == "gaussian_to_raw_linear" and ep < CURRICULUM_SWITCH_EPOCH + CURRICULUM_BLEND_EPOCHS:
                curriculum_phase = "blend"
                curriculum_alpha = (ep - CURRICULUM_SWITCH_EPOCH + 1) / CURRICULUM_BLEND_EPOCHS
            else:
                curriculum_phase = "raw"
                curriculum_alpha = 1.0
            if ep == CURRICULUM_SWITCH_EPOCH:
                raw_lr = LR * CURRICULUM_RAW_LR_SCALE
                if CURRICULUM_FREEZE_TRUNKS:
                    base_model.bar5m.requires_grad_(False)
                    base_model.bar1m.requires_grad_(False)
                    base_model.bar5m.eval()
                    base_model.bar1m.eval()
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if not trainable_params:
                    raise RuntimeError("curriculum switch left no trainable parameters")
                opt = torch.optim.AdamW(trainable_params, lr=raw_lr, weight_decay=WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=max(EPOCHS - ep, 1),
                    eta_min=min(LR_MIN, raw_lr),
                )
                last_progress_epoch = ep
                _log_info(
                    "curriculum phase switched",
                    epoch=ep + 1,
                    freeze_trunks=CURRICULUM_FREEZE_TRUNKS,
                    phase="raw",
                    lr=raw_lr,
                    trainable_params=sum(p.numel() for p in trainable_params),
                )
            if TARGET_CURRICULUM == "gaussian_to_raw_linear" and ep == CURRICULUM_SWITCH_EPOCH + CURRICULUM_BLEND_EPOCHS:
                last_progress_epoch = ep
        reset_selection = CURRICULUM_SELECT_POST_ONLY and (
            (TARGET_CURRICULUM == "gaussian_to_raw" and ep == CURRICULUM_SWITCH_EPOCH)
            or (
                TARGET_CURRICULUM == "gaussian_to_raw_linear"
                and ep == CURRICULUM_SWITCH_EPOCH + CURRICULUM_BLEND_EPOCHS
            )
        )
        if reset_selection:
            best_epoch = 0
            best_state_dict = None
            best_val_loss = float("inf")
            last_progress_epoch = ep
            _log_info("checkpoint selector reset for post-curriculum phase", epoch=ep + 1)
        if CURRICULUM_FREEZE_TRUNKS and ep >= CURRICULUM_SWITCH_EPOCH:
            # Validation restores model.train(); keep frozen dropout disabled.
            base_model.bar5m.eval()
            base_model.bar1m.eval()
        t0, total, primary_total, aux_total, listwise_total, nb = time.time(), 0.0, 0.0, 0.0, 0.0, 0
        for x1b, yrawb, ysecondaryb, yauxb, yauxmaskb in loader:
            x1b = x1b.to(device, non_blocking=True)
            x1b = _apply_training_channel_mask(x1b, CHANNEL_MASK_PROB)
            yrawb = yrawb.to(device, non_blocking=True)
            ysecondaryb = ysecondaryb.to(device, non_blocking=True)
            yauxb = yauxb.to(device, non_blocking=True)
            yauxmaskb = yauxmaskb.to(device, non_blocking=True)
            if curriculum_phase == "gaussian":
                yb = ysecondaryb
            elif curriculum_phase == "blend":
                yb = ysecondaryb.lerp(yrawb, curriculum_alpha)
            else:
                yb = yrawb
            opt.zero_grad(set_to_none=True)
            pred, aux_pred = model(x1b, return_aux=True)
            primary_loss = loss_fn(pred, yb)
            listwise_loss = task_aligned_rank_loss(
                pred, yb, RANK_LOSS_KIND, LISTWISE_TEMPERATURE
            )
            if LOSS_MODE == "mse":
                aux_elements = (aux_pred - yauxb).square()
            else:
                aux_elements = F.smooth_l1_loss(aux_pred, yauxb, beta=0.01, reduction="none")
            aux_loss = (aux_elements * yauxmaskb).sum() / yauxmaskb.sum().clamp_min(1.0)
            loss = primary_loss + effective_aux_weight * aux_loss + LISTWISE_WEIGHT * listwise_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler_interval == "step":
                scheduler.step()
            if ema_state is not None:
                with torch.no_grad():
                    for key, value in base_model.state_dict().items():
                        ema_state[key].mul_(EMA_DECAY).add_(value.detach(), alpha=1.0 - EMA_DECAY)
            total += float(loss.item())
            primary_total += float(primary_loss.item())
            aux_total += float(aux_loss.item())
            listwise_total += float(listwise_loss.item())
            nb += 1
        train_loss = total / max(nb, 1)
        current_lr = opt.param_groups[0]["lr"]
        msg = (
            f"epoch {ep + 1}/{EPOCHS} loss={train_loss:.8f} "
            f"primary_loss={primary_total / max(nb, 1):.8f} "
            f"rank_loss={listwise_total / max(nb, 1):.8f} rank_kind={RANK_LOSS_KIND} "
            f"rank_weight={LISTWISE_WEIGHT:.6f} "
            f"aux_loss={aux_total / max(nb, 1):.8f} aux_weight={effective_aux_weight:.6f} "
            f"phase={curriculum_phase} target_alpha={curriculum_alpha:.6f} "
            f"lr={current_lr:.8g}"
        )
        val_improved = False
        if x1val is not None and yval is not None and idx_val is not None:
            if ema_state is not None:
                raw_state = copy.deepcopy(base_model.state_dict())
                base_model.load_state_dict(ema_state, strict=True)
            pred = _predict_array(model, x1val, device, TRAIN_BATCH)
            if ema_state is not None:
                base_model.load_state_dict(raw_state, strict=True)
            if LOSS_MODE == "mse":
                val_loss = float(np.mean(np.square(pred - yval)))
            else:
                val_loss = smooth_l1_numpy(pred, yval, beta=0.01)
            gap = val_loss - train_loss
            msg = f"{msg} ValLoss={val_loss:.8f} Gap={gap:+.8f}"
            if val_loss < best_val_loss - VAL_LOSS_MIN_DELTA:
                best_val_loss = val_loss
                val_improved = True
                best_state_dict = copy.deepcopy(ema_state if ema_state is not None else base_model.state_dict())
                best_epoch = ep + 1
            if SAVE_EPOCH_STATES:
                state_dir = HERE / "epoch_states"
                state_dir.mkdir(parents=True, exist_ok=True)
                source_state = ema_state if ema_state is not None else base_model.state_dict()
                torch.save(
                    {
                        "epoch": ep + 1,
                        "optimization": {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "gap": gap,
                            "lr": current_lr,
                        },
                        "official_metrics": None,
                        "state_dict": {
                            key: value.detach().cpu().clone()
                            for key, value in source_state.items()
                        },
                    },
                    state_dir / f"epoch_{ep + 1:02d}.pt",
                )
            _launch_async_official_eval(idx_val, pred, ep + 1)
        _log_info("epoch complete", epoch=ep + 1, loss=round(total / max(nb, 1), 8), elapsed=round(time.time() - t0, 2))
        print(msg, flush=True)
        if scheduler_interval == "epoch":
            scheduler.step()
        completed_epochs = ep + 1
        if val_improved:
            last_progress_epoch = ep + 1
        stop_epoch = CURRICULUM_SWITCH_EPOCH
        if TARGET_CURRICULUM == "gaussian_to_raw_linear":
            stop_epoch += CURRICULUM_BLEND_EPOCHS
        curriculum_can_stop = TARGET_CURRICULUM == "none" or ep >= stop_epoch
        if curriculum_can_stop and EARLY_STOP_PATIENCE > 0 and completed_epochs - last_progress_epoch >= EARLY_STOP_PATIENCE:
            print(
                f"early stopping epoch={completed_epochs} patience={EARLY_STOP_PATIENCE} "
                f"best_val_loss={best_val_loss:.8f}",
                flush=True,
            )
            break

    if best_epoch:
        print(
            f"provisional best by ValLoss epoch={best_epoch} val_loss={best_val_loss:.8f}; "
            "final checkpoint requires completed official v4 results",
            flush=True,
        )

    state_to_save = base_model.state_dict() if SAVE_LAST else (best_state_dict if best_state_dict is not None else base_model.state_dict())

    return save_model(
        {
            "state_dict": state_to_save,
            "model_cfg": MODEL_CFG,
            "feature_cols": FEATURE_COLS,
            "mean": [0.0] * len(FEATURE_COLS),
            "std": [1.0] * len(FEATURE_COLS),
            "bar5m_seq_len": BAR5M_SEQ_LEN,
            "bar1m_seq_len": BAR1M_SEQ_LEN,
            "bar5m_patch_len": BAR5M_PATCH_LEN,
            "bar1m_patch_len": BAR1M_PATCH_LEN,
            "seed": SEED,
            "n_params": n_params,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "val_start": VAL_START,
            "val_end": VAL_END,
            "best_epoch": best_epoch,
            "best_holdout_metrics": None,
            "best_select_metric": SELECT_METRIC,
            "select_min_ic": SELECT_MIN_IC,
            "best_select_score": None,
            "selection_status": "pending_official_v4",
            "async_official_eval": ASYNC_OFFICIAL_EVAL,
            "save_last": SAVE_LAST,
            "save_epoch_states": SAVE_EPOCH_STATES,
            "label_mode": LABEL_MODE,
            "loss_mode": LOSS_MODE,
            "listwise_weight": LISTWISE_WEIGHT,
            "listwise_temperature": LISTWISE_TEMPERATURE,
            "rank_loss_kind": RANK_LOSS_KIND,
            "channel_mask_probability": CHANNEL_MASK_PROB,
            "lr": LR,
            "lr_min": LR_MIN,
            "lr_schedule": SCHED_MODE,
            "warmup_frac": WARMUP_FRAC if SCHED_MODE == "step_warmup_cosine" else 0.0,
            "target_curriculum": TARGET_CURRICULUM,
            "curriculum_switch_epoch": CURRICULUM_SWITCH_EPOCH,
            "curriculum_raw_lr_scale": CURRICULUM_RAW_LR_SCALE,
            "curriculum_blend_epochs": CURRICULUM_BLEND_EPOCHS,
            "curriculum_select_post_only": CURRICULUM_SELECT_POST_ONLY,
            "curriculum_freeze_trunks": CURRICULUM_FREEZE_TRUNKS,
            "use_ema": USE_EMA,
            "ema_decay": EMA_DECAY if USE_EMA else None,
            "aux_horizon": AUX_HORIZON,
            "aux_weight": AUX_WEIGHT,
            "aux_anneal_last_frac": AUX_ANNEAL_LAST_FRAC,
            "best_val_loss": best_val_loss,
            "completed_epochs": completed_epochs,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        model_path,
    )


def main(datasources: dict, start_date: str, end_date: str) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and SUBMISSION_TORCH_THREADS > 0:
        torch.set_num_threads(SUBMISSION_TORCH_THREADS)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found; run train_and_save(...) first")

    ckpt = load_model(MODEL_PATH, map_location=device)
    model = DualScaleTimesFM(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _log_info("model loaded", path=str(MODEL_PATH), device=str(device), n_params=ckpt.get("n_params"))
    del ckpt
    _release_inference_memory()

    universe = read_instruments(start_date, end_date)
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["instrument"] = universe["instrument"].astype(str)
    universe = universe.drop_duplicates(["date", "instrument"])
    loader = build_inference_dataloader(
        datasources[BAR1M_KEY],
        start_date,
        end_date,
        universe["instrument"].unique(),
        universe["date"].unique(),
        BATCH,
    )
    score_parts = []
    with torch.inference_mode():
        for xb, date_ns, instruments in loader:
            scores = model(xb.to(device)).cpu().numpy().astype(np.float64)
            score_parts.append(
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(date_ns.numpy()),
                        "instrument": instruments,
                        "score": scores,
                    }
                )
            )
    if not score_parts:
        raise RuntimeError(f"submission inference produced no scores for {start_date}~{end_date}")
    scores = pd.concat(score_parts, ignore_index=True)
    result = (
        scores.merge(universe, on=["date", "instrument"], how="inner")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])[["date", "instrument", "score"]]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    _log_info("scores built", rows=len(result), days=result["date"].nunique(), instruments=result["instrument"].nunique())
    return result
