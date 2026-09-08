# -*- coding: utf-8 -*-
"""BigQuant E2E V38: tail-locked body ranker.

This file is intentionally self-contained for BigQuant Notebook submission.

Public inference entry:
    main(datasources, start_date, end_date) -> DataFrame[date, instrument, score]

Training entries:
    train_development_and_save(datasources) -> 2019-2023 development checkpoint
    train_and_save(datasources) -> 2019-2024 submission checkpoint

V38 first trains the proven V37 posterior-guided dynamic-factor anchor for six
epochs on V37's original fourteen-epoch learning-rate time axis. It then freezes
the complete anchor and trains a lightweight residual body-ranking head on the
anchor-defined middle 80 percent of each daily cross-section. Public inference
keeps the standardized anchor values in the bottom and top deciles and maps the
learned body ordering strictly between the two anchor-tail boundaries. This is
one model trained from scratch; no V24 weight or second-model inference is used.
"""
from __future__ import annotations

import base64
import json
import hashlib
import math
import os
import random
import shutil
import time
import gc
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "v38_tail_locked_body_model.json")
CACHE_DIR = os.environ.get(
    "BIGQUANT_V38_CACHE_DIR",
    os.environ.get("BIGQUANT_V24_CACHE_DIR", os.path.join(HERE, ".v24_cache")),
)
CHECKPOINT_DIR = os.environ.get(
    "BIGQUANT_V38_CHECKPOINT_DIR", os.path.join(HERE, "v38_checkpoints")
)
# V38 uses exactly the same raw fields, windows, and preprocessing as V24/V37.
CACHE_SCHEMA = "bigquant_v24_compact_daily_patch_mmap_v1"
CHECKPOINT_STATE_FORMAT = "torch_state_zlib_base64_v1"
SUBMISSION_FILE_LIMIT_BYTES = 50_000_000

# Submission training dates are intentionally hardcoded, following the official template.
TRAIN_START = "2019-01-01 00:00:00"
TRAIN_END = "2024-12-31 23:59:59"
DEV_TRAIN_START = "2019-01-01 00:00:00"
DEV_TRAIN_END = "2023-12-31 23:59:59"
DEV_VALID_START = "2024-01-01 00:00:00"
DEV_VALID_END = "2024-12-31 23:59:59"
ROLLING_FOLDS = (
    {
        "name": "fold_2019_2021_valid_2022",
        "train_start": "2019-01-01 00:00:00",
        "train_end": "2021-12-31 23:59:59",
        "valid_start": "2022-01-01 00:00:00",
        "valid_end": "2022-12-31 23:59:59",
    },
    {
        "name": "fold_2019_2022_valid_2023",
        "train_start": "2019-01-01 00:00:00",
        "train_end": "2022-12-31 23:59:59",
        "valid_start": "2023-01-01 00:00:00",
        "valid_end": "2023-12-31 23:59:59",
    },
)

TABLE_5M_KEYS = ("bar5m", "bar_5m", "e2e_bar5m")
TABLE_1M_KEYS = ("bar1m", "bar_1m", "e2e_bar1m")
DEFAULT_TABLE_5M = "bigalpha_2026_stock_bar5m"
DEFAULT_TABLE_1M = "bigalpha_2026_stock_bar1m"

FEATURE_FIELDS_1M = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "deal_number",
    "bid_price1",
    "ask_price1",
    "bid_volume1",
    "ask_volume1",
    "bid_num_orders1",
    "ask_num_orders1",
    "bid_price2",
    "ask_price2",
    "bid_volume2",
    "ask_volume2",
    "bid_num_orders2",
    "ask_num_orders2",
    "bid_price3",
    "ask_price3",
    "bid_volume3",
    "ask_volume3",
    "bid_num_orders3",
    "ask_num_orders3",
)
FEATURE_FIELDS_5M = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "adjust_factor",
)
FEATURE_FIELDS = FEATURE_FIELDS_1M + FEATURE_FIELDS_5M

PRICE_LOG_FIELDS_1M = (
    "open",
    "high",
    "low",
    "close",
    "bid_price1",
    "ask_price1",
    "bid_price2",
    "ask_price2",
    "bid_price3",
    "ask_price3",
)
ACTIVITY_LOG_FIELDS_1M = (
    "volume",
    "amount",
    "deal_number",
    "bid_volume1",
    "ask_volume1",
    "bid_num_orders1",
    "ask_num_orders1",
    "bid_volume2",
    "ask_volume2",
    "bid_num_orders2",
    "ask_num_orders2",
    "bid_volume3",
    "ask_volume3",
    "bid_num_orders3",
    "ask_num_orders3",
)
PRICE_LOG_FIELDS_5M = ("open", "high", "low", "close", "pre_close")
ADJUST_LOG_FIELDS_5M = ("adjust_factor",)

# 2 trading days x 48 five-minute intervals; every interval owns five 1m rows.
SEQ_LEN = 96
BARS_PER_DAY = 48
MINUTES_PER_BAR = 5
MINUTE_SEQ_LEN = SEQ_LEN * MINUTES_PER_BAR
INFER_BUFFER_DAYS = 20
INFER_INSTRUMENT_CHUNK = 250
INFER_CONTEXT_BATCH = 1024
INFER_STOCK_BATCH = 1000
TRAIN_INSTRUMENT_CHUNK = 100
TRAIN_PROGRESS_EVERY = 100
LABEL_FORWARD_BUFFER_DAYS = 10
ANCHOR_EPOCHS = 6
ANCHOR_SCHEDULE_EPOCHS = 14
BODY_EPOCHS = 4
EPOCHS = ANCHOR_EPOCHS + BODY_EPOCHS
DEV_EPOCHS = EPOCHS
WARMUP_EPOCHS = 1
KL_WARMUP_EPOCHS = 4
MAX_LR = 6e-4
MIN_LR = 5e-5
LR = MAX_LR
SEED = 20260731
SCORE_SIGN = 1.0
PRIOR_HUBER_WEIGHT = 0.65
PRIOR_CORRELATION_WEIGHT = 0.35
POSTERIOR_TEACHER_WEIGHT = 0.25
KL_MAX_WEIGHT = 0.03
EXPOSURE_ORTHOGONALITY_WEIGHT = 0.01
PORTFOLIO_DIVERSITY_WEIGHT = 0.002
PORTFOLIO_LOGIT_TEMPERATURE = 1.50
HUBER_BETA = 0.50
GAUSSIAN_TARGET_CLIP = 3.0
NUM_FACTORS = 8
MIN_DAILY_SAMPLES = 600
TAIL_FRACTION = 0.10
BODY_HUBER_WEIGHT = 0.60
BODY_CORRELATION_WEIGHT = 0.40
BODY_HUBER_BETA = 0.50
BODY_MAX_LR = 5e-4
BODY_MIN_LR = 5e-5
BODY_GRADIENT_ACCUMULATION_DAYS = 5
BODY_HIDDEN_DIM = 256

MODEL_CFG = {
    "n_feat_1m": len(FEATURE_FIELDS_1M),
    "n_feat_5m": len(FEATURE_FIELDS_5M),
    "seq_len": SEQ_LEN,
    "minutes_per_bar": MINUTES_PER_BAR,
    "trade_dim": 32,
    "book_dim": 32,
    "minute_dim": 96,
    "bar_dim": 128,
    "gru_hidden": 128,
    "gru_layers": 1,
    "stock_dim": 192,
    "factor_dim": 128,
    "num_factors": NUM_FACTORS,
    "body_hidden_dim": BODY_HIDDEN_DIM,
    "dropout": 0.10,
}

MIN_PARAMS = 100_000
MAX_PARAMS = 100_000_000


def _logger():
    try:
        import structlog

        return structlog.get_logger()
    except Exception:
        class SimpleLogger:
            def info(self, *args, **kwargs):
                print(*args, kwargs if kwargs else "")

            def warning(self, *args, **kwargs):
                print(*args, kwargs if kwargs else "")

        return SimpleLogger()


logger = _logger()
_COLUMN_CACHE: Dict[str, set] = {}


def _has_checkpoint_signature(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError, TypeError):
        return False
    required = {
        "state_dict",
        "model_cfg",
        "mean_1m",
        "std_1m",
        "mean_5m",
        "std_5m",
    }
    return isinstance(payload, dict) and required.issubset(payload)


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve the platform checkpoint while rejecting ambiguous packages."""
    requested = os.path.abspath(model_path or MODEL_PATH)
    if os.path.isfile(requested):
        return requested

    # Explicit local evaluation paths must remain strict. Automatic discovery is
    # only for platform main(), whose default filename may differ from the
    # uploaded development/final artifact name.
    if model_path is not None and requested != os.path.abspath(MODEL_PATH):
        raise FileNotFoundError(f"Explicit V38 checkpoint does not exist: {requested}")

    search_dirs = []
    for directory in (os.path.dirname(requested), HERE, os.getcwd()):
        absolute = os.path.abspath(directory)
        if absolute not in search_dirs and os.path.isdir(absolute):
            search_dirs.append(absolute)

    candidates = []
    visible_json = []
    for directory in search_dirs:
        for name in sorted(os.listdir(directory)):
            if not (name.startswith("v38_") and name.endswith(".json")):
                continue
            path = os.path.join(directory, name)
            visible_json.append(path)
            if name.endswith("_config.json"):
                continue
            if _has_checkpoint_signature(path):
                candidates.append(os.path.abspath(path))
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        logger.warning(
            "default V38 model filename missing; resolved unique checkpoint",
            requested=requested,
            resolved=candidates[0],
        )
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            "Ambiguous V38 submission: multiple valid checkpoints found: "
            + ", ".join(candidates)
        )
    raise FileNotFoundError(
        f"Missing V38 checkpoint {requested}; scanned JSON files: {visible_json}"
    )


class DayAwareOrderFlowEncoder(nn.Module):
    """Encode minute order flow without allowing convolutions across day boundaries."""

    def __init__(
        self,
        n_feat_1m: int,
        n_feat_5m: int,
        seq_len: int,
        minutes_per_bar: int = MINUTES_PER_BAR,
        trade_dim: int = 32,
        book_dim: int = 32,
        minute_dim: int = 96,
        bar_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if n_feat_1m != 25:
            raise ValueError(f"V37 expects 25 ordered 1m fields, got {n_feat_1m}")
        if seq_len != SEQ_LEN or minutes_per_bar != MINUTES_PER_BAR:
            raise ValueError("V37 training windows must contain two 48-bar days")
        self.n_feat_1m = int(n_feat_1m)
        self.n_feat_5m = int(n_feat_5m)
        self.trade_dim = int(trade_dim)
        self.book_dim = int(book_dim)
        self.minute_dim = int(minute_dim)
        self.bar_dim = int(bar_dim)

        self.raw_projection = nn.Conv1d(n_feat_1m, minute_dim, kernel_size=1)
        self.trade_projection = nn.Sequential(
            nn.Conv1d(7, trade_dim, kernel_size=1),
            nn.GELU(),
        )
        self.book_projection = nn.Sequential(
            nn.Conv1d(6, book_dim, kernel_size=1),
            nn.GELU(),
        )
        self.level_embedding = nn.Parameter(torch.empty(3, book_dim))
        self.book_gate = nn.Conv1d(book_dim + trade_dim, 1, kernel_size=1)
        self.minute_fusion = nn.Sequential(
            nn.Conv1d(
                minute_dim + trade_dim + book_dim,
                2 * minute_dim,
                kernel_size=1,
            ),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
        )
        self.minute_depthwise = nn.Conv1d(
            minute_dim,
            minute_dim,
            kernel_size=5,
            padding=2,
            groups=minute_dim,
            bias=False,
        )
        self.minute_pointwise = nn.Conv1d(minute_dim, minute_dim, kernel_size=1)
        self.minute_norm = nn.GroupNorm(1, minute_dim)
        self.downsample = nn.Sequential(
            nn.Conv1d(
                minute_dim,
                minute_dim,
                kernel_size=MINUTES_PER_BAR,
                stride=MINUTES_PER_BAR,
                groups=minute_dim,
                bias=False,
            ),
            nn.Conv1d(minute_dim, bar_dim, kernel_size=1),
            nn.GELU(),
        )
        self.macro_projection = nn.Sequential(
            nn.LayerNorm(n_feat_5m),
            nn.Linear(n_feat_5m, bar_dim),
            nn.GELU(),
        )
        self.bar_fusion = nn.Sequential(
            nn.Linear(2 * bar_dim, 2 * bar_dim),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(bar_dim)
        nn.init.normal_(self.level_embedding, std=0.02)

    def forward(self, x_1m: torch.Tensor, x_5m: torch.Tensor) -> torch.Tensor:
        if (
            x_1m.ndim != 4
            or x_1m.shape[2] != MINUTES_PER_BAR
            or x_1m.shape[1] % BARS_PER_DAY != 0
        ):
            raise ValueError(
                "expected whole-day 1m input [B, D*48, 5, F], "
                f"got {tuple(x_1m.shape)}"
            )
        if x_1m.shape[-1] != self.n_feat_1m:
            raise ValueError("V37 1m field count does not match the model")
        n_bars = int(x_1m.shape[1])
        n_days = n_bars // BARS_PER_DAY
        if x_5m.ndim != 3 or x_5m.shape[1:] != (
            n_bars,
            self.n_feat_5m,
        ):
            raise ValueError(
                f"expected 5m input [B, {n_bars}, {self.n_feat_5m}], "
                f"got {tuple(x_5m.shape)}"
            )

        batch = x_1m.shape[0]
        # [B, D, 48, 5, F] -> [DB, F, 240]. Each day is independent.
        day_minutes = (
            x_1m.reshape(
                batch,
                n_days,
                BARS_PER_DAY,
                MINUTES_PER_BAR,
                self.n_feat_1m,
            )
            .reshape(
                batch * n_days,
                BARS_PER_DAY * MINUTES_PER_BAR,
                self.n_feat_1m,
            )
        )
        channels = day_minutes.transpose(1, 2).contiguous()
        raw_state = self.raw_projection(channels)
        trade_state = self.trade_projection(channels[:, :7])

        book_channels = (
            day_minutes[:, :, 7:]
            .reshape(batch * n_days, BARS_PER_DAY * MINUTES_PER_BAR, 3, 6)
            .permute(0, 2, 3, 1)
            .reshape(batch * n_days * 3, 6, BARS_PER_DAY * MINUTES_PER_BAR)
        )
        book_state = self.book_projection(book_channels).reshape(
            batch * n_days, 3, self.book_dim, BARS_PER_DAY * MINUTES_PER_BAR
        )
        book_state = book_state + self.level_embedding.view(1, 3, -1, 1)
        trade_context = trade_state.unsqueeze(1).expand(-1, 3, -1, -1)
        gate_input = torch.cat([book_state, trade_context], dim=2).reshape(
            batch * n_days * 3,
            self.book_dim + self.trade_dim,
            BARS_PER_DAY * MINUTES_PER_BAR,
        )
        gate_logits = self.book_gate(gate_input).reshape(
            batch * n_days, 3, BARS_PER_DAY * MINUTES_PER_BAR
        )
        gate_weights = torch.softmax(gate_logits, dim=1)
        aggregate_book = (gate_weights.unsqueeze(2) * book_state).sum(dim=1)

        minute_state = self.minute_fusion(
            torch.cat([raw_state, trade_state, aggregate_book], dim=1)
        )
        local = self.minute_pointwise(self.minute_depthwise(minute_state))
        minute_state = self.minute_norm(minute_state + torch.nn.functional.gelu(local))
        micro_bars = self.downsample(minute_state).transpose(1, 2)
        if micro_bars.shape[1] != BARS_PER_DAY:
            raise RuntimeError("V37 minute downsampling did not produce 48 bars")

        macro = self.macro_projection(
            x_5m.reshape(
                batch, n_days, BARS_PER_DAY, self.n_feat_5m
            ).reshape(
                batch * n_days, BARS_PER_DAY, self.n_feat_5m
            )
        )
        fused = self.bar_fusion(torch.cat([micro_bars, macro], dim=-1))
        bars = self.output_norm(micro_bars + fused)
        return bars.reshape(batch, n_bars, self.bar_dim)


class DynamicFactorLayer(nn.Module):
    """Infer prior factors from inputs and posterior factors from training labels."""

    def __init__(
        self,
        stock_dim: int = 192,
        factor_dim: int = 128,
        num_factors: int = NUM_FACTORS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if num_factors < 2:
            raise ValueError("V37 requires at least two dynamic factors")
        self.stock_dim = int(stock_dim)
        self.factor_dim = int(factor_dim)
        self.num_factors = int(num_factors)
        self.stock_norm = nn.LayerNorm(stock_dim)
        self.factor_queries = nn.Parameter(torch.empty(num_factors, factor_dim))
        self.factor_key = nn.Linear(stock_dim, factor_dim, bias=False)
        self.factor_value = nn.Linear(stock_dim, factor_dim, bias=False)
        self.prior_distribution = nn.Sequential(
            nn.LayerNorm(factor_dim),
            nn.Linear(factor_dim, factor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(factor_dim, 2),
        )
        self.posterior_distribution = nn.Sequential(
            nn.LayerNorm(factor_dim + 2),
            nn.Linear(factor_dim + 2, factor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(factor_dim, 2),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(stock_dim),
            nn.Linear(stock_dim, stock_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(stock_dim // 2, 1),
        )
        self.beta_head = nn.Sequential(
            nn.LayerNorm(stock_dim),
            nn.Linear(stock_dim, stock_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(stock_dim, num_factors),
        )
        nn.init.orthogonal_(self.factor_queries)

    @staticmethod
    def _split_distribution(parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = parameters.unbind(dim=-1)
        return mean, log_std.clamp(-3.0, 1.0)

    @staticmethod
    def _diagonal_gaussian_kl(
        posterior_mean: torch.Tensor,
        posterior_log_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_std: torch.Tensor,
    ) -> torch.Tensor:
        posterior_var = torch.exp(2.0 * posterior_log_std)
        prior_var = torch.exp(2.0 * prior_log_std).clamp_min(1e-6)
        kl = (
            prior_log_std
            - posterior_log_std
            + (posterior_var + (posterior_mean - prior_mean).square())
            / (2.0 * prior_var)
            - 0.5
        )
        return kl.mean()

    def forward(
        self,
        stocks: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        return_training_outputs: bool = False,
    ):
        if stocks.ndim != 2 or stocks.shape[1] != self.stock_dim:
            raise ValueError(
                f"expected stock states [N, {self.stock_dim}], got {tuple(stocks.shape)}"
            )
        normalized = self.stock_norm(stocks)
        centered = normalized - normalized.mean(dim=0, keepdim=True)

        alpha = self.alpha_head(centered).squeeze(-1)
        alpha = alpha - alpha.mean()
        beta = self.beta_head(centered)
        beta = beta - beta.mean(dim=0, keepdim=True)
        beta = beta / beta.square().mean(dim=0, keepdim=True).add(1e-5).sqrt()

        keys = self.factor_key(normalized)
        values = self.factor_value(normalized)
        portfolio_logits = torch.einsum(
            "nd,kd->nk", keys, self.factor_queries
        ) / math.sqrt(self.factor_dim)
        portfolio_logits = portfolio_logits.float()
        portfolio_logits = portfolio_logits - portfolio_logits.mean(
            dim=0, keepdim=True
        )
        portfolio_logits = portfolio_logits / portfolio_logits.square().mean(
            dim=0, keepdim=True
        ).add(1e-4).sqrt()
        portfolio_logits = portfolio_logits / PORTFOLIO_LOGIT_TEMPERATURE
        prior_weights = torch.softmax(portfolio_logits, dim=0)
        market_states = torch.einsum(
            "nk,nd->kd", prior_weights.to(values.dtype), values
        )
        prior_mean, prior_log_std = self._split_distribution(
            self.prior_distribution(market_states)
        )
        factor_prior = torch.einsum("nk,k->n", beta, prior_mean)
        factor_prior = factor_prior / math.sqrt(self.num_factors)
        prior_score = alpha + factor_prior
        prior_score = prior_score - prior_score.mean()

        if not return_training_outputs:
            return prior_score
        if targets is None or target_mask is None:
            raise ValueError("V37 training requires targets and target_mask")
        if targets.shape != prior_score.shape or target_mask.shape != prior_score.shape:
            raise ValueError("V37 targets and mask must match the stock cross-section")
        target_mask = target_mask.bool()
        if int(target_mask.sum()) < 2:
            raise ValueError("V37 posterior requires at least two valid labels")

        masked_logits = portfolio_logits.masked_fill(
            ~target_mask.unsqueeze(-1),
            torch.finfo(portfolio_logits.dtype).min,
        )
        posterior_weights = torch.softmax(masked_logits, dim=0)
        safe_target = torch.where(
            target_mask, targets.float(), torch.zeros_like(targets).float()
        )
        portfolio_return = torch.einsum(
            "nk,n->k", posterior_weights, safe_target
        )
        portfolio_second = torch.einsum(
            "nk,n->k", posterior_weights, safe_target.square()
        ).clamp_min(1e-6).sqrt()
        posterior_input = torch.cat(
            [
                market_states.float(),
                portfolio_return[:, None],
                portfolio_second[:, None],
            ],
            dim=-1,
        )
        posterior_mean, posterior_log_std = self._split_distribution(
            self.posterior_distribution(posterior_input)
        )
        factor_posterior = torch.einsum("nk,k->n", beta, posterior_mean)
        factor_posterior = factor_posterior / math.sqrt(self.num_factors)
        posterior_score = alpha + factor_posterior
        posterior_score = posterior_score - posterior_score.mean()

        kl_loss = self._diagonal_gaussian_kl(
            posterior_mean,
            posterior_log_std,
            prior_mean,
            prior_log_std,
        )
        # CUDA eigvalsh has no bfloat16 implementation. Keep this small K x K
        # regularization/diagnostic block in true FP32 even under outer autocast.
        with torch.autocast(device_type=beta.device.type, enabled=False):
            beta_fp32 = beta.float()
            exposure_gram = beta_fp32.transpose(0, 1) @ beta_fp32
            exposure_gram = exposure_gram / float(max(beta.shape[0], 1))
            off_diagonal = ~torch.eye(
                self.num_factors, dtype=torch.bool, device=beta.device
            )
            exposure_orthogonality_loss = (
                exposure_gram[off_diagonal].square().mean()
            )

            market_weight = 1.0 / float(max(prior_weights.shape[0], 1))
            prior_weights_fp32 = prior_weights.float()
            centered_portfolios = prior_weights_fp32 - market_weight
            normalized_portfolios = torch.nn.functional.normalize(
                centered_portfolios.transpose(0, 1), dim=-1
            )
            portfolio_cosine = (
                normalized_portfolios @ normalized_portfolios.transpose(0, 1)
            )
            portfolio_diversity_loss = (
                portfolio_cosine[off_diagonal].square().mean()
            )
            portfolio_entropy = -(
                prior_weights_fp32.clamp_min(1e-12)
                * prior_weights_fp32.clamp_min(1e-12).log()
            ).sum(dim=0)
            eigenvalues = torch.linalg.eigvalsh(
                exposure_gram.detach()
            ).clamp_min(1e-8)
            eigen_probability = eigenvalues / eigenvalues.sum()
            beta_effective_rank = torch.exp(
                -(eigen_probability * eigen_probability.log()).sum()
            )
        diagnostics = {
            "posterior_score": posterior_score,
            "kl_loss": kl_loss,
            "exposure_orthogonality_loss": exposure_orthogonality_loss,
            "portfolio_diversity_loss": portfolio_diversity_loss,
            "portfolio_effective_stocks": portfolio_entropy.detach().exp().mean(),
            "portfolio_max_weight": prior_weights.detach().amax(dim=0).mean(),
            "portfolio_mean_abs_cosine": portfolio_cosine[off_diagonal].detach().abs().mean(),
            "beta_effective_rank": beta_effective_rank,
            "factor_score_ratio": (
                factor_prior.detach().float().std(unbiased=False)
                / prior_score.detach().float().std(unbiased=False).clamp_min(1e-6)
            ),
            "alpha_score_ratio": (
                alpha.detach().float().std(unbiased=False)
                / prior_score.detach().float().std(unbiased=False).clamp_min(1e-6)
            ),
            "prior_factor_std": prior_mean.detach().float().std(unbiased=False),
            "posterior_factor_std": posterior_mean.detach().float().std(unbiased=False),
        }
        return prior_score, diagnostics


class V38TailLockedBodyRanker(nn.Module):
    """V37 anchor plus a residual head dedicated to middle-80% ordering."""

    def __init__(
        self,
        n_feat_1m: int,
        n_feat_5m: int,
        seq_len: int,
        minutes_per_bar: int = MINUTES_PER_BAR,
        trade_dim: int = 32,
        book_dim: int = 32,
        minute_dim: int = 96,
        bar_dim: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        stock_dim: int = 192,
        factor_dim: int = 128,
        num_factors: int = NUM_FACTORS,
        body_hidden_dim: int = BODY_HIDDEN_DIM,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if minutes_per_bar != MINUTES_PER_BAR or seq_len != SEQ_LEN:
            raise ValueError("V37 requires two 48-bar trading days")
        self.seq_len = int(seq_len)
        self.minutes_per_bar = int(minutes_per_bar)
        self.bar_dim = int(bar_dim)
        self.global_dim = int(stock_dim)
        self.stock_dim = int(stock_dim)
        self.interval_encoder = DayAwareOrderFlowEncoder(
            n_feat_1m=n_feat_1m,
            n_feat_5m=n_feat_5m,
            seq_len=seq_len,
            trade_dim=trade_dim,
            book_dim=book_dim,
            minute_dim=minute_dim,
            bar_dim=bar_dim,
            dropout=dropout,
        )
        self.intraday_position = nn.Parameter(torch.zeros(1, BARS_PER_DAY, bar_dim))
        self.day_position = nn.Parameter(torch.zeros(1, 2, bar_dim))
        nn.init.normal_(self.intraday_position, std=0.02)
        nn.init.normal_(self.day_position, std=0.02)
        self.temporal_gru = nn.GRU(
            input_size=bar_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.stock_projection = nn.Sequential(
            nn.LayerNorm(3 * gru_hidden),
            nn.Linear(3 * gru_hidden, 2 * stock_dim),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.LayerNorm(stock_dim),
        )
        self.factor_layer = DynamicFactorLayer(
            stock_dim=stock_dim,
            factor_dim=factor_dim,
            num_factors=num_factors,
            dropout=dropout,
        )
        # Do not advance the RNG seen by the V37-compatible anchor training.
        # This keeps anchor initialization/dropout trajectories comparable to
        # the already-evaluated V37 epoch-6 run.
        with torch.random.fork_rng(devices=[]):
            self.body_head = nn.Sequential(
                nn.LayerNorm(stock_dim),
                nn.Linear(stock_dim, body_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(body_hidden_dim, body_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(body_hidden_dim // 2, 1),
            )
            # Stage B starts from the exact anchor ordering.
            nn.init.zeros_(self.body_head[-1].weight)
            nn.init.zeros_(self.body_head[-1].bias)

    def encode_bars(
        self, x_1m: torch.Tensor, x_5m: torch.Tensor
    ) -> torch.Tensor:
        return self.interval_encoder(x_1m, x_5m)

    def _temporal_position(self) -> torch.Tensor:
        return (
            self.day_position.unsqueeze(2)
            + self.intraday_position.unsqueeze(1)
        ).reshape(1, self.seq_len, self.bar_dim)

    def encode_temporal_from_bars(self, bars: torch.Tensor) -> torch.Tensor:
        if bars.ndim != 3 or bars.shape[1:] != (
            self.seq_len,
            self.bar_dim,
        ):
            raise ValueError(
                f"expected bar states [B, {self.seq_len}, {self.bar_dim}], "
                f"got {tuple(bars.shape)}"
            )
        hidden, last = self.temporal_gru(bars + self._temporal_position())
        final_state = last[-1]
        summary = torch.cat(
            [final_state, hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1
        )
        return self.stock_projection(summary)

    def encode_temporal(
        self, x_1m: torch.Tensor, x_5m: torch.Tensor
    ) -> torch.Tensor:
        if x_1m.ndim != 4 or x_1m.shape[1:3] != (
            self.seq_len,
            self.minutes_per_bar,
        ):
            raise ValueError(
                "expected 1m input "
                f"[B, {self.seq_len}, {self.minutes_per_bar}, F], "
                f"got {tuple(x_1m.shape)}"
            )
        if x_5m.ndim != 3 or x_5m.shape[1] != self.seq_len:
            raise ValueError(
                f"expected 5m input [B, {self.seq_len}, F], got {tuple(x_5m.shape)}"
            )
        return self.encode_temporal_from_bars(self.encode_bars(x_1m, x_5m))

    def forward_from_temporal(
        self,
        temporal: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        return_training_outputs: bool = False,
    ):
        return self.factor_layer(
            temporal,
            targets=targets,
            target_mask=target_mask,
            return_training_outputs=return_training_outputs,
        )

    def forward_components(
        self,
        x_1m: torch.Tensor,
        x_5m: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        return_training_outputs: bool = False,
    ):
        temporal = self.encode_temporal(x_1m, x_5m)
        return self.forward_from_temporal(
            temporal,
            targets=targets,
            target_mask=target_mask,
            return_training_outputs=return_training_outputs,
        )

    def body_score_from_temporal(
        self,
        temporal: torch.Tensor,
        anchor_score: torch.Tensor,
    ) -> torch.Tensor:
        if temporal.ndim != 2 or temporal.shape[1] != self.stock_dim:
            raise ValueError(
                f"expected temporal states [N, {self.stock_dim}], "
                f"got {tuple(temporal.shape)}"
            )
        if anchor_score.ndim != 1 or anchor_score.shape[0] != temporal.shape[0]:
            raise ValueError("anchor_score must have one value per stock")
        correction = self.body_head(temporal).squeeze(-1)
        correction = correction - correction.mean()
        body_score = anchor_score + correction
        return body_score - body_score.mean()

    def forward_scores_from_temporal(
        self,
        temporal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_score = self.forward_from_temporal(temporal)
        body_score = self.body_score_from_temporal(temporal, anchor_score)
        return anchor_score, body_score

    def forward(self, x_1m: torch.Tensor, x_5m: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x_1m, x_5m)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model_step(
    n_stocks: int = 1000,
    steps_per_epoch: int = 1213,
) -> Dict[str, object]:
    """Measure one exact full-cross-section V37 training step."""
    _seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V38TailLockedBodyRanker(**MODEL_CFG).to(device).train()
    x_1m = torch.randn(
        int(n_stocks),
        SEQ_LEN,
        MINUTES_PER_BAR,
        len(FEATURE_FIELDS_1M),
        device=device,
    )
    x_5m = torch.randn(
        int(n_stocks), SEQ_LEN, len(FEATURE_FIELDS_5M), device=device
    )
    rank_targets = torch.linspace(-1.0, 1.0, int(n_stocks), device=device)
    targets = _gaussianize_rank_targets(rank_targets)
    target_mask = torch.ones(int(n_stocks), dtype=torch.bool, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.time()
    with _autocast_context(device):
        score, diagnostics = model.forward_components(
            x_1m,
            x_5m,
            targets=targets,
            target_mask=target_mask,
            return_training_outputs=True,
        )
        losses = _single_day_loss(
            score,
            targets,
            target_mask,
            diagnostics,
            kl_weight=KL_MAX_WEIGHT,
        )
        loss = losses["loss"]
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_gib = None
    step_seconds = time.time() - started
    result = {
        "device": str(device),
        "n_stocks": int(n_stocks),
        "five_minute_patches": int(n_stocks) * SEQ_LEN,
        "temporal_steps": int(n_stocks) * SEQ_LEN,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach().cpu().item()),
        "prior_task_loss": float(losses["prior_task_loss"].cpu().item()),
        "prior_huber_loss": float(losses["prior_huber_loss"].cpu().item()),
        "prior_correlation": float(losses["prior_correlation"].cpu().item()),
        "posterior_task_loss": float(
            losses["posterior_task_loss"].cpu().item()
        ),
        "posterior_correlation": float(
            losses["posterior_correlation"].cpu().item()
        ),
        "kl_loss": float(losses["kl_loss"].cpu().item()),
        "portfolio_effective_stocks": float(
            diagnostics["portfolio_effective_stocks"].cpu().item()
        ),
        "portfolio_max_weight": float(
            diagnostics["portfolio_max_weight"].cpu().item()
        ),
        "portfolio_mean_abs_cosine": float(
            diagnostics["portfolio_mean_abs_cosine"].cpu().item()
        ),
        "beta_effective_rank": float(
            diagnostics["beta_effective_rank"].cpu().item()
        ),
        "step_seconds": round(step_seconds, 4),
        "epoch_minutes_lower_bound": round(
            step_seconds * int(steps_per_epoch) / 60.0, 2
        ),
        "peak_cuda_gib": None if peak_gib is None else round(peak_gib, 4),
    }
    del model, x_1m, x_5m, rank_targets, targets, target_mask
    del score, diagnostics, losses, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def benchmark_inference_step(n_stocks: int = 1000) -> Dict[str, object]:
    """Measure one full-day pass using submission inference precision."""
    _seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V38TailLockedBodyRanker(**MODEL_CFG).to(device).eval()
    x_1m = torch.randn(
        int(n_stocks),
        SEQ_LEN,
        MINUTES_PER_BAR,
        len(FEATURE_FIELDS_1M),
        device=device,
    )
    x_5m = torch.randn(
        int(n_stocks), SEQ_LEN, len(FEATURE_FIELDS_5M), device=device
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.time()
    with torch.inference_mode(), _autocast_context(device):
        temporal = model.encode_temporal(x_1m, x_5m)
        anchor, body = model.forward_scores_from_temporal(temporal)
    score = _value_interval_score(
        anchor.float().cpu().numpy(),
        body.float().cpu().numpy(),
        np.arange(int(n_stocks)).astype(str),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_gib = None
    result = {
        "device": str(device),
        "n_stocks": int(n_stocks),
        "five_minute_patches": int(n_stocks) * SEQ_LEN,
        "temporal_steps": int(n_stocks) * SEQ_LEN,
        "seconds": round(time.time() - started, 4),
        "peak_cuda_gib": None if peak_gib is None else round(peak_gib, 4),
        "score_std": float(np.std(score)),
    }
    del model, x_1m, x_5m, temporal, anchor, body, score
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def validate_model_invariants(
    n_stocks: int = 24,
    tolerance: float = 1e-5,
) -> Dict[str, object]:
    """Fast CPU checks for day isolation and cross-section equivariance."""
    if int(n_stocks) < 8:
        raise ValueError("n_stocks must be at least 8")
    _seed_everything(SEED)
    device = torch.device("cpu")
    model = V38TailLockedBodyRanker(**MODEL_CFG).to(device).eval()
    x_1m = torch.randn(
        int(n_stocks),
        SEQ_LEN,
        MINUTES_PER_BAR,
        len(FEATURE_FIELDS_1M),
    )
    x_5m = torch.randn(
        int(n_stocks), SEQ_LEN, len(FEATURE_FIELDS_5M)
    )
    mutated_1m = x_1m.clone()
    mutated_5m = x_5m.clone()
    mutated_1m[:, BARS_PER_DAY:] += 7.0
    mutated_5m[:, BARS_PER_DAY:] -= 5.0
    rank_target = torch.linspace(-1.0, 1.0, int(n_stocks))
    target = _gaussianize_rank_targets(rank_target)
    target_mask = torch.ones(int(n_stocks), dtype=torch.bool)

    with torch.inference_mode():
        bars = model.encode_bars(x_1m, x_5m)
        mutated_bars = model.encode_bars(mutated_1m, mutated_5m)
        daily_bars = model.encode_bars(
            x_1m[:, :BARS_PER_DAY], x_5m[:, :BARS_PER_DAY]
        )
        temporal = model.encode_temporal_from_bars(bars)
        prior_score = model.forward_from_temporal(temporal)
        initial_body_score = model.body_score_from_temporal(
            temporal, prior_score
        )
        training_prior, diagnostics = model.forward_from_temporal(
            temporal,
            targets=target,
            target_mask=target_mask,
            return_training_outputs=True,
        )
        permutation = torch.randperm(int(n_stocks))
        permuted_score = model.forward_from_temporal(temporal[permutation])

    day_isolation_error = float(
        (bars[:, :BARS_PER_DAY] - mutated_bars[:, :BARS_PER_DAY])
        .abs()
        .max()
        .item()
    )
    daily_reuse_error = float(
        (bars[:, :BARS_PER_DAY] - daily_bars).abs().max().item()
    )
    permutation_error = float(
        (permuted_score - prior_score[permutation]).abs().max().item()
    )
    prior_path_error = float(
        (training_prior - prior_score).abs().max().item()
    )
    body_zero_start_error = float(
        (initial_body_score - prior_score).abs().max().item()
    )
    failures = {
        "day_isolation": day_isolation_error,
        "daily_reuse": daily_reuse_error,
        "stock_permutation_equivariance": permutation_error,
        "training_and_inference_prior_identity": prior_path_error,
        "body_head_zero_start": body_zero_start_error,
    }
    invalid = {
        name: value for name, value in failures.items() if value > tolerance
    }
    if invalid:
        raise AssertionError(f"V37 invariant checks failed: {invalid}")
    return {
        "status": "PASS",
        "trainable_parameters": count_trainable_parameters(model),
        "day_isolation_max_abs_error": day_isolation_error,
        "daily_reuse_max_abs_error": daily_reuse_error,
        "stock_permutation_max_abs_error": permutation_error,
        "prior_path_max_abs_error": prior_path_error,
        "body_zero_start_max_abs_error": body_zero_start_error,
        "Gaussian_target_mean": float(target.mean().item()),
        "Gaussian_target_std": float(target.std(unbiased=False).item()),
        "portfolio_effective_stocks": float(
            diagnostics["portfolio_effective_stocks"].item()
        ),
        "beta_effective_rank": float(
            diagnostics["beta_effective_rank"].item()
        ),
    }


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_tables(
    datasources: Optional[Dict[str, str]],
) -> Tuple[str, str]:
    if not datasources:
        return DEFAULT_TABLE_5M, DEFAULT_TABLE_1M
    table_5m = next(
        (datasources[key] for key in TABLE_5M_KEYS if key in datasources), None
    )
    table_1m = next(
        (datasources[key] for key in TABLE_1M_KEYS if key in datasources), None
    )
    if table_5m is None or table_1m is None:
        raise ValueError(
            "V37 requires both official bar5m and bar1m datasources; "
            f"received keys={sorted(datasources)}"
        )
    return str(table_5m), str(table_1m)


def _iter_chunks(items: Sequence[str], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _import_dai():
    try:
        import dai

        return dai
    except ImportError as exc:
        raise ImportError(
            "This baseline must be run inside BigQuant Notebook, where `dai` is available."
        ) from exc


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["instrument"] = out["instrument"].astype(str)
    return out.dropna(subset=["date", "instrument"])


def query_pool(start_date: str, end_date: str) -> pd.DataFrame:
    """Historical CSI 1000 constituent pool for the requested interval."""
    dai = _import_dai()
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    if pool.empty:
        return pd.DataFrame(columns=["date", "instrument"])
    pool = _normalize_date_column(pool)
    return pool.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"])


def _available_columns(table: str, start_date: str, end_date: str) -> set:
    if table in _COLUMN_CACHE:
        return _COLUMN_CACHE[table]
    dai = _import_dai()
    try:
        sample = dai.query(
            f"SELECT * FROM {table} LIMIT 1",
            filters={"date": [start_date, end_date]},
        ).df()
        columns = set(sample.columns)
    except Exception as exc:
        logger.warning("failed to inspect table columns, falling back to configured fields", error=str(exc))
        columns = {
            "date",
            "instrument",
            *FEATURE_FIELDS_1M,
            *FEATURE_FIELDS_5M,
        }
    _COLUMN_CACHE[table] = columns
    return columns


def _query_raw_bars(
    table: str,
    fields: Sequence[str],
    start_date: str,
    end_date: str,
    instruments: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    dai = _import_dai()
    columns = _available_columns(table, start_date, end_date)
    missing = sorted(set(fields) - columns)
    if missing:
        raise ValueError(f"official table {table} is missing V37 fields: {missing}")
    select_cols = ["date", "instrument", *fields]
    # Window assembly is keyed by timestamp slots and does not depend on row order.
    # Avoiding a database-wide sort lowers public-inference query cost.
    sql = f"SELECT {', '.join(select_cols)} FROM {table}"
    filters = {"date": [start_date, end_date]}
    if instruments:
        filters["instrument"] = list(instruments)
    raw = dai.query(sql, filters=filters, compression=True).df()
    if raw.empty:
        return pd.DataFrame(columns=["date", "instrument", *fields])
    return raw[["date", "instrument", *fields]]


def _query_raw_bars_with_memory_fallback(
    table: str,
    fields: Sequence[str],
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
) -> pd.DataFrame:
    """Split only memory-failing DAI queries; propagate all other errors."""
    instrument_list = list(instruments)
    try:
        return _query_raw_bars(
            table,
            fields,
            start_date,
            end_date,
            instrument_list,
        )
    except Exception as exc:
        message = str(exc).lower()
        is_memory_error = (
            "out of memory" in message
            or "failed to allocate" in message
            or "memory limit" in message
        )
        if not is_memory_error or len(instrument_list) <= 25:
            raise
        midpoint = len(instrument_list) // 2
        logger.warning(
            "v37 DAI query memory fallback",
            table=table,
            instruments=len(instrument_list),
            retry_left=midpoint,
            retry_right=len(instrument_list) - midpoint,
        )
        left = _query_raw_bars_with_memory_fallback(
            table,
            fields,
            start_date,
            end_date,
            instrument_list[:midpoint],
        )
        right = _query_raw_bars_with_memory_fallback(
            table,
            fields,
            start_date,
            end_date,
            instrument_list[midpoint:],
        )
        return pd.concat([left, right], ignore_index=True)


def _base_preprocess(
    raw: pd.DataFrame,
    fields: Sequence[str],
) -> pd.DataFrame:
    work = raw.copy()
    work["timestamp"] = pd.to_datetime(work["date"], errors="coerce")
    work["instrument"] = work["instrument"].astype(str)
    work = work.dropna(subset=["timestamp", "instrument"])
    work = work.drop_duplicates(["timestamp", "instrument"], keep="last")
    for field in fields:
        work[field] = pd.to_numeric(work[field], errors="coerce")
    return work.sort_values(["instrument", "timestamp"])


def _transform_price_fields(
    work: pd.DataFrame,
    fields: Sequence[str],
) -> None:
    for field in fields:
        values = work[field].to_numpy(dtype="float64", copy=True)
        invalid = ~np.isfinite(values) | (values <= 0.0)
        values[invalid] = np.nan
        work[field] = np.log(values)


def _transform_activity_fields(
    work: pd.DataFrame,
    fields: Sequence[str],
) -> None:
    for field in fields:
        values = work[field].to_numpy(dtype="float64", copy=True)
        invalid = ~np.isfinite(values) | (values < 0.0)
        values[invalid] = np.nan
        work[field] = np.log1p(values)


def _preprocess_1m_bars(raw: pd.DataFrame) -> pd.DataFrame:
    work = _base_preprocess(raw, FEATURE_FIELDS_1M)
    _transform_price_fields(work, PRICE_LOG_FIELDS_1M)
    _transform_activity_fields(work, ACTIVITY_LOG_FIELDS_1M)
    return work


def _preprocess_5m_bars(raw: pd.DataFrame) -> pd.DataFrame:
    work = _base_preprocess(raw, FEATURE_FIELDS_5M)
    raw_close = work["close"].to_numpy(dtype="float64", copy=True)
    raw_close[~np.isfinite(raw_close) | (raw_close <= 0.0)] = np.nan
    work["_raw_close_for_label"] = raw_close
    _transform_price_fields(work, PRICE_LOG_FIELDS_5M)
    for field in ADJUST_LOG_FIELDS_5M:
        values = work[field].to_numpy(dtype="float64", copy=False)
        valid = np.isfinite(values) & (values > 0.0)
        transformed = np.full_like(values, np.nan, dtype="float64")
        transformed[valid] = np.log(values[valid])
        work[field] = transformed
    return work


def _standardize_1m(
    x: np.ndarray, stats: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    mean, std = stats
    values = np.asarray(x, dtype="float32")
    mean_view = mean.reshape(1, 1, 1, -1)
    std_view = np.maximum(std, 1e-6).reshape(1, 1, 1, -1)
    values = np.where(np.isfinite(values), values, mean_view)
    return ((values - mean_view) / std_view).astype("float32")


def _standardize_5m(
    x: np.ndarray, stats: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    mean, std = stats
    values = np.asarray(x, dtype="float32")
    mean_view = mean.reshape(1, 1, -1)
    std_view = np.maximum(std, 1e-6).reshape(1, 1, -1)
    values = np.where(np.isfinite(values), values, mean_view)
    return ((values - mean_view) / std_view).astype("float32")


def _month_ranges(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end = cursor + pd.offsets.MonthEnd(1)
        yield max(start, cursor), min(end, month_end)
        cursor = cursor + pd.offsets.MonthBegin(1)


def _cache_path(cache_dir: str, month_start: pd.Timestamp) -> str:
    return os.path.join(cache_dir, month_start.strftime("%Y-%m"))


def _cache_file(path: str, name: str) -> str:
    return os.path.join(path, name)


V38_CACHE_FILES = (
    "day_1m.npy",
    "day_5m.npy",
    "window_index.npy",
    "returns.npy",
    "dates_ns.npy",
    "target_dates_ns.npy",
    "instruments.npy",
    "meta.json",
)


def _cache_is_valid(path: str, table_5m: str, table_1m: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        if not all(os.path.exists(_cache_file(path, name)) for name in V38_CACHE_FILES):
            return False
        with open(_cache_file(path, "meta.json"), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return (
            meta.get("schema") == CACHE_SCHEMA
            and meta.get("table_5m") == table_5m
            and meta.get("table_1m") == table_1m
            and meta.get("feature_fields_1m") == list(FEATURE_FIELDS_1M)
            and meta.get("feature_fields_5m") == list(FEATURE_FIELDS_5M)
            and int(meta.get("seq_len", -1)) == SEQ_LEN
            and int(meta.get("minutes_per_bar", -1)) == MINUTES_PER_BAR
        )
    except Exception:
        return False


def _minute_slot(timestamp: pd.Timestamp) -> int:
    minute = int(timestamp.hour) * 60 + int(timestamp.minute)
    if 9 * 60 + 31 <= minute <= 11 * 60 + 30:
        return minute - (9 * 60 + 31)
    if 13 * 60 + 1 <= minute <= 15 * 60:
        return 120 + minute - (13 * 60 + 1)
    return -1


def _five_minute_slot(timestamp: pd.Timestamp) -> int:
    minute = int(timestamp.hour) * 60 + int(timestamp.minute)
    morning_start = 9 * 60 + 35
    afternoon_start = 13 * 60 + 5
    if morning_start <= minute <= 11 * 60 + 30 and (minute - morning_start) % 5 == 0:
        return (minute - morning_start) // 5
    if afternoon_start <= minute <= 15 * 60 and (minute - afternoon_start) % 5 == 0:
        return 24 + (minute - afternoon_start) // 5
    return -1


def _vectorized_slots(
    timestamps: pd.Series, slots_per_day: int
) -> np.ndarray:
    hours = timestamps.dt.hour.to_numpy(dtype="int16")
    minutes = timestamps.dt.minute.to_numpy(dtype="int16")
    clock = hours * 60 + minutes
    slots = np.full(len(clock), -1, dtype="int16")
    if slots_per_day == 240:
        morning = (clock >= 9 * 60 + 31) & (clock <= 11 * 60 + 30)
        afternoon = (clock >= 13 * 60 + 1) & (clock <= 15 * 60)
        slots[morning] = clock[morning] - (9 * 60 + 31)
        slots[afternoon] = 120 + clock[afternoon] - (13 * 60 + 1)
        return slots
    morning_delta = clock - (9 * 60 + 35)
    afternoon_delta = clock - (13 * 60 + 5)
    morning = (
        (morning_delta >= 0)
        & (clock <= 11 * 60 + 30)
        & (morning_delta % 5 == 0)
    )
    afternoon = (
        (afternoon_delta >= 0)
        & (clock <= 15 * 60)
        & (afternoon_delta % 5 == 0)
    )
    slots[morning] = morning_delta[morning] // 5
    slots[afternoon] = 24 + afternoon_delta[afternoon] // 5
    return slots


def _daily_matrices(
    sub: pd.DataFrame,
    fields: Sequence[str],
    slots_per_day: int,
    slot_function,
) -> Dict[pd.Timestamp, np.ndarray]:
    result: Dict[pd.Timestamp, np.ndarray] = {}
    if sub.empty:
        return result
    work = sub.copy()
    work["_day"] = work["timestamp"].dt.normalize()
    work["_slot"] = _vectorized_slots(work["timestamp"], slots_per_day)
    work = work[work["_slot"] >= 0]
    for day, frame in work.groupby("_day", sort=False):
        matrix = np.full(
            (slots_per_day, len(fields)), np.nan, dtype="float32"
        )
        slots = frame["_slot"].to_numpy(dtype="int64")
        values = frame.loc[:, fields].to_numpy(dtype="float32")
        matrix[slots] = values
        result[pd.Timestamp(day)] = matrix
    return result


def _daily_raw_closes(sub: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    result: Dict[pd.Timestamp, float] = {}
    if sub.empty:
        return result
    work = sub.copy()
    work["_day"] = work["timestamp"].dt.normalize()
    work["_slot"] = _vectorized_slots(work["timestamp"], BARS_PER_DAY)
    work = work[work["_slot"] >= 0].sort_values("timestamp")
    for day, frame in work.groupby("_day", sort=False):
        values = frame["_raw_close_for_label"].to_numpy(dtype="float64")
        valid = values[np.isfinite(values) & (values > 0.0)]
        if len(valid):
            result[pd.Timestamp(day)] = float(valid[-1])
    return result


def _build_month_arrays(
    table_5m: str,
    table_1m: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    label_limit: Optional[pd.Timestamp],
    require_labels: bool,
    instrument_chunk_size: int = TRAIN_INSTRUMENT_CHUNK,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build unique stock-day tensors plus two integer references per sample."""
    target_start_s = month_start.strftime("%Y-%m-%d 00:00:00")
    target_end_s = month_end.strftime("%Y-%m-%d 23:59:59")
    target_pool = query_pool(target_start_s, target_end_s).sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
    if target_pool.empty:
        return (
            np.empty(
                (0, BARS_PER_DAY, MINUTES_PER_BAR, len(FEATURE_FIELDS_1M)),
                dtype="float32",
            ),
            np.empty((0, BARS_PER_DAY, len(FEATURE_FIELDS_5M)), dtype="float32"),
            np.empty((0, 2), dtype="int32"),
            np.empty(0, dtype="float32"),
            np.empty(0, dtype="float32"),
            np.empty(0, dtype="int64"),
            np.empty(0, dtype="U16"),
        )

    target_pool["date"] = pd.to_datetime(target_pool["date"]).dt.normalize()
    target_pool["instrument"] = target_pool["instrument"].astype(str)
    target_pool = target_pool.drop_duplicates(["date", "instrument"]).reset_index(
        drop=True
    )
    n_rows = len(target_pool)
    returns = np.full(n_rows, np.nan, dtype="float32")
    dates_ns = target_pool["date"].astype("int64").to_numpy(dtype="int64")
    target_dates_ns = np.full(n_rows, np.iinfo("int64").min, dtype="int64")
    instruments_array = target_pool["instrument"].to_numpy(dtype="U16")

    rows_by_instrument: Dict[str, List[Tuple[pd.Timestamp, int]]] = {}
    for row, item in target_pool[["date", "instrument"]].iterrows():
        rows_by_instrument.setdefault(str(item["instrument"]), []).append(
            (pd.Timestamp(item["date"]), int(row))
        )

    context_start = month_start - pd.Timedelta(days=INFER_BUFFER_DAYS)
    if require_labels:
        context_start = max(
            context_start, pd.Timestamp(TRAIN_START).normalize()
        )
    forward_end = month_end
    if require_labels:
        forward_end = month_end + pd.Timedelta(days=LABEL_FORWARD_BUFFER_DAYS)
        if label_limit is not None:
            forward_end = min(forward_end, label_limit.normalize())
    calendar = query_pool(
        context_start.strftime("%Y-%m-%d 00:00:00"),
        forward_end.strftime("%Y-%m-%d 23:59:59"),
    )
    trading_days = sorted(
        pd.to_datetime(calendar["date"]).dt.normalize().unique().tolist()
    )
    trading_days = [pd.Timestamp(value) for value in trading_days]
    previous_day = {
        trading_days[index]: trading_days[index - 1]
        for index in range(1, len(trading_days))
    }
    next_day = {
        trading_days[index]: trading_days[index + 1]
        for index in range(len(trading_days) - 1)
    }
    target_days = sorted(target_pool["date"].unique().tolist())
    target_days = [pd.Timestamp(value) for value in target_days]
    required_prior_days = [
        previous_day[day] for day in target_days if day in previous_day
    ]
    raw_context_start = (
        min(required_prior_days) if required_prior_days else context_start
    )
    if require_labels:
        raw_context_start = max(
            raw_context_start, pd.Timestamp(TRAIN_START).normalize()
        )

    context_index: Dict[Tuple[pd.Timestamp, str], int] = {}
    contexts_by_instrument: Dict[str, List[Tuple[pd.Timestamp, int]]] = {}
    window_index = np.full((n_rows, 2), -1, dtype="int32")
    for row, item in target_pool[["date", "instrument"]].iterrows():
        day = pd.Timestamp(item["date"])
        instrument = str(item["instrument"])
        prior = previous_day.get(day)
        if prior is None:
            continue
        for column, context_day in enumerate((prior, day)):
            key = (context_day, instrument)
            context_id = context_index.get(key)
            if context_id is None:
                context_id = len(context_index)
                context_index[key] = context_id
                contexts_by_instrument.setdefault(instrument, []).append(
                    (context_day, context_id)
                )
            window_index[int(row), column] = context_id

    n_contexts = len(context_index)
    day_1m = np.full(
        (
            n_contexts,
            BARS_PER_DAY,
            MINUTES_PER_BAR,
            len(FEATURE_FIELDS_1M),
        ),
        np.nan,
        dtype="float32",
    )
    day_5m = np.full(
        (n_contexts, BARS_PER_DAY, len(FEATURE_FIELDS_5M)),
        np.nan,
        dtype="float32",
    )

    instruments = sorted(rows_by_instrument)
    if int(instrument_chunk_size) <= 0:
        raise ValueError("instrument_chunk_size must be positive")
    chunks = list(_iter_chunks(instruments, int(instrument_chunk_size)))
    for chunk_id, instrument_chunk in enumerate(chunks, start=1):
        chunk_started = time.time()
        raw_1m = _query_raw_bars_with_memory_fallback(
            table_1m,
            FEATURE_FIELDS_1M,
            raw_context_start.strftime("%Y-%m-%d 00:00:00"),
            month_end.strftime("%Y-%m-%d 23:59:59"),
            instrument_chunk,
        )
        raw_5m = _query_raw_bars_with_memory_fallback(
            table_5m,
            FEATURE_FIELDS_5M,
            raw_context_start.strftime("%Y-%m-%d 00:00:00"),
            forward_end.strftime("%Y-%m-%d 23:59:59"),
            instrument_chunk,
        )
        work_1m = _preprocess_1m_bars(raw_1m)
        work_5m = _preprocess_5m_bars(raw_5m)
        one_groups = {
            str(instrument): frame
            for instrument, frame in work_1m.groupby("instrument", sort=False)
        }
        five_groups = {
            str(instrument): frame
            for instrument, frame in work_5m.groupby("instrument", sort=False)
        }
        for instrument in instrument_chunk:
            target_rows = rows_by_instrument.get(str(instrument), ())
            if not target_rows:
                continue
            one_daily = _daily_matrices(
                one_groups.get(str(instrument), pd.DataFrame()),
                FEATURE_FIELDS_1M,
                240,
                _minute_slot,
            )
            five_frame = five_groups.get(str(instrument), pd.DataFrame())
            five_daily = _daily_matrices(
                five_frame,
                FEATURE_FIELDS_5M,
                BARS_PER_DAY,
                _five_minute_slot,
            )
            closes = _daily_raw_closes(five_frame)
            for context_day, context_id in contexts_by_instrument.get(
                str(instrument), ()
            ):
                one_context = one_daily.get(context_day)
                five_context = five_daily.get(context_day)
                if one_context is not None:
                    day_1m[context_id] = one_context.reshape(
                        BARS_PER_DAY,
                        MINUTES_PER_BAR,
                        len(FEATURE_FIELDS_1M),
                    )
                if five_context is not None:
                    day_5m[context_id] = five_context
            for day, output_row in target_rows:
                prior = previous_day.get(day)
                if prior is None:
                    continue
                one_prior = one_daily.get(prior)
                one_current = one_daily.get(day)
                five_prior = five_daily.get(prior)
                five_current = five_daily.get(day)
                window_available = (
                    one_prior is not None
                    and one_current is not None
                    and five_prior is not None
                    and five_current is not None
                )
                if not require_labels or not window_available:
                    continue
                target_day = next_day.get(day)
                if target_day is None:
                    continue
                target_dates_ns[output_row] = int(target_day.value)
                c0 = closes.get(day)
                c1 = closes.get(target_day)
                if (
                    c0 is not None
                    and c1 is not None
                    and np.isfinite(c0)
                    and np.isfinite(c1)
                    and c0 > 1e-12
                ):
                    returns[output_row] = np.float32(c1 / c0 - 1.0)

        logger.info(
            "v37 month chunk built",
            month=month_start.strftime("%Y-%m"),
            chunk=chunk_id,
            chunks=len(chunks),
            instruments=len(instrument_chunk),
            elapsed=round(time.time() - chunk_started, 2),
        )
        del raw_1m, raw_5m, work_1m, work_5m, one_groups, five_groups
        gc.collect()

    return (
        day_1m,
        day_5m,
        window_index,
        returns,
        dates_ns,
        target_dates_ns,
        instruments_array,
    )


def _write_month_cache(
    path: str,
    table_5m: str,
    table_1m: str,
    month_start: pd.Timestamp,
    arrays: Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    import shutil

    temporary = f"{path}.tmp-{os.getpid()}"
    shutil.rmtree(temporary, ignore_errors=True)
    os.makedirs(temporary, exist_ok=True)
    (
        day_1m,
        day_5m,
        window_index,
        returns,
        dates_ns,
        target_dates_ns,
        instruments,
    ) = arrays
    np.save(_cache_file(temporary, "day_1m.npy"), day_1m, allow_pickle=False)
    np.save(_cache_file(temporary, "day_5m.npy"), day_5m, allow_pickle=False)
    np.save(
        _cache_file(temporary, "window_index.npy"),
        window_index,
        allow_pickle=False,
    )
    np.save(_cache_file(temporary, "returns.npy"), returns, allow_pickle=False)
    np.save(_cache_file(temporary, "dates_ns.npy"), dates_ns, allow_pickle=False)
    np.save(
        _cache_file(temporary, "target_dates_ns.npy"),
        target_dates_ns,
        allow_pickle=False,
    )
    np.save(
        _cache_file(temporary, "instruments.npy"), instruments, allow_pickle=False
    )
    meta = {
        "schema": CACHE_SCHEMA,
        "month": month_start.strftime("%Y-%m"),
        "table_5m": table_5m,
        "table_1m": table_1m,
        "feature_fields_1m": list(FEATURE_FIELDS_1M),
        "feature_fields_5m": list(FEATURE_FIELDS_5M),
        "seq_len": SEQ_LEN,
        "minutes_per_bar": MINUTES_PER_BAR,
        "samples": int(len(dates_ns)),
    }
    with open(_cache_file(temporary, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(temporary, path)


def prepare_monthly_cache(
    table_5m: str,
    table_1m: str,
    start_date: str,
    end_date: str,
    cache_dir: str = CACHE_DIR,
    force: bool = False,
) -> List[str]:
    """Build reusable uncompressed monthly raw-window caches."""
    os.makedirs(cache_dir, exist_ok=True)
    label_limit = pd.Timestamp(end_date).normalize()
    paths: List[str] = []
    for month_start, month_end in _month_ranges(start_date, end_date):
        path = _cache_path(cache_dir, month_start)
        paths.append(path)
        if not force and _cache_is_valid(path, table_5m, table_1m):
            logger.info(
                "v37 month cache reused",
                month=month_start.strftime("%Y-%m"),
                path=path,
            )
            continue
        started = time.time()
        arrays = _build_month_arrays(
            table_5m=table_5m,
            table_1m=table_1m,
            month_start=month_start,
            month_end=month_end,
            label_limit=label_limit,
            require_labels=True,
        )
        _write_month_cache(
            path, table_5m, table_1m, month_start, arrays
        )
        size_bytes = sum(
            os.path.getsize(_cache_file(path, name))
            for name in V38_CACHE_FILES
            if name != "meta.json"
        )
        logger.info(
            "v37 month cache saved",
            month=month_start.strftime("%Y-%m"),
            samples=len(arrays[3]),
            contexts=len(arrays[0]),
            context_to_naive_ratio=round(
                len(arrays[0]) / max(2 * len(arrays[3]), 1), 4
            ),
            size_gib=round(size_bytes / (1024**3), 3),
            elapsed=round(time.time() - started, 2),
        )
        del arrays
        gc.collect()
    return paths


def build_training_cache_month(
    datasources: Optional[Dict[str, str]] = None,
    month: str = "2019-01",
    cache_dir: str = CACHE_DIR,
    force: bool = False,
) -> Dict[str, object]:
    """Build one production-compatible month before committing to the full cache."""
    table_5m, table_1m = _pick_tables(datasources)
    month_start = pd.Timestamp(f"{month}-01").normalize()
    allowed_start = pd.Timestamp(TRAIN_START).normalize().replace(day=1)
    allowed_end = pd.Timestamp(TRAIN_END).normalize().replace(day=1)
    if not (allowed_start <= month_start <= allowed_end):
        raise ValueError(f"month must be inside the fixed training interval: {month}")
    month_end = min(
        month_start + pd.offsets.MonthEnd(1),
        pd.Timestamp(TRAIN_END).normalize(),
    )
    path = _cache_path(cache_dir, month_start)
    os.makedirs(cache_dir, exist_ok=True)
    started = time.time()
    reused = not force and _cache_is_valid(path, table_5m, table_1m)
    if not reused:
        arrays = _build_month_arrays(
            table_5m,
            table_1m,
            month_start,
            month_end,
            label_limit=pd.Timestamp(TRAIN_END).normalize(),
            require_labels=True,
        )
        _write_month_cache(path, table_5m, table_1m, month_start, arrays)
        del arrays
        gc.collect()
    dates_ns = np.load(_cache_file(path, "dates_ns.npy"), mmap_mode="r")
    returns = np.load(_cache_file(path, "returns.npy"), mmap_mode="r")
    day_1m = np.load(_cache_file(path, "day_1m.npy"), mmap_mode="r")
    size_bytes = sum(
        os.path.getsize(_cache_file(path, name))
        for name in V38_CACHE_FILES
        if name != "meta.json"
    )
    report = {
        "month": month,
        "path": path,
        "reused": reused,
        "samples": int(len(dates_ns)),
        "contexts": int(len(day_1m)),
        "context_to_naive_ratio": round(
            len(day_1m) / max(2 * len(dates_ns), 1), 4
        ),
        "trading_days": int(len(np.unique(dates_ns))),
        "label_coverage": float(np.isfinite(returns).mean()) if len(returns) else 0.0,
        "size_gib": round(size_bytes / (1024**3), 3),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _input_mask(
    dates_ns: np.ndarray, start_date: str, end_date: str
) -> np.ndarray:
    start_ns = int(pd.Timestamp(start_date).normalize().value)
    end_ns = int(pd.Timestamp(end_date).normalize().value)
    return (dates_ns >= start_ns) & (dates_ns <= end_ns)


def _label_mask(
    dates_ns: np.ndarray,
    target_dates_ns: np.ndarray,
    returns: np.ndarray,
    start_date: str,
    end_date: str,
) -> np.ndarray:
    end_ns = int(pd.Timestamp(end_date).normalize().value)
    return (
        _input_mask(dates_ns, start_date, end_date)
        & (target_dates_ns <= end_ns)
        & np.isfinite(returns)
    )


def _fit_frequency_stats(
    cache_paths: Sequence[str],
    filename: str,
    n_features: int,
    start_date: str,
    end_date: str,
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(n_features, dtype="float64")
    sum_squares = np.zeros(n_features, dtype="float64")
    counts = np.zeros(n_features, dtype="int64")
    for path in cache_paths:
        dates_ns = np.load(_cache_file(path, "dates_ns.npy"), mmap_mode="r")
        target_dates_ns = np.load(
            _cache_file(path, "target_dates_ns.npy"), mmap_mode="r"
        )
        returns = np.load(_cache_file(path, "returns.npy"), mmap_mode="r")
        mask = _label_mask(
            dates_ns, target_dates_ns, returns, start_date, end_date
        )
        window_index = np.load(
            _cache_file(path, "window_index.npy"), mmap_mode="r"
        )
        indices = np.unique(np.asarray(window_index[mask]).reshape(-1))
        indices = indices[indices >= 0]
        x = np.load(_cache_file(path, filename), mmap_mode="r")
        for start in range(0, len(indices), 64):
            block = np.asarray(x[indices[start : start + 64]], dtype="float32")
            flat = block.reshape(-1, n_features)
            finite = np.isfinite(flat)
            values = np.where(finite, flat, 0.0).astype("float64", copy=False)
            sums += values.sum(axis=0)
            sum_squares += np.square(values).sum(axis=0)
            counts += finite.sum(axis=0)
        del dates_ns, target_dates_ns, returns, window_index, x
        gc.collect()
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(
        sum_squares, counts, out=np.ones_like(sum_squares), where=counts > 0
    ) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-6))
    return mean.astype("float32"), std.astype("float32")


def _stats_path(cache_dir: str, start_date: str, end_date: str) -> str:
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d")
    return os.path.join(cache_dir, f"v24_stats_{start}_{end}.json")


def load_or_fit_streaming_stats(
    cache_paths: Sequence[str],
    cache_dir: str,
    table_5m: str,
    table_1m: str,
    start_date: str,
    end_date: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    path = _stats_path(cache_dir, start_date, end_date)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if (
                payload.get("schema") == CACHE_SCHEMA
                and payload.get("table_5m") == table_5m
                and payload.get("table_1m") == table_1m
                and payload.get("feature_fields_1m") == list(FEATURE_FIELDS_1M)
                and payload.get("feature_fields_5m") == list(FEATURE_FIELDS_5M)
            ):
                return (
                    (
                        np.asarray(payload["mean_1m"], dtype="float32"),
                        np.asarray(payload["std_1m"], dtype="float32"),
                    ),
                    (
                        np.asarray(payload["mean_5m"], dtype="float32"),
                        np.asarray(payload["std_5m"], dtype="float32"),
                    ),
                )
        except Exception as exc:
            logger.warning("invalid V37 statistics cache", error=repr(exc))
    stats_1m = _fit_frequency_stats(
        cache_paths,
        "day_1m.npy",
        len(FEATURE_FIELDS_1M),
        start_date,
        end_date,
    )
    stats_5m = _fit_frequency_stats(
        cache_paths,
        "day_5m.npy",
        len(FEATURE_FIELDS_5M),
        start_date,
        end_date,
    )
    payload = {
        "schema": CACHE_SCHEMA,
        "table_5m": table_5m,
        "table_1m": table_1m,
        "feature_fields_1m": list(FEATURE_FIELDS_1M),
        "feature_fields_5m": list(FEATURE_FIELDS_5M),
        "train_start": start_date,
        "train_end": end_date,
        "mean_1m": stats_1m[0].tolist(),
        "std_1m": stats_1m[1].tolist(),
        "mean_5m": stats_5m[0].tolist(),
        "std_5m": stats_5m[1].tolist(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return stats_1m, stats_5m


def _stats_signature(
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
) -> str:
    digest = hashlib.sha1()
    for array in (*stats_1m, *stats_5m):
        digest.update(np.asarray(array, dtype="float32").tobytes())
    return digest.hexdigest()[:12]


def _standardized_cache_file(
    path: str, signature: str, frequency: str
) -> str:
    return _cache_file(path, f"std_{signature}_{frequency}_f16.npy")


def _materialize_standardized_frequency(
    path: str,
    source_name: str,
    output_path: str,
    stats: Tuple[np.ndarray, np.ndarray],
    frequency: str,
) -> None:
    source = np.load(_cache_file(path, source_name), mmap_mode="r")
    if os.path.exists(output_path):
        try:
            existing = np.load(output_path, mmap_mode="r")
            valid = existing.shape == source.shape and existing.dtype == np.float16
            del existing
            if valid:
                return
        except Exception:
            pass
    temporary = f"{output_path}.tmp-{os.getpid()}"
    if os.path.exists(temporary):
        os.remove(temporary)
    target = np.lib.format.open_memmap(
        temporary, mode="w+", dtype="float16", shape=source.shape
    )
    standardize = _standardize_1m if frequency == "1m" else _standardize_5m
    for start in range(0, len(source), 128):
        stop = min(start + 128, len(source))
        target[start:stop] = standardize(source[start:stop], stats).astype(
            "float16"
        )
    target.flush()
    del target, source
    os.replace(temporary, output_path)


def materialize_standardized_cache(
    cache_paths: Sequence[str],
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
) -> Tuple[str, str]:
    """Write fill-and-standardize-once float16 tensors for all later epochs."""
    signature = _stats_signature(stats_1m, stats_5m)
    name_1m = os.path.basename(
        _standardized_cache_file(cache_paths[0], signature, "1m")
    )
    name_5m = os.path.basename(
        _standardized_cache_file(cache_paths[0], signature, "5m")
    )
    for position, path in enumerate(cache_paths, start=1):
        started = time.time()
        _materialize_standardized_frequency(
            path,
            "day_1m.npy",
            _cache_file(path, name_1m),
            stats_1m,
            "1m",
        )
        _materialize_standardized_frequency(
            path,
            "day_5m.npy",
            _cache_file(path, name_5m),
            stats_5m,
            "5m",
        )
        logger.info(
            "standardized cache ready",
            month=os.path.basename(path),
            month_index=position,
            months=len(cache_paths),
            elapsed=round(time.time() - started, 2),
        )
    return name_1m, name_5m


def build_training_cache(
    datasources: Optional[Dict[str, str]] = None,
    cache_dir: str = CACHE_DIR,
    force: bool = False,
    development: bool = False,
) -> Dict[str, object]:
    """Build all monthly mmap files and training-only normalization statistics."""
    table_5m, table_1m = _pick_tables(datasources)
    train_end = DEV_TRAIN_END if development else TRAIN_END
    cache_end = DEV_VALID_END if development else TRAIN_END
    started = time.time()
    paths = prepare_monthly_cache(
        table_5m,
        table_1m,
        TRAIN_START,
        cache_end,
        cache_dir=cache_dir,
        force=force,
    )
    stats_1m, stats_5m = load_or_fit_streaming_stats(
        paths,
        cache_dir,
        table_5m,
        table_1m,
        TRAIN_START,
        train_end,
    )
    materialize_standardized_cache(paths, stats_1m, stats_5m)
    report = {
        "cache_dir": cache_dir,
        "schema": CACHE_SCHEMA,
        "months": len(paths),
        "development": bool(development),
        "statistics_train_end": train_end,
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    logger.info("v37 training cache ready", **report)
    return report


def _load_training_month(
    path: str,
    start_date: str,
    end_date: str,
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    dates_map = np.load(_cache_file(path, "dates_ns.npy"), mmap_mode="r")
    mask = _input_mask(dates_map, start_date, end_date)
    indices = np.flatnonzero(mask)
    dates_ns = np.asarray(dates_map[indices], dtype="int64")
    returns_map = np.load(_cache_file(path, "returns.npy"), mmap_mode="r")
    target_map = np.load(
        _cache_file(path, "target_dates_ns.npy"), mmap_mode="r"
    )
    returns = np.asarray(returns_map[indices], dtype="float32")
    target_dates_ns = np.asarray(target_map[indices], dtype="int64")
    if len(indices) == 0:
        return (
            np.empty(
                (0, BARS_PER_DAY, MINUTES_PER_BAR, len(FEATURE_FIELDS_1M)),
                dtype="float16",
            ),
            np.empty(
                (0, BARS_PER_DAY, len(FEATURE_FIELDS_5M)), dtype="float16"
            ),
            np.empty((0, 2), dtype="int32"),
            np.empty(0, dtype="int64"),
            np.empty(0, dtype="float32"),
            dates_ns,
            returns,
        )
    end_ns = int(pd.Timestamp(end_date).normalize().value)
    returns[target_dates_ns > end_ns] = np.nan
    signature = _stats_signature(stats_1m, stats_5m)
    standardized_1m = _standardized_cache_file(path, signature, "1m")
    standardized_5m = _standardized_cache_file(path, signature, "5m")
    if not os.path.exists(standardized_1m) or not os.path.exists(standardized_5m):
        materialize_standardized_cache([path], stats_1m, stats_5m)
    x_1m = np.load(standardized_1m, mmap_mode="r")
    x_5m = np.load(standardized_5m, mmap_mode="r")
    window_map = np.load(_cache_file(path, "window_index.npy"), mmap_mode="r")
    window_index = np.asarray(window_map[indices], dtype="int32")
    targets, sample_weights = _cross_section_rank_targets(dates_ns, returns)
    return (
        x_1m,
        x_5m,
        window_index,
        targets,
        sample_weights,
        dates_ns.astype("int64"),
        returns,
    )


def _day_row_groups(dates_ns: np.ndarray) -> List[np.ndarray]:
    """Return daily row positions without repeatedly scanning the full month."""
    if len(dates_ns) == 0:
        return []
    order = np.argsort(dates_ns, kind="stable")
    ordered_dates = dates_ns[order]
    boundaries = np.flatnonzero(
        np.r_[True, ordered_dates[1:] != ordered_dates[:-1], True]
    )
    return [order[left:right] for left, right in zip(boundaries[:-1], boundaries[1:])]


def _assemble_compact_windows(
    day_1m: np.ndarray,
    day_5m: np.ndarray,
    window_index: np.ndarray,
    rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    references = np.asarray(window_index[rows], dtype="int64")
    valid = references >= 0
    safe = np.where(valid, references, 0)
    one = np.asarray(day_1m[safe.reshape(-1)], dtype="float16").reshape(
        len(rows),
        2,
        BARS_PER_DAY,
        MINUTES_PER_BAR,
        len(FEATURE_FIELDS_1M),
    )
    five = np.asarray(day_5m[safe.reshape(-1)], dtype="float16").reshape(
        len(rows), 2, BARS_PER_DAY, len(FEATURE_FIELDS_5M)
    )
    if not bool(valid.all()):
        one[~valid] = 0.0
        five[~valid] = 0.0
    return (
        np.ascontiguousarray(
            one.reshape(
                len(rows), SEQ_LEN, MINUTES_PER_BAR, len(FEATURE_FIELDS_1M)
            )
        ),
        np.ascontiguousarray(
            five.reshape(len(rows), SEQ_LEN, len(FEATURE_FIELDS_5M))
        ),
    )


def _prefetched_window_batches(
    day_1m: np.ndarray,
    day_5m: np.ndarray,
    window_index: np.ndarray,
    day_groups: Sequence[np.ndarray],
):
    """Overlap the next mmap gather with the current CUDA training step."""
    if not day_groups:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _assemble_compact_windows,
            day_1m,
            day_5m,
            window_index,
            day_groups[0],
        )
        for position, rows in enumerate(day_groups):
            one, five = future.result()
            if position + 1 < len(day_groups):
                future = executor.submit(
                    _assemble_compact_windows,
                    day_1m,
                    day_5m,
                    window_index,
                    day_groups[position + 1],
                )
            yield rows, one, five


def _to_training_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            pass
        return tensor.to(device, non_blocking=tensor.is_pinned())
    return tensor.float() if tensor.dtype == torch.float16 else tensor


def _cross_section_rank_targets(
    date_values: np.ndarray,
    returns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map next-day returns to deterministic daily percentiles in [-1, 1]."""
    frame = pd.DataFrame(
        {
            "date_key": np.asarray(date_values),
            "raw_return": np.asarray(returns, dtype="float64"),
        }
    )
    valid = np.isfinite(frame["raw_return"].to_numpy())
    targets = np.full(len(frame), np.nan, dtype="float32")
    sample_weights = np.zeros(len(frame), dtype="float32")
    if not valid.any():
        return targets, sample_weights

    valid_frame = frame.loc[valid].copy()
    grouped = valid_frame.groupby("date_key", sort=False)["raw_return"]
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    denominator = np.maximum(counts.to_numpy("float64") - 1.0, 1.0)
    percentile = (ranks.to_numpy("float64") - 1.0) / denominator
    rank_targets = np.clip(2.0 * percentile - 1.0, -1.0, 1.0)

    valid_positions = np.flatnonzero(valid)
    targets[valid_positions] = rank_targets.astype("float32")
    sample_weights[valid_positions] = 1.0
    return targets, sample_weights


def _state_dict_to_compact_payload(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    """Pack exact tensor bytes into one compressed JSON-safe payload."""
    chunks: List[bytes] = []
    tensor_meta: List[Dict[str, object]] = []
    offset = 0
    for name, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu().contiguous()
        # NumPy cannot represent bfloat16. Model parameters are float32, but
        # converting an unexpected bfloat16 buffer is safer than failing to save.
        if cpu_tensor.dtype == torch.bfloat16:
            cpu_tensor = cpu_tensor.float()
        array = np.ascontiguousarray(cpu_tensor.numpy())
        raw = array.tobytes(order="C")
        tensor_meta.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        chunks.append(raw)
        offset += len(raw)

    raw_state = b"".join(chunks)
    compressed = zlib.compress(raw_state, level=9)
    return {
        "__format__": CHECKPOINT_STATE_FORMAT,
        "encoding": "base64",
        "compression": "zlib",
        "tensor_count": len(tensor_meta),
        "raw_nbytes": len(raw_state),
        "compressed_nbytes": len(compressed),
        "raw_sha256": hashlib.sha256(raw_state).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "tensors": tensor_meta,
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _compact_payload_to_state_dict(
    compact: Dict[str, object],
    map_location: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    if compact.get("__format__") != CHECKPOINT_STATE_FORMAT:
        raise ValueError(
            f"unsupported V37 checkpoint state format: {compact.get('__format__')}"
        )
    if compact.get("encoding") != "base64" or compact.get("compression") != "zlib":
        raise ValueError("unsupported V37 checkpoint encoding")
    try:
        compressed = base64.b64decode(compact["data"], validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 V37 checkpoint payload") from exc
    expected_compressed_hash = str(compact.get("compressed_sha256", ""))
    if hashlib.sha256(compressed).hexdigest() != expected_compressed_hash:
        raise ValueError("V37 checkpoint compressed payload checksum mismatch")
    try:
        raw_state = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ValueError("invalid zlib V37 checkpoint payload") from exc
    if len(raw_state) != int(compact.get("raw_nbytes", -1)):
        raise ValueError("V37 checkpoint raw byte count mismatch")
    if hashlib.sha256(raw_state).hexdigest() != str(compact.get("raw_sha256", "")):
        raise ValueError("V37 checkpoint raw payload checksum mismatch")

    state_dict: Dict[str, torch.Tensor] = {}
    tensor_meta = compact.get("tensors", [])
    if len(tensor_meta) != int(compact.get("tensor_count", -1)):
        raise ValueError("V37 checkpoint tensor count mismatch")
    for meta in tensor_meta:
        offset = int(meta["offset"])
        nbytes = int(meta["nbytes"])
        shape = tuple(int(value) for value in meta["shape"])
        dtype = np.dtype(meta["dtype"])
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != expected_nbytes or offset < 0 or offset + nbytes > len(raw_state):
            raise ValueError(f"invalid tensor layout for {meta['name']}")
        array = np.frombuffer(
            raw_state,
            dtype=dtype,
            count=expected_nbytes // dtype.itemsize,
            offset=offset,
        ).reshape(shape)
        # Copy detaches the tensor from the temporary decompressed byte buffer.
        state_dict[str(meta["name"])] = torch.from_numpy(array.copy()).to(map_location)
    return state_dict


def save_checkpoint(payload: Dict[str, object], model_path: str = MODEL_PATH) -> str:
    state_dict = payload["state_dict"]
    serializable = {key: value for key, value in payload.items() if key != "state_dict"}
    serializable["checkpoint_state_format"] = CHECKPOINT_STATE_FORMAT
    serializable["state_dict"] = _state_dict_to_compact_payload(state_dict)
    parent = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(parent, exist_ok=True)
    temporary_path = f"{model_path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(
                serializable,
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return model_path


def load_checkpoint(
    model_path: str = MODEL_PATH,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, object]:
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    state_payload = payload["state_dict"]
    if state_payload.get("__format__") == CHECKPOINT_STATE_FORMAT:
        state_dict = _compact_payload_to_state_dict(state_payload, map_location)
    else:
        # Backward compatibility lets an already-trained >50 MB V37 checkpoint
        # be compacted without retraining.
        state_dict = {}
        for name, meta in state_payload.items():
            dtype = getattr(torch, meta.get("dtype", "float32"))
            tensor = torch.tensor(meta["data"], dtype=dtype).reshape(meta["shape"])
            state_dict[name] = tensor.to(map_location)
    payload["state_dict"] = state_dict
    return payload


def checkpoint_file_info(model_path: str = MODEL_PATH) -> Dict[str, object]:
    """Return upload-relevant checkpoint metadata without decoding weights."""
    absolute_path = os.path.abspath(model_path)
    size_bytes = os.path.getsize(absolute_path)
    with open(absolute_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_payload = payload.get("state_dict", {})
    state_format = (
        state_payload.get("__format__", "legacy_decimal_json")
        if isinstance(state_payload, dict)
        else "invalid"
    )
    return {
        "path": absolute_path,
        "size_bytes": size_bytes,
        "size_mib": round(size_bytes / (1024.0**2), 3),
        "state_format": state_format,
        "under_50mb_submission_limit": size_bytes < SUBMISSION_FILE_LIMIT_BYTES,
    }


def compact_checkpoint_file(
    source_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    """Convert a legacy V37 JSON checkpoint losslessly, without retraining.

    When output_path is omitted, the source is replaced atomically only after
    the compact staging file has passed a strict tensor equality check.
    """
    source_path = os.path.abspath(source_path)
    output_path = os.path.abspath(output_path or source_path)
    before_bytes = os.path.getsize(source_path)
    payload = load_checkpoint(source_path, map_location="cpu")
    original_state = payload["state_dict"]
    staging_path = f"{output_path}.compact.{os.getpid()}.json"
    try:
        save_checkpoint(payload, staging_path)
        restored = load_checkpoint(staging_path, map_location="cpu")
        restored_state = restored["state_dict"]
        if original_state.keys() != restored_state.keys():
            raise ValueError("V37 compact conversion changed state-dict keys")
        for name, original in original_state.items():
            candidate = restored_state[name]
            if original.shape != candidate.shape or not torch.equal(
                original.cpu(), candidate.cpu()
            ):
                raise ValueError(
                    f"V37 compact conversion changed tensor values: {name}"
                )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.replace(staging_path, output_path)
    finally:
        if os.path.exists(staging_path):
            os.remove(staging_path)

    report = checkpoint_file_info(output_path)
    report.update(
        {
            "source_size_mib": round(before_bytes / (1024.0**2), 3),
            "compression_ratio": round(
                int(report["size_bytes"]) / max(before_bytes, 1), 4
            ),
            "exact_tensor_roundtrip": True,
        }
    )
    logger.info("V37 checkpoint compacted", **report)
    return report


def _cross_section_standardize(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    centered = values - values.mean()
    scale = centered.square().mean().clamp_min(1e-6).sqrt()
    return centered / scale


def _gaussianize_rank_targets(rank_targets: torch.Tensor) -> torch.Tensor:
    """Map the cached [-1, 1] daily rank to finite normal scores."""
    rank_targets = rank_targets.float()
    n_samples = int(rank_targets.numel())
    if n_samples < 2:
        return torch.zeros_like(rank_targets)
    percentile = 0.5 * (rank_targets + 1.0)
    plotting_position = (
        percentile * float(n_samples - 1) + 0.5
    ) / float(n_samples)
    plotting_position = plotting_position.clamp(1e-5, 1.0 - 1e-5)
    normal_score = math.sqrt(2.0) * torch.erfinv(
        2.0 * plotting_position - 1.0
    )
    return normal_score.clamp(-GAUSSIAN_TARGET_CLIP, GAUSSIAN_TARGET_CLIP)


def _correlation_objective(
    score: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score_z = _cross_section_standardize(score)
    target_z = _cross_section_standardize(targets)
    correlation = (score_z * target_z).mean().clamp(-1.0, 1.0)
    return 1.0 - correlation, correlation


def _tail_count(n_samples: int, tail_fraction: float = TAIL_FRACTION) -> int:
    if not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must lie strictly between 0 and 0.5")
    count = max(1, int(math.floor(int(n_samples) * float(tail_fraction))))
    if 2 * count >= int(n_samples):
        raise ValueError("cross-section is too small for the tail split")
    return count


def _anchor_middle_mask(anchor_score: torch.Tensor) -> torch.Tensor:
    """Select the anchor-defined middle 80%; labels never define the tails."""
    if anchor_score.ndim != 1:
        raise ValueError("anchor_score must be one-dimensional")
    count = _tail_count(int(anchor_score.numel()))
    order = torch.argsort(anchor_score.detach().float())
    middle = torch.ones_like(anchor_score, dtype=torch.bool)
    middle[order[:count]] = False
    middle[order[-count:]] = False
    return middle


def _stable_order(values: np.ndarray, instruments: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    instruments = np.asarray(instruments).astype(str)
    return np.lexsort((instruments, values))


def _rank_slot_fallback(
    anchor: np.ndarray,
    body: np.ndarray,
    instruments: np.ndarray,
) -> np.ndarray:
    n_samples = len(anchor)
    count = _tail_count(n_samples)
    anchor_order = _stable_order(anchor, instruments)
    bottom = anchor_order[:count]
    middle = anchor_order[count:-count]
    top = anchor_order[-count:]
    middle_order = middle[_stable_order(body[middle], instruments[middle])]
    position = np.empty(n_samples, dtype="float64")
    position[bottom] = np.arange(count, dtype="float64")
    position[middle_order] = np.arange(count, n_samples - count, dtype="float64")
    position[top] = np.arange(n_samples - count, n_samples, dtype="float64")
    return 2.0 * position / float(max(n_samples - 1, 1)) - 1.0


def _value_interval_score(
    anchor: np.ndarray,
    body: np.ndarray,
    instruments: np.ndarray,
) -> np.ndarray:
    """Preserve anchor tails and map learned body ranks between their bounds."""
    anchor = np.asarray(anchor, dtype="float64")
    body = np.asarray(body, dtype="float64")
    instruments = np.asarray(instruments).astype(str)
    if len(anchor) != len(body) or len(anchor) != len(instruments):
        raise ValueError("anchor, body, and instruments must have equal length")
    if len(anchor) < 3 or not np.isfinite(anchor).all() or not np.isfinite(body).all():
        raise ValueError("V38 value-interval inputs must be finite")
    count = _tail_count(len(anchor))
    centered = anchor - float(anchor.mean())
    scale = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return _rank_slot_fallback(anchor, body, instruments)
    anchor_z = centered / scale
    anchor_order = _stable_order(anchor_z, instruments)
    bottom = anchor_order[:count]
    middle = anchor_order[count:-count]
    top = anchor_order[-count:]
    lower = float(anchor_z[bottom].max())
    upper = float(anchor_z[top].min())
    if not np.isfinite(lower + upper) or upper - lower <= 1e-8:
        return _rank_slot_fallback(anchor_z, body, instruments)
    middle_order = middle[_stable_order(body[middle], instruments[middle])]
    middle_values = np.linspace(
        lower, upper, len(middle_order) + 2, dtype="float64"
    )[1:-1]
    result = anchor_z.copy()
    result[middle_order] = middle_values
    return result


def _compose_value_interval_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "instrument", "anchor_score", "body_score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing V38 score columns: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    final_score = np.empty(len(result), dtype="float64")
    dates_ns = pd.to_datetime(result["date"]).to_numpy(dtype="datetime64[ns]").astype(
        "int64"
    )
    anchor_values = result["anchor_score"].to_numpy("float64")
    body_values = result["body_score"].to_numpy("float64")
    instrument_values = result["instrument"].astype(str).to_numpy()
    for rows in _day_row_groups(dates_ns):
        final_score[rows] = _value_interval_score(
            anchor_values[rows],
            body_values[rows],
            instrument_values[rows],
        )
    result["score"] = final_score
    return result[["date", "instrument", "score"]]


def _v38_kl_weight(update_index: int, steps_per_epoch: int) -> float:
    warmup_updates = max(int(KL_WARMUP_EPOCHS) * int(steps_per_epoch), 1)
    fraction = min(float(update_index + 1) / float(warmup_updates), 1.0)
    return float(KL_MAX_WEIGHT) * fraction


def _single_day_loss(
    prior_score: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    diagnostics: Dict[str, torch.Tensor],
    kl_weight: float,
) -> Dict[str, torch.Tensor]:
    target_mask = target_mask.bool()
    valid_target = targets.float()[target_mask]
    valid_prior = prior_score.float()[target_mask]
    valid_posterior = diagnostics["posterior_score"].float()[target_mask]

    prior_correlation_loss, prior_correlation = _correlation_objective(
        valid_prior, valid_target
    )
    posterior_correlation_loss, posterior_correlation = _correlation_objective(
        valid_posterior, valid_target
    )
    prior_huber_loss = nn.functional.smooth_l1_loss(
        valid_prior,
        valid_target,
        beta=HUBER_BETA,
        reduction="mean",
    )
    posterior_huber_loss = nn.functional.smooth_l1_loss(
        valid_posterior,
        valid_target,
        beta=HUBER_BETA,
        reduction="mean",
    )
    prior_task_loss = (
        PRIOR_HUBER_WEIGHT * prior_huber_loss
        + PRIOR_CORRELATION_WEIGHT * prior_correlation_loss
    )
    posterior_task_loss = (
        PRIOR_HUBER_WEIGHT * posterior_huber_loss
        + PRIOR_CORRELATION_WEIGHT * posterior_correlation_loss
    )
    kl_loss = diagnostics["kl_loss"].float()
    orthogonality_loss = diagnostics["exposure_orthogonality_loss"].float()
    diversity_loss = diagnostics["portfolio_diversity_loss"].float()
    loss = (
        prior_task_loss
        + POSTERIOR_TEACHER_WEIGHT * posterior_task_loss
        + float(kl_weight) * kl_loss
        + EXPOSURE_ORTHOGONALITY_WEIGHT * orthogonality_loss
        + PORTFOLIO_DIVERSITY_WEIGHT * diversity_loss
    )
    return {
        "loss": loss,
        "prior_task_loss": prior_task_loss.detach(),
        "prior_huber_loss": prior_huber_loss.detach(),
        "prior_correlation_loss": prior_correlation_loss.detach(),
        "prior_correlation": prior_correlation.detach(),
        "posterior_task_loss": posterior_task_loss.detach(),
        "posterior_huber_loss": posterior_huber_loss.detach(),
        "posterior_correlation_loss": posterior_correlation_loss.detach(),
        "posterior_correlation": posterior_correlation.detach(),
        "kl_loss": kl_loss.detach(),
        "exposure_orthogonality_loss": orthogonality_loss.detach(),
        "portfolio_diversity_loss": diversity_loss.detach(),
        "prior_score_std": valid_prior.detach().std(unbiased=False),
        "posterior_score_std": valid_posterior.detach().std(unbiased=False),
    }


class _FactorEpochAccumulator:
    def __init__(self) -> None:
        self.keys = (
            "portfolio_effective_stocks",
            "portfolio_max_weight",
            "portfolio_mean_abs_cosine",
            "beta_effective_rank",
            "factor_score_ratio",
            "alpha_score_ratio",
            "prior_factor_std",
            "posterior_factor_std",
        )
        self.totals: Optional[torch.Tensor] = None
        self.count = 0

    def update(self, diagnostics: Optional[Dict[str, torch.Tensor]]) -> None:
        if diagnostics is None:
            return
        values = torch.stack(
            [diagnostics[key].detach().float() for key in self.keys]
        )
        if self.totals is None:
            self.totals = values
        else:
            self.totals += values
        self.count += 1

    def summary(self) -> Dict[str, float]:
        if self.count == 0 or self.totals is None:
            return {key: float("nan") for key in self.keys}
        values = (self.totals / self.count).cpu().tolist()
        return dict(zip(self.keys, map(float, values)))


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _v38_learning_rate(
    update_index: int,
    total_updates: int,
    warmup_updates: int,
) -> float:
    """One-epoch linear warmup followed by update-level cosine decay."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    update_index = min(max(int(update_index), 0), total_updates - 1)
    warmup_updates = min(max(int(warmup_updates), 0), total_updates)
    if warmup_updates and update_index < warmup_updates:
        fraction = float(update_index + 1) / float(warmup_updates)
        return MIN_LR + fraction * (MAX_LR - MIN_LR)
    decay_updates = max(total_updates - warmup_updates, 1)
    decay_index = max(update_index - warmup_updates, 0)
    fraction = min(float(decay_index) / float(max(decay_updates - 1, 1)), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return MIN_LR + cosine * (MAX_LR - MIN_LR)


def _new_adamw(
    parameters: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> torch.optim.Optimizer:
    kwargs: Dict[str, object] = {
        "lr": MIN_LR,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def _eligible_training_steps(
    cache_paths: Sequence[str],
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    train_start: str,
    train_end: str,
) -> int:
    steps = 0
    for path in cache_paths:
        (
            day_1m,
            day_5m,
            window_index,
            targets,
            sample_weights,
            dates_ns,
            raw_returns,
        ) = _load_training_month(
            path, train_start, train_end, stats_1m, stats_5m
        )
        for rows in _day_row_groups(dates_ns):
            if int(np.isfinite(targets[rows]).sum()) >= MIN_DAILY_SAMPLES:
                steps += 1
        del (
            day_1m,
            day_5m,
            window_index,
            targets,
            sample_weights,
            dates_ns,
            raw_returns,
        )
    if steps <= 0:
        raise RuntimeError("V37 found no complete training cross-section")
    return steps


def _v38_checkpoint_payload(
    model: nn.Module,
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    train_start: str,
    train_end: str,
    history: Sequence[Dict[str, object]],
    completed_epoch: int,
    completed_stage: str,
) -> Dict[str, object]:
    return {
        "schema_version": "bigquant_v38_tail_locked_body_v1",
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_fields_1m": list(FEATURE_FIELDS_1M),
        "feature_fields_5m": list(FEATURE_FIELDS_5M),
        "mean_1m": stats_1m[0].tolist(),
        "std_1m": stats_1m[1].tolist(),
        "mean_5m": stats_5m[0].tolist(),
        "std_5m": stats_5m[1].tolist(),
        "score_sign": SCORE_SIGN,
        "train_start": train_start,
        "train_end": train_end,
        "completed_epoch": int(completed_epoch),
        "completed_stage": str(completed_stage),
        "trainable_parameters": count_trainable_parameters(model),
        "seed": SEED,
        "history": list(history),
    }


def _train_anchor_model(
    cache_paths: Sequence[str],
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    train_start: str,
    train_end: str,
    device: torch.device,
    run_name: str,
    epochs: int = ANCHOR_EPOCHS,
    schedule_epochs: int = ANCHOR_SCHEDULE_EPOCHS,
    checkpoint_dir: Optional[str] = None,
    initial_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    initial_global_epoch: int = 0,
    initial_history: Optional[Sequence[Dict[str, object]]] = None,
) -> Tuple[nn.Module, List[Dict[str, object]]]:
    """Train the V37-compatible anchor without shortening its LR horizon."""
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    model = V38TailLockedBodyRanker(**MODEL_CFG).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    n_params = count_trainable_parameters(model)
    if not MIN_PARAMS <= n_params <= MAX_PARAMS:
        raise ValueError(
            f"V37 parameter count {n_params} violates [{MIN_PARAMS}, {MAX_PARAMS}]"
        )
    for parameter in model.body_head.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = _new_adamw(trainable, device)
    history: List[Dict[str, object]] = list(initial_history or [])
    global_epoch = int(initial_global_epoch)
    rng = np.random.default_rng(SEED + global_epoch)

    steps_per_epoch = _eligible_training_steps(
        cache_paths,
        stats_1m,
        stats_5m,
        train_start,
        train_end,
    )
    if int(schedule_epochs) < int(epochs):
        raise ValueError("anchor schedule_epochs cannot be shorter than epochs")
    total_updates = int(schedule_epochs) * steps_per_epoch
    warmup_updates = min(WARMUP_EPOCHS, int(epochs)) * steps_per_epoch
    update_index = 0
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(
        "V38 anchor initialized",
        run=run_name,
        device=str(device),
        precision="bfloat16" if device.type == "cuda" else "float32",
        trainable_parameters=n_params,
        epochs=int(epochs),
        schedule_epochs=int(schedule_epochs),
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=min(WARMUP_EPOCHS, int(epochs)),
        max_lr=MAX_LR,
        min_lr=MIN_LR,
        kl_warmup_epochs=KL_WARMUP_EPOCHS,
        kl_max_weight=KL_MAX_WEIGHT,
    )

    metric_keys = (
        "loss",
        "prior_task_loss",
        "prior_huber_loss",
        "prior_correlation_loss",
        "prior_correlation",
        "posterior_task_loss",
        "posterior_huber_loss",
        "posterior_correlation_loss",
        "posterior_correlation",
        "kl_loss",
        "exposure_orthogonality_loss",
        "portfolio_diversity_loss",
        "prior_score_std",
        "posterior_score_std",
    )

    for local_epoch in range(int(epochs)):
        global_epoch += 1
        epoch_started = time.time()
        path_order = list(cache_paths)
        rng.shuffle(path_order)
        metric_totals = torch.zeros(
            len(metric_keys), dtype=torch.float32, device=device
        )
        factor_accumulator = _FactorEpochAccumulator()
        n_steps = 0
        n_samples = 0
        model.train()

        for path in path_order:
            (
                day_1m,
                day_5m,
                window_index,
                targets,
                sample_weights,
                dates_ns,
                raw_returns,
            ) = _load_training_month(
                path,
                train_start,
                train_end,
                stats_1m,
                stats_5m,
            )
            if len(day_1m) == 0:
                continue
            day_groups = _day_row_groups(dates_ns)
            rng.shuffle(day_groups)
            for rows, x1_array, x5_array in _prefetched_window_batches(
                day_1m, day_5m, window_index, day_groups
            ):
                valid_label = np.isfinite(targets[rows])
                valid_count = int(valid_label.sum())
                if valid_count < MIN_DAILY_SAMPLES:
                    continue
                x1b = _to_training_tensor(x1_array, device)
                x5b = _to_training_tensor(x5_array, device)
                rank_target = _to_training_tensor(
                    np.ascontiguousarray(targets[rows][valid_label]), device
                )
                label_mask = _to_training_tensor(
                    np.ascontiguousarray(valid_label), device
                ).bool()
                gaussian_target = _gaussianize_rank_targets(rank_target)
                target_full = torch.zeros(
                    len(rows), dtype=torch.float32, device=device
                )
                target_full[label_mask] = gaussian_target

                learning_rate = _v38_learning_rate(
                    update_index, total_updates, warmup_updates
                )
                kl_weight = _v38_kl_weight(update_index, steps_per_epoch)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device):
                    score, diagnostics = model.forward_components(
                        x1b,
                        x5b,
                        targets=target_full,
                        target_mask=label_mask,
                        return_training_outputs=True,
                    )
                    losses = _single_day_loss(
                        score,
                        target_full,
                        label_mask,
                        diagnostics,
                        kl_weight=kl_weight,
                    )
                    loss = losses["loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                with torch.no_grad():
                    metric_totals += torch.stack(
                        [losses[key].detach().float() for key in metric_keys]
                    )
                    factor_accumulator.update(diagnostics)
                update_index += 1
                n_steps += 1
                n_samples += valid_count
                del (
                    x1b,
                    x5b,
                    rank_target,
                    gaussian_target,
                    target_full,
                    label_mask,
                    x1_array,
                    x5_array,
                    score,
                    diagnostics,
                    losses,
                    loss,
                )
                if n_steps % TRAIN_PROGRESS_EVERY == 0:
                    logger.info(
                        "V37 epoch progress",
                        run=run_name,
                        epoch=global_epoch,
                        steps=n_steps,
                        steps_per_epoch=steps_per_epoch,
                        learning_rate=round(learning_rate, 8),
                        month=os.path.basename(path),
                        elapsed=round(time.time() - epoch_started, 2),
                    )
            del (
                day_1m,
                day_5m,
                window_index,
                targets,
                sample_weights,
                dates_ns,
                raw_returns,
            )
            gc.collect()

        if n_steps != steps_per_epoch:
            raise RuntimeError(
                f"V38 expected {steps_per_epoch} anchor steps but ran {n_steps}"
            )
        values = (metric_totals / float(n_steps)).cpu().tolist()
        metric_summary = dict(zip(metric_keys, map(float, values)))
        epoch_log: Dict[str, object] = {
            "epoch": global_epoch,
            "stage": "anchor",
            **metric_summary,
            "kl_weight_end": _v38_kl_weight(
                update_index - 1, steps_per_epoch
            ),
            "learning_rate_start": _v38_learning_rate(
                update_index - n_steps, total_updates, warmup_updates
            ),
            "learning_rate_end": _v38_learning_rate(
                update_index - 1, total_updates, warmup_updates
            ),
            "steps": n_steps,
            "samples": n_samples,
            **factor_accumulator.summary(),
        }
        history.append(epoch_log)
        logger.info(
            "V38 anchor epoch finished",
            run=run_name,
            epoch=global_epoch,
            epochs=initial_global_epoch + int(epochs),
            loss=round(float(epoch_log["loss"]), 8),
            prior_corr=round(float(epoch_log["prior_correlation"]), 6),
            posterior_corr=round(float(epoch_log["posterior_correlation"]), 6),
            prior_huber=round(float(epoch_log["prior_huber_loss"]), 8),
            posterior_huber=round(
                float(epoch_log["posterior_huber_loss"]), 8
            ),
            kl_loss=round(float(epoch_log["kl_loss"]), 8),
            kl_weight=round(float(epoch_log["kl_weight_end"]), 6),
            prior_score_std=round(float(epoch_log["prior_score_std"]), 6),
            posterior_score_std=round(
                float(epoch_log["posterior_score_std"]), 6
            ),
            portfolio_effective_stocks=round(
                float(epoch_log["portfolio_effective_stocks"]), 2
            ),
            portfolio_max_weight=round(
                float(epoch_log["portfolio_max_weight"]), 6
            ),
            portfolio_mean_abs_cosine=round(
                float(epoch_log["portfolio_mean_abs_cosine"]), 4
            ),
            beta_effective_rank=round(
                float(epoch_log["beta_effective_rank"]), 4
            ),
            factor_score_ratio=round(
                float(epoch_log["factor_score_ratio"]), 4
            ),
            learning_rate_end=round(float(epoch_log["learning_rate_end"]), 8),
            steps=n_steps,
            samples=n_samples,
            elapsed=round(time.time() - epoch_started, 2),
        )
        if checkpoint_dir and local_epoch + 1 == int(epochs):
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"v38_epoch_{global_epoch:02d}_anchor.json",
            )
            save_checkpoint(
                _v38_checkpoint_payload(
                    model,
                    stats_1m,
                    stats_5m,
                    train_start,
                    train_end,
                    history,
                    global_epoch,
                    "anchor",
                ),
                checkpoint_path,
            )
            logger.info(
                "V38 anchor checkpoint saved",
                checkpoint=checkpoint_path,
                epoch=global_epoch,
            )
    return model, history


def _frozen_body_cache_root(
    cache_paths: Sequence[str], run_name: str
) -> str:
    cache_root = os.path.commonpath(
        [os.path.abspath(path) for path in cache_paths]
    )
    if os.path.basename(cache_root)[:4].isdigit():
        cache_root = os.path.dirname(cache_root)
    safe_run = "".join(
        character if character.isalnum() else "_" for character in run_name
    )
    return os.path.join(cache_root, f".v38_body_{safe_run}_{os.getpid()}")


def _build_frozen_body_cache(
    model: V38TailLockedBodyRanker,
    cache_paths: Sequence[str],
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    train_start: str,
    train_end: str,
    device: torch.device,
    run_name: str,
) -> Tuple[str, Dict[str, str]]:
    """Encode the frozen anchor once; four body epochs reuse ~0.5 GiB FP16."""
    output_root = _frozen_body_cache_root(cache_paths, run_name)
    shutil.rmtree(output_root, ignore_errors=True)
    os.makedirs(output_root, exist_ok=True)
    outputs: Dict[str, str] = {}
    started = time.time()
    model.eval()
    try:
        with torch.inference_mode():
            for month_number, path in enumerate(cache_paths, start=1):
                (
                    day_1m,
                    day_5m,
                    window_index,
                    targets,
                    sample_weights,
                    dates_ns,
                    raw_returns,
                ) = _load_training_month(
                    path, train_start, train_end, stats_1m, stats_5m
                )
                if len(dates_ns) == 0:
                    del (
                        day_1m,
                        day_5m,
                        window_index,
                        targets,
                        sample_weights,
                        dates_ns,
                        raw_returns,
                    )
                    continue
                month_output = os.path.join(output_root, os.path.basename(path))
                os.makedirs(month_output, exist_ok=True)
                temporal_path = _cache_file(month_output, "temporal_f16.npy")
                anchor_path = _cache_file(month_output, "anchor_f32.npy")
                temporal_map = np.lib.format.open_memmap(
                    temporal_path,
                    mode="w+",
                    dtype="float16",
                    shape=(len(dates_ns), model.stock_dim),
                )
                anchor_map = np.lib.format.open_memmap(
                    anchor_path,
                    mode="w+",
                    dtype="float32",
                    shape=(len(dates_ns),),
                )
                for rows, x1_array, x5_array in _prefetched_window_batches(
                    day_1m,
                    day_5m,
                    window_index,
                    _day_row_groups(dates_ns),
                ):
                    x1b = _to_training_tensor(x1_array, device)
                    x5b = _to_training_tensor(x5_array, device)
                    with _autocast_context(device):
                        temporal = model.encode_temporal(x1b, x5b)
                        anchor = model.forward_from_temporal(temporal)
                    temporal_map[rows] = temporal.float().cpu().numpy().astype(
                        "float16"
                    )
                    anchor_map[rows] = anchor.float().cpu().numpy()
                    del x1b, x5b, x1_array, x5_array, temporal, anchor
                temporal_map.flush()
                anchor_map.flush()
                del temporal_map, anchor_map
                outputs[os.path.abspath(path)] = month_output
                logger.info(
                    "V38 frozen body month built",
                    run=run_name,
                    month=os.path.basename(path),
                    month_number=month_number,
                    months=len(cache_paths),
                    samples=len(dates_ns),
                    elapsed=round(time.time() - started, 2),
                )
                del (
                    day_1m,
                    day_5m,
                    window_index,
                    targets,
                    sample_weights,
                    dates_ns,
                    raw_returns,
                )
                gc.collect()
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    logger.info(
        "V38 frozen body cache built",
        run=run_name,
        months=len(outputs),
        cache_dir=output_root,
        elapsed=round(time.time() - started, 2),
    )
    return output_root, outputs


def _body_learning_rate(update_index: int, total_updates: int) -> float:
    if total_updates <= 1:
        return BODY_MIN_LR
    fraction = min(max(float(update_index) / float(total_updates - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return BODY_MIN_LR + cosine * (BODY_MAX_LR - BODY_MIN_LR)


def _single_body_day_loss(
    body_score: torch.Tensor,
    anchor_score: torch.Tensor,
    rank_targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    middle_mask = _anchor_middle_mask(anchor_score)
    training_mask = middle_mask & target_mask.bool()
    if int(training_mask.sum()) < MIN_DAILY_SAMPLES // 2:
        raise ValueError("V38 body stage has too few labeled middle stocks")
    body_valid = body_score.float()[training_mask]
    anchor_valid = anchor_score.float()[training_mask]
    target_valid = rank_targets.float()[training_mask]
    body_z = _cross_section_standardize(body_valid)
    target_z = _cross_section_standardize(target_valid)
    huber_loss = nn.functional.smooth_l1_loss(
        body_z,
        target_z,
        beta=BODY_HUBER_BETA,
        reduction="mean",
    )
    correlation_loss, correlation = _correlation_objective(
        body_valid, target_valid
    )
    _, anchor_correlation = _correlation_objective(anchor_valid, target_valid)
    loss = (
        BODY_HUBER_WEIGHT * huber_loss
        + BODY_CORRELATION_WEIGHT * correlation_loss
    )
    correction = body_valid - anchor_valid
    correction_ratio = correction.std(unbiased=False) / anchor_valid.std(
        unbiased=False
    ).clamp_min(1e-6)
    body_anchor_correlation = (
        _cross_section_standardize(body_valid)
        * _cross_section_standardize(anchor_valid)
    ).mean().clamp(-1.0, 1.0)
    return {
        "loss": loss,
        "huber_loss": huber_loss.detach(),
        "correlation_loss": correlation_loss.detach(),
        "body_correlation": correlation.detach(),
        "anchor_correlation": anchor_correlation.detach(),
        "body_anchor_correlation": body_anchor_correlation.detach(),
        "correction_std_ratio": correction_ratio.detach(),
        "body_score_std": body_valid.detach().std(unbiased=False),
        "middle_samples": training_mask.sum().detach().float(),
    }


def _train_body_model(
    model: V38TailLockedBodyRanker,
    cache_paths: Sequence[str],
    frozen_cache_paths: Dict[str, str],
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    train_start: str,
    train_end: str,
    device: torch.device,
    run_name: str,
    epochs: int,
    initial_global_epoch: int,
    history: Sequence[Dict[str, object]],
    checkpoint_dir: Optional[str],
) -> Tuple[nn.Module, List[Dict[str, object]]]:
    """Train only the residual body head from frozen stock embeddings."""
    if int(epochs) <= 0:
        return model, list(history)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.body_head.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.body_head.parameters()]
    optimizer = _new_adamw(trainable, device)
    steps_per_epoch = _eligible_training_steps(
        cache_paths, stats_1m, stats_5m, train_start, train_end
    )
    updates_per_epoch = int(
        math.ceil(steps_per_epoch / BODY_GRADIENT_ACCUMULATION_DAYS)
    )
    total_updates = int(epochs) * updates_per_epoch
    update_index = 0
    global_epoch = int(initial_global_epoch)
    result_history: List[Dict[str, object]] = list(history)
    rng = np.random.default_rng(SEED + 10_000 + global_epoch)
    metric_keys = (
        "loss",
        "huber_loss",
        "correlation_loss",
        "body_correlation",
        "anchor_correlation",
        "body_anchor_correlation",
        "correction_std_ratio",
        "body_score_std",
        "middle_samples",
    )
    logger.info(
        "V38 body stage initialized",
        run=run_name,
        epochs=int(epochs),
        steps_per_epoch=steps_per_epoch,
        optimizer_updates_per_epoch=updates_per_epoch,
        gradient_accumulation_days=BODY_GRADIENT_ACCUMULATION_DAYS,
        trainable_parameters=sum(p.numel() for p in trainable),
        max_lr=BODY_MAX_LR,
        min_lr=BODY_MIN_LR,
    )

    for _ in range(int(epochs)):
        global_epoch += 1
        epoch_started = time.time()
        path_order = list(cache_paths)
        rng.shuffle(path_order)
        metric_totals = torch.zeros(
            len(metric_keys), dtype=torch.float32, device=device
        )
        n_steps = 0
        n_samples = 0
        pending_days = 0
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for path in path_order:
            (
                day_1m,
                day_5m,
                window_index,
                targets,
                sample_weights,
                dates_ns,
                raw_returns,
            ) = _load_training_month(
                path, train_start, train_end, stats_1m, stats_5m
            )
            if len(dates_ns) == 0:
                del (
                    day_1m,
                    day_5m,
                    window_index,
                    targets,
                    sample_weights,
                    dates_ns,
                    raw_returns,
                )
                continue
            frozen_path = frozen_cache_paths[os.path.abspath(path)]
            temporal_map = np.load(
                _cache_file(frozen_path, "temporal_f16.npy"), mmap_mode="r"
            )
            anchor_map = np.load(
                _cache_file(frozen_path, "anchor_f32.npy"), mmap_mode="r"
            )
            day_groups = _day_row_groups(dates_ns)
            rng.shuffle(day_groups)
            for rows in day_groups:
                valid_label = np.isfinite(targets[rows])
                if int(valid_label.sum()) < MIN_DAILY_SAMPLES:
                    continue
                temporal = _to_training_tensor(
                    np.asarray(temporal_map[rows], dtype="float16"), device
                )
                anchor = _to_training_tensor(
                    np.asarray(anchor_map[rows], dtype="float32"), device
                )
                target = _to_training_tensor(
                    np.asarray(targets[rows], dtype="float32"), device
                )
                target_mask = _to_training_tensor(
                    np.asarray(valid_label), device
                ).bool()
                with _autocast_context(device):
                    body_score = model.body_score_from_temporal(
                        temporal, anchor
                    )
                    losses = _single_body_day_loss(
                        body_score, anchor, target, target_mask
                    )
                    scaled_loss = (
                        losses["loss"]
                        / float(BODY_GRADIENT_ACCUMULATION_DAYS)
                    )
                scaled_loss.backward()
                pending_days += 1
                should_step = (
                    pending_days == BODY_GRADIENT_ACCUMULATION_DAYS
                )
                if should_step:
                    learning_rate = _body_learning_rate(
                        update_index, total_updates
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_index += 1
                    pending_days = 0
                with torch.no_grad():
                    metric_totals += torch.stack(
                        [losses[key].detach().float() for key in metric_keys]
                    )
                n_steps += 1
                n_samples += int(losses["middle_samples"].item())
                del (
                    temporal,
                    anchor,
                    target,
                    target_mask,
                    body_score,
                    losses,
                    scaled_loss,
                )
                if n_steps % TRAIN_PROGRESS_EVERY == 0:
                    logger.info(
                        "V38 body epoch progress",
                        run=run_name,
                        epoch=global_epoch,
                        steps=n_steps,
                        steps_per_epoch=steps_per_epoch,
                        elapsed=round(time.time() - epoch_started, 2),
                    )
            del (
                day_1m,
                day_5m,
                window_index,
                targets,
                sample_weights,
                dates_ns,
                raw_returns,
                temporal_map,
                anchor_map,
            )

        if pending_days:
            correction = float(BODY_GRADIENT_ACCUMULATION_DAYS) / float(
                pending_days
            )
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            learning_rate = _body_learning_rate(update_index, total_updates)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_index += 1

        if n_steps != steps_per_epoch:
            raise RuntimeError(
                f"V38 expected {steps_per_epoch} body steps but ran {n_steps}"
            )
        values = (metric_totals / float(n_steps)).cpu().tolist()
        summary = dict(zip(metric_keys, map(float, values)))
        epoch_log: Dict[str, object] = {
            "epoch": global_epoch,
            "stage": "body_ranker",
            **summary,
            "learning_rate_end": _body_learning_rate(
                max(update_index - 1, 0), total_updates
            ),
            "optimizer_updates": update_index,
            "steps": n_steps,
            "samples": n_samples,
        }
        result_history.append(epoch_log)
        logger.info(
            "V38 body epoch finished",
            run=run_name,
            epoch=global_epoch,
            epochs=initial_global_epoch + int(epochs),
            loss=round(float(epoch_log["loss"]), 8),
            body_corr=round(float(epoch_log["body_correlation"]), 6),
            anchor_corr=round(float(epoch_log["anchor_correlation"]), 6),
            body_anchor_corr=round(
                float(epoch_log["body_anchor_correlation"]), 6
            ),
            correction_std_ratio=round(
                float(epoch_log["correction_std_ratio"]), 6
            ),
            learning_rate_end=round(
                float(epoch_log["learning_rate_end"]), 8
            ),
            steps=n_steps,
            samples=n_samples,
            elapsed=round(time.time() - epoch_started, 2),
        )
        if checkpoint_dir:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"v38_epoch_{global_epoch:02d}_body.json",
            )
            save_checkpoint(
                _v38_checkpoint_payload(
                    model,
                    stats_1m,
                    stats_5m,
                    train_start,
                    train_end,
                    result_history,
                    global_epoch,
                    "body_ranker",
                ),
                checkpoint_path,
            )
            logger.info(
                "V38 body checkpoint saved",
                checkpoint=checkpoint_path,
                epoch=global_epoch,
            )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()
    return model, result_history


def _mean_ir(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std, float(arr.mean() / std) if std > 1e-12 else 0.0


def _evaluate_streaming_model(
    model: nn.Module,
    cache_paths: Sequence[str],
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    valid_start: str,
    valid_end: str,
    device: torch.device,
) -> Dict[str, float]:
    pearson_values: List[float] = []
    rank_values: List[float] = []
    n_samples = 0
    model.eval()
    with torch.inference_mode():
        for path in cache_paths:
            (
                day_1m,
                day_5m,
                window_index,
                targets,
                sample_weights,
                dates_ns,
                raw_returns,
            ) = _load_training_month(
                path,
                valid_start,
                valid_end,
                stats_1m,
                stats_5m,
            )
            if len(day_1m) == 0:
                continue
            for idx in _day_row_groups(dates_ns):
                valid_label = np.isfinite(targets[idx]) & np.isfinite(
                    raw_returns[idx]
                )
                if int(valid_label.sum()) < MIN_DAILY_SAMPLES:
                    continue
                x1_array, x5_array = _assemble_compact_windows(
                    day_1m, day_5m, window_index, idx
                )
                x1b = _to_training_tensor(x1_array, device)
                x5b = _to_training_tensor(x5_array, device)
                with _autocast_context(device):
                    temporal = model.encode_temporal(x1b, x5b)
                    anchor_score, body_score = model.forward_scores_from_temporal(
                        temporal
                    )
                pred_all = _value_interval_score(
                    anchor_score.float().cpu().numpy(),
                    body_score.float().cpu().numpy(),
                    np.arange(len(idx)).astype(str),
                )
                pred = pred_all[valid_label]
                target = raw_returns[idx][valid_label].astype("float64")
                if pred.std(ddof=0) <= 1e-12 or target.std(ddof=0) <= 1e-12:
                    continue
                pearson_values.append(float(np.corrcoef(pred, target)[0, 1]))
                pred_rank = pd.Series(pred).rank(method="average").to_numpy("float64")
                target_rank = pd.Series(target).rank(method="average").to_numpy(
                    "float64"
                )
                rank_values.append(
                    float(np.corrcoef(pred_rank, target_rank)[0, 1])
                )
                n_samples += int(valid_label.sum())
            del (
                day_1m,
                day_5m,
                window_index,
                targets,
                sample_weights,
                dates_ns,
                raw_returns,
            )
            gc.collect()

    ic_mean, ic_std, ic_ir = _mean_ir(pearson_values)
    rank_ic_mean, rank_ic_std, rank_ic_ir = _mean_ir(rank_values)
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_ic_ir": rank_ic_ir,
        "n_days": len(pearson_values),
        "valid_samples": n_samples,
        "recommended_score_sign": 1.0 if ic_mean >= 0.0 else -1.0,
    }


def run_rolling_validation(
    datasources: Optional[Dict[str, str]] = None,
    cache_dir: str = CACHE_DIR,
    force_cache: bool = False,
) -> pd.DataFrame:
    """Train 2019-2021 -> 2022 and 2019-2022 -> 2023 with one shared raw cache."""
    _seed_everything(SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    table_5m, table_1m = _pick_tables(datasources)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_paths = prepare_monthly_cache(
        table_5m,
        table_1m,
        TRAIN_START,
        TRAIN_END,
        cache_dir=cache_dir,
        force=force_cache,
    )
    results: List[Dict[str, object]] = []
    for fold in ROLLING_FOLDS:
        fold_started = time.time()
        logger.info("fold started", **fold)
        stats_1m, stats_5m = load_or_fit_streaming_stats(
            cache_paths,
            cache_dir,
            table_5m,
            table_1m,
            fold["train_start"],
            fold["train_end"],
        )
        materialize_standardized_cache(cache_paths, stats_1m, stats_5m)
        model, history = _train_anchor_model(
            cache_paths=cache_paths,
            stats_1m=stats_1m,
            stats_5m=stats_5m,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            device=device,
            run_name=fold["name"],
            epochs=ANCHOR_EPOCHS,
            schedule_epochs=ANCHOR_SCHEDULE_EPOCHS,
        )
        frozen_root = None
        try:
            frozen_root, frozen_paths = _build_frozen_body_cache(
                model,
                cache_paths,
                stats_1m,
                stats_5m,
                fold["train_start"],
                fold["train_end"],
                device,
                fold["name"],
            )
            model, history = _train_body_model(
                model,
                cache_paths,
                frozen_paths,
                stats_1m,
                stats_5m,
                fold["train_start"],
                fold["train_end"],
                device,
                fold["name"],
                BODY_EPOCHS,
                ANCHOR_EPOCHS,
                history,
                None,
            )
        finally:
            if frozen_root:
                shutil.rmtree(frozen_root, ignore_errors=True)
        metrics = _evaluate_streaming_model(
            model=model,
            cache_paths=cache_paths,
            stats_1m=stats_1m,
            stats_5m=stats_5m,
            valid_start=fold["valid_start"],
            valid_end=fold["valid_end"],
            device=device,
        )
        metrics.update(
            {
                "fold": fold["name"],
                "last_train_loss": float(history[-1]["loss"]),
                "elapsed": round(time.time() - fold_started, 2),
            }
        )
        results.append(metrics)
        logger.info("fold finished", **metrics)
        del model, stats_1m, stats_5m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(results)


def _train_range_and_save(
    datasources: Optional[Dict[str, str]] = None,
    *,
    model_path: str,
    cache_dir: str,
    force_cache: bool,
    train_start: str,
    train_end: str,
    cache_end: str,
    anchor_epochs: int,
    body_epochs: int,
    run_name: str,
    checkpoint_dir: Optional[str],
) -> str:
    """Train one V38 anchor/body model and save one compact checkpoint."""
    _seed_everything(SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    started = time.time()
    table_5m, table_1m = _pick_tables(datasources)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "V38 training started",
        run=run_name,
        device=str(device),
        table_5m=table_5m,
        table_1m=table_1m,
        train_start=train_start,
        train_end=train_end,
        anchor_epochs=int(anchor_epochs),
        anchor_schedule_epochs=ANCHOR_SCHEDULE_EPOCHS,
        body_epochs=int(body_epochs),
        cache_dir=cache_dir,
        cache_schema=CACHE_SCHEMA,
    )
    cache_paths = prepare_monthly_cache(
        table_5m,
        table_1m,
        train_start,
        cache_end,
        cache_dir=cache_dir,
        force=force_cache,
    )
    stats_1m, stats_5m = load_or_fit_streaming_stats(
        cache_paths,
        cache_dir,
        table_5m,
        table_1m,
        train_start,
        train_end,
    )
    materialize_standardized_cache(cache_paths, stats_1m, stats_5m)
    model, history = _train_anchor_model(
        cache_paths=cache_paths,
        stats_1m=stats_1m,
        stats_5m=stats_5m,
        train_start=train_start,
        train_end=train_end,
        device=device,
        run_name=run_name,
        epochs=int(anchor_epochs),
        schedule_epochs=ANCHOR_SCHEDULE_EPOCHS,
        checkpoint_dir=checkpoint_dir,
    )
    frozen_root: Optional[str] = None
    try:
        frozen_root, frozen_paths = _build_frozen_body_cache(
            model,
            cache_paths,
            stats_1m,
            stats_5m,
            train_start,
            train_end,
            device,
            run_name,
        )
        model, history = _train_body_model(
            model=model,
            cache_paths=cache_paths,
            frozen_cache_paths=frozen_paths,
            stats_1m=stats_1m,
            stats_5m=stats_5m,
            train_start=train_start,
            train_end=train_end,
            device=device,
            run_name=run_name,
            epochs=int(body_epochs),
            initial_global_epoch=int(anchor_epochs),
            history=history,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        if frozen_root:
            shutil.rmtree(frozen_root, ignore_errors=True)
            logger.info("V38 temporary body cache removed", path=frozen_root)
    payload = _v38_checkpoint_payload(
        model,
        stats_1m,
        stats_5m,
        train_start,
        train_end,
        history,
        int(anchor_epochs) + int(body_epochs),
        "body_ranker",
    )
    payload.update(
        {
            "table_1m": table_1m,
            "table_5m": table_5m,
            "seq_len": SEQ_LEN,
            "minutes_per_bar": MINUTES_PER_BAR,
            "label": {
                "return": "next_global_trading_day_raw_close_to_close_return",
                "cached_target": "daily_average_rank_mapped_to_minus1_plus1",
                "anchor_training_target": "finite_Gaussianized_daily_rank",
                "body_training_target": "linear_daily_rank_on_anchor_middle_80pct",
                "ties": "average_rank",
            },
            "preprocessing": {
                "price": "nonfinite_or_nonpositive_missing_then_log",
                "activity": "negative_or_nonfinite_missing_then_log1p_real_zero_kept",
                "statistics": "separate_1m_5m_training_only_per_field",
                "missing_fill": "zero_in_standardized_space",
                "cross_field_features": False,
            },
            "architecture": {
                "minute_encoder": "day_aware_trade_book_raw_convolution",
                "five_minute_aggregation": "depthwise_stride_five_plus_raw_5m_macro",
                "temporal": "single_layer_GRU_over_96_five_minute_states",
                "stock_summary": "last_mean_max_projection",
                "cross_section": "eight_supervised_dynamic_factor_portfolios",
                "portfolio_logit_temperature": PORTFOLIO_LOGIT_TEMPERATURE,
                "portfolio_diversity_reference": "equal_weight_market",
                "anchor_score": "stock_alpha_plus_exposure_times_prior_factor_mean",
                "body_head": "zero_initialized_anchor_plus_residual_MLP",
                "tail_lock": "anchor_bottom_and_top_10pct",
                "final_score": "anchor_tail_values_plus_body_value_interval_ranks",
                "training_only_teacher": "target_conditioned_factor_posterior",
                "inference": "historical_input_factor_prior_only",
            },
            "loss": {
                "prior_rank_huber": PRIOR_HUBER_WEIGHT,
                "prior_daily_correlation": PRIOR_CORRELATION_WEIGHT,
                "posterior_teacher": POSTERIOR_TEACHER_WEIGHT,
                "kl_max": KL_MAX_WEIGHT,
                "kl_warmup_epochs": KL_WARMUP_EPOCHS,
                "exposure_orthogonality": EXPOSURE_ORTHOGONALITY_WEIGHT,
                "portfolio_diversity": PORTFOLIO_DIVERSITY_WEIGHT,
                "huber_beta": HUBER_BETA,
                "Gaussian_target_clip": GAUSSIAN_TARGET_CLIP,
                "body_rank_huber": BODY_HUBER_WEIGHT,
                "body_daily_correlation": BODY_CORRELATION_WEIGHT,
                "body_huber_beta": BODY_HUBER_BETA,
            },
            "training": {
                "epochs": int(anchor_epochs) + int(body_epochs),
                "anchor_epochs": int(anchor_epochs),
                "anchor_schedule_epochs": ANCHOR_SCHEDULE_EPOCHS,
                "body_epochs": int(body_epochs),
                "optimizer": "fused_AdamW_when_available",
                "anchor_schedule": "V37_14_epoch_time_axis_stopped_after_epoch_6",
                "body_schedule": "cosine_on_frozen_FP16_temporal_cache",
                "max_lr": MAX_LR,
                "min_lr": MIN_LR,
                "body_max_lr": BODY_MAX_LR,
                "body_min_lr": BODY_MIN_LR,
                "weight_decay": 1e-4,
                "optimizer_resets": 1,
                "body_gradient_accumulation_days": BODY_GRADIENT_ACCUMULATION_DAYS,
            },
            "batching": "one_complete_historical_CSI1000_day_per_step",
        }
    )
    path = save_checkpoint(payload, model_path=model_path)
    file_info = checkpoint_file_info(path)
    if not bool(file_info["under_50mb_submission_limit"]):
        raise RuntimeError(
            f"V38 checkpoint is {file_info['size_mib']} MiB and exceeds 50 MB"
        )
    logger.info(
        "V38 training finished",
        run=run_name,
        model_path=path,
        trainable_parameters=count_trainable_parameters(model),
        checkpoint_mib=file_info["size_mib"],
        elapsed=round(time.time() - started, 2),
    )
    return path


def train_development_and_save(
    datasources: Optional[Dict[str, str]] = None,
    model_path: Optional[str] = None,
    cache_dir: str = CACHE_DIR,
    force_cache: bool = False,
    checkpoint_dir: Optional[str] = CHECKPOINT_DIR,
) -> str:
    """Train on 2019-2023 for architecture development and 2024 evaluation."""
    if model_path is None:
        model_path = os.path.join(HERE, "v38_development_2019_2023.json")
    return _train_range_and_save(
        datasources,
        model_path=model_path,
        cache_dir=cache_dir,
        force_cache=force_cache,
        train_start=DEV_TRAIN_START,
        train_end=DEV_TRAIN_END,
        cache_end=DEV_VALID_END,
        anchor_epochs=ANCHOR_EPOCHS,
        body_epochs=BODY_EPOCHS,
        run_name="development_2019_2023",
        checkpoint_dir=checkpoint_dir,
    )


def train_and_save(
    datasources: Optional[Dict[str, str]] = None,
    model_path: str = MODEL_PATH,
    cache_dir: str = CACHE_DIR,
    force_cache: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> str:
    """Train the fixed 2019-2024 final model and save its JSON checkpoint."""
    return _train_range_and_save(
        datasources,
        model_path=model_path,
        cache_dir=cache_dir,
        force_cache=force_cache,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        cache_end=TRAIN_END,
        anchor_epochs=ANCHOR_EPOCHS,
        body_epochs=BODY_EPOCHS,
        run_name="final_2019_2024",
        checkpoint_dir=checkpoint_dir,
    )


def _predict_compact_arrays(
    model: V38TailLockedBodyRanker,
    day_1m_raw: np.ndarray,
    day_5m_raw: np.ndarray,
    window_index: np.ndarray,
    dates_ns: np.ndarray,
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode each stock-day once and return anchor/body raw scores."""
    anchor_scores = np.zeros(len(dates_ns), dtype="float64")
    body_scores = np.zeros(len(dates_ns), dtype="float64")
    if len(day_1m_raw) == 0:
        return anchor_scores, body_scores
    bar_states = np.empty(
        (len(day_1m_raw), BARS_PER_DAY, model.bar_dim), dtype="float32"
    )
    context_batch = INFER_CONTEXT_BATCH
    context_start = 0
    model.eval()
    with torch.inference_mode():
        while context_start < len(day_1m_raw):
            stop = min(context_start + context_batch, len(day_1m_raw))
            one = _standardize_1m(day_1m_raw[context_start:stop], stats_1m)
            five = _standardize_5m(day_5m_raw[context_start:stop], stats_5m)
            x1b = x5b = encoded = None
            try:
                x1b = _to_training_tensor(one.astype("float16"), device)
                x5b = _to_training_tensor(five.astype("float16"), device)
                with _autocast_context(device):
                    encoded = model.encode_bars(x1b, x5b)
                bar_states[context_start:stop] = encoded.float().cpu().numpy()
                context_start = stop
                del x1b, x5b, encoded, one, five
            except RuntimeError as exc:
                message = str(exc).lower()
                is_oom = device.type == "cuda" and (
                    "out of memory" in message or "cuda oom" in message
                )
                if not is_oom or context_batch <= 32:
                    raise
                context_batch = max(32, context_batch // 2)
                del x1b, x5b, encoded, one, five
                gc.collect()
                torch.cuda.empty_cache()
                logger.warning(
                    "v38 context encoding OOM; reducing batch",
                    retry_batch=context_batch,
                )

        for rows in _day_row_groups(dates_ns):
            references = np.asarray(window_index[rows], dtype="int64")
            valid = references >= 0
            safe = np.where(valid, references, 0)
            bars = np.asarray(
                bar_states[safe.reshape(-1)], dtype="float32"
            ).reshape(len(rows), SEQ_LEN, model.bar_dim)
            if not bool(valid.all()):
                bars_view = bars.reshape(
                    len(rows), 2, BARS_PER_DAY, model.bar_dim
                )
                bars_view[~valid] = 0.0
            stock_batch = INFER_STOCK_BATCH
            while True:
                temporal_parts: List[torch.Tensor] = []
                bar_batch = None
                try:
                    for start in range(0, len(rows), stock_batch):
                        stop = min(start + stock_batch, len(rows))
                        bar_batch = _to_training_tensor(bars[start:stop], device)
                        with _autocast_context(device):
                            temporal_parts.append(
                                model.encode_temporal_from_bars(bar_batch)
                            )
                        del bar_batch
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    is_oom = device.type == "cuda" and (
                        "out of memory" in message or "cuda oom" in message
                    )
                    if not is_oom or stock_batch <= 32:
                        raise
                    del temporal_parts, bar_batch
                    stock_batch = max(32, stock_batch // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.warning(
                        "v38 temporal inference OOM; reducing stock batch",
                        retry_batch=stock_batch,
                    )
            temporal = torch.cat(temporal_parts, dim=0)
            with _autocast_context(device):
                anchor, body = model.forward_scores_from_temporal(temporal)
            anchor_scores[rows] = anchor.float().cpu().numpy().astype("float64")
            body_scores[rows] = body.float().cpu().numpy().astype("float64")
            del temporal_parts, temporal, anchor, body, bars
    del bar_states
    return anchor_scores, body_scores


def _predict_month_full_cross_section(
    model: nn.Module,
    table_5m: str,
    table_1m: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    stats_1m: Tuple[np.ndarray, np.ndarray],
    stats_5m: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    (
        day_1m_raw,
        day_5m_raw,
        window_index,
        _,
        dates_ns,
        _,
        instruments,
    ) = _build_month_arrays(
        table_5m=table_5m,
        table_1m=table_1m,
        month_start=month_start,
        month_end=month_end,
        label_limit=None,
        require_labels=False,
        instrument_chunk_size=INFER_INSTRUMENT_CHUNK,
    )
    if len(day_1m_raw) == 0:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    anchor_scores, body_scores = _predict_compact_arrays(
        model,
        day_1m_raw,
        day_5m_raw,
        window_index,
        dates_ns,
        stats_1m,
        stats_5m,
        device,
    )
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_ns),
            "instrument": instruments.astype(str),
            "anchor_score": anchor_scores,
            "body_score": body_scores,
        }
    )
    result = _compose_value_interval_frame(result)
    del (
        day_1m_raw,
        day_5m_raw,
        window_index,
        dates_ns,
        instruments,
        anchor_scores,
        body_scores,
    )
    gc.collect()
    return result

def _run_inference(
    datasources: Dict[str, str],
    start_date: str,
    end_date: str,
    *,
    model_path: str,
) -> pd.DataFrame:
    """Shared exact full-cross-section inference implementation."""
    started = time.time()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    model_path = resolve_model_path(model_path)
    table_5m, table_1m = _pick_tables(datasources)
    pool = query_pool(start_date, end_date)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(model_path, map_location=device)
    if list(ckpt.get("feature_fields_1m", [])) != list(FEATURE_FIELDS_1M):
        raise ValueError("V38 checkpoint 1m field order does not match code")
    if list(ckpt.get("feature_fields_5m", [])) != list(FEATURE_FIELDS_5M):
        raise ValueError("V38 checkpoint 5m field order does not match code")
    inference_model_cfg = dict(ckpt["model_cfg"])
    model = V38TailLockedBodyRanker(**inference_model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    logger.info(
        "v38 inference model initialized",
        device=str(device),
        inference_instrument_chunk=INFER_INSTRUMENT_CHUNK,
        inference_context_batch=INFER_CONTEXT_BATCH,
        inference_stock_batch=INFER_STOCK_BATCH,
    )
    stats_1m = (
        np.asarray(ckpt["mean_1m"], dtype="float32"),
        np.asarray(ckpt["std_1m"], dtype="float32"),
    )
    stats_5m = (
        np.asarray(ckpt["mean_5m"], dtype="float32"),
        np.asarray(ckpt["std_5m"], dtype="float32"),
    )
    parts: List[pd.DataFrame] = []
    for month_start, month_end in _month_ranges(start_date, end_date):
        month_started = time.time()
        part = _predict_month_full_cross_section(
            model,
            table_5m,
            table_1m,
            month_start,
            month_end,
            stats_1m,
            stats_5m,
            device,
        )
        parts.append(part)
        logger.info(
            "inference month finished",
            month=month_start.strftime("%Y-%m"),
            rows=len(part),
            elapsed=round(time.time() - month_started, 2),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predictions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["date", "instrument", "score"]
    )
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.normalize()
    predictions["instrument"] = predictions["instrument"].astype(str)
    predictions = predictions.drop_duplicates(["date", "instrument"], keep="last")

    if pool.empty:
        result = predictions
    else:
        result = pool.merge(predictions, on=["date", "instrument"], how="left")
    result["score"] = (
        pd.to_numeric(result["score"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float64")
    )
    result["score"] *= float(ckpt.get("score_sign", SCORE_SIGN))
    result = result.sort_values(["date", "instrument"]).reset_index(drop=True)
    logger.info(
        "v38 prediction finished",
        score_path="tail_locked_value_interval",
        rows=len(result),
        days=result["date"].nunique() if not result.empty else 0,
        elapsed=round(time.time() - started, 2),
    )
    return result[["date", "instrument", "score"]]


def main_with_model_path(
    datasources: Dict[str, str],
    start_date: str,
    end_date: str,
    model_path: str,
) -> pd.DataFrame:
    """Local evaluation helper for an explicit development checkpoint."""
    return _run_inference(
        datasources,
        start_date,
        end_date,
        model_path=model_path,
    )


def main(datasources: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    """Platform inference: exact full-day final-score pass and three columns."""
    return _run_inference(
        datasources,
        start_date,
        end_date,
        model_path=MODEL_PATH,
    )


if __name__ == "__main__":
    train_and_save(
        {"bar5m": DEFAULT_TABLE_5M, "bar1m": DEFAULT_TABLE_1M}
    )
