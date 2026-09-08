"""Standalone BigAlpha TCN-GRU-ISAB training, preprocessing, and JSON loading.

Rule-text constraints: a submission may contain multiple files but exactly one notebook; the
notebook must expose the evaluation entry point; training code, inference code, and pretrained
weights must be included; the public board loads weights without retraining; the private board
rebuilds weights from zero through ``train_and_save``.

Uploader observations and a teammate's successful package, not rule text: the archive accepts
UTF-8 text only, is smaller than 50 MiB, and contains three flat root files without directories.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import random
import re
import warnings
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "bid_price1",
    "ask_price1",
    "bid_volume1",
    "ask_volume1",
    "bid_price2",
    "bid_price3",
    "bid_price4",
    "bid_price5",
    "ask_price2",
    "ask_price3",
    "ask_price4",
    "ask_price5",
    "bid_volume2",
    "bid_volume3",
    "bid_volume4",
    "bid_volume5",
    "ask_volume2",
    "ask_volume3",
    "ask_volume4",
    "ask_volume5",
    "bid_num_orders1",
    "bid_num_orders2",
    "bid_num_orders3",
    "bid_num_orders4",
    "bid_num_orders5",
    "ask_num_orders1",
    "ask_num_orders2",
    "ask_num_orders3",
    "ask_num_orders4",
    "ask_num_orders5",
    "deal_number",
]
PRICE_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    *[f"bid_price{level}" for level in range(1, 6)],
    *[f"ask_price{level}" for level in range(1, 6)],
]
BARRA_COLUMNS = [
    "SIZE",
    "BETA",
    "MOMENTUM",
    "RESVOL",
    "SIZENL",
    "BTOP",
    "LIQUIDTY",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
]


def _session_end_times(interval_minutes: int) -> list[str]:
    """Return exchange-session bar end times for one exact minute frequency."""
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be positive")
    minutes = [
        *range(9 * 60 + 30 + interval_minutes, 11 * 60 + 30 + 1, interval_minutes),
        *range(13 * 60 + interval_minutes, 15 * 60 + 1, interval_minutes),
    ]
    return [f"{value // 60:02d}:{value % 60:02d}" for value in minutes]


BAR_END_TIMES_BY_FREQUENCY = {
    "5m": _session_end_times(5),
    "30m": _session_end_times(30),
}
FREQUENCY = "30m"
BAR_END_TIMES = BAR_END_TIMES_BY_FREQUENCY[FREQUENCY]
BOOK_DEPTH = 5
FEATURE_COUNT = len(FEATURE_COLS)
LOOKBACK_DAYS = 60
MODEL_HEADS = ("h1", "h5", "h20")
MODEL_CFG = {
    "lookback_days": LOOKBACK_DAYS,
    "n_bars": len(BAR_END_TIMES),
    "n_fields": FEATURE_COUNT,
    "d_intra": 64,
    "d_model": 128,
    "n_market_tokens": 16,
    "n_heads": 4,
    "dropout": 0.25,
    "feature_dropout": 0.1,
    "random_lookback_min": 40,
    "tcn_kernels": [3, 5],
    "tcn_dilations": [1, 2],
    "min_parameters": 100_000,
    "max_parameters": 100_000_000,
}
PREPROCESS_CONTRACT = {
    "price_transform": "log",
    "default_transform": "log1p",
    "scaler": "standard",
    "clip": 5.0,
    "fill": 0.0,
    "price_zero_to_nan": True,
}
PUBLIC_BAR_TABLES = {
    "5m": "bigalpha_2026_stock_bar5m",
    "30m": "bigalpha_2026_stock_bar30m",
}
PUBLIC_BAR30M_TABLE = PUBLIC_BAR_TABLES["30m"]
PUBLIC_INSTRUMENTS_TABLE = "bigalpha_2026_instruments"
PUBLIC_EXPOSURE_TABLE = "bigalpha_2026_exposure"
DEFAULT_MODEL_PATH = Path("tcn_gru_isab_model.json")
PREDICT_QUERY_DAYS = 5
PREDICT_INSTRUMENT_CHUNK = 50
TRAIN_QUERY_DAYS = 31
TRAIN_INSTRUMENT_CHUNK = 250
TRAIN_EPOCHS = 10
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_EMA_DECAY = 0.999
TRAIN_MSE_WEIGHT = 0.1
TRAIN_MAX_GRAD_NORM = 1.0
FULL_PRECISION_LIMIT_BYTES = 40 * 1024**2
SIGNIFICANT_DIGITS = 6
ILLEGAL_ERROR_RATIO = 0.01
ILLEGAL_SAMPLE_SIZE = 5
_TABLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONVOLUTIONS_PER_TCN_BLOCK = 2


def prediction_calendar_buffer_days(lookback_days: int) -> int:
    """Convert a metadata trading-day lookback into a conservative calendar-day buffer."""
    if not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or lookback_days < 1:
        raise ValueError("lookback_days must be a positive integer")
    return lookback_days * 2


def _assert_static_contract() -> None:
    """Fail at import time if the flattened field/model contract drifts."""
    if FEATURE_COUNT != 37 or len(set(FEATURE_COLS)) != FEATURE_COUNT:
        raise AssertionError("FEATURE_COLS must contain exactly 37 unique ordered fields")
    if MODEL_CFG["n_fields"] != FEATURE_COUNT:
        raise AssertionError("MODEL_CFG.n_fields must equal FEATURE_COUNT")
    if MODEL_CFG["n_bars"] != len(BAR_END_TIMES) or BOOK_DEPTH != 5:
        raise AssertionError("default frequency, book depth, and bar count contract drifted")
    if set(BAR_END_TIMES_BY_FREQUENCY) != set(PUBLIC_BAR_TABLES):
        raise AssertionError(
            "bar time and public table frequency contracts must have identical keys"
        )
    if not set(PRICE_FIELDS).issubset(FEATURE_COLS):
        raise AssertionError("PRICE_FIELDS must be a subset of FEATURE_COLS")


_assert_static_contract()


def tcn_receptive_field(kernels: Sequence[int], dilations: Sequence[int]) -> int:
    """Return the causal receptive field of the stacked two-convolution TCN blocks."""
    normalized_kernels = tuple(kernels)
    normalized_dilations = tuple(dilations)
    if not normalized_kernels or len(normalized_kernels) != len(normalized_dilations):
        raise ValueError("TCN kernels and dilations must be non-empty equal-length sequences")
    values = (*normalized_kernels, *normalized_dilations)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError("TCN kernels and dilations must contain positive integers")
    return 1 + _CONVOLUTIONS_PER_TCN_BLOCK * sum(
        (kernel - 1) * dilation
        for kernel, dilation in zip(normalized_kernels, normalized_dilations, strict=True)
    )


class CausalConv1d(nn.Module):
    """Apply one temporal convolution with left-only padding."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        """Build a length-preserving causal convolution."""
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve without exposing a bar to any later bar."""
        return self.conv(functional.pad(x, (self.left_padding, 0)))


class CausalTCNBlock(nn.Module):
    """Use two causal convolutions in a regularized residual block."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
        """Build one causal residual scale."""
        super().__init__()
        self.conv1 = CausalConv1d(channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a causal residual sequence with unchanged length."""
        residual = x
        hidden = self.conv1(x).transpose(1, 2)
        hidden = self.dropout(self.activation(self.norm1(hidden))).transpose(1, 2)
        hidden = self.conv2(hidden).transpose(1, 2)
        hidden = self.dropout(self.activation(self.norm2(hidden))).transpose(1, 2)
        return residual + hidden


class AttentionPool(nn.Module):
    """Pool a short ordered sequence through learned scalar attention weights."""

    def __init__(self, dimension: int, dropout: float) -> None:
        """Build the attention scorer."""
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a weighted sum over the penultimate sequence dimension."""
        scores = self.scorer(sequence).squeeze(-1)
        if padding_mask is not None:
            padding = padding_mask.to(device=sequence.device, dtype=torch.bool)
            if padding.shape != sequence.shape[:-1]:
                raise ValueError(
                    "attention padding_mask must match sequence dimensions before features: "
                    f"expected={tuple(sequence.shape[:-1])}, actual={tuple(padding.shape)}"
                )
            scores = scores.masked_fill(padding, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        if padding_mask is not None:
            weights = torch.nan_to_num(weights, nan=0.0).masked_fill(padding, 0.0)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=-2)


class IntraDayEncoder(nn.Module):
    """Project fields, apply causal multi-scale TCNs, then pool bars early."""

    def __init__(
        self,
        n_fields: int,
        d_intra: int,
        *,
        kernels: Sequence[int],
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        """Build the input projection and all configured causal TCN scales."""
        super().__init__()
        self.input_projection = nn.Conv1d(n_fields, d_intra, kernel_size=1)
        self.input_norm = nn.LayerNorm(d_intra)
        self.tcn = nn.ModuleList(
            CausalTCNBlock(
                d_intra,
                kernel_size=kernel,
                dilation=dilation,
                dropout=dropout,
            )
            for kernel, dilation in zip(kernels, dilations, strict=True)
        )
        self.pool = AttentionPool(d_intra, dropout)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return one causal representation at every bar."""
        if x.ndim != 3:
            raise ValueError("intra-day input must have shape [samples, bars, fields]")
        hidden = self.input_projection(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.input_norm(hidden).transpose(1, 2)
        for block in self.tcn:
            hidden = block(hidden)
        return hidden.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress every stock-day to one fixed-width vector."""
        return self.pool(self.encode_sequence(x), padding_mask)


def _feed_forward(dimension: int, dropout: float) -> nn.Sequential:
    """Build a compact transformer-style feed-forward sublayer."""
    return nn.Sequential(
        nn.Linear(dimension, dimension * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dimension * 2, dimension),
        nn.Dropout(dropout),
    )


class InducedSetAttention(nn.Module):
    """Exchange cross-sectional information through a small learned token set."""

    def __init__(
        self,
        dimension: int,
        *,
        n_market_tokens: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        """Build two O(NK) cross-attention passes without stock identity embeddings."""
        super().__init__()
        self.market_tokens = nn.Parameter(torch.empty(n_market_tokens, dimension))
        nn.init.normal_(self.market_tokens, mean=0.0, std=dimension**-0.5)
        self.token_query_norm = nn.LayerNorm(dimension)
        self.stock_key_norm = nn.LayerNorm(dimension)
        self.token_attention = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.token_ffn_norm = nn.LayerNorm(dimension)
        self.token_ffn = _feed_forward(dimension, dropout)
        self.stock_query_norm = nn.LayerNorm(dimension)
        self.token_key_norm = nn.LayerNorm(dimension)
        self.stock_attention = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.stock_ffn_norm = nn.LayerNorm(dimension)
        self.stock_ffn = _feed_forward(dimension, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stocks: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Return a correction in which padded stocks neither write nor affect peers."""
        if stocks.ndim != 2:
            raise ValueError("ISAB input must have shape [stocks, dimension]")
        if padding_mask is None:
            padding = torch.zeros(len(stocks), dtype=torch.bool, device=stocks.device)
        else:
            padding = padding_mask.to(device=stocks.device, dtype=torch.bool)
            if padding.ndim != 1 or len(padding) != len(stocks):
                raise ValueError("padding_mask must have shape [stocks]")

        stock_batch = stocks.unsqueeze(0)
        tokens = self.market_tokens.unsqueeze(0)
        normalized_stocks = self.stock_key_norm(stock_batch)
        token_delta, _ = self.token_attention(
            self.token_query_norm(tokens),
            normalized_stocks,
            normalized_stocks,
            key_padding_mask=padding.unsqueeze(0),
            need_weights=False,
        )
        tokens = tokens + self.dropout(token_delta)
        tokens = tokens + self.token_ffn(self.token_ffn_norm(tokens))
        normalized_tokens = self.token_key_norm(tokens)
        stock_delta, _ = self.stock_attention(
            self.stock_query_norm(stock_batch),
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )
        correction = stock_delta + self.stock_ffn(self.stock_ffn_norm(stock_delta))
        return correction.squeeze(0).masked_fill(padding.unsqueeze(-1), 0.0)


class TCNGRUISAB(nn.Module):
    """Causally compress bars, encode days, and correct scores through market tokens."""

    requires_full_cross_section = True
    supports_multi_horizon = True
    supported_horizons = (1, 5, 20)

    def __init__(
        self,
        *,
        lookback_days: int,
        n_bars: int,
        n_fields: int,
        d_intra: int,
        d_model: int,
        n_market_tokens: int,
        n_heads: int,
        dropout: float,
        feature_dropout: float,
        random_lookback_min: int,
        tcn_kernels: Sequence[int],
        tcn_dilations: Sequence[int],
        min_parameters: int,
        max_parameters: int,
    ) -> None:
        """Build the compact model and enforce dimensions and parameter count."""
        super().__init__()
        dimensions = {
            "lookback_days": lookback_days,
            "n_bars": n_bars,
            "n_fields": n_fields,
            "d_intra": d_intra,
            "d_model": d_model,
            "n_market_tokens": n_market_tokens,
            "n_heads": n_heads,
            "random_lookback_min": random_lookback_min,
        }
        invalid = {
            name: value
            for name, value in dimensions.items()
            if not isinstance(value, int) or isinstance(value, bool) or value < 1
        }
        if invalid:
            raise ValueError(f"model dimensions must be positive integers: {invalid}")
        if random_lookback_min > lookback_days:
            raise ValueError("random_lookback_min must not exceed lookback_days")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= float(dropout) < 1.0 or not 0.0 <= float(feature_dropout) < 1.0:
            raise ValueError("dropout and feature_dropout must be in [0, 1)")
        kernels = tuple(tcn_kernels)
        dilations = tuple(tcn_dilations)
        receptive_field = tcn_receptive_field(kernels, dilations)
        if receptive_field < n_bars:
            raise ValueError(
                f"TCN receptive field {receptive_field} is smaller than n_bars={n_bars}; "
                "each residual block contains two causal convolutions"
            )

        self.lookback_days = lookback_days
        self.n_bars = n_bars
        self.n_fields = n_fields
        self.random_lookback_min = random_lookback_min
        self.feature_dropout = float(feature_dropout)
        self.tcn_receptive_field = receptive_field
        self._lookback_rng = random.Random(random.randrange(2**63))
        self.intra_encoder = IntraDayEncoder(
            n_fields,
            d_intra,
            kernels=kernels,
            dilations=dilations,
            dropout=float(dropout),
        )
        self.temporal_encoder = nn.GRU(d_intra, d_model, batch_first=True)
        self.temporal_pool = AttentionPool(d_model, float(dropout))
        self.cross_section = InducedSetAttention(
            d_model,
            n_market_tokens=n_market_tokens,
            n_heads=n_heads,
            dropout=float(dropout),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.output_norm = nn.LayerNorm(d_model)
        self.head_h1 = nn.Linear(d_model, 1)
        self.head_h5 = nn.Linear(d_model, 1)
        self.head_h20 = nn.Linear(d_model, 1)
        self.primary_head = f"h{max(self.supported_horizons)}"
        self.active_heads = MODEL_HEADS
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if not min_parameters <= self.parameter_count <= max_parameters:
            raise AssertionError(
                "TCN-GRU-ISAB parameter count violates contract: "
                f"count={self.parameter_count}, bounds=[{min_parameters}, {max_parameters}]"
            )

    def configure_output_horizons(
        self,
        horizons: Sequence[int],
        *,
        primary_horizon: int,
    ) -> None:
        """Keep only configured heads so aux_horizons=[] is one shared single-head path."""
        normalized = tuple(dict.fromkeys(int(value) for value in horizons))
        if not normalized or any(value not in self.supported_horizons for value in normalized):
            raise ValueError(
                f"configured horizons must be a subset of {self.supported_horizons}: {normalized}"
            )
        if primary_horizon not in normalized:
            raise ValueError("primary_horizon must be included in configured horizons")
        active = tuple(f"h{value}" for value in normalized)
        for value in self.supported_horizons:
            name = f"h{value}"
            if name not in active and hasattr(self, f"head_{name}"):
                delattr(self, f"head_{name}")
        self.active_heads = active
        self.primary_head = f"h{primary_horizon}"
        self.supports_multi_horizon = len(active) > 1
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())

    def apply_feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Drop whole fields during training and remain identity in evaluation."""
        if not self.training or self.feature_dropout == 0.0:
            return x
        keep_probability = 1.0 - self.feature_dropout
        keep = (
            torch.rand((1, 1, 1, self.n_fields), device=x.device, dtype=x.dtype) < keep_probability
        )
        return x * keep / keep_probability

    def encode_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Return one temporal vector per stock from the trailing input window."""
        finite = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        finite = self.apply_feature_dropout(finite)
        if self.training and self.random_lookback_min < finite.shape[1]:
            sampled = self._lookback_rng.randint(self.random_lookback_min, finite.shape[1])
            finite = finite[:, -sampled:]
        stocks, days, bars, fields = finite.shape
        daily = self.intra_encoder(finite.reshape(stocks * days, bars, fields))
        daily = daily.reshape(stocks, days, -1)
        encoded, _ = self.temporal_encoder(daily)
        return self.temporal_pool(encoded)

    def score_heads(
        self, representation: torch.Tensor, *, return_all_heads: bool
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return the configured primary head or every active training head."""
        normalized = self.output_norm(representation)
        if not return_all_heads:
            head = getattr(self, f"head_{self.primary_head}")
            return head(normalized).squeeze(-1)
        outputs = {
            name: getattr(self, f"head_{name}")(normalized).squeeze(-1)
            for name in self.active_heads
        }
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_all_heads: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return primary-head scores by default or all heads for multi-task training."""
        expected = (self.lookback_days, self.n_bars, self.n_fields)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"model input trailing shape must be {expected}, got {tuple(x.shape)}")
        temporal = self.encode_temporal(x)
        cross = self.cross_section(temporal, padding_mask)
        return self.score_heads(temporal + self.alpha * cross, return_all_heads=return_all_heads)


def _validate_table_name(table: object, key: str) -> str:
    """Return a safe injected SQL table identifier."""
    if not isinstance(table, str) or not _TABLE_IDENTIFIER.fullmatch(table):
        raise ValueError(f"datasources[{key!r}] must be a safe non-empty table name")
    return table


def _bar_source_key(frequency: object) -> str:
    """Return the injected datasource key for one supported minute frequency."""
    normalized = str(frequency)
    if normalized not in BAR_END_TIMES_BY_FREQUENCY:
        raise ValueError(
            f"unsupported submission frequency {normalized!r}; "
            f"available={sorted(BAR_END_TIMES_BY_FREQUENCY)}"
        )
    return f"bar{normalized}"


def public_bar_table(frequency: object) -> str:
    """Return the public fallback table for one supported frequency."""
    normalized = str(frequency)
    _bar_source_key(normalized)
    return PUBLIC_BAR_TABLES[normalized]


def _resolve_bar_end_times(
    frequency: object,
    values: Sequence[object] | None = None,
) -> tuple[str, ...]:
    """Validate or derive the exact ordered bar-end-time contract."""
    normalized = str(frequency)
    _bar_source_key(normalized)
    expected = tuple(BAR_END_TIMES_BY_FREQUENCY[normalized])
    actual = expected if values is None else tuple(str(value) for value in values)
    if actual != expected:
        raise ValueError(
            f"bar_end_times for {normalized} differ from the exchange-session contract: "
            f"expected={list(expected)}, actual={list(actual)}"
        )
    return actual


def _primary_horizon(contract: Mapping[str, Any]) -> int:
    """Return the integer horizon named by the validated primary-head contract."""
    head = str(contract["primary_head"])
    if not head.startswith("h") or not head[1:].isdigit():
        raise ValueError(f"invalid label_contract.primary_head={head!r}; expected hN")
    return int(head[1:])


def _query(
    sql: str, *, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Execute one DAI query and require a DataFrame result."""
    import dai

    if (start is None) != (end is None):
        raise ValueError("query start and end must be supplied together")
    if start is None:
        result = dai.query(sql).df()
    else:
        result = dai.query(
            sql,
            filters={
                "date": [
                    pd.Timestamp(start).strftime("%Y-%m-%d"),
                    (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d"),
                ]
            },
        ).df()
    if not isinstance(result, pd.DataFrame):
        raise TypeError("dai.query(...).df() must return a pandas DataFrame")
    return result


def _calendar_chunks(
    start: pd.Timestamp, end: pd.Timestamp, days: int
) -> Sequence[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return deterministic inclusive calendar chunks."""
    chunks = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=days - 1))
        chunks.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return chunks


def _instrument_chunks(instruments: Sequence[str], size: int) -> Sequence[list[str]]:
    """Return stable instrument chunks without changing caller order."""
    return [list(instruments[start : start + size]) for start in range(0, len(instruments), size)]


def _sql_string(value: str) -> str:
    """Quote one string literal for an instrument IN clause."""
    return "'" + value.replace("'", "''") + "'"


def _query_bar_chunks(
    table: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    instruments: Sequence[str],
    *,
    query_days: int,
    instrument_chunk: int,
) -> pd.DataFrame:
    """Query projected bars in bounded date and instrument chunks."""
    columns = ["date", "instrument", *FEATURE_COLS]
    parts: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _calendar_chunks(start, end, query_days):
        for instrument_values in _instrument_chunks(instruments, instrument_chunk):
            literals = ", ".join(_sql_string(value) for value in instrument_values)
            sql = f"SELECT {', '.join(columns)} FROM {table} WHERE instrument IN ({literals})"
            frame = _query(sql, start=chunk_start, end=chunk_end)
            if not frame.empty:
                parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def clean_illegal_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Inline canonical cleaning required before any log or log1p transform."""
    missing = set(FEATURE_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"cannot clean a frame missing feature fields: {sorted(missing)}")
    output = frame.copy()
    report: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for field in FEATURE_COLS:
        values = pd.to_numeric(output[field], errors="raise")
        array = values.to_numpy(dtype=np.float64, copy=False)
        inf_mask = np.isinf(array)
        negative_mask = np.isfinite(array) & (array < 0.0)
        illegal_mask = inf_mask | negative_mask
        zero_price_mask = (
            np.isfinite(array) & (array == 0.0)
            if field in PRICE_FIELDS
            else np.zeros(len(array), dtype=bool)
        )
        count = int(illegal_mask.sum())
        total = int(len(array))
        ratio = float(count / total) if total else 0.0
        samples = _flat_illegal_value_samples(output, illegal_mask)
        report[field] = {
            "count": count,
            "total": total,
            "ratio": ratio,
            "inf_count": int(inf_mask.sum()),
            "negative_count": int(negative_mask.sum()),
            "zero_price_count": int(zero_price_mask.sum()),
            "samples": samples,
        }
        if illegal_mask.any():
            output.loc[illegal_mask, field] = np.nan
            detail = (
                f"field={field!r}, count={count}, total={total}, ratio={ratio:.6%}, "
                f"samples={samples}"
            )
            print(f"*** WARNING: ILLEGAL CANONICAL VALUES — {detail}", flush=True)
            if ratio > ILLEGAL_ERROR_RATIO:
                failures.append(detail)
        if zero_price_mask.any():
            output.loc[zero_price_mask, field] = np.nan
    output.attrs["illegal_values"] = report
    if failures:
        raise ValueError(
            f"illegal value ratio exceeds configured {ILLEGAL_ERROR_RATIO:.2%} safety threshold: "
            + "; ".join(failures)
        )
    return output


def _flat_illegal_value_samples(
    frame: pd.DataFrame,
    illegal_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Return identifier-only samples for flattened cleaning diagnostics."""
    if not illegal_mask.any():
        return []
    columns = [
        column for column in ("trading_date", "date", "instrument") if column in frame.columns
    ]
    samples = frame.loc[illegal_mask, columns].drop_duplicates().head(ILLEGAL_SAMPLE_SIZE)
    output: list[dict[str, Any]] = []
    for row in samples.to_dict(orient="records"):
        output.append(
            {
                key: pd.Timestamp(value).isoformat()
                if key in {"trading_date", "date"}
                else str(value)
                for key, value in row.items()
            }
        )
    return output


def _canonicalize_bars(
    raw: pd.DataFrame,
    *,
    frequency: str = FREQUENCY,
    bar_end_times: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply the cloud adapter contract without scaling or feature transforms."""
    resolved_times = _resolve_bar_end_times(frequency, bar_end_times)
    required_order = ["date", "instrument", *FEATURE_COLS]
    missing = set(required_order) - set(raw.columns)
    if missing:
        raise ValueError(f"bar source is missing required columns: {sorted(missing)}")
    actual_feature_cols = [column for column in raw.columns if column in FEATURE_COLS]
    if actual_feature_cols != FEATURE_COLS:
        raise ValueError(
            "actual feature columns must exactly equal FEATURE_COLS in order: "
            f"actual={actual_feature_cols}"
        )
    frame = raw.loc[:, required_order].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["trading_date"] = frame["date"].dt.normalize()
    time_to_sequence = {value: index for index, value in enumerate(resolved_times)}
    actual_times = frame["date"].dt.strftime("%H:%M")
    unknown = sorted(set(actual_times) - set(time_to_sequence))
    if unknown:
        raise ValueError(f"bar timestamps contain unknown {frequency} end times: {unknown}")
    frame["bar_seq"] = actual_times.map(time_to_sequence).astype(np.int8)
    frame["instrument"] = frame["instrument"].astype(str)
    for field in FEATURE_COLS:
        frame[field] = pd.to_numeric(frame[field], errors="raise").astype(np.float32)
    frame = clean_illegal_values(frame)
    duplicated = frame.duplicated(["trading_date", "bar_seq", "instrument"], keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, ["trading_date", "bar_seq", "instrument"]].head(10)
        raise ValueError(f"bar primary key is duplicated: {sample.to_dict(orient='records')}")
    return frame.loc[:, ["trading_date", "bar_seq", "instrument", *FEATURE_COLS]]


def _dense_bars(
    canonical: pd.DataFrame,
    instruments: Sequence[str],
    trading_dates: Sequence[object] | None,
    *,
    frequency: str = FREQUENCY,
    bar_end_times: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densify canonical rows and invalidate every partially missing stock-day."""
    resolved_times = _resolve_bar_end_times(frequency, bar_end_times)
    observed_dates = pd.DatetimeIndex(canonical["trading_date"].unique()).normalize()
    if trading_dates is not None:
        required_dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize()
        dates = observed_dates.union(required_dates).sort_values()
    else:
        dates = observed_dates.sort_values()
    if len(dates) == 0:
        raise ValueError("bar source contains no trading dates")
    instrument_values = list(instruments)
    date_positions = {value: index for index, value in enumerate(dates)}
    instrument_positions = {value: index for index, value in enumerate(instrument_values)}
    x = np.full(
        (len(dates), len(instrument_values), len(resolved_times), FEATURE_COUNT),
        np.nan,
        dtype=np.float32,
    )
    for row in canonical.itertuples(index=False):
        if row.instrument not in instrument_positions:
            continue
        day_index = date_positions[pd.Timestamp(row.trading_date)]
        stock_index = instrument_positions[row.instrument]
        values = np.asarray([getattr(row, field) for field in FEATURE_COLS], dtype=np.float32)
        x[day_index, stock_index, int(row.bar_seq)] = values
    counts = canonical.groupby(["trading_date", "instrument"], sort=False).size()
    mask = np.zeros(x.shape[:2], dtype=bool)
    for (day, instrument), count in counts.items():
        if instrument not in instrument_positions:
            continue
        day_index = date_positions[pd.Timestamp(day)]
        stock_index = instrument_positions[str(instrument)]
        if int(count) == len(resolved_times) and not np.isnan(x[day_index, stock_index]).all():
            mask[day_index, stock_index] = True
        else:
            x[day_index, stock_index] = np.nan
    x[~mask] = np.nan
    return x, mask, dates.to_numpy(dtype="datetime64[D]")


def transform_raw(x_raw: np.ndarray) -> np.ndarray:
    """Apply the fixed per-field log recipe while preserving NaN."""
    output = np.asarray(x_raw, dtype=np.float32).copy()
    if output.shape[-1] != FEATURE_COUNT:
        raise ValueError(f"feature count {output.shape[-1]} != {FEATURE_COUNT}")
    for field_index, field in enumerate(FEATURE_COLS):
        values = output[..., field_index]
        finite = np.isfinite(values)
        if field in PRICE_FIELDS:
            if np.any(values[finite] <= 0.0):
                raise ValueError(f"log transform requires positive values for {field!r}")
            values[finite] = np.log(values[finite])
        else:
            if np.any(values[finite] <= -1.0):
                raise ValueError(f"log1p transform requires values greater than -1 for {field!r}")
            values[finite] = np.log1p(values[finite])
    return output


def fit_preprocess(x_raw: np.ndarray, mask: np.ndarray) -> dict[str, list[float]]:
    """Fit one standard-scaler center and scale per field from mask-true blocks."""
    transformed = transform_raw(x_raw)
    valid_blocks = transformed[np.asarray(mask, dtype=bool)]
    if len(valid_blocks) == 0:
        raise ValueError("training data contains no mask-true stock-day blocks")
    axes = tuple(range(valid_blocks.ndim - 1))
    mean = np.nanmean(valid_blocks, axis=axes, dtype=np.float64).astype(np.float32)
    std = np.nanstd(valid_blocks, axis=axes, dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("preprocess statistics must be finite for all fields")
    std[std <= np.finfo(np.float32).eps] = np.float32(1.0)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_preprocess(x_raw: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Transform, standardize, clip, then fill in the project-defined order."""
    output = transform_raw(x_raw)
    mean = np.asarray(stats.get("mean"), dtype=np.float32)
    std = np.asarray(stats.get("std"), dtype=np.float32)
    if mean.shape != (FEATURE_COUNT,) or std.shape != (FEATURE_COUNT,):
        raise ValueError("preprocess mean/std must each have shape [feature_count]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("preprocess mean/std must be finite and std strictly positive")
    output -= mean
    output /= std
    np.clip(
        output, -float(PREPROCESS_CONTRACT["clip"]), float(PREPROCESS_CONTRACT["clip"]), out=output
    )
    np.copyto(output, np.float32(PREPROCESS_CONTRACT["fill"]), where=~np.isfinite(output))
    return output.astype(np.float32, copy=False)


def build_dataset(
    table: str,
    start: object,
    end: object,
    mode: str,
    instruments: Sequence[str],
    stats: Mapping[str, Any] | None = None,
    *,
    trading_dates: Sequence[object] | None = None,
    frequency: str = FREQUENCY,
    bar_end_times: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Query bounded bars and return dense canonical raw or preprocessed tensors."""
    source_key = _bar_source_key(frequency)
    resolved_times = _resolve_bar_end_times(frequency, bar_end_times)
    table_name = _validate_table_name(table, source_key)
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if pd.isna(start_day) or pd.isna(end_day) or start_day > end_day:
        raise ValueError(f"invalid dataset date range: {start!r} to {end!r}")
    if mode not in {"train", "predict"}:
        raise ValueError("mode must be 'train' or 'predict'")
    instrument_values = [str(value) for value in instruments]
    if not instrument_values or len(instrument_values) != len(set(instrument_values)):
        raise ValueError("instruments must be a non-empty unique ordered sequence")
    query_days = PREDICT_QUERY_DAYS if mode == "predict" else TRAIN_QUERY_DAYS
    instrument_chunk = PREDICT_INSTRUMENT_CHUNK if mode == "predict" else TRAIN_INSTRUMENT_CHUNK
    raw = _query_bar_chunks(
        table_name,
        start_day,
        end_day,
        instrument_values,
        query_days=query_days,
        instrument_chunk=instrument_chunk,
    )
    if raw.empty:
        raise ValueError("bar query returned zero rows")
    canonical = _canonicalize_bars(
        raw,
        frequency=frequency,
        bar_end_times=resolved_times,
    )
    x_raw, mask, dates = _dense_bars(
        canonical,
        instrument_values,
        trading_dates,
        frequency=frequency,
        bar_end_times=resolved_times,
    )
    x = x_raw if stats is None else apply_preprocess(x_raw, stats)
    if x.shape[-1] != FEATURE_COUNT or x.shape[-2] != len(resolved_times):
        raise AssertionError("dataset feature contract differs from the model input dimension")
    return {
        "x": x,
        "x_raw": x_raw,
        "mask": mask,
        "trading_dates": dates,
        "instruments": np.asarray(instrument_values, dtype=object),
        "stats": stats,
        "feature_cols": list(FEATURE_COLS),
        "feature_count": FEATURE_COUNT,
        "frequency": frequency,
        "bar_end_times": list(resolved_times),
        "book_depth": BOOK_DEPTH,
        "raw_row_count": int(len(canonical)),
    }


def _discover_date_range(table: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Discover an injected table's available range without using inference dates."""
    frame = _query(f"SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM {table}")
    required = {"min_date", "max_date"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("date-range query returned no min_date/max_date")
    start = pd.Timestamp(frame.iloc[0]["min_date"]).normalize()
    end = pd.Timestamp(frame.iloc[0]["max_date"]).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError(f"injected bar table has invalid date range: {start} to {end}")
    return start, end


def _query_exposure(table: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Query projected exposure rows across the full selected training range."""
    columns = ["date", "instrument", "ret", *BARRA_COLUMNS]
    parts = [
        _query(f"SELECT {', '.join(columns)} FROM {table}", start=left, end=right)
        for left, right in _calendar_chunks(start, end, 92)
    ]
    exposure = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    missing = set(columns) - set(exposure.columns)
    if missing or exposure.empty:
        raise ValueError(f"exposure source is empty or missing columns: {sorted(missing)}")
    exposure = exposure.loc[:, columns].copy()
    exposure["date"] = pd.to_datetime(exposure["date"], errors="raise").dt.normalize()
    exposure["instrument"] = exposure["instrument"].astype(str)
    for column in ["ret", *BARRA_COLUMNS]:
        exposure[column] = pd.to_numeric(exposure[column], errors="raise")
    duplicates = exposure.duplicated(["date", "instrument"], keep=False)
    if duplicates.any():
        sample = exposure.loc[duplicates, ["date", "instrument"]].head(10)
        raise ValueError(f"exposure keys are duplicated: {sample.to_dict(orient='records')}")
    return exposure.sort_values(["date", "instrument"]).reset_index(drop=True)


def _validate_label_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the config-derived label contract used by private retraining."""
    required = {
        "label_horizon",
        "aux_horizons",
        "ret_alignment",
        "barra_columns",
        "industry_included",
        "regression_weighting",
        "exposure_date",
        "processing_order",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"label_contract is missing keys: {sorted(missing)}")
    normalized = dict(contract)
    horizon = int(normalized["label_horizon"])
    auxiliary = [int(value) for value in normalized["aux_horizons"]]
    heads = {f"h{value}" for value in [*auxiliary, horizon]}
    if not heads or not heads.issubset(MODEL_HEADS):
        raise ValueError(
            f"label horizons {sorted(heads)} must be a non-empty subset of {MODEL_HEADS}"
        )
    if "primary_head" not in normalized:
        # Compatibility is intentionally generic: old 30m JSON predates the explicit field and
        # used label_horizon as the selected head. Every newly saved artifact writes primary_head.
        normalized["primary_head"] = f"h{horizon}"
        warnings.warn(
            "legacy label_contract has no primary_head; deriving it from label_horizon",
            RuntimeWarning,
            stacklevel=2,
        )
    primary = str(normalized["primary_head"])
    if primary not in heads:
        raise ValueError(
            f"label_contract.primary_head={primary!r} is not among configured heads {sorted(heads)}"
        )
    if normalized["ret_alignment"] != "current":
        raise ValueError("ret_alignment must be config-confirmed 'current'")
    if list(normalized["barra_columns"]) != BARRA_COLUMNS:
        raise ValueError("label_contract.barra_columns order differs from BARRA_COLUMNS")
    if bool(normalized["industry_included"]):
        raise ValueError("industry_included must remain false")
    if normalized["regression_weighting"] != "unweighted_ols":
        raise ValueError("only unweighted_ols label residualization is supported")
    if normalized["exposure_date"] != "t":
        raise ValueError("label exposure_date must be prediction day 't'")
    expected_order = [
        "compound_ret_t_plus_1_through_t_plus_h",
        "residualize_on_t_day_barra10",
        "winsorize_cross_section",
        "zscore_cross_section",
    ]
    if list(normalized["processing_order"]) != expected_order:
        raise ValueError(
            "label_contract.processing_order differs from the implemented label pipeline"
        )
    return normalized


def _forward_returns(returns: np.ndarray, horizon: int, alignment: str) -> np.ndarray:
    """Compound t+1 through t+h because exposure.ret[t] is the current return."""
    if alignment != "current":
        raise ValueError("unknown ret alignment would create label leakage")
    result = np.full(returns.shape, np.nan, dtype=np.float64)
    for day_index in range(max(0, len(returns) - horizon)):
        future = returns[day_index + 1 : day_index + horizon + 1]
        complete = np.isfinite(future).all(axis=0)
        result[day_index, complete] = np.prod(1.0 + future[:, complete], axis=0) - 1.0
    return result


def _residualize_styles(values: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Regress one daily cross section on t-day styles and an intercept."""
    output = np.full(len(values), np.nan, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(factors).all(axis=1)
    if int(finite.sum()) < 100:
        return output
    design = np.column_stack((np.ones(int(finite.sum())), factors[finite]))
    coefficients = np.linalg.lstsq(design, values[finite], rcond=None)[0]
    output[finite] = values[finite] - design @ coefficients
    return output


def _postprocess_label(values: np.ndarray) -> np.ndarray:
    """Winsorize then z-score one residual label cross section."""
    output = values.copy()
    finite = np.isfinite(output)
    if not finite.any():
        return output
    lower, upper = np.quantile(output[finite], [0.01, 0.99])
    output[finite] = np.clip(output[finite], lower, upper)
    scale = float(output[finite].std(ddof=0))
    if not np.isfinite(scale) or scale == 0.0:
        output[finite] = np.nan
    else:
        output[finite] = (output[finite] - float(output[finite].mean())) / scale
    return output


def build_labels(
    exposure: pd.DataFrame,
    trading_dates: np.ndarray,
    instruments: Sequence[str],
    label_contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build all config-derived horizons and the daily exposure-universe mask."""
    contract = _validate_label_contract(label_contract)
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize()
    instrument_values = list(instruments)
    date_positions = {value: index for index, value in enumerate(dates)}
    stock_positions = {value: index for index, value in enumerate(instrument_values)}
    returns = np.full((len(dates), len(instrument_values)), np.nan, dtype=np.float64)
    factors = np.full(
        (len(dates), len(instrument_values), len(BARRA_COLUMNS)), np.nan, dtype=np.float64
    )
    universe = np.zeros(returns.shape, dtype=bool)
    for row in exposure.itertuples(index=False):
        day = pd.Timestamp(row.date)
        if day not in date_positions or row.instrument not in stock_positions:
            continue
        day_index = date_positions[day]
        stock_index = stock_positions[row.instrument]
        returns[day_index, stock_index] = float(row.ret)
        factors[day_index, stock_index] = np.asarray(
            [getattr(row, name) for name in BARRA_COLUMNS], dtype=np.float64
        )
        universe[day_index, stock_index] = True
    horizons = [*contract["aux_horizons"], contract["label_horizon"]]
    labels: dict[str, np.ndarray] = {}
    for horizon_value in dict.fromkeys(int(value) for value in horizons):
        values = _forward_returns(returns, horizon_value, str(contract["ret_alignment"]))
        for day_index in range(len(dates) - horizon_value):
            values[day_index] = _postprocess_label(
                _residualize_styles(values[day_index], factors[day_index])
            )
        labels[f"h{horizon_value}"] = values.astype(np.float32)
    return labels, universe


def _weighted_pearson(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return graph-connected weighted Pearson correlation or zero when degenerate."""
    zero = torch.nan_to_num(prediction).sum() * 0.0
    if prediction.numel() < 2:
        return zero
    weight_sum = weights.sum()
    centered_prediction = prediction - (weights * prediction).sum() / weight_sum
    centered_target = target - (weights * target).sum() / weight_sum
    prediction_variance = (weights * centered_prediction.square()).sum()
    target_variance = (weights * centered_target.square()).sum()
    epsilon = torch.finfo(prediction.dtype).eps
    if bool((prediction_variance <= epsilon).item()) or bool((target_variance <= epsilon).item()):
        return zero
    covariance = (weights * centered_prediction * centered_target).sum()
    return covariance / torch.sqrt(prediction_variance * target_variance)


def _head_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return daily negative IC plus the configured MSE anchor."""
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any().item()):
        return torch.nan_to_num(prediction).sum() * 0.0
    selected_prediction = prediction[valid].float()
    selected_target = target[valid].float()
    weights = torch.ones_like(selected_prediction)
    ic = _weighted_pearson(selected_prediction, selected_target, weights)
    mse = (selected_prediction - selected_target).square().mean()
    return -ic + TRAIN_MSE_WEIGHT * mse


class ExponentialMovingAverage:
    """Maintain an EMA of trainable model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        """Snapshot parameters before the first update."""
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend one optimizer update into the EMA state."""
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(parameters[name].detach(), 1.0 - self.decay)

    def state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return a full CPU state dict with EMA trainable parameters."""
        output = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        for name, average in self.shadow.items():
            output[name] = average.detach().cpu().clone()
        return output


def seed_everything(seed: int, *, strict_determinism: bool = False) -> None:
    """Seed all RNGs and optionally trade fused CUDA kernels for bitwise determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not strict_determinism
    torch.backends.cudnn.deterministic = strict_determinism
    torch.use_deterministic_algorithms(strict_determinism)
    if strict_determinism:
        print("*** WARNING: strict determinism enabled: expect 10-100x slowdown ***", flush=True)


def _train_one_seed(
    seed: int,
    device: torch.device,
    x: np.ndarray,
    mask: np.ndarray,
    labels: Mapping[str, np.ndarray],
    *,
    model_cfg: Mapping[str, Any],
    primary_head: str,
    epochs: int,
    loss_weights: Mapping[str, float],
    strict_determinism: bool,
) -> dict[str, torch.Tensor]:
    """Train one seeded model on full daily cross sections and return EMA weights."""
    seed_everything(seed, strict_determinism=strict_determinism)
    normalized_model_cfg = dict(model_cfg)
    model = TCNGRUISAB(**normalized_model_cfg).to(device)
    active_heads = tuple(labels)
    model.configure_output_horizons(
        tuple(int(name.removeprefix("h")) for name in active_heads),
        primary_horizon=_primary_horizon({"primary_head": primary_head}),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    ema = ExponentialMovingAverage(model, TRAIN_EMA_DECAY)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if primary_head not in labels:
        raise ValueError(
            f"primary_head={primary_head!r} is not available in labels {sorted(labels)}"
        )
    lookback_days = int(normalized_model_cfg["lookback_days"])
    usable_dates = [
        day_index
        for day_index in range(lookback_days - 1, len(x))
        if np.any(mask[day_index] & np.isfinite(labels[primary_head][day_index]))
    ]
    if not usable_dates:
        raise ValueError(f"training data has no usable {primary_head} daily cross sections")
    generator = np.random.default_rng(seed)
    total_updates = epochs * len(usable_dates)
    update_index = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for day_index in generator.permutation(usable_dates):
            selected = mask[day_index] & np.isfinite(labels[primary_head][day_index])
            stock_indices = np.flatnonzero(selected)
            window = np.ascontiguousarray(
                x[day_index - lookback_days + 1 : day_index + 1, stock_indices].transpose(
                    1, 0, 2, 3
                )
            )
            model_input = torch.from_numpy(window).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(model_input, return_all_heads=True)
                if not isinstance(predictions, Mapping):
                    raise TypeError("multi-horizon training requires a mapping of model heads")
                loss = torch.zeros((), dtype=torch.float32, device=device)
                for head in active_heads:
                    target = torch.from_numpy(labels[head][day_index, stock_indices]).to(device)
                    loss = loss + float(loss_weights[head]) * _head_loss(predictions[head], target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            update_index += 1
            progress = update_index / max(1, total_updates)
            for group in optimizer.param_groups:
                group["lr"] = TRAIN_LR * 0.5 * (1.0 + math.cos(math.pi * progress))
            ema.update(model)
            losses.append(float(loss.detach().cpu()))
        print(
            f"seed={seed} epoch={epoch + 1}/{epochs} "
            f"train_loss={float(np.mean(losses)):.8f} device={device}"
        )
    return ema.state_dict(model)


def _training_devices(n_seeds: int) -> list[torch.device]:
    """Assign one seed per visible GPU, or serialize all seeds on one device."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [
            torch.device(f"cuda:{index}")
            for index in range(min(n_seeds, torch.cuda.device_count()))
        ]
    return [torch.device("cpu")]


def _train_seed_ensemble(
    seeds: Sequence[int],
    x: np.ndarray,
    mask: np.ndarray,
    labels: Mapping[str, np.ndarray],
    *,
    model_cfg: Mapping[str, Any],
    primary_head: str,
    epochs: int,
    loss_weights: Mapping[str, float],
    strict_determinism: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    """Train GPU-sized waves in parallel and CPU/single-GPU waves serially."""
    devices = _training_devices(len(seeds))
    output: dict[str, dict[str, torch.Tensor]] = {}
    for wave_start in range(0, len(seeds), len(devices)):
        wave = list(seeds[wave_start : wave_start + len(devices)])
        arguments = [(seed, devices[index]) for index, seed in enumerate(wave)]
        if len(arguments) == 1:
            seed, device = arguments[0]
            states = [
                _train_one_seed(
                    seed,
                    device,
                    x,
                    mask,
                    labels,
                    model_cfg=model_cfg,
                    primary_head=primary_head,
                    epochs=epochs,
                    loss_weights=loss_weights,
                    strict_determinism=strict_determinism,
                )
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(arguments)) as executor:
                futures = [
                    executor.submit(
                        _train_one_seed,
                        seed,
                        device,
                        x,
                        mask,
                        labels,
                        model_cfg=model_cfg,
                        primary_head=primary_head,
                        epochs=epochs,
                        loss_weights=loss_weights,
                        strict_determinism=strict_determinism,
                    )
                    for seed, device in arguments
                ]
                states = [future.result() for future in futures]
        for (seed, _device), state in zip(arguments, states, strict=True):
            output[f"seed_{seed}"] = state
    return output


def _encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Encode a tensor as dtype, shape, and flattened JSON data."""
    value = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(value.dtype).replace("torch.", ""),
        "shape": list(value.shape),
        "data": value.reshape(-1).tolist(),
    }


def _decode_tensor(spec: Mapping[str, Any], name: str) -> torch.Tensor:
    """Decode and validate one JSON tensor."""
    if set(spec) != {"dtype", "shape", "data"}:
        raise ValueError(f"tensor {name!r} must contain dtype, shape, and data")
    dtype = getattr(torch, str(spec["dtype"]), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"tensor {name!r} has unsupported dtype {spec['dtype']!r}")
    shape = tuple(int(value) for value in spec["shape"])
    data = spec["data"]
    if not isinstance(data, list) or len(data) != math.prod(shape):
        raise ValueError(f"tensor {name!r} flattened data length differs from shape {shape}")
    return torch.tensor(data, dtype=dtype).reshape(shape)


def _decode_seed_states(payload: Mapping[str, Any]) -> list[dict[str, torch.Tensor]]:
    """Decode every seed state and require the declared ensemble size."""
    encoded = payload.get("state_dict")
    if not isinstance(encoded, Mapping):
        raise TypeError("model state_dict must be a mapping")
    n_seeds = int(payload.get("n_seeds", -1))
    if n_seeds < 1 or len(encoded) != n_seeds:
        raise ValueError(f"state_dict seed count {len(encoded)} != declared n_seeds {n_seeds}")
    states = []
    for seed_name in sorted(encoded):
        layers = encoded[seed_name]
        if not isinstance(layers, Mapping):
            raise TypeError(f"state_dict[{seed_name!r}] must map layers to tensors")
        states.append(
            {
                str(name): _decode_tensor(spec, f"{seed_name}.{name}")
                for name, spec in layers.items()
            }
        )
    return states


def _round_significant(value: float, digits: int) -> float:
    """Round a JSON floating value to a fixed number of significant digits."""
    number = float(value)
    return 0.0 if number == 0.0 else float(f"{number:.{digits}g}")


def _round_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce only floating sequences when the full JSON exceeds 40 MiB."""
    rounded = dict(payload)
    state_dict: dict[str, Any] = {}
    for seed_name, state in payload["state_dict"].items():
        rounded_state = {}
        for layer_name, spec_value in state.items():
            spec = dict(spec_value)
            if str(spec["dtype"]).startswith(("float", "bfloat")):
                spec["data"] = [
                    _round_significant(value, SIGNIFICANT_DIGITS) for value in spec["data"]
                ]
            rounded_state[layer_name] = spec
        state_dict[seed_name] = rounded_state
    rounded["state_dict"] = state_dict
    rounded["mean"] = [_round_significant(value, SIGNIFICANT_DIGITS) for value in payload["mean"]]
    rounded["std"] = [_round_significant(value, SIGNIFICANT_DIGITS) for value in payload["std"]]
    return rounded


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize compact UTF-8 JSON and forbid NaN literals."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def save_model(
    path: str | Path,
    seed_states: Mapping[str, Mapping[str, torch.Tensor]],
    stats: Mapping[str, Any],
    label_contract: Mapping[str, Any],
    *,
    training_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any] | None = None,
    frequency: str = FREQUENCY,
    bar_end_times: Sequence[str] | None = None,
    book_depth: int = BOOK_DEPTH,
) -> dict[str, Any]:
    """Write all seed weights and strict field/label contracts as text JSON."""
    contract = _validate_label_contract(label_contract)
    normalized_frequency = str(frequency)
    resolved_times = _resolve_bar_end_times(normalized_frequency, bar_end_times)
    selected_model_cfg = dict(MODEL_CFG if model_cfg is None else model_cfg)
    if int(selected_model_cfg.get("n_fields", -1)) != FEATURE_COUNT:
        raise ValueError("model_cfg.n_fields differs from the flattened feature contract")
    if int(selected_model_cfg.get("n_bars", -1)) != len(resolved_times):
        raise ValueError("model_cfg.n_bars differs from the selected frequency bar count")
    if int(book_depth) != BOOK_DEPTH:
        raise ValueError("book_depth differs from the flattened five-level feature contract")
    lookback_days = int(selected_model_cfg.get("lookback_days", -1))
    if lookback_days < 1:
        raise ValueError("model_cfg.lookback_days must be positive")
    if not seed_states:
        raise ValueError("at least one seed state is required")
    active_horizons = tuple(
        int(value) for value in [*contract["aux_horizons"], contract["label_horizon"]]
    )
    model = TCNGRUISAB(**selected_model_cfg)
    model.configure_output_horizons(
        active_horizons,
        primary_horizon=_primary_horizon(contract),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    for seed_name, state in seed_states.items():
        model.load_state_dict(state, strict=True)
        if sum(value.numel() for value in state.values()) < parameter_count:
            raise ValueError(f"state {seed_name!r} contains fewer values than model parameters")
    mean = np.asarray(stats.get("mean"), dtype=np.float32)
    std = np.asarray(stats.get("std"), dtype=np.float32)
    if mean.shape != (FEATURE_COUNT,) or std.shape != (FEATURE_COUNT,):
        raise ValueError("model mean/std must each match FEATURE_COUNT")
    payload: dict[str, Any] = {
        "state_dict": {
            str(seed_name): {name: _encode_tensor(tensor) for name, tensor in state.items()}
            for seed_name, state in seed_states.items()
        },
        "model_cfg": selected_model_cfg,
        "feature_cols": list(FEATURE_COLS),
        "feature_count": FEATURE_COUNT,
        "seq_len": lookback_days,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "frequency": normalized_frequency,
        "bar_end_times": list(resolved_times),
        "book_depth": int(book_depth),
        "lookback_days": lookback_days,
        "label_contract": contract,
        "preprocess_contract": dict(PREPROCESS_CONTRACT),
        "training_cfg": dict(training_cfg),
        "parameter_count": parameter_count,
        "n_seeds": len(seed_states),
    }
    encoded = _json_bytes(payload)
    if len(encoded) > FULL_PRECISION_LIMIT_BYTES:
        payload = _round_payload(payload)
        encoded = _json_bytes(payload)
    destination = Path(path)
    destination.write_bytes(encoded)
    return payload


def load_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[list[TCNGRUISAB], dict[str, Any]]:
    """Load all seed models after strict portable data/model contract checks."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "state_dict",
        "model_cfg",
        "feature_cols",
        "feature_count",
        "frequency",
        "book_depth",
        "lookback_days",
        "label_contract",
        "parameter_count",
        "n_seeds",
        "mean",
        "std",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"model JSON is missing keys: {sorted(missing)}")
    saved_fields = list(payload["feature_cols"])
    if saved_fields != FEATURE_COLS or int(payload["feature_count"]) != FEATURE_COUNT:
        raise ValueError("saved feature_cols/feature_count differ from flattened FEATURE_COLS")
    model_cfg = dict(payload["model_cfg"])
    if int(model_cfg.get("n_fields", -1)) != FEATURE_COUNT:
        raise ValueError("saved model input dimension differs from actual feature count")
    frequency = str(payload["frequency"])
    resolved_times = _resolve_bar_end_times(frequency, payload.get("bar_end_times"))
    if int(model_cfg.get("n_bars", -1)) != len(resolved_times):
        raise ValueError("saved model n_bars differs from saved frequency/bar_end_times")
    if int(payload["book_depth"]) != BOOK_DEPTH:
        raise ValueError("saved book_depth differs from the flattened feature contract")
    if int(payload["lookback_days"]) != int(model_cfg.get("lookback_days", -1)):
        raise ValueError("saved lookback_days differs from saved model_cfg.lookback_days")
    if "seq_len" in payload and int(payload["seq_len"]) != int(payload["lookback_days"]):
        raise ValueError("saved seq_len differs from saved lookback_days")
    contract = _validate_label_contract(payload["label_contract"])
    payload["label_contract"] = contract
    payload["bar_end_times"] = list(resolved_times)
    active_horizons = tuple(
        int(value) for value in [*contract["aux_horizons"], contract["label_horizon"]]
    )
    states = _decode_seed_states(payload)
    models = []
    for state in states:
        model = TCNGRUISAB(**model_cfg)
        model.configure_output_horizons(
            active_horizons,
            primary_horizon=_primary_horizon(contract),
        )
        model.load_state_dict(state, strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(payload["parameter_count"]):
            raise ValueError(
                f"loaded parameter_count {parameter_count} != saved {payload['parameter_count']}"
            )
        models.append(model.to(device).eval())
    return models, payload


def _resolve_optional_table(datasources: Mapping[str, Any], key: str, public_table: str) -> str:
    """Prefer an injected auxiliary table and warn before a public-table fallback."""
    value = datasources.get(key)
    if value is None:
        warnings.warn(
            f"{key} not injected; available={list(datasources)}; falling back to {public_table}",
            RuntimeWarning,
            stacklevel=2,
        )
        value = public_table
    return _validate_table_name(value, key)


def train_and_save(
    datasources: Mapping[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    train_start: object | None = None,
    train_end: object | None = None,
) -> str:
    """Retrain on the injected configured-frequency table's full available range."""
    if not isinstance(datasources, Mapping):
        raise TypeError("datasources must be a mapping")
    print(f"datasources keys: {list(datasources.keys())}", flush=True)

    existing_path = Path(model_path)
    if not existing_path.is_file() and not existing_path.is_absolute():
        sibling = Path(__file__).resolve().with_name(existing_path.name)
        if sibling.is_file():
            existing_path = sibling
    if not existing_path.is_file():
        raise FileNotFoundError(
            "train_and_save requires the bundled pretrained JSON to obtain its portable "
            f"model/data/label contract: {existing_path}"
        )
    metadata = json.loads(existing_path.read_text(encoding="utf-8"))
    frequency = str(metadata.get("frequency", ""))
    source_key = _bar_source_key(frequency)
    bar_end_times = _resolve_bar_end_times(frequency, metadata.get("bar_end_times"))
    model_cfg = dict(metadata.get("model_cfg", {}))
    if int(model_cfg.get("n_fields", -1)) != FEATURE_COUNT:
        raise ValueError("bundled model_cfg.n_fields differs from flattened FEATURE_COLS")
    if int(model_cfg.get("n_bars", -1)) != len(bar_end_times):
        raise ValueError("bundled model_cfg.n_bars differs from configured frequency")
    lookback_days = int(model_cfg.get("lookback_days", -1))
    if lookback_days != int(metadata.get("lookback_days", -2)):
        raise ValueError("bundled model/lookback metadata is inconsistent")

    bar_value = datasources.get(source_key)
    if bar_value is None:
        raise KeyError(f"{source_key} missing; available keys = {list(datasources)}")
    bar_table = _validate_table_name(bar_value, source_key)
    exposure_table = _resolve_optional_table(datasources, "exposure", PUBLIC_EXPOSURE_TABLE)
    instruments_table = _resolve_optional_table(
        datasources, "instruments", PUBLIC_INSTRUMENTS_TABLE
    )
    print(
        f"training tables: {source_key}={bar_table}, exposure={exposure_table}, "
        f"instruments={instruments_table}",
        flush=True,
    )
    label_contract = _validate_label_contract(metadata.get("label_contract", {}))
    active_heads = {
        f"h{int(value)}"
        for value in [*label_contract["aux_horizons"], label_contract["label_horizon"]]
    }
    n_seeds = int(metadata.get("n_seeds", 1))
    training_cfg = dict(metadata.get("training_cfg", {}))
    if int(training_cfg.get("n_seeds", n_seeds)) != n_seeds:
        raise ValueError("training_cfg.n_seeds differs from model JSON n_seeds")
    if "seed" not in training_cfg:
        raise ValueError("training_cfg.seed must be saved explicitly in model JSON")
    seed_base = int(training_cfg["seed"])
    strict_determinism = training_cfg.get("strict_determinism", False)
    if not isinstance(strict_determinism, bool):
        raise TypeError("training_cfg.strict_determinism must be boolean")
    epochs = int(training_cfg.get("fixed_epochs") or training_cfg.get("epochs") or TRAIN_EPOCHS)
    raw_loss_weights = training_cfg.get("loss_weights")
    if not isinstance(raw_loss_weights, Mapping):
        raise ValueError("training_cfg.loss_weights must be saved explicitly in model JSON")
    loss_weights = dict(raw_loss_weights)
    if set(loss_weights) != active_heads or not math.isclose(
        sum(float(value) for value in loss_weights.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError(
            "training_cfg.loss_weights must exactly cover configured heads and sum to 1"
        )

    available_start, available_end = _discover_date_range(bar_table)
    start = available_start if train_start is None else pd.Timestamp(train_start).normalize()
    end = available_end if train_end is None else pd.Timestamp(train_end).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError(f"invalid private training range: {start} to {end}")
    if start < available_start or end > available_end:
        raise ValueError(
            f"requested training range {start.date()}..{end.date()} is outside injected range "
            f"{available_start.date()}..{available_end.date()}"
        )

    exposure = _query_exposure(exposure_table, start, end)
    instruments = sorted(exposure["instrument"].unique().tolist())
    trading_dates = pd.DatetimeIndex(exposure["date"].unique()).sort_values()
    dataset = build_dataset(
        bar_table,
        start,
        end,
        "train",
        instruments,
        stats=None,
        trading_dates=trading_dates,
        frequency=frequency,
        bar_end_times=bar_end_times,
    )
    actual_dates = pd.DatetimeIndex(pd.to_datetime(dataset["trading_dates"]))
    if len(actual_dates) == 0 or actual_dates.min() < start or actual_dates.max() > end:
        raise ValueError("built training dataset has empty or out-of-range dates")
    print(
        "private training data: "
        f"min_date={actual_dates.min().date()}, max_date={actual_dates.max().date()}, "
        f"bar_rows={dataset['raw_row_count']}, days={len(actual_dates)}, "
        f"instruments={len(instruments)}",
        flush=True,
    )
    print(
        f"private training history: requested_lookback={lookback_days} trading days, "
        "calendar_buffer=0 days (full injected range), "
        f"actual_history_trading_days={len(actual_dates)}",
        flush=True,
    )
    labels, exposure_mask = build_labels(
        exposure,
        dataset["trading_dates"],
        instruments,
        label_contract,
    )
    training_mask = np.asarray(dataset["mask"], dtype=bool) & exposure_mask
    stats = fit_preprocess(dataset["x_raw"], training_mask)
    x = apply_preprocess(dataset["x_raw"], stats)
    seeds = [seed_base + index for index in range(n_seeds)]
    states = _train_seed_ensemble(
        seeds,
        x,
        training_mask,
        labels,
        model_cfg=model_cfg,
        primary_head=str(label_contract["primary_head"]),
        epochs=epochs,
        loss_weights={name: float(value) for name, value in loss_weights.items()},
        strict_determinism=strict_determinism,
    )
    save_model(
        existing_path,
        states,
        stats,
        label_contract,
        training_cfg={**training_cfg, "epochs_used": epochs, "loss_weights": loss_weights},
        model_cfg=model_cfg,
        frequency=frequency,
        bar_end_times=bar_end_times,
        book_depth=int(metadata.get("book_depth", BOOK_DEPTH)),
    )
    print(
        f"private retraining complete: model={existing_path}, frequency={frequency}, "
        f"primary_head={label_contract['primary_head']}, seeds={seeds}, epochs={epochs}",
        flush=True,
    )
    return str(existing_path)


if __name__ == "__main__":
    raise SystemExit(
        "This module is a platform library. Call train_and_save(datasources, model_path)."
    )
