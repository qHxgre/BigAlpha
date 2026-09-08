"""Frozen hierarchical per-stock causal TCN for WIP22."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral
import re
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn
from torch.nn import functional as F

MINIMAL_15M_RAW_FIELDS = ('adjust_factor', 'high', 'open', 'low', 'close', 'deal_number', 'volume', 'amount', 'ask_price1', 'ask_price2', 'ask_price3', 'bid_price1', 'bid_price2', 'bid_price3', 'ask_volume1', 'ask_volume2', 'ask_volume3', 'bid_volume1', 'bid_volume2', 'bid_volume3', 'ask_num_orders1', 'ask_num_orders2', 'ask_num_orders3', 'bid_num_orders1', 'bid_num_orders2', 'bid_num_orders3')


WIP22_EXPERIMENT_ID = "WIP22-15M-FULLFIELD-HIERARCHICAL-TCN"
WIP22_PARAMETER_COUNT = 121_729
WIP22_EMA_DECAY = 0.999
WIP22_LOOKBACK_DAYS = 20
WIP22_BARS_PER_DAY = 16
WIP22_RAW_FIELD_COUNT = 26


@dataclass(frozen=True, slots=True)
class WIP22TCNConfig:
    """The one preregistered WIP22 architecture and optimizer configuration."""

    lookback_days: int = WIP22_LOOKBACK_DAYS
    bars_per_day: int = WIP22_BARS_PER_DAY
    raw_field_count: int = WIP22_RAW_FIELD_COUNT
    channels: int = 64
    kernel_size: int = 3
    intraday_dilations: tuple[int, int] = (1, 2)
    interday_dilations: tuple[int, int, int, int] = (1, 2, 4, 8)
    head_hidden_dim: int = 64
    dropout: float = 0.0
    learning_rate: float = 3.0e-3
    batch_size: int = 256
    epochs: int = 3
    l2: float = 1.0e-6
    gradient_clip_norm: float = 5.0
    ema_decay: float = WIP22_EMA_DECAY

    def validate(self) -> None:
        frozen = {
            "lookback_days": WIP22_LOOKBACK_DAYS,
            "bars_per_day": WIP22_BARS_PER_DAY,
            "raw_field_count": WIP22_RAW_FIELD_COUNT,
            "channels": 64,
            "kernel_size": 3,
            "intraday_dilations": (1, 2),
            "interday_dilations": (1, 2, 4, 8),
            "head_hidden_dim": 64,
            "dropout": 0.0,
            "learning_rate": 3.0e-3,
            "batch_size": 256,
            "l2": 1.0e-6,
            "gradient_clip_norm": 5.0,
            "ema_decay": WIP22_EMA_DECAY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"WIP22 {name} is frozen at {expected!r}")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise ValueError("WIP22 epochs must be an integer")
        if self.epochs <= 0 or self.epochs > 3:
            raise ValueError("WIP22 epochs must be between one and three")


FROZEN_WIP22_TCN_CONFIG = WIP22TCNConfig()


class _CausalConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = F.pad(values, (self.left_padding, 0))
        receptive_field = self.left_padding + 1
        windows = padded.unfold(2, receptive_field, 1)
        contexts = windows[..., :: self.convolution.dilation[0]]
        batch, channels, length, kernel = contexts.shape
        flattened = contexts.permute(0, 2, 1, 3).reshape(
            batch * length, channels * kernel
        )
        output = F.linear(
            flattened,
            self.convolution.weight.reshape(
                self.convolution.out_channels, -1
            ),
            self.convolution.bias,
        )
        return output.reshape(
            batch, length, self.convolution.out_channels
        ).transpose(1, 2)


class _IntradayEncoder(nn.Module):
    def __init__(self, config: WIP22TCNConfig) -> None:
        super().__init__()
        self.config = config
        self.first = _CausalConv1d(
            config.raw_field_count,
            config.channels,
            kernel_size=config.kernel_size,
            dilation=config.intraday_dilations[0],
        )
        self.second = _CausalConv1d(
            config.channels,
            config.channels,
            kernel_size=config.kernel_size,
            dilation=config.intraday_dilations[1],
        )
        self.normalization = nn.LayerNorm(config.channels)

    def forward_sequence(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("WIP22 network input must be four-dimensional")
        batch, days, bars, fields = values.shape
        if (days, bars, fields) != (
            self.config.lookback_days,
            self.config.bars_per_day,
            self.config.raw_field_count,
        ):
            raise ValueError("WIP22 network input shape changed")
        flattened = values.reshape(batch * days, bars, fields).transpose(1, 2)
        hidden = F.gelu(self.first(flattened))
        hidden = self.second(hidden).transpose(1, 2)
        hidden = F.gelu(self.normalization(hidden))
        return hidden.reshape(batch, days, bars, self.config.channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, days, bars, fields = values.shape
        if (days, bars, fields) != (
            self.config.lookback_days,
            self.config.bars_per_day,
            self.config.raw_field_count,
        ):
            raise ValueError("WIP22 network input shape changed")
        flattened = values.reshape(batch * days, bars, fields)
        first_contexts = torch.stack(
            [
                flattened[:, position - 2 : position + 1, :].transpose(1, 2)
                for position in (11, 13, 15)
            ],
            dim=1,
        )
        first_hidden = F.linear(
            first_contexts.reshape(batch * days * 3, -1),
            self.first.convolution.weight.reshape(self.config.channels, -1),
            self.first.convolution.bias,
        )
        first_hidden = F.gelu(first_hidden).reshape(
            batch * days, 3, self.config.channels
        )
        second_context = first_hidden.transpose(1, 2).reshape(batch * days, -1)
        final_hidden = F.linear(
            second_context,
            self.second.convolution.weight.reshape(self.config.channels, -1),
            self.second.convolution.bias,
        )
        final_hidden = F.gelu(self.normalization(final_hidden))
        return final_hidden.reshape(batch, days, self.config.channels)


class _InterdayResidualBlock(nn.Module):
    def __init__(self, config: WIP22TCNConfig, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.first = _CausalConv1d(
            config.channels,
            config.channels,
            kernel_size=config.kernel_size,
            dilation=dilation,
        )
        self.first_normalization = nn.LayerNorm(config.channels)
        self.second = _CausalConv1d(
            config.channels,
            config.channels,
            kernel_size=config.kernel_size,
            dilation=dilation,
        )
        self.second_normalization = nn.LayerNorm(config.channels)

    @staticmethod
    def _normalize(
        values: torch.Tensor, normalization: nn.LayerNorm
    ) -> torch.Tensor:
        return normalization(values.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def _sparse_convolution(
        values: torch.Tensor,
        input_positions: tuple[int, ...],
        output_positions: tuple[int, ...],
        convolution: _CausalConv1d,
        dilation: int,
    ) -> torch.Tensor:
        position_to_index = {
            position: index for index, position in enumerate(input_positions)
        }
        zero = torch.zeros_like(values[:, 0, :])
        contexts = torch.stack(
            [
                torch.stack(
                    [
                        values[:, position_to_index[source], :]
                        if source in position_to_index
                        else zero
                        for source in (
                            output_position - 2 * dilation,
                            output_position - dilation,
                            output_position,
                        )
                    ],
                    dim=2,
                )
                for output_position in output_positions
            ],
            dim=1,
        )
        batch, positions, channels, kernel = contexts.shape
        output = F.linear(
            contexts.reshape(batch * positions, channels * kernel),
            convolution.convolution.weight.reshape(
                convolution.convolution.out_channels, -1
            ),
            convolution.convolution.bias,
        )
        return output.reshape(
            batch, positions, convolution.convolution.out_channels
        )

    def forward_sparse(
        self,
        values: torch.Tensor,
        input_positions: tuple[int, ...],
        output_positions: tuple[int, ...],
    ) -> torch.Tensor:
        intermediate_positions = tuple(
            sorted(
                {
                    position - offset * self.dilation
                    for position in output_positions
                    for offset in (0, 1, 2)
                    if position - offset * self.dilation >= 0
                }
            )
        )
        first = self._sparse_convolution(
            values,
            input_positions,
            intermediate_positions,
            self.first,
            self.dilation,
        )
        first = F.gelu(self.first_normalization(first))
        second = self._sparse_convolution(
            first,
            intermediate_positions,
            output_positions,
            self.second,
            self.dilation,
        )
        second = F.gelu(self.second_normalization(second))
        index = {position: item for item, position in enumerate(input_positions)}
        residual = torch.stack(
            [values[:, index[position], :] for position in output_positions],
            dim=1,
        )
        return residual + second

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.first(values)
        hidden = F.gelu(self._normalize(hidden, self.first_normalization))
        hidden = self.second(hidden)
        hidden = F.gelu(self._normalize(hidden, self.second_normalization))
        return values + hidden


class _HierarchicalTCNNetwork(nn.Module):
    def __init__(self, config: WIP22TCNConfig) -> None:
        super().__init__()
        self.config = config
        self.intraday = _IntradayEncoder(config)
        self.interday = nn.ModuleList(
            _InterdayResidualBlock(config, dilation)
            for dilation in config.interday_dilations
        )
        self.head_normalization = nn.LayerNorm(config.channels)
        self.head_hidden = nn.Linear(config.channels, config.head_hidden_dim)
        self.head_output = nn.Linear(config.head_hidden_dim, 1)

    def encode_day_sequence(self, values: torch.Tensor) -> torch.Tensor:
        day_tokens = self.intraday(values)
        hidden = day_tokens.transpose(1, 2)
        for block in self.interday:
            hidden = block(hidden)
        return hidden.transpose(1, 2)

    def encode_final_token(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.intraday(values)
        input_positions = tuple(range(self.config.lookback_days))
        output_positions_by_block = (
            tuple(range(1, 20, 2)),
            (3, 7, 11, 15, 19),
            (3, 11, 19),
            (19,),
        )
        for block, output_positions in zip(
            self.interday, output_positions_by_block, strict=True
        ):
            hidden = block.forward_sparse(
                hidden, input_positions, output_positions
            )
            input_positions = output_positions
        return hidden[:, 0, :]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        final_token = self.encode_final_token(values)
        hidden = self.head_normalization(final_token)
        hidden = F.gelu(self.head_hidden(hidden))
        return self.head_output(hidden).squeeze(-1)


BatchLoader = Callable[
    [NDArray[np.int64]], tuple[NDArray[np.float32], NDArray[np.bool_]]
]


class FullFieldHierarchicalPerStockTCN:
    """Train and serve the frozen WIP22 model on CPU with final-step EMA."""

    ARTIFACT_TYPE = "wip22_fullfield_hierarchical_per_stock_tcn"
    SERIALIZATION_VERSION = 1
    scaler_batch_size = 1024
    prediction_batch_size = 2048

    def __init__(
        self,
        *,
        seed: int = 20260713,
        device: str = "cpu",
        config: WIP22TCNConfig = FROZEN_WIP22_TCN_CONFIG,
    ) -> None:
        config.validate()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("WIP22 seed must be an integer")
        if device != "cpu":
            raise ValueError("WIP22 device is frozen at CPU")
        self.seed = seed
        self.device = torch.device(device)
        self.config = config
        self._reset_network()
        if self.parameter_count() != WIP22_PARAMETER_COUNT:
            raise AssertionError("WIP22 TCN parameter count changed")

    def _set_determinism(self) -> None:
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _reset_network(self) -> None:
        self._set_determinism()
        self.network = _HierarchicalTCNNetwork(self.config).to(self.device)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.network.parameters()))

    def trainable_parameter_count(self) -> int:
        return int(
            sum(
                parameter.numel()
                for parameter in self.network.parameters()
                if parameter.requires_grad
            )
        )

    @staticmethod
    def _validate_values(
        X: ArrayLike, mask: ArrayLike | None
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 4 or values.shape[1:] != (
            WIP22_LOOKBACK_DAYS,
            WIP22_BARS_PER_DAY,
            WIP22_RAW_FIELD_COUNT,
        ):
            raise ValueError("WIP22 X must have shape [samples, 20, 16, 6]")
        if not len(values):
            raise ValueError("WIP22 X must contain at least one sample")
        valid = np.isfinite(values)
        if mask is not None:
            declared = np.asarray(mask, dtype=bool)
            if declared.shape != values.shape:
                raise ValueError("WIP22 mask must match X")
            valid &= declared
        return values, np.asarray(valid, dtype=bool)

    def _set_field_scaler(
        self,
        means: NDArray[np.float64],
        scales: NDArray[np.float64],
        counts: NDArray[np.int64],
    ) -> None:
        if means.shape != (26,) or scales.shape != (26,) or counts.shape != (26,):
            raise ValueError("WIP22 field scaler must contain six fields")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)):
            raise ValueError("WIP22 field scaler must be finite")
        if np.any(scales <= 0.0) or np.any(counts < 0):
            raise ValueError("WIP22 field scales and counts are invalid")
        self.field_mean_ = means.copy()
        self.field_scale_ = scales.copy()
        self.field_observation_count_ = counts.copy()

    def _set_target_scaler(self, targets: NDArray[np.float64]) -> None:
        if targets.ndim != 1 or not len(targets) or not np.all(np.isfinite(targets)):
            raise ValueError("WIP22 targets must be finite and one-dimensional")
        self.target_mean_ = float(np.mean(targets, dtype=np.float64))
        scale = float(np.std(targets, dtype=np.float64))
        self.target_scale_ = (
            scale if scale >= np.finfo(np.float64).eps else 1.0
        )

    @staticmethod
    def _field_statistics(
        values: NDArray[np.float32], valid: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        axes = (0, 1, 2)
        counts = valid.sum(axis=axes, dtype=np.int64)
        sums = np.where(valid, values, 0.0).sum(axis=axes, dtype=np.float64)
        means = np.divide(
            sums,
            counts,
            out=np.zeros(WIP22_RAW_FIELD_COUNT, dtype=np.float64),
            where=counts > 0,
        )
        centered = np.where(valid, values - means[None, None, None, :], 0.0)
        squared = np.square(centered).sum(axis=axes, dtype=np.float64)
        variances = np.divide(
            squared,
            counts,
            out=np.zeros(WIP22_RAW_FIELD_COUNT, dtype=np.float64),
            where=counts > 0,
        )
        scales = np.sqrt(np.maximum(variances, 0.0))
        scales[(counts == 0) | (scales < np.finfo(np.float64).eps)] = 1.0
        return means, scales, counts

    def _standardize(
        self, values: NDArray[np.float32], valid: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        standardized = np.where(
            valid,
            (values - self.field_mean_.astype(np.float32)[None, None, None, :])
            / self.field_scale_.astype(np.float32)[None, None, None, :],
            0.0,
        )
        return np.asarray(standardized, dtype=np.float32)

    @staticmethod
    def _reshape_indexed(
        values: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        expected = (WIP22_LOOKBACK_DAYS * WIP22_BARS_PER_DAY, 6)
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError("WIP22 indexed materialization shape changed")
        shape = (len(values), WIP22_LOOKBACK_DAYS, WIP22_BARS_PER_DAY, 6)
        return values.reshape(shape), mask.reshape(shape)

    def _optimize(
        self,
        indices: NDArray[np.int64],
        standardized_targets: NDArray[np.float32],
        loader: BatchLoader,
    ) -> None:
        self._reset_network()
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-7,
        )
        ema_parameters = [
            parameter.detach().clone() for parameter in self.network.parameters()
        ]
        rng = np.random.default_rng(self.seed)
        self.training_history_: list[float] = []
        update_step = 0
        for _ in range(self.config.epochs):
            self.network.train()
            permutation = rng.permutation(indices)
            squared_error = 0.0
            observed = 0
            for start in range(0, len(permutation), self.config.batch_size):
                batch_indices = np.asarray(
                    permutation[start : start + self.config.batch_size],
                    dtype=np.int64,
                )
                values, valid = loader(batch_indices)
                batch = torch.from_numpy(self._standardize(values, valid)).to(
                    self.device
                )
                target = torch.from_numpy(
                    standardized_targets[batch_indices]
                ).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(batch)
                mse = torch.mean(torch.square(prediction - target))
                regularization = torch.zeros((), device=self.device)
                if self.config.l2 > 0.0:
                    regularization = sum(
                        torch.sum(torch.square(parameter))
                        for parameter in self.network.parameters()
                        if parameter.ndim >= 2
                    )
                loss = mse + self.config.l2 * regularization
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.gradient_clip_norm
                )
                optimizer.step()
                with torch.no_grad():
                    for shadow, parameter in zip(
                        ema_parameters, self.network.parameters(), strict=True
                    ):
                        shadow.mul_(self.config.ema_decay).add_(
                            parameter,
                            alpha=1.0 - self.config.ema_decay,
                        )
                update_step += 1
                squared_error += float(mse.detach().cpu()) * len(batch_indices)
                observed += len(batch_indices)
            self.training_history_.append(squared_error / observed)
        with torch.no_grad():
            for parameter, shadow in zip(
                self.network.parameters(), ema_parameters, strict=True
            ):
                parameter.copy_(shadow)
        self.ema_update_steps_ = update_step
        self.ema_finalized_ = True

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        mask: ArrayLike | None = None,
    ) -> "FullFieldHierarchicalPerStockTCN":
        values, valid = self._validate_values(X, mask)
        targets = np.asarray(y, dtype=np.float64)
        if targets.shape != (len(values),) or not np.all(np.isfinite(targets)):
            raise ValueError("WIP22 y must align with X and be finite")
        self._set_field_scaler(*self._field_statistics(values, valid))
        self._set_target_scaler(targets)
        standardized_targets = np.asarray(
            (targets - self.target_mean_) / self.target_scale_, dtype=np.float32
        )
        indices = np.arange(len(values), dtype=np.int64)

        def loader(batch_indices: NDArray[np.int64]) -> tuple[
            NDArray[np.float32], NDArray[np.bool_]
        ]:
            return values[batch_indices], valid[batch_indices]

        self._optimize(indices, standardized_targets, loader)
        return self

    @staticmethod
    def _validate_indices(
        dataset: IndexedMultiDayDataset,
        sample_indices: Sequence[int] | NDArray[np.int64],
    ) -> NDArray[np.int64]:
        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("WIP22 sample indices must be non-empty and one-dimensional")
        if indices.min() < 0 or indices.max() >= len(dataset):
            raise IndexError("WIP22 sample indices are outside the dataset")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("WIP22 sample indices must be unique")
        return indices

    def fit_indexed(
        self,
        dataset: IndexedMultiDayDataset,
        sample_indices: Sequence[int] | NDArray[np.int64],
        training_targets: NDArray[np.float64],
    ) -> "FullFieldHierarchicalPerStockTCN":
        if dataset.lookback_days != WIP22_LOOKBACK_DAYS:
            raise ValueError("WIP22 indexed lookback changed")
        indices = self._validate_indices(dataset, sample_indices)
        targets = np.asarray(training_targets, dtype=np.float64)
        if targets.shape != dataset.y.shape or not np.all(np.isfinite(targets)):
            raise ValueError("WIP22 training targets must align with the dataset")
        counts = np.zeros(6, dtype=np.int64)
        sums = np.zeros(6, dtype=np.float64)
        for start in range(0, len(indices), self.scaler_batch_size):
            values, mask = self._reshape_indexed(
                *dataset.materialize(indices[start : start + self.scaler_batch_size])
            )
            valid = mask & np.isfinite(values)
            counts += valid.sum(axis=(0, 1, 2), dtype=np.int64)
            sums += np.where(valid, values, 0.0).sum(
                axis=(0, 1, 2), dtype=np.float64
            )
        means = np.divide(
            sums,
            counts,
            out=np.zeros(6, dtype=np.float64),
            where=counts > 0,
        )
        squared = np.zeros(6, dtype=np.float64)
        for start in range(0, len(indices), self.scaler_batch_size):
            values, mask = self._reshape_indexed(
                *dataset.materialize(indices[start : start + self.scaler_batch_size])
            )
            valid = mask & np.isfinite(values)
            centered = np.where(
                valid, values - means[None, None, None, :], 0.0
            )
            squared += np.square(centered).sum(
                axis=(0, 1, 2), dtype=np.float64
            )
        variances = np.divide(
            squared,
            counts,
            out=np.zeros(6, dtype=np.float64),
            where=counts > 0,
        )
        scales = np.sqrt(np.maximum(variances, 0.0))
        scales[(counts == 0) | (scales < np.finfo(np.float64).eps)] = 1.0
        self._set_field_scaler(means, scales, counts)
        self._set_target_scaler(targets[indices])
        standardized_targets = np.asarray(
            (targets - self.target_mean_) / self.target_scale_, dtype=np.float32
        )

        def loader(batch_indices: NDArray[np.int64]) -> tuple[
            NDArray[np.float32], NDArray[np.bool_]
        ]:
            return self._reshape_indexed(*dataset.materialize(batch_indices))

        self._optimize(indices, standardized_targets, loader)
        return self

    def _require_fitted(self) -> None:
        required = (
            "field_mean_",
            "field_scale_",
            "target_mean_",
            "target_scale_",
            "ema_finalized_",
        )
        if not all(hasattr(self, name) for name in required):
            raise RuntimeError("WIP22 model must be fitted before use")

    def predict(
        self,
        X: ArrayLike,
        *,
        mask: ArrayLike | None = None,
        batch_size: int | None = None,
    ) -> NDArray[np.float64]:
        self._require_fitted()
        values, valid = self._validate_values(X, mask)
        size = self.prediction_batch_size if batch_size is None else int(batch_size)
        if size <= 0:
            raise ValueError("WIP22 prediction batch size must be positive")
        parts: list[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), size):
                batch = torch.from_numpy(
                    self._standardize(
                        values[start : start + size], valid[start : start + size]
                    )
                ).to(self.device)
                parts.append(self.network(batch).detach().cpu().numpy())
        standardized = np.concatenate(parts).astype(np.float64, copy=False)
        return standardized * self.target_scale_ + self.target_mean_

    def predict_indexed(
        self,
        dataset: IndexedMultiDayDataset,
        sample_indices: Sequence[int] | NDArray[np.int64],
    ) -> NDArray[np.float64]:
        self._require_fitted()
        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("WIP22 sample indices must be one-dimensional")
        parts: list[np.ndarray] = []
        for start in range(0, len(indices), self.prediction_batch_size):
            values, mask = self._reshape_indexed(
                *dataset.materialize(indices[start : start + self.prediction_batch_size])
            )
            parts.append(self.predict(values, mask=mask))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def save(self, path: str | PathLike[str]) -> None:
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": self.ARTIFACT_TYPE,
            "serialization_version": self.SERIALIZATION_VERSION,
            "config": asdict(self.config),
            "seed": self.seed,
            "field_mean": self.field_mean_,
            "field_scale": self.field_scale_,
            "field_observation_count": self.field_observation_count_,
            "target_mean": self.target_mean_,
            "target_scale": self.target_scale_,
            "training_history": self.training_history_,
            "ema_update_steps": self.ema_update_steps_,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.network.state_dict().items()
            },
        }
        torch.save(payload, target)

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        device: str = "cpu",
    ) -> "FullFieldHierarchicalPerStockTCN":
        payload: dict[str, Any] = torch.load(
            Path(path), map_location="cpu", weights_only=False
        )
        if payload.get("artifact_type") != cls.ARTIFACT_TYPE:
            raise ValueError("invalid WIP22 model artifact type")
        if int(payload.get("serialization_version", -1)) != cls.SERIALIZATION_VERSION:
            raise ValueError("unsupported WIP22 model serialization version")
        config = WIP22TCNConfig(**payload["config"])
        model = cls(seed=int(payload["seed"]), device=device, config=config)
        model.network.load_state_dict(payload["state_dict"], strict=True)
        model.network.to(model.device)
        model._set_field_scaler(
            np.asarray(payload["field_mean"], dtype=np.float64),
            np.asarray(payload["field_scale"], dtype=np.float64),
            np.asarray(payload["field_observation_count"], dtype=np.int64),
        )
        model.target_mean_ = float(payload["target_mean"])
        model.target_scale_ = float(payload["target_scale"])
        model.training_history_ = [float(item) for item in payload["training_history"]]
        model.ema_update_steps_ = int(payload["ema_update_steps"])
        model.ema_finalized_ = True
        if model.parameter_count() != WIP22_PARAMETER_COUNT:
            raise ValueError("loaded WIP22 parameter count changed")
        return model


__all__ = [
    "FROZEN_WIP22_TCN_CONFIG",
    "FullFieldHierarchicalPerStockTCN",
    "WIP22TCNConfig",
    "WIP22_EMA_DECAY",
    "WIP22_EXPERIMENT_ID",
    "WIP22_PARAMETER_COUNT",
]


RAW_FIELDS = ('adjust_factor', 'high', 'open', 'low', 'close', 'deal_number', 'volume', 'amount', 'ask_price1', 'ask_price2', 'ask_price3', 'bid_price1', 'bid_price2', 'bid_price3', 'ask_volume1', 'ask_volume2', 'ask_volume3', 'bid_volume1', 'bid_volume2', 'bid_volume3', 'ask_num_orders1', 'ask_num_orders2', 'ask_num_orders3', 'bid_num_orders1', 'bid_num_orders2', 'bid_num_orders3')
OUTPUT_COLUMNS = ("date", "instrument", "score")
BAR_END_TIMES = (
    "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30",
    "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00",
)
BARS_PER_DAY = len(BAR_END_TIMES)
MODEL_FILENAME = "wip22_model_15m.json"
OFFICIAL_INSTRUMENTS_TABLE = "bigalpha_2026_instruments"
MAX_MISSING_RATIO = 0.40
HISTORY_CALENDAR_DAYS = 240
_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SubmissionError(ValueError):
    """Raised when platform data or model bytes violate the frozen contract."""


def _as_array(payload, name, dtype, shape):
    if not isinstance(payload, dict) or set(payload) != {"dtype", "shape", "data"}:
        raise SubmissionError(f"invalid array metadata for {name}")
    if payload["dtype"] != str(np.dtype(dtype)) or payload["shape"] != list(shape):
        raise SubmissionError(f"invalid array shape or dtype for {name}")
    values = np.asarray(payload["data"], dtype=dtype)
    if values.ndim != 1 or values.size != math.prod(shape):
        raise SubmissionError(f"invalid array length for {name}")
    return values.reshape(shape).copy()


def _array_payload(values):
    array = np.asarray(values)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.reshape(-1).tolist(),
    }


def _model_payload(model):
    model._require_fitted()
    return {
        "artifact_type": model.ARTIFACT_TYPE,
        "serialization_version": model.SERIALIZATION_VERSION,
        "contract": {
            "frequency_minutes": 15,
            "lookback_days": WIP22_LOOKBACK_DAYS,
            "bars_per_day": WIP22_BARS_PER_DAY,
            "raw_field_names": list(RAW_FIELDS),
            "parameter_count": WIP22_PARAMETER_COUNT,
            "uses_instrument_identifier": False,
            "maximum_training_year": 2023,
        },
        "architecture": asdict(model.config),
        "state": {
            "seed": model.seed,
            "field_mean": _array_payload(model.field_mean_),
            "field_scale": _array_payload(model.field_scale_),
            "field_observation_count": _array_payload(model.field_observation_count_),
            "target_mean": model.target_mean_,
            "target_scale": model.target_scale_,
            "training_history": list(model.training_history_),
            "ema_update_steps": model.ema_update_steps_,
            "network": {
                name: _array_payload(value.detach().cpu().numpy())
                for name, value in sorted(model.network.state_dict().items())
            },
        },
    }


def save_model(model, path):
    if model.parameter_count() != WIP22_PARAMETER_COUNT:
        raise SubmissionError("WIP22 parameter count changed")
    Path(path).write_text(
        json.dumps(
            _model_payload(model),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )


def load_model(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SubmissionError("unable to read the WIP22 UTF-8 JSON model") from error
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_type", "serialization_version", "contract", "architecture", "state"
    }:
        raise SubmissionError("invalid WIP22 model top-level schema")
    if (
        payload["artifact_type"] != FullFieldHierarchicalPerStockTCN.ARTIFACT_TYPE
        or payload["serialization_version"] != FullFieldHierarchicalPerStockTCN.SERIALIZATION_VERSION
    ):
        raise SubmissionError("unsupported WIP22 model artifact")
    expected_contract = {
        "frequency_minutes": 15,
        "lookback_days": WIP22_LOOKBACK_DAYS,
        "bars_per_day": WIP22_BARS_PER_DAY,
        "raw_field_names": list(RAW_FIELDS),
        "parameter_count": WIP22_PARAMETER_COUNT,
        "uses_instrument_identifier": False,
        "maximum_training_year": 2023,
    }
    if payload["contract"] != expected_contract:
        raise SubmissionError("WIP22 model violates the frozen contract")
    architecture = dict(payload["architecture"])
    architecture["intraday_dilations"] = tuple(architecture["intraday_dilations"])
    architecture["interday_dilations"] = tuple(architecture["interday_dilations"])
    state = payload["state"]
    if not isinstance(state, dict) or set(state) != {
        "seed", "field_mean", "field_scale", "field_observation_count",
        "target_mean", "target_scale", "training_history", "ema_update_steps", "network"
    }:
        raise SubmissionError("invalid WIP22 model state")
    model = FullFieldHierarchicalPerStockTCN(
        seed=int(state["seed"]), device="cpu", config=WIP22TCNConfig(**architecture)
    )
    expected = model.network.state_dict()
    network = state["network"]
    if not isinstance(network, dict) or set(network) != set(expected):
        raise SubmissionError("WIP22 network state keys changed")
    restored = {
        name: torch.from_numpy(
            _as_array(network[name], name, np.float32, tuple(value.shape))
        )
        for name, value in expected.items()
    }
    model.network.load_state_dict(restored, strict=True)
    model._set_field_scaler(
        _as_array(state["field_mean"], "field_mean", np.float64, (26,)),
        _as_array(state["field_scale"], "field_scale", np.float64, (26,)),
        _as_array(
            state["field_observation_count"], "field_observation_count", np.int64, (26,)
        ),
    )
    model.target_mean_ = float(state["target_mean"])
    model.target_scale_ = float(state["target_scale"])
    model.training_history_ = [float(value) for value in state["training_history"]]
    model.ema_update_steps_ = int(state["ema_update_steps"])
    model.ema_finalized_ = True
    return model
def _normalize_date(value, name):
    try:
        if isinstance(value, Integral) and not isinstance(value, bool):
            parsed = pd.to_datetime(str(int(value)), format="%Y%m%d")
        else:
            parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SubmissionError(f"invalid {name}") from error
    if pd.isna(parsed):
        raise SubmissionError(f"invalid {name}")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def _derive_15m_table(source):
    if not isinstance(source, str) or source.count("bar1m") != 1:
        raise SubmissionError("bar1m fallback must be one official physical table name")
    table = source.replace("bar1m", "bar15m")
    if not _TABLE_NAME.fullmatch(table):
        raise SubmissionError("unsafe derived table name")
    return table


def _select_source(datasources):
    if isinstance(datasources, pd.DataFrame):
        return datasources
    if isinstance(datasources, str):
        if "bar15m" in datasources:
            return datasources
        if "bar1m" in datasources:
            return _derive_15m_table(datasources)
        raise SubmissionError("physical datasource is not the official 15-minute table")
    if not isinstance(datasources, Mapping):
        raise SubmissionError("datasources must map keys to physical tables")
    for key in ("bar15m", "bar_15m", "stock_bar15m"):
        if key in datasources:
            source = datasources[key]
            if isinstance(source, str) and "bar15m" not in source:
                raise SubmissionError("15-minute key points to a wrong table")
            return source
    candidates = [key for key in datasources if "15m" in str(key).lower() or "15min" in str(key).lower()]
    if len(candidates) == 1:
        source = datasources[candidates[0]]
        if isinstance(source, str) and "bar15m" not in source:
            raise SubmissionError("15-minute key points to a wrong table")
        return source
    one_minute = [key for key in datasources if "bar1m" in str(key).lower()]
    if len(one_minute) == 1:
        source = datasources[one_minute[0]]
        if isinstance(source, str) and "bar15m" in source:
            return source
        return _derive_15m_table(source)
    raise SubmissionError("no unambiguous 15-minute datasource")


def _default_query(sql, *, filters, compression):
    try:
        import dai
    except ImportError:
        try:
            from bigquant import dai
        except ImportError as error:
            raise SubmissionError("BigQuant dai is required for physical table names") from error
    result = dai.query(sql, filters=dict(filters), compression=compression)
    return result.df() if hasattr(result, "df") else result


def _query_table(table, columns, start, end):
    if not isinstance(table, str) or not _TABLE_NAME.fullmatch(table):
        raise SubmissionError("unsafe physical table name")
    sql = f"SELECT {', '.join(columns)} FROM {table} ORDER BY instrument, date"
    result = _default_query(
        sql,
        filters={
            "date": [
                start.strftime("%Y-%m-%d 00:00:00"),
                end.strftime("%Y-%m-%d 23:59:59"),
            ]
        },
        compression=True,
    )
    if hasattr(result, "df") and callable(result.df):
        result = result.df()
    if not isinstance(result, pd.DataFrame):
        raise SubmissionError("dai query did not return a DataFrame")
    return result.copy(deep=True)


def _materialize_bars(source, start, end):
    if isinstance(source, str):
        return _query_table(source, ("date", "instrument", *RAW_FIELDS), start, end)
    current = source
    for _ in range(3):
        if isinstance(current, pd.DataFrame):
            return current.copy(deep=True)
        method = getattr(current, "to_pandas", None) or getattr(current, "read", None)
        if not callable(method):
            break
        updated = method()
        if updated is current:
            break
        current = updated
    raise SubmissionError("15-minute datasource did not materialize to a DataFrame")


def _parse_timestamps(values):
    if pd.api.types.is_numeric_dtype(values):
        text = values.astype("Int64").astype("string")
        if bool(text.str.fullmatch(r"\d{8}", na=False).all()):
            return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    # Official tables use one uniform timestamp representation per query.
    # Omitting the pandas-2-only ``format="mixed"`` keeps the upload runtime
    # compatible with older BigQuant images as well.
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _prepare_bars(frame, start, end):
    required = {"date", "instrument", *RAW_FIELDS}
    if not required.issubset(frame.columns) or frame.empty:
        raise SubmissionError("15-minute datasource is empty or missing OHLCV/amount")
    bars = frame.loc[:, ["date", "instrument", *RAW_FIELDS]].copy()
    bars["_timestamp"] = _parse_timestamps(bars["date"])
    if bars["_timestamp"].isna().any():
        raise SubmissionError("15-minute datasource contains invalid timestamps")
    bars["instrument"] = bars["instrument"].astype("string").str.strip()
    if bars["instrument"].isna().any() or bars["instrument"].eq("").any():
        raise SubmissionError("15-minute datasource contains invalid instruments")
    bars["_trade_date"] = bars["_timestamp"].dt.normalize()
    bars = bars.loc[bars["_trade_date"].between(start, end)].copy()
    if bars.empty:
        raise SubmissionError("no 15-minute rows in the requested range")
    if bars.duplicated(["_timestamp", "instrument"]).any():
        raise SubmissionError("duplicate timestamp/instrument bars")
    slot_map = {clock: index for index, clock in enumerate(BAR_END_TIMES)}
    times = bars["_timestamp"].dt.strftime("%H:%M")
    if (~times.isin(slot_map)).any():
        raise SubmissionError("bars are not on the official 15-minute grid")
    bars["_slot"] = times.map(slot_map).astype(np.int16)
    deltas = (
        bars.sort_values(["instrument", "_trade_date", "_timestamp"])
        .groupby(["instrument", "_trade_date"], observed=True)["_timestamp"]
        .diff().dt.total_seconds().div(60).dropna()
    )
    if deltas.empty or not bool(np.isclose(deltas, 15.0).any()):
        raise SubmissionError("15-minute cadence could not be verified")
    for field in RAW_FIELDS:
        bars[field] = pd.to_numeric(bars[field], errors="coerce")
    values = bars.loc[:, list(RAW_FIELDS)].to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    positions = {field: index for index, field in enumerate(RAW_FIELDS)}
    for field in ("high", "open", "low", "close"):
        column = positions[field]
        values[values[:, column] <= 0.0, column] = np.nan
    for field in RAW_FIELDS:
        if field not in ("high", "open", "low", "close"):
            column = positions[field]
            values[values[:, column] < 0.0, column] = np.nan
    bars.loc[:, list(RAW_FIELDS)] = values
    return bars.sort_values(["_trade_date", "instrument", "_slot"], kind="mergesort")


def _prepare_universe(frame, start, end):
    if not {"date", "instrument"}.issubset(frame.columns):
        raise SubmissionError("official universe is missing date/instrument")
    universe = frame.loc[:, ["date", "instrument"]].copy()
    universe["_date"] = _parse_timestamps(universe["date"]).dt.normalize()
    universe["instrument"] = universe["instrument"].astype("string").str.strip()
    if universe["_date"].isna().any() or universe["instrument"].isna().any() or universe["instrument"].eq("").any():
        raise SubmissionError("official universe contains invalid keys")
    universe = universe.loc[universe["_date"].between(start, end)].copy()
    universe["date"] = universe["_date"].dt.strftime("%Y-%m-%d")
    universe = universe.loc[:, ["date", "instrument"]]
    if universe.empty or universe.duplicated(["date", "instrument"]).any():
        raise SubmissionError("official universe is empty or has duplicate keys")
    return universe.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)


def _validate_output(frame, universe):
    if tuple(frame.columns) != OUTPUT_COLUMNS or frame.empty:
        raise SubmissionError("output must contain exactly date, instrument, score")
    if frame.duplicated(["date", "instrument"]).any():
        raise SubmissionError("output contains duplicate keys")
    score = pd.to_numeric(frame["score"], errors="coerce")
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise SubmissionError("output scores must be finite")
    expected = universe.loc[:, ["date", "instrument"]].drop_duplicates()
    submitted = frame.loc[:, ["date", "instrument"]].drop_duplicates()
    extra = submitted.merge(expected, how="left", on=["date", "instrument"], indicator=True)
    if (extra["_merge"] == "left_only").any():
        raise SubmissionError("output contains keys outside the official universe")
    aligned = expected.merge(submitted.assign(_present=True), how="left", on=["date", "instrument"])
    per_date = aligned.groupby("date", sort=True)["_present"].agg(["size", "count"])
    if ((per_date["size"] - per_date["count"]) / per_date["size"] > MAX_MISSING_RATIO + 1e-12).any():
        raise SubmissionError("output coverage exceeds the maximum missing ratio")
    result = frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.strftime("%Y-%m-%d")
    result["instrument"] = result["instrument"].astype("string").str.strip()
    result["score"] = score
    return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)




@dataclass(frozen=True)
class _IndexedBars:
    daily_values: np.ndarray
    daily_mask: np.ndarray
    window_starts: np.ndarray
    y: np.ndarray
    feature_dates: np.ndarray
    lookback_days: int = WIP22_LOOKBACK_DAYS

    def __len__(self):
        return int(len(self.window_starts))

    def materialize(self, sample_indices):
        indices = np.asarray(sample_indices, dtype=np.int64)
        starts = self.window_starts[indices]
        positions = starts[:, None] + np.arange(self.lookback_days, dtype=np.int64)
        values = self.daily_values[positions].reshape(
            len(indices), self.lookback_days * BARS_PER_DAY, len(RAW_FIELDS)
        )
        mask = self.daily_mask[positions].reshape(values.shape)
        return values, mask


def _daily_cross_section_target(raw_targets, feature_dates):
    targets = np.asarray(raw_targets, dtype=np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime(feature_dates, errors="coerce")).normalize()
    if targets.ndim != 1 or dates.shape[0] != targets.shape[0] or dates.isna().any():
        raise SubmissionError("WIP22 training targets and dates must align")
    transformed = np.empty_like(targets)
    for date in dates.unique():
        positions = np.flatnonzero(dates == date)
        if positions.size < 2:
            raise SubmissionError("WIP22 requires two labels per feature date")
        day = targets[positions]
        lower, upper = np.quantile(day, (0.01, 0.99), method="linear")
        clipped = np.clip(day, lower, upper)
        scale = float(np.std(clipped, ddof=0, dtype=np.float64))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            raise SubmissionError("WIP22 daily target scale must be positive")
        transformed[positions] = (clipped - float(np.mean(clipped))) / scale
    return transformed


def _daily_matrices(bars, max_instruments):
    selected = sorted(bars["instrument"].astype(str).unique())[: int(max_instruments)]
    bars = bars.loc[bars["instrument"].isin(selected)].copy()
    empty = np.full((BARS_PER_DAY, len(RAW_FIELDS)), np.nan, dtype=np.float32)
    for instrument, instrument_bars in bars.groupby(
        "instrument", sort=True, observed=True
    ):
        daily = []
        for date, group in instrument_bars.groupby(
            "_trade_date", sort=True, observed=True
        ):
            matrix = empty.copy()
            matrix[group["_slot"].to_numpy(dtype=np.int64)] = group.loc[
                :, list(RAW_FIELDS)
            ].to_numpy(dtype=np.float32)
            daily.append((pd.Timestamp(date), matrix))
        yield str(instrument), daily


def _build_indexed_training_data(bars, max_instruments):
    daily_values, window_starts, targets, feature_dates = [], [], [], []
    for _, daily in _daily_matrices(bars, max_instruments):
        base = len(daily_values)
        daily_values.extend(matrix for _, matrix in daily)
        for final in range(WIP22_LOOKBACK_DAYS - 1, len(daily) - 1):
            current_close = float(daily[final][1][-1, RAW_FIELDS.index("close")])
            next_close = float(daily[final + 1][1][-1, RAW_FIELDS.index("close")])
            if np.isfinite(current_close) and current_close > 0.0 and np.isfinite(next_close):
                target = next_close / current_close - 1.0
                if np.isfinite(target):
                    window_starts.append(base + final - WIP22_LOOKBACK_DAYS + 1)
                    targets.append(target)
                    feature_dates.append(daily[final][0])
    if len(targets) < 2:
        raise SubmissionError("WIP22 training data produced fewer than two windows")
    values = np.asarray(daily_values, dtype=np.float32)
    starts = np.asarray(window_starts, dtype=np.int64)
    y = np.asarray(targets, dtype=np.float64)
    dates = np.asarray(feature_dates, dtype="datetime64[ns]")
    order = np.argsort(dates, kind="stable")
    return _IndexedBars(
        daily_values=values,
        daily_mask=np.isfinite(values),
        window_starts=starts[order],
        y=y[order],
        feature_dates=dates[order],
    )


@dataclass(frozen=True)
class _InferenceTarget:
    date: str
    instrument: str
    sequence: np.ndarray


def _targets_from_history(bars, universe):
    by_instrument = dict(_daily_matrices(bars, 10_000_000))
    targets = []
    for row in universe.itertuples(index=False):
        requested = pd.Timestamp(row.date).normalize()
        daily = by_instrument.get(str(row.instrument), ())
        positions = {date.normalize(): index for index, (date, _) in enumerate(daily)}
        final = positions.get(requested)
        if final is None or final + 1 < WIP22_LOOKBACK_DAYS:
            continue
        start = final - WIP22_LOOKBACK_DAYS + 1
        values = np.stack([daily[index][1] for index in range(start, final + 1)])
        targets.append(
            _InferenceTarget(
                requested.strftime("%Y-%m-%d"), str(row.instrument), values
            )
        )
    if not targets:
        raise SubmissionError("no WIP22 20-day prediction targets were built")
    return targets


def _model_path():
    adjacent = Path(__file__).resolve().with_name(MODEL_FILENAME)
    return adjacent if adjacent.is_file() else Path(MODEL_FILENAME)


def _run(datasources, start_date, end_date):
    start = _normalize_date(start_date, "start_date")
    end = _normalize_date(end_date, "end_date")
    if start > end:
        raise SubmissionError("start_date must be on or before end_date")
    history_start = start - pd.Timedelta(days=HISTORY_CALENDAR_DAYS)
    source = _select_source(datasources)
    bars = _prepare_bars(
        _materialize_bars(source, history_start, end), history_start, end
    )
    universe = _prepare_universe(
        _query_table(
            OFFICIAL_INSTRUMENTS_TABLE, ("date", "instrument"), start, end
        ),
        start,
        end,
    )
    targets = _targets_from_history(bars, universe)
    model = load_model(_model_path())
    rows = []
    for offset in range(0, len(targets), 512):
        batch = targets[offset : offset + 512]
        values = np.stack([target.sequence for target in batch]).astype(np.float32)
        scores = model.predict(values, mask=np.isfinite(values))
        rows.extend(
            (target.date, target.instrument, float(score))
            for target, score in zip(batch, scores)
        )
    return _validate_output(
        pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS)), universe
    )


def train_and_save(
    datasources,
    model_path=MODEL_FILENAME,
    train_start="2019-01-01",
    train_end="2023-12-31",
    max_instruments=10000,
    epochs=3,
):
    """Private-board entry: reproduce the frozen 2019-2023 WIP22 fit."""

    start = _normalize_date(train_start, "train_start")
    end = _normalize_date(train_end, "train_end")
    if (
        start != pd.Timestamp("2019-01-01")
        or end != pd.Timestamp("2023-12-31")
        or int(max_instruments) != 10000
        or int(epochs) != 3
    ):
        raise SubmissionError("WIP22 private training configuration is frozen")
    source = _select_source(datasources)
    bars = _prepare_bars(_materialize_bars(source, start, end), start, end)
    dataset = _build_indexed_training_data(bars, max_instruments)
    targets = _daily_cross_section_target(dataset.y, dataset.feature_dates)
    model = FullFieldHierarchicalPerStockTCN(seed=20260713, device="cpu")
    model.fit_indexed(
        dataset, np.arange(len(dataset), dtype=np.int64), targets
    )
    save_model(model, model_path)
    return {
        "status": "trained",
        "sample_count": int(len(dataset)),
        "parameter_count": model.parameter_count(),
        "model_path": Path(model_path).name,
        "training_year_minimum": 2019,
        "training_year_maximum": 2023,
    }


def main(datasources, start_date, end_date):
    """Official BigQuant evaluator entry point; returns exactly three columns."""
    return _run(datasources, start_date, end_date)


__all__ = ["main", "train_and_save"]
