# ===================================================================
# core.py: 配置加载 / 数据 IO / 特征变换 / 指标 / 分布式工具（合并）
# ===================================================================
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


def date_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def dataset_for_table(data_root: str | Path, table_name: str):
    return ds.dataset(Path(data_root) / table_name, format="parquet", partitioning="hive")


def read_table_filtered(
    data_root: str | Path,
    table_name: str,
    columns: list[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    instruments: list[int] | list[str] | None = None,
) -> pd.DataFrame:
    dataset = dataset_for_table(data_root, table_name)
    flt = (ds.field("date") >= start) & (ds.field("date") < end_exclusive)
    if instruments is not None:
        names = set(dataset.schema.names)
        if "instrument" in names and any(isinstance(value, str) for value in instruments):
            values = [str(value) for value in instruments]
            flt = flt & ds.field("instrument").isin(values)
        elif "instrument_id" in names:
            values = [int(value) for value in instruments]
            flt = flt & ds.field("instrument_id").isin(values)
        else:
            values = [str(value) for value in instruments]
            flt = flt & ds.field("instrument").isin(values)
    table = dataset.to_table(columns=columns, filter=flt)
    frame = table.to_pandas()
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
        if "instrument" in frame.columns:
            frame["instrument"] = frame["instrument"].astype(str)
        if "instrument_id" in frame.columns:
            frame["instrument_id"] = frame["instrument_id"].astype(int)
    return frame


def load_daily_frame(
    data_root: str | Path,
    table_name: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    columns: list[str],
) -> pd.DataFrame:
    frame = read_table_filtered(data_root, table_name, columns, start, end_exclusive)
    if frame.empty:
        return frame
    frame["day"] = frame["date"].dt.normalize()
    return frame


# BigAlpha e2e bars: 3-level order book, integer prices, instrument_id.
ID_COLS = ["date", "instrument_id"]
PRICE_BASE_COLS = ["open", "high", "low", "close"]
ASK_PRICE_COLS = [f"ask_price{i}" for i in range(1, 4)]
BID_PRICE_COLS = [f"bid_price{i}" for i in range(1, 4)]
ASK_VOLUME_COLS = [f"ask_volume{i}" for i in range(1, 4)]
BID_VOLUME_COLS = [f"bid_volume{i}" for i in range(1, 4)]
ASK_ORDER_COLS = [f"ask_num_orders{i}" for i in range(1, 4)]
BID_ORDER_COLS = [f"bid_num_orders{i}" for i in range(1, 4)]
NUM_ORDER_COLS = ASK_ORDER_COLS + BID_ORDER_COLS
TRADE_COLS = ["deal_number", "volume", "amount"]
ADJUST_COLS = ["adjust_factor"]

BAR_COLS_3LEVEL = (
    ID_COLS
    + ADJUST_COLS
    + PRICE_BASE_COLS
    + TRADE_COLS
    + ASK_PRICE_COLS
    + BID_PRICE_COLS
    + ASK_VOLUME_COLS
    + BID_VOLUME_COLS
    + NUM_ORDER_COLS
)


PRICE_SCALE = 100.0
SCALE_FIELDS = [
    "open", "high", "low", "close", "amount",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
OHLC_COLS = ["open", "high", "low", "close"]


def to_canonical(df: pd.DataFrame, *, is_local: bool, keep_five_levels: bool = True) -> pd.DataFrame:
    """本地压缩 e2e 表 / 云端原始表 -> 同一份 canonical 表示。

    is_local=True  ：分 -> 元、OHLC -1 -> NaN、instrument_id 作为分组键。
    is_local=False ：云端原始表（元、字符串 instrument）；keep_five_levels=False
                     时丢弃 4/5 档以对齐本地 3 档。
    """
    df = df.copy()
    if is_local:
        for c in OHLC_COLS:
            if c in df.columns:
                df.loc[df[c] == -1, c] = np.nan
        for c in SCALE_FIELDS:
            if c in df.columns:
                df[c] = df[c].astype("float64") / PRICE_SCALE
        df["key"] = df["instrument_id"]
    else:
        if not keep_five_levels:
            drop_cols = [
                c for c in df.columns
                if any(
                    c.startswith(p) and c[-1] in "45"
                    for p in ("ask_price", "bid_price", "ask_volume",
                              "bid_volume", "ask_num_orders", "bid_num_orders")
                )
            ]
            df = df.drop(columns=drop_cols, errors="ignore")
        df["key"] = df["instrument"]
    return df


import numpy as np
import pandas as pd


class FeatureTransform:
    """Deterministic transforms for BigAlpha stock 1-minute bars."""

    def __init__(self, config: dict):
        feature_cfg = config["features"]
        self.price_cols = list(feature_cfg["price_cols"])
        self.volume_cols = list(feature_cfg["volume_cols"])
        self.num_order_cols = list(feature_cfg.get("num_order_cols", []))
        if not feature_cfg.get("use_num_orders", False):
            self.num_order_cols = []
        self.required_cols = (
            ["date", "instrument", "pre_close"]
            + self.price_cols
            + self.volume_cols
            + self.num_order_cols
        )
        self.feature_names = (
            [f"{col}_rel_pre_close" for col in self.price_cols]
            + [f"log1p_{col}" for col in self.volume_cols]
            + [f"log1p_{col}" for col in self.num_order_cols]
        )

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def frame_to_features(self, frame: pd.DataFrame) -> np.ndarray:
        pre_close = frame["pre_close"].astype("float32").replace(0, np.nan).to_numpy()[:, None]
        price = frame[self.price_cols].astype("float32").to_numpy()
        price_rel = price / pre_close - 1.0

        volume = frame[self.volume_cols].clip(lower=0).astype("float32").to_numpy()
        volume_log = np.log1p(volume)
        pieces = [price_rel, volume_log]
        if self.num_order_cols:
            order_count = frame[self.num_order_cols].clip(lower=0).astype("float32").to_numpy()
            pieces.append(np.log1p(order_count))
        out = np.concatenate(pieces, axis=1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32)


from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd



@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std).astype(np.float32)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, payload: dict) -> "Normalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def fit_normalizer_for_frequency(
    data_root: str | Path,
    table_name: str,
    transform: FeatureTransform,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    max_rows: int,
    batch_size: int,
) -> Normalizer:
    dataset = dataset_for_table(data_root, table_name)
    import pyarrow.dataset as ds

    flt = (ds.field("date") >= train_start) & (ds.field("date") < train_end + pd.Timedelta(days=1))
    cols = [c for c in transform.required_cols if c not in {"instrument", "instrument_id"}]
    scanner = dataset.scanner(columns=cols, filter=flt, batch_size=batch_size)

    total = 0
    sum_x = np.zeros(transform.n_features, dtype=np.float64)
    sum_x2 = np.zeros(transform.n_features, dtype=np.float64)

    for batch in scanner.to_batches():
        if total >= max_rows:
            break
        frame = batch.to_pandas()
        if len(frame) == 0:
            # Scanner may emit leading empty batches under I/O contention;
            # skip them instead of aborting the fit.
            continue
        take = min(len(frame), max_rows - total)
        if take <= 0:
            break
        x = transform.frame_to_features(frame.iloc[:take].copy()).astype(np.float64)
        sum_x += x.sum(axis=0)
        sum_x2 += np.square(x).sum(axis=0)
        total += len(x)

    if total == 0:
        raise RuntimeError(f"No rows available to fit normalizer for {table_name}")

    mean = sum_x / total
    var = np.maximum(sum_x2 / total - np.square(mean), 1e-12)
    return Normalizer(mean.astype(np.float32), np.sqrt(var).astype(np.float32))


import numpy as np
import pandas as pd


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(xr, yr)


def icir(values: list[float] | np.ndarray, annualization: float | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    std = np.std(arr, ddof=1)
    if std == 0:
        return float("nan")
    out = float(np.mean(arr) / std)
    if annualization is not None:
        out *= float(np.sqrt(annualization))
    return out


import numpy as np
import torch


DEFAULT_STYLE_EXPOSURE_COLS = (
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
)


def platform_alignment_enabled(config: dict) -> bool:
    return bool(config.get("platform_alignment", {}).get("enabled", False))


def metrics_platform_score_enabled(config: dict) -> bool:
    """Whether reported IC/RankIC metrics should use the style-neutralized score."""
    return bool(config.get("metrics", {}).get("platform_score", False))


def style_exposure_cols(config: dict) -> tuple[str, ...]:
    values = config.get("platform_alignment", {}).get(
        "exposure_cols",
        config.get("metrics", {}).get(
            "exposure_cols",
            DEFAULT_STYLE_EXPOSURE_COLS,
        ),
    )
    if isinstance(values, str):
        values = [values]
    cols = tuple(str(value) for value in values)
    if not cols:
        raise ValueError("platform_alignment.exposure_cols cannot be empty")
    return cols


def platform_score_torch(
    score: torch.Tensor,
    exposure: torch.Tensor,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    ridge: float = 1e-6,
    min_obs: int = 50,
) -> torch.Tensor:
    """Apply BigQuant-style cross-sectional score preprocessing.

    The returned residual remains differentiable with respect to ``score``.
    Exposure values and winsorization bounds are treated as fixed controls.
    """

    if score.ndim != 1:
        raise ValueError(f"score must be 1D, got {tuple(score.shape)}")
    if exposure.ndim != 2 or exposure.shape[0] != score.numel():
        raise ValueError(
            "exposure must have shape [N, R] aligned with score; "
            f"got score={tuple(score.shape)} exposure={tuple(exposure.shape)}"
        )
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("winsorization quantiles must satisfy 0 <= lower < upper <= 1")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    work_score = score.float()
    work_exposure = exposure.detach().to(device=score.device, dtype=torch.float32)
    if not bool(torch.isfinite(work_score).all()):
        raise ValueError("score contains non-finite values")
    if not bool(torch.isfinite(work_exposure).all()):
        raise ValueError("exposure contains non-finite values")

    bounds = torch.quantile(
        work_score.detach(),
        torch.tensor(
            [lower_quantile, upper_quantile],
            device=work_score.device,
            dtype=work_score.dtype,
        ),
    )
    clipped = work_score.clamp(min=bounds[0], max=bounds[1])
    standardized = (
        clipped - clipped.mean()
    ) / clipped.std(unbiased=False).clamp_min(1e-6)

    required = max(int(min_obs), int(work_exposure.shape[1]) + 2)
    if score.numel() < required:
        return standardized

    exposure_std = work_exposure.std(dim=0, unbiased=False)
    active = exposure_std > 1e-8
    if not bool(active.any()):
        return standardized
    controls = work_exposure[:, active]
    controls = (
        controls - controls.mean(dim=0, keepdim=True)
    ) / controls.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    design = torch.cat(
        (
            torch.ones(
                (controls.shape[0], 1),
                device=controls.device,
                dtype=controls.dtype,
            ),
            controls,
        ),
        dim=1,
    )
    regularizer = torch.eye(
        design.shape[1],
        device=design.device,
        dtype=design.dtype,
    ) * float(ridge)
    regularizer[0, 0] = 0.0
    beta = torch.linalg.solve(
        design.transpose(0, 1) @ design + regularizer,
        design.transpose(0, 1) @ standardized,
    )
    return standardized - design @ beta


def platform_score_numpy(
    score: np.ndarray,
    exposure: np.ndarray,
    **kwargs,
) -> np.ndarray:
    with torch.no_grad():
        residual = platform_score_torch(
            torch.as_tensor(np.asarray(score), dtype=torch.float32),
            torch.as_tensor(np.asarray(exposure), dtype=torch.float32),
            **kwargs,
        )
    return residual.cpu().numpy().astype(np.float64, copy=False)


def platform_score_kwargs(config: dict) -> dict[str, float | int]:
    cfg = config.get("platform_alignment", {})
    quantiles = cfg.get("score_winsor_quantiles", [0.01, 0.99])
    if len(quantiles) != 2:
        raise ValueError("score_winsor_quantiles must contain [lower, upper]")
    return {
        "lower_quantile": float(quantiles[0]),
        "upper_quantile": float(quantiles[1]),
        "ridge": float(cfg.get("ridge", 1e-6)),
        "min_obs": int(cfg.get("min_obs", 50)),
    }


def metrics_score_kwargs(config: dict) -> dict[str, float | int]:
    cfg = config.get("metrics", {})
    quantiles = cfg.get("score_winsor_quantiles", [0.01, 0.99])
    if len(quantiles) != 2:
        raise ValueError("metrics.score_winsor_quantiles must contain [lower, upper]")
    return {
        "lower_quantile": float(quantiles[0]),
        "upper_quantile": float(quantiles[1]),
        "ridge": float(cfg.get("ridge", 1e-6)),
        "min_obs": int(cfg.get("min_obs", 50)),
    }


__all__ = [
    "DEFAULT_STYLE_EXPOSURE_COLS",
    "metrics_platform_score_enabled",
    "metrics_score_kwargs",
    "platform_alignment_enabled",
    "platform_score_kwargs",
    "platform_score_numpy",
    "platform_score_torch",
    "style_exposure_cols",
]


import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(config: dict) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        timeout_minutes = int(
            config.get("training", {}).get("distributed_timeout_minutes", 1440)
        )
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
            )
        return DistributedContext(rank, world_size, local_rank, device)

    requested = config.get("training", {}).get("device", "auto")
    if torch.cuda.is_available() and requested != "cpu":
        if isinstance(requested, str) and requested.startswith("cuda:"):
            device = torch.device(requested)
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return DistributedContext(0, 1, device.index or 0, device)


def wrap_ddp(model: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if not context.enabled:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=context.local_rank if device_ids else None,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        if context.device.type == "cuda":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_counts(counts: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    counts = counts.detach().clone()
    if context.enabled:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts


def shard_cross_section(x: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if not context.enabled:
        return x
    return x[context.rank :: context.world_size].contiguous()


def gather_cross_section(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: DistributedContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not context.enabled:
        return prediction, target

    local_size = torch.tensor([prediction.numel()], device=prediction.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(context.world_size)]
    dist.all_gather(sizes, local_size)
    lengths = [int(value.item()) for value in sizes]
    max_size = max(lengths)
    if max_size == 0:
        raise RuntimeError("Every DDP rank received an empty cross-section")

    pred_padded = F_pad_1d(prediction, max_size).contiguous()
    target_padded = F_pad_1d(target, max_size).contiguous()
    gathered_pred = differentiable_all_gather(pred_padded)
    gathered_target = [torch.zeros_like(target_padded) for _ in range(context.world_size)]
    dist.all_gather(gathered_target, target_padded)
    full_prediction = torch.cat(
        [value[:length] for value, length in zip(gathered_pred, lengths)],
        dim=0,
    )
    full_target = torch.cat(
        [value[:length] for value, length in zip(gathered_target, lengths)],
        dim=0,
    )
    return full_prediction, full_target


def gather_cross_section_features(
    features: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    """Gather a non-trainable [N, ...] tensor in the same order as predictions."""
    if not context.enabled:
        return features
    if features.ndim < 1:
        raise ValueError("features must have a leading cross-sectional dimension")

    local_size = torch.tensor(
        [features.shape[0]],
        device=features.device,
        dtype=torch.long,
    )
    sizes = [torch.zeros_like(local_size) for _ in range(context.world_size)]
    dist.all_gather(sizes, local_size)
    lengths = [int(value.item()) for value in sizes]
    max_size = max(lengths)
    if max_size == 0:
        raise RuntimeError("Every DDP rank received an empty cross-section")

    if features.shape[0] < max_size:
        padding = features.new_zeros((max_size - features.shape[0], *features.shape[1:]))
        features = torch.cat((features, padding), dim=0)
    gathered = [torch.zeros_like(features) for _ in range(context.world_size)]
    dist.all_gather(gathered, features.contiguous())
    return torch.cat(
        [value[:length] for value, length in zip(gathered, lengths)],
        dim=0,
    )


def F_pad_1d(value: torch.Tensor, size: int) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got {tuple(value.shape)}")
    if value.numel() == size:
        return value
    return torch.cat([value, value.new_zeros(size - value.numel())], dim=0)


__all__ = [
    "DistributedContext",
    "all_reduce_counts",
    "barrier",
    "cleanup_distributed",
    "gather_cross_section",
    "gather_cross_section_features",
    "initialize_distributed",
    "shard_cross_section",
    "unwrap_model",
    "wrap_ddp",
]
