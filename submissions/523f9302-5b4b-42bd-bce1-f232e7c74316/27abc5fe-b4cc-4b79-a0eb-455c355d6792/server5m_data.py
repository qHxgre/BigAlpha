"""从本地 Parquet 构建并读取五档训练缓存。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from server5m_constants import (
    FEATURE_COLUMNS,
    PRICE_INDICES,
    SEQ_LEN,
    SPLIT_RANGES,
    TARGETS,
    VOLUME_INDICES,
)

READ_COLUMNS = (
    "trade_date",
    "datetime",
    "stock_code",
    "close",
    *FEATURE_COLUMNS,
)


@dataclass
class DayRecord:
    date: str
    features: np.ndarray
    stocks: np.ndarray
    open_price: np.ndarray
    close: np.ndarray


def _setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bigalpha_local.build")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _day_record(frame: pd.DataFrame) -> DayRecord:
    frame = frame.sort_values(["stock_code", "datetime"], kind="stable")
    date = str(frame["trade_date"].iloc[0])[:10]
    stocks = np.sort(frame["stock_code"].astype(str).unique())
    stock_index = {stock: i for i, stock in enumerate(stocks)}

    # 5 分钟标准时点固定为全市场当日出现的 48 个时间；停牌或缺帧保留 NaN。
    times = np.sort(frame["datetime"].dt.strftime("%H:%M:%S").unique())
    if len(times) != SEQ_LEN:
        raise ValueError(f"{date}: expected {SEQ_LEN} market timestamps, got {len(times)}")
    time_index = {time: i for i, time in enumerate(times)}
    row_stock = frame["stock_code"].astype(str).map(stock_index).to_numpy()
    row_time = frame["datetime"].dt.strftime("%H:%M:%S").map(time_index).to_numpy()

    features = np.full(
        (len(stocks), SEQ_LEN, len(FEATURE_COLUMNS)), np.nan, dtype=np.float32
    )
    values = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    values[:, PRICE_INDICES] = np.where(
        values[:, PRICE_INDICES] > 0, values[:, PRICE_INDICES], np.nan
    )
    features[row_stock, row_time] = values

    grouped_close = frame.groupby("stock_code", sort=False)["close"]
    open_price = grouped_close.first().reindex(stocks).to_numpy(
        dtype=np.float64, copy=True
    )
    close = grouped_close.last().reindex(stocks).to_numpy(
        dtype=np.float64, copy=True
    )
    open_price[open_price <= 0] = np.nan
    close[close <= 0] = np.nan
    return DayRecord(
        date=date, features=features, stocks=stocks,
        open_price=open_price, close=close,
    )


def iter_day_records(parquet_path: str | Path, logger: logging.Logger):
    """按 row group 流式读取；跨 row group 的最后一个交易日会正确拼接。"""
    parquet = pq.ParquetFile(parquet_path)
    pending = None
    for row_group in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(
            row_group, columns=list(READ_COLUMNS), use_threads=True
        ).to_pandas()
        if pending is not None:
            frame = pd.concat((pending, frame), ignore_index=True)
            pending = None
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        dates = np.sort(frame["trade_date"].unique())
        last_date = dates[-1]
        for value in dates[:-1]:
            yield _day_record(frame.loc[frame["trade_date"] == value])
        pending = frame.loc[frame["trade_date"] == last_date].copy()
        logger.info(
            "row_group=%d/%d rows=%d through=%s",
            row_group + 1,
            parquet.metadata.num_row_groups,
            parquet.metadata.row_group(row_group).num_rows,
            str(last_date)[:10],
        )
    if pending is not None and not pending.empty:
        yield _day_record(pending)


def _split_for_date(date: str) -> str | None:
    for split, (start, end) in SPLIT_RANGES.items():
        if start <= date <= end:
            return split
    return None


def _transform_features(
    features: np.ndarray,
    price_transform: str,
    volume_transform: str,
) -> np.ndarray:
    result = features.astype(np.float32, copy=True)
    if price_transform == "log":
        result[..., PRICE_INDICES] = np.log(
            np.maximum(result[..., PRICE_INDICES], 1e-12)
        )
    elif price_transform != "none":
        raise ValueError(f"Unsupported price_transform: {price_transform}")
    if volume_transform == "log1p":
        result[..., VOLUME_INDICES] = np.log1p(
            np.maximum(result[..., VOLUME_INDICES], 0.0)
        )
    elif volume_transform != "none":
        raise ValueError(f"Unsupported volume_transform: {volume_transform}")
    return result


def _relative_cs_features(record: DayRecord) -> np.ndarray:
    """开盘相对价格 + 时点字段截面 z-score + 十个量字段最大值归一化。"""
    result = record.features.astype(np.float32, copy=True)

    prices = result[..., PRICE_INDICES]
    open_price = record.open_price.astype(np.float32)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        prices = prices / open_price - 1.0
    with np.errstate(all="ignore"):
        cs_mean = np.nanmean(prices, axis=0, keepdims=True)
        cs_std = np.nanstd(prices, axis=0, keepdims=True)
    cs_std = np.where(cs_std < 1e-8, 1.0, cs_std)
    result[..., PRICE_INDICES] = (prices - cs_mean) / cs_std

    volumes = result[..., VOLUME_INDICES]
    finite_volumes = np.where(np.isfinite(volumes), volumes, -np.inf)
    max_volume = finite_volumes.max(axis=-1, keepdims=True)
    max_volume = np.where(max_volume > 0, max_volume, np.nan)
    result[..., VOLUME_INDICES] = volumes / max_volume
    return result


def _fit_standardizer(
    records: list[DayRecord],
    price_transform: str,
    volume_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    total_sq = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    count = np.zeros(len(FEATURE_COLUMNS), dtype=np.int64)
    for record in records:
        if _split_for_date(record.date) != "train":
            continue
        transformed = _transform_features(
            record.features, price_transform, volume_transform
        )
        flat = transformed.reshape(-1, len(FEATURE_COLUMNS)).astype(np.float64)
        finite = np.isfinite(flat)
        clean = np.where(finite, flat, 0.0)
        total += clean.sum(axis=0)
        total_sq += (clean * clean).sum(axis=0)
        count += finite.sum(axis=0)
    mean = total / np.maximum(count, 1)
    variance = total_sq / np.maximum(count, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _target_panel(records: list[DayRecord], horizon: int) -> pd.DataFrame:
    close = pd.DataFrame(
        {
            record.date: pd.Series(record.close, index=record.stocks)
            for record in records
        }
    ).transpose()
    close.index.name = "date"
    return close.shift(-horizon).divide(close) - 1.0


def _classification_labels(values: np.ndarray, bins: int) -> np.ndarray:
    labels = np.full(len(values), -1, dtype=np.int16)
    valid_index = np.flatnonzero(np.isfinite(values))
    if len(valid_index) < bins:
        return labels
    order = valid_index[np.argsort(values[valid_index], kind="stable")]
    labels[order] = np.minimum(
        np.arange(len(order), dtype=np.int64) * bins // len(order), bins - 1
    ).astype(np.int16)
    return labels


def build_cache(
    parquet_path: str | Path,
    cache_dir: str | Path,
    force: bool = False,
    price_transform: str = "log",
    volume_transform: str = "log1p",
    feature_mode: str = "log_standard",
) -> Path:
    cache = Path(cache_dir)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and not force:
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(cache / "build.log")
    logger.info("Building five-level cache from %s", parquet_path)

    records = list(iter_day_records(parquet_path, logger))
    dates = [record.date for record in records]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("Parquet records are not uniquely ordered by date")
    logger.info("Loaded %d trading days: %s through %s", len(dates), dates[0], dates[-1])

    if feature_mode == "log_standard":
        mean, std = _fit_standardizer(records, price_transform, volume_transform)
        np.savez(cache / "feature_stats.npz", mean=mean, std=std)
        logger.info("Fitted train-only field standardizer")
    elif feature_mode == "relative_cs":
        mean = std = None
        logger.info(
            "Feature mode: open-relative price + cross-sectional zscore; "
            "per-snapshot max-normalized volume"
        )
    elif feature_mode == "pre_normalized":
        mean = std = None
        logger.info("Feature mode: values already normalized by source SQL")
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    split_records: dict[str, list[DayRecord]] = {name: [] for name in SPLIT_RANGES}
    for record in records:
        split = _split_for_date(record.date)
        if split is not None:
            split_records[split].append(record)

    for split, selected in split_records.items():
        features = np.concatenate(
            [
                np.nan_to_num(
                    (
                        (
                            _transform_features(
                                record.features, price_transform, volume_transform
                            )
                            - mean
                        )
                        / std
                        if feature_mode == "log_standard"
                        else (
                            _relative_cs_features(record)
                            if feature_mode == "relative_cs"
                            else record.features
                        )
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
                for record in selected
            ]
        )
        date_ids = np.concatenate(
            [
                np.full(len(record.stocks), dates.index(record.date), dtype=np.int32)
                for record in selected
            ]
        )
        stocks = np.concatenate([record.stocks.astype("U16") for record in selected])
        np.save(cache / f"{split}_X.npy", features)
        np.save(cache / f"{split}_date_ids.npy", date_ids)
        np.save(cache / f"{split}_stocks.npy", stocks)
        logger.info(
            "%s: days=%d samples=%d X=%s",
            split,
            len(selected),
            len(features),
            tuple(features.shape),
        )
        del features, date_ids, stocks

    record_by_date = {record.date: record for record in records}
    for horizon in (1, 2, 3):
        panel = _target_panel(records, horizon)
        for split, selected in split_records.items():
            raw_parts = [
                panel.loc[record.date].reindex(record.stocks).to_numpy(dtype=np.float32, copy=True)
                for record in selected
            ]
            raw = np.concatenate(raw_parts)
            np.save(cache / f"{split}_ret_{horizon}.npy", raw)
            for bins in (5, 10):
                labels = np.concatenate(
                    [_classification_labels(values, bins) for values in raw_parts]
                )
                np.save(cache / f"{split}_cls{bins}_{horizon}.npy", labels)
            logger.info(
                "%s horizon=%d valid_targets=%d/%d",
                split,
                horizon,
                int(np.isfinite(raw).sum()),
                len(raw),
            )

    manifest = {
        "version": 1,
        "source": str(Path(parquet_path).resolve()),
        "feature_columns": list(FEATURE_COLUMNS),
        "sequence_length": SEQ_LEN,
        "feature_mode": feature_mode,
        "price_transform": price_transform,
        "volume_transform": volume_transform,
        "dates": dates,
        "split_ranges": SPLIT_RANGES,
        "samples": {
            split: int(sum(len(record.stocks) for record in selected))
            for split, selected in split_records.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Cache complete: %s", cache)
    return cache


def cross_sectional_zscore(values: np.ndarray, date_ids: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float32)
    for date_id in np.unique(date_ids):
        mask = (date_ids == date_id) & np.isfinite(values)
        if not mask.any():
            continue
        section = values[mask].astype(np.float64)
        scale = section.std()
        result[mask] = ((section - section.mean()) / max(scale, 1e-8)).astype(
            np.float32
        )
    return result


def cross_sectional_rank(values: np.ndarray, date_ids: np.ndarray) -> np.ndarray:
    """逐日精确百分位秩，中心化到 [-0.5, 0.5]。"""
    result = np.full(len(values), np.nan, dtype=np.float32)
    for date_id in np.unique(date_ids):
        mask = (date_ids == date_id) & np.isfinite(values)
        count = int(mask.sum())
        if count < 2:
            continue
        order = np.argsort(values[mask], kind="stable")
        ranks = np.empty(count, dtype=np.float32)
        ranks[order] = np.arange(count, dtype=np.float32)
        result[mask] = ranks / max(count - 1, 1) - 0.5
    return result


class LocalDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        target: str,
        max_samples: int | None = None,
        purge_boundary: bool = False,
    ):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; choices: {sorted(TARGETS)}")
        cache = Path(cache_dir)
        self.X = np.load(cache / f"{split}_X.npy", mmap_mode="r")
        self.date_ids_all = np.load(cache / f"{split}_date_ids.npy", mmap_mode="r")
        self.stocks_all = np.load(cache / f"{split}_stocks.npy", mmap_mode="r")
        task, horizon, _ = TARGETS[target]
        self.task = task
        self.horizon = horizon
        self.raw_return_all = np.load(
            cache / f"{split}_ret_{horizon}.npy", mmap_mode="r"
        )
        if task == "regression":
            target_all = cross_sectional_zscore(
                self.raw_return_all, self.date_ids_all
            )
            valid = np.isfinite(target_all)
        else:
            target_all = np.load(cache / f"{split}_{target}.npy", mmap_mode="r")
            valid = target_all >= 0
        if purge_boundary and split in ("train", "val"):
            unique_dates = np.unique(self.date_ids_all)
            boundary_dates = unique_dates[-horizon:]
            valid &= ~np.isin(self.date_ids_all, boundary_dates)
        indices = np.flatnonzero(valid)
        if max_samples is not None:
            indices = indices[: int(max_samples)]
        self.indices = indices
        self.target = np.asarray(target_all[indices])
        self.date_ids = np.asarray(self.date_ids_all[indices])
        self.stocks = np.asarray(self.stocks_all[indices])
        self.raw_return = np.asarray(self.raw_return_all[indices])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        x = torch.from_numpy(np.array(self.X[index], copy=True))
        if self.task == "regression":
            y = torch.tensor([self.target[item]], dtype=torch.float32)
        else:
            y = torch.tensor(int(self.target[item]), dtype=torch.long)
        return x, y


class CombinedDataset(Dataset):
    """把多个时序 split 组合成一个逻辑数据集，不复制大型 X 数组。"""

    def __init__(self, datasets: list[LocalDataset]):
        if not datasets:
            raise ValueError("CombinedDataset needs at least one dataset")
        if len({dataset.task for dataset in datasets}) != 1:
            raise ValueError("All combined datasets must have the same task")
        self.datasets = datasets
        self.task = datasets[0].task
        self.cumulative = np.cumsum([len(dataset) for dataset in datasets])
        self.target = np.concatenate([dataset.target for dataset in datasets])
        self.date_ids = np.concatenate([dataset.date_ids for dataset in datasets])
        self.stocks = np.concatenate([dataset.stocks for dataset in datasets])
        self.raw_return = np.concatenate([dataset.raw_return for dataset in datasets])

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def __getitem__(self, item: int):
        dataset_index = int(np.searchsorted(self.cumulative, item, side="right"))
        previous = 0 if dataset_index == 0 else int(self.cumulative[dataset_index - 1])
        return self.datasets[dataset_index][item - previous]


class MultiTargetDataset(Dataset):
    """共享一份 X，同时提供 day1/day2/day3 的回归与五分类标签。"""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        max_samples: int | None = None,
        purge_boundary: bool = True,
        num_classes: int = 5,
    ):
        cache = Path(cache_dir)
        self.X = np.load(cache / f"{split}_X.npy", mmap_mode="r")
        date_ids_all = np.load(cache / f"{split}_date_ids.npy", mmap_mode="r")
        stocks_all = np.load(cache / f"{split}_stocks.npy", mmap_mode="r")
        raw_returns = [
            np.load(cache / f"{split}_ret_{horizon}.npy", mmap_mode="r")
            for horizon in (1, 2, 3)
        ]
        regression = np.column_stack(
            [cross_sectional_zscore(values, date_ids_all) for values in raw_returns]
        )
        rank_target = np.column_stack(
            [cross_sectional_rank(values, date_ids_all) for values in raw_returns]
        )
        classification = np.column_stack(
            [
                np.load(cache / f"{split}_cls{num_classes}_{horizon}.npy", mmap_mode="r")
                for horizon in (1, 2, 3)
            ]
        )
        valid = np.isfinite(regression).all(axis=1)
        valid &= (classification >= 0).all(axis=1)
        if purge_boundary and split in ("train", "val"):
            unique_dates = np.unique(date_ids_all)
            valid &= ~np.isin(date_ids_all, unique_dates[-3:])
        indices = np.flatnonzero(valid)
        if max_samples is not None:
            indices = indices[: int(max_samples)]
        self.indices = indices
        self.regression = regression[indices].astype(np.float32, copy=False)
        self.rank_target = rank_target[indices].astype(np.float32, copy=False)
        self.classification = classification[indices].astype(np.int64, copy=False)
        self.date_ids = np.asarray(date_ids_all[indices])
        self.stocks = np.asarray(stocks_all[indices])
        self.raw_return = np.asarray(raw_returns[0][indices])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        return (
            torch.from_numpy(np.array(self.X[index], copy=True)),
            torch.from_numpy(self.regression[item]),
            torch.from_numpy(self.classification[item]),
            torch.from_numpy(self.rank_target[item]),
        )


def load_manifest(cache_dir: str | Path) -> dict:
    return json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))
