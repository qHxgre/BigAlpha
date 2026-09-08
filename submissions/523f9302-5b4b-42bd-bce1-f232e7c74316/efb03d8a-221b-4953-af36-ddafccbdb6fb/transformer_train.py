# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端赛道：合规的分块 Transformer 训练与推理模块。

合规原则：
1. 模型输入只来自官方分钟行情表中的原始字段；
2. 不使用 PE、ROE、市值、ST 标签、人工因子或跨字段衍生特征；
3. 预处理仅包含缺失值填充、逐字段 log1p 和训练集 StandardScaler；
4. 模型参数量在 10 万到 1 亿之间；
5. 分块查询股票，避免一次读取全部分钟数据。

在 BigQuant 工作站训练：python transformer_train.py
训练结束后会在同目录生成 transformer_model.json。
"""

import gc
import json
import math
import os
import re
import time

import dai
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# 固定配置
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

DEFAULT_BAR_TABLE = "bigalpha_2026_stock_bar1m"
INSTRUMENT_TABLE = "bigalpha_2026_instruments"
EXPOSURE_TABLE = "bigalpha_2026_exposure"

# 覆盖完整公开训练区间；固定容量抽样会把最终训练规模控制在可接受范围。
TRAIN_START = "2019-01-01 00:00:00"
TRAIN_END = "2023-12-31 23:59:59"

SEQ_LEN = 64
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2
FALLBACK_EPOCHS = 3
VALIDATION_FRACTION = 0.20
SEED = 42

CHUNK_SIZE = 25
# 为控制 CPU 训练时长，从训练期成分股并集中按固定种子抽取 600 只；
# 这是与行情数值无关的可复现抽样，推理仍覆盖全部中证1000成分股。
MAX_TRAIN_INSTRUMENTS = 600
MAX_TRAIN_SAMPLES = 260_000

MIN_MODEL_PARAMETERS = 100_000
MAX_MODEL_PARAMETERS = 100_000_000

# 这 25 列全部是官方 bar1m 表中的原始字段。字段在模型外不做相减、相除、
# 滚动统计、因子合成或降维。
INPUT_FIELDS = [
    "open",
    "high",
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
LOG_INPUT_FIELDS = [
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
INPUT_VERSION = 4
INPUT_TRANSFORM = "missing_zero_log_volume_count_then_standardize_v4"
N_INPUTS = len(INPUT_FIELDS)

# Transformer 主干保持适中的计算量；额外的 MLP 输出头使总参数量超过 10 万，
# 比单纯增加 Transformer 层数更节省序列计算。
MODEL_CFG = {
    "n_inputs": N_INPUTS,
    "d_model": 64,
    "nhead": 4,
    "nlayers": 2,
    "dim_ff": 128,
    "attention_hidden": 32,
    "head_hidden": 512,
    "seq_len": SEQ_LEN,
    "dropout": 0.15,
}


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class StockTransformer(nn.Module):
    """直接从原始分钟字段序列输出下一交易日的截面收益分数。"""

    def __init__(
        self,
        n_inputs,
        d_model=64,
        nhead=4,
        nlayers=2,
        dim_ff=128,
        attention_hidden=32,
        head_hidden=512,
        seq_len=64,
        dropout=0.15,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_inputs, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        hidden = self.input_projection(x) + self.position_embedding
        hidden = self.encoder(hidden)
        attention = torch.softmax(self.attention_pool(hidden), dim=1)
        pooled = torch.sum(hidden * attention, dim=1)
        return self.output_head(pooled).squeeze(-1)


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def validate_model_size(model):
    count = count_trainable_parameters(model)
    if not MIN_MODEL_PARAMETERS <= count <= MAX_MODEL_PARAMETERS:
        raise RuntimeError(
            f"模型可训练参数量 {count:,} 不符合比赛范围 "
            f"[{MIN_MODEL_PARAMETERS:,}, {MAX_MODEL_PARAMETERS:,}]"
        )
    return count


# ---------------------------------------------------------------------------
# 数据读取与合规预处理
# ---------------------------------------------------------------------------

def _safe_table_name(table):
    if not isinstance(table, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", table):
        raise ValueError(f"非法数据表名: {table!r}")
    return table


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _normalize_keys(frame):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype("string").astype(str)
    return frame


def membership(start_date, end_date):
    """读取历史时点上的中证1000成分股。"""
    frame = dai.query(
        f"SELECT date, instrument FROM {INSTRUMENT_TABLE}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    if frame.empty:
        raise RuntimeError(f"股票池为空: {start_date} ~ {end_date}")
    frame = _normalize_keys(frame)
    return frame.drop_duplicates(["date", "instrument"]).reset_index(drop=True)


def _query_bar_chunk(table, start_date, end_date, instruments):
    table = _safe_table_name(table)
    fields = ", ".join(INPUT_FIELDS)
    frame = dai.query(
        f"SELECT date, instrument, {fields} "
        f"FROM {table} ORDER BY instrument, date",
        filters={
            "date": [start_date, end_date],
            "instrument": instruments,
        },
        compression=True,
    ).df()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype("string").astype(str)
    for field in INPUT_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame.sort_values(["instrument", "date"]).reset_index(drop=True)


def _preprocess_raw(values):
    """只做规则允许的缺失填充和指定原始字段的 log1p。"""
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # 官方本地压缩表可能以 -1 表示 OHLC 缺失；云端表通常为 NaN。
    for field in ["open", "high", "low", "close"]:
        field_index = INPUT_FIELDS.index(field)
        values[:, field_index] = np.where(values[:, field_index] < 0, 0.0, values[:, field_index])
    for field in LOG_INPUT_FIELDS:
        field_index = INPUT_FIELDS.index(field)
        values[:, field_index] = np.log1p(np.clip(values[:, field_index], 0.0, None))
    return values.astype(np.float32, copy=False)


def _next_trading_day_map(members):
    days = np.sort(members["date"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    return {
        pd.Timestamp(days[index]).value: pd.Timestamp(days[index + 1]).value
        for index in range(len(days) - 1)
    }


def _windows_from_instrument(
    frame,
    start_day,
    end_day,
    next_day_by_value=None,
    need_label=False,
):
    """每个交易日仅使用当日最后 SEQ_LEN 条原始分钟记录。"""
    windows, labels, keys = [], [], []
    if len(frame) < SEQ_LEN:
        return windows, labels, keys

    instrument = str(frame["instrument"].iloc[0])
    transformed = _preprocess_raw(frame[INPUT_FIELDS].to_numpy())
    minute_days = frame["date"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    day_end_positions = np.flatnonzero(np.append(minute_days[1:] != minute_days[:-1], True))
    closing_prices = frame["close"].to_numpy(dtype=np.float64)[day_end_positions]
    closing_days = minute_days[day_end_positions]

    for index, end_position in enumerate(day_end_positions):
        day = pd.Timestamp(closing_days[index]).normalize()
        day_start_position = 0 if index == 0 else day_end_positions[index - 1] + 1
        if day < start_day or day > end_day:
            continue
        if end_position - day_start_position + 1 < SEQ_LEN:
            continue

        label = np.float32(0.0)
        if need_label:
            if index + 1 >= len(day_end_positions):
                continue
            expected_next = next_day_by_value.get(day.value)
            actual_next = pd.Timestamp(closing_days[index + 1]).value
            current_close = closing_prices[index]
            next_close = closing_prices[index + 1]
            if expected_next is None or actual_next != expected_next:
                continue
            if not np.isfinite(current_close) or not np.isfinite(next_close) or current_close <= 0:
                continue
            raw_return = next_close / current_close - 1.0
            if not np.isfinite(raw_return):
                continue
            label = np.float32(raw_return)

        windows.append(
            transformed[end_position - SEQ_LEN + 1:end_position + 1].copy()
        )
        labels.append(label)
        keys.append((day, instrument))

    return windows, labels, keys


class _RunningStats:
    """逐字段累计训练集均值和标准差，避免复制完整样本。"""

    def __init__(self, width):
        self.width = int(width)
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.square_total = np.zeros(width, dtype=np.float64)

    def update(self, values):
        matrix = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        self.count += len(matrix)
        self.total += matrix.sum(axis=0)
        self.square_total += np.square(matrix).sum(axis=0)

    def finalize(self):
        if self.count == 0:
            raise RuntimeError("没有可用于计算标准化参数的训练样本")
        mean = self.total / self.count
        variance = self.square_total / self.count - np.square(mean)
        std = np.sqrt(np.clip(variance, 0.0, None))
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


def _sample_key(day, instrument):
    return pd.Timestamp(day).normalize().value, str(instrument)


def _validation_start(members):
    days = np.sort(members["date"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    if len(days) < 5:
        raise RuntimeError("训练区间的交易日过少")
    split_index = int(len(days) * (1.0 - VALIDATION_FRACTION))
    split_index = min(max(split_index, 1), len(days) - 1)
    return pd.Timestamp(days[split_index]).normalize()


def _query_exposure_frame(start_date, end_date, instruments):
    """风险暴露仅用于训练标签残差化，绝不进入模型输入。"""
    table = _safe_table_name(EXPOSURE_TABLE)
    frame = dai.query(
        f"SELECT * FROM {table}",
        filters={
            "date": [start_date, end_date],
            "instrument": instruments,
        },
        compression=True,
    ).df()
    if frame.empty:
        raise RuntimeError("风险暴露表为空")
    return _normalize_keys(frame).drop_duplicates(["date", "instrument"], keep="last")


def _select_style_exposure_columns(frame):
    """从官方暴露表中识别连续风格列，排除行业和可能的收益标签列。"""
    style_tokens = (
        "size",
        "beta",
        "momentum",
        "resvol",
        "volatility",
        "liquidity",
        "value",
        "btop",
        "earn",
        "profit",
        "growth",
        "leverage",
        "nonlinear",
        "nl_size",
        "nlsize",
    )
    blocked_tokens = (
        "return",
        "ret_",
        "label",
        "target",
        "future",
        "next",
        "industry",
        "sector",
        "sw_",
        "date",
        "instrument",
    )
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not any(token in column.lower() for token in blocked_tokens)
    ]
    selected = [
        column
        for column in numeric_columns
        if any(token in column.lower() for token in style_tokens)
    ]
    # 某些版本的暴露表采用简写列名；若无法按名称识别，则仅回退到连续数值列。
    if len(selected) < 2:
        selected = [
            column
            for column in numeric_columns
            if frame[column].nunique(dropna=True) > 10
        ]
    return sorted(selected)[:20]


def _daily_style_residual(one_day, exposure_columns):
    raw_return = one_day["next_return"].to_numpy(dtype=np.float64)
    finite_return = np.isfinite(raw_return)
    result = np.full(len(one_day), np.nan, dtype=np.float64)
    if finite_return.sum() < 20:
        result[finite_return] = raw_return[finite_return] - np.mean(raw_return[finite_return])
        return result

    y = raw_return[finite_return]
    lower, upper = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lower, upper)
    x = one_day.loc[finite_return, exposure_columns].to_numpy(dtype=np.float64)
    usable_columns = []
    normalized_columns = []
    for column_index in range(x.shape[1]):
        values = x[:, column_index]
        finite = np.isfinite(values)
        if finite.sum() < max(10, int(0.50 * len(values))):
            continue
        median = float(np.median(values[finite]))
        values = np.where(finite, values, median)
        std = float(values.std())
        if not np.isfinite(std) or std < 1e-8:
            continue
        usable_columns.append(column_index)
        normalized_columns.append((values - values.mean()) / std)

    centered_y = y - y.mean()
    if not usable_columns:
        residual = centered_y
    else:
        design = np.column_stack(normalized_columns)
        ridge = 1e-3 * np.eye(design.shape[1], dtype=np.float64)
        try:
            coefficients = np.linalg.solve(
                design.T @ design + ridge,
                design.T @ centered_y,
            )
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(design, centered_y, rcond=None)[0]
        residual = centered_y - design @ coefficients
    result[finite_return] = residual
    return result


def _build_target_labels(targets_frame, start_date, end_date):
    """优先构造官方风格暴露残差收益排名，失败时安全退回原始收益排名。"""
    target_basis = targets_frame["next_return"].astype(np.float64).copy()
    mode = "cross_sectional_return_rank"
    exposure_columns = []
    try:
        instruments = targets_frame["instrument"].drop_duplicates().tolist()
        exposures = _query_exposure_frame(start_date, end_date, instruments)
        exposure_columns = _select_style_exposure_columns(exposures)
        if len(exposure_columns) < 2:
            raise RuntimeError("风险暴露表中未识别到足够的风格列")

        merged = (
            targets_frame.reset_index()
            .rename(columns={"index": "sample_order"})
            .merge(
                exposures[["date", "instrument"] + exposure_columns],
                on=["date", "instrument"],
                how="left",
            )
            .sort_values("sample_order")
            .reset_index(drop=True)
        )
        residual = np.full(len(merged), np.nan, dtype=np.float64)
        for _, positions in merged.groupby("date", sort=False).groups.items():
            position_array = np.asarray(list(positions), dtype=np.int64)
            residual[position_array] = _daily_style_residual(
                merged.iloc[position_array], exposure_columns
            )
        if np.isfinite(residual).sum() < int(0.80 * len(residual)):
            raise RuntimeError("风格残差标签有效覆盖率低于80%")
        fallback = merged.groupby("date")["next_return"].transform(
            lambda values: values - values.mean()
        ).to_numpy(dtype=np.float64)
        residual = np.where(np.isfinite(residual), residual, fallback)
        target_basis = pd.Series(residual, index=targets_frame.index)
        mode = "official_style_exposure_residual_rank"
        logger.info(
            "风格残差训练标签构建完成",
            exposure_fields=exposure_columns,
            samples=len(targets_frame),
        )
    except Exception as error:
        exposure_columns = []
        logger.warning(
            "风格暴露标签不可用，退回收益截面排名",
            error=str(error),
        )

    rank_source = pd.DataFrame(
        {"date": targets_frame["date"], "target_basis": target_basis}
    )
    target_rank = rank_source.groupby("date")["target_basis"].rank(
        method="average", pct=True
    )
    targets = (2.0 * target_rank.to_numpy(dtype=np.float32) - 1.0).astype(np.float32)
    metadata = {
        "target_mode": mode,
        "exposure_table": EXPOSURE_TABLE if exposure_columns else None,
        "exposure_fields": exposure_columns,
    }
    return targets, metadata


def build_train_dataset(table, start_date, end_date, members, instruments, validation_start):
    """按股票分块并以固定容量水库抽样构建跨年份训练样本。"""
    started = time.time()
    start_day = pd.Timestamp(start_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()
    validation_day = pd.Timestamp(validation_start).normalize()
    allowed_keys = {
        _sample_key(day, instrument)
        for day, instrument in members[["date", "instrument"]].itertuples(index=False, name=None)
    }
    next_day_by_value = _next_trading_day_map(members)

    all_windows, all_returns, all_dates, all_instruments = [], [], [], []
    train_stats = _RunningStats(N_INPUTS)
    full_stats = _RunningStats(N_INPUTS)
    chunk_count = math.ceil(len(instruments) / CHUNK_SIZE)
    reservoir_random = np.random.default_rng(SEED + 1009)
    seen_samples = 0

    for chunk_index, instrument_chunk in enumerate(
        _chunks(instruments, CHUNK_SIZE), start=1
    ):
        frame = _query_bar_chunk(table, start_date, end_date, instrument_chunk)
        accepted = 0
        if not frame.empty:
            for _, one_stock in frame.groupby("instrument", sort=False, observed=True):
                windows, returns, keys = _windows_from_instrument(
                    one_stock,
                    start_day,
                    end_day,
                    next_day_by_value=next_day_by_value,
                    need_label=True,
                )
                selected = [
                    position
                    for position, key in enumerate(keys)
                    if _sample_key(*key) in allowed_keys
                ]
                if not selected:
                    continue

                selected_windows = [windows[position] for position in selected]
                selected_returns = [returns[position] for position in selected]
                selected_dates = [keys[position][0] for position in selected]
                stacked = np.stack(selected_windows).astype(np.float32, copy=False)
                full_stats.update(stacked)
                train_mask = np.asarray(
                    [day < validation_day for day in selected_dates], dtype=bool
                )
                if train_mask.any():
                    train_stats.update(stacked[train_mask])

                for position, window in enumerate(selected_windows):
                    seen_samples += 1
                    if MAX_TRAIN_SAMPLES is None or len(all_windows) < MAX_TRAIN_SAMPLES:
                        reservoir_position = len(all_windows)
                    else:
                        reservoir_position = int(reservoir_random.integers(0, seen_samples))
                    if MAX_TRAIN_SAMPLES is not None and reservoir_position >= MAX_TRAIN_SAMPLES:
                        continue

                    sample_return = selected_returns[position]
                    sample_date = selected_dates[position]
                    sample_instrument = keys[selected[position]][1]
                    if reservoir_position == len(all_windows):
                        all_windows.append(window)
                        all_returns.append(sample_return)
                        all_dates.append(sample_date)
                        all_instruments.append(sample_instrument)
                    else:
                        all_windows[reservoir_position] = window
                        all_returns[reservoir_position] = sample_return
                        all_dates[reservoir_position] = sample_date
                        all_instruments[reservoir_position] = sample_instrument
                accepted += len(selected)

        del frame
        gc.collect()
        logger.info(
            "训练分钟分块完成",
            chunk=f"{chunk_index}/{chunk_count}",
            accepted=accepted,
            seen=seen_samples,
            stored=len(all_windows),
        )

    if not all_windows:
        raise RuntimeError(f"训练区间没有可用序列: {start_date} ~ {end_date}")

    inputs = np.stack(all_windows).astype(np.float32, copy=False)
    targets_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(all_dates)).dt.normalize(),
            "instrument": pd.Series(all_instruments, dtype="string").astype(str),
            "next_return": np.asarray(all_returns, dtype=np.float32),
        }
    )
    targets, target_metadata = _build_target_labels(targets_frame, start_date, end_date)
    dates = targets_frame["date"].to_numpy(dtype="datetime64[ns]")

    del all_windows, all_returns, all_dates, all_instruments, targets_frame
    gc.collect()
    logger.info(
        "训练集构建完成",
        samples=len(inputs),
        seen_samples=seen_samples,
        train_samples=int((dates < np.datetime64(validation_day)).sum()),
        validation_samples=int((dates >= np.datetime64(validation_day)).sum()),
        elapsed=round(time.time() - started, 2),
    )
    return (
        inputs,
        targets,
        dates,
        train_stats.finalize(),
        full_stats.finalize(),
        target_metadata,
    )


def _normalize_inplace(inputs, mean, std):
    inputs -= np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    inputs /= np.asarray(std, dtype=np.float32).reshape(1, 1, -1)


def _denormalize_inplace(inputs, mean, std):
    inputs *= np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    inputs += np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)


# ---------------------------------------------------------------------------
# 训练与验证
# ---------------------------------------------------------------------------

def _seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_loader(dataset, indices, shuffle, device):
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )


def _train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss, sample_count = 0.0, 0
    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(batch_inputs), batch_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_inputs)
        sample_count += len(batch_inputs)
    return total_loss / max(sample_count, 1)


def _validation_metrics(model, loader, dates, device):
    model.eval()
    predictions, targets = [], []
    squared_error, sample_count = 0.0, 0
    loss_function = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            output = model(batch_inputs.to(device, non_blocking=True)).cpu()
            squared_error += float(loss_function(output, batch_targets).item())
            sample_count += len(batch_targets)
            predictions.append(output.numpy())
            targets.append(batch_targets.numpy())

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    metric = pd.DataFrame(
        {"date": dates, "prediction": prediction, "target": target}
    )
    daily_ic = []
    for _, one_day in metric.groupby("date", sort=False):
        prediction_rank = one_day["prediction"].rank(method="average", pct=True)
        value = prediction_rank.corr(one_day["target"])
        if np.isfinite(value):
            daily_ic.append(float(value))
    rank_ic = float(np.mean(daily_ic)) if daily_ic else np.nan
    return squared_error / max(sample_count, 1), rank_ic


def _select_epoch(inputs, targets, dates, validation_start, device):
    split = np.datetime64(pd.Timestamp(validation_start).normalize())
    train_indices = np.flatnonzero(dates < split)
    validation_indices = np.flatnonzero(dates >= split)
    if len(train_indices) == 0 or len(validation_indices) == 0:
        logger.warning(
            "缺少内部训练或验证样本，采用固定轮数",
            epochs=FALLBACK_EPOCHS,
        )
        return FALLBACK_EPOCHS

    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    train_loader = _make_loader(dataset, train_indices, True, device)
    validation_loader = _make_loader(dataset, validation_indices, False, device)

    _seed_everything()
    model = StockTransformer(**MODEL_CFG).to(device)
    parameter_count = validate_model_size(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    best_epoch, best_score, stale_epochs = 1, -np.inf, 0

    logger.info("模型参数检查通过", trainable_parameters=parameter_count)
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time()
        train_mse = _train_epoch(model, train_loader, optimizer, loss_function, device)
        validation_mse, validation_rank_ic = _validation_metrics(
            model, validation_loader, dates[validation_indices], device
        )
        score = validation_rank_ic if np.isfinite(validation_rank_ic) else -validation_mse
        logger.info(
            "验证训练轮次完成",
            epoch=epoch,
            train_mse=round(train_mse, 8),
            validation_mse=round(validation_mse, 8),
            validation_rank_ic=(
                round(validation_rank_ic, 6)
                if np.isfinite(validation_rank_ic)
                else None
            ),
            elapsed=round(time.time() - started, 2),
        )
        if score > best_score + 1e-5:
            best_score, best_epoch, stale_epochs = score, epoch, 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break

    logger.info(
        "最佳训练轮数已确定",
        best_epoch=best_epoch,
        best_validation_rank_ic=best_score,
    )
    del model, dataset, train_loader, validation_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return best_epoch


def _fit_full_dataset(inputs, targets, epochs, device):
    _seed_everything()
    model = StockTransformer(**MODEL_CFG).to(device)
    validate_model_size(model)
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        started = time.time()
        mse = _train_epoch(model, loader, optimizer, loss_function, device)
        logger.info(
            "全量重训轮次完成",
            epoch=epoch,
            epochs=epochs,
            mse=round(mse, 8),
            elapsed=round(time.time() - started, 2),
        )
    return model


# ---------------------------------------------------------------------------
# 分块推理与提交输出
# ---------------------------------------------------------------------------

def predict_scores(model, table, start_date, end_date, instruments, stats, device):
    """分块预测模型原始分数，不加载任何因子表。"""
    mean, std = stats
    start_day = pd.Timestamp(start_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()
    outputs = []
    chunk_count = math.ceil(len(instruments) / CHUNK_SIZE)
    model.eval()

    for chunk_index, instrument_chunk in enumerate(
        _chunks(instruments, CHUNK_SIZE), start=1
    ):
        frame = _query_bar_chunk(table, start_date, end_date, instrument_chunk)
        windows, keys = [], []
        if not frame.empty:
            for _, one_stock in frame.groupby("instrument", sort=False, observed=True):
                one_windows, _, one_keys = _windows_from_instrument(
                    one_stock,
                    start_day,
                    end_day,
                    next_day_by_value=None,
                    need_label=False,
                )
                windows.extend(one_windows)
                keys.extend(one_keys)
        del frame

        if not windows:
            logger.warning("推理分块没有有效窗口", chunk=f"{chunk_index}/{chunk_count}")
            continue

        inputs = np.stack(windows).astype(np.float32, copy=False)
        _normalize_inplace(inputs, mean, std)
        tensor_inputs = torch.from_numpy(inputs)
        predictions = []
        with torch.no_grad():
            for batch_start in range(0, len(tensor_inputs), BATCH_SIZE):
                batch = tensor_inputs[batch_start:batch_start + BATCH_SIZE].to(device)
                predictions.append(model(batch).cpu().numpy())

        one_output = pd.DataFrame(keys, columns=["date", "instrument"])
        one_output["score"] = np.concatenate(predictions).astype(np.float64)
        outputs.append(one_output)
        cumulative = sum(len(item) for item in outputs)
        del windows, keys, inputs, tensor_inputs, predictions
        gc.collect()
        logger.info(
            "推理分钟分块完成",
            chunk=f"{chunk_index}/{chunk_count}",
            cumulative=cumulative,
        )

    if not outputs:
        raise RuntimeError(f"推理区间没有可用序列: {start_date} ~ {end_date}")
    result = pd.concat(outputs, ignore_index=True)
    result = _normalize_keys(result)
    return result.drop_duplicates(["date", "instrument"], keep="last")


def compose_submission_scores(raw_scores, members):
    """补齐股票池覆盖并严格返回 date、instrument、score 三列。"""
    raw_scores = _normalize_keys(raw_scores)
    members = _normalize_keys(members)
    result = members[["date", "instrument"]].drop_duplicates().merge(
        raw_scores[["date", "instrument", "score"]],
        on=["date", "instrument"],
        how="left",
    )
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result["score"] = result["score"].replace([np.inf, -np.inf], np.nan)

    daily_neutral = result.groupby("date")["score"].transform("median")
    finite_scores = result["score"].dropna()
    global_neutral = float(finite_scores.median()) if len(finite_scores) else 0.0
    result["score"] = result["score"].fillna(daily_neutral).fillna(global_neutral)
    result["score"] = result["score"].astype(np.float64)
    result = result[["date", "instrument", "score"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)

    expected_days = members["date"].nunique()
    if list(result.columns) != ["date", "instrument", "score"]:
        raise RuntimeError("提交列名不符合 date、instrument、score")
    if result.duplicated(["date", "instrument"]).any():
        raise RuntimeError("提交结果存在重复的 date、instrument")
    if result["date"].nunique() != expected_days:
        raise RuntimeError("提交结果缺少交易日")
    if not np.isfinite(result["score"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("提交结果仍包含非有限 score")
    return result


# ---------------------------------------------------------------------------
# JSON 权重读写
# ---------------------------------------------------------------------------

def save_model(checkpoint, model_path=MODEL_PATH):
    tensors = {}
    for name, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    payload = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, allow_nan=False)
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    state_dict = {}
    for name, metadata in payload["state_dict"].items():
        dtype = getattr(torch, metadata["dtype"])
        tensor = torch.tensor(metadata["data"], dtype=dtype)
        state_dict[name] = tensor.reshape(metadata["shape"]).to(map_location)
    checkpoint = {key: value for key, value in payload.items() if key != "state_dict"}
    checkpoint["state_dict"] = state_dict
    return checkpoint


def train_and_save(
    datasources,
    model_path=MODEL_PATH,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
):
    """训练、内部验证、全量重训并生成新的合规 JSON 权重。"""
    bar_table = datasources["bar1m"]
    _safe_table_name(bar_table)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "合规训练开始",
        bar_table=bar_table,
        start=train_start,
        end=train_end,
        device=str(device),
        chunk_size=CHUNK_SIZE,
        batch_size=BATCH_SIZE,
        input_fields=len(INPUT_FIELDS),
        max_train_samples=MAX_TRAIN_SAMPLES,
    )

    members = membership(train_start, train_end)
    instruments = members["instrument"].drop_duplicates().tolist()
    if MAX_TRAIN_INSTRUMENTS is not None:
        sample_size = min(int(MAX_TRAIN_INSTRUMENTS), len(instruments))
        random_generator = np.random.default_rng(SEED)
        instruments = sorted(
            random_generator.choice(
                np.asarray(instruments, dtype=object),
                size=sample_size,
                replace=False,
            ).tolist()
        )
        members = members[members["instrument"].isin(instruments)].copy()

    validation_start = _validation_start(members)
    (
        inputs,
        targets,
        dates,
        train_stats,
        full_stats,
        target_metadata,
    ) = build_train_dataset(
        bar_table,
        train_start,
        train_end,
        members,
        instruments,
        validation_start,
    )

    _normalize_inplace(inputs, *train_stats)
    best_epoch = _select_epoch(
        inputs, targets, dates, validation_start, device
    )

    _denormalize_inplace(inputs, *train_stats)
    _normalize_inplace(inputs, *full_stats)
    model = _fit_full_dataset(inputs, targets, best_epoch, device)
    parameter_count = validate_model_size(model)
    mean, std = full_stats

    save_model(
        {
            "state_dict": model.state_dict(),
            "format_version": 4,
            "model_cfg": MODEL_CFG,
            "parameter_count": parameter_count,
            "input_version": INPUT_VERSION,
            "input_fields": INPUT_FIELDS,
            "input_transform": INPUT_TRANSFORM,
            "seq_len": SEQ_LEN,
            "mean": np.asarray(mean, dtype=np.float32).tolist(),
            "std": np.asarray(std, dtype=np.float32).tolist(),
            "train_start": str(train_start),
            "train_end": str(train_end),
            "validation_start": str(validation_start),
            "best_epoch": int(best_epoch),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "max_train_instruments": MAX_TRAIN_INSTRUMENTS,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            **target_metadata,
        },
        model_path,
    )
    logger.info(
        "合规模型训练并保存完成",
        path=model_path,
        size_mb=round(os.path.getsize(model_path) / 1024 / 1024, 2),
        best_epoch=best_epoch,
        trainable_parameters=parameter_count,
    )
    del inputs, targets, dates, model
    gc.collect()
    return model_path


if __name__ == "__main__":
    data_sources = {"bar1m": DEFAULT_BAR_TABLE}
    train_and_save(data_sources)
