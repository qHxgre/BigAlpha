# -*- coding: utf-8 -*-
"""BigAlpha 单变量优化 07：Step02 使用完整公榜训练期。

设计目的
--------
1. 分钟序列只负责提取每只股票当日的原始量价表征。
2. 模型内部根据当日分钟表征计算股票间余弦相似度，不引入人工特征。
3. 每只股票先聚合最相似的 32 只股票，再进入全局截面注意力。
4. 沿用优化 01 的 Daily IC-Aligned 损失，不再改变训练目标。
5. 保持 Step02 的 600 只固定训练股票不变，只把训练结束日期延长到
   2024-12-31。
6. 从零训练后保存独立 JSON；推理入口严格返回
   date/instrument/score。

只使用主办方原始字段以及允许的逐字段 log1p/StandardScaler 预处理。
"""

import gc
import json
import math
import os
import random
import time

# 在 torch 初始化 CUDA 前声明确定性工作空间。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import dai
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = structlog.get_logger()
_HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 冻结实验配置
# =============================================================================

EXPERIMENT_ID = "step_next_07_step02_train2019_2024"
VERSION = "step_next_07_step02_train2019_2024_public_full_fit"
TARGET_NAME = "next_available_stock_day_return_daily_rank_minus1_plus1"
LOSS_NAME = (
    "0p25_daily_rank_mse_plus_0p75_daily_pearson_ic"
    "_plus_0p05_same_day_ranknet"
)

TRAIN_START = "2019-01-01 00:00:00"
TRAIN_END = "2024-12-31 23:59:59"
# 为保证本轮只改变训练结束日期，仍按 Step02 的原训练区间固定抽取
# 同一逻辑下的 600 只历史股票；下一轮再单独测试完整股票池。
INSTRUMENT_SELECTION_END = "2023-12-31 23:59:59"
# 公榜验证区间由平台隐藏，不能在本地硬编码。
VALID_START = None
VALID_END = None

MODEL_PATH = os.path.join(
    _HERE,
    "step_next_07_step02_train2019_2024_model.json",
)

SEQ_LEN = 64
EPOCHS = 5
LR = 5e-4
WEIGHT_DECAY = 1e-5
SEED = 20260719
MSE_WEIGHT = 0.25
DAILY_IC_WEIGHT = 0.75
PAIRWISE_WEIGHT = 0.05
PAIR_BUDGET_PER_DAY = 128
DAYS_PER_STEP = 4
MIN_DAILY_TRAIN_SAMPLES = 20
MAX_TRAIN_INSTRUMENTS = 600
TRAIN_INSTRUMENT_CHUNK_SIZE = 150
INFERENCE_INSTRUMENT_CHUNK_SIZE = 150
TEMPORAL_INFERENCE_BATCH = 1024
MIN_INFERENCE_COVERAGE = 0.60

PRICE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "bid_price1",
    "ask_price1",
]
VOL_COLS = [
    "volume",
    "amount",
    "bid_volume1",
    "ask_volume1",
]
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

MODEL_CFG = {
    "n_feat": N_FEAT,
    "d_model": 64,
    "temporal_heads": 4,
    "temporal_layers": 3,
    "temporal_ff": 128,
    "cross_heads": 4,
    "cross_layers": 1,
    "cross_ff": 128,
    "relation_dim": 32,
    "relation_topk": 32,
    "relation_temperature": 0.25,
    "seq_len": SEQ_LEN,
    "dropout": 0.10,
}


# =============================================================================
# 2. 确定性与模型
# =============================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class DailyCrossSectionTransformer(nn.Module):
    """分钟编码 → 相似股票图聚合 → 全局截面注意力。"""

    def __init__(
        self,
        n_feat,
        d_model=64,
        temporal_heads=4,
        temporal_layers=3,
        temporal_ff=128,
        cross_heads=4,
        cross_layers=1,
        cross_ff=128,
        relation_dim=32,
        relation_topk=32,
        relation_temperature=0.25,
        seq_len=SEQ_LEN,
        dropout=0.10,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.relation_topk = int(relation_topk)
        self.relation_temperature = float(relation_temperature)
        if self.relation_topk < 1:
            raise ValueError("relation_topk 必须为正整数")
        if self.relation_temperature <= 0:
            raise ValueError("relation_temperature 必须为正数")

        self.input_projection = nn.Linear(n_feat, d_model)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=temporal_heads,
            dim_feedforward=temporal_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.pool_query = nn.Parameter(torch.zeros(d_model))

        # 相似度完全由模型内部的当日分钟表征学习，不使用股票代码、
        # 行业、市值或预先计算的相关系数。对角线会被屏蔽，避免复制自身。
        self.relation_projection = nn.Linear(
            d_model,
            relation_dim,
            bias=False,
        )
        self.relation_value = nn.Linear(d_model, d_model)
        self.relation_output = nn.Linear(d_model, d_model)
        self.relation_gate = nn.Parameter(torch.tensor(-2.0))
        self.relation_norm = nn.LayerNorm(d_model)

        # 不加入股票顺序位置编码，保证股票排列变化时输出同步排列。
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cross_heads,
            dim_feedforward=cross_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(
            cross_layer,
            num_layers=cross_layers,
            enable_nested_tensor=False,
        )
        self.cross_gate = nn.Parameter(torch.tensor(0.0))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        nn.init.normal_(self.temporal_position, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    def encode_sequence(self, x):
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError(
                "输入必须为 [股票数, {}, 字段数]".format(self.seq_len)
            )
        hidden = self.input_projection(x) + self.temporal_position
        hidden = self.temporal_encoder(hidden)
        hidden = self.temporal_norm(hidden)
        attention_logits = torch.matmul(
            hidden,
            self.pool_query,
        ) / math.sqrt(self.d_model)
        attention_weight = torch.softmax(attention_logits, dim=1)
        return torch.sum(hidden * attention_weight.unsqueeze(-1), dim=1)

    def aggregate_similar_stocks(self, temporal_embedding):
        stock_count = len(temporal_embedding)
        if stock_count <= 1:
            return temporal_embedding

        relation_embedding = F.normalize(
            self.relation_projection(temporal_embedding),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        similarity = torch.matmul(
            relation_embedding,
            relation_embedding.transpose(0, 1),
        ) / self.relation_temperature
        diagonal_mask = torch.eye(
            stock_count,
            dtype=torch.bool,
            device=similarity.device,
        )
        similarity = similarity.masked_fill(diagonal_mask, float("-inf"))

        neighbor_count = min(self.relation_topk, stock_count - 1)
        neighbor_logits, neighbor_indices = torch.topk(
            similarity,
            k=neighbor_count,
            dim=-1,
            largest=True,
            sorted=False,
        )
        neighbor_weight = torch.softmax(neighbor_logits, dim=-1)
        relation_value = self.relation_value(temporal_embedding)
        neighbor_value = relation_value[neighbor_indices]
        aggregated = torch.sum(
            neighbor_value * neighbor_weight.unsqueeze(-1),
            dim=1,
        )
        relation_update = self.relation_output(aggregated)
        relation_gate = torch.sigmoid(self.relation_gate)
        return self.relation_norm(
            temporal_embedding + relation_gate * relation_update
        )

    def score_cross_section(self, temporal_embedding):
        if temporal_embedding.ndim != 2:
            raise ValueError("截面表征必须为 [股票数, d_model]")
        if len(temporal_embedding) == 0:
            raise ValueError("截面不能为空")
        relation_fused = self.aggregate_similar_stocks(
            temporal_embedding
        )
        cross_output = self.cross_encoder(
            relation_fused.unsqueeze(0)
        ).squeeze(0)
        gate = torch.sigmoid(self.cross_gate)
        fused = self.fusion_norm(
            relation_fused
            + gate * (cross_output - relation_fused)
        )
        score = self.score_head(fused).squeeze(-1)
        # 每日截面均值不参与排名，中心化可稳定训练和推理。
        return score - score.mean()

    def forward_day(self, x):
        return self.score_cross_section(self.encode_sequence(x))

    def forward(self, x):
        return self.forward_day(x)


def trainable_parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def validate_parameter_count(model):
    count = trainable_parameter_count(model)
    if not 100_000 <= count <= 100_000_000:
        raise RuntimeError("模型参数量不合规: {:,}".format(count))
    return count


# =============================================================================
# 3. 股票池与数据读取
# =============================================================================

def official_pool_frame(start_date, end_date):
    frame = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    if frame.empty:
        raise RuntimeError("官方股票池为空")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return (
        frame.dropna(subset=["date", "instrument"])
        .drop_duplicates(["date", "instrument"])
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def pool(start_date, end_date):
    frame = official_pool_frame(start_date, end_date)
    return sorted(frame["instrument"].unique().tolist())


def select_training_instruments(
    start_date,
    end_date,
    limit=MAX_TRAIN_INSTRUMENTS,
):
    """固定种子抽取广覆盖股票，避免只训练代码排序最前的一小块市场。"""
    instruments = np.asarray(pool(start_date, end_date), dtype=object)
    if limit is None or len(instruments) <= int(limit):
        return sorted(instruments.tolist())
    rng = np.random.default_rng(SEED)
    selected = rng.choice(
        instruments,
        size=int(limit),
        replace=False,
    )
    return sorted(selected.tolist())


def _query_and_prepare_frame(table, start_date, end_date, instruments):
    if not instruments:
        raise RuntimeError("查询股票列表为空")
    buffer_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=20)
    ).strftime("%Y-%m-%d")
    sql = (
        "SELECT date, instrument, {} FROM {} "
        "ORDER BY instrument, date"
    ).format(", ".join(FEATURE_COLS), table)
    frame = dai.query(
        sql,
        filters={
            "date": [buffer_start, end_date],
            "instrument": instruments,
        },
        compression=True,
    ).df()
    if frame.empty:
        raise RuntimeError(
            "查询无数据: table={}, {}~{}".format(
                table,
                buffer_start,
                end_date,
            )
        )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype(str)
    for column in FEATURE_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in VOL_COLS:
        frame[column] = np.log1p(frame[column].clip(lower=0))
    return frame


def _raw_training_chunk(table, start_date, end_date, instruments):
    frame = _query_and_prepare_frame(
        table,
        start_date,
        end_date,
        instruments,
    )
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    windows = []
    raw_returns = []
    keys = []

    for instrument, subframe in frame.groupby(
        "instrument",
        sort=False,
        observed=True,
    ):
        subframe = subframe.sort_values("date")
        if len(subframe) < SEQ_LEN:
            continue
        features = subframe[FEATURE_COLS].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        trading_days = subframe["date"].dt.normalize().to_numpy()
        day_end_positions = np.flatnonzero(
            np.append(trading_days[1:] != trading_days[:-1], True)
        )
        close_prices = subframe["close"].to_numpy(
            dtype=np.float64
        )[day_end_positions]
        dates = trading_days[day_end_positions]

        for day_number, end_position in enumerate(day_end_positions):
            date = pd.Timestamp(dates[day_number])
            if end_position + 1 < SEQ_LEN:
                continue
            if date < start_ts or date > end_ts:
                continue
            if day_number + 1 >= len(day_end_positions):
                continue
            current_close = close_prices[day_number]
            next_close = close_prices[day_number + 1]
            if current_close <= 0:
                continue
            future_return = next_close / current_close - 1.0
            if not np.isfinite(future_return):
                continue
            windows.append(
                features[end_position - SEQ_LEN + 1:end_position + 1]
            )
            raw_returns.append(np.float32(future_return))
            keys.append((date.normalize(), str(instrument)))

    del frame
    gc.collect()
    if not keys:
        return None, None, None
    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(raw_returns, dtype=np.float32),
        pd.DataFrame(keys, columns=["date", "instrument"]),
    )


def _raw_inference_chunk(table, start_date, end_date, instruments):
    frame = _query_and_prepare_frame(
        table,
        start_date,
        end_date,
        instruments,
    )
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    windows = []
    keys = []

    for instrument, subframe in frame.groupby(
        "instrument",
        sort=False,
        observed=True,
    ):
        subframe = subframe.sort_values("date")
        if len(subframe) < SEQ_LEN:
            continue
        features = subframe[FEATURE_COLS].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        trading_days = subframe["date"].dt.normalize().to_numpy()
        day_end_positions = np.flatnonzero(
            np.append(trading_days[1:] != trading_days[:-1], True)
        )
        dates = trading_days[day_end_positions]
        for end_position, raw_date in zip(day_end_positions, dates):
            date = pd.Timestamp(raw_date)
            if end_position + 1 < SEQ_LEN:
                continue
            if date < start_ts or date > end_ts:
                continue
            windows.append(
                features[end_position - SEQ_LEN + 1:end_position + 1]
            )
            keys.append((date.normalize(), str(instrument)))

    del frame
    gc.collect()
    if not keys:
        return None, None
    return (
        np.stack(windows).astype(np.float32, copy=False),
        pd.DataFrame(keys, columns=["date", "instrument"]),
    )


def _select_official_rows(x_data, index_frame, pool_frame):
    index_frame = index_frame.copy()
    index_frame["_row_id"] = np.arange(len(index_frame), dtype=np.int64)
    selected = index_frame.merge(
        pool_frame,
        on=["date", "instrument"],
        how="inner",
    )
    selected = selected.sort_values("_row_id").reset_index(drop=True)
    rows = selected.pop("_row_id").to_numpy(dtype=np.int64)
    return x_data[rows], selected


def _fit_standardization_stats(x_data):
    flat = x_data.reshape(-1, N_FEAT)
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("训练统计量包含 NaN 或无穷值")
    return mean, np.maximum(std, np.float32(1e-6))


def _apply_standardization(x_data, stats):
    mean, std = stats
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(
        np.asarray(std, dtype=np.float32),
        np.float32(1e-6),
    )
    if mean.shape != (N_FEAT,) or std.shape != (N_FEAT,):
        raise RuntimeError("标准化统计维度错误")
    x_data = np.where(np.isfinite(x_data), x_data, mean)
    return ((x_data - mean) / std).astype(np.float32, copy=False)


def build_training_dataset(table, start_date, end_date, instruments):
    started = time.time()
    pool_frame = official_pool_frame(start_date, end_date)
    x_parts = []
    index_parts = []

    total_chunks = (
        len(instruments) + TRAIN_INSTRUMENT_CHUNK_SIZE - 1
    ) // TRAIN_INSTRUMENT_CHUNK_SIZE
    for chunk_number, begin in enumerate(
        range(0, len(instruments), TRAIN_INSTRUMENT_CHUNK_SIZE),
        start=1,
    ):
        chunk = instruments[
            begin:begin + TRAIN_INSTRUMENT_CHUNK_SIZE
        ]
        raw_x, raw_return, chunk_index = _raw_training_chunk(
            table,
            start_date,
            end_date,
            chunk,
        )
        if chunk_index is None:
            continue
        chunk_index["raw_return"] = raw_return
        raw_x, chunk_index = _select_official_rows(
            raw_x,
            chunk_index,
            pool_frame,
        )
        if len(chunk_index) == 0:
            continue
        x_parts.append(raw_x)
        index_parts.append(chunk_index)
        logger.info(
            "相似股票图训练数据分块完成",
            chunk="{}/{}".format(chunk_number, total_chunks),
            samples=len(chunk_index),
        )

    if not index_parts:
        raise RuntimeError("训练集没有有效样本")
    raw_x = np.concatenate(x_parts, axis=0)
    index_frame = pd.concat(index_parts, ignore_index=True)
    del x_parts, index_parts
    gc.collect()

    daily_count = index_frame.groupby(
        "date",
        observed=True,
    )["instrument"].transform("count")
    valid = daily_count >= MIN_DAILY_TRAIN_SAMPLES
    raw_x = raw_x[valid.to_numpy()]
    index_frame = index_frame.loc[valid].reset_index(drop=True)

    ordinal_rank = index_frame.groupby(
        "date",
        observed=True,
    )["raw_return"].rank(method="average")
    group_count = index_frame.groupby(
        "date",
        observed=True,
    )["raw_return"].transform("count")
    target = (
        2.0
        * (ordinal_rank.to_numpy(dtype=np.float32) - 1.0)
        / (group_count.to_numpy(dtype=np.float32) - 1.0)
        - 1.0
    ).astype(np.float32, copy=False)
    if not np.isfinite(target).all():
        raise RuntimeError("每日排名标签包含非有限值")

    index_frame["_row_id"] = np.arange(len(index_frame), dtype=np.int64)
    ordered = index_frame.sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
    rows = ordered.pop("_row_id").to_numpy(dtype=np.int64)
    raw_x = raw_x[rows]
    target = target[rows]
    index_frame = ordered

    stats = _fit_standardization_stats(raw_x)
    x_data = _apply_standardization(raw_x, stats)
    del raw_x
    gc.collect()

    logger.info(
        "相似股票图训练集构建完成",
        samples=len(index_frame),
        days=index_frame["date"].nunique(),
        instruments=index_frame["instrument"].nunique(),
        min_daily_samples=int(
            index_frame.groupby("date", observed=True).size().min()
        ),
        max_daily_samples=int(
            index_frame.groupby("date", observed=True).size().max()
        ),
        x_mb=round(x_data.nbytes / 1024 ** 2, 2),
        elapsed=round(time.time() - started, 2),
    )
    return x_data, target, index_frame, stats


def _day_slices(index_frame):
    dates = index_frame["date"].to_numpy()
    if len(dates) == 0:
        return []
    boundaries = np.flatnonzero(
        np.append(True, dates[1:] != dates[:-1])
    )
    ends = np.append(boundaries[1:], len(dates))
    return [
        (int(begin), int(end), pd.Timestamp(dates[begin]))
        for begin, end in zip(boundaries, ends)
    ]


# =============================================================================
# 4. 损失与训练
# =============================================================================

def _daily_ic_loss(scores, labels, eps=1e-6):
    """直接优化单日横截面 Pearson 相关性。

    labels 已是当日未来收益排名，因此这里的 Pearson 相关性等价于
    对 Spearman/Rank IC 的平滑近似。每个交易日独立计算，避免大截面
    交易日在目标中获得额外权重。
    """
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("scores 和 labels 必须是一维张量")
    if len(scores) != len(labels) or len(scores) < 2:
        raise ValueError("每日 IC 至少需要两个等长样本")
    centered_scores = scores - scores.mean()
    centered_labels = labels - labels.mean()
    numerator = torch.sum(centered_scores * centered_labels)
    denominator = torch.sqrt(
        torch.sum(centered_scores.square()) + eps
    ) * torch.sqrt(
        torch.sum(centered_labels.square()) + eps
    )
    correlation = torch.clamp(
        numerator / denominator,
        min=-1.0,
        max=1.0,
    )
    return 1.0 - correlation, correlation


def _same_day_pairwise_loss(scores, labels, pair_budget, rng):
    count = len(labels)
    if count < 2 or pair_budget <= 0:
        return scores.sum() * 0.0, 0
    draw = max(int(pair_budget) * 3, int(pair_budget))
    left = rng.integers(0, count, size=draw, endpoint=False)
    right = rng.integers(0, count, size=draw, endpoint=False)
    valid = left != right
    left = left[valid]
    right = right[valid]
    if len(left) == 0:
        return scores.sum() * 0.0, 0
    left = left[:pair_budget]
    right = right[:pair_budget]
    left_tensor = torch.as_tensor(left, device=scores.device)
    right_tensor = torch.as_tensor(right, device=scores.device)
    label_difference = labels[left_tensor] - labels[right_tensor]
    non_tie = label_difference.abs() > 1e-7
    if int(non_tie.sum()) == 0:
        return scores.sum() * 0.0, 0
    left_tensor = left_tensor[non_tie]
    right_tensor = right_tensor[non_tie]
    direction = torch.sign(label_difference[non_tie])
    margin = scores[left_tensor] - scores[right_tensor]
    return F.softplus(-direction * margin).mean(), int(len(margin))


def train_and_save(datasources, model_path=MODEL_PATH):
    if "bar1m" not in datasources:
        raise KeyError("datasources 必须包含 bar1m")
    seed_everything(SEED)
    table = datasources["bar1m"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instruments = select_training_instruments(
        TRAIN_START,
        INSTRUMENT_SELECTION_END,
        MAX_TRAIN_INSTRUMENTS,
    )
    if len(instruments) != MAX_TRAIN_INSTRUMENTS:
        raise RuntimeError(
            "训练股票数量不足: {}/{}".format(
                len(instruments),
                MAX_TRAIN_INSTRUMENTS,
            )
        )

    logger.info(
        "开始训练相似股票图注意力模型",
        device=str(device),
        table=table,
        instruments=len(instruments),
        train_start=TRAIN_START,
        train_end=TRAIN_END,
    )
    x_train, y_train, train_index, stats = build_training_dataset(
        table,
        TRAIN_START,
        TRAIN_END,
        instruments,
    )
    day_slices = _day_slices(train_index)
    if len(day_slices) < 100:
        raise RuntimeError("有效训练日过少: {}".format(len(day_slices)))

    model = DailyCrossSectionTransformer(**MODEL_CFG).to(device)
    parameter_count = validate_parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    epoch_history = []
    training_started = time.time()

    model.train()
    for epoch in range(EPOCHS):
        epoch_started = time.time()
        rng = np.random.default_rng(SEED + epoch + 1)
        shuffled_days = rng.permutation(len(day_slices))
        total_mse = 0.0
        total_samples = 0
        total_daily_ic = 0.0
        ic_day_count = 0
        negative_ic_days = 0
        total_pairwise = 0.0
        total_pairs = 0
        optimizer_steps = 0

        for group_begin in range(0, len(shuffled_days), DAYS_PER_STEP):
            group = shuffled_days[
                group_begin:group_begin + DAYS_PER_STEP
            ]
            optimizer.zero_grad(set_to_none=True)
            group_loss = None

            for day_id in group:
                begin, end, _ = day_slices[int(day_id)]
                features = torch.from_numpy(x_train[begin:end]).to(
                    device,
                    non_blocking=True,
                )
                labels = torch.from_numpy(y_train[begin:end]).to(
                    device,
                    non_blocking=True,
                )
                predictions = model.forward_day(features)
                mse = F.mse_loss(predictions, labels)
                ic_loss, daily_ic = _daily_ic_loss(
                    predictions,
                    labels,
                )
                pairwise, pair_count = _same_day_pairwise_loss(
                    predictions,
                    labels,
                    PAIR_BUDGET_PER_DAY,
                    rng,
                )
                day_loss = (
                    MSE_WEIGHT * mse
                    + DAILY_IC_WEIGHT * ic_loss
                    + PAIRWISE_WEIGHT * pairwise
                )
                group_loss = (
                    day_loss if group_loss is None else group_loss + day_loss
                )

                samples = end - begin
                total_mse += float(mse.detach()) * samples
                total_samples += samples
                daily_ic_value = float(daily_ic.detach())
                total_daily_ic += daily_ic_value
                ic_day_count += 1
                negative_ic_days += int(daily_ic_value < 0.0)
                total_pairwise += float(pairwise.detach()) * pair_count
                total_pairs += pair_count

            group_loss = group_loss / max(len(group), 1)
            group_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps += 1

        mean_mse = total_mse / max(total_samples, 1)
        mean_daily_ic = total_daily_ic / max(ic_day_count, 1)
        mean_pairwise = total_pairwise / max(total_pairs, 1)
        record = {
            "epoch": epoch + 1,
            "total_loss": float(
                MSE_WEIGHT * mean_mse
                + DAILY_IC_WEIGHT * (1.0 - mean_daily_ic)
                + PAIRWISE_WEIGHT * mean_pairwise
            ),
            "mse": float(mean_mse),
            "daily_ic": float(mean_daily_ic),
            "negative_ic_days": int(negative_ic_days),
            "ic_days": int(ic_day_count),
            "pairwise": float(mean_pairwise),
            "pairs": int(total_pairs),
            "steps": int(optimizer_steps),
            "cross_gate": float(
                torch.sigmoid(model.cross_gate).detach().cpu()
            ),
            "relation_gate": float(
                torch.sigmoid(model.relation_gate).detach().cpu()
            ),
            "elapsed_seconds": round(time.time() - epoch_started, 2),
        }
        epoch_history.append(record)
        logger.info("相似股票图注意力 epoch 完成", **record)

    mean, std = stats
    daily_sizes = train_index.groupby("date", observed=True).size()
    checkpoint = {
        "state_dict": model.state_dict(),
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "target": TARGET_NAME,
        "loss": LOSS_NAME,
        "mse_weight": MSE_WEIGHT,
        "daily_ic_weight": DAILY_IC_WEIGHT,
        "pairwise_weight": PAIRWISE_WEIGHT,
        "pair_budget_per_day": PAIR_BUDGET_PER_DAY,
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
        "parameter_count": int(parameter_count),
        "seed": SEED,
        "deterministic_algorithms": True,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "instrument_selection_end": INSTRUMENT_SELECTION_END,
        "validation_start": VALID_START,
        "validation_end": VALID_END,
        "train_instruments": instruments,
        "data_info": {
            "samples": int(len(train_index)),
            "days": int(train_index["date"].nunique()),
            "instruments": int(train_index["instrument"].nunique()),
            "min_daily_samples": int(daily_sizes.min()),
            "mean_daily_samples": float(daily_sizes.mean()),
            "max_daily_samples": int(daily_sizes.max()),
        },
        "epoch_history": epoch_history,
        "training_seconds": round(time.time() - training_started, 2),
    }
    save_model(checkpoint, model_path)
    loaded = load_model(model_path, map_location="cpu")
    validate_checkpoint(loaded)
    logger.info(
        "相似股票图 JSON 权重保存并回读完成",
        path=model_path,
        size_mb=round(os.path.getsize(model_path) / 1024 ** 2, 2),
    )

    del x_train, y_train, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model_path


# =============================================================================
# 5. JSON 权重保存与检查
# =============================================================================

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def save_model(checkpoint, model_path=MODEL_PATH):
    tensors = {}
    for name, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise RuntimeError("权重包含非有限值: {}".format(name))
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    payload = {
        key: value
        for key, value in checkpoint.items()
        if key != "state_dict"
    }
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    state_dict = {}
    for name, metadata in payload["state_dict"].items():
        dtype_name = metadata["dtype"]
        if dtype_name not in _DTYPE_MAP:
            raise RuntimeError("不支持的权重 dtype: {}".format(dtype_name))
        tensor = torch.tensor(
            metadata["data"],
            dtype=_DTYPE_MAP[dtype_name],
        )
        state_dict[name] = tensor.reshape(
            metadata["shape"]
        ).to(map_location)
    checkpoint = {
        key: value
        for key, value in payload.items()
        if key != "state_dict"
    }
    checkpoint["state_dict"] = state_dict
    return checkpoint


def validate_checkpoint(checkpoint):
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "target": TARGET_NAME,
        "loss": LOSS_NAME,
        "feature_cols": FEATURE_COLS,
        "model_cfg": MODEL_CFG,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "instrument_selection_end": INSTRUMENT_SELECTION_END,
    }
    for key, expected_value in expected.items():
        if checkpoint.get(key) != expected_value:
            raise RuntimeError(
                "权重身份检查失败: {}={!r}, expected={!r}".format(
                    key,
                    checkpoint.get(key),
                    expected_value,
                )
            )
    if abs(
        float(checkpoint.get("mse_weight", -1.0))
        - MSE_WEIGHT
    ) > 1e-12:
        raise RuntimeError("权重 MSE 系数错误")
    if abs(
        float(checkpoint.get("daily_ic_weight", -1.0))
        - DAILY_IC_WEIGHT
    ) > 1e-12:
        raise RuntimeError("权重 Daily IC 系数错误")
    if abs(
        float(checkpoint.get("pairwise_weight", -1.0))
        - PAIRWISE_WEIGHT
    ) > 1e-12:
        raise RuntimeError("权重 Pairwise 系数错误")


# =============================================================================
# 6. 隐藏区间推理：先分块编码，再按完整每日股票池做截面交互
# =============================================================================

def _encode_chunk(model, x_data, device):
    embeddings = []
    x_tensor = torch.from_numpy(x_data)
    with torch.inference_mode():
        for begin in range(0, len(x_tensor), TEMPORAL_INFERENCE_BATCH):
            features = x_tensor[
                begin:begin + TEMPORAL_INFERENCE_BATCH
            ].to(device, non_blocking=True)
            embeddings.append(
                model.encode_sequence(features).cpu().numpy()
            )
    if not embeddings:
        raise RuntimeError("时序编码没有输出")
    return np.concatenate(embeddings).astype(np.float32, copy=False)


def predict_scores(
    datasources,
    start_date,
    end_date,
    model_path=MODEL_PATH,
    instrument_chunk_size=INFERENCE_INSTRUMENT_CHUNK_SIZE,
):
    if "bar1m" not in datasources:
        raise KeyError("datasources 必须包含 bar1m")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "未找到相似股票图模型权重: {}".format(model_path)
        )
    table = datasources["bar1m"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_model(model_path, map_location="cpu")
    validate_checkpoint(checkpoint)

    model = DailyCrossSectionTransformer(**checkpoint["model_cfg"])
    parameter_count = validate_parameter_count(model)
    if parameter_count != int(checkpoint.get("parameter_count", -1)):
        raise RuntimeError("权重参数量记录不一致")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    stats = (
        np.asarray(checkpoint["mean"], dtype=np.float32),
        np.asarray(checkpoint["std"], dtype=np.float32),
    )

    stock_pool = official_pool_frame(start_date, end_date)
    instruments = sorted(stock_pool["instrument"].unique().tolist())
    index_parts = []
    embedding_parts = []
    total_chunks = (
        len(instruments) + instrument_chunk_size - 1
    ) // instrument_chunk_size

    for chunk_number, begin in enumerate(
        range(0, len(instruments), instrument_chunk_size),
        start=1,
    ):
        chunk = instruments[begin:begin + instrument_chunk_size]
        raw_x, chunk_index = _raw_inference_chunk(
            table,
            start_date,
            end_date,
            chunk,
        )
        if chunk_index is None:
            continue
        raw_x, chunk_index = _select_official_rows(
            raw_x,
            chunk_index,
            stock_pool,
        )
        if len(chunk_index) == 0:
            continue
        x_data = _apply_standardization(raw_x, stats)
        chunk_embedding = _encode_chunk(model, x_data, device)
        index_parts.append(chunk_index)
        embedding_parts.append(chunk_embedding)
        logger.info(
            "相似股票图推理时序编码分块完成",
            chunk="{}/{}".format(chunk_number, total_chunks),
            samples=len(chunk_index),
        )
        del raw_x, x_data, chunk_embedding, chunk_index
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not index_parts:
        raise RuntimeError("相似股票图推理没有有效样本")
    index_frame = pd.concat(index_parts, ignore_index=True)
    embeddings = np.concatenate(embedding_parts, axis=0)
    del index_parts, embedding_parts
    gc.collect()

    index_frame["_row_id"] = np.arange(len(index_frame), dtype=np.int64)
    index_frame = index_frame.sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
    rows = index_frame.pop("_row_id").to_numpy(dtype=np.int64)
    embeddings = embeddings[rows]

    scores = np.empty(len(index_frame), dtype=np.float64)
    day_slices = _day_slices(index_frame)
    with torch.inference_mode():
        for begin, end, _ in day_slices:
            day_embedding = torch.from_numpy(
                embeddings[begin:end]
            ).to(device, non_blocking=True)
            scores[begin:end] = (
                model.score_cross_section(day_embedding)
                .cpu()
                .numpy()
                .astype(np.float64)
            )

    result = index_frame[["date", "instrument"]].copy()
    result["score"] = scores
    result = (
        result.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    if list(result.columns) != ["date", "instrument", "score"]:
        raise RuntimeError("输出列不符合要求")
    if result.duplicated(["date", "instrument"]).any():
        raise RuntimeError("输出存在重复键")
    if not np.isfinite(result["score"].to_numpy(np.float64)).all():
        raise RuntimeError("输出包含非有限分数")

    expected = stock_pool.groupby(
        "date",
        observed=True,
    )["instrument"].nunique()
    actual = result.groupby(
        "date",
        observed=True,
    )["instrument"].nunique()
    coverage = (actual / expected).fillna(0.0)
    if set(expected.index) != set(actual.index):
        raise RuntimeError("输出缺少交易日")
    if coverage.empty or float(coverage.min()) < MIN_INFERENCE_COVERAGE:
        raise RuntimeError(
            "最低覆盖率不足: {:.2%}".format(float(coverage.min()))
        )
    logger.info(
        "相似股票图分数生成完成",
        rows=len(result),
        days=result["date"].nunique(),
        instruments=result["instrument"].nunique(),
        min_coverage="{:.2%}".format(float(coverage.min())),
    )
    return result


def main(datasources, start_date, end_date):
    return predict_scores(datasources, start_date, end_date)


if __name__ == "__main__":
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})
