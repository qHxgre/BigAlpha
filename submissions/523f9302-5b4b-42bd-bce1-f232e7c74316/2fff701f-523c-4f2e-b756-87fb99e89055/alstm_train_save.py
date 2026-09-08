# -*- coding: utf-8 -*-
"""ALSTM 端到端 demo —— 训练侧脚本。

保留官方模板中的主要接口名称：
    MODEL_PATH
    FEATURE_COLS
    SEQ_LEN
    build_dataset(...)
    save_model(...)
    load_model(...)
    train_and_save(...)

保留原有数据读取、特征处理和训练流程，
公榜阶段可直接加载本地训练得到的 Transformer_model.json；
私榜阶段平台可调用 train_and_save(...) 从零重训。
"""

import os
import json
import math
import time
import random
import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

try:
    from bigquant import dai
except ImportError:
    dai = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Sampler
import structlog

logger = structlog.get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))


# ALSTM 模型权重文件。
MODEL_PATH = os.path.join(_HERE, "alstm_model.json")


# ---------- 配置 ----------
TRAIN_START = "2022-01-01"
TRAIN_END = "2024-06-30 23:59:59"
VALID_START = "2024-07-01"
VALID_END = "2024-12-31 23:59:" \
"59"
LOCAL_DATA_ROOT = Path(os.environ.get("BIGALPHA_DATA_ROOT", "/home/hzf/bigalpha/data")).expanduser().resolve()

SEQ_LEN = 240
PRED_LEN = 1

EPOCHS = 100
EARLY_STOP_PATIENCE = 10
BATCH = 1000
LR = 0.0001
WEIGHT_DECAY = 1e-3
SEED = 42
GRAD_CLIP = 1.0


# 正式训练建议为 None；调试时可设为较小整数。
MAX_TRAIN_INSTRUMENTS=None

PRICE_COLS = [
    "open",
    "high",
    "low",
    "close",
    # "bid_price1",
    # "ask_price1",
]

VOL_COLS = [
    # "volume",
    # "amount",
    # "bid_volume1",
    # "ask_volume1",
]

FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

# ALSTM 模型超参数。
#
# 输入仍为 [B, SEQ_LEN, N_FEAT]，输出仍为 [B]，
# 因此数据集、标签和训练循环保持不变。
MODEL_CFG = {
    "n_feat": N_FEAT,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "rnn_type": "GRU",
}


# =============================================================================
# 随机种子
# =============================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# =============================================================================
# StockALSTM 模型
# =============================================================================

class StockALSTM(nn.Module):
    """带时间注意力机制的 ALSTM，用最近 SEQ_LEN 根 bar 预测下一交易日median price return。

    输入:
        x: [B, L, F]
    输出:
        score: [B]
    """

    def __init__(
        self,
        n_feat: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "GRU",
    ):
        super().__init__()

        if hidden_size < 2:
            raise ValueError("hidden_size 必须大于等于 2")

        try:
            rnn_class = getattr(nn, rnn_type.upper())
        except AttributeError as exc:
            raise ValueError(f"不支持的 rnn_type：{rnn_type}") from exc

        self.n_feat = int(n_feat)
        self.hidden_size = int(hidden_size)
        self.rnn_type = str(rnn_type).upper()

        # 与标准 ALSTM 一致：先将每个时间步映射到隐藏空间。
        self.input_norm = nn.LayerNorm(self.n_feat)
        self.fc_in = nn.Linear(self.n_feat, self.hidden_size)
        self.act = nn.Tanh()

        self.rnn = rnn_class(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 对所有时间步生成归一化注意力权重。
        self.att_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # 拼接最后一个时间步状态与注意力汇聚状态。
        self.fc_out = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"StockALSTM 期望输入形状 [B, L, F]，实际为 {tuple(x.shape)}"
            )
        if x.size(-1) != self.n_feat:
            raise ValueError(
                f"输入特征维度不一致：模型={self.n_feat}，输入={x.size(-1)}"
            )

        x = self.input_norm(x)
        x = self.act(self.fc_in(x))

        rnn_out, _ = self.rnn(x)
        attention_score = self.att_net(rnn_out)
        attention_output = torch.sum(rnn_out * attention_score, dim=1)

        combined = torch.cat(
            [rnn_out[:, -1, :], attention_output],
            dim=1,
        )
        return self.fc_out(combined).squeeze(-1)


# =============================================================================
# 数据
# =============================================================================

def local_table_dir(table: str) -> Path:
    return LOCAL_DATA_ROOT / table


def all_local_parquet_files(table: str) -> List[Path]:
    """仅返回完整的 .parquet 文件，忽略 manifest、.part 和其他文件。

    同时兼容：
    - 1m：按天保存，例如 *_2024-01-02.parquet
    - 5m/15m/30m：按月保存，例如 *_2024-01.parquet
    """
    folder = local_table_dir(table)

    if not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.glob("*.parquet")
        if path.is_file() and path.stat().st_size > 0
    )


def _file_period(path: Path, table: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """从文件名解析该 Parquet 覆盖的日期范围。

    解析失败时返回 None；调用方会保守地保留该文件。
    """
    prefix = f"{table}_"
    name = path.stem

    if not name.startswith(prefix):
        return None

    label = name[len(prefix):]

    try:
        # 1m 按天：YYYY-MM-DD
        if len(label) == 10:
            day = pd.Timestamp(label).normalize()
            return day, day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        # 其他频率按月：YYYY-MM
        if len(label) == 7:
            period = pd.Period(label, freq="M")
            return period.start_time, period.end_time
    except Exception:
        return None

    return None


def local_parquet_files(
    table: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Path]:
    """返回与查询区间相交的本地 Parquet 文件。

    文件按天或按月存储均可；非交易日没有文件也不会造成问题。
    """
    files = all_local_parquet_files(table)

    if not files:
        return []

    if start is None or end is None:
        return files

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    selected: List[Path] = []

    for path in files:
        period = _file_period(path, table)

        # 文件名无法解析时保守保留，后续仍由 Arrow 的 date filter 筛选。
        if period is None:
            selected.append(path)
            continue

        file_start, file_end = period

        if file_end >= start_ts and file_start <= end_ts:
            selected.append(path)

    return selected


def has_local_table(table: str) -> bool:
    return bool(all_local_parquet_files(table))


def bars_per_day(table: str) -> int:
    if table.endswith("bar1m"):
        return 240
    if table.endswith("bar5m"):
        return 48
    if table.endswith("bar15m"):
        return 16
    if table.endswith("bar30m"):
        return 8
    raise ValueError(f"无法从表名判断频率：{table}")


def canonicalize_local(df: pd.DataFrame) -> pd.DataFrame:
    """把本地 e2e 压缩表统一到云端 stock 表的量纲。"""
    df = df.copy()

    # 在任何清洗、缩放和填充之前记录原始无效值。
    # 训练窗口只要包含任一特征值 0 或 -1，就整条样本不参与 loss。
    mask_cols = [col for col in FEATURE_COLS if col in df.columns]
    raw_numeric = df[mask_cols].apply(pd.to_numeric, errors="coerce")
    df["_invalid_zero_or_minus_one"] = (
        raw_numeric.eq(0) | raw_numeric.eq(-1)
    ).any(axis=1)

    # 本地 OHLC 中 -1 和 0 均表示缺失。
    # BigAlpha e2e 数据中部分股票存在 close=0 的缺失区间，
    # 不应作为真实价格参与模型输入。
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    # 本地价格及 amount 以“分”为单位，恢复成“元”。
    cent_cols = [
        "open", "high", "low", "close", "amount",
        "ask_price1", "ask_price2", "ask_price3",
        "bid_price1", "bid_price2", "bid_price3",
    ]

    for col in cent_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .astype("float64")
                / 100.0
            )

    df["date"] = pd.to_datetime(df["date"])

    # 本地训练仅要求稳定且唯一的股票键，不必先映射回字符串代码。
    if "instrument_id" in df.columns:
        df["instrument_key"] = pd.to_numeric(
            df["instrument_id"],
            errors="coerce",
        ).astype("Int32")
    elif "instrument" in df.columns:
        df["instrument_key"] = df["instrument"].astype(str)
    else:
        raise KeyError("本地数据既没有 instrument_id，也没有 instrument")

    return df


def read_local_data(
    table: str,
    start: str,
    end: str,
    columns: Sequence[str],
    instruments: Optional[Sequence[Union[int, str]]] = None,
) -> pd.DataFrame:
    """像 dai.query 一样按日期、字段和股票读取本地 Parquet。

    支持：
    - 1m 按天文件；
    - 5m/15m/30m 按月文件；
    - 非交易日没有文件；
    - 下载目录中存在 manifest 或 .part 文件。
    """
    folder = local_table_dir(table)

    if not folder.exists():
        raise FileNotFoundError(f"本地表目录不存在：{folder}")

    files = local_parquet_files(
        table=table,
        start=start,
        end=end,
    )

    if not files:
        logger.warning(
            "查询区间没有匹配的本地 Parquet 文件",
            table=table,
            start=str(start),
            end=str(end),
            folder=str(folder),
        )
        return pd.DataFrame(columns=list(columns))

    # 显式传入完整 .parquet 文件列表，避免读取 manifest、.part 等文件。
    dataset = pads.dataset(
        [str(path) for path in files],
        format="parquet",
    )

    schema_names = set(dataset.schema.names)
    requested = list(columns)

    # 云端接口使用 instrument，本地 e2e 表通常使用 instrument_id。
    if (
        "instrument" in requested
        and "instrument" not in schema_names
        and "instrument_id" in schema_names
    ):
        requested.remove("instrument")
        requested.append("instrument_id")

    requested = list(dict.fromkeys(requested))

    missing = [
        col
        for col in requested
        if col not in schema_names
    ]

    if missing:
        raise KeyError(
            f"本地表 {table} 缺少字段：{missing}\n"
            f"实际字段：{sorted(schema_names)}"
        )

    start_ts = pd.Timestamp(start).to_pydatetime()
    end_ts = pd.Timestamp(end).to_pydatetime()

    expression = (
        (pads.field("date") >= start_ts)
        & (pads.field("date") <= end_ts)
    )

    instrument_col = (
        "instrument_id"
        if "instrument_id" in schema_names
        else "instrument"
    )

    if instruments is not None:
        values = list(instruments)

        if values:
            # 避免 Int32/字符串类型不一致造成过滤为空。
            if instrument_col == "instrument_id":
                values = [int(value) for value in values]
            else:
                values = [str(value) for value in values]

            expression = (
                expression
                & pads.field(instrument_col).isin(values)
            )

    logger.info(
        "读取本地数据",
        table=table,
        files=len(files),
        first_file=files[0].name,
        last_file=files[-1].name,
        start=str(start),
        end=str(end),
        columns=requested,
        instruments=(
            None
            if instruments is None
            else len(instruments)
        ),
    )

    arrow_table = dataset.to_table(
        columns=requested,
        filter=expression,
    )

    if arrow_table.num_rows == 0:
        return pd.DataFrame(columns=requested)

    df = arrow_table.to_pandas(
        split_blocks=True,
        self_destruct=True,
    )

    return canonicalize_local(df)


def read_cloud_data(
    table: str,
    start: str,
    end: str,
    columns: Sequence[str],
    instruments: Optional[Sequence[Union[int, str]]] = None,
) -> pd.DataFrame:
    """
    云端读取，与本地 e2e 数据处理对齐：

    1. 数据源:
       bigalpha_2026_stock_bar1m

    2. 差异处理:
       - instrument(str) -> instrument_key
       - OHLC 已经是 float 元，无需 /100
       - amount 已经是 float 元，无需 /100
       - OHLC NaN 视为缺失
       - 盘口缺失 0 保持一致

    3. 大数据读取:
       - 日期按月切片
       - 股票按 chunk 分批
       避免一次加载整个 2019-2024 bar1m。
    """

    if dai is None:
        raise RuntimeError("没有安装 BigQuant SDK")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    month_list = pd.date_range(
        start_ts.strftime("%Y-%m-01"),
        end_ts.strftime("%Y-%m-01"),
        freq="MS",
    )

    if instruments is None:
        instrument_chunks = [None]
    else:
        chunk_size = 200
        instrument_chunks = [
            list(instruments[i:i + chunk_size])
            for i in range(0, len(instruments), chunk_size)
        ]

    outputs = []

    for month in month_list:
        month_start = max(month, start_ts)
        month_end = min(
            month + pd.offsets.MonthEnd(1),
            end_ts,
        )

        for chunk in instrument_chunks:

            sql = f"""
            SELECT {','.join(columns)}
            FROM {table}
            """

            if chunk is not None:
                values = ",".join(
                    "'" + str(x) + "'"
                    for x in chunk
                )
                sql += f"""
                WHERE instrument IN ({values})
                """

            # 固定云端读取顺序，与本地训练保持一致：
            # instrument -> date
            # 避免不同 chunk / month 查询返回顺序变化影响后续数据处理
            sql += """
            ORDER BY instrument, date
            """

            logger.info(
                "cloud query",
                table=table,
                start=str(month_start),
                end=str(month_end),
                instruments=None if chunk is None else len(chunk),
            )

            df = dai.query(
                sql,
                filters={
                    "date": [
                        str(month_start),
                        str(month_end),
                    ]
                },
                compression=True,
            ).df()

            if not df.empty:
                outputs.append(df)

    if not outputs:
        return pd.DataFrame(columns=list(columns))

    df = pd.concat(outputs, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["instrument_key"] = df["instrument"].astype(str)

    # 与本地 canonicalize_local 对齐：
    # 在任何填充前记录 invalid mask
    raw_numeric = df[
        [c for c in FEATURE_COLS if c in df.columns]
    ].apply(pd.to_numeric, errors="coerce")

    # 与本地训练保持一致：
    # 原始数据阶段记录 invalid mask。
    # 云端数据中 close/open/high/low 的 NaN、0、-1 均视为无效，
    # 对应样本窗口不参与训练。
    df["_invalid_zero_or_minus_one"] = (
        raw_numeric.eq(0)
        |
        raw_numeric.eq(-1)
        |
        raw_numeric.isna()
    ).any(axis=1)

    # 云端价格已经为元
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


def read_market_data(table: str, start: str, end: str,
                     columns: Sequence[str],
                     instruments: Optional[Sequence[Union[int, str]]] = None) -> pd.DataFrame:
    if has_local_table(table):
        return read_local_data(table, start, end, columns, instruments)
    return read_cloud_data(table, start, end, columns, instruments)


def pool(table: str, sd: str, ed: str) -> List[Union[int, str]]:
    if has_local_table(table):
        df = read_local_data(table, sd, ed, ["date", "instrument_id"], None)
        values = df["instrument_key"].dropna().drop_duplicates().sort_values().tolist()
    else:
        if dai is None:
            raise RuntimeError("无法访问本地表或云端 dai")
        df = dai.query(
            f"""
            SELECT DISTINCT instrument
            FROM {table}
            ORDER BY instrument
            """,
            filters={"date": [sd, ed]},
        ).df()
        values = df["instrument"].dropna().astype(str).drop_duplicates().sort_values().tolist()

    if not values:
        raise RuntimeError(f"股票池为空：table={table}, {sd}~{ed}")
    return values


def build_dataset(
    table: str,
    sd: str,
    ed: str,
    mode: str,
    instruments: Sequence,
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """训练保持原数据与标签逻辑；推理可由单文件Notebook负责。"""
    if mode not in {"train", "infer"}:
        raise ValueError("mode 必须为 train 或 infer")

    started = time.time()
    # ==============================
    # Warmup history
    # ==============================
    # 固定读取最近30个自然日历史。
    #
    # 设计原因:
    # 1. 官方测试端可能使用不同频率(bar1m/bar5m/bar15m/bar30m)，
    #    无法依赖 bars_per_day 推算。
    # 2. 240天warmup会导致高频数据(bar1m)一次读取过多，
    #    增加内存压力。
    # 3. 当前模型SEQ_LEN=240:
    #    - bar1m/bar5m/bar15m有充足历史；
    #    - bar30m极端情况下由后续padding机制兜底。
    #
    # 与测试端保持一致:
    WARMUP_DAYS = 30

    buffer_days = WARMUP_DAYS
    buffer_start = (
        pd.Timestamp(sd)
        - pd.Timedelta(days=buffer_days)
    ).strftime("%Y-%m-%d")

    df = read_market_data(
        table,
        buffer_start,
        ed,
        ["date", "instrument", *FEATURE_COLS],
        instruments,
    )
    if df.empty:
        raise RuntimeError(f"未读取到数据：table={table}, {buffer_start}~{ed}")

    if "_invalid_zero_or_minus_one" not in df.columns:
        raw_numeric = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        df["_invalid_zero_or_minus_one"] = (
            raw_numeric.eq(0)
            | raw_numeric.eq(-1)
            | raw_numeric.isna()
        ).any(axis=1)

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # volume / amount:
    # -1 和 0 均视为无成交，log1p 后保持为 0。
    for col in VOL_COLS:
        df[col] = np.log1p(df[col].clip(lower=0))

    # ==============================
    # 缺失值处理
    # 保持原有缺失值处理流程:
    # 1. 按股票 forward fill 最近有效价格
    # 2. 开头无历史价格的位置再填 0
    # ==============================
    df = df.sort_values(
        ["instrument_key", "date"]
    )

    for col in PRICE_COLS:
        df[col] = (
            df.groupby("instrument_key")[col]
            .ffill()
        )

    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    sd_ts, ed_ts = pd.Timestamp(sd), pd.Timestamp(ed)
    windows, labels, keys = [], [], []

    for instrument, sub in df.groupby("instrument_key", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        features = sub[FEATURE_COLS].to_numpy(np.float32)
        invalid_rows = sub["_invalid_zero_or_minus_one"].fillna(True).to_numpy(bool)
        natural_dates = sub["date"].dt.normalize().to_numpy()
        close_positions = np.flatnonzero(np.append(natural_dates[1:] != natural_dates[:-1], True))
        close_dates = natural_dates[close_positions]
        close_prices = sub["close"].to_numpy(np.float64)[close_positions]

        for k, position in enumerate(close_positions):
            date = pd.Timestamp(close_dates[k])
            if date < sd_ts or date > ed_ts or position + 1 < SEQ_LEN:
                continue

            label = None

            # ==========================================
            # Future median price return label
            #
            # Use the median price of all minute bars
            # in the next trading day as the future price.
            #
            # y =
            # median(price_{t+1}) / close_t - 1
            #
            # This reduces sensitivity to:
            # - closing auction noise
            # - extreme intraday spikes
            # - abnormal last-bar movements
            # ==========================================

            if (
                k + 1 < len(close_positions)
                and np.isfinite(close_prices[k])
                and close_prices[k] > 0
            ):
                next_start = close_positions[k] + 1
                next_end = close_positions[k + 1] + 1

                if next_end > next_start:
                    next_prices = sub["close"].to_numpy(np.float64)[
                        next_start:next_end
                    ]

                    valid_prices = (
                        np.isfinite(next_prices)
                        &
                        (next_prices > 0)
                    )

                    if valid_prices.any():
                        future_median_price = np.median(
                            next_prices[valid_prices]
                        )

                        value = (
                            future_median_price
                            /
                            close_prices[k]
                            -
                            1.0
                        )

                        if np.isfinite(value):
                            label = np.float32(value)

            if mode == "train" and label is None:
                continue

            window_start = position - SEQ_LEN + 1
            window = features[window_start: position + 1]
            if window.shape != (SEQ_LEN, N_FEAT):
                continue

            # 样本级 mask：原始窗口中任一输入特征出现 0 或 -1，
            # 该训练样本不进入 DataLoader，因此完全不参与 loss。
            if mode == "train" and invalid_rows[window_start: position + 1].any():
                continue

            windows.append(window)
            labels.append(label if label is not None else np.float32(0.0))
            keys.append((date, instrument))

    if not windows:
        raise RuntimeError(f"build_dataset 无样本：mode={mode}, table={table}, {sd}~{ed}")

    X = np.stack(windows).astype(np.float32)

    if stats is None:
        if mode != "train":
            raise ValueError("推理模式必须传入训练集 stats")

        flat = X.reshape(-1, N_FEAT)
        mean = np.nanmean(flat, axis=0).astype(np.float32)
        std = np.nanstd(flat, axis=0).astype(np.float32)
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
        stats = (mean, std)

    mean, std = stats

    # 前面已经完成 ffill + 0 填充，
    # 这里只作为异常保护。
    X = np.where(
        np.isfinite(X),
        X,
        0.0
    )

    X = ((X - mean) / std).astype(np.float32)

    logger.info(f"{mode} 集构建完成", samples=len(keys), elapsed=round(time.time() - started, 2))

    if mode == "train":
        return (
            X,
            np.asarray(labels, np.float32),
            pd.DataFrame(keys, columns=["date", "instrument"]),
            stats,
        )
    return X, None, pd.DataFrame(keys, columns=["date", "instrument"]), stats


# =============================================================================
# 模型 JSON 保存与加载
# =============================================================================

def save_model(
    checkpoint: Dict[str, object],
    model_path: str = MODEL_PATH,
) -> str:
    """将 checkpoint 保存为纯文本 JSON。"""
    tensors = {}

    for name, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu()

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
            separators=(",", ":"),
        )

    return model_path


def load_model(
    model_path: str = MODEL_PATH,
    map_location: str = "cpu",
) -> Dict[str, object]:
    """读取 JSON checkpoint，并恢复 state_dict。"""
    with open(model_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    state_dict = {}

    for name, metadata in payload["state_dict"].items():
        dtype_name = metadata["dtype"]

        if not hasattr(torch, dtype_name):
            raise ValueError(f"不支持的 dtype：{dtype_name}")

        tensor = torch.tensor(
            metadata["data"],
            dtype=getattr(torch, dtype_name),
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


# =============================================================================
# Rank Loss 与按交易日分组的 batch sampler
# =============================================================================

class DateGroupedBatchSampler(Sampler[List[int]]):
    """保证每个 batch 仅包含同一交易日的股票样本。

    Rank loss 必须在同一横截面内比较股票排序，不能把不同日期混在一起。
    """

    def __init__(self, dates: Sequence, batch_size: int, shuffle: bool = True):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

        date_series = pd.to_datetime(pd.Series(dates)).dt.normalize()
        self.groups = [
            group.index.to_numpy(dtype=np.int64)
            for _, group in date_series.groupby(date_series, sort=True)
        ]

    def __iter__(self):
        batches: List[np.ndarray] = []
        for indices in self.groups:
            indices = indices.copy()
            if self.shuffle:
                np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) >= 2:
                    batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch.tolist()

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.groups
            if len(indices) >= 2
        )



def rank_percentile_by_day(y, dates):
    """按照交易日将未来收益转换为截面 percentile 标签。

    原始 return -> 每日横截面排序 -> percentile rank

    与 Rank IC 评估目标保持一致：
    - 不改变股票排序关系；
    - 保留连续排序信息；
    - 标签范围固定在 (0, 1)。

    percentile 定义:
        最低收益股票 -> 接近 0
        最高收益股票 -> 接近 1
    """
    y = np.asarray(y, dtype=np.float32)
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()

    out = np.zeros_like(y, dtype=np.float32)

    for d in np.unique(dates):
        mask = dates == d
        values = y[mask]

        if len(values) < 2 or not np.isfinite(values).all():
            out[mask] = 0.5
            continue

        # 稳定排序，避免收益相同时随机扰动
        order = np.argsort(values, kind="mergesort")

        ranks = np.empty(len(values), dtype=np.float32)
        ranks[order] = np.arange(len(values), dtype=np.float32)

        # percentile rank in (0,1)
        out[mask] = (ranks + 0.5) / len(values)

    return out.astype(np.float32)




def ic_loss(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """最大化截面 Pearson IC。

    batch 必须来自同一个交易日，因此 prediction 和 target
    代表同一横截面的股票排序。

    loss = -IC
    """
    prediction = prediction - prediction.mean()
    target = target - target.mean()

    numerator = torch.sum(prediction * target)
    denominator = (
        torch.sqrt(torch.sum(prediction ** 2) + eps)
        *
        torch.sqrt(torch.sum(target ** 2) + eps)
    )

    ic = numerator / denominator

    if not torch.isfinite(ic):
        return prediction.sum() * 0.0

    return -ic


def pairwise_rank_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """RankNet 风格的成对排序损失。

    一个 batch 必须来自同一交易日。标签相同的 pair 不参与损失。
    """
    n = prediction.numel()
    if n < 2:
        return prediction.sum() * 0.0

    i, j = torch.triu_indices(n, n, offset=1, device=prediction.device)
    target_diff = target[i] - target[j]
    valid = torch.isfinite(target_diff) & target_diff.ne(0)

    if not valid.any():
        return prediction.sum() * 0.0

    logits = prediction[i[valid]] - prediction[j[valid]]
    pair_target = (target_diff[valid] > 0).to(prediction.dtype)
    return F.binary_cross_entropy_with_logits(logits, pair_target)




def ic_rank_loss(
    prediction: Tensor,
    target: Tensor,
    ic_weight: float = 0.5,
    rank_weight: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """IC + Pairwise RankNet 混合损失。

    loss = ic_weight * (-IC) + rank_weight * RankNet

    - IC: 优化整体横截面相关性
    - RankNet: 强化 top/bottom 股票排序关系

    batch 必须来自同一个交易日。
    """
    ic_loss_value = ic_loss(prediction, target)
    rank_loss_value = pairwise_rank_loss(prediction, target)

    loss = (
        ic_weight * ic_loss_value
        +
        rank_weight * rank_loss_value
    )

    return loss, ic_loss_value, rank_loss_value


def evaluate_mixed_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """计算验证集损失：与 PatchMixer 保持一致，以 Rank IC 对齐目标为主。"""
    model.eval()
    total_loss = 0.0
    total_rank_loss = 0.0
    total_ic_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            prediction = model(X_batch)
            loss, ic_loss_value, rank_loss_value = ic_rank_loss(
                prediction,
                y_batch,
                ic_weight=0.2,
                rank_weight=0.8,
            )

            if torch.isfinite(loss):
                total_loss += float(loss.detach().cpu())
                total_rank_loss += float(rank_loss_value.detach().cpu())
                total_ic_loss += float(ic_loss_value.detach().cpu())
                num_batches += 1

    denominator = max(num_batches, 1)
    return (
        total_loss / denominator,
        total_rank_loss / denominator,
        total_ic_loss / denominator,
    )


# =============================================================================
# 训练
# =============================================================================

def train_and_save(
    datasources: Dict[str, str],
    model_path: str = MODEL_PATH,
) -> str:
    """从零训练 ALSTM，并按验证集混合损失早停、保存最佳权重。"""
    seed_everything(SEED)

    table = datasources["bar1m"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "训练设备",
        device=str(device),
        table=table,
        train_period=f"{TRAIN_START}~{TRAIN_END}",
        valid_period=f"{VALID_START}~{VALID_END}",
        local=has_local_table(table),
        local_root=str(LOCAL_DATA_ROOT),
    )

    # 股票池取训练期与验证期的并集；每个样本仍按自身日期独立构建。
    instruments = pool(table, TRAIN_START, VALID_END)
    if MAX_TRAIN_INSTRUMENTS is not None:
        instruments = instruments[:MAX_TRAIN_INSTRUMENTS]

    X_train, y_train, train_keys, stats = build_dataset(
        table=table,
        sd=TRAIN_START,
        ed=TRAIN_END,
        mode="train",
        instruments=instruments,
        stats=None,
    )

    X_valid, y_valid, valid_keys, _ = build_dataset(
        table=table,
        sd=VALID_START,
        ed=VALID_END,
        mode="train",
        instruments=instruments,
        stats=stats,
    )

    # 仅使用训练集分位点去极值，避免验证信息泄漏。
    lower, upper = np.nanpercentile(y_train, [1, 99])
    y_train = np.clip(y_train, lower, upper).astype(np.float32)
    y_valid = np.clip(y_valid, lower, upper).astype(np.float32)

    # 与 PatchMixer 保持一致：
    # 原始收益标签 -> 每日截面 percentile rank 标签
    y_train = rank_percentile_by_day(
        y_train,
        train_keys["date"],
    )
    y_valid = rank_percentile_by_day(
        y_valid,
        valid_keys["date"],
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(X_valid),
        torch.from_numpy(y_valid),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=DateGroupedBatchSampler(
            train_keys["date"], BATCH, shuffle=True
        ),
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=DateGroupedBatchSampler(
            valid_keys["date"], BATCH, shuffle=False
        ),
        pin_memory=(device.type == "cuda"),
    )

    model = StockALSTM(**MODEL_CFG).to(device)
    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info("可训练参数量", n_params=n_params)

    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(
            f"模型参数量不符合比赛要求：{n_params:,}；"
            "必须位于 [100,000, 100,000,000]。"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_valid_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        epoch_started = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_rank_loss = 0.0
        total_train_ic_loss = 0.0
        num_train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss, ic_loss_value, rank_loss_value = ic_rank_loss(
                prediction,
                y_batch,
                ic_weight=0.2,
                rank_weight=0.8,
            )

            if not torch.isfinite(loss):
                logger.warning("跳过非有限损失", epoch=epoch + 1)
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_train_loss += float(loss.detach().cpu())
            total_train_rank_loss += float(rank_loss_value.detach().cpu())
            total_train_ic_loss += float(ic_loss_value.detach().cpu())
            num_train_batches += 1

        denominator = max(num_train_batches, 1)
        train_loss = total_train_loss / denominator
        train_rank_loss = total_train_rank_loss / denominator
        train_ic_loss = total_train_ic_loss / denominator

        valid_loss, valid_rank_loss, valid_ic_loss = evaluate_mixed_loss(
            model,
            valid_loader,
            device,
        )

        improved = valid_loss < best_valid_loss - 1e-8
        if improved:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "epoch 完成",
            epoch=epoch + 1,
            train_loss=round(train_loss, 8),
            train_rank_loss=round(train_rank_loss, 8),
            train_ic_loss=round(train_ic_loss, 8),
            valid_loss=round(valid_loss, 8),
            valid_rank_loss=round(valid_rank_loss, 8),
            valid_ic_loss=round(valid_ic_loss, 8),
            best_valid_loss=round(best_valid_loss, 8),
            best_epoch=best_epoch,
            patience=f"{epochs_without_improvement}/{EARLY_STOP_PATIENCE}",
            elapsed=round(time.time() - epoch_started, 2),
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            logger.info(
                "触发 Early Stopping",
                epoch=epoch + 1,
                best_epoch=best_epoch,
                best_valid_loss=best_valid_loss,
            )
            break

    model.load_state_dict(best_state_dict)
    mean, std = stats

    save_model(
        {
            "state_dict": model.state_dict(),
            "model_name": "StockALSTM",
            "model_cfg": MODEL_CFG,
            "feature_cols": FEATURE_COLS,
            "price_cols": PRICE_COLS,
            "vol_cols": VOL_COLS,
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "valid_start": VALID_START,
            "valid_end": VALID_END,
            "loss_name": "0.5*ic_loss+0.5*pairwise_ranknet_loss_percentile_target",
            "rank_loss_weight": 0.5,
            "ic_loss_weight": 0.5,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "train_table": table,
            "local_data_root": str(LOCAL_DATA_ROOT),
            "seed": SEED,
            "n_params": n_params,
        },
        model_path=model_path,
    )

    logger.info(
        "ALSTM 最佳模型已保存，请随 Notebook 一并上传",
        path=model_path,
        best_epoch=best_epoch,
        best_valid_loss=best_valid_loss,
    )
    return model_path


if __name__ == "__main__":
    datasources = {
        "bar1m": "bigalpha_2026_stock_bar1m",
    }

    train_and_save(datasources)
