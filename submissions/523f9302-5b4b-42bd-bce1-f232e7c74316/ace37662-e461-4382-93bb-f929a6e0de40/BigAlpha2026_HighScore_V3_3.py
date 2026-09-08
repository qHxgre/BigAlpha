# %% [markdown]
# # BigAlpha 2026 端到端大模型赛道：HighScore V3.3 DirectIC
#
# 这是一份自包含的 BigQuant Notebook 源文件（通过 `scripts/build_notebook.py` 生成 `.ipynb`）。
#
# 设计原则：
#
# - 仅使用比赛提供的原始 5 分钟 K 线与盘口字段；不构造技术因子、滚动统计或跨字段衍生特征。
# - 以 5 分钟原始量价与线上可用的最深十档盘口为输入，模型内部学习 20 分钟 patch、日内结构和 80 日跨周期结构。
# - 按官方模板对齐：日期 `t` 的分数使用截至 `t` 收盘的原始序列，监督目标为 `t+1` 残差收益。
# - 仅做规则允许的缺失值填充、逐字段 signed-log 和逐字段 StandardScaler。
# - 个股时序编码后，在完整单日中证1000截面上用置换等变的潜变量注意力学习相对强弱与市场状态。
# - 只优化实际提交的 `t+1` 单一输出，移除 V3 中可能稀释短期方向的 5 日辅助 head。
# - 截面标准化预测后计算 FP32 IC/排序/软多空目标，并显式约束输出分散度，防止零预测局部最优。
# - 将一次性研发验证 `develop_model` 与正式全量训练 `train_and_save` 分离；私榜不会现场选模。
# - 同一训练轨迹上对第二/三轮 EMA 检查点做等权平均，用单份稳定权重降低 regime 偶然性。
# - 验证阶段复现平台 1%/99% 去极值、z-score 与十个 BARRA 风格剔除。
# - 正式训练固定从零使用 2019–2024 全量样本，并用全量训练数据重新拟合 scaler。
# - 模型架构、训练数据、损失函数、训练轮数和权重数值与 V3.2 完全一致。
# - 仅改变推理内存调度：按交易日分块取数，每块保留完整 80 日历史，推理后立即释放。
# - 可将已训练完成的 V3.2 float16 JSON 无重训升级为 V3.3，始终保留原 V3.2 文件。
# - 归一化统计量只用训练集拟合。
# - Transformer 权重从随机初始化开始训练，不加载任何外部预训练权重。
# - 模型参数量自动校验在 100,000 到 100,000,000 之间。
# - `main(datasources, start_date, end_date)` 严格返回 `date/instrument/score` 三列。
#
# 当前工作口径：训练数据可选 `2019–2024` 区间；GPU Notebook 训练预算不超过 3 小时，公榜推理不超过 12 小时。

# %%
from __future__ import annotations

import copy
import base64
import gc
import json
import math
import os
import random
import re
import time
import warnings
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import dai
except ImportError:
    dai = None

warnings.filterwarnings("ignore", category=RuntimeWarning)


# %% [markdown]
# ## 1. 集中配置
#
# 首次在 BigQuant 运行时，只需要先核对以下配置。HighScore 版本针对 A100 80G ×1 优化。

# %%
@dataclass
class Config:
    seed: int = 20260715
    data_source: str = "bigalpha_2026_stock_bar5m"
    datasource_key: str = "bar5m"
    train_start: str = "2019-01-01"
    train_end: str = "2024-12-31"
    artifact_path: str = "bigalpha_e2e_highscore_v3_3.json"
    artifact_storage_dtype: str = "float16"
    # 只影响推理内存调度，不影响模型输入、参数或分数。
    inference_chunk_trading_days: int = 40
    inference_history_calendar_days: int = 190
    exposure_source: str = "bigalpha_2026_exposure"
    use_exposure_residualization: bool = True

    # 5m: 每日 48 根 bar；4 根组成一个可学习 20m patch；回看 80 个交易日。
    bars_per_day: int = 48
    patch_size: int = 4
    lookback_days: int = 80
    min_history_fraction: float = 0.75
    validation_days: int = 240
    target_horizons: Tuple[int, ...] = (1,)
    horizon_weights: Tuple[float, ...] = (1.0,)
    training_label_trim_fraction: float = 0.02

    # 训练预算。2.65 小时主动停止，为 3 小时 Notebook 上限预留保存与校验时间。
    max_train_seconds: int = 2 * 3600 + 39 * 60
    development_epochs: int = 3
    formal_epochs: int = 3
    # 模型包含截面注意力，训练、验证和推理必须都以完整单日中证1000
    # 为一批，避免 V2 中“训练子截面/推理另一截面”造成的结构漂移。
    max_stocks_per_day: int = 1000
    train_stocks_per_batch: int = 1000
    eval_stocks_per_batch: int = 1000
    learning_rate: float = 1.0e-4
    weight_decay: float = 1e-4
    regression_loss_weight: float = 0.22
    ic_loss_weight: float = 0.56
    rank_loss_weight: float = 0.10
    tail_loss_weight: float = 0.04
    dispersion_loss_weight: float = 0.08
    prediction_std_target: float = 1.0
    rank_pairings: int = 3
    rank_temperature: float = 0.75
    tail_temperature: float = 0.80
    warmup_ratio: float = 0.08
    grad_clip: float = 1.0
    num_workers: int = 0
    max_validation_samples: int = 0
    ema_decay: float = 0.990
    checkpoint_average_start_epoch: int = 2
    minimum_development_ic: float = 0.0
    minimum_development_rank_ic: float = 0.0
    minimum_development_ls_sharpe: float = 0.0
    minimum_development_stress_ic: float = -0.02
    minimum_development_ic_t_stat: float = 1.0
    minimum_development_positive_ic_ratio: float = 0.50
    minimum_training_batch_ic: float = 0.0
    minimum_training_pred_std: float = 0.05
    maximum_training_pred_std: float = 3.0
    minimum_full_train_memory_gib: float = 128.0
    minimum_full_gpu_memory_gib: float = 70.0

    # 按 38 个实际字段计算约 1705 万参数；十档字段全部可用时约 1709 万。
    d_model: int = 320
    nhead: int = 8
    intraday_layers: int = 2
    num_layers: int = 6
    dim_feedforward: int = 1280
    stock_mlp_layers: int = 2
    cross_sectional_layers: int = 2
    cross_sectional_latents: int = 16
    # PyTorch SDPA 无法一次启动 1000股×80日×8头的日内 batch 网格。
    # 只分块计算相互独立的“股票-交易日”日内编码，再无损拼回。
    intraday_chunk_size: int = 4096
    dropout: float = 0.08
    target_parameter_min: int = 15_000_000
    target_parameter_max: int = 20_000_000

    # 仅列出原始字段。实际使用列会与线上表结构取交集并写入模型产物。
    feature_candidates: Tuple[str, ...] = (
        "pre_close", "open", "high", "low", "close", "price",
        "volume", "amount", "num_trades", "deal_number",
        "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",
        "bid_price6", "bid_price7", "bid_price8", "bid_price9", "bid_price10",
        "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5",
        "ask_price6", "ask_price7", "ask_price8", "ask_price9", "ask_price10",
        "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",
        "bid_volume6", "bid_volume7", "bid_volume8", "bid_volume9", "bid_volume10",
        "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5",
        "ask_volume6", "ask_volume7", "ask_volume8", "ask_volume9", "ask_volume10",
        "bid_num_orders1", "bid_num_orders2", "bid_num_orders3", "bid_num_orders4", "bid_num_orders5",
        "bid_num_orders6", "bid_num_orders7", "bid_num_orders8", "bid_num_orders9", "bid_num_orders10",
        "ask_num_orders1", "ask_num_orders2", "ask_num_orders3", "ask_num_orders4", "ask_num_orders5",
        "ask_num_orders6", "ask_num_orders7", "ask_num_orders8", "ask_num_orders9", "ask_num_orders10",
        "total_bid_volume", "total_ask_volume", "bid_avg_price", "ask_avg_price",
    )
    log_field_patterns: Tuple[str, ...] = (
        "price", "open", "high", "low", "close",
        "volume", "amount", "num_trades", "deal_number", "num_orders",
    )


CFG = Config()
print(json.dumps(asdict(CFG), ensure_ascii=False, indent=2))


# %% [markdown]
# ## 2. 可复现性与合规检查

# %%
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(name)):
        raise ValueError(f"非法数据源或字段名: {name!r}")
    return str(name)


def assert_config_compliance(cfg: Config) -> None:
    if not 1 <= cfg.lookback_days <= 240:
        raise ValueError("lookback_days 必须在 1 到 240 个交易日之间")
    if len(cfg.feature_candidates) > 100:
        raise ValueError("候选输入字段超过 100 个")
    if not 0.0 < cfg.min_history_fraction <= 1.0:
        raise ValueError("min_history_fraction 必须在 (0, 1] 区间")
    if cfg.d_model % cfg.nhead != 0:
        raise ValueError("d_model 必须能被 nhead 整除")
    if cfg.bars_per_day % cfg.patch_size != 0:
        raise ValueError("bars_per_day 必须能被 patch_size 整除")
    if (
        cfg.intraday_layers < 1 or cfg.num_layers < 1
        or cfg.stock_mlp_layers < 1 or cfg.cross_sectional_layers < 1
    ):
        raise ValueError("日内、跨日、个股与截面层数必须至少为 1")
    if cfg.cross_sectional_latents < 4:
        raise ValueError("截面潜变量数至少为 4")
    if cfg.development_epochs < 1 or cfg.formal_epochs < 1:
        raise ValueError("研发验证与正式训练至少需要 1 个 epoch")
    if cfg.development_epochs != cfg.formal_epochs:
        raise ValueError("研发验证必须完整复现正式训练轮数")
    loss_weight_sum = (
        cfg.regression_loss_weight + cfg.ic_loss_weight
        + cfg.rank_loss_weight + cfg.tail_loss_weight + cfg.dispersion_loss_weight
    )
    if not np.isclose(loss_weight_sum, 1.0):
        raise ValueError(f"五项损失权重之和必须为 1，当前为 {loss_weight_sum}")
    if not cfg.target_horizons or len(cfg.target_horizons) != len(cfg.horizon_weights):
        raise ValueError("target_horizons 与 horizon_weights 必须非空且等长")
    if cfg.target_horizons[0] != 1 or tuple(sorted(set(cfg.target_horizons))) != cfg.target_horizons:
        raise ValueError("target_horizons 必须是以 1 开头的严格递增正整数")
    if not np.isclose(sum(cfg.horizon_weights), 1.0) or min(cfg.horizon_weights) <= 0:
        raise ValueError("horizon_weights 必须全为正数且之和为 1")
    if not 50 <= cfg.train_stocks_per_batch <= cfg.max_stocks_per_day:
        raise ValueError("train_stocks_per_batch 必须位于 [50, max_stocks_per_day]")
    if not 50 <= cfg.eval_stocks_per_batch <= cfg.max_stocks_per_day:
        raise ValueError("eval_stocks_per_batch 必须位于 [50, max_stocks_per_day]")
    if (
        cfg.train_stocks_per_batch != cfg.max_stocks_per_day
        or cfg.eval_stocks_per_batch != cfg.max_stocks_per_day
    ):
        raise ValueError("截面模块要求训练、验证与推理都使用完整单日股票池")
    if cfg.rank_pairings < 1 or cfg.tail_temperature <= 0:
        raise ValueError("排序配对数和软尾部温度必须大于 0")
    if not 0.0 < cfg.ema_decay < 1.0:
        raise ValueError("ema_decay 必须位于 (0, 1)")
    if not 1 <= cfg.checkpoint_average_start_epoch <= cfg.development_epochs:
        raise ValueError("checkpoint_average_start_epoch 必须位于训练轮数内")
    if cfg.minimum_development_ic_t_stat <= 0:
        raise ValueError("研发 IC t-stat 门槛必须为正")
    if not 0.0 <= cfg.minimum_development_positive_ic_ratio <= 1.0:
        raise ValueError("正 IC 日比例门槛必须位于 [0, 1]")
    if not 0.0 <= cfg.training_label_trim_fraction < 0.10:
        raise ValueError("training_label_trim_fraction 必须位于 [0, 0.10)")
    if cfg.prediction_std_target <= 0:
        raise ValueError("prediction_std_target 必须大于 0")
    if cfg.rank_temperature <= 0:
        raise ValueError("rank_temperature 必须大于 0")
    if not 0 < cfg.minimum_training_pred_std < cfg.maximum_training_pred_std:
        raise ValueError("训练输出标准差上下界配置非法")
    if not 256 <= cfg.intraday_chunk_size <= 8191:
        raise ValueError("intraday_chunk_size 必须位于 [256, 8191]，避免 SDPA CUDA 网格越界")
    if cfg.max_stocks_per_day < 50:
        raise ValueError("正式配置每日至少需要 50 只股票来计算截面目标")
    if not 100_000 <= cfg.target_parameter_min <= cfg.target_parameter_max <= 100_000_000:
        raise ValueError("策略目标参数范围必须位于比赛允许的 10万到1亿之间")
    if cfg.artifact_storage_dtype not in {"float16", "float32"}:
        raise ValueError("artifact_storage_dtype 只能是 float16 或 float32")
    if not 1 <= cfg.inference_chunk_trading_days <= 120:
        raise ValueError("inference_chunk_trading_days 必须位于 [1, 120]")
    if cfg.inference_history_calendar_days < cfg.lookback_days * 2:
        raise ValueError("推理日历缓冲不足以覆盖回看交易日")
    validate_identifier(cfg.data_source)


def system_memory_gib() -> float:
    """不依赖 psutil 获取当前 Notebook 的主机总内存。"""
    try:
        import psutil
        return float(psutil.virtual_memory().total / 1024**3)
    except Exception:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size / 1024**3)
    return float("nan")


def assert_training_resources(cfg: Config, allow_cpu: bool = False) -> None:
    memory_gib = system_memory_gib()
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
    print(
        f"runtime resources: system_memory={memory_gib:.1f} GiB, "
        f"cuda={cuda_available}, gpu={gpu_name}"
    )
    if not allow_cpu and not cuda_available:
        raise RuntimeError(
            "HighScore 完整训练要求 GPU；当前是 CPU 环境。低配资源只适合字段检查，"
            "请切换到 A100 80G ×1 后重启内核。"
        )
    if cuda_available:
        gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"gpu memory: {gpu_memory_gib:.1f} GiB")
        if not allow_cpu and gpu_memory_gib < cfg.minimum_full_gpu_memory_gib:
            raise RuntimeError(
                f"V3.3 完整分钟序列训练至少需要 "
                f"{cfg.minimum_full_gpu_memory_gib:.0f} GiB GPU 显存；"
                f"当前只有 {gpu_memory_gib:.1f} GiB。请使用 A100 80G ×1。"
            )
    if not allow_cpu and np.isfinite(memory_gib) and memory_gib < cfg.minimum_full_train_memory_gib:
        raise RuntimeError(
            f"完整训练至少需要约 {cfg.minimum_full_train_memory_gib:.0f} GiB 主机内存；"
            f"当前只有 {memory_gib:.1f} GiB。六年分钟数据在转为 DataFrame/张量时会导致 Kernel OOM。"
        )


def make_smoke_config(base: Optional[Config] = None) -> Config:
    """低成本验证 DAI→张量→模型→JSON 全链路，不代表正式训练。"""
    cfg = copy.deepcopy(base if base is not None else Config())
    cfg.train_start = "2024-01-01"
    cfg.train_end = "2024-03-31"
    cfg.lookback_days = 5
    cfg.validation_days = 5
    cfg.target_horizons = (1,)
    cfg.horizon_weights = (1.0,)
    cfg.development_epochs = 1
    cfg.formal_epochs = 1
    cfg.train_stocks_per_batch = 128
    cfg.eval_stocks_per_batch = 128
    cfg.max_stocks_per_day = 128
    cfg.max_validation_samples = 500
    cfg.max_train_seconds = 10 * 60
    cfg.d_model = 128
    cfg.nhead = 4
    cfg.intraday_layers = 1
    cfg.num_layers = 2
    cfg.dim_feedforward = 384
    cfg.stock_mlp_layers = 1
    cfg.cross_sectional_layers = 1
    cfg.cross_sectional_latents = 8
    cfg.checkpoint_average_start_epoch = 1
    # smoke 只验证功能链路，不用几十次更新的方向决定是否允许落盘。
    cfg.minimum_training_batch_ic = -1.0
    cfg.target_parameter_min = 500_000
    cfg.target_parameter_max = 3_000_000
    cfg.artifact_path = "bigalpha_e2e_highscore_v3_3_smoke.json"
    return cfg


def make_capacity_smoke_config(base: Optional[Config] = None) -> Config:
    """用正式 80 日窗口和完整中证1000日截面验证显存与吞吐。"""
    cfg = copy.deepcopy(base if base is not None else Config())
    cfg.train_start = "2023-01-01"
    cfg.train_end = "2024-02-29"
    cfg.lookback_days = 80
    cfg.validation_days = 30
    cfg.development_epochs = 1
    cfg.formal_epochs = 1
    cfg.train_stocks_per_batch = 1000
    cfg.eval_stocks_per_batch = 1000
    cfg.max_stocks_per_day = 1000
    cfg.max_validation_samples = 0
    cfg.checkpoint_average_start_epoch = 1
    # capacity smoke 的目标是显存/吞吐/保存/推理，方向由完整 development 门判定。
    cfg.minimum_training_batch_ic = -1.0
    cfg.max_train_seconds = 30 * 60
    cfg.artifact_path = "bigalpha_e2e_highscore_v3_3_capacity_smoke.json"
    return cfg


set_global_seed(CFG.seed)
assert_config_compliance(CFG)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)


# %% [markdown]
# ## 3. DAI 数据读取
#
# 先以极小查询探测表结构，再只读取实际存在的白名单原始字段。`main` 推理时使用训练产物里冻结的字段列表，避免训练和推理列错位。

# %%
def require_dai() -> None:
    if dai is None:
        raise RuntimeError("当前环境没有 dai；请在 BigQuant AIStudio Notebook 中运行数据读取与训练。")


def query_filters(start_date: str, end_date: str) -> Dict[str, List[str]]:
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d 00:00:00")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d 23:59:59")
    return {"date": [start, end]}


def probe_columns(data_source: str, date_hint: str) -> List[str]:
    require_dai()
    table = validate_identifier(data_source)
    hint = pd.Timestamp(date_hint)
    probe_start = (hint - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    probe_end = (hint + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    sample = dai.query(
        f"SELECT * FROM {table} LIMIT 1",
        filters=query_filters(probe_start, probe_end),
        compression=True,
    ).df()
    if sample.empty:
        raise RuntimeError(f"无法在 {probe_start} 至 {probe_end} 探测 {table} 的字段，请换一个交易日重试。")
    return list(sample.columns)


def resolve_feature_columns(
    data_source: str,
    date_hint: str,
    candidates: Sequence[str],
    minimum: int = 8,
) -> Tuple[List[str], str]:
    available = set(probe_columns(data_source, date_hint))
    features = [c for c in candidates if c in available]
    if len(features) > 100:
        raise ValueError(f"实际输入字段数 {len(features)} 超过 100")
    if len(features) < minimum:
        raise RuntimeError(
            f"仅找到 {len(features)} 个候选原始字段：{features}。"
            f"线上字段为：{sorted(available)}。请先更新 CFG.feature_candidates。"
        )
    close_column = "close" if "close" in available else "price" if "price" in available else ""
    if not close_column:
        raise RuntimeError("数据表既没有 close 也没有 price，无法构造训练标签。")
    if close_column not in features:
        features.append(close_column)
    return features, close_column


def assert_full_order_book_depth(feature_columns: Sequence[str], depth: int = 5) -> None:
    """至少要求五档；若线上表提供六至十档，则自动一并使用。"""
    required = {
        f"{side}_{kind}{level}"
        for side in ("bid", "ask")
        for kind in ("price", "volume", "num_orders")
        for level in range(1, depth + 1)
    }
    missing = sorted(required.difference(feature_columns))
    if missing:
        raise RuntimeError(
            f"{depth} 档盘口字段不完整，缺少 {missing}。"
            "V3.3 不允许静默退化为前三档输入，请核对 bar5m 数据权限与字段。"
        )
    complete_depth = 0
    available = set(feature_columns)
    for candidate_depth in range(1, 11):
        level_fields = {
            f"{side}_{kind}{candidate_depth}"
            for side in ("bid", "ask")
            for kind in ("price", "volume", "num_orders")
        }
        if level_fields.issubset(available):
            complete_depth = candidate_depth
        else:
            break
    print(
        f"raw feature coverage: {len(feature_columns)} fields; "
        f"full {complete_depth}-level order book"
    )


def load_market_frame(
    data_source: str,
    start_date: str,
    end_date: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    require_dai()
    table = validate_identifier(data_source)
    columns = ["date", "instrument"] + [validate_identifier(c) for c in feature_columns]
    columns = list(dict.fromkeys(columns))
    sql = f"SELECT {', '.join(columns)} FROM {table} ORDER BY date, instrument"
    df = dai.query(sql, filters=query_filters(start_date, end_date), compression=True).df()
    if df.empty:
        raise RuntimeError(f"{table} 在 {start_date} 至 {end_date} 没有返回数据")
    required = {"date", "instrument", *feature_columns}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"DAI 查询结果缺少字段: {sorted(missing)}")
    # DAI 已只返回上述列；浅拷贝元数据即可，禁止整表深拷贝。
    df = df.loc[:, columns].copy(deep=False)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    # compression=True 通常已经是 category；统一保持紧凑编码。
    if not isinstance(df["instrument"].dtype, pd.CategoricalDtype):
        df["instrument"] = df["instrument"].astype("category")
    for col in feature_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.dtype != np.dtype("float32"):
            df[col] = values.astype("float32", copy=False)
    return df


# %% [markdown]
# ## 4. 原始字段张量化
#
# 每个样本形状为 `[lookback_days × bars_per_day, raw_fields]`。这里只做排序、缺失位置保留和定长装箱；没有构造任何技术指标或跨字段算子。

# %%
@dataclass
class MarketStore:
    features: np.ndarray          # [instrument, trading_day, bar, field], float32
    token_present: np.ndarray     # [instrument, trading_day, bar], bool
    day_present: np.ndarray       # [instrument, trading_day], bool
    daily_close: np.ndarray       # [instrument, trading_day], float32
    instruments: np.ndarray       # [instrument]
    days: np.ndarray              # datetime64[D]
    feature_columns: List[str]
    close_column: str


def build_market_store(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    close_column: str,
    bars_per_day: int,
) -> MarketStore:
    if bars_per_day <= 0:
        raise ValueError("bars_per_day 必须大于 0")
    # frame 在调用后立即丢弃，直接接管，不再复制 38 列完整行情。
    df = frame
    timestamp_values = df["date"].to_numpy(dtype="datetime64[ns]", copy=False)
    unique_timestamps = np.unique(timestamp_values)
    timestamp_days = unique_timestamps.astype("datetime64[D]")
    days, first_positions, slot_counts = np.unique(
        timestamp_days, return_index=True, return_counts=True
    )
    if len(slot_counts) and int(slot_counts.max()) > bars_per_day:
        raise RuntimeError(
            f"行情表单日最多出现 {int(slot_counts.max())} 个时点，"
            f"超过 bars_per_day={bars_per_day}；拒绝静默截断。"
        )

    # 时间位置仍由全市场当日时钟决定，但用紧凑整数映射取代 DataFrame merge。
    timestamp_ids = np.searchsorted(unique_timestamps, timestamp_values).astype(
        np.int32, copy=False
    )
    timestamp_day_ids = np.repeat(
        np.arange(len(days), dtype=np.int32), slot_counts
    )
    timestamp_slots = (
        np.arange(len(unique_timestamps), dtype=np.int32)
        - np.repeat(first_positions.astype(np.int32), slot_counts)
    ).astype(np.int16, copy=False)
    d_idx = timestamp_day_ids[timestamp_ids]
    s_idx = timestamp_slots[timestamp_ids]
    del timestamp_ids, timestamp_day_ids, timestamp_slots, timestamp_values, timestamp_days

    instrument_values = df["instrument"].astype("category")
    raw_codes = instrument_values.cat.codes.to_numpy(np.int32, copy=False)
    if (raw_codes < 0).any():
        raise RuntimeError("instrument 包含缺失值")
    raw_categories = instrument_values.cat.categories.astype(str).to_numpy()
    order = np.argsort(raw_categories)
    remap = np.empty(len(order), dtype=np.int32)
    remap[order] = np.arange(len(order), dtype=np.int32)
    i_idx = remap[raw_codes]
    instruments = raw_categories[order]
    del instrument_values, raw_codes, raw_categories, order, remap

    shape = (len(instruments), len(days), bars_per_day, len(feature_columns))
    tensor = np.full(shape, np.nan, dtype=np.float32)
    token_present = np.zeros(shape[:-1], dtype=bool)
    positions = (i_idx, d_idx, s_idx)
    # 逐字段填充，避免 pandas 将 38 列合并为额外的连续 5.38 GiB 数组。
    for feature_idx, column in enumerate(feature_columns):
        values = df[column].to_numpy(dtype=np.float32, copy=False)
        tensor[i_idx, d_idx, s_idx, feature_idx] = values
        observed = np.isfinite(values)
        token_present[positions] = token_present[positions] | observed
    day_present = token_present.any(axis=-1)

    close_pos = list(feature_columns).index(close_column)
    daily_close = np.full((len(instruments), len(days)), np.nan, dtype=np.float32)
    close_values = tensor[..., close_pos]
    # 每日最后一个有效原始成交价用于构造 t->t+1 标签；t+1 数据绝不进入 t 日输入。
    for slot in range(bars_per_day):
        valid = np.isfinite(close_values[:, :, slot])
        daily_close[valid] = close_values[:, :, slot][valid]

    del df, values, observed, positions, i_idx, d_idx, s_idx, close_values
    gc.collect()
    return MarketStore(
        features=tensor,
        token_present=token_present,
        day_present=day_present,
        daily_close=daily_close,
        instruments=instruments,
        days=days,
        feature_columns=list(feature_columns),
        close_column=close_column,
    )


def load_official_universe(start_date: str, end_date: str) -> pd.DataFrame:
    """读取历史时点中证1000成分，训练和推理都不得把池外股票送入模型。"""
    require_dai()
    universe = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters=query_filters(start_date, end_date),
        compression=True,
    ).df()
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["instrument"] = universe["instrument"].astype(str)
    return universe.drop_duplicates(["date", "instrument"])


def build_universe_mask(store: MarketStore, universe: pd.DataFrame) -> np.ndarray:
    mask = np.zeros((len(store.instruments), len(store.days)), dtype=bool)
    instrument_to_id = {name: idx for idx, name in enumerate(store.instruments.astype(str))}
    day_to_id = {pd.Timestamp(day): idx for idx, day in enumerate(store.days)}
    instrument_ids = universe["instrument"].map(instrument_to_id)
    day_ids = universe["date"].map(day_to_id)
    valid = instrument_ids.notna() & day_ids.notna()
    mask[
        instrument_ids[valid].astype(np.int32).to_numpy(),
        day_ids[valid].astype(np.int32).to_numpy(),
    ] = True
    return mask


def restrict_store_to_universe(
    store: MarketStore,
    universe_mask: np.ndarray,
    earliest_allowed_day: Optional[str] = None,
) -> None:
    """屏蔽历史时点池外数据；训练时同时屏蔽训练区间之前的数据。"""
    if universe_mask.shape != store.day_present.shape:
        raise ValueError("universe_mask 形状与市场张量不一致")
    allowed = universe_mask.copy()
    if earliest_allowed_day is not None:
        allowed[:, store.days < np.datetime64(pd.Timestamp(earliest_allowed_day).date())] = False
    invalid = ~allowed
    store.features[invalid] = np.nan
    store.token_present[invalid] = False
    store.day_present[invalid] = False
    store.daily_close[invalid] = np.nan
    counts = allowed.sum(axis=0)
    populated = counts[counts > 0]
    if len(populated) == 0:
        raise RuntimeError("官方中证1000股票池与行情张量没有任何可对齐日期")
    print(
        f"official universe applied: days={len(populated):,}; "
        f"min={int(populated.min())}; median={int(np.median(populated))}; "
        f"max={int(populated.max())}"
    )


def build_forward_return_labels(
    store: MarketStore,
    horizons: Sequence[int],
) -> np.ndarray:
    """构造 t 收盘到 t+h 收盘的多周期监督目标，不向模型输入任何未来数据。"""
    close = store.daily_close.astype(np.float64, copy=False)
    labels = np.full((*close.shape, len(horizons)), np.nan, dtype=np.float32)
    for horizon_idx, horizon in enumerate(horizons):
        if horizon <= 0 or horizon >= close.shape[1]:
            raise ValueError(f"非法监督周期: {horizon}")
        before, after = close[:, :-horizon], close[:, horizon:]
        valid = np.isfinite(after) & np.isfinite(before) & (before > 0)
        raw = np.full_like(before, np.nan, dtype=np.float64)
        raw[valid] = after[valid] / before[valid] - 1.0
        raw[(raw < -0.50) | (raw > 0.50)] = np.nan
        labels[:, :-horizon, horizon_idx] = raw.astype(np.float32)
        for day_idx in range(labels.shape[1] - horizon):
            column = labels[:, day_idx, horizon_idx]
            if np.isfinite(column).sum() >= 10:
                column -= np.nanmedian(column)
    return labels


OFFICIAL_STYLE_EXPOSURES: Tuple[str, ...] = (
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


def resolve_exposure_columns(table: str, date_hint: str) -> List[str]:
    """严格复现官方评分阶段剔除的十个 BARRA 风格暴露。"""
    require_dai()
    table = validate_identifier(table)
    hint = pd.Timestamp(date_hint)
    sample = dai.query(
        f"SELECT * FROM {table} LIMIT 50",
        filters=query_filters(
            (hint - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            (hint + pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
        ),
        compression=True,
    ).df()
    available = {str(column).upper(): str(column) for column in sample.columns}
    missing = [name for name in OFFICIAL_STYLE_EXPOSURES if name not in available]
    if missing:
        raise RuntimeError(
            f"{table} 缺少官方评分所需的 BARRA 风格暴露 {missing}；"
            f"实际字段为 {list(sample.columns)}"
        )
    columns = [available[name] for name in OFFICIAL_STYLE_EXPOSURES]
    print("label residualization exposures:", columns)
    return columns


def load_exposure_frame(
    table: str,
    start_date: str,
    end_date: str,
    exposure_columns: Sequence[str],
) -> pd.DataFrame:
    table = validate_identifier(table)
    columns = ["date", "instrument", *[validate_identifier(c) for c in exposure_columns]]
    frame = dai.query(
        f"SELECT {', '.join(columns)} FROM {table} ORDER BY date, instrument",
        filters=query_filters(start_date, end_date),
        compression=True,
    ).df()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    for column in exposure_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates(["date", "instrument"], keep="last")


def residualize_labels_with_exposure(
    labels: np.ndarray,
    store: MarketStore,
    exposure_frame: pd.DataFrame,
    exposure_columns: Sequence[str],
    ridge: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """在日期 t 用当日 BARRA 暴露回归 t→t+1 收益，只改变监督标签。"""
    result = labels.copy()
    instrument_to_id = {name: idx for idx, name in enumerate(store.instruments.astype(str))}
    day_to_id = {pd.Timestamp(day): idx for idx, day in enumerate(store.days)}
    if labels.ndim != 3:
        raise ValueError("多周期标签张量必须为 [instrument, day, horizon]")
    processed = 0
    failed = 0
    for day, sub in exposure_frame.groupby("date", sort=False, observed=True):
        day_idx = day_to_id.get(pd.Timestamp(day))
        if day_idx is None:
            continue
        instrument_ids = sub["instrument"].map(instrument_to_id)
        keep = instrument_ids.notna().to_numpy()
        if keep.sum() < 50:
            failed += 1
            continue
        ids = instrument_ids[keep].astype(np.int32).to_numpy()
        x = sub.loc[keep, exposure_columns].to_numpy(np.float64, copy=True)
        for horizon_idx in range(labels.shape[-1]):
            y = result[ids, day_idx, horizon_idx].astype(np.float64, copy=True)
            valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
            if valid.sum() < max(50, len(exposure_columns) * 5):
                failed += 1
                continue
            valid_ids = ids[valid]
            valid_x = x[valid]
            y = y[valid]
            mean = valid_x.mean(axis=0)
            scale = valid_x.std(axis=0)
            useful = np.isfinite(scale) & (scale > 1e-8)
            valid_x = (valid_x[:, useful] - mean[useful]) / scale[useful]
            design = np.column_stack([np.ones(len(valid_x)), valid_x])
            penalty = ridge * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
            residual = y - design @ beta
            residual -= np.median(residual)
            result[valid_ids, day_idx, horizon_idx] = residual.astype(np.float32)
            processed += 1
    if processed == 0:
        raise RuntimeError("没有任何交易日成功完成风险暴露残差化")
    stats = {"processed_days": float(processed), "failed_days": float(failed)}
    print("label residualization:", stats)
    return result, stats


def normalize_labels_by_day(labels: np.ndarray) -> np.ndarray:
    """标签侧每日 1%/99% 去极值与 z-score，使训练尺度和官方评估一致。"""
    result = labels.copy()
    if result.ndim == 2:
        result = result[..., None]
    for horizon_idx in range(result.shape[-1]):
        for day_idx in range(result.shape[1]):
            values = result[:, day_idx, horizon_idx]
            finite = np.isfinite(values)
            if finite.sum() < 50:
                continue
            lower, upper = np.quantile(values[finite], [0.01, 0.99])
            clipped = np.clip(values[finite], lower, upper)
            scale = clipped.std()
            if np.isfinite(scale) and scale > 1e-8:
                result[finite, day_idx, horizon_idx] = (
                    (clipped - clipped.mean()) / scale
                ).astype(np.float32)
    return result


def trim_training_label_extremes(
    labels: np.ndarray,
    target_day_mask: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """仅在训练日删除双侧极端标签；验证标签保持完整。

    MASTER 的公开实现采用同类的 DropExtremeLabel + CSZScore 训练口径。
    这里使用更保守的双侧 2%，降低涨跌停、复牌等尾部噪声对梯度的支配，
    不改变模型输入，也不使用任何验证或未来信息。
    """
    if not 0.0 <= fraction < 0.10:
        raise ValueError("fraction 必须位于 [0, 0.10)")
    result = labels.copy()
    if fraction == 0:
        return result
    for day_idx in np.flatnonzero(target_day_mask):
        for horizon_idx in range(result.shape[-1]):
            values = result[:, day_idx, horizon_idx]
            finite = np.isfinite(values)
            if finite.sum() < 100:
                continue
            lower, upper = np.quantile(values[finite], [fraction, 1.0 - fraction])
            extreme = finite & ((values < lower) | (values > upper))
            result[extreme, day_idx, horizon_idx] = np.nan
    return result


def build_exposure_tensor(
    store: MarketStore,
    exposure_frame: pd.DataFrame,
    exposure_columns: Sequence[str],
) -> np.ndarray:
    """将官方十个风格暴露对齐到 instrument × date，供验证阶段模拟平台剔除。"""
    tensor = np.full(
        (len(store.instruments), len(store.days), len(exposure_columns)),
        np.nan,
        dtype=np.float32,
    )
    instrument_to_id = {name: idx for idx, name in enumerate(store.instruments.astype(str))}
    day_to_id = {pd.Timestamp(day): idx for idx, day in enumerate(store.days)}
    instrument_ids = exposure_frame["instrument"].map(instrument_to_id)
    day_ids = exposure_frame["date"].map(day_to_id)
    valid = instrument_ids.notna() & day_ids.notna()
    if valid.any():
        tensor[
            instrument_ids[valid].astype(np.int32).to_numpy(),
            day_ids[valid].astype(np.int32).to_numpy(),
        ] = exposure_frame.loc[valid, exposure_columns].to_numpy(np.float32, copy=True)
    return tensor


# %% [markdown]
# ## 5. 规则允许的逐字段预处理

# %%
@dataclass
class FieldPreprocessor:
    feature_columns: List[str]
    log_mask: np.ndarray
    fill_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @staticmethod
    def _signed_log(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * np.log1p(np.abs(values))

    @classmethod
    def fit(
        cls,
        tensor: np.ndarray,
        feature_columns: Sequence[str],
        log_patterns: Sequence[str],
    ) -> "FieldPreprocessor":
        n_features = tensor.shape[-1]
        log_mask = np.array(
            [any(pattern in col for pattern in log_patterns) for col in feature_columns],
            dtype=bool,
        )
        fills = np.zeros(n_features, dtype=np.float32)
        means = np.zeros(n_features, dtype=np.float32)
        scales = np.ones(n_features, dtype=np.float32)
        for j in range(n_features):
            values = tensor[..., j].astype(np.float64, copy=True)
            if log_mask[j]:
                values = cls._signed_log(values)
            finite = np.isfinite(values)
            if not finite.any():
                raise RuntimeError(f"训练集中字段 {feature_columns[j]} 全部缺失")
            fills[j] = np.nanmedian(values)
            # 统计量只由真实观测值决定，不能让池外/停牌 padding 的数量改变字段尺度。
            observed = values[finite]
            means[j] = observed.mean()
            scale = observed.std()
            scales[j] = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
        return cls(list(feature_columns), log_mask, fills, means, scales)

    def transform_inplace(self, tensor: np.ndarray) -> None:
        if tensor.shape[-1] != len(self.feature_columns):
            raise ValueError("推理张量字段数与训练产物不一致")
        for j in range(tensor.shape[-1]):
            values = tensor[..., j]
            if self.log_mask[j]:
                np.copyto(values, self._signed_log(values), casting="unsafe")
            missing = ~np.isfinite(values)
            values[missing] = self.fill_values[j]
            values -= self.means[j]
            values /= self.scales[j]
            np.clip(values, -12.0, 12.0, out=values)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_columns": self.feature_columns,
            "log_mask": self.log_mask.tolist(),
            "fill_values": self.fill_values.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FieldPreprocessor":
        return cls(
            feature_columns=list(payload["feature_columns"]),
            log_mask=np.asarray(payload["log_mask"], dtype=bool),
            fill_values=np.asarray(payload["fill_values"], dtype=np.float32),
            means=np.asarray(payload["means"], dtype=np.float32),
            scales=np.asarray(payload["scales"], dtype=np.float32),
        )


# %% [markdown]
# ## 6. 无复制滑窗 Dataset

# %%
def history_counts(day_present: np.ndarray, lookback_days: int) -> np.ndarray:
    """返回以目标日结尾的 lookback_days 窗口内有数据的天数；包含目标日。"""
    cumsum = np.concatenate([[0], np.cumsum(day_present.astype(np.int32))])
    counts = np.zeros_like(day_present, dtype=np.int16)
    for day_idx in range(lookback_days - 1, len(day_present)):
        counts[day_idx] = cumsum[day_idx + 1] - cumsum[day_idx + 1 - lookback_days]
    return counts


def make_training_indices(
    store: MarketStore,
    labels: np.ndarray,
    target_day_mask: np.ndarray,
    lookback_days: int,
    min_history_fraction: float,
) -> np.ndarray:
    minimum = math.ceil(lookback_days * min_history_fraction)
    pairs: List[np.ndarray] = []
    for instrument_idx in range(len(store.instruments)):
        counts = history_counts(store.day_present[instrument_idx], lookback_days)
        valid = (
            target_day_mask
            & store.day_present[instrument_idx]
            & np.isfinite(labels[instrument_idx]).all(axis=-1)
            & (counts >= minimum)
        )
        day_ids = np.flatnonzero(valid)
        if len(day_ids):
            pairs.append(np.column_stack([np.full(len(day_ids), instrument_idx), day_ids]))
    if not pairs:
        raise RuntimeError("没有构造出有效训练样本，请检查日期范围、回看窗口和数据完整性。")
    return np.concatenate(pairs).astype(np.int32, copy=False)


def make_inference_indices(
    store: MarketStore,
    start_date: str,
    end_date: str,
    lookback_days: int,
    min_history_fraction: float,
) -> np.ndarray:
    start = np.datetime64(pd.Timestamp(start_date).date())
    end = np.datetime64(pd.Timestamp(end_date).date())
    target_day_mask = (store.days >= start) & (store.days <= end)
    minimum = math.ceil(lookback_days * min_history_fraction)
    pairs: List[np.ndarray] = []
    for instrument_idx in range(len(store.instruments)):
        counts = history_counts(store.day_present[instrument_idx], lookback_days)
        valid = target_day_mask & store.day_present[instrument_idx] & (counts >= minimum)
        day_ids = np.flatnonzero(valid)
        if len(day_ids):
            pairs.append(np.column_stack([np.full(len(day_ids), instrument_idx), day_ids]))
    if not pairs:
        raise RuntimeError("推理区间没有有效样本；请增加查询回看区间或检查数据源。")
    return np.concatenate(pairs).astype(np.int32, copy=False)


def deterministic_subsample(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(indices), size=maximum, replace=False))
    return indices[chosen]


def date_balanced_order(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """按交易日均衡下采样并打乱日期顺序，让大多数 batch 保持单日截面结构。"""
    rng = np.random.default_rng(seed)
    day_ids = np.unique(indices[:, 1])
    rng.shuffle(day_ids)
    quota = max(1, maximum // max(len(day_ids), 1)) if maximum > 0 else len(indices)
    groups = []
    for day_id in day_ids:
        group = indices[indices[:, 1] == day_id]
        if len(group) > quota:
            group = group[rng.choice(len(group), size=quota, replace=False)]
        else:
            group = group[rng.permutation(len(group))]
        groups.append(group)
    ordered = np.concatenate(groups).astype(np.int32, copy=False)
    return ordered[:maximum] if maximum > 0 and len(ordered) > maximum else ordered


def select_stocks_per_day(indices: np.ndarray, maximum_per_day: int, seed: int) -> np.ndarray:
    """保留所有交易日，并在每个日期内确定性抽样股票；0 表示不抽样。"""
    if maximum_per_day <= 0:
        return indices
    rng = np.random.default_rng(seed)
    groups: List[np.ndarray] = []
    for day_id in np.unique(indices[:, 1]):
        group = indices[indices[:, 1] == day_id]
        if len(group) > maximum_per_day:
            group = group[rng.choice(len(group), size=maximum_per_day, replace=False)]
        groups.append(group)
    return np.concatenate(groups).astype(np.int32, copy=False)


class DayChunkBatchSampler:
    """交易日感知的 batch sampler。

    V3.3 正式路径会再用 `assert_full_day_batches` 强制每日只产生一批；
    保留通用切块能力只用于小型单元测试。
    """
    def __init__(self, indices: np.ndarray, chunk_size: int, seed: int) -> None:
        self.indices = indices
        self.chunk_size = int(chunk_size)
        self.seed = seed
        self.epoch = 0
        self.groups = [np.flatnonzero(indices[:, 1] == day_id) for day_id in np.unique(indices[:, 1])]
        if self.chunk_size < 50:
            raise ValueError("截面 batch 至少需要 50 只股票")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(math.ceil(len(group) / self.chunk_size) for group in self.groups)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = rng.permutation(len(self.groups))
        for group_id in group_order:
            group = self.groups[group_id][rng.permutation(len(self.groups[group_id]))]
            chunks = [group[start:start + self.chunk_size] for start in range(0, len(group), self.chunk_size)]
            # 避免不足50只的尾批使相关性目标失效：并入前一批，不丢样本。
            if len(chunks) > 1 and len(chunks[-1]) < 50:
                chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
                chunks.pop()
            for chunk in chunks:
                yield chunk.tolist()


def assert_full_day_batches(
    indices: np.ndarray,
    sampler: DayChunkBatchSampler,
    stage: str,
) -> None:
    """截面模块必须一次看到该日所有可用股票。"""
    expected_days = int(len(np.unique(indices[:, 1])))
    if len(sampler) != expected_days:
        maximum = int(pd.Series(indices[:, 1]).value_counts().max())
        raise RuntimeError(
            f"{stage} 会把单日截面拆成多个 batch："
            f"最大日截面={maximum}，batch={sampler.chunk_size}。"
            "这会使截面注意力的训练/推理口径不一致。"
        )
    print(f"{stage} full-day batches: days={expected_days:,}; batches={len(sampler):,}")


class SequenceDataset(Dataset):
    def __init__(
        self,
        store: MarketStore,
        indices: np.ndarray,
        lookback_days: int,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.store = store
        self.indices = indices
        self.lookback_days = lookback_days
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        instrument_idx, target_day_idx = self.indices[item]
        start = target_day_idx - self.lookback_days + 1
        x = self.store.features[instrument_idx, start:target_day_idx + 1]
        present = self.store.token_present[instrument_idx, start:target_day_idx + 1]
        x_tensor = torch.from_numpy(x.reshape(-1, x.shape[-1]))
        padding_mask = torch.from_numpy((~present).reshape(-1))
        if self.labels is None:
            return x_tensor, padding_mask, int(instrument_idx), int(target_day_idx)
        y = torch.tensor(self.labels[instrument_idx, target_day_idx], dtype=torch.float32)
        return x_tensor, padding_mask, y, int(instrument_idx), int(target_day_idx)


# %% [markdown]
# ## 7. 从零初始化的 Transformer

# %%
class StockResidualBlock(nn.Module):
    """不依赖 batch 成员的个股残差 FFN。"""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.ffn(self.norm(state))


class CrossSectionLatentBlock(nn.Module):
    """用少量可学习潜变量在完整单日截面上交换信息。

    没有股票位置编码，因此对同日股票的输入排列保持置换等变；
    复杂度为 O(NK) 而非 O(N²)，N≈1000、K 为少量潜变量。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        latent_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_tokens = nn.Parameter(torch.empty(1, latent_count, d_model))
        self.latent_query_norm = nn.LayerNorm(d_model)
        self.stock_key_norm = nn.LayerNorm(d_model)
        self.latent_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.latent_residual = StockResidualBlock(d_model, dim_feedforward, dropout)
        self.stock_query_norm = nn.LayerNorm(d_model)
        self.latent_key_norm = nn.LayerNorm(d_model)
        self.stock_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.stock_residual = StockResidualBlock(d_model, dim_feedforward, dropout)

    def forward(self, stock_state: torch.Tensor) -> torch.Tensor:
        stocks = stock_state.unsqueeze(0)
        normalized_stocks = self.stock_key_norm(stocks)
        latents = self.latent_tokens.expand(stocks.shape[0], -1, -1)
        latent_update, _ = self.latent_attention(
            self.latent_query_norm(latents),
            normalized_stocks,
            normalized_stocks,
            need_weights=False,
        )
        latents = self.latent_residual(latents + latent_update)
        normalized_latents = self.latent_key_norm(latents)
        stock_update, _ = self.stock_attention(
            self.stock_query_norm(stocks),
            normalized_latents,
            normalized_latents,
            need_weights=False,
        )
        stocks = self.stock_residual(stocks + stock_update)
        return stocks.squeeze(0)


class MarketPatchTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        lookback_days: int,
        bars_per_day: int,
        patch_size: int,
        d_model: int,
        nhead: int,
        intraday_layers: int,
        num_layers: int,
        dim_feedforward: int,
        stock_mlp_layers: int,
        cross_sectional_layers: int,
        cross_sectional_latents: int,
        intraday_chunk_size: int,
        output_horizons: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lookback_days = lookback_days
        self.bars_per_day = bars_per_day
        self.patch_size = patch_size
        self.n_features = n_features
        self.patches_per_day = bars_per_day // patch_size
        self.output_horizons = output_horizons
        self.intraday_chunk_size = intraday_chunk_size
        self.patch_projection = nn.Linear(n_features * patch_size, d_model)
        self.day_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.intraday_position = nn.Parameter(torch.empty(1, self.patches_per_day + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal_position = nn.Parameter(torch.empty(1, lookback_days + 1, d_model))

        intraday_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.intraday_encoder = nn.TransformerEncoder(intraday_layer, num_layers=intraday_layers)
        # 可学习的局部时间混合位于模型内部，不生成或缓存任何人工因子。
        self.temporal_local = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)
        # CLS、全窗口均值和最新有效日分别代表全局、长期与最近状态。
        self.summary_projection = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stock_blocks = nn.ModuleList(
            StockResidualBlock(d_model, dim_feedforward, dropout)
            for _ in range(stock_mlp_layers)
        )
        self.cross_sectional_blocks = nn.ModuleList(
            CrossSectionLatentBlock(
                d_model, nhead, dim_feedforward, cross_sectional_latents, dropout
            )
            for _ in range(cross_sectional_layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_horizons),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.intraday_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        nn.init.trunc_normal_(self.day_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for block in self.cross_sectional_blocks:
            nn.init.trunc_normal_(block.latent_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # 小幅初始化最终回归层，避免训练早期输出尺度先于方向学习而爆炸。
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        expected = self.lookback_days * self.bars_per_day
        if x.shape[1] != expected:
            raise ValueError(f"输入 token 数 {x.shape[1]} 与配置 {expected} 不一致")

        # 第一级：把相邻 4 根 5m bar 交给可训练线性层学习为 20m patch，而非人工聚合。
        raw = x.reshape(
            batch_size * self.lookback_days, self.bars_per_day, self.n_features
        )
        raw_mask = padding_mask.reshape(
            batch_size * self.lookback_days, self.bars_per_day
        )
        patches = raw.reshape(
            raw.shape[0], self.patches_per_day, self.patch_size * self.n_features
        )
        tokens = self.patch_projection(patches)
        intraday_mask = raw_mask.reshape(
            raw_mask.shape[0], self.patches_per_day, self.patch_size
        ).all(dim=-1)
        day_token = self.day_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([day_token, tokens], dim=1) + self.intraday_position
        day_token_mask = torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=x.device)
        intraday_full_mask = torch.cat([day_token_mask, intraday_mask], dim=1)
        # 各股票-交易日的日内序列相互独立，因此这里分块不改变结果。
        # 4096×8 heads 低于 BigQuant PyTorch SDPA 的 CUDA launch grid 上限。
        encoded_chunks = []
        for start in range(0, tokens.shape[0], self.intraday_chunk_size):
            stop = min(start + self.intraday_chunk_size, tokens.shape[0])
            encoded_chunks.append(
                self.intraday_encoder(
                    tokens[start:stop],
                    src_key_padding_mask=intraday_full_mask[start:stop],
                )[:, 0]
            )
        day_encoded = torch.cat(encoded_chunks, dim=0)
        day_encoded = day_encoded.reshape(batch_size, self.lookback_days, -1)
        day_padding_mask = padding_mask.reshape(
            batch_size, self.lookback_days, self.bars_per_day
        ).all(dim=-1)
        day_encoded = day_encoded.masked_fill(day_padding_mask.unsqueeze(-1), 0.0)
        local = self.temporal_local(day_encoded.transpose(1, 2)).transpose(1, 2)
        day_encoded = day_encoded + local.masked_fill(
            day_padding_mask.unsqueeze(-1),
            0.0,
        )

        # 第二级：跨交易日建模；全缺失交易日仅作为 padding，不影响有效日期。
        cls = self.cls_token.expand(batch_size, -1, -1)
        temporal_tokens = torch.cat([cls, day_encoded], dim=1) + self.temporal_position
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        temporal_mask = torch.cat([cls_mask, day_padding_mask], dim=1)
        encoded = self.temporal_encoder(temporal_tokens, src_key_padding_mask=temporal_mask)
        valid_days = (~day_padding_mask).unsqueeze(-1)
        mean_state = (encoded[:, 1:] * valid_days).sum(dim=1) / valid_days.sum(dim=1).clamp_min(1)
        # 输入窗口以目标日结尾；最后位置是最新交易日，历史缺口不会改变其位置。
        latest_state = encoded[:, -1]
        stock_state = self.summary_projection(
            torch.cat([encoded[:, 0], mean_state, latest_state], dim=-1)
        )
        for block in self.stock_blocks:
            stock_state = block(stock_state)
        # 调用方强制一个 batch 对应完整的单日中证1000截面。
        # 训练、验证与推理因此不会出现截面成员口径漂移。
        for block in self.cross_sectional_blocks:
            stock_state = block(stock_state)
        return self.head(stock_state)


def build_model(cfg: Config, n_features: int) -> MarketPatchTransformer:
    return MarketPatchTransformer(
        n_features=n_features,
        lookback_days=cfg.lookback_days,
        bars_per_day=cfg.bars_per_day,
        patch_size=cfg.patch_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        intraday_layers=cfg.intraday_layers,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        stock_mlp_layers=cfg.stock_mlp_layers,
        cross_sectional_layers=cfg.cross_sectional_layers,
        cross_sectional_latents=cfg.cross_sectional_latents,
        intraday_chunk_size=cfg.intraday_chunk_size,
        output_horizons=len(cfg.target_horizons),
        dropout=cfg.dropout,
    )


def assert_parameter_count(model: nn.Module, cfg: Optional[Config] = None) -> int:
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not 100_000 <= count <= 100_000_000:
        raise ValueError(f"可训练参数量 {count:,} 不在 [100,000, 100,000,000] 内")
    if cfg is not None and not cfg.target_parameter_min <= count <= cfg.target_parameter_max:
        raise ValueError(
            f"可训练参数量 {count:,} 未达到本方案目标范围 "
            f"[{cfg.target_parameter_min:,}, {cfg.target_parameter_max:,}]"
        )
    print(f"trainable parameters: {count:,}")
    return count


# %% [markdown]
# ## 8. 训练、验证与产物保存

# %%
def daily_ic(y_true: np.ndarray, y_pred: np.ndarray, day_ids: np.ndarray) -> float:
    values: List[float] = []
    for day_id in np.unique(day_ids):
        mask = day_ids == day_id
        if mask.sum() < 20:
            continue
        x = y_true[mask]
        y = y_pred[mask]
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def cross_section_ic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    day_ids: torch.Tensor,
    minimum_size: int = 20,
) -> torch.Tensor:
    losses = []
    for day_id in torch.unique(day_ids):
        mask = day_ids == day_id
        if int(mask.sum()) < minimum_size:
            continue
        pred = prediction[mask] - prediction[mask].mean()
        true = target[mask] - target[mask].mean()
        denominator = torch.sqrt(pred.square().sum() * true.square().sum()).clamp_min(1e-8)
        losses.append(1.0 - (pred * true).sum() / denominator)
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    day_ids: torch.Tensor,
    minimum_size: int = 20,
    temperature: float = 0.75,
    pairings: int = 1,
) -> torch.Tensor:
    """同日多次随机配对，直接学习收益的次序关系。"""
    losses = []
    for day_id in torch.unique(day_ids):
        mask = day_ids == day_id
        pred = prediction[mask]
        true = target[mask]
        if len(pred) < minimum_size:
            continue
        for _ in range(pairings):
            order = torch.randperm(len(pred), device=pred.device)
            half = len(order) // 2
            left, right = order[:half], order[half:half * 2]
            direction = torch.sign(true[left] - true[right])
            valid = direction != 0
            if valid.any():
                margin = direction[valid] * (pred[left][valid] - pred[right][valid])
                losses.append(torch.nn.functional.softplus(-margin / temperature).mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def soft_tail_spread_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    day_ids: torch.Tensor,
    temperature: float,
    minimum_size: int = 50,
) -> torch.Tensor:
    """可微的多空尾部目标，与评分中的十分组多空方向一致。"""
    losses = []
    for day_id in torch.unique(day_ids):
        mask = day_ids == day_id
        if int(mask.sum()) < minimum_size:
            continue
        pred = prediction[mask]
        true = target[mask]
        pred = (pred - pred.mean()) / pred.std(unbiased=False).clamp_min(1e-6)
        long_weight = torch.softmax(pred / temperature, dim=0)
        short_weight = torch.softmax(-pred / temperature, dim=0)
        spread = (long_weight * true).sum() - (short_weight * true).sum()
        losses.append(-spread)
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def cross_section_standardize(
    values: torch.Tensor,
    day_ids: torch.Tensor,
    minimum_scale: float = 1e-4,
) -> torch.Tensor:
    """逐日中心化和标准化预测，只用于损失，不改变输入或落盘分数。"""
    normalized = torch.empty_like(values, dtype=torch.float32)
    for day_id in torch.unique(day_ids):
        mask = day_ids == day_id
        section = values[mask].float()
        scale = section.std(unbiased=False).clamp_min(minimum_scale)
        normalized[mask] = (section - section.mean()) / scale
    return normalized


def multihorizon_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    day_ids: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """所有截面损失强制在 FP32 中计算，避免 BF16 破坏微小相对差异。"""
    prediction = prediction.float()
    target = target.float()
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if target.ndim == 1:
        target = target[:, None]
    if prediction.shape != target.shape or prediction.shape[1] != len(cfg.horizon_weights):
        raise ValueError("模型输出与多周期监督目标形状不一致")
    components = {
        name: prediction.sum() * 0.0
        for name in ("reg", "ic", "rank", "tail", "dispersion")
    }
    for horizon_idx, horizon_weight in enumerate(cfg.horizon_weights):
        pred = prediction[:, horizon_idx]
        true = target[:, horizon_idx]
        pred_z = cross_section_standardize(pred, day_ids)
        true_z = cross_section_standardize(true, day_ids)
        # 原尺度回归把输出均值和方差锚定在每日 z-score 标签量级；
        # IC/rank 仍使用标准化预测，专注于截面方向。
        components["reg"] = components["reg"] + horizon_weight * torch.nn.functional.smooth_l1_loss(
            pred, true, beta=0.50
        )
        components["ic"] = components["ic"] + horizon_weight * cross_section_ic_loss(
            pred_z, true_z, day_ids
        )
        components["rank"] = components["rank"] + horizon_weight * pairwise_rank_loss(
            pred_z, true_z, day_ids,
            temperature=cfg.rank_temperature,
            pairings=cfg.rank_pairings,
        )
        components["tail"] = components["tail"] + horizon_weight * soft_tail_spread_loss(
            pred_z, true_z, day_ids, temperature=cfg.tail_temperature
        )
        section_scales = []
        for day_id in torch.unique(day_ids):
            section = pred[day_ids == day_id]
            if len(section) >= 20:
                section_scales.append(section.float().std(unbiased=False))
        if section_scales:
            scale = torch.stack(section_scales).mean().clamp_min(1e-4)
            target_scale = prediction.new_tensor(cfg.prediction_std_target)
            # 对数尺度损失同时惩罚塌缩和爆炸；std=0.1 与 std=10 受到同等惩罚。
            components["dispersion"] = components["dispersion"] + horizon_weight * (
                torch.log(scale / target_scale).square()
            )
    total = (
        cfg.regression_loss_weight * components["reg"]
        + cfg.ic_loss_weight * components["ic"]
        + cfg.rank_loss_weight * components["rank"]
        + cfg.tail_loss_weight * components["tail"]
        + cfg.dispersion_loss_weight * components["dispersion"]
    )
    return total, components


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    instrument_ids: np.ndarray,
    day_ids: np.ndarray,
    exposure_tensor: Optional[np.ndarray],
) -> Dict[str, float]:
    pearson, rank_ic, long_short, ordered_days = [], [], [], []
    for day_id in np.unique(day_ids):
        mask = day_ids == day_id
        if mask.sum() < 50:
            continue
        true = y_true[mask]
        pred = y_pred[mask].astype(np.float64, copy=True)
        instruments = instrument_ids[mask]
        # 与平台顺序一致：截面 winsorize → z-score → 对十个 BARRA 风格回归取残差。
        lower, upper = np.quantile(pred, [0.01, 0.99])
        pred = np.clip(pred, lower, upper)
        pred_scale = pred.std()
        if pred_scale <= 1e-12:
            continue
        pred = (pred - pred.mean()) / pred_scale
        if exposure_tensor is not None:
            exposure = exposure_tensor[instruments, int(day_id)].astype(np.float64, copy=False)
            complete = np.isfinite(exposure).all(axis=1) & np.isfinite(true)
            if complete.sum() < 50:
                continue
            true, pred, exposure = true[complete], pred[complete], exposure[complete]
            mean, scale = exposure.mean(axis=0), exposure.std(axis=0)
            useful = np.isfinite(scale) & (scale > 1e-8)
            design = np.column_stack([
                np.ones(len(exposure)),
                (exposure[:, useful] - mean[useful]) / scale[useful],
            ])
            penalty = 1e-4 * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(design.T @ design + penalty, design.T @ pred)
            pred = pred - design @ beta
        if np.std(true) <= 1e-12 or np.std(pred) <= 1e-12:
            continue
        pearson.append(float(np.corrcoef(true, pred)[0, 1]))
        rank_ic.append(float(np.corrcoef(_rank(true), _rank(pred))[0, 1]))
        count = max(10, len(pred) // 10)
        order = np.argsort(pred)
        long_short.append(float(true[order[-count:]].mean() - true[order[:count]].mean()))
        ordered_days.append(int(day_id))
    if not pearson:
        return {key: float("nan") for key in (
            "ic_mean", "rank_ic_mean", "ic_ir", "ic_t_stat", "positive_ic_ratio",
            "ls_sharpe", "stress_ic", "median_block_ic", "worst_block_index",
            "selection_score"
        )}
    ic = np.asarray(pearson)
    ric = np.asarray(rank_ic)
    ls = np.asarray(long_short)
    ic_ir = float(ic.mean() / (ic.std(ddof=1) + 1e-12))
    ls_sharpe = float(np.sqrt(252.0) * ls.mean() / (ls.std(ddof=1) + 1e-12))
    # 每 20 个验证交易日视为一个 regime，取最差块作为压力期代理。
    block_ic = [
        ic[start:start + 20].mean()
        for start in range(0, len(ic), 20)
        if len(ic[start:start + 20]) >= 10
    ]
    stress_ic = float(min(block_ic)) if block_ic else float(np.quantile(ic, 0.20))
    worst_block_index = int(np.argmin(block_ic)) if block_ic else -1
    # 与官方四个排名维度等权；RankIC 只作诊断，不额外改变选模方向。
    score = 0.25 * (
        np.tanh(ic.mean() / 0.025)
        + np.tanh(ic_ir / 0.40)
        + np.tanh(ls_sharpe / 2.0)
        + np.tanh(stress_ic / 0.02)
    )
    return {
        "ic_mean": float(ic.mean()),
        "rank_ic_mean": float(ric.mean()),
        "ic_ir": ic_ir,
        "ic_t_stat": float(ic_ir * np.sqrt(len(ic))),
        "positive_ic_ratio": float((ic > 0).mean()),
        "ls_sharpe": ls_sharpe,
        "stress_ic": stress_ic,
        "median_block_ic": float(np.median(block_ic)) if block_ic else float("nan"),
        "worst_block_index": float(worst_block_index),
        "selection_score": float(score),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    exposure_tensor: Optional[np.ndarray],
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.01, reduction="sum")
    total_loss = 0.0
    total_n = 0
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    instrument_ids: List[np.ndarray] = []
    day_ids: List[np.ndarray] = []
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    for x, mask, y, instrument_idx, day_idx in loader:
        if int(torch.unique(day_idx).numel()) != 1:
            raise RuntimeError("截面 Transformer 的验证 batch 混入了多个交易日")
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda", dtype=amp_dtype
        ):
            pred = model(x, mask)
        if pred.ndim == 1:
            pred = pred[:, None]
        if y.ndim == 1:
            y = y[:, None]
        total_loss += float(criterion(pred[:, 0], y[:, 0]).item())
        total_n += len(y)
        ys.append(y[:, 0].cpu().numpy())
        # NumPy 不支持 BigQuant 当前 PyTorch 返回的 BFloat16 张量。
        preds.append(pred[:, 0].float().cpu().numpy())
        instrument_ids.append(np.asarray(instrument_idx))
        day_ids.append(np.asarray(day_idx))
    y_all = np.concatenate(ys)
    pred_all = np.concatenate(preds)
    instrument_all = np.concatenate(instrument_ids)
    day_all = np.concatenate(day_ids)
    return total_loss / max(total_n, 1), validation_metrics(
        y_all, pred_all, instrument_all, day_all, exposure_tensor
    )


def artifact_payload(
    model: nn.Module,
    cfg: Config,
    preprocessor: FieldPreprocessor,
    feature_columns: Sequence[str],
    close_column: str,
    parameter_count: int,
    validation_ic: float,
) -> Dict[str, object]:
    return {
        "format_version": 33,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "preprocessor": preprocessor.to_dict(),
        "feature_columns": list(feature_columns),
        "close_column": close_column,
        "parameter_count": int(parameter_count),
        "validation_ic": float(validation_ic) if np.isfinite(validation_ic) else None,
        "training_policy": "random initialization; competition data only; no external pretrained weights",
    }


def save_artifact_json(payload: Dict[str, object], path: str) -> str:
    """压缩并原子写入文本 JSON；中途崩溃不会破坏上一份最佳 checkpoint。"""
    storage_dtype = str(
        payload.get("config", {}).get("artifact_storage_dtype", "float16")
    )
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(f"不支持的权重存储 dtype: {storage_dtype}")
    floating_storage_dtype = np.dtype(storage_dtype)
    serializable = {key: value for key, value in payload.items() if key != "state_dict"}
    serializable["state_dict"] = {}
    for name, value in payload["state_dict"].items():
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(floating_storage_dtype, copy=False)
        compressed = zlib.compress(array.tobytes(order="C"), level=6)
        serializable["state_dict"][name] = {
            "encoding": "zlib-base64",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(compressed).decode("ascii"),
        }
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return str(destination)


def read_artifact_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dict = {}
    for name, metadata in payload["state_dict"].items():
        if metadata.get("encoding") == "zlib-base64":
            raw = zlib.decompress(base64.b64decode(metadata["data"]))
            array = np.frombuffer(raw, dtype=np.dtype(metadata["dtype"])).reshape(metadata["shape"])
            tensor = torch.from_numpy(array.copy())
        else:  # 兼容早期 JSON list 格式。
            tensor = torch.tensor(
                metadata["data"], dtype=getattr(torch, metadata["dtype"])
            ).reshape(metadata["shape"])
        state_dict[name] = tensor
    payload["state_dict"] = state_dict
    return payload


def verify_saved_state(model: nn.Module, path: str) -> float:
    """校验落盘权重；V3.3 float16 存储只允许标准舍入误差。"""
    payload = read_artifact_json(path)
    restored = payload["state_dict"]
    storage_dtype = str(
        payload.get("config", {}).get("artifact_storage_dtype", "float16")
    )
    maximum_error = 0.0
    maximum_scaled_error = 0.0
    for name, expected in model.state_dict().items():
        actual = restored[name].to(dtype=expected.dtype)
        expected_cpu = expected.detach().cpu()
        error = float((expected_cpu - actual).abs().max().item())
        scale = max(float(expected_cpu.abs().max().item()), 1.0)
        maximum_error = max(maximum_error, error)
        maximum_scaled_error = max(maximum_scaled_error, error / scale)
    failed = (
        maximum_scaled_error > 1e-3
        if storage_dtype == "float16"
        else maximum_error > 1e-7
    )
    if failed:
        raise RuntimeError(
            f"保存后权重校验失败，dtype={storage_dtype}, "
            f"最大绝对误差={maximum_error:.3e}, "
            f"最大尺度误差={maximum_scaled_error:.3e}"
        )
    print(
        f"artifact reload verified: storage_dtype={storage_dtype}; "
        f"max_abs_error={maximum_error:.3e}; "
        f"max_scaled_error={maximum_scaled_error:.3e}"
    )
    return maximum_error


def convert_v3_2_artifact_to_v3_3(
    source_path: str = "bigalpha_e2e_highscore_v3_2.json",
    destination_path: str = "bigalpha_e2e_highscore_v3_3.json",
) -> Dict[str, object]:
    """将 V3.2 float16 正式权重无损升级为 V3.3，不重训也不覆盖 V3.2。"""
    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    if source == destination:
        raise ValueError("输出文件不能与 V3.2 源权重相同")
    if destination.name != "bigalpha_e2e_highscore_v3_3.json":
        raise ValueError(
            "V3.3 正式输出必须命名为 bigalpha_e2e_highscore_v3_3.json"
        )
    if not source.exists():
        raise FileNotFoundError(f"找不到 V3.2 权重: {source}")

    payload = read_artifact_json(str(source))
    if int(payload.get("format_version", -1)) != 32:
        raise RuntimeError("输入不是 format_version=32 的 V3.2 权重")
    if not bool(payload.get("final_training_complete", False)):
        raise RuntimeError("只允许升级 final_training_complete=True 的 V3.2 正式权重")

    source_state = payload["state_dict"]
    config = dict(payload["config"])
    config["artifact_path"] = destination.name
    config["artifact_storage_dtype"] = "float16"
    config["inference_chunk_trading_days"] = CFG.inference_chunk_trading_days
    config["inference_history_calendar_days"] = CFG.inference_history_calendar_days
    payload["config"] = config
    payload["format_version"] = 33
    payload["converted_from"] = source.name
    payload["converted_at"] = pd.Timestamp.utcnow().isoformat()
    save_artifact_json(payload, str(destination))

    restored = read_artifact_json(str(destination))["state_dict"]
    maximum_error = 0.0
    maximum_scaled_error = 0.0
    for name, expected in source_state.items():
        actual = restored[name].to(dtype=expected.dtype)
        expected_cpu = expected.detach().cpu()
        error = float((expected_cpu - actual).abs().max().item())
        scale = max(float(expected_cpu.abs().max().item()), 1.0)
        maximum_error = max(maximum_error, error)
        maximum_scaled_error = max(maximum_scaled_error, error / scale)
    if maximum_scaled_error > 1e-3:
        raise RuntimeError(
            f"V3.3 float16 转换误差过大: "
            f"abs={maximum_error:.3e}, scaled={maximum_scaled_error:.3e}"
        )

    source_mib = source.stat().st_size / 1024 ** 2
    destination_mib = destination.stat().st_size / 1024 ** 2
    summary = {
        "source_path": str(source),
        "destination_path": str(destination),
        "source_mib": source_mib,
        "destination_mib": destination_mib,
        "size_ratio": destination_mib / max(source_mib, 1e-12),
        "maximum_weight_error": maximum_error,
        "maximum_scaled_weight_error": maximum_scaled_error,
        "v3_2_preserved": source.exists(),
    }
    print(
        f"V3.2 preserved: {source}\n"
        f"V3.3 saved: {destination}\n"
        f"artifact size: {source_mib:.1f} MiB -> {destination_mib:.1f} MiB\n"
        f"max_weight_error={maximum_error:.3e}; "
        f"max_scaled_error={maximum_scaled_error:.3e}"
    )
    return summary


class ExponentialMovingAverage:
    """只对浮点参数做 EMA；验证和最终产物使用平滑后的权重。"""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if torch.is_floating_point(value):
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        if self.backup is not None:
            raise RuntimeError("EMA 权重已经应用")
        self.backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            raise RuntimeError("没有可恢复的在线权重")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None


class CheckpointAverager:
    """对同一条训练轨迹的 epoch EMA 检查点做等权平均。

    这仍是一个单模型、单份权重；目的是降低单一 epoch 终点对压力
    regime 的偶然敏感性，不增加参数量，也不在推理时做多模型拼接。
    """

    def __init__(self) -> None:
        self.average: Dict[str, torch.Tensor] = {}
        self.count = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        self.count += 1
        if self.count == 1:
            self.average = {
                name: value.detach().clone()
                for name, value in state.items()
            }
            return
        weight = 1.0 / float(self.count)
        for name, value in state.items():
            averaged = self.average[name]
            if torch.is_floating_point(value):
                averaged.lerp_(value.detach(), weight)
            else:
                averaged.copy_(value.detach())

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        if self.count <= 0:
            raise RuntimeError("尚无可应用的 epoch EMA 检查点")
        model.load_state_dict(self.average, strict=True)


def resolve_datasource(datasources: Optional[Dict[str, str]], key: str, fallback: str) -> str:
    if datasources is None:
        return validate_identifier(fallback)
    if not isinstance(datasources, dict) or key not in datasources:
        raise ValueError(f"datasources 必须是包含 {key!r} 的字典")
    return validate_identifier(datasources[key])


def _run_training(
    datasources: Optional[Dict[str, str]] = None,
    cfg: Config = CFG,
    allow_cpu: bool = False,
    mode: str = "formal",
) -> Dict[str, object]:
    """运行研发验证或正式全量训练；两个阶段绝不复用模型权重。"""
    if mode not in {"development", "formal"}:
        raise ValueError("mode 必须是 'development' 或 'formal'")
    set_global_seed(cfg.seed)
    assert_config_compliance(cfg)
    assert_training_resources(cfg, allow_cpu=allow_cpu)
    started = time.monotonic()

    train_table = resolve_datasource(datasources, cfg.datasource_key, cfg.data_source)
    feature_columns, close_column = resolve_feature_columns(
        train_table, cfg.train_start, cfg.feature_candidates
    )
    assert_full_order_book_depth(feature_columns, depth=5)
    # 训练严格不查询 train_start 之前的数据。初始 lookback 不足的日期会由
    # make_training_indices 自动排除，避免即使“读取后掩码”也被视为越界访问。
    query_start = pd.Timestamp(cfg.train_start).strftime("%Y-%m-%d")
    print("features:", feature_columns)
    print("loading:", query_start, "to", cfg.train_end)
    frame = load_market_frame(train_table, query_start, cfg.train_end, feature_columns)
    store = build_market_store(frame, feature_columns, close_column, cfg.bars_per_day)
    del frame
    gc.collect()

    # 规则要求只使用历史时点中证1000成分，并且训练不得借用 2019 年以前的数据。
    training_universe = load_official_universe(query_start, cfg.train_end)
    universe_mask = build_universe_mask(store, training_universe)
    restrict_store_to_universe(store, universe_mask, earliest_allowed_day=cfg.train_start)
    del training_universe, universe_mask
    gc.collect()

    labels = build_forward_return_labels(store, cfg.target_horizons)
    residualization_stats: Dict[str, float] = {}
    exposure_tensor: Optional[np.ndarray] = None
    if cfg.use_exposure_residualization:
        exposure_columns = resolve_exposure_columns(cfg.exposure_source, cfg.train_start)
        exposure_frame = load_exposure_frame(
            cfg.exposure_source, cfg.train_start, cfg.train_end, exposure_columns
        )
        labels, residualization_stats = residualize_labels_with_exposure(
            labels, store, exposure_frame, exposure_columns
        )
        if mode == "development":
            exposure_tensor = build_exposure_tensor(store, exposure_frame, exposure_columns)
        del exposure_frame
        gc.collect()
    labels = normalize_labels_by_day(labels)
    end_day = np.datetime64(pd.Timestamp(cfg.train_end).date())
    start_day = np.datetime64(pd.Timestamp(cfg.train_start).date())
    eligible_days = np.flatnonzero((store.days >= start_day) & (store.days <= end_day))
    full_day_mask = (store.days >= start_day) & (store.days <= end_day)
    validation_loader: Optional[DataLoader] = None
    validation_indices = np.empty((0, 2), dtype=np.int32)
    if mode == "development":
        if len(eligible_days) <= cfg.validation_days + cfg.lookback_days:
            raise RuntimeError("训练区间不足以完成回看和时序验证切分")
        validation_start_idx = eligible_days[-cfg.validation_days]
        # 所有监督标签都不得跨入验证区间。
        embargo = max(cfg.target_horizons)
        train_day_mask = (store.days >= start_day) & (
            np.arange(len(store.days)) < validation_start_idx - embargo
        )
        validation_day_mask = (
            (np.arange(len(store.days)) >= validation_start_idx) & (store.days <= end_day)
        )
        scaler_day_mask = train_day_mask
        training_day_mask = train_day_mask
        epochs = cfg.development_epochs
    else:
        # 正式产物的 scaler 和监督样本都使用完整 2019–2024 训练区间。
        scaler_day_mask = full_day_mask
        training_day_mask = full_day_mask
        validation_day_mask = np.zeros(len(store.days), dtype=bool)
        epochs = cfg.formal_epochs

    preprocessor = FieldPreprocessor.fit(
        store.features[:, scaler_day_mask, :, :], feature_columns, cfg.log_field_patterns
    )
    preprocessor.transform_inplace(store.features)
    training_labels = trim_training_label_extremes(
        labels, training_day_mask, cfg.training_label_trim_fraction
    )
    training_indices = make_training_indices(
        store, training_labels, training_day_mask, cfg.lookback_days, cfg.min_history_fraction
    )
    training_indices = select_stocks_per_day(
        training_indices, cfg.max_stocks_per_day, cfg.seed
    )
    if mode == "development":
        validation_indices = make_training_indices(
            store, labels, validation_day_mask, cfg.lookback_days, cfg.min_history_fraction
        )
        validation_indices = deterministic_subsample(
            validation_indices, cfg.max_validation_samples, cfg.seed + 1
        )
        validation_dataset = SequenceDataset(store, validation_indices, cfg.lookback_days, labels)
        validation_sampler = DayChunkBatchSampler(
            validation_indices, cfg.eval_stocks_per_batch, cfg.seed + 1
        )
        assert_full_day_batches(validation_indices, validation_sampler, "validation")
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_sampler,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        # 用“标签预测自身”审计验证口径与符号；正常应接近 +1。
        audit_indices = deterministic_subsample(
            validation_indices, 100_000, cfg.seed + 17
        )
        audit_target = labels[
            audit_indices[:, 0], audit_indices[:, 1], 0
        ].astype(np.float64)
        audit_metrics = validation_metrics(
            audit_target,
            audit_target.copy(),
            audit_indices[:, 0],
            audit_indices[:, 1],
            exposure_tensor,
        )
        print("label direction audit:", audit_metrics)
        if audit_metrics.get("ic_mean", float("nan")) < 0.80:
            raise RuntimeError(
                "标签方向/风格剔除审计失败；停止训练，避免用错误口径解释负 IC。"
            )
    print(
        f"mode={mode}; training samples={len(training_indices):,}; "
        f"validation samples={len(validation_indices):,}; "
        f"stocks/train-batch={cfg.train_stocks_per_batch}"
    )
    data_seconds = time.monotonic() - started
    print(f"data pipeline ready: elapsed={data_seconds:.1f}s")

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    def make_optimizer(model: nn.Module, total_steps: int):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return max(1e-3, (step + 1) / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        scaler = torch.cuda.amp.GradScaler(
            enabled=torch.cuda.is_available() and not use_bf16
        )
        return optimizer, scheduler, scaler

    def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer,
        scheduler,
        scaler,
        ema: ExponentialMovingAverage,
        deadline_seconds: float,
    ) -> Tuple[Dict[str, float], int, int, bool]:
        model.train()
        running = {
            name: 0.0
            for name in ("loss", "reg", "ic", "rank", "tail", "dispersion", "batch_ic", "pred_std")
        }
        seen, updates = 0, 0
        completed = True
        for x, mask, y, instrument_idx, day_idx in loader:
            if time.monotonic() - started >= deadline_seconds:
                completed = False
                break
            if int(torch.unique(day_idx).numel()) != 1:
                raise RuntimeError("截面 Transformer 的训练 batch 混入了多个交易日")
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            day_idx = day_idx.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available(),
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
            ):
                pred = model(x, mask)
            # 模型前向使用 BF16 提速，但相关性、排序和尾部权重必须使用 FP32。
            with torch.cuda.amp.autocast(enabled=False):
                loss, components = multihorizon_objective(pred, y, day_idx, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            running["loss"] += float(loss.item()) * len(y)
            for name, value in components.items():
                running[name] += float(value.item()) * len(y)
            with torch.no_grad():
                primary_pred = pred[:, 0].float()
                primary_true = y[:, 0].float()
                centered_pred = primary_pred - primary_pred.mean()
                centered_true = primary_true - primary_true.mean()
                correlation = (
                    (centered_pred * centered_true).sum()
                    / torch.sqrt(
                        centered_pred.square().sum() * centered_true.square().sum()
                    ).clamp_min(1e-8)
                )
                running["batch_ic"] += float(correlation.item()) * len(y)
                running["pred_std"] += float(primary_pred.std(unbiased=False).item()) * len(y)
            seen += len(y)
            updates += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        averages = {name: value / max(seen, 1) for name, value in running.items()}
        return averages, seen, updates, completed

    training_dataset = SequenceDataset(
        store, training_indices, cfg.lookback_days, training_labels
    )
    training_sampler = DayChunkBatchSampler(
        training_indices, cfg.train_stocks_per_batch, cfg.seed
    )
    assert_full_day_batches(training_indices, training_sampler, "training")
    training_loader = DataLoader(
        training_dataset,
        batch_sampler=training_sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(cfg, len(feature_columns)).to(DEVICE)
    parameter_count = assert_parameter_count(model, cfg)
    optimizer, scheduler, scaler = make_optimizer(
        model, max(1, len(training_loader) * epochs)
    )
    ema = ExponentialMovingAverage(model, cfg.ema_decay)
    checkpoint_averager = CheckpointAverager()
    best_score = -float("inf")
    best_metrics: Dict[str, float] = {}
    last_metrics: Dict[str, float] = {}
    last_train_stats: Dict[str, float] = {}
    best_epoch = 0
    peak_gpu_gib = 0.0
    samples_seen = 0
    optimizer_updates = 0
    training_started = time.monotonic()
    training_complete = True
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        training_sampler.set_epoch(epoch)
        train_stats, seen, updates, completed = run_epoch(
            model, training_loader, optimizer, scheduler, scaler, ema, cfg.max_train_seconds
        )
        samples_seen += seen
        optimizer_updates += updates
        last_train_stats = dict(train_stats)
        if not completed:
            training_complete = False
            print(f"{mode} stage reached the configured time budget")
            break
        metrics: Dict[str, float] = {}
        validation_loss = float("nan")
        # 训练前期尚未收敛的权重不进入最终平均；从固定的起始轮次起，
        # 将 epoch EMA 纳入同轨迹检查点平均。研发评估与正式落盘同口径。
        ema.apply(model)
        try:
            if epoch >= cfg.checkpoint_average_start_epoch:
                checkpoint_averager.update(model)
                checkpoint_averager.apply(model)
            if mode == "development":
                if validation_loader is None:
                    raise RuntimeError("研发模式缺少验证 DataLoader")
                validation_loss, metrics = evaluate_model(
                    model, validation_loader, DEVICE, exposure_tensor
                )
        finally:
            ema.restore(model)
        print(
            f"{mode} epoch={epoch}/{epochs} updates={updates:,} samples={seen:,} "
            f"averaged_checkpoints={checkpoint_averager.count} "
            + " ".join(f"train_{key}={value:.6f}" for key, value in train_stats.items())
            + (f" val_loss={validation_loss:.6f} " + " ".join(
                f"{key}={value:.6f}" for key, value in metrics.items()
            ) if metrics else "")
        )
        if metrics:
            last_metrics = dict(metrics)
            score = metrics.get("selection_score", float("nan"))
            if not np.isfinite(score):
                score = -validation_loss
            if score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                best_epoch = epoch
        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / 1024**3
            peak_gpu_gib = max(peak_gpu_gib, current_peak)
            print(
                "cuda memory: "
                f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
                f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB, "
                f"peak={current_peak:.2f} GiB"
            )

    training_seconds = time.monotonic() - training_started
    common_summary = {
        "mode": mode,
        "parameter_count": parameter_count,
        "feature_columns": feature_columns,
        "eligible_days": int(len(eligible_days)),
        "samples_seen": int(samples_seen),
        "optimizer_updates": int(optimizer_updates),
        "checkpoint_average_count": int(checkpoint_averager.count),
        "checkpoint_average_start_epoch": int(cfg.checkpoint_average_start_epoch),
        "data_seconds": data_seconds,
        "training_seconds": training_seconds,
        "peak_gpu_gib": peak_gpu_gib,
        "elapsed_seconds": time.monotonic() - started,
    }
    if mode == "development":
        if not last_metrics:
            raise RuntimeError("研发训练未产生有效验证指标")
        gate_checks = {
            "ic": last_metrics.get("ic_mean", -float("inf")) > cfg.minimum_development_ic,
            "rank_ic": last_metrics.get("rank_ic_mean", -float("inf"))
            > cfg.minimum_development_rank_ic,
            "long_short": last_metrics.get("ls_sharpe", -float("inf"))
            > cfg.minimum_development_ls_sharpe,
            "stress": last_metrics.get("stress_ic", -float("inf"))
            > cfg.minimum_development_stress_ic,
            "ic_t_stat": last_metrics.get("ic_t_stat", -float("inf"))
            >= cfg.minimum_development_ic_t_stat,
            "positive_ic_ratio": last_metrics.get("positive_ic_ratio", -float("inf"))
            > cfg.minimum_development_positive_ic_ratio,
            "train_ic": last_train_stats.get("batch_ic", -float("inf"))
            > cfg.minimum_training_batch_ic,
            "prediction_scale": cfg.minimum_training_pred_std
            <= last_train_stats.get("pred_std", float("inf"))
            <= cfg.maximum_training_pred_std,
        }
        direction_passed = bool(all(gate_checks[key] for key in (
            "ic", "rank_ic", "long_short", "stress", "ic_t_stat", "positive_ic_ratio"
        )))
        scale_passed = bool(all(gate_checks[key] for key in (
            "train_ic", "prediction_scale"
        )))
        validation_day_ids = np.unique(validation_indices[:, 1])
        worst_block_index = int(last_metrics.get("worst_block_index", -1))
        worst_start = worst_block_index * 20
        worst_stress_window: Optional[Dict[str, str]] = None
        if 0 <= worst_start < len(validation_day_ids):
            worst_end = min(worst_start + 20, len(validation_day_ids)) - 1
            worst_stress_window = {
                "start": str(store.days[validation_day_ids[worst_start]]),
                "end": str(store.days[validation_day_ids[worst_end]]),
            }
        common_summary.update({
            "validation_metrics": last_metrics,
            "best_validation_metrics": best_metrics,
            "best_epoch": int(best_epoch),
            "gate_checks": gate_checks,
            "worst_stress_window": worst_stress_window,
            "direction_passed": direction_passed,
            "training_stability_passed": scale_passed,
            "ready_for_formal_training": bool(
                direction_passed and scale_passed and training_complete
            ),
        })
        print("development gate checks:", gate_checks)
        print("worst stress window:", worst_stress_window)
        if direction_passed and scale_passed:
            print("DEVELOPMENT GATE PASSED: 可以使用冻结的 CFG 运行正式训练。")
        else:
            print(
                "DEVELOPMENT GATE FAILED: 方向、压力期或训练输出尺度未达标；"
                "禁止正式训练和提交。"
            )
        return common_summary

    if not training_complete:
        raise RuntimeError("正式训练未在时间预算内完成；旧正式产物（如存在）未被覆盖。")
    expected_checkpoint_count = epochs - cfg.checkpoint_average_start_epoch + 1
    if checkpoint_averager.count != expected_checkpoint_count:
        raise RuntimeError(
            f"epoch EMA 检查点不完整（{checkpoint_averager.count}/"
            f"{expected_checkpoint_count}），"
            "拒绝保存和提交。"
        )
    final_batch_ic = last_train_stats.get("batch_ic", -float("inf"))
    final_pred_std = last_train_stats.get("pred_std", float("inf"))
    if final_batch_ic <= cfg.minimum_training_batch_ic:
        raise RuntimeError(
            f"正式训练未建立正训练 IC（train_batch_ic={final_batch_ic:.6f}），"
            "拒绝保存和提交。"
        )
    if not cfg.minimum_training_pred_std <= final_pred_std <= cfg.maximum_training_pred_std:
        raise RuntimeError(
            f"正式训练输出尺度异常（train_pred_std={final_pred_std:.6f}，"
            f"要求 [{cfg.minimum_training_pred_std}, {cfg.maximum_training_pred_std}]），"
            "拒绝保存和提交。"
        )

    save_started = time.monotonic()
    checkpoint_averager.apply(model)
    payload = artifact_payload(
        model, cfg, preprocessor, feature_columns, close_column, parameter_count,
        float("nan"),
    )
    payload.update({
        "train_table": train_table,
        "label_residualization": residualization_stats,
        "formal_epochs": int(cfg.formal_epochs),
        "checkpoint_average_count": int(checkpoint_averager.count),
        "final_train_stats": last_train_stats,
        "final_training_complete": True,
    })
    save_artifact_json(payload, cfg.artifact_path)
    verify_saved_state(model, cfg.artifact_path)
    print(f"saved formal artifact: {Path(cfg.artifact_path).resolve()}")
    common_summary.update({
        "artifact_path": str(Path(cfg.artifact_path).resolve()),
        "formal_epochs": int(cfg.formal_epochs),
        "final_train_stats": last_train_stats,
        "final_training_complete": True,
        "serialization_seconds": time.monotonic() - save_started,
        "elapsed_seconds": time.monotonic() - started,
    })
    return common_summary


def develop_model(
    datasources: Optional[Dict[str, str]] = None,
    cfg: Config = CFG,
    allow_cpu: bool = False,
) -> Dict[str, object]:
    """一次性严格时序验证；不生成可提交权重。"""
    return _run_training(datasources, cfg, allow_cpu, mode="development")


def train_and_save(
    datasources: Optional[Dict[str, str]] = None,
    cfg: Config = CFG,
    allow_cpu: bool = False,
) -> Dict[str, object]:
    """冻结配置后，从零使用全部公开训练区间训练一次并保存正式权重。"""
    return _run_training(datasources, cfg, allow_cpu, mode="formal")


# %% [markdown]
# ## 9. 推理与比赛 `main` 入口

# %%
def load_artifact(path: str, device: torch.device = DEVICE) -> Tuple[Dict[str, object], Config, FieldPreprocessor, nn.Module]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"未找到模型产物 {path!r}。公榜提交前请先在 BigQuant Notebook 执行 train_and_save()，"
            "并将生成的权重文件与本 Notebook 一并提交。"
        )
    payload = read_artifact_json(path)
    if payload.get("format_version") in {31, 32, 33} and not payload.get("final_training_complete", False):
        raise RuntimeError(
            "V3.3 权重的正式全量训练尚未完成，禁止用于提交；"
            "请重新运行 train_and_save 并确认 final_training_complete=True。"
        )
    cfg_payload = dict(payload["config"])
    for key in (
        "feature_candidates", "log_field_patterns", "target_horizons", "horizon_weights"
    ):
        if key in cfg_payload:
            cfg_payload[key] = tuple(cfg_payload[key])
    artifact_cfg = Config(**cfg_payload)
    preprocessor = FieldPreprocessor.from_dict(payload["preprocessor"])
    model = build_model(artifact_cfg, len(payload["feature_columns"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    assert_parameter_count(model, artifact_cfg)
    model.to(device).eval()
    return payload, artifact_cfg, preprocessor, model


@torch.no_grad()
def predict_store(
    model: nn.Module,
    store: MarketStore,
    indices: np.ndarray,
    cfg: Config,
    device: torch.device = DEVICE,
) -> pd.DataFrame:
    dataset = SequenceDataset(store, indices, cfg.lookback_days, labels=None)
    sampler = DayChunkBatchSampler(indices, cfg.eval_stocks_per_batch, cfg.seed)
    assert_full_day_batches(indices, sampler, "inference")
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    rows: List[pd.DataFrame] = []
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    for x, mask, instrument_idx, day_idx in loader:
        if int(torch.unique(day_idx).numel()) != 1:
            raise RuntimeError("截面 Transformer 的推理 batch 混入了多个交易日")
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda", dtype=amp_dtype
        ):
            pred = model(
                x.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
            )
        pred = pred.float().cpu().numpy()
        if pred.ndim == 2:
            pred = pred[:, 0]
        instrument_idx = np.asarray(instrument_idx)
        day_idx = np.asarray(day_idx)
        rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(store.days[day_idx]),
                    "instrument": store.instruments[instrument_idx].astype(str),
                    "score": pred.astype(np.float64),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    return validate_submission_frame(result)


def validate_submission_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected = ["date", "instrument", "score"]
    if list(frame.columns) != expected:
        raise ValueError(f"输出列必须严格为 {expected}，当前为 {list(frame.columns)}")
    if frame.duplicated(["date", "instrument"]).any():
        raise ValueError("输出存在重复的 date/instrument")
    if not np.isfinite(frame["score"].to_numpy()).all():
        raise ValueError("score 包含 NaN 或无穷值")
    frame = frame.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    coverage = frame.groupby("date", observed=True)["instrument"].nunique()
    if coverage.empty:
        raise ValueError("输出为空")
    dispersion = frame.groupby("date", observed=True)["score"].agg(["count", "nunique", "std"])
    collapsed = dispersion[
        (dispersion["count"] >= 50)
        & ((dispersion["nunique"] < 20) | (dispersion["std"].fillna(0.0) < 1e-8))
    ]
    if len(collapsed):
        raise ValueError(
            "模型截面分数塌缩，无法形成可靠排序："
            + repr(collapsed.head().to_dict("index"))
        )
    print(
        f"submission rows={len(frame):,}; dates={len(coverage):,}; "
        f"min instruments/day={int(coverage.min())}; median={int(coverage.median())}; "
        f"median score std={float(dispersion['std'].median()):.6g}"
    )
    return frame


def assert_store_coverage(
    result: pd.DataFrame,
    store: MarketStore,
    start_date: str,
    end_date: str,
    minimum_ratio: float = 0.60,
) -> None:
    start = np.datetime64(pd.Timestamp(start_date).date())
    end = np.datetime64(pd.Timestamp(end_date).date())
    target_ids = np.flatnonzero((store.days >= start) & (store.days <= end))
    actual = result.groupby("date", observed=True)["instrument"].nunique()
    failures = []
    for day_idx in target_ids:
        day = pd.Timestamp(store.days[day_idx])
        universe = int(store.day_present[:, day_idx].sum())
        produced = int(actual.get(day, 0))
        ratio = produced / universe if universe else 0.0
        if universe and ratio < minimum_ratio:
            failures.append((day.strftime("%Y-%m-%d"), produced, universe, ratio))
    if failures:
        raise ValueError(
            "截面覆盖率低于 60%，前几项为 "
            + repr(failures[:5])
            + "。请检查查询回看缓冲或 min_history_fraction。"
        )


def align_official_universe(
    result: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    universe = load_official_universe(start_date, end_date)
    aligned = result.merge(universe, on=["date", "instrument"], how="inner")
    return aligned.loc[:, ["date", "instrument", "score"]], universe


def assert_official_coverage(
    result: pd.DataFrame,
    universe: pd.DataFrame,
    minimum_ratio: float = 0.60,
) -> None:
    expected = universe.groupby("date", observed=True)["instrument"].nunique()
    actual = result.groupby("date", observed=True)["instrument"].nunique()
    failures = []
    for day, universe_size in expected.items():
        produced = int(actual.get(day, 0))
        ratio = produced / int(universe_size) if universe_size else 0.0
        if universe_size and ratio < minimum_ratio:
            failures.append((day.strftime("%Y-%m-%d"), produced, int(universe_size), ratio))
    if failures:
        raise ValueError("官方中证1000截面覆盖率低于60%：" + repr(failures[:5]))


def _predict_inference_chunk(
    table: str,
    chunk_start: str,
    chunk_end: str,
    feature_columns: Sequence[str],
    close_column: str,
    artifact_cfg: Config,
    preprocessor: FieldPreprocessor,
    model: nn.Module,
    require_competition_coverage: bool,
) -> pd.DataFrame:
    """只保留一个目标日期块及其历史窗口；返回后调用者立即释放整块内存。"""
    query_start = (
        pd.Timestamp(chunk_start)
        - pd.Timedelta(days=artifact_cfg.inference_history_calendar_days)
    ).strftime("%Y-%m-%d")
    print(f"inference chunk: target={chunk_start} to {chunk_end}; query_start={query_start}")
    frame = load_market_frame(
        table, query_start, chunk_end, feature_columns
    )
    store = build_market_store(
        frame, feature_columns, close_column, artifact_cfg.bars_per_day
    )
    del frame
    gc.collect()

    history_universe = load_official_universe(query_start, chunk_end)
    universe_mask = build_universe_mask(store, history_universe)
    restrict_store_to_universe(store, universe_mask)
    del history_universe, universe_mask
    gc.collect()

    preprocessor.transform_inplace(store.features)
    indices = make_inference_indices(
        store,
        chunk_start,
        chunk_end,
        artifact_cfg.lookback_days,
        artifact_cfg.min_history_fraction,
    )
    indices = select_stocks_per_day(
        indices, artifact_cfg.max_stocks_per_day, artifact_cfg.seed
    )
    result = predict_store(model, store, indices, artifact_cfg, DEVICE)
    result = result[
        (result["date"] >= pd.Timestamp(chunk_start).normalize())
        & (result["date"] <= pd.Timestamp(chunk_end).normalize())
    ].reset_index(drop=True)
    result = validate_submission_frame(
        result.loc[:, ["date", "instrument", "score"]]
    )
    if require_competition_coverage:
        assert_store_coverage(result, store, chunk_start, chunk_end)

    del store, indices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_inference(
    datasources: Dict[str, str],
    start_date: str,
    end_date: str,
    artifact_path: str,
    *,
    require_competition_coverage: bool = True,
) -> pd.DataFrame:
    """共享分块推理实现；模型只加载一次，行情块逐块构建与释放。

    `require_competition_coverage=False` 只供每日仅抽样少量股票的快速 smoke
    检查训练→保存→重载→推理链路。容量 smoke 和比赛 `main` 必须保持默认值。
    """
    # 官方约定：训练读固定开发表；推理必须读平台注入的物理表。
    table = resolve_datasource(datasources, CFG.datasource_key, CFG.data_source)
    payload, artifact_cfg, preprocessor, model = load_artifact(artifact_path, DEVICE)
    feature_columns = list(payload["feature_columns"])
    close_column = str(payload["close_column"])

    # 用体积很小的官方股票池确定目标交易日，不预读取整段分钟行情。
    universe = load_official_universe(start_date, end_date)
    target_days = np.sort(universe["date"].dropna().unique())
    if not len(target_days):
        raise RuntimeError(f"评估区间 {start_date} 至 {end_date} 没有交易日")

    chunk_size = int(artifact_cfg.inference_chunk_trading_days)
    print(
        f"chunked inference: target_days={len(target_days):,}; "
        f"chunk_days={chunk_size}; history_calendar_days="
        f"{artifact_cfg.inference_history_calendar_days}"
    )
    chunks: List[pd.DataFrame] = []
    for chunk_number, offset in enumerate(range(0, len(target_days), chunk_size), 1):
        chunk_days = target_days[offset:offset + chunk_size]
        chunk_start = pd.Timestamp(chunk_days[0]).strftime("%Y-%m-%d")
        chunk_end = pd.Timestamp(chunk_days[-1]).strftime("%Y-%m-%d")
        print(
            f"chunk {chunk_number}/{math.ceil(len(target_days) / chunk_size)}"
        )
        chunks.append(
            _predict_inference_chunk(
                table,
                chunk_start,
                chunk_end,
                feature_columns,
                close_column,
                artifact_cfg,
                preprocessor,
                model,
                require_competition_coverage,
            )
        )

    result = pd.concat(chunks, ignore_index=True)
    result = validate_submission_frame(
        result.loc[:, ["date", "instrument", "score"]]
    )
    result = result.merge(universe, on=["date", "instrument"], how="inner")
    result = result.loc[:, ["date", "instrument", "score"]]
    result = validate_submission_frame(result)
    if require_competition_coverage:
        assert_official_coverage(result, universe)
    else:
        print(
            "quick smoke: skipped >=60% competition coverage assertion; "
            "this sampled output cannot be submitted"
        )
    return result


def main(datasources: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    """比赛评估入口：返回且只返回 date、instrument、score 三列。"""
    return run_inference(datasources, start_date, end_date, CFG.artifact_path)


# %% [markdown]
# ## 10. 运行方式
#
# 训练时取消下一格对应调用的注释并运行。正式产物为 `bigalpha_e2e_highscore_v3_3.json`。
# 公榜提交前保持训练调用为注释状态，将 `.ipynb` 与权重文件放在同一提交包中。
# 只有 `training_summary["final_training_complete"] == True` 时才可以提交。
# 已有 V3.2 正式权重时可无损升级为 V3.3，无需重新训练；升级不会覆盖 V3.2。
#
# **注意**：私榜重训的具体平台调用协议尚未出现在现有官方文本中；获得官方 Transformer 模板后，
# 只需要适配这一格的调用方式，数据管线、模型与 `train_and_save` 无需重写。
# 新版增加完整日截面模块并由两轮增至三轮，旧时间不能直接外推。
# 必须先用 capacity smoke 实测峰值显存和每轮吞吐，完整路径以 `elapsed_seconds` 为准。

# %%
# 第一步：快速功能 smoke（验证 DAI→训练→保存→重载）：
# smoke_cfg = make_smoke_config(CFG)
# smoke_summary = train_and_save(cfg=smoke_cfg, allow_cpu=True)
# smoke_summary
# smoke_preview = run_inference(
#     {CFG.datasource_key: CFG.data_source}, "2024-02-01", "2024-02-29",
#     smoke_cfg.artifact_path,
#     require_competition_coverage=False,
# )
# smoke_preview.head(), smoke_preview.shape

# 第二步：A100 80G 容量 smoke（正式80日窗口、完整1000股截面）：
# capacity_cfg = make_capacity_smoke_config(CFG)
# capacity_summary = train_and_save(cfg=capacity_cfg)
# capacity_summary
# capacity_preview = run_inference(
#     {CFG.datasource_key: CFG.data_source}, "2024-02-01", "2024-02-29",
#     capacity_cfg.artifact_path,
# )
# capacity_preview.head(), capacity_preview.shape

# 第三步（只做一次）：严格时序研发验证，不生成可提交权重。
# development_summary = develop_model(cfg=CFG)
# development_summary
# assert development_summary["ready_for_formal_training"]

# 第四步：研发门通过后，冻结 CFG，使用完整 2019–2024 从零正式训练：
# training_summary = train_and_save(cfg=CFG)
# training_summary
# assert training_summary["final_training_complete"]
# preview = main({CFG.datasource_key: CFG.data_source}, "2024-01-02", "2024-01-31")
# preview.head(), preview.shape

# 第五步（已有 V3.2 正式权重时才用）：无损升级为独立 V3.3 JSON：
# conversion_summary = convert_v3_2_artifact_to_v3_3(
#     "bigalpha_e2e_highscore_v3_2.json",
#     "bigalpha_e2e_highscore_v3_3.json",
# )
# conversion_summary
# converted_preview = main(
#     {CFG.datasource_key: CFG.data_source}, "2024-01-02", "2024-01-05"
# )
# converted_preview.head(), converted_preview.shape
