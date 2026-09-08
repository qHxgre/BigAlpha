# -*- coding: utf-8 -*-
"""BigQuant 端到端提交 —— 自包含单文件版 (训练 + 推理)。

**本文件不依赖 code/ 目录**: code/ 下用到的全部实现 (configs / dense_dataloader /
models / losses / optim / scripts.train / backtest.predictions) 已按依赖顺序内联在
下方, 用到的 yaml 配置也以字符串常量内联。提交时只需上传:

    competition_all_in_one.py                                  <- 本文件
    final.json                                                  <- 训练好的权重

两个 notebook 只需 `from competition_all_in_one import main` / `train_and_save`。

契约与 scripts/example_{train,predict}.py 一致:
    train_and_save(datasources, model_path)      -> model_path  (训练侧)
    main(datasources, start_date, end_date)      -> ['date','instrument','score']
    save_model / load_model                       同一套 {dtype, shape, data} JSON 格式

打分锚点: 每个交易日只取 slot_idx == PREDICT_SLOT (=47, 5m 面板一天 48 根, 47 是
15:00 收盘), 即以当日收盘为右端的回看窗口; 该 slot 必须在 checkpoint 训练时用过的
anchor_slots 里 (本权重是 [46,47]), 见 _check_checkpoint_matches_config。

本文件由 /tmp/build_bundle.py 从 code/ 自动生成 (机械搬运, 未改动任何算法):
剥掉包内 import、消解跨模块同名冲突 (_day_idx_bounds -> _eval_day_idx_bounds,
train.py 的 _HERE -> _TRAIN_HERE)、剥掉各模块的 __main__ 段。code/ 改动后重新
生成即可。
"""
# ==== 外部依赖 (平台环境已具备) ====
import argparse
import contextlib
import gc
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.utils.checkpoint as cp
import yaml
from bigquant import dai
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset
from tqdm import tqdm

logger = structlog.get_logger()



# ==============================================================================
# 内联配置 (原 code/configs/*.yaml 的逐字原文)
# load_config 在单文件版里先查这张表, 查不到再退回读磁盘文件
# ==============================================================================
EMBEDDED_CONFIGS = {
    'rwkv_256_softspearman_all_excluding_2024q1_1gpu.yaml': r'''
# rwkv_256 + SoftSpearman，全量训练（原训练集 + 原验证集），单卡版本。
# 基于 rwkv_256_softspearman_all_excluding_2024q1.yaml，唯一实质差异是
# train.grad_accum_steps: 8——GPU 4 物理掉出总线导致 NCCL 拓扑探测必崩，
# 多卡 torchrun 暂时用不了，改用单进程 (不触发 dist.init_process_group,
# 与 GPU 4 是否存活无关) + 梯度累积 8 步, 有效全局 batch
# (world_size(1) * batch_size(1) * grad_accum_steps(8) = 8) 与原 8 卡/
# 无累积打平；steps_per_epoch/total_steps/warmup 等调度节奏据此自动换算,
# 不需要再改 optim.noam_params.warmup。
#
# 代价: 单卡顺序算 8 次前向+反向再更新一次参数, 同一个"有效 step"的墙钟
# 时间大约是原 8 卡版本的 8 倍左右, 会明显更慢。
#
# 实际训练日期: 2019-01-02~2023-12-31 和 2024-03-01~2024-12-31
# 2024-01-01~2024-02-29 保持隔离，不进入训练。
#
# cache_mode: read 要求磁盘上已有匹配缓存 (只看 date_range/train_end/
# val_start/sample_interval/lookback_days/step/stride/keys/horizon_days/
# source/anchor_slots/feature_cols/stats 这些数据侧字段, 不受 batch_size/
# grad_accum_steps/num_workers/GPU 数量影响); 与现有缓存
# /data0/jyli/bigquant/cache/dense_arrays/b6e1da1dec2a7fa89966 的
# manifest.json 完全一致 (train_anchors=19392, val_anchors=3264)。
data:
  train_start: "2019-01-02"
  train_end: "2024-01-01"
  val_start: "2024-03-01"
  val_end: "2024-12-31"
  sample_interval: 5
  lookback_days: 2
  step: 5
  stride: 3
  horizon_days: 1
  train_keys: null
  train_keys_range: null
  source: local
  cache_dir: "/data0/jyli/bigquant/cache/dense_arrays"
  cache_mode: read

train:
  epochs: 1
  seed: 42
  batch_size: 1
  num_workers: 8
  grad_accum_steps: 8
  sub_epochs: 1
  enable_valid: false
  include_val_in_train: true
  max_steps: 1500
  val_every_steps: null

model:
  type: rwkv
  rwkv_params:
    d_model: 256
    n_layers: 2
    d_inner: 512
    dropout: 0.2
    pooling: last
    mlp_dims: [128, 64, 1]

loss:
  type: softspearman
  softspearman_params:
    regularization_strength: 1.0

optim:
  type: noam
  noam_params:
    model_size: 19
    warmup: 300
    peak_lr: 7.5e-4
    decay_power: 1.0

output:
  run_name: "rwkv_256_softspearman_all_excluding_2024q1_1gpu"
''',
    'rwkv_256_softspearman_last2_finetune.yaml': r'''
# 从 rwkv_256_softspearman 预训练 checkpoint 微调：每天只使用最后两个 5m 锚点。
# slot 46/47 = 14:55/15:00；每个样本仍保留完整的 2 日 lookback 窗口。
# 全量数据微调：不含 2024-01/02 (隔离期，既不进训练也不进验证)。

data:
  train_start: "2019-01-01"
  train_end: "2024-01-01"
  val_start: "2024-03-01"
  # 不包含 2024-12-31，避免末日 forward label 依赖后续交易日数据。
  val_end: "2024-12-30 23:59:59"
  sample_interval: 5
  lookback_days: 2
  step: 5
  stride: 48
  horizon_days: 1
  train_keys: null
  train_keys_range: null
  source: local
  cache_dir: "/data0/jyli/bigquant/cache/dense_arrays"
  # 该签名 (val_start=2024-03-01) 的缓存已构建完成 (READY)，直接复用。
  cache_mode: read

train:
  epochs: 2
  seed: 42
  batch_size: 1
  num_workers: 0
  sub_epochs: 1
  # include_val_in_train=true 时验证集被并入训练、val_anchors 清空，
  # enable_valid 失去意义，故一并关闭。
  enable_valid: false
  include_val_in_train: true
  max_steps: 5000
  val_every_steps: null

model:
  type: rwkv
  rwkv_params:
    d_model: 256
    n_layers: 2
    d_inner: 512
    dropout: 0.2
    pooling: last
    mlp_dims: [128, 64, 1]

loss:
  type: softspearman
  softspearman_params:
    regularization_strength: 1.0

# optim.type=adam 在 plain_adam.py 中使用 torch.optim.AdamW，学习率固定。
optim:
  type: adam
  adam_params:
    lr: 1.0e-4

finetune:
  checkpoint_path: "/data0/jyli/bigquant/models/rwkv_256_softspearman_all_excluding_2024q1_1gpu_20260805-221950/step1210.pt"
  anchor_slots: [46, 47]
  strict_checkpoint: true

output:
  run_name: "rwkv_256_softspearman_last2_finetune"
''',
    'rwkv_256_softspearman.yaml': r'''
# rwkv_256 + SoftSpearman
# SoftSpearman 是截面排序损失；必须 batch_size=1，使每次 loss 只在一个 anchor
# 的股票截面上排序。模型输入仍是完整 lookback_days 窗口。
data:
  train_start: "2019-01-02"
  train_end: "2024-01-01"
  val_start: "2024-03-01"
  val_end: "2024-12-31"
  sample_interval: 5
  lookback_days: 2
  step: 5
  stride: 3
  horizon_days: 1
  train_keys: null
  train_keys_range: null
  source: local
  cache_dir: "/data0/jyli/bigquant/cache/dense_arrays"
  cache_mode: readwrite

train:
  epochs: 1
  seed: 42
  batch_size: 1
  num_workers: 0
  sub_epochs: 1
  enable_valid: true
  include_val_in_train: false
  max_steps: null
  val_every_steps: null

model:
  type: rwkv
  rwkv_params:
    d_model: 256
    n_layers: 2
    d_inner: 512
    dropout: 0.2
    pooling: last
    mlp_dims: [128, 64, 1]

loss:
  type: softspearman
  softspearman_params:
    regularization_strength: 1.0

optim:
  type: noam
  noam_params:
    model_size: 19
    warmup: 400
    peak_lr: 7.5e-4
    decay_power: 1.0

output:
  run_name: "rwkv_256_softspearman"
''',
}


# ==============================================================================
# configs/configs.py  —  配置: yaml -> 结构化 Config
# ==============================================================================

# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _HERE / "default.yaml"


@dataclass
class DataConfig:
    train_start: str
    train_end: str
    val_end: str
    sample_interval: int
    lookback_days: int
    step: int
    horizon_days: float
    stride: Optional[int] = None
    train_keys: Optional[List[int]] = None
    # 验证集起始日期 (含); None 时等于 train_end (训练/验证紧邻切分)。晚于
    # train_end 时, [train_end, val_start) 之间的数据会被弃用 (既不进训练也不
    # 进验证), 见 dense_dataloader.dataloader.get_dataloaders 的同名参数。
    val_start: Optional[str] = None
    # 数据源: "local" (默认) 读本地 aligned parquet; "online" 现场 dai.query
    # 线上表 (bigalpha_2026_stock_bar5m/bar1m/bigalpha_2026_instruments,
    # 见 code/dense_dataloader/online_source.py)。train_keys 语义不变, 只是
    # source="online" 时里面装的是真实字符串股票代码, 不是本地匿名 instrument_id
    # (两者不可互相映射)。
    source: str = "local"
    # 最终 X/label/mask/anchors 的持久化磁盘缓存。local 推荐使用，online 默认
    # 不启用（远程数据可能变化，除非显式指定）。模式：off/read/write/readwrite。
    cache_dir: Optional[str] = None
    cache_mode: str = "off"


@dataclass
class TrainConfig:
    epochs: int
    seed: int
    batch_size: int
    # dense_dataloader.get_dataloaders/build_panel_arrays 的 num_workers 透传
    # (DataLoader 的 worker 进程数)
    num_workers: int = 0
    # 每次参数更新聚合的 micro-batch 数。DDP 下有效全局 batch =
    # world_size * batch_size * grad_accum_steps。
    grad_accum_steps: int = 1
    # 分布式训练 (torchrun) 的 NCCL/gloo 进程组超时, 单位分钟。默认的 10 分钟
    # 对 DDP 场景来说太短——rank0 独自构建全历史全股票面板可能要几十分钟甚至更久
    # (取决于机器负载), 其余 rank 会一直卡在 rank0 发布共享内存之后的那次 broadcast
    # 上等它, 超时太短会被 watchdog 误判成"卡死"直接杀掉整个进程组。单进程模式下
    # 这个字段不生效。
    ddp_timeout_minutes: int = 180
    # 训练循环划成多少个 sub-epoch (驱动 OptimStrategy 的 is_sub_e/sub_e_passed,
    # WarmupCosineStrategy 的两段式调度需要它; NoamStrategy 忽略这两个参数)
    sub_epochs: int = 1
    enable_valid: bool = True
    # 总训练步数上限; None 时用 epochs * len(train_loader) (即跑满 epochs 整个数据集)
    max_steps: Optional[int] = None
    # 每多少步跑一次验证集; None 时退化成每个 sub-epoch 边界验证一次
    val_every_steps: Optional[int] = None
    # 将原验证锚点并入训练集；适用于最终全量训练，验证集保持为空。
    include_val_in_train: bool = False


@dataclass
class ModelConfig:
    type: str
    # 对应 models/__init__.py 约定的 model.<type>_params: 只解析出当前 type 那一份
    params: dict = field(default_factory=dict)


@dataclass
class LossConfig:
    # LOSS_REGISTRY (code/losses) 的 key; 不配置 loss 段时默认 mse
    type: str = "mse"
    params: dict = field(default_factory=dict)


@dataclass
class OptimConfig:
    # OPTIM_REGISTRY (code/optim) 的 key; 不配置 optim 段时默认 adam (固定学习率)
    type: str = "adam"
    params: dict = field(default_factory=dict)


@dataclass
class OutputConfig:
    # checkpoint 目录名前缀; None 时调用方 (train.py) 用 config 文件名 (不含扩展名)
    # 兜底。checkpoint 根目录固定是 /data0/jyli/bigquant/models (不在 yaml 里配),
    # 每次训练会在其下新建一个带时间戳的子目录 <run_name>_<timestamp>, 保证不同
    # 次训练的产物不会互相覆盖, 见 scripts/train.py。
    run_name: Optional[str] = None


@dataclass
class FineTuneConfig:
    checkpoint_path: str
    anchor_slots: List[int] = field(default_factory=lambda: [45, 46, 47])
    strict_checkpoint: bool = True


@dataclass
class Config:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    loss: LossConfig
    optim: OptimConfig
    output: OutputConfig
    finetune: Optional[FineTuneConfig] = None


def _deep_merge(base: dict, override: dict) -> dict:
    """override 逐 key 覆盖 base: 两边都是 dict 的 key 递归合并, 否则整体替换
    (列表等非 dict 值不做元素级合并, 直接用 override 的值)。"""
    merged = dict(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_raw(path: Path, _seen: Optional[set] = None) -> dict:
    path = path.resolve()
    _seen = set() if _seen is None else _seen
    if path in _seen:
        raise ValueError(f"配置文件 extends 出现循环引用: {path}")
    _seen.add(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.get("extends")
    if extends:
        base_path = (path.parent / extends).resolve()
        base_raw = _load_raw(base_path, _seen)
        raw = _deep_merge(base_raw, raw)
    return raw


def _resolve_path(path) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidate = _HERE / p
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"找不到配置文件: {path} (尝试过 {p} 和 {candidate})")


def _expand_train_keys(data_raw: Dict[str, Any]) -> Optional[List[int]]:
    """train_keys 支持两种写法: 显式列表 `train_keys: [1,2,3]`, 或区间
    `train_keys_range: [lo, hi]` (对应 python 的 range(lo, hi)); 都不写 (或都为
    null) 时返回 None, dense_dataloader 会用全历史全部股票 (~2074 只)。"""
    explicit = data_raw.get("train_keys")
    if explicit is not None:
        return list(explicit)
    rng = data_raw.get("train_keys_range")
    if rng is not None:
        lo, hi = rng
        return list(range(lo, hi))
    return None


def load_config(path=DEFAULT_CONFIG_PATH) -> Config:
    """读取 yaml 配置文件 (支持 extends 继承), 返回结构化的 Config。

    path 可以是绝对路径、相对当前工作目录的路径, 或相对 configs/ 目录的文件名
    (如 "rwkv.yaml")——三种都会尝试解析。
    """
    return _build_config(_load_raw(_resolve_path(path)))


def _build_config(raw: dict) -> Config:
    """raw dict -> Config。单文件版把原 load_config 切成两半的后半段, 使内联
    yaml (EMBEDDED_CONFIGS) 能走同一段解析逻辑, 不必先落盘再读。解析代码与
    原 configs.py 逐字一致。"""
    data_raw = dict(raw["data"])
    data_cfg = DataConfig(
        train_start=data_raw["train_start"],
        train_end=data_raw["train_end"],
        val_end=data_raw["val_end"],
        sample_interval=data_raw["sample_interval"],
        lookback_days=data_raw["lookback_days"],
        step=data_raw["step"],
        stride=data_raw.get("stride"),
        horizon_days=data_raw["horizon_days"],
        train_keys=_expand_train_keys(data_raw),
        val_start=data_raw.get("val_start"),
        source=data_raw.get("source", "local"),
        cache_dir=data_raw.get("cache_dir"),
        cache_mode=data_raw.get("cache_mode", "off"),
    )
    train_cfg = TrainConfig(**raw["train"])

    model_raw = raw["model"]
    model_type = model_raw["type"]
    model_cfg = ModelConfig(type=model_type, params=model_raw.get(f"{model_type}_params", {}))

    loss_raw = raw.get("loss", {})
    loss_type = loss_raw.get("type", "mse")
    loss_cfg = LossConfig(type=loss_type, params=loss_raw.get(f"{loss_type}_params", {}))

    optim_raw = raw.get("optim", {})
    optim_type = optim_raw.get("type", "adam")
    optim_cfg = OptimConfig(type=optim_type, params=optim_raw.get(f"{optim_type}_params", {}))

    output_cfg = OutputConfig(**raw.get("output", {}))
    finetune_raw = raw.get("finetune")
    finetune_cfg = FineTuneConfig(**finetune_raw) if finetune_raw else None

    return Config(
        data=data_cfg, train=train_cfg, model=model_cfg,
        loss=loss_cfg, optim=optim_cfg, output=output_cfg,
        finetune=finetune_cfg,
    )


# ==============================================================================
# dense_dataloader/config.py  —  数据: 字段/路径常量
# ==============================================================================

# -*- coding: utf-8 -*-
from pathlib import Path

# 仓库根目录: code/dense_dataloader/config.py -> parents[2] = /data0/jyli/bigquant
REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_DIR = REPO_ROOT / "download_data" / "local_data" / "aligned_data"
# 仅 labels.py 的 VWAP 前瞻窗口计算需要 bar1m；面板/日历已改为直接读 bar5m。
BAR1M_GLOB = str(ALIGNED_DIR / "bigalpha_2026_e2e_bar1m" / "*.parquet")
BAR5M_GLOB = str(ALIGNED_DIR / "bigalpha_2026_e2e_bar5m" / "*.parquet")

BARS_PER_TRADING_DAY = 240  # 每个交易日的 1 分钟 bar 数 (09:31-11:30, 13:01-15:00)

# 价格类列: 日内前向填充 (停牌期间价格视为延续不变，但不跨天延续)
PRICE_COLS = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
# 量类列: 缺失时填 0 (停牌/无成交时真实成交量就是 0，不做前向填充)
VOL_COLS = [
    "volume", "amount", "deal_number",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
]
FEATURE_COLS = PRICE_COLS + VOL_COLS
# 送入模型的特征维度: 不再 +1 -- 缺失现在是独立的 X_mask 输出，不拼进特征维
N_FEAT = len(FEATURE_COLS)

RAW_COLS = ["date", "key", "adjust_factor"] + FEATURE_COLS


# ==============================================================================
# dense_dataloader/canonical.py  —  数据: 原始列 -> 规范列
# ==============================================================================

# -*- coding: utf-8 -*-
import polars as pl

# OHLC: 本地原始表用字面 -1 表示缺失 (÷100 后会变成 -0.01 元这种"看起来像
# 真实价格"的假值, 必须先转 null 再缩放, 否则会被当成真实价格污染后续计算)。
# 线上表缺失已经是 NaN (alignment.md §1 记录的官方口径), 不需要这步。
_OHLC_COLS = ["open", "high", "low", "close"]
_BOOK_PRICE_COLS = [f"{s}_price{i}" for s in ("ask", "bid") for i in (1, 2, 3)]
_BOOK_VOLUME_COLS = [f"{s}_volume{i}" for s in ("ask", "bid") for i in (1, 2, 3)]
_BOOK_ORDER_COLS = [f"{s}_num_orders{i}" for s in ("ask", "bid") for i in (1, 2, 3)]

# 本地这些列的原始单位是"分"(int), 元 = 分/100; amount 同理。
_LOCAL_SCALE_COLS = _OHLC_COLS + _BOOK_PRICE_COLS + ["amount"]

# 线上 canonical 输出保留的列: 只取前 3 档盘口 (线上原始表有 5 档, 这里的
# select 天然完成"云端 5 档 -> 3 档"的截断, 见 alignment.md §1/§4)。
# adjust_factor/instrument_id 用 `if c in df.columns` 兜底 (alignment.md §6
# 未知项①: 线上是否含 adjust_factor 列未经真实验证), 缺列时由调用方
# (panel.py/labels.py) 决定兜底策略, 这里不代为假设。
_ONLINE_KEEP = (
    ["date", "key", "instrument_id", "adjust_factor"]
    + _OHLC_COLS
    + ["volume", "amount", "deal_number"]
    + _BOOK_PRICE_COLS
    + _BOOK_VOLUME_COLS
    + _BOOK_ORDER_COLS
)


def to_canonical(df: pl.DataFrame, is_local: bool) -> pl.DataFrame:
    """把本地 (e2e) 或线上 (stock) 的原始行情行统一到 canonical 表示。

    输入:
        df: pl.DataFrame。
            is_local=True 时, 传 raw_data 分片原样读出的表 (列即
            download_data/data.md 记录的本地 e2e_bar schema, 见
            download_data/align_data.py)。
            is_local=False 时, 传 dai.query 查
            bigalpha_2026_stock_bar{1m,5m,15m,30m} 的结果转成的
            pl.DataFrame (见 online_source.py)。
        is_local: True 走本地整数编码 -> canonical 的转换 (÷100 还原成元,
            OHLC 的 -1 转 null); False 假定线上数据已经是 canonical 单位/
            缺失约定 (元/NaN, 见 alignment.md §1), 只做列名统一和盘口截断,
            不做二次判断。
    返回:
        pl.DataFrame。
            is_local=True 时保留原表的全部列 (只是价格/金额/key 被转换),
            列顺序为 [date, key, instrument_id, adjust_factor, ...其余原始
            列不变...]——与 download_data/align_data.py 原有输出逐值一致,
            用于回归测试。
            is_local=False 时列固定为 _ONLINE_KEEP 中实际存在的那些
            (adjust_factor/instrument_id 可能因线上未验证而缺失)。
    """
    if is_local:
        exprs = [
            pl.when(pl.col(c) == -1).then(None).otherwise(pl.col(c) / 100.0).cast(pl.Float64).alias(c)
            for c in _OHLC_COLS
        ] + [
            (pl.col(c) / 100.0).cast(pl.Float64).alias(c)
            for c in _LOCAL_SCALE_COLS
            if c not in _OHLC_COLS
        ] + [
            pl.col("instrument_id").alias("key"),
        ]
        df = df.with_columns(exprs)
        front = ["date", "key", "instrument_id", "adjust_factor"]
        front = [c for c in front if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        return df.select(front + rest)

    # 线上: 假定已经是 canonical 单位/缺失约定 (元、NaN), 只需要统一分组键 +
    # 显式 select 到前 3 档盘口对应的列 (见模块顶部 _ONLINE_KEEP 说明)。
    df = df.with_columns(pl.col("instrument").alias("key"))
    keep = [c for c in _ONLINE_KEEP if c in df.columns]
    return df.select(keep)


# ==============================================================================
# dense_dataloader/online_source.py  —  数据: 线上 dai.query 数据源
# ==============================================================================

# -*- coding: utf-8 -*-
from typing import Optional, Sequence, Tuple

import polars as pl
import structlog
from bigquant import dai


logger = structlog.get_logger()

ONLINE_BAR5M_TABLE = "bigalpha_2026_stock_bar5m"
ONLINE_BAR1M_TABLE = "bigalpha_2026_stock_bar1m"
ONLINE_INSTRUMENTS_TABLE = "bigalpha_2026_instruments"

# 实际使用的表名，可被 configure_tables() 覆盖 (对齐 scripts/transformer_train.py/
# example_predict.py 一直沿用的约定: 平台注入的 datasources 字典里存的就是真实
# 表名，不应该在本模块里写死忽略掉)。默认值就是上面三个模块常量。
_TABLES = {
    "bar5m": ONLINE_BAR5M_TABLE,
    "bar1m": ONLINE_BAR1M_TABLE,
    "instruments": ONLINE_INSTRUMENTS_TABLE,
}

# resolve_full_universe_online/build_global_calendar_online 用: 必须扫全历史
# (不能从某次训练窗口反推股票集合，见 panel.resolve_full_universe 的同一条
# 设计原则)，上界留够余量覆盖到查询当天。
FULL_HISTORY_RANGE = ("2019-01-01 00:00:00", "2030-01-01 00:00:00")

# 显式列投影: 线上表有 43 列 (已用 dai.get_datasource_schema 核实)，下游只用
# 其中一部分，`SELECT *` 会把 pre_close / 4-5 档盘口 / __PARTITION__ 一路拉进
# 内存再丢掉——全历史全股票量级下这是几十 GiB 级的无谓开销。local 分支靠
# `pl.scan_parquet(...).select(RAW_COLS)` 让 polars 把列裁剪下推进 parquet
# reader，天然没这个问题；online 只能在 SQL 里显式写列名来对齐这个行为。
#
# bar5m: 与 canonical._ONLINE_KEEP 逐项对应 (那里的 `key` 是 to_canonical 内部
# 由 instrument 派生的，不是表里的列)。刻意保留 num_orders1-3 而不砍到
# panel.RAW_COLS 那 22 列: 线上 canonical 输出要和本地 aligned parquet 保持同一
# 套 schema，alignment.md §6 的"同 key 同 date 逐值吻合"复核才做得下去。
_BAR5M_COLS = (
    ["date", "instrument", "instrument_id", "adjust_factor"]
    + ["open", "high", "low", "close"]
    + ["volume", "amount", "deal_number"]
    + [f"{side}_price{i}" for side in ("ask", "bid") for i in (1, 2, 3)]
    + [f"{side}_volume{i}" for side in ("ask", "bid") for i in (1, 2, 3)]
    + [f"{side}_num_orders{i}" for side in ("ask", "bid") for i in (1, 2, 3)]
)

# bar1m: 只服务 labels._compute_vwap_adj 的前瞻 VWAP 聚合，那边 select 的就是
# 这 6 列 (date/key/close/volume/amount/adjust_factor，key 由 instrument 派生)。
# bar1m 的行数是 bar5m 的 5 倍，这条路的列裁剪收益比 bar5m 更大。
_BAR1M_COLS = ["date", "instrument", "close", "volume", "amount", "adjust_factor"]


def configure_tables(datasources: dict) -> None:
    """用平台注入的 datasources 覆盖默认表名 (key: "bar5m"/"bar1m"/"instruments")。

    不含的 key 保留原值 (默认是 ONLINE_BAR5M_TABLE 等模块常量)。train_and_save/
    main 入口应在构建面板之前调用一次，见 scripts/transformer_train.py。
    """
    for k in ("bar5m", "bar1m", "instruments"):
        if k in datasources:
            _TABLES[k] = datasources[k]


def get_table(name: str) -> str:
    """读取当前生效的表名 ("bar5m"/"bar1m"/"instruments")，可能已被
    configure_tables() 覆盖过，供调用方自己拼 SQL 时使用 (如
    scripts/example_predict.py 里对齐 CSI1000 成分池那一步)。"""
    return _TABLES[name]


def _query_bar(
    table: str,
    date_range: Tuple,
    keys: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """查线上 bar 表, 返回 pl.DataFrame。

    columns: 显式列投影 (见 _BAR5M_COLS/_BAR1M_COLS)。None 时退回 `SELECT *`。
        filters 用到的 date/instrument 都在两个列集内, 投影不影响谓词下推。

    结果用 QueryResult.pl() 直接拿 polars, 不走 .df() + pl.from_pandas ——
    后者会先整份物化成 pandas 再复制一份 polars, 峰值是必要内存的两倍。
    """
    lo, hi = date_range
    filters = {"date": [str(lo), str(hi)]}
    if keys is not None:
        filters["instrument"] = list(keys)
    select = "*" if columns is None else ", ".join(columns)
    result = dai.query(f"SELECT {select} FROM {table}", filters=filters)
    try:
        return result.pl()
    except (AttributeError, NotImplementedError) as exc:
        # 平台 SDK 版本差异时的兜底: 语义完全一致, 只是多一份 pandas 中间副本
        logger.warning(
            "QueryResult.pl() 不可用, 退回 .df() + pl.from_pandas (内存峰值更高)",
            table=table, error=repr(exc),
        )
        return pl.from_pandas(result.df())


def fetch_bar5m_canonical(date_range: Tuple, keys: Optional[Sequence[str]] = None) -> pl.DataFrame:
    """线上 bar5m -> canonical DataFrame，供 panel.py 的 online 分支使用。"""
    raw = _query_bar(_TABLES["bar5m"], date_range, keys, columns=_BAR5M_COLS)
    return to_canonical(raw, is_local=False)


def fetch_bar1m_canonical(date_range: Tuple, keys: Optional[Sequence[str]] = None) -> pl.DataFrame:
    """线上 bar1m -> canonical DataFrame，供 labels.py 的 online 分支算前瞻 VWAP 用。

    只取 VWAP 聚合需要的 6 列 (_BAR1M_COLS)，因此返回值比 bar5m 的 canonical
    结果窄——to_canonical 的 `keep = [c for c in _ONLINE_KEEP if c in df.columns]`
    本身就是"缺列则跳过"，下游 labels._compute_vwap_adj 也只 select 这几列。
    """
    raw = _query_bar(_TABLES["bar1m"], date_range, keys, columns=_BAR1M_COLS)
    return to_canonical(raw, is_local=False)


def resolve_full_universe_online(keys: Optional[Sequence[str]] = None) -> list:
    """全历史 (bigalpha_2026_instruments 全表) 出现过的 distinct instrument，
    独立于任何 date_range——和 panel.resolve_full_universe (本地版本) 同一个
    设计原则: 不能从某次查询窗口反推股票集合，否则会漏掉不在这个区间内的
    股票。

    返回真实字符串股票代码 (如 "600000.SH")，与本地匿名 instrument_id 无
    对应关系，不能跨表混用，只能各自在自己的数据源内部分组/排序。

    输入:
        keys: 可选，若给定则与全历史股票集合取交集 (用于小规模调试)。
    返回:
        list[str]，升序排列、去重后的股票代码列表。
    """
    df = dai.query(
        f"SELECT DISTINCT instrument FROM {_TABLES['instruments']}",
        filters={"date": list(FULL_HISTORY_RANGE)},
    ).df()
    universe = sorted(df["instrument"].dropna().unique().tolist())
    if keys is not None:
        wanted = set(keys)
        universe = [k for k in universe if k in wanted]
    return universe


def build_global_calendar_online() -> pl.DataFrame:
    """全历史交易日历: bigalpha_2026_instruments 的 date 去重转 day。

    比全历史扫 bar5m 更省流量 (instruments 本身就是逐交易日的成分表，不像
    bar5m 那样每天有上千只股票 x 48 根 bar)。

    返回:
        pl.DataFrame，一列 day (pl.Date)，按升序排列——schema 与本地
        calendar.build_global_calendar() 的输出一致。
    """
    df = dai.query(
        f"SELECT DISTINCT date FROM {_TABLES['instruments']}",
        filters={"date": list(FULL_HISTORY_RANGE)},
    ).df()
    days = pl.from_pandas(df).select(pl.col("date").dt.date().alias("day")).unique().sort("day")
    return days


# ==============================================================================
# dense_dataloader/calendar.py  —  数据: 全局交易日历
# ==============================================================================

# -*- coding: utf-8 -*-
import polars as pl



def build_global_calendar(source: str = "local") -> pl.DataFrame:
    """全历史出现过的所有交易日，去重、升序。

    输入:
        source: "local" (默认) 扫本地 bar5m parquet; "online" 改用
            online_source.build_global_calendar_online() (查
            bigalpha_2026_instruments)，见该函数 docstring。
    返回:
        pl.DataFrame，一列 day (pl.Date)，按升序排列。
    """
    if source == "online":
        pass  # (原为包内 import, 单文件版无需)
        return build_global_calendar_online()

    lf = pl.scan_parquet(BAR5M_GLOB)
    days = lf.select(pl.col("date").dt.date().alias("day")).unique().sort("day").collect()
    return days


# ==============================================================================
# dense_dataloader/panel.py  —  数据: 稠密面板 (N,T,F)
# ==============================================================================

# -*- coding: utf-8 -*-
import gc
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
import structlog


logger = structlog.get_logger()


@dataclass
class DensePanel:
    """规整面板，由 build_dense_panel() 产出，未做任何缺失值填充。

    字段:
        keys: 长度 N 的升序股票 id 列表 (全历史出现过的股票，或与 keys 参数
            的交集)，行下标 i 对应 keys[i]。
        time_grid: 长度 T 的时间网格 (pl.DataFrame)，列为
            date/day_idx/slot_idx/t_idx，按 t_idx 升序排列，
            t_idx = day_idx * bars_per_day + slot_idx。
        bars_per_day: 每个交易日的时间片数 (T = 天数 * bars_per_day)。
        values: float32 (N, T, len(FEATURE_COLS))，原始 (未前向填充) 特征。
            PRICE_COLS 在 is_missing 处已被置为 NaN；VOL_COLS 是 bar5m 表
            原生值 (窗口尾快照，不是真实 5 分钟聚合量，不做置空处理)。
        is_missing: bool (N, T)，该 (key, t) 是否缺失
            (原始 close 为 NaN 或 <=0)，与是否做过填充无关。
        adj_close: float32 (N, T)，(已置空的) close * adjust_factor，
            专供 label 计算使用，在 is_missing 处天然是 NaN。
    """
    keys: list
    time_grid: pl.DataFrame
    bars_per_day: int
    values: np.ndarray
    is_missing: np.ndarray
    adj_close: np.ndarray


def resolve_full_universe(keys: Optional[Sequence] = None, source: str = "local") -> list:
    """全历史出现过的 distinct key 列表，独立于任何 date_range。

    这是"股票轴恒定为全历史"这一设计要求的关键: 不能从某次按 date_range
    过滤后的查询结果反推股票集合 (那样会漏掉不在这个区间内交易的股票，
    比如只查 2024-06 一个月只能反推出约 998 只，而不是全历史 2074 只)。

    输入:
        keys: 可选，若给定则与全历史股票集合取交集 (用于小规模调试)，
            不是替代全历史集合。
        source: "local" (默认) 扫本地 bar5m parquet，key 是匿名 int
            instrument_id; "online" 改用
            online_source.resolve_full_universe_online() (查
            bigalpha_2026_instruments)，key 是真实字符串股票代码，
            与本地 key 不可互相映射。
    返回:
        list，升序排列、去重后的股票 key 列表；source="local" 且 keys=None
        时长度为全历史 distinct key 数 (已验证 2019-01-02~2024-12-30 为
        2074，范围 6~5493)。
    """
    if source == "online":
        pass  # (原为包内 import, 单文件版无需)
        return resolve_full_universe_online(keys)

    universe = (
        pl.scan_parquet(BAR5M_GLOB).select(pl.col("key").unique().sort())
        .collect()["key"].to_list()
    )
    if keys is not None:
        wanted = set(keys)
        universe = [k for k in universe if k in wanted]
    return universe


def year_chunks(date_range: tuple) -> list:
    """把 [lo, hi] 闭区间按自然年切成若干闭区间子段 (升序, 不重不漏)。

    供 build_dense_panel / labels._compute_vwap_adj 分块取数用: 全区间一次性
    物化长表 (bar1m 全历史量级 5 亿行以上) 会和最终的 (N,T,F) 面板同时存活,
    是内存峰值的主要来源; 分块后每块散射进最终数组即可释放。

    按自然年 (而不是固定天数) 切是为了让分块边界稳定可复现, 与数据量无关。
    单个块内的行为与整体一致: 面板散射按 (key_idx, t_idx) 定位, 与拉取顺序
    无关; bar1m 的 5 分钟聚合窗口锚在绝对时间网格上, 不随每帧起点漂移
    (已验证), 所以分块聚合与整体聚合逐值等价。

    输入:
        date_range: (lo, hi)，可以是 str / datetime / pl 时间标量。
    返回:
        list[(lo_i, hi_i)]，闭区间，首段起点 == lo、末段终点 == hi。
    """
    # pd.Timestamp 而非 pl.Series(...).cast(): 后者不接受 str -> datetime。
    # dataloader.build_panel_arrays 处理同样的入参用的也是 pd.Timestamp, 保持一致。
    lo_ts, hi_ts = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    if hi_ts < lo_ts:
        return []
    out = []
    for year in range(lo_ts.year, hi_ts.year + 1):
        seg_lo = lo_ts if year == lo_ts.year else pd.Timestamp(datetime(year, 1, 1))
        seg_hi = hi_ts if year == hi_ts.year else pd.Timestamp(datetime(year, 12, 31, 23, 59, 59, 999999))
        out.append((seg_lo.to_pydatetime(), seg_hi.to_pydatetime()))
    return out


def _ensure_adjust_factor(df: pl.DataFrame, ctx: str) -> pl.DataFrame:
    """线上表若缺 adjust_factor 列 (alignment.md §6 未知项①未经真实验证)，
    整列兜底填 1.0 (视为不复权) 并告警，不让 KeyError 直接炸穿管线。"""
    if "adjust_factor" not in df.columns:
        logger.warning(
            f"{ctx}: 线上查询结果没有 adjust_factor 列，兜底填 1.0 (视为不复权)，"
            "见 alignment.md §6 未知项①，提交前需在有权限的环境复核",
        )
        df = df.with_columns(pl.lit(1.0).alias("adjust_factor"))
    return df


def _load_bar5m(date_range: tuple, keys: Optional[Sequence] = None, source: str = "local") -> pl.DataFrame:
    """读 bar5m 数据 (本地直接读 parquet 原生表，线上现场查 dai)。

    输入:
        date_range: (start, end) 二元组，按 date 列过滤 (闭区间)。
        keys: 可选，只保留这些股票；为 None 时不过滤。
        source: "local" (默认) 或 "online"，见 build_dense_panel。
    返回:
        pl.DataFrame，列为 config.RAW_COLS (date/key/adjust_factor/
        FEATURE_COLS)。VOL_COLS 是 bar5m 表原生值，本地已知是窗口尾快照
        (见模块顶部说明)；线上按用户决定直接信任其为真实窗口聚合，不再
        现场从 bar1m 聚合。
    """
    if source == "online":
        pass  # (原为包内 import, 单文件版无需)
        df = fetch_bar5m_canonical(date_range, keys)
        df = _ensure_adjust_factor(df, "panel._load_bar5m")
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"线上 bar5m canonical 结果缺少必需列 {missing}，无法构建面板")
        return df.select(RAW_COLS)

    lo, hi = date_range
    lf = pl.scan_parquet(BAR5M_GLOB).filter(pl.col("date").is_between(pl.lit(lo), pl.lit(hi)))
    if keys is not None:
        lf = lf.filter(pl.col("key").is_in(list(keys)))
    return lf.select(RAW_COLS).collect()


def build_time_grid(date_range: tuple, sample_interval: int, source: str = "local") -> pl.DataFrame:
    """构造 T 长度的规整时间网格: 全局交易日历 (来自 bar5m 或线上 instruments) x 日内时间片。

    日内时间片按 sample_interval 分钟的规律，从上午 09:30 和下午 13:00
    开盘时刻起，每隔 sample_interval 分钟一个右闭标签，分别到 11:30 / 15:00
    收盘为止 (与 download_data/data.md §2 记录的时间戳约定一致)；时间片数
    恒为 240 / sample_interval，不依赖某次查询实际有没有观测到每个时间片。

    输入:
        date_range: (start, end) 二元组，用于从全局交易日历里筛出要覆盖
            的交易日 (闭区间)。
        sample_interval: int，分钟，必须整除 240。
        source: "local" (默认) 或 "online"，透传给 build_global_calendar。
    返回:
        pl.DataFrame，按 t_idx 升序排列，列: date/day_idx/slot_idx/t_idx。
    """
    assert 240 % sample_interval == 0, f"sample_interval={sample_interval} 必须整除 240"
    lo, hi = date_range
    calendar = build_global_calendar(source=source)
    days = (
        calendar.filter(pl.col("day").is_between(pl.lit(lo).cast(pl.Date), pl.lit(hi).cast(pl.Date)))
        .select("day")
        .sort("day")
        .with_row_index("day_idx")
    )

    morning = pl.datetime_range(
        pl.datetime(2000, 1, 1, 9, 30) + pl.duration(minutes=sample_interval),
        pl.datetime(2000, 1, 1, 11, 30),
        interval=f"{sample_interval}m",
        eager=True,
    )
    afternoon = pl.datetime_range(
        pl.datetime(2000, 1, 1, 13, 0) + pl.duration(minutes=sample_interval),
        pl.datetime(2000, 1, 1, 15, 0),
        interval=f"{sample_interval}m",
        eager=True,
    )
    times = pl.concat([morning, afternoon]).dt.time()
    slots = pl.DataFrame({"time": times}).sort("time").with_row_index("slot_idx")
    assert slots.height == 240 // sample_interval, (
        f"日内时间片数 {slots.height} != 240/sample_interval={240 // sample_interval}"
    )

    grid = days.join(slots, how="cross").with_columns(
        pl.col("day").dt.combine(pl.col("time")).cast(pl.Datetime("ns")).alias("date"),
        (pl.col("day_idx") * slots.height + pl.col("slot_idx")).alias("t_idx"),
    )
    return grid.select("date", "day_idx", "slot_idx", "t_idx").sort("t_idx")


def build_dense_panel(
    sample_interval: int, date_range: tuple, keys: Optional[Sequence] = None, source: str = "local",
) -> DensePanel:
    """顶层入口: 重采样 -> 全历史股票 + 全局时间网格 -> reindex 成 (N,T,F)。

    不做任何缺失值填充 (前向填充/填 0 一律交给 fill_within_day，且必须在
    label 算完之后才能调用)，但会把 PRICE_COLS 在 is_missing 处置成 NaN
    (见模块 docstring)，保证返回值可以直接、安全地喂给
    labels.compute_forward_return_label 和 fill_within_day。

    输入:
        sample_interval: int，必须为 5 (面板现在直接读 bar5m 原生表/线上
            bar5m 表，不再现场重采样，因此不支持其他频率)。
        date_range: (start, end) 二元组，覆盖的日期区间 (闭区间)；如果需要
            用到区间末尾的 label，调用方应自行把 date_range 的结束时间
            往后扩一段 (本函数不负责加缓冲区)。
        keys: 可选，只使用这些股票 (与全历史股票集合取交集)；为 None 时
            使用全历史全部股票 (本地约 2074 只；线上取决于
            bigalpha_2026_instruments 的全历史成分数)。
        source: "local" (默认) 读本地 aligned parquet；"online" 现场
            dai.query bigalpha_2026_stock_bar5m + bigalpha_2026_instruments
            (见 online_source.py)，两者算法完全一致，只有取数这一步不同。
    返回:
        DensePanel，见该类的字段说明。
    """
    assert sample_interval == 5, (
        f"sample_interval={sample_interval}: 面板直接读 bar5m 原生表，只支持 5"
    )
    universe = resolve_full_universe(keys, source=source)
    key_to_idx = {k: i for i, k in enumerate(universe)}
    n = len(universe)

    time_grid = build_time_grid(date_range, sample_interval, source=source)
    bars_per_day = time_grid.filter(pl.col("day_idx") == 0).height
    t = time_grid.height

    values = np.full((n, t, len(FEATURE_COLS)), np.nan, dtype=np.float32)
    adjust_factor = np.full((n, t), np.nan, dtype=np.float32)
    grid_lookup = time_grid.select("date", "t_idx")

    # 按自然年分块取数 + 增量散射: 全区间一次性物化长表会和上面两个大数组
    # 同时存活 (全历史量级下长表本身就有几十 GiB), 是内存峰值的主要来源。
    # 散射按 (key_idx, t_idx) 绝对定位, 与拉取顺序/分块方式无关, 因此分块与
    # 整体逐值等价; 每块处理完就释放, 峰值只由单块大小决定。
    for chunk_lo, chunk_hi in year_chunks(date_range):
        df = _load_bar5m((chunk_lo, chunk_hi), keys, source=source)
        # 股票轴和行数据可能来自两张不同的表: source="online" 时 universe 来自
        # bigalpha_2026_instruments (中证1000 成分, 约 2216 只), 而 bar5m 表是全部
        # A 股 (5000+)，两者不是同一集合，不过滤会让下面的 replace_strict 撞上
        # "incomplete mapping" 直接报错。这里统一把行数据裁到股票轴之内 (语义上就是
        # "只在成分股上训练/打分", 与 scripts/example_predict.py 最后 merge 成分表
        # 对齐同一个意图)。source="local" 时两边都来自同一份 bar5m parquet, 集合
        # 天然一致, 这一步是 no-op。labels.py::_compute_vwap_adj 在同样的
        # replace_strict 之前也做了对应的 is_in 过滤。
        df = df.filter(pl.col("key").is_in(universe))
        # 把 (key, date) 映射到 (key_idx, t_idx)，用于把长表"散射"进规整数组，
        # 不去物化 N x T 的笛卡尔积骨架 (那样对大范围查询会非常吃内存)。
        df = df.join(grid_lookup, on="date", how="inner")
        if df.height == 0:
            del df
            continue
        key_idx = df["key"].replace_strict(key_to_idx, return_dtype=pl.Int64).to_numpy()
        t_idx = df["t_idx"].to_numpy()
        values[key_idx, t_idx, :] = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
        adjust_factor[key_idx, t_idx] = df["adjust_factor"].to_numpy().astype(np.float32)
        del df, key_idx, t_idx
        gc.collect()

    close_idx = FEATURE_COLS.index("close")
    close_raw = values[:, :, close_idx]
    is_missing = np.isnan(close_raw) | (close_raw <= 0)

    # 关键一步: 把全部 PRICE_COLS (不只 close) 在 is_missing 处置成 NaN，
    # 再算 adj_close -- 见模块 docstring。必须在这里做，不能留给下游。
    price_idxs = [FEATURE_COLS.index(c) for c in PRICE_COLS]
    for i in price_idxs:
        values[:, :, i] = np.where(is_missing, np.nan, values[:, :, i])

    close = values[:, :, close_idx]  # 此时已经在 is_missing 处是 NaN
    adj_close = (close * adjust_factor).astype(np.float32)

    return DensePanel(
        keys=universe,
        time_grid=time_grid,
        bars_per_day=bars_per_day,
        values=values,
        is_missing=is_missing,
        adj_close=adj_close,
    )


def _ffill_within_day(values: np.ndarray, bars_per_day: int) -> np.ndarray:
    """对 (N, T) 数组做"日内前向填充": 按 bars_per_day 分块，块内独立前向
    填充，不跨块。块的第一个位置永远不会被上一块的值填充；若本身缺失且
    块内此前也没有有效值，保持 NaN。用 numpy 向量化实现 (对每个块沿时间
    轴求"最近一次有效值的下标"，没有则为 -1)，不写 Python 双重循环。

    输入:
        values: float32/float64 (N, T)，T 必须能整除 bars_per_day。
        bars_per_day: int，每个块 (交易日) 的长度。
    返回:
        与输入同形状的新数组 (float32)，已做块内前向填充。
    """
    n, t = values.shape
    assert t % bars_per_day == 0, f"T={t} 不能整除 bars_per_day={bars_per_day}"
    num_days = t // bars_per_day
    v = values.reshape(n, num_days, bars_per_day)

    valid = ~np.isnan(v)
    slot_idx = np.arange(bars_per_day, dtype=np.int64)[None, None, :]
    last_valid_idx = np.where(valid, slot_idx, -1)
    last_valid_idx = np.maximum.accumulate(last_valid_idx, axis=2)

    filled = np.take_along_axis(v, np.clip(last_valid_idx, 0, None), axis=2)
    filled = np.where(last_valid_idx >= 0, filled, np.nan)
    return filled.reshape(n, t).astype(np.float32)


def fill_within_day(panel: DensePanel) -> np.ndarray:
    """生成供模型输入 X 使用的填充后特征数组。**必须先用 panel (原始未填充)
    算完 label/y_mask 之后再调用本函数**——本函数不修改 panel 本身，只返回
    一份新数组，避免把填充后的数据误传给 label 计算。

    - 价格类列 (config.PRICE_COLS): 日内前向填充 (_ffill_within_day)，延续
      当天更早时刻的有效价格; 不跨天延续。当天第一个时间片就缺失、且当天
      此前也没有任何有效值可延续，保持 NaN (不会被前一天的值填充)。
    - 量类列 (config.VOL_COLS): 缺失位置直接填 0 (停牌当天真实成交量就是
      0，不是"延续上一根的量")。

    输入:
        panel: DensePanel，build_dense_panel() 的返回值。
    返回:
        float32 (N, T, len(FEATURE_COLS))，与 panel.values 同形状的新数组。
    """
    filled = panel.values.copy()
    for c in PRICE_COLS:
        i = FEATURE_COLS.index(c)
        filled[:, :, i] = _ffill_within_day(panel.values[:, :, i], panel.bars_per_day)
    for c in VOL_COLS:
        i = FEATURE_COLS.index(c)
        filled[:, :, i] = np.nan_to_num(panel.values[:, :, i], nan=0.0)
    return filled


# ==============================================================================
# dense_dataloader/labels.py  —  数据: 前瞻 VWAP 收益率 label
# ==============================================================================

# -*- coding: utf-8 -*-
import gc

import numpy as np
import polars as pl
import structlog


logger = structlog.get_logger()


def _compute_vwap_adj(panel: DensePanel, source: str = "local") -> np.ndarray:
    """对 panel 的时间网格上每个 (key, t) 算"锚点 t 之后下一个可成交 5 分钟
    窗口"的 VWAP (复权)，见模块 docstring。

    输入:
        panel: DensePanel，用其 keys/time_grid/bars_per_day 确定要覆盖的
            股票和日期范围 (不需要调用方另外传 date_range/keys)。
        source: "local" (默认) 读本地 bar1m parquet；"online" 现场
            dai.query bigalpha_2026_stock_bar1m (见 online_source.py)，
            算法 (5 分钟窗口聚合/VWAP 公式/有效性判定) 完全一致，只有取数
            这一步不同。
    返回:
        float32 (N, T)，无法计算的位置 (停牌/总成交量为0/没有下一格) 为 NaN。
    """
    keys = panel.keys
    time_grid = panel.time_grid
    lo, hi = time_grid["date"].min(), time_grid["date"].max()

    key_to_idx = {k: i for i, k in enumerate(keys)}
    n, t = len(keys), time_grid.height
    grid_lookup = time_grid.select("date", "t_idx")

    # vwap_adj[k, t] = 锚点 t 之后"下一个可成交 5 分钟窗口"的 VWAP。T 轴是扁平的
    # 全局交易日历索引 (只含真实交易日)，所以 t+1 天然就是下一个可成交窗口，
    # 自动跨过午休 (11:30 -> 13:00-13:05) 和隔夜 (15:00 -> 次日 09:30-09:35)，
    # 不需要为这两个锚点写任何特判。最后一格没有 t+1，保持 NaN。
    #
    # 这里直接散射到 t_idx-1 (而不是先填一份 bucket_vwap 再整体 vwap_adj[:, :-1]
    # = bucket_vwap[:, 1:])，两者逐值等价但只需要一份 (N,T)。
    vwap_adj = np.full((n, t), np.nan, dtype=np.float32)

    # 按自然年分块: bar1m 是全流程最大的一次取数 (行数是 bar5m 的 5 倍，全历史
    # 量级下长表本身就有几十 GiB)，而它最终只为了填上面这个 (N,T)。每块聚合完
    # 立即散射并释放，峰值只由单块决定。5 分钟聚合窗口锚在绝对时间网格上、不随
    # 每帧起点漂移 (已验证)，且交易时段不跨午夜，所以窗口不会被年边界切开，
    # 分块聚合与整体聚合逐值等价；散射按 (key_idx, t_idx-1) 绝对定位，跨块的
    # 左移 (某年首格的 VWAP 赋给上一年末格的锚点) 也天然正确。
    for chunk_lo, chunk_hi in year_chunks((lo, hi)):
        if source == "online":
            pass  # (原为包内 import, 单文件版无需)
            raw = fetch_bar1m_canonical((chunk_lo, chunk_hi), keys)
            raw = _ensure_adjust_factor(raw, "labels._compute_vwap_adj")
            lf = (
                raw.lazy()
                .select("date", "key", "close", "volume", "amount", "adjust_factor")
                .filter(pl.col("key").is_in(keys))
                .with_columns((pl.col("close").is_null() | (pl.col("close") <= 0)).alias("is_missing"))
            )
        else:
            lf = (
                pl.scan_parquet(BAR1M_GLOB)
                .filter(pl.col("date").is_between(pl.lit(chunk_lo), pl.lit(chunk_hi)))
                .filter(pl.col("key").is_in(keys))
                .select("date", "key", "close", "volume", "amount", "adjust_factor")
                .with_columns((pl.col("close").is_null() | (pl.col("close") <= 0)).alias("is_missing"))
            )

        agg = (
            lf.sort("date")
            .group_by_dynamic("date", every="5m", closed="right", label="right", group_by="key")
            .agg(
                amount_sum=pl.col("amount").sum(),
                volume_sum=pl.col("volume").sum(),
                n_bars=pl.len(),
                n_bad=pl.col("is_missing").sum(),
                adjust_factor=pl.col("adjust_factor").last(),
            )
            .collect()
        )

        # 不重标窗口标签: group_by_dynamic 产出的右标签 (窗口 (X-5, X] 标在 X)
        # 恰好就等于 time_grid 的槽位时刻，所以直接按原标签 join 拿到 t_idx，
        # 得到"结束于槽位 t 的那个 5 分钟窗口"的 VWAP。
        valid = (pl.col("n_bars") == 5) & (pl.col("n_bad") == 0) & (pl.col("volume_sum") > 0)
        agg = agg.with_columns(
            pl.when(valid)
            .then(pl.col("amount_sum") / pl.col("volume_sum") * pl.col("adjust_factor"))
            .otherwise(None)
            .alias("vwap_adj")
        )

        agg = agg.join(grid_lookup, on="date", how="inner")
        if agg.height == 0:
            del agg
            continue

        key_idx = agg["key"].replace_strict(key_to_idx, return_dtype=pl.Int64).to_numpy()
        t_idx = agg["t_idx"].to_numpy()
        vwap_adj_flat = agg["vwap_adj"].to_numpy().astype(np.float32)
        # 左移一格: 结束于槽位 t_idx 的窗口，服务的是锚点 t_idx-1。t_idx==0 的
        # 窗口没有对应锚点 (原实现里 bucket_vwap[:, 0] 同样从未被读取)，丢弃。
        shift = t_idx >= 1
        vwap_adj[key_idx[shift], t_idx[shift] - 1] = vwap_adj_flat[shift]
        del agg, key_idx, t_idx, vwap_adj_flat, shift
        gc.collect()

    return vwap_adj


def compute_forward_return_label(panel: DensePanel, horizon_days: int = 1, source: str = "local"):
    """默认 target: 未来 horizon_days 个交易日、同一时间片的前瞻 VWAP 窗口收益率。

    label[k, t] = vwap_adj[k, t + n] / vwap_adj[k, t] - 1,
    n = horizon_days * panel.bars_per_day, vwap_adj 见 _compute_vwap_adj。

    T 轴本身就是全局交易日历 (不是某只股票自己的日期)，所以沿 T 轴整体
    平移 n 个位置，天然对应真实的"下一个交易日同一时间片"，不需要
    segment_id 之类的分段逻辑——只要 t 和 t+n 两个位置的 vwap_adj 都有效
    (非 NaN)，label 就是有效的；缺失的地方 (不管是因为停牌，还是股票当时
    不在股票池里，还是取数范围末尾没有下一格) 都会自然算出 NaN。

    输入:
        panel: DensePanel，**必须是 build_dense_panel() 的原始 (未填充)
            返回值**，不能传 fill_within_day() 填充后的数组。
        horizon_days: int，往未来看多少个交易日，默认 1 (下一个交易日)。
        source: "local" (默认) 或 "online"，透传给 _compute_vwap_adj，
            必须和构建 panel 时用的 source 一致。
    返回:
        (label, y_mask) 二元组:
            - label: float32 (N, T)，值 = 未来收益率；无法计算的位置为 NaN
              (自身或目标位置的 vwap_adj 无效、或目标位置超出 T 范围)。
            - y_mask: bool (N, T)，label 是否有效 (= np.isfinite(label))。
    """
    n = round(horizon_days * panel.bars_per_day)
    vwap_adj = _compute_vwap_adj(panel, source=source)
    _, t = vwap_adj.shape

    future = np.full_like(vwap_adj, np.nan)
    if n < t:
        future[:, : t - n] = vwap_adj[:, n:]

    with np.errstate(invalid="ignore", divide="ignore"):
        label = future / vwap_adj - 1.0
    y_mask = np.isfinite(label)
    label = np.where(y_mask, label, np.nan).astype(np.float32)
    return label, y_mask


# ==============================================================================
# dense_dataloader/dataset.py  —  数据: 锚点/mask/标准化统计 + 截面 Dataset
# ==============================================================================

# -*- coding: utf-8 -*-
from typing import Optional, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset



def compute_still_missing(filled: np.ndarray, price_col_idxs: Sequence[int]) -> np.ndarray:
    """日内前向填充之后，某个 (key,t) 是否仍有残留缺失。

    只检查 PRICE_COLS 对应的列: VOL_COLS 经 fill_within_day 的
    np.nan_to_num 处理后永远不再是 NaN，查了也没有意义。

    输入:
        filled: float32 (N, T, F)，panel.fill_within_day() 的输出。
        price_col_idxs: PRICE_COLS 各列在 F 维上的下标。
    返回:
        bool (N, T)，True = 该 (key,t) 在 PRICE_COLS 里仍有 NaN
        (即当天到这一刻为止都没有任何有效值可延续)。
    """
    # 逐列 OR 累积，不写成 np.isnan(filled[:, :, price_col_idxs]).any(-1):
    # 后者的 fancy index 会先物化一份 (N, T, len(PRICE_COLS)) 的副本 (全历史
    # 全股票量级约 5.4GiB)，而 filled[:, :, i] 是基本切片、只是 view，
    # 整个过程除了最终的 (N, T) bool 之外不再分配大数组。逐值等价。
    out = np.zeros(filled.shape[:2], dtype=bool)
    for i in price_col_idxs:
        out |= np.isnan(filled[:, :, i])
    return out


def compute_x_mask_full(still_missing: np.ndarray, lookback_bars: int) -> np.ndarray:
    """(N,T) bool，仅 t >= lookback_bars-1 处有意义: True 表示窗口
    [t-L+1, t] 内没有任何残留缺失 (可以作为该股票在这个锚点的有效样本)。

    用前缀和一次性算出所有 (key,t) 的结果，而不是对每个锚点现算窗口求和
    (那样是 O(N*T*L)，对全量数据太慢)。

    下标关系: cum[:,j] = sum(still_missing[:, 0:j])，
    窗口 [t-L+1, t] (闭区间，长度 L) 的和 = cum[:,t+1] - cum[:,t-L+1]。

    输入:
        still_missing: bool (N, T)，compute_still_missing() 的输出。
        lookback_bars: int，回看窗口长度 L。
    返回:
        bool (N, T)，t < lookback_bars-1 处恒为 False (窗口本身就不完整)。
    """
    n, t = still_missing.shape
    L = lookback_bars
    # int32 而非 int64: 前缀和的上界就是 T (每个位置最多累加 1)，全历史口径
    # T≈7e4 远小于 int32 上限 2.1e9，不可能溢出；(N,T) 这一份省一半内存。
    cum = np.zeros((n, t + 1), dtype=np.int32)
    np.cumsum(still_missing, axis=1, out=cum[:, 1:])
    x_mask_full = np.full((n, t), False)
    if L <= t:
        window_missing_count = cum[:, L:] - cum[:, : t - L + 1]
        x_mask_full[:, L - 1:] = window_missing_count == 0
    return x_mask_full


def build_anchors(
    time_grid: pl.DataFrame,
    day_lo_idx: int,
    day_hi_idx: int,
    lookback_bars: int,
    anchor_stride: int,
    anchor_slots: Optional[Sequence[int]] = None,
    stride_by_day: bool = False,
    day_stride: Optional[int] = None,
) -> np.ndarray:
    """某个 split 的锚点 t_idx 列表: 只按"历史够不够、落在哪个 split"筛，
    不按缺失筛 (缺失交给 X_mask/y_mask 处理，不做样本级别排除)。

    输入:
        time_grid: DensePanel.time_grid。
        day_lo_idx, day_hi_idx: 闭区间，该 split 对应的 day_idx 范围
            (缓冲区的 day_idx 不应包含在内，由调用方算好传入)。
        lookback_bars: int，回看窗口长度 L，要求 t_idx >= L-1。
        anchor_stride: int，相邻锚点之间跨越多少个时间片。
        anchor_slots: 可选的日内 slot_idx 白名单。例如 5m 数据一天有 48 根，
            [45, 46, 47] 代表每天最后三根 (14:50/14:55/15:00)。指定时先按
            slot_idx 过滤；stride_by_day=True 时按交易日抽样，不改变每天保留的槽位。
            否则按全局时间片抽样。
    返回:
        np.ndarray[int64]，升序排列的 t_idx 列表，len(dataset) == 其长度。
    """
    sub = time_grid.filter(
        (pl.col("day_idx") >= day_lo_idx) & (pl.col("day_idx") <= day_hi_idx)
    )
    if anchor_slots is not None:
        slots = list(anchor_slots)
        if not slots:
            raise ValueError("anchor_slots 不能为空；不筛选时请传 None")
        if anchor_stride < 1:
            raise ValueError(f"anchor_stride 必须为正整数，得到 {anchor_stride}")
        if day_stride is None or day_stride < 1:
            raise ValueError(f"day_stride 必须为正整数，得到 {day_stride}")
        max_slot = int(time_grid["slot_idx"].max())
        invalid = [slot for slot in slots if not isinstance(slot, (int, np.integer)) or slot < 0 or slot > max_slot]
        if invalid:
            raise ValueError(f"anchor_slots={invalid} 超出有效 slot_idx 范围 [0, {max_slot}]")
        sub = sub.filter(pl.col("slot_idx").is_in(slots))
        if stride_by_day:
            days = sub.select("day_idx").unique().sort("day_idx")[::day_stride]
            sub = sub.join(days, on="day_idx", how="inner")
        else:
            sub = sub.sort("t_idx")[::anchor_stride]
    else:
        sub = sub.sort("t_idx")[::anchor_stride]
    t_idx = sub.sort("t_idx")["t_idx"].to_numpy()
    t_idx = t_idx[t_idx >= lookback_bars - 1]
    return t_idx.astype(np.int64)


def compute_train_stats(panel: DensePanel, t_lo: int, t_hi: int):
    """在训练区间 [t_lo, t_hi] (闭区间，T 轴下标) 上算每个特征列的
    mean/std，用于标准化。

    - 用**原始未填充**的 panel.values (不是 fill_within_day 的输出)，
      限定在 ~panel.is_missing 的位置——避免同一个真实观测值被前向填充
      复制多次，从而拉低方差估计。
    - VOL_COLS 先做 log1p (跟最终喂给模型的 X 保持同一套变换)，
      再参与统计。

    输入:
        panel: DensePanel。
        t_lo, t_hi: 训练区间在 T 轴上的下标范围 (闭区间)。
    返回:
        (mean, std): 均为 float32 (F,)。
    """
    values_train = panel.values[:, t_lo: t_hi + 1, :].copy()
    vol_idxs = [FEATURE_COLS.index(c) for c in VOL_COLS]
    for i in vol_idxs:
        values_train[:, :, i] = np.log1p(np.clip(values_train[:, :, i], 0, None))

    valid_train = ~panel.is_missing[:, t_lo: t_hi + 1]
    flat = values_train[valid_train]  # (n_obs, F)
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def compute_label_train_stats(label: np.ndarray, y_mask: np.ndarray, t_lo: int, t_hi: int):
    """在训练区间 [t_lo, t_hi] (闭区间，T 轴下标) 上算 label 的 mean/std，
    用于标准化。跟 compute_train_stats 对称，只是 label 没有特征维，
    mean/std 是标量。

    输入:
        label: float32 (N, T)，compute_forward_return_label() 的**原始**
            返回值 (还没有 nan_to_num 过)。
        y_mask: bool (N, T)，label 是否有效，同样来自
            compute_forward_return_label()。
        t_lo, t_hi: 训练区间在 T 轴上的下标范围 (闭区间)，与
            compute_train_stats 共用同一段。
    返回:
        (mean, std): 均为标量 float32。
    """
    label_train = label[:, t_lo: t_hi + 1]
    valid_train = y_mask[:, t_lo: t_hi + 1]
    flat = label_train[valid_train]
    mean = flat.mean().astype(np.float32)
    std = (flat.std() + 1e-6).astype(np.float32)
    return mean, std


class CrossSectionalWindowDataset(Dataset):
    """单条样本 = 某一时间锚点 t，全部股票联合。

    X/label/y_mask/x_mask_full 全部是常驻内存的规整数组 (不按样本展开)，
    __getitem__ 只做一次切片，不重新计算任何东西。
    """

    def __init__(
        self,
        X: np.ndarray,
        label: np.ndarray,
        y_mask: np.ndarray,
        x_mask_full: np.ndarray,
        anchors: np.ndarray,
        lookback_bars: int,
    ):
        self.X = X
        self.label = label
        self.y_mask = y_mask
        self.x_mask_full = x_mask_full
        self.anchors = anchors
        self.L = lookback_bars

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int):
        t = int(self.anchors[idx])
        L = self.L
        X = self.X[:, t - L + 1: t + 1, :].copy()   # (N, L, F)
        Xm = self.x_mask_full[:, t].copy()          # (N,)
        y = self.label[:, t].copy()                 # (N,)
        ym = self.y_mask[:, t].copy()                # (N,)
        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(Xm),
            torch.from_numpy(ym),
        )


# ==============================================================================
# dense_dataloader/shared_panel.py  —  数据: 面板共享内存 (DDP)
# ==============================================================================

# -*- coding: utf-8 -*-
from multiprocessing import shared_memory
from typing import Any, Dict, List, Tuple

import numpy as np

# CrossSectionalWindowDataset 需要的 4 个数组 (见 dense_dataloader/dataset.py)
_PANEL_KEYS = ["X", "label", "y_mask", "x_mask_full"]


def publish_array(arr: np.ndarray, name: str) -> Tuple[Dict[str, Any], shared_memory.SharedMemory]:
    """新建一块具名共享内存，把 arr 的数据拷进去。

    返回 (manifest, shm): manifest 是还原这个数组需要的元信息 (name/shape/dtype，
    体积很小，可以直接塞进 dist.broadcast_object_list); shm 是底层 SharedMemory
    对象，调用方必须一直持有 (见模块 docstring)。
    """
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr[:]
    manifest = {"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
    return manifest, shm


def attach_array(manifest: Dict[str, Any]) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    """挂载 publish_array() 发布的共享内存块，返回同一块物理内存的 numpy 视图。"""
    shm = shared_memory.SharedMemory(name=manifest["name"])
    arr = np.ndarray(tuple(manifest["shape"]), dtype=np.dtype(manifest["dtype"]), buffer=shm.buf)
    return arr, shm


def publish_panel(arrays: Dict[str, np.ndarray], run_id: str) -> Tuple[Dict[str, Any], List[shared_memory.SharedMemory]]:
    """把 build_panel_arrays() 产出的 X/label/y_mask/x_mask_full 四个数组都发布
    到共享内存。run_id 会拼进每块共享内存的名字，保证不同训练进程组之间不冲突
    (同一次 torchrun 内部的多个 rank 共用同一个 run_id, 所以指向同一批内存块)。
    """
    manifest: Dict[str, Any] = {}
    handles: List[shared_memory.SharedMemory] = []
    for key in _PANEL_KEYS:
        sub_manifest, shm = publish_array(arrays[key], name=f"{run_id}_{key}")
        manifest[key] = sub_manifest
        handles.append(shm)
    return manifest, handles


def attach_panel(manifest: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], List[shared_memory.SharedMemory]]:
    """挂载 publish_panel() 发布的四个数组，返回 {X/label/y_mask/x_mask_full: ndarray}。"""
    arrays: Dict[str, np.ndarray] = {}
    handles: List[shared_memory.SharedMemory] = []
    for key in _PANEL_KEYS:
        arr, shm = attach_array(manifest[key])
        arrays[key] = arr
        handles.append(shm)
    return arrays, handles


def close_all(handles: List[shared_memory.SharedMemory], unlink: bool) -> None:
    """训练结束后释放共享内存引用。unlink=True (只有发布方/rank0 该传 True) 才会
    真正释放 /dev/shm 里的空间；其余 rank 只应该 close 自己的引用 (unlink=False)。
    """
    for shm in handles:
        shm.close()
        if unlink:
            shm.unlink()


# ==============================================================================
# dense_dataloader/cache.py  —  数据: X/label/mask 磁盘缓存
# ==============================================================================

# -*- coding: utf-8 -*-
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

CACHE_VERSION = 1
_ARRAY_NAMES = ("X", "label", "y_mask", "x_mask_full", "train_anchors", "val_anchors")


def cache_signature(
    *,
    date_range,
    train_end,
    val_start,
    sample_interval,
    lookback_days,
    step,
    stride,
    keys,
    horizon_days,
    source,
    anchor_slots,
    feature_cols,
    stats,
) -> Dict[str, Any]:
    """生成决定数组语义的稳定签名；任何字段变化都会命中不同缓存目录。"""
    return {
        "version": CACHE_VERSION,
        "date_range": [str(date_range[0]), str(date_range[1])],
        "train_end": str(train_end),
        "val_start": str(val_start) if val_start is not None else None,
        "sample_interval": sample_interval,
        "lookback_days": lookback_days,
        "step": step,
        "stride": stride,
        "keys": list(keys) if keys is not None else None,
        "horizon_days": horizon_days,
        "source": source,
        "anchor_slots": list(anchor_slots) if anchor_slots is not None else None,
        "feature_cols": list(feature_cols),
        "stats": None if stats is None else {
            "mean": np.asarray(stats[0]).tolist(),
            "std": np.asarray(stats[1]).tolist(),
            "y_mean": float(stats[2]),
            "y_std": float(stats[3]),
        },
    }


def cache_key(signature: Dict[str, Any]) -> str:
    encoded = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def cache_path(cache_dir: str, signature: Dict[str, Any]) -> Path:
    return Path(cache_dir).expanduser().resolve() / cache_key(signature)


def load_arrays_cache(cache_dir: str, signature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验 READY+manifest 后以只读 memory-map 返回缓存；任何不完整/不匹配
    缓存都返回 None，由调用方决定重建或报错。"""
    root = cache_path(cache_dir, signature)
    ready = root / "READY"
    manifest_path = root / "manifest.json"
    if not ready.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            return None
        arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in _ARRAY_NAMES}
        for name, arr in arrays.items():
            expected = manifest["arrays"][name]
            if list(arr.shape) != expected["shape"] or str(arr.dtype) != expected["dtype"]:
                return None
        stats = np.load(root / "stats.npz")
        time_grid = pl.read_parquet(root / "time_grid.parquet")
        return {
            **arrays,
            "lookback_bars": int(manifest["lookback_bars"]),
            "stats": (stats["mean"], stats["std"], np.float32(stats["y_mean"]), np.float32(stats["y_std"])),
            "keys": manifest["keys"],
            "time_grid": time_grid,
            "cache_path": str(root),
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_arrays_cache(cache_dir: str, signature: Dict[str, Any], arrays: Dict[str, Any]) -> Path:
    """原子写入缓存。完成前没有 READY，因此异常中断不会被后续训练误读。"""
    root = cache_path(cache_dir, signature)
    tmp = root.with_name(f"{root.name}.writing-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        array_meta = {}
        for name in _ARRAY_NAMES:
            arr = np.asarray(arrays[name])
            np.save(tmp / f"{name}.npy", arr, allow_pickle=False)
            array_meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        mean, std, y_mean, y_std = arrays["stats"]
        np.savez(tmp / "stats.npz", mean=np.asarray(mean), std=np.asarray(std), y_mean=y_mean, y_std=y_std)
        arrays["time_grid"].write_parquet(tmp / "time_grid.parquet")
        manifest = {
            "signature": signature,
            "arrays": array_meta,
            "lookback_bars": int(arrays["lookback_bars"]),
            "keys": list(arrays["keys"]),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp / "READY").write_text("ok\n", encoding="utf-8")
        if root.exists():
            shutil.rmtree(root)
        os.replace(tmp, root)
        return root
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ==============================================================================
# dense_dataloader/dataloader.py  —  数据: 训练/验证 DataLoader 组装
# ==============================================================================

# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm


Stats = Tuple[np.ndarray, np.ndarray, np.float32, np.float32]

_STAGES = [
    "构建稠密面板 (读 bar5m)",
    "计算前瞻 VWAP label (读 bar1m)",
    "日内前向填充 + 量纲变换",
    "计算残留缺失 mask + 标准化统计量",
    "标准化 X / label",
    "构建训练/验证锚点",
]


def _day_idx_bounds(time_grid: pl.DataFrame, lo, hi) -> Tuple[int, int]:
    """time_grid 里日期落在 [lo, hi] (闭区间) 的行对应的 day_idx 范围。"""
    sub = time_grid.filter(pl.col("date").is_between(pl.lit(lo), pl.lit(hi)))
    return int(sub["day_idx"].min()), int(sub["day_idx"].max())


def _t_idx_bounds(time_grid: pl.DataFrame, day_lo_idx: int, day_hi_idx: int) -> Tuple[int, int]:
    """time_grid 里 day_idx 落在 [day_lo_idx, day_hi_idx] 的行对应的 t_idx 范围。"""
    sub = time_grid.filter((pl.col("day_idx") >= day_lo_idx) & (pl.col("day_idx") <= day_hi_idx))
    return int(sub["t_idx"].min()), int(sub["t_idx"].max())


def build_panel_arrays(
    date_range: Tuple[str, str],
    train_end: str,
    sample_interval: int = 5,
    lookback_days: int = 3,
    step: int = 5,
    keys: Optional[Sequence] = None,
    horizon_days: Union[int, float] = 1,
    stats: Optional[Stats] = None,
    val_start: Optional[str] = None,
    show_progress: bool = True,
    source: str = "local",
    anchor_slots: Optional[Sequence[int]] = None,
    stride: Optional[int] = None,
    cache_dir: Optional[str] = None,
    cache_mode: str = "off",
) -> Dict[str, Any]:
    """把面板构建到标准化、切锚点这一整套重活跑一遍，返回原始 numpy 数组
    (不包 DataLoader)，供 get_dataloaders() 和 DDP 场景下的共享内存发布
    (shared_panel.py) 复用同一份逻辑，不重复实现。

    参数含义与 get_dataloaders() 完全一致 (见其 docstring)，多一个
    show_progress: 是否显示按阶段更新的 tqdm 进度条 (全历史全股票量级下这一步
    要跑数分钟, DDP 场景下应只在 rank0 打开，其余 rank 传 False)。

    source: "local" (默认) 读本地 aligned parquet；"online" 现场 dai.query
        线上表 (见 panel.py/labels.py/calendar.py 的同名参数)，产出的
        X/label/mask/锚点/标准化统计与 local 路径完全同一套算法，只有
        "怎么拿原始行" 这一步不同。
    anchor_slots: 可选日内 slot_idx 白名单。例如 5m 数据下 [45,46,47] 只
        选择每天 14:50/14:55/15:00 三个预测时点；不改变每个样本的完整
        lookback 历史窗口。指定时 stride 按交易日抽样，保留选中日期的全部槽位。
    stride: 相邻锚点间隔。未指定 anchor_slots 时，单位为 sample_interval 时间片；
        指定 anchor_slots 时，单位为交易日。缺省时沿用 step 的旧行为。

    返回 dict，键:
        X, label, y_mask, x_mask_full: 见 CrossSectionalWindowDataset 的同名参数
        train_anchors, val_anchors, lookback_bars: 见 build_anchors / dataset.py
        stats: (mean, std, y_mean, y_std)
    """
    cache_mode = cache_mode.lower()
    if cache_mode not in {"off", "read", "write", "readwrite"}:
        raise ValueError(f"cache_mode 必须是 off/read/write/readwrite，得到 {cache_mode}")
    if cache_mode != "off" and not cache_dir:
        raise ValueError(f"cache_mode={cache_mode} 时必须设置 cache_dir")
    if source != "local" and cache_mode != "off":
        raise ValueError("online 数据源默认不支持持久化缓存；避免将可能变化的线上数据误当稳定快照")

    signature = cache_signature(
        date_range=date_range, train_end=train_end, val_start=val_start,
        sample_interval=sample_interval, lookback_days=lookback_days, step=step,
        stride=stride, keys=keys, horizon_days=horizon_days, source=source,
        anchor_slots=anchor_slots, feature_cols=FEATURE_COLS, stats=stats,
    )
    if cache_mode in {"read", "readwrite"}:
        cached = load_arrays_cache(cache_dir, signature)
        if cached is not None:
            return cached
        if cache_mode == "read":
            raise FileNotFoundError("没有找到匹配当前数据配置的 dense arrays 缓存")

    assert 240 % sample_interval == 0, f"sample_interval={sample_interval} 必须整除 240"
    assert step % sample_interval == 0, f"step={step} 必须是 sample_interval={sample_interval} 的整数倍"
    if stride is None:
        stride = step // sample_interval
    if not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride 必须为正整数，得到 {stride}")
    bars_per_day = 240 // sample_interval
    lookback_bars = lookback_days * bars_per_day
    if anchor_slots is None:
        anchor_stride = stride
        stride_by_day = False
        day_stride = None
    else:
        if stride % (240 // sample_interval) != 0:
            raise ValueError(
                f"指定 anchor_slots 时 stride={stride} 必须是一天时间片数 "
                f"{240 // sample_interval} 的整数倍"
            )
        anchor_stride = 1
        stride_by_day = True
        day_stride = stride // (240 // sample_interval)

    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    train_end_ts = pd.Timestamp(train_end)
    val_start_ts = pd.Timestamp(val_start) if val_start is not None else train_end_ts

    # 缓冲区: 前向缓冲保证回看窗口历史足够，后向缓冲保证区间末尾的 label
    # 能取到未来数据 (按自然日粗略换算 *1.6 覆盖周末/节假日，再加安全余量)。
    # +10 (而不是 +5): slot 47 (15:00 收盘锚点) 的 vwap_adj 现在取自次日开盘
    # (见 labels.py::_compute_vwap_adj 的"下一个可成交窗口"语义)，label 本身
    # 还要再往后 horizon_days 天，等于比其余 slot 多吃掉将近 1 个交易日的
    # 后向缓冲，长假期间富余量不够会导致区间末尾这一个 slot 的 label 提前
    # 退化成 NaN (优雅降级，不是错误，但富余量给够更省心)。
    back_buffer = pd.Timedelta(days=int(lookback_days * 1.6) + 10)
    fwd_buffer = pd.Timedelta(days=int(horizon_days * 1.6) + 10)
    fetch_range = (start - back_buffer, end + fwd_buffer)

    pbar = tqdm(total=len(_STAGES), desc=_STAGES[0], disable=not show_progress)

    panel = build_dense_panel(sample_interval, fetch_range, keys, source=source)
    pbar.update(1)
    pbar.set_description(_STAGES[1])

    label, y_mask = compute_forward_return_label(panel, horizon_days=horizon_days, source=source)
    pbar.update(1)
    pbar.set_description(_STAGES[2])

    filled = fill_within_day(panel)
    vol_idxs = [FEATURE_COLS.index(c) for c in VOL_COLS]
    for i in vol_idxs:
        filled[:, :, i] = np.log1p(np.clip(filled[:, :, i], 0, None))
    pbar.update(1)
    pbar.set_description(_STAGES[3])

    price_idxs = [FEATURE_COLS.index(c) for c in PRICE_COLS]
    still_missing = compute_still_missing(filled, price_idxs)
    x_mask_full = compute_x_mask_full(still_missing, lookback_bars)
    del still_missing
    import gc
    gc.collect()

    day_lo_idx, day_hi_idx = _day_idx_bounds(panel.time_grid, start, end)
    # 训练/验证按 train_end/val_start 切分 day_idx: [day_lo_idx, train_hi_idx] 训练，
    # [val_lo_idx, day_hi_idx] 验证；val_start 缺省时 val_start_ts == train_end_ts，
    # 两段紧邻 (旧行为)。val_start_ts > train_end_ts 时 [train_end, val_start) 之间
    # 的日期既不进训练也不进验证 (人为空出的隔离期)。
    train_days = panel.time_grid.filter(
        (pl.col("day_idx") >= day_lo_idx) & (pl.col("day_idx") <= day_hi_idx) & (pl.col("date") < train_end_ts)
    )
    val_days = panel.time_grid.filter(
        (pl.col("day_idx") >= day_lo_idx) & (pl.col("day_idx") <= day_hi_idx) & (pl.col("date") >= val_start_ts)
    )
    train_hi_idx = int(train_days["day_idx"].max())
    val_lo_idx = int(val_days["day_idx"].min())

    if stats is None:
        t_lo, t_hi = _t_idx_bounds(panel.time_grid, day_lo_idx, train_hi_idx)
        mean, std = compute_train_stats(panel, t_lo, t_hi)
        y_mean, y_std = compute_label_train_stats(label, y_mask, t_lo, t_hi)
        stats = (mean, std, y_mean, y_std)
    mean, std, y_mean, y_std = stats
    pbar.update(1)
    pbar.set_description(_STAGES[4])

    panel.values = None
    panel.adj_close = None
    import gc
    gc.collect()

    # 原地标准化: 不写成 X = np.nan_to_num((filled - mean) / std).astype(np.float32)。
    # 那个写法会同时持有 filled、(filled-mean)/std 的临时结果、nan_to_num 的输出
    # 三份 (N,T,F) (全历史全股票量级每份约 10.2GiB)，是整个构建流程的峰值来源。
    # filled 已经是 float32 且此后不再被单独使用，直接就地改写成 X:
    # -= / /= 不分配新数组, nan_to_num(copy=False) 也就地填 0。逐值等价。
    filled -= mean
    filled /= std
    np.nan_to_num(filled, nan=0.0, copy=False)
    X = filled          # 同一块内存, 只是换个名字表达"已经是标准化后的 X"
    del filled
    gc.collect()

    # label 同理就地标准化 ((N,T) 每份约 0.54GiB)。label 由
    # compute_forward_return_label 现算返回, 是本函数独占的临时数组, 可安全就地改。
    label -= y_mean
    label /= y_std
    np.nan_to_num(label, nan=0.0, copy=False)
    label = label.astype(np.float32, copy=False)
    pbar.update(1)
    pbar.set_description(_STAGES[5])

    train_anchors = build_anchors(
        panel.time_grid, day_lo_idx, train_hi_idx, lookback_bars, anchor_stride,
        anchor_slots=anchor_slots, stride_by_day=stride_by_day, day_stride=day_stride,
    )
    val_anchors = build_anchors(
        panel.time_grid, val_lo_idx, day_hi_idx, lookback_bars, anchor_stride,
        anchor_slots=anchor_slots, stride_by_day=stride_by_day, day_stride=day_stride,
    )
    pbar.update(1)
    pbar.close()

    if len(train_anchors) == 0:
        raise RuntimeError("build_panel_arrays: 训练集锚点为空，检查 train_end 是否过早")

    result = {
        "X": X,
        "label": label,
        "y_mask": y_mask,
        "x_mask_full": x_mask_full,
        "train_anchors": train_anchors,
        "val_anchors": val_anchors,
        "lookback_bars": lookback_bars,
        "stats": stats,
        "keys": panel.keys,
        "time_grid": panel.time_grid,
    }
    if cache_mode in {"write", "readwrite"}:
        cache_root = save_arrays_cache(cache_dir, signature, result)
        result["cache_path"] = str(cache_root)
    return result


def get_dataloaders(
    date_range: Tuple[str, str],
    train_end: str,
    sample_interval: int = 5,
    lookback_days: int = 3,
    step: int = 5,
    keys: Optional[Sequence] = None,
    horizon_days: Union[int, float] = 1,
    batch_size: int = 8,
    stats: Optional[Stats] = None,
    num_workers: int = 0,
    shuffle_train: bool = True,
    val_start: Optional[str] = None,
    show_progress: bool = True,
    source: str = "local",
    anchor_slots: Optional[Sequence[int]] = None,
    stride: Optional[int] = None,
    cache_dir: Optional[str] = None,
    cache_mode: str = "off",
    include_val_in_train: bool = False,
) -> Tuple[DataLoader, DataLoader, Stats]:
    """构建训练/验证 DataLoader。

    - date_range: (start, end)，锚点要落在的区间 (内部会自动向前/向后各多
      取一段缓冲数据，分别用于回看窗口历史和 label 的未来数据，缓冲区
      本身不会产生锚点)。
    - train_end: train_end 之前 (不含) 为训练集。
    - val_start: 验证集从这天开始 (含)；不传时默认等于 train_end (训练/验证
      紧邻切分，向后兼容旧行为)。传入且晚于 train_end 时，[train_end, val_start)
      这段日期会被排除在训练和验证锚点之外 (不产生任何样本)，但仍会被拉取、
      参与回看窗口/label 的计算 (供 val_start 附近的锚点回看历史用)——用于人为
      空出一段"训练截止"和"验证开始"之间的隔离期 (比如弃用某几个月的数据)。
    - sample_interval/lookback_days/step: 见 panel.build_time_grid /
      dataset.build_anchors；lookback_bars = lookback_days * bars_per_day，
      anchor_stride = step / sample_interval (必须整除)。
    - keys: 可选，限制股票子集 (调试用)；为 None 时用全历史全部股票
      (local ~2074 只；online 取决于 bigalpha_2026_instruments 的全历史成分数)。
    - horizon_days: label 往未来看多少个交易日，默认 1。
    - batch_size: 默认 8 -- 注意每个样本已经带着全部历史股票 (~2074 只)，
      不是旧设计里"一个样本=一只股票一个时间点"，batch_size 不能沿用
      旧默认的 256，否则单个 batch 会几个 GB 起步。
    - stats: 若为 None，用训练集算 (mean, std, y_mean, y_std) 并标准化
      X 和 label；否则复用传入的统计量 (推理阶段应传入训练时保存的 stats，
      不要重新估计)。y_mean/y_std 是标量 (label 没有特征维)，模型预测值要
      还原成真实收益率时用 pred * y_std + y_mean。
    - show_progress: 是否显示 build_panel_arrays() 内部的按阶段进度条。
    - source: "local" (默认) 读本地 aligned parquet；"online" 现场
      dai.query 线上表 (bigalpha_2026_stock_bar5m/bar1m/bigalpha_2026_instruments,
      见 online_source.py)，产出与 local 路径同构的 X/label/mask，供
      code/scripts/train.py 和最终提交模板 (scripts/transformer_train.py)
      共用同一套下游训练逻辑。
    - anchor_slots: 可选日内 slot_idx 白名单，同时限制 train/validation anchors；
      5m 数据最后三根为 [45,46,47]。每个 anchor 仍保留完整 lookback 窗口。
        (train_loader, val_loader, (mean, std, y_mean, y_std))
    """
    arrays = build_panel_arrays(
        date_range=date_range, train_end=train_end, sample_interval=sample_interval,
        lookback_days=lookback_days, step=step, keys=keys, horizon_days=horizon_days,
        stats=stats, val_start=val_start, show_progress=show_progress, source=source,
        anchor_slots=anchor_slots, stride=stride,
        cache_dir=cache_dir, cache_mode=cache_mode,
    )

    train_anchors = arrays["train_anchors"]
    val_anchors = arrays["val_anchors"]
    if include_val_in_train:
        train_anchors = np.concatenate((train_anchors, val_anchors))
        val_anchors = val_anchors[:0]

    train_ds = CrossSectionalWindowDataset(
        arrays["X"], arrays["label"], arrays["y_mask"], arrays["x_mask_full"],
        train_anchors, arrays["lookback_bars"],
    )
    val_ds = CrossSectionalWindowDataset(
        arrays["X"], arrays["label"], arrays["y_mask"], arrays["x_mask_full"],
        val_anchors, arrays["lookback_bars"],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, arrays["stats"]


# ==============================================================================
# models/RWKV/seq_attention.py  —  模型: RWKV 可选序列注意力模块
# ==============================================================================

#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """
    时序自注意力机制 - 用于金融时间序列的局部时序加权

    输入：h = (B, L, D) - backbone 输出
    输出：output = (B, D) - 注意力加权后的序列表示

    工作原理:
        1. 沿时间轴做局部注意力（滑动窗口）
        2. 生成注意力权重 (B, L, 1) 或 (B, L, D)
        3. 加权聚合：output = Σ(attention_t × h_t)

    与卷积门控的区别:
        - 卷积门控: 用卷积核生成门控，感受野=kernel_size
        - Attention: 用 QKV 计算注意力，感受野=window_size（可全局）
        - Attention 更灵活，可以学习长程依赖

    参数量:
        - num_heads=4, head_dim=64: 约 D×(3×head_dim×num_heads + D) 个参数
        - 比卷积门控多，但表达能力更强
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        window_size: int = 5,  # 局部窗口大小，-1 表示全局注意力
        attention_type: str = 'local',  # 'local' 或 'global'
        use_scalar_gate: bool = True,  # True=标量门控 (B,L,1), False=向量门控 (B,L,D)
        dropout: float = 0.1
    ):
        super(TemporalSelfAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert attention_type in ['local', 'global'], "attention_type must be 'local' or 'global'"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.attention_type = attention_type
        self.use_scalar_gate = use_scalar_gate

        # QKV 投影
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # 温度参数（可选，用于缩放注意力分数）
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, D) - backbone 输出
        return: (B, D) - 注意力加权后的序列表示
        """
        B, L, D = h.shape

        # 1. QKV 投影
        qkv = self.qkv_proj(h)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个 (B, L, D)

        # 2. 多头拆分
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)

        # 3. 计算注意力分数
        scale = self.temperature * (self.head_dim ** -0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

        # 4. 局部注意力掩码（如果是 local 模式）
        if self.attention_type == 'local' and self.window_size > 0:
            mask = self._create_local_mask(L, self.window_size, h.device)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 5. Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 加权求和
        attn_output = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)

        # 7. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        # 8. 输出投影
        attn_output = self.out_proj(attn_output)  # (B, L, D)
        attn_output = self.out_dropout(attn_output)

        # 9. 生成注意力门控（用于最终加权）
        if self.use_scalar_gate:
            # 标量门控: 对所有头做平均，得到 (B, L, 1)
            gate = attn_weights.mean(dim=1, keepdim=True)  # (B, 1, L, L) → 取对角线
            # 取每个位置的平均注意力权重
            gate = gate.diagonal(dim1=-2, dim2=-1).mean(dim=1, keepdim=True)  # (B, 1, L)
            gate = gate.transpose(1, 2)  # (B, L, 1)
        else:
            # 向量门控: 使用输出投影前的值
            gate = attn_output  # (B, L, D)

        # 10. 注意力加权聚合
        # 方法 1: 直接用注意力输出（已经过加权）
        # output = attn_output.mean(dim=1)  # (B, D)

        # 方法 2: 用注意力权重对原始输入加权（更标准）
        attn_weights_summary = attn_weights.mean(dim=1)  # (B, L, L)
        # 取每个位置的总注意力权重
        gate_for_weighting = attn_weights_summary.diagonal(dim1=-2, dim2=-1)  # (B, L)
        gate_for_weighting = gate_for_weighting.unsqueeze(-1)  # (B, L, 1)

        # 用注意力权重对原始输入加权
        h_weighted = h * gate_for_weighting  # (B, L, D) × (B, L, 1)
        output = h_weighted.sum(dim=1)  # (B, D)

        return output

    def _create_local_mask(self, L: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        创建局部注意力掩码
        每个位置只能看到前后 window_size//2 个位置
        """
        # 创建距离矩阵
        i = torch.arange(L, device=device)
        j = torch.arange(L, device=device)
        dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs()  # (L, L)

        # 距离超过 window_size//2 的位置掩码为 0
        mask = (dist <= window_size // 2).float()  # (L, L)

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


class GlobalTemporalAttention(nn.Module):
    """
    全局时序注意力 - 简化版本

    与 TemporalSelfAttention 的区别:
    - 只支持全局注意力
    - 参数更少，适合序列较短的场景
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super(GlobalTemporalAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # QKV 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, D)
        return: (B, D)
        """
        B, L, D = h.shape

        # QKV 投影
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scale = self.temperature * (self.head_dim ** -0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)

        # 对输出加权平均（用注意力权重）
        attn_weights_summary = attn_weights.mean(dim=1)  # (B, L, L)
        # 取最后一个位置的注意力权重（因为我们要聚合整个序列）
        last_attn = attn_weights_summary[:, -1, :]  # (B, L)
        last_attn = last_attn.unsqueeze(-1)  # (B, L, 1)

        output = (h * last_attn).sum(dim=1)  # (B, D)

        return output


# 测试代码


# ==============================================================================
# models/RWKV/conv_gate.py  —  模型: RWKV 可选卷积门控模块
# ==============================================================================

#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvGating(nn.Module):
    """
    时序卷积门控机制 - Depthwise Separable Conv + Gating 版本

    输入：h = (B, L, D) - backbone 输出
    输出：output = (B, D) - 门控加权后的序列表示

    Padding 模式说明:

    1. replication（复制边界）:
       原始: [a, b, c, d, e, f, g]
       padded: [a, a, b, c, d, e, f, g, g]  (kernel_size=3, padding=1)
       特点: 两侧用边界值填充，保持信息完整

    2. causal（因果卷积）:
       原始: [a, b, c, d, e]
       padded: [0, 0, a, b, c, d, e]  (kernel_size=3, padding_left=2)
       特点: 只在左侧补 0，符合因果性，尾部信息完整

    3. causal_replication（因果 + 复制）:
       原始: [a, b, c, d, e]
       padded: [a, a, a, b, c, d, e]  (kernel_size=3, padding_left=2)
       特点: 左侧复制第一个值，右侧不补，尾部完整且无 0 污染
    """

    def __init__(self, d_model: int, kernel_size: int = 3, num_layers: int = 1,
                 dilation: int = 1, padding_mode: str = 'replication'):
        """
        参数:
            d_model: 输入/输出维度
            kernel_size: 卷积核大小（必须为奇数）
            num_layers: 卷积层数
            dilation: 膨胀系数
            padding_mode: padding 模式，可选 ['replication', 'causal', 'causal_replication']
        """
        super(TemporalConvGating, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert num_layers >= 1, "num_layers must be >= 1"
        assert padding_mode in ['replication', 'causal', 'causal_replication'], \
            f"padding_mode must be 'replication', 'causal', or 'causal_replication', got {padding_mode}"

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation = dilation
        self.padding_mode = padding_mode

        # 计算 padding
        if padding_mode == 'replication':
            # 对称 padding
            self.padding_left = kernel_size // 2
            self.padding_right = kernel_size // 2
        elif padding_mode == 'causal':
            # 因果 padding: 只在左侧补 0
            self.padding_left = (kernel_size - 1) * dilation
            self.padding_right = 0
        elif padding_mode == 'causal_replication':
            # 因果 + 复制: 左侧复制第一个值，右侧不补
            self.padding_left = (kernel_size - 1) * dilation
            self.padding_right = 0

        # Depthwise Conv: 每个通道独立卷积
        # 输入 (B, D, L) → Depthwise Conv → (B, D, L)
        # groups=d_model, 输出通道 i 只与输入通道 i 相关
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=0,  # 手动 padding，不使用 nn.Conv1d 的 padding
                dilation=dilation,
                bias=True,
                groups=d_model  # ← 关键: Depthwise Conv, 每个通道独立
            ))

        # 激活函数
        self.gate_activation = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """初始化卷积层权重"""
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def _apply_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        根据 padding_mode 应用相应的 padding

        参数:
            x: 输入张量 (B, D, L)
        返回:
            padded_x: padding 后的张量 (B, D, L')
        """
        if self.padding_mode == 'replication':
            # 两侧用边界值复制
            return F.pad(x, (self.padding_left, self.padding_right), mode='replicate')

        elif self.padding_mode == 'causal':
            # 左侧补 0，右侧不补
            return F.pad(x, (self.padding_left, self.padding_right), mode='constant', value=0)

        elif self.padding_mode == 'causal_replication':
            # 左侧复制第一个值，右侧不补
            # 先获取第一个时间步的值
            first_val = x[:, :, :1]  # (B, D, 1)
            # 复制 padding_left 次
            repeated = first_val.repeat(1, 1, self.padding_left)  # (B, D, padding_left)
            # 拼接到左侧
            return torch.cat([repeated, x], dim=2)

        else:
            raise ValueError(f"Unknown padding_mode: {self.padding_mode}")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, D) - backbone 输出
        return: (B, D) - 门控加权后的序列表示
        """
        B, L, D = h.shape

        # 1. 重排维度: Conv1d 期望 (B, C, L)
        x = h.permute(0, 2, 1)  # (B, L, D) → (B, D, L)

        # 2. 应用 padding
        x = self._apply_padding(x)  # (B, D, L) → (B, D, L + padding_left + padding_right)

        # 3. 多层 Depthwise 时序卷积，生成门控
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)  # (B, D, L') → (B, D, L')
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)  # 中间层用 ReLU

        # 4. Sigmoid 激活: gate ∈ (0, 1), 形状 (B, D, L_out)
        gate = self.gate_activation(x)

        # 5. 重排回 (B, L_out, D)
        gate = gate.permute(0, 2, 1)  # (B, D, L_out) → (B, L_out, D)

        # 6. 将 gate 插值回原始长度 L（如果 padding 导致长度变化）
        if gate.shape[1] != L:
            # 使用线性插值调整长度
            gate = F.interpolate(
                gate.permute(0, 2, 1),  # (B, D, L_out)
                size=L,                  # 目标长度
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # (B, L, D)

        # 7. Element-wise 乘法: 门控加权
        h_gated = h * gate  # (B, L, D) × (B, L, D)

        # 8. 时序加权平均
        gate_sum = gate.sum(dim=1) + 1e-6  # (B, D) - 归一化因子
        h_weighted = h_gated.sum(dim=1)    # (B, D) - 加权求和
        output = h_weighted / gate_sum     # (B, D)

        return output

    def get_gate_stats(self, h: torch.Tensor) -> dict:
        """
        获取门控统计信息（用于调试和分析）
        """
        B, L, D = h.shape
        x = h.permute(0, 2, 1)  # (B, D, L)

        # 应用 padding
        x = self._apply_padding(x)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)

        gate = self.gate_activation(x).permute(0, 2, 1)  # (B, L_out, D)

        # 插值回原始长度
        if gate.shape[1] != L:
            gate = F.interpolate(
                gate.permute(0, 2, 1),
                size=L,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

        return {
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
            'gate_min': gate.min().item(),
            'gate_max': gate.max().item(),
            'gate_shape': gate.shape
        }


# 测试代码


# ==============================================================================
# models/RWKV/rwkv.py  —  模型: RWKV backbone + 适配壳
# ==============================================================================

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# 卷积门控模块

# 序列 Attention 模块


class RWKVTimeMixing(nn.Module):
    """
    RWKV时间混合层 - 实现Receptance-Weighted Key-Value
    """
    def __init__(self, d_model: int, layer_id: int = 0, chunk_size: int = 8):
        super().__init__()
        self.d_model = d_model  # 隐藏层维度
        self.layer_id = layer_id  # 当前块在整个网络中的层索引 (0-based)
        self.chunk_size = chunk_size  # 并行 WKV 分块大小, 见 _compute_wkv_chunked

        # 这里的 ratio 即为 Token Shift 插值系数 μ (论文式 11、12、13)
        # 每个通道独立持有一个可学习标量 μ ∈ (0,1)，控制:
        #  "当前时刻 x_t" 与 "上一时刻 x_{t-1}" 的混合权重:
        #  混合后输入 = μ ⊙ x_t + (1-μ) ⊙ x_{t-1}
        # 对不同通道给予不同初始偏置，使模型各通道一开始就带有差异化的"时间感受野，加速收敛
        ratio = torch.linspace(0, 1, d_model).reshape(1, 1, -1)  # 论文附录 E: μ_ki = (i/S)^{1-L/l}, 此处用 linspace 近似
        self.time_mix_k = nn.Parameter(ratio)
        self.time_mix_v = nn.Parameter(ratio ** 2)
        self.time_mix_r = nn.Parameter(ratio)

        # 线性变换层
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # 时间衰减参数
        self.long_time_decay = nn.Parameter(torch.empty(d_model))
        self.long_time_first = nn.Parameter(torch.empty(d_model))

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # 正交初始化
            nn.init.orthogonal_(self.key.weight, gain=0.8)
            nn.init.orthogonal_(self.value.weight, gain=0.8)
            nn.init.orthogonal_(self.receptance.weight, gain=0.8)
            nn.init.orthogonal_(self.output.weight, gain=1.0)

            # 时间decay初始化: 初始化的是 raw 参数, 实际衰减率为 self._log_w = -exp(raw)
            # (见下方 property), 这里取 raw = log(u), u~Uniform(1,5), 使得初始
            # -exp(raw) = -u ~ Uniform(-5,-1), 与套保护之前的初始分布完全一致。
            self.long_time_decay.uniform_(0.0, math.log(5.0))
            self.long_time_first.normal_(-1, 0.5)

    @property
    def _log_w(self) -> torch.Tensor:
        """有效衰减率 (D,), 恒为负 -> w=exp(_log_w)∈(0,1) 恒成立。

        套一层 -exp() (原版 RWKV 的做法): 无论 long_time_decay 这个可训练 raw
        参数被优化到什么实数值, -exp(raw) 恒为负, 从根本上防止训练把衰减率推成
        正值导致 WKV 在长序列/多 chunk 递推下数值炸裂 (实测 raw 直接用不加保护时,
        推到 ~2.0 附近 L=48 就出 NaN)。
        """
        return -torch.exp(self.long_time_decay)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        B, L, D = x.shape

        if state is None:
            x_prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
            a = b = None
        else:
            prev_x, a, b = state
            x_prev = prev_x

        # 时间混合
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        ek = torch.exp(k)

        if state is None:
            wkv = self._compute_wkv_chunked(ek, v)
            rwkv = r * wkv
            output = self.output(rwkv)
            return output
        else:
            wkv, new_a, new_b = self._wkv_step(ek[:, 0], v[:, 0], a, b)
            rwkv = r[:, 0] * wkv
            out = self.output(rwkv.unsqueeze(1))
            new_state = (x, new_a, new_b)
            return out, new_state

    def _compute_wkv_seq(self, k: torch.Tensor, v: torch.Tensor):
        """并行版 WKV (数学上与循环版逐项等价)

        递推 a_t = Σ_{s<t} w^{t-1-s}·k_s·v_s, b_t = Σ_{s<t} w^{t-1-s}·k_s 是线性的,
        闭式化为下三角衰减矩阵 D[t,s] = w^{t-1-s} (s<t) 与一次 einsum:
            a = D @ (k*v),  b = D @ k
        D 是 L*L*D 大小, 显存/计算随 L² 增长; forward() 默认已改用下面的
        _compute_wkv_chunked (O(L)), 本方法保留作数值对照/回归测试用。
        """
        B, L, D = k.shape
        time_first = torch.exp(self.long_time_first)  # u, (D,)

        # 下三角衰减矩阵: exponent[t,s] = t-1-s (仅 s<t 有效)
        idx = torch.arange(L, device=k.device)
        expo = idx.unsqueeze(1) - 1 - idx.unsqueeze(0)          # L*L
        causal = idx.unsqueeze(0) < idx.unsqueeze(1)            # L*L, s<t
        # 先把无效位置的指数钳为 0 再取 exp, 避免负指数×负 log_w 溢出; 之后用 mask 置零
        expo = expo.clamp(min=0).unsqueeze(-1)                  # L*L*1
        decay = torch.exp(expo * self._log_w)                   # L*L*D, w^{t-1-s}
        decay = decay * causal.unsqueeze(-1)                    # 仅保留 s<t

        kv = k * v
        a = torch.einsum("tsd,bsd->btd", decay, kv)             # B*L*D
        b = torch.einsum("tsd,bsd->btd", decay, k)              # B*L*D

        wkv = (a + time_first * kv) / (b + time_first * k + 1e-6)
        return wkv

    def _compute_wkv_chunked(self, k: torch.Tensor, v: torch.Tensor):
        """分块 (chunk) 矩阵乘 + 跨块递推版 WKV (与 _compute_wkv_seq 数值等价)

        _compute_wkv_seq 的 O(L²) 来自把整条序列一次性摊平成 (L,L,D) 矩阵。这里改为
        按 chunk_size 切块: 块内仍用同样的下三角衰减矩阵公式 (指数 τ-1-σ∈[0,ℓ-1] 天然
        非负、被块长限死, 不会溢出), 块间只传递一个 (B,D) 的小状态 (a,b 在块边界的值)
        做递推, 整体复杂度/显存降为 O(L)。

        没有直接用"提出公因式 w^{t-1-s}=w^{t-1}·w^{-s} 再 cumsum"的写法, 是因为 w^{-s}
        在 s 较大时会显式算出一个巨大的数 (long_time_decay 无约束, 典型取值下 s 到 60+
        就能让 w^{-s} 溢出 float32), 而分块版从不出现负指数, 数值上更安全。
        """
        B, L, D = k.shape
        c = self.chunk_size
        time_first = torch.exp(self.long_time_first)          # u, (D,)
        w = torch.exp(self._log_w)                              # (D,), 兼容 _wkv_step 的定义

        s_a = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        s_b = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        outs = []

        for s0 in range(0, L, c):
            length = min(c, L - s0)
            k_chunk = k[:, s0:s0 + length, :]                  # (B,ℓ,D)
            v_chunk = v[:, s0:s0 + length, :]
            kv_chunk = k_chunk * v_chunk

            idx = torch.arange(length, device=k.device)
            expo = idx.unsqueeze(1) - 1 - idx.unsqueeze(0)      # (ℓ,ℓ), τ-1-σ
            causal = idx.unsqueeze(0) < idx.unsqueeze(1)        # σ<τ
            expo = expo.clamp(min=0).unsqueeze(-1)              # (ℓ,ℓ,1), 非负
            local_decay = torch.exp(expo * self._log_w) * causal.unsqueeze(-1)  # (ℓ,ℓ,D)

            a_local = torch.einsum("tsd,bsd->btd", local_decay, kv_chunk)  # (B,ℓ,D)
            b_local = torch.einsum("tsd,bsd->btd", local_decay, k_chunk)

            w_pow = torch.exp(idx.unsqueeze(-1) * self._log_w)   # (ℓ,D) = w^τ, τ∈[0,ℓ-1]
            a_prev = w_pow.unsqueeze(0) * s_a.unsqueeze(1)                # (B,ℓ,D)
            b_prev = w_pow.unsqueeze(0) * s_b.unsqueeze(1)

            a_chunk = a_local + a_prev
            b_chunk = b_local + b_prev
            wkv_chunk = (a_chunk + time_first * kv_chunk) / (b_chunk + time_first * k_chunk + 1e-6)
            outs.append(wkv_chunk)

            # 把状态推进到本块末尾 (t=s0+length), 供下一块使用
            end_expo = (length - 1 - idx).unsqueeze(-1)                  # (ℓ,1), 非负: ℓ-1-σ
            end_decay = torch.exp(end_expo * self._log_w)       # (ℓ,D)
            end_a = torch.einsum("sd,bsd->bd", end_decay, kv_chunk)      # (B,D)
            end_b = torch.einsum("sd,bsd->bd", end_decay, k_chunk)
            w_full = w ** length                                         # (D,) = w^ℓ, ℓ 很小, 安全
            s_a = w_full * s_a + end_a
            s_b = w_full * s_b + end_b

        return torch.cat(outs, dim=1)

    def _compute_wkv_seq_loop(self, k: torch.Tensor, v: torch.Tensor):
        """原始循环版 (保留作数值对照/回退), 与 lj 源码逐行一致"""
        B, L, D = k.shape
        time_w = torch.exp(self._log_w)
        time_first = torch.exp(self.long_time_first)
        wkv = torch.zeros_like(k)

        a = torch.zeros(B, D, device=k.device)
        b = torch.zeros(B, D, device=k.device)

        for t in range(L):
            kt = k[:, t]
            vt = v[:, t]

            wkv[:, t] = (a + time_first * kt * vt) / (b + time_first * kt + 1e-6)

            a = time_w * a + kt * vt
            b = time_w * b + kt

        return wkv

    def _wkv_step(self, ek: torch.Tensor, v: torch.Tensor,
                  a: torch.Tensor, b: torch.Tensor):
        u = torch.exp(self.long_time_first)
        w = torch.exp(self._log_w)
        kv = ek * v

        wkv = (a + u * kv) / (b + u * ek + 1e-6)

        new_a = w * a + kv
        new_b = w * b + ek
        return wkv, new_a, new_b


class RWKVChannelMixing(nn.Module):
    """
    RWKV通道混合层
    """
    def __init__(self, d_model: int, d_inner: int = None, layer_id: int = 0):
        super().__init__()
        d_inner = d_inner or 4 * d_model

        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        self.key = nn.Linear(d_model, d_inner, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.key.weight, gain=0.8)
            nn.init.orthogonal_(self.receptance.weight, gain=0.8)
            nn.init.orthogonal_(self.value.weight, gain=0.8)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        # 时间偏移
        x_prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

        # 时间混合
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)

        r = torch.sigmoid(self.receptance(xr))
        output = r * kv

        return output


class RWKVBlock(nn.Module):
    """
    完整的 RWKV 块，包含时间混合和通道混合
    支持 ResGate (ReZero-style) 门控残差连接
    """
    def __init__(self, d_model: int, d_inner: int = None, layer_id: int = 0,
                 dropout: float = 0.1, use_resgate: bool = False, gate_init: float = 0.01,
                 chunk_size: int = 8):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_resgate = use_resgate

        self.time_mixing = RWKVTimeMixing(d_model, layer_id, chunk_size=chunk_size)
        self.channel_mixing = RWKVChannelMixing(d_model, d_inner, layer_id)

        self.dropout = nn.Dropout(dropout)

        # ResGate: learnable residual gate parameters
        if use_resgate:
            print("Use ResGate: ", use_resgate)
            self.gate_time = nn.Parameter(torch.tensor(gate_init))
            self.gate_channel = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        # 时间混合分支
        residual = x
        x = self.ln1(x)
        x = self.time_mixing(x, state)
        x = self.dropout(x)
        if self.use_resgate:
            x = residual + self.gate_time * x
        else:
            x = residual + x

        # 通道混合分支
        residual = x
        x = self.ln2(x)
        x = self.channel_mixing(x)
        x = self.dropout(x)
        if self.use_resgate:
            x = residual + self.gate_channel * x
        else:
            x = residual + x

        return x


class RWKVBackbone(nn.Module):
    """
    RWKV骨干网络，堆叠多个RWKV块
    """
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 6,
        d_inner: int = None,
        dropout: float = 0.1,
        pooling: str = 'mean',
        use_attention: bool = False,
        attention_type: str = 'local',
        num_heads: int = 4,
        window_size: int = 5,
        use_conv_gating: bool = False,
        conv_gating_kernel: int = 3,
        conv_gating_layers: int = 1,
        conv_gating_padding_mode: str = 'replication',
        use_resgate: bool = False,
        gate_init: float = 0.01,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pooling = pooling.lower()
        self.d_model = d_model
        self.use_attention = use_attention
        self.use_conv_gating = use_conv_gating
        self.use_resgate = use_resgate

        self.input_proj = nn.Linear(num_features, d_model)

        self.blocks = nn.ModuleList([
            RWKVBlock(d_model, d_inner, layer_id=i, dropout=dropout, use_resgate=use_resgate,
                      gate_init=gate_init, chunk_size=chunk_size)
            for i in range(n_layers)
        ])

        # 归一化
        self.ln_out = nn.LayerNorm(d_model)

        # Attention 模块 (可选)
        if use_attention:
            if attention_type == 'global':
                self.attention = GlobalTemporalAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            else:  # local
                self.attention = TemporalSelfAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    window_size=window_size,
                    attention_type=attention_type,
                    dropout=dropout
                )
            print(f"启用 Attention 模块 type={attention_type}, num_heads={num_heads}, window_size={window_size}")

        # 卷积门控 (可选)
        if use_conv_gating:
            self.conv_gating = TemporalConvGating(
                d_model=d_model,
                kernel_size=conv_gating_kernel,
                num_layers=conv_gating_layers,
                padding_mode=conv_gating_padding_mode
            )
            print(f"启用 ConvGating 模块 kernel_size={conv_gating_kernel}, layers={conv_gating_layers}, padding_mode={conv_gating_padding_mode}")

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.input_proj.weight, gain=0.8)

    def forward(self, x: torch.Tensor):
        B, L, F = x.shape
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)

        # 使用 Attention 模块、卷积门控或传统 pooling
        if self.use_attention:
            return self.attention(h)  # (B, D)
        elif self.use_conv_gating:
            return self.conv_gating(h)  # (B, D)
        elif self.pooling == 'mean':
            return h.mean(dim=1)
        elif self.pooling == 'last':
            return h[:, -1, :]
        elif self.pooling == 'first':
            return h[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")


class RWKVModel(nn.Module):
    """RWKV backbone + MLP 头, 适配本 pipeline 的模型契约。

    组装方式复刻 lj 的 Kraken_RWKV_noamopt._NetRWKV:
      backbone(B'*L*F -> B'*d_model) → MLP 头 (dims, 层间 GELU+Dropout, 末层无激活)
    外壳负责 pipeline 契约: forward(x: B*N*L*C, pad_mask) -> B*N
    (把股票维折进 batch: B' = B*N, F = C, 与 BaseEncoder 相同的折叠方式)

    Parameters
    ----------
    N, L, C : int
        pipeline 的 input_shape (股票数 / 回看长度 / 特征数)。
        L 即 lj 的 c_mlpdays/seq_len 语义 (backbone 吃的序列长度)。
    d_model : int
        RWKV 隐层维度 (lj: c_rwkv_d_model)
    n_layers : int
        RWKV 块数 (lj: c_rwkv_layers)
    d_inner : int | None
        通道混合隐层, None -> 4*d_model (lj: c_rwkv_d_inner)
    dropout : float
        (lj: c_rwkv_dropout; MLP 头层间 Dropout 同用此值, 与 lj 一致)
    pooling : str
        'mean' / 'last' / 'first' (lj: c_rwkv_pooling)
    mlp_dims : list[int]
        MLP 头各层输出维度, 末层须为 1 (lj: c_dims)
    chunk_size : int
        并行 WKV 计算 (RWKVTimeMixing._compute_wkv_chunked) 的分块大小, 默认 8;
        复杂度/显存随此值线性增长 (而非 L 的平方), 数值稳定性也随其减小而提升。
    其余可选参数与 RWKVBackbone 一致 (attention / conv_gating / resgate)。
    """

    def __init__(
        self,
        N: int,
        L: int,
        C: int,
        d_model: int = 256,
        n_layers: int = 2,
        d_inner: int = None,
        dropout: float = 0.2,
        pooling: str = 'last',
        mlp_dims: list = None,
        use_attention: bool = False,
        attention_type: str = 'local',
        num_heads: int = 4,
        window_size: int = 5,
        use_conv_gating: bool = False,
        conv_gating_kernel: int = 3,
        conv_gating_layers: int = 1,
        conv_gating_padding_mode: str = 'replication',
        use_resgate: bool = False,
        gate_init: float = 0.01,
        chunk_size: int = 8,
    ) -> None:
        super().__init__()
        self.N, self.L, self.C = N, L, C
        mlp_dims = [300, 200, 100, 100, 1] if mlp_dims is None else list(mlp_dims)
        if mlp_dims[-1] != 1:
            raise ValueError(f"mlp_dims must end with 1 (scalar output), got {mlp_dims}")

        self.backbone = RWKVBackbone(
            num_features=C,
            seq_len=L,
            d_model=d_model,
            n_layers=n_layers,
            d_inner=d_inner,
            dropout=dropout,
            pooling=pooling,
            use_attention=use_attention,
            attention_type=attention_type,
            num_heads=num_heads,
            window_size=window_size,
            use_conv_gating=use_conv_gating,
            conv_gating_kernel=conv_gating_kernel,
            conv_gating_layers=conv_gating_layers,
            conv_gating_padding_mode=conv_gating_padding_mode,
            use_resgate=use_resgate,
            gate_init=gate_init,
            chunk_size=chunk_size,
        )
        # MLP 头: 复刻 _NetRWKV (层间 GELU+Dropout, 末层裸 Linear)
        mlp_layers = []
        in_dim = d_model
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != mlp_dims[-1]:
                mlp_layers.append(nn.GELU())
                mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        x : torch.Tensor
            Features, B*N*L*C
        pad_mask : torch.Tensor, optional
            Padding mask, B*N (当前未接入 backbone, 与 BaseEncoder 行为一致;
            缺失股票由 ignore_index 在 loss/回测层剔除)

        Returns
        -------
        torch.Tensor
            Predicted labels, B*N
        """
        h = x.view(-1, self.L, self.C)      # B*N*L*C -> (B·N)*L*C
        h = self.backbone(h)                # (B·N)*d_model
        h = self.mlp_head(h).squeeze(-1)    # (B·N)
        return h.view(-1, self.N)           # B*N


# ==============================================================================
# models/RWKV/RWKV_TS.py  —  模型: RWKV-TS backbone + 适配壳
# ==============================================================================

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 rwkv.py 同款可选门控模块


class InstanceNorm(nn.Module):
    """RevIN 式实例归一化 (仅输入侧)。

    对每个实例沿时间轴、逐特征通道做 零均值/单位方差 归一化, 缓解训练/测试分布漂移。
    论文会把 mean/std 反注入输出预测; 本任务输出是标量收益故不反归一化 (见文件头说明)。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> 沿 L 归一化
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class PatchEmbedding(nn.Module):
    """Patching + 线性投影 (论文式 1)。

    多变量分块: 把 (B, L, C) 沿时间切成长度 P、步长 S 的 patch, 每个 patch 展平成
    (P*C) 再线性投影到 d_model, 得到 (B, num_patches, d_model)。
    末端 replication padding 使 num_patches = ⌊(L-P)/S⌋ + 2, 与论文一致。

    若 patch_len<=0 则退化为**不分块**: 逐时刻 Linear(C -> d_model), num_patches=L
    (等价于 rwkv.py 的 input_proj, 便于短回看长度直接用)。
    """

    def __init__(self, num_features: int, seq_len: int, d_model: int,
                 patch_len: int = 0, stride: int = 0):
        super().__init__()
        self.use_patching = patch_len is not None and patch_len > 0
        self.C = num_features
        if self.use_patching:
            self.patch_len = patch_len
            self.stride = stride if stride and stride > 0 else patch_len
            self.pad = nn.ReplicationPad1d((0, self.stride))          # 末端补 stride
            self.num_patches = (seq_len + self.stride - patch_len) // self.stride + 1
            self.proj = nn.Linear(patch_len * num_features, d_model)
        else:
            self.num_patches = seq_len
            self.proj = nn.Linear(num_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if not self.use_patching:
            return self.proj(x)                                       # (B, L, d_model)
        B = x.shape[0]
        x = x.transpose(1, 2)                                         # (B, C, L)
        x = self.pad(x)                                               # (B, C, L+S)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, C, num, P)
        x = x.permute(0, 2, 1, 3).contiguous()                       # (B, num, C, P)
        x = x.reshape(B, x.shape[1], self.C * self.patch_len)        # (B, num, C*P)
        return self.proj(x)                                          # (B, num, d_model)


class RWKVTSTimeMixing(nn.Module):
    """RWKV-TS 时间混合 (Multi-head WKV + Token Shift + Output Gating)。

    对应论文式 2-9。相比 rwkv.py 的单头 WKV, 这里:
      - 多头矩阵态 WKV (RWKV5/6 式): 每头维护 (head_size×head_size) 的 KV 态;
      - Token Shift 多出一路 g (门控), 与 r/k/v 一起用可学习 µ 插值 x_t 与 x_{t-1};
      - 输出门控 o = (SiLU(g) ⊙ GroupNorm_per_head(r·wkv)) W_o。
    并行实现: 小序列 (patch 后 num_patches 很小) 直接用下三角衰减做 O(T^2) einsum。
    """

    def __init__(self, d_model: int, n_head: int = 4, layer_id: int = 0):
        super().__init__()
        assert d_model % n_head == 0, f"d_model({d_model}) 必须能被 n_head({n_head}) 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.head_size = d_model // n_head
        self.layer_id = layer_id

        # Token Shift 插值系数 µ (逐通道), 对 g/r/k/v 各一套 (式 2-5)
        ratio = torch.linspace(0, 1, d_model).reshape(1, 1, -1)
        self.time_mix_g = nn.Parameter(ratio)
        self.time_mix_r = nn.Parameter(ratio)
        self.time_mix_k = nn.Parameter(ratio)
        self.time_mix_v = nn.Parameter(ratio ** 2)

        # 线性投影 (无 bias)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # 每头逐通道的时间衰减 w 与首现 bonus u (式 6-7); w = exp(-exp(time_decay)) ∈ (0,1)
        self.time_decay = nn.Parameter(torch.empty(n_head, self.head_size))
        self.time_first = nn.Parameter(torch.empty(n_head, self.head_size))

        # 每头分组归一化 (= 对每个 head 单独 LayerNorm, 式 9)
        self.group_norm = nn.GroupNorm(n_head, d_model, eps=1e-5)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.gate.weight, gain=1.0)
            nn.init.orthogonal_(self.receptance.weight, gain=0.8)
            nn.init.orthogonal_(self.key.weight, gain=0.8)
            nn.init.orthogonal_(self.value.weight, gain=0.8)
            nn.init.orthogonal_(self.output.weight, gain=1.0)
            # 衰减 speed 在各通道铺开, 使不同通道有不同时间感受野
            decay = torch.linspace(-3.0, 3.0, self.head_size)
            self.time_decay.copy_(decay.unsqueeze(0).repeat(self.n_head, 1))
            self.time_first.fill_(1.0)  # 当前 token bonus 初始为 1

    def _multihead_wkv(self, r, k, v):
        """并行多头 WKV。r/k/v: (B, T, H, P)。返回 (B, T, H, P) = r_t · wkv_t。"""
        B, T, H, P = r.shape
        logw = -torch.exp(self.time_decay)                 # (H, P), log(w) <= 0
        u = self.time_first                                # (H, P)

        idx = torch.arange(T, device=r.device)
        expo = (idx.unsqueeze(1) - 1 - idx.unsqueeze(0))   # (T, T) = t-1-i
        causal = (idx.unsqueeze(0) < idx.unsqueeze(1))     # (T, T), i<t
        eye = torch.eye(T, device=r.device, dtype=torch.bool)

        # D[t,i,h,p] = w^{t-1-i} (i<t) + u (i==t); 无效位 (i>t) 为 0
        expo_c = expo.clamp(min=0).view(T, T, 1, 1).float()
        decay = torch.exp(expo_c * logw.view(1, 1, H, P))  # (T,T,H,P)
        decay = decay * causal.view(T, T, 1, 1)
        D = decay + eye.view(T, T, 1, 1) * u.view(1, 1, H, P)

        # score[b,t,i,h] = Σ_p r[b,t,h,p]·D[t,i,h,p]·k[b,i,h,p]
        rD = torch.einsum("bthp,tihp->btihp", r, D)
        score = torch.einsum("btihp,bihp->btih", rD, k)
        # out[b,t,h,p'] = Σ_i score[b,t,i,h]·v[b,i,h,p']
        out = torch.einsum("btih,bihp->bthp", score, v)
        return out

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        # Token Shift: 用上一时刻插值 (state 仅用于兼容签名, 训练走并行模式)
        x_prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        xg = x * self.time_mix_g + x_prev * (1 - self.time_mix_g)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)

        g = self.gate(xg)                                  # (B,T,D)
        r = self.receptance(xr).view(B, T, self.n_head, self.head_size)
        k = self.key(xk).view(B, T, self.n_head, self.head_size)
        v = self.value(xv).view(B, T, self.n_head, self.head_size)

        wkv = self._multihead_wkv(r, k, v)                 # (B,T,H,P)
        wkv = wkv.reshape(B, T, D)

        # 输出门控 (式 9): (SiLU(g) ⊙ GroupNorm_per_head(r·wkv)) W_o
        wkv = self.group_norm(wkv.reshape(B * T, D)).reshape(B, T, D)
        out = self.output(F.silu(g) * wkv)
        return out


class RWKVTSChannelMixing(nn.Module):
    """RWKV-TS 通道混合 (Token Shift + squared-ReLU + Sigmoid 门控), 论文式 10-13。"""

    def __init__(self, d_model: int, d_inner: int = None, layer_id: int = 0):
        super().__init__()
        d_inner = d_inner or 4 * d_model
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        self.key = nn.Linear(d_model, d_inner, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_inner, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.key.weight, gain=0.8)
            nn.init.orthogonal_(self.receptance.weight, gain=0.8)
            nn.init.orthogonal_(self.value.weight, gain=0.8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = torch.relu(self.key(xk)) ** 2                  # squared ReLU (式 12)
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))             # Sigmoid 门控 (式 13)
        return r * kv


class RWKVTSBlock(nn.Module):
    """RWKV-TS 残差块: (LN → Time-mixing) + (LN → Channel-mixing), 支持 ResGate。"""

    def __init__(self, d_model: int, n_head: int = 4, d_inner: int = None,
                 layer_id: int = 0, dropout: float = 0.1,
                 use_resgate: bool = False, gate_init: float = 0.01):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_resgate = use_resgate

        self.time_mixing = RWKVTSTimeMixing(d_model, n_head, layer_id)
        self.channel_mixing = RWKVTSChannelMixing(d_model, d_inner, layer_id)
        self.dropout = nn.Dropout(dropout)

        if use_resgate:
            self.gate_time = nn.Parameter(torch.tensor(gate_init))
            self.gate_channel = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        residual = x
        x = self.dropout(self.time_mixing(self.ln1(x), state))
        x = residual + (self.gate_time * x if self.use_resgate else x)

        residual = x
        x = self.dropout(self.channel_mixing(self.ln2(x)))
        x = residual + (self.gate_channel * x if self.use_resgate else x)
        return x


class RWKVTSBackbone(nn.Module):
    """RWKV-TS 骨干: 实例归一化 + Patching + 堆叠 RWKV-TS 块 + 输出聚合。

    输出聚合方式 (pooling):
      - 'flatten' (论文默认): 展平所有 patch token 再线性投影到 d_model;
      - 'mean'/'last'/'first': 传统池化;
      - 另可用 use_attention / use_conv_gating 门控替代池化 (与 rwkv.py 一致)。
    统一输出 (B, d_model), 供适配壳的 MLP 头接续。
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_head: int = 4,
        d_inner: int = None,
        dropout: float = 0.1,
        patch_len: int = 0,
        patch_stride: int = 0,
        use_instance_norm: bool = True,
        pooling: str = 'flatten',
        use_attention: bool = False,
        attention_type: str = 'local',
        num_heads: int = 4,
        window_size: int = 5,
        use_conv_gating: bool = False,
        conv_gating_kernel: int = 3,
        conv_gating_layers: int = 1,
        conv_gating_padding_mode: str = 'replication',
        use_resgate: bool = False,
        gate_init: float = 0.01,
    ):
        super().__init__()
        self.pooling = pooling.lower()
        self.d_model = d_model
        self.use_attention = use_attention
        self.use_conv_gating = use_conv_gating
        self.use_instance_norm = use_instance_norm

        # 输入模块
        self.instance_norm = InstanceNorm(num_features) if use_instance_norm else None
        self.patch_embed = PatchEmbedding(num_features, seq_len, d_model,
                                          patch_len=patch_len, stride=patch_stride)
        self.num_patches = self.patch_embed.num_patches

        # RWKV-TS 块
        self.blocks = nn.ModuleList([
            RWKVTSBlock(d_model, n_head=n_head, d_inner=d_inner, layer_id=i,
                        dropout=dropout, use_resgate=use_resgate, gate_init=gate_init)
            for i in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

        # flatten 头 (论文): 展平 (num_patches*d_model) -> d_model
        if self.pooling == 'flatten':
            self.flatten_proj = nn.Linear(self.num_patches * d_model, d_model)

        # 可选门控 (与 rwkv.py 完全一致)
        if use_attention:
            if attention_type == 'global':
                self.attention = GlobalTemporalAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            else:
                self.attention = TemporalSelfAttention(
                    d_model=d_model, num_heads=num_heads, window_size=window_size,
                    attention_type=attention_type, dropout=dropout)
            print(f"[RWKV-TS] 启用 Attention type={attention_type}, heads={num_heads}, window={window_size}")
        if use_conv_gating:
            self.conv_gating = TemporalConvGating(
                d_model=d_model, kernel_size=conv_gating_kernel,
                num_layers=conv_gating_layers, padding_mode=conv_gating_padding_mode)
            print(f"[RWKV-TS] 启用 ConvGating kernel={conv_gating_kernel}, layers={conv_gating_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        h = self.patch_embed(x)                            # (B, num_patches, d_model)
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)

        if self.use_attention:
            return self.attention(h)
        if self.use_conv_gating:
            return self.conv_gating(h)
        if self.pooling == 'flatten':
            return self.flatten_proj(h.reshape(h.shape[0], -1))
        if self.pooling == 'mean':
            return h.mean(dim=1)
        if self.pooling == 'last':
            return h[:, -1, :]
        if self.pooling == 'first':
            return h[:, 0, :]
        raise ValueError(f"Unknown pooling method: {self.pooling}")


class RWKVTSModel(nn.Module):
    """RWKV-TS backbone + MLP 头, 适配本 pipeline 契约。

    与 `rwkv.py:RWKVModel` 平行: 把股票维折进 batch (B'=B*N, F=C),
    forward(x: B*N*L*C, pad_mask) -> B*N。

    Parameters
    ----------
    N, L, C : int
        pipeline 的 input_shape (股票数 / 回看长度 / 特征数)。
    d_model : int
        RWKV 隐层维度 (= 论文 backbone 维度 D)。
    n_layers : int
        RWKV-TS 块数。
    n_head : int
        Multi-head WKV 的头数 (d_model 须被其整除)。
    d_inner : int | None
        通道混合隐层, None -> 4*d_model。
    dropout : float
        块内 & MLP 头层间 Dropout。
    patch_len / patch_stride : int
        Patching 的块长 P 与步长 S; patch_len<=0 则不分块 (逐时刻投影)。
    use_instance_norm : bool
        是否对输入做实例归一化 (RevIN 输入侧)。
    pooling : str
        'flatten'(论文默认) / 'mean' / 'last' / 'first'。
    mlp_dims : list[int]
        MLP 头各层输出维度, 末层须为 1。
    其余门控参数 (attention / conv_gating / resgate) 与 rwkv.py 一致。
    """

    def __init__(
        self,
        N: int,
        L: int,
        C: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_head: int = 4,
        d_inner: int = None,
        dropout: float = 0.2,
        patch_len: int = 0,
        patch_stride: int = 0,
        use_instance_norm: bool = True,
        pooling: str = 'flatten',
        mlp_dims: list = None,
        use_attention: bool = False,
        attention_type: str = 'local',
        num_heads: int = 4,
        window_size: int = 5,
        use_conv_gating: bool = False,
        conv_gating_kernel: int = 3,
        conv_gating_layers: int = 1,
        conv_gating_padding_mode: str = 'replication',
        use_resgate: bool = False,
        gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.N, self.L, self.C = N, L, C
        mlp_dims = [300, 200, 100, 100, 1] if mlp_dims is None else list(mlp_dims)
        if mlp_dims[-1] != 1:
            raise ValueError(f"mlp_dims must end with 1 (scalar output), got {mlp_dims}")

        self.backbone = RWKVTSBackbone(
            num_features=C,
            seq_len=L,
            d_model=d_model,
            n_layers=n_layers,
            n_head=n_head,
            d_inner=d_inner,
            dropout=dropout,
            patch_len=patch_len,
            patch_stride=patch_stride,
            use_instance_norm=use_instance_norm,
            pooling=pooling,
            use_attention=use_attention,
            attention_type=attention_type,
            num_heads=num_heads,
            window_size=window_size,
            use_conv_gating=use_conv_gating,
            conv_gating_kernel=conv_gating_kernel,
            conv_gating_layers=conv_gating_layers,
            conv_gating_padding_mode=conv_gating_padding_mode,
            use_resgate=use_resgate,
            gate_init=gate_init,
        )
        # MLP 头 (与 rwkv.py 一致: 层间 GELU+Dropout, 末层裸 Linear)
        mlp_layers = []
        in_dim = d_model
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != mlp_dims[-1]:
                mlp_layers.append(nn.GELU())
                mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """forward(x: B*N*L*C, pad_mask: B*N) -> B*N"""
        h = x.view(-1, self.L, self.C)      # (B·N)*L*C
        h = self.backbone(h)                # (B·N)*d_model
        h = self.mlp_head(h).squeeze(-1)    # (B·N)
        return h.view(-1, self.N)           # B*N


# ==============================================================================
# models/Mamba/mamba.py  —  模型: Mamba backbone + 适配壳
# ==============================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.utils.checkpoint as cp


# ====================
# 模块级小工具 (内联自 lj 源码, 供 Mamba 组件使用)
# ====================
def _activation(act_name: str) -> nn.Module:
    name = (act_name or "").strip().lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu" or name == "swish":
        return nn.SiLU()
    # 默认 GELU
    return nn.GELU()


def _safe_softplus(x: torch.Tensor) -> torch.Tensor:
    return Fnn.softplus(x)


def _left_pad_causal_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    # x: (B, C, L), 只在左侧补零实现因果卷积
    if pad <= 0:
        return x
    return Fnn.pad(x, (pad, 0))


# ====================
# 选择性 SSM (Mamba 核心) - 纯 PyTorch 实现（无就地写入版本）
# ====================
class MambaSelectiveSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        activation: str = 'GELU',
        scan_mode: str = 'parallel',
        use_dwconv: bool = True,
        internal_amp: bool = False,
    ):
        super().__init__()
        assert d_state >= 1, "d_state must be >= 1"
        assert conv_kernel >= 1, "conv_kernel must be >= 1"
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * max(1, int(expand))  # 内部通道
        self.conv_kernel = conv_kernel
        self.dropout = nn.Dropout(dropout)
        self.act = _activation(activation)
        # 时间维扫描方式: parallel(并行前缀扫描, 默认) / sequential(整段逐步循环) / lowmem(流式逐步, 省显存)
        self.scan_mode = str(scan_mode).lower()
        assert self.scan_mode in ('parallel', 'sequential', 'lowmem'), \
            f"scan_mode must be parallel/sequential/lowmem, got {scan_mode}"
        self.use_dwconv = bool(use_dwconv)
        self.internal_amp = bool(internal_amp)

        # 归一化（Pre-LN）
        self.norm = nn.LayerNorm(d_model)

        # 1) 生成卷积分支的通道：x_u = W_u x
        self.x_u_proj = nn.Linear(d_model, self.d_inner)

        # 2) 生成选择性参数：A(x), b(x), c(x)
        self.x_proj_params = nn.Linear(d_model, 3 * self.d_inner)

        # 3) 门控 g(x)
        self.x_gate = nn.Linear(d_model, self.d_inner)

        # 4) 深度可分离因果卷积 (groups = d_inner)，可禁用
        if self.use_dwconv:
            self.dw_conv_weight = nn.Parameter(
                torch.randn(self.d_inner, 1, conv_kernel) * (1.0 / math.sqrt(conv_kernel))
            )
            self.dw_conv_bias = nn.Parameter(torch.zeros(self.d_inner))
        else:
            self.register_parameter('dw_conv_weight', None)
            self.register_parameter('dw_conv_bias', None)

        # 5) SSM 核心参数
        self.logA = nn.Parameter(torch.empty(self.d_inner, self.d_state))  # 对角，稳定化
        self.B = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.zeros(self.d_inner))  # 直接项

        # 6) 输出投影回 d_model
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # 7) Δ 的偏置项（控制初始时间尺度）
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))
        self.dt_min  = 1e-4  # 防止 Δ 过小

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.uniform_(self.logA, -2.0, 0.0)  # A = -exp(logA) 稳定
            nn.init.xavier_uniform_(self.B)
            nn.init.xavier_uniform_(self.C)
            self.D.zero_()
            if self.use_dwconv:
                nn.init.constant_(self.dw_conv_weight, 0.0)
                self.dw_conv_weight.data[:, :, -1] = 1e-3
                self.dw_conv_bias.zero_()
            nn.init.constant_(self.dt_bias, -2.0)

    # 非 low_mem 时的整段卷积
    def _causal_depthwise_conv1d(self, x_u):
        """
        x_u: (B, L, d_inner) -> (B, L, d_inner)
        """
        if not self.use_dwconv:
            return x_u
        B, L, C = x_u.shape
        x_ch_first = x_u.transpose(1, 2)  # (B, C, L)
        x_pad = _left_pad_causal_1d(x_ch_first, self.conv_kernel - 1)
        y = Fnn.conv1d(
            x_pad, self.dw_conv_weight, self.dw_conv_bias,
            stride=1, padding=0, groups=self.d_inner
        )
        return y.transpose(1, 2)

    def forward(self, x):
        """
        x: (B, L, d_model) -> (B, L, d_model)
        """
        if self.internal_amp and x.is_cuda:
            with torch.cuda.amp.autocast(enabled=True):
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
        B, L, D = x.shape
        assert D == self.d_model
        h = self.norm(x)

        if self.scan_mode == 'lowmem':
            return self._forward_low_mem(x, h)   # 逐步流式扫描，**无就地写入**，省显存
        if self.scan_mode == 'parallel':
            return self._forward_parallel(x, h)  # 并行前缀扫描（默认）
        else:
            # 'sequential': 整段实现，时间维仍是 Python 循环（显存较大）
            u = self.x_u_proj(h)                         # (B, L, d_inner)
            u = self._causal_depthwise_conv1d(u)          # (B, L, d_inner)
            u = self.act(u)

            params = self.x_proj_params(h)                # (B, L, 3*d_inner)
            delta, b_scale, c_scale = torch.chunk(params, 3, dim=-1)
            delta = _safe_softplus(delta + self.dt_bias) + self.dt_min

            g = torch.sigmoid(self.x_gate(h))             # (B, L, d_inner)

            y_ssm = self._selective_scan(u, delta, b_scale, c_scale)  # (B, L, d_inner)
            y = self.out_proj(self.dropout(g * y_ssm))                # (B, L, d_model)
            return x + y

    def _forward_parallel(self, x, h):
        """并行前缀扫描版本（默认）。

        选择性 SSM 的状态递推 x_t = dA_t ⊙ x_{t-1} + dB_u_t 是一阶线性递推,
        可用结合律前缀扫描 (affine 组合的 Hillis-Steele scan) 一次性并行求解整段序列,
        把时间维从 O(L) 顺序步降到 O(log L) 深度; 数学上与 _selective_scan 逐步版等价。
        代价: 需 materialize (B, L, d_inner, d_state) 的中间张量 (以显存换速度);
        配合 grad_checkpoint=True 可在反传时重算, 缓解峰值显存。
        """
        u = self.x_u_proj(h)                          # (B, L, C)
        u = self._causal_depthwise_conv1d(u)          # (B, L, C)
        u = self.act(u)

        params = self.x_proj_params(h)                # (B, L, 3C)
        delta, b_scale, c_scale = torch.chunk(params, 3, dim=-1)
        delta = _safe_softplus(delta + self.dt_bias) + self.dt_min   # (B, L, C)

        g = torch.sigmoid(self.x_gate(h))             # (B, L, C)

        A = -torch.exp(self.logA)                     # (C, N)
        A_safe = torch.where(A == 0.0, A - 1e-6, A)   # (C, N)

        # 离散化（逐时刻、逐样本的选择性系数），与 _selective_scan 完全一致
        dA    = torch.exp(delta.unsqueeze(-1) * A)                    # (B, L, C, N)
        num   = dA - 1.0
        B_eff = self.B * b_scale.unsqueeze(-1)                        # (B, L, C, N)
        dB_u  = num * (B_eff / A_safe) * u.unsqueeze(-1)             # (B, L, C, N)

        # 并行前缀扫描: x_state[:, t] = dA[:, t] * x_state[:, t-1] + dB_u[:, t]
        x_state = self._parallel_scan(dA, dB_u)                      # (B, L, C, N)

        C_eff = self.C * c_scale.unsqueeze(-1)                       # (B, L, C, N)
        y_ssm = (C_eff * x_state).sum(dim=-1)                        # (B, L, C)
        y_ssm = y_ssm + self.D * u                                   # (B, L, C)

        y = self.out_proj(self.dropout(g * y_ssm))                   # (B, L, d_model)
        return x + y

    @staticmethod
    def _parallel_scan(a, b):
        """affine 组合的并行前缀扫描 (Hillis-Steele, 沿时间维 dim=1)。

        输入 a, b: (B, L, C, N), 表示递推 x_t = a_t * x_{t-1} + b_t (x_{-1}=0)。
        返回 x: (B, L, C, N)。做 ceil(log2(L)) 轮, 每轮把"窗口"翻倍地组合仿射变换
        f(x)=a·x+b (组合律: (a_L,b_L)∘(a_R,b_R) = (a_L·a_R, a_L·b_R+b_L))。
        全程只有乘加、无除法 -> 数值稳定 (优于 cumprod 闭式解, 后者对衰减系统会下溢)。
        """
        B, L, C, N = a.shape
        a_acc = a
        b_acc = b
        d = 1
        while d < L:
            # 取 t-d 处的累积仿射; 前 d 个位置用单位元 (a=1, b=0) 左填充
            ones = a_acc.new_ones(B, d, C, N)
            zeros = b_acc.new_zeros(B, d, C, N)
            a_sh = torch.cat([ones, a_acc[:, :L - d]], dim=1)    # a_{t-d}
            b_sh = torch.cat([zeros, b_acc[:, :L - d]], dim=1)   # b_{t-d}
            # 当前(晚)∘移位(早): 当前作用在后
            b_acc = a_acc * b_sh + b_acc
            a_acc = a_acc * a_sh
            d *= 2
        return b_acc

    def _forward_low_mem(self, x, h):
        """
        逐时间步流式版本（无就地切片）：
          - 卷积用长度为 K 的 Python 列表做“窗口”，每步 stack 生成 (B,C,K)。
          - 输出用列表累积，最后 stack 成 (B,L,D)。
        """
        B, L, Dm = x.shape
        C = self.d_inner
        N = self.d_state

        device = x.device
        dtype  = x.dtype

        # SSM 参数与安全项
        A      = -torch.exp(self.logA)                         # (C, N) <= 0
        A_safe = torch.where(A == 0.0, A - 1e-6, A)            # (C, N)
        B_base = self.B                                        # (C, N)
        C_base = self.C                                        # (C, N)
        D_skip = self.D                                        # (C,)

        # 状态 (B, C, N)
        x_state = torch.zeros(B, C, N, device=device, dtype=dtype)

        # 卷积核参数
        if self.use_dwconv:
            Wk = self.dw_conv_weight.squeeze(1)                # (C, K)
            b_conv = self.dw_conv_bias                         # (C,)
            K = self.conv_kernel
            # 初始化窗口：左侧用 0 填充，保持因果
            zero_frame = torch.zeros(B, C, device=device, dtype=dtype)
            buf_list = [zero_frame for _ in range(K-1)]
        else:
            Wk = None
            b_conv = None
            K = 1
            buf_list = None

        y_out_list = []  # 每步的 (B, d_model)

        for t in range(L):
            h_t = h[:, t, :]                                  # (B, d_model)

            # 1) 卷积分支（逐步，**无就地**）
            u_raw_t = self.x_u_proj(h_t)                      # (B, C)
            if self.use_dwconv:
                # 维护长度为 K 的窗口，右端是当前帧
                cur_list = buf_list + [u_raw_t]
                # (B, C, K)
                stack = torch.stack(cur_list, dim=-1)
                # 逐通道点积 + bias
                u_t = (stack * Wk.unsqueeze(0)).sum(dim=-1) + b_conv.unsqueeze(0)  # (B, C)
                # 更新窗口（去掉最左，加入当前）
                buf_list = buf_list[1:] + [u_raw_t]
            else:
                u_t = u_raw_t

            u_t = self.act(u_t)                                  # (B, C)

            # 2) 选择性参数（逐步）
            params_t = self.x_proj_params(h_t)                   # (B, 3C)
            delta_t, b_t, c_t = torch.chunk(params_t, 3, dim=-1)  # each (B, C)
            delta_t = _safe_softplus(delta_t + self.dt_bias) + self.dt_min  # (B, C)

            # 3) 门控（逐步）
            g_t = torch.sigmoid(self.x_gate(h_t))                # (B, C)

            # 4) SSM 离散化与递推（逐步）
            dA_t = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))                       # (B, C, N)
            num   = (dA_t - 1.0)
            B_eff = B_base.unsqueeze(0) * b_t.unsqueeze(-1)                                # (B, C, N)
            dB_u  = num * (B_eff / A_safe.unsqueeze(0)) * u_t.unsqueeze(-1)                # (B, C, N)
            x_state = dA_t * x_state + dB_u                                                # (B, C, N)

            C_eff = C_base.unsqueeze(0) * c_t.unsqueeze(-1)                                # (B, C, N)
            y_ssm_t = (C_eff * x_state).sum(dim=-1)                                        # (B, C)
            y_ssm_t = y_ssm_t + D_skip.unsqueeze(0) * u_t                                  # (B, C)

            # 5) 门控与输出 + 残差（逐步，**无就地**）
            y_t = self.out_proj(self.dropout(g_t * y_ssm_t))                               # (B, d_model)
            y_out_list.append(x[:, t, :] + y_t)                                            # (B, d_model)

        # (B, L, d_model)
        out = torch.stack(y_out_list, dim=1)
        return out

    def _selective_scan(self, u, delta, b_scale, c_scale):
        """
        非 low_mem 模式保留（这里我们也避免就地写入：用 list 累积再 stack）
        """
        B, L, C = u.shape
        N = self.d_state

        A = -torch.exp(self.logA)                       # (C, N)
        A_safe = torch.where(A == 0.0, A - 1e-6, A)
        B_base = self.B                                 # (C, N)
        C_base = self.C                                 # (C, N)
        D_skip = self.D                                 # (C,)

        x_state = u.new_zeros(B, C, N)
        y_list = []

        for t in range(L):
            delta_t = delta[:, t, :].unsqueeze(-1)   # (B, C, 1)
            b_t     = b_scale[:, t, :].unsqueeze(-1) # (B, C, 1)
            c_t     = c_scale[:, t, :].unsqueeze(-1) # (B, C, 1)
            u_t     = u[:, t, :].unsqueeze(-1)       # (B, C, 1)

            dA_t = torch.exp(delta_t * A.unsqueeze(0))                  # (B, C, N)
            num = (dA_t - 1.0)
            B_eff = B_base.unsqueeze(0) * b_t                           # (B, C, N)
            dB_u = num * (B_eff / A_safe.unsqueeze(0)) * u_t            # (B, C, N)

            x_state = dA_t * x_state + dB_u                             # (B, C, N)

            C_eff = C_base.unsqueeze(0) * c_t                           # (B, C, N)
            y_t = torch.sum(C_eff * x_state, dim=-1)                    # (B, C)
            y_t = y_t + D_skip.unsqueeze(0) * u[:, t, :]                # (B, C)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)                                  # (B, L, C)
        return y


# ====================
# Mamba 块 / 骨干
# ====================
class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        activation: str = 'GELU',
        use_ffn: bool = True,
        ffn_expand: int = 2,
        scan_mode: str = 'parallel',
        use_dwconv: bool = True,
        internal_amp: bool = False,
    ):
        super().__init__()
        self.ssm = MambaSelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            conv_kernel=conv_kernel,
            dropout=dropout,
            activation=activation,
            scan_mode=scan_mode,
            use_dwconv=use_dwconv,
            internal_amp=internal_amp,
        )
        self.dropout = nn.Dropout(dropout)

        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            d_ff = int(d_model * max(1, int(ffn_expand)))
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

    def forward(self, x):
        x = self.ssm(x)  # 残差包含在 ssm 内
        if self.use_ffn:
            h = self.norm2(x)
            h = self.ffn(h)
            x = x + self.dropout(h)
        return x


class MambaBackbone(nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        activation: str = 'GELU',
        pooling: str = 'mean',
        use_ffn: bool = True,
        ffn_expand: int = 2,
        scan_mode: str = 'parallel',
        use_dwconv: bool = True,
        internal_amp: bool = False,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pooling = pooling.lower()
        self.d_model = d_model
        self.grad_checkpoint = bool(grad_checkpoint)

        self.input_proj = nn.Linear(num_features, d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                conv_kernel=conv_kernel,
                dropout=dropout,
                activation=activation,
                use_ffn=use_ffn,
                ffn_expand=ffn_expand,
                scan_mode=scan_mode,
                use_dwconv=use_dwconv,
                internal_amp=internal_amp,
            ) for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.input_proj.weight, gain=0.8)

    def forward(self, x):
        h = self.input_proj(x)                    # (B, L, d_model)
        if self.grad_checkpoint and self.training:
            for blk in self.blocks:
                h = cp.checkpoint(blk, h)
        else:
            for blk in self.blocks:
                h = blk(h)

        h = self.ln_out(h)
        if self.pooling == 'mean':
            return h.mean(dim=1)
        elif self.pooling == 'last':
            return h[:, -1, :]
        elif self.pooling == 'first':
            return h[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")


class MambaModel(nn.Module):
    """Mamba backbone + MLP 头, 适配本 pipeline 的模型契约。

    组装方式复刻 lj 的 Kraken_Mamba._NetMamba:
      backbone(B'*L*F -> B'*d_model) → MLP 头 (dims, 层间激活+Dropout, 末层无激活)
    外壳负责 pipeline 契约: forward(x: B*N*L*C, pad_mask) -> B*N
    (把股票维折进 batch: B' = B*N, F = C, 与 BaseEncoder / RWKVModel 相同的折叠方式)

    Parameters
    ----------
    N, L, C : int
        pipeline 的 input_shape (股票数 / 回看长度 / 特征数)。
        L 即 lj 的 c_mlpdays/seq_len 语义 (backbone 吃的序列长度)。
    d_model : int
        Mamba 隐层维度 (lj: _c_mamba_d_model)
    n_layers : int
        Mamba 块数 (lj: _c_mamba_layers)
    d_state : int
        SSM 状态维度 N (lj: _c_mamba_d_state)
    expand : int
        内部通道扩张倍数 d_inner = expand*d_model (lj: _c_mamba_expand)
    conv_kernel : int
        深度可分离因果卷积核长 (lj: _c_mamba_conv_kernel)
    dropout : float
        (lj: _c_mamba_dropout; MLP 头层间 Dropout 同用此值, 与 lj 一致)
    activation : str
        激活函数名 GELU/ReLU/SiLU (lj: _c_activation)
    pooling : str
        'mean' / 'last' / 'first' (lj: _c_mamba_pooling)
    use_ffn : bool
        Mamba 块内是否带 FFN 子层 (lj: _c_mamba_use_ffn)
    ffn_expand : int
        FFN 隐层扩张倍数 (lj: _c_mamba_ffn_expand)
    scan_mode : str
        时间维扫描方式, 默认 'parallel':
          'parallel'   -> 并行前缀扫描 (O(log L) 深度, 默认, 最快)
          'sequential' -> 整段逐步 Python 循环 (显存中等)
          'lowmem'     -> 流式逐步扫描 (最省显存, 最慢)
        (lj 原 _c_mamba_low_mem 布尔开关的超集; 数学上三者等价)
    use_dwconv : bool
        是否启用深度可分离因果卷积分支 (lj: _c_mamba_use_dwconv)
    internal_amp : bool
        SSM 内部是否开 autocast (lj: _c_mamba_internal_amp)
    grad_checkpoint : bool
        训练时是否对每个块做梯度检查点 (lj: _c_mamba_grad_checkpoint)
    mlp_dims : list[int]
        MLP 头各层输出维度, 末层须为 1 (lj: _c_dims)
    """

    def __init__(
        self,
        N: int,
        L: int,
        C: int,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        activation: str = 'GELU',
        pooling: str = 'mean',
        use_ffn: bool = True,
        ffn_expand: int = 2,
        scan_mode: str = 'parallel',
        use_dwconv: bool = True,
        internal_amp: bool = False,
        grad_checkpoint: bool = True,
        mlp_dims: list = None,
    ) -> None:
        super().__init__()
        self.N, self.L, self.C = N, L, C
        mlp_dims = [300, 200, 100, 100, 1] if mlp_dims is None else list(mlp_dims)
        if mlp_dims[-1] != 1:
            raise ValueError(f"mlp_dims must end with 1 (scalar output), got {mlp_dims}")

        self.backbone = MambaBackbone(
            num_features=C,
            seq_len=L,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand=expand,
            conv_kernel=conv_kernel,
            dropout=dropout,
            activation=activation,
            pooling=pooling,
            use_ffn=use_ffn,
            ffn_expand=ffn_expand,
            scan_mode=scan_mode,
            use_dwconv=use_dwconv,
            internal_amp=internal_amp,
            grad_checkpoint=grad_checkpoint,
        )
        # MLP 头: 复刻 _NetMamba (层间激活+Dropout, 末层裸 Linear)
        mlp_layers = []
        in_dim = d_model
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != mlp_dims[-1]:
                mlp_layers.append(_activation(activation))
                mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        x : torch.Tensor
            Features, B*N*L*C
        pad_mask : torch.Tensor, optional
            Padding mask, B*N (当前未接入 backbone, 与 BaseEncoder 行为一致;
            缺失股票由 ignore_index 在 loss/回测层剔除)

        Returns
        -------
        torch.Tensor
            Predicted labels, B*N
        """
        h = x.view(-1, self.L, self.C)      # B*N*L*C -> (B·N)*L*C
        h = self.backbone(h)                # (B·N)*d_model
        h = self.mlp_head(h).squeeze(-1)    # (B·N)
        return h.view(-1, self.N)           # B*N


# ==============================================================================
# losses/power_mse.py  —  损失: PowerMSE
# ==============================================================================

import torch
import torch.nn as nn

__all__ = ["PowerMSELoss"]


class PowerMSELoss(nn.Module):
    """MSE 的变体: Loss = reduce(|pred - target|^power)。

    power=2 时等价于 MSE; power>2 加重大残差, power<2 减轻。
    """

    def __init__(self, power: float = 2.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), (
            f"reduction must be 'mean', 'sum' or 'none', got {reduction}"
        )
        self.power = float(power)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.pow(diff, self.power)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ==============================================================================
# losses/directional.py  —  损失: 方向性
# ==============================================================================

import torch
import torch.nn as nn

__all__ = ["MeanAbsoluteDirectionalLoss", "DirectionalWeightedMSE"]


class MeanAbsoluteDirectionalLoss(nn.Module):
    """按 |target| 加权的方向损失: loss = -direction * |target|。

    - mode='hard': direction = sign(target * pred)
    - mode='soft': direction 用 (target*pred/alpha) 的软符号 (可微)
    方向一致 (target 与 pred 同号) 时 loss 更小 (更负)。
    """

    def __init__(self, mode: str = "soft", alpha: float = 0.0005,
                 reduction: str = "sum"):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "hard":
            direction = torch.sign(target * pred)
        else:
            s = target * pred / self.alpha
            direction = s / (torch.abs(s) + 0.0001)

        loss = -direction * target.abs()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DirectionalWeightedMSE(nn.Module):
    """Direction-Weighted MSE (DW-MSE): 方向相反的样本 MSE 权重更大。

    weight = 1 + beta * (1 - tanh(k * pred * target)) / 2  ∈ [1, 1+beta]
    loss   = weight * (pred - target)^2

    beta=0 退化为 MSE; k->∞ 权重近似 sign 门控, k->0 权重≈常数。
    """

    def __init__(self, beta: float = 0.2, k: float = 2.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.beta = float(beta)
        self.k = float(k)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.beta * (1 - torch.tanh(self.k * pred * target)) / 2.0
        loss = weight * torch.square(pred - target)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ==============================================================================
# losses/huber.py  —  损失: Power/Soft/Stochastic Huber
# ==============================================================================

import torch
import torch.nn as nn

__all__ = [
    "PowerHuberLoss",
    "SoftPowerHuberLoss",
    "AutoDeltaPowerHuberLoss",
    "StochasticPowerHuberLoss",
    "StochasticPowerHuberLossPlus",
]


class PowerHuberLoss(nn.Module):
    """piecewise (C1 连续):
      if |e| <= delta:  0.5 * e^2
      else:             0.5*delta^2 + (delta^2 / p) * ((|e|/delta)^p - 1)

    - 小残差保持二次; 超过阈值后按 |e|^p (p>2) 加重大残差
    - 建议: p∈[2.2, 2.6]; delta≈标签的稳健尺度 (如 IQR/1.349); 标签已标准化时先用 1.0
    - 默认 fp32 计算以提升数值稳定性 (force_fp32=True)
    """

    def __init__(self, delta: float = 1.0, p: float = 2.4, force_fp32: bool = True):
        super().__init__()
        assert p >= 0.0, "p must be >= 2 for super-quadratic tails"
        assert delta > 0.0, "delta must be > 0"
        self.delta = float(delta)
        self.p = float(p)
        self.force_fp32 = bool(force_fp32)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 统一 dtype/device; 默认切 fp32 避免 p>2 时半精度不稳
        dtype = torch.float32 if self.force_fp32 else preds.dtype
        device = preds.device

        d = torch.as_tensor(self.delta, dtype=dtype, device=device)
        p = torch.as_tensor(self.p, dtype=dtype, device=device)

        e = preds.to(dtype) - targets.detach().to(dtype)
        a = e.abs()

        u = a / d
        quad = 0.5 * (e * e)
        tail = 0.5 * d * d + (d * d / p) * (u.pow(p) - 1.0)

        loss = torch.where(a <= d, quad, tail)
        return loss.mean()


class SoftPowerHuberLoss(nn.Module):
    """二次 -> p 次的「软」过渡版本:
      loss = (1 - w) * (0.5 * e^2) + w * [0.5*δ^2 + (δ^2/p)*((|e|/δ)^p - 1)]
      w = sigmoid(kappa * (|e| - δ))

    - 边界附近平滑过渡而非硬切换, 降低对 δ 的敏感
    - 在 |e|=δ 处函数值与导数都更平滑
    """

    def __init__(self, delta: float = 0.5, p: float = 2.2, kappa: float = 10.0,
                 force_fp32: bool = True):
        super().__init__()
        assert p > 0.0 and delta > 0.0
        self.delta = float(delta)
        self.p = float(p)
        self.kappa = float(kappa)
        self.force_fp32 = bool(force_fp32)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dtype = torch.float32 if self.force_fp32 else preds.dtype
        device = preds.device

        d = torch.as_tensor(self.delta, dtype=dtype, device=device)
        p = torch.as_tensor(self.p, dtype=dtype, device=device)
        k = torch.as_tensor(self.kappa, dtype=dtype, device=device)

        e = preds.to(dtype) - targets.detach().to(dtype)
        a = e.abs()

        u = a / d
        quad = 0.5 * (e * e)
        tail = 0.5 * d * d + (d * d / p) * (u.pow(p) - 1.0)

        w = torch.sigmoid(k * (a - d))
        loss = (1.0 - w) * quad + w * tail
        return loss.mean()


class AutoDeltaPowerHuberLoss(nn.Module):
    """PowerHuber 的自适应 δ 版本:
      δ_t = clamp(k * R_t, min_delta, max_delta)
    R_t 是 EMA 平滑的稳健尺度估计:
      - mode='target'   : 用 targets 的分布 (更稳定, 不依赖当前模型好坏)
      - mode='residual' : 用 |preds-targets| (跟随训练进度, 需设 min_delta)

    scale_stat: 'iqr'->IQR/1.349, 'mad'->1.4826*MAD, 'mae'->mean(|x|),
                'std'->标准差, 'pXX'->abs(x) 的 XX 分位 (如 'p90')
    """

    def __init__(self, p: float = 2.2, k: float = 0.5, mode: str = "target",
                 scale_stat: str = "iqr", ema_beta: float = 0.95,
                 init_scale: float = None, min_delta: float = 1e-4,
                 max_delta: float = None, soft: bool = True, kappa: float = 10.0,
                 force_fp32: bool = True):
        super().__init__()
        assert p > 0.0 and k > 0.0
        assert mode in ("target", "residual")
        self.p = float(p)
        self.k = float(k)
        self.mode = mode
        self.scale_stat = scale_stat.lower()
        self.ema_beta = float(ema_beta)
        self.min_delta = float(min_delta)
        self.max_delta = None if max_delta is None else float(max_delta)
        self.soft = bool(soft)
        self.kappa = float(kappa)
        self.force_fp32 = bool(force_fp32)

        init = 1.0 if init_scale is None else float(init_scale)
        # 运行中的稳健尺度与步数 (buffer 不参与梯度, 随模型迁移设备)
        self.register_buffer("running_scale", torch.tensor(init, dtype=torch.float32))
        self.register_buffer("inited", torch.tensor(0, dtype=torch.int32))

    @torch.no_grad()
    def _robust_scale(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().flatten()
        if x.numel() == 0:
            return torch.tensor(self.running_scale.item(), dtype=x.dtype, device=x.device)
        if self.scale_stat == "iqr":
            q75 = torch.quantile(x, 0.75)
            q25 = torch.quantile(x, 0.25)
            scale = (q75 - q25) / 1.349
        elif self.scale_stat == "mad":
            med = x.median()
            mad = (x - med).abs().median()
            scale = 1.4826 * mad
        elif self.scale_stat == "mae":
            scale = x.abs().mean()
        elif self.scale_stat == "std":
            scale = x.std(unbiased=False)
        elif self.scale_stat.startswith("p"):
            try:
                q = float(self.scale_stat[1:]) / 100.0
            except ValueError:
                q = 0.9
            scale = torch.quantile(x.abs(), q)
        else:
            q75 = torch.quantile(x, 0.75)
            q25 = torch.quantile(x, 0.25)
            scale = (q75 - q25) / 1.349
        return torch.clamp(scale, min=1e-8)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dtype = torch.float32 if self.force_fp32 else preds.dtype
        device = preds.device
        p = torch.as_tensor(self.p, dtype=dtype, device=device)
        self.running_scale = self.running_scale.to(device)
        self.inited = self.inited.to(device)

        with torch.no_grad():
            src = targets if self.mode == "target" else (preds - targets).abs()
            scale_now = self._robust_scale(src.to(torch.float32))
            if self.inited.item() == 0:
                self.running_scale.copy_(scale_now)
                self.inited.fill_(1)
            else:
                self.running_scale.mul_(self.ema_beta).add_(scale_now * (1.0 - self.ema_beta))

            delta = self.k * self.running_scale
            delta = torch.clamp(delta, min=self.min_delta)
            if self.max_delta is not None:
                delta = torch.clamp(delta, max=self.max_delta)

        d = delta.to(dtype=dtype, device=device)

        e = preds.to(dtype) - targets.detach().to(dtype)
        a = e.abs()
        u = a / d

        quad = 0.5 * (e * e)
        tail = 0.5 * d * d + (d * d / p) * (u.pow(p) - 1.0)

        if self.soft:
            kappa = torch.as_tensor(self.kappa, dtype=dtype, device=device)
            w = torch.sigmoid(kappa * (a - d))
            loss = (1.0 - w) * quad + w * tail
        else:
            loss = torch.where(a <= d, quad, tail)
        return loss.mean()


class StochasticPowerHuberLoss(nn.Module):
    """训练时随机采样 p 和 δ (小范围), 相当于对损失函数做数据增强。
    建议围绕当前最优点做很小的扰动即可。
    """

    def __init__(self, p_min: float = 2.1, p_max: float = 2.4,
                 delta_min: float = 0.4, delta_max: float = 0.6,
                 soft: bool = True, kappa: float = 10.0, force_fp32: bool = True):
        super().__init__()
        assert p_min > 0 and delta_min > 0
        assert p_max >= p_min and delta_max >= delta_min
        self.p_min, self.p_max = float(p_min), float(p_max)
        self.d_min, self.d_max = float(delta_min), float(delta_max)
        self.soft = bool(soft)
        self.kappa = float(kappa)
        self.force_fp32 = bool(force_fp32)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 每个 batch 采样一次 p, δ
        p_val = torch.empty((), device=preds.device).uniform_(self.p_min, self.p_max).item()
        d_val = torch.empty((), device=preds.device).uniform_(self.d_min, self.d_max).item()

        dtype = torch.float32 if self.force_fp32 else preds.dtype
        device = preds.device
        p = torch.as_tensor(p_val, dtype=dtype, device=device)
        d = torch.as_tensor(d_val, dtype=dtype, device=device)

        e = preds.to(dtype) - targets.detach().to(dtype)
        a = e.abs()
        u = a / d

        quad = 0.5 * (e * e)
        tail = 0.5 * d * d + (d * d / p) * (u.pow(p) - 1.0)

        if self.soft:
            kappa = torch.as_tensor(self.kappa, dtype=dtype, device=device)
            w = torch.sigmoid(kappa * (a - d))
            loss = (1.0 - w) * quad + w * tail
        else:
            loss = torch.where(a <= d, quad, tail)
        return loss.mean()


class StochasticPowerHuberLossPlus(nn.Module):
    """StochasticPowerHuberLoss 的增强版, 针对 A 股做多为主的特点:

    1. 非对称惩罚: pred > threshold (做多) 权重 *long_weight_ratio;
       否则 *short_weight_ratio。
    2. threshold_mode: 'zero'/'mean'/'median'/'target_mean'/'target_median'。
    3. 可选 clamp: 对 pred < threshold+clamp_threshold 的部分限制 penalty 上界。
    """

    def __init__(self, p_min: float = 2.1, p_max: float = 2.5,
                 delta_min: float = 0.4, delta_max: float = 0.6,
                 soft: bool = True, kappa: float = 10.0, force_fp32: bool = True,
                 long_weight_ratio: float = 2.0, short_weight_ratio: float = 1.0,
                 threshold_mode: str = "zero", use_clamp: bool = False,
                 clamp_threshold: float = -0.5, clamp_max_penalty: float = None):
        super().__init__()
        assert p_min > 0 and delta_min > 0
        assert p_max >= p_min and delta_max >= delta_min
        assert long_weight_ratio > 0 and short_weight_ratio > 0
        assert threshold_mode in ("zero", "mean", "median", "target_mean", "target_median")

        self.p_min, self.p_max = float(p_min), float(p_max)
        self.d_min, self.d_max = float(delta_min), float(delta_max)
        self.soft = bool(soft)
        self.kappa = float(kappa)
        self.force_fp32 = bool(force_fp32)

        self.long_weight_ratio = float(long_weight_ratio)
        self.short_weight_ratio = float(short_weight_ratio)
        self.threshold_mode = threshold_mode

        self.use_clamp = bool(use_clamp)
        self.clamp_threshold = float(clamp_threshold)
        self.clamp_max_penalty = None if clamp_max_penalty is None else float(clamp_max_penalty)

    def _compute_threshold(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.threshold_mode == "zero":
            return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        elif self.threshold_mode == "mean":
            return preds.mean()
        elif self.threshold_mode == "median":
            return preds.median()
        elif self.threshold_mode == "target_mean":
            return targets.mean()
        elif self.threshold_mode == "target_median":
            return targets.median()
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p_val = torch.empty((), device=preds.device).uniform_(self.p_min, self.p_max).item()
        d_val = torch.empty((), device=preds.device).uniform_(self.d_min, self.d_max).item()

        dtype = torch.float32 if self.force_fp32 else preds.dtype
        device = preds.device
        p = torch.as_tensor(p_val, dtype=dtype, device=device)
        d = torch.as_tensor(d_val, dtype=dtype, device=device)

        e = preds.to(dtype) - targets.detach().to(dtype)
        a = e.abs()
        u = a / d

        quad = 0.5 * (e * e)
        tail = 0.5 * d * d + (d * d / p) * (u.pow(p) - 1.0)

        if self.soft:
            kappa = torch.as_tensor(self.kappa, dtype=dtype, device=device)
            w = torch.sigmoid(kappa * (a - d))
            base_loss = (1.0 - w) * quad + w * tail
        else:
            base_loss = torch.where(a <= d, quad, tail)

        threshold = self._compute_threshold(preds, targets)

        is_long = (preds > threshold).to(dtype=dtype)
        weight_map = is_long * self.long_weight_ratio + (1.0 - is_long) * self.short_weight_ratio
        weighted_loss = base_loss * weight_map

        if self.use_clamp:
            clamp_mask = (preds < threshold + self.clamp_threshold).to(dtype=dtype)
            if self.clamp_max_penalty is not None:
                max_penalty = torch.tensor(self.clamp_max_penalty, dtype=dtype, device=device)
                weighted_loss = torch.where(clamp_mask > 0,
                                            torch.min(weighted_loss, max_penalty),
                                            weighted_loss)
            else:
                avg_loss = weighted_loss.mean().detach()
                weighted_loss = torch.where(clamp_mask > 0,
                                            torch.min(weighted_loss, avg_loss),
                                            weighted_loss)

        return weighted_loss.mean()


# ==============================================================================
# losses/ranking.py  —  损失: SoftSpearman / RankMSE
# ==============================================================================

import torch
import torch.nn as nn

__all__ = ["soft_rank", "SoftSpearmanLoss", "RankMSE"]


def soft_rank(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """可微软排序 (成对 sigmoid 松弛), 见模块 docstring。

    Parameters
    ----------
    x : torch.Tensor
        B*N, 沿最后一维排序 (逐行独立)。
    regularization_strength : float
        sigmoid 温度 τ, 越大越平滑/越可微, 越小越接近硬排序 (但梯度越易饱和)。

    Returns
    -------
    torch.Tensor
        B*N, 近似秩 (允许整体常数偏移, 见模块 docstring)。
    """
    if x.dim() != 2:
        raise ValueError(f"'x' should be a 2d-tensor but got {x.shape}")
    diff = (x.unsqueeze(-1) - x.unsqueeze(-2)) / regularization_strength  # B*N*N: (x_i-x_j)/τ
    return torch.sigmoid(diff).sum(dim=-1) + 0.5


class SoftSpearmanLoss(nn.Module):
    """1 - 软 Spearman 相关。用 soft_rank 得到可微秩, 再算 Pearson 相关。

    regularization_strength 越大, 软排序越平滑 (越可微, 越偏离硬排序)。
    """

    def __init__(self, regularization_strength: float = 1.0):
        super().__init__()
        self.reg_strength = float(regularization_strength)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = preds.flatten().unsqueeze(0)
        t = targets.flatten().unsqueeze(0).detach()

        p_rank = soft_rank(p, regularization_strength=self.reg_strength)
        t_rank = soft_rank(t, regularization_strength=self.reg_strength)

        p_rank = p_rank - p_rank.mean(dim=1, keepdim=True)
        t_rank = t_rank - t_rank.mean(dim=1, keepdim=True)

        cov = (p_rank * t_rank).mean()
        p_std = p_rank.std(unbiased=False) + 1e-8
        t_std = t_rank.std(unbiased=False) + 1e-8
        rho = cov / (p_std * t_std)
        return 1.0 - rho


class RankMSE(nn.Module):
    """预测秩与标签秩的 MSE: mean((soft_rank(pred) - soft_rank(target))^2)。"""

    def __init__(self, regularization_strength: float = 1.0):
        super().__init__()
        self.reg_strength = float(regularization_strength)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = preds.flatten().unsqueeze(0)
        t = targets.flatten().unsqueeze(0).detach()

        p_rank = soft_rank(p, regularization_strength=self.reg_strength)
        t_rank = soft_rank(t, regularization_strength=self.reg_strength)
        return torch.mean((p_rank - t_rank) ** 2)


# ==============================================================================
# losses/combine.py  —  损失: 组合
# ==============================================================================

import torch
import torch.nn as nn

# 直接从子模块导入 (不经过包 __init__, 避免与 registry 的循环依赖)

__all__ = ["CombineLoss"]


class CombineLoss(nn.Module):
    """通用组合损失: 任意多种子损失的加权和 (权重固定, 不参与训练)。"""

    def __init__(self, loss_list, params=None):
        """
        Parameters
        ----------
        loss_list : list[dict]
            每个 dict: {'name': str, 'weight': float, ...该子损失的其他参数}
        params : dict | None
            全局默认参数字典 (可选), 子损失缺参时回退到此处的 c_* 键
        """
        super().__init__()
        self.loss_list = nn.ModuleList()
        self.weights = nn.ParameterList()
        self.loss_names = []

        for loss_cfg in loss_list:
            name = loss_cfg["name"]
            weight = loss_cfg.get("weight", 1.0)
            loss_params = {k: v for k, v in loss_cfg.items() if k not in ("name", "weight")}

            loss_fn = self._create_loss(name, loss_params, params)
            self.loss_list.append(loss_fn)
            self.weights.append(nn.Parameter(torch.tensor(float(weight)), requires_grad=False))
            self.loss_names.append(name)

    def _create_loss(self, name, loss_params, params=None):
        name_lower = name.lower()

        def get_param(key, default):
            if params and key in params:
                return params[key]
            return default

        if name_lower == "mse":
            return nn.MSELoss()
        elif name_lower == "huber":
            return nn.SmoothL1Loss()
        elif name_lower == "softspearman":
            pass  # (原为包内 import, 单文件版无需)
            return SoftSpearmanLoss(
                regularization_strength=get_param("c_spearman_reg", 1.0)
            )
        elif name_lower == "rankmse":
            pass  # (原为包内 import, 单文件版无需)
            return RankMSE(regularization_strength=get_param("c_spearman_reg", 1.0))
        elif name_lower == "powerhuber":
            return PowerHuberLoss(
                delta=loss_params.get("delta", get_param("c_delta", 1.0)),
                p=loss_params.get("p", get_param("c_p", 2.4)),
                force_fp32=True,
            )
        elif name_lower == "softhuber":
            return SoftPowerHuberLoss(
                delta=loss_params.get("delta", get_param("c_delta", 0.5)),
                p=loss_params.get("p", get_param("c_p", 2.2)),
                kappa=loss_params.get("kappa", get_param("c_kappa", 10)),
                force_fp32=True,
            )
        elif name_lower == "autohuber":
            return AutoDeltaPowerHuberLoss(
                p=loss_params.get("p", get_param("c_p", 2.2)),
                k=loss_params.get("k", get_param("c_k", 0.5)),
            )
        elif name_lower == "shuber":
            return StochasticPowerHuberLoss(
                p_min=loss_params.get("p_min", get_param("c_p_min", 2.1)),
                p_max=loss_params.get("p_max", get_param("c_p_max", 2.4)),
                delta_min=loss_params.get("delta_min", get_param("c_delta_min", 0.4)),
                delta_max=loss_params.get("delta_max", get_param("c_delta_max", 0.6)),
            )
        elif name_lower == "shuberplus":
            return StochasticPowerHuberLossPlus(**loss_params)
        elif name_lower == "madl":
            return MeanAbsoluteDirectionalLoss(
                mode=loss_params.get("mode", get_param("c_madl_mode", "soft")),
                alpha=loss_params.get("alpha", get_param("c_madl_alpha", 1.0)),
                reduction="mean",
            )
        elif name_lower == "dwmse":
            return DirectionalWeightedMSE()
        elif name_lower == "powermse":
            return PowerMSELoss(
                power=loss_params.get("power", get_param("c_power", 2.0)),
                reduction="mean",
            )
        raise NotImplementedError(f"Unsupported loss in CombineLoss: {name}")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for weight, loss_fn in zip(self.weights, self.loss_list):
            total_loss = total_loss + weight * loss_fn(preds, targets)
        return total_loss

    def get_loss_names(self):
        return self.loss_names


# ==============================================================================
# optim/base.py  —  优化: 策略基类 + 参数分组
# ==============================================================================

from abc import ABC, abstractmethod

import torch.nn as nn


def build_param_groups(model: nn.Module, weight_decay: float) -> list:
    """按维度分组做差异化 weight decay。

    ≥2 维参数 (矩阵乘/embedding) 施加 weight_decay; 1 维参数 (bias/LayerNorm) 不施加。
    抽自原 pipeline._init_optimizer, 供需要 weight decay 的策略复用。
    """
    grad_params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in grad_params if p.dim() >= 2]
    nondecay_params = [p for p in grad_params if p.dim() < 2]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nondecay_params, "weight_decay": 0.0},
    ]


class OptimStrategy(ABC):
    """优化策略抽象基类。

    生命周期: 构造 (吃 config 的 <type>_params) -> build (吃 model 与运行期上下文)
    -> 训练循环里反复 zero_grad / step。
    """

    def __init__(self, params: dict = None) -> None:
        self.params = params or {}

    @abstractmethod
    def build(
        self,
        model: nn.Module,
        *,
        C: int,
        steps_per_epoch: int,
        sub_epochs: int,
        epochs: int,
        total_steps: int,
        use_cuda: bool,
        verbose: bool = False,
    ) -> None:
        """构建内部 optimizer / scheduler。

        Parameters
        ----------
        model : nn.Module
            已搬到设备并包好 DDP/DP 的模型
        C : int
            模型有效输入特征数 (factor_mode 决定; Noam 默认用它当 model_size)
        steps_per_epoch : int
            每个 epoch 的 batch 数 (= len(dl_train)); warmup 跨度用得到
        sub_epochs, epochs : int
            sub-epoch 数与 epoch 数; 两段式调度的时钟
        total_steps : int
            本次训练实际会跑的总 step 数 (= min(epochs*steps_per_epoch, max_steps));
            Noam 策略默认按它的 1/5 算 warmup 步数 (见 noam.py)。
        use_cuda : bool
            是否在 CUDA 上 (决定 fused 等)
        verbose : bool
            是否打印诊断
        """
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """清空梯度"""
        ...

    @abstractmethod
    def step(self, *, batch_idx: int, is_sub_e: bool, sub_e_passed: int) -> None:
        """执行一步参数更新 + lr 调度。

        Parameters
        ----------
        batch_idx : int
            当前 epoch 内的 batch 下标 (从 0 起)
        is_sub_e : bool
            当前是否踩在 sub-epoch 边界
        sub_e_passed : int
            已经过的 sub-epoch 数 (含当前); 两段式调度据此切 warmup/anneal
        """
        ...

    def lr(self) -> float:
        """当前学习率 (用于日志/进度条); 默认取第一个 param group。"""
        return self.optimizer.param_groups[0]["lr"]


# ==============================================================================
# optim/noam.py  —  优化: Noam
# ==============================================================================

import torch.nn as nn
import torch.optim as optim



class NoamOpt:
    """Optimizer 包装器: 逐 step 按 Noam 公式改写 lr (原样迁移自 lj-code)。"""

    def __init__(self, model_size: int, factor: float, warmup: int, optimizer,
                 peak_lr: float = None, decay_power: float = 0.5) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        # peak_lr: 显式指定峰值 LR (解耦 factor/model_size); None 时回退原 noam 公式的峰值
        self.peak_lr = peak_lr
        # decay_power: 退火幂指数, 衰减段 lr ∝ (warmup/step)^p; p=0.5=原 noam, p 越大衰减越快
        self.decay_power = decay_power
        self._rate = 0

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        if step is None:
            step = self._step
        t = max(1, step)
        # 峰值: 显式 peak_lr 优先, 否则用原 noam 公式在 step=warmup 处的值
        peak = self.peak_lr if self.peak_lr is not None else (
            self.factor * self.model_size ** (-0.5) * self.warmup ** (-0.5)
        )
        # 线性 warmup 到峰值, 之后按 (warmup/t)^decay_power 衰减;
        # peak_lr=None & decay_power=0.5 时逐位等价原 noam 公式。
        return peak * min(t / self.warmup, (self.warmup / t) ** self.decay_power)


class NoamStrategy(OptimStrategy):
    """Noam 调度策略: 内层 Adam(lr=0) 由 NoamOpt 逐 batch 接管 lr。"""

    def build(
        self,
        model: nn.Module,
        *,
        C: int,
        steps_per_epoch: int,
        sub_epochs: int,
        epochs: int,
        total_steps: int,
        use_cuda: bool,
        verbose: bool = False,
    ) -> None:
        p = self.params
        # model_size: 显式配置优先, 否则用模型有效输入特征数 C (factor_mode 决定)
        model_size = p.get("model_size") or C
        factor = p.get("factor", 2)
        # warmup: 显式配置优先, 否则默认按本次训练总步数的 1/5 算 (不再是固定 300 步,
        # 这样不同规模的训练 (max_steps 不同) warmup 跨度会自动跟着缩放)
        warmup = p.get("warmup") or max(1, total_steps // 5)
        peak_lr = p.get("peak_lr")            # 显式峰值 LR (优先); None -> 用 factor/model_size 推
        decay_power = p.get("decay_power", 0.5)  # 退火幂; 0.5=原 noam, 越大衰减越快
        betas = tuple(p.get("betas", (0.9, 0.98)))
        eps = p.get("eps", 1e-9)
        weight_decay = p.get("weight_decay", 0.0)
        # 内层用 AdamW + 维度分组 (仅 2D 参数施 weight_decay, 与 warmup_cosine 一致);
        # weight_decay=0.0 时 AdamW 数值等价于原裸 Adam, 保持向后兼容。lr 从 0 起由 NoamOpt 接管。
        adam = optim.AdamW(
            build_param_groups(model, weight_decay), lr=0, betas=betas, eps=eps
        )
        self.opt = NoamOpt(model_size, factor, warmup, adam,
                           peak_lr=peak_lr, decay_power=decay_power)
        self.optimizer = adam  # 供基类 lr() 兜底 (NoamOpt.step 会写回 adam 的 lr)
        if verbose:
            peak = peak_lr if peak_lr is not None else factor * model_size ** (-0.5) * warmup ** (-0.5)
            print(f"[noam] peak_lr={peak:.3e} (指定={peak_lr}), warmup={warmup}, "
                  f"decay_power={decay_power}, wd={weight_decay}, betas={betas}, eps={eps}")

    def zero_grad(self) -> None:
        self.opt.zero_grad()

    def step(self, *, batch_idx: int, is_sub_e: bool, sub_e_passed: int) -> None:
        # Noam 逐 batch 步进, 内部改写 lr; sub-epoch 相关参数无用
        self.opt.step()

    def lr(self) -> float:
        return self.opt._rate


# ==============================================================================
# optim/plain_adam.py  —  优化: 固定 lr AdamW
# ==============================================================================

import torch.nn as nn
import torch.optim as optim



class PlainAdamStrategy(OptimStrategy):
    """固定学习率的 AdamW, 逐 batch 直接 step, 不理会 is_sub_e/sub_e_passed。"""

    def build(
        self,
        model: nn.Module,
        *,
        C: int,
        steps_per_epoch: int,
        sub_epochs: int,
        epochs: int,
        total_steps: int,
        use_cuda: bool,
        verbose: bool = False,
    ) -> None:
        p = self.params
        lr = p.get("lr", 0.001)
        betas = tuple(p.get("betas", (0.9, 0.999)))
        eps = p.get("eps", 1e-8)
        weight_decay = p.get("weight_decay", 0.0)
        self.optimizer = optim.AdamW(
            build_param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps
        )
        if verbose:
            print(f"[adam] lr={lr}, betas={betas}, eps={eps}, wd={weight_decay}")

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, *, batch_idx: int, is_sub_e: bool, sub_e_passed: int) -> None:
        self.optimizer.step()


# ==============================================================================
# optim/warmup_cosine.py  —  优化: warmup + cosine
# ==============================================================================

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR



class WarmupCosineStrategy(OptimStrategy):
    """两段式调度: 前 warmup_end 个 sub-epoch 指数 warmup, anneal_start 起余弦退火。"""

    def build(
        self,
        model: nn.Module,
        *,
        C: int,
        steps_per_epoch: int,
        sub_epochs: int,
        epochs: int,
        total_steps: int,
        use_cuda: bool,
        verbose: bool = False,
    ) -> None:
        p = self.params
        self.warmup_end = p["warmup_end"]
        self.warmup_base = p["warmup_base"]
        self.anneal_start = p["anneal_start"]

        # optimizer: 2D 参数施加 wd, 1D 不施加
        optim_groups = build_param_groups(model, p["weight_decay"])
        if verbose:
            n_decay = sum(x.numel() for x in optim_groups[0]["params"])
            n_nondecay = sum(x.numel() for x in optim_groups[1]["params"])
            print(f"number of decayed tensors {len(optim_groups[0]['params'])}, "
                  f"with {n_decay} parameters ")
            print(f"number of non-decayed tensors {len(optim_groups[1]['params'])}, "
                  f"with {n_nondecay} parameters")
        self.optimizer = optim.AdamW(
            optim_groups,
            lr=p["lr"],
            betas=tuple(p["betas"]),
            fused=p["fused"] if use_cuda else False,
        )

        # warmup: 首个 sub-epoch 内逐 batch 指数爬升 (lambda 同原 _init_scheduler)
        T_warmup = self.warmup_end * steps_per_epoch // sub_epochs
        self.warmup = LambdaLR(
            self.optimizer,
            lr_lambda=lambda t: self.warmup_base ** (t / T_warmup) / self.warmup_base,
        )
        # anneal: 按 sub-epoch 步进的余弦退火
        T_anneal = sub_epochs * epochs - self.anneal_start
        self.anneal = CosineAnnealingLR(
            self.optimizer, T_max=T_anneal, eta_min=p["anneal_eta_min"]
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, *, batch_idx: int, is_sub_e: bool, sub_e_passed: int) -> None:
        self.optimizer.step()
        # warmup 每 batch 步进 (仅在 warmup 阶段)
        if sub_e_passed < self.warmup_end:
            self.warmup.step()
        # anneal 每 sub-epoch 步进
        if is_sub_e and sub_e_passed >= self.anneal_start:
            self.anneal.step()


# ==============================================================================
# models/__init__.py  —  MODEL_REGISTRY (model.type -> 模型类)
# ==============================================================================

MODEL_REGISTRY = {
    "rwkv": RWKVModel,
    "rwkv_ts": RWKVTSModel,
    "mamba": MambaModel,
}


# ==============================================================================
# losses/__init__.py  —  LOSS_REGISTRY + build_loss
# ==============================================================================

import torch.nn as nn


LOSS_REGISTRY = {
    "mse": nn.MSELoss,
    "huber": nn.SmoothL1Loss,
    "powermse": PowerMSELoss,
    "powerhuber": PowerHuberLoss,
    "softhuber": SoftPowerHuberLoss,
    "autohuber": AutoDeltaPowerHuberLoss,
    "shuber": StochasticPowerHuberLoss,
    "shuberplus": StochasticPowerHuberLossPlus,
    "madl": MeanAbsoluteDirectionalLoss,
    "dwmse": DirectionalWeightedMSE,
    "softspearman": SoftSpearmanLoss,
    "rankmse": RankMSE,
    "combine": CombineLoss,
}


def build_loss(loss_type: str = "mse", loss_params: dict = None) -> nn.Module:
    """按 config 构建损失模块。

    Parameters
    ----------
    loss_type : str
        LOSS_REGISTRY 的键, by default "mse"
    loss_params : dict | None
        该损失的构造参数 (loss.<type>_params); None -> {}

    Returns
    -------
    nn.Module
        损失模块, 调用签名 (preds, targets) -> 标量
    """
    if loss_type not in LOSS_REGISTRY:
        raise KeyError(
            f"unknown loss.type '{loss_type}'; 可选: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[loss_type](**(loss_params or {}))


__all__ = [
    "LOSS_REGISTRY",
    "build_loss",
    "PowerMSELoss",
    "PowerHuberLoss",
    "SoftPowerHuberLoss",
    "AutoDeltaPowerHuberLoss",
    "StochasticPowerHuberLoss",
    "StochasticPowerHuberLossPlus",
    "MeanAbsoluteDirectionalLoss",
    "DirectionalWeightedMSE",
    "SoftSpearmanLoss",
    "RankMSE",
    "CombineLoss",
]


# ==============================================================================
# optim/__init__.py  —  OPTIM_REGISTRY
# ==============================================================================

OPTIM_REGISTRY = {
    "adam": PlainAdamStrategy,
    "warmup_cosine": WarmupCosineStrategy,
    "noam": NoamStrategy,
}

__all__ = [
    "OptimStrategy",
    "build_param_groups",
    "PlainAdamStrategy",
    "NoamStrategy",
    "WarmupCosineStrategy",
    "OPTIM_REGISTRY",
]


# ==============================================================================
# scripts/train.py  —  训练循环 (run_training / build_model_from_checkpoint)
# ==============================================================================

# -*- coding: utf-8 -*-
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import structlog
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

logger = structlog.get_logger()

# 单文件版: 本文件不在 code/scripts/ 下, 不能靠 __file__ 往上两级找仓库根,
# 也没有 code/ 需要加进 sys.path (所有实现都已内联)。
_TRAIN_HERE = os.path.dirname(os.path.abspath(__file__))   # 本文件所在目录


# checkpoint 根目录: 固定在仓库根目录下的 models/ (不是 code/models/ 那个 backbone
# 源码包), 不在 yaml 里配, 每次训练在其下新建一个带时间戳的独立子目录。
# 训练期间的原生 .pt 存档目录 (本文件同目录下的 models/); 提交用的是
# save_model 写出的 JSON, 这里只是训练过程的中间产物。
CHECKPOINT_ROOT = Path(_TRAIN_HERE) / "models"


# ==== DDP 启停 ====
def _ddp_env() -> bool:
    """torchrun 会注入 RANK/LOCAL_RANK/WORLD_SIZE 环境变量; 没有这些变量 (或
    WORLD_SIZE<=1) 时按单进程跑, 不初始化分布式。"""
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def _setup_ddp(timeout_minutes: int):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"  # 没有 CUDA 时退化成 gloo, 可在 CPU 上跑通分布式逻辑
    # 默认的 NCCL/gloo collective 超时只有 10 分钟——rank0 独自构建全历史全股票
    # 面板 (build_panel_arrays) 可能要几十分钟甚至更久 (取决于机器负载), 其余
    # rank 会一直卡在 rank0 发布共享内存之后的那次 broadcast 上等它, 不放宽这个
    # 超时会被 watchdog 误判成"卡死"直接杀掉整个进程组 (实测踩过这个坑)。
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _unwrap(model: nn.Module) -> nn.Module:
    """DDP 包过的模型取 .module, 保证 state_dict 的 key 和不用 DDP 时一致;
    评估时也用它拿到原始模块, 绕开 DDP 的 forward hook (纯前向不需要)。"""
    return model.module if isinstance(model, DistributedDataParallel) else model


def _winsorize_bounds(dataset, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """训练集有效 label 上取分位数用于裁剪极端值, 对齐 example.py 对 ytr 的 winsorize。
    dataset.label 已经是标准化过的 z-score, 这里取到的边界和训练循环里
    y.clamp(y_lo,y_hi) 用的是同一套单位。"""
    valid_label = dataset.label[dataset.y_mask]
    lo, hi = np.percentile(valid_label, [lo_pct, hi_pct])
    return float(lo), float(hi)


def _periodic_checkpoint_path(run_dir: Path, step: int) -> Path:
    """周期性 checkpoint 各自独立留一个文件: run_dir/stepN.pt, 不覆盖之前的存档点。"""
    return run_dir / f"step{step}.pt"


def _forward_masked_loss(model, loss_fn, X, y, X_mask, y_mask):
    """(B,N,L,F) 前向 + 截面 mask: 只在 X_mask & y_mask 为 True 的 (股票,锚点)
    位置参与损失。对 mse/huber/power*/shuber*/madl/dwmse 等 pointwise 损失完全
    等价于以前手写的 masked-mse; 对 softspearman/rankmse 这类要在同一截面内排序
    的损失, 这样展平会破坏"同一锚点内排序"的语义 (code/losses/ranking.py 的
    docstring 里也提到 batch_size>1 时需要按截面分组, 这里未处理, 见 run_training
    里的 warning)。batch 内没有任何有效样本时返回 (None, valid)。
    """
    pred = model(X, pad_mask=X_mask)
    valid = X_mask & y_mask
    if not valid.any():
        return None, valid
    return loss_fn(pred[valid], y[valid]), valid


def _evaluate(model, loader, device, loss_fn, y_lo: float, y_hi: float) -> float:
    """验证集上的平均 masked loss (纯评估, 不反传)。传入未包 DDP 的原始模块即可。"""
    model.eval()
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for X, y, X_mask, y_mask in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).clamp(y_lo, y_hi)
            X_mask = X_mask.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)
            loss, _ = _forward_masked_loss(model, loss_fn, X, y, X_mask, y_mask)
            if loss is not None:
                total += loss.item()
                n_batches += 1
    return total / max(n_batches, 1)


# ==== 模型存/读: 原生 .pt (torch.save/torch.load), 不再是 JSON ====
def save_checkpoint(payload: dict, path) -> str:
    """torch.save 原生支持 tensor/ndarray, 不需要像 JSON 版那样手动摊平。"""
    path = str(path)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """加载本地 .pt checkpoint。JSON 提交模型请用 load_finetune_checkpoint。"""
    return torch.load(path, map_location=map_location, weights_only=False)


def load_finetune_checkpoint(path: str, map_location="cpu") -> dict:
    """加载微调起点，支持本地 `.pt` 和提交侧文本 `.json` 两种格式。

    JSON 格式与 scripts/transformer_train.py::save_model 一致：state_dict 中每个
    张量按 dtype/shape/扁平 data 保存。返回值统一成普通 checkpoint dict，供
    run_training 的微调分支严格校验后加载；这里不依赖 scripts/ 目录，避免核心
    训练代码反向依赖提交脚本。
    """
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        state_dict = {}
        for name, meta in payload["state_dict"].items():
            dtype = getattr(torch, meta["dtype"])
            state_dict[name] = torch.tensor(meta["data"], dtype=dtype).reshape(meta["shape"]).to(map_location)
        payload["state_dict"] = state_dict
        return payload
    return load_checkpoint(path, map_location=map_location)


def _validate_finetune_checkpoint(ckpt: dict, model_type: str, model_cfg: dict, seq_len: int) -> None:
    """在开始数据构建/训练前验证微调起点的不可变契约，防止静默错配。"""
    required = {"state_dict", "model_type", "model_cfg", "feature_cols", "seq_len", "mean", "std", "y_mean", "y_std"}
    missing = sorted(required - set(ckpt))
    if missing:
        raise ValueError(f"微调 checkpoint 缺少必需字段: {missing}")
    if ckpt["model_type"] != model_type:
        raise ValueError(f"微调模型类型不匹配: checkpoint={ckpt['model_type']}, config={model_type}")
    if list(ckpt["feature_cols"]) != list(FEATURE_COLS):
        raise ValueError("微调 checkpoint 的 feature_cols 顺序与当前 FEATURE_COLS 不一致")
    if int(ckpt["seq_len"]) != seq_len:
        raise ValueError(f"微调序列长度不匹配: checkpoint={ckpt['seq_len']}, 当前={seq_len}")
    for name, expected in (("mean", N_FEAT), ("std", N_FEAT)):
        values = np.asarray(ckpt[name], dtype=np.float32)
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(f"微调 checkpoint 的 {name} 必须是长度 {expected} 的有限数组")
    if not np.isfinite(float(ckpt["y_mean"])) or not np.isfinite(float(ckpt["y_std"])) or float(ckpt["y_std"]) <= 0:
        raise ValueError("微调 checkpoint 的 y_mean/y_std 非法")
    expected_cfg = dict(model_cfg)
    expected_cfg.update(N=int(ckpt["model_cfg"]["N"]), L=seq_len, C=N_FEAT)
    if dict(ckpt["model_cfg"]) != expected_cfg:
        raise ValueError("微调 checkpoint 的完整 model_cfg 与当前配置不一致；微调不允许更换模型结构")



def build_model_from_checkpoint(ckpt: dict, map_location="cpu") -> nn.Module:
    """按 checkpoint 里的 model_type/model_cfg 重建 nn.Module 并加载权重,
    供推理端复用。"""
    model_cls = MODEL_REGISTRY[ckpt["model_type"]]
    model = model_cls(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(map_location)


# ==== 训练并持久化 ====
def run_training(cfg: Config, run_name: str) -> str:
    """按已加载好的 Config 从零训练, 把 权重+模型/loss/optim 类型+结构超参+标准化
    统计 一并存盘, 训练期间按总步数的 1/10 周期性存档。

    train_and_save() 的实际工作函数, 拆出来是为了让调用方可以先 load_config()
    再在内存里改几个字段做快速试验, 不用为此另外写一份 yaml。

    是否分布式由启动方式决定 (见文件头), 不是 cfg 里的字段。
    """
    assert cfg.model.type in MODEL_REGISTRY, (
        f"未知 model.type={cfg.model.type!r}, 可选: {sorted(MODEL_REGISTRY)}"
    )
    assert cfg.optim.type in OPTIM_REGISTRY, (
        f"未知 optim.type={cfg.optim.type!r}, 可选: {sorted(OPTIM_REGISTRY)}"
    )

    distributed = _ddp_env()
    if distributed:
        rank, local_rank, world_size = _setup_ddp(cfg.train.ddp_timeout_minutes)
    else:
        rank, local_rank, world_size = 0, 0, 1
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def _info(event, **kwargs):
        if is_main:
            logger.info(event, **kwargs)

    def _warn(event, **kwargs):
        if is_main:
            logger.warning(event, **kwargs)

    # 同一个 seed 保证所有 rank 初始化出一模一样的模型权重 (DDP 要求各 rank 起点一致)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    _info(
        "训练设备", device=str(device), distributed=distributed, world_size=world_size,
        model_type=cfg.model.type, loss_type=cfg.loss.type, optim_type=cfg.optim.type,
    )

    shm_handles = []  # 分布式 + 共享内存场景下需要在 finally 里清理
    try:
        # ---------- run_id: 各 rank 必须完全一致 (共享内存命名 + checkpoint 目录都要用) ----------
        if is_main:
            run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        else:
            run_id = None
        if distributed:
            obj = [run_id]
            dist.broadcast_object_list(obj, src=0)
            run_id = obj[0]
        run_dir = CHECKPOINT_ROOT / run_id
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
        _info("checkpoint 目录", run_dir=str(run_dir))

        ready_path = Path("/dev/shm") / f"bigquant_panel_{run_id}.ready"
        if is_main:
            ready_path.unlink(missing_ok=True)
        if distributed:
            dist.barrier()

        # ---------- 构建训练/验证集 ----------
        d = cfg.data
        is_finetune = cfg.finetune is not None
        anchor_slots = cfg.finetune.anchor_slots if is_finetune else None
        if is_finetune:
            if distributed:
                raise ValueError("微调模式暂不支持 DDP；请使用单进程 python 运行")
            if d.step != d.sample_interval:
                raise ValueError(
                    "微调指定每日 anchor_slots 时 data.step 必须等于 data.sample_interval；"
                    "最后三根 5m K 线应使用 step: 5"
                )
            if not os.path.isfile(cfg.finetune.checkpoint_path):
                raise FileNotFoundError(f"微调起点 checkpoint 不存在: {cfg.finetune.checkpoint_path}")
            parent_ckpt = load_finetune_checkpoint(cfg.finetune.checkpoint_path)
            _validate_finetune_checkpoint(
                parent_ckpt, cfg.model.type, cfg.model.params,
                d.lookback_days * (240 // d.sample_interval),
            )
            if d.train_keys is None and parent_ckpt.get("train_keys") is not None:
                d.train_keys = list(parent_ckpt["train_keys"])
                _info("微调复用 checkpoint 保存的股票轴", n_instruments=len(d.train_keys))
            stats_from_checkpoint = (
                np.asarray(parent_ckpt["mean"], dtype=np.float32),
                np.asarray(parent_ckpt["std"], dtype=np.float32),
                np.float32(parent_ckpt["y_mean"]),
                np.float32(parent_ckpt["y_std"]),
            )
            _info(
                "微调模式: 复用模型权重和训练期标准化统计",
                parent=cfg.finetune.checkpoint_path, anchor_slots=anchor_slots,
            )
        else:
            parent_ckpt = None
            stats_from_checkpoint = None
        _info(
            "构建 dense_dataloader",
            date_range=(d.train_start, d.val_end), train_end=d.train_end, val_start=d.val_start,
            sample_interval=d.sample_interval, lookback_days=d.lookback_days, step=d.step,
            n_keys=(len(d.train_keys) if d.train_keys is not None else "全历史"),
        )
        t0 = time.time()

        ready_path = Path("/dev/shm") / f"bigquant_panel_{run_id}.ready"
        if is_main:
            ready_path.unlink(missing_ok=True)
        if distributed:
            dist.barrier()

        if distributed:
            # rank0 是唯一构建者；其他 rank 在 host 上低频等待 ready 文件，避免在
            # rank0 进行几十分钟 CPU/内存密集工作时提前进入 NCCL broadcast。
            if is_main:
                # rank0 构建阶段是 CPU/内存密集操作；限制线程数避免 32 个构建线程
                # 和其他系统进程争抢内存带宽并放大 swap 压力。
                n_cpus = min(os.cpu_count() or 8, 16)
                for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    os.environ[var] = str(n_cpus)
                _info("rank0 开始构建面板", n_cpus_for_build=n_cpus)
                try:
                    arrays = build_panel_arrays(
                        date_range=(d.train_start, d.val_end), train_end=d.train_end, val_start=d.val_start,
                        sample_interval=d.sample_interval, lookback_days=d.lookback_days, step=d.step,
                        stride=d.stride,
                        keys=d.train_keys, horizon_days=d.horizon_days, show_progress=True, source=d.source,
                        anchor_slots=anchor_slots, cache_dir=d.cache_dir, cache_mode=d.cache_mode,
                    )
                    shm_bytes = sum(arrays[k].nbytes for k in ("X", "label", "y_mask", "x_mask_full"))
                    logger.info("共享内存段大小 (X+label+y_mask+x_mask_full)", total_gib=round(shm_bytes / 1024 ** 3, 3))
                    manifest, shm_handles = publish_panel(arrays, run_id=run_id)
                    ready_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
                    train_anchors_bc = arrays["train_anchors"]
                    val_anchors_bc = arrays["val_anchors"]
                    if cfg.train.include_val_in_train:
                        train_anchors_bc = np.concatenate((train_anchors_bc, val_anchors_bc))
                        val_anchors_bc = val_anchors_bc[:0]
                    broadcast_obj = [manifest, arrays["stats"], train_anchors_bc,
                                     val_anchors_bc, arrays["lookback_bars"]]
                except Exception as exc:
                    ready_path.write_text(json.dumps({"ok": False, "error": repr(exc)}), encoding="utf-8")
                    raise
            else:
                arrays = None
                broadcast_obj = [None, None, None, None, None]
                deadline = time.monotonic() + cfg.train.ddp_timeout_minutes * 60
                while not ready_path.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"等待 rank0 面板构建超时: {ready_path}")
                    time.sleep(2.0)
                status = json.loads(ready_path.read_text(encoding="utf-8"))
                if not status.get("ok"):
                    raise RuntimeError(f"rank0 面板构建失败: {status.get('error')}")

            # ready 文件只表示 rank0 已发布共享内存；此处 collective 顺序固定。
            dist.broadcast_object_list(broadcast_obj, src=0)
            manifest, stats, train_anchors, val_anchors, lookback_bars = broadcast_obj

            if is_main:
                X, label, y_mask, x_mask_full = (arrays["X"], arrays["label"],
                                                  arrays["y_mask"], arrays["x_mask_full"])
            else:
                shared, shm_handles = attach_panel(manifest)
                X, label, y_mask, x_mask_full = (shared["X"], shared["label"],
                                                  shared["y_mask"], shared["x_mask_full"])
            dist.barrier()

            mean, std, y_mean, y_std = stats
            train_ds = CrossSectionalWindowDataset(X, label, y_mask, x_mask_full, train_anchors, lookback_bars)
            val_ds = CrossSectionalWindowDataset(X, label, y_mask, x_mask_full, val_anchors, lookback_bars)

            # 训练集按 rank 分片; 验证只在 rank0 跑 (不分布式验证, 避免额外 all-reduce)
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed,
            )
            train_loader = DataLoader(
                train_ds, batch_size=cfg.train.batch_size, sampler=train_sampler,
                num_workers=cfg.train.num_workers,
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers,
            )
        else:
            train_sampler = None
            train_loader, val_loader, (mean, std, y_mean, y_std) = get_dataloaders(
                date_range=(d.train_start, d.val_end),
                train_end=d.train_end,
                val_start=d.val_start,
                sample_interval=d.sample_interval,
                lookback_days=d.lookback_days,
                step=d.step,
                stride=d.stride,
                keys=d.train_keys,
                horizon_days=d.horizon_days,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                stats=stats_from_checkpoint, source=d.source,
                anchor_slots=anchor_slots, cache_dir=d.cache_dir, cache_mode=d.cache_mode,
                include_val_in_train=cfg.train.include_val_in_train,
            )

            n_keys = train_loader.dataset.X.shape[0]
            if parent_ckpt is not None:
                expected_n = int(parent_ckpt["model_cfg"]["N"])
                if n_keys != expected_n:
                    raise ValueError(
                        f"微调股票轴不匹配: checkpoint N={expected_n}, 当前面板 N={n_keys}; "
                        "请使用与初始训练完全相同的 data.train_keys/source"
                    )
        # 实际入面板的股票数 (train_keys 与全历史交集), 即模型构造参数里的 N
        n_keys = train_loader.dataset.X.shape[0]
        bars_per_day = 240 // d.sample_interval
        seq_len = d.lookback_days * bars_per_day  # 喂给模型的回看长度 L
        steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum_steps)
        _info(
            "dataloader 构建完成",
            n_keys=n_keys, seq_len=seq_len, steps_per_epoch=steps_per_epoch,
            train_samples=len(train_loader.dataset), val_samples=len(val_loader.dataset),
            elapsed=round(time.time() - t0, 2),
        )

        y_lo, y_hi = _winsorize_bounds(train_loader.dataset)
        _info("label winsorize 边界 (1%/99%)", y_lo=round(y_lo, 6), y_hi=round(y_hi, 6))

        if cfg.loss.type in ("softspearman", "rankmse") and cfg.train.batch_size != 1:
            _warn(
                "loss.type 是截面排序损失但 batch_size != 1: soft_rank 会把这个 batch 里"
                "多个锚点(不同交易日)的股票混在一起排序, 语义不正确 (见 code/losses/ranking.py)",
                loss_type=cfg.loss.type, batch_size=cfg.train.batch_size,
            )

        # ---------- 建模型: models.MODEL_REGISTRY[model.type](N,L,C,**params) ----------
        model_cfg = dict(N=n_keys, L=seq_len, C=N_FEAT, **cfg.model.params)
        model = MODEL_REGISTRY[cfg.model.type](**model_cfg).to(device)
        if parent_ckpt is not None:
            model.load_state_dict(parent_ckpt["state_dict"], strict=True)
            _info("已加载微调起点权重", model_type=cfg.model.type)
        if distributed:
            ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
            model = DistributedDataParallel(model, **ddp_kwargs)
        _info(
            "可训练参数量", model_type=cfg.model.type,
            n_params=sum(p.numel() for p in model.parameters()),
        )

        # ---------- 总步数规划 (Noam 的默认 warmup 需要 total_steps, 必须先算) ----------
        total_steps = steps_per_epoch * cfg.train.epochs
        if cfg.train.max_steps is not None:
            total_steps = min(total_steps, cfg.train.max_steps)
        sub_epoch_steps = max(1, total_steps // max(1, cfg.train.sub_epochs))
        checkpoint_every = max(1, total_steps // 10)  # "总步数的 1/10" 存一次
        val_every = cfg.train.val_every_steps or sub_epoch_steps
        _info(
            "训练步数规划", total_steps=total_steps, sub_epoch_steps=sub_epoch_steps,
            checkpoint_every=checkpoint_every, val_every=val_every,
        )

        # ---------- 建 loss / optimizer 策略 ----------
        loss_fn = build_loss(cfg.loss.type, cfg.loss.params)
        strategy = OPTIM_REGISTRY[cfg.optim.type](params=cfg.optim.params)
        strategy.build(
            model, C=N_FEAT, steps_per_epoch=steps_per_epoch, sub_epochs=cfg.train.sub_epochs,
            epochs=cfg.train.epochs, total_steps=total_steps, use_cuda=(device.type == "cuda"),
            verbose=is_main,
        )

        def _save(path):
            payload = {
                "state_dict": _unwrap(model).state_dict(),
                "model_type": cfg.model.type,
                "model_cfg": model_cfg,
                "loss_type": cfg.loss.type,
                "optim_type": cfg.optim.type,
                "feature_cols": FEATURE_COLS,
                "seq_len": seq_len,
                "mean": np.asarray(mean, np.float32),
                "std": np.asarray(std, np.float32),
                "y_mean": float(y_mean),
                "y_std": float(y_std),
                "training_mode": "finetune" if parent_ckpt is not None else "train",
                "parent_checkpoint": cfg.finetune.checkpoint_path if parent_ckpt is not None else None,
                "anchor_slots": list(anchor_slots) if anchor_slots is not None else None,
                "data_source": d.source,
                "train_keys": list(d.train_keys) if d.train_keys is not None else None,
            }
            save_checkpoint(payload, path)

        # ---------- 训练循环 (按 global_step 驱动, 不是单纯的 epoch 循环) ----------
        # global_step 现在等于"优化器更新次数", 不是 micro-batch 数: 每
        # grad_accum_steps 个 micro-batch 才 zero_grad/clip/strategy.step 一次
        # (DDP 下非边界 micro-batch 用 model.no_sync() 跳过 all-reduce), 使
        # --nproc_per_node=W + grad_accum_steps=A 的有效全局 batch 和调度节奏
        # 等价于 --nproc_per_node=(W*A) + grad_accum_steps=1。
        accum = cfg.train.grad_accum_steps
        global_step = 0
        sub_e_passed = 0
        t_start = time.time()
        stop = False
        pbar = tqdm(total=total_steps, desc=f"训练[{run_id}]", disable=not is_main)
        for ep in range(cfg.train.epochs):
            if stop:
                break
            if distributed:
                train_sampler.set_epoch(ep)  # 保证每个 epoch 重新洗牌, 各 rank 分片不重叠
            model.train()
            n_micro = len(train_loader)
            group_size = None
            accum_loss_sum = 0.0
            accum_has_loss = False
            for micro_idx, (X, y, X_mask, y_mask) in enumerate(train_loader):
                if micro_idx % accum == 0:
                    strategy.zero_grad()
                    # 末尾若 micro-batch 数不是 accum 的整数倍, 最后一组按实际大小
                    # 缩放 (而不是固定除以 accum), 避免尾组梯度被系统性削弱。
                    group_size = min(accum, n_micro - micro_idx)
                    accum_loss_sum = 0.0
                    accum_has_loss = False
                is_boundary = (micro_idx % accum == group_size - 1) or (micro_idx == n_micro - 1)

                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).clamp(y_lo, y_hi)
                X_mask = X_mask.to(device, non_blocking=True)
                y_mask = y_mask.to(device, non_blocking=True)

                sync_ctx = (
                    model.no_sync() if (distributed and not is_boundary) else contextlib.nullcontext()
                )
                with sync_ctx:
                    loss, valid = _forward_masked_loss(model, loss_fn, X, y, X_mask, y_mask)
                    if loss is not None:
                        (loss / group_size).backward()
                        accum_loss_sum += loss.item()
                        accum_has_loss = True
                    else:
                        _warn("batch 内没有任何有效 (X_mask & y_mask) 样本, 跳过本 micro-batch",
                              global_step=global_step + 1, micro_idx=micro_idx)

                if not is_boundary:
                    continue

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                global_step += 1
                is_sub_e = (global_step % sub_epoch_steps == 0)
                if is_sub_e:
                    sub_e_passed += 1
                strategy.step(batch_idx=micro_idx, is_sub_e=is_sub_e, sub_e_passed=sub_e_passed)

                if is_main:
                    n_in_group = (micro_idx % accum) + 1
                    disp_loss = accum_loss_sum / n_in_group if accum_has_loss else None
                    pbar.set_postfix(loss=f"{disp_loss:.4f}" if disp_loss is not None else "n/a",
                                      lr=f"{strategy.lr():.2e}")
                    pbar.update(1)

                if cfg.train.enable_valid and is_main and global_step % val_every == 0:
                    val_loss = _evaluate(_unwrap(model), val_loader, device, loss_fn, y_lo, y_hi)
                    logger.info(
                        "验证", global_step=global_step, total_steps=total_steps,
                        val_loss=round(val_loss, 8), lr=round(strategy.lr(), 8),
                        elapsed=round(time.time() - t_start, 2),
                    )
                    model.train()

                if is_main and global_step % checkpoint_every == 0:
                    ckpt_path = _periodic_checkpoint_path(run_dir, global_step)
                    _save(ckpt_path)
                    logger.info("周期性 checkpoint 已保存", path=str(ckpt_path), global_step=global_step)

                if global_step >= total_steps:
                    stop = True
                    break
        pbar.close()

        # ---------- 持久化: 训练结束后的最终 checkpoint ----------
        final_path = run_dir / "final.pt"
        if is_main:
            _save(final_path)
        _info(
            "模型已保存 (纯本地研究产物, 不保证可直接提交)", path=str(final_path),
            model_type=cfg.model.type, total_steps=global_step,
            elapsed=round(time.time() - t_start, 2),
        )
        return str(final_path)
    finally:
        if distributed:
            dist.barrier()
            close_all(shm_handles, unlink=is_main)
            dist.destroy_process_group()


def train_and_save(config_path=DEFAULT_CONFIG_PATH, run_name: str = None) -> str:
    """CLI/最常见用法的入口: 读 yaml 配置, 训练, 存盘。

    run_name 不传时用配置里 output.run_name，再不设就用 config 文件名 (不含扩展名)。
    """
    cfg = load_config(config_path)
    if run_name is None:
        run_name = cfg.output.run_name or Path(config_path).stem
    return run_training(cfg, run_name)


# ==============================================================================
# backtest/predictions.py  —  推理: 评测区间 X/label/mask/anchors
# ==============================================================================

# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



def _eval_day_idx_bounds(time_grid: pl.DataFrame, lo, hi) -> Tuple[int, int]:
    """time_grid 里日期落在 [lo, hi] (闭区间) 的行对应的 day_idx 范围。

    跟 dense_dataloader.dataloader._eval_day_idx_bounds 是同一段逻辑, 那边是模块
    私有函数, 这里独立写一份 (两行), 不跨模块引用下划线开头的私有接口。
    """
    sub = time_grid.filter(pl.col("date").is_between(pl.lit(lo), pl.lit(hi)))
    return int(sub["day_idx"].min()), int(sub["day_idx"].max())


def build_eval_arrays(
    cfg, stats: Tuple, start: Optional[str] = None, end: Optional[str] = None, source: str = "local",
) -> dict:
    """构建评测区间 [start,end] 的 X/label/mask/anchors, 标准化用传入的 stats
    (训练时的 mean/std/y_mean/y_std)，不重新估计。

    输入:
        cfg: 训练该 checkpoint 时用的同一份 Config (决定 train_keys/
            sample_interval/lookback_days/step/horizon_days)。
        stats: (mean, std, y_mean, y_std)，来自 checkpoint。
        start/end: 评测窗口 (闭区间)；不传时默认 cfg.data.val_start~val_end。
        source: "local" (默认) 或 "online"，透传给 build_dense_panel/
            compute_forward_return_label，见 dense_dataloader.dataloader 的
            同名参数；提交模板 (scripts/example_predict.py) 用 "online"。
    返回 dict: X, label (已标准化), y_mask, x_mask_full, anchors, lookback_bars, keys, time_grid。
    """
    d = cfg.data
    mean, std, y_mean, y_std = stats

    bars_per_day = 240 // d.sample_interval
    lookback_bars = d.lookback_days * bars_per_day
    anchor_stride = d.stride if d.stride is not None else d.step // d.sample_interval
    anchor_slots = cfg.finetune.anchor_slots if cfg.finetune is not None else None
    day_stride = None
    if anchor_slots is not None:
        bars_per_day = 240 // d.sample_interval
        if anchor_stride % bars_per_day != 0:
            raise ValueError(f"指定 anchor_slots 时 stride={anchor_stride} 必须是 {bars_per_day} 的整数倍")
        anchor_stride = 1
        day_stride = (d.stride if d.stride is not None else d.step // d.sample_interval) // bars_per_day

    start = start or d.val_start or d.train_end
    end = end or d.val_end
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    # 缓冲区换算和 build_panel_arrays 完全一致: 前向缓冲保证回看窗口历史足够,
    # 后向缓冲保证区间末尾也能取到 horizon_days 天之后的 label (+10 而不是 +5,
    # 原因同 dataloader.py: slot 47 收盘锚点的 vwap_adj 取自次日开盘，多吃掉
    # 将近 1 个交易日的富余量)。
    back_buffer = pd.Timedelta(days=int(d.lookback_days * 1.6) + 10)
    fwd_buffer = pd.Timedelta(days=int(d.horizon_days * 1.6) + 10)
    fetch_range = (
        (start_ts - back_buffer).strftime("%Y-%m-%d"),
        (end_ts + fwd_buffer).strftime("%Y-%m-%d"),
    )

    panel = build_dense_panel(d.sample_interval, fetch_range, keys=d.train_keys, source=source)

    label, y_mask = compute_forward_return_label(panel, horizon_days=d.horizon_days, source=source)

    filled = fill_within_day(panel)
    vol_idxs = [FEATURE_COLS.index(c) for c in VOL_COLS]
    for i in vol_idxs:
        filled[:, :, i] = np.log1p(np.clip(filled[:, :, i], 0, None))

    X = np.nan_to_num((filled - mean) / std, nan=0.0).astype(np.float32)
    label_std = np.nan_to_num((label - y_mean) / y_std, nan=0.0).astype(np.float32)

    price_idxs = [FEATURE_COLS.index(c) for c in PRICE_COLS]
    still_missing = compute_still_missing(filled, price_idxs)
    x_mask_full = compute_x_mask_full(still_missing, lookback_bars)

    day_lo_idx, day_hi_idx = _eval_day_idx_bounds(panel.time_grid, start_ts, end_ts)
    anchors = build_anchors(
        panel.time_grid, day_lo_idx, day_hi_idx, lookback_bars, anchor_stride,
        anchor_slots=anchor_slots, stride_by_day=anchor_slots is not None, day_stride=day_stride,
    )
    if len(anchors) == 0:
        raise RuntimeError(f"build_eval_arrays: 评测区间锚点为空 ({start}~{end})")

    return {
        "X": X,
        "label": label_std,
        "y_mask": y_mask,
        "x_mask_full": x_mask_full,
        "anchors": anchors,
        "lookback_bars": lookback_bars,
        "keys": panel.keys,
        "time_grid": panel.time_grid,
    }


def build_prediction_table(
    cfg,
    model: torch.nn.Module,
    ckpt: dict,
    start: Optional[str] = None,
    end: Optional[str] = None,
    batch_size: int = 8,
    device: str = "cpu",
    show_progress: bool = True,
    source: str = "local",
) -> pd.DataFrame:
    """跑一遍评测区间的推理，返回长表 (ts, key, pred, label)。

    输入:
        cfg: 训练该 checkpoint 时用的**同一份** Config。
        model: build_model_from_checkpoint(ckpt) 构建好的模型 (已加载权重)。
        ckpt: load_checkpoint(...) 读到的原始 checkpoint dict (取
            mean/std/y_mean/y_std/feature_cols/seq_len/model_cfg 做一致性校验)。
        start/end: 评测窗口 (闭区间)；不传时默认用 cfg.data.val_start~val_end。
        batch_size: 推理时每批打包多少个锚点 (每个锚点自带全部股票)。
        source: "local" (默认) 或 "online"，透传给 build_eval_arrays。
    返回:
        pd.DataFrame，列 ts/key/pred/label，每行是某个锚点某只股票的
        (还原成真实收益率单位的预测值, 还原成真实收益率单位的训练 label)，
        只包含 X_mask 和 y_mask 都有效的 (股票, 锚点) —— 和训练时
        _forward_masked_loss 的 valid 判定完全一致。**这是为回测/自评估设计的
        (需要真实 label 才能算指标)；纯线上打分 (没有真实 label 可比对) 应该
        只按 X_mask 过滤，不要求 y_mask，见 scripts/example_predict.py 的做法。**
    """
    assert list(ckpt["feature_cols"]) == FEATURE_COLS, (
        "checkpoint 的特征列顺序和当前 dense_dataloader.config.FEATURE_COLS 不一致，"
        "很可能是用了不同版本的 dense_dataloader 训练/回测，预测会错位"
    )

    d = cfg.data
    bars_per_day = 240 // d.sample_interval
    lookback_bars = d.lookback_days * bars_per_day
    assert int(ckpt["seq_len"]) == lookback_bars, (
        f"checkpoint 的 seq_len={ckpt['seq_len']} 和当前 config 算出的 "
        f"lookback_bars={lookback_bars} 不一致，请确认 --config 传的是训练该 "
        "checkpoint 时用的同一份配置"
    )

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    y_mean, y_std = float(ckpt["y_mean"]), float(ckpt["y_std"])

    arrays = build_eval_arrays(cfg, (mean, std, y_mean, y_std), start=start, end=end, source=source)
    assert len(arrays["keys"]) == ckpt["model_cfg"]["N"], (
        f"面板股票数 {len(arrays['keys'])} 和 checkpoint 训练时的 "
        f"N={ckpt['model_cfg']['N']} 不一致，请确认 --config 里的 "
        "train_keys/train_keys_range 和训练时一致"
    )

    anchors = arrays["anchors"]
    anchor_ts = (
        arrays["time_grid"].filter(pl.col("t_idx").is_in(anchors.tolist()))
        .sort("t_idx")["date"].to_list()
    )
    t_to_ts = dict(zip(anchors.tolist(), anchor_ts))

    dataset = CrossSectionalWindowDataset(
        arrays["X"], arrays["label"], arrays["y_mask"], arrays["x_mask_full"],
        anchors, arrays["lookback_bars"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device).eval()
    keys = arrays["keys"]
    rows = []
    anchor_pos = 0
    with torch.no_grad():
        it = tqdm(loader, desc="回测推理", disable=not show_progress)
        for Xb, yb, Xm, ym in it:
            Xb_dev = Xb.to(device)
            Xm_dev = Xm.to(device)
            pred = model(Xb_dev, pad_mask=Xm_dev).cpu().numpy() * y_std + y_mean  # (B,N)，还原成真实收益率单位
            label_real = yb.numpy() * y_std + y_mean                              # (B,N)
            valid = (Xm.numpy() & ym.numpy())                                     # (B,N)
            b = pred.shape[0]
            for bi in range(b):
                t = int(anchors[anchor_pos + bi])
                ts = t_to_ts[t]
                for ki in np.flatnonzero(valid[bi]):
                    rows.append((ts, keys[ki], float(pred[bi, ki]), float(label_real[bi, ki])))
            anchor_pos += b

    if not rows:
        raise RuntimeError("build_prediction_table: 没有任何 (X_mask & y_mask) 都为 True 的样本")
    return pd.DataFrame(rows, columns=["ts", "key", "pred", "label"])


# ==============================================================================
# 提交层: 配置常量 + JSON 存读 + train_and_save / main
# ==============================================================================
# 训练区间与超参一律写死 (对齐 example_train.py), 切勿使用平台注入的测试区间训练。
#
# CONFIG_NAME 必须是**训练该 checkpoint 的那一份**配置: finetune 段的 anchor_slots
# 决定锚点落在日内哪些 slot 上。用预训练的 rwkv_256_softspearman.yaml (stride=3,
# 无 finetune 段) 会把锚点铺成全局每 3 根一个、每天最后一个是 slot 45 (14:50),
# 与本权重训练时的 slot 46/47 口径不符。
CONFIG_NAME = 'rwkv_256_softspearman_last2_finetune.yaml'
RUN_NAME = 'rwkv_256_softspearman_last2_finetune'

# 第一阶段 (从零预训练) 的配置: stride=3 全局铺锚点、Noam 调度、1500 步。
# CONFIG_NAME 那份带 finetune 段, 只是"从这个 checkpoint 接着训"的第二阶段;
# 平台从零重训时两段必须串起来跑, 见 run_training_chain / train_and_save。
PRETRAIN_CONFIG_NAME = 'rwkv_256_softspearman_all_excluding_2024q1_1gpu.yaml'
PRETRAIN_RUN_NAME = 'rwkv_256_softspearman_all_excluding_2024q1_1gpu'

# 训练好的权重 (文本类 JSON, 平台不接受 .pt 二进制)。默认与本文件同目录。
# 这一份就是"预训练 -> 微调"两段串行跑完后的最终产物: train_and_save 把第二段
# (微调) 的 final.pt 转成 JSON 写到这里, 推理侧 predict_scores 也从这里读,
# 训练侧写、推理侧读同一个常量, 不会出现"训完存 A、推理读 B"的漂移。
MODEL_NAME = 'final.json'
MODEL_PATH = Path(__file__).resolve().parent / MODEL_NAME

# 每天用哪个 slot 的回看窗口打分 (5m 数据一天 48 根, slot 47 = 15:00 收盘)
PREDICT_SLOT = 47

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# code/ 的管线需要三张表: bar5m (稠密面板) / bar1m (前瞻 VWAP label) /
# instruments (全历史股票池 + 交易日历), 三个都经 configure_tables() 生效。
DATASOURCES = {
    'bar5m': 'bigalpha_2026_stock_bar5m',
    'bar1m': 'bigalpha_2026_stock_bar1m',
    'instruments': 'bigalpha_2026_instruments',
}


def load_embedded_config(name=None):
    """按名字加载内联的 yaml 配置 (单文件版不依赖 code/configs/ 目录)。

    走的是 configs.py 原本的 yaml -> Config 解析路径 (_build_config), 只是原文
    来自 EMBEDDED_CONFIGS 而不是磁盘文件, 所以字段语义与本地训练完全一致。
    name 也可以是磁盘上的 yaml 路径 (本地研究时方便), 此时退回 load_config。
    """
    name = CONFIG_NAME if name is None else str(name)
    if name in EMBEDDED_CONFIGS:
        return _build_config(yaml.safe_load(EMBEDDED_CONFIGS[name]) or {})
    if os.path.isfile(name):
        return load_config(name)
    raise KeyError(f'未知配置 {name!r}; 内联可选: {sorted(EMBEDDED_CONFIGS)}')


# ==== 模型存/读: 一律用文本类文件 (JSON), 不使用 .pt 等二进制 ====
# 与 scripts/example_train.py::save_model / load_model 同一套格式: state_dict 里
# 每个张量转成 {dtype, shape, data(扁平 list)}, 其余字段原样写入; 加载时按
# dtype/shape 还原。训练侧写、推理侧读, 同一份定义, 不会漂移。
def _json_safe(value):
    """checkpoint 里的 tensor/ndarray/np 标量/Path 转成 JSON 可序列化对象。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_model(ckpt, model_path=None):
    """把 checkpoint 存成 JSON 文本文件 (格式同 example_train.py::save_model)。"""
    model_path = str(MODEL_PATH if model_path is None else model_path)
    tensors = {}
    for k, v in ckpt['state_dict'].items():
        t = v.detach().cpu()
        tensors[k] = {
            'dtype': str(t.dtype).replace('torch.', ''),   # 如 'float32'
            'shape': list(t.shape),
            'data': t.reshape(-1).tolist(),                # 扁平存, 加载时按 shape 还原
        }
    payload = {k: _json_safe(v) for k, v in ckpt.items() if k != 'state_dict'}
    payload['state_dict'] = tensors
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=None, map_location='cpu'):
    """读取 save_model 写出的 JSON, 把 state_dict 还原为张量 dict。"""
    model_path = str(MODEL_PATH if model_path is None else model_path)
    with open(model_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload['state_dict'].items():
        t = torch.tensor(meta['data'], dtype=getattr(torch, meta['dtype']))
        sd[k] = t.reshape(meta['shape']).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != 'state_dict'}
    ckpt['state_dict'] = sd
    return ckpt


def _load_online_config(config_name=None):
    """加载配置并切成线上口径 (训练/推理共用: 两侧数据源必须一致)。

    yaml 里 data.source=local 是本地研究用 (直接读 aligned parquet, 比现场
    dai.query 快得多); 平台隔离环境访问不到这些文件, 必须切成 online 现场查表。
    build_panel_arrays 不允许 online + 磁盘缓存, 故同时关缓存。
    """
    cfg = load_embedded_config(config_name)
    cfg.data.source = 'online'
    cfg.data.cache_mode = 'off'
    cfg.data.cache_dir = None
    return cfg


def print_config_summary(config_name=None, model_path=None) -> None:
    """训练/推理前把关键量摊开确认。"""
    cfg = load_embedded_config(config_name)
    mp = MODEL_PATH if model_path is None else model_path
    print('CONFIG =', CONFIG_NAME if config_name is None else config_name)
    print('model.type =', cfg.model.type, '| loss.type =', cfg.loss.type,
          '| optim.type =', cfg.optim.type)
    print('训练区间 =', cfg.data.train_start, '->', cfg.data.train_end)
    print('验证区间 =', cfg.data.val_start, '->', cfg.data.val_end)
    print('seq_len =', cfg.data.lookback_days * (240 // cfg.data.sample_interval))
    if cfg.finetune is not None:
        print('finetune.anchor_slots =', cfg.finetune.anchor_slots)
        pre = load_embedded_config(PRETRAIN_CONFIG_NAME)
        print('训练链 = 两段串行:', PRETRAIN_CONFIG_NAME,
              f'(max_steps={pre.train.max_steps}, optim={pre.optim.type})', '->',
              CONFIG_NAME if config_name is None else config_name,
              f'(max_steps={cfg.train.max_steps}, optim={cfg.optim.type})')
    print('PREDICT_SLOT =', PREDICT_SLOT, '(5m 面板一天 48 根, 47 = 15:00 收盘)')
    print('MODEL_PATH =', mp, '| exists =', Path(mp).exists())
    print('DEVICE =', DEVICE)


# ==== 训练侧: 预训练 -> 微调 两段串行 ====
def run_training_chain(
    pretrain_cfg, finetune_cfg,
    pretrain_run_name=None, finetune_run_name=None,
) -> str:
    """串行跑两次 run_training: 先从零预训练, 再拿它的产物做微调, 返回微调 .pt。

    两段之间唯一的传递就是权重文件路径: 第一段的 final.pt 覆写进
    finetune_cfg.finetune.checkpoint_path, 所以 yaml 里那个写死的本地父
    checkpoint 路径在链式调用下不生效 (平台隔离环境也没有那个文件)。

    第二段 run_training 的微调分支会自己做全部一致性校验 (model_type /
    model_cfg / feature_cols / seq_len / 股票轴 N), 并复用第一段存下的
    mean/std/y_mean/y_std——微调不重新估计标准化统计, 两段的量纲因此严格一致。

    本地研究用法 (不走 online, 直接吃 yaml 里的 local + 磁盘缓存):
        run_training_chain(
            load_embedded_config(PRETRAIN_CONFIG_NAME),
            load_embedded_config(CONFIG_NAME),
        )
    """
    if pretrain_cfg.finetune is not None:
        raise ValueError(
            '第一阶段配置不应带 finetune 段 (那会变成"微调的微调"); '
            f'请用 {PRETRAIN_CONFIG_NAME}'
        )
    if finetune_cfg.finetune is None:
        raise ValueError(f'第二阶段配置必须带 finetune 段; 请用 {CONFIG_NAME}')

    pretrain_run_name = pretrain_run_name or pretrain_cfg.output.run_name or PRETRAIN_RUN_NAME
    finetune_run_name = finetune_run_name or finetune_cfg.output.run_name or RUN_NAME

    logger.info('阶段 1/2: 从零预训练', run_name=pretrain_run_name,
                epochs=pretrain_cfg.train.epochs, max_steps=pretrain_cfg.train.max_steps,
                optim_type=pretrain_cfg.optim.type,
                grad_accum_steps=pretrain_cfg.train.grad_accum_steps)
    pretrain_pt = run_training(pretrain_cfg, pretrain_run_name)
    logger.info('阶段 1/2 完成', pretrain_pt=pretrain_pt)

    # 两段的面板 (X/label/mask, 几十 GiB 级) 不能同时留在内存里; 第一段的
    # dataloader/dataset 出了 run_training 作用域就没人引用了, 这里显式回收。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetune_cfg.finetune.checkpoint_path = pretrain_pt
    logger.info('阶段 2/2: 微调', run_name=finetune_run_name,
                parent=pretrain_pt, anchor_slots=finetune_cfg.finetune.anchor_slots,
                epochs=finetune_cfg.train.epochs, max_steps=finetune_cfg.train.max_steps,
                optim_type=finetune_cfg.optim.type)
    finetune_pt = run_training(finetune_cfg, finetune_run_name)
    logger.info('阶段 2/2 完成', finetune_pt=finetune_pt)
    return finetune_pt


# ==== 训练侧 (签名/返回值与 example_train.py::train_and_save 一致) ====
def train_and_save(datasources, model_path=None, config_name=None, run_name=None,
                   pretrain_config_name=None):
    """在写死的训练区间上从零训练, 把 权重 + 标准化统计 + 结构超参 存成 JSON。

    落盘的是**两段跑完之后**的模型 (预训练 -> 微调, 微调那段的 final.pt), 默认
    写到 MODEL_PATH (即本文件同目录下的 final.json); 第一段预训练的权重只作为
    第二段的起点, 不会覆盖这个文件。推理侧 predict_scores 读同一个 MODEL_PATH。

    公榜阶段平台不重训, 直接加载该文件推理; 私榜阶段平台调本函数从零重训,
    故训练逻辑保持可复现 (seed 固定在配置的 train.seed)。

    config_name (默认 CONFIG_NAME) 带 finetune 段时自动走两段串行:
    先按 pretrain_config_name (默认 PRETRAIN_CONFIG_NAME) 从零预训练, 再拿
    第一段的 final.pt 做微调, 见 run_training_chain。传 config_name 为无
    finetune 段的配置 (如 'rwkv_256_softspearman.yaml') 则退回单段训练。

    pretrain_config_name 传 False 可以强制单段: 只跑 config_name 那一段微调,
    父 checkpoint 用 yaml 里写死的本地路径 (仅本地研究可用, 平台上没有该文件)。
    """
    model_path = str(MODEL_PATH if model_path is None else model_path)
    run_name = RUN_NAME if run_name is None else run_name
    configure_tables(datasources)   # 平台注入的表名生效 (bar5m/bar1m/instruments)

    cfg = _load_online_config(config_name)
    logger.info('训练配置', config=CONFIG_NAME if config_name is None else config_name,
                source=cfg.data.source, model_type=cfg.model.type,
                loss_type=cfg.loss.type, optim_type=cfg.optim.type, seed=cfg.train.seed,
                is_finetune=cfg.finetune is not None)

    if cfg.finetune is not None and pretrain_config_name is not False:
        pre_name = PRETRAIN_CONFIG_NAME if pretrain_config_name is None else pretrain_config_name
        pre_cfg = _load_online_config(pre_name)
        logger.info('预训练配置', config=pre_name, source=pre_cfg.data.source,
                    model_type=pre_cfg.model.type, loss_type=pre_cfg.loss.type,
                    optim_type=pre_cfg.optim.type, seed=pre_cfg.train.seed)
        pt_path = run_training_chain(pre_cfg, cfg, finetune_run_name=run_name)
    else:
        pt_path = run_training(cfg, run_name)   # 完整训练循环, 产出原生 .pt

    # pt_path 是链式训练里**第二段 (微调) 的 final.pt**: run_training_chain 返回的
    # 就是微调那段的产物, 预训练那段的 .pt 只被当作起点传给第二段。
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    save_model(ckpt, model_path)                # 再转成平台要求的文本类文件
    logger.info('模型已保存, 请随本文件一并上传', path=model_path, source_pt=pt_path,
                training_mode=ckpt.get('training_mode'),
                parent_checkpoint=ckpt.get('parent_checkpoint'),
                anchor_slots=ckpt.get('anchor_slots'))
    return model_path


# ==== 推理侧 ====
def _check_checkpoint_matches_config(ckpt, cfg, config_name) -> int:
    """配置与 checkpoint 漂移时直接报错, 不静默算出错误预测。返回 bars_per_day。"""
    assert list(ckpt['feature_cols']) == list(FEATURE_COLS), (
        'checkpoint 的 feature_cols 与当前 dense_dataloader 不一致, 预测会错位')
    lookback = cfg.data.lookback_days * (240 // cfg.data.sample_interval)
    assert int(ckpt['seq_len']) == lookback, (
        f"checkpoint 的 seq_len={ckpt['seq_len']} 与 {config_name} 算出的 "
        f'lookback_bars={lookback} 不一致, 请确认用的是训练该 checkpoint 的同一份配置')

    # 打分用的 slot 必须在 checkpoint 训练时用过的 anchor_slots 里, 否则是在
    # 一个模型没见过的日内时刻上外推 (5m 面板一天 48 根, PREDICT_SLOT=47 是收盘)
    bars_per_day = 240 // cfg.data.sample_interval
    assert 0 <= PREDICT_SLOT < bars_per_day, (
        f'PREDICT_SLOT={PREDICT_SLOT} 超出日内 slot 范围 [0, {bars_per_day - 1}]')
    ckpt_slots = ckpt.get('anchor_slots')
    assert ckpt_slots and PREDICT_SLOT in list(ckpt_slots), (
        f'PREDICT_SLOT={PREDICT_SLOT} 不在 checkpoint 训练用的 anchor_slots='
        f'{ckpt_slots} 里, 模型没在该日内时刻上训练过')
    # 配置的 anchor_slots 决定 build_eval_arrays 会不会生成该 slot 的锚点
    cfg_slots = cfg.finetune.anchor_slots if cfg.finetune is not None else None
    assert cfg_slots is not None and PREDICT_SLOT in list(cfg_slots), (
        f'{config_name} 的 finetune.anchor_slots={cfg_slots} 不含 PREDICT_SLOT='
        f'{PREDICT_SLOT}, 请确认用的是训练该 checkpoint 的同一份配置')
    return bars_per_day


def _select_daily_anchors(arrays, start_date, end_date):
    """每个交易日只取 slot_idx == PREDICT_SLOT (47 = 15:00 收盘) 这一个锚点, 即
    "以当日收盘为窗口右端的回看窗口", 使输出粒度与 example 的"每天每只股票一个
    分数"一致 (内部锚点本身是日内多个采样时刻: 这里 anchor_slots=[46,47])。

    按 slot_idx 精确取, 不用"每天取最大 t_idx": 后者依赖锚点集合恰好以 47 收尾,
    换配置 (如 stride=3 的预训练配置, 每天最后一个锚点是 slot 45=14:50) 会静默
    打错时刻; 按 slot_idx 取则口径错时直接报空、不会算出错误分数。

    返回 (sel_day, sel_t); 没有锚点时返回 (None, None)。
    """
    anchors = np.asarray(arrays['anchors'], dtype=np.int64)
    grid = (arrays['time_grid']
            .filter(pl.col('t_idx').is_in(anchors.tolist()))
            .filter(pl.col('slot_idx') == PREDICT_SLOT)
            .sort('t_idx'))
    if grid.height == 0:
        logger.warning('区间内没有任何 slot 锚点, 返回空表',
                       slot=PREDICT_SLOT, start=str(start_date), end=str(end_date))
        return None, None
    daily = grid.with_columns(pl.col('date').dt.date().alias('day'))
    # slot_idx 在 time_grid 里对每个交易日唯一, 过滤后每天必然恰好一行
    assert daily['day'].n_unique() == daily.height, (
        f'每个交易日应恰好一个 slot {PREDICT_SLOT} 锚点, 实得 {daily.height} 行 / '
        f"{daily['day'].n_unique()} 天")
    sel_day = daily['day'].to_list()
    sel_t = daily['t_idx'].to_numpy().astype(np.int64)
    logger.info('打分锚点已选定', slot=PREDICT_SLOT, days=len(sel_day),
                first=str(sel_day[0]), last=str(sel_day[-1]))
    return sel_day, sel_t


def predict_scores(datasources, start_date, end_date, model_path=None, config_name=None):
    """加载已训练好的模型, 在样本外测试区间上推理打分。

    本函数**不训练**: 权重来自随本文件上传的 MODEL_PATH。start_date~end_date 为
    平台注入的【测试集区间】, 输出 ['date','instrument','score'] —— 与
    example_predict.py::main 完全一致。
    """
    model_path = Path(MODEL_PATH if model_path is None else model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f'未找到模型文件 {model_path}; 请先跑 train_and_save (预训练 -> 微调两段, '
            f'产出 {MODEL_NAME}), 再随本文件一起上传')

    configure_tables(datasources)
    cfg = _load_online_config(config_name)
    name = CONFIG_NAME if config_name is None else config_name

    ckpt = load_model(model_path, map_location=DEVICE)
    bars_per_day = _check_checkpoint_matches_config(ckpt, cfg, name)
    # 逐日打分: 每个交易日都要有 slot 47 的锚点, 不能按 day_stride 抽稀交易日
    cfg.data.stride = bars_per_day

    # 标准化统计量随权重存盘、此处直接复用, 不用测试区间自己的数据重新估计
    logger.info('构建测试集并预测', start=str(start_date), end=str(end_date))
    arrays = build_eval_arrays(
        cfg,
        (np.asarray(ckpt['mean'], np.float32), np.asarray(ckpt['std'], np.float32),
         np.float32(ckpt['y_mean']), np.float32(ckpt['y_std'])),
        start=str(start_date), end=str(end_date), source='online',
    )

    # model_cfg['N'] 只是 RWKVModel.forward 末尾 h.view(-1, N) 的 reshape 参数:
    # backbone 不接收 N (每只股票的窗口独立前向, 无跨股票混合), 所以 N 不影响
    # 参数量/state_dict 形状, 换个 N 后同一份权重仍能 strict 加载、逐值给出相同
    # 的单股票预测。这里按面板实际股票数覆盖它——训练时是本地股票轴 (2074),
    # 线上股票轴来自 instruments 表, 两者数量通常不同, 不覆盖会直接 reshape 报错。
    # 位置->股票的对应关系用面板自己返回的 arrays['keys'], 覆盖 N 不引入错位。
    n_keys = len(arrays['keys'])
    n_ckpt = int(ckpt['model_cfg']['N'])
    if n_keys != n_ckpt:
        logger.info('按面板股票数覆盖 model_cfg.N', n_checkpoint=n_ckpt, n_panel=n_keys)
        ckpt['model_cfg'] = {**ckpt['model_cfg'], 'N': n_keys}

    model = build_model_from_checkpoint(ckpt, map_location=DEVICE).eval()
    logger.info('已加载模型', path=str(model_path), device=str(DEVICE),
                model_type=ckpt['model_type'], N=ckpt['model_cfg']['N'])

    sel_day, sel_t = _select_daily_anchors(arrays, start_date, end_date)
    if sel_day is None:
        return pd.DataFrame(columns=['date', 'instrument', 'score'])

    L, keys = arrays['lookback_bars'], arrays['keys']
    rows = []
    with torch.no_grad():
        for day, t in zip(sel_day, sel_t):
            t = int(t)
            X = torch.from_numpy(arrays['X'][:, t - L + 1:t + 1]).unsqueeze(0).to(DEVICE)
            Xm = torch.from_numpy(arrays['x_mask_full'][:, t]).unsqueeze(0).to(DEVICE)
            pred = model(X, pad_mask=Xm).cpu().numpy()[0]           # (N,)
            # 训练时 label 做过 z-score, 这里还原成真实收益率量纲
            pred = pred * float(ckpt['y_std']) + float(ckpt['y_mean'])
            # 只输出回看窗口无残留缺失的股票 (未来 label 不参与筛选)
            for ki in np.flatnonzero(arrays['x_mask_full'][:, t]):
                rows.append((pd.Timestamp(day), keys[ki], float(pred[ki])))
    idx_df = pd.DataFrame(rows, columns=['date', 'instrument', 'score'])

    # ---------- 对齐中证 1000 + 规范输出 (与 example_predict.py 同一段逻辑) ----------
    stk = dai.query(f'SELECT date, instrument FROM {get_table("instruments")}',
                    filters={'date': [start_date, end_date]}).df()
    stk['date'] = pd.to_datetime(stk['date']).dt.normalize()   # 与 pd.Timestamp(day) 对齐
    result = (pd.merge(idx_df, stk, on=['date', 'instrument'], how='inner')
                .replace([np.inf, -np.inf], np.nan).dropna(subset=['score'])
                .drop_duplicates(['date', 'instrument'])[['date', 'instrument', 'score']]
                .reset_index(drop=True))
    logger.info('分数构建完成', rows=len(result), days=result['date'].nunique(),
                instruments=result['instrument'].nunique())
    return result


# 平台推理侧的契约函数名是 main(datasources, start_date, end_date)
main = predict_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigQuant 端到端提交 (自包含单文件版)')
    parser.add_argument('mode', nargs='?', default='train',
                        choices=['train', 'predict', 'summary'])
    parser.add_argument('--config', default=None, help=f'配置名 (默认 {CONFIG_NAME})')
    parser.add_argument('--pretrain-config', default=None,
                        help=f'第一阶段配置名 (默认 {PRETRAIN_CONFIG_NAME})')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='跳过第一阶段, 只跑 --config 那一段微调 (父 checkpoint 用 yaml 里写死的路径)')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--start', default='2024-01-01 00:00:00')
    parser.add_argument('--end', default='2024-12-31 23:59:59')
    _a = parser.parse_args()

    print_config_summary(_a.config, _a.model_path)
    if _a.mode == 'train':
        train_and_save(DATASOURCES, model_path=_a.model_path, config_name=_a.config,
                       pretrain_config_name=False if _a.no_pretrain else _a.pretrain_config)
    elif _a.mode == 'predict':
        _df = main(DATASOURCES, _a.start, _a.end,
                   model_path=_a.model_path, config_name=_a.config)
        print(_df.head())
        print('天数 =', _df['date'].nunique(), '| 股票数 =', _df['instrument'].nunique())
