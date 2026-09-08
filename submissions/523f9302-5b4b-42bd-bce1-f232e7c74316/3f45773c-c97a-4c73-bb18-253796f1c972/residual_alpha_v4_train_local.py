# Runtime dependencies are preinstalled in BigQuant: numpy, pandas, torch, dai.
# -*- coding: utf-8 -*-
"""
BigQuant 端到端 1分钟 V3 ResidualAlpha —— 官方 modelsave 模式训练/推理共享脚本。

提交时与唯一 Notebook、transformer_model.json 一并上传。

公开阶段：
    Notebook 调用 main(...)，本脚本只加载 transformer_model.json 推理。

私有阶段 / 复现：
    平台或审核方可调用 train_and_save(datasources) 从固定训练期重新训练。
    本脚本同时支持：
      1) BIGQUANT_E2E_LOCAL_DATA_ROOT 指向本地72个月 Feather；
      2) datasources["bar1m"] 指向云端 bigalpha_2026_stock_bar1m。

模型：
    风格抑制自适应字段归一化
    + DeepLOB式三档盘口空间编码
    + 多尺度时序卷积与跨日Shape Mixer
    + 稳定DeepSets横截面上下文
    + 排序/尾部/IC联合损失
    + 时间切分验证、EMA与尾段低学习率微调
"""
from __future__ import annotations

import os
import json
import math
import time
import random
import shutil
import hashlib
import tempfile
import base64
import gc
import signal
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
MODEL_PATH = str(_HERE / "residual_alpha_v4_model.json")

TRAIN_START = "2019-01-01"
TRAIN_END = "2024-12-31 23:59:59"
# 私榜由平台用不公开的训练集重训。写死区间要么取空、要么白丢数据，
# 打开后实际训练区间 = [TRAIN_START, TRAIN_END] ∩ 数据源真实可用区间。
ADAPT_TRAIN_RANGE_TO_DATA = True
# 重训时间保险：逼近预算就在当前 epoch 收尾并保存，宁可少训也不要被平台杀进程。
TRAIN_TIME_BUDGET_HOURS = float(os.environ.get("BIGQUANT_E2E_TRAIN_HOURS", "5.0"))

SEED = 42
# V3 到第 4 轮时验证集五个指标仍在单调上升，明显欠训练，默认上调到 6。
# 平台私榜重训受 6 小时限制，靠 TRAIN_TIME_BUDGET_HOURS 提前收尾兜底。
EPOCHS = int(os.environ.get("BIGQUANT_E2E_EPOCHS", "6"))
LOOKBACK_DAYS = 20
MINUTES_PER_DAY = 240
# 预测周期。改这里标签会自动重建（只重算日频收益，不动分钟特征缓存）。
HORIZONS = (1, 3, 5)
MAX_HORIZON = max(HORIZONS)
MIN_HISTORY_RATIO = 0.80
# 评估区间开头拿不到前置数据，训练时随机截断窗口让模型适应短历史输入。
SHORT_HISTORY_AUG_PROB = 0.25
# 按验证目标取前 K 个 epoch 的 EMA 权重做集成（K=1 等价于 V3 的行为）。
ENSEMBLE_TOP_K = int(os.environ.get("BIGQUANT_E2E_TOPK", "3"))
# 尾段微调只用验证集最后一段，前面一段保持干净用于选模型。
TAIL_ADAPTATION_ENABLED = os.environ.get("BIGQUANT_E2E_TAIL_ADAPT", "0") == "1"
TAIL_ADAPTATION_FRACTION = 0.40
# 选模同时奖励粗风格中性化后仍存活的IC；只影响epoch选择，不改变推理。
NEUTRAL_SELECTION_WEIGHT = float(os.environ.get("BIGQUANT_E2E_NEUTRAL_WEIGHT", "0.20"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("BIGQUANT_E2E_PATIENCE", "2"))
EARLY_STOPPING_MIN_DELTA = float(os.environ.get("BIGQUANT_E2E_MIN_DELTA", "0.0002"))
# 输出端跨日指数平滑（0 关闭）。只用过去的分数，不引入未来信息。
SCORE_SMOOTHING_ALPHA = 0.35

LEARNING_RATE = 2.5e-4
MIN_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 3e-4
WARMUP_RATIO = 0.08
GRAD_CLIP = 1.0
EMA_DECAY = 0.996
SAVE_EVERY_STEPS = 250
BATCH = 1  # 每个 step 是一个完整交易日横截面；保留官方模板变量名

PREDICTION_WEIGHTS = np.asarray([0.55, 0.30, 0.15], dtype=np.float32)


FEATURE_COLS = [
    "adjust_factor",
    "open", "high", "low", "close",
    "deal_number", "volume", "amount",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
OHLC_COLS = ["open", "high", "low", "close"]
BOOK_PRICE_COLS = [
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
SCALE_COLS = OHLC_COLS + ["amount"] + BOOK_PRICE_COLS
LOG_COLS = [
    "deal_number", "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
LOG_FEATURE_INDICES = [FEATURE_COLS.index(c) for c in LOG_COLS]

def _minute_slot(ts: pd.Series) -> np.ndarray:
    minute = ts.dt.hour * 60 + ts.dt.minute
    morning = minute - (9 * 60 + 31)
    afternoon = 120 + minute - (13 * 60 + 1)
    return np.where(
        minute.between(9 * 60 + 31, 11 * 60 + 30),
        morning,
        np.where(
            minute.between(13 * 60 + 1, 15 * 60),
            afternoon,
            -1,
        ),
    ).astype("int16")

def to_canonical(df: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    """本地压缩表和云端原始表收敛到完全一致的 canonical 表。"""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")

    if is_local:
        for c in OHLC_COLS:
            if c in out:
                out.loc[pd.to_numeric(out[c], errors="coerce").eq(-1), c] = np.nan
        for c in SCALE_COLS:
            if c in out:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32") / 100.0
        if "instrument_id" not in out:
            raise KeyError("本地表缺少 instrument_id")
        out["key"] = out["instrument_id"]
    else:
        # 云端价格已为元；只保留本地拥有的前三档。
        drop_cols = [
            c for c in out.columns
            if any(
                c.startswith(prefix) and c[-1:] in {"4", "5"}
                for prefix in (
                    "ask_price", "bid_price", "ask_volume", "bid_volume",
                    "ask_num_orders", "bid_num_orders",
                )
            )
        ]
        out = out.drop(columns=drop_cols, errors="ignore")
        if "instrument" not in out:
            raise KeyError("云端表缺少 instrument")
        out["key"] = out["instrument"].astype(str)

    for c in BOOK_PRICE_COLS:
        if c in out:
            out.loc[pd.to_numeric(out[c], errors="coerce").le(0), c] = np.nan

    missing = [c for c in FEATURE_COLS if c not in out]
    if missing:
        raise KeyError(f"缺少模型字段：{missing}")

    out["minute_slot"] = _minute_slot(out["date"])
    out["trade_date"] = out["date"].dt.normalize()
    out = out[out["minute_slot"].between(0, 239)].copy()
    return out

def transform_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    for j in LOG_FEATURE_INDICES:
        values[:, j] = np.log1p(np.clip(values[:, j], 0, None))
    return values

def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (values.astype(np.float32, copy=False) - mean) / np.maximum(std, 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class AttentiveTokenPool(nn.Module):
   """
   对可变数量的时序 token 做可学习加权汇总。
   """

   def __init__(
       self,
       dim: int,
       hidden_dim: int = 64,
   ):
       super().__init__()

       self.score = nn.Sequential(
           nn.LayerNorm(dim),
           nn.Linear(dim, hidden_dim),
           nn.GELU(),
           nn.Linear(hidden_dim, 1),
       )

   def forward(
       self,
       tokens: torch.Tensor,
   ) -> torch.Tensor:
       # tokens: (B, T, D)

       logits = self.score(tokens)

       weights = torch.softmax(
           logits,
           dim=1,
       )

       return (
           tokens * weights
       ).sum(dim=1)


class PatchConvBranch(nn.Module):
   """
   单个可学习时间尺度分支。

   patch_size=5 / 15 / 30
   只属于模型内部运算，不输出人工特征。
   """

   def __init__(
       self,
       n_features: int,
       branch_dim: int,
       patch_size: int,
   ):
       super().__init__()

       self.patch_embedding = nn.Sequential(
           nn.Conv1d(
               in_channels=n_features,
               out_channels=branch_dim,
               kernel_size=patch_size,
               stride=patch_size,
               padding=0,
           ),
           nn.GroupNorm(
               num_groups=8,
               num_channels=branch_dim,
           ),
           nn.GELU(),
       )

       # 深度可分离局部混合
       self.local_mixer = nn.Sequential(
           nn.Conv1d(
               branch_dim,
               branch_dim,
               kernel_size=3,
               padding=1,
               groups=branch_dim,
           ),
           nn.Conv1d(
               branch_dim,
               branch_dim,
               kernel_size=1,
           ),
           nn.GroupNorm(
               num_groups=8,
               num_channels=branch_dim,
           ),
           nn.GELU(),
       )

       self.pool = AttentiveTokenPool(
           dim=branch_dim,
       )

   def forward(
       self,
       x: torch.Tensor,
   ) -> torch.Tensor:
       # x: (B, F, 240)

       h = self.patch_embedding(x)

       h = h + self.local_mixer(h)

       # (B, D, T) -> (B, T, D)
       h = h.transpose(1, 2)

       return self.pool(h)


class MultiScaleIntradayEncoder(nn.Module):
   """
   一天240分钟原始序列
       ↓
   5 / 15 / 30分钟可学习Patch分支
       ↓
   每天一个隐藏状态
   """

   def __init__(
       self,
       n_features: int,
       branch_dim: int = 96,
       d_model: int = 192,
       dropout: float = 0.10,
   ):
       super().__init__()

       # 模型内部字段门控
       self.feature_gate = nn.Sequential(
           nn.Linear(
               n_features,
               64,
           ),
           nn.GELU(),
           nn.Linear(
               64,
               n_features,
           ),
           nn.Sigmoid(),
       )

       self.branch_5m = PatchConvBranch(
           n_features=n_features,
           branch_dim=branch_dim,
           patch_size=5,
       )

       self.branch_15m = PatchConvBranch(
           n_features=n_features,
           branch_dim=branch_dim,
           patch_size=15,
       )

       self.branch_30m = PatchConvBranch(
           n_features=n_features,
           branch_dim=branch_dim,
           patch_size=30,
       )

       # 尾盘原始状态保留一条直接通路
       self.last_minute_projection = nn.Sequential(
           nn.Linear(
               n_features,
               branch_dim,
           ),
           nn.GELU(),
       )

       self.output_projection = nn.Sequential(
           nn.LayerNorm(
               branch_dim * 4
           ),
           nn.Linear(
               branch_dim * 4,
               d_model,
           ),
           nn.GELU(),
           nn.Dropout(dropout),
       )

   def forward(
       self,
       x: torch.Tensor,
   ) -> torch.Tensor:
       """
       x:
           (S, L, 240, F)

       输出:
           (S, L, d_model)
       """

       n_stocks, n_days, n_minutes, n_features = (
           x.shape
       )

       # 用一天内的原始字段状态生成字段门控
       daily_context = x.mean(dim=2)

       gate = self.feature_gate(
           daily_context
       )

       x = x * (
           0.5
           + gate.unsqueeze(2)
       )

       flat = x.reshape(
           n_stocks * n_days,
           n_minutes,
           n_features,
       )

       # Conv1d 输入格式：(B,F,T)
       conv_input = flat.transpose(
           1,
           2,
       )

       representation_5m = self.branch_5m(
           conv_input
       )

       representation_15m = self.branch_15m(
           conv_input
       )

       representation_30m = self.branch_30m(
           conv_input
       )

       last_minute = (
           self.last_minute_projection(
               flat[:, -1, :]
           )
       )

       combined = torch.cat(
           [
               representation_5m,
               representation_15m,
               representation_30m,
               last_minute,
           ],
           dim=-1,
       )

       day_embedding = (
           self.output_projection(
               combined
           )
       )

       return day_embedding.reshape(
           n_stocks,
           n_days,
           -1,
       )

class TemporalMixerBlock(nn.Module):
   """
   使用深度卷积混合相邻交易日信息，再使用通道MLP。

   复杂度低于纯注意力，适合与Transformer组合。
   """

   def __init__(
       self,
       d_model: int,
       kernel_size: int = 5,
       expansion: int = 4,
       dropout: float = 0.10,
   ):
       super().__init__()

       self.norm_temporal = nn.LayerNorm(
           d_model
       )

       self.temporal_conv = nn.Conv1d(
           in_channels=d_model,
           out_channels=d_model,
           kernel_size=kernel_size,
           padding=kernel_size // 2,
           groups=d_model,
       )

       self.norm_channel = nn.LayerNorm(
           d_model
       )

       self.channel_mlp = nn.Sequential(
           nn.Linear(
               d_model,
               d_model * expansion,
           ),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(
               d_model * expansion,
               d_model,
           ),
           nn.Dropout(dropout),
       )

   def forward(
       self,
       x: torch.Tensor,
       day_mask: torch.Tensor,
   ) -> torch.Tensor:
       # x: (S,L,D)

       h = self.norm_temporal(x)

       h = self.temporal_conv(
           h.transpose(1, 2)
       ).transpose(1, 2)

       x = x + h

       x = x + self.channel_mlp(
           self.norm_channel(x)
       )

       return x * day_mask.unsqueeze(-1)


class CrossDayEncoder(nn.Module):

   def __init__(
       self,
       lookback_days: int,
       d_model: int = 192,
       n_heads: int = 6,
       n_transformer_layers: int = 2,
       n_mixer_layers: int = 2,
       dim_feedforward: int = 576,
       dropout: float = 0.10,
   ):
       super().__init__()

       self.position_embedding = nn.Parameter(
           torch.zeros(
               1,
               lookback_days,
               d_model,
           )
       )

       self.mixer_blocks = nn.ModuleList(
           [
               TemporalMixerBlock(
                   d_model=d_model,
                   kernel_size=5,
                   expansion=4,
                   dropout=dropout,
               )
               for _ in range(
                   n_mixer_layers
               )
           ]
       )

       transformer_layer = (
           nn.TransformerEncoderLayer(
               d_model=d_model,
               nhead=n_heads,
               dim_feedforward=dim_feedforward,
               dropout=dropout,
               activation="gelu",
               batch_first=True,
               norm_first=True,
           )
       )

       self.transformer = (
           nn.TransformerEncoder(
               transformer_layer,
               num_layers=n_transformer_layers,
               norm=nn.LayerNorm(d_model),
               enable_nested_tensor=False,
           )
       )

       self.fusion = nn.Sequential(
           nn.LayerNorm(
               d_model * 2
           ),
           nn.Linear(
               d_model * 2,
               d_model,
           ),
           nn.GELU(),
           nn.Dropout(dropout),
       )

       nn.init.normal_(
           self.position_embedding,
           mean=0.0,
           std=0.02,
       )

   def forward(
       self,
       day_embedding: torch.Tensor,
       day_mask: torch.Tensor,
   ) -> torch.Tensor:
       """
       day_embedding:
           (S,L,D)

       day_mask:
           (S,L)
       """

       h = (
           day_embedding
           + self.position_embedding[
               :,
               :day_embedding.shape[1],
               :,
           ]
       )

       h = h * day_mask.unsqueeze(-1)

       for block in self.mixer_blocks:
           h = block(
               h,
               day_mask,
           )

       padding_mask = ~day_mask.bool()

       h = self.transformer(
           h,
           src_key_padding_mask=padding_mask,
       )

       # 当前日应当为有效交易日
       final_state = h[:, -1, :]

       masked_sum = (
           h
           * day_mask.unsqueeze(-1)
       ).sum(dim=1)

       masked_count = (
           day_mask
           .sum(dim=1, keepdim=True)
           .clamp_min(1.0)
       )

       history_mean = (
           masked_sum / masked_count
       )

       return self.fusion(
           torch.cat(
               [
                   final_state,
                   history_mean,
               ],
               dim=-1,
           )
       )

class CrossSectionLatentAttention(nn.Module):

   def __init__(
       self,
       d_model: int = 192,
       n_heads: int = 6,
       n_latents: int = 12,
       dropout: float = 0.10,
   ):
       super().__init__()

       self.market_latents = nn.Parameter(
           torch.randn(
               1,
               n_latents,
               d_model,
           ) * 0.02
       )

       self.stock_norm_1 = nn.LayerNorm(
           d_model
       )

       self.latent_norm_1 = nn.LayerNorm(
           d_model
       )

       self.latents_read_stocks = (
           nn.MultiheadAttention(
               embed_dim=d_model,
               num_heads=n_heads,
               dropout=dropout,
               batch_first=True,
           )
       )

       self.latent_self_attention = (
           nn.MultiheadAttention(
               embed_dim=d_model,
               num_heads=n_heads,
               dropout=dropout,
               batch_first=True,
           )
       )

       self.latent_norm_2 = nn.LayerNorm(
           d_model
       )

       self.stock_norm_2 = nn.LayerNorm(
           d_model
       )

       self.stocks_read_latents = (
           nn.MultiheadAttention(
               embed_dim=d_model,
               num_heads=n_heads,
               dropout=dropout,
               batch_first=True,
           )
       )

       self.stock_ffn = nn.Sequential(
           nn.LayerNorm(d_model),
           nn.Linear(
               d_model,
               d_model * 3,
           ),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(
               d_model * 3,
               d_model,
           ),
           nn.Dropout(dropout),
       )

   def forward(
       self,
       stock_states: torch.Tensor,
   ) -> torch.Tensor:
       """
       stock_states:
           (S,D)

       当前batch只有一个交易日，因此加一个batch维：
           (1,S,D)
       """

       stocks = stock_states.unsqueeze(0)

       latents = self.market_latents.expand(
           stocks.shape[0],
           -1,
           -1,
       )

       stock_keys = self.stock_norm_1(
           stocks
       )

       latent_query = self.latent_norm_1(
           latents
       )

       latent_update, _ = (
           self.latents_read_stocks(
               query=latent_query,
               key=stock_keys,
               value=stock_keys,
               need_weights=False,
           )
       )

       latents = latents + latent_update

       latent_self_input = (
           self.latent_norm_2(
               latents
           )
       )

       latent_self_update, _ = (
           self.latent_self_attention(
               query=latent_self_input,
               key=latent_self_input,
               value=latent_self_input,
               need_weights=False,
           )
       )

       latents = (
           latents
           + latent_self_update
       )

       stock_query = self.stock_norm_2(
           stocks
       )

       stock_update, _ = (
           self.stocks_read_latents(
               query=stock_query,
               key=latents,
               value=latents,
               need_weights=False,
           )
       )

       stocks = stocks + stock_update

       stocks = stocks + self.stock_ffn(
           stocks
       )

       return stocks.squeeze(0)

class CompetitionE2EModel(nn.Module):

   def __init__(
       self,
       n_features: int,
       lookback_days: int,
       n_targets: int = 3,
       branch_dim: int = 96,
       d_model: int = 192,
       n_heads: int = 6,
       n_latents: int = 12,
       dropout: float = 0.10,
   ):
       super().__init__()

       self.intraday = (
           MultiScaleIntradayEncoder(
               n_features=n_features,
               branch_dim=branch_dim,
               d_model=d_model,
               dropout=dropout,
           )
       )

       self.cross_day = CrossDayEncoder(
           lookback_days=lookback_days,
           d_model=d_model,
           n_heads=n_heads,
           n_transformer_layers=2,
           n_mixer_layers=2,
           dim_feedforward=576,
           dropout=dropout,
       )

       self.cross_section = (
           CrossSectionLatentAttention(
               d_model=d_model,
               n_heads=n_heads,
               n_latents=n_latents,
               dropout=dropout,
           )
       )

       # 三个预测周期使用独立预测头
       self.target_heads = nn.ModuleList(
           [
               nn.Sequential(
                   nn.LayerNorm(
                       d_model
                   ),
                   nn.Linear(
                       d_model,
                       d_model,
                   ),
                   nn.GELU(),
                   nn.Dropout(dropout),
                   nn.Linear(
                       d_model,
                       1,
                   ),
               )
               for _ in range(n_targets)
           ]
       )

   def forward(
       self,
       x: torch.Tensor,
       day_mask: torch.Tensor,
   ) -> torch.Tensor:
       """
       x:
           (S,L,240,26)

       day_mask:
           (S,L)

       输出:
           (S,3)
       """

       day_embedding = self.intraday(x)

       day_embedding = (
           day_embedding
           * day_mask.unsqueeze(-1)
       )

       stock_state = self.cross_day(
           day_embedding,
           day_mask,
       )

       stock_state = self.cross_section(
           stock_state
       )

       outputs = [
           head(stock_state)
           for head in self.target_heads
       ]

       return torch.cat(
           outputs,
           dim=1,
       )


MODEL_CONFIG = {
   "n_features": 26,
   "lookback_days": 20,
   "n_targets": 3,
   "branch_dim": 96,
   "d_model": 192,
   "n_heads": 6,
   "n_latents": 12,
   "dropout": 0.10,
}


HORIZON_WEIGHTS = torch.tensor(
    [0.55, 0.30, 0.15],
    dtype=torch.float32,
)


def stable_zscore(
    values: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return (
        values - values.mean()
    ) / (
        values.std(
            unbiased=False
        ) + eps
    )


def robust_target_zscore(
    target: torch.Tensor,
) -> torch.Tensor:
    """
    只在Loss内部对标签做截面去极值和标准化。
    不会改变模型输入字段。
    """

    lower = torch.quantile(
        target.detach(),
        0.01,
    )

    upper = torch.quantile(
        target.detach(),
        0.99,
    )

    clipped = target.clamp(
        min=lower,
        max=upper,
    )

    return stable_zscore(
        clipped
    )


def pearson_ic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_centered = (
        prediction
        - prediction.mean()
    )

    target_centered = (
        target
        - target.mean()
    )

    covariance = (
        pred_centered
        * target_centered
    ).mean()

    denominator = torch.sqrt(
        pred_centered
        .pow(2)
        .mean()
        * target_centered
        .pow(2)
        .mean()
        + eps
    )

    return covariance / denominator


def weighted_pairwise_loss(
    prediction: torch.Tensor,
    target_z: torch.Tensor,
) -> torch.Tensor:
    """
    对当前截面全部股票对进行排序训练。

    512只股票时约26万个关系，
    A100可轻松处理。
    """

    prediction_difference = (
        prediction[:, None]
        - prediction[None, :]
    )

    target_difference = (
        target_z[:, None]
        - target_z[None, :]
    )

    n = prediction.shape[0]

    pair_mask = torch.triu(
        torch.ones(
            n,
            n,
            dtype=torch.bool,
            device=prediction.device,
        ),
        diagonal=1,
    )

    pair_mask &= (
        target_difference.abs()
        > 1e-6
    )

    selected_prediction = (
        prediction_difference[
            pair_mask
        ]
    )

    selected_target = (
        target_difference[
            pair_mask
        ]
    )

    direction = torch.sign(
        selected_target
    )

    # 收益差越大的股票对权重越高
    pair_weight = (
        selected_target
        .abs()
        .clamp(
            min=0.25,
            max=3.0,
        )
    )

    return (
        F.softplus(
            -direction
            * selected_prediction
        )
        * pair_weight
    ).mean()


def tail_margin_loss(
    prediction: torch.Tensor,
    target_z: torch.Tensor,
    tail_fraction: float = 0.10,
    margin: float = 0.75,
) -> torch.Tensor:
    n = prediction.shape[0]

    tail_count = max(
        8,
        int(
            n * tail_fraction
        ),
    )

    order = torch.argsort(
        target_z
    )

    bottom = order[
        :tail_count
    ]

    top = order[
        -tail_count:
    ]

    prediction_z = stable_zscore(
        prediction
    )

    predicted_gap = (
        prediction_z[top].mean()
        - prediction_z[bottom].mean()
    )

    return F.softplus(
        margin - predicted_gap
    )


def competition_multitask_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
):
    horizon_weights = (
        HORIZON_WEIGHTS
        .to(
            predictions.device
        )
    )

    total_loss = predictions.new_tensor(
        0.0
    )

    metric_records = {}

    for horizon_index in range(3):
        prediction = predictions[
            :,
            horizon_index,
        ]

        target = targets[
            :,
            horizon_index,
        ]

        target_z = robust_target_zscore(
            target
        )

        prediction_z = stable_zscore(
            prediction
        )

        ic = pearson_ic(
            prediction,
            target_z,
        )

        ic_loss = 1.0 - ic

        pair_loss = weighted_pairwise_loss(
            prediction_z,
            target_z,
        )

        tail_loss = tail_margin_loss(
            prediction,
            target_z,
        )

        huber_loss = F.smooth_l1_loss(
            prediction_z,
            target_z,
        )

        horizon_loss = (
            0.50 * ic_loss
            + 0.25 * pair_loss
            + 0.15 * tail_loss
            + 0.10 * huber_loss
        )

        total_loss = (
            total_loss
            + horizon_weights[
                horizon_index
            ]
            * horizon_loss
        )

        metric_records[
            f"ic_{horizon_index}"
        ] = ic.detach()

        metric_records[
            f"pair_{horizon_index}"
        ] = pair_loss.detach()

        metric_records[
            f"tail_{horizon_index}"
        ] = tail_loss.detach()

    metric_records["loss"] = total_loss

    return metric_records


# ---------------------------------------------------------------------------
# V3 architecture/loss override. The legacy V2 definitions above are retained
# only to keep the shared data/inference pipeline auditable; all training and
# inference below use ResidualAlpha V3.
# ---------------------------------------------------------------------------
from residual_alpha_v4_model import (
    CompetitionE2EModel as ResidualAlphaV4Model,
    MODEL_CONFIG as RESIDUAL_ALPHA_V4_CONFIG,
    competition_multitask_loss as residual_alpha_v4_loss,
    PREDICTION_WEIGHTS as RESIDUAL_ALPHA_V4_PREDICTION_WEIGHTS,
)

CompetitionE2EModel = ResidualAlphaV4Model
MODEL_CONFIG = dict(RESIDUAL_ALPHA_V4_CONFIG)
MODEL_CONFIG["n_targets"] = len(HORIZONS)
MODEL_CONFIG["lookback_days"] = LOOKBACK_DAYS
competition_multitask_loss = residual_alpha_v4_loss


def _default_horizon_weights(n: int) -> list[float]:
    """周期越长权重越低（沿用 V3 的 0.72/0.20/0.08 衰减形态）。"""
    raw = [0.72 * (0.28 ** i) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


if len(RESIDUAL_ALPHA_V4_PREDICTION_WEIGHTS) == len(HORIZONS):
    _prediction_weights = list(RESIDUAL_ALPHA_V4_PREDICTION_WEIGHTS)
else:
    # 换了预测周期数，权重按默认形态重建，同时同步模型侧的损失权重
    _prediction_weights = _default_horizon_weights(len(HORIZONS))
    import residual_alpha_v4_model as _v4_module
    _v4_module.HORIZON_WEIGHTS = torch.tensor(
        _default_horizon_weights(len(HORIZONS)), dtype=torch.float32
    )
    print(f"[config] 预测周期改为 {HORIZONS}，权重重建为 "
          f"{[round(w, 4) for w in _prediction_weights]}")

PREDICTION_WEIGHTS = np.asarray(_prediction_weights, dtype=np.float32)
PREDICTION_WEIGHTS = PREDICTION_WEIGHTS / max(float(PREDICTION_WEIGHTS.sum()), 1e-12)


class ModelEMA:
    def __init__(self, model, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(value)

    def apply(self, model):
        if self.backup is not None:
            raise RuntimeError("EMA 已应用但尚未恢复")
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self.backup is None:
            raise RuntimeError("没有 EMA 备份")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None

    def cpu_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_cpu_state_dict(self, state, model):
        current = model.state_dict()
        self.shadow = {
            k: v.to(device=current[k].device, dtype=current[k].dtype)
            for k, v in state.items()
        }



def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)



_TORCH_NUMPY_DTYPE = {
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "int8": np.dtype("i1"),
    "uint8": np.dtype("u1"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "bool": np.dtype("?"),
}


def _tensor_to_json(tensor: torch.Tensor) -> dict:
    """把张量以 base64 原始字节写入 JSON。

    与逐元素 list 相比，避免 json.load 创建数百万个 Python float，
    显著降低公榜推理的 CPU 内存峰值。产物仍是纯文本 JSON。
    """
    value = tensor.detach().cpu().contiguous()
    dtype_name = str(value.dtype).replace("torch.", "")
    if dtype_name == "bfloat16":
        value = value.float()
        dtype_name = "float32"
    if dtype_name not in _TORCH_NUMPY_DTYPE:
        raise ValueError(f"不支持的张量 dtype: {value.dtype}")
    array = value.numpy().astype(_TORCH_NUMPY_DTYPE[dtype_name], copy=False)
    encoded = base64.b64encode(array.tobytes(order="C")).decode("ascii")
    return {
        "dtype": dtype_name,
        "shape": list(array.shape),
        "encoding": "base64_raw",
        "data_b64": encoded,
    }


def _json_to_tensor(spec: dict, map_location: Any = "cpu") -> torch.Tensor:
    dtype_name = str(spec["dtype"])
    if dtype_name not in _TORCH_NUMPY_DTYPE:
        raise ValueError(f"不支持的torch dtype: {dtype_name}")

    if spec.get("encoding") == "base64_raw":
        raw_bytes = base64.b64decode(spec["data_b64"], validate=True)
        array = np.frombuffer(
            raw_bytes,
            dtype=_TORCH_NUMPY_DTYPE[dtype_name],
        ).reshape(spec["shape"]).copy()
        tensor = torch.from_numpy(array)
    else:
        # 兼容官方模板及旧版逐元素 list JSON。
        if not hasattr(torch, dtype_name):
            raise ValueError(f"不支持的torch dtype: {dtype_name}")
        tensor = torch.tensor(spec["data"], dtype=getattr(torch, dtype_name))
        tensor = tensor.reshape(spec["shape"])

    return tensor.to(map_location)


def save_model(payload: dict, path: str | os.PathLike = MODEL_PATH) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = payload.get("model_state_dicts") or [payload["model_state_dict"]]
    state = states[-1]
    serializable = {
        "format_version": 5,
        "weight_encoding": "base64_raw",
        "model_config": payload["model_config"],
        "feature_cols": list(payload["feature_cols"]),
        "log_cols": list(payload["log_cols"]),
        "mean": np.asarray(payload["mean"], dtype=np.float32).tolist(),
        "std": np.asarray(payload["std"], dtype=np.float32).tolist(),
        "prediction_weights": np.asarray(
            payload.get("prediction_weights", PREDICTION_WEIGHTS),
            dtype=np.float32,
        ).tolist(),
        "lookback_days": int(payload.get("lookback_days", LOOKBACK_DAYS)),
        "minutes_per_day": int(payload.get("minutes_per_day", MINUTES_PER_DAY)),
        "max_horizon": int(payload.get("max_horizon", MAX_HORIZON)),
        "seed": int(payload.get("seed", SEED)),
        "epochs": int(payload.get("epochs", EPOCHS)),
        "model_name": str(payload.get("model_name", "ResidualAlphaV4")),
        "horizons": list(payload.get("horizons", HORIZONS)),
        "n_snapshots": len(states),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "best_validation_objective": float(payload.get("best_validation_objective", float("nan"))),
        "train_date_start": str(payload.get("train_date_start", TRAIN_START)),
        "train_date_end": str(payload.get("train_date_end", TRAIN_END)),
        "validation_date_start": str(payload.get("validation_date_start", "")),
        "validation_date_end": str(payload.get("validation_date_end", "")),
        "training_history": payload.get("training_history", []),
        # V4.1只保存快照列表，避免把最后一份权重重复写入JSON。
        "state_dicts": [
            {name: _tensor_to_json(v) for name, v in one.items()}
            for one in states
        ],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)
    return str(path)


def load_model(
    path: str | os.PathLike = MODEL_PATH,
    map_location: Any = "cpu",
) -> dict:
    """加载单文件或拆分后的V4模型JSON。"""
    model_path = Path(path)

    with model_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)

    split_files = raw.pop(
        "state_dict_files",
        None,
    )

    if split_files:
        raw_states = []

        for filename in split_files:
            part_path = (
                model_path.parent
                / str(filename)
            )

            if not part_path.exists():
                raise FileNotFoundError(
                    f"找不到拆分权重文件：{part_path}"
                )

            with part_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                part = json.load(handle)

            part_states = part.get(
                "state_dicts"
            )

            if not isinstance(
                part_states,
                list,
            ):
                raise ValueError(
                    f"权重分片格式错误：{part_path}"
                )

            raw_states.extend(
                part_states
            )

    else:
        raw_states = raw.pop(
            "state_dicts",
            None,
        )

        single = raw.pop(
            "state_dict",
            None,
        )

        if raw_states is None:
            if single is None:
                raise KeyError(
                    "模型JSON同时缺少 "
                    "state_dicts 和 state_dict"
                )

            raw_states = [single]

    expected = int(
        raw.get(
            "n_snapshots",
            len(raw_states),
        )
    )

    if len(raw_states) != expected:
        raise ValueError(
            f"快照数量错误："
            f"读取到{len(raw_states)}，"
            f"预期{expected}"
        )

    raw["state_dicts"] = [
        {
            name: _json_to_tensor(
                spec,
                map_location=map_location,
            )
            for name, spec in one.items()
        }
        for one in raw_states
    ]

    raw["state_dict"] = (
        raw["state_dicts"][-1]
    )

    return raw


def convert_pt_to_json(
    pt_path: str | os.PathLike,
    json_path: str | os.PathLike = MODEL_PATH,
) -> str:
    artifact = torch.load(pt_path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "model_config", "feature_cols", "log_cols",
        "mean", "std", "prediction_weights", "lookback_days",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise KeyError(f"final_model.pt 缺少字段: {missing}")
    return save_model(artifact, json_path)

def _table_reference(datasources: Any):
    if isinstance(datasources, dict):
        for key in (
            "bar1m",
            "bigalpha_2026_stock_bar1m",
            "bigalpha_2026_e2e_bar1m",
            "stock_bar1m",
        ):
            if key in datasources:
                return datasources[key]
        if datasources:
            return next(iter(datasources.values()))
    return datasources


def _materialize_cloud(
    datasource: Any,
    *,
    start_date: str,
    end_date: str,
    columns: list[str],
    instruments: list[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """按官方 DAI filters 接口读取区间数据。

    推理阶段可额外传 instrument 列表，只拉取评估股票池，避免把无关股票
    的分钟数据带入 Python 内存。DataFrame / datasource.df() 路径也执行同样过滤。
    """
    if isinstance(datasource, pd.DataFrame):
        frame = datasource.copy()
        date_values = pd.to_datetime(frame["date"])
        mask = (
            (date_values >= pd.Timestamp(start_date))
            & (date_values <= pd.Timestamp(end_date))
        )
        if instruments is not None and "instrument" in frame:
            instrument_set = set(map(str, instruments))
            mask &= frame["instrument"].astype(str).isin(instrument_set)
        return frame.loc[mask, columns].copy()

    if hasattr(datasource, "df") and callable(datasource.df):
        frame = datasource.df()
        date_values = pd.to_datetime(frame["date"])
        mask = (
            (date_values >= pd.Timestamp(start_date))
            & (date_values <= pd.Timestamp(end_date))
        )
        if instruments is not None and "instrument" in frame:
            instrument_set = set(map(str, instruments))
            mask &= frame["instrument"].astype(str).isin(instrument_set)
        return frame.loc[mask, columns].copy()

    import dai

    table = str(datasource)
    filters = {"date": [start_date, end_date]}
    if instruments is not None:
        filters["instrument"] = list(map(str, instruments))
    return dai.query(
        f"SELECT {','.join(columns)} FROM {table} ORDER BY instrument,date",
        filters=filters,
    ).df()


def _month_ranges(start: str, end: str):
    start_ts = pd.Timestamp(start).normalize().replace(day=1)
    end_ts = pd.Timestamp(end)
    for month_start in pd.date_range(start_ts, end_ts, freq="MS"):
        month_end = min(month_start + pd.offsets.MonthEnd(1), end_ts)
        yield month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d 23:59:59")


def _parameter_groups(model: nn.Module) -> list[dict]:
    """把归一化权重、偏置和字段门控排除出 weight decay。

    V3 里 mix_logit 训练完几乎没动（0.741~0.760，初始 0.750），
    正是因为 AdamW 对它施加了 3e-4 的衰减，把它往 level 方向拖。
    这些参数本来就不该被衰减。
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        skip = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "mix_logit" in name
            or "output_scale" in name
            or "output_bias" in name
            or "norm" in name.lower()
        )
        (no_decay if skip else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _learning_rate(step: int, total_steps: int) -> float:
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    if step < warmup_steps:
        return LEARNING_RATE * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine


def _combine_prediction_heads(predictions: np.ndarray) -> np.ndarray:
    columns = []
    for j in range(predictions.shape[1]):
        x = predictions[:, j].astype(np.float64)
        scale = max(float(np.nanstd(x)), 1e-8)
        columns.append((x - float(np.nanmean(x))) / scale)
    weights = np.asarray(PREDICTION_WEIGHTS, dtype=np.float64)[: len(columns)]
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.column_stack(columns).dot(weights)


def _cross_section_zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)
    center = float(np.mean(values[finite]))
    scale = max(float(np.std(values[finite])), 1e-8)
    return np.where(finite, (values - center) / scale, 0.0)



class DenseSampler:
    def __init__(self, cache_dir: str | os.PathLike):
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        self.cache_dir = cache_dir
        self.dates = np.load(cache_dir / meta["dates_file"], allow_pickle=False)
        self.keys = np.load(cache_dir / meta["keys_file"], allow_pickle=False)
        self.shape = tuple(meta["shape"])
        feature_dtype = np.dtype(meta.get("feature_dtype", "float32"))
        self.features = np.memmap(
            cache_dir / meta["features_file"], mode="r",
            dtype=feature_dtype, shape=self.shape,
        )
        self.presence = np.memmap(
            cache_dir / meta["presence_file"], mode="r",
            dtype=np.uint8, shape=self.shape[:2],
        )
        self.horizons = list(meta.get("horizons", [1, 3, 5]))
        self.labels = np.memmap(
            cache_dir / meta["labels_file"], mode="r",
            dtype=np.float32,
            shape=(self.shape[0], self.shape[1], len(self.horizons)),
        )

    def date_pool(self) -> np.ndarray:
        return np.arange(
            LOOKBACK_DAYS - 1,
            len(self.dates) - MAX_HORIZON,
            dtype=np.int64,
        )


def _minimal_daily(frame: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if is_local:
        out.loc[pd.to_numeric(out["close"], errors="coerce").eq(-1), "close"] = np.nan
        out["close"] = pd.to_numeric(out["close"], errors="coerce").astype("float32") / 100.0
        out["key"] = out["instrument_id"]
    else:
        out["close"] = pd.to_numeric(out["close"], errors="coerce").astype("float32")
        out["key"] = out["instrument"].astype(str)
    out["minute_slot"] = _minute_slot(out["date"])
    out["trade_date"] = out["date"].dt.normalize()
    out = out[out["minute_slot"].between(0, 239)].copy()
    valid = out["close"].notna() & out["close"].gt(0)
    return (
        out.loc[valid, ["trade_date", "key", "date", "close", "adjust_factor"]]
        .sort_values(["trade_date", "key", "date"])
        .groupby(["trade_date", "key"], as_index=False)
        .tail(1)
    )


def _build_dense_cache_local(data_root: Path, cache_dir: Path) -> Path:
    files = sorted(data_root.glob("*.feather"))
    if not files:
        raise FileNotFoundError(f"{data_root} 下没有 feather 文件")
    date_set, key_set, daily_parts = set(), set(), []
    for file in files:
        probe = pd.read_feather(
            file,
            columns=["date", "instrument_id", "close", "adjust_factor"],
        )
        daily = _minimal_daily(probe, is_local=True)
        date_set.update(daily["trade_date"].unique().tolist())
        key_set.update(daily["key"].unique().tolist())
        daily_parts.append(daily)

    dates = np.asarray(sorted(pd.to_datetime(list(date_set))), dtype="datetime64[ns]")
    keys = np.asarray(sorted(key_set), dtype=np.int64)
    return _allocate_and_fill_cache(
        cache_dir=cache_dir,
        dates=dates,
        keys=keys,
        daily_parts=daily_parts,
        source_iter=[
            ("local", file, None, None)
            for file in files
        ],
        datasource=None,
    )


def _effective_train_range(datasource: Any) -> tuple[str, str]:
    """取 [TRAIN_START, TRAIN_END] 与数据源真实可用区间的交集。

    私榜平台会换一套不公开的训练集；写死区间要么取空，要么白扔掉可用数据。
    """
    if not ADAPT_TRAIN_RANGE_TO_DATA or datasource is None:
        return TRAIN_START, TRAIN_END
    try:
        import dai
        probe = dai.query(
            f"SELECT min(date) AS lo, max(date) AS hi FROM {datasource}"
        ).df()
        lo = pd.Timestamp(probe["lo"].iloc[0])
        hi = pd.Timestamp(probe["hi"].iloc[0])
    except Exception as error:
        print(f"[range] 探测数据区间失败，沿用写死区间: {error}")
        return TRAIN_START, TRAIN_END
    if pd.isna(lo) or pd.isna(hi):
        return TRAIN_START, TRAIN_END
    start = max(pd.Timestamp(TRAIN_START), lo).strftime("%Y-%m-%d")
    end = min(pd.Timestamp(TRAIN_END), hi).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[range] 实际训练区间 {start} ~ {end}")
    return start, end


def _fit_dates_to_disk(dates: np.ndarray, n_keys: int, cache_dir: Path) -> np.ndarray:
    """磁盘放不下整段历史时从最早的日期开始截，保留最近的部分。

    私榜环境磁盘未知；直接抛错会让整次重训作废，用较短训练期跑完远好过没有结果。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_date = n_keys * MINUTES_PER_DAY * len(FEATURE_COLS) * 4
    free = shutil.disk_usage(cache_dir).free
    affordable = int(free * 0.90) // max(per_date, 1)
    if affordable >= len(dates):
        return dates
    keep = max(int(affordable), LOOKBACK_DAYS + MAX_HORIZON + 300)
    if keep >= len(dates):
        return dates
    print(f"[disk] 空间只够 {affordable} 个交易日，训练期从 {len(dates)} 天截到最近 {keep} 天")
    return dates[-keep:]


def _daily_close_panel(
    dates: np.ndarray,
    keys: np.ndarray,
    *,
    local_root: Path | None,
    datasource: Any,
) -> pd.DataFrame:
    """只读 date/close/adjust_factor 拼出日频复权收盘价面板（几分钟就能跑完）。"""
    parts = []
    if local_root is not None:
        for file in sorted(Path(local_root).glob("*.feather")):
            probe = pd.read_feather(
                file, columns=["date", "instrument_id", "close", "adjust_factor"]
            )
            parts.append(_minimal_daily(probe, is_local=True))
    else:
        range_start, range_end = _effective_train_range(datasource)
        for sd, ed in _month_ranges(range_start, range_end):
            probe = _materialize_cloud(
                datasource, start_date=sd, end_date=ed,
                columns=["date", "instrument", "close", "adjust_factor"],
            )
            parts.append(_minimal_daily(probe, is_local=False))

    daily = pd.concat(parts, ignore_index=True)
    is_text = keys.dtype.kind in {"U", "S"}
    daily["key_norm"] = daily["key"].astype(str) if is_text else daily["key"]
    daily["adjusted_close"] = (
        pd.to_numeric(daily["close"], errors="coerce")
        * pd.to_numeric(daily["adjust_factor"], errors="coerce")
    )
    panel = daily.pivot_table(
        index="trade_date", columns="key_norm", values="adjusted_close", aggfunc="last"
    )
    columns = [str(v) for v in keys.tolist()] if is_text else keys.tolist()
    return panel.reindex(index=pd.to_datetime(dates), columns=columns)


def rebuild_labels_if_needed(
    cache_dir: Path, *, local_root: Path | None, datasource: Any
) -> None:
    """缓存里的 horizons 与当前配置不一致时只重算标签，分钟特征原样复用。"""
    manifest_path = cache_dir / "manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (list(meta.get("horizons", [1, 3, 5])) == list(HORIZONS)
            and (cache_dir / meta["labels_file"]).exists()):
        return

    dates = np.load(cache_dir / meta["dates_file"], allow_pickle=False)
    keys = np.load(cache_dir / meta["keys_file"], allow_pickle=False)
    tag = "-".join(str(h) for h in HORIZONS)
    label_path = cache_dir / f"labels_f32_h{tag}.dat"
    print(f"[cache] 预测周期变为 {HORIZONS}，重建标签 -> {label_path.name}")

    panel = _daily_close_panel(
        dates, keys, local_root=local_root, datasource=datasource
    )
    labels = np.memmap(
        label_path, mode="w+", dtype=np.float32,
        shape=(len(dates), len(keys), len(HORIZONS)),
    )
    labels[:] = np.nan
    for target_idx, horizon in enumerate(HORIZONS):
        labels[:, :, target_idx] = (
            panel.shift(-horizon).div(panel).sub(1.0).to_numpy(dtype=np.float32)
        )
    labels.flush()
    meta["horizons"] = list(HORIZONS)
    meta["labels_file"] = label_path.name
    manifest_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_dense_cache_cloud(datasource: Any, cache_dir: Path) -> Path:
    date_set, key_set, daily_parts = set(), set(), []
    range_start, range_end = _effective_train_range(datasource)
    ranges = list(_month_ranges(range_start, range_end))
    for sd, ed in ranges:
        probe = _materialize_cloud(
            datasource,
            start_date=sd,
            end_date=ed,
            columns=["date", "instrument", "close", "adjust_factor"],
        )
        daily = _minimal_daily(probe, is_local=False)
        date_set.update(daily["trade_date"].unique().tolist())
        key_set.update(daily["key"].unique().tolist())
        daily_parts.append(daily)

    dates = np.asarray(sorted(pd.to_datetime(list(date_set))), dtype="datetime64[ns]")
    max_len = max((len(str(k)) for k in key_set), default=16)
    keys = np.asarray(sorted(str(k) for k in key_set), dtype=f"<U{max_len}")
    dates = _fit_dates_to_disk(dates, len(keys), cache_dir)
    return _allocate_and_fill_cache(
        cache_dir=cache_dir,
        dates=dates,
        keys=keys,
        daily_parts=daily_parts,
        source_iter=[
            ("cloud", None, sd, ed)
            for sd, ed in ranges
        ],
        datasource=datasource,
    )


def _allocate_and_fill_cache(
    *,
    cache_dir: Path,
    dates: np.ndarray,
    keys: np.ndarray,
    daily_parts: list[pd.DataFrame],
    source_iter: list[tuple],
    datasource: Any,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_dates, n_keys = len(dates), len(keys)
    shape = (n_dates, n_keys, MINUTES_PER_DAY, len(FEATURE_COLS))
    feature_path = cache_dir / "features_f32.dat"
    presence_path = cache_dir / "presence_u8.dat"
    label_path = cache_dir / "labels_f32.dat"
    expected = int(np.prod(shape, dtype=np.int64)) * 4
    if shutil.disk_usage(cache_dir).free < int(expected * 1.08):
        raise OSError(
            f"缓存需要约 {expected/1024**3:.1f} GiB，当前磁盘空间不足；"
            f"请缩短 TRAIN_START 或改用更低频的行情表"
        )

    np.save(cache_dir / "dates.npy", dates)
    np.save(cache_dir / "keys.npy", keys)
    features = np.memmap(feature_path, mode="w+", dtype=np.float32, shape=shape)
    presence = np.memmap(
        presence_path, mode="w+", dtype=np.uint8, shape=(n_dates, n_keys)
    )
    labels = np.memmap(
        label_path, mode="w+", dtype=np.float32,
        shape=(n_dates, n_keys, len(HORIZONS)),
    )
    features[:] = np.nan
    presence[:] = 0
    labels[:] = np.nan

    date_map = {pd.Timestamp(v): i for i, v in enumerate(dates)}
    key_map = {str(v) if keys.dtype.kind in {"U", "S"} else v: i for i, v in enumerate(keys.tolist())}
    local_columns = ["date", "instrument_id", *FEATURE_COLS]
    cloud_columns = ["date", "instrument", *FEATURE_COLS]

    for source_type, file, sd, ed in source_iter:
        if source_type == "local":
            raw = pd.read_feather(file, columns=local_columns)
            frame = to_canonical(raw, is_local=True)
        else:
            raw = _materialize_cloud(
                datasource,
                start_date=sd,
                end_date=ed,
                columns=cloud_columns,
            )
            frame = to_canonical(raw, is_local=False)

        values = transform_features(frame)
        di = frame["trade_date"].map(date_map).to_numpy()
        key_values = frame["key"].astype(str) if keys.dtype.kind in {"U", "S"} else frame["key"]
        ki = key_values.map(key_map).to_numpy()
        mi = frame["minute_slot"].to_numpy()
        valid = pd.notna(di) & pd.notna(ki) & (mi >= 0) & (mi < MINUTES_PER_DAY)
        di = di[valid].astype(int)
        ki = ki[valid].astype(int)
        mi = mi[valid].astype(int)
        features[di, ki, mi, :] = values[valid]
        presence[di, ki] = 1
        features.flush()
        presence.flush()

    daily = pd.concat(daily_parts, ignore_index=True)
    daily["key_norm"] = (
        daily["key"].astype(str) if keys.dtype.kind in {"U", "S"} else daily["key"]
    )
    daily["adjusted_close"] = (
        pd.to_numeric(daily["close"], errors="coerce")
        * pd.to_numeric(daily["adjust_factor"], errors="coerce")
    )
    panel = daily.pivot(
        index="trade_date", columns="key_norm", values="adjusted_close"
    )
    panel_columns = [str(v) for v in keys.tolist()] if keys.dtype.kind in {"U", "S"} else keys.tolist()
    panel = panel.reindex(index=pd.to_datetime(dates), columns=panel_columns)
    for target_idx, horizon in enumerate(HORIZONS):
        target = panel.shift(-horizon).div(panel).sub(1.0)
        labels[:, :, target_idx] = target.to_numpy(dtype=np.float32)
    labels.flush()

    manifest = {
        "shape": list(shape),
        "horizons": list(HORIZONS),
        "dates_file": "dates.npy",
        "keys_file": "keys.npy",
        "features_file": feature_path.name,
        "presence_file": presence_path.name,
        "labels_file": label_path.name,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_dir


def _compute_stats(sampler: DenseSampler) -> tuple[np.ndarray, np.ndarray]:
    n_features = sampler.shape[-1]
    count = np.zeros(n_features, dtype=np.int64)
    total = np.zeros(n_features, dtype=np.float64)
    total_sq = np.zeros(n_features, dtype=np.float64)
    for date_idx in range(sampler.shape[0]):
        x = np.asarray(sampler.features[date_idx], dtype=np.float32).reshape(-1, n_features)
        finite = np.isfinite(x)
        safe = np.where(finite, x, 0.0)
        count += finite.sum(axis=0)
        total += safe.sum(axis=0, dtype=np.float64)
        total_sq += (safe.astype(np.float64) ** 2).sum(axis=0)
    mean = total / np.maximum(count, 1)
    var = total_sq / np.maximum(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def _cross_section(
    sampler: DenseSampler,
    date_idx: int,
    mean: np.ndarray,
    std: np.ndarray,
    require_labels: bool = True,
    rng: np.random.Generator | None = None,
):
    start = date_idx - LOOKBACK_DAYS + 1
    current = sampler.presence[date_idx].astype(bool)
    history = sampler.presence[start:date_idx + 1].sum(axis=0)
    eligible = current & (
        history >= int(math.ceil(LOOKBACK_DAYS * MIN_HISTORY_RATIO))
    )
    if require_labels:
        all_y = np.asarray(sampler.labels[date_idx], dtype=np.float32)
        eligible &= np.isfinite(all_y).all(axis=1)
    selected = np.flatnonzero(eligible)
    if len(selected) < 100:
        raise RuntimeError(
            f"{pd.Timestamp(sampler.dates[date_idx]).date()} 有效股票仅 {len(selected)}"
        )
    x = np.asarray(
        sampler.features[start:date_idx + 1, selected], dtype=np.float32
    ).transpose(1, 0, 2, 3)
    mask = np.asarray(
        sampler.presence[start:date_idx + 1, selected], dtype=np.float32
    ).T
    x = (x - mean) / np.maximum(std, 1e-6)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x *= mask[:, :, None, None]

    # 评估区间开头没有前置数据，模型必须能在只有几天历史时给分。
    # 训练时随机截断窗口最前面若干天，制造同样的输入分布。
    if rng is not None and rng.random() < SHORT_HISTORY_AUG_PROB:
        drop = int(rng.integers(1, max(2, LOOKBACK_DAYS - 2)))
        mask[:, :drop] = 0.0
        x[:, :drop] = 0.0

    output = {
        "x": x.astype(np.float32, copy=False),
        "mask": mask,
        "selected": selected,
    }
    if require_labels:
        output["y"] = np.asarray(sampler.labels[date_idx, selected], dtype=np.float32)
    return output



def _rank_percentile_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    return ranks


def _safe_corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 20 or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


_CLOSE_INDEX = FEATURE_COLS.index("close")
_AMOUNT_INDEX = FEATURE_COLS.index("amount")


def _style_proxies(batch: dict) -> np.ndarray:
    """用粗代理近似 BARRA 风格暴露：价格水平 / 流动性水平 / 波动 / 动量。

    纯离线诊断，只在本地验证里使用，绝不进入推理路径，也不作为模型输入，
    所以不触及「禁止特征工程」那条。目的是估计：分数被风格回归剃掉之后还剩多少。
    """
    x = batch["x"]
    mask = batch["mask"]
    weight = mask[:, :, None]
    denom = np.maximum(weight.sum(axis=(1, 2), keepdims=False), 1.0)

    close = x[:, :, :, _CLOSE_INDEX]
    amount = x[:, :, :, _AMOUNT_INDEX]
    level_price = (close * weight).sum(axis=(1, 2)) / denom
    level_liquidity = (amount * weight).sum(axis=(1, 2)) / denom
    centered = (close - level_price[:, None, None]) * weight
    volatility = np.sqrt(np.maximum(
        (centered ** 2).sum(axis=(1, 2)) / denom, 0.0
    ))
    momentum = close[:, -1, -1] - close[:, 0, 0]
    proxies = np.column_stack([
        level_price, level_liquidity, volatility, momentum
    ]).astype(np.float64)
    return np.nan_to_num(proxies, nan=0.0, posinf=0.0, neginf=0.0)


def _residualize(score: np.ndarray, proxies: np.ndarray) -> tuple[np.ndarray, float]:
    """把分数对风格代理做截面回归，返回残差和被解释掉的方差比例。"""
    score = np.asarray(score, dtype=np.float64)
    design = np.column_stack([np.ones(len(score)), proxies])
    try:
        coefficient, *_ = np.linalg.lstsq(design, score, rcond=None)
    except np.linalg.LinAlgError:
        return score, 0.0
    residual = score - design.dot(coefficient)
    total = float(np.var(score))
    explained = 0.0 if total < 1e-18 else 1.0 - float(np.var(residual)) / total
    return residual, float(np.clip(explained, 0.0, 1.0))


@torch.no_grad()
def _evaluate_validation(
    model: nn.Module,
    sampler: DenseSampler,
    date_pool: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    amp_context,
) -> dict:
    model.eval()
    daily_ic, daily_spread = [], []
    daily_residual_ic, daily_style_r2 = [], []
    for date_idx in date_pool:
        batch = _cross_section(sampler, int(date_idx), mean, std, True)
        xb = torch.from_numpy(batch["x"]).to(device, non_blocking=True)
        mb = torch.from_numpy(batch["mask"]).to(device, non_blocking=True)
        with amp_context():
            prediction = model(xb, mb)
        score = _combine_prediction_heads(
            prediction.float().detach().cpu().numpy()
        )
        target = np.asarray(batch["y"][:, 0], dtype=np.float64)
        lo, hi = np.nanquantile(target, [0.01, 0.99])
        target = np.clip(target, lo, hi)
        target_rank = _rank_percentile_np(target)
        daily_ic.append(_safe_corr_np(score, target_rank))

        # 离线诊断：预测和目标都对同一组粗风格代理残差化。
        # 这比“残差预测 vs 原始目标”更接近平台风格中性化后的IC。
        proxies = _style_proxies(batch)
        residual_score, style_r2 = _residualize(score, proxies)
        residual_target, _ = _residualize(target_rank, proxies)
        daily_residual_ic.append(_safe_corr_np(residual_score, residual_target))
        daily_style_r2.append(style_r2)

        k = max(10, int(len(score) * 0.10))
        order = np.argsort(score)
        daily_spread.append(
            float(np.nanmean(target[order[-k:]]) - np.nanmean(target[order[:k]]))
        )
        del xb, mb, prediction

    ic = np.asarray(daily_ic, dtype=np.float64)
    spread = np.asarray(daily_spread, dtype=np.float64)
    mean_ic = float(np.nanmean(ic))
    ic_std = float(np.nanstd(ic) + 1e-8)
    icir = mean_ic / ic_std
    spread_sr = float(
        np.nanmean(spread) / (np.nanstd(spread) + 1e-8) * np.sqrt(252.0)
    )
    stress = float(np.nanquantile(ic, 0.20))
    block_means = [
        float(np.nanmean(ic[i:i + 21]))
        for i in range(0, len(ic), 21)
        if len(ic[i:i + 21]) >= 10
    ]
    worst_block = float(min(block_means)) if block_means else stress
    raw_objective = (
        0.45 * mean_ic
        + 0.020 * np.tanh(icir)
        + 0.020 * np.tanh(spread_sr / 3.0)
        + 0.10 * stress
        + 0.10 * worst_block
    )
    residual_ic = np.asarray(daily_residual_ic, dtype=np.float64)
    residual_mean = float(np.nanmean(residual_ic))
    residual_ir = residual_mean / float(np.nanstd(residual_ic) + 1e-8)
    neutral_weight = float(np.clip(NEUTRAL_SELECTION_WEIGHT, 0.0, 0.50))
    objective = (1.0 - neutral_weight) * raw_objective + neutral_weight * residual_mean
    return {
        "objective": float(objective),
        "raw_objective": float(raw_objective),
        "ic_mean": mean_ic,
        "ic_ir": float(icir),
        "ls_sr_proxy": spread_sr,
        "stress_q20": stress,
        "worst_21d_ic": worst_block,
        # residual_ic_mean 以小权重参与选模；其余风格指标只用于诊断。
        # residual_ic_mean 明显低于 ic_mean 说明分数里有较多可被剃掉的风格暴露
        "residual_ic_mean": residual_mean,
        "residual_ic_ir": float(residual_ir),
        "style_r2_mean": float(np.nanmean(daily_style_r2)),
        "n_days": int(len(ic)),
    }

def _atomic_checkpoint_save(payload: dict, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_rng_state(rng: np.random.Generator) -> dict:
    payload = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "numpy_generator": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = [v.cpu() for v in torch.cuda.get_rng_state_all()]
    return payload


def _restore_rng_state(state: dict | None, rng: np.random.Generator) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy_legacy"])
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"].to(dtype=torch.uint8, device="cpu"))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        for idx, value in enumerate(state["torch_cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(
                value.to(dtype=torch.uint8, device="cpu"), device=idx
            )


def train_and_save(
    datasources: Any = None,
    *,
    local_data_root: str | os.PathLike | None = None,
    model_path: str | os.PathLike = MODEL_PATH,
    resume: bool = True,
    checkpoint_path: str | os.PathLike | None = None,
) -> str:
    """Train ResidualAlpha V4.2 with exact mid-epoch resume.

    The checkpoint stores the model, optimizer, EMA, exact shuffled date order,
    next position, running epoch statistics, RNG states, history and Top-K EMA
    snapshots. Re-running the same call continues the interrupted epoch without
    changing the data, model, loss, learning-rate schedule or cross-section.
    """
    seed_everything(SEED)
    local_root = local_data_root or os.environ.get("BIGQUANT_E2E_LOCAL_DATA_ROOT")
    runtime_root = Path(
        os.environ.get("BIGQUANT_E2E_RUNTIME_ROOT", str(_HERE / "_v4_runtime"))
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    cache_dir = runtime_root / "dense_cache"
    checkpoint_path = Path(
        checkpoint_path
        or os.environ.get(
            "BIGQUANT_E2E_CHECKPOINT_PATH",
            str(_HERE / "v4_resume_checkpoint.pt"),
        )
    )
    crash_path = checkpoint_path.with_name(
        checkpoint_path.stem + ".crash" + checkpoint_path.suffix
    )
    save_every_steps = max(
        1, int(os.environ.get("BIGQUANT_E2E_SAVE_EVERY", "20"))
    )

    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        if local_root:
            _build_dense_cache_local(Path(local_root), cache_dir)
        else:
            datasource = _table_reference(datasources)
            if datasource is None:
                raise ValueError("未提供 local_data_root，也未提供 datasources['bar1m']")
            _build_dense_cache_cloud(datasource, cache_dir)

    rebuild_labels_if_needed(
        cache_dir,
        local_root=Path(local_root) if local_root else None,
        datasource=None if local_root else _table_reference(datasources),
    )
    sampler = DenseSampler(cache_dir)
    stats_path = runtime_root / "stats.npz"
    if stats_path.exists():
        stats = np.load(stats_path)
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)
    else:
        mean, std = _compute_stats(sampler)
        np.savez(stats_path, mean=mean, std=std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompetitionE2EModel(**MODEL_CONFIG).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not MIN_PARAMETER_COUNT <= parameter_count <= MAX_PARAMETER_COUNT:
        raise RuntimeError(f"可训练参数量越界：{parameter_count:,}")
    print(f"[model] 单份权重可训练参数量 {parameter_count:,}")
    optimizer = torch.optim.AdamW(
        _parameter_groups(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    ema = ModelEMA(model, decay=EMA_DECAY)

    full_pool = sampler.date_pool()
    split = max(LOOKBACK_DAYS + 40, int(len(full_pool) * 0.85))
    train_pool = full_pool[:max(1, split - MAX_HORIZON)]
    val_pool = full_pool[split:]
    if len(val_pool) < 40:
        raise RuntimeError("验证集交易日不足，请检查训练数据范围")

    n_tail = int(len(val_pool) * TAIL_ADAPTATION_FRACTION)
    if not TAIL_ADAPTATION_ENABLED or n_tail < 20:
        n_tail = 0
    val_select = val_pool[: len(val_pool) - n_tail] if n_tail else val_pool
    val_tail = val_pool[len(val_pool) - n_tail:] if n_tail else np.asarray([], dtype=np.int64)
    print(f"[split] 训练 {len(train_pool)} 天 / 选模型 {len(val_select)} 天 / "
          f"尾段微调 {len(val_tail)} 天")
    if len(val_tail):
        print("[warning] 当前精确断点覆盖主训练；建议保持 BIGQUANT_E2E_TAIL_ADAPT=0")

    total_steps = len(train_pool) * EPOCHS
    time_budget = TRAIN_TIME_BUDGET_HOURS * 3600.0
    rng = np.random.default_rng(SEED)
    global_step = 0
    history = []
    epoch_states: list[tuple[float, int, dict]] = []
    best_seen_objective = -float("inf")
    epochs_without_improvement = 0
    training_start = time.time()

    current_epoch = 1
    next_position = 0
    epoch_order: np.ndarray | None = None
    epoch_losses: list[float] = []
    epoch_ic: list[list[float]] = [[] for _ in HORIZONS]

    def _cpu_model_state() -> dict:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _trim_epoch_states(states: list[tuple[float, int, dict]]) -> list:
        # Keeping the current Top-K is mathematically sufficient for the final
        # Top-K ensemble: a discarded lower-ranked snapshot can never re-enter.
        return sorted(states, key=lambda item: item[0], reverse=True)[
            : max(1, ENSEMBLE_TOP_K)
        ]

    def _checkpoint_payload(reason: str) -> dict:
        return {
            "format_version": 2,
            "model_name": "ResidualAlphaV4.2",
            "reason": reason,
            "saved_at": pd.Timestamp.utcnow().isoformat(),
            "phase": "main",
            "epoch": int(current_epoch),
            "next_position": int(next_position),
            "epoch_order": None if epoch_order is None else np.asarray(epoch_order, dtype=np.int64),
            "epoch_losses": list(epoch_losses),
            "epoch_ic": [list(values) for values in epoch_ic],
            "global_step": int(global_step),
            "model_state_dict": _cpu_model_state(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state_dict": {
                k: v.detach().cpu().clone() for k, v in ema.shadow.items()
            },
            "history": history,
            "epoch_states": _trim_epoch_states(epoch_states),
            "best_seen_objective": float(best_seen_objective),
            "epochs_without_improvement": int(epochs_without_improvement),
            "elapsed_seconds": float(time.time() - training_start),
            "rng_state": _capture_rng_state(rng),
            "config_signature": {
                "epochs": int(EPOCHS),
                "train_days": int(len(train_pool)),
                "seed": int(SEED),
                "lookback": int(LOOKBACK_DAYS),
                "horizons": tuple(int(v) for v in HORIZONS),
                "model_config": MODEL_CONFIG,
            },
        }

    def _save_checkpoint(path: Path, reason: str) -> None:
        _atomic_checkpoint_save(_checkpoint_payload(reason), path)
        print(
            f"[checkpoint] {reason} | epoch={current_epoch} | "
            f"next_pos={next_position} | global_step={global_step} | {path}"
        )

    # Prefer the newest valid normal/crash checkpoint.
    candidates = [p for p in (checkpoint_path, crash_path) if p.exists()]
    resume_path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    if resume and resume_path is not None:
        print(f"[resume] 读取checkpoint：{resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location="cpu")
        if checkpoint.get("model_name") not in {"ResidualAlphaV4.2", "ResidualAlphaV4.1"}:
            raise RuntimeError("checkpoint不是ResidualAlphaV4")
        signature = checkpoint.get("config_signature")
        if signature is not None:
            if int(signature.get("epochs", EPOCHS)) != int(EPOCHS):
                raise RuntimeError("续训时EPOCHS发生变化，会改变学习率计划")
            if int(signature.get("train_days", len(train_pool))) != int(len(train_pool)):
                raise RuntimeError("续训时训练交易日数量发生变化")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_state_to_device(optimizer, device)
        ema.load_cpu_state_dict(checkpoint["ema_state_dict"], model)
        global_step = int(checkpoint.get("global_step", 0))
        history = checkpoint.get("history", [])
        epoch_states = checkpoint.get("epoch_states", [])
        best_seen_objective = float(
            checkpoint.get("best_seen_objective", -float("inf"))
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        current_epoch = int(
            checkpoint.get("epoch", checkpoint.get("next_epoch", 1))
        )
        next_position = int(checkpoint.get("next_position", 0))
        saved_order = checkpoint.get("epoch_order")
        epoch_order = None if saved_order is None else np.asarray(saved_order, dtype=np.int64)
        epoch_losses = [float(v) for v in checkpoint.get("epoch_losses", [])]
        saved_ic = checkpoint.get("epoch_ic", [[] for _ in HORIZONS])
        epoch_ic = [[float(v) for v in values] for values in saved_ic]
        while len(epoch_ic) < len(HORIZONS):
            epoch_ic.append([])
        training_start = time.time() - float(checkpoint.get("elapsed_seconds", 0.0))
        _restore_rng_state(checkpoint.get("rng_state"), rng)
        print(
            f"[resume] 主训练从 epoch={current_epoch}, "
            f"position={next_position}, global_step={global_step} 继续"
        )

    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    stop_requested = {"value": False, "signal": None}
    old_handlers = {}

    def _request_stop(signum, _frame):
        stop_requested["value"] = True
        stop_requested["signal"] = int(signum)
        print("\n[signal] 收到中断请求；将在当前step安全结束后保存断点。")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):
            pass

    try:
        while current_epoch <= EPOCHS:
            if epoch_order is None:
                epoch_order = train_pool.copy()
                rng.shuffle(epoch_order)
                next_position = 0
                epoch_losses = []
                epoch_ic = [[] for _ in HORIZONS]

            model.train()
            for position in range(next_position, len(epoch_order)):
                date_idx = int(epoch_order[position])
                step_completed = False
                try:
                    for group in optimizer.param_groups:
                        group["lr"] = _learning_rate(global_step, total_steps)
                    batch = _cross_section(
                        sampler, date_idx, mean, std, True, rng=rng
                    )
                    xb = torch.from_numpy(batch["x"]).to(device, non_blocking=True)
                    mb = torch.from_numpy(batch["mask"]).to(device, non_blocking=True)
                    yb = torch.from_numpy(batch["y"]).to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with amp_context():
                        prediction = model(xb, mb)
                    parts = competition_multitask_loss(prediction.float(), yb.float())
                    loss = parts["loss"]
                    if not torch.isfinite(loss):
                        raise RuntimeError("训练出现NaN/Inf loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    ema.update(model)

                    epoch_losses.append(float(loss.detach().cpu()))
                    for horizon_index in range(len(HORIZONS)):
                        epoch_ic[horizon_index].append(
                            float(parts[f"ic_{horizon_index}"].detach().cpu())
                        )
                    global_step += 1
                    next_position = position + 1
                    step_completed = True
                    del xb, mb, yb, prediction, loss, parts, batch
                except BaseException:
                    # Re-run this date unless the optimizer step completed.
                    next_position = position + 1 if step_completed else position
                    _save_checkpoint(crash_path, "exception")
                    raise

                if global_step % save_every_steps == 0:
                    _save_checkpoint(checkpoint_path, "periodic")

                elapsed = time.time() - training_start
                # Pause before the environment's hard time limit. Resuming keeps
                # the same exact epoch order, LR position and optimizer state.
                if elapsed >= max(60.0, time_budget - 10 * 60.0):
                    _save_checkpoint(checkpoint_path, "budget_pause")
                    print("[paused] 接近本次时间预算，已安全暂停；再次运行同一单元格即可继续。")
                    return str(checkpoint_path)

                if stop_requested["value"]:
                    _save_checkpoint(checkpoint_path, "signal_pause")
                    print("[paused] 已安全保存；再次运行同一单元格即可继续。")
                    return str(checkpoint_path)

            # Exact epoch finished; validation remains identical to the base V4.1.
            ema.apply(model)
            validation = _evaluate_validation(
                model, sampler, val_select, mean, std, device, amp_context
            )
            current_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            ema.restore(model)

            record = {
                "epoch": current_epoch,
                "loss": float(np.mean(epoch_losses)),
                "last_lr": float(optimizer.param_groups[0]["lr"]),
                "validation": validation,
                "seconds_total": float(time.time() - training_start),
            }
            for horizon_index, horizon in enumerate(HORIZONS):
                record[f"train_ic_{horizon}d"] = float(
                    np.mean(epoch_ic[horizon_index])
                )
            history.append(record)
            print(record)

            current_objective = float(validation["objective"])
            epoch_states.append((current_objective, current_epoch, current_state))
            epoch_states = _trim_epoch_states(epoch_states)

            if current_objective > best_seen_objective + EARLY_STOPPING_MIN_DELTA:
                best_seen_objective = current_objective
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(
                    f"[early-stop] 连续 {epochs_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE} 轮未显著提升"
                )

            current_epoch += 1
            next_position = 0
            epoch_order = None
            epoch_losses = []
            epoch_ic = [[] for _ in HORIZONS]
            _save_checkpoint(checkpoint_path, "epoch_end")
            crash_path.unlink(missing_ok=True)

            if (
                EARLY_STOPPING_PATIENCE > 0
                and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
            ):
                print("[early-stop] 触发提前停止，开始Top-K选模并保存")
                break

        if not epoch_states:
            raise RuntimeError("没有获得有效验证checkpoint")

        epoch_states.sort(key=lambda item: item[0], reverse=True)
        selected = epoch_states[: max(1, ENSEMBLE_TOP_K)]
        best_objective = selected[0][0]
        best_epoch = selected[0][1]
        print(f"[ensemble] 选用 epoch {[e for _, e, _ in selected]}，"
              f"验证目标 {[round(o, 5) for o, _, _ in selected]}")

        final_states = []
        for objective, epoch, state in selected:
            if len(val_tail) == 0:
                final_states.append(state)
                continue
            # Kept for compatibility, but V4.2 notebook defaults tail adaptation off.
            model.load_state_dict(state, strict=True)
            tail_optimizer = torch.optim.AdamW(
                _parameter_groups(model), lr=3e-5, weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.98),
            )
            tail_ema = ModelEMA(model, decay=0.990)
            model.train()
            tail_losses = []
            for date_idx in val_tail:
                batch = _cross_section(sampler, int(date_idx), mean, std, True)
                xb = torch.from_numpy(batch["x"]).to(device, non_blocking=True)
                mb = torch.from_numpy(batch["mask"]).to(device, non_blocking=True)
                yb = torch.from_numpy(batch["y"]).to(device, non_blocking=True)
                tail_optimizer.zero_grad(set_to_none=True)
                with amp_context():
                    prediction = model(xb, mb)
                parts = competition_multitask_loss(prediction.float(), yb.float())
                loss = parts["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                tail_optimizer.step()
                tail_ema.update(model)
                tail_losses.append(float(loss.detach().cpu()))
                del xb, mb, yb, prediction, loss, parts, batch
            final_states.append(tail_ema.cpu_state_dict())
            history.append({
                "stage": "latest_regime_tail_adaptation",
                "source_epoch": int(epoch),
                "mean_loss": float(np.mean(tail_losses)),
                "n_days": int(len(val_tail)),
                "seconds_total": float(time.time() - training_start),
            })

        final_state = final_states[-1]
        payload = {
            "format_version": 6,
            "model_name": "ResidualAlphaV4.2",
            "model_state_dict": final_state,
            "model_state_dicts": final_states,
            "horizons": list(HORIZONS),
            "model_config": MODEL_CONFIG,
            "feature_cols": FEATURE_COLS,
            "log_cols": LOG_COLS,
            "mean": mean,
            "std": std,
            "prediction_weights": PREDICTION_WEIGHTS,
            "lookback_days": LOOKBACK_DAYS,
            "minutes_per_day": MINUTES_PER_DAY,
            "max_horizon": MAX_HORIZON,
            "seed": SEED,
            "epochs": EPOCHS,
            "best_epoch": int(best_epoch),
            "best_validation_objective": float(best_objective),
            "train_date_start": str(pd.Timestamp(sampler.dates[train_pool[0]]).date()),
            "train_date_end": str(pd.Timestamp(sampler.dates[train_pool[-1]]).date()),
            "validation_date_start": str(pd.Timestamp(sampler.dates[val_select[0]]).date()),
            "validation_date_end": str(pd.Timestamp(sampler.dates[val_select[-1]]).date()),
            "training_history": history,
        }
        result = save_model(payload, model_path)
        print(f"[completed] 最终模型已保存：{result}")
        print(f"[completed] 精确断点保留：{checkpoint_path}")
        return result
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass



INFER_STOCK_CHUNK = 32
INFER_QUERY_INSTRUMENT_CHUNK = 500
POOL_LOOKBACK_CALENDAR_DAYS = 120
INFER_STORAGE_DTYPE = np.float16
# Hidden/public tables can contain extreme finite values. Clip standardized
# inference inputs before float16 storage to prevent overflow -> Inf -> NaN.
INFER_VALUE_CLIP = 20.0
MIN_PARAMETER_COUNT = 100_000
MAX_PARAMETER_COUNT = 100_000_000


def _empty_daily_snapshot() -> dict:
    return {
        "index": {},
        "values": np.zeros(
            (0, MINUTES_PER_DAY, len(FEATURE_COLS)),
            dtype=INFER_STORAGE_DTYPE,
        ),
    }


def _build_daily_snapshot(
    raw: pd.DataFrame,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    """把单个交易日压缩成按股票索引的 dense float16 张量。

    单日处理后立即释放原始 DataFrame，避免一次物化数十个交易日的分钟表。
    """
    if raw is None or len(raw) == 0:
        return _empty_daily_snapshot()

    required = ["date", "instrument", *FEATURE_COLS]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise KeyError(f"云端分钟表缺少字段: {missing}")

    dates = pd.to_datetime(raw["date"], errors="raise")
    minute_slot = _minute_slot(dates)
    valid_time = (minute_slot >= 0) & (minute_slot < MINUTES_PER_DAY)
    if not np.any(valid_time):
        return _empty_daily_snapshot()

    frame = raw.loc[valid_time, ["instrument", *FEATURE_COLS]]
    minute_slot = minute_slot[valid_time]
    key_values = frame["instrument"].astype(str).to_numpy()
    key_codes, unique_keys = pd.factorize(key_values, sort=True)

    values = frame[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    for column in BOOK_PRICE_COLS:
        index = FEATURE_COLS.index(column)
        bad = values[:, index] <= 0
        values[bad, index] = np.nan
    for index in LOG_FEATURE_INDICES:
        values[:, index] = np.log1p(np.clip(values[:, index], 0, None))
    values = (values - mean) / np.maximum(std, 1e-6)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, -INFER_VALUE_CLIP, INFER_VALUE_CLIP)
    values = values.astype(INFER_STORAGE_DTYPE, copy=False)

    dense = np.zeros(
        (len(unique_keys), MINUTES_PER_DAY, len(FEATURE_COLS)),
        dtype=INFER_STORAGE_DTYPE,
    )
    dense[key_codes.astype(np.int64), minute_slot.astype(np.int64), :] = values
    keys = np.asarray(unique_keys, dtype=str)
    return {
        "index": {key: index for index, key in enumerate(keys.tolist())},
        "values": dense,
    }


def _make_history_chunk(
    history: deque,
    keys: np.ndarray,
    *,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_stocks = len(keys)
    x = np.zeros(
        (n_stocks, lookback_days, MINUTES_PER_DAY, len(FEATURE_COLS)),
        dtype=INFER_STORAGE_DTYPE,
    )
    mask = np.zeros((n_stocks, lookback_days), dtype=np.float32)
    snapshots = list(history)[-lookback_days:]
    offset = lookback_days - len(snapshots)

    for day_position, snapshot in enumerate(snapshots, start=offset):
        index_map = snapshot["index"]
        source_indices = np.fromiter(
            (index_map.get(str(key), -1) for key in keys),
            dtype=np.int64,
            count=n_stocks,
        )
        valid = source_indices >= 0
        if np.any(valid):
            x[valid, day_position, :, :] = snapshot["values"][source_indices[valid]]
            mask[valid, day_position] = 1.0

    # TransformerEncoder can emit NaN when every position is padding.
    # Also, the trained architecture reads h[:, -1], so make the final slot
    # deterministic even for suspended/newly-added stocks with no current-day bar.
    for stock_index in range(n_stocks):
        valid_positions = np.flatnonzero(mask[stock_index] > 0)
        if valid_positions.size == 0:
            # Neutral all-zero observation, but one valid token prevents all-padding.
            mask[stock_index, -1] = 1.0
        elif mask[stock_index, -1] == 0:
            last_valid = int(valid_positions[-1])
            x[stock_index, -1, :, :] = x[stock_index, last_valid, :, :]
            mask[stock_index, -1] = 1.0
    return x, mask


def _predict_one_cross_section(
    model: CompetitionE2EModel,
    history: deque,
    current_keys: np.ndarray,
    *,
    device: torch.device,
    lookback_days: int,
) -> np.ndarray:
    """分股票块完成日内/跨日编码，再一次性做横截面注意力。

    这与 model.forward 的计算图等价，但不需要同时把约 1000 只股票的
    20×240×26 输入全部放入 CPU/GPU 内存。
    """
    stock_states = []
    with torch.inference_mode():
        for begin in range(0, len(current_keys), INFER_STOCK_CHUNK):
            keys_chunk = current_keys[begin: begin + INFER_STOCK_CHUNK]
            x_np, mask_np = _make_history_chunk(
                history,
                keys_chunk,
                lookback_days=lookback_days,
            )
            xb = torch.from_numpy(x_np).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            mb = torch.from_numpy(mask_np).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            day_embedding = model.intraday(xb, mb)
            day_embedding = day_embedding * mb.unsqueeze(-1)
            state = model.cross_day(day_embedding, mb)
            stock_states.append(state)
            del x_np, mask_np, xb, mb, day_embedding, state

        all_states = torch.cat(stock_states, dim=0)
        all_states = torch.nan_to_num(all_states, nan=0.0, posinf=0.0, neginf=0.0)
        all_states = model.cross_section(all_states)
        all_states = torch.nan_to_num(all_states, nan=0.0, posinf=0.0, neginf=0.0)
        hidden = model.shared(all_states)
        base = model.base_head(hidden)
        offsets = torch.cat(
            [head(hidden) for head in model.horizon_heads],
            dim=1,
        )
        prediction = base + offsets * model.horizon_scale.view(1, -1)
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        result = prediction.float().cpu().numpy()

    del stock_states, all_states, hidden, base, offsets, prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _load_pool_frame(start_date: str, end_date: str) -> pd.DataFrame:
    """官方成分股表。私榜区间可能查不到，这里绝不允许抛异常中断推理。"""
    try:
        import dai
        pool_frame = dai.query(
            "SELECT date,instrument FROM bigalpha_2026_instruments "
            "ORDER BY date,instrument",
            filters={"date": [start_date, end_date]},
        ).df()
    except Exception as error:
        print(f"[pool] 成分股表不可用，改用分钟表推导股票池: {error}")
        return pd.DataFrame(columns=["date", "instrument"])
    if pool_frame is None or len(pool_frame) == 0:
        return pd.DataFrame(columns=["date", "instrument"])
    pool_frame["date"] = pd.to_datetime(pool_frame["date"]).dt.normalize()
    pool_frame["instrument"] = pool_frame["instrument"].astype(str)
    return pool_frame.drop_duplicates(["date", "instrument"])


def _grid_from_datasource(
    datasource: Any, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """从平台注入的分钟表直接取 (交易日, 股票) 网格。

    注入表只含评估区间，因此它是「评估区间到底有哪些交易日」的权威来源；
    成分股表缺日或私榜区间查不到时用它兜底，杜绝缺失交易日。
    """
    lo = pd.Timestamp(start_date).normalize()
    hi = pd.Timestamp(end_date).normalize()
    try:
        if isinstance(datasource, pd.DataFrame) or (
            hasattr(datasource, "df") and callable(datasource.df)
        ):
            frame = (
                datasource if isinstance(datasource, pd.DataFrame) else datasource.df()
            )
            grid = frame.loc[:, ["date", "instrument"]].copy()
        else:
            import dai
            grid = dai.query(
                "SELECT DISTINCT date::DATE::DATETIME AS date, instrument "
                f"FROM {datasource}",
                filters={
                    "date": [
                        lo.strftime("%Y-%m-%d 00:00:00"),
                        hi.strftime("%Y-%m-%d 23:59:59"),
                    ]
                },
            ).df()
    except Exception as error:
        print(f"[grid] 分钟表交易日探测失败: {error}")
        return pd.DataFrame(columns=["date", "instrument"])

    if grid is None or len(grid) == 0:
        return pd.DataFrame(columns=["date", "instrument"])
    grid["date"] = pd.to_datetime(grid["date"]).dt.normalize()
    grid["instrument"] = grid["instrument"].astype(str)
    grid = grid[(grid["date"] >= lo) & (grid["date"] <= hi)]
    return grid.drop_duplicates(["date", "instrument"])


def _postprocess_scores(
    raw_result: pd.DataFrame, skeleton: pd.DataFrame, evaluation_dates: list
) -> pd.DataFrame:
    """把逐日分数铺回完整骨架：补缺、跨日平滑、逐日重新标准化。

    保证每个评估交易日都有一个有判别力的截面，既不缺交易日，
    也不会出现全零/常数截面（那会让平台的 z-score 变成 NaN）。
    """
    merged = skeleton.merge(raw_result, how="left", on=["date", "instrument"])
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged["score"] = merged["score"].replace([np.inf, -np.inf], np.nan)

    wide = merged.pivot_table(
        index="date", columns="instrument", values="score", aggfunc="last"
    )
    wide = wide.reindex(index=pd.DatetimeIndex(evaluation_dates))
    wide.index.name = "date"
    wide.columns.name = "instrument"

    center = wide.mean(axis=1)
    scale = wide.std(axis=1).replace(0.0, np.nan)
    wide = wide.sub(center, axis=0).div(scale, axis=0)
    wide = wide.ffill(limit=5).fillna(0.0)

    empty_days = wide.std(axis=1) < 1e-9
    if bool(empty_days.any()):
        wide.loc[empty_days.to_numpy(), :] = np.nan
        # 只允许使用过去交易日。若评估区间开头尚无历史分数，
        # 使用稳定的极小确定性截面，避免常数分数和未来信息。
        wide = wide.ffill()
        leading_empty = wide.isna().all(axis=1)
        if bool(leading_empty.any()):
            base = np.arange(len(wide.columns), dtype=np.float64)
            base = (base - base.mean()) / max(base.std(), 1.0)
            for day in wide.index[leading_empty]:
                wide.loc[day, :] = base * 1e-6
        wide = wide.fillna(0.0)
        print(f"[fill] {int(empty_days.sum())} 个交易日截面为空，已按仅使用过去信息的规则回填")

    if SCORE_SMOOTHING_ALPHA > 0.0:
        wide = wide.ewm(alpha=1.0 - SCORE_SMOOTHING_ALPHA, adjust=False).mean()

    center = wide.mean(axis=1)
    scale = wide.std(axis=1).replace(0.0, 1.0)
    wide = wide.sub(center, axis=0).div(scale, axis=0)

    tidy = wide.stack().rename("score").reset_index()
    tidy.columns = ["date", "instrument", "score"]
    tidy["instrument"] = tidy["instrument"].astype(str)

    result = skeleton.merge(tidy, how="left", on=["date", "instrument"])
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result["score"] = result["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (
        result[["date", "instrument", "score"]]
        .drop_duplicates(["date", "instrument"], keep="last")
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def _merge_daily_snapshots(parts: list[dict]) -> dict:
    """合并分批读取的单日快照；不同 instrument 分批，不会重复。"""
    nonempty = [part for part in parts if len(part["index"]) > 0]
    if not nonempty:
        return _empty_daily_snapshot()

    keys: list[str] = []
    values: list[np.ndarray] = []
    for part in nonempty:
        ordered = sorted(part["index"].items(), key=lambda item: item[1])
        keys.extend(str(key) for key, _ in ordered)
        values.append(part["values"])
    dense = np.concatenate(values, axis=0)
    return {
        "index": {key: idx for idx, key in enumerate(keys)},
        "values": dense,
    }


def _load_daily_snapshot_filtered(
    datasource: Any,
    *,
    day_start: str,
    day_end: str,
    instruments: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    """按股票分批读取单日分钟数据，避免 DAI/Pandas 单次结果过大。"""
    columns = ["date", "instrument", *FEATURE_COLS]
    parts: list[dict] = []
    for begin in range(0, len(instruments), INFER_QUERY_INSTRUMENT_CHUNK):
        chunk = instruments[begin: begin + INFER_QUERY_INSTRUMENT_CHUNK]
        raw = _materialize_cloud(
            datasource,
            start_date=day_start,
            end_date=day_end,
            columns=columns,
            instruments=chunk,
        )
        parts.append(_build_daily_snapshot(raw, mean=mean, std=std))
        del raw
        gc.collect()
    return _merge_daily_snapshots(parts)


def predict_scores(
    datasources: Any,
    start_date: str,
    end_date: str,
    *,
    model_path: str | os.PathLike = MODEL_PATH,
) -> pd.DataFrame:
    """加载多份 EMA 快照，按交易日流式推理并做集成。"""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"缺少 {model_path}；请把训练生成的权重 JSON 与代码一起提交"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_model(model_path, map_location="cpu")

    required_metadata = {
        "state_dicts", "model_config", "feature_cols", "mean", "std",
        "lookback_days", "minutes_per_day", "seed",
    }
    missing_metadata = sorted(required_metadata.difference(ckpt))
    if missing_metadata:
        raise KeyError(f"模型文件缺少元数据: {missing_metadata}")
    if list(ckpt["feature_cols"]) != FEATURE_COLS:
        raise RuntimeError("模型权重的26个输入字段与推理脚本不一致")
    if int(ckpt["minutes_per_day"]) != MINUTES_PER_DAY:
        raise RuntimeError("模型权重的分钟长度与推理脚本不一致")

    state_dicts = ckpt.pop("state_dicts")
    ckpt.pop("state_dict", None)
    models = []
    parameter_count = 0
    for state_dict in state_dicts:
        one = CompetitionE2EModel(**ckpt["model_config"]).to(device)
        parameter_count = sum(p.numel() for p in one.parameters())
        if not MIN_PARAMETER_COUNT <= parameter_count <= MAX_PARAMETER_COUNT:
            raise RuntimeError(f"可训练参数量越界: {parameter_count:,}")
        one.load_state_dict(state_dict, strict=True)
        one.eval()
        models.append(one)
    print(f"[model] 载入 {len(models)} 份快照，单份参数量 {parameter_count:,}")
    del state_dicts
    gc.collect()

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    lookback = int(ckpt["lookback_days"])
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    if requested_end < requested_start:
        raise ValueError("end_date 早于 start_date")

    datasource = _table_reference(datasources)
    if datasource is None:
        raise ValueError("datasources 中未找到 bar1m 数据源")

    # ---- 交易日与股票池：成分股表为主，注入的分钟表兜底 ----
    pool_query_start = requested_start - pd.Timedelta(
        days=POOL_LOOKBACK_CALENDAR_DAYS
    )
    pool_frame = _load_pool_frame(
        pool_query_start.strftime("%Y-%m-%d"),
        requested_end.strftime("%Y-%m-%d 23:59:59"),
    )
    source_grid = _grid_from_datasource(datasource, requested_start, requested_end)

    pool_eval = pool_frame[
        pool_frame["date"].between(requested_start, requested_end)
    ][["date", "instrument"]]

    missing_days = sorted(
        set(source_grid["date"].unique()) - set(pool_eval["date"].unique())
    )
    if missing_days:
        print(f"[grid] 成分股表缺 {len(missing_days)} 个交易日，用分钟表补齐")
    # 真正的逐日股票池并集，而不只是补“整天缺失”的日期。
    skeleton = pd.concat(
        [pool_eval, source_grid],
        ignore_index=True,
    ).drop_duplicates(["date", "instrument"])
    if skeleton.empty:
        raise RuntimeError("评估区间既取不到成分股，也取不到分钟数据")
    skeleton["instrument"] = skeleton["instrument"].astype(str)

    evaluation_dates = sorted(skeleton["date"].unique())
    universe_by_date = {
        pd.Timestamp(date): np.asarray(
            sorted(group["instrument"].unique()), dtype=str
        )
        for date, group in skeleton.groupby("date", sort=True)
    }

    prior_dates = sorted(
        d for d in pool_frame["date"].unique() if d < requested_start
    )[-max(lookback - 1, 0):]
    processing_dates = list(prior_dates) + list(evaluation_dates)

    universe_instruments = np.asarray(
        sorted(
            {i for arr in universe_by_date.values() for i in arr.tolist()}
            | set(pool_frame["instrument"].astype(str).tolist())
        ),
        dtype=str,
    )
    if len(universe_instruments) == 0:
        raise RuntimeError("评估股票池为空")

    history = deque(maxlen=lookback)
    outputs: list[pd.DataFrame] = []

    for trade_date in processing_dates:
        trade_date = pd.Timestamp(trade_date)
        snapshot = _load_daily_snapshot_filtered(
            datasource,
            day_start=trade_date.strftime("%Y-%m-%d 00:00:00"),
            day_end=trade_date.strftime("%Y-%m-%d 23:59:59"),
            instruments=universe_instruments,
            mean=mean,
            std=std,
        )
        history.append(snapshot)

        if trade_date < requested_start:
            continue

        current_keys = universe_by_date[trade_date]
        try:
            # 同一份行情喂给全部快照，DAI 取数只付一次
            blended = np.zeros(len(current_keys), dtype=np.float64)
            for one in models:
                prediction = _predict_one_cross_section(
                    one, history, current_keys,
                    device=device, lookback_days=lookback,
                )
                blended += _cross_section_zscore(
                    _combine_prediction_heads(prediction)
                )
                del prediction
            score = blended / float(len(models))
        except Exception as error:
            # 单日推理失败绝不能拖掉整段评估，留空交给后处理回填
            print(f"[predict] {trade_date.date()} 推理失败，留空回填: {error}")
            score = np.full(len(current_keys), np.nan, dtype=np.float64)

        outputs.append(pd.DataFrame({
            "date": trade_date,
            "instrument": current_keys,
            "score": score,
        }))
        gc.collect()

    raw_result = (
        pd.concat(outputs, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .drop_duplicates(["date", "instrument"], keep="last")
    )

    result = _postprocess_scores(raw_result, skeleton, evaluation_dates)

    produced = {pd.Timestamp(v).normalize() for v in result["date"].unique()}
    still_missing = [d for d in evaluation_dates if pd.Timestamp(d) not in produced]
    if still_missing:
        raise RuntimeError(f"推理结果仍缺失交易日: {still_missing[:5]}")
    if list(result.columns) != ["date", "instrument", "score"]:
        raise RuntimeError("输出列不严格等于 date/instrument/score")
    if not np.isfinite(result["score"].to_numpy()).all():
        raise RuntimeError("推理分数存在NaN/Inf")

    daily = result.groupby("date")["score"]
    print(
        f"[output] {result['date'].nunique()} 个交易日, {len(result)} 行, "
        f"日均 {daily.size().mean():.0f} 只, 截面 std 均值 {daily.std().mean():.4f}"
    )
    return result


if __name__ == "__main__":
    local_root = os.environ.get("BIGQUANT_E2E_LOCAL_DATA_ROOT")
    if local_root:
        print(train_and_save(local_data_root=local_root))
    else:
        print(
            "请设置 BIGQUANT_E2E_LOCAL_DATA_ROOT 后训练，"
            "或由平台调用 train_and_save(datasources)。"
        )
