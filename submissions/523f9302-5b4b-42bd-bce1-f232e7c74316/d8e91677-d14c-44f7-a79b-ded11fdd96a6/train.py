#!/usr/bin/env python3
"""Self-contained BigAlpha 2026 training/inference runtime.

This source is materialized into each submission as ``train.py``.  The public
leaderboard loads the bundled ``model.json``; private evaluation can call
``train_and_save`` to train the same architecture from scratch.  Only official
BigAlpha datasources are queried and the random seed/configuration are fixed.

Generated submission: V8_R12_shared_depth_mlp_rank_mse_F3_epoch005
Selected public checkpoint epoch: 5
Input representation: shared_depth_mlp
Preprocessing profile: preclose_ratio
Training loss: mse
Training target: cs_rank
Feature schema: raw37
"""

import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import dai
except ImportError:
    dai = None


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.json")
SUBMISSION_NAME = "V8_R12_shared_depth_mlp_rank_mse_F3_epoch005"
BAR_KEY = "bar5m"
DEFAULT_BAR_TABLE = "bigalpha_2026_stock_bar5m"
TRAIN_START, TRAIN_END = "2019-01-01", "2023-12-31"
SEQ_LEN = 96
TRAIN_EPOCHS = 5
BATCH_SIZE = 1024
INFERENCE_BATCH_SIZE = 4096
INSTRUMENT_CHUNK = 20
INFERENCE_INSTRUMENT_CHUNK = 200
SEED = 42
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
WARMUP_EPOCHS = 3
LR_T0, LR_T_MULT, LR_ETA_MIN = 10, 2, 1e-5
INPUT_REPRESENTATION = "shared_depth_mlp"
PREPROCESS_PROFILE = "preclose_ratio"
LOSS_NAME = "mse"
TARGET_TYPE = "cs_rank"
FEATURE_SCHEMA = "raw37"
DEPENDENCIES = {
    "python": "3.11",
    "numpy": "AIStudio built-in",
    "pandas": "AIStudio built-in",
    "torch": "AIStudio built-in",
    "dai": "AIStudio built-in",
}

OHLC_FIELDS = ["open", "high", "low", "close"]
TRADE_FIELDS = ["volume", "amount", "deal_number"]
BOOK_PRICE_FIELDS = [
    *(f"ask_price{i}" for i in range(1, 6)),
    *(f"bid_price{i}" for i in range(1, 6)),
]
BOOK_VOLUME_FIELDS = [
    *(f"ask_volume{i}" for i in range(1, 6)),
    *(f"bid_volume{i}" for i in range(1, 6)),
]
BOOK_COUNT_FIELDS = [
    *(f"ask_num_orders{i}" for i in range(1, 6)),
    *(f"bid_num_orders{i}" for i in range(1, 6)),
]
if FEATURE_SCHEMA == "raw37":
    FEATURE_COLS = (
        OHLC_FIELDS + TRADE_FIELDS + BOOK_PRICE_FIELDS +
        BOOK_VOLUME_FIELDS + BOOK_COUNT_FIELDS)
elif FEATURE_SCHEMA == "raw38":
    FEATURE_COLS = (
        OHLC_FIELDS + ["pre_close"] + TRADE_FIELDS + BOOK_PRICE_FIELDS +
        BOOK_VOLUME_FIELDS + BOOK_COUNT_FIELDS)
else:
    raise ValueError(f"Unsupported submission feature schema: {FEATURE_SCHEMA}")
LOG_FIELDS = ["volume", "amount", "deal_number"] + BOOK_VOLUME_FIELDS + BOOK_COUNT_FIELDS
PRE_CLOSE_RATIO_FIELDS = OHLC_FIELDS + BOOK_PRICE_FIELDS
MODEL_CONFIG = {
    "input_representation": INPUT_REPRESENTATION,
    "d_model": 128,
    "gru_hidden": 128,
    "gru_layers": 2,
    "nlayers": 2,
    "dropout": 0.1,
    "masked_feature_groups": [],
}


def _require_dai():
    if dai is None:
        raise RuntimeError("dai is available only in the AIStudio runtime")


def feature_group_indices(feature_cols):
    market_fields = OHLC_FIELDS + (["pre_close"] if "pre_close" in feature_cols else [])
    groups = [
        market_fields, TRADE_FIELDS, BOOK_PRICE_FIELDS,
        BOOK_VOLUME_FIELDS, BOOK_COUNT_FIELDS,
    ]
    return [[feature_cols.index(name) for name in group] for group in groups]


def book_layout(feature_cols):
    layout = []
    for side in ("bid", "ask"):
        side_layout = []
        for kind in ("price", "volume", "num_orders"):
            side_layout.append([
                feature_cols.index(f"{side}_{kind}{level}")
                for level in range(1, 6)
            ])
        layout.append(side_layout)
    market = [feature_cols.index(name) for name in OHLC_FIELDS + TRADE_FIELDS]
    return layout, market


class GroupedFieldStem(nn.Module):
    def __init__(self, group_indices, d_model=128, dropout=0.1):
        super().__init__()
        dim_per_group = max(d_model // 5, 16)
        group_dims = [max(len(indices), dim_per_group) for indices in group_indices]
        self.group_indices = group_indices
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), dim), nn.LayerNorm(dim),
                nn.GELU(), nn.Dropout(dropout))
            for indices, dim in zip(group_indices, group_dims)
        ])
        total = sum(group_dims)
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(total), nn.Linear(total, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        values = [projection(x[..., indices])
                  for projection, indices in zip(self.projections, self.group_indices)]
        return self.channel_mixer(torch.cat(values, dim=-1))


class ChannelMixerStem(nn.Module):
    def __init__(self, group_indices, d_model=128, token_dim=32, dropout=0.1):
        super().__init__()
        self.group_indices = group_indices
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
            )
            for indices in group_indices
        ])
        n_groups = len(group_indices)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_groups, n_groups * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(n_groups * 2, n_groups))
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(token_dim * 2, token_dim))
        self.output = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.LayerNorm(n_groups * token_dim),
            nn.Linear(n_groups * token_dim, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        hidden = torch.stack([
            projection(x[..., indices])
            for projection, indices in zip(self.projections, self.group_indices)
        ], dim=-2)
        token_update = self.token_mlp(
            self.token_norm(hidden).transpose(-1, -2)).transpose(-1, -2)
        hidden = hidden + token_update
        hidden = hidden + self.channel_mlp(self.channel_norm(hidden))
        return self.output(hidden)


def side_type_groups(feature_cols):
    """Market/trade plus separate bid/ask price/volume/order-count groups."""
    cols = list(feature_cols)
    market_fields = ["open", "high", "low", "close"]
    if "pre_close" in cols:
        market_fields.append("pre_close")
    names = [
        tuple(market_fields),
        ("volume", "amount", "deal_number"),
    ]
    for side in ("bid", "ask"):
        for kind in ("price", "volume", "num_orders"):
            names.append(tuple(
                f"{side}_{kind}{level}" for level in range(1, 6)))
    groups = []
    for fields in names:
        missing = [field for field in fields if field not in cols]
        if missing:
            raise ValueError(f"side_type_mixer missing raw fields: {missing}")
        groups.append([cols.index(field) for field in fields])
    return groups


class SideTypeMixerStem(nn.Module):
    """Exact V6/V7 side-type token mixer used by the selected checkpoint."""

    def __init__(self, feature_cols, d_model=128, token_dim=24, dropout=0.1):
        super().__init__()
        self.group_indices = side_type_groups(feature_cols)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), token_dim),
                nn.LayerNorm(token_dim), nn.GELU())
            for indices in self.group_indices
        ])
        n_groups = len(self.group_indices)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_groups, n_groups * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(n_groups * 2, n_groups))
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(token_dim * 2, token_dim))
        self.output = nn.Sequential(
            nn.Flatten(start_dim=-2), nn.LayerNorm(n_groups * token_dim),
            nn.Linear(n_groups * token_dim, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        tokens = torch.stack([
            projection(x[..., indices])
            for projection, indices in zip(
                self.projections, self.group_indices)
        ], dim=-2)
        token_update = self.token_mlp(
            self.token_norm(tokens).transpose(-1, -2)).transpose(-1, -2)
        tokens = tokens + token_update
        tokens = tokens + self.channel_mlp(self.channel_norm(tokens))
        return self.output(tokens)


class GroupTemporalInceptionStem(nn.Module):
    """Exact independent within-group multi-scale temporal stem used by R14."""

    def __init__(self, group_indices, d_model=128, token_dim=24,
                 kernels=(3, 7, 15), dropout=0.1):
        super().__init__()
        self.group_indices = group_indices
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), token_dim),
                nn.LayerNorm(token_dim), nn.GELU())
            for indices in group_indices
        ])
        branch_dim = max(8, token_dim // len(kernels))
        self.temporal_branches = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(token_dim, branch_dim, kernel_size=kernel,
                          padding=kernel // 2)
                for kernel in kernels
            ])
            for _ in group_indices
        ])
        temporal_dim = branch_dim * len(kernels)
        self.temporal_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(temporal_dim, token_dim, kernel_size=1),
                nn.GELU(), nn.Dropout(dropout))
            for _ in group_indices
        ])
        total = len(group_indices) * token_dim
        self.output = nn.Sequential(
            nn.LayerNorm(total), nn.Linear(total, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        encoded = []
        for indices, projection, branches, output in zip(
                self.group_indices, self.projections,
                self.temporal_branches, self.temporal_outputs):
            token = projection(x[..., indices])
            stream = token.transpose(1, 2)
            temporal = torch.cat(
                [branch(stream) for branch in branches], dim=1)
            token = token + output(temporal).transpose(1, 2)
            encoded.append(token)
        return self.output(torch.cat(encoded, dim=-1))


class FieldTokenStem(nn.Module):
    """Exact scalar-to-token field representation used by R16."""

    def __init__(self, n_feat, d_model=128, token_dim=16, dropout=0.1):
        super().__init__()
        self.n_feat = int(n_feat)
        self.value_weight = nn.Parameter(torch.empty(n_feat, token_dim))
        self.value_bias = nn.Parameter(torch.zeros(n_feat, token_dim))
        self.field_identity = nn.Parameter(torch.empty(n_feat, token_dim))
        nn.init.normal_(self.value_weight, std=0.02)
        nn.init.normal_(self.field_identity, std=0.02)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_feat, n_feat * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(n_feat * 2, n_feat))
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim))
        self.output = nn.Sequential(
            nn.Flatten(start_dim=-2), nn.LayerNorm(n_feat * token_dim),
            nn.Linear(n_feat * token_dim, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        tokens = (
            x.unsqueeze(-1) * self.value_weight.unsqueeze(0).unsqueeze(0)
            + self.value_bias.unsqueeze(0).unsqueeze(0)
            + self.field_identity.unsqueeze(0).unsqueeze(0))
        update = self.token_mlp(
            self.token_norm(tokens).transpose(-1, -2))
        tokens = tokens + update.transpose(-1, -2)
        tokens = tokens + self.channel_mlp(tokens)
        return self.output(tokens)


class SharedDepthStem(nn.Module):
    def __init__(self, feature_cols, d_model=128, depth_channels=16, dropout=0.1):
        super().__init__()
        self.book_layout, self.market_indices = book_layout(feature_cols)
        self.depth = 5
        self.depth_encoder = nn.Sequential(
            nn.Conv1d(3, depth_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, depth_channels), nn.GELU(),
            nn.Conv1d(depth_channels, depth_channels, kernel_size=3, padding=1),
            nn.GELU())
        market_dim = max(16, d_model // 4)
        self.market = nn.Sequential(
            nn.Linear(len(self.market_indices), market_dim),
            nn.LayerNorm(market_dim), nn.GELU())
        self.output = nn.Sequential(
            nn.LayerNorm(depth_channels * self.depth * 2 + market_dim),
            nn.Linear(depth_channels * self.depth * 2 + market_dim, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        batch, length = x.shape[:2]
        sides = []
        for side_layout in self.book_layout:
            book = torch.stack([x[..., indices] for indices in side_layout], dim=-2)
            encoded = self.depth_encoder(book.reshape(batch * length, 3, self.depth))
            sides.append(encoded.reshape(batch, length, -1))
        market = self.market(x[..., self.market_indices])
        return self.output(torch.cat(sides + [market], dim=-1))


class SharedDepthMLPStem(nn.Module):
    """Exact V8 R12 shared bid/ask capacity-control stem."""

    def __init__(self, feature_cols, d_model=128, side_hidden=64, dropout=0.1):
        super().__init__()
        self.book_layout, self.market_indices = book_layout(feature_cols)
        self.depth = 5
        side_input = 3 * self.depth
        self.side_encoder = nn.Sequential(
            nn.LayerNorm(side_input), nn.Linear(side_input, side_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(side_hidden, side_hidden), nn.GELU())
        market_dim = max(16, d_model // 4)
        self.market = nn.Sequential(
            nn.Linear(len(self.market_indices), market_dim),
            nn.LayerNorm(market_dim), nn.GELU())
        self.output = nn.Sequential(
            nn.LayerNorm(side_hidden * 2 + market_dim),
            nn.Linear(side_hidden * 2 + market_dim, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        sides = []
        for side_layout in self.book_layout:
            book = torch.stack(
                [x[..., indices] for indices in side_layout], dim=-2)
            sides.append(self.side_encoder(book.flatten(start_dim=-2)))
        market = self.market(x[..., self.market_indices])
        return self.output(torch.cat(sides + [market], dim=-1))


class LOBMatrixStem(nn.Module):
    def __init__(self, feature_cols, d_model=128, matrix_channels=16, dropout=0.1):
        super().__init__()
        self.book_layout, self.market_indices = book_layout(feature_cols)
        self.depth = 5
        self.encoder = nn.Sequential(
            nn.Conv2d(3, matrix_channels, kernel_size=(2, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(matrix_channels, matrix_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU())
        market_dim = max(16, d_model // 4)
        self.market = nn.Sequential(
            nn.Linear(len(self.market_indices), market_dim),
            nn.LayerNorm(market_dim), nn.GELU())
        self.output = nn.Sequential(
            nn.LayerNorm(matrix_channels * self.depth + market_dim),
            nn.Linear(matrix_channels * self.depth + market_dim, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        batch, length = x.shape[:2]
        sides = [torch.stack([x[..., indices] for indices in side], dim=-2)
                 for side in self.book_layout]
        book = torch.stack(sides, dim=-3).permute(0, 1, 3, 2, 4)
        encoded = self.encoder(book.reshape(batch * length, 3, 2, self.depth))
        encoded = encoded.reshape(batch, length, -1)
        market = self.market(x[..., self.market_indices])
        return self.output(torch.cat([encoded, market], dim=-1))


class SharedSideMLPMixerStem(nn.Module):
    """Exact shared bid/ask encoder and side/market mixer used by L60."""

    def __init__(self, feature_cols, d_model=128, token_dim=64, dropout=0.1):
        super().__init__()
        self.book_layout, self.market_indices = book_layout(feature_cols)
        self.depth = 5
        side_input = 3 * self.depth
        self.side_encoder = nn.Sequential(
            nn.LayerNorm(side_input), nn.Linear(side_input, token_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim), nn.GELU())
        self.market_encoder = nn.Sequential(
            nn.Linear(len(self.market_indices), token_dim),
            nn.LayerNorm(token_dim), nn.GELU())
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(3, 6), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(6, 3))
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(token_dim * 2, token_dim))
        self.output = nn.Sequential(
            nn.Flatten(start_dim=-2), nn.LayerNorm(3 * token_dim),
            nn.Linear(3 * token_dim, d_model),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        sides = []
        for side_layout in self.book_layout:
            book = torch.stack(
                [x[..., indices] for indices in side_layout], dim=-2)
            sides.append(self.side_encoder(book.flatten(start_dim=-2)))
        market = self.market_encoder(x[..., self.market_indices])
        tokens = torch.stack([sides[0], sides[1], market], dim=-2)
        token_update = self.token_mlp(
            self.token_norm(tokens).transpose(-1, -2)).transpose(-1, -2)
        tokens = tokens + token_update
        tokens = tokens + self.channel_mlp(self.channel_norm(tokens))
        return self.output(tokens)


class HybridGlobalSharedSideStem(nn.Module):
    """Submission counterpart of the raw-bypass V8 hybrid stem."""

    def __init__(self, feature_cols, d_model=128, dropout=0.1,
                 fusion_mode="concat"):
        super().__init__()
        if fusion_mode not in {"concat", "gate", "tri_gate"}:
            raise ValueError(f"Unknown hybrid fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.global_branch = nn.Sequential(
            nn.Linear(len(feature_cols), d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.structural_branch = SharedSideMLPMixerStem(
            feature_cols, d_model=d_model, dropout=dropout)
        self.output_norm = nn.LayerNorm(d_model)
        if fusion_mode == "concat":
            self.fusion = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        elif fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.Sigmoid())
        else:
            self.temporal_branch = GroupTemporalInceptionStem(
                feature_group_indices(feature_cols),
                d_model=d_model, dropout=dropout)
            self.branch_gate = nn.Sequential(
                nn.LayerNorm(3 * d_model),
                nn.Linear(3 * d_model, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, 3))

    def forward(self, x):
        global_embedding = self.global_branch(x)
        structural_embedding = self.structural_branch(x)
        joined = torch.cat([global_embedding, structural_embedding], dim=-1)
        if self.fusion_mode == "concat":
            update = self.fusion(joined)
        elif self.fusion_mode == "gate":
            update = self.gate(joined) * structural_embedding
        else:
            temporal_embedding = self.temporal_branch(x)
            joined = torch.cat(
                [global_embedding, structural_embedding, temporal_embedding], dim=-1)
            weights = torch.softmax(self.branch_gate(joined), dim=-1).unsqueeze(-1)
            branches = torch.stack(
                [global_embedding, structural_embedding, temporal_embedding], dim=-2)
            update = (branches * weights).sum(dim=-2)
        return self.output_norm(global_embedding + update)


class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels=128, out_channels_per_kernel=32,
                 kernels=(3, 5, 11), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels_per_kernel, kernel,
                          padding=kernel // 2),
                nn.GroupNorm(min(8, out_channels_per_kernel), out_channels_per_kernel),
                nn.GELU())
            for kernel in kernels
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(out_channels_per_kernel * len(kernels), in_channels, 1),
            nn.Dropout(dropout))

    def forward(self, x):
        transposed = x.transpose(1, 2)
        encoded = torch.cat([layer(transposed) for layer in self.convs], dim=1)
        return self.proj(encoded).transpose(1, 2) + x


class ConvHeadGRU(nn.Module):
    def __init__(self, feature_cols, input_representation,
                 d_model=128, gru_hidden=128, gru_layers=2, dropout=0.1):
        super().__init__()
        if input_representation == "flat":
            self.stem = nn.Sequential(
                nn.Linear(len(feature_cols), d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Dropout(dropout))
        elif input_representation == "grouped":
            self.stem = GroupedFieldStem(
                feature_group_indices(feature_cols), d_model, dropout)
        elif input_representation == "channel_mixer":
            self.stem = ChannelMixerStem(
                feature_group_indices(feature_cols), d_model=d_model,
                dropout=dropout)
        elif input_representation == "shared_depth":
            self.stem = SharedDepthStem(feature_cols, d_model, dropout=dropout)
        elif input_representation == "shared_depth_mlp":
            self.stem = SharedDepthMLPStem(
                feature_cols, d_model=d_model, dropout=dropout)
        elif input_representation == "lob_matrix":
            self.stem = LOBMatrixStem(feature_cols, d_model, dropout=dropout)
        elif input_representation == "side_type_mixer":
            self.stem = SideTypeMixerStem(
                feature_cols, d_model=d_model, dropout=dropout)
        elif input_representation == "shared_side_mlp_mixer":
            self.stem = SharedSideMLPMixerStem(
                feature_cols, d_model=d_model, dropout=dropout)
        elif input_representation == "group_temporal_inception":
            self.stem = GroupTemporalInceptionStem(
                feature_group_indices(feature_cols),
                d_model=d_model, dropout=dropout)
        elif input_representation == "hybrid_global_shared_side_concat":
            self.stem = HybridGlobalSharedSideStem(
                feature_cols, d_model=d_model, dropout=dropout,
                fusion_mode="concat")
        elif input_representation == "hybrid_global_shared_side_gate":
            self.stem = HybridGlobalSharedSideStem(
                feature_cols, d_model=d_model, dropout=dropout,
                fusion_mode="gate")
        elif input_representation == "hybrid_global_shared_side_temporal":
            self.stem = HybridGlobalSharedSideStem(
                feature_cols, d_model=d_model, dropout=dropout,
                fusion_mode="tri_gate")
        elif input_representation == "field_token":
            self.stem = FieldTokenStem(
                len(feature_cols), d_model=d_model, dropout=dropout)
        else:
            raise ValueError(f"Unsupported submission representation: {input_representation}")
        self.input_representation = input_representation
        self.conv = MultiScaleConv1d(
            d_model, d_model // 4, (3, 5, 11), dropout)
        self.gru = nn.GRU(
            d_model, gru_hidden, gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden), nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(gru_hidden // 2, 1))

    def forward(self, x):
        hidden = self.stem(x)
        hidden = self.conv(hidden)
        hidden, _ = self.gru(hidden)
        return self.head(hidden[:, -1]).squeeze(-1)


def create_model(feature_cols=FEATURE_COLS, model_config=None):
    config = dict(MODEL_CONFIG if model_config is None else model_config)
    return ConvHeadGRU(
        feature_cols=list(feature_cols),
        input_representation=config["input_representation"],
        d_model=int(config.get("d_model", 128)),
        gru_hidden=int(config.get("gru_hidden", 128)),
        gru_layers=int(config.get("gru_layers", 2)),
        dropout=float(config.get("dropout", 0.1)))


def load_model_json(path=MODEL_PATH, map_location="cpu"):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dict = {}
    for name, meta in payload["state_dict"].items():
        tensor = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        state_dict[name] = tensor.reshape(meta["shape"]).to(map_location)
    checkpoint = {key: value for key, value in payload.items() if key != "state_dict"}
    checkpoint["state_dict"] = state_dict
    return checkpoint


def save_model_json(model, mean, std, path=MODEL_PATH):
    tensors = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    payload = {
        "model_name": "conv_gru",
        "n_feat": len(FEATURE_COLS),
        "seq_len": SEQ_LEN,
        "freq": "5m",
        "feature_cols": FEATURE_COLS,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
        "preprocess_profile": PREPROCESS_PROFILE,
        "feature_schema": FEATURE_SCHEMA,
        "loss_name": LOSS_NAME,
        "target_type": TARGET_TYPE,
        "model_config": MODEL_CONFIG,
        "checkpoint_metadata": {
            "artifact_role": "private_retrain_output",
            "selected_public_epoch": TRAIN_EPOCHS,
            "seed": SEED,
            "submission_name": SUBMISSION_NAME,
        },
        "state_dict": tensors,
    }
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary, path)
    return path


def to_canonical(frame, feature_cols, preprocess_profile):
    frame = frame.copy()
    required = set(feature_cols)
    if preprocess_profile == "preclose_ratio":
        required.add("pre_close")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Raw datasource is missing fields: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame.sort_values(["instrument", "date"]).reset_index(drop=True)
    for column in OHLC_FIELDS:
        frame[column] = frame.groupby("instrument", sort=False)[column].ffill()
    if preprocess_profile == "preclose_ratio":
        denominator = pd.to_numeric(frame["pre_close"], errors="coerce").to_numpy(float)
        denominator_valid = np.isfinite(denominator) & (denominator > 0)
        for column in PRE_CLOSE_RATIO_FIELDS:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            valid = denominator_valid & np.isfinite(values) & (values > 0)
            fill = np.nan if column in OHLC_FIELDS else 0.0
            ratio = np.full(len(values), fill, dtype=np.float64)
            np.divide(values, denominator, out=ratio, where=valid)
            frame[column] = ratio
    for column in LOG_FIELDS:
        frame[column] = np.log1p(
            pd.to_numeric(frame[column], errors="coerce").clip(lower=0).astype(np.float64))
    return frame[["date", "instrument"] + list(feature_cols)]


def raw_query_fields(feature_cols, preprocess_profile):
    fields = list(feature_cols)
    if preprocess_profile == "preclose_ratio":
        fields.append("pre_close")
    return list(dict.fromkeys(fields))


def build_windows(frame, start_date, end_date, feature_cols, seq_len):
    lower = pd.Timestamp(start_date).normalize()
    upper = pd.Timestamp(end_date).normalize()
    windows, keys = [], []
    for instrument, sub in frame.groupby("instrument", sort=False):
        sub = sub.reset_index(drop=True)
        features = sub[feature_cols].to_numpy(np.float32)
        days = sub["date"].dt.normalize().to_numpy()
        end_positions = np.flatnonzero(np.append(days[1:] != days[:-1], True))
        for position in end_positions:
            sample_date = pd.Timestamp(days[position])
            if (position + 1 < seq_len or sample_date < lower or sample_date > upper):
                continue
            window = features[position - seq_len + 1:position + 1]
            if np.isfinite(window).all():
                windows.append(window)
                keys.append((sample_date, instrument))
    if not windows:
        return (np.empty((0, seq_len, len(feature_cols)), np.float32),
                pd.DataFrame(columns=["sample_date", "instrument"]))
    return (np.stack(windows).astype(np.float32),
            pd.DataFrame(keys, columns=["sample_date", "instrument"]))


def query_instruments(start_date, end_date):
    _require_dai()
    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]}).df()
    return sorted(frame["instrument"].astype(str).drop_duplicates().tolist())


def query_daily_universe(start_date, end_date):
    frame = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]}).df()
    frame["sample_date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame[["sample_date", "instrument"]]
    if frame.duplicated(["sample_date", "instrument"]).any():
        raise ValueError("Official universe contains duplicate daily keys")
    return frame


def query_adjusted_labels(table, start_date, end_date):
    buffer_start = (pd.Timestamp(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    buffer_end = (pd.Timestamp(end_date) + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    sql = f"""SELECT
        strftime(date, '%Y-%m-%d') AS sample_date,
        instrument,
        last(close ORDER BY date) AS raw_eod_close,
        last(adjust_factor ORDER BY date) AS daily_adjust_factor
      FROM {table}
      WHERE close > 0
      GROUP BY strftime(date, '%Y-%m-%d'), instrument
      ORDER BY instrument, sample_date"""
    labels = dai.query(sql, filters={"date": [buffer_start, buffer_end]}).df()
    labels["sample_date"] = pd.to_datetime(labels["sample_date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.sort_values(["instrument", "sample_date"]).reset_index(drop=True)
    close = pd.to_numeric(labels["raw_eod_close"], errors="coerce")
    factor = pd.to_numeric(labels["daily_adjust_factor"], errors="coerce")
    labels["adjusted_close"] = np.where(
        np.isfinite(close) & np.isfinite(factor) & (close > 0) & (factor > 0),
        close * factor, np.nan)
    labels["label_date"] = labels.groupby("instrument")["sample_date"].shift(-1)
    labels["next_adjusted_close"] = labels.groupby("instrument")["adjusted_close"].shift(-1)
    labels["target"] = labels["next_adjusted_close"] / labels["adjusted_close"] - 1.0
    labels = labels[
        (labels["sample_date"] >= pd.Timestamp(start_date)) &
        (labels["sample_date"] <= pd.Timestamp(end_date)) &
        (labels["label_date"] > pd.Timestamp(start_date)) &
        (labels["label_date"] <= pd.Timestamp(end_date)) &
        np.isfinite(labels["target"])
    ][["sample_date", "instrument", "label_date", "target"]].copy()
    if labels.duplicated(["sample_date", "instrument"]).any():
        raise ValueError("Adjusted labels contain duplicate daily keys")
    return labels


class RunningStats:
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, np.float64)
        self.m2 = np.zeros(n_features, np.float64)

    def update(self, array, max_rows=1_000_000):
        rows_per_sample = int(np.prod(array.shape[1:-1]))
        samples_per_chunk = max(1, max_rows // rows_per_sample)
        for start in range(0, len(array), samples_per_chunk):
            chunk = array[start:start + samples_per_chunk].reshape(-1, array.shape[-1])
            chunk = chunk.astype(np.float64, copy=False)
            count = len(chunk)
            mean = chunk.mean(axis=0, dtype=np.float64)
            centered = chunk - mean
            m2 = np.einsum("ij,ij->j", centered, centered, optimize=False)
            if self.n == 0:
                self.n, self.mean, self.m2 = count, mean.copy(), m2.copy()
            else:
                delta = mean - self.mean
                total = self.n + count
                self.m2 += m2 + delta ** 2 * (self.n * count / total)
                self.mean += delta * (count / total)
                self.n = total

    def finalize(self):
        std = np.sqrt(self.m2 / (self.n - 1))
        return self.mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)


def normalize_inplace(array, mean, std, chunk_size=100_000):
    for start in range(0, len(array), chunk_size):
        stop = min(start + chunk_size, len(array))
        np.subtract(array[start:stop], mean, out=array[start:stop])
        np.divide(array[start:stop], std, out=array[start:stop])
    return array


def build_training_dataset(table):
    _require_dai()
    labels = query_adjusted_labels(table, TRAIN_START, TRAIN_END)
    universe = query_daily_universe(TRAIN_START, TRAIN_END)
    instruments = query_instruments(TRAIN_START, TRAIN_END)
    fields = raw_query_fields(FEATURE_COLS, PREPROCESS_PROFILE)
    arrays, targets, sample_dates = [], [], []
    started = time.time()
    for year in range(2019, 2024):
        year_start, year_end = f"{year}-01-01", f"{year}-12-31"
        for offset in range(0, len(instruments), INSTRUMENT_CHUNK):
            chunk = instruments[offset:offset + INSTRUMENT_CHUNK]
            sql = (f"SELECT date, instrument, {', '.join(fields)} FROM {table} "
                   "ORDER BY instrument, date")
            frame = dai.query(
                sql, filters={"date": [year_start, year_end], "instrument": chunk}).df()
            if frame is None or frame.empty:
                continue
            frame = to_canonical(frame, FEATURE_COLS, PREPROCESS_PROFILE)
            X, index = build_windows(frame, year_start, year_end, FEATURE_COLS, SEQ_LEN)
            del frame
            if len(X) == 0:
                continue
            index["_row"] = np.arange(len(index), dtype=np.int64)
            selected = (index.merge(labels, on=["sample_date", "instrument"], how="inner")
                        .merge(universe, on=["sample_date", "instrument"], how="inner")
                        .sort_values("_row"))
            row_ids = selected["_row"].to_numpy(np.int64)
            arrays.append(X[row_ids])
            targets.append(selected["target"].to_numpy(np.float32))
            sample_dates.append(
                selected["sample_date"].to_numpy(copy=True))
            del X, index, selected, row_ids
            gc.collect()
        print(f"[train] built year={year}, elapsed={time.time() - started:.1f}s", flush=True)
    if not arrays:
        raise RuntimeError("No training samples were built")
    X = np.concatenate(arrays, axis=0)
    y = np.concatenate(targets).astype(np.float32)
    dates = np.concatenate(sample_dates)
    del arrays, targets, sample_dates, labels, universe
    gc.collect()
    if len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError("Training feature/label alignment or finiteness failure")
    stats = RunningStats(len(FEATURE_COLS))
    stats.update(X)
    mean, std = stats.finalize()
    normalize_inplace(X, mean, std)
    if TARGET_TYPE in {"raw", "clipped", "winsorized"}:
        lower, upper = np.percentile(y, [1, 99])
        y = np.clip(y, lower, upper).astype(np.float32)
    elif TARGET_TYPE == "cs_rank":
        rank_frame = pd.DataFrame({
            "sample_date": pd.to_datetime(dates),
            "target": y.astype(np.float64, copy=False),
        })
        percentile = rank_frame.groupby(
            "sample_date", sort=False)["target"].rank(
                method="average", pct=True)
        y = (percentile.to_numpy(np.float64) * 2.0 - 1.0).astype(np.float32)
        if not np.isfinite(y).all():
            raise RuntimeError("Cross-sectional rank target contains NaN/Inf")
    elif TARGET_TYPE != "raw_unclipped":
        raise ValueError(f"Unsupported submission target: {TARGET_TYPE}")
    print(f"[train] samples={len(X):,}, scaler_count={stats.n:,}", flush=True)
    return X, y, mean, std


def set_determinism(device):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_float32_matmul_precision("high")


def train_and_save(datasources, model_path=MODEL_PATH):
    """Train the declared architecture from scratch and write model.json."""
    _require_dai()
    table = datasources.get(BAR_KEY)
    if table is None:
        raise KeyError(f"Required datasource {BAR_KEY!r}; got {sorted(datasources)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(device)
    X, y, mean, std = build_training_dataset(table)
    model = create_model().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=LR_T0, T_mult=LR_T_MULT, eta_min=LR_ETA_MIN)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(TRAIN_EPOCHS):
        started = time.time()
        if epoch < WARMUP_EPOCHS:
            for group in optimizer.param_groups:
                group["lr"] = LR * (epoch + 1) / WARMUP_EPOCHS
        model.train()
        total, batches = 0.0, 0
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = model(features)
                if LOSS_NAME == "mse":
                    loss = nn.functional.mse_loss(prediction, target)
                elif LOSS_NAME == "huber":
                    loss = nn.functional.smooth_l1_loss(
                        prediction, target, beta=0.01)
                else:
                    raise ValueError(f"Unsupported submission loss: {LOSS_NAME}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            batches += 1
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
        print(
            f"[train] epoch={epoch + 1}/{TRAIN_EPOCHS}, "
            f"loss={total / max(batches, 1):.8f}, elapsed={time.time() - started:.1f}s",
            flush=True)
    save_model_json(model, mean, std, model_path)
    print(f"[train] saved={model_path}", flush=True)
    return model_path


def build_inference_dataset(table, start_date, end_date, instruments,
                            checkpoint, mean, std):
    buffer_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fields = raw_query_fields(
        checkpoint["feature_cols"], checkpoint.get("preprocess_profile", "standard"))
    sql = (f"SELECT date, instrument, {', '.join(fields)} FROM {table} "
           "ORDER BY instrument, date")
    frame = dai.query(
        sql, filters={"date": [buffer_start, end_date], "instrument": instruments}).df()
    frame = to_canonical(
        frame, checkpoint["feature_cols"],
        checkpoint.get("preprocess_profile", "standard"))
    X, index = build_windows(
        frame, start_date, end_date,
        checkpoint["feature_cols"], int(checkpoint["seq_len"]))
    del frame
    if len(X) == 0:
        raise RuntimeError(f"No inference samples for {start_date}~{end_date}")
    normalize_inplace(X, mean, std)
    return X, index


def infer_scores(datasources, start_date, end_date, model_path=MODEL_PATH):
    """Public leaderboard entry: load bundled weights and return three columns."""
    _require_dai()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_model_json(model_path, map_location=device)
    if checkpoint["feature_cols"] != FEATURE_COLS:
        raise ValueError("Bundled checkpoint feature order differs from train.py")
    if checkpoint.get("preprocess_profile", "standard") != PREPROCESS_PROFILE:
        raise ValueError("Bundled checkpoint preprocessing differs from train.py")
    if checkpoint.get("model_config", {}).get("input_representation") != INPUT_REPRESENTATION:
        raise ValueError("Bundled checkpoint representation differs from train.py")
    model = create_model(
        checkpoint["feature_cols"], checkpoint.get("model_config")).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    mean = np.asarray(checkpoint["mean"], np.float32)
    std = np.asarray(checkpoint["std"], np.float32)
    if mean.shape != (len(FEATURE_COLS),) or std.shape != mean.shape or not np.all(std > 0):
        raise ValueError("Invalid bundled training scaler")
    table = datasources.get(BAR_KEY)
    if table is None:
        raise KeyError(f"Required datasource {BAR_KEY!r}; got {sorted(datasources)}")
    instruments = query_instruments(start_date, end_date)
    parts = []
    for offset in range(0, len(instruments), INFERENCE_INSTRUMENT_CHUNK):
        chunk = instruments[offset:offset + INFERENCE_INSTRUMENT_CHUNK]
        X, index = build_inference_dataset(
            table, start_date, end_date, chunk, checkpoint, mean, std)
        tensor = torch.from_numpy(X)
        predictions = []
        with torch.inference_mode():
            for row in range(0, len(tensor), INFERENCE_BATCH_SIZE):
                predictions.append(
                    model(tensor[row:row + INFERENCE_BATCH_SIZE].to(device)).cpu().numpy())
        index["score"] = np.concatenate(predictions).astype(np.float64)
        parts.append(index.rename(columns={"sample_date": "date"}))
        del X, tensor, predictions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.concat(parts, ignore_index=True)
    universe = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]}).df()
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["instrument"] = universe["instrument"].astype(str)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result = (result.merge(universe, on=["date", "instrument"], how="inner")
              .replace([np.inf, -np.inf], np.nan)
              .dropna(subset=["score"])
              .drop_duplicates(["date", "instrument"])
              [["date", "instrument", "score"]]
              .reset_index(drop=True))
    if result.empty or result.duplicated(["date", "instrument"]).any():
        raise RuntimeError("Invalid inference keys or empty output")
    if not np.isfinite(result["score"]).all():
        raise RuntimeError("Inference produced NaN/Inf")
    print(
        f"[infer] submission={SUBMISSION_NAME}, rows={len(result):,}, "
        f"dates={result['date'].nunique()}", flush=True)
    return result


if __name__ == "__main__":
    train_and_save({BAR_KEY: DEFAULT_BAR_TABLE})
