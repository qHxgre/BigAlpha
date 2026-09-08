# -*- coding: utf-8 -*-
"""ResidualAlpha V4 model for BigAlpha 2026 E2E track.

V4 相对 V3 只改一处：字段归一化的门控。
V3 的 mix_logit 训练后 26 个字段全落在 0.741~0.760（初始 0.750），几乎没动，
退化成固定的 75% shape / 25% level 混合，且被 weight decay 往 level 方向拽。
那 25% 的 level 通路是纯市值/价格代理，恰好被评估端的风格回归剃掉。

Design goals:
1. Remove stock-level absolute price/liquidity style information inside the network.
2. Respect the 3-level LOB spatial layout instead of treating all fields identically.
3. Use a stable temporal-convolution backbone with one lightweight attention layer.
4. Use cross-sectional market context without expensive/universe-fragile full stock attention.
5. Optimize daily ranking/tails, which is closer to the competition's neutralized factor score.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Field order must match transformer_train_local.py
BOOK_PRICE_INDICES = [8, 11, 9, 12, 10, 13]   # ask1,bid1,ask2,bid2,ask3,bid3
BOOK_VOLUME_INDICES = [14, 17, 15, 18, 16, 19]
BOOK_ORDER_INDICES = [20, 23, 21, 24, 22, 25]
TRADE_INDICES = list(range(8))


# level 通路的最大占比。评估端会剃掉风格暴露，绝对价格/流动性水平基本是
# 被剃掉的那部分，所以给它设一个硬上限，而不是让门控自由漂。
LEVEL_PATH_MAX = 0.20


class MaskedAdaptiveFieldNorm(nn.Module):
    """Per-stock, per-field normalization over the complete lookback sequence.

    门控范围被限制在 [1 - LEVEL_PATH_MAX, 1.0]，初始化接近纯 shape 通路。
    训练侧必须把 mix_logit / output_scale / output_bias 排除出 weight decay，
    否则 AdamW 会把 mix_logit 拉向 0，反而放大被中性化剃掉的 level 分量。
    """

    def __init__(self, n_features: int):
        super().__init__()
        # sigmoid(1.5)=0.818 -> gate = 0.80 + 0.20*0.818 = 0.964，起步几乎纯 shape
        self.mix_logit = nn.Parameter(torch.full((n_features,), 1.50))
        self.output_scale = nn.Parameter(torch.ones(n_features))
        self.output_bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
        # x: (S,L,M,F), day_mask: (S,L)
        mask = day_mask[:, :, None, None].to(dtype=x.dtype)
        count = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0) * x.shape[2]
        mean = (x * mask).sum(dim=(1, 2), keepdim=True) / count
        var = ((x - mean).pow(2) * mask).sum(dim=(1, 2), keepdim=True) / count
        shape_x = (x - mean) / torch.sqrt(var + 1e-5)
        shape_x = torch.clamp(shape_x, -8.0, 8.0)
        level_x = torch.tanh(torch.clamp(x, -10.0, 10.0))
        gate = (
            1.0 - LEVEL_PATH_MAX
            + LEVEL_PATH_MAX * torch.sigmoid(self.mix_logit)
        ).view(1, 1, 1, -1)
        out = gate * shape_x + (1.0 - gate) * level_x
        out = out * self.output_scale.view(1, 1, 1, -1)
        out = out + self.output_bias.view(1, 1, 1, -1)
        return out * mask


class AttentivePool(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)


class GatedDepthwiseBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(8, dim)
        self.dw = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding,
            dilation=dilation, groups=dim,
        )
        self.pw = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(self.norm(x))
        a, b = self.pw(h).chunk(2, dim=1)
        h = F.gelu(a) * torch.sigmoid(b)
        h = self.dropout(self.out(h))
        return x + self.res_scale * h


class TradeTemporalEncoder(nn.Module):
    def __init__(self, out_dim: int = 96, dropout: float = 0.08):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(len(TRADE_INDICES), out_dim, kernel_size=5, stride=5),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            GatedDepthwiseBlock1D(out_dim, 3, 1, dropout),
            GatedDepthwiseBlock1D(out_dim, 3, 2, dropout),
            GatedDepthwiseBlock1D(out_dim, 5, 2, dropout),
        ])
        self.pool = AttentivePool(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,240,F)
        h = x[:, :, TRADE_INDICES].transpose(1, 2)
        h = self.stem(h)
        for block in self.blocks:
            h = block(h)
        return self.pool(h.transpose(1, 2))


class BookSpatialEncoder(nn.Module):
    """DeepLOB-inspired 3-level order-book encoder.

    Width layout is [ask1,bid1,ask2,bid2,ask3,bid3], so a width-2 kernel learns
    price/volume/order interactions within each level and across the two sides.
    """

    def __init__(self, out_dim: int = 96, dropout: float = 0.08):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(5, 2), stride=(5, 2)),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48),
            nn.Conv2d(48, out_dim * 2, kernel_size=1),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.pool = AttentivePool(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,240,F)
        price = x[:, :, BOOK_PRICE_INDICES]
        volume = x[:, :, BOOK_VOLUME_INDICES]
        orders = x[:, :, BOOK_ORDER_INDICES]
        book = torch.stack([price, volume, orders], dim=1)  # (B,3,T,6)
        h = self.stem(book)
        a, b = self.local(h).chunk(2, dim=1)
        h = h.mean(dim=1, keepdim=True) + self.proj(F.gelu(a) * torch.sigmoid(b))
        tokens = h.flatten(2).transpose(1, 2)
        return self.pool(tokens)


class PatchShapeBranch(nn.Module):
    def __init__(self, n_features: int, out_dim: int, patch_size: int, dropout: float):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(n_features, out_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )
        self.block = GatedDepthwiseBlock1D(out_dim, 3, 1, dropout)
        self.pool = AttentivePool(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(self.embed(x.transpose(1, 2)))
        return self.pool(h.transpose(1, 2))


class StructuredIntradayEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 192,
        trade_dim: int = 96,
        book_dim: int = 96,
        patch_dim: int = 64,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.field_norm = MaskedAdaptiveFieldNorm(n_features)
        self.trade = TradeTemporalEncoder(trade_dim, dropout)
        self.book = BookSpatialEncoder(book_dim, dropout)
        self.patch_15 = PatchShapeBranch(n_features, patch_dim, 15, dropout)
        self.patch_30 = PatchShapeBranch(n_features, patch_dim, 30, dropout)
        total = trade_dim + book_dim + 2 * patch_dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.skip = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
        # x: (S,L,240,F)
        x = self.field_norm(x, day_mask)
        s, l, m, f = x.shape
        flat = x.reshape(s * l, m, f)
        fused = torch.cat([
            self.trade(flat), self.book(flat),
            self.patch_15(flat), self.patch_30(flat),
        ], dim=-1)
        last_state = self.skip(flat[:, -1, :])
        day = self.out_norm(self.fuse(fused) + 0.25 * last_state)
        return day.reshape(s, l, -1) * day_mask.unsqueeze(-1)


class GatedTemporalMixer(nn.Module):
    def __init__(self, d_model: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=dilation,
            dilation=dilation, groups=d_model,
        )
        self.gate = nn.Linear(d_model, d_model * 2)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
            nn.Dropout(dropout),
        )
        self.temporal_scale = nn.Parameter(torch.tensor(0.10))
        self.ffn_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        a, b = self.gate(h).chunk(2, dim=-1)
        h = self.out(F.gelu(a) * torch.sigmoid(b))
        x = x + self.temporal_scale * h
        x = x + self.ffn_scale * self.ffn(x)
        return x * mask.unsqueeze(-1)


class CrossDayShapeEncoder(nn.Module):
    def __init__(
        self,
        lookback_days: int,
        d_model: int = 192,
        n_heads: int = 6,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, lookback_days, d_model))
        self.mixers = nn.ModuleList([
            GatedTemporalMixer(d_model, 1, dropout),
            GatedTemporalMixer(d_model, 2, dropout),
            GatedTemporalMixer(d_model, 4, dropout),
            GatedTemporalMixer(d_model, 8, dropout),
        ])
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            layer, num_layers=1, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.pool = AttentivePool(d_model)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = (x + self.position[:, :x.shape[1]]) * mask.unsqueeze(-1)
        for block in self.mixers:
            h = block(h, mask)
        h = self.attention(h, src_key_padding_mask=~mask.bool())
        pooled = self.pool(h, mask)
        mean = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        final = h[:, -1]
        return self.fuse(torch.cat([final, pooled, mean], dim=-1))


class StableMarketContext(nn.Module):
    """DeepSets-style cross-sectional context, robust to changing stock count."""

    def __init__(self, d_model: int = 192, dropout: float = 0.08):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 5),
            nn.Linear(d_model * 5, d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.context_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=0, keepdim=True) + 1e-5)
        weights = torch.softmax(self.attn_score(x).squeeze(-1), dim=0).unsqueeze(-1)
        attn = (x * weights).sum(dim=0, keepdim=True)
        n = x.shape[0]
        context = torch.cat([
            x,
            x - mean,
            mean.expand(n, -1),
            std.expand(n, -1),
            attn.expand(n, -1),
        ], dim=-1)
        h = x + self.context_scale * self.fuse(context)
        return h + 0.10 * self.ffn(h)


class CompetitionE2EModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        lookback_days: int,
        n_targets: int = 3,
        d_model: int = 192,
        n_heads: int = 6,
        dropout: float = 0.08,
        **_: object,
    ):
        super().__init__()
        self.intraday = StructuredIntradayEncoder(
            n_features=n_features, d_model=d_model, dropout=dropout,
        )
        self.cross_day = CrossDayShapeEncoder(
            lookback_days=lookback_days, d_model=d_model,
            n_heads=n_heads, dropout=dropout,
        )
        self.cross_section = StableMarketContext(d_model, dropout)
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(d_model, 1)
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(n_targets)
        ])
        self.horizon_scale = nn.Parameter(torch.full((n_targets,), 0.10))

    def forward(self, x: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
        day = self.intraday(x, day_mask)
        stock = self.cross_day(day, day_mask)
        stock = self.cross_section(stock)
        h = self.shared(stock)
        base = self.base_head(h)
        offsets = torch.cat([head(h) for head in self.horizon_heads], dim=1)
        return base + offsets * self.horizon_scale.view(1, -1)


MODEL_CONFIG = {
    "n_features": 26,
    "lookback_days": 20,
    "n_targets": 3,
    "d_model": 192,
    "n_heads": 6,
    "dropout": 0.08,
}

# 预测周期与权重。评估口径未公开，默认沿用 V3 已验证过的 1/3/5 日配置；
# 想试 (1,5,10,20) 只需同时改这里和 train_local 的 HORIZONS，标签会自动重建。
HORIZONS = (1, 3, 5)
HORIZON_WEIGHTS = torch.tensor([0.72, 0.20, 0.08], dtype=torch.float32)
PREDICTION_WEIGHTS = [0.78, 0.17, 0.05]


def stable_zscore(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    n = values.numel()
    order = torch.argsort(values.detach())
    ranks = torch.empty_like(values)
    ranks[order] = torch.linspace(-1.0, 1.0, n, device=values.device, dtype=values.dtype)
    return ranks


def robust_rank_target(target: torch.Tensor) -> torch.Tensor:
    lo = torch.quantile(target.detach(), 0.01)
    hi = torch.quantile(target.detach(), 0.99)
    return percentile_rank(target.clamp(lo, hi))


def pearson_ic(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = prediction - prediction.mean()
    t = target - target.mean()
    return (p * t).mean() / torch.sqrt(p.pow(2).mean() * t.pow(2).mean() + eps)


def focused_pairwise_loss(prediction: torch.Tensor, target_rank: torch.Tensor) -> torch.Tensor:
    n = prediction.numel()
    k = max(24, int(n * 0.18))
    order = torch.argsort(target_rank)
    selected = torch.cat([order[:k], order[-k:]])
    p = stable_zscore(prediction[selected])
    t = target_rank[selected]
    pdiff = p[:, None] - p[None, :]
    tdiff = t[:, None] - t[None, :]
    mask = torch.triu(torch.ones_like(pdiff, dtype=torch.bool), diagonal=1)
    direction = torch.sign(tdiff[mask])
    weight = tdiff[mask].abs().clamp(0.20, 2.0)
    return (F.softplus(-direction * pdiff[mask]) * weight).mean()


def listwise_tail_loss(prediction: torch.Tensor, target_rank: torch.Tensor) -> torch.Tensor:
    n = prediction.numel()
    k = max(16, int(n * 0.10))
    order = torch.argsort(target_rank)
    bottom, top = order[:k], order[-k:]
    p = stable_zscore(prediction)
    spread = p[top].mean() - p[bottom].mean()
    # Directional softmax puts probability mass on the true upper and lower tails.
    top_ce = -torch.log_softmax(p, dim=0)[top].mean()
    bottom_ce = -torch.log_softmax(-p, dim=0)[bottom].mean()
    return 0.55 * F.softplus(1.20 - spread) + 0.225 * top_ce + 0.225 * bottom_ce


def competition_multitask_loss(predictions: torch.Tensor, targets: torch.Tensor):
    weights = HORIZON_WEIGHTS.to(predictions.device)[: predictions.shape[1]]
    total = predictions.new_tensor(0.0)
    records = {}
    head_z = []
    for j in range(predictions.shape[1]):
        p = predictions[:, j]
        t = robust_rank_target(targets[:, j])
        pz = stable_zscore(p)
        ic = pearson_ic(pz, t)
        pair = focused_pairwise_loss(pz, t)
        tail = listwise_tail_loss(pz, t)
        huber = F.smooth_l1_loss(pz, t)
        loss_j = 0.58 * (1.0 - ic) + 0.20 * pair + 0.17 * tail + 0.05 * huber
        total = total + weights[j] * loss_j
        records[f"ic_{j}"] = ic.detach()
        records[f"pair_{j}"] = pair.detach()
        records[f"tail_{j}"] = tail.detach()
        head_z.append(pz)

    # Keep horizons related but not identical; this reduces unstable head cancellation at inference.
    consistency = predictions.new_tensor(0.0)
    if len(head_z) > 1:
        pairs = [
            F.smooth_l1_loss(head_z[i], head_z[i + 1])
            for i in range(len(head_z) - 1)
        ]
        consistency = torch.stack(pairs).mean()
    total = total + 0.025 * consistency
    records["consistency"] = consistency.detach()
    records["loss"] = total
    return records
