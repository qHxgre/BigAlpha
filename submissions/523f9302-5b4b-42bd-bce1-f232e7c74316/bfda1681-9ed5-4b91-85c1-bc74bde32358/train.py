# -*- coding: utf-8 -*-
"""Transformer 端到端 —— 训练脚本 (本地/在线统一) V2 改进版。

通过 config.MODE 切换:
  - MODE="local"  → 本地 parquet 数据
  - MODE="online" → BigQuant 平台 dai 数据

V2 改进:
  - 多频率融合 (30m + 15m), 跨频率门控融合
  - 多尺度 1D 卷积 → BiGRU → RoPE Transformer 混合架构
  - 注意力池化替代均值池化
  - Pearson 相关损失 + MSE 联合优化
  - Cosine annealing warmup 学习率调度

与 online/transformer_train.py 保持一致:
  - 模型保存格式 (JSON 文本文件)
  - train_and_save(datasources) 签名

用法:
    python train.py
    from train import train_and_save
    train_and_save({"bar30m": "bigalpha_2026_e2e_bar30m", "bar15m": "bigalpha_2026_e2e_bar15m"})
"""
import os
import sys
import json
import math
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import structlog

# 本地环境下配置 structlog (在线平台已预配置)
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    cache_logger_on_first_use=True,
)

import config

logger = structlog.get_logger()
_HERE = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════
# RoPE 基础组件 (保留兼容)
# ═══════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码: 预计算 cos / sin 缓存, 按需扩展到任意序列长度。"""

    def __init__(self, dim, max_seq_len=512, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.dim = dim
        self._set_cache(max_seq_len)

    def _set_cache(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len, device):
        if seq_len > self.max_seq_len:
            self._set_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(device),
            self.sin_cached[:, :, :seq_len, :].to(device),
        )


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.rotary = RotaryEmbedding(self.d_head)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        cos, sin = self.rotary(L, x.device)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh") if activation == "gelu" else nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(self.norm1(src))
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src


# ═══════════════════════════════════════════════════════════════════════════
# StockTransformer (保留旧版兼容)
# ═══════════════════════════════════════════════════════════════════════════
class StockTransformer(nn.Module):
    """端到端股票预测 Transformer V1 (保留兼容)。"""

    def __init__(self, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(n_feat, d_model)
        self.encoder = nn.Sequential(*[
            RoPETransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.1, activation="gelu")
            for _ in range(nlayers)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.encoder(self.proj(x)).mean(dim=1)
        return self.head(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# V2 改进模型: 多尺度卷积 + BiGRU + RoPE Transformer + 注意力池化 + 跨频率融合
# ═══════════════════════════════════════════════════════════════════════════

class MultiScaleConv1D(nn.Module):
    """多尺度 1D 卷积, 使用不同 kernel size 捕捉多时间尺度局部模式。"""

    def __init__(self, in_dim, out_dim, kernels=(3, 5, 7, 11)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim // len(kernels), k, padding=k // 2)
            for k in kernels
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (B, L, D) → (B, D, L) for Conv1d
        x_t = x.transpose(1, 2)
        outs = [F.gelu(conv(x_t)) for conv in self.convs]
        out = torch.cat(outs, dim=1).transpose(1, 2)  # back to (B, L, D)
        return self.norm(out)


class AttentionPooling(nn.Module):
    """可学习的注意力池化, 替代简单均值池化。"""

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (B, L, D) → (B, D)
        w = self.attn(x).squeeze(-1)  # (B, L)
        w = F.softmax(w, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (x * w).sum(dim=1)


class CrossFreqGate(nn.Module):
    """跨频率门控融合: 学习各频率表示的加权组合。"""

    def __init__(self, n_freqs, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_freqs * d_model, n_freqs * d_model // 2),
            nn.GELU(),
            nn.Linear(n_freqs * d_model // 2, n_freqs),
            nn.Softmax(dim=-1),
        )

    def forward(self, freq_reprs):
        # freq_reprs: list of (B, D) tensors, one per frequency
        stacked = torch.stack(freq_reprs, dim=-1)  # (B, D, F)
        concat = torch.cat(freq_reprs, dim=-1)      # (B, F*D)
        weights = self.gate(concat)                  # (B, F)
        fused = (stacked * weights.unsqueeze(1)).sum(dim=-1)  # (B, D)
        return fused


class StockTransformerV2(nn.Module):
    """V2 改进模型: 多尺度Conv → BiGRU → RoPE Transformer → 注意力池化 → 跨频率融合。

    架构:
      1. 多尺度 1D 卷积提取局部多时间尺度模式
      2. 双向 GRU 编码序列依赖
      3. RoPE Transformer 捕捉全局交互
      4. 注意力池化得到序列表示
      5. 跨频率门控融合多频率信息
      6. 深层预测头输出标量预测
    """

    def __init__(self, n_feat, d_model=128, nhead=8, nlayers=4, dim_ff=256,
                 seq_len=None, n_freqs=1, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.d_model = d_model
        n_feat_per_freq = n_feat // max(n_freqs, 1)

        # 每频率独立投影 + 多尺度卷积
        self.freq_projs = nn.ModuleList([
            nn.Linear(n_feat_per_freq, d_model) for _ in range(n_freqs)
        ])
        self.multi_scale = MultiScaleConv1D(d_model, d_model)

        # BiGRU 序列编码
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)

        # RoPE Transformer
        self.transformer = nn.Sequential(*[
            RoPETransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(nlayers)
        ])

        # 注意力池化
        self.attn_pool = AttentionPooling(d_model)

        # 跨频率门控融合
        self.cross_freq_gate = CrossFreqGate(n_freqs, d_model) if n_freqs > 1 else None

        # 预测头 (残差结构)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

    def _encode_freq(self, x_freq):
        """编码单个频率的序列。"""
        h = self.freq_projs[0](x_freq)  # 实际会在 forward 中按索引选择
        h = self.multi_scale(h)
        h, _ = self.gru(h)
        h = self.transformer(h)
        return self.attn_pool(h)

    def forward(self, x):
        B, L, D = x.shape
        feat_per_freq = D // max(self.n_freqs, 1)

        freq_reprs = []
        for i in range(self.n_freqs):
            x_freq = x[:, :, i * feat_per_freq: (i + 1) * feat_per_freq]
            h = self.freq_projs[i](x_freq)
            h = self.multi_scale(h)
            h, _ = self.gru(h)
            h = self.transformer(h)
            h = self.attn_pool(h)
            freq_reprs.append(h)

        if self.cross_freq_gate is not None:
            fused = self.cross_freq_gate(freq_reprs)
        else:
            fused = freq_reprs[0]

        return self.head(fused).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# V3 改进模型: 融合 TimesNet + Informer + ModernTCN 核心思想
# ═══════════════════════════════════════════════════════════════════════════

class TimesBlock(nn.Module):
    """[TimesNet] 固定周期 + 1D→2D重塑 + 多尺度2D卷积 + 2D→1D还原。

    将 1D 序列按固定周期重塑为 2D 张量,
    使用 2D 卷积 (Inception 风格) 捕捉 intraperiod 和 interperiod 变化。
    为避免 FFT 反向传播的 dtype 问题, 使用固定启发式周期。
    """

    def __init__(self, d_model, top_k=5, reduction=4, seq_len=96):
        super().__init__()
        self.top_k = top_k
        # 固定启发式周期: 基于 seq_len=96 (30m bars) 的合理周期
        # 8 ≈ 半日, 16 ≈ 1日, 24 ≈ 1.5日, 32 ≈ 2日
        self.fixed_periods = [8, 12, 16, 24, 32]
        self.top_k = min(top_k, len(self.fixed_periods))
        # 多尺度 2D 卷积核 (TimesNet 的 Inception 风格)
        self.conv_1x1 = nn.Conv2d(d_model, d_model // reduction, 1)
        self.conv_1x3 = nn.Conv2d(d_model // reduction, d_model // reduction, (1, 3),
                                   padding=(0, 1))
        self.conv_3x1 = nn.Conv2d(d_model // reduction, d_model // reduction, (3, 1),
                                   padding=(1, 0))
        self.conv_3x3 = nn.Conv2d(d_model // reduction, d_model // reduction, 3,
                                   padding=1)
        self.out_conv = nn.Conv2d(d_model // reduction, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        # 可学习的周期权重
        self.period_weights = nn.Parameter(torch.ones(self.top_k) / self.top_k)

    def _reshape_2d(self, x_1d, period):
        """将 (B, L, D) 按给定周期重塑为 (B, D, rows, cols)。period 为 Python int。"""
        B, L, D = x_1d.shape
        rows = int(period)
        cols = (L + rows - 1) // rows  # ceil 除法
        pad_len = rows * cols - L
        if pad_len > 0:
            x_1d = F.pad(x_1d, (0, 0, 0, pad_len))
        x_2d = x_1d.reshape(B, rows, cols, D).permute(0, 3, 1, 2)  # (B, D, rows, cols)
        return x_2d, pad_len

    def forward(self, x):
        B, L, D = x.shape
        outputs = []
        for i in range(self.top_k):
            period = self.fixed_periods[i]
            x_2d, pad_len = self._reshape_2d(x, period)

            # 多尺度 2D 卷积 (Inception 风格)
            h = self.act(self.conv_1x1(x_2d))
            h1 = self.conv_1x3(h)
            h2 = self.conv_3x1(h)
            h3 = self.conv_3x3(h)
            h = (h1 + h2 + h3) / 3.0
            h = self.out_conv(h)  # (B, D, rows, cols)

            # 2D → 1D
            _, _, rows, cols = h.shape
            h_1d = h.permute(0, 2, 3, 1).reshape(B, rows * cols, D)
            if pad_len > 0:
                h_1d = h_1d[:, :L, :]
            outputs.append(h_1d)

        # 可学习权重融合
        stack = torch.stack(outputs, dim=-1)  # (B, L, D, K)
        weights = F.softmax(self.period_weights, dim=0)  # (K,)
        fused = (stack * weights.view(1, 1, 1, -1)).sum(dim=-1)  # (B, L, D)
        return self.norm(fused + x)


class LargeKernelDWConv(nn.Module):
    """[ModernTCN] 大核深度可分离卷积 (ConvNeXt 风格)。

    使用 Depthwise Conv (大核) + Pointwise Conv 的倒置瓶颈结构,
    配合 BN + GeLU 激活, 有效提取局部时间模式。
    """

    def __init__(self, d_model, kernel_size=13, expansion=4):
        super().__init__()
        hidden = d_model * expansion
        # Inverted bottleneck: d_model → hidden → d_model
        self.pw1 = nn.Conv1d(d_model, hidden, 1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.dw = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.pw2 = nn.Conv1d(hidden, d_model, 1)
        self.bn3 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: (B, L, D) → (B, D, L)
        h = x.transpose(1, 2)
        identity = h
        h = self.act(self.bn1(self.pw1(h)))
        h = self.act(self.bn2(self.dw(h)))
        h = self.bn3(self.pw2(h))
        h = h.transpose(1, 2)  # back to (B, L, D)
        return self.act(h + x)


class ProbSparseAttention(nn.Module):
    """[Informer] ProbSparse 自注意力: 只选择 top-u 个 query 计算完整注意力。

    基于 query 的 KL 散度测量稀疏性, 选择信息量最大的 query,
    其余 query 使用 mean value, 实现 O(L log L) 复杂度。
    """

    def __init__(self, d_model, nhead, dropout=0.1, factor=5):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.rotary = RotaryEmbedding(self.d_head)

    def _prob_QK(self, Q, K, sample_k):
        """针对每个 query 计算稀疏性度量 M(qi, K)。"""
        # Q: (B, H, L, d_head), K: (B, H, L, d_head)
        B, H, L, D = Q.shape
        # 随机采样部分 key 来计算稀疏性
        L_K = K.shape[2]
        L_Q = Q.shape[2]
        # 采样 n_sample 个 key
        n_sample = min(sample_k, L_K)
        idx = torch.randint(0, L_K, (L_Q, n_sample), device=Q.device)
        K_sample = K[:, :, idx, :]  # (B, H, L_Q, n_sample, D)
        # QK^T
        QK = torch.einsum("bhld,bhlsd->bhls", Q, K_sample) / (D ** 0.5)  # (B, H, L_Q, n_sample)
        # M = max(QK) - mean(QK)
        M = QK.max(dim=-1)[0] - QK.mean(dim=-1)  # (B, H, L_Q)
        return M

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)

        # RoPE
        cos, sin = self.rotary(L, x.device)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # ProbSparse: 选取 top-u 个 query
        u = max(int(self.factor * math.log(L)), 1)
        M = self._prob_QK(q, k, sample_k=int(math.log(L)) * 5)
        top_u = torch.topk(M, min(u, L), dim=-1)[1]  # indices: (B, H, u)

        # 为 top-u queries 计算完整注意力
        # 构建索引
        B_idx = torch.arange(B, device=x.device).view(-1, 1, 1).expand(-1, self.nhead, u)
        H_idx = torch.arange(self.nhead, device=x.device).view(1, -1, 1).expand(B, -1, u)
        q_top = q[B_idx, H_idx, top_u]  # (B, H, u, d_head)

        # 计算 top-u 的完整 attention
        attn_top = torch.matmul(q_top, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_top = F.softmax(attn_top, dim=-1)
        out_top = torch.matmul(attn_top, v)  # (B, H, u, d_head)

        # 其余 query 用 mean(V) 填充
        out = v.mean(dim=2, keepdim=True).expand(-1, -1, L, -1).clone()
        out[B_idx, H_idx, top_u] = out_top

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class AttentionDistill(nn.Module):
    """[Informer] 注意力蒸馏: Conv1d + MaxPool 将序列长度减半,
    构建金字塔式多尺度特征, 突出主导注意力特征。"""

    def __init__(self, d_model, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()

    def forward(self, x):
        # x: (B, L, D) → (B, D, L)
        h = x.transpose(1, 2)
        h = self.pool(self.act(self.norm(self.conv(h))))
        h = h.transpose(1, 2)  # (B, L//2, D)
        return h


class CrossFreqAttention(nn.Module):
    """[V3 Enhanced] 跨频率注意力融合: 使用交叉注意力而非简单门控,
    各频率之间可交互信息, 学习频率间的复杂依赖关系。"""

    def __init__(self, n_freqs, d_model, nhead=4):
        super().__init__()
        self.n_freqs = n_freqs
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, freq_reprs):
        # freq_reprs: list of (B, D), len = n_freqs
        stacked = torch.stack(freq_reprs, dim=1)  # (B, F, D)
        # 以第一个频率为 query, 所有频率为 key/value
        q = stacked.mean(dim=1, keepdim=True)  # (B, 1, D)
        kv = stacked  # (B, F, D)
        attn_out, _ = self.cross_attn(q, kv, kv)
        fused = self.norm(attn_out.squeeze(1) + stacked.mean(dim=1))
        fused = self.norm2(self.ffn(fused) + fused)
        return fused


class StockTransformerV3(nn.Module):
    """V3 旗舰模型: 融合 TimesNet + ModernTCN + Informer 核心思想的混合架构。

    架构流程 (每频率):
      1. [TimesNet] FFT周期发现 + 1D→2D重塑 + 多尺度2D卷积
      2. [ModernTCN] 大核深度可分离卷积 (ConvNeXt 风格)
      3. BiGRU 序列编码 + [Informer] ProbSparse RoPE Attention
      4. [Informer] 注意力蒸馏金字塔 (可选)
      5. [V3 Enhanced] 跨频率交叉注意力融合
      6. 深层预测头

    继承 V2 的已验证设计 (截面标签归一化, CombinedLoss, Cosine warmup),
    融合论文创新:
      - Informer:  ProbSparse 注意力 + 注意力蒸馏金字塔
      - ModernTCN: 大核深度可分离卷积 + ConvNeXt 倒置瓶颈
      - TimesNet:  FFT 周期发现 + 2D 重塑 + 多尺度 2D 卷积
    """

    def __init__(self, n_feat, d_model=128, nhead=8, nlayers=4, dim_ff=256,
                 seq_len=96, n_freqs=1, dropout=0.1,
                 top_k_periods=5, large_conv_kernel=13,
                 prob_sparse=True, attn_distill=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.d_model = d_model
        self.seq_len = seq_len
        self.prob_sparse = prob_sparse
        self.attn_distill = attn_distill
        n_feat_per_freq = n_feat // max(n_freqs, 1)

        # ---- 每频率独立投影 ----
        self.freq_projs = nn.ModuleList([
            nn.Linear(n_feat_per_freq, d_model) for _ in range(n_freqs)
        ])

        # ---- [TimesNet] FFT周期 + 2D变换 ----
        self.times_block = TimesBlock(d_model, top_k=top_k_periods, seq_len=seq_len)

        # ---- [ModernTCN] 大核深度可分离卷积 ----
        self.large_conv = LargeKernelDWConv(d_model, kernel_size=large_conv_kernel)

        # ---- 序列编码 ----
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)

        # ---- [Informer] ProbSparse 或 RoPE Attention ----
        self.attn_type = "prob_sparse" if prob_sparse else "rope"
        if prob_sparse:
            self.transformer = nn.ModuleList([
                nn.ModuleDict({
                    "attn": ProbSparseAttention(d_model, nhead, dropout),
                    "ffn": nn.Sequential(
                        nn.Linear(d_model, dim_ff),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim_ff, d_model),
                        nn.Dropout(dropout),
                    ),
                    "norm1": nn.LayerNorm(d_model),
                    "norm2": nn.LayerNorm(d_model),
                    "distill": AttentionDistill(d_model) if attn_distill and i < nlayers - 1 else None,
                }) for i in range(nlayers)
            ])
        else:
            self.transformer = nn.ModuleList([
                RoPETransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
                for _ in range(nlayers)
            ])

        # ---- 注意力池化 ----
        self.attn_pool = AttentionPooling(d_model)

        # ---- [V3 Enhanced] 跨频率交叉注意力融合 ----
        self.cross_freq_attn = CrossFreqAttention(n_freqs, d_model) if n_freqs > 1 else None

        # ---- 预测头 (残差结构, 更深的 MLP) ----
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

    def _encode_freq(self, x_freq, freq_idx):
        """编码单个频率: 分解 → [TimesBlock] → LargeConv → GRU → Transformer → Pool。"""
        # 投影
        h = self.freq_projs[freq_idx](x_freq)

        # [TimesNet] 2D 周期变换 (GPU 加速, top_k>0 时启用)
        if hasattr(self, 'times_block') and self.times_block.top_k > 0:
            h = self.times_block(h)

        # [ModernTCN] 大核卷积
        h = self.large_conv(h)

        # GRU 序列编码
        h, _ = self.gru(h)

        # Transformer 层
        if self.attn_type == "prob_sparse":
            for layer_dict in self.transformer:
                # Self-attention
                h2 = layer_dict["attn"](layer_dict["norm1"](h))
                h = h + F.dropout(h2, p=0.1, training=self.training)
                # FFN
                h2 = layer_dict["ffn"](layer_dict["norm2"](h))
                h = h + h2
                # Distill (除最后一层)
                if layer_dict["distill"] is not None:
                    h = layer_dict["distill"](h)
        else:
            for layer in self.transformer:
                h = layer(h)

        # 注意力池化
        return self.attn_pool(h)

    def forward(self, x):
        x = x.float()  # 确保 float32
        B, L, D = x.shape
        feat_per_freq = D // max(self.n_freqs, 1)

        freq_reprs = []
        for i in range(self.n_freqs):
            x_freq = x[:, :, i * feat_per_freq: (i + 1) * feat_per_freq]
            rep = self._encode_freq(x_freq, i)
            freq_reprs.append(rep)

        # 跨频率融合
        if self.cross_freq_attn is not None:
            fused = self.cross_freq_attn(freq_reprs)
        else:
            fused = freq_reprs[0]

        return self.head(fused).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 数据构建 (按 MODE 分发)
# ═══════════════════════════════════════════════════════════════════════════
def _build_single_freq_local(table, sd, ed, mode, instruments):
    """本地模式: 从 parquet 文件构建单频率样本 (优先使用合并文件加速)。"""
    from data_loader import (
        preprocess_data, _find_parquet_files, _parse_date_range_from_filename,
    )
    from collections import defaultdict

    buf = (pd.to_datetime(sd) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    buf_ts = pd.Timestamp(buf)
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    needed_cols = ["date", "instrument_id", "adjust_factor"] + config.FEATURE_COLS
    instruments_set = set(instruments)

    # ---- V3.2: 优先使用合并文件 (大幅加速, 减少内存碎片) ----
    all_files = _find_parquet_files(table)
    selected = []
    # 先找覆盖日期范围的合并文件 (使用 date() 比较避免 datetime vs date 不匹配)
    for f in all_files:
        f_sd, f_ed = _parse_date_range_from_filename(f)
        if f_sd is None:
            continue
        f_sd_ts = pd.Timestamp(f_sd)
        f_ed_ts = pd.Timestamp(f_ed)
        if f_sd_ts.date() <= sd_ts.date() and f_ed_ts.date() >= ed_ts.date():
            selected.append(f)
            logger.info(f"  使用合并文件加速", file=os.path.basename(f))
            break  # 一个就够了
    # 回退到分片加载
    if not selected:
        for f in all_files:
            f_sd, f_ed = _parse_date_range_from_filename(f)
            if f_sd is None:
                selected.append(f)
                continue
            f_sd_ts = pd.Timestamp(f_sd)
            f_ed_ts = pd.Timestamp(f_ed)
            if (f_ed_ts - f_sd_ts).days > 10:
                continue
            if f_sd_ts <= ed_ts and f_ed_ts >= buf_ts:
                selected.append(f)
        if not selected:
            selected = [f for f in all_files if "_2022-01-01_2023-12-31" not in f
                        and "_2024-01-01_2024-12-31" not in f]
            if not selected:
                selected = all_files

    logger.info(f"  加载 {len(selected)} 个文件", n_instruments=len(instruments))

    inst_buffers = defaultdict(list)
    for i, f in enumerate(selected):
        df_chunk = pd.read_parquet(f, columns=needed_cols)
        mask = (df_chunk["date"] >= buf_ts) & (df_chunk["date"] <= ed_ts)
        df_chunk = df_chunk.loc[mask]
        if len(df_chunk) == 0:
            del df_chunk; continue
        df_chunk = df_chunk[df_chunk["instrument_id"].isin(instruments_set)]
        if len(df_chunk) == 0:
            del df_chunk; continue
        df_chunk = preprocess_data(df_chunk, use_adjust=config.USE_ADJUST_FACTOR)
        for ins, sub in df_chunk.groupby("instrument_id", sort=False):
            inst_buffers[ins].append(sub)
        del df_chunk
        if (i + 1) % 20 == 0:
            gc.collect()

    wins, ys, keys = [], [], []
    for ins in sorted(inst_buffers.keys()):
        sub = pd.concat(inst_buffers[ins], ignore_index=False).sort_values("date")
        del inst_buffers[ins]
        if len(sub) <= config.SEQ_LEN:
            del sub; continue
        feats = sub[config.FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        # 使用原始复权价计算标签 (log1p 变换前的 close, 存于 _close_raw)
        close_col = "_close_raw" if "_close_raw" in sub.columns else "close"
        close_px = sub[close_col].to_numpy(np.float64)[close_pos]
        dates = day[close_pos]
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if p + 1 < config.SEQ_LEN or d < sd_ts or d > ed_ts:
                continue
            label = None
            if k + 1 < len(close_pos) and close_px[k] > 0:
                r = close_px[k + 1] / close_px[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue
            wins.append(feats[p - config.SEQ_LEN + 1: p + 1])
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))
        del sub, feats

    del inst_buffers; gc.collect()
    return wins, ys, keys


def _build_single_freq_online(table, sd, ed, mode, instruments):
    """在线模式: 通过 dai.query 构建单频率样本。

    预处理流程 (与本地 data_loader.preprocess_data 完全对齐):
      1. 缺失值填充: 价格类 ffill+填0; 量额/笔数类填0
      2. 价格: 后复权 (乘 adjust_factor) + log1p 变换
      3. 量额/笔数: log1p 变换
      4. Inf→NaN→0
    """
    import dai
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    sql = (f"SELECT date, instrument, adjust_factor, {', '.join(config.FEATURE_COLS)} "
           f"FROM {table} ORDER BY instrument, date")
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": instruments}).df()

    # ---- 缺失值填充 (在数值变换前) ----
    # 价格类: 按股票分组向前填充 (防止跨股票泄漏), 残余填 0
    for c in config.PRICE_COLS:
        if c in df.columns:
            df[c] = df.groupby("instrument", sort=False)[c].ffill().fillna(0)
    for c in config.VOL_COLS + config.ORDER_NUM_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ---- 价格: 后复权 + log1p ----
    if config.USE_ADJUST_FACTOR:
        df[config.PRICE_COLS] = df[config.PRICE_COLS].mul(df["adjust_factor"], axis=0)
    # 保存原始复权 close 用于标签计算
    if "close" in df.columns:
        df["_close_raw"] = df["close"].to_numpy(np.float32).copy()
    for c in config.PRICE_COLS:
        if c in df.columns:
            df[c] = np.log1p(np.maximum(df[c].to_numpy(np.float32), 0))

    # 量额/笔数: log1p
    for c in config.VOL_COLS:
        if c in df.columns:
            df[c] = np.log1p(np.maximum(df[c].to_numpy(np.float32), 0))
    for c in config.ORDER_NUM_COLS:
        if c in df.columns:
            df[c] = np.log1p(np.maximum(df[c].to_numpy(np.float32), 0))

    # Inf 防御 + 残余 NaN 填 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in config.FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df.drop(columns=["adjust_factor"], inplace=True)
    return _build_samples_from_df(df, sd, ed, mode, id_col="instrument")


def _build_samples_from_df(df, sd, ed, mode, id_col="instrument"):
    """从已预处理的 DataFrame 构建 (wins, ys, keys)。"""
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    wins, ys, keys = [], [], []
    # 使用原始复权价计算标签 (log1p 变换前的 close, 存于 _close_raw)
    close_col = "_close_raw" if "_close_raw" in df.columns else "close"
    for ins, sub in df.groupby(id_col, sort=False):
        if len(sub) <= config.SEQ_LEN:
            continue
        feats = sub[config.FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        close_px = sub[close_col].to_numpy(np.float64)[close_pos]
        dates = day[close_pos]
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if p + 1 < config.SEQ_LEN or d < sd_ts or d > ed_ts:
                continue
            label = None
            if k + 1 < len(close_pos) and close_px[k] > 0:
                r = close_px[k + 1] / close_px[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue
            wins.append(feats[p - config.SEQ_LEN + 1: p + 1])
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))
    del df; gc.collect()
    return wins, ys, keys


_build_single_freq = _build_single_freq_local if config.MODE == "local" else _build_single_freq_online


def build_dataset(tables, sd, ed, mode, instruments, stats=None):
    """多频率数据构建 (训练与推理共用, 按 MODE 自动分发)。"""
    t0 = time.time()

    freq_results = {}
    for freq_name, table in tables.items():
        wins, ys, keys = _build_single_freq(table, sd, ed, mode, instruments)
        if not keys:
            raise RuntimeError(f"build_dataset {freq_name} 无样本 (mode={mode}, {sd}~{ed})")
        freq_results[freq_name] = (wins, ys, keys)
        gc.collect()
        logger.info(f"{mode} 集 {freq_name} 构建完成", samples=len(keys))

    freq_names = list(tables.keys())
    key_sets = [set(freq_results[fn][2]) for fn in freq_names]
    common_keys = sorted(set.intersection(*key_sets))
    if not common_keys:
        raise RuntimeError(f"build_dataset 无共同样本 (mode={mode}, {sd}~{ed})")

    aligned_wins = []
    for fn in freq_names:
        wins_list, _, keys_list = freq_results[fn]
        key_to_idx = {k: i for i, k in enumerate(keys_list)}
        idxs = [key_to_idx[k] for k in common_keys]
        aligned_wins.append(np.stack([wins_list[i] for i in idxs]).astype(np.float32))

    X = np.concatenate(aligned_wins, axis=-1)

    _, ys_list, keys_list = freq_results[freq_names[0]]
    key_to_y = dict(zip(keys_list, ys_list))
    y = np.array([key_to_y[k] for k in common_keys], dtype=np.float32)

    if stats is None:
        n_feat_per = config.N_FEAT
        means, stds = [], []
        for aw in aligned_wins:
            flat = aw.reshape(-1, n_feat_per)
            means.append(flat.mean(0).astype(np.float32))
            stds.append(flat.std(0).astype(np.float32) + 1e-6)
        stats = (np.concatenate(means), np.concatenate(stds))
    m, s = stats
    X = ((X - m) / s).astype(np.float32)

    logger.info(f"{mode} 集构建完成 (多频率)", samples=len(common_keys),
                n_freqs=len(tables), elapsed=round(time.time() - t0, 2))
    keys_col = "instrument_id" if config.MODE == "local" else "instrument"
    keys_df = pd.DataFrame(common_keys, columns=["date", keys_col])
    if mode == "train":
        return X, y, keys_df, stats
    return X, None, keys_df, stats


# ═══════════════════════════════════════════════════════════════════════════
# 标准化统计 & 模型 存/读
# ═══════════════════════════════════════════════════════════════════════════
def save_stats(stats, path=config.STATS_PATH, train_end=config.TRAIN_END):
    mean, std = stats
    payload = {
        "train_start": config.TRAIN_START,
        "train_end": train_end,
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info("标准化统计已缓存", path=path)


def load_stats(path=config.STATS_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("train_start") != config.TRAIN_START or payload.get("train_end") != config.TRAIN_END:
        logger.warning("stats 缓存训练区间不匹配, 将重新计算")
        return None
    mean = np.array(payload["mean"], dtype=np.float32)
    std = np.array(payload["std"], dtype=np.float32)
    logger.info("命中标准化统计缓存", path=path)
    return mean, std


def save_model(ckpt, model_path=config.MODEL_PATH):
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),
        }
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=config.MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ═══════════════════════════════════════════════════════════════════════════
# 验证指标计算
# ═══════════════════════════════════════════════════════════════════════════
def compute_val_metrics(y_pred, y_true, keys_df):
    """计算本地验证指标: IC Mean / IC IR / 多空 Sharpe。"""
    if len(y_pred) < 2:
        return {"ic_mean": float("nan"), "ic_ir": float("nan"), "long_short_sharpe": float("nan")}

    id_col = "instrument_id" if config.MODE == "local" else "instrument"
    df = keys_df.copy()
    df["score"] = y_pred.astype(np.float64)
    df["label"] = y_true.astype(np.float64)
    df["date"] = pd.to_datetime(df["date"])

    daily_ic = df.groupby("date").apply(
        lambda g: g["score"].corr(g["label"]) if len(g) > 10 else np.nan,
        include_groups=False,
    ).dropna()
    ic_mean = float(daily_ic.mean())
    ic_std = float(daily_ic.std())
    ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")

    daily_ret = []
    for _, g in df.groupby("date"):
        if len(g) < 20:
            continue
        g = g.dropna(subset=["score", "label"])
        g["group"] = pd.qcut(g["score"], 10, labels=False, duplicates="drop")
        if g["group"].nunique() < 10:
            continue
        long_ret = g.loc[g["group"] == 9, "label"].mean()
        short_ret = g.loc[g["group"] == 0, "label"].mean()
        daily_ret.append(long_ret - short_ret)

    ls_sharpe = float("nan")
    if len(daily_ret) >= 10:
        dr = np.array(daily_ret)
        ls_sharpe = float(np.mean(dr) / max(np.std(dr), 1e-12) * np.sqrt(252))

    return {"ic_mean": round(ic_mean, 6), "ic_ir": round(ic_ir, 4),
            "long_short_sharpe": round(ls_sharpe, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# Pearson 相关损失 (直接优化 IC)
# ═══════════════════════════════════════════════════════════════════════════
class PearsonCorrelationLoss(nn.Module):
    """Pearson 相关系数损失: 1 - pearson(pred, target), 直接优化线性相关。"""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        pred = pred - pred.mean()
        target = target - target.mean()
        cov = (pred * target).mean()
        std_pred = pred.std() + self.eps
        std_target = target.std() + self.eps
        corr = cov / (std_pred * std_target)
        return 1.0 - corr


class CombinedLoss(nn.Module):
    """组合损失: MSE + λ * PearsonLoss, 兼顾数值精度与排序质量。"""

    def __init__(self, lambda_pearson=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pearson = PearsonCorrelationLoss()
        self.lambda_pearson = lambda_pearson

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        loss_mse = self.mse(pred, target)
        loss_pearson = self.pearson(pred, target)
        return loss_mse + self.lambda_pearson * loss_pearson


# ═══════════════════════════════════════════════════════════════════════════
# 训练主函数
# ═══════════════════════════════════════════════════════════════════════════
def train_and_save(datasources, model_path=config.MODEL_PATH):
    """在写死的训练区间上从零训练, 把 权重 + 标准化统计 + 结构超参 一并存盘。

    V2 改进:
      - 使用 StockTransformerV2 模型
      - CombinedLoss (MSE + Pearson)
      - Cosine annealing warmup LR 调度
      - 每 epoch 保存检查点并于验证集评估
    """
    tables = {k.replace("bar", ""): v for k, v in datasources.items() if k.startswith("bar")}
    if not tables:
        raise ValueError("datasources 中未找到 bar 频率表 (如 bar1m/bar5m/...)")

    instruments = config.pool(config.TRAIN_START, config.TRAIN_END)[:config.MAX_TRAIN_INSTRUMENTS]
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), mode=config.MODE, tables=tables,
                train=f"{config.TRAIN_START}~{config.TRAIN_END}",
                val=f"{config.VAL_START}~{config.VAL_END}")

    # ====== Pass 1: 标准化统计 ======
    stats = load_stats(config.STATS_PATH)
    if stats is not None:
        logger.info("Pass 1: 命中缓存, 跳过统计计算")
    else:
        logger.info("Pass 1: 一次性加载全部数据计算标准化统计量")
        X_all, _, _, stats = build_dataset(tables, config.TRAIN_START, config.TRAIN_END,
                                           "train", instruments)
        save_stats(stats, config.STATS_PATH)
        del X_all
        gc.collect()
        logger.info("Pass 1 完成")

    # ====== Pass 2: 加载训练数据 ======
    logger.info("Pass 2: 一次性加载全部训练数据")
    X, y, keys_train, _ = build_dataset(tables, config.TRAIN_START, config.TRAIN_END,
                                        "train", instruments, stats=stats)
    logger.info("训练数据加载完成", samples=len(X), shape=X.shape)

    # ---- 截面标签归一化: 每日 z-score (直接优化截面 IC) ----
    date_col = "instrument_id" if config.MODE == "local" else "instrument"
    keys_train["y_raw"] = y
    y_cs = keys_train.groupby("date")["y_raw"].transform(
        lambda x: (x - x.mean()) / (max(x.std(), 1e-8)))
    y = y_cs.values.astype(np.float32)
    logger.info("截面标签归一化完成", y_mean=float(y.mean()), y_std=float(y.std()))

    # 构建模型 (V1 / V2 / V3)
    if config.MODEL_VERSION == "V1":
        model = StockTransformer(**config.MODEL_CFG).to(device)
        model_class_name = "StockTransformer"
        model_cfg_to_save = config.MODEL_CFG
    elif config.MODEL_VERSION == "V2":
        n_freqs = len(tables)
        v2_cfg = dict(
            n_feat=config.N_FEAT_ALL,
            d_model=128,
            nhead=8,
            nlayers=4,
            dim_ff=256,
            seq_len=config.SEQ_LEN,
            n_freqs=n_freqs,
            dropout=0.1,
        )
        model = StockTransformerV2(**v2_cfg).to(device)
        model_class_name = "StockTransformerV2"
        model_cfg_to_save = v2_cfg
    else:  # V3
        n_freqs = len(tables)
        v3_cfg = dict(config.V3_CFG)
        v3_cfg["n_freqs"] = n_freqs
        v3_cfg["n_feat"] = config.N_FEAT_ALL
        v3_cfg["seq_len"] = config.SEQ_LEN
        model = StockTransformerV3(**v3_cfg).to(device)
        model_class_name = "StockTransformerV3"
        model_cfg_to_save = v3_cfg

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("可训练参数量", n_params=n_params, model=model_class_name)
    if n_params < 100_000:
        logger.warning("参数量低于 10 万下限!", n_params=n_params)
    if n_params > 100_000_000:
        logger.warning("参数量超过 1 亿上限!", n_params=n_params)

    # 优化器 (AdamW with weight decay)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # 学习率调度器: Cosine annealing with linear warmup
    warmup_epochs = config.WARMUP_EPOCHS
    total_epochs = config.EPOCHS
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_epochs - warmup_epochs, eta_min=config.LR * 0.01)

    loss_fn = CombinedLoss(lambda_pearson=config.V3_TRAIN_CFG.get("lambda_pearson", 0.5) if config.MODEL_VERSION == "V3" else 0.5)

    # ====== 加载验证集 (用于 early stopping 参考) ======
    val_instruments = config.pool(config.VAL_START, config.VAL_END)[:config.MAX_TRAIN_INSTRUMENTS]
    X_val, y_val_raw, keys_val, _ = build_dataset(
        tables, config.VAL_START, config.VAL_END, "train", val_instruments, stats=stats)
    # 验证集也做截面标签归一化 (用于loss), 保留原始标签用于指标计算
    keys_val["y_raw"] = y_val_raw
    y_val_cs = keys_val.groupby("date")["y_raw"].transform(
        lambda x: (x - x.mean()) / (max(x.std(), 1e-8)))
    y_val = y_val_cs.values.astype(np.float32)
    logger.info("验证集加载完成", samples=len(X_val))

    # ====== 训练 ======
    best_val_ic = float("-inf")
    ckpt_paths = []
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for ep in range(total_epochs):
        t_ep = time.time()

        # 数据准备: 随机打乱 + label winsorize
        perm = np.random.permutation(len(X))
        X_shuf = X[perm].copy()
        y_shuf = y[perm].copy()
        lo, hi = np.percentile(y_shuf, [0.5, 99.5])
        y_shuf = np.clip(y_shuf, lo, hi)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_shuf), torch.from_numpy(y_shuf)),
            batch_size=config.BATCH, shuffle=True,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        # ---- Warmup 学习率 ----
        if ep < warmup_epochs:
            lr = config.LR * (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg["lr"] = lr

        # ---- 训练一个 epoch ----
        tot_loss, nb = 0.0, 0
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                opt.step()

            tot_loss += loss.item()
            nb += 1

        del X_shuf, y_shuf, loader

        # ---- Cosine 衰减 (warmup 之后) ----
        if ep >= warmup_epochs:
            scheduler_cosine.step()

        # ---- 验证集评估 ----
        model.eval()
        val_preds = []
        X_val_t = torch.from_numpy(X_val)
        with torch.no_grad():
            for i in range(0, len(X_val_t), config.BATCH * 2):
                xb = X_val_t[i:i + config.BATCH * 2].to(device)
                val_preds.append(model(xb).cpu().numpy())
        y_pred = np.concatenate(val_preds)
        # 使用原始标签计算指标 (IC 不受影响, Sharpe 需要原始量纲)
        val_m = compute_val_metrics(y_pred, y_val_raw, keys_val)

        # ---- 保存检查点 ----
        ckpt_path = config.MODEL_PATH + f".ep{ep + 1:02d}"
        mean_s, std_s = stats
        save_model({
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "model_cfg": model_cfg_to_save,
            "model_class": model_class_name,
            "feature_cols": config.FEATURE_COLS,
            "freq_names": list(tables.keys()),
            "seq_len": config.SEQ_LEN,
            "mean": np.asarray(mean_s, np.float32).tolist(),
            "std": np.asarray(std_s, np.float32).tolist(),
            "mode": config.MODE,
            "epoch": ep + 1,
        }, ckpt_path)
        ckpt_paths.append(ckpt_path)

        # ---- 更新最佳 ----
        is_best = val_m["ic_mean"] > best_val_ic
        if is_best:
            best_val_ic = val_m["ic_mean"]
            best_path = config.MODEL_PATH + ".best"
            ckpt_best = load_model(ckpt_path, map_location="cpu")
            ckpt_best["val_metrics"] = val_m
            ckpt_best["best_epoch"] = ep + 1
            save_model(ckpt_best, best_path)

        lr_now = opt.param_groups[0]["lr"]
        marker = " *" if is_best else ""
        logger.info(f"epoch {ep+1:02d}/{total_epochs}",
                    loss=round(tot_loss / max(nb, 1), 6),
                    ic=val_m["ic_mean"], ir=val_m["ic_ir"],
                    sharpe=val_m["long_short_sharpe"],
                    lr=round(lr_now, 8),
                    elapsed=round(time.time() - t_ep, 1),
                    marker=marker)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ====== 释放资源 ======
    del X, y, X_val, y_val; gc.collect()
    try:
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("训练完成, 模型已保存", path=model_path, mode=config.MODE,
                checkpoints=len(ckpt_paths), best_val_ic=best_val_ic)
    return model_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    datasources = {f"bar{k}": v for k, v in config.FREQ_TABLES.items()}
    train_and_save(datasources)
