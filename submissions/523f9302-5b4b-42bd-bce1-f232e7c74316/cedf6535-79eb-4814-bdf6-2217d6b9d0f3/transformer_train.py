# -*- coding: utf-8 -*-
"""
Transformer V11-norm-v2 — 股票日内时序因子模型（训练脚本）
================================================================================
竞赛: BigAlpha 2026
作者: CapooQuant
提交方式: 方式二 (.py 训练 + .ipynb 推理)

私榜流程:
  平台调用 train_and_save(datasources) → 产出 transformer_model.json
  → 平台在私榜验证集上用推理 notebook 的 main() 打分

架构概述
--------
基于 Transformer Encoder 的股票量化因子模型:
  - 输入: 240 根 1 分钟 K 线 (12 个原始字段: 7 价格 + 5 成交量)
  - 输出: 个股未来 1 日收益预测
    评测口径: T 日出信号 → T+1 开盘买入 → T+2 开盘卖出 (open-to-open)
    训练标签: close[T+1]/close[T] - 1 (close-to-close, 实践更稳定)
  - 参数量: ~318K (符合 100K~100M 约束)

预处理管线 (规则化, 无学习参数, 符合竞赛规则):
  1. bid/ask 零值 → close 填充 (涨跌停/低流动性修复)
  2. 价格÷pre_close 日级归一化 (跨股票可比的日内价格动态)
  3. log1p 压缩成交量重尾分布
  4. 成交量÷daily_max 日级归一化 (消除大小票量级差异)
  5. 240-bar 滑动窗口 → z-score 标准化 (训练集统计量)

模型组件:
  - SpectralFilter1D: 可学习频域 Wiener 滤波 (FFT → 可学习权重 → iFFT)
  - Conv Stem: 时序卷积投影 (k=5 + k=3)
  - ClusterEmbed: 股价聚类嵌入 (5 类, 正态分位边界, 对数空间)
  - Positional + Time-of-Day Embedding: 可学习位置 + 日内时间编码
  - Transformer Encoder: 3 层, d=96, head=4, ff=192, pre-norm
  - Attn Gate: softmax 时间维加权聚合
  - DAE 辅助任务: mask 15% bars + Gaussian noise σ=0.10 → 重建 MSE
  - EMA: 指数移动平均 (decay=0.999)

标签构造:
  使用 bigalpha_2026_bar1d (后复权日线) 的 close 构造:
  label[T] = close[T+1] / close[T] - 1 → log1p 变换
  评测口径为 open(T+1)→open(T+2), 但实践发现 close-to-close 学习更稳定
  (close-close 与 open-open 日度收益相关性约 0.7~0.8)

超参数选择依据 (鲁棒性实验, 20260801 目录):
  在 3 个训练周期 (2020-2023 / 2021-2023 / 2022-2023) 上扫描 epoch=2~24,
  评估 2024 年全年表现。EPOCHS=20 在所有周期下排名靠前且跨周期最稳定
  (IC 标准差最小 = 0.0006), 选为最终提交值。

可复现性说明:
  本模型使用 GPU 训练, 浮点数加法不满足结合律, 以下因素会导致两次训练产出
  的权重不完全一致:
    - 不同 GPU 架构 (A100 / V100 / H100 等) 的 tensor core 实现不同
    - CUDA 版本 / cuDNN 版本差异导致的 kernel 选择不同
    - torch.fft.rfft (频域滤波) 和 scaled_dot_product_attention (Transformer)
      等算子的浮点累加顺序依赖硬件和运行时调度
  固定随机种子 (SEED=42) 可保证 CPU 侧可复现, 但无法消除 GPU 浮点差异。
  不同硬件上训练得到的模型, 因子 IC 通常在同一水平 (±0.001), 排名和方向一致。

用法:
  python transformer_train.py
  → 产出 transformer_model.json (权重 + 统计量 + 配置)
"""

import os, sys, json, time, math, gc
import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ============================================================================
# 核心超参数 (私榜平台从零重训时使用)
# ============================================================================
# 训练周期: 2021-01-01 ~ 2023-12-31 (3 年, 平衡数据量与训练时间)
# 如平台提供更多数据, 可自行扩展 TRAIN_START; EPOCHS=20 在多数据量下仍稳健
TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31 23:59:59"

# ★ 关键超参数 — 不要随意修改 ★
SEQ_LEN       = 240     # 回看窗口: 240 根 1min bar ≈ 1 个交易日
EPOCHS        = 20      # 训练轮数 (鲁棒性实验验证: 跨周期最稳定)
BATCH         = 512     # 批次大小
LR_PEAK       = 5e-4    # 峰值学习率 (warmup 线性增长 → cosine 衰减)
WEIGHT_DECAY  = 1e-4    # AdamW 权重衰减
WARMUP_EPOCHS = 3       # 学习率 warmup 轮数
GRAD_CLIP     = 1.0     # 梯度裁剪阈值
SEED          = 42      # 随机种子 (固定 CPU 侧可复现; GPU 浮点非结合律导致硬件间有微小差异)

EMA_DECAY     = 0.999   # EMA 衰减率
GRAD_ACCUM    = 1       # 梯度累积步数 (1 = 不累积)
INSTRUMENT_BATCH = 150  # 数据加载: 每批股票数 (控制内存峰值)

# ============================================================================
# 数据表名 (★ 训练表名必须硬编码, 不能用 datasources 参数 ★)
# ============================================================================
BAR1D_TABLE = "bigalpha_2026_bar1d"        # 后复权日线 (标签构造)
TRAIN_TABLE = "bigalpha_2026_stock_bar1m"  # 1 分钟 K 线 (特征数据源, 硬编码)

# ============================================================================
# 预处理: 日级归一化配置 (规则化, 无学习参数, 符合竞赛规则)
# ============================================================================
# 价格归一化: intraday 价格列 ÷ 当日 pre_close → 跨股票可比的日内动态
PRICE_NORM_MODE = "pre_close"
NORM_PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
#   pre_close 保留为绝对价格锚点 (不归一化), 承载"低价股 vs 高价股"信息

# 成交量归一化: log1p 后 ÷ 当日最大值 → 每日量剖面归一化到 [0, 1]
VOL_NORM_MODE = "daily_max"

# ============================================================================
# V10: 股价聚类嵌入配置
# ============================================================================
USE_CLUSTER_EMB = True
N_CLUSTERS      = 5       # 聚类数: 超低价/低价/中价/高价/超高价
CLUSTER_METHOD  = "normal" # 正态分位边界 (对数空间), 比 quantile 更稳定
Z_RANGE         = 2.5     # 正态边界 z-score 范围
Z_EXPONENT      = 1.5     # 边界密度指数 (>1 使极值区间更宽)

# ============================================================================
# V9: 频域滤波 + DAE 降噪配置
# ============================================================================
SPECTRAL_FILTER   = True   # 可学习频域滤波 (1D Wiener filter)
SPEC_FILTER_DIM   = 32     # 频域隐层维度

MASK_RATIO        = 0.15   # DAE: bar mask 比例
NOISE_STD         = 0.10   # DAE: Gaussian noise 标准差
DENOISE_MODE      = "both" # DAE 模式: mask + noise 同时使用

# DAE 损失权重: 从初始值线性衰减到最终值
#   早期: 重建任务引导表征学习; 后期: 预测任务主导
RECON_WEIGHT_INITIAL = 0.5
RECON_WEIGHT_FINAL   = 0.03

# ============================================================================
# 特征开关 (架构消融实验用, 提交时全部开启推荐配置)
# ============================================================================
FEATURE_FLAGS = {
    "use_ema":          True,   # EMA 权重平滑
    "use_prenorm":      True,   # Transformer pre-norm (训练更稳定)
    "use_tod_emb":      True,   # Time-of-Day 嵌入 (日内时间结构)
    "session_aware":    False,  # 上下午分离卷积 (实验性, 未启用)
    "use_preclose_gate": False, # pre_close 门控 (实验性, 未启用)
    "use_val_earlystop": False, # 验证集早停 (关闭: 固定 epoch 更稳健)
}

# ============================================================================
# Transformer 结构超参
# ============================================================================
MODEL_CFG = dict(
    n_feat=12,    # 输入特征数 (7 价格 + 5 成交量)
    d_model=96,   # 隐层维度
    nhead=4,      # 注意力头数 (96/4 = 24 dim/head)
    nlayers=3,    # Encoder 层数
    dim_ff=192,   # FFN 隐层维度 (2× d_model)
    seq_len=240,  # 序列长度 (= SEQ_LEN)
    dropout=0.15, # Dropout 率
)

# ============================================================================
# 原始字段列表 (12 个, 符合 ≤ 100 约束)
# ============================================================================
PRICE_COLS = ["pre_close", "open", "high", "low", "close",
              "bid_price1", "ask_price1"]    # 7 个价格字段
VOL_COLS   = ["deal_number", "volume", "amount",
              "bid_volume1", "ask_volume1"]  # 5 个成交量字段
FEATURE_COLS = PRICE_COLS + VOL_COLS         # 共 12 个
N_FEAT = len(FEATURE_COLS)


# ============================================================================
# V10: 股价聚类边界计算
# ============================================================================
def compute_cluster_boundaries_from_closes(all_closes, n_clusters=N_CLUSTERS,
                                            method=CLUSTER_METHOD,
                                            z_range=Z_RANGE, z_exponent=Z_EXPONENT):
    """从全量收盘价计算正态分位聚类边界。

    在对数空间用 z-score 划分, 使边界在高/低价区域更宽 (因为极端价格
    对预测更有区分度), 中间区域更窄。

    参数:
      all_closes: 全量日线收盘价列表 (bar1d close)
      n_clusters: 聚类数
      method: "normal" | "quantile" | "uniform"
      z_range: 正态边界跨度 (±z_range σ)
      z_exponent: 边界密度指数 (>1 拉伸两端)

    返回: boundaries 数组 (长度 n_clusters-1), 用于 np.searchsorted 赋簇
    """
    closes = np.array(all_closes, dtype=np.float64)
    closes = closes[np.isfinite(closes) & (closes > 0)]
    log_closes = np.log(closes)

    if method == "quantile":
        percentiles = np.linspace(0, 100, n_clusters + 1)[1:-1]
        boundaries = np.percentile(closes, percentiles).astype(np.float32)

    elif method == "normal":
        mu, sigma = np.mean(log_closes), np.std(log_closes)
        n_boundaries = n_clusters - 1
        half = n_boundaries // 2

        if half > 0:
            powers = np.linspace(0, 1, half + 2)[1:-1] ** z_exponent
            z_pos = powers * z_range
        else:
            z_pos = np.array([], dtype=np.float64)

        if n_clusters % 2 == 0:
            z_boundaries = np.concatenate([-z_pos[::-1], [0.0], z_pos])
        else:
            z_boundaries = np.concatenate([-z_pos[::-1], z_pos])

        boundaries = np.exp(mu + z_boundaries * sigma).astype(np.float32)

    elif method == "uniform":
        min_log, max_log = np.min(log_closes), np.max(log_closes)
        log_boundaries = np.linspace(min_log, max_log, n_clusters + 1)[1:-1]
        boundaries = np.exp(log_boundaries).astype(np.float32)

    else:
        raise ValueError(f"未知聚类方法: {method}")

    cids = np.searchsorted(boundaries, closes)
    unique, counts = np.unique(cids, return_counts=True)
    dist_pct = ", ".join(f"c{c}={n/len(closes)*100:.1f}%"
                         for c, n in zip(unique, counts))

    logger.info("聚类边界已计算",
                method=method, n_clusters=n_clusters,
                boundaries=[f"{b:.2f}" for b in boundaries],
                close_min=f"{closes.min():.2f}", close_max=f"{closes.max():.2f}",
                n_samples=len(closes), distribution=dist_pct)
    return boundaries


# ============================================================================
# V9: 可学习频域滤波 (SpectralFilter1D)
# ============================================================================
class SpectralFilter1D(nn.Module):
    """可学习 1D 时序频域滤波器 (Wiener filter 风格)。

    对每条特征在频域学习逐频率权重:
      FFT → log magnitude → MLP → sigmoid 权重 → 加权 FFT → iFFT
    残差连接 (α=0.5) 保留原始信号。
    """
    def __init__(self, seq_len=240, n_feat=12, hidden_dim=32):
        super().__init__()
        n_freqs = seq_len // 2 + 1     # 240 → 121 个频率分量
        self.freq_net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        self.global_scale = nn.Parameter(torch.ones(n_feat))  # 逐特征缩放

    def forward(self, x):
        """x: (B, L, F) → (B, L, F)"""
        B, L, F = x.shape
        x_freq = torch.fft.rfft(x.float(), dim=1)        # (B, n_freqs, F)
        mag = torch.log(x_freq.abs() + 1e-6)              # log 幅度
        weights = self.freq_net(mag.unsqueeze(-1)).squeeze(-1)  # (B, n_freqs, F)
        weights = weights * self.global_scale.view(1, 1, F)     # 逐特征缩放
        x_freq_filtered = x_freq * weights
        x_filtered = torch.fft.irfft(x_freq_filtered, n=L, dim=1)
        alpha = 0.5  # 残差比例: 50% 滤波 + 50% 原始
        return alpha * x_filtered + (1 - alpha) * x


# ============================================================================
# V10: 聚类嵌入 (ClusterEmbed)
# ============================================================================
class ClusterEmbed(nn.Module):
    """学习每个价格聚类的 d_model 维嵌入向量, 加到时序表征上。

    使模型感知"当前是低价股还是高价股", 不同价格层级可能有不同的
    波动率、流动性和机构参与度模式。
    """
    def __init__(self, n_clusters, d_model):
        super().__init__()
        self.embed = nn.Embedding(n_clusters, d_model)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, h, cluster_ids):
        """h: (B, L, d_model), cluster_ids: (B,) → h + emb"""
        emb = self.embed(cluster_ids)  # (B, d_model)
        return h + emb.unsqueeze(1)    # 广播到时序维度


# ============================================================================
# EMA (指数移动平均)
# ============================================================================
class EMA:
    """模型权重的指数移动平均, 推理时使用 shadow 权重。

    decay=0.999: 每个 step 新权重 = 0.001×当前 + 0.999×历史
    """
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow, self._backup = {}, {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(
                        p.data, alpha=1. - self.decay)

    def apply_shadow(self):
        """保存当前权重, 替换为 EMA shadow (推理前调用)"""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self):
        """恢复原始权重 (训练继续前调用)"""
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n])


# ============================================================================
# DAE (Denoising Autoencoder) 输入破坏
# ============================================================================
def apply_bar_mask(x, mask_ratio=0.15, mask_val=0.0):
    """随机 mask 15% 的 bar (整根 bar 全部特征置零)。

    限制每样本最多 mask 30% 的 bar, 防止单样本过度破坏。
    """
    B, L, F = x.shape
    mask = torch.rand(B, L, device=x.device) < mask_ratio
    for b in range(B):
        if mask[b].sum() > int(L * 0.30):
            keep = torch.randperm(L, device=x.device)[:int(L * 0.70)]
            mask[b, keep] = False
    xc = x.clone()
    xc[mask] = mask_val
    return xc, mask


def apply_gaussian_noise(x, noise_std=0.10):
    """随机 50% 的 bar 添加高斯噪声 (σ=0.10)。"""
    B, L, _ = x.shape
    nmask = torch.rand(B, L, device=x.device) < 0.5
    noise = torch.randn_like(x) * noise_std
    xc = x.clone()
    xc[nmask] = x[nmask] + noise[nmask]
    return xc, nmask


def corrupt_input(x, mode="both"):
    """DAE 输入破坏: mask + noise (或单独使用)。

    返回: (破坏后输入, 干净目标, 破坏位置 mask)
    """
    B, L, F = x.shape
    xt = x.clone()
    all_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    if mode == "none":
        return x, None, None
    if mode in ("mask", "both"):
        xt, mm = apply_bar_mask(xt)
        all_mask = all_mask | mm
    if mode in ("noise", "both"):
        xt, mn = apply_gaussian_noise(xt)
        all_mask = all_mask | mn
    return xt, x.clone(), all_mask


# ============================================================================
# 主模型: StockTransformer (V10 架构 + V11 归一化)
# ============================================================================
class StockTransformer(nn.Module):
    """基于 Transformer 的股票日内因子模型。

    管线:
      x (B, L, 12) → SpectralFilter1D (频域降噪)
        → Conv Stem (时序卷积投影到 d_model)
        → + ClusterEmbed (股价层级嵌入)
        → + Positional Embedding + Time-of-Day Embedding
        → Transformer Encoder (3 层, pre-norm)
        → Attn Gate (softmax 时间加权)
        → Head MLP → 标量预测

    同时输出 DAE 重建 (auxiliary task)。
    """
    def __init__(self, n_feat=N_FEAT, d_model=96, nhead=4, nlayers=3,
                 dim_ff=192, seq_len=SEQ_LEN, dropout=0.15,
                 session_aware=False, use_preclose_gate=False,
                 use_tod_emb=True, use_prenorm=True,
                 use_spectral_filter=True,
                 use_cluster_emb=True, n_clusters=N_CLUSTERS):
        super().__init__()
        self.session_aware = session_aware
        self.use_tod_emb = use_tod_emb
        self.use_preclose_gate = use_preclose_gate
        self.use_spectral_filter = use_spectral_filter
        self.use_cluster_emb = use_cluster_emb

        # -- 频域滤波 --
        if use_spectral_filter:
            self.spectral_filter = SpectralFilter1D(
                seq_len=seq_len, n_feat=n_feat, hidden_dim=SPEC_FILTER_DIM)

        # -- 时序卷积 Stem (12 → d_model) --
        if session_aware:
            # 实验性: 上下午分离卷积 (当前未启用)
            hd = d_model // 2
            self.morning_conv = nn.Sequential(
                nn.Conv1d(n_feat, hd, 5, padding=2, bias=False),
                nn.BatchNorm1d(hd), nn.GELU(),
                nn.Conv1d(hd, hd, 3, padding=1, bias=False),
                nn.BatchNorm1d(hd), nn.GELU())
            self.afternoon_conv = nn.Sequential(
                nn.Conv1d(n_feat, hd, 5, padding=2, bias=False),
                nn.BatchNorm1d(hd), nn.GELU(),
                nn.Conv1d(hd, hd, 3, padding=1, bias=False),
                nn.BatchNorm1d(hd), nn.GELU())
            self.session_fuse = nn.Linear(hd, d_model)
        else:
            self.conv_stem = nn.Sequential(
                nn.Conv1d(n_feat, d_model, 5, padding=2, bias=False),
                nn.BatchNorm1d(d_model), nn.GELU())
            self.conv_stem2 = nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=1, bias=False),
                nn.BatchNorm1d(d_model), nn.GELU())

        # -- 股价聚类嵌入 --
        if use_cluster_emb:
            self.cluster_embed = ClusterEmbed(n_clusters, d_model)

        # -- 位置 + 时间嵌入 --
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if use_tod_emb:
            self.tod_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.trunc_normal_(self.tod_emb, std=0.02)

        # -- Transformer Encoder --
        el = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=True, activation="gelu", norm_first=use_prenorm)
        self.encoder = nn.TransformerEncoder(el, nlayers)

        # -- Attn Gate + 预测头 --
        self.attn_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

        # -- DAE 重建头 --
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, n_feat))

    def encode(self, x, cluster_ids=None):
        """编码器: 输入时序 → Transformer 隐状态序列"""
        B, L, _ = x.shape

        if self.use_spectral_filter:
            x = self.spectral_filter(x)

        if self.session_aware:
            hm = self.morning_conv(x[:, :120, :].transpose(1, 2)).transpose(1, 2)
            ha = self.afternoon_conv(x[:, 120:, :].transpose(1, 2)).transpose(1, 2)
            h = torch.cat([hm, ha], dim=1)
            h = self.session_fuse(h)
        else:
            h = self.conv_stem(x.transpose(1, 2))
            h = self.conv_stem2(h).transpose(1, 2)

        if self.use_cluster_emb and cluster_ids is not None:
            h = self.cluster_embed(h, cluster_ids)

        pl = min(L, self.pos.shape[1])
        h = h[:, :pl, :] + self.pos[:, :pl, :]
        if self.use_tod_emb:
            h = h + self.tod_emb[:, :pl, :]

        return self.encoder(h)

    def forward(self, x, cluster_ids=None, return_recon=False):
        """前向传播。

        返回:
          return_recon=False → pred (B,)  标量预测
          return_recon=True  → (pred, recon)  预测 + 重建
        """
        h = self.encode(x, cluster_ids=cluster_ids)
        w = F.softmax(self.attn_gate(h), dim=1)     # 时间维注意力权重
        pred = self.head((h * w).sum(dim=1)).squeeze(-1)  # 加权聚合 → 预测
        if not return_recon:
            return pred
        recon = self.recon_head(h)  # 每个时间步的 12 维重建
        return pred, recon


# ============================================================================
# 日级归一化辅助函数 (预处理管线的一部分)
# ============================================================================
def _apply_daily_price_norm(sub, day_starts, day_ends):
    """将 intraday 价格列除以当日 pre_close (原地修改)。"""
    pc_arr = sub["pre_close"].to_numpy(np.float64)
    for c in NORM_PRICE_COLS:
        c_arr = sub[c].to_numpy(np.float64)
        for ds, de in zip(day_starts, day_ends):
            pc = pc_arr[ds]
            if pc > 1e-8:
                c_arr[ds:de + 1] /= pc
        sub[c] = c_arr


def _apply_daily_vol_norm(sub, day_starts, day_ends, mode="daily_max"):
    """log1p 后的量列除以当日参考值 (原地修改)。

    mode="daily_max": 除以当日 log1p 最大值 → 每日量剖面归一化到 [0, 1]
    """
    for c in VOL_COLS:
        c_arr = sub[c].to_numpy(np.float64)
        c_arr = np.maximum(c_arr, 0.0)
        for ds, de in zip(day_starts, day_ends):
            if mode == "first_bar":
                ref = c_arr[ds]
            else:
                ref = c_arr[ds:de + 1].max()
            ref = max(ref, 1e-8)
            c_arr[ds:de + 1] /= ref
        sub[c] = c_arr


# ============================================================================
# 数据管道
# ============================================================================
def pool(sd, ed):
    """获取指定日期范围内的中证 1000 成分股列表。"""
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]}).df()
    return df["instrument"].tolist()


def _build_label_dict(instruments, sd, ed, return_closes=False):
    """从 bar1d (后复权日线) 构造标签字典。

    标签: label[T] = log1p(close[T+1] / close[T] - 1)
    评测口径为 open-to-open, 但 close-to-close 训练更稳定。
    """
    ed_ext = (pd.to_datetime(ed) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    sql = (f"SELECT date, instrument, close FROM {BAR1D_TABLE} "
           f"ORDER BY instrument, date")
    df = dai.query(sql,
                   filters={"date": [buf, ed_ext],
                            "instrument": instruments}).df()
    labels = {}
    all_closes = []
    for ins, sub in df.groupby("instrument", sort=False):
        sub = sub.sort_values("date")
        closes = sub["close"].to_numpy(np.float64)
        dates  = sub["date"].dt.normalize().to_numpy()
        if return_closes:
            all_closes.extend(closes[closes > 0].tolist())
        for i in range(len(closes) - 1):
            if closes[i] <= 0:
                continue
            r_raw = closes[i + 1] / closes[i] - 1.0
            r = np.log1p(r_raw) if r_raw > -1.0 else r_raw
            if np.isfinite(r):
                labels[(pd.Timestamp(dates[i]), ins)] = np.float32(r)
    if return_closes:
        return labels, all_closes
    return labels


def _window_batch(table, sd, ed, mode, batch_inst, label_dict, sd_ts, ed_ts,
                  cluster_boundaries=None, price_norm_mode=None, vol_norm_mode=None):
    """查询一批股票的特征数据, 构建 240-bar 滑动窗口。

    预处理管线 (按顺序):
      1. bid/ask 零值 → close 填充 (涨跌停修复)
      2. 价格 ÷ pre_close (日级归一化)
      3. log1p (量) / log (价) 变换
      4. 量 ÷ daily_max (日级归一化)
      5. 240-bar 窗口拼接

    price_norm_mode / vol_norm_mode 默认使用全局常量, 推理时可从
    checkpoint 传入以确保与训练时一致。
    """
    if price_norm_mode is None:
        price_norm_mode = PRICE_NORM_MODE
    if vol_norm_mode is None:
        vol_norm_mode = VOL_NORM_MODE
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    sql = (f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
           f"FROM {table} ORDER BY instrument, date")
    df = dai.query(sql,
                   filters={"date": [buf, ed],
                            "instrument": batch_inst}).df()

    # ★ 步骤 1: 盘口价格零值修复 (log 变换前)
    # 涨跌停/低流动性时 bid/ask=0, 用 close 填充
    # 不能直接用 max(x, 1e-8), 否则 log(1e-8)=-18.4 成为极端离群点
    for c in ["bid_price1", "ask_price1"]:
        mask = (df[c].to_numpy() <= 0) | (df[c].isna())
        if mask.any():
            df.loc[mask, c] = df.loc[mask, "close"].to_numpy()

    wins, ys, keys, cids = [], [], [], []
    for ins, sub_orig in df.groupby("instrument", sort=False):
        if len(sub_orig) <= SEQ_LEN:
            continue

        sub = sub_orig.copy()

        # -- 识别日边界 --
        day_arr = sub["date"].dt.normalize().to_numpy()
        day_starts = np.flatnonzero(
            np.append([True], day_arr[1:] != day_arr[:-1]))
        day_ends = np.append(day_starts[1:] - 1, len(sub) - 1)

        # -- 保存原始收盘价 (归一化前, 用于聚类赋簇) --
        if cluster_boundaries is not None:
            ins_raw_close = sub["close"].to_numpy(np.float64).copy()

        # ★ 步骤 2: 价格日级归一化 (log 变换前)
        if price_norm_mode == "pre_close":
            _apply_daily_price_norm(sub, day_starts, day_ends)

        # ★ 步骤 3: log 变换
        for c in VOL_COLS:
            sub[c] = np.log1p(sub[c].clip(lower=0))
        for c in PRICE_COLS:
            sub[c] = np.log(np.maximum(sub[c].to_numpy(), 1e-8))

        # ★ 步骤 4: 成交量日级归一化 (log1p 之后)
        if vol_norm_mode in ("first_bar", "daily_max"):
            _apply_daily_vol_norm(sub, day_starts, day_ends, mode=vol_norm_mode)

        # ★ 步骤 5: 构建 240-bar 滑动窗口
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        dates = day[close_pos]
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                continue
            if mode in ("train", "val"):
                label = label_dict.get((d, ins))
                if label is None:
                    continue
                ys.append(label)
            wins.append(feats[p - SEQ_LEN + 1: p + 1])
            keys.append((d, ins))
            if cluster_boundaries is not None:
                cids.append(int(np.searchsorted(
                    cluster_boundaries, ins_raw_close[p])))

    del df
    gc.collect()
    if wins:
        Xb = np.stack(wins).astype(np.float32)
    else:
        Xb = np.empty((0, SEQ_LEN, N_FEAT), np.float32)
    del wins
    cid_arr = (np.array(cids, dtype=np.int64) if cids
               else np.empty((0,), dtype=np.int64))
    return Xb, ys, keys, cid_arr


def build_dataset(table, sd, ed, mode, instruments, stats=None,
                  cluster_boundaries=None, label_dict=None,
                  price_norm_mode=None, vol_norm_mode=None):
    """构建训练/推理张量 (按 instrument 分批加载, 控制内存峰值)。

    参数:
      table: 数据表名
      sd, ed: 起止日期
      mode: "train" | "val" | "infer"
      instruments: 标的列表
      stats: (mean, std) 元组, infer 时复用训练集统计量
      cluster_boundaries: 聚类边界数组
      label_dict: 标签字典 (train/val 模式)
      price_norm_mode: 价格归一化模式 (默认使用全局 PRICE_NORM_MODE)
      vol_norm_mode: 量归一化模式 (默认使用全局 VOL_NORM_MODE)

    返回: (X, y, idx_df, stats, cluster_ids)
    """
    t0 = time.time()
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)

    if mode in ("train", "val") and label_dict is None:
        label_dict = _build_label_dict(instruments, sd, ed,
                                       return_closes=False)
        logger.info("标签字典构建完成", n_labels=len(label_dict))

    # 按 instrument 分批加载 (避免全量物化导致 OOM)
    inst_sorted = sorted(instruments)
    n_batches = (len(inst_sorted) + INSTRUMENT_BATCH - 1) // INSTRUMENT_BATCH
    X_parts, ys, keys, all_cids = [], [], [], []
    for bi in range(n_batches):
        batch_inst = inst_sorted[bi * INSTRUMENT_BATCH:
                                 (bi + 1) * INSTRUMENT_BATCH]
        Xb, yb, kb, cidb = _window_batch(
            table, sd, ed, mode, batch_inst,
            label_dict, sd_ts, ed_ts,
            cluster_boundaries=cluster_boundaries,
            price_norm_mode=price_norm_mode,
            vol_norm_mode=vol_norm_mode)
        if len(kb):
            X_parts.append(Xb)
            ys.extend(yb)
            keys.extend(kb)
            if cluster_boundaries is not None:
                all_cids.append(cidb)
        logger.info("批次完成",
                    batch=f"{bi+1}/{n_batches}",
                    batch_samples=len(kb), cum_samples=len(keys))

    if not keys:
        raise RuntimeError(
            f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")

    # 预分配最终 X, 逐批拷入并释放 (避免 np.concatenate 2× 内存翻倍)
    total = sum(p.shape[0] for p in X_parts)
    X = np.empty((total, SEQ_LEN, N_FEAT), np.float32)
    off = 0
    for j in range(len(X_parts)):
        p = X_parts[j]
        X[off:off + p.shape[0]] = p
        off += p.shape[0]
        X_parts[j] = None
    del X_parts
    gc.collect()

    # z-score 标准化 (训练集计算 mean/std, 推理集复用)
    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        stats = (flat.mean(0).astype(np.float32),
                 flat.std(0).astype(np.float32) + 1e-6)
    X -= stats[0]
    X /= stats[1]

    cluster_ids = (np.concatenate(all_cids) if all_cids
                   else np.empty((total,), dtype=np.int64))

    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])
    logger.info(f"{mode} 集构建完成", samples=len(keys),
                elapsed=round(time.time() - t0, 2),
                n_clusters=len(np.unique(cluster_ids))
                    if len(cluster_ids) > 0 else 0)
    return (X,
            np.array(ys, np.float32) if mode in ("train", "val") else None,
            idx_df, stats, cluster_ids)


def process_labels(ytr, train_dates):
    """标签截面处理: 每日去均值 + 1%/99% 截尾。

    平台评估会做截面 z-score 中性化, 这里只做轻量去均值。
    """
    train_dates = train_dates.copy()
    train_dates["y_raw"] = ytr
    ds = train_dates.groupby("date")["y_raw"].agg(["mean", "std"])
    train_dates["dm"] = train_dates["date"].map(ds["mean"])
    train_dates["ds"] = train_dates["date"].map(ds["std"])
    y_cs = ((ytr - train_dates["dm"].values)
            / (train_dates["ds"].values + 1e-6)).astype(np.float32)
    y_cs = np.nan_to_num(y_cs, nan=0.0)
    lo, hi = np.percentile(y_cs, [1., 99.])
    y_cs = np.clip(y_cs, lo, hi)
    logger.info("标签处理",
                raw_mean=round(float(ytr.mean()), 6),
                raw_std=round(float(ytr.std()), 6))
    return y_cs


# ============================================================================
# 模型保存 (JSON 文本格式, 符合竞赛规则)
# ============================================================================
def save_model(ckpt, path=MODEL_PATH):
    """保存模型权重 + 配置 + 统计量到 JSON 文件。

    结构:
      {
        "state_dict": {param_name: {dtype, shape, data[flat list]}},
        "model_cfg": {...},
        "feature_cols": [...],
        "seq_len": 240,
        "mean": [...], "std": [...],
        "cluster_boundaries": [...],
        "price_norm_mode": "pre_close",
        "vol_norm_mode": "daily_max",
        "target_transform": "log1p"
      }
    """
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        tensors[k] = {
            "dtype": str(v.detach().cpu().dtype).replace("torch.", ""),
            "shape": list(v.shape),
            "data": v.detach().cpu().reshape(-1).tolist(),
        }
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    payload["feature_flags"] = FEATURE_FLAGS
    payload["spectral_filter"] = SPECTRAL_FILTER
    payload["use_cluster_emb"] = USE_CLUSTER_EMB
    payload["n_clusters"] = N_CLUSTERS
    payload["cluster_method"] = CLUSTER_METHOD
    payload["price_norm_mode"] = PRICE_NORM_MODE
    payload["vol_norm_mode"] = VOL_NORM_MODE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def load_model(path=MODEL_PATH, map_location="cpu"):
    """从 JSON 文件加载模型 (推理使用)。"""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"],
                         dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ============================================================================
# 训练入口 (平台私榜调用此函数从零重训)
# ============================================================================
def get_recon_weight(epoch, total_epochs):
    """DAE 重建损失权重: 从初始值线性衰减到最终值。"""
    progress = epoch / max(total_epochs - 1, 1)
    return (RECON_WEIGHT_INITIAL
            + (RECON_WEIGHT_FINAL - RECON_WEIGHT_INITIAL) * progress)


def train_and_save(datasources, model_path=MODEL_PATH):
    """从零训练模型并保存到 JSON 文件。

    平台私榜阶段调用此函数:
      - datasources["bar1m"]: 平台提供的训练集表名
      - 训练完成后产出 model_path 指定的 JSON 文件
      - 推理 notebook 的 main() 加载此文件进行打分

    注意: 函数内部训练表名已硬编码为 TRAIN_TABLE,
    datasources 参数保留用于兼容平台接口但不用于训练数据查询。
    """
    table = TRAIN_TABLE  # ★ 训练表名硬编码, 不使用 datasources

    # -- 随机种子 (确保可复现) --
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("V11-v2 训练启动",
                train_start=TRAIN_START, train_end=TRAIN_END,
                epochs=EPOCHS, price_norm=PRICE_NORM_MODE,
                vol_norm=VOL_NORM_MODE,
                spectral_filter=SPECTRAL_FILTER,
                denoise_mode=DENOISE_MODE,
                use_cluster_emb=USE_CLUSTER_EMB,
                n_clusters=N_CLUSTERS)

    # ---- 训练集构建 ----
    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    train_inst = pool(TRAIN_START, TRAIN_END)

    cluster_boundaries = None
    label_dict = None
    if USE_CLUSTER_EMB:
        # 标签字典 + 收盘价收集 (用于聚类边界计算)
        label_dict, train_closes = _build_label_dict(
            train_inst, TRAIN_START, TRAIN_END, return_closes=True)
        logger.info("标签字典构建完成 (含收盘价收集)",
                    n_labels=len(label_dict))
        cluster_boundaries = compute_cluster_boundaries_from_closes(
            train_closes, N_CLUSTERS, method=CLUSTER_METHOD,
            z_range=Z_RANGE, z_exponent=Z_EXPONENT)
        del train_closes

    Xtr, ytr_raw, train_dates_df, stats, cluster_ids = build_dataset(
        table, TRAIN_START, TRAIN_END, "train", train_inst,
        cluster_boundaries=cluster_boundaries, label_dict=label_dict)
    ytr = process_labels(ytr_raw, train_dates_df)
    logger.info("训练集样本数", n_train=len(Xtr))

    if USE_CLUSTER_EMB and len(cluster_ids) > 0:
        unique, counts = np.unique(cluster_ids, return_counts=True)
        dist_str = ", ".join(f"c{c}={n}" for c, n in zip(unique, counts))
        logger.info("聚类分布", clusters=dist_str)

    # ---- 模型初始化 ----
    flags = FEATURE_FLAGS
    model = StockTransformer(
        **MODEL_CFG,
        session_aware=flags["session_aware"],
        use_preclose_gate=flags["use_preclose_gate"],
        use_tod_emb=flags["use_tod_emb"],
        use_prenorm=flags["use_prenorm"],
        use_spectral_filter=SPECTRAL_FILTER,
        use_cluster_emb=USE_CLUSTER_EMB,
        n_clusters=N_CLUSTERS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("可训练参数量", n_params=f"{n_params:,}")

    # ---- DataLoader ----
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr),
                      torch.from_numpy(ytr),
                      torch.from_numpy(cluster_ids)),
        batch_size=BATCH, shuffle=True,
        pin_memory=(device.type == "cuda"), drop_last=True)

    # ---- 优化器 + 损失 + EMA ----
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    recon_fn = nn.MSELoss(reduction="none")
    ema = EMA(model, decay=EMA_DECAY) if flags["use_ema"] else None

    total_steps = len(loader) * EPOCHS
    warmup_steps = len(loader) * WARMUP_EPOCHS

    # ---- 训练循环 ----
    model.train()
    global_step = 0
    logger.info("开始训练", epochs=EPOCHS,
                steps_per_epoch=len(loader),
                denoise_mode=DENOISE_MODE,
                spectral_filter=SPECTRAL_FILTER,
                cluster_emb=USE_CLUSTER_EMB,
                price_norm=PRICE_NORM_MODE,
                vol_norm=VOL_NORM_MODE)

    for ep in range(EPOCHS):
        t_ep = time.time()
        tot_loss = tot_pred = tot_recon = 0.0
        tot_corr, nb = 0.0, 0
        opt.zero_grad()
        recon_weight = get_recon_weight(ep, EPOCHS)

        for step, (xb, yb, cid) in enumerate(loader):
            # -- 学习率调度: warmup → cosine 衰减 --
            if global_step < warmup_steps:
                lr = LR_PEAK * (global_step + 1) / max(warmup_steps, 1)
            else:
                prog = ((global_step - warmup_steps)
                        / max(total_steps - warmup_steps, 1))
                lr = LR_PEAK * 0.5 * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = lr

            xb, yb, cid = (xb.to(device, non_blocking=True),
                           yb.to(device, non_blocking=True),
                           cid.to(device, non_blocking=True))

            # -- DAE: 破坏输入 → 预测 + 重建 --
            if DENOISE_MODE != "none":
                xb_c, xb_clean, recon_mask = corrupt_input(
                    xb, mode=DENOISE_MODE)
                pred, recon = model(xb_c, cluster_ids=cid, return_recon=True)
                pred_loss = loss_fn(pred, yb)
                rpe = recon_fn(recon, xb_clean).mean(dim=-1)
                recon_loss = ((rpe * recon_mask.float()).sum()
                              / (recon_mask.float().sum() + 1e-8))
                loss = pred_loss + recon_weight * recon_loss
                tot_recon += recon_loss.item()
            else:
                pred = model(xb, cluster_ids=cid, return_recon=False)
                pred_loss = loss_fn(pred, yb)
                loss = pred_loss

            # -- 梯度累积 + 反向传播 --
            loss = loss / max(GRAD_ACCUM, 1)
            loss.backward()
            if (step + 1) % max(GRAD_ACCUM, 1) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad()
                if ema is not None:
                    ema.update()

            # -- 批次内 Pearson 相关 (监控训练) --
            with torch.no_grad():
                pc, yc = pred - pred.mean(), yb - yb.mean()
                bc = ((pc * yc).sum()
                      / ((pc.norm() + 1e-8) * (yc.norm() + 1e-8)))
            tot_loss += loss.item() * GRAD_ACCUM
            tot_pred += pred_loss.item()
            tot_corr += bc.item()
            nb += 1
            global_step += 1

        # -- Epoch 日志 --
        avg_loss = tot_loss / max(nb, 1)
        avg_pred = tot_pred / max(nb, 1)
        avg_recon = tot_recon / max(nb, 1)
        avg_corr = tot_corr / max(nb, 1)
        elapsed = time.time() - t_ep

        recon_str = (f"recon={avg_recon:.4f} λ={recon_weight:.3f}"
                     if DENOISE_MODE != "none" else "")
        logger.info("epoch 完成", epoch=ep + 1,
                    loss=round(avg_loss, 6),
                    pred_loss=round(avg_pred, 6),
                    batch_corr=round(avg_corr, 4),
                    lr=round(lr, 8), elapsed=round(elapsed, 1),
                    extra=recon_str)
        sys.stdout.flush()  # 确保日志实时输出

    # ---- 保存模型 (使用 EMA 权重) ----
    if ema is not None:
        ema.apply_shadow()
    mean, std = stats
    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
        "cluster_boundaries": (
            np.asarray(cluster_boundaries, np.float32).tolist()
            if cluster_boundaries is not None else []),
        "target_transform": "log1p",
    }, model_path)
    logger.info("模型已保存", path=model_path,
                price_norm=PRICE_NORM_MODE,
                vol_norm=VOL_NORM_MODE)
    sys.stdout.flush()
    return model_path


# ============================================================================
# 本地测试入口
# ============================================================================
if __name__ == "__main__":
    # ★ 本地测试: 硬编码训练表名, 不使用 datasources 变量
    # 平台提交后, 私榜阶段会调用 train_and_save(datasources)
    # 函数内部使用 TRAIN_TABLE 而非 datasources["bar1m"]
    train_and_save({"bar1m": TRAIN_TABLE})
