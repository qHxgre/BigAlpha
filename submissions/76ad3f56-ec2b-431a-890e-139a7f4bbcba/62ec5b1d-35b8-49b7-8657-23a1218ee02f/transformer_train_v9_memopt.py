# -*- coding: utf-8 -*-
"""Transformer V9 — 频域滤波 + DAE 降噪 (V7 有效部分保留, V8 失败部分移除)。

V8 失败根因:
  1. Span masking 破坏太狠 → 模型学到模糊均值, 非有用结构
  2. 对比损失与预测冲突 → 强制忽略高频信号
  3. 课程式 → 训练分布漂移
  4. λ 0.8 太高太久 → 压制预测目标

V9 = V7 DAE 核心 + 可学习频域滤波 (信号处理层面降噪, 互补不冲突):
  1. Learnable Spectral Filter: FFT → 学习频率权重 → IFFT
     - 1.5K 参数, 本质是 Wiener filter
     - 高频 = microstructure noise → 应抑制
     - 低频 = 日内趋势 → 应保留
  2. 保留 V7 的随机 mask + 高斯噪声 DAE
  3. 移除 V8 的 span mask/对比损失/课程式/高λ

用法:
    python transformer_train_v9_memopt.py
    # 关闭频域滤波: SPECTRAL_FILTER = False → 回退到 V7

V9-memopt (2026/07/11): 仅优化 build_dataset 的内存峰值, 训练逻辑/输出不变。
    详见 build_dataset docstring 与 MARKDOWN.md。
"""

import os, json, time, math, gc
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
# 配置
# ============================================================================
TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31 23:59:59"    # 全量数据训练
VAL_START   = "2023-07-01"
VAL_END     = "2023-12-31 23:59:59"
BAR1D_TABLE = "bigalpha_2026_bar1d"
TRAIN_TABLE = "bigalpha_2026_stock_bar1m"

SEQ_LEN       = 240
EPOCHS        = 32
BATCH         = 512
LR_PEAK       = 5e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 3
GRAD_CLIP     = 1.0
SEED          = 42

EMA_DECAY     = 0.999
EARLY_STOP_PATIENCE = 15
GRAD_ACCUM = 1

# —— 分批加载 (V9-memopt): 每次只 query 这么多只股票, 避免全量 df 物化 OOM ——
# 全量 5 年 1min 单块 df ≈ 40GB, 经 arrow→pandas 物化瞬时峰值 ≈ 100GB → OOM。
# 按 instrument 分批后, 峰值 ≈ 单批 df + 累积窗口 ≈ 20~30GB。调小更安全, 调大更快。
INSTRUMENT_BATCH = 150

# —— 5 日模型配置 (1200 根 1min bar, 与 240-bar 模型集成) ——
SEQ_LEN_5D    = 1200   # 5 天 × 240 min/天
BATCH_5D      = 16     # 长序列用小 batch 控制显存
LR_PEAK_5D    = 3e-4
GRAD_ACCUM_5D = 4      # 梯度累积模拟更大 batch (16×4=64)
SUBSAMPLE_5D  = 5      # 窗口降采样: 每 5 个交易日取 1 窗 (邻窗重叠 4/5 天)
# MODEL_CFG_5D 定义在 N_FEAT 之后 (需要引用 N_FEAT)

# ============================================================================
# V9 降噪配置
# ============================================================================

# —— 频域滤波 (V9 核心新增) ——
SPECTRAL_FILTER   = True    # 是否启用可学习频域滤波
SPEC_FILTER_DIM   = 32      # 频域滤波的中间维度 (轻量)

# —— DAE 降噪 (V7 验证有效的方案, 保留) ——
MASK_RATIO        = 0.15
NOISE_STD         = 0.10
DENOISE_MODE      = "both"  # "mask" | "noise" | "both" | "none"

# —— 重建权重 (V7 式: 简单线性衰减) ——
RECON_WEIGHT_INITIAL = 0.5
RECON_WEIGHT_FINAL   = 0.03

# ============================================================================
# 特征开关
# ============================================================================
FEATURE_FLAGS = {
    "use_ema":          True,
    "use_prenorm":      True,
    "use_tod_emb":      True,
    "session_aware":    False,
    "use_preclose_gate": False,
    "use_val_earlystop": False,   # 全量训练时关闭早停
}

PRICE_COLS = ["pre_close", "open", "high", "low", "close", "vwap",
               "bid_price1", "ask_price1"]
VOL_COLS   = ["deal_number", "volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)   # 13 (8 price + 5 volume)

# SQL 只需查询数据表中存在的列 (vwap 在 Python 中计算)
SQL_COLS = ["pre_close", "open", "high", "low", "close",
            "bid_price1", "ask_price1",
            "deal_number", "volume", "amount",
            "bid_volume1", "ask_volume1"]

MODEL_CFG = dict(
    n_feat=N_FEAT, d_model=96, nhead=4, nlayers=3,
    dim_ff=192, seq_len=SEQ_LEN, dropout=0.15,
)

MODEL_CFG_5D = dict(
    n_feat=N_FEAT, d_model=96, nhead=4, nlayers=3,
    dim_ff=192, seq_len=SEQ_LEN_5D, dropout=0.15,
)

MODEL_PATH_5D = os.path.join(_HERE, "transformer_model_5d.json")


# ============================================================================
# V9 核心: 可学习频域滤波
# ============================================================================

class SpectralFilter1D(nn.Module):
    """可学习频域滤波器 — 1D 时序 Wiener filter。

    原理:
      - FFT 将 bar 序列变换到频域
      - 可学习权重对每个频率分量加权 (抑制高频噪声, 保留低频趋势)
      - IFFT 还原为时域信号

    参数量: ~1.5K (几乎可忽略)
    灵感: Kronos 量化是离散瓶颈去噪, 这是频域去噪 — 不同路径, 同样哲学
    """
    def __init__(self, seq_len=240, n_feat=12, hidden_dim=32):
        super().__init__()
        n_freqs = seq_len // 2 + 1  # rfft 的频率分量数 = 121

        # 方法: 对频率维度做轻量 MLP, 输出每个频率的权重
        # 输入: log(magnitude + eps) — 更平滑的输入空间
        # 输出: sigmoid → [0, 1] 权重
        self.freq_net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        # 全局可学习缩放 (每个特征不同)
        self.global_scale = nn.Parameter(torch.ones(n_feat))

    def forward(self, x):
        """x: (B, L, F) → (B, L, F)"""
        B, L, F = x.shape

        # FFT 沿时间维度
        x_freq = torch.fft.rfft(x.float(), dim=1)  # (B, L//2+1, F), complex

        # 计算每个频率分量的权重
        # 用 log magnitude 作为输入来预测该频率的重要性
        mag = torch.log(x_freq.abs() + 1e-6)        # (B, L//2+1, F)

        # 对每个频率位置的每个特征计算权重
        weights = self.freq_net(mag.unsqueeze(-1)).squeeze(-1)  # (B, L//2+1, F)

        # 乘上全局缩放
        weights = weights * self.global_scale.view(1, 1, F)

        # 应用频域滤波
        x_freq_filtered = x_freq * weights

        # IFFT 还原
        x_filtered = torch.fft.irfft(x_freq_filtered, n=L, dim=1)

        # 残差连接: 滤波后 + 原始 (让模型学习"滤多少")
        alpha = 0.5  # 可调, 或改为可学习
        return alpha * x_filtered + (1 - alpha) * x


# ============================================================================
# EMA
# ============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow, self._backup = {}, {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.data.clone()
    def update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.-self.decay)
    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._backup: p.data.copy_(self._backup[n])


# ============================================================================
# V7 式 DAE 破坏 (验证有效, 保留)
# ============================================================================

def apply_bar_mask(x, mask_ratio=0.15, mask_val=0.0):
    B, L, F = x.shape
    mask = torch.rand(B, L, device=x.device) < mask_ratio
    for b in range(B):
        if mask[b].sum() > int(L * 0.30):
            keep = torch.randperm(L, device=x.device)[:int(L * 0.70)]
            mask[b, keep] = False
    xc = x.clone(); xc[mask] = mask_val
    return xc, mask

def apply_gaussian_noise(x, noise_std=0.10):
    B, L, _ = x.shape
    nmask = torch.rand(B, L, device=x.device) < 0.5
    noise = torch.randn_like(x) * noise_std
    xc = x.clone()
    xc[nmask] = x[nmask] + noise[nmask]
    return xc, nmask

def corrupt_input(x, mode="both"):
    B, L, F = x.shape
    xt = x.clone()
    all_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    if mode == "none": return x, None, None
    if mode in ("mask", "both"):
        xt, mm = apply_bar_mask(xt); all_mask = all_mask | mm
    if mode in ("noise", "both"):
        xt, mn = apply_gaussian_noise(xt); all_mask = all_mask | mn
    return xt, x.clone(), all_mask  # corrupted, clean_target, mask


# ============================================================================
# 模型: StockTransformer V9
# ============================================================================

class StockTransformerV9(nn.Module):
    """V9 = 频域滤波 + V7 DAE 骨干。

    新增: SpectralFilter1D (频域降噪, 信号处理层面)
    保留: V7 的 DAE 框架 (Conv Stem + Encoder + Pred Head + Recon Head)
    移除: V8 的 span mask / 对比损失 / 课程式 / 高 λ warmup
    """
    def __init__(self, n_feat=N_FEAT, d_model=96, nhead=4, nlayers=3,
                 dim_ff=192, seq_len=SEQ_LEN, dropout=0.15,
                 session_aware=False, use_preclose_gate=False,
                 use_tod_emb=True, use_prenorm=True,
                 use_spectral_filter=True):
        super().__init__()
        self.session_aware = session_aware
        self.use_tod_emb = use_tod_emb
        self.use_preclose_gate = use_preclose_gate
        self.use_spectral_filter = use_spectral_filter

        # —— 频域滤波 (V9 新增) ——
        if use_spectral_filter:
            self.spectral_filter = SpectralFilter1D(
                seq_len=seq_len, n_feat=n_feat, hidden_dim=SPEC_FILTER_DIM)

        # —— Conv Stem (V7 同款) ——
        if session_aware:
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

        if use_preclose_gate:
            self.pc_gate = nn.Sequential(
                nn.Linear(1, d_model // 4), nn.GELU(),
                nn.Linear(d_model // 4, d_model), nn.Sigmoid())

        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if use_tod_emb:
            self.tod_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.trunc_normal_(self.tod_emb, std=0.02)

        el = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                        batch_first=True, activation="gelu",
                                        norm_first=use_prenorm)
        self.encoder = nn.TransformerEncoder(el, nlayers)

        self.attn_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

        # 重建头: V7 同款 (简单有效)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, n_feat))

    def encode(self, x):
        B, L, _ = x.shape

        # V9 新增: 频域滤波 (在 conv stem 之前)
        if self.use_spectral_filter:
            x = self.spectral_filter(x)

        if self.use_preclose_gate:
            ratio = (x[:, :, 1] / (x[:, :, 0] + 1e-8)).unsqueeze(-1)
            pc_gate = self.pc_gate(ratio).unsqueeze(1)

        if self.session_aware:
            hm = self.morning_conv(x[:, :120, :].transpose(1, 2)).transpose(1, 2)
            ha = self.afternoon_conv(x[:, 120:, :].transpose(1, 2)).transpose(1, 2)
            h = torch.cat([hm, ha], dim=1)
            h = self.session_fuse(h)
        else:
            h = self.conv_stem(x.transpose(1, 2))
            h = self.conv_stem2(h).transpose(1, 2)

        if self.use_preclose_gate:
            h = h * pc_gate.squeeze(1)

        pl = min(L, self.pos.shape[1])
        h = h[:, :pl, :] + self.pos[:, :pl, :]
        if self.use_tod_emb:
            h = h + self.tod_emb[:, :pl, :]

        return self.encoder(h)

    def forward(self, x, return_recon=False):
        h = self.encode(x)
        w = F.softmax(self.attn_gate(h), dim=1)
        pred = self.head((h * w).sum(dim=1)).squeeze(-1)
        if not return_recon:
            return pred
        recon = self.recon_head(h)
        return pred, recon


# ============================================================================
# 数据管道
# ============================================================================

def pool(sd, ed):
    df = dai.query("SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
                   filters={"date": [sd, ed]}).df()
    return df["instrument"].tolist()

def _build_label_dict(instruments, sd, ed):
    ed_ext = (pd.to_datetime(ed) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    sql = f"SELECT date, instrument, close FROM {BAR1D_TABLE} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed_ext], "instrument": instruments}).df()
    labels = {}
    for ins, sub in df.groupby("instrument", sort=False):
        sub = sub.sort_values("date")
        closes = sub["close"].to_numpy(np.float64)
        dates  = sub["date"].dt.normalize().to_numpy()
        for i in range(len(closes) - 1):
            if closes[i] <= 0: continue
            r_raw = closes[i + 1] / closes[i] - 1.0
            r = np.log1p(r_raw) if r_raw > -1.0 else r_raw
            if np.isfinite(r): labels[(pd.Timestamp(dates[i]), ins)] = np.float32(r)
    return labels

def _window_batch(table, sd, ed, mode, batch_inst, label_dict, sd_ts, ed_ts):
    """查询一批股票的特征并构建窗口。窗口逻辑与原版逐行相同, 只是作用在一批股票上。

    返回 (Xb, ys_list, keys_list): Xb 是本批堆叠好的 (nb, SEQ_LEN, N_FEAT) 未标准化数组。
    本批 df 用完即 del, 因此峰值只与"单批股票数"有关, 与总股票数无关。
    """
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    sql = f"SELECT date, instrument, {', '.join(SQL_COLS)} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": batch_inst}).df().fillna(0)

    # —— 盘口价格缺失填充: 数据源可能用 0 填充缺失的 bid/ask 价格,
    #     在取对数前用 close 填充, 避免 log(0) = -inf ——
    for c in ["bid_price1", "ask_price1"]:
        bad = df[c] <= 0
        if bad.any():
            df.loc[bad, c] = df.loc[bad, "close"]

    # —— 日内 VWAP: 累计成交额 / 累计成交量, 日内重置 ——
    #     VWAP 是重要的日内基准价, 捕捉价格与成交量加权均价的位置关系
    df["_day"] = df["date"].dt.normalize()
    cum_amt = df.groupby(["instrument", "_day"])["amount"].cumsum()
    cum_vol = df.groupby(["instrument", "_day"])["volume"].cumsum()
    df["vwap"] = cum_amt / cum_vol.clip(lower=1e-8)
    # 极端情形回退到 close (如全天无成交或首笔 bar 之前)
    bad = ~np.isfinite(df["vwap"])
    if bad.any():
        df.loc[bad, "vwap"] = df.loc[bad, "close"]
    df.drop(columns=["_day"], inplace=True)

    # —— 量字段: log1p ——
    for c in VOL_COLS:   df[c] = np.log1p(df[c].clip(lower=0))
    # —— 价格字段归一化: 除以 pre_close 消除股票面值差异, 再取对数 ——
    # pre_close 在日内保持不变 (前一交易日收盘价), 归一化后所有价格变为相对比率
    # open,high,low,close,vwap,bid/ask_price → log(price / pre_close)   (量纲无关)
    # pre_close 自身保留 log(pre_close) 以提供绝对价格水平信息
    pc_safe = np.maximum(df["pre_close"].to_numpy(), 1e-8)
    for c in PRICE_COLS:
        if c == "pre_close":
            df[c] = np.log(pc_safe)
        else:
            ratio = df[c].to_numpy() / pc_safe
            df[c] = np.log(np.maximum(ratio, 1e-8))

    wins, ys, keys = [], [], []
    for ins, sub in df.groupby("instrument", sort=False):
        if len(sub) <= SEQ_LEN: continue
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        dates = day[close_pos]
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts: continue
            if mode in ("train", "val"):
                label = label_dict.get((d, ins))
                if label is None: continue
                ys.append(label)
            wins.append(feats[p - SEQ_LEN + 1: p + 1])
            keys.append((d, ins))
    del df
    gc.collect()
    if wins:
        Xb = np.stack(wins).astype(np.float32)   # 本批小, 堆叠峰值可控
    else:
        Xb = np.empty((0, SEQ_LEN, N_FEAT), np.float32)
    del wins
    return Xb, ys, keys


def _window_batch_5d(table, sd, ed, mode, batch_inst, label_dict, sd_ts, ed_ts):
    """5 日窗口版本: 每窗 1200 根 1min bar, 价格归一化除以首日 pre_close。

    与 _window_batch 的关键差异:
      1. 窗口长度 = 5 天 (1200 bar), 而非 1 天 (240 bar)
      2. 价格归一化: 窗口内所有 bar 的价格除以首日 pre_close, 而非每日各自的 pre_close
         → 编码 5 日累计收益, 捕捉跨日跳空和趋势
      3. 价格字段保持原始值直到建窗, 再按窗口归一化 (无法预先全局归一化)
      4. VWAP 仍每日分别计算, 但不除以当日 pre_close, 统一除以首日 pre_close
    """
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    sql = f"SELECT date, instrument, {', '.join(SQL_COLS)} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": batch_inst}).df()

    # —— 盘口价格缺失填充 ——
    for c in ["bid_price1", "ask_price1"]:
        bad = df[c] <= 0
        if bad.any():
            df.loc[bad, c] = df.loc[bad, "close"]

    # —— 日内 VWAP (每日独立计算) ——
    df["_day"] = df["date"].dt.normalize()
    cum_amt = df.groupby(["instrument", "_day"])["amount"].cumsum()
    cum_vol = df.groupby(["instrument", "_day"])["volume"].cumsum()
    df["vwap"] = cum_amt / cum_vol.clip(lower=1e-8)
    bad = ~np.isfinite(df["vwap"])
    if bad.any():
        df.loc[bad, "vwap"] = df.loc[bad, "close"]
    df.drop(columns=["_day"], inplace=True)

    # —— 量字段 log1p (与 240-bar 一致) ——
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))

    # 价格字段保持原始值 — 建窗时按首日 pre_close 归一化
    pre_close_idx = FEATURE_COLS.index("pre_close")
    price_indices = [FEATURE_COLS.index(c) for c in PRICE_COLS]

    wins, ys, keys = [], [], []
    for ins, sub in df.groupby("instrument", sort=False):
        feats = sub[FEATURE_COLS].to_numpy(np.float32)  # raw prices + log volumes
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        dates = day[close_pos]

        for k, p in enumerate(close_pos):
            if k < 4: continue          # 需要至少 5 个交易日
            # 降采样: 邻窗重叠 4/5 天, 每 SUBSAMPLE_5D 个交易日取 1 窗
            if (k - 4) % SUBSAMPLE_5D != 0: continue
            d = pd.Timestamp(dates[k])
            if d < sd_ts or d > ed_ts: continue

            # 窗口: 日 [k-4, k] → 5 个完整交易日
            first_bar = 0 if k == 4 else close_pos[k - 5] + 1
            if p - first_bar + 1 != SEQ_LEN_5D:
                continue                  # 跳过不完整的 5 日窗口

            if mode in ("train", "val"):
                label = label_dict.get((d, ins))
                if label is None: continue
                ys.append(label)

            win = feats[first_bar: p + 1].copy()  # (1200, N_FEAT)

            # —— 5d 归一化: 所有价格除以首日 pre_close ——
            first_pc = max(win[0, pre_close_idx], 1e-8)
            for j in price_indices:
                c = FEATURE_COLS[j]
                if c == "pre_close":
                    win[:, j] = np.log(first_pc)
                else:
                    ratio = win[:, j] / first_pc
                    win[:, j] = np.log(np.maximum(ratio, 1e-8))

            wins.append(win)
            keys.append((d, ins))

    del df
    gc.collect()
    if wins:
        Xb = np.stack(wins).astype(np.float32)
    else:
        Xb = np.empty((0, SEQ_LEN_5D, N_FEAT), np.float32)
    del wins
    return Xb, ys, keys


def build_dataset(table, sd, ed, mode, instruments, stats=None):
    """构建训练/推理张量 —— 分批(按 instrument)加载版, 避免全量 df 物化 OOM。

    真正的 OOM 位置是 `dai.query(...).df()` 物化全量 5 年 1min df (≈103GB anon-rss),
    发生在任何窗口拼接代码之前。因此这里改为按 instrument 分批 query + 建窗, 增量拼接:
      - 单批 df 用完即释放 → 峰值只与 INSTRUMENT_BATCH 有关 (≈20~30GB), 与总股票数无关;
      - 标准化统计量仍在"全部窗口"上计算, 与原版一致;
      - 预分配最终 X 并逐批拷入后释放各批, 拼接峰值 ≈ 1×X (~17GB), 无 concatenate 翻倍。

    输出 (X, ys, idx_df, stats) 与原版在"窗口集合与统计量"上一致 (行顺序按 instrument 排序,
    与原版单条 SQL `ORDER BY instrument,date` 的顺序相同)。训练逻辑不变。
    """
    t0 = time.time()
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)

    label_dict = None
    if mode in ("train", "val"):
        label_dict = _build_label_dict(instruments, sd, ed)   # bar1d 日线, 体量小, 单次查询
        logger.info("标签字典构建完成", n_labels=len(label_dict))

    inst_sorted = sorted(instruments)
    n_batches = (len(inst_sorted) + INSTRUMENT_BATCH - 1) // INSTRUMENT_BATCH
    X_parts, ys, keys = [], [], []
    for bi in range(n_batches):
        batch_inst = inst_sorted[bi * INSTRUMENT_BATCH:(bi + 1) * INSTRUMENT_BATCH]
        Xb, yb, kb = _window_batch(table, sd, ed, mode, batch_inst, label_dict, sd_ts, ed_ts)
        if len(kb):
            X_parts.append(Xb)
            ys.extend(yb)
            keys.extend(kb)
        logger.info("批次完成", batch=f"{bi+1}/{n_batches}",
                    batch_samples=len(kb), cum_samples=len(keys))
    if not keys:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")

    # 预分配最终 X, 逐批拷入并释放, 峰值 ≈ 1×X (不做 concatenate 翻倍)
    total = sum(p.shape[0] for p in X_parts)
    X = np.empty((total, SEQ_LEN, N_FEAT), np.float32)
    off = 0
    for j in range(len(X_parts)):
        p = X_parts[j]
        X[off:off + p.shape[0]] = p
        off += p.shape[0]
        X_parts[j] = None            # 逐批释放
    del X_parts
    gc.collect()

    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        stats = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32) + 1e-6)
    # 原地标准化, 避免全量临时数组
    X -= stats[0]
    X /= stats[1]
    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])
    logger.info(f"{mode} 集构建完成", samples=len(keys), elapsed=round(time.time()-t0, 2))
    return (X, np.array(ys, np.float32) if mode in ("train","val") else None, idx_df, stats)


def build_dataset_5d(table, sd, ed, mode, instruments, stats=None):
    """5 日窗口版 build_dataset: 每窗 1200 bar, 归一化除以首日 pre_close。

    与 build_dataset 关键差异:
      - 调用 _window_batch_5d 建窗
      - 使用 SEQ_LEN_5D (1200) 而非 SEQ_LEN (240)
      - 输出 X 形状为 (N, 1200, N_FEAT)
    """
    t0 = time.time()
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)

    label_dict = None
    if mode in ("train", "val"):
        label_dict = _build_label_dict(instruments, sd, ed)
        logger.info("5d 标签字典构建完成", n_labels=len(label_dict))

    inst_sorted = sorted(instruments)
    n_batches = (len(inst_sorted) + INSTRUMENT_BATCH - 1) // INSTRUMENT_BATCH
    X_parts, ys, keys = [], [], []
    for bi in range(n_batches):
        batch_inst = inst_sorted[bi * INSTRUMENT_BATCH:(bi + 1) * INSTRUMENT_BATCH]
        Xb, yb, kb = _window_batch_5d(table, sd, ed, mode, batch_inst, label_dict, sd_ts, ed_ts)
        if len(kb):
            X_parts.append(Xb)
            ys.extend(yb)
            keys.extend(kb)
        logger.info("5d 批次完成", batch=f"{bi+1}/{n_batches}",
                    batch_samples=len(kb), cum_samples=len(keys))
    if not keys:
        raise RuntimeError(f"build_dataset_5d 无样本 (mode={mode}, {sd}~{ed})")

    total = sum(p.shape[0] for p in X_parts)
    X = np.empty((total, SEQ_LEN_5D, N_FEAT), np.float32)
    off = 0
    for j in range(len(X_parts)):
        p = X_parts[j]
        X[off:off + p.shape[0]] = p
        off += p.shape[0]
        X_parts[j] = None
    del X_parts
    gc.collect()

    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        stats = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32) + 1e-6)
    X -= stats[0]
    X /= stats[1]
    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])
    logger.info(f"5d {mode} 集构建完成", samples=len(keys), elapsed=round(time.time()-t0, 2))
    return (X, np.array(ys, np.float32) if mode in ("train","val") else None, idx_df, stats)


def process_labels(ytr, train_dates):
    train_dates = train_dates.copy(); train_dates["y_raw"] = ytr
    ds = train_dates.groupby("date")["y_raw"].agg(["mean", "std"])
    train_dates["dm"] = train_dates["date"].map(ds["mean"])
    train_dates["ds"] = train_dates["date"].map(ds["std"])
    y_cs = ((ytr - train_dates["dm"].values) / (train_dates["ds"].values + 1e-6)).astype(np.float32)
    y_cs = np.nan_to_num(y_cs, nan=0.0)
    lo, hi = np.percentile(y_cs, [1., 99.]); y_cs = np.clip(y_cs, lo, hi)
    logger.info("标签处理", raw_mean=round(float(ytr.mean()),6), raw_std=round(float(ytr.std()),6))
    return y_cs

def save_model(ckpt, path=MODEL_PATH):
    sd = ckpt["state_dict"]
    tensors = {k: {"dtype": str(v.detach().cpu().dtype).replace("torch.",""),
                   "shape": list(v.shape), "data": v.detach().cpu().reshape(-1).tolist()}
               for k, v in sd.items()}
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    payload["feature_flags"] = FEATURE_FLAGS
    payload["spectral_filter"] = SPECTRAL_FILTER
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path

def load_model(path=MODEL_PATH, map_location="cpu"):
    with open(path, "r", encoding="utf-8") as f: payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ============================================================================
# 训练入口
# ============================================================================

def get_recon_weight(epoch, total_epochs):
    """V7 式简单线性衰减。"""
    progress = epoch / max(total_epochs - 1, 1)
    return RECON_WEIGHT_INITIAL + (RECON_WEIGHT_FINAL - RECON_WEIGHT_INITIAL) * progress


def train_and_save(datasources, model_path=MODEL_PATH):
    table = datasources["bar1m"]
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("V9 频域滤波+DAE 训练启动",
                spectral_filter=SPECTRAL_FILTER, denoise_mode=DENOISE_MODE)

    # —— 训练集 ——
    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    train_inst = pool(TRAIN_START, TRAIN_END)
    Xtr, ytr_raw, train_dates_df, stats = build_dataset(
        table, TRAIN_START, TRAIN_END, "train", train_inst)
    ytr = process_labels(ytr_raw, train_dates_df)
    logger.info("训练集样本数", n_train=len(Xtr))

    # —— 模型 ——
    flags = FEATURE_FLAGS
    model = StockTransformerV9(
        **MODEL_CFG, session_aware=flags["session_aware"],
        use_preclose_gate=flags["use_preclose_gate"],
        use_tod_emb=flags["use_tod_emb"], use_prenorm=flags["use_prenorm"],
        use_spectral_filter=SPECTRAL_FILTER,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("可训练参数量", n_params=f"{n_params:,}")

    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True,
                        pin_memory=(device.type=="cuda"), drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    recon_fn = nn.MSELoss(reduction="none")
    ema = EMA(model, decay=EMA_DECAY) if flags["use_ema"] else None

    total_steps = len(loader) * EPOCHS
    warmup_steps = len(loader) * WARMUP_EPOCHS

    model.train()
    global_step = 0
    best_val_ic, best_epoch = -999., 0
    best_state_dict = None

    logger.info("开始训练", epochs=EPOCHS, steps_per_epoch=len(loader),
                denoise_mode=DENOISE_MODE, spectral_filter=SPECTRAL_FILTER)

    for ep in range(EPOCHS):
        t_ep = time.time()
        tot_loss = tot_pred = tot_recon = 0.0
        tot_corr, nb = 0.0, 0
        opt.zero_grad()
        recon_weight = get_recon_weight(ep, EPOCHS)

        for step, (xb, yb) in enumerate(loader):
            if global_step < warmup_steps:
                lr = LR_PEAK * (global_step + 1) / max(warmup_steps, 1)
            else:
                prog = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = LR_PEAK * 0.5 * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups: pg["lr"] = lr

            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if DENOISE_MODE != "none":
                xb_c, xb_clean, recon_mask = corrupt_input(xb, mode=DENOISE_MODE)
                pred, recon = model(xb_c, return_recon=True)
                pred_loss = loss_fn(pred, yb)
                rpe = recon_fn(recon, xb_clean).mean(dim=-1)
                recon_loss = (rpe * recon_mask.float()).sum() / (recon_mask.float().sum() + 1e-8)
                loss = pred_loss + recon_weight * recon_loss
                tot_recon += recon_loss.item()
            else:
                pred = model(xb, return_recon=False)
                pred_loss = loss_fn(pred, yb)
                loss = pred_loss

            loss = loss / max(GRAD_ACCUM, 1)
            loss.backward()
            if (step + 1) % max(GRAD_ACCUM, 1) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step(); opt.zero_grad()
                if ema is not None: ema.update()

            with torch.no_grad():
                pc, yc = pred - pred.mean(), yb - yb.mean()
                bc = (pc * yc).sum() / ((pc.norm()+1e-8)*(yc.norm()+1e-8))
            tot_loss += loss.item() * GRAD_ACCUM
            tot_pred += pred_loss.item()
            tot_corr += bc.item(); nb += 1; global_step += 1

        avg_loss = tot_loss/max(nb,1); avg_pred = tot_pred/max(nb,1)
        avg_recon = tot_recon/max(nb,1); avg_corr = tot_corr/max(nb,1)
        elapsed = time.time() - t_ep

        recon_str = f"recon={avg_recon:.4f} λ={recon_weight:.3f}" if DENOISE_MODE != "none" else ""
        logger.info("epoch 完成", epoch=ep+1, loss=round(avg_loss,6),
                    pred_loss=round(avg_pred,6), batch_corr=round(avg_corr,4),
                    lr=round(lr,8), elapsed=round(elapsed,1),
                    extra=recon_str)

    # —— 保存 (EMA 权重) ——
    if ema is not None: ema.apply_shadow()
    mean, std = stats
    save_model({"state_dict": model.state_dict(), "model_cfg": MODEL_CFG,
                "feature_cols": FEATURE_COLS, "seq_len": SEQ_LEN,
                "mean": np.asarray(mean, np.float32).tolist(),
                "std":  np.asarray(std, np.float32).tolist(),
                "target_transform": "log1p"}, model_path)
    logger.info("V9 模型已保存", path=model_path)
    return model_path


def train_and_save_5d(datasources, model_path=MODEL_PATH_5D):
    """训练 5 日窗口模型 (1200 bar / 窗), 与 240-bar 模型集成使用。

    与 train_and_save 的核心差异:
      - 窗口: 5 天 (1200 bar) vs 1 天 (240 bar)
      - 归一化: 除以首日 pre_close vs 每日各自 pre_close
      - 小 batch + 梯度累积 (长序列显存占用大)
    """
    table = datasources["bar1m"]
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("V9 5d 频域滤波+DAE 训练启动",
                seq_len=SEQ_LEN_5D, spectral_filter=SPECTRAL_FILTER,
                denoise_mode=DENOISE_MODE)

    # —— 训练集 ——
    logger.info("5d 构建训练集", start=TRAIN_START, end=TRAIN_END)
    train_inst = pool(TRAIN_START, TRAIN_END)
    Xtr, ytr_raw, train_dates_df, stats = build_dataset_5d(
        table, TRAIN_START, TRAIN_END, "train", train_inst)
    ytr = process_labels(ytr_raw, train_dates_df)
    logger.info("5d 训练集样本数", n_train=len(Xtr))

    # —— 模型 ——
    flags = FEATURE_FLAGS
    model = StockTransformerV9(
        **MODEL_CFG_5D, session_aware=flags["session_aware"],
        use_preclose_gate=flags["use_preclose_gate"],
        use_tod_emb=flags["use_tod_emb"], use_prenorm=flags["use_prenorm"],
        use_spectral_filter=SPECTRAL_FILTER,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("5d 可训练参数量", n_params=f"{n_params:,}")

    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH_5D, shuffle=True,
                        pin_memory=(device.type == "cuda"), drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK_5D, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    recon_fn = nn.MSELoss(reduction="none")
    ema = EMA(model, decay=EMA_DECAY) if flags["use_ema"] else None

    total_steps = len(loader) * EPOCHS
    warmup_steps = len(loader) * WARMUP_EPOCHS

    model.train()
    global_step = 0

    logger.info("5d 开始训练", epochs=EPOCHS, steps_per_epoch=len(loader),
                batch_size=BATCH_5D, grad_accum=GRAD_ACCUM_5D,
                denoise_mode=DENOISE_MODE, spectral_filter=SPECTRAL_FILTER)

    for ep in range(EPOCHS):
        t_ep = time.time()
        tot_loss = tot_pred = tot_recon = 0.0
        tot_corr, nb = 0.0, 0
        opt.zero_grad()
        recon_weight = get_recon_weight(ep, EPOCHS)

        for step, (xb, yb) in enumerate(loader):
            if global_step < warmup_steps:
                lr = LR_PEAK_5D * (global_step + 1) / max(warmup_steps, 1)
            else:
                prog = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = LR_PEAK_5D * 0.5 * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups: pg["lr"] = lr

            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if DENOISE_MODE != "none":
                xb_c, xb_clean, recon_mask = corrupt_input(xb, mode=DENOISE_MODE)
                pred, recon = model(xb_c, return_recon=True)
                pred_loss = loss_fn(pred, yb)
                rpe = recon_fn(recon, xb_clean).mean(dim=-1)
                recon_loss = (rpe * recon_mask.float()).sum() / (recon_mask.float().sum() + 1e-8)
                loss = pred_loss + recon_weight * recon_loss
                tot_recon += recon_loss.item()
            else:
                pred = model(xb, return_recon=False)
                pred_loss = loss_fn(pred, yb)
                loss = pred_loss

            loss = loss / max(GRAD_ACCUM_5D, 1)
            loss.backward()
            if (step + 1) % max(GRAD_ACCUM_5D, 1) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step(); opt.zero_grad()
                if ema is not None: ema.update()

            with torch.no_grad():
                pc, yc = pred - pred.mean(), yb - yb.mean()
                bc = (pc * yc).sum() / ((pc.norm() + 1e-8) * (yc.norm() + 1e-8))
            tot_loss += loss.item() * GRAD_ACCUM_5D
            tot_pred += pred_loss.item()
            tot_corr += bc.item(); nb += 1; global_step += 1

        avg_loss = tot_loss / max(nb, 1); avg_pred = tot_pred / max(nb, 1)
        avg_recon = tot_recon / max(nb, 1); avg_corr = tot_corr / max(nb, 1)
        elapsed = time.time() - t_ep

        recon_str = f"recon={avg_recon:.4f} λ={recon_weight:.3f}" if DENOISE_MODE != "none" else ""
        logger.info("5d epoch 完成", epoch=ep + 1, loss=round(avg_loss, 6),
                    pred_loss=round(avg_pred, 6), batch_corr=round(avg_corr, 4),
                    lr=round(lr, 8), elapsed=round(elapsed, 1),
                    extra=recon_str)

    # —— 保存 (EMA 权重) ——
    if ema is not None: ema.apply_shadow()
    mean, std = stats
    save_model({"state_dict": model.state_dict(), "model_cfg": MODEL_CFG_5D,
                "feature_cols": FEATURE_COLS, "seq_len": SEQ_LEN_5D,
                "mean": np.asarray(mean, np.float32).tolist(),
                "std":  np.asarray(std, np.float32).tolist(),
                "target_transform": "log1p"}, model_path)
    logger.info("V9 5d 模型已保存", path=model_path)
    return model_path


if __name__ == "__main__":
    train_and_save({"bar1m": TRAIN_TABLE})
