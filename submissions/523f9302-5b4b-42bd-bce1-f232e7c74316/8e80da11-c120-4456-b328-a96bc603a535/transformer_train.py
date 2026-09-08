# -*- coding: utf-8 -*-
"""端到端量价时序预测 -- v2 训练侧脚本 (训练/推理单一事实来源 + 从零训练并持久化)。

v2 相对 v1 的唯一改动: **标签按日截面 rank 归一化** (替代 v1 的 1/99 winsorize)。
- 动机: 评分全为 rank 指标 (RankIC/ICIR/SR/Stress), MSE 在原始收益上会被极端值主导,
  而按日 rank 归一化让模型直接学习截面**排序**, 与 RankIC 对齐, 训练更稳、对尾部更鲁棒。
- 做法: 每个交易日内把所有成分股的次日收益转为 percentile rank ∈ [-1,1] 作回归目标。
- 其余与 v1 完全一致 (5m/5日/自动适配表schema/价格归一/z-score/Pre-LN Transformer+注意力池化/padding+mask/JSON存盘/CPU-GPU自适应)。

设计要点
--------------
1. 频率与回看: 用 5 分钟 K 线 (bar5m), 单样本回看 SEQ_LEN=240 根 bar = 5 个交易日
   (每交易日 48 根 5m bar)。5 分钟兼顾日内微观结构与数据量, 5 日窗口覆盖短动量/反转,
   同时序列长度对 Transformer 友好。
2. 原始字段: 15 个原始量价/盘口字段 (≤100 约束), 不做衍生因子工程。
   - 价格类: open/high/low/close/bid_price1/ask_price1/bid_avg_price/ask_avg_price
   - 量额类: volume/amount/bid_volume1/ask_volume1/total_bid_volume/total_ask_volume/num_trades
3. 预处理 (仅允许的: 缺失填充/统一归一化/log/符号变换):
   - 量额列 log1p; 价格列按"每个窗口末日收盘"做尺度归一 (跨股可比, 端到端必备);
   - 再用训练集统计做全局标准化 (mean/std), 随权重存盘, 推理复用, 杜绝泄漏;
   - 缺失填 0; 窗口不足 SEQ_LEN 时左侧零填充 + mask (见下)。
4. 边界鲁棒性: 公榜注入表只含测试区间, 回看窗口在区间起始处无历史 -> 左侧 padding + mask,
   保证评估区间"每个交易日都不缺"。训练区间内的窗口通常为满窗, 极少数数据缺口同样 padding。
5. 模型: Linear 投影 + 可学习位置编码 + Pre-LN Transformer Encoder (4 层) +
   带掩码的注意力池化 + 回归头。可训练参数 ~90 万 (满足 [1e5, 1e8])。
6. 训练: 写死 2019-01~2023-10 训练 / 2023-11~2023-12 验证; 标签为次日收盘收益, **按日截面 rank 归一化到 [-1,1]**;
   MSE 损失, Adam + 余弦退火, 梯度裁剪; 每 epoch 记录验证集日均 RankIC (与评分指标对齐)。
7. 持久化: 权重 + 标准化统计 + 结构超参 一律存为 JSON 文本 (state_dict 转 {dtype,shape,data})。

平台约定: 公榜平台只加载本文件产出的 transformer_model.json 做推理 (见 transformer_predict.py);
私榜平台用 train_and_save 在隔离环境从零重训, 故训练逻辑须可复现 (固定 SEED)。
"""
import os
import json
import math
import time

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_HERE) == "__pycache__":            # 平台可能从 __pycache__ 加载, 回到源码目录
    _HERE = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ---------- 配置 (写死, 不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2019-01-01", "2023-10-31 23:59:59"   # 训练区间 (写死)
VAL_START,   VAL_END   = "2023-11-01", "2023-12-31 23:59:59"   # 验证区间 (监控过拟合)
SEQ_LEN = 240                  # 5 个交易日 × 48 根/日 (5 分钟 K 线)
BUFFER_DAYS = 25               # 取数前推日历日, 凑回看窗口 (5 交易日 ≈ 8 日历日, 留余量)
LR, SEED = 5e-4, 42
# CPU/GPU 自适应: 平台实测 device=cpu, 全量训练无法在 3h 内完成; 有 GPU 时用全量
_GPU = torch.cuda.is_available()
EPOCHS = 15 if _GPU else 5
BATCH = 512 if _GPU else 256
MAX_TRAIN_SAMPLES = 250000 if _GPU else 120000           # 训练样本上限 (控内存/时长)
PER_YEAR_CAP = 50000 if _GPU else 25000                  # 每个年度分片抽样上限
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

# 候选字段 (按优先级; 实际取"表里存在"的子集, 自动适配 bar5m 等与 bar1m 的 schema 差异)
# 平台实测: bar5m 无 bid_avg_price/ask_avg_price, 但有 bid/ask_price1-5 多档
CANDIDATE_PRICE_COLS = ["open", "high", "low", "close",
                        "bid_price1", "ask_price1", "bid_price2", "ask_price2",
                        "bid_price3", "ask_price3", "bid_price4", "ask_price4",
                        "bid_price5", "ask_price5",
                        "bid_avg_price", "ask_avg_price", "pre_close"]
CANDIDATE_VOL_COLS = ["volume", "amount",
                      "bid_volume1", "ask_volume1", "bid_volume2", "ask_volume2",
                      "bid_volume3", "ask_volume3", "bid_volume4", "ask_volume4",
                      "bid_volume5", "ask_volume5",
                      "total_bid_volume", "total_ask_volume", "num_trades"]

# 实际使用的字段 -- load_bars 首次调用时按表 schema 解析 (训练/推理同表则一致, 存盘校验)
FEATURE_COLS, PRICE_COLS, VOL_COLS = None, None, None
N_FEAT, CLOSE_IDX, PRICE_IDX = None, None, None

# 模型结构超参 (n_feat 在 resolve_features 后填入; 训练与推理必须一致, 一并存入权重文件)
MODEL_CFG = dict(d_model=128, nhead=4, nlayers=4,
                 dim_ff=512, seq_len=SEQ_LEN, dropout=DROPOUT)


def resolve_features(table):
    """查表的实际列, 从候选里取存在的子集 (幂等)。bar5m 与 bar1m schema 不同时自动适配。"""
    global FEATURE_COLS, PRICE_COLS, VOL_COLS, N_FEAT, CLOSE_IDX, PRICE_IDX
    if FEATURE_COLS is not None:
        return
    avail = set(dai.query(f"SELECT * FROM {table} LIMIT 1",
                          full_db_scan=True).df().columns)
    price = [c for c in CANDIDATE_PRICE_COLS if c in avail]
    vol = [c for c in CANDIDATE_VOL_COLS if c in avail]
    if "close" not in price:
        raise RuntimeError(f"表 {table} 缺少 close 列, 无法训练")
    feats = price + vol
    FEATURE_COLS, PRICE_COLS, VOL_COLS = feats, price, vol
    N_FEAT = len(feats)
    CLOSE_IDX = feats.index("close")
    PRICE_IDX = [feats.index(c) for c in price]
    logger.info("解析特征字段", table=table, n_feat=N_FEAT, price=price, vol=vol)


def set_features(feature_cols, price_cols, vol_cols):
    """推理时直接用 checkpoint 存的特征列 (跳过对注入表的 schema 探测, 更稳)。"""
    global FEATURE_COLS, PRICE_COLS, VOL_COLS, N_FEAT, CLOSE_IDX, PRICE_IDX
    FEATURE_COLS, PRICE_COLS, VOL_COLS = feature_cols, price_cols, vol_cols
    N_FEAT = len(feature_cols)
    CLOSE_IDX = feature_cols.index("close")
    PRICE_IDX = [feature_cols.index(c) for c in price_cols]

# 年度分片 (控单次取数内存; 每片自带前推 buffer)
YEAR_CHUNKS = [
    ("2019-01-01", "2019-12-31 23:59:59"),
    ("2020-01-01", "2020-12-31 23:59:59"),
    ("2021-01-01", "2021-12-31 23:59:59"),
    ("2022-01-01", "2022-12-31 23:59:59"),
    ("2023-01-01", "2023-10-31 23:59:59"),
]


# ---------- 模型 ----------
class AttnPool(nn.Module):
    """带 key_padding_mask 的单查询注意力池化: 把变长序列压成一个向量。"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, x, mask):                       # x:(B,L,D) mask:(B,L) 1=有效 0=填充
        kpm = (mask == 0)                              # True = 忽略
        q = self.query.expand(x.size(0), 1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=kpm, need_weights=False)
        return out.squeeze(1)                          # (B, D)


class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=128, nhead=4, nlayers=4,
                 dim_ff=512, seq_len=SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.pool = AttnPool(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):                        # x:(B,L,F) mask:(B,L)
        h = self.drop(self.proj(x) + self.pos)
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        p = self.pool(h, mask)
        return self.head(self.norm(p)).squeeze(-1)     # (B,)


# ---------- 数据 ----------
def pool(sd, ed):
    """区间内中证 1000 成分股代码。"""
    df = dai.query("SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
                   filters={"date": [sd, ed]}).df()
    return df["instrument"].tolist()


def load_bars(table, sd, ed, instruments):
    """取数 -> 量额列 log1p -> 缺失填 0 -> 按 instrument 分组为 {ins: {feats, dates}}。

    取数区间前推 BUFFER_DAYS 日历日, 以凑齐回看窗口 (训练分片间互相补历史;
    推理时注入表无前推数据则窗口自动左侧 padding, 由 build_windows 处理)。"""
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=BUFFER_DAYS)).strftime("%Y-%m-%d")
    resolve_features(table)                               # 按表 schema 解析可用字段 (幂等)
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": instruments},
                   compression=True).df()
    if df.empty:
        return {}
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))         # 量纲大, log1p
    df = df.replace([np.inf, -np.inf], np.nan)
    raw = {}
    for ins, sub in df.groupby("instrument", sort=False):
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        feats = np.nan_to_num(feats, nan=0.0)          # 缺失填 0
        dates = sub["date"].dt.normalize().to_numpy()  # 自然日 (datetime64)
        raw[ins] = {"feats": feats, "dates": dates}
    return raw


def build_windows(raw, sd, ed, mode, stats=None, max_samples=None, rng=None):
    """从 per-stock 原始序列切 (左填充) 窗口 + mask + 标签。

    mode='train' 返回 (X, mask, y, idx_df, stats); 'infer' 返回 (X, mask, idx_df, stats)。
    每个样本 = 某日 (决策点=当日最后一根 bar) 回看 SEQ_LEN 根 bar。
    - 价格列按窗口最后一根 close 做尺度归一; 量额列已在 load_bars 做 log1p。
    - 窗口不足 SEQ_LEN 时左侧零填充, mask 标 0 (公榜区间起始日/数据缺口均走此路径)。
    - 标签 = 次日 close 收益 (训练/验证需要; 推理不需要)。
    两段式: 先轻量扫描候选 (ins,pos,date,label), 抽样后再建窗口 -- 内存只随抽样规模增长,
    不随 universe×days 爆炸 (1932 成分股 × 全量窗口会 OOM)。
    stats 为 (mean, std), 训练集首次算出, 推理/后续分片复用。"""
    t0 = time.time()
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    # ---- Pass 1: 收集候选 (不建窗口数组) ----
    cands = []                                            # (ins, pos, date, label)
    for ins, data in raw.items():
        feats = data["feats"]
        dates = data["dates"]
        if len(feats) == 0:
            continue
        day = dates
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))  # 每日最后一根 bar
        close_px = feats[close_pos, CLOSE_IDX].astype(np.float64)         # 每日收盘 (原始)
        day_dates = day[close_pos]
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(day_dates[k])
            if d < sd_ts or d > ed_ts:
                continue
            ref = feats[p, CLOSE_IDX]
            if not np.isfinite(ref) or ref <= 0:
                continue
            label = None
            if k + 1 < len(close_pos) and close_px[k] > 0:
                r = close_px[k + 1] / close_px[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue
            cands.append((ins, p, d, label))
    if not cands:
        raise RuntimeError(f"build_windows 无样本 (mode={mode}, {sd}~{ed})")
    # ---- 抽样 (训练分片控内存; 验证/推理不抽) ----
    if max_samples is not None and rng is not None and len(cands) > max_samples:
        idx = rng.choice(len(cands), max_samples, replace=False)
        cands = [cands[i] for i in idx]
    # ---- Pass 2: 仅对候选建窗口 ----
    X_list, m_list, y_list, keys = [], [], [], []
    for ins, p, d, label in cands:
        feats = raw[ins]["feats"]
        start = max(0, p - SEQ_LEN + 1)
        window = feats[start:p + 1].copy()               # (L, N_FEAT), L<=SEQ_LEN
        ref = window[-1, CLOSE_IDX]
        window[:, PRICE_IDX] = window[:, PRICE_IDX] / ref       # 价格尺度归一 (跨股可比)
        L = window.shape[0]
        x = np.zeros((SEQ_LEN, N_FEAT), np.float32)
        m = np.zeros((SEQ_LEN,), np.float32)
        x[SEQ_LEN - L:] = window
        m[SEQ_LEN - L:] = 1.0                             # 左填充, 有效位在后段
        X_list.append(x)
        m_list.append(m)
        y_list.append(label if label is not None else np.float32(0.0))
        keys.append((d, ins))

    X = np.stack(X_list).astype(np.float32)              # (N, SEQ_LEN, N_FEAT)
    M = np.stack(m_list).astype(np.float32)              # (N, SEQ_LEN)

    # 全局标准化 (仅在有效位上算统计; 推理复用训练统计)
    if stats is None:
        valid = M.astype(bool)
        flat = X[valid]                                   # (有效位数, N_FEAT)
        stats = (flat.mean(0).astype(np.float32),
                 flat.std(0).astype(np.float32) + 1e-6)
    mean, std = stats
    X = ((X - mean) / std).astype(np.float32)
    X = X * M[:, :, None]                                 # 填充位归零, 避免扰动

    logger.info(f"{mode} 窗口构建完成", samples=len(keys),
                elapsed=round(time.time() - t0, 2))
    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])
    if mode == "train":
        y = np.array(y_list, np.float32)
        return X, M, y, idx_df, stats
    return X, M, idx_df, stats


def daily_rank_ic(pred, true, dates):
    """日均 Spearman RankIC (与评分指标对齐)。"""
    df = pd.DataFrame({"p": pred, "t": true, "d": dates})

    def _ic(g):
        if len(g) < 2:
            return np.nan
        return g["p"].corr(g["t"], method="spearman")

    return float(df.groupby("d").apply(_ic).mean())


def rank_normalize(y, dates):
    """按日截面 rank 归一化: 每个交易日内把收益转为 percentile rank ∈ [-1, 1]。

    与 RankIC 评分指标对齐: 让模型学习截面排序而非收益绝对值, 对尾部极端值更鲁棒。
    单日成分股不足 2 只时该日置 0 (不影响训练)。"""
    df = pd.DataFrame({"y": y, "d": dates})
    r = df.groupby("d")["y"].rank(pct=True)              # 0..1
    out = (r.to_numpy() * 2.0 - 1.0).astype(np.float32)  # -1..1
    out = np.where(np.isfinite(out), out, np.float32(0.0))
    return out


# ==== 模型存/读: 一律用文本类文件 (JSON), 不使用 .pt 等二进制 ====
def save_model(ckpt, model_path=MODEL_PATH):
    """checkpoint 存成 JSON 文本: state_dict 张量转 {dtype, shape, data(扁平 list)}。"""
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


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    """读取 save_model 写出的 JSON, 把 state_dict 还原为张量 dict。"""
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ==== 训练并持久化 ====
def train_and_save(datasources, model_path=MODEL_PATH):
    """在写死的训练区间上从零训练, 把 权重 + 标准化统计 + 结构超参 一并存盘。"""
    table = datasources["bar5m"]
    rng = np.random.RandomState(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=table)

    instruments = pool(TRAIN_START, VAL_END)
    logger.info("成分股数", n=len(instruments))

    # ---------- 构建训练集 (按年度分片取数 + build_windows 内抽样, 控内存) ----------
    stats = None
    X_parts, M_parts, y_parts, d_parts = [], [], [], []
    n_train_days = 0
    for (ys, ye) in YEAR_CHUNKS:
        t1 = time.time()
        raw = load_bars(table, ys, ye, instruments)
        Xc, Mc, yc, kdf, stats = build_windows(raw, ys, ye, "train", stats,
                                               max_samples=PER_YEAR_CAP, rng=rng)
        n_train_days += kdf["date"].nunique()
        dates_c = kdf["date"].to_numpy()
        X_parts.append(Xc)
        M_parts.append(Mc)
        y_parts.append(yc)
        d_parts.append(dates_c)
        del raw, Xc, Mc, kdf, dates_c
        logger.info("年度分片完成", chunk=f"{ys}~{ye}", kept=len(X_parts[-1]),
                    elapsed=round(time.time() - t1, 2))

    Xtr = np.concatenate(X_parts)
    Mtr = np.concatenate(M_parts)
    ytr = np.concatenate(y_parts)
    dtr = np.concatenate(d_parts)
    del X_parts, M_parts, y_parts, d_parts
    if len(Xtr) > MAX_TRAIN_SAMPLES:                     # 安全阀: 全局再抽 (一般不触发)
        idx = rng.choice(len(Xtr), MAX_TRAIN_SAMPLES, replace=False)
        Xtr, Mtr, ytr, dtr = Xtr[idx], Mtr[idx], ytr[idx], dtr[idx]
    ytr = rank_normalize(ytr, dtr)                        # 按日截面 rank 归一化 (替代 winsorize)
    logger.info("训练集就绪", samples=len(Xtr), days=n_train_days)

    # ---------- 构建验证集 (用同一套 stats 标准化; 不抽样) ----------
    raw_val = load_bars(table, VAL_START, VAL_END, instruments)
    Xval, Mval, yval, kdf_val, _ = build_windows(raw_val, VAL_START, VAL_END, "train", stats)
    yval = rank_normalize(yval, kdf_val["date"].to_numpy())  # 验证集同样 rank 归一化
    del raw_val
    logger.info("验证集就绪", samples=len(Xval))

    # ---------- 从零训练 ----------
    model_cfg = dict(MODEL_CFG, n_feat=N_FEAT)            # n_feat 在 resolve_features 后已知
    model = StockTransformer(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("可训练参数量", n_params=n_params)
    assert 1e5 <= n_params <= 1e8, f"参数量 {n_params} 越界 [1e5, 1e8]"

    ds = TensorDataset(
        torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        pin_memory=(device.type == "cuda"), drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    Xval_t = torch.from_numpy(Xval)
    Mval_t = torch.from_numpy(Mval)
    val_dates = kdf_val["date"].to_numpy()

    best_ic = -9.0
    for ep in range(EPOCHS):
        model.train()
        t, tot, nb = time.time(), 0.0, 0
        for xb, mb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        # 验证集 RankIC
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xval_t), BATCH):
                xb = Xval_t[i:i + BATCH].to(device)
                mb = Mval_t[i:i + BATCH].to(device)
                preds.append(model(xb, mb).cpu().numpy())
        pred_val = np.concatenate(preds)
        ic = daily_rank_ic(pred_val, yval, val_dates)
        logger.info("epoch 完成", epoch=ep + 1, mse=round(tot / max(nb, 1), 8),
                    val_rankic=round(ic, 4), lr=round(sched.get_last_lr()[0], 6),
                    elapsed=round(time.time() - t, 2))

        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    logger.info("训练结束", best_val_rankic=round(best_ic, 4))

    # ---------- 持久化: 权重 + 统计 + 结构超参 ----------
    mean, std = stats
    save_model({
        "state_dict": best_state,
        "model_cfg": model_cfg,
        "feature_cols": FEATURE_COLS,
        "price_cols": PRICE_COLS,
        "vol_cols": VOL_COLS,
        "seq_len": SEQ_LEN,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
    }, model_path)
    logger.info("模型已保存, 请随 notebook 一并上传", path=model_path)
    return model_path


if __name__ == "__main__":
    datasources = {"bar5m": "bigalpha_2026_stock_bar5m"}
    train_and_save(datasources)
