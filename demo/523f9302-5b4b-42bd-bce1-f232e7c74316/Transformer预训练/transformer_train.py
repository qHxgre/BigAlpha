# -*- coding: utf-8 -*-
"""Transformer 端到端 demo —— 训练侧脚本 (共享定义 + 从零训练并持久化)。

本文件承担两件事:
  1. 沉淀 **训练与推理共用** 的定义 (配置 / 模型结构 / 数据构建), 作为单一事实来源;
     配套 notebook 在推理时直接 `from transformer_train import ...` 复用, 避免两边漂移。
  2. 提供 `train_and_save(...)`: 在写死的训练区间上从零训练, 把
     **权重 + 标准化统计 + 结构超参** 一并保存到 `transformer_model.pt`。

用法 (参赛者本地运行一次, 产物随 notebook 一起上传):
    python transformer_train.py
或在其它脚本/notebook 中:
    from transformer_train import train_and_save
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})

公榜阶段平台不会重训, 直接加载该文件做推理 (见 notebook 的 `main`);
私榜阶段平台用 `train_and_save` 在隔离环境从零重训, 故训练逻辑需保持可复现 (固定随机种子)。
"""
import os
import time

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

# 训练好的模型保存路径; 参赛者本地训练后, 把该文件随 notebook 一并上传
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.pt")

# ---------- 配置 (写死, 不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"  # 训练区间写死, 切勿用平台注入的测试区间训练
SEQ_LEN = 64                  # 每条样本回看多少个 bar
EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42
MAX_TRAIN_INSTRUMENTS = 200   # demo 限制训练标的数控制时长, 正式可放开

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS   = ["volume", "amount", "bid_volume1", "ask_volume1"]  # 量纲大, 先 log1p
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

# 模型结构超参 (训练与推理必须一致, 会一并存入权重文件供推理端重建模型)
MODEL_CFG = dict(n_feat=N_FEAT, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN)


# ---------- 模型: 单条 Transformer 编码 -> 池化 -> 回归头 ----------
class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)                  # 每个 bar -> token 向量
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))  # 可学习位置编码
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, 0.1,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                                       # (B, L, N_FEAT) -> (B,)
        h = self.encoder(self.proj(x) + self.pos).mean(dim=1)
        return self.head(h).squeeze(-1)


def pool(sd, ed):
    """区间内中证 1000 成分股代码。"""
    df = dai.query("SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
                   filters={"date": [sd, ed]}).df()
    return df["instrument"].tolist()


# ---------- 数据: 直接用原始字段切窗口, 只做 量log + 标准化 ----------
def build_dataset(table, sd, ed, mode, instruments, stats=None):
    """切窗口并标准化 (训练与推理共用)。
    mode='train' 返回 (X, y, None, stats); 'infer' 返回 (X, None, idx_df, stats)。
    X 为 (N, SEQ_LEN, N_FEAT); stats 为 (mean, std), 训练集上算好, 推理复用。"""
    t0 = time.time()
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")  # 缓冲凑回看窗口
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": instruments}).df()
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))                   # 量纲大的字段先 log1p

    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    wins, ys, keys = [], [], []
    for ins, sub in df.groupby("instrument", sort=False):
        if len(sub) <= SEQ_LEN:
            continue
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()            # 1m bar 时间戳取自然日
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))  # 每日最后一根 bar
        close_px = sub["close"].to_numpy(np.float64)[close_pos]
        dates = day[close_pos]
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                continue                                        # 历史不足 或 落在缓冲区
            label = None
            if k + 1 < len(close_pos) and close_px[k] > 0:
                r = close_px[k + 1] / close_px[k] - 1.0         # 未来 1 日收益
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue                                        # 训练集需要标签
            wins.append(feats[p - SEQ_LEN + 1: p + 1])
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))
    if not keys:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")

    X = np.stack(wins).astype(np.float32)                       # (N, SEQ_LEN, N_FEAT)
    if stats is None:                                           # 训练集上算, 推理复用
        flat = X.reshape(-1, N_FEAT)
        stats = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32) + 1e-6)
    m, s = stats
    X = ((X - m) / s).astype(np.float32)                        # 按字段标准化
    logger.info(f"{mode} 集构建完成", samples=len(keys), elapsed=round(time.time() - t0, 2))
    if mode == "train":
        return X, np.array(ys, np.float32), None, stats
    return X, None, pd.DataFrame(keys, columns=["date", "instrument"]), stats


# ==== 训练并持久化 (参赛者本地运行一次, 产物随 notebook 一起上传) ====
def train_and_save(datasources, model_path=MODEL_PATH):
    """在写死的训练区间上从零训练, 把 权重 + 标准化统计 + 结构超参 一并存盘。

    公榜阶段平台不会重训, 直接加载该文件做推理; 私榜阶段平台用本函数从零重训,
    故训练逻辑需保持可复现 (固定随机种子)。"""
    table = datasources["bar1m"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=table)

    # ---------- 构建训练集 (写死训练区间) ----------
    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    Xtr, ytr, _, stats = build_dataset(
        table, TRAIN_START, TRAIN_END, "train",
        pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS])
    lo, hi = np.percentile(ytr, [1, 99])
    ytr = np.clip(ytr, lo, hi)                                  # winsorize 标签

    # ---------- 从零训练 ----------
    model = StockTransformer(**MODEL_CFG).to(device)
    logger.info("可训练参数量", n_params=sum(p.numel() for p in model.parameters()))
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True, pin_memory=(device.type == "cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(EPOCHS):
        t, tot, nb = time.time(), 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        logger.info("epoch 完成", epoch=ep + 1, mse=round(tot / max(nb, 1), 8),
                    elapsed=round(time.time() - t, 2))

    # ---------- 持久化: 权重 + 统计 + 结构超参 (推理端据此重建并复用) ----------
    mean, std = stats
    torch.save({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "mean": np.asarray(mean, np.float32).tolist(),         # 存为 list, 加载更稳健
        "std": np.asarray(std, np.float32).tolist(),
    }, model_path)
    logger.info("模型已保存, 请随 notebook 一并上传", path=model_path)
    return model_path


if __name__ == "__main__":
    # 本地训练入口: 在写死的训练区间上从零训练并保存 transformer_model.pt
    datasources = {"bar1m": "bigalpha_2026_stock_bar1m"}
    train_and_save(datasources)
