# -*- coding: utf-8 -*-
"""Transformer 端到端 demo —— 训练侧脚本。

本文件负责：
1. 定义训练和推理共用的配置、模型结构、数据构建函数；
2. 从零训练 Transformer；
3. 保存模型权重、标准化统计、模型结构超参到 transformer_model.json。

本版本的输入序列改为：
    每只股票每天的【早盘前 32 根 1分钟bar】 + 【尾盘最后 32 根 1分钟bar】

模型结构和原版保持不变：
    SEQ_LEN = 64
    d_model = 64
    nhead = 4
    nlayers = 2
    dim_ff = 128
"""

import os
import json
import time

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

# ========== 模型保存路径 ==========
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ========== 配置：保持原模型超参数不变 ==========
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"

SEQ_LEN = 64
MORNING_LEN = 32
TAIL_LEN = 32
assert MORNING_LEN + TAIL_LEN == SEQ_LEN

EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42
MAX_TRAIN_INSTRUMENTS = 200

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS = ["volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

# 模型结构超参保持不变
MODEL_CFG = dict(
    n_feat=N_FEAT,
    d_model=64,
    nhead=4,
    nlayers=2,
    dim_ff=128,
    seq_len=SEQ_LEN,
)


# ========== 模型结构：不改 ==========
class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN):
        super().__init__()

        # 每根 1分钟bar 的 10个字段，映射成 d_model 维 token
        self.proj = nn.Linear(n_feat, d_model)

        # 可学习位置编码，长度仍然是 64
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer Encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)

        # 回归头：输出一个预测分数
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        """
        x: (B, 64, 10)

        64 的含义现在变成：
            前 32 个 token：早盘前 32 根 1分钟bar
            后 32 个 token：尾盘最后 32 根 1分钟bar
        """
        h = self.encoder(self.proj(x) + self.pos).mean(dim=1)
        return self.head(h).squeeze(-1)


def pool(sd, ed):
    """获取区间内中证1000成分股代码。"""
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return df["instrument"].tolist()


# ========== 数据构建：核心改动在这里 ==========
def build_dataset(table, sd, ed, mode, instruments, stats=None):
    """构建训练集或推理集。

    原版逻辑：
        每天取收盘前 64 根 1分钟bar。

    当前版本：
        每天取早盘前 32 根 1分钟bar + 尾盘最后 32 根 1分钟bar。

    返回：
        mode='train':
            X, y, None, stats

        mode='infer':
            X, None, idx_df, stats

    X shape:
        (N, 64, 10)

    其中 64 个 token 的含义：
        token 0~31：当天早盘前 32 根 1分钟bar
        token 32~63：当天尾盘最后 32 根 1分钟bar
    """
    t0 = time.time()

    sql = f"""
        SELECT date, instrument, {', '.join(FEATURE_COLS)}
        FROM {table}
        ORDER BY instrument, date
    """

    df = dai.query(
        sql,
        filters={"date": [sd, ed], "instrument": instruments},
    ).df()

    if df.empty:
        raise RuntimeError(f"查询结果为空：mode={mode}, {sd}~{ed}")

    # 成交量、成交额、盘口量做 log1p，避免量纲过大
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))

    sd_day = pd.to_datetime(sd).normalize()
    ed_day = pd.to_datetime(ed).normalize()

    wins, ys, keys = [], [], []

    for ins, sub in df.groupby("instrument", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)

        if len(sub) < SEQ_LEN:
            continue

        feats = sub[FEATURE_COLS].to_numpy(np.float32)

        # 每根分钟bar对应的自然日
        day = sub["date"].dt.normalize().to_numpy()

        # 找到每个交易日的起始位置和结束位置
        # starts: 每天第一根bar的位置
        # ends:   每天最后一根bar的位置
        day_change = np.r_[True, day[1:] != day[:-1]]
        starts = np.flatnonzero(day_change)
        ends = np.r_[starts[1:] - 1, len(sub) - 1]

        # 每日收盘价，用于构造下一交易日收益标签
        close_px = sub["close"].to_numpy(np.float64)[ends]
        dates = day[ends]

        for k, (s_pos, e_pos) in enumerate(zip(starts, ends)):
            d = pd.Timestamp(dates[k])

            # 只保留目标区间内的日期
            if d < sd_day or d > ed_day:
                continue

            # 当天分钟bar数量
            n_bars_today = e_pos - s_pos + 1

            # 必须至少有 64 根bar，才能取早盘32 + 尾盘32
            if n_bars_today < SEQ_LEN:
                continue

            # 早盘前32根bar
            morning_part = feats[s_pos: s_pos + MORNING_LEN]

            # 尾盘最后32根bar
            tail_part = feats[e_pos - TAIL_LEN + 1: e_pos + 1]

            if len(morning_part) != MORNING_LEN or len(tail_part) != TAIL_LEN:
                continue

            # 拼成一个长度为64的序列
            # 前32根是早盘，后32根是尾盘
            window = np.concatenate([morning_part, tail_part], axis=0)

            if window.shape != (SEQ_LEN, N_FEAT):
                continue

            # 标签：下一交易日收盘价 / 当日收盘价 - 1
            label = None
            if k + 1 < len(ends) and close_px[k] > 0:
                r = close_px[k + 1] / close_px[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)

            # 训练集必须有标签
            if mode == "train" and label is None:
                continue

            wins.append(window)
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))

    if not keys:
        raise RuntimeError(f"build_dataset 无样本：mode={mode}, {sd}~{ed}")

    X = np.stack(wins).astype(np.float32)

    # 标准化统计只在训练集上计算
    # 推理阶段复用训练集 mean/std
    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32) + 1e-6
        stats = (mean, std)

    mean, std = stats
    X = ((X - mean) / std).astype(np.float32)

    logger.info(
        f"{mode} 集构建完成",
        samples=len(keys),
        elapsed=round(time.time() - t0, 2),
        input_format="morning32_tail32",
    )

    if mode == "train":
        return X, np.array(ys, np.float32), None, stats

    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])
    return X, None, idx_df, stats


# ========== 模型保存：JSON 文本格式 ==========
def save_model(ckpt, model_path=MODEL_PATH):
    """保存模型为 JSON 文本文件。

    保存内容包括：
        state_dict：模型权重
        model_cfg：模型结构配置
        feature_cols：输入字段
        seq_len：序列长度
        mean/std：训练集标准化参数
    """
    state_dict = ckpt["state_dict"]
    tensors = {}

    for k, v in state_dict.items():
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
    """读取 JSON 模型文件，并还原 state_dict。"""
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state_dict = {}

    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        state_dict[k] = t.reshape(meta["shape"]).to(map_location)

    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = state_dict

    return ckpt


# ========== 训练并保存 ==========
def train_and_save(datasources, model_path=MODEL_PATH):
    """从零训练模型，并保存到 transformer_model.json。"""
    table = datasources["bar1m"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=table)

    # 构建训练集
    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)

    Xtr, ytr, _, stats = build_dataset(
        table=table,
        sd=TRAIN_START,
        ed=TRAIN_END,
        mode="train",
        instruments=pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS],
        stats=None,
    )

    # 标签缩尾，防止极端收益影响训练
    lo, hi = np.percentile(ytr, [1, 99])
    ytr = np.clip(ytr, lo, hi)

    # 创建模型
    model = StockTransformer(**MODEL_CFG).to(device)

    logger.info(
        "可训练参数量",
        n_params=sum(p.numel() for p in model.parameters()),
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=BATCH,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()

    for ep in range(EPOCHS):
        t0 = time.time()
        total_loss = 0.0
        nb = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()

            # 防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            total_loss += loss.item()
            nb += 1

        logger.info(
            "epoch 完成",
            epoch=ep + 1,
            mse=round(total_loss / max(nb, 1), 8),
            elapsed=round(time.time() - t0, 2),
        )

    # 保存训练结果
    mean, std = stats

    save_model(
        {
            "state_dict": model.state_dict(),
            "model_cfg": MODEL_CFG,
            "feature_cols": FEATURE_COLS,
            "seq_len": SEQ_LEN,
            "morning_len": MORNING_LEN,
            "tail_len": TAIL_LEN,
            "input_format": "morning32_tail32",
            "mean": np.asarray(mean, np.float32).tolist(),
            "std": np.asarray(std, np.float32).tolist(),
        },
        model_path=model_path,
    )

    logger.info("模型已保存，请随 notebook 一并上传", path=model_path)

    return model_path


if __name__ == "__main__":
    datasources = {
        "bar1m": "bigalpha_2026_stock_bar1m",
    }

    train_and_save(datasources)
