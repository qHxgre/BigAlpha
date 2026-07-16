# -*- coding: utf-8 -*-
"""Transformer 端到端 demo (本地数据版) —— 用下载解压的本地行情从零训练并持久化。

与 demo 里的 `transformer_train.py` 相比, 只有 **数据来源** 不同:
  - 训练阶段: 直接读取参赛者下载解压到本地的压缩行情 (feather 分区文件),
    不再走 BigQuant SDK / dai.query, 从根本上规避下载流量与 quota 限制。
  - 推理阶段 (平台公榜 / 私榜, 见 main): 仍用平台注入的 datasources 通过 dai
    读取 **云端未压缩的原始数据**。

本地表 (e2e, 已压缩) 与 云端表 (stock, 原始) 的差异, 全部收敛到 `to_canonical`
一处消化: 价格/金额 分->元、OHLC 缺失 -1->NaN、instrument_id->key、盘口只留 3 档。
两边先归一到同一 canonical 表示, 再走同一套 `build_windows`, 从而杜绝
"本地训练分高、云端预测对不上"。

模型一律存为文本类文件 (JSON, 见 save_model/load_model), 不用 .pt 二进制。

用法:
  # 1) 本地: 用下载解压的数据从零训练并保存 transformer_model.json
  E2E_DATA_ROOT=/path/to/e2e_data python transformer_train_local.py
  # 2) 提交: 把本文件 + transformer_model.json 随推理 notebook 一并上传;
  #    平台公榜阶段加载权重、在云端原始数据上调用 main() 推理打分。
"""
import os
import glob
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# 下载解压后的本地数据根目录: 每个频率一个子目录 (即 datasource_id),
# 目录内是按月分区的 feather 文件 (YYYYMM.0.feather) + bdb 元数据 bmeta.bin。
# 改成你本机解压后的实际路径, 或用环境变量 E2E_DATA_ROOT 覆盖。
LOCAL_DATA_ROOT = os.environ.get("E2E_DATA_ROOT", "./e2e_data")

# 本地训练用哪个频率 (对应解压出来的目录名); 云端 main 用平台注入的 datasources。
LOCAL_TABLE = "bigalpha_2026_e2e_bar5m"

# ---------- 配置 (写死, 不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"  # 训练区间写死, 切勿用平台注入的测试区间训练
SEQ_LEN = 48                  # 每条样本回看多少个 bar (5 分钟频率, 一个交易日 48 根)
EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42
MAX_TRAIN_INSTRUMENTS = 200   # demo 限制训练标的数控制时长, 正式可放开

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS   = ["volume", "amount", "bid_volume1", "ask_volume1"]  # 量纲大, 先 log1p
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

# 存储压缩约定 (本地 e2e 表): 价格/金额以"分"(元×100)的整数存储, 读取时 /100 还原;
# OHLC 缺失以 -1 表示 (盘口价缺失沿用 0)。云端原始表则是"元"的 float、NaN 缺失。
PRICE_SCALE = 100.0
SCALE_FIELDS = ["open", "high", "low", "close", "amount",
                "ask_price1", "ask_price2", "ask_price3",
                "bid_price1", "bid_price2", "bid_price3"]
OHLC_COLS = ["open", "high", "low", "close"]

# 模型结构超参 (训练与推理必须一致, 会一并存入权重文件供推理端重建模型)
MODEL_CFG = dict(n_feat=N_FEAT, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN)


# ---------- 模型: 单条 Transformer 编码 -> 池化 -> 回归头 ----------
class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)                     # 每个 bar -> token 向量
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))  # 可学习位置编码
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, 0.1,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                                          # (B, L, N_FEAT) -> (B,)
        h = self.encoder(self.proj(x) + self.pos).mean(dim=1)
        return self.head(h).squeeze(-1)


# ---------- 数据统一: 把本地表 / 云端表归一到同一 canonical 表示 ----------
def to_canonical(df: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    """把 本地 e2e 表(压缩) 或 云端 stock 表(原始) 统一成同一份表示, 供后续特征复用。

    统一后约定: 价格/金额单位为"元"(float), OHLC 缺失为 NaN, 盘口只保留 3 档,
    并有一列 `key` 作为标的键 (本地=instrument_id 整数, 云端=instrument 字符串)。
    build_windows 只认这份 canonical, 从而两阶段特征口径严格一致。"""
    df = df.copy()
    if is_local:
        # 1) OHLC 的 -1 (停牌/无成交) 视为缺失
        for c in OHLC_COLS:
            df.loc[df[c] == -1, c] = np.nan
        # 2) "分" -> "元" (价格/金额 /100 还原为浮点)
        for c in SCALE_FIELDS:
            if c in df.columns:
                df[c] = df[c].astype("float64") / PRICE_SCALE
        # 3) 标的键: 本地只有整数 instrument_id, 直接作为分组键
        df["key"] = df["instrument_id"]
    else:
        # 云端原始表: 价格已是"元"、缺失已是 NaN; 只需丢掉 4/5 档使列与本地对齐
        drop_cols = [c for c in df.columns
                     if any(c.startswith(p) and c[-1] in "45"
                            for p in ("ask_price", "bid_price",
                                      "ask_volume", "bid_volume",
                                      "ask_num_orders", "bid_num_orders"))]
        df = df.drop(columns=drop_cols, errors="ignore")
        df["key"] = df["instrument"]
    return df


# ---------- 本地读取: 直接读解压后的 feather 分区, 不走 SDK ----------
def read_local(table: str, sd: str, ed: str, columns: list[str]) -> pd.DataFrame:
    """从本地解压目录读取一个频率表在 [sd, ed] 区间的数据。

    数据按月分区存为 `<root>/<table>/YYYYMM.0.feather`; 我们先按文件名的 YYYYMM
    做粗筛 (只读区间覆盖到的月份, 避免全量 IO), 再按精确时间戳裁剪。
    这里带出 instrument_id 供 to_canonical 生成分组键; bmeta.bin 是 bdb 元数据,
    纯 feather 读取用不到, 跳过即可。"""
    root = os.path.join(LOCAL_DATA_ROOT, table)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"未找到本地数据目录 {root}; 请先下载并解压对应频率的 zip, "
            f"或用环境变量 E2E_DATA_ROOT 指向解压根目录")

    lo, hi = int(pd.Timestamp(sd).strftime("%Y%m")), int(pd.Timestamp(ed).strftime("%Y%m"))
    need = set(columns) | {"date", "instrument_id"}
    parts = []
    for fp in sorted(glob.glob(os.path.join(root, "*.feather"))):
        stem = os.path.basename(fp).split(".")[0]            # 'YYYYMM'
        if not (stem.isdigit() and lo <= int(stem) <= hi):
            continue
        sub = pd.read_feather(fp, columns=[c for c in need])  # 只读需要的列
        parts.append(sub)
    if not parts:
        raise RuntimeError(f"本地区间 {sd}~{ed} 无 feather 分区命中 (table={table})")

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= pd.Timestamp(sd)) & (df["date"] <= pd.Timestamp(ed))]


# ---------- 特征/切窗口: 只认 canonical, 训练与推理共用 ----------
def build_windows(df_canon: pd.DataFrame, sd, ed, mode, stats=None):
    """从 canonical 表切滑动窗口并标准化 (本地/云端两条路径都走这里)。

    mode='train' 返回 (X, y, None, stats); 'infer' 返回 (X, None, idx_df, stats)。
    X 为 (N, SEQ_LEN, N_FEAT); stats=(mean,std) 训练集上算好, 推理复用。
    标签: 每个交易日最后一根 bar 的 收盘价 -> 未来 1 日收益。"""
    t0 = time.time()
    df = df_canon.copy()
    # 量纲大的字段先 log1p (canonical 已把量统一, OHLC 的 NaN 不参与 log)
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))
    # OHLC 缺失前向填充, 让停牌 bar 不至于打断窗口 (首值仍可能 NaN, 下面丢弃)
    df = df.sort_values(["key", "date"])
    for c in OHLC_COLS:
        df[c] = df.groupby("key")[c].ffill()

    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    wins, ys, keys = [], [], []
    for k, sub in df.groupby("key", sort=False):
        if len(sub) <= SEQ_LEN:
            continue
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))  # 每日最后一根 bar
        close_px = sub["close"].to_numpy(np.float64)[close_pos]
        dates = day[close_pos]
        for j, p in enumerate(close_pos):
            d = pd.Timestamp(dates[j])
            if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                continue                                    # 历史不足 或 落在缓冲区
            win = feats[p - SEQ_LEN + 1: p + 1]
            if not np.isfinite(win).all():
                continue                                    # 窗口内仍有缺失, 跳过
            label = None
            if j + 1 < len(close_pos) and close_px[j] > 0:
                r = close_px[j + 1] / close_px[j] - 1.0     # 未来 1 日收益
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue                                    # 训练集需要标签
            wins.append(win)
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, k))
    if not keys:
        raise RuntimeError(f"build_windows 无样本 (mode={mode}, {sd}~{ed})")

    X = np.stack(wins).astype(np.float32)                   # (N, SEQ_LEN, N_FEAT)
    if stats is None:                                       # 训练集上算, 推理复用
        flat = X.reshape(-1, N_FEAT)
        stats = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32) + 1e-6)
    m, s = stats
    X = ((X - m) / s).astype(np.float32)                    # 按字段标准化
    logger.info(f"{mode} 集构建完成", samples=len(keys), elapsed=round(time.time() - t0, 2))
    if mode == "train":
        return X, np.array(ys, np.float32), None, stats
    idx_df = pd.DataFrame(keys, columns=["date", "key"])
    return X, None, idx_df, stats


# ==== 模型存/读: 一律用文本类文件 (JSON), 不使用 .pt 等二进制 ====
def save_model(ckpt, model_path=MODEL_PATH):
    """把 checkpoint 存成 JSON: state_dict 每个张量转 {dtype, shape, data(扁平)},
    其余字段 (结构超参 / 标准化统计) 原样写入, 加载时用 load_model 还原。"""
    tensors = {}
    for k, v in ckpt["state_dict"].items():
        t = v.detach().cpu()
        tensors[k] = {"dtype": str(t.dtype).replace("torch.", ""),
                      "shape": list(t.shape), "data": t.reshape(-1).tolist()}
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    """读回 save_model 写出的 JSON, 把 state_dict 还原为 {name: Tensor}。"""
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ==== 训练并持久化 (参赛者本地运行一次, 读本地 feather, 产物随 notebook 上传) ====
def train_and_save(model_path=MODEL_PATH):
    """读本地解压数据, 在写死的训练区间上从零训练, 存 权重 + 标准化统计 + 结构超参。"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=LOCAL_TABLE, root=LOCAL_DATA_ROOT)

    # ---------- 读本地 -> canonical -> 切窗口 ----------
    buf = (pd.to_datetime(TRAIN_START) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")  # 缓冲凑回看窗口
    raw = read_local(LOCAL_TABLE, buf, TRAIN_END, FEATURE_COLS)
    canon = to_canonical(raw, is_local=True)
    # demo 限制训练标的数控制时长 (取样本量最多的前若干只), 正式可放开
    top = canon["key"].value_counts().index[:MAX_TRAIN_INSTRUMENTS]
    canon = canon[canon["key"].isin(top)]

    Xtr, ytr, _, stats = build_windows(canon, TRAIN_START, TRAIN_END, "train")
    lo, hi = np.percentile(ytr, [1, 99])
    ytr = np.clip(ytr, lo, hi)                              # winsorize 标签

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
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        logger.info("epoch 完成", epoch=ep + 1, mse=round(tot / max(nb, 1), 8),
                    elapsed=round(time.time() - t, 2))

    # ---------- 持久化: 权重 + 统计 + 结构超参 ----------
    mean, std = stats
    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
    }, model_path)
    logger.info("模型已保存, 请随 notebook 一并上传", path=model_path)
    return model_path


# ==== 云端推理 (平台公榜阶段调用, 读云端原始数据, 只替换 datasources/日期) ====
def main(datasources, start_date, end_date):
    """加载本地训练好的权重, 在平台注入的测试区间上推理打分。

    注意: 这里读的是**云端未压缩的原始数据** (dai.query), 经同一个 to_canonical
    归一到与本地训练一致的表示后, 复用同一套 build_windows + 训练集算好的 mean/std,
    保证 train/infer 口径严格一致。输出 ['date','instrument','score']。"""
    import dai                                              # 云端环境才有, 本地训练不依赖

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型文件 {MODEL_PATH}; 请先本地运行 train_and_save() 训练并随 notebook 上传")

    ckpt = load_model(MODEL_PATH, map_location=device)
    stats = (np.asarray(ckpt["mean"], np.float32), np.asarray(ckpt["std"], np.float32))
    model = StockTransformer(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("已加载模型", path=MODEL_PATH, device=str(device))

    # 云端表: 频率与本地训练所用一致 (键名去掉 e2e_ 前缀由平台注入)
    table = datasources.get("bar5m") or next(iter(datasources.values()))
    buf = (pd.to_datetime(start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    sql = (f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
           f"FROM {table} ORDER BY instrument, date")
    raw = dai.query(sql, filters={"date": [buf, str(end_date)]}).df()
    canon = to_canonical(raw, is_local=False)               # 云端 -> 同一 canonical

    Xte, _, idx_df, _ = build_windows(canon, start_date, end_date, "infer", stats)
    preds, Xte_t = [], torch.from_numpy(Xte)
    with torch.no_grad():
        for i in range(0, len(idx_df), BATCH):
            preds.append(model(Xte_t[i:i + BATCH].to(device)).cpu().numpy())
    idx_df["score"] = np.concatenate(preds).astype(np.float64)
    # 云端 canonical 的 key 即字符串 instrument
    return (idx_df.rename(columns={"key": "instrument"})[["date", "instrument", "score"]]
                  .sort_values(["date", "instrument"]).reset_index(drop=True))


if __name__ == "__main__":
    # 本地训练入口: 读解压后的 feather, 从零训练并保存 transformer_model.json
    train_and_save()
