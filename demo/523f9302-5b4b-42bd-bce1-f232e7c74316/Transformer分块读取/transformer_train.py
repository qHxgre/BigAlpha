# -*- coding: utf-8 -*-
"""Transformer 端到端 demo —— 训练侧脚本 (分块读取版, 防止读数据时内存溢出)。

本文件与「Transformer预训练」目录下的同名脚本**模型结构完全一致**, 唯一区别是
**取数方式**: 原版在 build_dataset 里对全区间、全体成分股 (中证 1000 约 1200 只) 一次性
`dai.query(...).df()`, 把海量 1 分钟原始 K 线整体 materialize 到内存 —— 提交到平台跑
推理时经常 OOM。1 分钟 bar 每只股票每天约 240 根, 1200 只 × 一年 ≈ 上千万行 × 十几列,
而真正需要的样本 (每天收盘决策点的 64 根回看窗口) 体量小得多, 却被前面那一大坨原始数据压垮。

解决办法就一句话: **按 instrument 分批查询, 每批切完窗口立刻丢弃原始行**, 峰值内存
只占「一批股票的原始数据」而不是「全体」。为什么按股票分批而不是按日期分批?
因为回看窗口需要单只股票时间上连续, 按日期切会让窗口跨越边界而损坏; 按股票切则
每只股票的历史保持完整, 互不影响。

两条路径分别处理:
  * **训练** —— build_dataset 分批读取、逐批切窗口累积 (窗口本身很小, 累积无压力),
    原始 1m 数据每批用完即 `del` 释放; 标准化统计用流式累加器 (sum/sumsq/count) 在线计算,
    数值上与「全量算一次」等价, 却无需把全部原始数据同时留在内存。
  * **推理** —— predict_scores 分批读取, 每批切窗口→标准化→立即预测, 只保留
    (date, instrument, score) 三列结果, 窗口与原始数据当批释放。峰值内存与总区间长度无关,
    只与 CHUNK_SIZE 有关, 这正是提交后最需要省内存的路径。

模型一律存为 **文本类文件 (JSON)** (见 save_model / load_model), 与原版一致。

用法 (参赛者本地运行一次, 产物随 notebook 一起上传):
    python transformer_train.py
公榜阶段平台不重训, 直接加载该文件推理 (见 notebook 的 main);
私榜阶段平台用 train_and_save 从零重训, 故训练逻辑保持可复现 (固定随机种子)。
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

# 训练好的模型保存路径; 参赛者本地训练后, 把该文件随 notebook 一并上传
# 平台限制只能提交文本类文件, 故存为 JSON (而非 torch 的 .pt 二进制)
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ---------- 配置 (写死, 不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"  # 训练区间写死, 切勿用平台注入的测试区间训练
SEQ_LEN = 64                  # 每条样本回看多少个 bar
EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42
MAX_TRAIN_INSTRUMENTS = 200   # demo 限制训练标的数控制时长, 正式可放开
CHUNK_SIZE = 50               # ★分块核心: 每次只查这么多只股票的原始 1m 数据, 用完即释放

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS   = ["volume", "amount", "bid_volume1", "ask_volume1"]  # 量纲大, 先 log1p
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

# 模型结构超参 (训练与推理必须一致, 会一并存入权重文件供推理端重建模型)
MODEL_CFG = dict(n_feat=N_FEAT, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN)


# ---------- 模型: 单条 Transformer 编码 -> 池化 -> 回归头 (与原版完全一致) ----------
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


def _chunks(seq, size):
    """把股票列表切成每份 size 只的小块 (分块读取的基础)。"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------- 单只股票切窗口: 从一段连续的 1m 数据里, 抽出每个收盘决策点的回看窗口 ----------
def _windows_from_one(sub, sd_ts, ed_ts, need_label):
    """对单只股票已排好序的 1m 数据切窗口。

    sub 为该股票在 [buf, ed] 内、按 date 升序的 DataFrame (量纲大的列已 log1p)。
    返回 (wins, ys, keys) 三个 list: 窗口 (SEQ_LEN, N_FEAT) / 标签 / (date, instrument)。
    与原版 build_dataset 内层逻辑逐行一致, 只是抽成函数以便按股票分批复用。"""
    wins, ys, keys = [], [], []
    if len(sub) <= SEQ_LEN:
        return wins, ys, keys
    ins = sub["instrument"].iloc[0]
    feats = sub[FEATURE_COLS].to_numpy(np.float32)
    day = sub["date"].dt.normalize().to_numpy()                # 1m bar 时间戳取自然日
    close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))  # 每日最后一根 bar
    close_px = sub["close"].to_numpy(np.float64)[close_pos]
    dates = day[close_pos]
    for k, p in enumerate(close_pos):
        d = pd.Timestamp(dates[k])
        if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
            continue                                           # 历史不足 或 落在缓冲区
        label = None
        if k + 1 < len(close_pos) and close_px[k] > 0:
            r = close_px[k + 1] / close_px[k] - 1.0            # 未来 1 日收益
            if np.isfinite(r):
                label = np.float32(r)
        if need_label and label is None:
            continue                                           # 训练集需要标签
        wins.append(feats[p - SEQ_LEN + 1: p + 1])
        ys.append(label if label is not None else np.float32(0.0))
        keys.append((d, ins))
    return wins, ys, keys


def _query_chunk(table, buf, ed, chunk):
    """查一小批股票的原始 1m 数据 (只有这一步会把原始行放进内存)。"""
    sql = (f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
           f"FROM {table} ORDER BY instrument, date")
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": chunk}).df()
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))                  # 量纲大的字段先 log1p
    return df


class _RunningStats:
    """流式标准化统计: 按字段累加 sum / sumsq / count, 最后一次算出 mean/std。

    这样每批窗口用完即可丢弃, 无需把全部训练窗口同时留在内存, 而结果与
    「全量堆到一起算一次 mean/std」在数值上等价。"""
    def __init__(self, n_feat):
        self.n = 0
        self.s = np.zeros(n_feat, np.float64)
        self.ss = np.zeros(n_feat, np.float64)

    def update(self, x):                                       # x: (m, N_FEAT)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x.astype(np.float64) ** 2).sum(0)

    def finalize(self):
        mean = (self.s / self.n)
        var = self.ss / self.n - mean ** 2
        std = np.sqrt(np.clip(var, 0, None)) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)


# ---------- 训练集: 分块读取 + 逐批切窗口累积 + 流式统计 ----------
def build_train_dataset(table, sd, ed, instruments):
    """分块构建训练集, 峰值内存只占「一批股票的原始 1m 数据」。

    窗口本身体量小 (每股每天 1 条 × 64×N_FEAT), 累积到内存无压力; 真正吃内存的是
    原始 1m 行, 故每批查完、切完窗口就 `del df` 释放。标准化统计用 _RunningStats
    在线累加, 避免为算 mean/std 把所有窗口再堆一次。
    返回 (X, y, stats): X 已标准化, stats=(mean, std) 供推理复用。"""
    t0 = time.time()
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")  # 缓冲凑回看窗口
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    all_wins, all_ys = [], []
    rs = _RunningStats(N_FEAT)
    n_chunk = (len(instruments) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci, chunk in enumerate(_chunks(instruments, CHUNK_SIZE)):
        df = _query_chunk(table, buf, ed, chunk)               # 只查这一小批
        for _, sub in df.groupby("instrument", sort=False):
            wins, ys, _ = _windows_from_one(sub, sd_ts, ed_ts, need_label=True)
            for w in wins:
                rs.update(w)                                   # 流式累加原始 (未标准化) 窗口
            all_wins.extend(wins)
            all_ys.extend(ys)
        del df                                                 # ★关键: 原始行用完立即释放
        logger.info("训练分块完成", chunk=f"{ci + 1}/{n_chunk}", cum_samples=len(all_wins))
    if not all_wins:
        raise RuntimeError(f"build_train_dataset 无样本 ({sd}~{ed})")

    X = np.stack(all_wins).astype(np.float32)                  # (N, SEQ_LEN, N_FEAT)
    y = np.array(all_ys, np.float32)
    mean, std = rs.finalize()                                  # 与全量算一次等价
    X = ((X - mean) / std).astype(np.float32)                  # 按字段标准化
    logger.info("训练集构建完成", samples=len(X), chunks=n_chunk,
                elapsed=round(time.time() - t0, 2))
    return X, y, (mean, std)


# ---------- 推理: 分块读取 + 每批即预测, 只留 (date, instrument, score) ----------
def predict_scores(model, table, sd, ed, instruments, stats, device):
    """分块推理: 每批股票 查数→切窗口→标准化→预测, 只累积三列结果。

    这是提交后最需要省内存的路径。峰值内存与总区间长度、股票总数无关, 只与
    CHUNK_SIZE 有关 —— 因为每批的窗口和原始数据当批就释放, 内存里始终只留下
    已经算好的、极小的 (date, instrument, score) 结果。"""
    mean, std = stats
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    out = []
    n_chunk = (len(instruments) + CHUNK_SIZE - 1) // CHUNK_SIZE
    model.eval()
    for ci, chunk in enumerate(_chunks(instruments, CHUNK_SIZE)):
        df = _query_chunk(table, buf, ed, chunk)
        wins, keys = [], []
        for _, sub in df.groupby("instrument", sort=False):
            w, _, k = _windows_from_one(sub, sd_ts, ed_ts, need_label=False)
            wins.extend(w)
            keys.extend(k)
        del df                                                 # ★原始行用完立即释放
        if not wins:
            continue
        X = np.stack(wins).astype(np.float32)
        X = ((X - mean) / std).astype(np.float32)
        Xt = torch.from_numpy(X)
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xt), BATCH):
                xb = Xt[i:i + BATCH].to(device)
                preds.append(model(xb).cpu().numpy())
        sc = np.concatenate(preds).astype(np.float64)
        chunk_df = pd.DataFrame(keys, columns=["date", "instrument"])
        chunk_df["score"] = sc
        out.append(chunk_df)                                   # 只留三列结果
        del wins, keys, X, Xt, preds                           # 窗口/张量当批释放
        logger.info("推理分块完成", chunk=f"{ci + 1}/{n_chunk}", cum_rows=sum(len(o) for o in out))
    if not out:
        raise RuntimeError(f"predict_scores 无样本 ({sd}~{ed})")
    return pd.concat(out, ignore_index=True)


# ==== 模型存/读: 一律用文本类文件 (JSON), 不使用 .pt 等二进制 (与原版一致) ====
def save_model(ckpt, model_path=MODEL_PATH):
    """把 checkpoint 存成 JSON 文本文件。

    state_dict 里每个张量转成 {dtype, shape, data(扁平 list)}, 其余字段原样写入;
    加载时用 load_model 按 dtype/shape 还原。"""
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),   # 如 'float32'
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),                # 扁平存, 加载时按 shape 还原
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


# ==== 训练并持久化 (参赛者本地运行一次, 产物随 notebook 一起上传) ====
def train_and_save(datasources, model_path=MODEL_PATH):
    """在写死的训练区间上从零训练, 把 权重 + 标准化统计 + 结构超参 一并存盘。

    唯一区别是训练集用 build_train_dataset 分块构建, 避免读数据时 OOM。"""
    table = datasources["bar1m"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=table, chunk_size=CHUNK_SIZE)

    # ---------- 构建训练集 (写死训练区间, 分块读取) ----------
    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    Xtr, ytr, stats = build_train_dataset(
        table, TRAIN_START, TRAIN_END,
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


if __name__ == "__main__":
    # 本地训练入口: 在写死的训练区间上从零训练并保存 transformer_model.json
    datasources = {"bar1m": "bigalpha_2026_stock_bar1m"}
    train_and_save(datasources)
