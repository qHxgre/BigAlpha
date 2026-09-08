# -*- coding: utf-8 -*-

# %%
"""Transformer 端到端 demo —— 训练侧脚本 (共享定义 + 从零训练并持久化)。

本文件承担两件事:
  1. 沉淀 **训练与推理共用** 的定义 (配置 / 模型结构 / 数据构建), 作为单一事实来源;
     配套 notebook 在推理时直接 `from transformer_train import ...` 复用, 避免两边漂移。
  2. 提供 `train_and_save(...)`: 在写死的训练区间上从零训练, 把
     **权重 + 标准化统计 + 结构超参** 一并保存到 `transformer_model.json` (纯文本)。

模型一律存为 **文本类文件 (JSON)**, 不使用 `.pt` 等二进制格式: state_dict 里的张量
会被转成 {dtype, shape, data(扁平 list)} 结构, 加载时按 dtype/shape 还原, 便于版本
管理、人工查阅与跨环境传输 (见 `save_model` / `load_model`)。

数据源切换: 通过 USE_LOCAL_DATA 控制
  - USE_LOCAL_DATA=False (默认): 使用 dai 平台查询
  - USE_LOCAL_DATA=True: 使用本地 Parquet 分区文件 (Hive 格式)

用法 (参赛者本地运行一次, 产物随 notebook 一起上传):
    python transformer_train.py
或在其它脚本/notebook 中:
    from transformer_train import train_and_save
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})

公榜阶段平台不会重训, 直接加载该文件做推理 (见 notebook 的 `main`);
私榜阶段平台用 `train_and_save` 在隔离环境从零重训, 故训练逻辑需保持可复现 (固定随机种子)。
"""
import os
import json
import time
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
import structlog
import gc

logger = structlog.get_logger()

# ---------- 数据源切换: False=dai 平台, True=本地 Parquet ----------
USE_LOCAL_DATA = False

# 训练好的模型保存路径; 参赛者本地训练后, 把该文件随 notebook 一并上传
# 平台限制只能提交文本类文件, 故存为 JSON (而非 torch 的 .pt 二进制)
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ---------- 配置 (写死, 不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"  # 训练区间写死, 切勿用平台注入的测试区间训练
SEQ_LEN = 64                  # 每条样本回看多少个 bar
EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42

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


# ---------- 本地数据读取工具函数 (仅 USE_LOCAL_DATA=True 时使用) ----------
if USE_LOCAL_DATA:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    DATA_DIR = os.path.join(_HERE, "..", "..", "data")

    def _read_local_data(sd, ed, instruments=None):
        """读取本地 Parquet 分区数据, 使用 pyarrow 谓词下推节省内存。

        数据格式: data/day=YYYY-MM-DD/data.parquet
        parquet 内部含 'date' 列 (timestamp[ns], 含时分秒) 用于日内定位。

        Parameters
        ----------
        sd, ed : str
            日期范围起点/终点 (含), 格式 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'.
        instruments : list[str] or None
            若不为 None, 只读取指定 instrument 的数据 (利用 parquet row-group 过滤).

        Returns
        -------
        pd.DataFrame
            按 instrument, date 排序后的数据, 含 'date' 列 (datetime).
        """
        sd_str = sd[:10]
        ed_str = ed[:10]
        # 按分区 day= 过滤, 提升谓词下推效率
        filters = (ds.field("day") >= sd_str) & (ds.field("day") <= ed_str)
        if instruments is not None:
            filters = filters & ds.field("instrument").isin(instruments)

        columns = FEATURE_COLS + ["date", "instrument"]
        table = ds.dataset(str(DATA_DIR), format="parquet", partitioning="hive").to_table(
            columns=columns,
            filter=filters,
        )
        if table.num_rows == 0:
            raise RuntimeError(f"本地数据无匹配记录 [{sd_str}, {ed_str}]")
        df = table.to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["instrument", "date"]).reset_index(drop=True)


# ---------- 成分股池 ----------
def pool(sd, ed):
    """区间内中证 1000 成分股代码。"""
    if USE_LOCAL_DATA:
        sd_str = sd[:10]
        pattern = os.path.join(DATA_DIR, f"day={sd_str}-*", "data.parquet")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(DATA_DIR, "day=*", "data.parquet")))[:1]
        pf = pq.ParquetFile(candidates[0])
        tbl = pf.read(columns=["instrument"])
        codes = pc.unique(tbl.column("instrument")).to_pylist()
        logger.info("本地数据提取 instrument", n=len(codes), start=sd, end=ed)
        return codes
    else:
        from bigquant import dai
        df = dai.query("SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
                       filters={"date": [sd, ed]}).df()
        return df["instrument"].tolist()


# ---------- 数据: 直接用原始字段切窗口, 只做 量log + 标准化 ----------
def build_dataset(table, sd, ed, mode, instruments, stats=None):
    """切窗口并标准化 (训练与推理共用).
    mode='train' 返回 (X, y, None, stats); 'infer' 返回 (X, None, idx_df, stats).
    X 为 (N, SEQ_LEN, N_FEAT); stats 为 (mean, std), 训练集上算好, 推理复用."""
    t0 = time.time()
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")  # 缓冲凑回看窗口

    if USE_LOCAL_DATA:
        df = _read_local_data(buf, ed, instruments=instruments)
    else:
        sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
        from bigquant import dai
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


# ---------- 分块扫描: 永远不一次加载全量标的, 避免 OOM ----------
CHUNK_SIZE = 250   # 每批最多加载 n 只标的的 df

def _read_chunk_data(buf, ed, chunk_instruments, table=None):
    """统一数据读取: USE_LOCAL_DATA=True 走本地 Parquet, 否则走 dai 查询。"""
    if USE_LOCAL_DATA:
        return _read_local_data(buf, ed, instruments=chunk_instruments)
    from bigquant import dai
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": chunk_instruments}).df()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["instrument", "date"]).reset_index(drop=True)


def _scan_chunk(sd, ed, chunk_instruments, table=None):
    """扫描一批 instrument 的窗口, 返回 (n_windows, sum_feat, sum2_feat, ys, meta)。"""
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    df = _read_chunk_data(buf, ed, chunk_instruments, table=table)
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))

    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    n_windows = 0
    sum_feat = np.zeros(N_FEAT, np.float64)
    sum2_feat = np.zeros(N_FEAT, np.float64)
    ys = []
    meta = []

    for ins, sub in df.groupby("instrument", sort=False):
        if len(sub) <= SEQ_LEN:
            continue
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        close_px = sub["close"].to_numpy(np.float64)[close_pos]
        dates = day[close_pos]
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                continue
            if k + 1 < len(close_pos) and close_px[k] > 0:
                r = close_px[k + 1] / close_px[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
                else:
                    continue
            else:
                continue
            win = feats[p - SEQ_LEN + 1: p + 1]
            n_windows += 1
            sum_feat += win.sum(axis=0)
            sum2_feat += (win ** 2).sum(axis=0)
            ys.append(label)
            meta.append((ins, p))

    del df
    gc.collect()
    return n_windows, sum_feat, sum2_feat, ys, meta


def compute_stats_and_y(sd, ed, instruments, table=None):
    """第一遍扫描: 分块读取, 流式计算标准化统计量 + 收集 labels (不存储 X)。"""
    t0 = time.time()
    total_n = 0
    total_sum = np.zeros(N_FEAT, np.float64)
    total_sum2 = np.zeros(N_FEAT, np.float64)
    all_ys = []
    all_meta = []

    for i in range(0, len(instruments), CHUNK_SIZE):
        chunk = instruments[i:i + CHUNK_SIZE]
        n, s, s2, ys, meta = _scan_chunk(sd, ed, chunk, table=table)
        total_n += n
        total_sum += s
        total_sum2 += s2
        all_ys.extend(ys)
        all_meta.extend(meta)
        logger.info("  块扫描完成", chunk=f"{i // CHUNK_SIZE + 1}/{(len(instruments) - 1) // CHUNK_SIZE + 1}",
                    instruments=len(chunk), samples=n)

    if total_n == 0:
        raise RuntimeError(f"compute_stats_and_y 无样本")

    count = total_n * SEQ_LEN
    mean = (total_sum / count).astype(np.float32)
    variance = total_sum2 / count - (total_sum / count) ** 2
    std = np.sqrt(np.maximum(variance, 0)).astype(np.float32) + 1e-6
    stats = (mean, std)

    y_arr = np.array(all_ys, np.float32)
    lo, hi = np.percentile(y_arr, [1, 99])
    y_clipped = np.clip(y_arr, lo, hi)

    logger.info("第一遍扫描完成", samples=total_n, elapsed=round(time.time() - t0, 2))
    return stats, y_clipped, lo, hi, all_meta


class StockWindowDataset(IterableDataset):
    """IterableDataset: 每 epoch 分块读取数据, 即时标准化后 yield (x, y)。

    永远不一次加载全量标的的数据, 峰值内存 = CHUNK_SIZE 只标的的数据。"""

    def __init__(self, sd, ed, instruments, stats, meta, y, table=None, shuffle=True):
        super().__init__()
        self.sd = sd
        self.ed = ed
        self.instruments = instruments
        self.m, self.s = stats
        self.meta = meta
        self.y = y
        self.table = table
        self.shuffle = shuffle
        self.n_samples = len(meta)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        idx = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)

        # 按 instrument 分组
        groups = {}
        for i in idx:
            ins, p = self.meta[i]
            groups.setdefault(ins, []).append((p, self.y[i]))

        ins_list = list(groups.keys())
        buf = (pd.to_datetime(self.sd) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")

        # 分块处理, 每块只加载部分标的的数据
        for i in range(0, len(ins_list), CHUNK_SIZE):
            chunk_ins = ins_list[i:i + CHUNK_SIZE]
            df = _read_chunk_data(buf, self.ed, chunk_ins, table=self.table)
            for c in VOL_COLS:
                df[c] = np.log1p(df[c].clip(lower=0))

            # 构建当前块内 feats 查找
            ins_feats = {}
            for ins, sub in df.groupby("instrument", sort=False):
                ins_feats[ins] = sub.sort_values("date")[FEATURE_COLS].to_numpy(np.float32)

            # yield 当前块的样本 (经 shuffle buffer 混洗, 避免同 instrument 连续)
            chunk_samples = []
            for ins in chunk_ins:
                feats = ins_feats[ins]
                for p, label in groups[ins]:
                    x = feats[p - SEQ_LEN + 1: p + 1]
                    x = (x - self.m) / self.s
                    chunk_samples.append((x.astype(np.float32), label))
            if self.shuffle:
                np.random.shuffle(chunk_samples)
            yield from chunk_samples

            del df, ins_feats
            gc.collect()

        del groups


# ==== 模型存/读: 一律用文本类文件 (JSON), 不使用 .pt 等二进制 ====
def save_model(ckpt, model_path=MODEL_PATH):
    """把 checkpoint 存成 JSON 文本文件。

    state_dict 里每个张量转成 {dtype, shape, data(扁平 list)}, 其余字段 (结构超参 /
    标准化统计等) 原样写入; 加载时用 load_model 按 dtype/shape 还原。"""
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
    """读取 save_model 写出的 JSON, 把 state_dict 还原为张量 dict。

    返回结构与原 torch.load(...) 的 checkpoint 一致 (state_dict 为 {name: Tensor})。"""
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

    公榜阶段平台不会重训, 直接加载该文件做推理; 私榜阶段平台用本函数从零重训,
    故训练逻辑需保持可复现 (固定随机种子)。

    统一使用流式处理: 第一遍分块扫描计算 stats + labels (不存储 X),
    之后每 epoch 通过 IterableDataset 分块重读数据, 峰值内存 = CHUNK_SIZE 只标的的 df。"""
    table = datasources["bar1m"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=table, use_local_data=USE_LOCAL_DATA)

    # ---------- 第一遍: 流式扫描, 不存储 X ----------
    logger.info("第一遍扫描: 计算 stats + 收集 labels", start=TRAIN_START, end=TRAIN_END)
    instruments = pool(TRAIN_START, TRAIN_END)
    stats, ytr, _, _, meta = compute_stats_and_y(TRAIN_START, TRAIN_END, instruments, table=table)

    # ---------- IterableDataset (每 epoch 重读数据, 不常驻 X) ----------
    dataset = StockWindowDataset(TRAIN_START, TRAIN_END, instruments, stats, meta, ytr, table=table, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH, num_workers=0)

    # ---------- 从零训练 ----------
    model = StockTransformer(**MODEL_CFG).to(device)
    logger.info("可训练参数量", n_params=sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(EPOCHS):
        t, tot, nb = time.time(), 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
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
    # 一律存为文本类文件 (JSON), 张量在 save_model 内转成 {dtype, shape, data}
    mean, std = stats
    save_model({
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
    # 本地训练入口: 在写死的训练区间上从零训练并保存 transformer_model.json
    datasources = {"bar1m": "bigalpha_2026_stock_bar1m"}
    train_and_save(datasources)
