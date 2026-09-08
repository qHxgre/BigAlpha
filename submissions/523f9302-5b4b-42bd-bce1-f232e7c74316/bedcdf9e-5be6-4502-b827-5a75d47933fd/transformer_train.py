# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端模型：Patch Transformer + 截面排序损失。

本文件保留官方模板的提交接口：

* ``train_and_save(datasources)`` 在固定训练区间从零训练；
* 模型权重、标准化统计和结构配置保存为 JSON 文本；
* 推理 Notebook 从本文件导入模型和 ``predict_scores``；
* 训练和推理均按股票分块查询，避免全市场分钟数据一次性进入内存。

输入端只进行比赛允许的逐字段变换：指定成交量字段 ``log1p``、缺失值填充和
训练集统一标准化。Patch 划分、字段交互和时间聚合全部在神经网络内部完成。
"""

import gc
import json
import os
import time

import dai
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = structlog.get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "patch_ic_model.json")

# 固定训练配置。第一版沿用官方示例的训练区间和股票数量，先验证结构增益与运行成本。
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31 23:59:59"
SEQ_LEN = 240
PATCH_LEN = 8
EPOCHS = 8
BATCH = 512
LR = 3e-4
WEIGHT_DECAY = 1e-3
SEED = 42
MAX_TRAIN_INSTRUMENTS = 200
CHUNK_SIZE = 20
MIN_CROSS_SECTION = 16

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS = ["volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)

MODEL_CFG = {
    "n_feat": N_FEAT,
    "seq_len": SEQ_LEN,
    "patch_len": PATCH_LEN,
    "d_model": 96,
    "nhead": 4,
    "nlayers": 3,
    "dim_ff": 256,
    "dropout": 0.10,
}

LOSS_CFG = {
    "huber_weight": 0.45,
    "ic_weight": 0.35,
    "rank_weight": 0.20,
    "huber_beta": 0.50,
}


class PatchStockTransformer(nn.Module):
    """把连续原始 bar 切成 Patch token，再做时间注意力编码。

    Patch 仅执行 reshape 和可训练线性投影，不在模型外构造任何人工特征。
    """

    def __init__(
        self,
        n_feat,
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        d_model=96,
        nhead=4,
        nlayers=3,
        dim_ff=256,
        dropout=0.10,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len 必须能被 patch_len 整除")
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")

        self.n_feat = int(n_feat)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.n_patches = self.seq_len // self.patch_len

        self.patch_proj = nn.Linear(self.patch_len * self.n_feat, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, x):
        # x: (batch, seq_len, n_feat)
        if x.ndim != 3:
            raise ValueError(f"模型输入应为三维张量，实际 shape={tuple(x.shape)}")
        if x.shape[1] != self.seq_len or x.shape[2] != self.n_feat:
            raise ValueError(
                f"模型输入 shape 不匹配，期望 (*,{self.seq_len},{self.n_feat})，"
                f"实际 {tuple(x.shape)}"
            )
        batch = x.shape[0]
        patches = x.reshape(batch, self.n_patches, self.patch_len * self.n_feat)
        tokens = self.patch_proj(patches)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.input_norm(tokens + self.pos)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0]).squeeze(-1)


# 保留官方示例中的类名，便于已有 Notebook 迁移。
StockTransformer = PatchStockTransformer


def _set_deterministic(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def pool(sd, ed):
    """返回区间内出现过的中证1000成分股，排序以保证可复现。"""
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return sorted(df["instrument"].dropna().astype(str).unique().tolist())


def _sample_train_instruments(instruments, max_count, seed=SEED):
    instruments = np.asarray(sorted(set(instruments)), dtype=object)
    if len(instruments) <= max_count:
        return instruments.tolist()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(instruments, size=max_count, replace=False)
    return sorted(chosen.tolist())


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _buffer_start(sd):
    # 240根1分钟bar通常约为一个交易日；20个自然日为停牌和节假日留出余量。
    return (pd.to_datetime(sd) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")


def _query_chunk(table, buf, ed, chunk):
    sql = (
        f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    df = dai.query(
        sql,
        filters={"date": [buf, ed], "instrument": list(chunk)},
    ).df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["instrument", "date"], inplace=True, kind="mergesort")
    for col in VOL_COLS:
        values = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.log1p(values.clip(lower=0))
    return df


def _windows_from_one(sub, sd_ts, ed_ts, need_label):
    """从单只股票连续分钟数据提取每日收盘决策点窗口。"""
    wins, ys, keys = [], [], []
    if len(sub) <= SEQ_LEN:
        return wins, ys, keys

    instrument = str(sub["instrument"].iloc[0])
    feats = sub[FEATURE_COLS].to_numpy(np.float32)
    days = sub["date"].dt.normalize().to_numpy()
    close_pos = np.flatnonzero(np.append(days[1:] != days[:-1], True))
    close_px = pd.to_numeric(sub["close"], errors="coerce").to_numpy(np.float64)[close_pos]
    dates = days[close_pos]

    for k, pos in enumerate(close_pos):
        date = pd.Timestamp(dates[k])
        if pos + 1 < SEQ_LEN or date < sd_ts or date > ed_ts:
            continue

        label = None
        if k + 1 < len(close_pos) and np.isfinite(close_px[k]) and close_px[k] > 0:
            value = close_px[k + 1] / close_px[k] - 1.0
            if np.isfinite(value):
                label = np.float32(value)
        if need_label and label is None:
            continue

        wins.append(feats[pos - SEQ_LEN + 1 : pos + 1])
        ys.append(label if label is not None else np.float32(0.0))
        keys.append((date, instrument))
    return wins, ys, keys


class RunningStats:
    """忽略非有限值的逐字段流式 mean/std。"""

    def __init__(self, n_feat):
        self.count = np.zeros(n_feat, np.int64)
        self.sum = np.zeros(n_feat, np.float64)
        self.sumsq = np.zeros(n_feat, np.float64)

    def update(self, x):
        values = np.asarray(x, np.float64).reshape(-1, x.shape[-1])
        mask = np.isfinite(values)
        safe = np.where(mask, values, 0.0)
        self.count += mask.sum(axis=0)
        self.sum += safe.sum(axis=0)
        self.sumsq += (safe * safe).sum(axis=0)

    def finalize(self):
        if np.any(self.count == 0):
            bad = np.flatnonzero(self.count == 0).tolist()
            raise RuntimeError(f"以下字段在训练集中没有有限值: {bad}")
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean * mean
        std = np.sqrt(np.clip(var, 0.0, None))
        std = np.maximum(std, 1e-6)
        return mean.astype(np.float32), std.astype(np.float32)


def _normalize_windows(x, mean, std):
    normalized = (x - mean) / std
    # 缺失值统一填为该字段训练集均值对应的标准化值0。
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_train_dataset(table, sd, ed, instruments):
    """分块查询并构建训练窗口，返回X、y、样本键和训练集统计。"""
    started = time.time()
    buf = _buffer_start(sd)
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    all_wins, all_ys, all_keys = [], [], []
    stats = RunningStats(N_FEAT)
    n_chunks = (len(instruments) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_no, chunk in enumerate(_chunks(instruments, CHUNK_SIZE), start=1):
        df = _query_chunk(table, buf, ed, chunk)
        if not df.empty:
            for _, sub in df.groupby("instrument", sort=False):
                wins, ys, keys = _windows_from_one(sub, sd_ts, ed_ts, need_label=True)
                for window in wins:
                    stats.update(window)
                all_wins.extend(wins)
                all_ys.extend(ys)
                all_keys.extend(keys)
        del df
        gc.collect()
        logger.info(
            "训练分块完成",
            chunk=f"{chunk_no}/{n_chunks}",
            cumulative_samples=len(all_wins),
        )

    if not all_wins:
        raise RuntimeError(f"build_train_dataset 无样本 ({sd}~{ed})")

    mean, std = stats.finalize()
    x = np.stack(all_wins).astype(np.float32)
    y = np.asarray(all_ys, np.float32)
    keys = pd.DataFrame(all_keys, columns=["date", "instrument"])
    x = _normalize_windows(x, mean, std)
    logger.info(
        "训练集构建完成",
        samples=len(x),
        dates=keys["date"].nunique(),
        chunks=n_chunks,
        elapsed=round(time.time() - started, 2),
    )
    return x, y, keys, (mean, std)


def _date_groups(keys):
    groups = []
    for _, frame in keys.groupby("date", sort=True):
        indices = frame.index.to_numpy(np.int64)
        if len(indices) >= MIN_CROSS_SECTION:
            groups.append(indices)
    if not groups:
        raise RuntimeError("没有满足最小截面样本数的交易日")
    return groups


def _standardize_tensor(values):
    centered = values - values.mean()
    scale = torch.sqrt(torch.mean(centered * centered) + 1e-6)
    return centered / scale


def cross_sectional_objective(pred, target, cfg=LOSS_CFG):
    """同一交易日截面的回归、IC和成对排序联合损失。"""
    target_z = _standardize_tensor(target)
    huber = F.smooth_l1_loss(pred, target_z, beta=cfg["huber_beta"])

    if pred.numel() < MIN_CROSS_SECTION:
        zero = pred.new_zeros(())
        return huber, {"huber": huber.detach(), "ic": zero, "rank": zero}

    pred_z = _standardize_tensor(pred)
    ic = torch.mean(pred_z * target_z)
    ic_loss = 1.0 - ic

    perm = torch.randperm(pred.numel(), device=pred.device)
    target_diff = target_z - target_z[perm]
    pred_diff = pred - pred[perm]
    direction = torch.sign(target_diff)
    valid = direction != 0
    if torch.any(valid):
        rank_loss = F.softplus(-direction[valid] * pred_diff[valid]).mean()
    else:
        rank_loss = pred.new_zeros(())

    total = (
        cfg["huber_weight"] * huber
        + cfg["ic_weight"] * ic_loss
        + cfg["rank_weight"] * rank_loss
    )
    terms = {
        "huber": huber.detach(),
        "ic": ic.detach(),
        "rank": rank_loss.detach(),
    }
    return total, terms


def _train_one_epoch(model, x, y, groups, optimizer, device, rng):
    model.train()
    sums = {"loss": 0.0, "huber": 0.0, "ic": 0.0, "rank": 0.0}
    n_batches = 0

    for group_pos in rng.permutation(len(groups)):
        indices = groups[group_pos].copy()
        rng.shuffle(indices)
        for start in range(0, len(indices), BATCH):
            batch_idx = indices[start : start + BATCH]
            if len(batch_idx) < MIN_CROSS_SECTION:
                continue
            xb = torch.from_numpy(x[batch_idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(y[batch_idx]).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, terms = cross_sectional_objective(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sums["loss"] += float(loss.detach().cpu())
            for name in ("huber", "ic", "rank"):
                sums[name] += float(terms[name].cpu())
            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("训练未产生有效的按日截面 batch")
    return {name: value / n_batches for name, value in sums.items()}


def predict_scores(model, table, sd, ed, instruments, stats, device):
    """按股票分块查询、切窗和预测，只累计三列结果。"""
    mean, std = stats
    buf = _buffer_start(sd)
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    output = []
    cumulative_rows = 0
    n_chunks = (len(instruments) + CHUNK_SIZE - 1) // CHUNK_SIZE
    model.eval()

    for chunk_no, chunk in enumerate(_chunks(instruments, CHUNK_SIZE), start=1):
        df = _query_chunk(table, buf, ed, chunk)
        wins, keys = [], []
        if not df.empty:
            for _, sub in df.groupby("instrument", sort=False):
                part_wins, _, part_keys = _windows_from_one(
                    sub, sd_ts, ed_ts, need_label=False
                )
                wins.extend(part_wins)
                keys.extend(part_keys)
        del df

        if not wins:
            gc.collect()
            continue

        x = _normalize_windows(np.stack(wins).astype(np.float32), mean, std)
        tensor = torch.from_numpy(x)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(tensor), BATCH):
                xb = tensor[start : start + BATCH].to(device, non_blocking=True)
                predictions.append(model(xb).cpu().numpy())

        scores = np.concatenate(predictions).astype(np.float64)
        chunk_frame = pd.DataFrame(keys, columns=["date", "instrument"])
        chunk_frame["score"] = scores
        output.append(chunk_frame)
        cumulative_rows += len(chunk_frame)

        del wins, keys, x, tensor, predictions, scores, chunk_frame
        gc.collect()
        logger.info(
            "推理分块完成",
            chunk=f"{chunk_no}/{n_chunks}",
            cumulative_rows=cumulative_rows,
        )

    if not output:
        raise RuntimeError(f"predict_scores 无样本 ({sd}~{ed})")
    return pd.concat(output, ignore_index=True)


def save_model(checkpoint, model_path=MODEL_PATH):
    """将checkpoint保存为可提交的JSON文本。"""
    tensors = {}
    for name, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    payload = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    """读取JSON模型并还原state_dict。"""
    with open(model_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dict = {}
    for name, meta in payload["state_dict"].items():
        tensor = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        state_dict[name] = tensor.reshape(meta["shape"]).to(map_location)
    checkpoint = {key: value for key, value in payload.items() if key != "state_dict"}
    checkpoint["state_dict"] = state_dict
    return checkpoint


def train_and_save(datasources, model_path=MODEL_PATH):
    """在固定训练区间从零训练并保存模型。"""
    _set_deterministic(SEED)
    table = datasources["bar1m"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instruments = _sample_train_instruments(
        pool(TRAIN_START, TRAIN_END),
        MAX_TRAIN_INSTRUMENTS,
        SEED,
    )
    logger.info(
        "开始训练",
        device=str(device),
        table=table,
        instruments=len(instruments),
        chunk_size=CHUNK_SIZE,
    )

    x_train, y_train, keys, stats = build_train_dataset(
        table,
        TRAIN_START,
        TRAIN_END,
        instruments,
    )
    lower, upper = np.percentile(y_train, [1, 99])
    y_train = np.clip(y_train, lower, upper).astype(np.float32)
    groups = _date_groups(keys)

    model = PatchStockTransformer(**MODEL_CFG).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(f"模型参数量不符合赛规: {n_params}")
    logger.info("模型初始化完成", n_params=n_params, date_groups=len(groups))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    rng = np.random.default_rng(SEED)

    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        metrics = _train_one_epoch(
            model,
            x_train,
            y_train,
            groups,
            optimizer,
            device,
            rng,
        )
        scheduler.step()
        logger.info(
            "epoch完成",
            epoch=epoch,
            loss=round(metrics["loss"], 6),
            huber=round(metrics["huber"], 6),
            ic=round(metrics["ic"], 6),
            rank=round(metrics["rank"], 6),
            lr=round(scheduler.get_last_lr()[0], 8),
            elapsed=round(time.time() - started, 2),
        )

    mean, std = stats
    save_model(
        {
            "state_dict": model.state_dict(),
            "model_name": "PatchStockTransformer",
            "model_cfg": MODEL_CFG,
            "loss_cfg": LOSS_CFG,
            "feature_cols": FEATURE_COLS,
            "seq_len": SEQ_LEN,
            "mean": np.asarray(mean, np.float32).tolist(),
            "std": np.asarray(std, np.float32).tolist(),
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "seed": SEED,
            "n_params": n_params,
        },
        model_path,
    )
    logger.info("模型已保存", path=model_path, n_params=n_params)
    return model_path


if __name__ == "__main__":
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})
