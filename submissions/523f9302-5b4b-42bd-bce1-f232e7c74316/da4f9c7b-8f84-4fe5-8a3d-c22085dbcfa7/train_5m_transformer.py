"""
本地 8 卡 A100 训练脚本 —— BigAlpha 2026 端到端赛道。

功能:
    1. 从本地 parquet（download_data.py 落盘的 e2e_bar* 文件）读取数据。
    2. 基于 features_raw.py 做本地/云端统一的 raw schema 对齐。
    3. 训练一个 StockTransformer（PatchTST 风格）。
    4. 损失函数: 截面 Pearson 相关损失（直接优化 IC），比 MSE 效果更好。
    5. 支持单卡运行和 8 卡 DDP（torchrun 启动）。
    6. 大数据（bar5m/bar1m）自动走 memmap 路径，避免一次性占满内存。
    7. 输出: transformer_model.json（权重 + 归一化统计 + 超参），供 predict.ipynb 加载。

用法:
    # 单卡（调试）
    python transformer_train.py --freq bar30m

    # 8 卡 DDP（正式训练 bar5m）
    torchrun --nproc_per_node=8 transformer_train.py --freq bar5m --epochs 20 --batch_size 2048

    # bar5m 大数据（自动走 memmap）
    torchrun --nproc_per_node=8 transformer_train.py --freq bar5m --seq_len 240 --epochs 20

前提:
    - 已运行 download_data.py 把数据落盘到 ./local_data/
    - 已运行 bq auth --apikey <AK.SK> 完成认证（下载阶段需要，训练不需要）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import tempfile
import time
from datetime import timedelta

from reproducibility import bootstrap_repro_env

bootstrap_repro_env()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from features_raw import (
    FEATURE_COLS, RAW_COLS, RAW25_COLS, N_FEAT,
    load_local_table, compute_stats, normalize,
)
from download_data import (
    LOCAL_DATA_DIR, TRAIN_START, TRAIN_END,
    local_parquet_paths, E2E_TABLES,
)
from reproducibility import make_torch_generator, seed_worker, set_global_seed, strict_repro_enabled

# ── 输出路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "transformer_model.json")

FEATURE_SETS = {
    "raw19": RAW_COLS,
    "raw25": RAW25_COLS,
}

# 大数据时切换到 memmap 的阈值（样本数 × 特征大小，单位 bytes）
MEMMAP_THRESHOLD_BYTES = 8 * 1024 ** 3  # 8 GB

# ── 训练超参（写死，私榜重训时平台使用相同值） ──
CFG = dict(
    freq         = "bar5m",
    train_start  = TRAIN_START,
    train_end    = TRAIN_END,
    seq_len      = 240,   # bar5m×240 = 5 个完整交易日
    patch_size   = 12,    # 240 / 12 = 20 patches
    d_model      = 256,
    nhead        = 8,
    nlayers      = 4,
    dim_ff       = 512,
    dropout      = 0.1,
    activation   = "gelu",   # gelu / relu
    norm         = "pre",    # pre = Pre-LN (norm_first)，post = Post-LN
    final_norm   = 0,        # 1 = encoder 出口再加一层 LayerNorm
    pool         = "mean",   # mean / last / attn
    revin        = 0,        # 1 = 窗口内实例归一化（RevIN 层，属模型结构）
    batch_size   = 2048,
    epochs       = 20,
    lr           = 5e-4,
    weight_decay = 1e-4,
    seed         = 42,
    label_winsor = (1, 99),
    label_mode   = "raw",    # raw = 全局 winsorize；cs_zscore = 按日截面标准化
    batch_mode   = "random", # random = 随机batch；by_date = 同一天一个batch（真截面IC）
    grad_clip    = 1.0,
    feature_set  = "raw19",
    output       = MODEL_PATH,
    num_workers  = 4,
    strict_repro = False,
    # ── 本地 OOS 验证（留空则不验证，行为与旧版一致）──
    val_start    = "",
    val_end      = "",
    val_stride   = 1,        # 验证日期抽样步长，>1 可省内存
    save_best    = 0,        # 1 = 按验证 RankIC 保存最优 epoch 权重
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  模型: PatchTST 风格
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, n_feat: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_feat * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        n_patch = L // self.patch_size
        x = x[:, :n_patch * self.patch_size, :]
        x = x.reshape(B, n_patch, self.patch_size * C)
        return self.proj(x)


class StockTransformer(nn.Module):
    def __init__(
        self,
        n_feat:     int   = N_FEAT,
        patch_size: int   = 12,
        d_model:    int   = 256,
        nhead:      int   = 8,
        nlayers:    int   = 4,
        dim_ff:     int   = 512,
        dropout:    float = 0.1,
        seq_len:    int   = 240,
        activation: str   = "gelu",
        norm:       str   = "pre",
        final_norm: int   = 0,
        pool:       str   = "mean",
        revin:      int   = 0,
    ):
        super().__init__()
        n_patch = seq_len // patch_size
        self.revin = bool(revin)
        self.pool = pool
        self.patch_embed = PatchEmbedding(n_feat, patch_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=True, activation=activation,
            norm_first=(norm == "pre"),
        )
        # Pre-LN 结构在 encoder 出口补一个 LayerNorm 通常更稳；默认关闭以保持与历史模型一致
        enc_norm = nn.LayerNorm(d_model) if final_norm else None
        self.encoder = nn.TransformerEncoder(
            layer, nlayers, norm=enc_norm, enable_nested_tensor=False
        )
        if pool == "attn":
            self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.pool_q, std=0.02)
            self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp(min=1e-5)
            x = (x - mu) / sd
        h = self.patch_embed(x) + self.pos
        h = self.encoder(h)
        if self.pool == "last":
            h = h[:, -1, :]
        elif self.pool == "attn":
            q = self.pool_q.expand(h.size(0), -1, -1)
            h = self.pool_attn(q, h, h, need_weights=False)[0].squeeze(1)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


# ─────────────────────────────────────────────
#  损失函数: 截面 Pearson 相关损失（直接优化 IC）
# ─────────────────────────────────────────────

def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c   = pred   - pred.mean()
    target_c = target - target.mean()
    cov      = (pred_c * target_c).mean()
    std_p    = pred_c.pow(2).mean().sqrt().clamp(min=1e-8)
    std_t    = target_c.pow(2).mean().sqrt().clamp(min=1e-8)
    return -cov / (std_p * std_t)


# ─────────────────────────────────────────────
#  模型序列化
# ─────────────────────────────────────────────

def save_model(ckpt: dict, path: str = MODEL_PATH) -> str:
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data":  t.reshape(-1).tolist(),
        }
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    log.info("模型已保存: %s  (%.1f MB)", path, os.path.getsize(path) / 1024 / 1024)
    return path


def load_model(path: str = MODEL_PATH, map_location: str = "cpu") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ─────────────────────────────────────────────
#  memmap Dataset — 支持大数据训练
# ─────────────────────────────────────────────

class MemmapDataset(Dataset):
    """从 numpy memmap 文件读取 (X, y)，避免一次性占满内存。"""

    def __init__(self, x_path: str, y_path: str, shape_x: tuple, n: int):
        self.X = np.memmap(x_path, dtype=np.float32, mode="r", shape=shape_x)
        self.y = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n,))
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx].copy()), torch.tensor(self.y[idx])

    @staticmethod
    def save(X: np.ndarray, y: np.ndarray, tmpdir: str) -> tuple[str, str]:
        x_path = os.path.join(tmpdir, "X.memmap")
        y_path = os.path.join(tmpdir, "y.memmap")
        xm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=X.shape)
        xm[:] = X
        xm.flush()
        ym = np.memmap(y_path, dtype=np.float32, mode="w+", shape=y.shape)
        ym[:] = y
        ym.flush()
        return x_path, y_path


# ─────────────────────────────────────────────
#  DDP 工具
# ─────────────────────────────────────────────

def setup_ddp() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int) -> None:
    set_global_seed(seed, deterministic_torch=True)


# ─────────────────────────────────────────────
#  数据构建
# ─────────────────────────────────────────────

def get_feature_cols(cfg: dict) -> list[str]:
    feature_set = cfg.get("feature_set", "raw19")
    if feature_set not in FEATURE_SETS:
        raise ValueError("未知 feature_set=%r，可选: %s" % (feature_set, sorted(FEATURE_SETS)))
    return list(FEATURE_SETS[feature_set])


def make_windows_for_cols(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    mode: str,
    sd,
    ed,
):
    sd_ts = pd.Timestamp(sd)
    ed_ts = pd.Timestamp(ed)
    wins, ys_list, keys = [], [], []

    for ins, sub in df_feat.groupby("instrument", sort=False):
        if len(sub) < seq_len + 1:
            continue
        sub = sub.sort_values("date").reset_index(drop=True)
        feats = sub[feature_cols].to_numpy(np.float32)
        day_arr = sub["date"].dt.normalize().to_numpy()
        close_np = sub["close"].to_numpy(np.float64)
        last_bar = np.flatnonzero(np.append(day_arr[1:] != day_arr[:-1], True))
        close_by_day = close_np[last_bar]
        dates_by_day = day_arr[last_bar]

        for k, p in enumerate(last_bar):
            d = pd.Timestamp(dates_by_day[k])
            if p + 1 < seq_len or d < sd_ts or d > ed_ts:
                continue
            label = None
            if k + 1 < len(last_bar) and close_by_day[k] > 0:
                r = close_by_day[k + 1] / close_by_day[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue
            wins.append(feats[p - seq_len + 1 : p + 1])
            ys_list.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))

    if not keys:
        raise RuntimeError(
            "make_windows_for_cols 无样本 (mode=%s, sd=%s, ed=%s)。请检查日期范围或数据完整性。"
            % (mode, sd, ed)
        )

    X = np.stack(wins, axis=0).astype(np.float32)
    y = np.array(ys_list, np.float32) if mode == "train" else None
    idx = pd.DataFrame(keys, columns=["date", "instrument"])
    return X, y, idx


def _load_feat_df(cfg: dict, sd, ed) -> pd.DataFrame:
    freq = cfg["freq"]
    table = next((t for t in E2E_TABLES if t.endswith(freq)), None)
    if table is None:
        raise ValueError(f"未知频率: {freq}")

    parquet_paths = local_parquet_paths(table, sd, ed)
    if not parquet_paths:
        raise FileNotFoundError(
            "未找到本地数据: %s [%s ~ %s]\n"
            "请先执行: python download_data.py --table %s --start %s --end %s"
            % (table, sd, ed, table, sd, ed)
        )

    log.info("加载并处理本地数据: %s", parquet_paths)
    df_parts = [load_local_table(path) for path in parquet_paths]
    df_feat = pd.concat(df_parts, ignore_index=True) if len(df_parts) > 1 else df_parts[0]
    lo = pd.Timestamp(sd)
    hi = pd.Timestamp(ed) + pd.Timedelta(days=1)
    return df_feat[(df_feat["date"] >= lo) & (df_feat["date"] < hi)].copy()


def _apply_label_mode(y: np.ndarray, idx: pd.DataFrame, cfg: dict) -> np.ndarray:
    """raw = 全局 winsorize（历史行为）；cs_zscore = 按日截面标准化后 clip。"""
    if cfg.get("label_mode", "raw") == "cs_zscore":
        s = pd.Series(y.astype(np.float64), index=idx.index)
        g = s.groupby(idx["date"])
        z = (s - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        return z.to_numpy(np.float32)
    lo, hi = np.percentile(y, cfg["label_winsor"])
    return np.clip(y, lo, hi).astype(np.float32)


def build_train_data(cfg: dict) -> tuple[np.ndarray, np.ndarray, tuple, pd.DataFrame]:
    """
    读本地 parquet → 特征构建 → 切窗口 → 归一化。
    返回 (X_norm, y, stats, idx)。
    """
    df_feat = _load_feat_df(cfg, cfg["train_start"], cfg["train_end"])
    feature_cols = get_feature_cols(cfg)

    log.info(
        "切时序窗口 (seq_len=%d, feature_set=%s, n_feat=%d) ...",
        cfg["seq_len"],
        cfg.get("feature_set", "raw19"),
        len(feature_cols),
    )
    X_raw, y, idx = make_windows_for_cols(
        df_feat,
        feature_cols,
        seq_len=cfg["seq_len"],
        mode="train",
        sd=cfg["train_start"],
        ed=cfg["train_end"],
    )
    log.info("原始样本数: %d  特征维度: %s", len(y), X_raw.shape)

    y = _apply_label_mode(y, idx, cfg)

    mean, std = compute_stats(X_raw)
    X_norm = normalize(X_raw, mean, std)

    return X_norm, y, (mean, std), idx


def build_val_data(cfg: dict, stats: tuple) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    构建 OOS 验证集。归一化统计量沿用训练集，标签保持原始收益率
    （验证指标是按日截面 RankIC，不需要再做标签变换）。
    """
    # 需要多读 seq_len 之前的历史 bar 才能在验证区间首日切出完整窗口
    lookback_days = int(cfg["seq_len"] / 48 * 2) + 30
    load_start = (pd.Timestamp(cfg["val_start"]) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df_feat = _load_feat_df(cfg, load_start, cfg["val_end"])
    feature_cols = get_feature_cols(cfg)

    X_raw, y, idx = make_windows_for_cols(
        df_feat,
        feature_cols,
        seq_len=cfg["seq_len"],
        mode="train",
        sd=cfg["val_start"],
        ed=cfg["val_end"],
    )

    stride = max(1, int(cfg.get("val_stride", 1)))
    if stride > 1:
        days = np.sort(idx["date"].unique())
        keep = set(days[::stride])
        mask = idx["date"].isin(keep).to_numpy()
        X_raw, y, idx = X_raw[mask], y[mask], idx[mask].reset_index(drop=True)

    mean, std = stats
    X_norm = normalize(X_raw, mean, std)
    log.info("验证集样本数: %d  交易日数: %d", len(y), idx["date"].nunique())
    return X_norm, y, idx


@torch.no_grad()
def evaluate_rank_ic(model, X_val, y_val, idx_val, device, batch_size: int = 4096) -> dict:
    """按日截面 RankIC，返回 mean / std / IR，与赛事 IC 类指标对齐。"""
    model.eval()
    preds = np.empty(len(X_val), dtype=np.float64)
    for i in range(0, len(X_val), batch_size):
        xb = torch.from_numpy(X_val[i : i + batch_size]).to(device, non_blocking=True)
        preds[i : i + batch_size] = model(xb).float().cpu().numpy()
    df = pd.DataFrame({"date": idx_val["date"].to_numpy(), "pred": preds, "y": y_val.astype(np.float64)})
    daily = df.groupby("date", sort=True).apply(
        lambda g: g["pred"].corr(g["y"], method="spearman") if len(g) >= 16 else np.nan
    )
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    m, s = float(daily.mean()), float(daily.std())
    model.train()
    return {"ic": m, "ic_std": s, "ir": m / s if s > 1e-12 else 0.0, "n_days": int(len(daily))}


class DateBatchSampler(torch.utils.data.Sampler):
    """同一交易日的样本组成一个 batch，使 ic_loss 成为真正的截面 IC。"""

    def __init__(self, dates: np.ndarray, max_batch: int, seed: int, rank: int = 0, world_size: int = 1):
        self.groups = [np.flatnonzero(dates == d) for d in np.sort(pd.unique(dates))]
        if world_size > 1:
            self.groups = self.groups[rank::world_size]
        self.max_batch = max_batch
        self.seed = seed
        self.epoch = 0
        self.batches = []
        for g in self.groups:
            for i in range(0, len(g), max_batch):
                chunk = g[i : i + max_batch]
                if len(chunk) >= 16:  # 截面太小算 IC 无意义
                    self.batches.append(chunk)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.batches))
        for i in order:
            b = self.batches[i]
            yield rng.permutation(b).tolist()

    def __len__(self) -> int:
        return len(self.batches)



def _make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    tmpdir: str,
) -> Dataset:
    """
    根据数据大小选择 TensorDataset（小数据）或 MemmapDataset（大数据）。
    """
    data_bytes = X.nbytes + y.nbytes
    if data_bytes <= MEMMAP_THRESHOLD_BYTES:
        log.info("数据 %.1f GB，使用 TensorDataset（全量内存）", data_bytes / 1024**3)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    else:
        log.info("数据 %.1f GB 超过阈值，写 memmap 到 %s ...", data_bytes / 1024**3, tmpdir)
        t0 = time.time()
        x_path, y_path = MemmapDataset.save(X, y, tmpdir)
        log.info("memmap 写入完成 %.1fs", time.time() - t0)
        return MemmapDataset(x_path, y_path, X.shape, len(y))


# ─────────────────────────────────────────────
#  主训练流程
# ─────────────────────────────────────────────

def train(cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = CFG
    cfg = dict(cfg)
    strict_repro = bool(cfg.get("strict_repro")) or strict_repro_enabled()
    feature_cols = get_feature_cols(cfg)
    output_path = cfg.get("output") or MODEL_PATH
    if not os.path.isabs(output_path):
        output_path = os.path.join(SCRIPT_DIR, output_path)

    rank, local_rank, world_size = setup_ddp()
    set_seed(cfg["seed"] + rank)

    use_val = bool(cfg.get("val_start")) and bool(cfg.get("val_end"))
    if world_size > 1 and (use_val or cfg.get("batch_mode") == "by_date"):
        raise RuntimeError(
            "val_start/val_end 与 batch_mode=by_date 目前只支持单卡运行："
            "请用 `CUDA_VISIBLE_DEVICES=<id> python transformer_train.py ...` 启动，"
            "不要用 torchrun。"
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main(rank):
        log.info("训练配置: %s", cfg)
        log.info("特征集: %s  n_feat=%d", cfg.get("feature_set", "raw19"), len(feature_cols))
        log.info("输出模型: %s", output_path)
        log.info("设备: %s  world_size: %d", device, world_size)
        log.info("强复现模式: %s", strict_repro)

    # ── 数据：只在 rank0 构建，通过 memmap 文件共享给其他进程 ──
    # memmap 比 broadcast 更省内存（broadcast 需要每卡一份）
    tmpdir = None
    x_path = y_path = None
    n_samples = shape_x = None
    stats = None

    if is_main(rank):
        t0 = time.time()
        X, y, stats, idx_tr = build_train_data(cfg)
        log.info("数据构建完成，耗时 %.1f s", time.time() - t0)
        n_samples = len(y)
        shape_x = X.shape

        if world_size > 1:
            # 写 memmap 供其他 rank 共享读
            tmpdir = tempfile.mkdtemp(prefix="bq_train_", dir=SCRIPT_DIR)
            x_path, y_path = MemmapDataset.save(X, y, tmpdir)
            log.info("共享 memmap 写入: %s", tmpdir)
            del X, y  # 释放内存，其他 rank 从 memmap 读

        # broadcast metadata: n_samples, shape dims, stats, paths
        if world_size > 1:
            # pack metadata into a simple dict via json broadcast over a tensor
            meta = {
                "n": n_samples,
                "shape": list(shape_x),
                "mean": stats[0].tolist(),
                "std":  stats[1].tolist(),
                "x_path": x_path,
                "y_path": y_path,
            }
            meta_str = json.dumps(meta)
            meta_bytes = meta_str.encode("utf-8")
            meta_len = torch.tensor([len(meta_bytes)], dtype=torch.long).cuda()
        else:
            meta_len = None
    else:
        meta_len = torch.zeros(1, dtype=torch.long).cuda()

    # broadcast metadata length then content
    if world_size > 1:
        dist.broadcast(meta_len, src=0)
        meta_len_val = meta_len.item()
        if is_main(rank):
            meta_t = torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8).cuda()
        else:
            meta_t = torch.zeros(meta_len_val, dtype=torch.uint8).cuda()
        dist.broadcast(meta_t, src=0)
        if not is_main(rank):
            meta = json.loads(meta_t.cpu().numpy().tobytes().decode("utf-8"))
            n_samples = meta["n"]
            shape_x   = tuple(meta["shape"])
            stats     = (
                np.array(meta["mean"], dtype=np.float32),
                np.array(meta["std"],  dtype=np.float32),
            )
            x_path = meta["x_path"]
            y_path = meta["y_path"]

    # ── DataLoader ──
    if world_size > 1:
        # 所有 rank 从 memmap 文件读
        dataset = MemmapDataset(x_path, y_path, shape_x, n_samples)
    else:
        # 单卡：根据大小选 TensorDataset 或 MemmapDataset
        tmpdirobj = tempfile.TemporaryDirectory(prefix="bq_train_single_", dir=SCRIPT_DIR)
        dataset = _make_dataset(X, y, tmpdirobj.name)

    sampler = DistributedSampler(dataset, shuffle=True, seed=cfg["seed"]) if world_size > 1 else None
    num_workers = int(os.environ.get("BQ_TORCH_NUM_WORKERS", "0" if strict_repro else str(cfg.get("num_workers", 4))))
    date_sampler = None
    if cfg.get("batch_mode") == "by_date":
        date_sampler = DateBatchSampler(
            idx_tr["date"].to_numpy(), cfg["batch_size"], cfg["seed"], rank, world_size
        )
        log.info("按日 batch: %d 个截面 batch", len(date_sampler))
        loader_kwargs = dict(
            dataset=dataset,
            batch_sampler=date_sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=make_torch_generator(cfg["seed"] + rank),
        )
    else:
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=make_torch_generator(cfg["seed"] + rank),
        )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_kwargs)

    # ── 验证集（单卡，可选）──
    X_val = y_val = idx_val = None
    if use_val:
        X_val, y_val, idx_val = build_val_data(cfg, stats)

    # ── 模型 ──
    model_cfg = dict(
        n_feat     = len(feature_cols),
        patch_size = cfg.get("patch_size", CFG["patch_size"]),
        d_model    = cfg.get("d_model", CFG["d_model"]),
        nhead      = cfg.get("nhead", CFG["nhead"]),
        nlayers    = cfg.get("nlayers", CFG["nlayers"]),
        dim_ff     = cfg.get("dim_ff", CFG["dim_ff"]),
        dropout    = cfg.get("dropout", CFG["dropout"]),
        seq_len    = cfg["seq_len"],
        activation = cfg.get("activation", CFG["activation"]),
        norm       = cfg.get("norm", CFG["norm"]),
        final_norm = int(cfg.get("final_norm", CFG["final_norm"])),
        pool       = cfg.get("pool", CFG["pool"]),
        revin      = int(cfg.get("revin", CFG["revin"])),
    )

    model = StockTransformer(**model_cfg).to(device)
    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters())
        log.info("可训练参数量: %d (%.2f M)", n_params, n_params / 1e6)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    # ── 训练循环 ──
    best = {"ic": -1e9, "epoch": -1, "state": None}
    for epoch in range(1, cfg["epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if date_sampler is not None:
            date_sampler.set_epoch(epoch)
        model.train()
        total_loss, n_batch = 0.0, 0
        t_ep = time.time()

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = ic_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        val_msg = ""
        if use_val:
            vm = evaluate_rank_ic(model, X_val, y_val, idx_val, device)
            val_msg = "  val_rank_ic=%.6f  val_ir=%.4f  val_days=%d" % (vm["ic"], vm["ir"], vm["n_days"])
            if vm["ic"] > best["ic"]:
                best = {
                    "ic": vm["ic"],
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

        if is_main(rank):
            log.info(
                "epoch %d/%d  ic_loss=%.6f  lr=%.2e  elapsed=%.1fs%s",
                epoch, cfg["epochs"],
                total_loss / max(n_batch, 1),
                scheduler.get_last_lr()[0],
                time.time() - t_ep,
                val_msg,
            )

    if use_val:
        log.info("最优 epoch=%d  val_rank_ic=%.6f", best["epoch"], best["ic"])

    # ── 保存（只在 rank0 保存） ──
    if is_main(rank):
        raw_model = model.module if world_size > 1 else model
        state = raw_model.state_dict()
        if use_val and int(cfg.get("save_best", 0)) and best["state"] is not None:
            state = best["state"]
            log.info("保存 val_rank_ic 最优权重（epoch %d）", best["epoch"])
        mean, std = stats
        save_model({
            "state_dict":   state,
            "model_cfg":    model_cfg,
            "train_cfg":    {k: v for k, v in cfg.items() if k != "output"},
            "strict_repro": strict_repro,
            "feature_set":  cfg.get("feature_set", "raw19"),
            "feature_cols": feature_cols,
            "seq_len":      cfg["seq_len"],
            "freq":         cfg["freq"],
            "val_rank_ic":  best["ic"] if use_val else None,
            "val_best_epoch": best["epoch"] if use_val else None,
            "mean": np.asarray(mean, np.float32).tolist(),
            "std":  np.asarray(std,  np.float32).tolist(),
        }, output_path)


    if world_size > 1:
        dist.barrier()
    if is_main(rank) and tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
        log.info("已清理训练临时目录: %s", tmpdir)

    cleanup_ddp(world_size)
    return output_path


# ─────────────────────────────────────────────
#  CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigAlpha 端到端模型本地训练")
    parser.add_argument("--freq",       default=CFG["freq"],
                        choices=["bar1m", "bar5m", "bar15m", "bar30m"])
    parser.add_argument("--train_start", default=CFG["train_start"])
    parser.add_argument("--train_end",   default=CFG["train_end"])
    parser.add_argument("--seq_len",    type=int,   default=CFG["seq_len"])
    parser.add_argument("--patch_size", type=int,   default=CFG["patch_size"])
    parser.add_argument("--epochs",     type=int,   default=CFG["epochs"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--d_model",    type=int,   default=CFG["d_model"])
    parser.add_argument("--nlayers",    type=int,   default=CFG["nlayers"])
    parser.add_argument("--nhead",      type=int,   default=CFG["nhead"])
    parser.add_argument("--dim_ff",     type=int,   default=CFG["dim_ff"])
    parser.add_argument("--dropout",    type=float, default=CFG["dropout"])
    parser.add_argument("--weight_decay", type=float, default=CFG["weight_decay"])
    parser.add_argument("--activation", default=CFG["activation"], choices=["gelu", "relu"])
    parser.add_argument("--norm",       default=CFG["norm"], choices=["pre", "post"])
    parser.add_argument("--final_norm", type=int, default=CFG["final_norm"], choices=[0, 1])
    parser.add_argument("--pool",       default=CFG["pool"], choices=["mean", "last", "attn"])
    parser.add_argument("--revin",      type=int, default=CFG["revin"], choices=[0, 1])
    parser.add_argument("--label_mode", default=CFG["label_mode"], choices=["raw", "cs_zscore"])
    parser.add_argument("--batch_mode", default=CFG["batch_mode"], choices=["random", "by_date"])
    parser.add_argument("--val_start",  default=CFG["val_start"])
    parser.add_argument("--val_end",    default=CFG["val_end"])
    parser.add_argument("--val_stride", type=int, default=CFG["val_stride"])
    parser.add_argument("--save_best",  type=int, default=CFG["save_best"], choices=[0, 1])
    parser.add_argument("--batch_size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--seed",       type=int,   default=CFG["seed"])
    parser.add_argument("--feature_set", default=CFG["feature_set"], choices=sorted(FEATURE_SETS))
    parser.add_argument("--output",      default=CFG["output"])
    parser.add_argument("--num_workers", type=int, default=CFG["num_workers"])
    parser.add_argument("--strict_repro", action="store_true")
    args = parser.parse_args()

    cfg = {**CFG, **{k: v for k, v in vars(args).items() if v is not None}}
    train(cfg)
