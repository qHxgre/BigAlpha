"""Train a long-horizon raw daily-bar cross-sectional model for BigAlpha E2E.

The input boundary is intentionally narrow:
- build daily bars from raw intraday bars using daily OHLCV-style aggregation;
- use only open/high/low/close/volume/amount/deal_number as model inputs;
- let the network learn temporal relations from a long daily sequence.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from download_data import E2E_TABLES, TRAIN_START, local_parquet_paths
from parquet_compat import read_parquet, write_parquet
from reproducibility import set_global_seed


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"

PRICE_COLS = ["open", "high", "low", "close"]
FLOW_COLS = ["volume", "amount", "deal_number"]
DAILY_COLS = PRICE_COLS + FLOW_COLS
LOG1P_COLS = FLOW_COLS

CFG = dict(
    freq="bar5m",
    train_start=TRAIN_START,
    train_end="2023-12-31",
    seq_len=120,
    d_model=192,
    nhead=6,
    ts_layers=3,
    cs_layers=1,
    dim_ff=512,
    dropout=0.12,
    batch_days=8,
    epochs=14,
    lr=7e-4,
    weight_decay=1e-4,
    seed=42,
    label_winsor=(1, 99),
    label_mode="rank",
    ic_weight=1.0,
    pairwise_weight=0.15,
    grad_clip=1.0,
    revin=1,
    output=str(SCRIPT_DIR / "transformer_model.dailyohlcv_bar5m_seq120_d192_seed42.json"),
    val_start="",
    val_end="",
    val_stride=1,
    save_best=0,
    amp=1,
    force_daily_cache=0,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _daily_cache_path(freq: str, start: str, end: str) -> Path:
    tag = f"{start}_{end}".replace("-", "")
    return LOCAL_DATA_DIR / f"daily_ohlcv_{freq}__{tag}.parquet"


def _table_for_freq(freq: str) -> str:
    table = next((t for t in E2E_TABLES if t.endswith(freq)), None)
    if table is None:
        raise ValueError(f"unknown freq={freq!r}")
    return table


def build_daily_bars(freq: str, start: str, end: str, *, force: bool = False) -> pd.DataFrame:
    cache = _daily_cache_path(freq, start, end)
    if cache.exists() and not force:
        log.info("reuse daily cache: %s", cache)
        return read_parquet(cache)

    table = _table_for_freq(freq)
    paths = local_parquet_paths(table, start, end)
    if not paths:
        raise FileNotFoundError(f"missing local parquet for {table} [{start} ~ {end}]")

    log.info("build daily bars from %s", paths)
    lf = pl.scan_parquet(paths).select(["date", "instrument_id"] + DAILY_COLS)
    lf = lf.filter((pl.col("date") >= pl.lit(pd.Timestamp(start))) & (pl.col("date") < pl.lit(pd.Timestamp(end) + pd.Timedelta(days=1))))
    lf = lf.with_columns(pl.col("date").dt.truncate("1d").alias("day"))

    price_exprs = [
        pl.when(pl.col(c) > 0)
        .then(pl.col(c).cast(pl.Float64) / 100.0)
        .otherwise(None)
        .alias(c)
        for c in PRICE_COLS
    ]
    lf = lf.with_columns(
        price_exprs
        + [
            pl.col("volume").cast(pl.Float64).fill_null(0.0).alias("volume"),
            (pl.col("amount").cast(pl.Float64).fill_null(0.0) / 100.0).alias("amount"),
            pl.col("deal_number").cast(pl.Float64).fill_null(0.0).alias("deal_number"),
        ]
    )

    daily = (
        lf.group_by(["day", "instrument_id"])
        .agg(
            [
                pl.col("open").sort_by("date").drop_nulls().first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").sort_by("date").drop_nulls().last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("amount").sum().alias("amount"),
                pl.col("deal_number").sum().alias("deal_number"),
            ]
        )
        .sort(["instrument_id", "day"])
    )
    out_pl = daily.collect(streaming=True)
    df = pd.DataFrame({c: out_pl[c].to_numpy() for c in out_pl.columns})
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["instrument"] = df["instrument_id"].astype(str)
    df = df[["date", "instrument", "instrument_id"] + DAILY_COLS]
    df = df.replace([np.inf, -np.inf], np.nan)
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, cache)
    log.info("saved daily cache: %s shape=%s", cache, df.shape)
    return df


def preprocess_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["instrument", "date"]).reset_index(drop=True).copy()
    for c in DAILY_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    for c in PRICE_COLS:
        out[c] = out.groupby("instrument", sort=False)[c].ffill()
    for c in LOG1P_COLS:
        out[c] = np.log1p(out[c].clip(lower=0))
    out[DAILY_COLS] = out[DAILY_COLS].fillna(0.0).astype("float32")
    return out


def compute_stats(df_feat: pd.DataFrame, sample_start: str, sample_end: str) -> tuple[np.ndarray, np.ndarray]:
    sd = pd.Timestamp(sample_start)
    ed = pd.Timestamp(sample_end)
    sample = df_feat[(df_feat["date"] >= sd) & (df_feat["date"] <= ed)]
    arr = sample[DAILY_COLS].to_numpy(np.float32)
    mean = arr.mean(axis=0).astype(np.float32)
    std = (arr.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def _transform_labels(y: np.ndarray, dates: pd.Series, mode: str, winsor: tuple[int, int]) -> np.ndarray:
    lo, hi = np.percentile(y, winsor)
    y = np.clip(y, lo, hi).astype(np.float32)
    if mode == "raw":
        return y

    out = np.zeros_like(y, dtype=np.float32)
    date_values = pd.Series(pd.to_datetime(dates).to_numpy(), index=np.arange(len(y)))
    for _, idxs in date_values.groupby(date_values, sort=False).groups.items():
        ix = np.asarray(list(idxs), dtype=np.int64)
        vals = y[ix].astype(np.float64)
        if len(vals) < 2:
            continue
        if mode == "zscore":
            sd = vals.std()
            if sd > 1e-12:
                out[ix] = ((vals - vals.mean()) / sd).astype(np.float32)
        elif mode == "rank":
            order = np.argsort(vals, kind="mergesort")
            ranks = np.empty(len(vals), dtype=np.float64)
            ranks[order] = np.arange(len(vals), dtype=np.float64)
            ranks = ranks / max(len(vals) - 1, 1) - 0.5
            out[ix] = ranks.astype(np.float32)
        else:
            raise ValueError(f"unknown label_mode={mode!r}")
    return out


class DailyDataset(torch.utils.data.Dataset):
    def __init__(self, daily_x: list[np.ndarray], daily_y: list[np.ndarray], daily_dates: list[pd.Timestamp]):
        self.daily_x = daily_x
        self.daily_y = daily_y
        self.daily_dates = daily_dates

    def __len__(self) -> int:
        return len(self.daily_x)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.daily_x[idx]), torch.from_numpy(self.daily_y[idx])


def collate_days(batch):
    xs, ys = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    seq_len = xs[0].shape[1]
    n_feat = xs[0].shape[2]
    x_pad = torch.zeros(len(xs), max_n, seq_len, n_feat, dtype=torch.float32)
    y_pad = torch.zeros(len(xs), max_n, dtype=torch.float32)
    mask = torch.zeros(len(xs), max_n, dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        mask[i, :n] = True
    return x_pad, y_pad, mask


def build_dataset(
    cfg: dict,
    *,
    load_start: str,
    load_end: str,
    sample_start: str,
    sample_end: str,
    stats: tuple[np.ndarray, np.ndarray] | None,
    train: bool,
) -> tuple[DailyDataset, tuple[np.ndarray, np.ndarray]]:
    df = build_daily_bars(
        cfg["freq"],
        load_start,
        load_end,
        force=bool(int(cfg.get("force_daily_cache", 0))),
    )
    df_feat = preprocess_daily_frame(df)
    if stats is None:
        stats = compute_stats(df_feat, sample_start, sample_end)
    mean, std = stats

    sd = pd.Timestamp(sample_start)
    ed = pd.Timestamp(sample_end)
    seq_len = int(cfg["seq_len"])
    records: list[tuple[pd.Timestamp, np.ndarray, float]] = []

    for _, sub in df_feat.groupby("instrument", sort=False):
        if len(sub) <= seq_len:
            continue
        sub = sub.sort_values("date").reset_index(drop=True)
        feats = ((sub[DAILY_COLS].to_numpy(np.float32) - mean) / std).astype(np.float32)
        close = sub["close"].to_numpy(np.float64)
        dates = pd.to_datetime(sub["date"]).to_numpy()
        for i in range(seq_len - 1, len(sub) - 1):
            d = pd.Timestamp(dates[i])
            if d < sd or d > ed:
                continue
            if close[i] <= 0 or not np.isfinite(close[i]) or not np.isfinite(close[i + 1]):
                continue
            y = close[i + 1] / close[i] - 1.0
            if not np.isfinite(y):
                continue
            win = feats[i - seq_len + 1 : i + 1]
            if np.isfinite(win).all():
                records.append((d, win, float(y)))

    if not records:
        raise RuntimeError("no samples built")

    dates = pd.Series([r[0] for r in records])
    y_raw = np.asarray([r[2] for r in records], dtype=np.float32)
    if train:
        y = _transform_labels(y_raw, dates, cfg.get("label_mode", "rank"), tuple(cfg["label_winsor"]))
    else:
        y = y_raw

    day_map: dict[pd.Timestamp, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        day_map[pd.Timestamp(d)].append(i)

    daily_x: list[np.ndarray] = []
    daily_y: list[np.ndarray] = []
    daily_dates: list[pd.Timestamp] = []
    for d in sorted(day_map):
        idxs = day_map[d]
        if len(idxs) < 16:
            continue
        daily_x.append(np.stack([records[i][1] for i in idxs], axis=0).astype(np.float32))
        daily_y.append(y[idxs].astype(np.float32))
        daily_dates.append(d)

    dataset = DailyDataset(daily_x, daily_y, daily_dates)
    log.info(
        "%s dataset days=%d samples=%d avg_stocks=%.0f",
        "train" if train else "valid",
        len(dataset),
        len(records),
        float(np.mean([len(x) for x in daily_x])) if daily_x else 0.0,
    )
    gc.collect()
    return dataset, stats


class TemporalEncoder(nn.Module):
    def __init__(self, n_feat: int, seq_len: int, d_model: int, nhead: int, nlayers: int, dim_ff: int, dropout: float, revin: int):
        super().__init__()
        self.revin = bool(revin)
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dw3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.dw7 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.conv_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_q, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp(min=1e-5)
            x = (x - mu) / sd
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        hc = h.transpose(1, 2)
        h = h + self.conv_scale * (self.dw3(hc) + self.dw7(hc)).transpose(1, 2)
        h = self.encoder(h)
        q = self.pool_q.expand(h.shape[0], -1, -1)
        pooled = self.pool_attn(q, h, h, need_weights=False)[0].squeeze(1)
        return self.norm(pooled)


class DailyOhlcvCrossTransformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        ts_layers: int,
        cs_layers: int,
        dim_ff: int,
        dropout: float,
        revin: int = 1,
    ):
        super().__init__()
        self.temporal = TemporalEncoder(n_feat, seq_len, d_model, nhead, ts_layers, dim_ff, dropout, revin)
        cs_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cs_encoder = nn.TransformerEncoder(cs_layer, cs_layers, enable_nested_tensor=False)
        self.cs_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, l, c = x.shape
        z = self.temporal(x.reshape(b * n, l, c)).reshape(b, n, -1)
        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=x.device)
        z = self.cs_encoder(z, src_key_padding_mask=~mask)
        z = self.cs_norm(z)
        return self.head(z).squeeze(-1)


def masked_ic_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for b in range(pred.shape[0]):
        valid = mask[b]
        if valid.sum() < 16:
            continue
        p = pred[b][valid]
        t = target[b][valid]
        p = p - p.mean()
        t = t - t.mean()
        cov = (p * t).mean()
        sp = p.pow(2).mean().sqrt().clamp(min=1e-8)
        st = t.pow(2).mean().sqrt().clamp(min=1e-8)
        losses.append(-(cov / (sp * st)))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for b in range(pred.shape[0]):
        valid = mask[b]
        if valid.sum() < 16:
            continue
        p = pred[b][valid]
        t = target[b][valid]
        order = torch.argsort(t, stable=True)
        ps = p[order]
        if ps.numel() >= 2:
            losses.append(F.softplus(-(ps[1:] - ps[:-1])).mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_rank_ic(model: nn.Module, dataset: DailyDataset, device: torch.device, batch_days: int) -> dict:
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_days, shuffle=False, collate_fn=collate_days, num_workers=0)
    day_pos = 0
    for xb, yb, mask in loader:
        xb = xb.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(xb, mask=mask).float().cpu().numpy()
        y_np = yb.numpy()
        mask_np = mask.cpu().numpy()
        for i in range(pred.shape[0]):
            d = dataset.daily_dates[day_pos + i]
            valid = mask_np[i]
            rows.append(pd.DataFrame({"date": d, "pred": pred[i, valid], "y": y_np[i, valid]}))
        day_pos += pred.shape[0]
    if not rows:
        return {"ic": np.nan, "std": np.nan, "ir": np.nan, "days": 0}
    df = pd.concat(rows, ignore_index=True)
    vals = []
    for _, sub in df.groupby("date", sort=True):
        corr = sub["pred"].corr(sub["y"], method="spearman")
        if np.isfinite(corr):
            vals.append(corr)
    vals = np.asarray(vals, dtype=np.float64)
    model.train()
    return {
        "ic": float(vals.mean()) if len(vals) else np.nan,
        "std": float(vals.std()) if len(vals) else np.nan,
        "ir": float(vals.mean() / (vals.std() + 1e-12)) if len(vals) else np.nan,
        "days": int(len(vals)),
    }


def _save_model(path: str | Path, model: nn.Module, model_cfg: dict, train_cfg: dict, stats: tuple[np.ndarray, np.ndarray]) -> None:
    state = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        state[k] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "data": t.reshape(-1).tolist()}
    mean, std = stats
    payload = {
        "state_dict": state,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "model_type": "daily_ohlcv_cross",
        "freq": train_cfg["freq"],
        "seq_len": train_cfg["seq_len"],
        "feature_cols": DAILY_COLS,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("saved model: %s %.1f MB", path, path.stat().st_size / 1024 / 1024)


def train(cfg: dict | None = None) -> str:
    cfg = dict(CFG if cfg is None else cfg)
    set_global_seed(int(cfg["seed"]), deterministic_torch=False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("config: %s", cfg)
    log.info("device=%s", device)

    train_ds, stats = build_dataset(
        cfg,
        load_start=cfg["train_start"],
        load_end=cfg["train_end"],
        sample_start=cfg["train_start"],
        sample_end=cfg["train_end"],
        stats=None,
        train=True,
    )

    val_ds = None
    if cfg.get("val_start") and cfg.get("val_end"):
        lookback = int(cfg["seq_len"]) * 3 + 30
        val_load_start = (pd.Timestamp(cfg["val_start"]) - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
        val_ds, _ = build_dataset(
            cfg,
            load_start=val_load_start,
            load_end=cfg["val_end"],
            sample_start=cfg["val_start"],
            sample_end=cfg["val_end"],
            stats=stats,
            train=False,
        )

    model_cfg = {
        "n_feat": len(DAILY_COLS),
        "seq_len": int(cfg["seq_len"]),
        "d_model": int(cfg["d_model"]),
        "nhead": int(cfg["nhead"]),
        "ts_layers": int(cfg["ts_layers"]),
        "cs_layers": int(cfg["cs_layers"]),
        "dim_ff": int(cfg["dim_ff"]),
        "dropout": float(cfg["dropout"]),
        "revin": int(cfg["revin"]),
    }
    model = DailyOhlcvCrossTransformer(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("params=%d %.2fM", n_params, n_params / 1e6)

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg["batch_days"]),
        shuffle=True,
        collate_fn=collate_days,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg["lr"]) * 0.01)
    use_amp = bool(int(cfg.get("amp", 1))) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    best = {"ic": -1e9, "epoch": -1, "state": None}

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        t0 = time.time()
        totals = {"loss": 0.0, "ic": 0.0, "rank": 0.0}
        n_batch = 0
        for xb, yb, mask in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(xb, mask=mask)
            ic = masked_ic_loss(pred.float(), yb.float(), mask)
            rank = pairwise_rank_loss(pred.float(), yb.float(), mask)
            loss = float(cfg["ic_weight"]) * ic + float(cfg["pairwise_weight"]) * rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["ic"] += float(ic.detach().cpu())
            totals["rank"] += float(rank.detach().cpu())
            n_batch += 1
        scheduler.step()

        val_msg = ""
        if val_ds is not None:
            vm = evaluate_rank_ic(model, val_ds, device, int(cfg["batch_days"]))
            val_msg = f" val_rank_ic={vm['ic']:.6f} val_ir={vm['ir']:.4f} val_days={vm['days']}"
            if vm["ic"] > best["ic"]:
                best = {"ic": vm["ic"], "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        log.info(
            "epoch %d/%d loss=%.6f ic=%.6f rank=%.6f lr=%.2e elapsed=%.1fs%s",
            epoch,
            int(cfg["epochs"]),
            totals["loss"] / max(n_batch, 1),
            totals["ic"] / max(n_batch, 1),
            totals["rank"] / max(n_batch, 1),
            scheduler.get_last_lr()[0],
            time.time() - t0,
            val_msg,
        )

    if val_ds is not None:
        log.info("best epoch=%d val_rank_ic=%.6f", best["epoch"], best["ic"])
        if int(cfg.get("save_best", 0)) and best["state"] is not None:
            model.load_state_dict(best["state"])
            log.info("loaded best validation state")

    output = cfg.get("output") or CFG["output"]
    _save_model(output, model, model_cfg, {k: v for k, v in cfg.items() if k != "output"}, stats)
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train daily raw OHLCV cross-sectional model")
    parser.add_argument("--freq", default=CFG["freq"], choices=["bar5m", "bar15m", "bar30m", "bar1m"])
    parser.add_argument("--train_start", default=CFG["train_start"])
    parser.add_argument("--train_end", default=CFG["train_end"])
    parser.add_argument("--seq_len", type=int, default=CFG["seq_len"])
    parser.add_argument("--d_model", type=int, default=CFG["d_model"])
    parser.add_argument("--nhead", type=int, default=CFG["nhead"])
    parser.add_argument("--ts_layers", type=int, default=CFG["ts_layers"])
    parser.add_argument("--cs_layers", type=int, default=CFG["cs_layers"])
    parser.add_argument("--dim_ff", type=int, default=CFG["dim_ff"])
    parser.add_argument("--dropout", type=float, default=CFG["dropout"])
    parser.add_argument("--batch_days", type=int, default=CFG["batch_days"])
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    parser.add_argument("--weight_decay", type=float, default=CFG["weight_decay"])
    parser.add_argument("--seed", type=int, default=CFG["seed"])
    parser.add_argument("--label_mode", default=CFG["label_mode"], choices=["raw", "zscore", "rank"])
    parser.add_argument("--pairwise_weight", type=float, default=CFG["pairwise_weight"])
    parser.add_argument("--revin", type=int, default=CFG["revin"], choices=[0, 1])
    parser.add_argument("--output", default=CFG["output"])
    parser.add_argument("--val_start", default=CFG["val_start"])
    parser.add_argument("--val_end", default=CFG["val_end"])
    parser.add_argument("--val_stride", type=int, default=CFG["val_stride"])
    parser.add_argument("--save_best", type=int, default=CFG["save_best"], choices=[0, 1])
    parser.add_argument("--amp", type=int, default=CFG["amp"], choices=[0, 1])
    parser.add_argument("--force_daily_cache", type=int, default=CFG["force_daily_cache"], choices=[0, 1])
    args = parser.parse_args()
    train({**CFG, **vars(args)})
