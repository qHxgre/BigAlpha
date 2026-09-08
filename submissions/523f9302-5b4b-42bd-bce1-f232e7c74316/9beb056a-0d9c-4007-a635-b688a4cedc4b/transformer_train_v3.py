# -*- coding: utf-8 -*-
"""Transformer v3: v2 model with low-memory BigQuant inference.

v3 keeps the v2 architecture and checkpoint JSON format.  The main change is
for cloud scoring: data is queried by monthly chunks and predictions are emitted
chunk by chunk, avoiding one huge dai.query(...).df() / build_dataset allocation.

If you already have a v2 checkpoint, you usually do not need to retrain. Copy it
as transformer_model_v3.json and upload it with this file and the v3 predict
notebook.
"""
import gc
import glob
import json
import os
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import dai  # type: ignore
except Exception:
    dai = None

try:
    import structlog  # type: ignore
    logger = structlog.get_logger()
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

LOCAL_DATA_ROOT = "/Users/wanux/Projects/bigalpha_2026_e2e/data/raw"
LOCAL_TABLE = "bigalpha_2026_e2e_bar30m"
TRAIN_START, TRAIN_END = "2019-01-01", "2023-12-31 23:59:59"
VAL_START, VAL_END = "2024-01-01", "2024-12-31 23:59:59"

SEQ_LEN = 64
EPOCHS = 40
BATCH = 384
INFER_BATCH = 1024
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.18
PATIENCE = 6
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
MAX_INSTRUMENTS = None
SEED = 42
LABEL_CLIP_Z = 3.0
QUERY_BUFFER_DAYS = 30
INFER_CHUNK_MONTHS = 1

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model_v3.json")

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS = ["volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT = len(FEATURE_COLS)
SCALE_FIELDS = ["open", "high", "low", "close", "amount", "bid_price1", "ask_price1"]
OHLC_COLS = ["open", "high", "low", "close"]
MODEL_CFG = dict(
    n_feat=N_FEAT,
    d_model=96,
    nhead=6,
    nlayers=3,
    dim_ff=192,
    seq_len=SEQ_LEN,
    dropout=DROPOUT,
)


class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model, nhead, nlayers, dim_ff, seq_len, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.in_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        h = self.in_norm(self.proj(x) + self.pos)
        h = self.encoder(self.drop(h)).mean(1)
        return self.head(h).squeeze(-1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_table(datasources):
    if isinstance(datasources, str):
        return datasources
    if not isinstance(datasources, dict):
        raise TypeError("datasources must be a dict or table name string")
    for key in ("bar30m", "bigalpha_2026_e2e_bar30m", "bar1m", "bar5m", "bar15m"):
        if key in datasources:
            return datasources[key]
    if datasources:
        return next(iter(datasources.values()))
    raise ValueError("datasources is empty")


def pool(sd, ed):
    if dai is None:
        raise RuntimeError("dai is required to query bigalpha_2026_instruments")
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return df["instrument"].tolist()


def query_official_pool(sd, ed):
    if dai is None:
        raise RuntimeError("dai is required to query bigalpha_2026_instruments")
    df = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.drop_duplicates(["date", "instrument"])


def _maybe_scale_compressed_prices(df):
    cols = [c for c in SCALE_FIELDS if c in df.columns]
    if not cols:
        return df
    probe = pd.to_numeric(df[cols[0]], errors="coerce").replace([-1, np.inf, -np.inf], np.nan)
    med = probe.abs().median()
    if np.isfinite(med) and med > 1000:
        df.loc[:, cols] = df.loc[:, cols] / 100.0
    return df


def to_canonical(df):
    df = df.copy()
    if "instrument" not in df.columns:
        if "instrument_id" in df.columns:
            df["instrument"] = df["instrument_id"]
        elif "key" in df.columns:
            df["instrument"] = df["key"]
        else:
            raise KeyError("input data needs instrument or instrument_id")
    df["date"] = pd.to_datetime(df["date"])
    for c in OHLC_COLS:
        if c in df.columns:
            df.loc[df[c] == -1, c] = np.nan
    df = _maybe_scale_compressed_prices(df)
    for c in VOL_COLS:
        df[c] = np.log1p(df[c].clip(lower=0))
    df = df.sort_values(["instrument", "date"])
    for c in OHLC_COLS:
        df[c] = df.groupby("instrument", sort=False)[c].ffill()
    return df[["date", "instrument"] + FEATURE_COLS]


def _standardize_labels_by_date(y, idx_df):
    s = pd.Series(y, index=idx_df.index, dtype="float32")
    dates = pd.to_datetime(idx_df["date"]).dt.normalize()
    mu = s.groupby(dates).transform("mean")
    sig = s.groupby(dates).transform("std").replace(0, np.nan)
    z = ((s - mu) / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.clip(-LABEL_CLIP_Z, LABEL_CLIP_Z).astype("float32").to_numpy()


def build_windows(df, sd, ed, mode="infer", stats=None):
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    wins, ys, keys = [], [], []
    for ins, sub in df.groupby("instrument", sort=False):
        if len(sub) <= SEQ_LEN:
            continue
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        eod = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        close_px = sub["close"].to_numpy(np.float64)[eod]
        dates = day[eod]
        for j, p in enumerate(eod):
            d = pd.Timestamp(dates[j])
            if p + 1 < SEQ_LEN or d < sd_ts or d > ed_ts:
                continue
            win = feats[p - SEQ_LEN + 1 : p + 1]
            if not np.isfinite(win).all():
                continue
            label = None
            if j + 1 < len(eod) and close_px[j] > 0:
                r = close_px[j + 1] / close_px[j] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue
            wins.append(win)
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))
    if not keys:
        return None, None, None, stats

    X = np.stack(wins).astype(np.float32)
    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        stats = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32) + 1e-6)
    mean, std = stats
    X = ((X - mean) / std).astype(np.float32)
    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])
    if mode == "train":
        y = _standardize_labels_by_date(np.asarray(ys, np.float32), idx_df)
        return X, y, None, stats
    return X, None, idx_df, stats


def _read_local_feathers(sd, ed, local_data_root=LOCAL_DATA_ROOT, local_table=LOCAL_TABLE):
    root = os.path.join(local_data_root, local_table)
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=QUERY_BUFFER_DAYS)).strftime("%Y%m")
    lo, hi = int(buf), int(pd.Timestamp(ed).strftime("%Y%m"))
    need = set(FEATURE_COLS) | {"date", "instrument_id"}
    parts = [
        pd.read_feather(fp, columns=list(need))
        for fp in sorted(glob.glob(os.path.join(root, "*.feather")))
        if os.path.basename(fp).split(".")[0].isdigit()
        and lo <= int(os.path.basename(fp).split(".")[0]) <= hi
    ]
    if not parts:
        raise FileNotFoundError(f"no local feather files found under {root} for {sd}~{ed}")
    return pd.concat(parts, ignore_index=True)


def _query_dai_table(table, sd, ed, instruments=None):
    if dai is None:
        raise RuntimeError("dai is not available; pass a DataFrame or use local feather training")
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=QUERY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table}"
    filters = {"date": [buf, ed]}
    if instruments is not None:
        filters["instrument"] = instruments
    return dai.query(sql, filters=filters).df()


def build_dataset(source=None, sd=None, ed=None, mode="infer", instruments=None, stats=None):
    if sd is None or ed is None:
        raise ValueError("sd and ed are required")
    t0 = time.time()
    if isinstance(source, pd.DataFrame):
        raw = source
    elif isinstance(source, str):
        raw = _query_dai_table(source, sd, ed, instruments)
    elif source is None:
        raw = _read_local_feathers(sd, ed)
        if instruments is not None and "instrument_id" in raw.columns:
            raw = raw[raw["instrument_id"].isin(instruments)]
    else:
        raise TypeError("source must be a table name, DataFrame, or None")
    df = to_canonical(raw)
    if instruments is not None:
        df = df[df["instrument"].isin(instruments)]
    result = build_windows(df, sd, ed, mode=mode, stats=stats)
    if result[0] is None:
        raise RuntimeError(f"build_dataset found no samples (mode={mode}, {sd}~{ed})")
    try:
        logger.info("dataset built", mode=mode, samples=len(result[0]), elapsed=round(time.time() - t0, 2))
    except TypeError:
        logger.info("dataset built: mode=%s samples=%s elapsed=%.2f", mode, len(result[0]), time.time() - t0)
    return result


def month_chunks(sd, ed, chunk_months=INFER_CHUNK_MONTHS):
    sd_ts = pd.Timestamp(sd).normalize()
    ed_ts = pd.Timestamp(ed)
    cur = sd_ts
    while cur <= ed_ts:
        nxt = cur + pd.DateOffset(months=chunk_months)
        chunk_end = min(nxt - pd.Timedelta(seconds=1), ed_ts)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d %H:%M:%S")
        cur = nxt.normalize()


def iter_infer_datasets(table, sd, ed, instruments=None, stats=None, chunk_months=INFER_CHUNK_MONTHS):
    """Yield (X, idx_df) by calendar chunks to reduce scoring memory."""
    for chunk_sd, chunk_ed in month_chunks(sd, ed, chunk_months):
        t0 = time.time()
        try:
            raw = _query_dai_table(table, chunk_sd, chunk_ed, instruments)
            df = to_canonical(raw)
            if instruments is not None:
                df = df[df["instrument"].isin(instruments)]
            X, _, idx_df, _ = build_windows(df, chunk_sd, chunk_ed, mode="infer", stats=stats)
        finally:
            raw = None
            df = None
            gc.collect()
        if X is None or idx_df is None or len(idx_df) == 0:
            logger.info("infer chunk skipped", start=chunk_sd, end=chunk_ed)
            continue
        try:
            logger.info("infer chunk built", start=chunk_sd, end=chunk_ed, samples=len(idx_df), elapsed=round(time.time() - t0, 2))
        except TypeError:
            logger.info("infer chunk built: %s~%s samples=%s elapsed=%.2f", chunk_sd, chunk_ed, len(idx_df), time.time() - t0)
        yield X, idx_df
        del X, idx_df
        gc.collect()


def predict_in_chunks(model, table, start_date, end_date, stats, device, instruments=None, batch_size=INFER_BATCH):
    pieces = []
    with torch.no_grad():
        for X, idx_df in iter_infer_datasets(table, start_date, end_date, instruments, stats):
            preds = []
            X_t = torch.from_numpy(X)
            for i in range(0, len(X_t), batch_size):
                xb = X_t[i : i + batch_size].to(device)
                preds.append(model(xb).cpu().numpy())
            out = idx_df.copy()
            out["score"] = np.concatenate(preds).astype(np.float64)
            pieces.append(out)
            del X_t, preds, out
            gc.collect()
    if not pieces:
        raise RuntimeError("no inference samples produced")
    result = pd.concat(pieces, ignore_index=True)
    del pieces
    gc.collect()
    return result


def save_model(ckpt, model_path=MODEL_PATH):
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
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


def _make_loader(X, y, device, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def _eval_loss(model, loader, loss_fn, device):
    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            total += loss_fn(model(xb), yb).item()
            nb += 1
    return total / max(nb, 1)


def train_and_save(datasources=None, model_path=MODEL_PATH):
    """Optional retraining. OOM fix does not require retraining existing v2 weights."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print("设备:", device)

    if datasources is None:
        source = None
        train_instruments = None
        val_instruments = None
    else:
        source = select_table(datasources)
        train_instruments = pool(TRAIN_START, TRAIN_END)
        val_instruments = pool(VAL_START, VAL_END) if VAL_START and VAL_END else None
        if MAX_INSTRUMENTS:
            train_instruments = train_instruments[:MAX_INSTRUMENTS]
            if val_instruments is not None:
                val_instruments = val_instruments[:MAX_INSTRUMENTS]

    Xtr, ytr, _, stats = build_dataset(source, TRAIN_START, TRAIN_END, "train", train_instruments, None)
    print("训练样本数:", len(ytr))

    has_val = bool(VAL_START and VAL_END)
    if has_val:
        Xva, yva, _, _ = build_dataset(source, VAL_START, VAL_END, "train", val_instruments, stats)
        print("验证样本数:", len(yva))
    else:
        Xva = yva = None

    model = StockTransformer(**MODEL_CFG).to(device)
    print("可训练参数量:", sum(p.numel() for p in model.parameters()))
    train_loader = _make_loader(Xtr, ytr, device, shuffle=True)
    val_loader = _make_loader(Xva, yva, device, shuffle=False) if has_val else None

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_state, best_metric, bad_epochs = None, float("inf"), 0
    for ep in range(EPOCHS):
        t, total, nb = time.time(), 0.0, 0
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += loss.item()
            nb += 1
        train_loss = total / max(nb, 1)
        metric = _eval_loss(model, val_loader, loss_fn, device) if has_val else train_loss
        scheduler.step(metric)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {ep + 1}/{EPOCHS} train={train_loss:.6f} valid={metric:.6f} lr={lr:.2e} {time.time() - t:.1f}s")
        if metric + MIN_DELTA < best_metric:
            best_metric = metric
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if has_val and bad_epochs >= PATIENCE:
                print(f"早停: best_valid={best_metric:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    mean, std = stats
    save_model(
        {
            "state_dict": model.state_dict(),
            "model_cfg": MODEL_CFG,
            "feature_cols": FEATURE_COLS,
            "seq_len": SEQ_LEN,
            "mean": np.asarray(mean, np.float32).tolist(),
            "std": np.asarray(std, np.float32).tolist(),
        },
        model_path,
    )
    print("模型已保存:", model_path)
    return model_path


if __name__ == "__main__":
    train_and_save()
