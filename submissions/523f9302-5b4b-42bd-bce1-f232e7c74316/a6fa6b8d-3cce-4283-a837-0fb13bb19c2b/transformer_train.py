# -*- coding: utf-8 -*-
"""Transformer 训练脚本 —— 30分钟K线，与 predict.ipynb 保持一致。"""
import os, json, time, tempfile
import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ======================== 配置 ========================
DATA_TABLE = "bigalpha_2026_stock_bar30m"

TRAIN_START = "2020-01-01"
TRAIN_END   = "2023-12-31 23:59:59"
VAL_START   = "2023-01-01"
VAL_END     = "2023-06-30 23:59:59"

SEQ_LEN = 80
EPOCHS  = 20
BATCH   = 512
LR      = 1e-3
SEED    = 42
MAX_TRAIN_STOCKS = 300

PRICE_COLS = ["pre_close", "open", "high", "low", "close"]
ASK_PX = [f"ask_price{i}" for i in range(1, 6)]
BID_PX = [f"bid_price{i}" for i in range(1, 6)]
ASK_VOL = [f"ask_volume{i}" for i in range(1, 6)]
BID_VOL = [f"bid_volume{i}" for i in range(1, 6)]
ASK_NORD = [f"ask_num_orders{i}" for i in range(1, 6)]
BID_NORD = [f"bid_num_orders{i}" for i in range(1, 6)]
TRADE_COLS = ["volume", "amount", "deal_number"]

FEATURE_COLS = PRICE_COLS + ASK_PX + BID_PX + ASK_VOL + BID_VOL + ASK_NORD + BID_NORD + TRADE_COLS
LOG_COLS = ASK_VOL + BID_VOL + ASK_NORD + BID_NORD + TRADE_COLS
N_FEAT = len(FEATURE_COLS)

MODEL_CFG = dict(
    n_feat=N_FEAT, d_model=128, nhead=8, nlayers=4,
    dim_ff=512, seq_len=SEQ_LEN, dropout=0.1,
)


class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=128, nhead=8, nlayers=4,
                 dim_ff=512, seq_len=SEQ_LEN, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        B = x.size(0)
        h = self.encoder(self.proj(x) + self.pos[:, :x.size(1), :])
        q = self.query.expand(B, -1, -1)
        out, _ = self.cross_attn(q, h, h)
        return self.head(out.squeeze(1)).squeeze(-1)


def pool(sd, ed, max_stocks=None):
    stocks = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]}, compression=True,
    ).df()["instrument"].tolist()
    return stocks[:max_stocks] if max_stocks else stocks


def _extract_stocks(table, buf_start, ed, instruments, sd_ts, ed_ts, mode):
    cols = ", ".join(FEATURE_COLS)
    sql = f"SELECT date, instrument, {cols} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf_start, ed], "instrument": instruments},
                   compression=True).df()
    if df.empty:
        return [], [], []
    for c in LOG_COLS:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    wins, ys_l, keys_l = [], [], []
    for ins, sub in df.groupby("instrument", sort=False, observed=True):
        if len(sub) <= SEQ_LEN:
            continue
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day_arr = sub["date"].dt.normalize().to_numpy()
        close_idx = np.flatnonzero(np.append(day_arr[1:] != day_arr[:-1], True))
        px_arr = sub["close"].to_numpy(np.float64)
        day_px = px_arr[close_idx]
        dates = day_arr[close_idx]
        for k, p in enumerate(close_idx):
            if p + 1 < SEQ_LEN:
                continue
            d = pd.Timestamp(dates[k])
            if d < sd_ts or d > ed_ts:
                continue
            if mode == "train":
                if not (k + 1 < len(close_idx) and day_px[k] > 0):
                    continue
                r = day_px[k + 1] / day_px[k] - 1.0
                if not np.isfinite(r):
                    continue
                ys_l.append(np.float32(r))
            else:
                if k + 1 < len(close_idx) and day_px[k] > 0:
                    r = day_px[k + 1] / day_px[k] - 1.0
                    ys_l.append(np.float32(r) if np.isfinite(r) else np.float32(np.nan))
                else:
                    ys_l.append(np.float32(np.nan))
            wins.append(feats[p - SEQ_LEN + 1: p + 1])
            keys_l.append((d, ins))
    del df
    return wins, ys_l, keys_l if wins else ([], [], [])


def build_dataset(table, sd, ed, mode, instruments, stats=None):
    t0 = time.time()
    buf_start = (pd.to_datetime(sd) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    CHUNK = 50
    tmpd = tempfile.mkdtemp(prefix="ds_")
    files = []
    total = 0
    all_ys = []
    all_keys = []
    for i in range(0, len(instruments), CHUNK):
        batch = instruments[i:i + CHUNK]
        wins, yl, kl = _extract_stocks(table, buf_start, ed, batch, sd_ts, ed_ts, mode)
        if not wins:
            continue
        Xb = np.stack(wins).astype(np.float32)
        del wins
        fname = os.path.join(tmpd, f"{i:04d}.npy")
        np.save(fname, Xb)
        files.append((fname, len(Xb)))
        total += len(Xb)
        all_ys.extend(yl)
        all_keys.extend(kl)
        del Xb, yl, kl
    if total == 0:
        raise RuntimeError(f"[{mode}] 无样本")

    mem_path = os.path.join(tmpd, "X.dat")
    shape = (total, SEQ_LEN, N_FEAT)
    X = np.memmap(mem_path, dtype=np.float32, mode="w+", shape=shape)
    off = 0
    for fname, n in sorted(files):
        X[off:off + n] = np.load(fname)
        off += n
        os.remove(fname)

    y = np.array(all_ys, np.float32) if all_ys else None

    if stats is None and mode == "train":
        fsum = np.zeros(N_FEAT, np.float64)
        fsq = np.zeros(N_FEAT, np.float64)
        bs = 2000
        for i in range(0, total, bs):
            batch = X[i:i+bs].reshape(-1, N_FEAT)
            fsum += batch.mean(0)
            fsq += (batch ** 2).mean(0)
        mean = fsum / (total // bs + 1)
        var = fsq / (total // bs + 1) - mean ** 2
        stats = (mean.astype(np.float32), np.sqrt(np.maximum(var, 0)).astype(np.float32) + 1e-6)

    if stats:
        m, s = stats
        bs = 5000
        for i in range(0, total, bs):
            X[i:i+bs] -= m
            X[i:i+bs] /= s

    idx_df = pd.DataFrame(all_keys, columns=["date", "instrument"])
    if y is not None:
        idx_df["label"] = y
    return X, y, idx_df, stats, tmpd


def save_model(ckpt, model_path=MODEL_PATH):
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "data": t.reshape(-1).tolist()}
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


def compute_ic(scores_df):
    g = scores_df.groupby("date", sort=False)
    ics = []
    for _, grp in g:
        grp = grp.dropna(subset=["score", "label"])
        if len(grp) < 10:
            continue
        ic = grp["score"].corr(grp["label"], method="spearman")
        if np.isfinite(ic):
            ics.append(ic)
    return np.mean(ics) if ics else 0.0, ics


def train_and_save(datasources, model_path=MODEL_PATH):
    table = DATA_TABLE  # 固定用 30m K 线
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练", device=str(device), table=table)

    train_pool = pool(TRAIN_START, TRAIN_END, MAX_TRAIN_STOCKS)
    val_pool = pool(VAL_START, VAL_END, max(len(train_pool) // 4, 50))
    logger.info("标的", train=len(train_pool), val=len(val_pool))

    Xtr, ytr, _, stats, _ = build_dataset(table, TRAIN_START, TRAIN_END, "train", train_pool)
    lo, hi = np.percentile(ytr, [1, 99])
    ytr = np.clip(ytr, lo, hi)

    Xva, _, va_idx, _, _ = build_dataset(table, VAL_START, VAL_END, "infer", val_pool, stats=stats)

    model = StockTransformer(**MODEL_CFG)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        pin_memory=True, drop_last=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    best_ic = -99.0
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            Xva_t = torch.from_numpy(Xva)
            ps = []
            for i in range(0, len(Xva_t), BATCH * 2):
                ps.append(model(Xva_t[i:i+BATCH*2].to(device)).cpu().numpy())
            va_idx["score"] = np.concatenate(ps).astype(np.float64)
        val_ic, _ = compute_ic(va_idx)
        print(f"ep {ep+1:2d} | loss={tot/max(nb,1):.6f} | IC={val_ic:.4f} | {time.time()-t0:.0f}s")
        if val_ic > best_ic:
            best_ic = val_ic

    print(f"\nbest_IC={best_ic:.4f} | {n_params:,} params")

    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    mean_arr, std_arr = stats
    save_model({
        "state_dict": sd, "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS, "log_cols": LOG_COLS, "seq_len": SEQ_LEN,
        "mean": np.asarray(mean_arr, np.float32).tolist(),
        "std": np.asarray(std_arr, np.float32).tolist(),
    }, model_path)
    logger.info("已保存", path=model_path)
    return model_path


if __name__ == "__main__":
    train_and_save({})
