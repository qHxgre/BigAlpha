# -*- coding: utf-8 -*-
"""Self-contained five-level C2C platform trainer.

The concrete arm is selected by ``package_config.json``.  Supported variants:
``base``, ``aux``, ``shrink05``, ``shrink05_aux``, ``sam``, ``price_trade`` and ``ssl``.
All arms keep the original adjusted C[T]→C[T+1] validation target, original
training boundary and PatchTST recipe.  Seeds and checkpoint count are fixed by
the package config; the raw-C2C validation composite selects checkpoints once
after training.  Scores are daily z-normalized, equally averaged and rank-gaussed.
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
try:
    import dai
except ModuleNotFoundError:
    from bigquant import dai
from scipy.stats import norm
import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger()
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "package_config.json"), encoding="utf-8") as f:
    PKG = json.load(f)

BAR_KEY = "bar1m"
SEQ_LEN = 120
FEATNORM = "logrel"
CONFIG_TAG = PKG["config_tag"]
VARIANT = PKG["variant"]
SUPPORTED_VARIANTS = {"base", "aux", "shrink05", "shrink05_aux", "sam", "price_trade", "ssl"}
if VARIANT not in SUPPORTED_VARIANTS:
    raise ValueError(f"unsupported variant: {VARIANT!r}")
ARM = str(PKG.get("arm", "control"))
if ARM not in {"control", "robust_nonprice"}:
    raise ValueError(f"unsupported arm: {ARM!r}")
IS_C6 = "arm" in PKG
USES_AUX = VARIANT in {"aux", "shrink05_aux"}
USES_SHRINK = VARIANT in {"shrink05", "shrink05_aux"}
USES_SAM = VARIANT == "sam"
SEEDS = [int(x) for x in PKG.get("seeds", [42, 123, 2024])]
TOP_K = int(PKG.get("top_k", 3))
EPOCHS = 30
FIXED_EPOCHS = [int(x) for x in PKG.get("fixed_epochs", [])]
if (not SEEDS or TOP_K < 1 or (FIXED_EPOCHS and len(FIXED_EPOCHS) != TOP_K)
        or any(epoch < 1 or epoch > EPOCHS for epoch in FIXED_EPOCHS)):
    raise ValueError("package_config requires valid seeds/top_k/fixed_epochs")
RUN_EPOCHS = (EPOCHS if PKG.get("run_full_schedule", False)
              else max(FIXED_EPOCHS) if FIXED_EPOCHS else EPOCHS)
SSL_EPOCHS = int(PKG.get("ssl_epochs", 5))
LR = 5e-4
AUX_WEIGHT = float(PKG.get("aux_weight", 0.2))
SHRINK_ALPHA = float(PKG.get("shrink_alpha", 0.5))
SAM_RHO = 0.05
WARMUP_FRAC = 0.08
TRAIN_START, TRAIN_END = "2019-07-01", "2024-12-31 23:59:59"
VAL_START, VAL_END = None, None
FULL_TRAIN = bool(PKG.get("full_train_no_test", False))
LOCAL_DATA_DIR = os.environ.get("BIGALPHA_LOCAL_DATA_DIR",
                                os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

PRICE_COLS = ["open", "high", "low", "close",
              "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5",
              "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5"]
VOL_COLS = ["deal_number", "volume", "amount",
            "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5",
            "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",
            "ask_num_orders1", "ask_num_orders2", "ask_num_orders3", "ask_num_orders4", "ask_num_orders5",
            "bid_num_orders1", "bid_num_orders2", "bid_num_orders3", "bid_num_orders4", "bid_num_orders5"]
FEATURE_COLS = PRICE_COLS + VOL_COLS
if len(FEATURE_COLS) != 37 or len(PRICE_COLS) != 14 or len(VOL_COLS) != 23:
    raise RuntimeError("five-level package feature contract is broken")
ACTIVE_IDX = ([0, 1, 2, 3, 10, 11, 12] if VARIANT == "price_trade" else list(range(37)))
ACTIVE_FEATURE_COLS = [FEATURE_COLS[i] for i in ACTIVE_IDX]
N_PRICE = len(PRICE_COLS)
CLOSE_IDX = PRICE_COLS.index("close")
if USES_SHRINK and not np.isclose(SHRINK_ALPHA, 0.5):
    raise ValueError("shrink05 variants require shrink_alpha=0.5")
if IS_C6 and (VARIANT != "shrink05_aux" or len(SEEDS) * TOP_K != 1):
    raise ValueError("C6 requires shrink05_aux and exactly one seed/checkpoint")
if ARM == "robust_nonprice" and (
        ACTIVE_IDX != list(range(len(FEATURE_COLS)))
        or ACTIVE_FEATURE_COLS[N_PRICE:] != VOL_COLS):
    raise ValueError("robust_nonprice requires the canonical 10 price + 15 non-price layout")
INPUT_PREPROCESS = (
    {"version": 1, "stage": "post_standardize", "kind": "clamp",
     "slice": [N_PRICE, len(FEATURE_COLS)], "bounds": [-3.0, 3.0],
     "cols": list(VOL_COLS)}
    if ARM == "robust_nonprice"
    else {"version": 1, "stage": "post_standardize", "kind": "identity"}
)
MODEL_CFG = dict(n_feat=len(ACTIVE_IDX), d_model=128, nhead=4, nlayers=3,
                 dim_ff=256, seq_len=SEQ_LEN, patch_len=16, stride=8, dropout=0.1)


def model_path(seed, k):
    return os.path.join(_HERE, f"model_s{seed}_k{k}.json")


class PatchEmbedding(nn.Module):
    def __init__(self, n_feat, patch_len=16, stride=8, d_model=128):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.proj = nn.Linear(patch_len * n_feat, d_model)

    def forward(self, x):
        b, length, n_feat = x.shape
        if length < self.patch_len:
            raise ValueError(f"seq_len={length} < patch_len={self.patch_len}")
        p = x.unfold(1, self.patch_len, self.stride)
        p = p.transpose(-1, -2).contiguous()
        return self.proj(p.reshape(b, p.size(1), self.patch_len * n_feat))


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                   nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(1)


class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=128, nhead=4, nlayers=3, dim_ff=256,
                 seq_len=120, patch_len=16, stride=8, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_feat, patch_len, stride, d_model)
        n_patch = 1 + (seq_len - patch_len) // stride
        self.pos = nn.Parameter(torch.zeros(1, n_patch, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.pool = AttentionPooling(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, 1))

    def forward(self, x):
        h = self.patch_embed(x)
        h = self.encoder(h + self.pos[:, :h.size(1), :])
        return self.head(self.pool(h)).squeeze(-1)


class AuxStockTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.base = StockTransformer(**cfg)
        d = cfg["d_model"]
        self.aux_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                      nn.Dropout(cfg["dropout"]), nn.Linear(d, 1))

    def forward(self, x):
        b = self.base
        h = b.patch_embed(x)
        h = b.encoder(h + b.pos[:, :h.size(1), :])
        z = b.pool(h)
        return b.head(z).squeeze(-1), self.aux_head(z).squeeze(-1)


class MaskedPatchAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbedding(cfg["n_feat"], cfg["patch_len"], cfg["stride"], cfg["d_model"])
        n_patch = 1 + (cfg["seq_len"] - cfg["patch_len"]) // cfg["stride"]
        self.pos = nn.Parameter(torch.zeros(1, n_patch, cfg["d_model"]))
        layer = nn.TransformerEncoderLayer(cfg["d_model"], cfg["nhead"], cfg["dim_ff"],
                                           cfg["dropout"], batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["nlayers"])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg["d_model"]))
        self.decoder = nn.Linear(cfg["d_model"], cfg["patch_len"] * cfg["n_feat"])
        self.patch_len, self.stride = cfg["patch_len"], cfg["stride"]

    def forward(self, x, ratio=0.30):
        target = x.unfold(1, self.patch_len, self.stride)
        target = target.transpose(-1, -2).contiguous().reshape(x.size(0), target.size(1), -1)
        h = self.patch_embed(x)
        mask = torch.rand(h.shape[:2], device=h.device) < ratio
        mask[:, 0] = True
        h = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(h), h)
        h = self.encoder(h + self.pos[:, :h.size(1), :])
        return self.decoder(h), target, mask


class SAM:
    def __init__(self, params, lr, rho=0.05):
        self.params = list(params)
        self.rho = rho
        self.base_optimizer = torch.optim.Adam(self.params, lr=lr)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        norms = [p.grad.norm(p=2) for p in self.params if p.grad is not None]
        grad_norm = torch.norm(torch.stack(norms), p=2)
        scale = self.rho / (grad_norm + 1e-12)
        for p in self.params:
            if p.grad is None:
                continue
            p._sam_e = p.grad * scale
            p.add_(p._sam_e)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for p in self.params:
            if hasattr(p, "_sam_e"):
                p.sub_(p._sam_e)
                del p._sam_e
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()


def _is_local(table):
    return "e2e" in table


def _local_path(table):
    return os.path.join(LOCAL_DATA_DIR, f"{table.split('_e2e_')[-1]}.parquet")


def _read_raw(table, sd, ed, instruments):
    cols = ["date", "adjust_factor"] + FEATURE_COLS
    if _is_local(table):
        raise RuntimeError("5-level package requires platform stock_bar1m; local e2e data has only 3 levels")
        flt = [("date", ">=", pd.Timestamp(sd)), ("date", "<=", pd.Timestamp(ed))]
        if instruments is not None:
            flt.append(("instrument_id", "in", list(instruments)))
        df = pd.read_parquet(_local_path(table), columns=["instrument_id"] + cols, filters=flt)
        df = df.rename(columns={"instrument_id": "instrument"})
        return df.sort_values(["instrument", "date"], kind="stable", ignore_index=True)
    sql = (f"SELECT date, instrument, adjust_factor, {', '.join(FEATURE_COLS)} "
           f"FROM {table} ORDER BY instrument, date")
    return dai.query(sql, filters={"date": [sd, ed], "instrument": list(instruments)}).df()


def _to_canonical(df, is_local):
    if is_local:
        for c in PRICE_COLS:
            v = df[c].astype(np.float32)
            df[c] = np.where(v < 0, np.nan, v / 100.0).astype(np.float32)
        amount = df["amount"].astype(np.float64)
        df["amount"] = np.where(amount < 0, np.nan, amount / 100.0)
    for c in PRICE_COLS:
        df[c] = (df[c] * df["adjust_factor"]).astype(np.float32)
    df[VOL_COLS] = df[VOL_COLS].clip(lower=0).fillna(0.0)
    for c in VOL_COLS:
        df[c] = np.log1p(df[c]).astype(np.float32)
    return df


def pool(sd, ed, table=None):
    if table is not None and _is_local(table):
        ids = pd.read_parquet(_local_path(table), columns=["instrument_id"])["instrument_id"].unique()
        return sorted(int(x) for x in ids)
    df = dai.query("SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
                   filters={"date": [sd, ed]}).df()
    return df["instrument"].tolist()


def build_dataset(table, sd, ed, mode, instruments, stats=None, label_end=None, query_batch=300):
    is_local = _is_local(table)
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    sd64 = np.datetime64(pd.Timestamp(sd).normalize())
    ed64 = np.datetime64(pd.Timestamp(ed).normalize())
    le64 = np.datetime64(pd.Timestamp(label_end).normalize()) if label_end is not None else None
    span = np.arange(SEQ_LEN)
    need_label = mode in ("train", "val")
    wins, cc_all, on_all, shrink_all, valid_all, kd, ki = [], [], [], [], [], [], []
    for bi in range(0, len(instruments), query_batch):
        df = _to_canonical(_read_raw(table, buf, ed, instruments[bi:bi + query_batch]), is_local)
        for ins, sub in df.groupby("instrument", sort=False):
            if len(sub) <= SEQ_LEN:
                continue
            sub = sub.copy()
            day = sub["date"].dt.normalize().to_numpy()
            # Original top1 feature/close semantics: price columns ffill by instrument.
            sub[PRICE_COLS] = sub[PRICE_COLS].ffill()
            feats = sub[FEATURE_COLS].to_numpy(np.float32)
            close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
            dates = day[close_pos]
            if need_label:
                open_raw = sub["open"].to_numpy(np.float64).copy()
                close_px = sub["close"].to_numpy(np.float64)[close_pos]
                open_map = pd.Series(open_raw).groupby(day, sort=False).first()
                open_px = open_map.reindex(dates).to_numpy(np.float64)
                n = len(dates)
                cc = np.full(n, np.nan, np.float32)
                on = np.full(n, np.nan, np.float32)
                shrink = np.full(n, np.nan, np.float32)
                valid = np.zeros(n, bool)
                if n >= 2:
                    c0, c1, o1 = close_px[:-1], close_px[1:], open_px[1:]
                    ok_cc = (c0 > 0) & (c1 > 0) & np.isfinite(c0) & np.isfinite(c1)
                    ok_comp = ok_cc & (o1 > 0) & np.isfinite(o1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cc_r = c1 / c0 - 1.0
                        on_r = np.log(o1 / c0)
                        id_r = np.log(c1 / o1)
                    cc_r[~ok_cc] = np.nan
                    on_r[~ok_comp] = np.nan
                    shr = on_r + SHRINK_ALPHA * id_r
                    shr[~ok_comp] = np.log1p(np.clip(cc_r[~ok_comp], -0.999999, None))
                    if le64 is not None:
                        blocked = dates[1:] >= le64
                        cc_r[blocked] = np.nan; on_r[blocked] = np.nan; shr[blocked] = np.nan
                        ok_comp[blocked] = False
                    cc[:-1] = cc_r; on[:-1] = on_r; shrink[:-1] = shr; valid[:-1] = ok_comp
            mask = (close_pos >= SEQ_LEN - 1) & (dates >= sd64) & (dates <= ed64)
            if need_label:
                mask &= np.isfinite(cc)
            sel = np.flatnonzero(mask)
            if not len(sel):
                continue
            rows = (close_pos[sel] - SEQ_LEN + 1)[:, None] + span
            wins.append(feats[rows])
            if need_label:
                cc_all.append(np.nan_to_num(cc[sel]).astype(np.float32))
                on_all.append(on[sel].astype(np.float32))
                shrink_all.append(shrink[sel].astype(np.float32))
                valid_all.append(valid[sel])
            kd.append(dates[sel]); ki.append(np.full(len(sel), ins, dtype=object))
        del df
    if not wins:
        raise RuntimeError(f"no samples: {mode} {sd} {ed}")
    X = np.concatenate(wins)
    del wins
    px = X[:, :, :N_PRICE]
    ref = np.clip(X[:, SEQ_LEN - 1, CLOSE_IDX], 1e-6, None)
    X[:, :, :N_PRICE] = (np.log(np.clip(px, 1e-6, None)) - np.log(ref)[:, None, None]).astype(np.float32)
    if ACTIVE_IDX != list(range(len(ACTIVE_IDX))):
        X = X[:, :, ACTIVE_IDX]
    if stats is None:
        flat = X.reshape(-1, len(ACTIVE_IDX))
        stats = (np.nanmean(flat, 0).astype(np.float32),
                 np.nanstd(flat, 0).astype(np.float32) + 1e-6)
    mean, std = stats
    X -= mean
    X /= std
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X = _post_standardize(X, copy=False)
    idx = pd.DataFrame({"date": np.concatenate(kd), "instrument": np.concatenate(ki)})
    if mode == "infer":
        return X, None, idx, stats
    labels = {"cc": np.concatenate(cc_all), "on": np.concatenate(on_all),
              "shrink05": np.concatenate(shrink_all), "component_valid": np.concatenate(valid_all)}
    return X, labels, idx, stats


def _post_standardize(x, copy=True):
    """Apply C6 after train-stat normalization; copy shared inputs by default."""
    if ARM == "control":
        return x
    if x.ndim != 3 or x.shape[-1] != len(FEATURE_COLS):
        raise ValueError(f"robust_nonprice expects [N,T,25], got {x.shape}")
    out = x.copy() if copy else x
    nonprice = out[..., N_PRICE:]
    np.clip(nonprice, -3.0, 3.0, out=nonprice)
    return out


def _cs_zscore(y, dates, clip=3.0):
    s = pd.DataFrame({"d": dates, "y": np.asarray(y, np.float64)})
    g = s.groupby("d")["y"]
    z = (s["y"] - g.transform("mean")) / (g.transform("std") + 1e-8)
    return z.clip(-clip, clip).fillna(0.0).to_numpy(np.float32)


def _loss_ic(pred, target):
    pc, tc = pred - pred.mean(), target - target.mean()
    return 1.0 - (pc * tc).sum() / (pc.norm() * tc.norm() + 1e-8)


def _day_groups(dates):
    order = np.argsort(dates, kind="stable")
    ds = dates[order]
    bounds = np.flatnonzero(np.append(ds[1:] != ds[:-1], True))
    groups, start = [], 0
    for b in bounds:
        groups.append(order[start:b + 1]); start = b + 1
    return groups


def _main(out):
    return out[0] if isinstance(out, tuple) else out


def _metrics(dates, pred, label):
    df = pd.DataFrame({"d": dates, "p": pred, "y": label})
    ics, ls = [], []
    for _, g in df.groupby("d", sort=True):
        ics.append(g["p"].corr(g["y"], method="spearman"))
        r = g["p"].rank(pct=True)
        ls.append(g.loc[r >= .9, "y"].mean() - g.loc[r <= .1, "y"].mean())
    ic = np.asarray(ics, float); lsv = np.asarray(ls, float)
    segs = np.array_split(ic, 3)
    return {"ic": float(np.nanmean(ic)),
            "icir": float(np.nanmean(ic) / (np.nanstd(ic, ddof=1) + 1e-12)),
            "sr": float(np.nanmean(lsv) / (np.nanstd(lsv, ddof=1) + 1e-12) * np.sqrt(252)),
            "stress": float(min(np.nanmean(x) for x in segs))}


def _composite(hist):
    m = np.array([[r[k] for k in ("ic", "icir", "sr", "stress")] for r in hist])
    return np.mean([pd.Series(m[:, j]).rank(pct=True).to_numpy() for j in range(4)], axis=0)


def _sched(opt, total):
    warm = max(1, int(WARMUP_FRAC * total))
    def fn(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return .5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def _predict(model, X, device, batch=1024):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        for j in range(0, len(X), batch):
            out.append(_main(model(torch.from_numpy(X[j:j + batch]).to(device))).float().cpu().numpy())
    return np.concatenate(out)


def _make_model(device):
    return (AuxStockTransformer(MODEL_CFG) if USES_AUX else StockTransformer(**MODEL_CFG)).to(device)


def _pretrain_ssl(model, X, groups, device):
    mae = MaskedPatchAE(MODEL_CFG).to(device)
    opt = torch.optim.Adam(mae.parameters(), lr=LR)
    sched = _sched(opt, SSL_EPOCHS * len(groups))
    for ep in range(SSL_EPOCHS):
        vals = []; mae.train()
        for gi in np.random.permutation(len(groups)):
            xb = torch.from_numpy(X[groups[gi]]).to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                rec, tgt, mask = mae(xb)
                loss = ((rec.float() - tgt.float()) ** 2).mean(-1)[mask].mean()
            loss.backward(); nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            opt.step(); sched.step(); vals.append(float(loss.detach().cpu()))
        logger.info("ssl pretrain", epoch=ep + 1, loss=round(float(np.mean(vals)), 6))
    model.patch_embed.load_state_dict(mae.patch_embed.state_dict())
    model.encoder.load_state_dict(mae.encoder.state_dict())
    model.pos.data.copy_(mae.pos.data)


def save_model(payload, path):
    tensors = {}
    for k, v in payload.pop("state_dict").items():
        t = v.detach().cpu()
        tensors[k] = {"dtype": str(t.dtype).replace("torch.", ""),
                      "shape": list(t.shape), "data": t.reshape(-1).tolist()}
    payload["state_dict"] = tensors
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_model(path, map_location="cpu"):
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    sd = {}
    for k, meta in p.pop("state_dict").items():
        sd[k] = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"])).reshape(meta["shape"]).to(map_location)
    p["state_dict"] = sd
    return p


def _validate_checkpoint(ck, seed, rank):
    expected = {
        "config_tag": CONFIG_TAG,
        "variant": VARIANT,
        "arm": ARM,
        "input_preprocess": INPUT_PREPROCESS,
        "feature_cols": ACTIVE_FEATURE_COLS,
        "model_cfg": MODEL_CFG,
        "package_config": PKG,
    }
    mismatched = [key for key, value in expected.items()
                  if ck.get(key) != value]
    if int(ck.get("seed", -1)) != seed:
        mismatched.append("seed")
    if int(ck.get("rank", -1)) != rank:
        mismatched.append("rank")
    if not 1 <= int(ck.get("epoch", -1)) <= EPOCHS:
        mismatched.append("epoch")
    if mismatched:
        raise ValueError(f"checkpoint/config mismatch: {sorted(set(mismatched))}")


def train_and_save(datasources):
    table = datasources[BAR_KEY]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instruments = pool(TRAIN_START, TRAIN_END, table)
    Xtr, ltr, itr, stats = build_dataset(table, TRAIN_START, TRAIN_END, "train", instruments,
                                          label_end=None)
    dtr = itr["date"].to_numpy()
    raw_target = ltr["shrink05"] if USES_SHRINK else ltr["cc"]
    target = torch.from_numpy(_cs_zscore(raw_target, dtr))
    on = torch.from_numpy(_cs_zscore(ltr["on"], dtr))
    valid = torch.from_numpy(ltr["component_valid"])
    groups = [g for g in _day_groups(dtr) if len(g) >= 2]
    mean, std = stats
    paths = []
    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed)
        model = _make_model(device)
        if VARIANT == "ssl":
            _pretrain_ssl(model, Xtr, groups, device)
        if USES_SAM:
            opt = SAM(model.parameters(), LR, SAM_RHO)
            sched = _sched(opt.base_optimizer, EPOCHS * len(groups))
        else:
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            sched = _sched(opt, EPOCHS * len(groups))
        states = {}
        for ep in range(RUN_EPOCHS):
            model.train(); t0 = time.time()
            for gi in np.random.permutation(len(groups)):
                g = groups[gi]
                xb = torch.from_numpy(Xtr[g]).to(device); yb = target[g].to(device)
                ob = on[g].to(device); vb = valid[g].to(device)
                def loss_now():
                    out = model(xb)
                    if USES_AUX:
                        main, aux = out
                        loss = _loss_ic(main.float(), yb)
                        if vb.sum() >= 2:
                            loss = loss + AUX_WEIGHT * _loss_ic(aux[vb].float(), ob[vb])
                        return loss
                    return _loss_ic(_main(out).float(), yb)
                if USES_SAM:
                    opt.zero_grad(); loss_now().backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.first_step(True); loss_now().backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.second_step(True)
                else:
                    opt.zero_grad(); loss_now().backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
            if not FIXED_EPOCHS or ep + 1 in FIXED_EPOCHS:
                states[ep] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info("epoch", variant=VARIANT, seed=seed, epoch=ep + 1,
                        mode="full_train_no_test", sec=round(time.time() - t0, 1))
        top = np.asarray(FIXED_EPOCHS, dtype=int) - 1
        for k, ep in enumerate(top, start=1):
            ep = int(ep)
            path = model_path(seed, k)
            save_model({"state_dict": states[ep], "model_cfg": MODEL_CFG,
                        "variant": VARIANT, "arm": ARM, "seed": seed, "rank": k,
                        "epoch": ep + 1,
                        "mean": np.asarray(mean, np.float32).tolist(),
                        "std": np.asarray(std, np.float32).tolist(),
                        "feature_cols": ACTIVE_FEATURE_COLS, "config_tag": CONFIG_TAG,
                        "input_preprocess": INPUT_PREPROCESS,
                        "package_config": PKG,
                        "data_contract": {
                            "feature_levels": 5,
                            "feature_count": len(FEATURE_COLS),
                            "platform_table": table,
                            "train": [TRAIN_START, TRAIN_END], "test": None,
                            "train_samples": len(Xtr), "val_samples": 0,
                            "instruments": len(instruments), "full_train": True},
                        "selection": {
                            "name": "fixed_epoch_full_train",
                            "history_role": "none_full_train",
                            "schedule_epochs": EPOCHS,
                            "executed_epochs": RUN_EPOCHS,
                            "history": [], "composite": [],
                            "selected_metrics": {}},
                        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                    "device": str(device)}}, path)
            paths.append(path)
    return paths

INFER_DATE_CHUNK_DAYS = 5
INFER_QUERY_BATCH = 100


def _date_blocks(start_date, end_date, days=INFER_DATE_CHUNK_DAYS):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    step = pd.Timedelta(days=int(days))
    cur = start
    while cur <= end:
        block_end = min(cur + step - pd.Timedelta(days=1), end)
        yield cur.strftime("%Y-%m-%d"), block_end.strftime("%Y-%m-%d 23:59:59")
        cur = block_end + pd.Timedelta(days=1)


def predict_ensemble(table, start_date, end_date, device, chunk=300, batch=1024):
    models, stats = [], None
    for seed in SEEDS:
        for k in range(1, TOP_K + 1):
            ck = load_model(model_path(seed, k), map_location=device)
            _validate_checkpoint(ck, seed, k)
            ck_stats = (np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32))
            if (any(x.shape != (len(ACTIVE_IDX),) for x in ck_stats)
                    or not all(np.isfinite(x).all() for x in ck_stats)
                    or not (ck_stats[1] > 0).all()):
                raise ValueError("invalid checkpoint normalization statistics")
            if stats is None:
                stats = ck_stats
            elif not all(np.array_equal(a, b) for a, b in zip(stats, ck_stats)):
                raise ValueError("checkpoint normalization statistics disagree")
            model = _make_model(device); model.load_state_dict(ck["state_dict"]); model.eval()
            models.append(model)
            del ck
    if IS_C6 and len(models) != 1:
        raise ValueError("C6 package must contain exactly one model")
    outs = []
    query_batch = max(1, min(int(chunk), INFER_QUERY_BATCH))
    for block_start, block_end in _date_blocks(start_date, end_date):
        instruments = pool(block_start, block_end, table)
        if not instruments:
            continue
        X, _, idx, _ = build_dataset(table, block_start, block_end, "infer",
                                     instruments, stats=stats, query_batch=query_batch)
        block = idx[["date", "instrument"]].copy()
        for j, model in enumerate(models):
            pred = pd.Series(_predict(model, X, device, batch).astype(np.float64),
                             index=block.index)
            g = pred.groupby(block["date"], sort=False)
            block[f"m{j}"] = (pred - g.transform("mean")) / (g.transform("std") + 1e-8)
            del pred, g
        cols = [f"m{j}" for j in range(len(models))]
        block["score"] = block[cols].mean(axis=1)
        r = block.groupby("date")["score"].rank(method="average")
        n = block.groupby("date")["score"].transform("size")
        block["score"] = norm.ppf(((r - 0.5) / n).to_numpy())
        outs.append(block[["date", "instrument", "score"]].copy())
        del X, idx, block, r, n
    if not outs:
        raise RuntimeError(f"no inference samples: {start_date} {end_date}")
    return pd.concat(outs, ignore_index=True)


if __name__ == "__main__":
    train_and_save({BAR_KEY: "bigalpha_2026_stock_bar1m"})
