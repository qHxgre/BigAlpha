# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端赛道 — PatchTCN 训练脚本（自包含）。

提交目录只允许 4 个文件，因此预处理 / 窗口构造 / 模型定义全部收敛在本文件，
predict.py 直接 import 本文件，保证训练与推理走同一套实现。

规则依据：
- 输入仅用主办方原始字段，禁止衍生特征（[介绍] L32）
- 预处理仅：缺失填充 / 按字段固定变换 / log / 训练集拟合的标准化（[介绍] L33）
- 回看窗口 ≤ 240 交易日（[介绍] L34）；本模型 5 个交易日
- 参数量 ∈ [1e5, 1e8]（[介绍] L37）
- 本地压缩表与云端原始表单位/档位/代码不同，必须在 to_canonical 对齐（[本地] L15）
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import dai
except ImportError:  # pragma: no cover - cloud provides one of the two
    try:
        from bigquant import dai
    except ImportError:
        dai = None

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_HERE, "model_weights.json")

# ----------------------------------------------------------------------------
# 1. 字段与常量（锁定，不可事后更改）
# ----------------------------------------------------------------------------
OHLC_COLS = ["open", "high", "low", "close"]
FULL_25 = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
    "volume", "amount", "deal_number",
]
PRICE_SCALE_COLS = [
    "open", "high", "low", "close", "amount",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
LOG1P_COLS = [
    "volume", "amount", "deal_number",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
NUM_ORDERS_COLS = [
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]

# 云端字段表与云端示例对该列命名不一致（RULES_LOCK 冲突 C2），两种都接受。
CLOUD_ALIASES = {
    "close": ["close", "price"],
    "deal_number": ["deal_number", "num_trades"],
}

PATCH_LEN = 16
N_PATCH = 15          # 15 * 16 = 240 根 1m bar = 一个交易日
WINDOW_DAYS = 5
# 全市场 5 个连续交易日的自然日跨度实测上限为 14 天（最长假期）。
# 超过则说明该标的自身缺交易日（停牌），此类窗口丢弃。
MAX_WINDOW_SPAN_DAYS = 14
N_FEAT = len(FULL_25)

# 每个日期块额外回看的自然日数。需覆盖 WINDOW_DAYS 个交易日，并为停牌留余量：
# 12 天曾导致停牌较多的标的在块首无法凑满窗口，全年漏掉 254 个 (date, instrument)。
CHUNK_DAYS = 20
HISTORY_DAYS = 30

D_MODEL = 128
N_LAYERS = 6
KERNEL = 3
DROPOUT = 0.1

EPOCHS = 5
BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 20260804
MIN_CS_SIZE = 30

TRAIN_START = "2019-01-02"
TRAIN_END = "2023-12-29"
TABLE_LOCAL = "bigalpha_2026_e2e_bar1m"
TABLE_CLOUD = "bigalpha_2026_stock_bar1m"


# ----------------------------------------------------------------------------
# 2. canonical：本地压缩表 / 云端原始表 → 统一表示（元、3 档、NaN 缺失）
# ----------------------------------------------------------------------------
def resolve_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for canon, names in CLOUD_ALIASES.items():
        if canon in out.columns:
            continue
        for alt in names:
            if alt in out.columns:
                out = out.rename(columns={alt: canon})
                break
    return out


def to_canonical(df: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    """统一本地 / 云端表示。

    本地：分→元、OHLC 0（文档称 -1）→NaN、int16 委托笔数回卷还原、保留 3 档。
    云端：丢弃 4/5 档、保持元与 NaN。
    """
    out = resolve_aliases(df).copy()
    if is_local:
        for c in OHLC_COLS:
            if c in out.columns:
                out.loc[out[c] == 0, c] = np.nan
                out.loc[out[c] == -1, c] = np.nan
        for c in NUM_ORDERS_COLS:
            if c in out.columns:
                v = out[c].to_numpy()
                neg = v < 0
                if neg.any():
                    v = v.astype(np.int32, copy=True)
                    v[neg] = v[neg] + 65536
                    out[c] = v
        for c in PRICE_SCALE_COLS:
            if c in out.columns:
                out[c] = out[c].astype("float64") / 100.0
        for c in FULL_25:
            if c in out.columns and c not in PRICE_SCALE_COLS:
                out[c] = out[c].astype("float64")
    else:
        drop = [
            c for c in out.columns
            if any(
                c.startswith(p) and c[-1] in "45"
                for p in ("ask_price", "bid_price", "ask_volume",
                          "bid_volume", "ask_num_orders", "bid_num_orders")
            )
        ]
        out = out.drop(columns=drop, errors="ignore")
        for c in FULL_25:
            if c in out.columns:
                out[c] = out[c].astype("float64")

    missing = [c for c in FULL_25 if c not in out.columns]
    if missing:
        raise KeyError(f"canonical 缺少必需原始字段: {missing}")
    return out


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in LOG1P_COLS:
        if c in out.columns:
            out[c] = np.log1p(np.clip(out[c].to_numpy(dtype="float64"), 0, None))
    return out


# ----------------------------------------------------------------------------
# 3. 日内 patch 与窗口
# ----------------------------------------------------------------------------
def bars_to_patches(feat: np.ndarray) -> np.ndarray:
    """(T, F) 日内 bar → (N_PATCH, F)，不足 240 根在前部补零。"""
    target = PATCH_LEN * N_PATCH
    t, f = feat.shape
    if t >= target:
        x = feat[-target:]
    else:
        x = np.vstack([np.zeros((target - t, f), dtype=np.float32), feat.astype(np.float32)])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x.reshape(N_PATCH, PATCH_LEN, f).mean(axis=1).astype(np.float32)


def _ffill_2d(feat: np.ndarray) -> np.ndarray:
    mask = np.isnan(feat)
    if not mask.any():
        return feat
    idx = np.where(~mask, np.arange(len(feat))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return feat[idx, np.arange(feat.shape[1])]


def build_day_table(df_canonical: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """canonical bar 表 → 每 (标的, 交易日) 一行的 patch 向量。"""
    df = apply_log1p(df_canonical)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.normalize()
    rows = []
    for (key, day), g in df.groupby([key_col, "day"], sort=False):
        g = g.sort_values("date")
        feat = _ffill_2d(g[FULL_25].to_numpy(dtype=np.float64))
        rows.append({key_col: key, "day": day, "patches": bars_to_patches(feat)})
    if not rows:
        return pd.DataFrame(columns=[key_col, "day", "patches"])
    return pd.DataFrame(rows)


def build_windows(day_table: pd.DataFrame, key_col: str, target_days=None):
    """日 patch 表 → (N, W, P, F) 窗口。窗口按标的自身连续 W 个可得交易日取。"""
    dt = day_table.sort_values([key_col, "day"], ignore_index=True)
    arr = np.stack(dt["patches"].to_numpy()).astype(np.float32)
    dt = dt.drop(columns=["patches"])
    dt["row"] = np.arange(len(dt), dtype=np.int64)
    dt["pos"] = dt.groupby(key_col).cumcount()

    sel = dt[dt["pos"] + 1 >= WINDOW_DAYS]
    if target_days is not None:
        sel = sel[sel["day"].isin(set(pd.to_datetime(list(target_days))))]
    if sel.empty:
        raise RuntimeError("无可用窗口")
    sel = sel.sort_values(["day", key_col], ignore_index=True)

    offsets = np.arange(-(WINDOW_DAYS - 1), 1, dtype=np.int64)
    gather = sel["row"].to_numpy()[:, None] + offsets[None, :]
    key_of_row = dt[key_col].to_numpy()
    ok = (key_of_row[gather] == sel[key_col].to_numpy()[:, None]).all(axis=1)

    # A name returning from a long suspension would otherwise get a "5-day"
    # window spanning years. 5 consecutive market sessions span at most
    # MAX_WINDOW_SPAN_DAYS calendar days, so anything longer means the name
    # itself missed sessions.
    day_of_row = dt["day"].to_numpy()
    span = (day_of_row[gather[:, -1]] - day_of_row[gather[:, 0]]) / np.timedelta64(1, "D")
    ok = ok & (span <= MAX_WINDOW_SPAN_DAYS)

    sel = sel[ok].reset_index(drop=True)
    gather = gather[ok]

    X = arr[gather].reshape(len(sel), WINDOW_DAYS, N_PATCH, N_FEAT)
    keys = sel[["day", key_col]].rename(columns={"day": "date"})
    return np.ascontiguousarray(X, dtype=np.float32), keys


# ----------------------------------------------------------------------------
# 4. 标准化（仅在训练区间拟合）
# ----------------------------------------------------------------------------
def fit_standardizer(x: np.ndarray):
    flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def transform_standardizer(x: np.ndarray, mean, std) -> np.ndarray:
    out = (np.asarray(x, dtype=np.float32) - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ----------------------------------------------------------------------------
# 5. 模型：因果空洞卷积
# ----------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, kernel, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(ch, ch, kernel, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x + self.net(x.transpose(1, 2)).transpose(1, 2))


class PatchTCN(nn.Module):
    def __init__(self, n_feat=N_FEAT, d_model=D_MODEL, n_layers=N_LAYERS,
                 kernel=KERNEL, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_feat, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.tcn = nn.Sequential(
            *[TCNBlock(d_model, kernel, 2**i, dropout) for i in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, day_patches):
        b, w, p, f = day_patches.shape
        h = self.input_proj(day_patches.reshape(b, w * p, f))
        return self.head(self.tcn(h)[:, -1, :]).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model() -> PatchTCN:
    model = PatchTCN()
    n = count_params(model)
    if not (100_000 <= n <= 100_000_000):
        raise RuntimeError(f"参数量 {n} 越界 [1e5, 1e8]")
    return model


# ----------------------------------------------------------------------------
# 6. 权重存取（JSON，< 50MB）
# ----------------------------------------------------------------------------
def _f32(x) -> float:
    """Shortest decimal form that round-trips through float32.

    Fixed-precision rounding silently zeroed the smallest weights (|w| ~ 5e-8),
    which showed up as a 1e-4 score drift versus the research model.
    """
    return float(f"{float(x):.9g}")


def save_weights(path: str, model: nn.Module, mean, std, extra: dict | None = None) -> str:
    tensors = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu().to(torch.float32)
        tensors[k] = {"shape": list(t.shape), "data": [_f32(x) for x in t.reshape(-1).tolist()]}
    payload = {
        "format": "patchtcn_v1",
        "state_dict": tensors,
        "mean": [_f32(x) for x in np.asarray(mean, np.float32).tolist()],
        "std": [_f32(x) for x in np.asarray(std, np.float32).tolist()],
        "feature_cols": FULL_25,
        "window_days": WINDOW_DAYS,
        "n_patch": N_PATCH,
        "patch_len": PATCH_LEN,
        "n_params": count_params(model),
        "seed": SEED,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return path


def load_weights(path: str, map_location="cpu") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=torch.float32)
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    payload["state_dict"] = sd
    return payload


# ----------------------------------------------------------------------------
# 7. 标签：全市场 t+1 复权收益 → 日截面 z-score
# ----------------------------------------------------------------------------
def build_labels(daily: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """daily 需含 key_col, day, close, adjust_factor（或云端已复权 close）。"""
    d = daily.sort_values([key_col, "day"]).reset_index(drop=True).copy()
    if "adjust_factor" in d.columns:
        d["adj_close"] = d["close"].astype("float64") * d["adjust_factor"].astype("float64")
    else:
        d["adj_close"] = d["close"].astype("float64")
    d.loc[~(d["adj_close"] > 0), "adj_close"] = np.nan

    cal = pd.DatetimeIndex(sorted(d["day"].unique()))
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    d["label_date"] = d["day"].map(nxt)

    right = d[[key_col, "day", "adj_close"]].rename(
        columns={"day": "join_day", "adj_close": "adj_next"}
    )
    m = d.merge(right, left_on=[key_col, "label_date"], right_on=[key_col, "join_day"], how="left")
    m = m.drop(columns=["join_day"])
    valid = m["adj_close"].gt(0) & m["adj_next"].gt(0)
    m["label"] = np.where(valid, m["adj_next"] / m["adj_close"] - 1.0, np.nan)

    def _z(s):
        sd = s.std(ddof=0)
        return s * 0.0 if not np.isfinite(sd) or sd < 1e-12 else (s - s.mean()) / sd

    m["y"] = m.groupby("day")["label"].transform(_z)
    return m[[key_col, "day", "label_date", "label", "y"]]


# ----------------------------------------------------------------------------
# 8. 数据读取（云端按短日期分块，保留最小历史窗口）
# ----------------------------------------------------------------------------
def query_bars(table: str, start: str, end: str) -> pd.DataFrame:
    if dai is None:
        raise RuntimeError("dai 不可用，无法从云端读取")
    cols = ["date", "instrument"] + FULL_25
    df = dai.query(
        f"SELECT * FROM {table}",
        filters={"date": [f"{start} 00:00:00", f"{end} 23:59:59"]},
        compression=True,
    ).df()
    if "instrument" in df.columns:
        df["instrument"] = df["instrument"].astype(str)
    keep = [c for c in df.columns if c in set(cols) | {"price", "num_trades", "adjust_factor"}]
    return df[keep]


def iter_date_chunks(start: str, end: str, chunk_days: int = CHUNK_DAYS,
                     history_days: int = HISTORY_DAYS):
    """按短日期块产出 (查询起点含历史缓冲, 查询终点, 该块实际目标起点)。

    history_days 必须覆盖 WINDOW_DAYS 个交易日在最长假期（春节约 11 天）下跨越的自然日，
    否则块首几天会因历史不足而丢样本。
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cur = s
    while cur <= e:
        stop = min(cur + pd.Timedelta(days=chunk_days - 1), e)
        q_start = cur - pd.Timedelta(days=history_days)
        yield q_start.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d"), cur.strftime("%Y-%m-%d")
        cur = stop + pd.Timedelta(days=1)


# ----------------------------------------------------------------------------
# 9. 训练入口
# ----------------------------------------------------------------------------
def train_and_save(table: str = TABLE_CLOUD, save_path: str = DEFAULT_WEIGHTS,
                   start: str = TRAIN_START, end: str = TRAIN_END,
                   is_local: bool = False, epochs: int = EPOCHS) -> str:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key_col = "instrument_id" if is_local else "instrument"

    t0 = time.time()
    day_parts, daily_parts = [], []
    for q_start, q_end, _ in iter_date_chunks(start, end):
        raw = query_bars(table, q_start, q_end)
        if raw.empty:
            continue
        can = to_canonical(raw, is_local=is_local)
        can["date"] = pd.to_datetime(can["date"])
        day_parts.append(build_day_table(can, key_col))

        can["day"] = can["date"].dt.normalize()
        last = (can.sort_values([key_col, "date"])
                   .groupby([key_col, "day"], sort=False).tail(1))
        cols = [key_col, "day", "close"] + (["adjust_factor"] if "adjust_factor" in can.columns else [])
        daily_parts.append(last[cols])
        print(f"[train] {q_start}~{q_end} bars={len(raw)}", flush=True)

    day_table = pd.concat(day_parts, ignore_index=True).drop_duplicates([key_col, "day"], keep="last")
    daily = pd.concat(daily_parts, ignore_index=True).drop_duplicates([key_col, "day"], keep="last")

    labels = build_labels(daily, key_col)
    labels = labels[labels["label"].notna() & (labels["label_date"] <= pd.Timestamp(end))]

    X, keys = build_windows(day_table, key_col)
    y_map = labels.set_index(["day", key_col])["y"]
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(keys["date"]), keys[key_col]])
    y = y_map.reindex(idx).to_numpy()
    ok = np.isfinite(y)
    X, y, keys = X[ok], y[ok].astype(np.float32), keys[ok].reset_index(drop=True)

    mean, std = fit_standardizer(X)
    X = transform_standardizer(X, mean, std)

    model = create_model().to(device)
    print(f"[train] samples={len(y)} n_params={count_params(model)} device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    model.train()
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        perm = rng.permutation(len(yt))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            sel = perm[i:i + BATCH]
            xb, yb = Xt[sel].to(device), yt[sel].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        print(f"[train] epoch {ep+1}/{epochs} loss={tot/max(nb,1):.6f}", flush=True)

    path = save_weights(save_path, model, mean, std,
                        extra={"train_start": start, "train_end": end,
                               "elapsed_sec": round(time.time() - t0, 1)})
    print(f"[train] saved {path} ({os.path.getsize(path)/2**20:.1f} MB)", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE_CLOUD)
    ap.add_argument("--start", default=TRAIN_START)
    ap.add_argument("--end", default=TRAIN_END)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=DEFAULT_WEIGHTS)
    a = ap.parse_args()
    train_and_save(a.table, a.out, a.start, a.end, is_local=a.local, epochs=a.epochs)
