# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端赛道 V1 提交模块 (训练与推理共用, 单一事实来源).

模型: 5m 原始 bar 序列 Transformer (6L/d256/8H/ffn1024, 4.82M 参数)
输入: 每(股票,日) 最近 5 个交易日 × 48 根 5m bar × 25 个原始字段
标签: 未来20日累计复权收益, 按日截面 1%/99% winsorize + zscore
预处理(仅规则允许三类): 缺失填充(0), 按字段 log/log1p 变换, 按字段全局标准化(训练集统计)

公榜: 平台调用 main(datasources, start_date, end_date), 加载 MODEL_JSON 只推理.
私榜: 平台调用 train_and_save(datasources) 从零重训 (固定种子).
提交文件: 本 .py + 权重 bigalpha_v1_model.json + 调用 notebook (共3个文件).
"""
import base64
import datetime as _dt
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(_HERE, "bigalpha_v1_model.json")

# ---------------- 配置 (训练/推理一致) ----------------
SEED = 42
WINDOW, N_BARS, N_FEAT = 5, 48, 25
D_MODEL, N_HEAD, N_LAYERS, FFN = 256, 8, 6, 1024
BATCH = 512
EPOCHS = 12
LR, WD = 3e-4, 1e-3
LABEL_HORIZON = 20
TRAIN_START, TRAIN_END = "2019-01-01", "2023-12-31"   # 私榜重训区间 (t+20 标签自动截断)

# 25 个输入字段 (固定顺序; 云端 5 档表只取前 3 档, 与本地 e2e 表对齐)
FEATURES = ["open", "high", "low", "close", "volume", "amount", "deal_number",
            "ask_price1", "ask_price2", "ask_price3",
            "bid_price1", "bid_price2", "bid_price3",
            "ask_volume1", "ask_volume2", "ask_volume3",
            "bid_volume1", "bid_volume2", "bid_volume3",
            "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
            "bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]
PRICE_IDX = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]     # log, <=0 视为缺失
COUNT_IDX = [4, 5, 6] + list(range(13, 25))        # log1p

# 5m bar 结束时刻 -> 槽位 0..47 (上午 09:35..11:30, 下午 13:05..15:00)
def _bar_slot(minutes):
    am = (minutes >= 575) & (minutes <= 690)
    pm = (minutes >= 785) & (minutes <= 900)
    slot = np.where(am, (minutes - 575) // 5, np.where(pm, 24 + (minutes - 785) // 5, -1))
    return slot.astype(np.int64)


# ---------------- 模型 ----------------
class BarTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(N_FEAT, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW * N_BARS, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, FFN, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos)
        return self.head(self.norm(h.mean(dim=1))).squeeze(-1)


# ---------------- 权重存取: 文本 JSON (fp16 -> base64) ----------------
def save_model_json(model, mean, std, path=MODEL_JSON):
    tensors = {}
    for k, v in model.state_dict().items():
        a = v.detach().cpu().numpy().astype(np.float16)
        tensors[k] = {"shape": list(a.shape),
                      "b64": base64.b64encode(a.tobytes()).decode("ascii")}
    payload = {
        "note": "自训权重, fp16 二进制经 base64 编码为 JSON 文本; 由本文件 train_and_save 产出",
        "config": dict(window=WINDOW, n_bars=N_BARS, n_feat=N_FEAT, d_model=D_MODEL,
                       n_head=N_HEAD, n_layers=N_LAYERS, ffn=FFN),
        "features": FEATURES,
        "mean": np.asarray(mean, np.float64).tolist(),
        "std": np.asarray(std, np.float64).tolist(),
        "state_dict": tensors,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def load_model_json(path=MODEL_JSON, device="cpu"):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    model = BarTransformer()
    sd = {}
    for k, meta in payload["state_dict"].items():
        a = np.frombuffer(base64.b64decode(meta["b64"]), dtype=np.float16).reshape(meta["shape"])
        sd[k] = torch.tensor(a.astype(np.float32))
    model.load_state_dict(sd)
    model.to(device).eval()
    mean = np.asarray(payload["mean"], np.float32)
    std = np.asarray(payload["std"], np.float32)
    return model, mean, std


# ---------------- 数据: 云端表 -> 稠密日张量 ----------------
def _pick_table(datasources):
    for k in ["bar5m", "bar_5m", "e2e_bar5m"]:
        if k in datasources:
            return datasources[k]
    for v in datasources.values():
        if "5m" in str(v):
            return v
    return list(datasources.values())[0]


def query_bars(table, start, end, instruments=None):
    import dai
    sql = f"SELECT date, instrument, {', '.join(FEATURES)} FROM {table}"
    filters = {"date": [f"{start} 00:00:00", f"{end} 23:59:59"]}
    if instruments is not None:
        filters["instrument"] = instruments
    return dai.query(sql, filters=filters).df()


def build_day_tensors(df):
    """长表 -> {day: (X(n,48,25) float32 变换前, stocks list)}; 兼容缺 bar (散射填充).

    df 需含 date(含时刻), instrument, FEATURES 列 (单位: 元, float).
    """
    df = df.copy()
    df["day"] = df["date"].dt.normalize()
    minutes = df["date"].dt.hour.to_numpy() * 60 + df["date"].dt.minute.to_numpy()
    df["slot"] = _bar_slot(minutes)
    df = df[df["slot"] >= 0]
    out = {}
    for day, sub in df.groupby("day", sort=True):
        stocks = np.sort(sub["instrument"].unique())
        sidx = pd.Series(np.arange(len(stocks)), index=stocks)
        rows = sidx[sub["instrument"]].to_numpy()
        X = np.full((len(stocks), N_BARS, N_FEAT), np.nan, np.float32)
        X[rows, sub["slot"].to_numpy()] = sub[FEATURES].to_numpy(np.float32)
        out[day] = (X, stocks)
    return out


def transform(X):
    """log/log1p (允许的按字段变换); OHLC/盘口价 <=0 或 -1 视为缺失."""
    X = X.copy()
    P = X[:, :, PRICE_IDX]
    P[P <= 0] = np.nan
    X[:, :, PRICE_IDX] = np.log(P)
    C = X[:, :, COUNT_IDX]
    X[:, :, COUNT_IDX] = np.log1p(np.clip(C, 0, None))
    return X


def standardize(Xt, mean, std):
    Xs = (Xt - mean) / std
    np.nan_to_num(Xs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs.astype(np.float32)


# ---------------- 推理 (公榜入口) ----------------
def main(datasources, start_date, end_date):
    import dai
    table = _pick_table(datasources)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_model_json(MODEL_JSON, device)

    start = str(start_date)[:10]
    end = str(end_date)[:10]
    # 预热: 覆盖 4 个历史交易日 (春节最长连休~8天, 取20自然日冗余)
    buf_start = (_dt.date.fromisoformat(start) - _dt.timedelta(days=20)).isoformat()

    pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",
                     filters={"date": [f"{start} 00:00:00", f"{end} 23:59:59"]}).df()
    pool["date"] = pool["date"].dt.normalize()

    t0 = time.time()
    df = query_bars(table, buf_start, end)
    print(f"query {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)
    tensors = build_day_tensors(df)
    days = sorted(tensors.keys())
    day_pos = {d: i for i, d in enumerate(days)}
    # 每日 lookup: instrument -> 行号
    lookups = {d: pd.Series(np.arange(len(s)), index=s) for d, (X, s) in tensors.items()}
    Xcache = {d: standardize(transform(X), mean, std) for d, (X, s) in tensors.items()}

    recs = []
    with torch.no_grad():
        for d in days:
            i = day_pos[d]
            if i < WINDOW - 1 or d < pd.Timestamp(start):
                continue
            win = days[i - WINDOW + 1: i + 1]
            stocks = tensors[d][1]
            ok = np.ones(len(stocks), bool)
            for w in win:
                ok &= pd.Index(stocks).isin(lookups[w].index)
            sel = stocks[ok]
            if len(sel) == 0:
                continue
            Xw = np.concatenate([Xcache[w][lookups[w][sel].to_numpy()] for w in win], axis=1)
            ps = []
            for j in range(0, len(sel), BATCH):
                xb = torch.from_numpy(Xw[j:j + BATCH]).to(device)
                with torch.autocast(device, torch.bfloat16, enabled=(device == "cuda")):
                    ps.append(model(xb).float().cpu().numpy())
            recs.append(pd.DataFrame({"date": d, "instrument": sel,
                                      "score": np.concatenate(ps).astype(np.float64)}))
    scores = pd.concat(recs, ignore_index=True)
    # 对齐成分股池: 保证不缺交易日; 池内缺失分数填 0 (覆盖度安全网)
    result = (pool.merge(scores, on=["date", "instrument"], how="left")
                  .assign(score=lambda x: x["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
                  .drop_duplicates(["date", "instrument"])
                  .sort_values(["date", "instrument"])[["date", "instrument", "score"]]
                  .reset_index(drop=True))
    print(f"scores: {len(result):,} rows, {result['date'].nunique()} days, "
          f"{result['instrument'].nunique()} instruments", flush=True)
    return result


# ---------------- 训练 (私榜重训入口) ----------------
def train_and_save(datasources, model_path=MODEL_JSON):
    import dai
    table = _pick_table(datasources)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train_and_save on {device}, table={table}", flush=True)

    # 按季度分块拉数 (服务端单次查询限 200MB)
    dfs = []
    q_starts = pd.date_range(TRAIN_START, TRAIN_END, freq="QS").tolist()
    for qs in q_starts:
        qe = min(qs + pd.offsets.QuarterEnd(), pd.Timestamp(TRAIN_END))
        t0 = time.time()
        d = query_bars(table, qs.strftime("%Y-%m-%d"), qe.strftime("%Y-%m-%d"))
        dfs.append(d)
        print(f"  {qs.date()}~{qe.date()}: {len(d):,} rows {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    # 复权因子单独取 (仅用于标签, 不进输入)
    adj = dai.query(f"SELECT date, instrument, adjust_factor FROM {table}",
                    filters={"date": [f"{TRAIN_START} 00:00:00", f"{TRAIN_END} 23:59:59"]}).df()

    tensors = build_day_tensors(df)
    del df
    days = sorted(tensors.keys())

    # 日收盘 (最后一根有效bar close × adjust_factor) -> t+20 标签
    adj["day"] = adj["date"].dt.normalize()
    adj_daily = adj.groupby(["day", "instrument"])["adjust_factor"].last()
    close_rows = []
    for d in days:
        X, stocks = tensors[d]
        close = X[:, :, 3].copy()
        close[close <= 0] = np.nan
        last = np.full(len(stocks), np.nan)
        for j in range(N_BARS - 1, -1, -1):
            m = np.isnan(last) & ~np.isnan(close[:, j])
            if m.any():
                last[m] = close[m, j]
        close_rows.append(pd.DataFrame({"day": d, "instrument": stocks, "close": last}))
    daily = pd.concat(close_rows, ignore_index=True)
    daily = daily.merge(adj_daily.rename("af"), on=["day", "instrument"], how="left")
    daily["close_adj"] = daily["close"] * daily["af"].fillna(1.0)
    daily = daily.sort_values(["instrument", "day"])
    g = daily.groupby("instrument", sort=False)["close_adj"]
    daily["ret"] = g.shift(-LABEL_HORIZON) / daily["close_adj"] - 1.0

    def zs(x):
        lo, hi = x.quantile([0.01, 0.99])
        xc = x.clip(lo, hi)
        return (xc - xc.mean()) / (xc.std() + 1e-12)
    daily["y"] = daily.groupby("day")["ret"].transform(zs)
    ymap = {d: s.set_index("instrument")["y"] for d, s in daily.groupby("day")}

    # 标准化统计 (仅训练区间, 每5天抽1天)
    s0 = np.zeros(N_FEAT); s1 = np.zeros(N_FEAT); cnt = np.zeros(N_FEAT)
    for d in days[::5]:
        Xt = transform(tensors[d][0])
        s0 += np.nansum(Xt, axis=(0, 1)); s1 += np.nansum(Xt * Xt, axis=(0, 1))
        cnt += (~np.isnan(Xt)).sum(axis=(0, 1))
    mean = (s0 / cnt).astype(np.float32)
    std = (np.sqrt(np.maximum(s1 / cnt - (s0 / cnt) ** 2, 1e-12)) + 1e-6).astype(np.float32)

    Xcache = {d: standardize(transform(X), mean, std).astype(np.float16)
              for d, (X, s) in tensors.items()}
    lookups = {d: pd.Series(np.arange(len(s)), index=s) for d, (X, s) in tensors.items()}
    stocks_of = {d: s for d, (X, s) in tensors.items()}

    # 训练样本日: 窗口齐 & t+20 标签不越出训练区间
    train_days = days[WINDOW - 1: len(days) - (LABEL_HORIZON + 1)]

    model = BarTransformer().to(device)
    print("params:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = len(train_days) * 2 * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                total_steps=total_steps, pct_start=0.05)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        order = train_days.copy()
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for d in order:
            i = days.index(d)
            win = days[i - WINDOW + 1: i + 1]
            stocks = stocks_of[d]
            y = ymap.get(d)
            if y is None:
                continue
            ok = pd.Index(stocks).isin(y.index) & ~pd.Index(stocks).map(
                lambda s: bool(np.isnan(y.get(s, np.nan))))
            for w in win:
                ok &= pd.Index(stocks).isin(lookups[w].index)
            sel = stocks[np.asarray(ok)]
            if len(sel) < 50:
                continue
            Xw = np.concatenate(
                [Xcache[w][lookups[w][sel].to_numpy()] for w in win], axis=1).astype(np.float32)
            yb = y[sel].to_numpy(np.float32)
            perm = rng.permutation(len(sel))
            for j in range(0, len(sel), BATCH):
                idx = perm[j:j + BATCH]
                xb = torch.from_numpy(Xw[idx]).to(device)
                tb = torch.from_numpy(yb[idx]).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device, torch.bfloat16, enabled=(device == "cuda")):
                    loss = loss_fn(model(xb), tb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                tot += loss.item(); nb += 1
        print(f"epoch {ep}: loss={tot/max(nb,1):.5f} {time.time()-t0:.0f}s", flush=True)

    save_model_json(model, mean, std, model_path)
    print("model saved:", model_path, flush=True)
    return model_path
