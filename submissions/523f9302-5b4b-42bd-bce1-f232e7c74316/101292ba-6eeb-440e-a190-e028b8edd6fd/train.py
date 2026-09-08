# train.py
import os, json, time, warnings, base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dai
warnings.filterwarnings("ignore")

TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31"
SEEDS = [42, 123, 2024]
MIN_STOCKS = 100
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-2
SEQ_30M = 40
SEQ_5M = 48
DAYS_1M = 5
N_DAILY_FEAT = 24
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.1

FEAT_30M = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
N_FEAT_30M = len(FEAT_30M)

FEAT_5M = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
]
N_FEAT_5M = len(FEAT_5M)

FEAT_1M = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]

SCALE_FIELDS = [
    "open", "high", "low", "close", "amount",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
OHLC_COLS = ["open", "high", "low", "close"]


def to_canonical(df, feat_cols, is_local):
    df = df.copy()
    if is_local:
        for c in OHLC_COLS:
            if c in df.columns:
                df[c] = df[c].astype("float64")
                df.loc[df[c] == -1.0, c] = np.nan
        for c in SCALE_FIELDS:
            if c in df.columns:
                df[c] = df[c].astype("float64") / 100.0
        df["key"] = df["instrument_id"].astype(str)
    else:
        drop_cols = []
        for c in df.columns:
            for p in ("ask_price", "bid_price", "ask_volume", "bid_volume", "ask_num_orders", "bid_num_orders"):
                if c.startswith(p) and c[-1] in "45":
                    drop_cols.append(c)
                    break
        df = df.drop(columns=drop_cols, errors="ignore")
        df["key"] = df["instrument"]
    for c in feat_cols:
        if c in df.columns:
            if "volume" in c or "num_orders" in c or c == "amount":
                df[c] = np.log1p(np.clip(df[c].astype("float64"), 0, None))
    return df


def compute_daily_summary(df_day):
    n = len(df_day)
    if n == 0:
        return np.zeros(N_DAILY_FEAT, dtype=np.float32)
    c = df_day["close"].values.astype(np.float64)
    v = df_day["volume"].values.astype(np.float64)
    a = df_day["amount"].values.astype(np.float64)
    bv = df_day["bid_volume1"].values.astype(np.float64)
    av = df_day["ask_volume1"].values.astype(np.float64)
    op = df_day["open"].values[0]
    cp = c[-1]
    hp = df_day["high"].values.max()
    lp = df_day["low"].values.min()
    pr = hp - lp + 1e-8
    raw_v = np.expm1(v)
    raw_a = np.expm1(a)
    total_v = raw_v.sum() + 1e-8
    vwap = raw_a.sum() / total_v
    log_ret = np.diff(np.log(np.clip(c, 1e-6, None)))
    rvol = log_ret.std() if len(log_ret) > 1 else 0.0
    seg = max(n // 4, 1)
    vol_segs = [raw_v[i * seg:(i + 1) * seg if i < 3 else n].sum() / total_v for i in range(4)]
    mid_imb = []
    for idx in [0, n // 4, n // 2, -1]:
        bvi = np.expm1(bv[idx])
        avi = np.expm1(av[idx])
        mid_imb.append((bvi - avi) / (bvi + avi + 1e-8))
    half = n // 2
    tail5 = min(5, n)
    bvt = np.expm1(bv[-tail5:]).mean()
    avt = np.expm1(av[-tail5:]).mean()
    feat = np.array([
        (op - lp) / pr,
        (cp - lp) / pr,
        cp / (op + 1e-8) - 1.0,
        rvol,
        (vwap - lp) / pr,
        vol_segs[0], vol_segs[1], vol_segs[2], vol_segs[3],
        mid_imb[0], mid_imb[1], mid_imb[2], mid_imb[3],
        (c[half - 1] - c[0]) / (c[0] + 1e-8),
        (c[-1] - c[half]) / (c[half] + 1e-8),
        (raw_v[half:].mean() - raw_v[:half].mean()) / (raw_v[:half].mean() + 1e-8),
        (bvt - avt) / (bvt + avt + 1e-8),
        c.mean() / (op + 1e-8) - 1.0,
        c.std() / (op + 1e-8),
        raw_v.max() / (raw_v.mean() + 1e-8),
        raw_v[:seg].sum() / total_v,
        raw_v[-seg:].sum() / total_v,
        (hp - op) / pr,
        (op - lp) / pr,
    ], dtype=np.float32)
    return np.clip(feat, -10.0, 10.0)


class HierarchicalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_30m = nn.Linear(N_FEAT_30M, D_MODEL)
        self.proj_5m = nn.Linear(N_FEAT_5M, D_MODEL)
        self.pos_30m = nn.Parameter(torch.zeros(1, SEQ_30M, D_MODEL))
        self.pos_5m = nn.Parameter(torch.zeros(1, SEQ_5M, D_MODEL))

        def make_enc(nl=NUM_LAYERS):
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
                dropout=DROPOUT, batch_first=True,
                activation="gelu", norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=nl, enable_nested_tensor=False)

        self.enc_30m = make_enc()
        self.enc_5m = make_enc()
        self.proj_1m_d = nn.Linear(N_DAILY_FEAT, D_MODEL)
        self.pos_1m_d = nn.Parameter(torch.zeros(1, DAYS_1M, D_MODEL))
        self.enc_1m_d = make_enc(nl=2)
        self.cross_attn = nn.MultiheadAttention(D_MODEL, NHEAD, dropout=DROPOUT, batch_first=True)
        self.cross_norm = nn.LayerNorm(D_MODEL)
        self.gate = nn.Sequential(nn.Linear(D_MODEL * 3, D_MODEL * 3), nn.Sigmoid())
        self.fusion_proj = nn.Linear(D_MODEL * 3, D_MODEL)
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 128), nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def _enc(self, x, proj, pos, enc):
        return enc(proj(x) + pos).mean(dim=1)

    def forward(self, x30, x5, x1d):
        h30 = self._enc(x30, self.proj_30m, self.pos_30m, self.enc_30m)
        h5 = self._enc(x5, self.proj_5m, self.pos_5m, self.enc_5m)
        h1d = self._enc(x1d, self.proj_1m_d, self.pos_1m_d, self.enc_1m_d)
        hs = torch.stack([h30, h5, h1d], dim=1)
        hc, _ = self.cross_attn(hs, hs, hs)
        hc = self.cross_norm(hc + hs)
        h30c, h5c, h1dc = hc.unbind(dim=1)
        hcat = torch.cat([h30c, h5c, h1dc], dim=-1)
        return self.head(self.fusion_proj(self.gate(hcat) * hcat)).squeeze(-1)


class SoftRankICLoss(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def soft_rank(self, x):
        return torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / self.tau).sum(dim=1)

    def forward(self, pred, target):
        rp = self.soft_rank(pred)
        rt = self.soft_rank(target)
        rp = rp - rp.mean()
        rt = rt - rt.mean()
        return -(rp * rt).sum() / (rp.norm() * rt.norm() + 1e-8)


def build_cross_sections(df_30m, df_5m, summaries_1m, start, end):
    sd, ed = pd.to_datetime(start), pd.to_datetime(end)

    df_30m = df_30m.sort_values(["key", "date"]).reset_index(drop=True)
    df_30m["day"] = df_30m["date"].dt.normalize()
    for c in FEAT_30M:
        if c in df_30m.columns:
            df_30m[c] = df_30m.groupby("key")[c].ffill()
    feat_mat_30m = df_30m[FEAT_30M].to_numpy(dtype=np.float32)
    mean_30m = np.nanmean(feat_mat_30m, axis=0).astype(np.float32)
    std_30m = (np.nanstd(feat_mat_30m, axis=0) + 1e-6).astype(np.float32)
    feat_mat_30m = np.nan_to_num((feat_mat_30m - mean_30m) / std_30m).astype(np.float32)
    keys_30m = df_30m["key"].to_numpy()
    days_30m = df_30m["day"].to_numpy()
    close_30m = df_30m["close"].to_numpy(dtype=np.float64)
    is_eod_30m = np.zeros(len(df_30m), dtype=bool)
    is_eod_30m[:-1] = (keys_30m[1:] != keys_30m[:-1]) | (days_30m[1:] != days_30m[:-1])
    is_eod_30m[-1] = True
    eod_30m = np.where(is_eod_30m)[0]

    df_5m = df_5m.sort_values(["key", "date"]).reset_index(drop=True)
    df_5m["day"] = df_5m["date"].dt.normalize()
    for c in FEAT_5M:
        if c in df_5m.columns:
            df_5m[c] = df_5m.groupby("key")[c].ffill()
    feat_mat_5m = df_5m[FEAT_5M].to_numpy(dtype=np.float32)
    mean_5m = np.nanmean(feat_mat_5m, axis=0).astype(np.float32)
    std_5m = (np.nanstd(feat_mat_5m, axis=0) + 1e-6).astype(np.float32)
    feat_mat_5m = np.nan_to_num((feat_mat_5m - mean_5m) / std_5m).astype(np.float32)
    keys_5m = df_5m["key"].to_numpy()
    days_5m = df_5m["day"].to_numpy()
    is_eod_5m = np.zeros(len(df_5m), dtype=bool)
    is_eod_5m[:-1] = (keys_5m[1:] != keys_5m[:-1]) | (days_5m[1:] != days_5m[:-1])
    is_eod_5m[-1] = True
    key_day_to_5m = {}
    for idx5 in np.where(is_eod_5m)[0]:
        key_day_to_5m[(keys_5m[idx5], days_5m[idx5])] = idx5

    cross_sections = {}
    for i in range(len(eod_30m) - 1):
        idx30 = eod_30m[i]
        next_idx30 = eod_30m[i + 1]
        if keys_30m[idx30] != keys_30m[next_idx30]:
            continue
        s30 = idx30 - SEQ_30M + 1
        if s30 < 0 or keys_30m[s30] != keys_30m[idx30]:
            continue
        day_val = days_30m[idx30]
        ts_val = pd.Timestamp(day_val)
        if ts_val < sd or ts_val > ed:
            continue
        curr_c = close_30m[idx30]
        next_c = close_30m[next_idx30]
        if curr_c <= 0 or not np.isfinite(curr_c) or not np.isfinite(next_c):
            continue
        ret = next_c / curr_c - 1.0
        if not np.isfinite(ret):
            continue
        key_val = keys_30m[idx30]
        idx5 = key_day_to_5m.get((key_val, day_val))
        if idx5 is None:
            continue
        s5 = idx5 - SEQ_5M + 1
        if s5 < 0 or keys_5m[s5] != keys_5m[idx5]:
            continue
        daily_seq = []
        day_count = 0
        j = i
        while j >= 0 and day_count < DAYS_1M:
            prev_day = str(pd.Timestamp(days_30m[eod_30m[j]]).date())
            s = summaries_1m.get((key_val, prev_day))
            if s is not None:
                daily_seq.append(s)
                day_count += 1
            prev_dv = days_30m[eod_30m[j]]
            while j >= 0 and days_30m[eod_30m[j]] == prev_dv:
                j -= 1
        if len(daily_seq) < DAYS_1M:
            continue
        daily_seq = np.stack(daily_seq[::-1], axis=0)
        if ts_val not in cross_sections:
            cross_sections[ts_val] = {"X30": [], "X5": [], "X1d": [], "y": [], "keys": []}
        cross_sections[ts_val]["X30"].append(feat_mat_30m[s30: idx30 + 1])
        cross_sections[ts_val]["X5"].append(feat_mat_5m[s5: idx5 + 1])
        cross_sections[ts_val]["X1d"].append(daily_seq)
        cross_sections[ts_val]["y"].append(np.float32(ret))
        cross_sections[ts_val]["keys"].append(key_val)

    for d in cross_sections:
        y_arr = np.array(cross_sections[d]["y"], dtype=np.float32)
        cross_sections[d]["y"] = (y_arr - y_arr.mean()) / (y_arr.std() + 1e-6)

    return cross_sections, (mean_30m, std_30m), (mean_5m, std_5m)


def train_one_seed(seed, cross_sections, timeline, stats_30m, stats_5m, device):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = HierarchicalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)
    criterion = SoftRankICLoss(tau=1.0)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")
    assert 100_000 <= n_params <= 100_000_000, f"参数量越界: {n_params}"

    best_ic = -np.inf
    best_sd = None

    for ep in range(EPOCHS):
        model.train()
        ep_ic, ep_steps = 0.0, 0
        ep_t = time.time()
        for i in np.random.permutation(len(timeline)):
            d = timeline[i]
            cs = cross_sections[d]
            X30_np = cs["X30"]
            X5_np = cs["X5"]
            X1d_np = cs["X1d"]
            y = cs["y"]
            n = len(y)
            if n > 500:
                sel = torch.randperm(n)[:500]
                X30_np = X30_np[sel.numpy()]
                X5_np = X5_np[sel.numpy()]
                X1d_np = X1d_np[sel.numpy()]
                y = y[sel]
            X30 = torch.tensor(X30_np, dtype=torch.float32).to(device)
            X5 = torch.tensor(X5_np, dtype=torch.float32).to(device)
            X1d = torch.tensor(X1d_np, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            pred = model(X30, X5, X1d)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_ic += -loss.item()
            ep_steps += 1
        scheduler.step()
        avg_ic = ep_ic / max(ep_steps, 1)
        print(f"  [seed={seed}] Epoch {ep+1:02d}/{EPOCHS} | RankIC={avg_ic:.4f} | {time.time()-ep_t:.1f}s")
        if avg_ic > best_ic:
            best_ic = avg_ic
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"  [seed={seed}] 最佳 RankIC={best_ic:.4f}")
    assert best_sd is not None
    return best_ic, best_sd


def train_and_save(datasources, model_path="transformer_model.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 设备: {device}")

    table_30m = datasources.get("bar30m", "bigalpha_2026_stock_bar30m")
    table_5m = datasources.get("bar5m", "bigalpha_2026_stock_bar5m")
    table_1m = datasources.get("bar1m", "bigalpha_2026_stock_bar1m")

    buf_start = (pd.to_datetime(TRAIN_START) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    buf_1m = (pd.to_datetime(TRAIN_START) - pd.Timedelta(days=DAYS_1M * 2)).strftime("%Y-%m-%d")

    cols_30m = ", ".join(FEAT_30M)
    cols_5m = ", ".join(FEAT_5M)
    cols_1m = ", ".join(FEAT_1M)

    print(f">>> 拉取 30m: {table_30m}")
    df_30m = dai.query(
        "SELECT date, instrument, " + cols_30m + " FROM " + table_30m + " ORDER BY instrument, date",
        filters={"date": [buf_start, TRAIN_END]}, compression=True,
    ).df()
    df_30m = to_canonical(df_30m, FEAT_30M, is_local=False)
    print(f"    行数={len(df_30m):,}")

    print(f">>> 拉取 5m: {table_5m}")
    df_5m = dai.query(
        "SELECT date, instrument, " + cols_5m + " FROM " + table_5m + " ORDER BY instrument, date",
        filters={"date": [buf_start, TRAIN_END]}, compression=True,
    ).df()
    df_5m = to_canonical(df_5m, FEAT_5M, is_local=False)
    print(f"    行数={len(df_5m):,}")

    print(f">>> 拉取 1m 并构建日内摘要: {table_1m}")
    summaries_1m = {}
    cur = pd.to_datetime(buf_1m)
    end_dt = pd.to_datetime(TRAIN_END)
    while cur <= end_dt:
        batch_end = min(cur + pd.DateOffset(months=1), end_dt)
        bdf = dai.query(
            "SELECT date, instrument, " + cols_1m + " FROM " + table_1m + " ORDER BY instrument, date",
            filters={"date": [
                cur.strftime("%Y-%m-%d 00:00:00"),
                batch_end.strftime("%Y-%m-%d 23:59:59"),
            ]}, compression=True,
        ).df()
        bdf = to_canonical(bdf, FEAT_1M, is_local=False)
        bdf["day"] = bdf["date"].dt.normalize()
        for (key, day), grp in bdf.groupby(["key", "day"], sort=False):
            summaries_1m[(key, str(day.date()))] = compute_daily_summary(grp.sort_values("date"))
        del bdf
        import gc
        gc.collect()
        cur = batch_end + pd.Timedelta(days=1)
    print(f"    日内摘要条数: {len(summaries_1m):,}")

    print(">>> 构建截面...")
    cross_sections, stats_30m, stats_5m = build_cross_sections(
        df_30m, df_5m, summaries_1m, TRAIN_START, TRAIN_END
    )
    timeline = sorted(d for d in cross_sections if len(cross_sections[d]["y"]) >= MIN_STOCKS)
    print(f"    有效截面数: {len(timeline)}")
    del df_30m, df_5m, summaries_1m
    import gc
    gc.collect()

    print(">>> 转换 Tensor...")
    for d in timeline:
        cs = cross_sections[d]
        cs["X30"] = np.stack(cs["X30"]).astype(np.float32)
        cs["X5"] = np.stack(cs["X5"]).astype(np.float32)
        cs["X1d"] = np.stack(cs["X1d"]).astype(np.float32)
        cs["y"] = torch.tensor(cs["y"], dtype=torch.float32).to(device)

    all_best_sd = []
    for seed in SEEDS:
        print(f"\n>>> 训练 seed={seed}")
        best_ic, best_sd = train_one_seed(seed, cross_sections, timeline, stats_30m, stats_5m, device)
        all_best_sd.append(best_sd)

    print("\n>>> 保存集成模型...")
    state_dicts_b64 = []
    for best_sd in all_best_sd:
        sd_b64 = {}
        for k, v in best_sd.items():
            arr = v.to(torch.float16).numpy()
            sd_b64[k] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            }
        state_dicts_b64.append(sd_b64)

    payload = {
        "model_type":   "ensemble",
        "n_models":     len(SEEDS),
        "seeds":        SEEDS,
        "state_dicts":  state_dicts_b64,
        "feat_30m":     FEAT_30M,
        "feat_5m":      FEAT_5M,
        "feat_1m":      FEAT_1M,
        "n_daily_feat": N_DAILY_FEAT,
        "seq_30m":      SEQ_30M,
        "seq_5m":       SEQ_5M,
        "days_1m":      DAYS_1M,
        "mean_30m":     stats_30m[0].tolist(),
        "std_30m":      stats_30m[1].tolist(),
        "mean_5m":      stats_5m[0].tolist(),
        "std_5m":       stats_5m[1].tolist(),
    }

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    mb = os.path.getsize(model_path) / 1024 / 1024
    print(f">>> 文件大小: {mb:.1f} MB")
    assert mb <= 50, f"超过50MB: {mb:.1f} MB"
    print(">>> 完成")
