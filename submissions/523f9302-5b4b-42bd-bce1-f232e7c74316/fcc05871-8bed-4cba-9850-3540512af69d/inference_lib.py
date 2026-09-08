"""BigAlpha 2026 submission, BIG-BARS family (v134/v135): hist_day seed-ENSEMBLE on the
COMPACT scoring core.

Geometry comes from the manifest (60 bars@14:01 x d60 or d120); per day, an intraday GRU
(8->256) encodes the last DAY_BARS one-minute bars, an outer GRU (256) runs across the
last DAYS trading days' day-vectors; qlib-HIST's hidden-concept + individual modules run
on each day's WHOLE cross-section (cosine-similarity concept graph, no external sector
data); Linear head, no LayerNorm. Score = MEAN of the packed seeds' raw scores.

Memory: this lib is the v9 compact-base core (int-key buffer, contiguous base matrix,
O(1) day slices, malloc_trim between months) -- the canonical month-buffer lib measured
6.25 GB peak RSS on 60xd60 in the same harness whose platform kill line is ~5.6; this
core measured 4.28 GB at d180x h384 when the standard path OOM'd at 5.48.

Normalization (must match training exactly):
  price levels <= 0 -> NaN, ffill within stock; flows/counts: log1p
  CROSS-SECTIONAL standardization per bar timestamp across stocks (computed from the
  live cross-section, no fitted parameters), then the train-fit per-field (x-mean)/std
  from the manifest.

Inference is DAY-WISE: each scoring day's whole cross-section goes through the model
together (the concept graph depends on who is in the batch).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
MODEL_PATH = os.path.join(_HERE, "final_model.json")
CLOUD_TABLE_1M = "bigalpha_2026_stock_bar1m"
# Geometry DEFAULTS ONLY -- load_manifest() OVERRIDES every one of these from the packed
# manifest (v134/v135 ship 60 bars@14:01), exactly like the canonical lib. A manifest
# without days/seq_len is refused there.
TAIL_TIME = "14:46"
FETCH_TIME = TAIL_TIME       # canonical-lib contract (preflight reads it); tail fetch
FREQ = "1m"
NORM = "cs"
DAY_BARS = 15
DAYS = 180
SEQ_LEN = DAY_BARS * DAYS
HIST_BUFFER_DAYS = int(DAYS * 1.6) + 40

PRICE_LEVEL_FIELDS = ["open", "high", "low", "close",
                      "ask_price1", "ask_price2", "ask_price3",
                      "bid_price1", "bid_price2", "bid_price3"]
FLOW_FIELDS = ["volume", "amount", "deal_number",
               "ask_volume1", "ask_volume2", "ask_volume3",
               "bid_volume1", "bid_volume2", "bid_volume3",
               "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
               "bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]
FEATURE_COLS = PRICE_LEVEL_FIELDS + FLOW_FIELDS          # 25 fields


class DayHIST(nn.Module):
    """Intraday GRU -> day vectors -> outer GRU -> HIST hidden-concept + individual
    modules over the day's whole cross-section -> Linear (no LayerNorm)."""

    def __init__(self, day_bars=15, n_feat=25, d_day=256, gru_hidden=256):
        super().__init__()
        self.day_bars = day_bars
        self.intraday = nn.GRU(n_feat, d_day, batch_first=True)
        self.rnn = nn.GRU(d_day, gru_hidden, batch_first=True)
        H = gru_hidden
        self.fc_is = nn.Linear(H, H)
        self.fc_is_back = nn.Linear(H, H)
        self.fc_is_fore = nn.Linear(H, H)
        self.fc_indi = nn.Linear(H, H)
        self.leaky = nn.LeakyReLU()
        self.head = nn.Linear(H, 1)

    @staticmethod
    def _cos(x, y):
        xy = x.mm(y.t())
        xn = torch.sqrt((x * x).sum(1)).reshape(-1, 1)
        yn = torch.sqrt((y * y).sum(1)).reshape(-1, 1)
        return xy / (xn.mm(yn.t()) + 1e-6)

    def forward(self, x):                                  # (N, K*day_bars, F): ONE day's stocks
        N, T, F = x.shape
        K = T // self.day_bars
        seqs = x.reshape(N * K, self.day_bars, F)
        # chunked intraday encode: one shot over N*K (~117k at d120) makes torch's CPU
        # GRU materialize the full per-step output + thread scratch (~12GB) -> platform
        # OOM. 8k-slice chunks cap the transient at ~150MB; identical outputs.
        hs = []
        for i in range(0, seqs.shape[0], 2048):
            _, h = self.intraday(seqs[i:i + 2048])
            hs.append(h[-1])
        _, h2 = self.rnn(torch.cat(hs).reshape(N, K, -1))
        x_hidden = h2[-1]
        dev = x.device
        if N <= 1:
            i_shared_back = torch.zeros_like(x_hidden)
            output_is = torch.zeros_like(x_hidden)
        else:
            g = self._cos(x_hidden, x_hidden)
            diag = g.diagonal(0)
            g = g * (torch.ones(N, N, device=dev) - torch.eye(N, device=dev))
            col = g.max(1)[1]; val = g.max(1)[0]
            marker = torch.zeros_like(g)
            marker[torch.arange(N, device=dev), col] = 10.0
            g = torch.where(marker == 10.0, val.reshape(-1, 1).expand_as(g), torch.zeros_like(g))
            g = g + torch.diag_embed((g.sum(0) != 0).float() * diag)
            hidden_i = x_hidden.t().mm(g).t()
            hidden_i = hidden_i[hidden_i.sum(1) != 0]
            if hidden_i.numel() == 0:
                i_shared_back = torch.zeros_like(x_hidden)
                output_is = torch.zeros_like(x_hidden)
            else:
                a = torch.softmax(self._cos(x_hidden, hidden_i), dim=1)
                iss = self.fc_is(a.mm(hidden_i))
                i_shared_back = self.fc_is_back(iss)
                output_is = self.leaky(self.fc_is_fore(iss))
        output_indi = self.leaky(self.fc_indi(x_hidden - i_shared_back))
        return self.head(output_is + output_indi).squeeze(-1)


def _decode_tensor(v):
    if "b64" in v:
        import base64
        arr = np.frombuffer(base64.b64decode(v["b64"]), dtype=np.dtype(v["dtype"]))
        return torch.from_numpy(arr.astype(np.float32).reshape(v["shape"]).copy())
    return torch.tensor(v["data"], dtype=getattr(torch, v["dtype"])).reshape(v["shape"])


def load_manifest(path=MODEL_PATH):
    """Load the manifest and OVERRIDE the module geometry globals from it, so this one
    lib scores whatever hist_day cell was packed (60x d60 / 60x d120 / ...). A manifest
    without days/seq_len would silently score the WRONG geometry -- refused."""
    global DAY_BARS, DAYS, SEQ_LEN, TAIL_TIME, HIST_BUFFER_DAYS
    global FEATURE_COLS, PRICE_LEVEL_FIELDS, FLOW_FIELDS
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m.get("arch") == "hist_day_v5", m.get("arch")
    cfg = m["model_cfg"]
    if m.get("days") is None or m.get("seq_len") is None:
        raise ValueError("manifest lacks days/seq_len -- repack (legacy manifests are "
                         "not supported by this lib)")
    if m.get("bar_windows"):
        raise ValueError("this lib has no BAR_WINDOWS support; the manifest asks for it")
    DAY_BARS = int(cfg["day_bars"])
    DAYS = int(m["days"])
    SEQ_LEN = int(m["seq_len"])
    assert SEQ_LEN == DAY_BARS * DAYS, (SEQ_LEN, DAY_BARS, DAYS)
    global FETCH_TIME, FREQ, NORM
    _tail = m.get("tail_time")
    if _tail:
        TAIL_TIME = _tail
    FETCH_TIME = TAIL_TIME
    FREQ = m.get("freq", FREQ)
    NORM = m.get("norm_scheme", m.get("norm", NORM))
    HIST_BUFFER_DAYS = int(DAYS * 1.6) + 40
    fcols = m.get("feature_cols")
    if fcols:
        FEATURE_COLS = list(fcols)
        PRICE_LEVEL_FIELDS = [c for c in FEATURE_COLS if c in (
            "open", "high", "low", "close",
            "ask_price1", "ask_price2", "ask_price3",
            "bid_price1", "bid_price2", "bid_price3")]
        FLOW_FIELDS = [c for c in FEATURE_COLS if c not in PRICE_LEVEL_FIELDS]
    models = []
    for sd_json in m["models"]:
        sd = {k: _decode_tensor(v) for k, v in sd_json.items()}
        net = DayHIST(cfg["day_bars"], cfg["n_feat"], cfg["d_day"], cfg["gru_hidden"])
        net.load_state_dict(sd)
        net.eval()
        models.append(net)
    stats = (np.asarray(m["mean"], np.float32), np.asarray(m["std"], np.float32))
    return models, stats, m


def canonical_prep(df):
    """NaN/ffill prices + log1p flows. NO scaling here (cs + manifest stats come later).
    MUTATES df: the sole caller passes a fresh pd.concat result, so a defensive copy
    would only double the 14-month buffer frame."""
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.normalize()
    df = df.sort_values(["key", "date"], kind="mergesort").reset_index(drop=True)
    for c in PRICE_LEVEL_FIELDS:
        df[c] = df[c].astype("float32")
        df.loc[df[c] <= 0.0, c] = np.nan
    df[PRICE_LEVEL_FIELDS] = df.groupby("key", sort=False)[PRICE_LEVEL_FIELDS].ffill()
    for c in FLOW_FIELDS:
        df[c] = np.log1p(df[c].clip(lower=0).astype("float32"))
    return df


def cs_standardize(df):
    """Per bar TIMESTAMP, standardize each field across that bar's stocks (live stats,
    no fitted parameters). numpy bincount per month slice: identical to the pandas
    groupby-transform (ddof=1 std, per-column NaN skipping, +1e-9 denom, float32 out)
    without its full-frame float64 intermediates."""
    per = df["day"].dt.to_period("M")
    pos = df.columns.get_indexer(FEATURE_COLS)
    for m in per.unique():
        ix = (per == m).to_numpy()
        codes, uniq = pd.factorize(df.loc[ix, "date"], sort=False)
        G = len(uniq)
        V = df.loc[ix, FEATURE_COLS].to_numpy(np.float64)
        out = np.empty(V.shape, np.float32)
        for c in range(V.shape[1]):
            col = V[:, c]
            ok = np.isfinite(col)
            cc, cv = codes[ok], col[ok]
            n1 = np.bincount(cc, minlength=G).astype(np.float64)
            s1 = np.bincount(cc, weights=cv, minlength=G)
            s2 = np.bincount(cc, weights=cv * cv, minlength=G)
            mean = np.divide(s1, n1, out=np.full(G, np.nan), where=n1 > 0)
            var = np.divide(s2 - n1 * mean * mean, n1 - 1,
                            out=np.full(G, np.nan), where=n1 > 1)
            std = np.sqrt(np.maximum(var, 0))
            out[:, c] = (col - mean[codes]) / (std[codes] + 1e-9)
        df.iloc[ix, pos] = out
    return df


def multi_day_windows(df, stats, score_lo, score_hi):
    """Per stock: for each end-of-day in [score_lo, score_hi], the last SEQ_LEN bars
    (across days, exactly like training's build_windows). All-finite required.
    Returns X (N, SEQ_LEN, F) normalized with the train-fit stats, keys, days."""
    mean, std = stats
    lo, hi = pd.Timestamp(score_lo), pd.Timestamp(score_hi)
    Xs, keys, days = [], [], []
    for k, sub in df.groupby("key", sort=False):
        if len(sub) < SEQ_LEN:
            continue
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        day = sub["day"].to_numpy()
        eod = np.nonzero(np.r_[day[1:] != day[:-1], True])[0]   # last row of each day
        eod_days = pd.to_datetime(day[eod])
        for j, p in enumerate(eod):
            d = eod_days[j]
            if p + 1 < SEQ_LEN or d < lo or d > hi:
                continue
            win = feats[p - SEQ_LEN + 1: p + 1]
            if not np.isfinite(win).all():
                continue
            Xs.append(win); keys.append(k); days.append(d)
    if not Xs:
        return np.empty((0, SEQ_LEN, len(FEATURE_COLS)), np.float32), [], []
    X = np.stack(Xs)
    X -= mean
    X /= std
    return X, keys, days


@torch.no_grad()
def score_frame(df, models, stats, score_lo, score_hi, inst_names=None):
    """Prepped+cs-standardized frame -> [date, instrument, score] via DAY-WISE forward.
    With inst_names, df["key"] holds int codes and names are mapped back at output."""
    X, keys, days = multi_day_windows(df, stats, score_lo, score_hi)
    if len(X) == 0:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = [n.to(dev) for n in models]            # SEED ENSEMBLE: mean of raw scores
    inst = np.asarray(inst_names)[np.asarray(keys)] if inst_names is not None else keys
    res = pd.DataFrame({"date": pd.to_datetime(days), "instrument": inst})
    scores = np.empty(len(X), dtype=np.float32)
    for d, ix in res.groupby("date").groups.items():
        ixs = np.asarray(ix, dtype=np.int64)
        xb = torch.from_numpy(X[ixs]).to(dev)
        scores[ixs] = np.mean([n(xb).cpu().numpy() for n in nets], axis=0)
    res["score"] = scores
    res["date"] = res["date"].dt.strftime("%Y-%m-%d")
    return res[["date", "instrument", "score"]]


def _assemble_base(frames):
    """Month frames (date, key:int32, features:f32) -> the same (base, w_ends, w_key,
    w_day) that concat+canonical_prep+cs_standardize+_compact_base produced, built in
    NUMPY with no concat frame, no sorted pandas copy and no groupby.ffill frame.
    Numerics are matched step for step: prices <=0 -> NaN -> per-stock ffill; flows
    log1p(clip 0); cs = per-timestamp per-month bincount (ddof=1, +1e-9); window
    validity = full SEQ_LEN within stock, all-finite."""
    n = sum(len(f) for f in frames)
    nf = len(FEATURE_COLS)
    date = np.empty(n, "datetime64[ns]")
    key = np.empty(n, np.int32)
    F = np.empty((n, nf), np.float32)
    o = 0
    for f in frames:
        m = len(f)
        date[o:o + m] = f["date"].to_numpy()
        key[o:o + m] = f["key"].to_numpy(np.int32)
        for j, c in enumerate(FEATURE_COLS):
            F[o:o + m, j] = f[c].to_numpy(np.float32)
        o += m
    perm = np.lexsort((date.view("i8"), key))          # (key, date) mergesort order
    date = date[perm]; key = key[perm]
    F = np.take(F, perm, axis=0)
    del perm
    day = date.astype("datetime64[D]")
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    bounds = np.r_[starts, n]
    price_idx = [FEATURE_COLS.index(c) for c in PRICE_LEVEL_FIELDS if c in FEATURE_COLS]
    flow_idx = [FEATURE_COLS.index(c) for c in FLOW_FIELDS if c in FEATURE_COLS]
    idx = np.arange(n, dtype=np.int64)
    for j in price_idx:
        col = F[:, j]
        col[col <= 0.0] = np.nan
        marker = np.where(np.isfinite(col), idx, -1)
        for s0, s1 in zip(bounds[:-1], bounds[1:]):    # ~1k stocks: cheap python loop
            np.maximum.accumulate(marker[s0:s1], out=marker[s0:s1])
        has = marker >= 0
        col[has] = col[marker[has]]                    # ffill; leading NaNs stay NaN
    for j in flow_idx:
        F[:, j] = np.log1p(np.clip(F[:, j], 0.0, None))
    # cs per bar timestamp, month by month (matches cs_standardize numerics exactly)
    mon = date.astype("datetime64[M]")
    for m in np.unique(mon):
        ix = np.flatnonzero(mon == m)
        codes, _u = pd.factorize(date[ix], sort=False)
        G = int(codes.max()) + 1 if len(codes) else 0
        V = F[ix].astype(np.float64)
        out = np.empty(V.shape, np.float32)
        for c in range(nf):
            col = V[:, c]
            ok = np.isfinite(col)
            cc, cv = codes[ok], col[ok]
            n1 = np.bincount(cc, minlength=G).astype(np.float64)
            s1 = np.bincount(cc, weights=cv, minlength=G)
            s2 = np.bincount(cc, weights=cv * cv, minlength=G)
            mean = np.divide(s1, n1, out=np.full(G, np.nan), where=n1 > 0)
            var = np.divide(s2 - n1 * mean * mean, n1 - 1,
                            out=np.full(G, np.nan), where=n1 > 1)
            std = np.sqrt(np.maximum(var, 0))
            out[:, c] = (col - mean[codes]) / (std[codes] + 1e-9)
        F[ix] = out
        del V, out
    # window ends: identical validity rule to _compact_base
    eod = np.flatnonzero(np.r_[(key[1:] != key[:-1]) | (day[1:] != day[:-1]), True])
    s0 = starts[np.searchsorted(starts, eod, side="right") - 1]
    badc = np.r_[0, np.cumsum(~np.isfinite(F).all(axis=1))]
    okhist = (eod - s0 + 1) >= SEQ_LEN
    p1 = eod + 1
    okfin = np.zeros(len(eod), bool)
    okfin[okhist] = (badc[p1[okhist]] - badc[p1[okhist] - SEQ_LEN]) == 0
    keep = okhist & okfin
    return F, p1[keep], key[eod[keep]], day[eod[keep]].astype("datetime64[ns]")


def _compact_base(df):
    """Sorted prepped frame -> (base f32 matrix, window end-offsets, end keys, end days).
    Replaces per-span groupby window building: one contiguous copy of the buffer's
    features; per-day scoring then slices base[e-SEQ_LEN:e] in O(1). Validity matches
    multi_day_windows exactly: full SEQ_LEN history within the stock, all-finite."""
    base = df[FEATURE_COLS].to_numpy(np.float32)
    key = df["key"].to_numpy()
    day = df["day"].to_numpy()
    eod = np.flatnonzero(np.r_[(key[1:] != key[:-1]) | (day[1:] != day[:-1]), True])
    stock_start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    s0 = stock_start[np.searchsorted(stock_start, eod, side="right") - 1]
    badc = np.r_[0, np.cumsum(~np.isfinite(base).all(axis=1))]
    okhist = (eod - s0 + 1) >= SEQ_LEN
    p1 = eod + 1
    okfin = np.zeros(len(eod), bool)
    okfin[okhist] = (badc[p1[okhist]] - badc[p1[okhist] - SEQ_LEN]) == 0
    keep = okhist & okfin
    return base, p1[keep], key[eod[keep]], day[eod[keep]]


def _slim(df):
    df["instrument"] = df["instrument"].astype(str)      # category (compression) -> str
    for c in FEATURE_COLS:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def _fetch_dai(table, start, end):
    """Memory-bounded fetch: push the auction-slice time filter into SQL (16x fewer
    rows); if the dialect rejects it, fall back to plain fetches in 10-day slices so
    a full month of 241-bar days is never materialized at once. Features float32."""
    import dai
    cols = ",".join(["date", "instrument"] + FEATURE_COLS)
    try:
        df = dai.query(f"SELECT {cols} FROM {table} WHERE CAST(date AS TIME) >= TIME '{TAIL_TIME}'",
                       filters={"date": [f"{start} 00:00:00", f"{end} 23:59:59"]},
                       compression=True).df()
        if len(df) == 0:
            raise ValueError("pushdown returned 0 rows")
        df = _slim(df)
    except Exception:
        parts, lo = [], pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        cutoff = pd.Timestamp(f"2000-01-01 {TAIL_TIME}").time()
        while lo <= end_ts:
            hi = min(lo + pd.Timedelta(days=9), end_ts)
            p = dai.query(f"SELECT {cols} FROM {table}",
                          filters={"date": [f"{lo.date()} 00:00:00", f"{hi.date()} 23:59:59"]},
                          compression=True).df()
            if len(p):
                p["date"] = pd.to_datetime(p["date"])
                p = p[p["date"].dt.time >= cutoff]        # slim BEFORE keeping
                if len(p):
                    parts.append(_slim(p))
            lo = hi + pd.Timedelta(days=1)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "instrument"] + FEATURE_COLS)
    df["key"] = df["instrument"]
    return df


def main(datasources, start_date, end_date, fetch_fn=None):
    """Platform entry point -> DataFrame [date, instrument, score] (higher = better).

    MEMORY-BOUNDED: scores month by month. Each scoring month is processed together
    with a rolling HIST_BUFFER_DAYS raw-bar buffer (for the 120-trading-day lookback),
    so peak memory is one ~4-month frame + one month of windows (~2 GB), independent
    of the evaluation span. Chunking does not change the scores: cs stats are per bar
    timestamp and windows depend only on each stock's trailing bars, both of which are
    fully contained in the buffer."""
    models, stats, _ = load_manifest(MODEL_PATH)
    if isinstance(datasources, str) and datasources.strip():
        table = datasources.strip()
    elif isinstance(datasources, dict) and datasources:
        table = None
        for k in ("bar1m", "bar_1m", "e2e_bar1m", "stock_bar1m"):
            if k in datasources:
                table = datasources[k]
                break
        table = table or next(iter(datasources.values()))
    else:
        table = CLOUD_TABLE_1M
    fetch = fetch_fn or (lambda s, e: _fetch_dai(table, s, e))
    cutoff = pd.Timestamp(f"2000-01-01 {TAIL_TIME}").time()

    # FACTORIZE AT FETCH (measured 2026-07-30): the str instrument column made every
    # cached 60-bar month ~260 MB deep -- five of them were +1.3 GB of the 6.24 GB smoke
    # peak. Encoding to int32 against a run-global registry the moment a month arrives
    # shrinks a cached month to ~60 MB, and every downstream stage (concat, sort, cs,
    # compact base) inherits the cut. Codes are consistent across months by construction.
    _icode, _inames = {}, []

    def _encode(inst):
        vals = inst.to_numpy()
        out = np.empty(len(vals), np.int32)
        get = _icode.get
        for i, v in enumerate(vals):
            c = get(v)
            if c is None:
                c = len(_inames)
                _icode[v] = c
                _inames.append(v)
            out[i] = c
        return out

    def fetch_month(mth):
        df = fetch(mth.start_time.strftime("%Y-%m-%d"), mth.end_time.strftime("%Y-%m-%d"))
        if df is None or len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.time >= cutoff]              # auction slice only
        df = df.drop(columns=["key"], errors="ignore")
        if len(df) == 0:
            return None
        df["key"] = _encode(df["instrument"].astype(str))
        df.drop(columns=["instrument"], inplace=True)
        return df

    fetch_lo = pd.Timestamp(start_date) - pd.Timedelta(days=HIST_BUFFER_DAYS)
    buf_months = int(np.ceil(HIST_BUFFER_DAYS / 28.0))     # raw months kept for lookback
    raw_cache = {}                                          # Period -> raw month frame
    outs = []
    for mth in pd.period_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="M"):
        need = pd.period_range(mth - buf_months, mth, freq="M")
        need = [p for p in need if p.end_time >= fetch_lo]
        for p in need:
            if p not in raw_cache:
                raw_cache[p] = fetch_month(p)
        for p in list(raw_cache):                           # drop months beyond the buffer
            if p not in need:
                del raw_cache[p]
        frames = [raw_cache[p] for p in need if raw_cache.get(p) is not None]
        if not frames:
            continue
        # ARRAY PIPELINE (v135 smoke forensics): no concat frame, no sorted pandas copy,
        # no groupby.ffill frame -- months are gathered straight into one preallocated
        # f32 matrix and prepped/cs'd/windowed in numpy. Score-identical to the old
        # concat+canonical_prep+cs_standardize+_compact_base path (replica-gated).
        inst_names = list(_inames)
        base, w_ends, w_key, w_day = _assemble_base(frames)
        import gc
        gc.collect()
        mean, std = stats
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nets = [n.to(dev) for n in models]     # SEED ENSEMBLE: mean of raw scores
        s = max(pd.Timestamp(start_date), mth.start_time)
        e = min(pd.Timestamp(end_date), mth.end_time)
        lo = s
        while lo <= e:                              # 1-day spans; HIST scores each day's
            dsel = np.flatnonzero(w_day == np.datetime64(lo.normalize()))   # whole cross-section
            if dsel.size:
                X = np.stack([base[e0 - SEQ_LEN:e0] for e0 in w_ends[dsel]])
                X -= mean
                X /= std
                with torch.no_grad():
                    xt = torch.from_numpy(X).to(dev)
                    sc = np.mean([n(xt).cpu().numpy() for n in nets], axis=0)
                    del xt
                outs.append(pd.DataFrame({"date": lo.strftime("%Y-%m-%d"),
                                          "instrument": np.asarray(inst_names)[w_key[dsel]],
                                          "score": sc}))
                del X
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            lo = lo + pd.Timedelta(days=1)
        del base, w_ends, w_key, w_day
        gc.collect()
        try:                                   # return freed heap to the OS between months
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    if not outs:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    out = pd.concat(outs, ignore_index=True)
    del outs

    # Coverage defense: OUTER-merge with the official universe; missing scores get the
    # day's median (neutral). Fail-soft off-platform.
    try:
        import dai
        pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",
                         filters={"date": [f"{pd.Timestamp(start_date).date()} 00:00:00",
                                           f"{pd.Timestamp(end_date).date()} 23:59:59"]}).df()
        pool["date"] = pd.to_datetime(pool["date"]).dt.strftime("%Y-%m-%d")
        pool["instrument"] = pool["instrument"].astype(str)
        pool = pool.drop_duplicates(["date", "instrument"])
        merged = pool.merge(out, on=["date", "instrument"], how="outer")
        med = merged.groupby("date")["score"].transform("median")
        merged["score"] = merged["score"].fillna(med).fillna(0.0)
        out = merged[["date", "instrument", "score"]].sort_values(["date", "instrument"]).reset_index(drop=True)
    except Exception:
        pass
    return out
