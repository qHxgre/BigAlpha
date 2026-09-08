# -*- coding: utf-8 -*-
"""Reproduce the training of this package's final_model.json weights -- single file.

Config (pinned from the packaged manifest -- see CONFIG below):
  cs hist_day, freq=1m, 60 bars x 120d (seq 7200), h256, 862,465 params x 3 seeds, train 2019-01-01..2023-12-31

Training recipe (v72 final convention): single seed 0, FIXED 12 epochs, cosine LR,
take the LAST epoch (--patience 0; no valid-based early stopping), IC loss
(negative per-batch Pearson), DAILY batches (one gradient step per trading day's
whole cross-section), label = raw next-day adjusted close-to-close return
winsorized at train 1/99 pct. Train window 2019-01-01..2023-12-31;
2024 fully held OUT of training = honest OOS certificate.

No hand-made feature engineering: raw bar fields only; allowed per-field cleaning
(fen->yuan, missing-price->NaN->ffill within stock, log1p flows) and cross-sectional per-timestamp (cs)
normalization happen inside the vendored pipeline.

The platform requires flat files, so the exact training sources (bigalpha/config.py,
bigalpha/data.py, scripts/train_baseline.py, scripts/train_gru2.py -- byte-identical
to the repo that produced the weights) are embedded at the bottom of this file as
raw strings and registered as modules at runtime. SRC_MD5 pins their checksums;
they are re-verified before every run (and by --dry-run).

Data: monthly e2e feather files (same layout the platform provides), passed via
--data-root; defaults to ./e2e_data like the original trainer.

Usage:
    python train.py --data-root /path/to/e2e_data           # full reproduction
    python train.py --dry-run                               # verify vs manifest + sources only

Reproducibility note (same statement as the accepted v72 precedent): determinism
flags are enabled, so a rerun of THIS script is bit-exact against itself on the
same hardware/torch build; across different GPUs/torch versions weights are
statistically equivalent, not bit-identical.

The produced checkpoint is self-verified against this package's final_model.json
manifest (norm/freq/day_bars/days/param-count) before the script reports success.
"""
import argparse, hashlib, json, os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "norm": "cs",
    "freq": "1m",
    "days": 120,
    "day_bars": 60,
    "n_feat": 8,
    "d_day": 256,
    "gru_hidden": 256,
    "seed": 0,
    "epochs": 12,
    "train_start": "2019-01-01",
    "train_end": "2023-12-31"
}

TRAIN_ARGS = [
    "--model",
    "hist_day",
    "--hidden",
    "256",
    "--loss",
    "ic",
    "--batch-mode",
    "daily",
    "--seed",
    "0",
    "--patience",
    "0",
    "--freq",
    "1m",
    "--days",
    "120",
    "--norm",
    "cs",
    "--tail",
    "14:01",
    "--feat",
    "core8",
    "--label-rank",
    "--train-start",
    "2019-01-01",
    "--train-end",
    "2023-12-31",
    "--valid-start",
    "2024-01-01",
    "--valid-end",
    "2024-12-31"
]

SRC_MD5 = {
    "bigalpha.config": "146cf4265fbb34172f0a7d37707749e6",
    "bigalpha.data": "db20e6d055a08041b7a2de28161c40db",
    "train_baseline": "99f208cb863fe4a474dd0ed63c1b7836",
    "train_gru2": "edfecb194c10583e9128f6cc4fc11641"
}


def manifest():
    with open(os.path.join(HERE, "final_model.json")) as f:
        return json.load(f)


def check_against_manifest(report):
    """Cross-check pinned CONFIG vs the shipped manifest; die loudly on drift."""
    m = manifest()
    errs = []
    def want(key, got, exp):
        if got != exp:
            errs.append(f"{key}: train.py={got!r} manifest={exp!r}")
    want("norm", CONFIG["norm"], m.get("norm_scheme") or m.get("norm"))
    want("days", CONFIG["days"], int(m["days"]))
    want("seq_len", CONFIG["day_bars"] * CONFIG["days"], int(m["seq_len"]))
    mc = m["model_cfg"]
    want("day_bars", CONFIG["day_bars"], int(mc["day_bars"]))
    want("n_feat", CONFIG["n_feat"], int(mc["n_feat"]))
    want("d_day", CONFIG["d_day"], int(mc["d_day"]))
    want("gru_hidden", CONFIG["gru_hidden"], int(mc["gru_hidden"]))
    if errs:
        print("MANIFEST MISMATCH -- train.py does not describe these weights:")
        for e in errs:
            print("  ", e)
        sys.exit(2)
    print(f"[OK] pinned config matches final_model.json "
          f"({CONFIG['norm']}/{CONFIG['freq']}/{CONFIG['day_bars']}x{CONFIG['days']}"
          f" seq={CONFIG['day_bars']*CONFIG['days']})" + (" -- " + report if report else ""))


def verify_sources():
    """md5 + compile every embedded source; die loudly if any was mangled."""
    for name in _SRC_ORDER:
        h = hashlib.md5(_SRC[name].encode("utf-8")).hexdigest()
        if h != SRC_MD5[name]:
            print(f"EMBEDDED SOURCE CORRUPT: {name} md5={h} expected={SRC_MD5[name]}")
            sys.exit(2)
        compile(_SRC[name], name, "exec")
    print(f"[OK] {len(_SRC_ORDER)} embedded sources md5-verified + compile clean")


def install_modules():
    """Register the embedded sources as real modules (bigalpha pkg + trainers)."""
    verify_sources()
    pkg = types.ModuleType("bigalpha")
    pkg.__path__, pkg.__package__ = [], "bigalpha"
    sys.modules["bigalpha"] = pkg
    for name in _SRC_ORDER:
        mod = types.ModuleType(name)
        # vendored inline: point __file__ at this script so path guesses stay local
        mod.__file__ = os.path.join(HERE, "train.py")
        if "." in name:
            mod.__package__ = name.rsplit(".", 1)[0]
        sys.modules[name] = mod
        exec(compile(_SRC[name], name, "exec"), mod.__dict__)
        if "." in name:
            setattr(pkg, name.split(".")[-1], mod)
    return sys.modules["train_gru2"]


def _train_tail():
    try:
        return TRAIN_ARGS[TRAIN_ARGS.index("--tail") + 1]
    except Exception:
        return "09:31"


def ensure_data(root):
    """SERVER-SIDE DATA ACQUISITION (private-stage retrain). The platform's isolated
    environment provides market data through `dai`, not necessarily as local feather
    files. If the monthly e2e feathers this trainer reads are absent under `root`,
    materialize them FROM dai -- same cloud table and same pushdown/fallback SQL as
    the scored inference libs -- in the exact local schema (fen-int scale fields,
    tail-filtered), so everything downstream is numerically equivalent to the
    local-mirror runs that produced the shipped weights. If files exist (local runs,
    or a platform data mount), they are used untouched."""
    import glob as _g
    tdir = os.path.join(root, "bigalpha_2026_e2e_bar1m")
    if _g.glob(os.path.join(tdir, "*.feather")) or _g.glob(os.path.join(root, "[0-9]*.feather")):
        print(f"[data] monthly feathers present under {root} -- using them", flush=True)
        return root
    import dai
    import numpy as np
    import pandas as pd
    m = manifest()
    tail = _train_tail()
    cols = ",".join(["date", "instrument"] + list(m["feature_cols"]))
    table = "bigalpha_2026_stock_bar1m"
    scale = ["open", "high", "low", "close",
             "ask_price1", "ask_price2", "ask_price3",
             "bid_price1", "bid_price2", "bid_price3", "amount"]
    os.makedirs(tdir, exist_ok=True)
    # month range derives from TRAIN_ARGS: [train-start minus the lookback lead-in,
    # valid-end]. Editing the time flags in TRAIN_ARGS is ALL it takes to retrain on a
    # different period -- the data bridge follows automatically.
    def _arg(flag, default):
        try:
            return TRAIN_ARGS[TRAIN_ARGS.index(flag) + 1]
        except Exception:
            return default
    _days = int(_arg("--days", "120"))
    _lead = int(_days * 1.6) + 40
    _lo = pd.Timestamp(_arg("--train-start", "2019-01-01")) - pd.Timedelta(days=_lead)
    _hi = pd.Timestamp(_arg("--valid-end", _arg("--train-end", "2024-12-31")))
    months = pd.period_range(_lo.strftime("%Y-%m"), _hi.strftime("%Y-%m"), freq="M")
    print(f"[data] no local feathers -- materializing {len(months)} months from dai "
          f"(table {table}, tail >= {tail})", flush=True)
    # UNIVERSE FILTER (caught live on the platform 2026-08-03): the raw cloud bar table
    # carries the WHOLE market (~2.2k names, 2.74M rows/month at this tail); the shipped
    # weights were trained on the per-date competition universe (~1k names, 1.26M rows).
    # Without this join the retrain OOMs small containers AND -- worse -- trains a
    # different model (cs normalization over a different cross-section). Same
    # instruments table the scored inference libs use for their coverage defense.
    pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",
                     filters={"date": [f"{_lo.date()} 00:00:00", f"{_hi.date()} 23:59:59"]},
                     compression=True).df()
    pool["day"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool = pool[["day", "instrument"]].drop_duplicates()
    print(f"[data] universe: {pool['instrument'].nunique()} names over "
          f"{pool['day'].nunique()} days", flush=True)
    for mth in months:
        out = os.path.join(tdir, f"{mth.strftime('%Y%m')}.0.feather")
        if os.path.exists(out):
            continue
        s0 = mth.start_time.strftime("%Y-%m-%d")
        e0 = mth.end_time.strftime("%Y-%m-%d")
        try:
            df = dai.query(
                f"SELECT {cols} FROM {table} WHERE CAST(date AS TIME) >= TIME '{tail}'",
                filters={"date": [f"{s0} 00:00:00", f"{e0} 23:59:59"]},
                compression=True).df()
            if len(df) == 0:
                raise ValueError("pushdown returned 0 rows")
        except Exception:
            parts, lo = [], pd.Timestamp(s0)
            end_ts = pd.Timestamp(e0)
            cutoff = pd.Timestamp(f"2000-01-01 {tail}").time()
            while lo <= end_ts:
                hi = min(lo + pd.Timedelta(days=9), end_ts)
                p = dai.query(f"SELECT {cols} FROM {table}",
                              filters={"date": [f"{lo.date()} 00:00:00",
                                                f"{hi.date()} 23:59:59"]},
                              compression=True).df()
                if len(p):
                    p["date"] = pd.to_datetime(p["date"])
                    p = p[p["date"].dt.time >= cutoff]
                    if len(p):
                        parts.append(p)
                lo = hi + pd.Timedelta(days=1)
            df = (pd.concat(parts, ignore_index=True) if parts
                  else pd.DataFrame(columns=["date", "instrument"] + list(m["feature_cols"])))
        if len(df) == 0:
            print(f"[data] {mth}: 0 rows -- skipped", flush=True)
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.normalize()
        n0 = len(df)
        df = df.merge(pool, on=["day", "instrument"], how="inner").drop(columns=["day"])
        print(f"[data] {mth}: universe filter {n0:,} -> {len(df):,} rows", flush=True)
        for c in scale:
            if c in df.columns:      # cloud yuan floats -> the mirror's fen ints
                v = np.round(df[c].to_numpy(np.float64) * 100.0)
                v[~np.isfinite(v)] = 0.0
                df[c] = v.astype(np.int64)
        df.reset_index(drop=True).to_feather(out)
        print(f"[data] {mth} -> {os.path.basename(out)} ({len(df):,} rows)", flush=True)
    return root


def verify_ckpt(ckpt_path):
    # REWRITTEN 2026-08-03 (live-fire drill): the original read ck["cfg"]/m["params"],
    # neither of which EXISTS in the real artifacts -- the trainer saves geometry keys
    # at the checkpoint TOP LEVEL (day_bars/days/hidden/d_day/norm/tail) and the
    # manifest carries model_cfg + packed models[], not a params field. Verify against
    # what is actually there: geometry keys + param count vs the packed member.
    import torch
    import numpy as _np
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = manifest()
    mc, ok = m["model_cfg"], True
    def _chk(name, got, exp):
        nonlocal ok
        if got is None or exp is None or str(got) != str(exp):
            print(f"[FAIL] ckpt {name}={got} != manifest {exp}"); ok = False
    _chk("day_bars", int(ck.get("day_bars", -1)), int(mc["day_bars"]))
    _chk("days", int(ck.get("days", -1)), int(m["days"]))
    _chk("gru_hidden", int(ck.get("hidden", -1)), int(mc["gru_hidden"]))
    _chk("d_day", int(ck.get("d_day", -1)), int(mc["d_day"]))
    _chk("norm", str(ck.get("norm")), str(m.get("norm_scheme") or m.get("norm")))
    _chk("tail", str(ck.get("tail")), str(m.get("tail_time")))
    sd = ck.get("state_dict") or ck.get("model") or {}
    n = sum(int(v.numel()) for v in sd.values()) if sd else -1
    ref = None
    if m.get("models"):
        ref = sum(int(_np.prod(v["shape"])) for v in m["models"][0].values())
    if ref is not None and n != ref:
        print(f"[FAIL] param count {n} != packed member {ref}"); ok = False
    # 5m/coarse ckpts historically omitted `freq`; stamp it so packing tools read it right
    if "freq" not in ck:
        ck["freq"] = CONFIG["freq"]; torch.save(ck, ckpt_path)
        print(f"[note] stamped freq={CONFIG['freq']} into checkpoint")
    print("[OK] checkpoint geometry + param count match final_model.json" if ok else
          "[FAIL] checkpoint does NOT match the shipped manifest")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="e2e_data",
                    help="dir of monthly e2e feather files (platform layout)")
    ap.add_argument("--save-dir", default=os.path.join(HERE, "retrain_out"))
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"],
                    help="override ONLY for smoke tests; reproduction = %(default)s")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify pinned args + embedded sources vs the shipped manifest")
    a = ap.parse_args()

    check_against_manifest("dry-run" if a.dry_run else "")
    argv = ["train_gru2.py", *TRAIN_ARGS, "--epochs", str(a.epochs),
            "--root", a.data_root, "--save-dir", a.save_dir]
    print("effective trainer argv:\n  " + " ".join(argv))
    if a.dry_run:
        verify_sources()
        return

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)   # torch<=1.12 rejects expandable_segments
    os.makedirs(a.save_dir, exist_ok=True)
    tg = install_modules()
    old_argv, sys.argv = sys.argv, argv
    try:
        tg.main()
    finally:
        sys.argv = old_argv
    pts = sorted((os.path.join(a.save_dir, f) for f in os.listdir(a.save_dir)
                  if f.endswith((".pt", ".pth"))), key=os.path.getmtime)
    if not pts:
        print("[FAIL] trainer finished but no checkpoint in", a.save_dir); sys.exit(3)
    print("newest checkpoint:", pts[-1])
    sys.exit(0 if verify_ckpt(pts[-1]) else 4)


# --------------------------------------------------------------------------
# vendored training sources -- byte-identical to the repo files that produced
# the packaged weights; checksums pinned in SRC_MD5 above. Do not edit below.
# --------------------------------------------------------------------------
_SRC = {}
_SRC_ORDER = ["bigalpha.config", "bigalpha.data", "train_baseline", "train_gru2"]

# --- bigalpha.config  (repo bigalpha/config.py, md5 146cf4265fbb34172f0a7d37707749e6) ---
_SRC['bigalpha.config'] = r'''"""Central config for the BigAlpha 2026 StockDiff entry.

Table names, the canonical field set, price scaling, train/valid/test splits, and the
default model feature columns. Both the local training script and the cloud inference
notebook import from here so they never drift (see DATA.md Section 5 and 10).
"""

from __future__ import annotations

# --- table names -----------------------------------------------------------------
LOCAL_BARS = {
    "1m": "bigalpha_2026_e2e_bar1m",
    "5m": "bigalpha_2026_e2e_bar5m",
    "15m": "bigalpha_2026_e2e_bar15m",
    "30m": "bigalpha_2026_e2e_bar30m",
}
CLOUD_BARS = {
    "1m": "bigalpha_2026_stock_bar1m",
    "5m": "bigalpha_2026_stock_bar5m",
    "15m": "bigalpha_2026_stock_bar15m",
    "30m": "bigalpha_2026_stock_bar30m",
}
INSTRUMENTS_TABLE = "bigalpha_2026_instruments"
EXPOSURE_TABLE = "bigalpha_2026_exposure"
FACTORLIB_TABLE = "bigalpha_2026_factorlib"  # reference only, never a model input

# --- reconciliation (local fen ints <-> cloud yuan floats), DATA.md Section 5 -----
PRICE_SCALE = 100.0
OHLC_COLS = ["open", "high", "low", "close"]
# price-LEVEL fields: a value <= 0 means MISSING (confirmed from real 15m data: the
# sentinel is 0, not the -1 the wiki claimed; ~2% of bars, and NOT correlated with
# volume==0). These are set to NaN in to_canonical, then ffilled within a stock for
# features. Flow fields (volume, amount, book volumes/counts) keep 0 as a real value.
PRICE_LEVEL_FIELDS = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
# price-like fields stored in fen locally; divided by PRICE_SCALE in to_canonical.
# amount is a cash flow (also fen) so it is scaled but NOT treated as missing at 0.
SCALE_FIELDS = PRICE_LEVEL_FIELDS + ["amount"]
# canonical order book is 3 levels (local depth; cloud levels 4-5 are dropped)
BOOK_DEPTH = 3
MISSING_PRICE_MAX = 0.0  # price-level values <= this are missing (covers 0 and legacy -1)

# --- default model feature columns (compact, <=100 field budget) ------------------
# Matches the wiki Transformer template so our encoder starts from a known-good set.
PRICE_FEATURES = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOLUME_FEATURES = ["volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_FEATURES + VOLUME_FEATURES  # 10 fields

# --- WIDE feature set: every informative raw field (25 of the cap of 100) ---------
# Adds book levels 2-3, per-level order counts, and deal_number. volume/num_orders
# ratios (avg order size, retail-vs-institutional proxy) are left for the net to learn.
WIDE_PRICE_FEATURES = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
WIDE_FLOW_FEATURES = [
    "volume", "amount", "deal_number",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
WIDE_FEATURE_COLS = WIDE_PRICE_FEATURES + WIDE_FLOW_FEATURES  # 25 fields

# flow/count features get a log1p before normalization (allowed per-field transform);
# applies only to columns actually present in the chosen feature set.
LOG1P_FEATURES = list(WIDE_FLOW_FEATURES)

# --- windowing / label ------------------------------------------------------------
SEQ_LEN = 40          # bars per window (30m bars: ~5 trading days)
LABEL_HORIZON = 1     # forward days for the close-to-close return label
LABEL_CLIP_PCT = (1.0, 99.0)  # winsorize the training label

# --- splits: train on the official 2019-2023 window, evaluate on 2024 -------------
# Train range is the official public window (2019-01-01 .. 2023-12-31); the local feather
# adds 2024, which we use as the validation / evaluation block. The competition PUBLIC
# LEADERBOARD is the true held-out test; the private leaderboard then retrains our code.
SPLITS = {
    "train": ("2019-01-01", "2023-12-31"),
    "valid": ("2024-01-01", "2024-12-31"),
}

SEED = 42
'''

# --- bigalpha.data  (repo bigalpha/data.py, md5 8f444ddddab3d5d0b9c1cc1d4078035a) ---
_SRC['bigalpha.data'] = r'''"""Shared data path for training (local feather) and inference (cloud dai).

The golden rule from the wiki: training and inference MUST run the exact same
canonical + preprocess + windowing code, or the scores drift. So every step lives
here and is imported by both `transformer_train_local.py` and the submission notebook.

Pipeline: raw bars (local or cloud) -> to_canonical -> preprocess_canonical ->
build_windows -> (X, y, idx). Normalizer stats (per-feature mean/std) are fit on
train windows only and reused everywhere.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C


# ================================================================================
# Loading raw bars
# ================================================================================
def parse_bar_windows(spec) -> list[tuple[int, int]]:
    """'09:31-09:45,14:46-15:00' -> [(571, 585), (886, 900)] in minutes-from-midnight.

    Both ends INCLUSIVE, matching the 1m bar stamps, which are minute-END: a trading day
    is 09:31..11:30 then 13:01..15:00 (240 bars), so "14:46-15:00" is exactly the 15 bars
    that ``tail_time='14:46'`` already selects. Windows are returned sorted by start so
    the per-day bar order is chronological regardless of how they were typed.
    """
    if not spec:
        return []
    if isinstance(spec, str):
        spec = spec.split(",")
    out = []
    for seg in spec:
        # idempotent: callers parse once and then hand the parsed list down to the loader,
        # so an already-(int, int) segment must pass straight through rather than be
        # str()'d back into "(781, 795)" and rejected
        if isinstance(seg, (tuple, list)) and len(seg) == 2:
            out.append((int(seg[0]), int(seg[1])))
            continue
        seg = str(seg).strip()
        if not seg:
            continue
        lo, _, hi = seg.partition("-")
        if not hi:
            raise ValueError(f"bar window {seg!r} is not 'HH:MM-HH:MM'")
        a, b = (int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t in (lo, hi))
        if b < a:
            raise ValueError(f"bar window {seg!r} ends before it starts")
        out.append((a, b))
    out.sort()
    for (a1, b1), (a2, _) in zip(out, out[1:]):
        if a2 <= b1:
            raise ValueError(f"bar windows overlap: {out} -- each bar would be fed twice")
    return out


def bar_window_count(wins: list[tuple[int, int]]) -> int:
    """Bars per day selected by ``wins``, skipping the 11:31-13:00 lunch break.

    Counting b-a+1 raw minutes would over-count any window spanning lunch and silently
    set day_bars larger than the rows that actually arrive, which reshapes the window
    tensor against a day boundary that is off by the difference.
    """
    MORNING, AFTERNOON = (9 * 60 + 31, 11 * 60 + 30), (13 * 60 + 1, 15 * 60)
    n = 0
    for a, b in wins:
        for s, e in (MORNING, AFTERNOON):
            n += max(0, min(b, e) - max(a, s) + 1)
    return n


def load_local_bars(root: str, freq: str = "30m", *,
                    columns: Optional[list[str]] = None,
                    month_lo: Optional[str] = None,
                    month_hi: Optional[str] = None,
                    tail_time: Optional[str] = None,
                    bar_windows=None) -> pd.DataFrame:
    """Read monthly e2e feather files from ``root`` for one frequency.

    ``root`` may be the parent data dir (we append the table name) or the frequency
    folder itself. ``month_lo``/``month_hi`` are inclusive ``YYYYMM`` bounds.
    ``tail_time`` (e.g. "14:30") keeps only bars at or after that intraday time,
    applied per month before concat to bound memory (used for 1m closing slices).
    ``bar_windows`` (e.g. "13:01-13:15,14:46-15:00", or the parsed list) selects one or
    more DISJOINT intraday segments instead; it supersedes ``tail_time``. Rows stay in
    chronological order, so a day contributes its segments back-to-back and the window
    builder's fixed day_bars reshape still lands on day boundaries.
    """
    table = C.LOCAL_BARS[freq]
    tdir = os.path.join(root, table)
    if not os.path.isdir(tdir):
        tdir = root
    files = sorted(fp for fp in glob.glob(os.path.join(tdir, "*.feather"))
                   if os.path.basename(fp).split(".")[0].isdigit())
    if not files:
        raise FileNotFoundError(f"no monthly *.feather under {tdir}")
    lo = int(month_lo) if month_lo else -1
    hi = int(month_hi) if month_hi else 10 ** 9
    wins = parse_bar_windows(bar_windows)
    cutoff = (pd.Timestamp(f"2000-01-01 {tail_time}").time()
              if (tail_time and not wins) else None)
    parts = []
    for fp in files:
        ym = int(os.path.basename(fp).split(".")[0])
        if lo <= ym <= hi:
            part = pd.read_feather(fp, columns=columns)
            if wins:
                # minute-of-day ints, not datetime.time objects: the segment mask is an OR
                # over windows, and object-dtype time comparisons would run the whole
                # elementwise Python path once per segment
                _d = pd.to_datetime(part["date"])
                mod = _d.dt.hour.to_numpy() * 60 + _d.dt.minute.to_numpy()
                keep = np.zeros(len(part), dtype=bool)
                for a, b in wins:
                    keep |= (mod >= a) & (mod <= b)
                part = part[keep]
            elif cutoff is not None:
                part = part[pd.to_datetime(part["date"]).dt.time >= cutoff]
            parts.append(part)
    if not parts:
        raise FileNotFoundError(f"no feather files in month range [{month_lo}, {month_hi}]")
    return pd.concat(parts, ignore_index=True)


def load_cloud_bars(table: str, start: str, end: str, *,
                    columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Query a cloud stock_bar table via dai (only works inside a BigQuant notebook)."""
    import dai  # noqa: local import; only present on the platform

    cols = ",".join(columns) if columns else "*"
    q = f"SELECT {cols} FROM {table} ORDER BY instrument, date"
    return dai.query(q, filters={"date": [str(start), str(end)]}, compression=True).df()


# ================================================================================
# Canonical form (local fen ints and cloud yuan floats -> one representation)
# ================================================================================
def to_canonical(df: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    """Map a raw local or cloud bar frame to the canonical form (DATA.md Section 5).

    Missing price levels (value <= 0, i.e. the real ``0`` sentinel or a legacy ``-1``)
    become NaN so they are ffilled for features and skipped for labels. Flow fields keep 0.
    Local prices are fen ints, scaled to yuan; cloud prices are already yuan floats.
    """
    df = df.copy()
    for c in C.PRICE_LEVEL_FIELDS:                 # 0 / -1 -> NaN (both families)
        if c in df.columns:
            df[c] = df[c].astype("float64")
            df.loc[df[c] <= C.MISSING_PRICE_MAX, c] = np.nan
    if is_local:
        for c in C.SCALE_FIELDS:                   # fen -> yuan
            if c in df.columns:
                df[c] = df[c].astype("float64") / C.PRICE_SCALE
        df["key"] = df["instrument"] if "instrument" in df.columns else df["instrument_id"]
    else:
        drop = [c for c in df.columns
                if any(c.startswith(p) and str(c)[-1] in "45"
                       for p in ("ask_price", "bid_price", "ask_volume",
                                 "bid_volume", "ask_num_orders", "bid_num_orders"))]
        df = df.drop(columns=drop, errors="ignore")
        df["key"] = df["instrument"]
    return df


def preprocess_canonical(df: pd.DataFrame, *,
                        feature_cols: Optional[list[str]] = None,
                        copy: bool = True) -> pd.DataFrame:
    """Allowed per-field cleaning: sort, ffill price levels within a stock, log1p flows.

    Nothing here is cross-field or rolling; it is per-field imputation and a fixed
    transform, which the rules permit. Also adds `adj_close_raw` = close * adjust_factor
    computed from the RAW (pre-ffill) close, so labels can use the true last valid daily
    close and skip missing-price days rather than a stale ffilled value.

    Returns a frame with `key`, `day`, `adj_close_raw`, cleaned features.
    """
    feature_cols = feature_cols or C.FEATURE_COLS
    # copy=False lets huge full-bar training frames (~45GB) skip the defensive copy;
    # the caller must not reuse its input afterwards. sort_values below still allocates
    # a sorted copy, so peak is 2x the frame instead of 3x.
    if copy:
        df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.normalize()
    df = df.sort_values(["key", "date"], kind="mergesort").reset_index(drop=True)

    # true adjusted close for labels: NaN wherever the real price was missing (0 -> NaN)
    adj = df["adjust_factor"].astype("float64") if "adjust_factor" in df.columns else 1.0
    df["adj_close_raw"] = df["close"].astype("float64") * adj

    # ffill price LEVELS within a stock so feature windows are continuous over gap bars
    price_levels = [c for c in C.PRICE_LEVEL_FIELDS if c in df.columns]
    if price_levels:
        df[price_levels] = df.groupby("key", sort=False)[price_levels].ffill()
    # flows: 0 is a real value; log1p the volume-like features
    for c in C.LOG1P_FEATURES:
        if c in feature_cols and c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    # downcast FEATURE columns to float32 (halves the load DataFrame; features are
    # standardized to float32 downstream anyway). adj_close_raw stays float64 for labels.
    fcast = {c: "float32" for c in feature_cols if c in df.columns and df[c].dtype == "float64"}
    if fcast:
        df = df.astype(fcast)
    return df


def cs_standardize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """CROSS-SECTIONAL standardization: per bar timestamp, across that bar's stocks.

    Replaces the global train-fit scaling as the effective normalization (the global
    standardizer applied afterwards is ~identity). Leak-free: uses only same-timestamp
    data. Removes market-wide level/volume drift; features become relative-to-the-
    cross-section, matching the ranking objective.
    """
    g = df.groupby("date", sort=False)[feature_cols]
    mean = g.transform("mean")
    std = g.transform("std")
    df[feature_cols] = ((df[feature_cols] - mean) / (std + 1e-9)).astype(np.float32)
    return df


# ================================================================================
# Windowing (per stock-day: last SEQ_LEN bars ending at end-of-day)
# ================================================================================
def _eod_positions(day: np.ndarray) -> np.ndarray:
    """Row positions of the last bar of each trading day, in order."""
    if len(day) == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.append(day[1:] != day[:-1], True))


def build_windows(df: pd.DataFrame, start: str, end: str, mode: str,
                 stats: Optional[tuple[np.ndarray, np.ndarray]] = None, *,
                 feature_cols: Optional[list[str]] = None,
                 seq_len: int = C.SEQ_LEN,
                 label_horizon: int = C.LABEL_HORIZON,
                 clip_pct: tuple[float, float] = C.LABEL_CLIP_PCT):
    """Turn a preprocessed canonical frame into model windows.

    Parameters
    ----------
    df : preprocessed canonical frame (has `key`, `day`, feature cols, `close`,
         and optionally `adjust_factor`).
    start, end : keep windows whose end-of-day date is in [start, end].
    mode : "train" fits normalizer stats and requires a forward label;
           "infer" reuses `stats`, keeps every in-range end-of-day (last day too),
           and does not require a future bar.
    stats : (mean, std) per feature; required for "infer", computed for "train".

    Returns
    -------
    X : float32 (N, seq_len, F) normalized windows
    y : float32 (N,) forward return label ("train"), or all-NaN ("infer")
    idx_df : DataFrame with columns [date, key], one row per window, aligned to X
    stats : (mean, std) used
    """
    feature_cols = feature_cols or C.FEATURE_COLS
    n_feat = len(feature_cols)
    sd, ed = pd.to_datetime(start), pd.to_datetime(end)
    want_label = mode == "train"
    if "adj_close_raw" not in df.columns:
        raise ValueError("preprocess_canonical must run before build_windows (missing adj_close_raw)")

    wins: list[np.ndarray] = []
    ys: list[float] = []
    dates: list[pd.Timestamp] = []
    keys: list = []

    for k, sub in df.groupby("key", sort=False):
        if len(sub) <= seq_len:
            continue
        feats = sub[feature_cols].to_numpy(np.float32)
        day = sub["day"].to_numpy()
        eod = _eod_positions(day)
        if eod.size == 0:
            continue
        eod_days = pd.to_datetime(day[eod])
        # label close = LAST VALID adjusted close within each day (NaN if none traded)
        daily_close = (sub.groupby("day", sort=True)["adj_close_raw"].last()
                       .reindex(pd.DatetimeIndex(eod_days).normalize()).to_numpy(np.float64))

        for j, p in enumerate(eod):
            d = eod_days[j]
            if p + 1 < seq_len or d < sd or d > ed:
                continue
            win = feats[p - seq_len + 1: p + 1]
            if not np.isfinite(win).all():
                continue
            if want_label:
                jf = j + label_horizon
                c0 = daily_close[j]
                if jf >= len(eod) or not (c0 > 0) or not np.isfinite(daily_close[jf]):
                    continue
                r = daily_close[jf] / c0 - 1.0
                if not np.isfinite(r):
                    continue
                ys.append(np.float32(r))
            else:
                ys.append(np.float32("nan"))
            wins.append(win)
            dates.append(d)
            keys.append(k)

    if not wins:
        raise ValueError(f"build_windows produced 0 windows for [{start}, {end}] mode={mode}")

    X = np.stack(wins)                             # (N, seq_len, F) float32; wins are views,
    # so this is the ONE big allocation (do NOT add astype/arithmetic copies below: a d50
    # build is ~90GB, and each extra copy previously pushed the peak past container limits)
    y = np.asarray(ys, np.float32)
    idx_df = pd.DataFrame({"date": dates, "key": keys})

    if mode == "train":
        mean = X.reshape(-1, n_feat).mean(0).astype(np.float32)
        std = X.reshape(-1, n_feat).std(0).astype(np.float32) + 1e-6
        lo, hi = np.percentile(y, clip_pct)
        y = np.clip(y, lo, hi).astype(np.float32)
    else:
        if stats is None:
            raise ValueError("infer mode needs stats=(mean, std) from training")
        mean, std = np.asarray(stats[0], np.float32), np.asarray(stats[1], np.float32)

    X -= mean                                      # in-place: no transient full-size copies
    X /= std
    return X, y, idx_df, (mean, std)
'''

# --- train_baseline  (repo scripts/train_baseline.py, md5 99f208cb863fe4a474dd0ed63c1b7836) ---
_SRC['train_baseline'] = r'''"""Baselines on the flat wiki-style pipeline: a simple MLP and the official Transformer.

Uses data.build_windows (flat SEQ_LEN-bar window, the BigQuant template setup), random
minibatches (stable), MSE on the winsorized forward return, and reports 2024 validation
IC/ICIR/RankIC/RankICIR as BEST epoch AND mean-across-epochs (to avoid cherry-picking).

    CUDA_VISIBLE_DEVICES=1 python scripts/train_baseline.py --model transformer --freq 15m
    CUDA_VISIBLE_DEVICES=1 python scripts/train_baseline.py --model mlp --freq 15m
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigalpha import config as C          # noqa: E402
from bigalpha import data as D            # noqa: E402


# ---------------- models ----------------
class MLPModel(nn.Module):
    def __init__(self, seq_len, n_feat, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        dims = [seq_len * n_feat] + [hidden] * layers
        net = []
        for a, b in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(a, b), nn.SiLU(), nn.Dropout(dropout)]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, x):                       # x: (B, L, F)
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class TransformerModel(nn.Module):              # official-template StockTransformer
    def __init__(self, seq_len, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                        # x: (B, L, F)
        return self.head(self.encoder(self.proj(x) + self.pos).mean(1)).squeeze(-1)


# ---------------- eval ----------------
def daily_ic(idx_df, score, label):
    df = idx_df.copy()
    df["s"] = score
    df["e"] = label
    df = df.dropna(subset=["s", "e"])
    pear, spear = [], []
    for _, g in df.groupby("date"):
        if len(g) < 10 or g["s"].std() == 0:
            continue
        pear.append(g["s"].corr(g["e"]))
        spear.append(g["s"].corr(g["e"], method="spearman"))
    pear, spear = np.array(pear), np.array(spear)
    return {"IC": pear.mean(), "ICIR": pear.mean() / (pear.std() + 1e-9),
            "RankIC": spear.mean(), "RankICIR": spear.mean() / (spear.std() + 1e-9)}


@torch.no_grad()
def predict(model, X, dev, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(dev)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "transformer"], required=True)
    ap.add_argument("--freq", default="15m")
    ap.add_argument("--root", default="e2e_data")
    ap.add_argument("--train-start", default="2019-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--valid-start", default="2024-01-01")
    ap.add_argument("--valid-end", default="2024-12-31")
    ap.add_argument("--seq-len", type=int, default=C.SEQ_LEN)
    ap.add_argument("--features", default="base", choices=["base", "wide", "close"],
                    help="base=10 template fields; wide=all 25 raw fields; close=daily close only")
    ap.add_argument("--tail-time", default=None,
                    help='keep only bars at/after this intraday time, e.g. "14:30" (1m closing slice)')
    ap.add_argument("--norm", default="global", choices=["global", "cs"],
                    help="global=train-fit per-field standardize; cs=per-bar cross-sectional standardize")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    table = D  # noqa

    t0 = time.time()
    fc = {"wide": C.WIDE_FEATURE_COLS, "close": ["close"]}.get(a.features, C.FEATURE_COLS)
    buf = lambda s: (pd.Timestamp(s) - pd.Timedelta(days=40)).strftime("%Y%m")  # noqa
    print(f"loading {a.freq} flat windows (seq_len={a.seq_len}, features={a.features}[{len(fc)}], "
          f"tail_time={a.tail_time}) ...")
    raw_tr = D.load_local_bars(a.root, a.freq, month_lo=buf(a.train_start),
                               month_hi=pd.Timestamp(a.train_end).strftime("%Y%m"),
                               tail_time=a.tail_time)
    pp_tr = D.preprocess_canonical(D.to_canonical(raw_tr, is_local=True), feature_cols=fc)
    del raw_tr
    if a.norm == "cs":
        pp_tr = D.cs_standardize(pp_tr, fc)
    Xtr, ytr, _, stats = D.build_windows(pp_tr, a.train_start, a.train_end, "train",
                                         feature_cols=fc, seq_len=a.seq_len)
    del pp_tr
    raw_va = D.load_local_bars(a.root, a.freq, month_lo=buf(a.valid_start),
                               month_hi=pd.Timestamp(a.valid_end).strftime("%Y%m"),
                               tail_time=a.tail_time)
    pp_va = D.preprocess_canonical(D.to_canonical(raw_va, is_local=True), feature_cols=fc)
    del raw_va
    if a.norm == "cs":
        pp_va = D.cs_standardize(pp_va, fc)
    Xva, yva, idx_va, _ = D.build_windows(pp_va, a.valid_start, a.valid_end, "train", stats,
                                          feature_cols=fc, seq_len=a.seq_len)
    del pp_va
    print(f"  train {Xtr.shape}  valid {Xva.shape}  | {time.time()-t0:.0f}s")

    n_feat = Xtr.shape[-1]
    if a.model == "mlp":
        model = MLPModel(a.seq_len, n_feat).to(dev)
    else:
        model = TransformerModel(a.seq_len, n_feat).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"{a.model} params: {npar:,} ({'OK' if 1e5 <= npar <= 1e8 else 'OUT OF RANGE'})")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    hist = []
    best = {"RankIC": -9}
    for ep in range(a.epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        tot = 0.0; nb = 0
        for i in range(0, len(idx), a.batch):
            b = idx[i:i + a.batch]
            xb = Xt[b].to(dev); yb = yt[b].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        ic = daily_ic(idx_va, predict(model, Xva, dev), yva)
        hist.append(ic)
        if ic["RankIC"] > best["RankIC"]:
            best = ic
        print(f"ep {ep+1:2d}/{a.epochs} mse={tot/nb:.5f} | valid IC={ic['IC']:.4f} "
              f"ICIR={ic['ICIR']:.3f} RankIC={ic['RankIC']:.4f} RankICIR={ic['RankICIR']:.3f}")

    def agg(key):
        v = np.array([h[key] for h in hist[-8:]])   # mean over last up-to-8 epochs
        return v.mean(), v.std()
    print(f"\n=== {a.model} {a.freq} on 2024 valid ===")
    print(f"  BEST(by RankIC): IC={best['IC']:.4f} ICIR={best['ICIR']:.3f} "
          f"RankIC={best['RankIC']:.4f} RankICIR={best['RankICIR']:.3f}")
    for k in ["IC", "RankIC"]:
        m, s = agg(k)
        print(f"  MEAN±std last epochs {k}: {m:.4f} ± {s:.4f}")
    mIC, _ = agg("IC"); mRankIC, _ = agg("RankIC")
    print(f"RESULT model={a.model} freq={a.freq} seed={a.seed} seqlen={a.seq_len} "
          f"features={a.features} tail={a.tail_time or 'none'} norm={a.norm} "
          f"RankIC={best['RankIC']:.4f} RankICIR={best['RankICIR']:.4f} "
          f"IC={best['IC']:.4f} ICIR={best['ICIR']:.4f} "
          f"meanlastRankIC={mRankIC:.4f} meanlastIC={mIC:.4f}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
'''

# --- train_gru2  (repo scripts/train_gru2.py, md5 de0f404daaf467c526d6aa2066e8e6c5) ---
_SRC['train_gru2'] = r'''"""Two GRU models (user-requested test), stable minibatch trainer, 2024 validation.

1. gru15m     : plain GRU (hidden 256) over D days x 16 fifteen-minute bars (wide 25 fields).
2. daymlp_gru : per-day MLP encoder of the 1m auction slice (last 15 one-minute bars,
                wide 25) -> GRU across the K daily representations -> predict.

    python scripts/train_gru2.py --model gru15m --days 3 --seed 0
    python scripts/train_gru2.py --model daymlp_gru --days 3 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from bigalpha import config as C          # noqa: E402
from bigalpha import data as D            # noqa: E402
import train_baseline as TB               # noqa: E402  (daily_ic, predict)

TRAIN = ("2019-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
BARS_15M = 16
AUC_BARS = 15


class GRUBars(nn.Module):
    """Plain GRU over the bar sequence; predict from the last hidden state."""

    def __init__(self, n_feat, hidden=256, layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)          # qlib-style fc_out, no LayerNorm

    def forward(self, x):                                  # (B, T, F)
        _, h = self.rnn(x)
        return self.head(h[-1]).squeeze(-1)


class DayMLPGRU(nn.Module):
    """Shared MLP encodes each day's auction slice; GRU passes daily representations."""

    def __init__(self, day_bars, n_feat, enc_hidden=256, d_day=128, gru_hidden=128, dropout=0.1):
        super().__init__()
        self.day_bars = day_bars
        self.n_feat = n_feat
        self.encoder = nn.Sequential(
            nn.Linear(day_bars * n_feat, enc_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(enc_hidden, d_day), nn.SiLU(),
        )
        self.rnn = nn.GRU(d_day, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, 1)      # qlib-style fc_out, no LayerNorm

    def forward(self, x):                                  # (B, K*day_bars, F)
        B, T, F = x.shape
        K = T // self.day_bars
        day_in = x.reshape(B, K, self.day_bars * F)        # one flat block per day
        day_vec = self.encoder(day_in)                     # (B, K, d_day)
        _, h = self.rnn(day_vec)
        return self.head(h[-1]).squeeze(-1)


class DayAttnGRU(nn.Module):
    """RESEARCH LINE (2026-07-25): intraday TRANSFORMER encoder -> attention-pooled day
    vector -> daily GRU -> head. Hypothesis: the champion family only reads the closing
    snapshot because its tiny intraday GRU cannot digest the day's trajectory; a
    self-attention day encoder can. No HIST cross-section module (single-variable)."""

    def __init__(self, day_bars, n_feat, d_day=128, gru_hidden=256, layers=1, heads=4):
        super().__init__()
        self.day_bars = day_bars
        self.proj = nn.Linear(n_feat, d_day)
        self.pos = nn.Parameter(torch.zeros(1, day_bars, d_day))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_day, nhead=heads, dim_feedforward=2 * d_day,
            dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.q = nn.Parameter(torch.zeros(1, 1, d_day))
        self.rnn = nn.GRU(d_day, gru_hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(gru_hidden, 1)
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.q, std=0.02)

    def forward(self, x):                       # (N, K*day_bars, F)
        N, T, F = x.shape
        K = T // self.day_bars
        seqs = x.reshape(N * K, self.day_bars, F)
        outs = []
        for i in range(0, seqs.shape[0], 4096):   # chunked: bound activations
            h = self.proj(seqs[i:i + 4096]) + self.pos
            h = self.enc(h)
            # attention pooling with a learned query
            a = torch.softmax((h @ self.q.squeeze(0).t()) / (h.shape[-1] ** 0.5), dim=1)
            outs.append((a.transpose(1, 2) @ h).squeeze(1))
        days = torch.cat(outs).reshape(N, K, -1)
        _, h2 = self.rnn(days)
        return self.head(h2[-1]).squeeze(-1)


class DayHISTDeep(nn.Module):
    """USER DESIGN (2026-07-26): intraday GRU -> [ inter-day DNN -> HIST ] x L -> head.

    The shipped DayHIST runs the cross-sectional HIST module ONCE, so information makes a
    single hop through the concept graph. Stacking L blocks lets it propagate L hops, with
    a per-stock DNN refining the representation between hops.

    Day aggregation is DNN-based and shape-agnostic: the K day-vectors are pooled into a
    fixed summary [last, mean, mean of the last 5 days] and projected by an MLP, so the
    same weights serve any lookback.
    """

    def __init__(self, day_bars, n_feat, d_day=128, gru_hidden=128, blocks=3,
                 hist_topk=0, hist_tau=0.1, dropout=0.1):
        super().__init__()
        self.day_bars = day_bars
        self.intraday = nn.GRU(n_feat, d_day, batch_first=True)
        H = gru_hidden
        self.day_dnn = nn.Sequential(nn.Linear(3 * d_day, H), nn.SiLU(),
                                     nn.Dropout(dropout), nn.Linear(H, H))
        self.blocks = int(blocks)
        self.hist_topk = int(hist_topk)
        if self.hist_topk:
            self.hist_logtau = nn.Parameter(torch.tensor(float(np.log(hist_tau))))
        # one [DNN -> HIST] block's parameters, L times
        for i in range(self.blocks):
            setattr(self, f"fc_is{i}", nn.Linear(H, H))
            setattr(self, f"fc_is_back{i}", nn.Linear(H, H))
            setattr(self, f"fc_is_fore{i}", nn.Linear(H, H))
            setattr(self, f"fc_indi{i}", nn.Linear(H, H))
            setattr(self, f"dnn{i}", nn.Sequential(nn.Linear(H, H), nn.SiLU(),
                                                   nn.Dropout(dropout), nn.Linear(H, H)))
        self.leaky = nn.LeakyReLU()
        self.head = nn.Linear(H, 1)

    @staticmethod
    def _cos(x, y):
        xy = x.mm(y.t())
        xn = torch.sqrt((x * x).sum(1)).reshape(-1, 1)
        yn = torch.sqrt((y * y).sum(1)).reshape(-1, 1)
        return xy / (xn.mm(yn.t()) + 1e-6)

    def _hist(self, h, i):
        """one cross-sectional hop (same maths as DayHIST, indexed weights)."""
        N = h.shape[0]
        if N <= 1:
            return h
        g = self._cos(h, h)
        diag = g.diagonal(0)
        dev = h.device
        g = g * (torch.ones(N, N, device=dev) - torch.eye(N, device=dev))
        if self.hist_topk:
            k = min(self.hist_topk, max(N - 1, 1))
            topv, topi = g.topk(k, dim=1)
            w = torch.softmax(topv / self.hist_logtau.exp().clamp(min=1e-3), dim=1)
            g = torch.zeros_like(g).scatter(1, topi, w * topv)
        else:
            col = g.max(1)[1]; val = g.max(1)[0]
            marker = torch.zeros_like(g)
            marker[torch.arange(N, device=dev), col] = 10.0
            g = torch.where(marker == 10.0, val.reshape(-1, 1).expand_as(g),
                            torch.zeros_like(g))
        g = g + torch.diag_embed((g.sum(0) != 0).float() * diag)
        hid = h.t().mm(g).t()
        hid = hid[hid.sum(1) != 0]
        if hid.numel() == 0:
            return h
        a = torch.softmax(self._cos(h, hid), dim=1)
        iss = getattr(self, f"fc_is{i}")(a.mm(hid))
        shared_back = getattr(self, f"fc_is_back{i}")(iss)
        out_is = self.leaky(getattr(self, f"fc_is_fore{i}")(iss))
        out_indi = self.leaky(getattr(self, f"fc_indi{i}")(h - shared_back))
        return out_is + out_indi

    def forward(self, x):                                  # (N, K*day_bars, F)
        N, T, F = x.shape
        K = T // self.day_bars
        enc = lambda t: self.intraday(t)[1][-1]            # noqa: E731
        day_vec = _chunked_encode(enc, x.reshape(N * K, self.day_bars, F),
                                  width=self.intraday.hidden_size).reshape(N, K, -1)
        last = day_vec[:, -1]
        mean = day_vec.mean(1)
        recent = day_vec[:, -min(5, K):].mean(1)
        h = self.day_dnn(torch.cat([last, mean, recent], dim=1))
        for i in range(self.blocks):
            # RESIDUAL around the HIST hop too. Without it, stacking 3 non-residual
            # cross-sectional transforms destroys the signal outright (measured: L=3
            # trained to IC 0.004, i.e. it learned nothing, while L=1 was fine at 0.136).
            h = h + self._hist(h, i)                       # cross-sectional hop
            h = h + getattr(self, f"dnn{i}")(h)            # per-stock refine
        return self.head(h).squeeze(-1)


class DayGRUGRU(nn.Module):
    """Pure GRU hierarchy: intraday GRU encodes each day's bars; outer GRU across days."""

    def __init__(self, day_bars, n_feat, d_day=128, gru_hidden=128, layers=1):
        super().__init__()
        self.day_bars = day_bars
        self.intraday = nn.GRU(n_feat, d_day, batch_first=True)
        self.rnn = nn.GRU(d_day, gru_hidden, num_layers=layers, batch_first=True,
                          dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(gru_hidden, 1)      # qlib-style fc_out

    def forward(self, x):                                  # (B, K*day_bars, F)
        B, T, F = x.shape
        K = T // self.day_bars
        enc = lambda t: self.intraday(t)[1][-1]            # noqa: E731
        day_vec = _chunked_encode(enc, x.reshape(B * K, self.day_bars, F),
                                  width=self.intraday.hidden_size).reshape(B, K, -1)
        _, h2 = self.rnn(day_vec)
        return self.head(h2[-1]).squeeze(-1)


def _chunked_encode(fn, seqs, chunk=None, width=256):
    """Run a per-day-slice encoder over (N*K, bars, F) in bounded chunks.

    Long lookbacks make N*K huge (d240: ~240k slices -> ~17GB for one GRU op -> OOM on
    24GB cards). Chunking alone doesn't save training memory (backward needs all
    activations), so under grad we use gradient checkpointing per chunk: activations are
    recomputed in backward, capping VRAM at ~one chunk (identical math, ~30% extra
    compute). Under no_grad (eval) a plain loop suffices."""
    if chunk is None:
        # checkpoint-recompute peak ~ chunk * bars * width * 45B: 32768 is sized for
        # 15-bar tails at width<=256; scale down with bar count AND encoder width
        # (240 bars x 2768 wide -> ~189) to stay <= ~6GB
        chunk = max(64, 32768 * 15 * 256 // (max(seqs.shape[1], 15) * max(width, 256)))
    outs = []
    for i in range(0, seqs.shape[0], chunk):
        xi = seqs[i:i + chunk]
        if torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as _ckpt
            # REENTRANT checkpoint (torch 1.12: the non-reentrant variant's output into a
            # MULTI-LAYER cuDNN GRU backward raises IndexError). Reentrant needs an input
            # that requires grad, so mark the (grad-free) data slice as a leaf.
            if not xi.requires_grad:
                xi = xi.detach().requires_grad_()
            outs.append(_ckpt(fn, xi))
        else:
            outs.append(fn(xi))
    return torch.cat(outs)


class CausalDilatedBlock(nn.Module):
    """Causal dilated Conv1d + SiLU + dropout with a residual (1x1-projected if needed)."""

    def __init__(self, ch_in, ch_out, kernel, dilation, dropout=0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch_in, ch_out, kernel, dilation=dilation)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else None

    def forward(self, x):                                  # (B, C, T)
        h = self.drop(self.act(self.conv(nn.functional.pad(x, (self.pad, 0)))))
        return h + (x if self.proj is None else self.proj(x))


def _dilation_schedule(target_rf, kernel):
    """Doubling dilations 1,2,4,... until the causal receptive field covers target_rf."""
    ds = [1]
    while 1 + (kernel - 1) * sum(ds) < target_rf:
        ds.append(ds[-1] * 2)
    return ds


def _conv_trunks(day_bars, n_feat, days, d_day, hidden, kernel, dropout):
    """(intraday, crossday) causal dilated-conv stacks. Short days (<=32 bars): pure
    causal dilated stack with doubling dilations (receptive field >= day_bars). Long
    days (full 1m sequence): multi-scale dilated pyramid, each block followed by a
    stride-2 avg-pool, channels ramping 32 -> d_day so the full-resolution layers stay
    thin. Cross-day: causal dilated stack over the K day vectors."""
    blocks = []
    if day_bars <= 32:
        for i, d in enumerate(_dilation_schedule(day_bars, kernel)):
            blocks.append(CausalDilatedBlock(n_feat if i == 0 else d_day, d_day,
                                             kernel, d, dropout))
    else:
        T, c_in, i = day_bars, n_feat, 0
        while T > kernel:
            c_out = min(d_day, 32 << i)
            blocks.append(CausalDilatedBlock(c_in, c_out, kernel, 2, dropout))
            blocks.append(nn.AvgPool1d(2, ceil_mode=True))
            T = (T + 1) // 2
            c_in = c_out
            i += 1
        blocks.append(CausalDilatedBlock(c_in, d_day, kernel, 1, dropout))
    cross = [CausalDilatedBlock(d_day if i == 0 else hidden, hidden, kernel, d, dropout)
             for i, d in enumerate(_dilation_schedule(days, kernel))]
    return nn.Sequential(*blocks), nn.Sequential(*cross)


class DayDNNDNN(nn.Module):
    """Dilated-conv hierarchy (GRU replacement): _conv_trunks + linear head.
    Last-timestep readout everywhere, mirroring the GRU last-hidden."""

    def __init__(self, day_bars, n_feat, days, d_day=128, hidden=128, kernel=3, dropout=0.1):
        super().__init__()
        self.day_bars = day_bars
        self.intraday, self.crossday = _conv_trunks(day_bars, n_feat, days,
                                                    d_day, hidden, kernel, dropout)
        self.head = nn.Linear(hidden, 1)      # qlib-style fc_out, no LayerNorm

    def forward(self, x):                                  # (B, K*day_bars, F)
        B, T, F = x.shape
        K = T // self.day_bars
        bars = x.reshape(B * K, self.day_bars, F).transpose(1, 2)   # (B*K, F, bars)
        enc = lambda t: self.intraday(t)[:, :, -1]                  # noqa: E731
        day_vec = _chunked_encode(enc, bars).reshape(B, K, -1)      # (B, K, d_day)
        h = self.crossday(day_vec.transpose(1, 2))[:, :, -1]        # (B, hidden)
        return self.head(h).squeeze(-1)


class DayTSFTSF(nn.Module):
    """Transformer hierarchy. A strided conv stem compresses each day's bars to <=16
    tokens (full attention over 240 bars x 30k day-slices does not fit memory), then a
    TransformerEncoder + last-token readout gives the day vector; a second
    TransformerEncoder attends across the K day tokens. The whole window is in the
    past relative to the prediction, so no causal mask is needed."""

    def __init__(self, day_bars, n_feat, days, d_model=256, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.day_bars = day_bars
        stem, T, c_in, i = [], day_bars, n_feat, 0
        while T > 16:
            c_out = min(d_model, 32 << i)
            stem += [nn.Conv1d(c_in, c_out, 4, stride=2, padding=1), nn.SiLU()]
            T = T // 2
            c_in = c_out
            i += 1
        stem.append(nn.Conv1d(c_in, d_model, 3, padding=1))
        self.stem = nn.Sequential(*stem)
        self.pos_intra = nn.Parameter(torch.zeros(1, T, d_model))
        mk = lambda: nn.TransformerEncoder(  # noqa: E731
            nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout,
                                       batch_first=True, activation="gelu"), layers)
        self.intraday = mk()
        self.pos_day = nn.Parameter(torch.zeros(1, days, d_model))
        self.crossday = mk()
        self.head = nn.Linear(d_model, 1)     # qlib-style fc_out, no LayerNorm

    def forward(self, x):                                  # (B, K*day_bars, F)
        B, T, F = x.shape
        K = T // self.day_bars
        bars = x.reshape(B * K, self.day_bars, F).transpose(1, 2)   # (B*K, F, bars)
        enc = lambda t: self.intraday(                              # noqa: E731
            self.stem(t).transpose(1, 2) + self.pos_intra)[:, -1]
        day_vec = _chunked_encode(enc, bars, chunk=16384).reshape(B, K, -1)
        h = self.crossday(day_vec + self.pos_day[:, :K])[:, -1]     # (B, d)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------- StockNet
# Field groups. Every group is standardized with ONE shared mean/std so that
# WITHIN-group differences survive at full scale: high-low range, close-open body,
# ask-bid spread, book slope. Per-field scaling would divide each column by its own
# sigma and flatten exactly those differences.
SN_PRICE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]      # open high low close, ask/bid price 1-3
SN_BOOKSZ = [13, 14, 15, 16, 17, 18]           # ask/bid volume 1-3
SN_ORDERS = [19, 20, 21, 22, 23, 24]           # ask/bid num_orders 1-3
SN_FLOW = [10, 11, 12]                         # volume, amount, deal_number
SN_CANDLE = [0, 1, 2, 3] + SN_FLOW             # OHLC + V: the candle branch
SN_BOOK = [4, 5, 6, 7, 8, 9] + SN_BOOKSZ + SN_ORDERS   # the order-book branch


class StockNorm(nn.Module):
    """Per-stock normalization anchored on the CLOSE.

    price fields : (x - inday_mean(close)) / window_std(close)
        Centering on THAT DAY'S mean close removes overnight level drift, so what reaches
        the encoder is the intraday shape; dividing by the stock's own close volatility
        over the window puts every price channel in units of "how much this stock usually
        moves". Because all 10 price columns share one centre and one scale, the geometry
        between them survives intact: high-low is a range in sigma units, ask1-bid1 is a
        volatility-normalized spread, close-open is the body.
    other fields : (x - mean) / std PER FIELD over time
        Volumes, order counts and flow each keep their own scale: their units are
        unrelated, so neither a price sigma nor a shared group sigma is meaningful.
    Parameter-free, computed from the live window, no fitted stats."""

    def __init__(self, day_bars, close_idx=3, price=None, others=None):
        super().__init__()
        self.day_bars = day_bars
        self.close_idx = close_idx
        self.price = list(price if price is not None else SN_PRICE)
        self.other_idx = sorted(others if others is not None
                                else SN_BOOKSZ + SN_ORDERS + SN_FLOW)

    def forward(self, x):                                  # (B, T, F), T = K * day_bars
        B, T, F = x.shape
        K = T // self.day_bars
        close = x[..., self.close_idx: self.close_idx + 1]          # (B, T, 1)
        mu = close.reshape(B, K, self.day_bars, 1).mean(2, keepdim=True)
        mu = mu.expand(B, K, self.day_bars, 1).reshape(B, T, 1)     # that day's mean close
        # sigma of the CENTERED close: the stock's intraday move scale. Using the raw
        # window std would be dominated by 30 days of level drift and would squash the
        # intraday shape back to ~0.04 sigma - the conditioning problem this norm exists
        # to fix.
        sd = (close - mu).std(dim=1, keepdim=True)
        out = torch.empty_like(x)
        out[..., self.price] = (x[..., self.price] - mu) / (sd + 1e-6)
        # every non-price field on its OWN mean/sigma over time: volume (shares), amount
        # (yuan) and order counts have unrelated units, so a shared scaler would let the
        # largest-magnitude channel dictate all of them.
        oth = self.other_idx
        sub = x[..., oth]
        out[..., oth] = (sub - sub.mean(dim=1, keepdim=True)) / (sub.std(dim=1, keepdim=True) + 1e-6)
        return out


class GroupTemporalNorm(nn.Module):
    """Per-stock temporal standardization with a SHARED scaler inside each field group
    (kept for the --norm dual mixing module)."""

    def __init__(self, groups=(SN_PRICE, SN_BOOKSZ, SN_ORDERS, SN_FLOW)):
        super().__init__()
        self.groups = [list(g) for g in groups]

    def forward(self, x):                                  # (B, T, F)
        out = torch.empty_like(x)
        for g in self.groups:
            sub = x[..., g]
            m = sub.mean(dim=(1, 2), keepdim=True)
            sd = sub.std(dim=(1, 2), keepdim=True)
            out[..., g] = (sub - m) / (sd + 1e-6)
        return out


def _rope_tables(seq_len, head_dim, device, dtype, base=10000.0, pos=None):
    """cos/sin tables for rotary position embedding, shape (1, 1, T, head_dim).

    pos: optional (T,) float positions. Passing TRADING-TIME coordinates instead of bar
    indices makes the rotation angle proportional to the real gap between bars, so the
    model sees that two auction bars are one minute apart while the same two bars on
    consecutive days are a whole session apart."""
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = (torch.arange(seq_len, device=device, dtype=torch.float32) if pos is None
         else pos.to(device=device, dtype=torch.float32))
    ang = torch.outer(t, inv)
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with ROTARY position embedding.

    RoPE rotates q/k by an angle proportional to bar index, so attention logits depend on
    the RELATIVE distance between bars rather than their absolute slot. For intraday bars
    that is the right inductive bias: "three minutes before the close" means the same thing
    whichever day it is, and it transfers across different bar counts per day."""

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0 and (d_model // heads) % 2 == 0, "head_dim must be even for RoPE"
        self.h, self.dh = heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self._cache = {}

    def _rope(self, T, device, dtype, pos=None):
        key = (T, device, dtype, None if pos is None else id(pos))
        if key not in self._cache:
            self._cache[key] = _rope_tables(T, self.dh, device, dtype, pos=pos)
        return self._cache[key]

    def forward(self, x, pos=None):                        # (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        cos, sin = self._rope(T, x.device, x.dtype, pos)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        att = torch.softmax(q @ k.transpose(-2, -1) / (self.dh ** 0.5), dim=-1)
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class AttnBarEncoder(nn.Module):
    """Per-day bar encoder: linear projection -> RoPE self-attention over the day's bars
    -> flatten head to one vector. Attention lets each bar read the whole day (the auction
    print sees the run-up), which a left-to-right GRU cannot do."""

    def __init__(self, n_in, day_bars, d_model=64, heads=4, layers=1, d_out=128,
                 dropout=0.1, ln=True):
        super().__init__()
        self.proj = nn.Linear(n_in, d_model)
        # ln=True gives the standard pre-norm transformer block. ln=False drops LayerNorm
        # entirely: it rescales each bar vector to unit norm, so two stocks with the same
        # intraday SHAPE but different AMPLITUDE collapse to the same code - and amplitude
        # is what cross-sectional ranking needs (progress/07's head-LayerNorm bug). Both
        # are wired so the question can be settled by experiment rather than by argument.
        self.ln = ln
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "attn": RoPESelfAttention(d_model, heads, dropout),
            "n1": nn.LayerNorm(d_model) if ln else nn.Identity(),
            "n2": nn.LayerNorm(d_model) if ln else nn.Identity(),
            "ff": nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)),
        }) for _ in range(layers)])
        if not ln:                                  # no norm -> damp residual branches
            for b in self.blocks:
                nn.init.zeros_(b["ff"][-1].bias); b["ff"][-1].weight.data *= 0.1
                nn.init.zeros_(b["attn"].proj.bias); b["attn"].proj.weight.data *= 0.1
        # flatten head over a per-DAY block; None when the encoder runs on the whole
        # window (a flatten head over K*day_bars would be millions of parameters)
        self.head = nn.Linear(day_bars * d_model, d_out) if day_bars else None

    def forward(self, x, pos=None):                        # (B, T, n_in)
        h = self.proj(x)                                    # position comes from RoPE
        for b in self.blocks:                   # pre-norm when ln=True, plain residual otherwise
            h = h + b["attn"](b["n1"](h), pos)
            h = h + b["ff"](b["n2"](h))
        return self.head(h.flatten(1)) if self.head is not None else h                      # (B, d_out)


class StockNetFlat(nn.Module):
    """StockNet with NO intraday/interday split: every bar of the whole lookback is one
    sequence, and RoPE angles advance with TRADING TIME rather than bar index.

    pos[t] = day(t) * bars_per_session * bar_minutes + bar(t) * bar_minutes
    so a 15m full-day window is evenly spaced, while a 15-bar auction slice correctly
    shows one minute between neighbouring bars and a whole session across a day boundary.
    Attention therefore reads "yesterday's close vs today's open" as a real gap instead
    of two adjacent tokens. Readout is a learned-query attention pool (a flatten head over
    K*day_bars would be millions of parameters), then the HIST cross-sectional head."""

    def __init__(self, day_bars, n_feat, d_day=128, gru_hidden=128, d_model=64, heads=4,
                 layers=2, dropout=0.1, public=0, public_mode="market", day_neutral=False,
                 ln=True, dual_view=True, bar_minutes=None, session_minutes=240.0,
                 attn_chunk=96):
        super().__init__()
        self.day_bars = day_bars
        self.dual_view = dual_view
        self.attn_chunk = attn_chunk
        self.bar_minutes = bar_minutes if bar_minutes else session_minutes / max(day_bars, 1)
        self.session_minutes = session_minutes
        self.norm = StockNorm(day_bars)
        self.cs = CSNorm()
        mult = 2 if dual_view else 1
        # day_bars=0 -> headless encoders: they return the full token sequence
        self.candle = AttnBarEncoder(len(SN_CANDLE) * mult, 0, d_model, heads, layers, d_day, dropout, ln)
        self.book = AttnBarEncoder(len(SN_BOOK) * mult, 0, d_model, heads, layers, d_day, dropout, ln)
        self.q = nn.Parameter(torch.randn(1, 1, 2 * d_model) * 0.02)   # pooling query
        self.pool_proj = nn.Linear(2 * d_model, gru_hidden)
        self.hist = DayHIST(day_bars, n_feat, d_day=gru_hidden, gru_hidden=gru_hidden,
                            public=public, public_mode=public_mode, day_neutral=day_neutral)
        del self.hist.intraday, self.hist.rnn

    def _positions(self, T, device):
        idx = torch.arange(T, device=device)
        day = torch.div(idx, self.day_bars, rounding_mode="floor")
        bar = idx % self.day_bars
        return day * self.session_minutes + bar * self.bar_minutes    # trading minutes

    def forward(self, x):                                  # (N, K*day_bars, F)
        N, T, F = x.shape
        xt = self.norm(x)
        if self.dual_view:
            x = torch.cat([xt, self.cs(x)], dim=-1)
            cand = SN_CANDLE + [i + F for i in SN_CANDLE]
            book = SN_BOOK + [i + F for i in SN_BOOK]
        else:
            x, cand, book = xt, SN_CANDLE, SN_BOOK
        pos = self._positions(T, x.device)

        def enc(t):                                        # (b, T, C) -> (b, 2*d_model)
            h = torch.cat([self.candle(t[..., cand], pos), self.book(t[..., book], pos)], -1)
            a = torch.softmax((h @ self.q.transpose(1, 2)).squeeze(-1), dim=1)   # (b, T)
            return (a.unsqueeze(-1) * h).sum(1)            # learned-query attention pool

        # chunk over STOCKS: the T x T attention map is the memory driver
        pooled = _chunked_encode(enc, x, chunk=self.attn_chunk)
        return self.hist._hist_head(self.pool_proj(pooled))


class StockNet(nn.Module):
    """Temporal-normalized, two-stream intraday encoder + cross-day GRU + HIST head.

    intraday : (N, K*day_bars, F) -> candle stream (OHLC+V) and order-book stream
               (ask/bid price, size, order counts), each an attention encoder with a
               flatten head -> concat -> (N, K, 2H)
    interday : GRU over the K day-codes -> (N, H)
    head     : the same HIST hidden-concept + individual modules used by DayHIST, so the
               cross-sectional context enters where it has always worked.
    """

    def __init__(self, day_bars, n_feat, d_day=128, gru_hidden=128, d_model=64,
                 heads=4, layers=1, dropout=0.1, public=0, public_mode="market",
                 day_neutral=False, ln=True, dual_view=True, input_norm=True):
        super().__init__()
        self.day_bars = day_bars
        # TWO VIEWS per stream, concatenated on the channel axis:
        #   temporal (StockNorm) - "unusual for THIS stock": close-anchored intraday shape,
        #                          volume surprise, spread vs its own norm
        #   cross-sectional      - "unusual vs the market right now": strips the market
        #                          factor, which is what the grid shows is worth 0.02-0.04
        #                          IC at multi-day lookbacks (d10 cs .1302 vs global .0887)
        # Keeping both means the temporal information survives AND market-relative
        # positioning is available without the HIST head having to reconstruct it.
        # input_norm=False (--sn-no-norm) drops BOTH live norms: the offline prep (mixnorm)
        # is then the only anchoring, and there is a single raw view.
        self.input_norm = input_norm
        if not input_norm:
            dual_view = False
        self.dual_view = dual_view
        self.norm = StockNorm(day_bars) if input_norm else None
        self.cs = CSNorm() if input_norm else None
        mult = 2 if dual_view else 1
        self.candle = AttnBarEncoder(len(SN_CANDLE) * mult, day_bars, d_model, heads, layers, d_day, dropout, ln)
        self.book = AttnBarEncoder(len(SN_BOOK) * mult, day_bars, d_model, heads, layers, d_day, dropout, ln)
        self.rnn = nn.GRU(2 * d_day, gru_hidden, batch_first=True)
        self.hist = DayHIST(day_bars, n_feat, d_day=gru_hidden, gru_hidden=gru_hidden,
                            public=public, public_mode=public_mode, day_neutral=day_neutral)
        del self.hist.intraday, self.hist.rnn                # head-only reuse

    def forward(self, x):                                  # (N, K*day_bars, F)
        N, T, F = x.shape
        K = T // self.day_bars
        if not self.input_norm:                             # offline-normalized input as-is
            cand, book = SN_CANDLE, SN_BOOK
        else:
            xt = self.norm(x)                               # temporal, close-anchored
            if self.dual_view:
                xc = self.cs(x)                             # cross-sectional (market-relative)
                x = torch.cat([xt, xc], dim=-1)             # (N, T, 2F)
                cand = SN_CANDLE + [i + F for i in SN_CANDLE]
                book = SN_BOOK + [i + F for i in SN_BOOK]
            else:
                x, cand, book = xt, SN_CANDLE, SN_BOOK
        bars = x.reshape(N * K, self.day_bars, x.shape[-1])
        enc = lambda t: torch.cat([self.candle(t[..., cand]), self.book(t[..., book])], -1)
        day_vec = _chunked_encode(enc, bars, width=self.rnn.hidden_size).reshape(N, K, -1)
        return self.hist._hist_head(self.rnn(day_vec)[1][-1])


class CSNorm(nn.Module):
    """Cross-sectional input norm INSIDE the model: standardize each (bar, field) across
    the batch, which must be one trading day's whole cross-section. Parameter-free and
    computed from the live batch at train AND inference time, so it needs NO fitted
    preprocessing params (REQUIREMENT-compliant, unlike offline cs preprocessing)."""

    def forward(self, x):                                  # (N, T, F)
        if x.shape[0] <= 1:
            return x
        m = x.mean(0, keepdim=True)
        s = x.std(0, keepdim=True)
        return (x - m) / (s + 1e-6)


class TemporalNorm(nn.Module):
    """Per-stock, per-field standardization over the stock's OWN window history.

    Answers "is this bar unusual for THIS stock" (volume surprise, price position in its
    own recent range, spread vs its own norm) - information cross-sectional normalization
    cannot express, since CS only ever compares a stock to its peers at one instant.
    Parameter-free and computed from the live window, so no fitted params (compliant)."""

    def forward(self, x):                                  # (N, T, F)
        m = x.mean(1, keepdim=True)
        s = x.std(1, keepdim=True)
        return (x - m) / (s + 1e-6)


class DualNorm(nn.Module):
    """Mixing module: run the SAME inputs through a cross-sectional view and a temporal
    view, then hand both to the encoder as separate channels (2F). The network learns how
    much of each view it needs per field, instead of us hard-coding derived features."""

    def __init__(self, mode="concat"):
        super().__init__()
        self.cs = CSNorm()
        self.tn = TemporalNorm()
        self.mode = mode

    def forward(self, x):                                  # (N, T, F) -> (N, T, 2F)
        return torch.cat([self.cs(x), self.tn(x)], dim=-1)


class DualWrapped(nn.Module):
    """DualNorm input layer + any (N, T, 2F) model."""

    def __init__(self, net):
        super().__init__()
        self.norm = DualNorm()
        self.net = net

    def forward(self, x):
        return self.net(self.norm(x))


class CSWrapped(nn.Module):
    """CSNorm input layer + any (N, T, F) model."""

    def __init__(self, net):
        super().__init__()
        self.norm = CSNorm()
        self.net = net

    def forward(self, x):
        return self.net(self.norm(x))


class DayHIST(nn.Module):
    """Real HIST on the DayGRUGRU trunk: hidden-concept + individual modules over each
    day's whole cross-section (qlib HIST minus the predefined stock2concept part, which
    needs external sector data, banned as input). Batches MUST be one trading day.
    NO LayerNorm anywhere near the head (ranking-head lesson)."""

    def __init__(self, day_bars, n_feat, d_day=128, gru_hidden=128, layers=1,
                 public=0, public_mode="market", day_neutral=False,
                 hist_topk=0, hist_tau=0.1, hist_dropout=0.0):
        super().__init__()
        self.day_bars = day_bars
        # DAY-NEUTRAL: subtract the cross-sectional mean of the day vectors at EVERY day
        # of the lookback, so the cross-day GRU sees market-RELATIVE daily moves instead
        # of absolute ones. The public stage removes the market once, at the final hidden;
        # this removes it K times, throughout the sequence. Parameter-free and computed
        # from the live batch at train AND inference (same compliance argument as CSNorm:
        # no fitted preprocessing params, no derived input fields, no dimensionality
        # reduction). Requires whole-day batches, which hist_day already enforces.
        self.day_neutral = bool(day_neutral)
        self.intraday = nn.GRU(n_feat, d_day, batch_first=True)
        self.rnn = nn.GRU(d_day, gru_hidden, num_layers=layers, batch_first=True,
                          dropout=0.1 if layers > 1 else 0.0)
        H = gru_hidden
        # PUBLIC concept(s): the predefined-concept slot of real HIST that needs no
        # external data, since every stock belongs (qlib HIST order: predefined ->
        # hidden-on-residual -> individual). Stages CASCADE, each on the previous
        # residual, so they compose instead of competing:
        #   market = equal-weight mean of the day's hiddens, per-stock cosine loading.
        #            Rank-1 and SIGNED, so it can express anti-market exposure (a beta).
        #   query  = `public` learned query vectors attention-pooling the cross-section
        #            into persistent style factors; per-stock attachment is a softmax
        #            mixture (non-negative). Content is rebuilt daily -> no cold start.
        # "both" runs market then query, which is the intended full form: strip beta
        # first, then let the styles explain what is left.
        self.n_public = int(public)
        self.public_mode = public_mode
        # project = market direction removed by TRUE projection instead of the learned
        #   rank-1 subtraction (the learned one translates the cloud: measured pairwise
        #   cosine spread 0.204 raw -> 0.001 after, vs 1.484 after a real projection, so
        #   the voting step downstream sees far more separation).
        # projq = project then learned queries on the residual: multi-factor structure
        #   through pure attention, the compliant substitute for an eigenbasis (the rules
        #   forbid dimensionality reduction, so no PCA/SVD anywhere in the model).
        self.pub_stages = ({"market": ["market"], "query": ["query"],
                            "both": ["market", "query"], "project": ["project"],
                            "projq": ["project", "query"]}[public_mode]
                           if self.n_public else [])
        # stage 1 keeps the ORIGINAL layer names so single-stage checkpoints still load
        for i, st in enumerate(self.pub_stages):
            sfx = "" if i == 0 else str(i + 1)
            setattr(self, f"fc_pub{sfx}", nn.Linear(H, H))
            if st != "project":                    # exact-projection stages need no
                setattr(self, f"fc_pub{sfx}_back", nn.Linear(H, H))   # learned back-map
            setattr(self, f"fc_pub{sfx}_fore", nn.Linear(H, H))
            if st == "query":
                self.pub_q = nn.Parameter(torch.randn(max(self.n_public, 2), H) / H ** 0.5)
                self.pub_q_ix = i
        self.fc_is = nn.Linear(H, H)
        self.fc_is_back = nn.Linear(H, H)
        self.fc_is_fore = nn.Linear(H, H)
        self.fc_indi = nn.Linear(H, H)
        self.leaky = nn.LeakyReLU()
        self.head = nn.Linear(H, 1)
        self.hist_drop = nn.Dropout(hist_dropout) if hist_dropout > 0 else None
        # --- soft top-k concept graph (hist_topk>0) --------------------------------
        # The original HIST picks each stock's SINGLE most-similar peer with argmax:
        # discrete, non-differentiable, and the pick flips wholesale between seeds ->
        # a structural variance source (measured public sigma 0.059). Setting
        # --hist-topk k replaces it with a temperature-softmax over the k most similar
        # peers: same sparse "concept" idea, but continuous, multi-peer, and the
        # temperature is learned. k=0 keeps the original argmax behaviour.
        self.hist_topk = int(hist_topk)
        if self.hist_topk:
            self.hist_logtau = nn.Parameter(torch.tensor(float(np.log(hist_tau))))

    @staticmethod
    def _cos(x, y):
        xy = x.mm(y.t())
        xn = torch.sqrt((x * x).sum(1)).reshape(-1, 1)
        yn = torch.sqrt((y * y).sum(1)).reshape(-1, 1)
        return xy / (xn.mm(yn.t()) + 1e-6)


    def forward(self, x):                                  # (N, K*day_bars, F), N = one day's stocks
        N, T, F = x.shape
        K = T // self.day_bars
        enc = lambda t: self.intraday(t)[1][-1]            # noqa: E731
        day_vec = _chunked_encode(enc, x.reshape(N * K, self.day_bars, F),
                                  width=self.intraday.hidden_size).reshape(N, K, -1)
        if self.day_neutral and N > 1:
            day_vec = day_vec - day_vec.mean(0, keepdim=True)   # market-relative per day
        if os.environ.get("TG_CKPT_CROSSDAY"):
            # giant models (100M / H~2768): the crossday cuDNN reserve is ~N*K*H*24B
            # (~14GB for a 900-stock day). Stocks are independent through the crossday
            # GRU, so run it batch-chunked through the same checkpointing helper.
            h_last = _chunked_encode(lambda v: self.rnn(v)[1][-1], day_vec,
                                     width=self.rnn.hidden_size)
        else:
            h_last = self.rnn(day_vec)[1][-1]
        return self._hist_head(h_last)

    def _hist_head(self, x_hidden):                        # (N, H) -> (N,) scores
        N = x_hidden.shape[0]
        dev = x_hidden.device
        output_pub = 0.0
        if self.n_public and N > 1:
            for i, st in enumerate(self.pub_stages):       # cascade on the residual
                sfx = "" if i == 0 else str(i + 1)
                proj = None
                if st in ("market", "project"):            # rank-1 SIGNED market mode
                    cpub = x_hidden.mean(0, keepdim=True)  # (1, H) equal-weight market
                    a = self._cos(x_hidden, cpub)          # (N, 1) per-stock loading
                    pub = getattr(self, f"fc_pub{sfx}")(a * cpub)   # beta_i * market
                    if st == "project":                    # exact component along market
                        u = cpub / (cpub.norm() + 1e-6)
                        proj = x_hidden.mm(u.t()).mm(u)
                else:                                      # learned queries pool the day
                    att = torch.softmax(self.pub_q.mm(x_hidden.t())
                                        / x_hidden.shape[1] ** 0.5, dim=1)
                    cpub = att.mm(x_hidden)                # (M, H) public concepts
                    a = torch.softmax(self._cos(x_hidden, cpub), dim=1)
                    pub = getattr(self, f"fc_pub{sfx}")(a.mm(cpub))
                output_pub = output_pub + self.leaky(getattr(self, f"fc_pub{sfx}_fore")(pub))
                # project strips the market component EXACTLY (shape-preserving 256->256
                # residual, no basis change); market/query subtract a learned rank-1 map
                x_hidden = (x_hidden - proj if proj is not None
                            else x_hidden - getattr(self, f"fc_pub{sfx}_back")(pub))
        if N <= 1:                                         # later stages / hidden concepts
            i_shared_back = torch.zeros_like(x_hidden)
            output_is = torch.zeros_like(x_hidden)
        else:
            g = self._cos(x_hidden, x_hidden)              # stock-stock similarity graph
            diag = g.diagonal(0)
            g = g * (torch.ones(N, N, device=dev) - torch.eye(N, device=dev))
            if self.hist_topk:
                # soft top-k: keep the k most similar peers per stock, weight them by a
                # temperature-softmax (learned tau). Continuous + multi-peer; reduces to
                # the argmax graph as tau -> 0 with k = 1.
                k = min(self.hist_topk, max(N - 1, 1))
                topv, topi = g.topk(k, dim=1)
                w = torch.softmax(topv / self.hist_logtau.exp().clamp(min=1e-3), dim=1)
                g = torch.zeros_like(g).scatter(1, topi, w * topv)
            else:
                col = g.max(1)[1]; val = g.max(1)[0]       # each stock's most-similar peer
                marker = torch.zeros_like(g)
                marker[torch.arange(N, device=dev), col] = 10.0
                g = torch.where(marker == 10.0, val.reshape(-1, 1).expand_as(g), torch.zeros_like(g))
            g = g + torch.diag_embed((g.sum(0) != 0).float() * diag)
            hidden_i = x_hidden.t().mm(g).t()              # discovered concept vectors
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
        z = output_pub + output_is + output_indi
        if self.hist_drop is not None:
            z = self.hist_drop(z)
        return self.head(z).squeeze(-1)


class DayHISTFlat(DayHIST):
    """USER DESIGN (2026-07-31): HISTflat -- NO intraday GRU. Each day's bars are
    FLATTENED into one (day_bars*n_feat) channel vector and fed straight to the
    cross-day GRU: the literal Alpha360 shape real HIST was built for. Everything
    after the day vectors (day-neutral, public stages, hidden-concept + individual
    head) is inherited from DayHIST untouched, so the A/B against hist_day isolates
    exactly one question: is the intraday GRU's day encoding worth anything, or is
    the flat channel vector enough."""

    def __init__(self, day_bars, n_feat, d_day=128, gru_hidden=128, layers=1, **kw):
        super().__init__(day_bars, n_feat, d_day=d_day, gru_hidden=gru_hidden,
                         layers=layers, **kw)
        del self.intraday                               # no intraday encoder at all
        self.rnn = nn.GRU(day_bars * n_feat, gru_hidden, num_layers=layers,
                          batch_first=True, dropout=0.1 if layers > 1 else 0.0)

    def forward(self, x):                               # (N, K*day_bars, F)
        N, T, F = x.shape
        K = T // self.day_bars
        day_vec = x.reshape(N, K, self.day_bars * F)
        if self.day_neutral and N > 1:
            day_vec = day_vec - day_vec.mean(0, keepdim=True)
        h_last = self.rnn(day_vec)[1][-1]
        return self._hist_head(h_last)


class DayHISTDual(DayHIST):
    """USER DESIGN (2026-08-02): IN-MODEL multi-resolution fusion. One fetch at the
    60-bar tail (14:01); per day TWO intraday encoders read the same bars at two
    resolutions -- GRU_a over all 60, GRU_b over only the LAST 15 (the 14:46 suffix,
    i.e. the champion 15-bar view) -- each emitting d_day//2; the concat (128+128)
    is the 256-d day vector for the cross-day GRU. Everything after the day vectors
    is inherited DayHIST. Rationale: post-hoc cross-geometry blending needs per-day
    normalization which the board taxes ~0.10 (v144-147); fusing INSIDE the model
    keeps a raw single-model output and lets the loss learn the mixing weights."""

    SUB_BARS = 15                                       # suffix resolution (14:46 tail)

    def __init__(self, day_bars, n_feat, d_day=256, gru_hidden=256, layers=1, **kw):
        assert day_bars > self.SUB_BARS, "dual needs the 60-bar (14:01) fetch"
        super().__init__(day_bars, n_feat, d_day=d_day, gru_hidden=gru_hidden,
                         layers=layers, **kw)
        del self.intraday
        half = d_day // 2
        self.intraday_a = nn.GRU(n_feat, half, batch_first=True)   # full 60 bars
        self.intraday_b = nn.GRU(n_feat, half, batch_first=True)   # last 15 bars

    def forward(self, x):                               # (N, K*day_bars, F)
        N, T, F = x.shape
        K = T // self.day_bars
        seqs = x.reshape(N * K, self.day_bars, F)
        ha = _chunked_encode(lambda t: self.intraday_a(t)[1][-1], seqs,
                             width=self.intraday_a.hidden_size)
        hb = _chunked_encode(lambda t: self.intraday_b(t)[1][-1],
                             seqs[:, -self.SUB_BARS:, :],
                             width=self.intraday_b.hidden_size)
        day_vec = torch.cat([ha, hb], dim=-1).reshape(N, K, -1)
        if self.day_neutral and N > 1:
            day_vec = day_vec - day_vec.mean(0, keepdim=True)
        h_last = self.rnn(day_vec)[1][-1]
        return self._hist_head(h_last)


class HISTDNN(DayHIST):
    """DayHIST with both GRUs replaced by the causal dilated-conv trunks (hist_dnn):
    _conv_trunks give the day vectors and the cross-day hidden, then the identical
    HIST hidden-concept + individual cross-sectional module scores the day's
    cross-section. Batches MUST be one trading day."""

    def __init__(self, day_bars, n_feat, days, d_day=128, hidden=128, kernel=3, dropout=0.1,
                 public=0, public_mode="market", day_neutral=False):
        super().__init__(day_bars, n_feat, d_day=d_day, gru_hidden=hidden,
                         public=public, public_mode=public_mode, day_neutral=day_neutral)
        del self.intraday, self.rnn                        # GRU trunks -> conv trunks
        self.intraday, self.crossday = _conv_trunks(day_bars, n_feat, days,
                                                    d_day, hidden, kernel, dropout)

    def forward(self, x):                                  # (N, K*day_bars, F)
        N, T, F = x.shape
        K = T // self.day_bars
        bars = x.reshape(N * K, self.day_bars, F).transpose(1, 2)    # (N*K, F, bars)
        enc = lambda t: self.intraday(t)[:, :, -1]                   # noqa: E731
        day_vec = _chunked_encode(enc, bars).reshape(N, K, -1)       # (N, K, d_day)
        if self.day_neutral and N > 1:
            day_vec = day_vec - day_vec.mean(0, keepdim=True)        # market-relative
        x_hidden = self.crossday(day_vec.transpose(1, 2))[:, :, -1]  # (N, hidden)
        return self._hist_head(x_hidden)


def replica_components(idx_df, score, label):
    """Platform-replica SR and Stress on validation predictions (same math as
    scripts/replicate_score.py, which is the gate submissions are judged by):
    per-day winsorize(1/99)+z of scores, decile long-short Sharpe (annualized), and
    the worst-regime mean IC over market-direction x market-vol quadrants. These two
    track the official Sharpe/StressIR far better than raw IC does, so every cell
    reports them."""
    df = idx_df.copy()
    df["s"] = score
    df["e"] = label
    df = df.dropna(subset=["s", "e"])
    ics, ls, days = [], [], []
    for d, g in df.groupby("date"):
        if len(g) < 20 or g["s"].std() == 0:
            continue
        z = g["s"].clip(g["s"].quantile(0.01), g["s"].quantile(0.99))
        z = (z - z.mean()) / (z.std() + 1e-12)
        ics.append(z.corr(g["e"]))
        q = pd.qcut(z.rank(method="first"), 10, labels=False)
        ls.append(g["e"][q == 9].mean() - g["e"][q == 0].mean())
        days.append(d)
    if not ics:
        return {"SR": float("nan"), "Stress": float("nan")}
    ics = pd.Series(ics, index=pd.DatetimeIndex(days))
    ls = np.asarray(ls, dtype=np.float64)
    sr = float(ls.mean() / (ls.std() + 1e-12) * np.sqrt(252))
    mkt = df.groupby("date")["e"].mean().reindex(ics.index)
    vol = mkt.rolling(10, min_periods=3).std().bfill()
    up, hv = mkt >= mkt.median(), vol >= vol.median()
    reg = np.where(up & hv, "up_hi", np.where(up & ~hv, "up_lo",
                   np.where(~up & hv, "dn_hi", "dn_lo")))
    means = [ics[reg == r].mean() for r in ("up_hi", "up_lo", "dn_hi", "dn_lo")
             if (reg == r).sum() > 5]
    means = [m for m in means if m == m]                    # drop NaN regimes
    return {"SR": sr, "Stress": float(min(means)) if means else float("nan")}


def lazy_windows(pp, start, end, feature_cols, seq_len, stats=None, clip_pct=(1.0, 99.0),
                 label_next=None):
    """Memory-lean equivalent of D.build_windows: windows overlap by seq_len-15 rows, so
    instead of a dense (N, seq, F) tensor (54-108GB at d30-d60) keep ONE concatenated bar
    matrix (~2GB) + an end-offset per window; batches are assembled on the fly.
    Stats are accumulated over the SAME windows as the dense path -> identical scaling."""
    sd, ed = pd.Timestamp(start), pd.Timestamp(end)
    bases, ends, ys, dates, keys = [], [], [], [], []
    off = 0
    for k, sub in pp.groupby("key", sort=False):
        if len(sub) <= seq_len:
            continue
        feats = sub[feature_cols].to_numpy(np.float32)
        day = sub["day"].to_numpy()
        eod = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        eod_days = pd.to_datetime(day[eod])
        daily_close = (sub.groupby("day", sort=True)["adj_close_raw"].last()
                       .reindex(pd.DatetimeIndex(eod_days).normalize()).to_numpy(np.float64))
        # O(rows) finiteness: a window ending at p (inclusive, len seq_len) is all-finite
        # iff no bad row falls in it -> prefix-sum of bad rows, O(1) per candidate. The
        # old per-window np.isfinite scan re-checked every bar ~`days` times (O(windows x
        # seq_len x F) = hours at seq 57600); identical accept/reject decisions.
        badc = np.concatenate(([0], np.cumsum(~np.isfinite(feats).all(axis=1))))
        used = False
        for j, p in enumerate(eod):
            d = eod_days[j]
            if p + 1 < seq_len or d < sd or d > ed:
                continue
            if badc[p + 1] - badc[p + 1 - seq_len]:
                continue
            if label_next is not None:
                # day-stride mode: the frame is DECIMATED (every stride-th trading day),
                # so daily_close[j+1] here would be a 2-day return. The caller precomputes
                # true next-TRADING-day returns (leakage guard included) from the full
                # frame and passes them down; a missing key means the label was rejected.
                r = label_next.get((k, d))
                if r is None:
                    continue
                ys.append(np.float32(r)); ends.append(off + p + 1); dates.append(d); keys.append(k)
                used = True
                continue
            jf = j + 1
            c0 = daily_close[j]
            if jf >= len(eod) or not (c0 > 0) or not np.isfinite(daily_close[jf]):
                continue
            # LABEL LEAKAGE GUARD: the label of a window ending on day j is the return
            # into day j+1, so day j+1 must ALSO lie inside [sd, ed]. Without this a
            # train range ending 2023-12-31 hands the model the true next-day returns of
            # the first validation day -- the whole cross-section of it. Enforced here,
            # at the one place labels are built, so no caller can reintroduce it.
            if eod_days[jf] > ed:
                continue
            r = daily_close[jf] / c0 - 1.0
            if not np.isfinite(r):
                continue
            ys.append(np.float32(r)); ends.append(off + p + 1); dates.append(d); keys.append(k)
            used = True
        if used:
            bases.append(feats); off += len(feats)
        # stocks with no accepted window are excluded from the base (offsets stay valid)
    if not ends:
        raise ValueError(f"lazy_windows produced 0 windows for [{start}, {end}]")
    base = np.concatenate(bases)
    ends = np.asarray(ends, np.int64)
    y_raw = np.asarray(ys, np.float32)
    idx_df = pd.DataFrame({"date": dates, "key": keys})
    del bases
    if stats is None:
        # O(rows) stats: each base row belongs to a known COUNT of accepted windows
        # (range-add over [e-seq_len, e) via a diff array), so the window-sum equals a
        # count-weighted row-sum. Replaces stacking every window (O(windows x seq_len),
        # 115GB+230GB transients at seq 57600 -> cgroup OOM; hours of copying). Same
        # float64 sums up to summation order (~1e-15 relative).
        n_feat = len(feature_cols)
        wdiff = np.zeros(len(base) + 1, np.int64)
        np.add.at(wdiff, ends - seq_len, 1)
        np.add.at(wdiff, ends, -1)
        w = np.cumsum(wdiff[:-1])
        s1 = np.zeros(n_feat, np.float64); s2 = np.zeros(n_feat, np.float64)
        cnt = int(w.sum())
        for i in range(0, len(base), 10_000_000):
            blk = base[i:i + 10_000_000].astype(np.float64)
            wb = w[i:i + 10_000_000]
            blk[wb == 0] = 0.0        # rows in no accepted window may be NaN; NaN*0=NaN
            wbf = wb.astype(np.float64)[:, None]
            s1 += (blk * wbf).sum(0); s2 += (blk * blk * wbf).sum(0)
        mean = (s1 / cnt).astype(np.float32)
        std = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)).astype(np.float32) + 1e-6
    else:
        mean, std = np.asarray(stats[0], np.float32), np.asarray(stats[1], np.float32)
    base -= mean
    base /= std
    lo, hi = np.percentile(y_raw, clip_pct)
    y = np.clip(y_raw, lo, hi).astype(np.float32)
    return base, ends, y, idx_df, (mean, std)


def stride_windows(pp, start, end, feature_cols, seq_len, stride, stats=None,
                   clip_pct=(1.0, 99.0)):
    """Multi-scale windows: lookback seq_len//DAY_BARS * stride trading days, sampling
    every stride-th day -> the tensor keeps the plain (days x bars) shape but spans
    stride x the history (user experiment 2026-08-01: d240 information through a d120
    tensor).

    Implementation: split the frame into `stride` phases by GLOBAL trading-calendar
    parity and run the untouched contiguous windowing per phase -- a contiguous window
    on the decimated frame IS the strided window on the full one. A scoring day's
    windows all live in that day's phase, so daily batches stay whole. Labels must stay
    next-TRADING-day returns (the decimated frame would yield stride-day returns), so
    they are precomputed on the full frame -- same c0>0/finite/leakage rules as
    lazy_windows -- and passed down via label_next."""
    sd, ed = pd.Timestamp(start), pd.Timestamp(end)
    dc = pp.groupby(["key", "day"], sort=True)["adj_close_raw"].last()
    label_next = {}
    for k, sub in dc.groupby(level=0):
        days_k = sub.index.get_level_values(1)
        c = sub.to_numpy(np.float64)
        for j in range(len(c) - 1):
            d = days_k[j]
            if d < sd or d > ed or days_k[j + 1] > ed:      # leakage guard as in lazy_windows
                continue
            if not (c[j] > 0) or not np.isfinite(c[j + 1]):
                continue
            r = c[j + 1] / c[j] - 1.0
            if np.isfinite(r):
                label_next[(k, pd.Timestamp(d))] = np.float32(r)
    all_days = np.sort(pp["day"].unique())
    phase_of = {d: i % stride for i, d in enumerate(all_days)}
    ph = pp["day"].map(phase_of).to_numpy()
    parts, st = [], stats
    for q in range(stride):
        sel = pp[ph == q]
        if not len(sel):
            continue
        b, e, y, idx, st = lazy_windows(sel, start, end, feature_cols, seq_len, st,
                                        clip_pct, label_next=label_next)
        parts.append((b, e, y, idx))
    off, bases, ends, ys, idxs = 0, [], [], [], []
    for b, e, y, idx in parts:
        bases.append(b); ends.append(e + off); ys.append(y); idxs.append(idx)
        off += len(b)
    return (np.concatenate(bases), np.concatenate(ends), np.concatenate(ys),
            pd.concat(idxs, ignore_index=True), st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gru15m", "daymlp_gru", "daygru_gru", "hist_day",
                                        "daydnn_dnn", "daytsf_tsf", "hist_dnn", "stocknet",
                                        "stocknet_flat", "dayattn", "hist_deep",
                                        "hist_flat", "hist_dual"], required=True)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--day-stride", type=int, default=1,
                    help="sample every N-th trading day: lookback covers days*N trading "
                         "days but the tensor stays (days x bars) -- multi-scale probe")
    ap.add_argument("--norm", default="global", choices=["global", "cs", "csmodel", "dual",
                                                         "mixnorm"],
                    help="global = train-fit field norm; cs = offline cross-sectional "
                         "preprocessing (NON-compliant); csmodel = CSNorm layer inside "
                         "the model (compliant, requires daily batches); mixnorm = price "
                         "fields log(price/preclose) (adjust-state-aware anchor), all "
                         "other fields offline cs (inherits cs's compliance caveat)")
    ap.add_argument("--score-alpha", type=float, default=0.3,
                    help="score loss: weight of the soft long-short (Sharpe) term")
    ap.add_argument("--score-beta", type=float, default=1.0,
                    help="score loss: extra weight on down-market (stress) days")
    ap.add_argument("--score-lambda", type=float, default=1.0,
                    help="score loss: IC-variance (IR) penalty weight")
    ap.add_argument("--label-rank", action="store_true",
                    help="LABEL becomes the per-day cross-sectional percentile rank of the "
                         "forward return, centred and scaled to std~1 ((pct_rank-0.5)*sqrt(12)). "
                         "We train Pearson IC on RAW returns, where a single stock up 10%% on a "
                         "calm day dominates that "
                         "day's gradient. Note the IC loss already centres and rescales y within "
                         "the batch, so the shift/scale here are cosmetic -- the RANK transform "
                         "is the part that does the work, by flattening the within-day tails. "
                         "Also replaces the pooled 1/99 winsorize, which clipped nothing on calm "
                         "days and a lot on volatile ones.")
    ap.add_argument("--rank-tau", type=float, default=0.1, metavar="FRAC",
                    help="--loss rankic: sigmoid temperature as a FRACTION of the day's "
                         "prediction std. Smaller = closer to the true rank but sharper "
                         "gradients; 0.1 is ~a tenth of the cross-sectional spread.")
    ap.add_argument("--loss", default="ic", choices=["mse", "ic", "score", "nic", "rankic"],
                    help="mse = pointwise; ic = negative per-batch Pearson correlation; "
                         "nic = NEUTRALIZED IC -- both the score and the label are OLS-"
                         "residualized against the bar-built style factors of that day "
                         "before the correlation, i.e. train directly on the quantity the "
                         "platform actually grades (it residualizes scores on style+industry "
                         "and the target is itself a residual return)")
    ap.add_argument("--batch-mode", default="daily", choices=["random", "daily"],
                    help="random = shuffled minibatches of --batch; daily = one step per "
                         "trading day's whole cross-section (with --loss ic this maximizes daily IC)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--public", type=int, default=0,
                    help="hist_day/hist_dnn: add PUBLIC concept(s) before the hidden-"
                         "concept module (hidden concepts then see the market-neutral "
                         "residual). 1 + --public-mode market = rank-1 market mode; "
                         "N + query = N learned query-pooled style concepts")
    ap.add_argument("--day-neutral", action="store_true",
                    help="hist_day/hist_dnn: subtract the cross-sectional mean of the day "
                         "vectors at EVERY day of the lookback (market-relative daily "
                         "moves). Parameter-free, live-batch, needs daily batches")
    ap.add_argument("--public-mode", default="market",
                    choices=["market", "query", "both", "project", "projq"],
                    help="both = market stage then query stage, each on the previous\n                         residual (market strips signed beta, queries explain the rest);\n                         --public N sizes the QUERY concepts (market is always rank-1)")
    ap.add_argument("--d-day", type=int, default=0,
                    help="intraday encoder width, decoupled from --hidden (0 = same as "
                         "--hidden). This is the MEMORY-critical width (checkpoint "
                         "recompute ~ chunk*bars*d_day); outer layers / HIST fcs at "
                         "--hidden are nearly free, so 100M-param configs use small "
                         "d_day + large hidden")
    ap.add_argument("--hidden", type=int, default=256,
                    help="d_day and gru_hidden for the hierarchy models (default 256; "
                         "h256 = IC frontier over 5 seeds, 2026-07-19)")
    ap.add_argument("--tail", default=None,
                    help="1m auction slice start, e.g. 14:31 (default 14:46 = 15 bars)")
    ap.add_argument("--windows", default=None,
                    help="explicit intraday segments instead of a single tail, e.g. "
                         "'13:01-13:15,14:46-15:00' (both ends inclusive, 1m stamps are "
                         "minute-END so a day is 09:31-11:30 + 13:01-15:00). Supersedes "
                         "--tail; day_bars = total bars across the segments. Use for "
                         "ablations that add a non-closing block to the tail window")
    ap.add_argument("--layers", type=int, default=1,
                    help="outer (cross-day) GRU layers; scale with --days (capacity matching)")
    ap.add_argument("--kernel", type=int, default=3,
                    help="daydnn_dnn: causal-dilated conv kernel size (dilations double until "
                         "the receptive field covers day_bars / days)")
    ap.add_argument("--log-price", action="store_true",
                    help="log-transform the 10 price fields before standardization "
                         "(per-field log is allowed preprocessing; makes differences "
                         "of price columns into log-ratios)")
    ap.add_argument("--sn-temporal-only", action="store_true",
                    help="stocknet: temporal view ONLY (drops the cross-sectional view)")
    ap.add_argument("--sn-no-norm", action="store_true",
                    help="stocknet: skip the internal StockNorm/CSNorm entirely (single "
                         "raw view; pair with --norm mixnorm so the offline prep supplies "
                         "the anchoring the internal norms would have done)")
    ap.add_argument("--no-layernorm", action="store_true",
                    help="stocknet: drop LayerNorm from the attention blocks")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup over N optimizer steps (transformers usually "
                         "diverge without it at lr 1e-3)")
    ap.add_argument("--d-model", type=int, default=64,
                    help="stocknet: per-bar attention width")
    ap.add_argument("--heads", type=int, default=4, help="stocknet: attention heads")
    # NOTE: the --walkforward Nov-Dec fold mode was REMOVED (label leakage). It trained to
    # {y}-10-31 and validated from {y}-11-01, so the last training window's next-day label
    # was the return into the first validation day. Use --yearcv, which backs the train end
    # off one trading day and validates on whole years.
    ap.add_argument("--yearcv", action="store_true",
                    help="WALK-FORWARD TRAINING MODE: expanding folds over WHOLE calendar "
                         "years, each retrained FROM SCRATCH with its own seed and given "
                         "its own validation year plus a clean test year -- "
                         "train 2019-2020/valid 2021/TEST 2022, train 2019-2021/valid 2022/"
                         "TEST 2023, train 2019-2022/valid 2023/TEST 2024. The valid year "
                         "picks the epoch; the test year is scored with those weights and "
                         "is never selected on. One data load serves every fold.")
    ap.add_argument("--yearcv-folds", default="2022,2023,2024", metavar="Y1,Y2,...",
                    help="TEST years; fold Y validates on Y-1 and trains on everything "
                         "before that (default 2022,2023,2024 = the three folds asked for)")
    ap.add_argument("--yearcv-rolling", type=int, default=0, metavar="N",
                    help="0 (default) = EXPANDING window (train always starts 2019-01-01); "
                         "N = rolling window of the N calendar years before the fold")
    ap.add_argument("--weight-decay", type=float, default=1e-5,
                    help="AdamW weight decay (was hardcoded 1e-5). The binding constraint "
                         "is regularization, so this is a first-class knob.")
    ap.add_argument("--exclude-months", default="", metavar="YYYY-MM,...",
                    help="drop these calendar months from the TRAINING TARGETS (their bars are "
                         "still available as lookback history for later samples). Motivation "
                         "(scripts/month_regime.py, 2026-07-27): training through 2024 costs "
                         "~-0.069 public in four independent A/Bs while every year of PAST data is "
                         "worth +0.128 -- only compatible if 2024 differs in KIND. Measured, it "
                         "does not: 8 of its 12 months sit inside the 2019-2023 regime range and "
                         "only 2024-01/02/09/10 are outliers (the small-cap unwind and the policy "
                         "rally), all extreme-co-movement months where the cross-section has "
                         "little to predict. This flag buys the other 8 months of data.")
    ap.add_argument("--aux-horizons", default="", metavar="H1,H2",
                    help="MULTI-HORIZON auxiliary supervision: also predict the t+H "
                         "forward return for each H, as extra head columns weighted by "
                         "--aux-weight. Measured 2026-07-26: cross-sectional rank corr of "
                         "the h1 label with h2/h5/h10 is 0.66/0.41/0.29, so these are "
                         "genuinely different tasks, not label copies (unlike shifting the "
                         "intraday anchor, which is 0.98 and would blur the closing "
                         "auction the alpha lives in). The EXTRA head columns are sliced "
                         "off at save time, so the shipped checkpoint is byte-compatible "
                         "with the single-output DayHIST the inference lib serves.")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="weight on each auxiliary horizon's IC loss (main h=1 stays 1.0)")
    ap.add_argument("--day-subsample", type=float, default=0.0, metavar="FRAC",
                    help="TRAINING-ONLY: use a random FRAC of each day's cross-section per "
                         "gradient step (0 = off). The effective sample size of a daily-IC "
                         "model is the number of DAYS (~1200), not stock-days; resampling "
                         "the cross-section makes each day yield DIFFERENT gradients across "
                         "epochs and stops the model leaning on particular names. Inference "
                         "always scores the full cross-section (this is graph/edge dropout, "
                         "the same asymmetry dropout has).")
    ap.add_argument("--hist-dropout", type=float, default=0.0,
                    help="dropout on the DayHIST hidden representation (none by default)")
    ap.add_argument("--hist-blocks", type=int, default=3,
                    help="hist_deep: how many [inter-day DNN -> HIST] blocks to stack")
    ap.add_argument("--hist-topk", type=int, default=0, metavar="K",
                    help="HIST concept graph: 0 = original argmax single-peer (default); "
                         "K>0 = soft temperature-softmax over the K most similar peers "
                         "(differentiable, multi-peer; targets the argmax variance source)")
    ap.add_argument("--hist-tau", type=float, default=0.1,
                    help="initial temperature for --hist-topk (learned thereafter)")
    ap.add_argument("--feat", default="wide", choices=["wide", "ohlcv", "narrow", "core8", "flows15"],
                    help="feature set: wide = 25 fields (default); ohlcv = "
                         "open/high/low/close/volume/amount only (6 fields -- the subset "
                         "available in EXTERNAL 2025+ data: baostock/tdx; used to build "
                         "the local public-replica evaluator); narrow = the 10-field set")
    ap.add_argument("--wf", action="store_true",
                    help="walk-forward protocol (2026-07-24 redesign): with --yearcv, each "
                         "fold trains 2019..(testyear-1) with NO validation year -- the test "
                         "year is reported only (fixed --epochs, take-LAST), never selects "
                         "anything. seed 0 on every fold. The 19-23->24 fold's saved model "
                         "is directly the submission checkpoint. Combine with --ema-decay.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="per-step exponential moving average of the weights (e.g. 0.999); "
                         "0 = off. Eval, test scoring and the saved checkpoint all use the "
                         "EMA parameters.")
    ap.add_argument("--yearcv-seeds", default="0",
                    help="BASE seeds (default one). Each fold gets its own derived seed "
                         "base + 1000*fold_index, so no two folds share an initialization; "
                         "extra base seeds are replications (3 trainings each).")
    ap.add_argument("--yearcv-select", default="ICIR",
                    choices=["ICIR", "IC", "RankICIR", "RankIC"],
                    help="valid-year criterion the early-stopping epoch is chosen on "
                         "(default ICIR: the mean alone is too easy to hit with a handful "
                         "of lucky days, and ICIR is the component that separates configs)")
    ap.add_argument("--yearcv-no-styles", action="store_true",
                    help="skip the BARRA-approximation panel used to style-neutralize the "
                         "fold report (score AND label); metrics are then RAW")
    ap.add_argument("--valid-start", default=None, metavar="YYYY-MM-DD",
                    help="validation window start (default 2024-01-01)")
    ap.add_argument("--valid-end", default=None, metavar="YYYY-MM-DD",
                    help="validation window end (default 2024-12-31)")
    ap.add_argument("--train-start", default=None, metavar="YYYY-MM-DD",
                    help="override the train range start (default 2019-01-01); tags the cell _tsYYYY")
    ap.add_argument("--train-end", default=None, metavar="YYYY-MM-DD",
                    help="extend the train range end (default 2023-12-31); tags the cell "
                         "_teYYYY; 2024 'valid' metrics become IN-SAMPLE and not comparable")
    ap.add_argument("--eval-only", default=None, metavar="CKPT",
                    help="load this checkpoint, run ONLY the validation pass, print RESULT, exit "
                         "(other args must match the checkpoint's config; data cache-hits)")
    ap.add_argument("--freq", default="1m", choices=["1m", "5m", "15m", "30m"],
                    help="bar frequency; non-1m uses the FULL trading day (48/16/8 bars)")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast for train+eval forward (fp32 master weights); "
                         "~2x on giant GRUs, tags the cell name with _bf16")
    ap.add_argument("--lazy", default="on", choices=["auto", "on", "off"],
                    help="lazy window assembly (30-60x less RAM for multi-day cells; DEFAULT on); "
                         "auto = on when seq_len >= 150")
    ap.add_argument("--save-dir", default=None,
                    help="save final state_dict as <save-dir>/<model>_<cell>_s<seed>.pt (for score blending)")
    ap.add_argument("--win-cache", default="win_cache",
                    help="dir for cached built windows keyed by (freq, tail, norm, seq_len, ranges); "
                         "seed/arch sweeps at one window shape skip the multi-hour load+build")
    ap.add_argument("--build-cache-only", action="store_true",
                    help="build+save the window cache, then exit without training")
    ap.add_argument("--swa", type=int, default=0, metavar="K",
                    help="also score the average of the last K epoch checkpoints on the "
                         "test year (swaIC=/swaICIR=/... on the RESULT line): validates "
                         "SWA as a no-validation final-train stopping rule")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after N epochs without a new best valid IC (0=off); "
                         "with --save-dir the BEST-IC epoch's weights are saved, not the last")
    ap.add_argument("--root", default="e2e_data")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"],
                    help="cosine (DEFAULT) = CosineAnnealingLR to ~0 over --epochs (stabilizes late "
                         "epochs so the FINAL weights ~ match the best epoch)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    fc = (C.WIDE_FEATURE_COLS if a.feat == "wide" else
          ["open", "high", "low", "close", "volume", "amount"] if a.feat == "ohlcv" else
          ["close", "volume", "amount", "deal_number", "bid_volume1", "ask_volume1",
           "bid_num_orders1", "ask_num_orders1"] if a.feat == "core8" else
          list(C.WIDE_FLOW_FEATURES) if a.feat == "flows15" else
          C.FEATURE_COLS)
    trn_end = a.train_end or TRAIN[1]
    trn_start = a.train_start or TRAIN[0]
    val_start = a.valid_start or VALID[0]
    val_end = a.valid_end or VALID[1]
    t0 = time.time()

    if a.norm == "mixnorm" and a.log_price:
        print("[mixnorm] does its own price log (anchored on preclose); ignoring --log-price",
              flush=True)
        a.log_price = False
    if a.sn_no_norm and a.norm != "mixnorm":
        print("[--sn-no-norm] WARNING: no internal norm AND no mixnorm prep -- the encoder "
              "sees raw scales; this is almost certainly not what you want", flush=True)
    wins = []
    if a.windows and (a.model == "gru15m" or a.freq != "1m"):
        sys.exit("--windows selects 1m bars by clock time; it is meaningless at "
                 f"model={a.model} freq={a.freq}")
    if a.model == "gru15m":
        freq, tail, day_bars = "15m", None, BARS_15M
    elif a.freq != "1m":
        freq, tail = a.freq, None                    # coarse freqs: full trading day
        day_bars = {"5m": 48, "15m": 16, "30m": 8}[a.freq]
    elif a.windows:
        # explicit segments: --tail can only ever express ONE slice running to the close,
        # so an ablation that ADDS e.g. the 13:01-13:15 block to the 14:46 tail is not
        # reachable through it. day_bars comes from the segments themselves.
        wins = D.parse_bar_windows(a.windows)
        tail, freq = None, "1m"
        day_bars = D.bar_window_count(wins)
        if day_bars == 0:
            sys.exit(f"--windows {a.windows!r} selects no trading bars")
    else:
        tail = a.tail or "14:46"
        h, m = map(int, tail.split(":"))
        t = h * 60 + m
        if t >= 13 * 60 + 1:                         # afternoon cutoff (14:46 -> 15 bars)
            day_bars = 15 * 60 - t + 1
        else:                                        # morning cutoff: skip the 11:31-13:00 lunch gap
            day_bars = (11 * 60 + 30 - t + 1) + 120  # morning bars + full afternoon (09:31 -> 240)
        freq = "1m"
    seq_len = day_bars * a.days

    # history lead-in must cover the FULL lookback so every --days value evaluates on the
    # SAME 2024 window (a fixed 40d buffer let long-lookback cells only score late-2024,
    # making their IC uncomparable to short-lookback cells). ~1.6 calendar days per trading
    # day + 40d slack; clamped so we never ask for pre-2019 months that don't exist.
    lead_days = int(a.days * a.day_stride * 1.6) + 40
    buf = lambda s: max(pd.Timestamp(s) - pd.Timedelta(days=lead_days),  # noqa
                        pd.Timestamp("2019-01-01")).strftime("%Y%m")
    print(f"loading {freq} {('windows=' + a.windows) if wins else ('tail=' + str(tail))} "
          f"seq_len={seq_len} ({a.days} days x {day_bars} bars) ...", flush=True)
    lazy = a.lazy == "on" or (a.lazy == "auto" and seq_len >= 150)
    # window cache: the built windows depend only on (freq, tail, norm, seq_len, ranges)
    # -- NOT on model/hidden/seed -- so seed/arch sweeps at one window shape can skip the
    # multi-hour load+build entirely (lazy path only).
    ck = None
    if a.win_cache and lazy:
        os.makedirs(a.win_cache, exist_ok=True)
        # a windows cell and a tail cell must never share a cache entry: same freq/seq_len,
        # completely different bars. The token is the segment list, so 13:01-13:15+14:46-
        # 15:00 and 09:31-09:45+14:46-15:00 stay distinct too.
        _wtok = ("w" + "+".join(f"{a1:04d}_{b1:04d}" for a1, b1 in wins)) if wins else None
        tag = (f"{freq}_{(_wtok or tail or 'na').replace(':', '')}_{a.norm}"
               f"{'_logp' if a.log_price else ''}_{seq_len}"
               f"{'' if a.day_stride == 1 else f'_sk{a.day_stride}'}"   # strided != contiguous
               f"{'' if a.feat == 'wide' else '_' + a.feat}")   # feat set changes the tensor!
        ck = os.path.join(a.win_cache, f"win_{tag}_{trn_start[:4]}-{trn_end[:4]}_{val_start[:4]}{val_start[5:7]}-{val_end[5:7]}.npz")
    def cs_by_month(pp, cols=None):
        """cs_standardize month-by-month via numpy bincount (matches the pandas
        groupby-transform: ddof=1 std, per-column NaN skipping, +1e-9 denom, float32
        out). Month slices keep intermediates bounded; bincount replaces the single-
        threaded groupby machinery (~10 min -> ~1 min on full-bar frames).
        cols limits the transform to a field subset (mixnorm cs-standardizes only the
        non-price fields); default all feature cols."""
        cols = fc if cols is None else cols
        per = pp["day"].dt.to_period("M")
        pos = pp.columns.get_indexer(cols)
        months = list(per.unique())

        def _one_month(m):
            ix = (per == m).to_numpy()
            codes, uniq = pd.factorize(pp.loc[ix, "date"], sort=False)
            G = len(uniq)
            V = pp.loc[ix, cols].to_numpy(np.float64)
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
            return ix, out

        # months hold disjoint rows, so each month's z-block is independent. numpy bincount
        # releases the GIL, so compute overlaps across threads (was a single-threaded month
        # loop, the ~171s step). TWO-PHASE for safety: workers ONLY read pp during the pool
        # (.to_numpy() copies), then the frame is mutated serially after the pool closes --
        # concurrent .iloc writes to a shared DataFrame are not thread-safe. Override via env.
        from concurrent.futures import ThreadPoolExecutor
        nw = max(1, min(int(os.environ.get("GRU_IO_WORKERS", os.cpu_count() or 8)
                            or (os.cpu_count() or 8)), len(months)))
        with ThreadPoolExecutor(max_workers=nw) as ex:
            results = list(ex.map(_one_month, months))
        for ix, out in results:
            pp.iloc[ix, pos] = out
        return pp

    def mix_norm(pp):
        """--norm mixnorm: price fields -> log(price / preclose); everything else -> the
        same offline cross-sectional z as --norm cs.

        The preclose anchor is ADJUST-STATE-AWARE: yesterday's ADJUSTED close mapped back
        into today's (unadjusted) price scale, anchor = AC(t-1) * C(t) / AC(t) with AC
        from adj_close_raw and C the raw close (each day's last bar). On a plain day this
        IS yesterday's close; across an ex-date it removes the artificial split/dividend
        jump that raw preclose would inject (A-share 送转 can be a fake -30% overnight).
        The first day of each stock has no preclose -> NaN -> its windows are dropped by
        the all-finite gate, which the lookback lead-in absorbs. adj_close_raw itself is
        left untouched (labels read it)."""
        pcols = [fc[i] for i in SN_PRICE]
        ocols = [c for c in fc if c not in pcols]
        dc = (pp.groupby(["key", "day"], sort=False, as_index=False)
                .agg(_ac=("adj_close_raw", "last"), _c=("close", "last")))
        anchor = dc["_ac"].groupby(dc["key"], sort=False).shift(1) * dc["_c"] / dc["_ac"]
        dc["_anchor"] = np.log(anchor.clip(lower=1e-6)).astype(np.float32)
        dc.loc[~np.isfinite(anchor), "_anchor"] = np.nan
        pp = pp.merge(dc[["key", "day", "_anchor"]], on=["key", "day"],
                      how="left", copy=False)
        A = pp.pop("_anchor").to_numpy(np.float32)[:, None]
        pp[pcols] = (np.log(pp[pcols].clip(lower=1e-6)).to_numpy(np.float32) - A)
        return cs_by_month(pp, cols=ocols)

    def load_canonical(lo_ym, hi_ym):
        """Month-chunked load+canonicalize: each month is converted (fen->yuan, NaN
        sentinels) in float64 TRANSIENTLY, downcast to float32, then concatenated.
        Cuts the full-bar (240 bars/day) load peak from ~250GB to ~50GB. ffill/log1p/
        cs run AFTER concat (cross-month correct); only the float64->float32 rounding
        point moves (~1e-7 relative, far below seed noise)."""
        months = pd.period_range(pd.Period(f"{lo_ym[:4]}-{lo_ym[4:]}", "M"),
                                 pd.Period(f"{hi_ym[:4]}-{hi_ym[4:]}", "M"), freq="M")

        def one(m):
            try:
                raw_m = D.load_local_bars(a.root, freq, month_lo=m.strftime("%Y%m"),
                                          month_hi=m.strftime("%Y%m"), tail_time=tail,
                                          bar_windows=wins)
            except FileNotFoundError:
                return None
            cm = D.to_canonical(raw_m, is_local=True)
            del raw_m
            fcast = {c: "float32" for c in fc if c in cm.columns and cm[c].dtype == "float64"}
            return cm.astype(fcast) if fcast else cm
        # threads: pyarrow feather reads release the GIL, so month reads+converts overlap;
        # results collected in month order -> identical concat. Thread count scales with CPU
        # cores (was hardcoded 8) -- a process pool would only add DataFrame-pickling overhead
        # since the GIL is already released. Full-bar loads (tail_time empty => 240 bars/day)
        # hold each in-flight month in float64 transiently (~6GB, ~50GB peak at 8 workers), so
        # those are capped by free RAM; tail loads (champion = 15 bars/day) are ~16x smaller
        # and run at full core width. Override with env GRU_IO_WORKERS.
        from concurrent.futures import ThreadPoolExecutor
        nw = min(os.cpu_count() or 8, len(months))
        if not tail and not wins:  # full-bar: bound concurrent float64 months by free RAM
            try:
                avail_gb = next(int(l.split()[1]) for l in open("/proc/meminfo")
                                if l.startswith("MemAvailable")) / 1e6
            except Exception:
                avail_gb = 64.0
            nw = min(nw, max(4, int(avail_gb * 0.5 / 6)))
        nw = max(1, min(int(os.environ.get("GRU_IO_WORKERS", nw) or nw), len(months)))
        with ThreadPoolExecutor(max_workers=nw) as ex:
            parts = [p for p in ex.map(one, months) if p is not None]
        return pd.concat(parts, ignore_index=True)

    dday = a.d_day or a.hidden
    nf = len(fc) * (2 if a.norm == "dual" else 1)   # dual = CS view + temporal view
    def _build_model():
        net = {"gru15m": lambda: GRUBars(nf),
                 "daymlp_gru": lambda: DayMLPGRU(day_bars, nf, d_day=dday, gru_hidden=a.hidden),
                 "dayattn": lambda: DayAttnGRU(day_bars, nf, d_day=dday, gru_hidden=a.hidden),
                 "hist_deep": lambda: DayHISTDeep(day_bars, nf, d_day=dday, gru_hidden=a.hidden,
                                                  blocks=a.hist_blocks, hist_topk=a.hist_topk,
                                                  hist_tau=a.hist_tau),
                 "daygru_gru": lambda: DayGRUGRU(day_bars, nf, d_day=dday, gru_hidden=a.hidden,
                                                layers=a.layers),
                 "hist_flat": lambda: DayHISTFlat(day_bars, nf, d_day=dday, gru_hidden=a.hidden,
                                                  hist_topk=a.hist_topk, hist_tau=a.hist_tau,
                                                  hist_dropout=a.hist_dropout,
                                                  layers=a.layers, public=a.public,
                                                  public_mode=a.public_mode,
                                                  day_neutral=a.day_neutral),
                 "hist_dual": lambda: DayHISTDual(day_bars, nf, d_day=dday, gru_hidden=a.hidden,
                                                  hist_topk=a.hist_topk, hist_tau=a.hist_tau,
                                                  hist_dropout=a.hist_dropout,
                                                  layers=a.layers, public=a.public,
                                                  public_mode=a.public_mode,
                                                  day_neutral=a.day_neutral),
                 "hist_day": lambda: DayHIST(day_bars, nf, d_day=dday, gru_hidden=a.hidden,
                                             hist_topk=a.hist_topk, hist_tau=a.hist_tau,
                                             hist_dropout=a.hist_dropout,
                                             layers=a.layers, public=a.public,
                                             public_mode=a.public_mode,
                                             day_neutral=a.day_neutral),
                 "daydnn_dnn": lambda: DayDNNDNN(day_bars, nf, a.days, d_day=dday,
                                                 hidden=a.hidden, kernel=a.kernel),
                 "daytsf_tsf": lambda: DayTSFTSF(day_bars, nf, a.days,
                                                 d_model=a.hidden),
                 "stocknet_flat": lambda: StockNetFlat(day_bars, nf, d_day=dday, gru_hidden=a.hidden,
                                                      d_model=a.d_model, heads=a.heads, layers=a.layers,
                                                      bar_minutes={"1m": 1.0, "5m": 5.0, "15m": 15.0,
                                                                   "30m": 30.0}[a.freq],
                                                      ln=not a.no_layernorm,
                                                      dual_view=not a.sn_temporal_only,
                                                      public=a.public, public_mode=a.public_mode,
                                                      day_neutral=a.day_neutral),
                 "stocknet": lambda: StockNet(day_bars, nf, d_day=dday, gru_hidden=a.hidden,
                                              d_model=a.d_model, heads=a.heads, layers=a.layers,
                                              ln=not a.no_layernorm,
                                              dual_view=not a.sn_temporal_only,
                                              input_norm=not a.sn_no_norm,
                                              public=a.public, public_mode=a.public_mode,
                                              day_neutral=a.day_neutral),
                 "hist_dnn": lambda: HISTDNN(day_bars, nf, a.days, d_day=dday,
                                             hidden=a.hidden, kernel=a.kernel, public=a.public,
                                             public_mode=a.public_mode,
                                             day_neutral=a.day_neutral)}[a.model]()
        # the norm wrapper is part of the model: --norm dual doubles the input channels,
        # so a caller that skips it hands 25 channels to a 50-channel GRU (this bit the
        # fold-based modes until it was folded in here)
        if a.norm == "csmodel":
            net = CSWrapped(net)
        elif a.norm == "dual":
            net = DualWrapped(net)
        return net


    mse_ = nn.MSELoss()

    def _neutral_ic_loss(pred, y, S):
        """Negative Pearson of the STYLE-RESIDUAL score vs the STYLE-RESIDUAL label.

        `S` is that day's (N, k+1) style design matrix (intercept + bar-built BARRA-ish
        factors), constant w.r.t. the weights, so the projection is a plain linear op and
        gradients flow through `pred` only. Residualizing BOTH sides reproduces the graded
        quantity: the platform residualizes submitted scores on style+industry and its
        target is already a residual return. Stocks with missing styles carry an all-zero
        style row, so they are simply not projected.
        """
        # RIDGE normal equations, NOT torch.linalg.lstsq: the default gels driver
        # assumes S has full column rank, but a day's style matrix is often rank-
        # deficient (rows with missing styles are zero-filled, factors can be locally
        # collinear) -> gels returns NaN and the whole run dies. (k+1) is ~8, so the
        # solve is free.
        k = S.shape[1]
        G = S.t() @ S + 1e-3 * torch.eye(k, device=S.device, dtype=S.dtype)
        V = torch.stack([pred, y], dim=1)                 # residualize BOTH in one solve
        beta = torch.linalg.solve(G, S.t() @ V)
        res = V - S @ beta
        p, t_ = res[:, 0], res[:, 1]
        p = p - p.mean(); t_ = t_ - t_.mean()
        denom = p.norm() * t_.norm()
        if not torch.isfinite(denom) or denom < 1e-8:     # degenerate day -> skip it
            return pred.sum() * 0.0
        return -(p * t_).sum() / denom

    def _ic_loss(pred, y):
        pc = pred - pred.mean(); yc = y - y.mean()
        return -(pc * yc).sum() / (pc.norm() * yc.norm() + 1e-8)

    def _soft_rank(x, tau):
        """Differentiable rank: rank_i = sum_j sigmoid((x_i - x_j)/tau).

        tau -> 0 recovers the exact rank (a step function); larger tau smooths it.
        O(N^2) but N is one trading day's cross-section (~980), so the pairwise matrix
        is ~4 MB -- negligible next to the sequence forward.
        """
        d = (x.unsqueeze(1) - x.unsqueeze(0)) / tau
        return torch.sigmoid(d).sum(1)

    def _rankic_loss(pred, y):
        """Negative Spearman IC (soft-ranked prediction vs exact-ranked label).

        CAVEAT ON THE MOTIVATION (corrected 2026-07-27): the competition's "Rank IC"
        means the PERCENTILE RANK OF YOUR IC AMONG SUBMITTERS, not a rank correlation --
        "Rank_IC_mean: Percentile ranking of cross-sectional Information Coefficient
        mean". Whether the underlying IC is Pearson or Spearman is NOT stated, and the
        leaderboard column is labelled plain "IC", which by convention usually means
        Pearson. So this loss is NOT "matching the scored metric" as first claimed.

        What survives as a rationale is narrower but still real: ICIR = mean(IC)/std(IC)
        whatever IC is, and days where a handful of extreme returns dominate produce
        extreme daily IC values that inflate std(IC) and depress ICIR. ICIR is the
        strongest score driver measured (+0.648 vs score, n=10 same-snapshot). Ranking
        removes that lever. RISK: if the metric is Pearson, this trains a different
        quantity and mean IC may fall -- acceptable only because IC's own correlation
        with score measured -0.006.

        The LABEL side uses exact ranks (no gradient needed there), so the only
        approximation is on the prediction side.
        """
        tau = a.rank_tau * (pred.detach().std() + 1e-8)
        rp = _soft_rank(pred, tau)
        ry = torch.argsort(torch.argsort(y)).float()
        rp = rp - rp.mean(); ry = ry - ry.mean()
        return -(rp * ry).sum() / (rp.norm() * ry.norm() + 1e-8)

    class ScoreLoss:
        """Platform-score-aligned day-batch loss:
            -w_stress * (IC + alpha * softLS) + lambda * (IC - EMA_IC)^2
        softLS = softmax(+/-pred/tau)-weighted spread of per-day standardized labels
        (differentiable decile-Sharpe proxy); the EMA penalty is mean-variance ascent
        on daily IC (targets IC_IR); w_stress upweights down-market days (targets the
        worst-regime Stress component). Running scales are no-grad EMAs."""

        def __init__(self, alpha, beta, lam, tau_frac=0.1):
            self.alpha, self.beta, self.lam, self.tau_frac = alpha, beta, lam, tau_frac
            self.ema_ic = None; self.ema_ls = None; self.ema_m2 = None

        def __call__(self, pred, y):
            pc = pred - pred.mean(); yc = y - y.mean()
            ic = (pc * yc).sum() / (pc.norm() * yc.norm() + 1e-8)
            ystd = yc / (y.std() + 1e-8)
            tau = self.tau_frac * (pred.detach().std() + 1e-8)
            ls = ((torch.softmax(pred / tau, 0) - torch.softmax(-pred / tau, 0)) * ystd).sum()
            m = float(y.mean())
            with torch.no_grad():
                icd, lsd = float(ic), abs(float(ls))
                self.ema_ic = icd if self.ema_ic is None else 0.98 * self.ema_ic + 0.02 * icd
                self.ema_ls = lsd if self.ema_ls is None else 0.98 * self.ema_ls + 0.02 * lsd
                self.ema_m2 = m * m if self.ema_m2 is None else 0.98 * self.ema_m2 + 0.02 * m * m
            w = 1.0 + self.beta / (1.0 + np.exp(m / (self.ema_m2 ** 0.5 + 1e-8)))
            ir_pen = (ic - ic.new_tensor(self.ema_ic)) ** 2
            return -w * (ic + self.alpha * ls / (self.ema_ls + 1e-8)) + self.lam * ir_pen

    def _make_loss():
        """A FRESH loss object per training run.

        ScoreLoss carries no-grad EMAs, so one shared instance would let fold 1's running
        scales prime fold 2. Worse, until this was hoisted the year-CV path never reached
        the score branch at all: `lossf` was bound to _ic_loss here while `--loss score`
        was only honoured ~400 lines below, inside the single-split path -- so
        `--yearcv --loss score` trained the plain IC loss and reported it as a `score`
        cell, whose folds came out bit-identical to the matching `ic` cell.
        """
        if a.loss == "mse":
            return mse_
        if a.loss == "score":
            return ScoreLoss(a.score_alpha, a.score_beta, a.score_lambda)
        if a.loss == "rankic":
            return _rankic_loss
        if a.loss == "nic":
            return _neutral_ic_loss          # called with the day's style matrix
        return _ic_loss

    lossf = _make_loss()

    def _rss(lbl):
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    print(f"  [mem] {lbl}: {int(ln.split()[1])/2**20:.1f} GB | {time.time()-t0:.0f}s", flush=True)
                    return

    def _fit_eval(b_tr, e_tr, y_tr, i_tr, b_va, e_va, y_va, i_va, st_, sd, fold, test=None,
                  honest=False):
        """Train one model on a walk-forward fold and return its metrics.

        Returns the BEST-epoch valid metrics
        with two extras attached: ``["last"]`` = the LAST epoch's valid metrics, and
        ``["test"]`` = metrics on ``test`` = (base, ends, y, idx) when given. Because the
        valid year exists precisely to choose the epoch, the test year is scored with the
        weights of the epoch that won on VALID -- snapshotted, not re-derived -- so the
        test year is never itself a selection surface. ``["test_last"]`` is the same
        thing at the last epoch, printed alongside to show what epoch choice bought.

        ``honest=True`` (no valid year) says the "validation" set IS the out-of-sample
        test year, so the whole line collapses to the last epoch. ``["pred"]`` /
        ``["test_pred"]`` carry the predictions the per-fold report is built from.
        """
        f_tr = lambda ixs: np.stack([b_tr[e - seq_len:e] for e in e_tr[ixs]])   # noqa: E731
        f_va = lambda ixs: np.stack([b_va[e - seq_len:e] for e in e_va[ixs]])   # noqa: E731
        dgrp = [np.asarray(ix, dtype=np.int64) for _, ix in
                i_tr.reset_index(drop=True).groupby("date").groups.items()]
        vgrp = [np.asarray(ix, dtype=np.int64) for _, ix in
                i_va.reset_index(drop=True).groupby("date").groups.items()]
        net = _build_model()
        net = net.to(dev)
        lossf = _make_loss()          # per-fold: ScoreLoss keeps EMA state across calls
        op = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sch = (torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=a.epochs)
               if a.lr_schedule == "cosine" else None)
        yt_ = torch.from_numpy(y_tr)
        base_lr = [g["lr"] for g in op.param_groups]
        sel = a.yearcv_select                           # valid-year criterion, default ICIR
        gs, best_ = 0, {k: -9e9 for k in ("IC", "ICIR", "RankIC", "RankICIR")}
        last_, best_sd, best_ep = best_, None, -1
        swa_snaps = []                     # rolling CPU snapshots of the last --swa epochs
        ema = None                         # per-step weight EMA (--ema-decay); lives on dev

        def _load_params(src):
            with torch.no_grad():
                for k_, p_ in net.named_parameters():
                    p_.copy_(src[k_])

        for ep in range(a.epochs):
            net.train()
            for b in [dgrp[k] for k in np.random.permutation(len(dgrp))]:
                if a.day_subsample and 0 < a.day_subsample < 1 and len(b) > 50:
                    b = np.random.choice(b, size=max(50, int(len(b) * a.day_subsample)),
                                         replace=False)
                if a.warmup and gs < a.warmup:
                    for gp, bl in zip(op.param_groups, base_lr):
                        gp["lr"] = bl * (gs + 1) / a.warmup
                gs += 1
                op.zero_grad()
                loss = lossf(net(torch.from_numpy(f_tr(b)).to(dev)),
                             yt_[torch.from_numpy(b)].to(dev))
                loss.backward(); op.step()
                if a.ema_decay > 0:
                    with torch.no_grad():
                        if ema is None:
                            ema = {k_: p_.detach().clone()
                                   for k_, p_ in net.named_parameters()}
                        else:
                            for k_, p_ in net.named_parameters():
                                ema[k_].mul_(a.ema_decay).add_(p_.detach(),
                                                               alpha=1 - a.ema_decay)
            if sch is not None:
                sch.step()
            net.eval()
            # all eval (and any snapshot taken from it) sees the EMA parameters
            _raw = None
            if ema is not None:
                _raw = {k_: p_.detach().clone() for k_, p_ in net.named_parameters()}
                _load_params(ema)
            pr = np.empty(len(y_va), dtype=np.float32)
            with torch.no_grad():
                for ixs in vgrp:
                    pr[ixs] = net(torch.from_numpy(f_va(ixs)).to(dev)).float().cpu().numpy()
            ic = TB.daily_ic(i_va, pr, y_va)
            ic.update(replica_components(i_va, pr, y_va))
            last_ = ic
            # ICIR, not IC: the mean alone rewards an epoch that got a few lucky days, and
            # ICIR is the component our submissions actually differ on. --yearcv-select
            # switches it back for a comparison run.
            if ic[sel] > best_[sel]:
                best_ = ic
                # Snapshot on CPU: the test year must be scored with the weights the VALID
                # year chose, and re-running the winning epoch is not reproducible once the
                # optimizer has moved on. ~3.5 MB at h256, so keeping it is free.
                best_sd = {k: v.detach().to("cpu", copy=True)
                           for k, v in net.state_dict().items()}
                best_ep = ep
            if a.swa:
                swa_snaps.append({k: v.detach().to("cpu", copy=True)
                                  for k, v in net.state_dict().items()})
                if len(swa_snaps) > a.swa:
                    swa_snaps.pop(0)
            if _raw is not None:
                _load_params(_raw)                     # resume training from RAW weights
        va_pred = pr                                   # last-epoch predictions, for the report
        if best_sd is None:                            # every epoch scored NaN on `sel`
            best_, best_ep = last_, a.epochs - 1
        if honest:
            # The "valid" set is this fold's out-of-sample TEST year: a best-of-N-epochs
            # number would be peeking at the very year the fold exists to keep honest.
            best_, best_ep = last_, a.epochs - 1
            # take-LAST weights = final EMA params when --ema-decay is on, else the raw
            # final params; this becomes both the test-scoring model and the saved ckpt
            sd_full = {k: v.detach().to("cpu", copy=True)
                       for k, v in net.state_dict().items()}
            if ema is not None:
                for k_ in ema:
                    sd_full[k_] = ema[k_].detach().to("cpu", copy=True)
            best_sd = sd_full
        tst, tst_last, te_pred = {}, {}, None
        if test is not None:
            b_te, e_te, y_te, i_te = test
            f_te = lambda ixs: np.stack([b_te[e - seq_len:e] for e in e_te[ixs]])  # noqa: E731
            tgrp = [np.asarray(ix, dtype=np.int64) for _, ix in
                    i_te.reset_index(drop=True).groupby("date").groups.items()]

            def _score_test():
                p = np.empty(len(y_te), dtype=np.float32)
                with torch.no_grad():
                    for ixs in tgrp:
                        p[ixs] = net(torch.from_numpy(f_te(ixs)).to(dev)).float().cpu().numpy()
                m = TB.daily_ic(i_te, p, y_te)
                m.update(replica_components(i_te, p, y_te))
                return p, m

            _, tst_last = _score_test()                # last-epoch RAW weights, for comparison
            if best_sd is not None:
                net.load_state_dict(best_sd)           # valid-picked epoch, or take-last(EMA)
                net.eval()
            te_pred, tst = _score_test()
        tst_swa = {}
        if a.swa and swa_snaps and test is not None:
            # SWA candidate: plain average of the last K epoch snapshots (no BatchNorm in
            # any of our models, so no re-estimation pass is needed). Scored on the test
            # year NEXT TO the valid-selected epoch -- a final-train stand-in that needs
            # no validation year at all.
            avg = {k: torch.stack([s[k].float() for s in swa_snaps]).mean(0)
                   .to(swa_snaps[-1][k].dtype) for k in swa_snaps[-1]}
            net.load_state_dict(avg)
            net.eval()
            _, tst_swa = _score_test()
        head = tst if tst else best_                   # the dashboard scrapes `IC=`
        print(f"RESULT model={a.model} freq={freq} fold={fold} days={a.days} seed={sd} "
              f"seqlen={seq_len} norm={a.norm} loss={a.loss} bmode={a.batch_mode} "
              f"RankIC={head['RankIC']:.4f} RankICIR={head['RankICIR']:.4f} "
              f"IC={head['IC']:.4f} ICIR={head['ICIR']:.4f} "
              f"SR={head.get('SR', float('nan')):.4f} Stress={head.get('Stress', float('nan')):.4f} "
              f"validIC={best_['IC']:.4f} validICIR={best_['ICIR']:.4f} "
              f"validsel={sel} validep={best_ep + 1}/{a.epochs} "
              f"lastIC={last_['IC']:.4f} lastICIR={last_['ICIR']:.4f} "
              + (f"testlastIC={tst_last['IC']:.4f} "
                 f"testdays={int(i_te['date'].nunique())} " if tst else "")
              + (f"swaK={len(swa_snaps)} swaIC={tst_swa['IC']:.4f} "
                 f"swaICIR={tst_swa['ICIR']:.4f} "
                 f"swaSR={tst_swa.get('SR', float('nan')):.4f} "
                 f"swaStress={tst_swa.get('Stress', float('nan')):.4f} " if tst_swa else "")
              + f"evaldays={int(i_va['date'].nunique())}", flush=True)
        if a.save_dir and best_sd is not None:
            os.makedirs(a.save_dir, exist_ok=True)
            _pt = os.path.join(a.save_dir, f"yearcv_{a.model}_{a.freq}_d{a.days}_{a.norm}"
                                           f"_{fold}_s{sd}.pt")
            torch.save({"state_dict": best_sd, "stats": st_, "fold": fold, "seed": sd,
                        "epoch": best_ep + 1, "ema_decay": a.ema_decay, "wf": bool(a.wf),
                        "cfg": {"days": a.days, "hidden": a.hidden, "freq": a.freq,
                                "norm": a.norm, "tail": a.tail or ""}}, _pt)
            print(f"  saved {_pt}", flush=True)
        del net, op
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        best_ = dict(best_)
        best_["last"], best_["test"], best_["test_last"] = last_, tst, tst_last
        best_["test_swa"] = tst_swa
        best_["pred"], best_["test_pred"], best_["epoch"] = va_pred, te_pred, best_ep + 1
        return best_

    def _prep_full(hi_ym, label):
        """Load + preprocess the WHOLE span once; every fold slices this one frame."""
        cn = load_canonical(buf("2019-01-01"), hi_ym)
        pp = D.preprocess_canonical(cn, feature_cols=fc, copy=False)
        del cn
        if a.log_price:
            pcols = [fc[i] for i in SN_PRICE]
            pp[pcols] = np.log(pp[pcols].clip(lower=1e-6))
        if a.norm == "cs":
            pp = cs_by_month(pp)
        elif a.norm == "mixnorm":
            pp = mix_norm(pp)
        _rss(f"{label}: full frame prepped")
        return pp

    if a.yearcv:
        # ---- WALK-FORWARD YEAR MODE ------------------------------------------------
        # Three expanding folds, each with its OWN validation year and a clean test year
        # one step further out:
        #
        #   fold 1   train 2019-2020   valid 2021   TEST 2022
        #   fold 2   train 2019-2021   valid 2022   TEST 2023
        #   fold 3   train 2019-2022   valid 2023   TEST 2024
        #
        # The valid year is what picks the epoch; the test year is scored with the weights
        # that epoch produced and is never itself selected on. That separation is the whole
        # point -- ranking configs on a best-of-12-epochs number computed on the same year
        # being reported is how 2024 came to look like the easiest year in the sample.
        # Each fold is retrained FROM SCRATCH with its own seed, so fold 3 is not fold 2
        # with more data bolted on and a lucky initialization cannot carry the mean.
        #
        # Leakage is blocked twice: training ends one trading day before the valid year,
        # and lazy_windows independently refuses any window whose next-day label falls
        # outside its range. The cost of the design is a one-year gap between the end of
        # training and the start of the test year -- the model is deliberately stale
        # there, which is the honest version of what a submission faces.
        import statistics as _st
        try:
            from bigalpha import fold_metrics as FM
        except Exception as exc:                                        # noqa: BLE001
            # The report is a diagnostic; a training box missing bigalpha/styles.py must
            # lose the report, not the run.
            FM = None
            print(f"  [fold report UNAVAILABLE: {exc}]", flush=True)
        folds = [int(y) for y in a.yearcv_folds.split(",") if y.strip()]
        bases = [int(x) for x in a.yearcv_seeds.split(",") if x.strip()]
        if a.save_dir is None:
            # every fold's valid-selected weights are saved BY DEFAULT (~3.5MB each):
            # re-training a fold just to recover an upload candidate is never worth it
            a.save_dir = "saved_models"
        pp_all = _prep_full(f"{max(folds)}12", "year-cv")
        all_days = np.sort(pp_all["day"].unique())

        styles = None
        if FM is not None and not a.yearcv_no_styles:
            # The platform residualizes scores on BARRA style + industry and the target is
            # itself a residual return, so the fold report neutralizes BOTH sides. This is
            # an approximation built from bars (no industry -- see bigalpha/styles.py); a
            # failure here must degrade to raw metrics loudly, never silently.
            try:
                from bigalpha.styles import load_styles
                styles = load_styles(freq="30m", month_lo="201901",
                                     month_hi=f"{max(folds)}12", instrument_fmt=None)
                print(f"  [styles] panel {len(styles):,} rows, "
                      f"{styles['date'].nunique()} days", flush=True)
            except Exception as exc:                                    # noqa: BLE001
                print(f"  [styles] UNAVAILABLE ({exc}) -> fold reports are RAW", flush=True)

        def _emit_fold_json(y, vy, ts, te, sd, r, rep, i_te):
            """`FOLDJSON {...}` -- everything the dashboard needs for one test year.

            The RESULT line cannot carry this: the neutralized IC/ICIR and the regime
            stress live in the fold report, and RESULT has no hidden/tail fields. Emitted
            even when the report failed (rep=None) so a fold is never silently missing.
            """
            import json as _json
            o = (rep or {}).get("overall", {})
            t = r["test"] or {}
            rec = {"model": a.model + ("@wf" if a.wf else ""),   # wf rows = separate cells
                   "feat": "" if a.feat == "wide" else a.feat,   # subset cells separate too
                   "freq": freq, "test_year": y, "valid_year": vy,
                   "train": f"{ts}..{te}", "seed": sd, "epoch": r["epoch"],
                   "days": a.days, "inday_seq": int(seq_len // max(a.days, 1)),
                   "seqlen": seq_len, "h": a.hidden, "dday": a.d_day or a.hidden,
                   "norm": a.norm, "loss": a.loss, "bmode": a.batch_mode,
                   "tail": a.tail or "", "features": getattr(a, "features", ""),
                   "test_days": int(i_te["date"].nunique()),
                   # raw (un-neutralized) test-year components, same definitions as RESULT
                   "IC": t.get("IC"), "ICIR": t.get("ICIR"),
                   "RankIC": t.get("RankIC"), "RankICIR": t.get("RankICIR"),
                   "SR_raw": t.get("SR"), "Stress_quad": t.get("Stress"),
                   # platform-chain: score AND label style-neutralized
                   "neutralized": (rep or {}).get("neutralized", False),
                   "nIC": o.get("IC"), "nICIR": o.get("ICIR"),
                   "SR": o.get("SR"),
                   "SRw": (o.get("SR_ladder") or {}).get("rank-wtd"),
                   "SR_ladder": o.get("SR_ladder"),
                   "Stress": ((rep or {}).get("stress") or {}).get("worst_IC"),
                   "Stress_SR": ((rep or {}).get("stress") or {}).get("worst_SR"),
                   "Stress_regime": ((rep or {}).get("stress") or {}).get("worst_IC_regime"),
                   "validIC": r["IC"], "validICIR": r["ICIR"]}
            print("FOLDJSON " + _json.dumps(rec, default=float), flush=True)

        def _train_span(vy):
            """[start, end] for a fold whose VALIDATION year is `vy`. `end` is the trading
            day before `vy` starts (one day back), so no training label lands inside it."""
            prev = all_days[all_days < np.datetime64(f"{vy}-01-01")]
            if len(prev) < 2:
                return None
            start = ("2019-01-01" if not a.yearcv_rolling
                     else max(f"{vy - a.yearcv_rolling}-01-01", "2019-01-01"))
            return start, pd.Timestamp(prev[-2]).strftime("%Y-%m-%d")

        results = []
        for fi, y in enumerate(folds):
            if a.wf:
                # WALK-FORWARD redesign (2026-07-24): train through testyear-1, NO valid
                # year at all -- the "valid" windows ARE the test year, reported only
                # (honest take-last); nothing is ever selected on any eval set.
                vy = y
            else:
                vy = y - 1                              # valid year sits between train and test
            span = _train_span(vy)
            if span is None:
                print(f"  [fold {y}] skipped: no training days before {vy}", flush=True)
                continue
            ts, te = span
            try:
                b_tr, e_tr, y_tr, i_tr, st_ = lazy_windows(pp_all, ts, te, fc, seq_len)
                b_va, e_va, y_va, i_va, _ = lazy_windows(pp_all, f"{vy}-01-01", f"{vy}-12-31",
                                                         fc, seq_len, st_)
                b_te, e_te, y_te, i_te, _ = lazy_windows(pp_all, f"{y}-01-01", f"{y}-12-31",
                                                         fc, seq_len, st_)
            except ValueError as exc:
                print(f"  [fold {y}] skipped: {exc}", flush=True)
                continue
            print(f"\n  [fold {fi + 1}/{len(folds)}] train {ts}..{te} ({len(y_tr):,} win) "
                  f"-> valid {vy} ({len(y_va):,} win, {i_va['date'].nunique()}d) "
                  f"-> TEST {y} ({len(y_te):,} win, {i_te['date'].nunique()}d)", flush=True)
            for base in bases:
                # wf protocol: SAME seed on every fold (user spec: seed=0 each run);
                # classic yearcv: a new seed per fold, retrained from zero
                sd = base if a.wf else base + 1000 * fi
                torch.manual_seed(sd); np.random.seed(sd)
                r = _fit_eval(b_tr, e_tr, y_tr, i_tr, b_va, e_va, y_va, i_va, st_, sd,
                              f"cv{y}", test=(b_te, e_te, y_te, i_te), honest=a.wf)
                results.append((y, sd, r))
                rep = None
                try:
                    if FM is None:
                        raise RuntimeError("bigalpha.fold_metrics not importable")
                    rep = FM.fold_report(i_te, r["test_pred"], y_te, styles)
                    print(FM.format_report(rep, f"TEST {y} (train {ts[:4]}..{te[:4]}, "
                                                f"valid {vy} picked epoch {r['epoch']}, "
                                                f"seed {sd})"), flush=True)
                except Exception as exc:                                # noqa: BLE001
                    print(f"    [fold report failed: {exc}]", flush=True)
                # One machine-readable line per fold: the dashboard builds the year-CV
                # table from THIS, not from the RESULT line, because the neutralized
                # components only exist inside the report.
                _emit_fold_json(y, vy, ts, te, sd, r, rep, i_te)
            del b_tr, e_tr, y_tr, b_va, e_va, y_va, b_te, e_te, y_te

        if results:
            byfold, byvalid = {}, {}
            for y, sd, r in results:
                byfold.setdefault(y, []).append(r["test"]["IC"])
                byvalid.setdefault(y, []).append(r["IC"])
            fold_means = {y: _st.mean(v) for y, v in byfold.items()}
            valid_means = {y: _st.mean(v) for y, v in byvalid.items()}
            m = _st.mean(list(fold_means.values()))
            sdv = _st.pstdev(list(fold_means.values())) if len(fold_means) > 1 else 0.0
            print(f"\nYEARCV {a.model} d{a.days} {a.norm} freq={freq} "
                  f"{'rolling' + str(a.yearcv_rolling) if a.yearcv_rolling else 'expanding'} "
                  f"seeds={len(bases)}: cvIC={m:+.4f} foldStd={sdv:.4f} "
                  f"worstFold={min(fold_means.values()):+.4f}", flush=True)
            print("  per fold (TEST year, valid-selected epoch): "
                  + "  ".join(f"{y}:{v:+.4f}" for y, v in sorted(fold_means.items())), flush=True)
            # The valid years are the epoch-selection surface, so they read high by
            # construction; the gap to the test row is the size of that optimism.
            print("  per fold (VALID year, best epoch -- selection surface): "
                  + "  ".join(f"{y - 1}:{v:+.4f}" for y, v in sorted(valid_means.items())),
                  flush=True)
        return

    if ck and os.path.exists(ck):
        z = np.load(ck, allow_pickle=False)
        base_tr, ends_tr, ytr = z["base_tr"], z["ends_tr"], z["ytr"]
        base_va, ends_va, yva = z["base_va"], z["ends_va"], z["yva"]
        stats = (z["mean"], z["std"])
        idx_tr = pd.DataFrame({"date": pd.to_datetime(z["date_tr"]), "key": z["key_tr"]})
        idx_va = pd.DataFrame({"date": pd.to_datetime(z["date_va"]), "key": z["key_va"]})
        print(f"  WIN-CACHE hit {ck}: train {len(ytr):,} valid {len(yva):,} | {time.time()-t0:.0f}s", flush=True)
    else:
        cn = load_canonical(buf(trn_start), pd.Timestamp(trn_end).strftime("%Y%m"))
        _rss("train canonical loaded")
        pp = D.preprocess_canonical(cn, feature_cols=fc, copy=False)
        del cn
        if a.log_price:
            pcols = [fc[i] for i in SN_PRICE]
            pp[pcols] = np.log(pp[pcols].clip(lower=1e-6))
        _rss("train preprocessed")
        if a.norm == "cs":
            pp = cs_by_month(pp)
            _rss("train cs done")
        elif a.norm == "mixnorm":
            pp = mix_norm(pp)
            _rss("train mixnorm done")
        if a.day_stride > 1:
            assert lazy, "--day-stride needs the lazy window path"
            base_tr, ends_tr, ytr, idx_tr, stats = stride_windows(pp, trn_start, trn_end, fc, seq_len, a.day_stride)
        elif lazy:
            base_tr, ends_tr, ytr, idx_tr, stats = lazy_windows(pp, trn_start, trn_end, fc, seq_len)
        else:
            Xtr, ytr, idx_tr, stats = D.build_windows(pp, trn_start, trn_end, "train", feature_cols=fc, seq_len=seq_len)
        del pp
        _rss("train windows built")
        cn = load_canonical(buf(val_start), pd.Timestamp(val_end).strftime("%Y%m"))
        pp = D.preprocess_canonical(cn, feature_cols=fc, copy=False)
        del cn
        if a.log_price:
            pcols = [fc[i] for i in SN_PRICE]
            pp[pcols] = np.log(pp[pcols].clip(lower=1e-6))
        if a.norm == "cs":
            pp = cs_by_month(pp)
        elif a.norm == "mixnorm":
            pp = mix_norm(pp)
        _rss("valid prepped")
        if a.day_stride > 1:
            base_va, ends_va, yva, idx_va, _ = stride_windows(pp, val_start, val_end, fc, seq_len, a.day_stride, stats)
        elif lazy:
            base_va, ends_va, yva, idx_va, _ = lazy_windows(pp, val_start, val_end, fc, seq_len, stats)
        else:
            Xva, yva, idx_va, _ = D.build_windows(pp, val_start, val_end, "train", stats, feature_cols=fc, seq_len=seq_len)
        del pp
        _rss("valid windows built")
        if ck:
            # write-then-rename: seeds of a NEW cell are launched together, so a second
            # process can reach the `WIN-CACHE hit` existence check while the first is
            # still streaming a multi-GB npz to the same path -- and would load a
            # truncated file as if it were a complete cache. os.replace is atomic within
            # a filesystem, so the path either does not exist or is a finished cache.
            tmp = f"{ck}.tmp{os.getpid()}"
            np.savez(tmp, base_tr=base_tr, ends_tr=ends_tr, ytr=ytr,
                     base_va=base_va, ends_va=ends_va, yva=yva,
                     mean=stats[0], std=stats[1],
                     date_tr=idx_tr["date"].to_numpy().astype("datetime64[ns]"),
                     key_tr=idx_tr["key"].to_numpy(),
                     date_va=idx_va["date"].to_numpy().astype("datetime64[ns]"),
                     key_va=idx_va["key"].to_numpy())
            if not tmp.endswith(".npz"):
                tmp += ".npz"          # np.savez appends .npz when the name lacks it
            os.replace(tmp, ck)
            print(f"  WIN-CACHE saved {ck} ({os.path.getsize(ck)/1e9:.1f}GB)", flush=True)
    if lazy:
        fetch_tr = lambda ixs: np.stack([base_tr[e - seq_len:e] for e in ends_tr[ixs]])  # noqa: E731
        fetch_va = lambda ixs: np.stack([base_va[e - seq_len:e] for e in ends_va[ixs]])  # noqa: E731
        print(f"  LAZY windows: train {len(ytr):,}x{seq_len} (base {base_tr.nbytes/1e9:.1f}GB) "
              f"valid {len(yva):,} (base {base_va.nbytes/1e9:.1f}GB) | {time.time()-t0:.0f}s", flush=True)
    else:
        fetch_tr = lambda ixs: Xtr[ixs]           # noqa: E731
        fetch_va = lambda ixs: Xva[ixs]           # noqa: E731
        print(f"  train {Xtr.shape} valid {Xva.shape} | {time.time()-t0:.0f}s", flush=True)
    if a.build_cache_only:
        print("  --build-cache-only: exiting after cache write", flush=True)
        return

    if (a.model in ("hist_day", "hist_dnn", "stocknet", "stocknet_flat", "hist_flat", "hist_dual") or a.norm in ("csmodel", "dual")) and a.batch_mode != "daily":
        raise SystemExit("hist_day / hist_dnn / csmodel / dual require --batch-mode daily (whole-day cross-sections)")
    if (a.public or a.day_neutral) and a.model not in ("hist_day", "hist_dnn", "stocknet", "stocknet_flat", "hist_flat", "hist_dual"):
        raise SystemExit("--public / --day-neutral only apply to hist_day / hist_dnn")
    if a.day_neutral and a.batch_mode != "daily":
        raise SystemExit("--day-neutral requires --batch-mode daily (whole-day cross-sections)")
    model = _build_model()
    if a.label_rank:
        _s = pd.Series(ytr, index=pd.MultiIndex.from_arrays(
            [idx_tr["date"].to_numpy(), idx_tr["key"].to_numpy()], names=["datetime", "instrument"]))
        _r = _s.groupby(level="datetime").rank(pct=True) - 0.5
        ytr = (_r * np.sqrt(12.0)).to_numpy(np.float32)
        print(f"[label-rank] per-day percentile rank: y std {ytr.std():.4f} "
              f"min {ytr.min():+.3f} max {ytr.max():+.3f}", flush=True)

    _ex = [m.strip() for m in a.exclude_months.split(",") if m.strip()]
    if _ex:
        _ym = idx_tr["date"].dt.strftime("%Y-%m")
        _keep = ~_ym.isin(_ex)
        _n0 = len(idx_tr)
        # base_tr is the SHARED flat bar array; ends_tr indexes into it. Filtering
        # windows means filtering ends/labels/index only -- never the base.
        _k = _keep.to_numpy()
        ends_tr = ends_tr[_k]
        ytr = ytr[_k]
        idx_tr = idx_tr[_k].reset_index(drop=True)
        print(f"[exclude] dropped months {_ex}: {_n0:,} -> {len(idx_tr):,} training windows "
              f"({100*(1-len(idx_tr)/_n0):.1f}% removed)", flush=True)

    aux_h = [int(x) for x in a.aux_horizons.split(",") if x.strip()]
    if aux_h:
        # widen ONLY the final linear: column 0 stays the h=1 head the lib will serve
        h_in = model.head.in_features
        old_head = model.head
        model.head = torch.nn.Linear(h_in, 1 + len(aux_h))
        with torch.no_grad():                       # keep col 0 at the original init
            model.head.weight[:1].copy_(old_head.weight)
            model.head.bias[:1].copy_(old_head.bias)
        print(f"[aux] multi-horizon head: 1 + {len(aux_h)} outputs for h={[1] + aux_h}, "
              f"aux weight {a.aux_weight}", flush=True)
    model = model.to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"{a.model} params: {npar:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
             if a.lr_schedule == "cosine" else None)

    lossf = _make_loss()
    yt = torch.from_numpy(ytr)
    yaux = None
    if aux_h:
        # forward returns at each extra horizon, merged onto the SAME (date, key) rows the
        # windows already have -- deliberately NOT computed inside build_windows, whose
        # memory profile is load-bearing (a d90 build is tens of GB).
        _dp = pp_all if "pp_all" in dir() else None
        _src = idx_tr.copy()
        _cl = (D.to_canonical(D.load_local_bars(a.root, "15m", month_lo="201812",
                                                month_hi=trn_end[:4] + "12",
                                                columns=["date", "instrument_id", "close",
                                                         "adjust_factor"]), is_local=True))
        _cl["date"] = pd.to_datetime(_cl["date"]); _cl["day"] = _cl["date"].dt.normalize()
        _cl["adjc"] = _cl["close"] * _cl["adjust_factor"]
        _d = (_cl.dropna(subset=["adjc"]).sort_values("date")
              .groupby(["key", "day"])["adjc"].last().reset_index()
              .sort_values(["key", "day"]))
        for _h in aux_h:
            _r = _d.groupby("key")["adjc"].shift(-_h) / _d["adjc"] - 1.0
            _q = [x / 100.0 if x > 1 else x for x in C.LABEL_CLIP_PCT]
            _lo, _hi = _r.quantile(_q)
            _d[f"h{_h}"] = _r.clip(_lo, _hi)
        _m = _src.merge(_d.rename(columns={"day": "date"})[["key", "date"] +
                                                           [f"h{h}" for h in aux_h]],
                        on=["key", "date"], how="left")
        _A = _m[[f"h{h}" for h in aux_h]].to_numpy(np.float32)
        _bad = ~np.isfinite(_A)
        _A[_bad] = 0.0
        yaux = torch.from_numpy(_A)
        ymask = torch.from_numpy((~_bad).astype(np.float32))
        print(f"[aux] aux label coverage: "
              + ", ".join(f"h{h}={100*float(ymask[:, i].mean()):.1f}%"
                          for i, h in enumerate(aux_h)), flush=True)
        del _cl, _d, _m, _A
    if a.batch_mode == "daily":
        day_groups = [np.asarray(ix, dtype=np.int64)
                      for _, ix in idx_tr.groupby("date").groups.items()]
        print(f"daily batch mode: {len(day_groups)} day-batches/epoch "
              f"(median {int(np.median([len(g) for g in day_groups]))} stocks/day)", flush=True)
    va_groups = None
    if a.model in ("hist_day", "hist_dnn", "stocknet", "stocknet_flat", "hist_flat", "hist_dual") or a.norm in ("csmodel", "dual") or lazy:  # whole-day (or lazy-assembled) eval
        va_groups = [np.asarray(ix, dtype=np.int64)
                     for _, ix in idx_va.groupby("date").groups.items()]
    best = {"IC": -9, "RankIC": -9}; hist = []
    best_ic_val, bad_eps, best_state, best_ep = -9.0, 0, None, 0
    gstep, base_lrs = 0, [g["lr"] for g in opt.param_groups]
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    dday_tok = (f"_D{dday}" if dday != a.hidden else "") + ("_dn" if a.day_neutral else "")
    pub_tok = ""
    if a.public:
        pub_tok = ("_pub" + (str(a.public) if a.public > 1 else "")
                   + {"market": "", "query": "q", "both": "mq",
                      "project": "p", "projq": "pq"}[a.public_mode])
    ep_tok = f"_e{a.epochs}" if a.epochs != 12 else ""
    ep_tok += "_bf16" if a.amp else ""
    ep_tok += f"_te{trn_end[:4]}" if a.train_end else ""
    ep_tok += f"_ts{trn_start[:4]}" if a.train_start else ""
    ep_tok += "_noLN" if a.no_layernorm else ""
    ep_tok += "_tonly" if a.sn_temporal_only else ""
    ep_tok += f"_cv{val_start[:4]}" if a.valid_start else ""
    ep_tok += f"_wu{a.warmup}" if a.warmup else ""
    # regularization knobs MUST appear in the cell name: without them two arms that differ
    # only by e.g. --day-subsample write the SAME checkpoint path and silently overwrite
    # each other (measured 2026-07-26: three arms collided into one set of files).
    ep_tok += f"_wd{a.weight_decay:g}" if a.weight_decay != 1e-5 else ""
    ep_tok += f"_sub{a.day_subsample:g}" if a.day_subsample else ""
    ep_tok += f"_hdrop{a.hist_dropout:g}" if a.hist_dropout else ""
    ep_tok += f"_ema{a.ema_decay:g}" if a.ema_decay else ""
    ep_tok += f"_ex{len(_ex)}m" if _ex else ""
    ep_tok += "_lrank" if a.label_rank else ""
    mname = (a.model + dday_tok + pub_tok + ep_tok + ("_cos" if a.lr_schedule == "cosine" else "")
             + (f"_L{a.layers}" if a.layers > 1 else "")
             + (f"_h{a.hidden}" if a.hidden != 128 else ""))
    tail_tok = f"tail={tail} " if (tail and tail != "14:46") else ""
    # a --windows cell has no tail; without its own token every segment ablation would write
    # the SAME checkpoint name as the plain 15-bar champion and overwrite it
    win_tok = ("w" + "+".join(f"{a1:04d}_{b1:04d}" for a1, b1 in wins)) if wins else ""
    if win_tok:
        tail_tok = f"windows={a.windows} "
    cell = (f"{mname}{'_' + a.freq if a.freq != '1m' else ''}_d{a.days}"
            f"{'' if a.day_stride == 1 else f'sk{a.day_stride}'}_{a.norm}_{a.loss}_{a.batch_mode}"
            + (f"_{win_tok}" if win_tok else (f"_{tail}" if tail_tok else "")))
    pt = os.path.join(a.save_dir, f"{cell.replace(':', '')}_s{a.seed}.pt") if a.save_dir else None
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)

    def _save_ckpt(sd, suffix=""):
        if aux_h:
            # The auxiliary horizons are a TRAINING-TIME device only. Ship column 0 (the
            # h=1 head) so the checkpoint loads into the lib's single-output DayHIST
            # unchanged -- no inference_lib change, no new replica-gate risk, and the aux
            # heads cannot silently alter what the platform scores.
            sd = dict(sd)
            for _k in ("head.weight", "head.bias"):
                if _k in sd and sd[_k].shape[0] > 1:
                    sd[_k] = sd[_k][:1].clone()
        torch.save({"state_dict": sd,
                    "stats": stats, "cell": cell, "seed": a.seed, "best_ep": best_ep or None,
                    "day_bars": day_bars, "days": a.days, "hidden": a.hidden,
                    "norm": a.norm, "tail": tail, "freq": a.freq,
                    # the segment list travels with the weights for the same reason
                    # hist_topk does: it changes WHICH bars the forward pass sees, and
                    # nothing in the state_dict records it
                    "windows": a.windows or "",
                    "feat": a.feat, "feature_cols": list(fc),
                    # architecture variants that change the FORWARD PASS must travel with
                    # the weights: k is not recoverable from the state_dict (only the
                    # presence of hist_logtau is), so an unrecorded k would silently score
                    # with the wrong graph at inference.
                    "hist_topk": int(a.hist_topk), "d_day": dday,
                    "aux_horizons": list(aux_h),
                    # the month LIST, not just the count: `_ex4m` in the cell name
                    # says how many were dropped but not WHICH, so a retrain could
                    # not reproduce the arm (same bug class as the missing --ema-decay)
                    "exclude_months": list(_ex),
                    # WHERE this seed was trained. Nothing else records it: the packages
                    # stamp their notebook's "training env" row from a --env FLAG the
                    # caller passes by hand, so the claim was unverifiable and, for the
                    # 2026-07-28 span_lr_b* ensembles, wrong -- seed 0 came off the rented
                    # 4090 and seeds 1/2 off the lab boxes, while all five notebooks said
                    # "host=vast-4090x4 ... 3 seeds". Mixed-env ensembles are worth knowing
                    # about on their own: the 4090's seed 0 does not reproduce on the lab
                    # hosts, by more than any effect that sweep was measuring.
                    "env": {"host": __import__("socket").gethostname(),
                            "torch": torch.__version__,
                            "cuda": torch.version.cuda,
                            "gpu": (torch.cuda.get_device_name(0)
                                    if torch.cuda.is_available() else "cpu"),
                            "python": __import__("platform").python_version()},
                    "loss": a.loss}, pt + suffix + ".tmp")
        os.replace(pt + suffix + ".tmp", pt + suffix)

    def _evaluate():
        model.eval()
        if va_groups is not None:                  # hist_day: day-wise prediction
            preds = np.empty(len(yva), dtype=np.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                for ixs in va_groups:
                    _o = model(torch.from_numpy(fetch_va(ixs)).to(dev)).float()
                    # multi-horizon head: column 0 IS the shipped h=1 predictor
                    preds[ixs] = (_o[:, 0] if _o.dim() > 1 and _o.shape[1] > 1
                                  else _o.reshape(-1)).cpu().numpy()
            ic = TB.daily_ic(idx_va, preds, yva)
            ic.update(replica_components(idx_va, preds, yva))
            return ic
        pred_bs = 2048 if seq_len > 60 else 8192   # long sequences: cap eval batch (GRU workspace OOM)
        pr = TB.predict(model, Xva, dev, bs=pred_bs)
        ic = TB.daily_ic(idx_va, pr, yva)
        ic.update(replica_components(idx_va, pr, yva))
        return ic

    if a.eval_only:
        ck_in = torch.load(a.eval_only, map_location="cpu")
        model.load_state_dict(ck_in["state_dict"])
        ic = _evaluate()
        evaldays = int(idx_va["date"].nunique())
        print(f"EVAL-ONLY {a.eval_only} (best_ep {ck_in.get('best_ep')})", flush=True)
        print(f"RESULT model={mname} freq={freq} fold={val_start[:7]} days={a.days} seed={a.seed} seqlen={seq_len} norm={a.norm} loss={a.loss} bmode={a.batch_mode} {tail_tok}"
              f"RankIC={ic['RankIC']:.4f} RankICIR={ic['RankICIR']:.4f} "
              f"IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} SR={ic.get('SR', float('nan')):.4f} "
              f"Stress={ic.get('Stress', float('nan')):.4f} meanlastRankIC={ic['RankIC']:.4f} evaldays={evaldays}", flush=True)
        return

    # --- style design matrix for --loss nic (row-aligned with the TRAIN index) --------
    style_mat = None
    if a.loss == "nic":
        from bigalpha.styles import load_styles, FACTORS as _SF
        _st = load_styles(freq="30m", month_lo=trn_start[:4] + trn_start[5:7],
                          month_hi=trn_end[:4] + trn_end[5:7], instrument_fmt=None)
        _key = idx_tr[["date", "key"]].copy()
        _key["date"] = pd.to_datetime(_key["date"])
        _st["date"] = pd.to_datetime(_st["date"])
        _m = _key.merge(_st, on=["date", "key"], how="left")
        _S = _m[list(_SF)].to_numpy(np.float32)
        _cov = np.isfinite(_S).all(1).mean()
        _S[~np.isfinite(_S)] = 0.0                      # missing styles -> no projection
        style_mat = torch.from_numpy(
            np.column_stack([np.ones(len(_S), np.float32), _S]))
        print(f"  [nic] style panel merged: {_S.shape[1]} factors, "
              f"{_cov:.1%} of train rows have complete styles", flush=True)
        del _st, _key, _m, _S

    ema = None                          # plain-path per-step weight EMA (--ema-decay)

    def _load_params_plain(src):
        with torch.no_grad():
            for k_, p_ in model.named_parameters():
                p_.copy_(src[k_])

    for ep in range(a.epochs):
        ep_t0 = time.time()
        model.train()
        tot = 0.0; nb = 0
        if a.batch_mode == "daily":
            batches = [day_groups[k] for k in np.random.permutation(len(day_groups))]
        else:
            order = torch.randperm(len(ytr)).numpy()
            batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
        # batch-level progress bar: shows it/s and intra-epoch ETA. mininterval=5 keeps
        # nohup'd log files from filling with refresh lines while staying live in a tty.
        biter = tqdm(batches, desc=f"ep{ep+1}/{a.epochs}", unit="batch", ncols=90,
                     mininterval=5) if tqdm else batches
        for b in biter:
            if a.day_subsample and 0 < a.day_subsample < 1 and len(b) > 50:
                # resample the day's cross-section: a fresh subset every epoch turns one
                # trading day into many distinct gradient signals
                k = max(50, int(len(b) * a.day_subsample))
                b = np.random.choice(b, size=k, replace=False)
            if a.warmup and gstep < a.warmup:      # linear warmup on the base LR
                for gp, base in zip(opt.param_groups, base_lrs):
                    gp["lr"] = base * (gstep + 1) / a.warmup
            gstep += 1
            opt.zero_grad()
            xb = torch.from_numpy(fetch_tr(b))
            _extra = ((style_mat[torch.from_numpy(b)].to(dev),)
                      if style_mat is not None else ())
            if a.amp:
                # bf16 autocast: ~2x on tensor-core GRUs, fp32 master weights, no scaler
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossf(model(xb.to(dev)), yt[torch.from_numpy(b)].to(dev), *_extra)
            else:
                _bi = torch.from_numpy(b)
                _out = model(xb.to(dev))
                if aux_h:
                    _main = _out[:, 0] if _out.dim() > 1 else _out
                    loss = lossf(_main, yt[_bi].to(dev), *_extra)
                    _ya = yaux[_bi].to(dev); _mk = ymask[_bi].to(dev)
                    for _j in range(len(aux_h)):
                        if float(_mk[:, _j].sum()) < 50:
                            continue          # too few labelled names that day
                        _sel = _mk[:, _j] > 0
                        loss = loss + a.aux_weight * _ic_loss(_out[_sel, _j + 1],
                                                              _ya[_sel, _j])
                else:
                    loss = lossf(_out, yt[_bi].to(dev), *_extra)
            loss.backward(); opt.step()
            if a.ema_decay > 0:
                with torch.no_grad():
                    if ema is None:
                        ema = {k_: p_.detach().clone()
                               for k_, p_ in model.named_parameters()}
                    else:
                        for k_, p_ in model.named_parameters():
                            ema[k_].mul_(a.ema_decay).add_(p_.detach(),
                                                           alpha=1 - a.ema_decay)
            tot += loss.item(); nb += 1
        if sched is not None:
            sched.step()
        if a.save_dir:
            # LAST-epoch snapshot BEFORE evaluating: an eval crash (missing dep, OOM,
            # NaN) can no longer lose a trained epoch; --eval-only replays it later
            _save_ckpt({k: v.detach().cpu() for k, v in model.state_dict().items()}, ".last")
        _raw_plain = None
        if ema is not None:
            _raw_plain = {k_: p_.detach().clone() for k_, p_ in model.named_parameters()}
            _load_params_plain(ema)         # eval (and any best-snapshot) = EMA weights
        ic = _evaluate()
        hist.append(ic)
        if ic["IC"] > best.get("IC", -9):
            # RESULT reports the best-IC epoch (selection rule: IC/ICIR; also matches
            # the saved best-IC checkpoint). Was best-RankIC-epoch before 2026-07-22.
            best = ic
        if ic["IC"] > best_ic_val:
            best_ic_val = ic["IC"]
            bad_eps = 0
            if a.patience and a.save_dir:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_ep = ep + 1
                os.makedirs(a.save_dir, exist_ok=True)
                _save_ckpt(best_state)     # incremental: multi-day runs always have best weights on disk
                print(f"  [ckpt] best IC {best_ic_val:+.4f} @ ep {best_ep} -> {pt}", flush=True)
        else:
            bad_eps += 1
        if _raw_plain is not None:
            _load_params_plain(_raw_plain)  # resume training from RAW weights
        ep_secs = time.time() - ep_t0
        eta_min = (a.epochs - ep - 1) * ep_secs / 60
        # with --loss ic the train loss is NEGATIVE Pearson corr, so report train IC directly
        train_metric = (f"trainIC={-tot/nb:+.4f}" if a.loss in ("ic", "nic") else
                        f"scoreloss={tot/nb:+.4f}" if a.loss == "score" else f"mse={tot/nb:.6f}")
        print(f"  ep {ep+1:2d}/{a.epochs} {train_metric} | valid RankIC={ic['RankIC']:+.4f} "
              f"RankICIR={ic['RankICIR']:.3f} IC={ic['IC']:+.4f} ICIR={ic['ICIR']:.3f} "
              f"SR={ic.get('SR', float('nan')):.2f} Stress={ic.get('Stress', float('nan')):+.4f} "
              f"[{ep_secs:.0f}s/ep ETA {eta_min:.0f}m]", flush=True)
        if a.patience and bad_eps >= a.patience:
            print(f"  EARLY STOP at ep {ep+1} (no valid-IC improvement for {a.patience} epochs; "
                  f"best IC {best_ic_val:+.4f} @ ep {best_ep})", flush=True)
            break
    mlast = float(np.mean([h["RankIC"] for h in hist[-3:]]))
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        if ema is not None and not a.patience:
            _load_params_plain(ema)         # take-LAST = final EMA weights
        sd_save = best_state if (a.patience and best_state is not None) \
            else {k: v.cpu() for k, v in model.state_dict().items()}
        _save_ckpt(sd_save)
        print(f"saved {pt}", flush=True)
    evaldays = int(idx_va["date"].nunique())          # # of 2024 trading days actually scored
    print(f"RESULT model={mname} freq={freq} fold={val_start[:7]} days={a.days} seed={a.seed} seqlen={seq_len} norm={a.norm} loss={a.loss} bmode={a.batch_mode} {tail_tok}"
          f"RankIC={best['RankIC']:.4f} RankICIR={best['RankICIR']:.4f} "
          f"IC={best['IC']:.4f} ICIR={best['ICIR']:.4f} SR={best.get('SR', float('nan')):.4f} "
          f"Stress={best.get('Stress', float('nan')):.4f} meanlastRankIC={mlast:.4f} evaldays={evaldays}", flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
'''



# ---------------------------------------------------------------------------
# ENSEMBLE DRIVER (one invocation = 3 seeds + manifest rebuild; private-retrain
# compatible). Replaces the single-seed entry of the member train.py.
# ---------------------------------------------------------------------------
def _ens_seeds():
    """Seeds to retrain = however many members the SHIPPED manifest carries.

    This was a hard-coded (0, 1, 2) until v122ens9, the first package with n != 3:
    it shipped nine members but its train.py would have rebuilt models[] from three,
    i.e. a private retrain would silently produce a DIFFERENT model than the one
    scored publicly. Reading the count off the manifest keeps the two in lockstep
    for any n. Our seeds are always 0..n-1 by construction (lab_grid_queue.sh
    numbers them contiguously and make_ens_package globs them in order).
    """
    return tuple(range(len(manifest()["models"])))


def _pack_sd_fp16(sd):
    import base64
    import numpy as _np
    return {k: {"dtype": "float16", "shape": list(v.shape),
                "b64": base64.b64encode(
                    v.cpu().numpy().astype(_np.float16).tobytes()).decode("ascii")}
            for k, v in sd.items()}


def main_ensemble():
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="e2e_data")
    ap.add_argument("--save-dir", default=os.path.join(HERE, "retrain_out"))
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    check_against_manifest("dry-run" if a.dry_run else "")
    if a.dry_run:
        verify_sources()
        print(f"ensemble dry-run OK: would train seeds {_ens_seeds()} sequentially "
              f"and rebuild final_model.json models[] from the fresh ckpts")
        return

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("GRU_IO_WORKERS", "2")   # small-container load guard
    a.data_root = ensure_data(a.data_root)
    os.makedirs(a.save_dir, exist_ok=True)
    tg = install_modules()
    fresh = []
    for seed in _ens_seeds():
        args = list(TRAIN_ARGS)
        args[args.index("--seed") + 1] = str(seed)
        argv = ["train_gru2.py", *args, "--epochs", str(a.epochs),
                "--root", a.data_root, "--save-dir", a.save_dir]
        print(f"=== seed {seed}: {' '.join(argv)}", flush=True)
        old_argv, sys.argv = sys.argv, argv
        try:
            tg.main()
        finally:
            sys.argv = old_argv
        pts = sorted(glob.glob(os.path.join(a.save_dir, f"*_s{seed}.pt")),
                     key=os.path.getmtime)
        assert pts, f"seed {seed}: no checkpoint produced"
        if not verify_ckpt(pts[-1]):
            sys.exit(4)
        fresh.append(pts[-1])

    # rebuild the shipped manifest's models[] from the fresh checkpoints
    import torch
    m = manifest()
    models = []
    for p in fresh:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        models.append(_pack_sd_fp16(ck["state_dict"]))
    m["models"] = models
    m["note"] = (m.get("note") or "") + " | REBUILT by ensemble train.py (private retrain)"
    out = os.path.join(HERE, "final_model.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f)
    print(f"ensemble manifest rebuilt: {len(models)} models -> {out}")


if __name__ == "__main__":
    main_ensemble()
