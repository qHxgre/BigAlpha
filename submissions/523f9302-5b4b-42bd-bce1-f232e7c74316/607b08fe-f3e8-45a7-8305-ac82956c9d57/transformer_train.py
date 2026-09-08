"""BigAlpha 2026 自包含训练/推理脚本；运行后生成 transformer_model.json。

双频架构 (15m 内生 + 30m 外生 concat)，约 220 万参数。
单文件自包含：配置、数据、标签、模型、损失、训练、推理、JSON 导出全在此。
"""

# ============================================================
# §1 全局配置
# ============================================================

import random

import numpy as np

# 字段清单依据平台实际 schema (bar15m / bar30m, 盘口 5 档,
# 无 num_trades/avg_price/total_volume 系字段; 成交笔数为 deal_number)。
PRICE_COLS = ["open", "high", "low", "close", "pre_close",
              "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",
              "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5"]
VOL_COLS = ["volume", "amount", "deal_number",
            "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",
            "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5"]

CONFIG = {
    "table": "bigalpha_2026_stock_bar15m",          # 高频内生表 (16 bar/天)
    "lowfreq_table": "bigalpha_2026_stock_bar30m",  # 低频外生表 (8 bar/天)
    "lowfreq_mode": "concat",  # "concat"—双频 bar 拼接走单一 IntradayEncoder
    "instruments_table": "bigalpha_2026_instruments",
    "exposure_table": "bigalpha_2026_exposure",
    "bars_per_day": 16,          # 高频 (15m) 日内 bar 数
    "lowfreq_bars_per_day": 8,   # 低频 (30m) 日内 bar 数
    "price_cols": PRICE_COLS,
    "vol_cols": VOL_COLS,
    "feature_cols": PRICE_COLS + VOL_COLS,
    # 复权因子仅用于标签收益计算 (close*adjust_factor), 不作为模型输入字段
    "adjust_col": "adjust_factor",
    "lookback_days": 60,
    "infer_buffer_natural_days": 130,
    "chunk_size": 250,  # 每块股票数; 内存紧张(<32G)时降回 100
    "train_frac": 0.8,
    "seed": 456,
    "tag": "submission",
    "cache_path": "train_cache.npz",
    "cache_path_low": "train_cache_low.npz",
    "checkpoint_path": "checkpoint.pt",
    "resume_checkpoint_path": "last_checkpoint.pt",
    # 官方提交的训练产物必须为 JSON；checkpoint.pt 仅用于本地断点/最佳权重。
    "artifact_path": "transformer_model.json",
    "label": {"mode": "residual", "horizon": 1, "winsor_pct": 1.0},
    "model": {"d_intra": 128, "heads_intra": 8, "layers_intra": 4, "ffn_intra": 512,
              "intra_chunk_size": 8192,
              "d_day": 256, "tau_cross": 8, "heads_cross": 4, "ffn_cross": 1024,
              "use_cross_attn": True, "gate_init": -5.0, "glu_bottleneck": 128,
              "dropout": 0.1,
              "n_global_tokens": 4},
    "loss": {"mse_w": 0.01, "listnet_w": 0.1, "listnet_temp": 1.0,
             "listmle_w": 0.1, "listmle_temp": 1.0,
             "min_pred_std": 0.05, "std_w": 1.0},
    "train": {"epochs": 40, "lr": 5e-4, "weight_decay_head": 0.01, "clip": 1.0,
              "patience": 10, "swa_frac": 0.9, "stock_sample_ratio": 1.0,
              "log_every": 5, "checkpoint_every": 100,
              #"time_budget_min": 160,  # 2h40m, 留 20min 余量应对平台 3h 硬限制
              "metric_w": {"ic": 1.0, "icir": 0.02, "sharpe": 0.02}},
}

PARAM_MIN, PARAM_MAX = 100_000, 100_000_000


def validate_config(cfg):
    assert len(cfg["feature_cols"]) <= 100, "字段数超过赛规上限 100"
    assert len(cfg["feature_cols"]) == len(set(cfg["feature_cols"])), "字段重复"
    assert cfg["lookback_days"] <= 240, "回看窗口超过赛规上限 240 交易日"
    assert set(cfg["price_cols"]).isdisjoint(cfg["vol_cols"])
    assert cfg.get("resume_checkpoint_path", "last_checkpoint.pt") != \
        cfg["checkpoint_path"], "断点文件与 best model 必须使用不同路径"


def assert_param_count(n):
    assert PARAM_MIN <= n <= PARAM_MAX, \
        f"参数量 {n} 超出赛规区间 [{PARAM_MIN}, {PARAM_MAX}]"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================
# §2 数据层
# ============================================================

import numpy as np
import pandas as pd


class RunningStats:
    """流式 mean/std（官方 OOM 指引方案），与全量一次计算数值等价。"""

    def __init__(self, n_feat):
        self.n = 0
        self.s = np.zeros(n_feat, np.float64)
        self.ss = np.zeros(n_feat, np.float64)

    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x ** 2).sum(0)

    def finalize(self):
        mean = self.s / self.n
        var = self.ss / self.n - mean ** 2
        std = np.sqrt(np.clip(var, 0, None)) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)


def apply_field_transforms(df, cfg):
    """按字段统一变换（赛规允许类）：价格 log、量 log1p。"""
    out = df.copy()
    for c in cfg["price_cols"]:
        if c in out:
            out[c] = np.log(out[c].clip(lower=1e-6))
    for c in cfg["vol_cols"]:
        if c in out:
            out[c] = np.log1p(out[c].clip(lower=0))
    return out


class Normalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def apply(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)


def pivot_to_days(df, feature_cols, bars_per_day, close_col="close"):
    """单只股票的 bar 级 df (按 date 升序) -> (dates (D,), X (D,B,F), day_close (D,))。
    向量化实现: 整只股票一次 ffill/bfill (缺失值填充), 按日期边界 numpy 切片;
    缺 bar 的天以当日首 bar 左侧补齐, 多余 bar 取最后 bars_per_day 根。"""
    n_feat = len(feature_cols)
    empty = (np.array([], "datetime64[D]"),
             np.zeros((0, bars_per_day, n_feat), np.float32),
             np.array([], np.float64))
    if len(df) == 0:
        return empty
    feats = df[feature_cols].to_numpy(np.float32)
    feats = pd.DataFrame(feats).ffill().bfill().to_numpy(np.float32)
    closes_all = df[close_col].to_numpy(np.float64)
    day = df["date"].dt.normalize().to_numpy().astype("datetime64[D]")
    starts = np.flatnonzero(np.concatenate([[True], day[1:] != day[:-1]]))
    ends = np.append(starts[1:], len(day))
    counts = ends - starts

    # 快速路径: 所有天都是标准 bar 数且无缺失
    if np.all(counts == bars_per_day) and np.isfinite(feats).all():
        return (day[starts],
                feats.reshape(-1, bars_per_day, n_feat),
                closes_all[ends - 1])

    dates, arrs, closes = [], [], []
    for s, e in zip(starts, ends):
        a = feats[s:e]
        if not np.isfinite(a).all():
            continue
        if len(a) >= bars_per_day:
            a = a[-bars_per_day:]
        else:
            a = np.concatenate([np.repeat(a[:1], bars_per_day - len(a), axis=0), a])
        dates.append(day[s])
        arrs.append(a)
        closes.append(closes_all[e - 1])
    if not dates:
        return empty
    return np.array(dates), np.stack(arrs), np.array(closes, np.float64)


def iter_chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def list_instruments(query_fn, table, start, end):
    df = query_fn(f"SELECT DISTINCT instrument FROM {table}",
                  {"date": [start, end]})
    return sorted(df["instrument"].tolist())


def load_stock_arrays(query_fn, table, cfg, start, end, instruments,
                     bars_per_day=None):
    """分块查询 -> 合规变换 -> 按日透视。每块用完即释放（官方 OOM 指引）。
    day_close 用于标签: 若表提供复权因子 (adjust_col), 用 close*adjust_factor
    计算复权收益, 避免分红送转污染标签; 复权因子不进入模型输入。"""
    import time as _time
    bpd = bars_per_day if bars_per_day is not None else cfg["bars_per_day"]
    adj = cfg.get("adjust_col")
    extra = [adj] if adj and adj not in cfg["feature_cols"] else []
    cols = ", ".join(["date", "instrument"] + cfg["feature_cols"] + extra)
    sql = f"SELECT {cols} FROM {table} ORDER BY instrument, date"
    out = {}
    chunks = list(iter_chunks(instruments, cfg["chunk_size"]))
    t_start = _time.time()
    for ci, chunk in enumerate(chunks):
        t_c = _time.time()
        df = query_fn(sql, {"date": [start, end], "instrument": list(chunk)})
        n_rows = 0 if df is None else len(df)
        elapsed = _time.time() - t_start
        eta = elapsed / (ci + 1) * (len(chunks) - ci - 1)
        table_tag = table.rsplit("_", 1)[-1] if "_" in table else table
        print(f"[data:{table_tag}] chunk {ci + 1}/{len(chunks)} rows={n_rows} "
              f"query={_time.time() - t_c:.0f}s elapsed={elapsed:.0f}s "
              f"eta={eta:.0f}s", flush=True)
        if df is None or len(df) == 0:
            continue
        t_p = _time.time()
        if adj and adj in df.columns:
            raw_close = df["close"].astype(np.float64) * df[adj].astype(np.float64)
        else:
            raw_close = df["close"].copy()
        df = apply_field_transforms(df, cfg)
        df["_raw_close"] = raw_close
        for ins, sub in df.groupby("instrument", sort=False):
            dates, X, close = pivot_to_days(sub, cfg["feature_cols"],
                                            bpd, close_col="_raw_close")
            if len(dates):
                out[str(ins)] = (dates, X.astype(np.float16), close)
        print(f"[data:{table_tag}] chunk {ci + 1}/{len(chunks)} processed "
              f"proc={_time.time() - t_p:.0f}s stocks_total={len(out)}", flush=True)
        del df
    return out


def save_cache(path, arrays):
    import time as _time
    t0 = _time.time()
    print(f"[cache] saving {len(arrays)} stocks -> {path} ...", flush=True)
    flat = {}
    for ins, (dates, X, close) in arrays.items():
        flat[f"{ins}::dates"] = dates
        flat[f"{ins}::X"] = X
        flat[f"{ins}::close"] = close
    # 不压缩: float16 行情数据压缩比低, savez_compressed 单线程要数分钟。
    # 先写临时文件再原子改名: 中断不会在目标路径留下残缺文件。
    import os as _os
    tmp = path + ".tmp.npz"
    np.savez(tmp, **flat)
    _os.replace(tmp, path)
    print(f"[cache] saved in {_time.time() - t0:.0f}s", flush=True)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    names = sorted({k.split("::")[0] for k in z.files})
    return {n: (z[f"{n}::dates"], z[f"{n}::X"], z[f"{n}::close"]) for n in names}


def load_multi_freq_arrays(query_fn, cfg, start, end, instruments):
    """加载双频数据（高频内生 + 低频外生），分别缓存。

    Returns:
        arrays_high: {ins: (dates, X_high, close)}
        arrays_low:  {ins: (dates, X_low, close)}  (共享同一 close)
    """
    import os as _os

    # ── 高频 (15m, 内生) ──
    arrays_high = None
    if _os.path.exists(cfg["cache_path"]):
        try:
            arrays_high = load_cache(cfg["cache_path"])
            print(f"[data] cache hit: {cfg['cache_path']}")
        except Exception as e:
            print(f"[data] cache corrupt ({e!r}), rebuilding ...", flush=True)
            arrays_high = None
    if arrays_high is None:
        arrays_high = load_stock_arrays(
            query_fn, cfg["table"], cfg, start, end, instruments,
            bars_per_day=cfg["bars_per_day"])
        save_cache(cfg["cache_path"], arrays_high)

    # ── 低频 (30m, 外生) ──
    arrays_low = None
    low_table = cfg.get("lowfreq_table")
    if low_table is None:
        return arrays_high, None

    cache_low = cfg["cache_path_low"]
    if _os.path.exists(cache_low):
        try:
            arrays_low = load_cache(cache_low)
            print(f"[data] cache hit: {cache_low}")
        except Exception as e:
            print(f"[data] cache corrupt ({e!r}), rebuilding ...", flush=True)
            arrays_low = None
    if arrays_low is None:
        arrays_low = load_stock_arrays(
            query_fn, low_table, cfg, start, end, instruments,
            bars_per_day=cfg["lowfreq_bars_per_day"])
        save_cache(cache_low, arrays_low)

    return arrays_high, arrays_low


# ============================================================
# §3 标签
# ============================================================

import numpy as np
import pandas as pd


_EXPOSURE_KEY_COLS = {"date", "instrument"}
_EXPOSURE_TARGET_COLS = {"ret", "return", "returns", "label", "target"}
_LABEL_COLUMNS = ["date", "instrument", "label", "residual_return"]


def forward_return(day_close, horizon):
    r = np.full(len(day_close), np.nan)
    if len(day_close) > horizon:
        base = day_close[:-horizon]
        r[:-horizon] = day_close[horizon:] / np.where(base > 0, base, np.nan) - 1.0
    return r


def residualize_day(r, X):
    A = np.column_stack([np.ones(len(r)), X])
    beta, *_ = np.linalg.lstsq(A, r, rcond=None)
    return r - A @ beta


def winsorize_zscore(x, pct):
    lo, hi = np.nanpercentile(x, [pct, 100 - pct])
    x = np.clip(x, lo, hi)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def build_exposure_lookup(exposure_df):
    """将 exposure DataFrame 转为按日索引的快速查找表。

    exposure_df: [date, instrument, f1..fK]（与 build_label_panel 输入一致）。
    返回 {np.datetime64[D]: {instrument_str: np.array([f1..fK], float64)}}
    用于验证时对预测值做 BARRA 残差化，对齐平台评估口径。
    """
    from pandas.api import types as pd_types

    if exposure_df is None:
        return None
    df = exposure_df.copy()
    missing = _EXPOSURE_KEY_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"exposure_df missing key columns: {sorted(missing)}")
    fac_cols = [
        c for c in df.columns
        if c.lower() not in _EXPOSURE_KEY_COLS | _EXPOSURE_TARGET_COLS
        and pd_types.is_numeric_dtype(df[c])
    ]
    if not fac_cols:
        return None
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
    lookup = {}
    # pandas 内部日期一律 datetime64[ns], groupby key 恒为 Timestamp;
    # 显式转 np.datetime64[D], 与 DaySampleIndex 的交易日 key 同型同单位,
    # 否则验证时 .get(d) 哈希不同永远 miss
    for d, g in df.groupby("date"):
        key = pd.Timestamp(d).to_numpy().astype("datetime64[D]")
        ins = g["instrument"].values
        X = g[fac_cols].to_numpy(np.float64)
        lookup[key] = dict(zip(ins, X))
    return lookup


def build_label_panel(close_map, exposure_df, cfg, universe_df=None):
    """close_map: {instrument: (dates, day_close)}; exposure_df: [date, instrument, f1..fK] 或 None。
    universe_df: 官方每日股票池 [date, instrument]；提供时先过滤再做截面残差化。
    返回 DataFrame[date, instrument, label, residual_return]：
    label 用于训练，residual_return 保留真实收益尺度用于验证 Sharpe。"""
    lab = cfg["label"]
    rows = []
    for ins, (dates, close) in close_map.items():
        r = forward_return(np.asarray(close, np.float64), lab["horizon"])
        for d, v in zip(dates, r):
            if np.isfinite(v):
                rows.append((d, ins, v))
    panel = pd.DataFrame(rows, columns=["date", "instrument", "ret"])
    if panel.empty:
        return pd.DataFrame(columns=_LABEL_COLUMNS)

    if universe_df is not None:
        missing = _EXPOSURE_KEY_COLS.difference(universe_df.columns)
        if missing:
            raise ValueError(f"universe_df missing key columns: {sorted(missing)}")
        universe = universe_df[["date", "instrument"]].copy()
        universe["date"] = pd.to_datetime(universe["date"]).values.astype("datetime64[D]")
        universe["instrument"] = universe["instrument"].astype(str)
        universe = universe.drop_duplicates(["date", "instrument"])
        panel = panel.merge(universe, on=["date", "instrument"], how="inner")
        if panel.empty:
            return pd.DataFrame(columns=_LABEL_COLUMNS)

    use_resid = lab["mode"] == "residual" and exposure_df is not None
    fac_cols = []
    if use_resid:
        exposure_df = exposure_df.copy()
        missing = _EXPOSURE_KEY_COLS.difference(exposure_df.columns)
        if missing:
            raise ValueError(f"exposure_df missing key columns: {sorted(missing)}")
        # SELECT * may also return the table's own return/label columns and
        # categorical metadata. They are not style exposures; including them
        # would either collide with panel["ret"] or leak a target into the OLS.
        fac_cols = [
            c for c in exposure_df.columns
            if c.lower() not in _EXPOSURE_KEY_COLS | _EXPOSURE_TARGET_COLS
            and pd.api.types.is_numeric_dtype(exposure_df[c])
        ]
        exposure_df["date"] = pd.to_datetime(exposure_df["date"]).values.astype("datetime64[D]")
        exposure_df = exposure_df[["date", "instrument", *fac_cols]]
        panel = panel.merge(exposure_df, on=["date", "instrument"], how="left")

    out = []
    for d, g in panel.groupby("date"):
        r = g["ret"].to_numpy(np.float64)
        if use_resid:
            X = g[fac_cols].to_numpy(np.float64)
            ok = np.isfinite(X).all(1) & np.isfinite(r)
            if ok.sum() >= max(30, X.shape[1] + 2):
                r2 = r.copy()
                r2[ok] = residualize_day(r[ok], X[ok])
                r2[~ok] = r[~ok] - np.nanmean(r[ok])  # 无暴露数据的股票退化为去均值
                r = r2
            else:
                r = r - np.nanmean(r)
        else:
            r = r - np.nanmean(r)
        if len(r) < 3:
            continue
        z = winsorize_zscore(r, lab["winsor_pct"])
        out.append(pd.DataFrame({"date": d, "instrument": g["instrument"].values,
                                 "label": z, "residual_return": r}))
    if not out:
        return pd.DataFrame(columns=_LABEL_COLUMNS)
    return pd.concat(out, ignore_index=True)


# ============================================================
# §4 训练集组装
# ============================================================

import numpy as np


def split_days(days, train_frac):
    days = sorted(days)
    k = int(len(days) * train_frac)
    return days[:k], days[k:]


class DaySampleIndex:
    """arrays: {instrument: (dates, X_high (D,B,F) float16, day_close)};
    arrays_low: 可选, 同上结构但 B 不同（低频外生表）。
    labels_df: DataFrame[date, instrument, label, residual_return]。
    batch(day) -> 有低频时 (X, X_low, y, instruments) 否则 (X, y, instruments)。
    保持 float16 到传入 GPU 后再转 float32，减少 CPU 内存复制和 PCIe 传输。"""

    def __init__(self, arrays, labels_df, cfg, arrays_low=None):
        self.arrays = arrays
        self.arrays_low = arrays_low
        self.has_low = arrays_low is not None
        self.L = cfg["lookback_days"]
        self.n_feat = len(cfg["feature_cols"])
        self.bars = cfg["bars_per_day"]
        self.bars_low = cfg.get("lowfreq_bars_per_day", 0) if self.has_low else 0
        self.pos = {ins: {d: i for i, d in enumerate(dates)}
                    for ins, (dates, _, _) in arrays.items()}
        if self.has_low:
            self.pos_low = {ins: {d: i for i, d in enumerate(dates)}
                           for ins, (dates, _, _) in arrays_low.items()
                           if ins in self.pos}
        else:
            self.pos_low = {}
        if "residual_return" not in labels_df.columns:
            raise ValueError("labels_df missing residual_return for validation Sharpe")
        by_day = {}
        columns = ["date", "instrument", "label", "residual_return"]
        for d, ins, v, raw_ret in labels_df[columns].itertuples(index=False):
            d = np.datetime64(d, "D")
            ok = ins in self.pos and d in self.pos[ins]
            if self.has_low:
                ok = ok and ins in self.pos_low and d in self.pos_low[ins]
            if ok:
                by_day.setdefault(d, []).append((ins, float(v), float(raw_ret)))
        # 截面太小的天丢弃（截面损失无意义）
        self.by_day = {d: v for d, v in by_day.items() if len(v) >= 2}
        self.days = sorted(self.by_day)

    def _window(self, ins, d, low=False):
        if low:
            arrs, pos, bars = self.arrays_low, self.pos_low, self.bars_low
        else:
            arrs, pos, bars = self.arrays, self.pos, self.bars
        _, X, _ = arrs[ins]
        i = pos[ins][d]
        lo = max(0, i + 1 - self.L)
        w = X[lo:i + 1]
        if len(w) < self.L:
            pad = np.zeros((self.L - len(w), bars, self.n_feat), dtype=X.dtype)
            w = np.concatenate([pad, w], axis=0)
        return w

    def batch(self, day):
        items = self.by_day[day]
        X = np.stack([self._window(ins, day, low=False) for ins, _, _ in items])
        y = np.array([v for _, v, _ in items], np.float32)
        insts = [ins for ins, _, _ in items]
        if self.has_low:
            X_low = np.stack([self._window(ins, day, low=True) for ins, _, _ in items])
            return X, X_low, y, insts
        return X, y, insts

    def eval_batch(self, day):
        items = self.by_day[day]
        X = np.stack([self._window(ins, day, low=False) for ins, _, _ in items])
        y = np.array([v for _, v, _ in items], np.float32)
        residual_return = np.array([r for _, _, r in items], np.float64)
        insts = [ins for ins, _, _ in items]
        if self.has_low:
            X_low = np.stack([self._window(ins, day, low=True) for ins, _, _ in items])
            return X, X_low, y, residual_return, insts
        return X, y, residual_return, insts


# ============================================================
# §5 模型
# ============================================================

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def alibi_bias(n_heads, length, device):
    slopes = torch.tensor([2 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)],
                          device=device)
    pos = torch.arange(length, device=device)
    dist = (pos[None, :] - pos[:, None]).abs().float()
    return -slopes[:, None, None] * dist[None]


class FieldMix(nn.Module):
    """字段维混合块 (StockMixer): 端到端学跨字段交互, 替代被禁止的人工跨字段算子。"""

    def __init__(self, n_feat):
        super().__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.fc1 = nn.Linear(n_feat, n_feat)
        self.fc2 = nn.Linear(n_feat, n_feat)
        self.act = nn.Hardswish()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class _EncoderLayer(nn.Module):
    def __init__(self, d, heads, ffn, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ffn(self.n2(x)))


class IntradayEncoder(nn.Module):
    """日内 bar 序列 -> 当日向量。ALiBi 距离衰减偏置 (TIPS) + global token 聚合 (TimeXer)。"""

    def __init__(self, n_feat, d, heads, layers, ffn, bars, dropout):
        super().__init__()
        self.proj = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, bars, d))
        self.glb = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList(_EncoderLayer(d, heads, ffn, dropout)
                                    for _ in range(layers))
        self.heads = heads
        self.out_norm = nn.LayerNorm(d)
        bias = torch.zeros(heads, bars + 1, bars + 1)
        bias[:, 1:, 1:] = alibi_bias(heads, bars, "cpu")
        self.register_buffer("attn_bias", bias, persistent=False)

    def forward(self, x):  # (B*, bars, F) -> (B*, d)
        h = self.proj(x) + self.pos
        h = torch.cat([self.glb.expand(h.shape[0], -1, -1), h], dim=1)
        mask = self.attn_bias.repeat(h.shape[0], 1, 1)
        for lyr in self.layers:
            h = lyr(h, mask)
        return self.out_norm(h[:, 0])


class VariateEncoder(nn.Module):
    """低频外生变量编码器 (TimeXer variate-wise):
    每个特征 → 1 个 token (不依赖时序长度), 特征间自注意力。"""

    def __init__(self, n_feat, bars, d, dropout):
        super().__init__()
        self.n_feat = n_feat
        self.bars = bars
        self.proj = nn.Linear(bars, d)       # 每个特征的整条 bar 序列 → d
        self.feat_embed = nn.Parameter(torch.zeros(1, n_feat, d))
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B*, bars, F) -> (B*, F, d)
        h = x.transpose(1, 2)                # (B*, F, bars)
        h = self.proj(h) + self.feat_embed   # (B*, F, d)
        return self.drop(self.norm(h))


class CrossFreqBridge(nn.Module):
    """可学习全局 token 通过 cross-attention 查询外生 feature token (TimeXer 桥接)。"""

    def __init__(self, d, heads, n_global=4, dropout=0.1):
        super().__init__()
        self.global_tokens = nn.Parameter(torch.zeros(1, n_global, d))
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

    def forward(self, feat_tokens):  # (B*, F, d) -> (B*, d)
        B = feat_tokens.shape[0]
        g = self.global_tokens.expand(B, -1, -1)           # (B*, n_global, d)
        enriched, _ = self.cross_attn(query=g, key=feat_tokens, value=feat_tokens)
        enriched = self.norm(g + self.drop(enriched))
        return enriched.mean(1)                            # pool → (B*, d)


class InterDayEncoder(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.gru = nn.GRU(d_in, d, batch_first=True)

    def forward(self, x):  # (N, L, d_in) -> (N, L, d)
        out, _ = self.gru(x)
        return out


class CrossSectionBlock(nn.Module):
    """截面注意力: batch 维=时间位置, 序列维=股票; 无位置编码 (排列不变, iTransformer);
    门控残差近零初始化 (WaveLSFormer): 先学单股基线, 训练中逐步启用截面信息。"""

    def __init__(self, d, heads, ffn, dropout, gate_init):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.gate_attn = nn.Parameter(torch.tensor(float(gate_init)))
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d))
        self.gate_ffn = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, h):  # (tau, N, d) -> (tau, N, d)
        x = self.norm(h)
        a, _ = self.attn(x, x, x, need_weights=False)
        h = h + torch.sigmoid(self.gate_attn) * a
        return h + torch.sigmoid(self.gate_ffn) * self.ffn(self.ffn_norm(h))


class TemporalAggregator(nn.Module):
    """以最新一天为 query 的时序注意力加权 (MASTER/DTML)。
    打分全程 fp32: fp16 下 q·k 内积可溢出为 ±inf, softmax 内 inf-inf → NaN
    (epoch 17 起逐股触发的根因, 由 code/diagnose_nan_layer.py 定位)。"""

    def __init__(self, d):
        super().__init__()
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, h):  # (N, tau, d) -> (N, d)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            w = torch.softmax((self.q(h[:, -1:]) @ self.k(h).transpose(1, 2))
                              * self.scale, -1)
            return (w @ h).squeeze(1)


class RankGLUHead(nn.Module):
    """线性直通 + γ·瓶颈 GLU 门控残差 (RankGLU): 稳定排序几何 + 有界低秩非线性。"""

    def __init__(self, d, b, gamma=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin = nn.Linear(d, 1)
        self.v, self.g = nn.Linear(d, b), nn.Linear(d, b)
        self.out = nn.Linear(b, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, e):
        e = self.norm(e)
        z = self.v(e) * torch.sigmoid(self.g(e))
        return (self.lin(e) + self.gamma * self.out(z)).squeeze(-1)


class StockScorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        n_feat = len(cfg["feature_cols"])
        self.tau = m["tau_cross"]
        self.use_cross = m["use_cross_attn"]
        self.intra_chunk_size = m.get("intra_chunk_size", 4096)
        self.has_lowfreq = cfg.get("lowfreq_table") is not None
        self.lowfreq_mode = cfg.get("lowfreq_mode", "cross_attn") if self.has_lowfreq else None
        d = m["d_intra"]

        if self.has_lowfreq and self.lowfreq_mode == "concat":
            # concat 模式：把两频 bar 拼成一个更宽的日内序列，走单一 IntradayEncoder
            total_bars = cfg["bars_per_day"] + cfg["lowfreq_bars_per_day"]
            self.fieldmix = FieldMix(n_feat)
            self.intra = IntradayEncoder(n_feat, d, m["heads_intra"],
                                         m["layers_intra"], m["ffn_intra"],
                                         total_bars, m["dropout"])
            self.fuse = None  # 不需要融合层
        else:
            self.fieldmix = FieldMix(n_feat)
            self.intra = IntradayEncoder(n_feat, d, m["heads_intra"],
                                         m["layers_intra"], m["ffn_intra"],
                                         cfg["bars_per_day"], m["dropout"])

            if self.has_lowfreq:
                lb = cfg["lowfreq_bars_per_day"]
                self.fieldmix_low = FieldMix(n_feat)
                self.variate = VariateEncoder(n_feat, lb, d, m["dropout"])
                self.bridge = CrossFreqBridge(d, m["heads_intra"],
                                              m.get("n_global_tokens", 4),
                                              m["dropout"])
                self.fuse = nn.Linear(d * 2, d)
            else:
                self.fuse = None

        self.inter = InterDayEncoder(d, m["d_day"])
        self.cross = CrossSectionBlock(m["d_day"], m["heads_cross"], m["ffn_cross"],
                                       m["dropout"], m["gate_init"])
        self.agg = TemporalAggregator(m["d_day"])
        self.head = RankGLUHead(m["d_day"], m["glu_bottleneck"])

    def encode_days(self, x, x_low=None):  # -> (N, L, d_day)
        N, L, B, F = x.shape

        # ── concat 模式：双频 bar 直接拼成宽序列 ──
        if self.has_lowfreq and self.lowfreq_mode == "concat" and x_low is not None:
            _, _, BL, _ = x_low.shape
            x = torch.cat([x, x_low], dim=2)   # (N, L, B+BL, F)
            B = B + BL

        # ── 高频路径（内生）/ concat 后的统一路径 ──
        flat = x.reshape(N * L, B, F)
        chunks = []
        for start in range(0, len(flat), self.intra_chunk_size):
            part = flat[start:start + self.intra_chunk_size]

            def _fn(p):
                return self.intra(self.fieldmix(p))

            chunks.append(_ckpt(_fn, part, use_reentrant=False))
        daily = torch.cat(chunks, dim=0)  # (N*L, d_intra)

        # ── 低频路径（外生, TimeXer variate-wise, 仅 cross_attn 模式）──
        if (self.has_lowfreq and self.lowfreq_mode != "concat"
                and x_low is not None):
            _, _, BL, _ = x_low.shape
            flat_low = x_low.reshape(N * L, BL, F)
            chunks_low = []
            for start in range(0, len(flat_low), self.intra_chunk_size):
                part = flat_low[start:start + self.intra_chunk_size]

                def _fn_low(p):
                    return self.variate(self.fieldmix_low(p))

                chunks_low.append(_ckpt(_fn_low, part, use_reentrant=False))
            feat_tokens = torch.cat(chunks_low, dim=0)  # (N*L, F, d_intra)
            context = self.bridge(feat_tokens)            # (N*L, d_intra)
            daily = self.fuse(torch.cat([daily, context], dim=-1))  # (N*L, d_intra)

        return self.inter(daily.reshape(N, L, -1))

    def score_day(self, H):  # (N, tau, d_day) -> (N,)
        if self.use_cross:
            H = self.cross(H.transpose(0, 1)).transpose(0, 1)
        return self.head(self.agg(H))

    def forward(self, x, x_low=None):
        H = self.encode_days(x, x_low)
        return self.score_day(H[:, -self.tau:])


# ============================================================
# §6 损失与验证指标
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as _sp_stats


def daily_ic_loss(pred, y, eps=1e-8):
    p = pred - pred.mean()
    t = y - y.mean()
    ic = (p * t).sum() / (p.norm() * t.norm() + eps)
    return 1.0 - ic


def pairwise_ranking_loss(pred, y, eps=1e-8):
    """RankNet logistic pairwise loss: 对同一截面内每对 label 不等的股票,
    若 y_i > y_j 但 pred_i < pred_j 则惩罚。

    天然抗坍缩: 输出全相等时 loss=log(2)≈0.693, 梯度按 label 排名方向推散。
    O(N^2), N=50~70 时单次计算 << 1ms。

    Args:
        pred, y: (N,) 同一交易日的预测和标签
    Returns:
        scalar, 平均每对的 loss
    """
    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # pred_diff[i,j] = pred_i - pred_j
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)  # (N, N)
    y_diff = y.unsqueeze(0) - y.unsqueeze(1)            # (N, N)

    # 只考虑 label 严格不等的 pair
    mask = (y_diff.abs() > eps).to(pred.dtype)
    sign = y_diff.sign()

    # RankNet: log(1 + exp(-sign * pred_diff))
    loss_per_pair = torch.log1p(torch.exp(-sign * pred_diff))

    n_pairs = mask.sum()
    if n_pairs == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return (loss_per_pair * mask).sum() / n_pairs


def listnet_loss(pred, y, temp):
    """数值安全版: softmax 加 eps 防止 float16 下 exp 下溢为精确 0 → log 为 -inf
    → 0*(-inf)=NaN。"""
    y_prob = torch.softmax(y / temp, 0).clamp(min=1e-12)
    pred_log = torch.log_softmax(pred / temp, 0)
    return -(y_prob * pred_log).sum()


def listmle_loss(pred, y, temp):
    """ListMLE: KL(softmax(y/temp) || softmax(pred/temp))。数值安全版（同上）。"""
    y_prob = torch.softmax(y / temp, dim=0).clamp(min=1e-12)
    pred_log = torch.log_softmax(pred / temp, dim=0)
    return (y_prob * (torch.log(y_prob) - pred_log)).sum()


def total_loss(pred, y, cfg):
    l = cfg["loss"]

    # 主损失：pairwise 优先（抗坍缩），回退到 daily_ic_loss
    if l.get("pairwise_w", 0) > 0:
        out = l["pairwise_w"] * pairwise_ranking_loss(pred, y)
    else:
        out = daily_ic_loss(pred, y)

    out = out + l["mse_w"] * F.mse_loss(pred, y)
    if l.get("listnet_w", 0) > 0:
        out = out + l["listnet_w"] * listnet_loss(pred, y, l["listnet_temp"])
    if l.get("listmle_w", 0) > 0:
        out = out + l["listmle_w"] * listmle_loss(pred, y, l["listmle_temp"])

    # 方差底线：pred_std 低于 min_pred_std 时施加二次惩罚，防止模型坍缩到常数输出
    min_std = l.get("min_pred_std", 0.0)
    if min_std > 0:
        pred_std = pred.std()
        if pred_std < min_std:
            out = out + l.get("std_w", 1.0) * (min_std - pred_std) ** 2

    # 末端防护：NaN → 用 MSE 兜底（保留梯度图，optimizer 可继续）
    if torch.isnan(out) or torch.isinf(out):
        out = 10.0 * F.mse_loss(pred, y)

    return out


def diagnose_nan(pred, y, cfg):
    """定位非有限 loss 的来源。仅在训练循环跳过 batch 时调用（无梯度, fp32）。

    返回 (cause, info): cause ∈ pred_nan / pred_inf / y_bad / loss_<分量> /
    loss_unknown; info 含 pred 有限性、坏元素数、最大幅值与各分量值。"""
    with torch.no_grad():
        pf = pred.detach().float()
        finite = torch.isfinite(pf)
        info = {
            "pred_finite": bool(finite.all()),
            "pred_bad": int((~finite).sum()),
            "pred_absmax": (float(pf[finite].abs().max())
                            if finite.any() else float("inf")),
            "y_finite": bool(torch.isfinite(y).all()),
        }
        if not info["pred_finite"]:
            return ("pred_nan" if torch.isnan(pf).any() else "pred_inf"), info
        if not info["y_finite"]:
            return "y_bad", info
        l = cfg["loss"]
        if l.get("pairwise_w", 0) > 0:
            comps = {"pairwise": pairwise_ranking_loss(pf, y)}
        else:
            comps = {"daily_ic": daily_ic_loss(pf, y)}
        if l.get("listnet_w", 0) > 0:
            comps["listnet"] = listnet_loss(pf, y, l["listnet_temp"])
        if l.get("listmle_w", 0) > 0:
            comps["listmle"] = listmle_loss(pf, y, l["listmle_temp"])
        info["components"] = {k: round(float(v), 6) for k, v in comps.items()}
        for name, v in comps.items():
            if not np.isfinite(float(v)):
                return f"loss_{name}", info
        return "loss_unknown", info


def rank_ic(pred, y):
    return float(_sp_stats.spearmanr(pred, y).statistic)


def long_short_sharpe(daily_pred, q=10):
    """daily_pred: [(pred (N,), realized_return (N,)), ...]。
    realized_return 必须保留真实收益尺度，不能使用逐日 z-score 标签。"""
    rets = []
    for pred, y in daily_pred:
        n = len(pred)
        if n < q * 2:
            continue
        order = np.argsort(pred)
        k = n // q
        rets.append(float(np.mean(y[order[-k:]]) - np.mean(y[order[:k]])))
    if len(rets) < 2:
        return 0.0
    r = np.array(rets)
    return float(r.mean() / (r.std() + 1e-9) * np.sqrt(252))


def composite_score(ic_series, sharpe, w):
    ic = np.asarray(ic_series, np.float64)
    icir = ic.mean() / (ic.std() + 1e-9)
    return float(w["ic"] * ic.mean() + w["icir"] * icir + w["sharpe"] * sharpe)


# ============================================================
# §7 训练入口
# ============================================================

import copy
import os
import random
import sys
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast


# 开发表可用区间的宽松包络 (窄于哨兵区间, 利于引擎按日期裁剪);
# 实际训练/验证切分仍是"取表内全部可用交易日的前80%/后20%"的相对逻辑
TRAIN_START, TRAIN_END = "2019-01-01 00:00:00", "2024-12-31 23:59:59"
_ACTIVE_PROGRESS = None


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _gpu_memory(device):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return (f" gpu_alloc={allocated:.1f}GiB"
            f" gpu_reserved={reserved:.1f}GiB"
            f" gpu_peak={peak:.1f}GiB")


def _gpu_memory_compact(device):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return f" GPU a/r/p={allocated:.1f}/{reserved:.1f}/{peak:.1f}G"


class _ProgressBar:
    """Dependency-free terminal progress bar with clean checkpoint messages."""

    def __init__(self, label, total, fallback_every=20, width=16):
        global _ACTIVE_PROGRESS
        self.label = label
        self.total = max(int(total), 1)
        self.fallback_every = max(int(fallback_every), 1)
        self.width = width
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.last_line = ""
        self.last_step = 0
        _ACTIVE_PROGRESS = self

    def update(self, step, suffix=""):
        step = min(int(step), self.total)
        pct = step / self.total
        filled = int(self.width * pct)
        bar = "#" * filled + "-" * (self.width - filled)
        self.last_line = (f"{self.label} [{bar}] {step}/{self.total} "
                          f"{pct * 100:5.1f}%{suffix}")
        self.last_step = step
        # Some hosted terminals report isatty=True but turn every carriage
        # return into a new log line. Throttle both interactive and captured
        # output so progress remains readable everywhere.
        should_render = (
            step == 1 or step % self.fallback_every == 0 or step == self.total
        )
        if not should_render:
            return
        if self.interactive:
            sys.stdout.write(f"\r\033[2K{self.last_line}")
            sys.stdout.flush()
        else:
            print(self.last_line, flush=True)

    def write(self, message):
        if self.interactive:
            sys.stdout.write(f"\r\033[2K{message}\n")
            sys.stdout.flush()
        else:
            print(message, flush=True)

    def close(self):
        global _ACTIVE_PROGRESS
        if self.interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if _ACTIVE_PROGRESS is self:
            _ACTIVE_PROGRESS = None


def _terminal_log(message):
    if _ACTIVE_PROGRESS is not None:
        _ACTIVE_PROGRESS.write(message)
    else:
        print(message, flush=True)


def _atomic_torch_save(payload, path):
    """Write a checkpoint without ever exposing a partially-written target."""
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added
        return torch.load(path, map_location=map_location)


def _normalize_to_device(x, torch_normalizer, device):
    mean, std = torch_normalizer
    out = torch.from_numpy(x).to(device=device, dtype=torch.float32,
                                 non_blocking=True)
    return out.sub_(mean).div_(std)


def _evaluate(model, index, days, torch_normalizer, device, metric_w,
              progress_label="val", log_every=20, exposure_lookup=None):
    """验证模型。复现平台三步预处理 (截面 winsorize 1%/99% → z-score → 对
    BARRA 风格截面回归取残差) 后, 对残差收益算 rank IC 与多空 Sharpe,
    使选模口径与平台一致。pred_std 报告原始预测尺度, 用于监控输出坍缩。"""
    model.eval()
    daily_pairs = []
    raw_stds = []
    started = time.time()
    total = len(days)
    progress = _ProgressBar(progress_label, total, log_every)
    try:
        with torch.inference_mode():
            with autocast():
                for step, d in enumerate(days, 1):
                    eval_data = index.eval_batch(d)
                    if index.has_low:
                        X, X_low, y, residual_return, instruments = eval_data
                    else:
                        X, y, residual_return, instruments = eval_data
                        X_low = None
                    X = _normalize_to_device(X, torch_normalizer, device)
                    if X_low is not None:
                        X_low = _normalize_to_device(X_low, torch_normalizer, device)
                    pred = model(X, X_low).float().cpu().numpy()
                    raw_stds.append(float(np.std(pred)))
                    # ── 平台口径: winsorize → z-score → 风格回归残差 ──
                    day_exposure = (exposure_lookup or {}).get(d)
                    Xf = None
                    mask = np.ones(len(pred), dtype=bool)
                    if day_exposure is not None:
                        facs = [day_exposure.get(ins) for ins in instruments]
                        mask = np.array([f is not None and np.isfinite(f).all()
                                         for f in facs])
                        if mask.sum() >= 30:
                            Xf = np.asarray([facs[i] for i in
                                             np.flatnonzero(mask)], np.float64)
                            if mask.sum() < Xf.shape[1] + 2:
                                Xf = None  # 截面不足以回归, 退化为前两步
                    if Xf is not None:
                        # 缺暴露的股票无法残差化, 当日指标中剔除
                        pred_proc = residualize_day(
                            winsorize_zscore(pred[mask], 1.0), Xf)
                        ret = residual_return[mask]
                    else:
                        pred_proc = winsorize_zscore(pred, 1.0)
                        ret = residual_return
                    if np.std(pred_proc) >= 1e-9 and np.std(ret) >= 1e-9:
                        daily_pairs.append((pred_proc, ret))
                    elapsed = time.time() - started
                    eta = elapsed / step * (total - step)
                    progress.update(
                        step,
                        f" | N={len(y)} valid={len(daily_pairs)}"
                        f" | {_format_duration(elapsed)} ETA {_format_duration(eta)}"
                        f" |{_gpu_memory_compact(device)}",
                    )
    finally:
        progress.close()
    ics = [rank_ic(p, r) for p, r in daily_pairs]
    ics = [v for v in ics if np.isfinite(v)]
    if not ics:
        return {"ic": 0.0, "icir": 0.0, "sharpe": 0.0, "pred_std": 0.0,
                "score": -1e9}
    sharpe = long_short_sharpe(daily_pairs)
    return {"ic": float(np.mean(ics)),
            "icir": float(np.mean(ics) / (np.std(ics) + 1e-9)),
            "sharpe": sharpe,
            "pred_std": float(np.mean(raw_stds)),
            "score": composite_score(ics, sharpe, metric_w)}


def run_training(cfg, query_fn, exposure_query_fn=None, universe_query_fn=None,
                 device=None):
    validate_config(cfg)
    set_seed(cfg["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---- 数据（双频: 高频内生 + 低频外生; 缓存优先） ----
    instruments = list_instruments(query_fn, cfg["table"], TRAIN_START, TRAIN_END)
    print(f"[data] querying {len(instruments)} instruments in chunks of "
          f"{cfg['chunk_size']} ...")
    arrays_high, arrays_low = load_multi_freq_arrays(
        query_fn, cfg, TRAIN_START, TRAIN_END, instruments)
    print(f"[data] stocks_high={len(arrays_high)} "
          f"stocks_low={len(arrays_low) if arrays_low else 0} "
          f"elapsed={time.time() - t0:.0f}s")
    has_lowfreq = arrays_low is not None

    # ---- 标签 ----
    exposure_df = None
    if exposure_query_fn is not None and cfg["label"]["mode"] == "residual":
        exposure_df = exposure_query_fn()
        print(f"[label] exposure rows={len(exposure_df)}", flush=True)
    universe_df = None
    if universe_query_fn is not None:
        universe_df = universe_query_fn()
        print(f"[universe] official rows={len(universe_df)}", flush=True)
    exposure_lookup = build_exposure_lookup(exposure_df)
    t_l = time.time()
    labels = build_label_panel({k: (v[0], v[2]) for k, v in arrays_high.items()},
                               exposure_df, cfg, universe_df)
    print(f"[label] panel rows={len(labels)} in {time.time() - t_l:.0f}s", flush=True)
    index = DaySampleIndex(arrays_high, labels, cfg, arrays_low)
    train_days, val_days = split_days(index.days, cfg["train_frac"])
    print(f"[data] days train={len(train_days)} val={len(val_days)} "
          f"has_lowfreq={has_lowfreq}", flush=True)

    # ---- 标准化统计（仅训练区间的按股数组, 合规: 参数只来自训练集;
    #      每个交易日计入一次, 不因窗口重叠重复计权, 且比逐日组 batch 快两个量级） ----
    t_s = time.time()
    rs = RunningStats(len(cfg["feature_cols"]))
    val_start = val_days[0] if val_days else None
    for dates, X, _ in arrays_high.values():
        Xm = X if val_start is None else X[dates < val_start]
        if len(Xm):
            rs.update(Xm)
    mean, std = rs.finalize()
    torch_normalizer = (torch.as_tensor(mean, device=device),
                        torch.as_tensor(std, device=device))
    print(f"[stats] done in {time.time() - t_s:.0f}s", flush=True)

    # ---- 模型 ----
    model = StockScorer(cfg).to(device)
    n_params = count_params(model)
    print(f"[model] device={device} params={n_params}{_gpu_memory(device)}", flush=True)
    if not cfg.get("skip_param_check"):
        assert_param_count(n_params)

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": body_params, "weight_decay": 0.0},
         {"params": head_params, "weight_decay": cfg["train"]["weight_decay_head"]}],
        lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    scaler = GradScaler()
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(cfg["train"]["epochs"] * (1 - cfg["train"]["swa_frac"]))

    rng = np.random.RandomState(cfg["seed"])
    best = {"score": -1e9, "state": None, "epoch": -1, "metrics": None}
    bad_epochs = 0
    ratio = cfg["train"]["stock_sample_ratio"]
    log_every = max(1, int(cfg["train"].get("log_every", 20)))
    checkpoint_every = max(1, int(cfg["train"].get("checkpoint_every", log_every)))
    metric_w = cfg["train"]["metric_w"]
    swa_updated = False
    budget_s = (cfg["train"].get("time_budget_min") or 0) * 60
    epoch_times = []
    resume_path = cfg.get("resume_checkpoint_path", "last_checkpoint.pt")
    elapsed_before = 0.0
    start_epoch = 0
    resume_phase = "train"
    resume_order = None
    resume_next_step = 0
    resume_tot = 0.0
    resume_nb = 0
    resume_n_skip = 0
    resume_epoch_elapsed = 0.0
    legacy_best_to_revalue = None
    data_signature = {
        "label_rows": len(labels),
        "train_days": len(train_days),
        "val_days": len(val_days),
        "official_universe": universe_df is not None,
    }

    def _save_best(announce=False):
        ckpt = {"state_dict": best["state"], "mean": mean, "std": std,
                "config": cfg, "best_epoch": best["epoch"],
                "val_metrics": best["metrics"], "metric_version": 3}
        _atomic_torch_save(ckpt, cfg["checkpoint_path"])
        epoch_label = best["epoch"] + 1 if best["epoch"] >= 0 else "SWA"
        if announce:
            m = best["metrics"]
            border = "=" * 72
            _terminal_log(
                f"{border}\n"
                f"*** NEW BEST - 请查看 ***\n"
                f"epoch={epoch_label}  score={best['score']:.6f}\n"
                f"IC={m['ic']:.6f}  ICIR={m['icir']:.4f}  "
                f"Sharpe={m['sharpe']:.4f}  pred_std={m['pred_std']:.6f}\n"
                f"saved={cfg['checkpoint_path']}\n"
                f"{border}"
            )
        else:
            _terminal_log(
                f"[ckpt:best] saved {cfg['checkpoint_path']} "
                f"(epoch={epoch_label} score={best['score']:.4f})"
            )
        return ckpt

    def _total_elapsed():
        return elapsed_before + time.time() - t0

    def _save_resume(epoch, phase, order=None, next_step=0, tot=0.0, nb=0,
                     epoch_elapsed=0.0, n_skip=0, complete=False, announce=True):
        state = {
            "format_version": 2,
            "complete": complete,
            "phase": phase,
            "epoch": int(epoch),
            "order": None if order is None else np.asarray(order),
            "next_step": int(next_step),
            "tot": float(tot),
            "nb": int(nb),
            "n_skip": int(n_skip),
            "epoch_elapsed": float(epoch_elapsed),
            "elapsed_s": float(_total_elapsed()),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "swa_state_dict": swa_model.state_dict(),
            "swa_updated": swa_updated,
            "best": best,
            "bad_epochs": bad_epochs,
            "epoch_times": epoch_times,
            "numpy_rng_state": rng.get_state(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
            "config": cfg,
            "data_signature": data_signature,
            "metric_version": 3,
        }
        _atomic_torch_save(state, resume_path)
        if announce:
            where = (f"epoch={epoch + 1} day={next_step}/{len(order)}"
                     if phase == "train" and order is not None
                     else f"epoch={epoch + 1} phase={phase}")
            _terminal_log(f"[ckpt:last] saved {resume_path} ({where})")

    if os.path.exists(resume_path):
        try:
            # Load on CPU first: model/optimizer loaders move tensors to their
            # parameter devices, while RNG states must remain CPU ByteTensors.
            saved = _torch_load(resume_path, "cpu")
            if saved.get("format_version") != 2:
                raise ValueError("unsupported resume checkpoint format")
            if saved.get("data_signature") != data_signature:
                raise ValueError(
                    f"training data changed: saved={saved.get('data_signature')} "
                    f"current={data_signature}"
                )
            if saved.get("complete"):
                print(f"[resume] training already complete: {resume_path}", flush=True)
                if os.path.exists(cfg["checkpoint_path"]):
                    return _torch_load(cfg["checkpoint_path"], "cpu")
                raise FileNotFoundError("best checkpoint is missing")
            model.load_state_dict(saved["model_state_dict"])
            opt.load_state_dict(saved["optimizer_state_dict"])
            sched.load_state_dict(saved["scheduler_state_dict"])
            if "scaler_state_dict" in saved:
                scaler.load_state_dict(saved["scaler_state_dict"])
            swa_model.load_state_dict(saved["swa_state_dict"])
            swa_updated = bool(saved["swa_updated"])
            best = saved["best"]
            bad_epochs = int(saved["bad_epochs"])
            if saved.get("metric_version", 1) != 3 and best["state"] is not None:
                print("[resume] validation metric upgraded to platform caliber "
                      "(winsorize→z-score→style residual); old best will be "
                      "re-evaluated before training resumes", flush=True)
                legacy_best_to_revalue = best
                bad_epochs = 0
            epoch_times = list(saved["epoch_times"])
            rng.set_state(saved["numpy_rng_state"])
            random.setstate(saved["python_rng_state"])
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
            elapsed_before = float(saved.get("elapsed_s", 0.0))
            start_epoch = int(saved["epoch"])
            resume_phase = saved["phase"]
            resume_order = saved.get("order")
            resume_next_step = int(saved.get("next_step", 0))
            resume_tot = float(saved.get("tot", 0.0))
            resume_nb = int(saved.get("nb", 0))
            resume_n_skip = int(saved.get("n_skip", 0))
            resume_epoch_elapsed = float(saved.get("epoch_elapsed", 0.0))
            resume_at = (f"next_day={resume_next_step + 1}"
                         if resume_phase == "train"
                         else "validation will restart")
            print(f"[resume] loaded: {resume_path} phase={resume_phase} "
                  f"epoch={start_epoch + 1}/{cfg['train']['epochs']} "
                  f"{resume_at}", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"resume checkpoint {resume_path!r} exists but cannot be loaded; "
                "move or delete it only if you intentionally want to restart"
            ) from e
    else:
        print(f"[resume] no checkpoint, starting fresh; autosave={resume_path} "
              f"every {checkpoint_every} training days", flush=True)

    if legacy_best_to_revalue is not None:
        # The current model may already be part-way through a later epoch.
        # Temporarily evaluate the previous best weights under metric v2, then
        # restore the exact in-progress weights and optimizer state.
        current_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(legacy_best_to_revalue["state"])
        legacy_epoch = legacy_best_to_revalue["epoch"]
        print(f"[metric migration] re-evaluating old best epoch={legacy_epoch + 1} "
              "with platform-caliber metrics", flush=True)
        migrated_metrics = _evaluate(
            model, index, val_days, torch_normalizer, device, metric_w,
            progress_label=f"revalue best e{legacy_epoch + 1}",
            log_every=log_every, exposure_lookup=exposure_lookup,
        )
        best = {
            "score": migrated_metrics["score"],
            "epoch": legacy_epoch,
            "state": copy.deepcopy(model.state_dict()),
            "metrics": migrated_metrics,
        }
        _save_best(announce=True)
        model.load_state_dict(current_state)
        _save_resume(
            start_epoch, resume_phase, resume_order, resume_next_step,
            resume_tot, resume_nb, resume_epoch_elapsed, announce=True,
        )
        print("[metric migration] current in-progress model restored; "
              "training will continue from the saved day", flush=True)

    active = {
        "epoch": start_epoch, "phase": resume_phase, "order": resume_order,
        "next_step": resume_next_step, "tot": resume_tot, "nb": resume_nb,
        "n_skip": resume_n_skip, "epoch_elapsed": resume_epoch_elapsed,
    }
    last_epoch = start_epoch
    stop_reason = None

    try:
        for ep in range(start_epoch, cfg["train"]["epochs"]):
            last_epoch = ep
            continuing = ep == start_epoch
            phase = resume_phase if continuing else "train"

            if phase == "train":
                model.train()
                order = (np.asarray(resume_order) if continuing and resume_order is not None
                         else rng.permutation(len(train_days)))
                next_step = resume_next_step if continuing else 0
                tot = resume_tot if continuing else 0.0
                nb = resume_nb if continuing else 0
                n_skip = resume_n_skip if continuing else 0
                skip_causes = {}
                prior_epoch_elapsed = resume_epoch_elapsed if continuing else 0.0
                te = time.time()
                n_steps = len(order)
                active.update(epoch=ep, phase="train", order=order,
                              next_step=next_step, tot=tot, nb=nb, n_skip=n_skip,
                              epoch_elapsed=prior_epoch_elapsed)
                print(f"[epoch {ep + 1}/{cfg['train']['epochs']}] train start "
                      f"days={n_steps} resume_at={next_step + 1} "
                      f"sample_ratio={ratio:.2f} lr={sched.get_last_lr()[0]:.2e} "
                      f"best={best['score']:.4f}", flush=True)
                progress = _ProgressBar(
                    f"Train {ep + 1:02d}/{cfg['train']['epochs']:02d}",
                    n_steps, log_every,
                )
                try:
                    for pos in range(next_step, n_steps):
                        step = pos + 1
                        j = order[pos]
                        ts = time.time()
                        batch_data = index.batch(train_days[j])
                        if has_lowfreq:
                            X, X_low, y, _ = batch_data
                        else:
                            X, y, _ = batch_data
                            X_low = None
                        if ratio < 1.0 and len(y) > 20:
                            keep = rng.choice(
                                len(y), max(20, int(len(y) * ratio)), replace=False
                            )
                            X, y = X[keep], y[keep]
                            if X_low is not None:
                                X_low = X_low[keep]
                        X = _normalize_to_device(X, torch_normalizer, device)
                        if X_low is not None:
                            X_low = _normalize_to_device(X_low, torch_normalizer,
                                                         device)
                        y = torch.from_numpy(y).to(device, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        with autocast():
                            pred = model(X, X_low)
                            loss = total_loss(pred.float(), y, cfg)
                        # NaN/Inf 保护：不 backward、不计入 loss 均值，
                        # 参数保持上一次有效状态；skips 计入 epoch 总结
                        loss_value = loss.detach().item()
                        if not np.isfinite(loss_value):
                            n_skip += 1
                            cause, diag = diagnose_nan(pred, y, cfg)
                            skip_causes[cause] = skip_causes.get(cause, 0) + 1
                            if n_skip <= 3:
                                _terminal_log(
                                    f"[nan-diag] epoch={ep + 1} day={step} "
                                    f"cause={cause} "
                                    f"pred_finite={diag['pred_finite']} "
                                    f"pred_bad={diag['pred_bad']} "
                                    f"pred_absmax={diag['pred_absmax']:.3e} "
                                    f"y_finite={diag['y_finite']}"
                                    + (f" comps={diag['components']}"
                                       if "components" in diag else ""))
                            epoch_elapsed = prior_epoch_elapsed + time.time() - te
                            active.update(next_step=step, n_skip=n_skip,
                                          epoch_elapsed=epoch_elapsed)
                            done_skip = step - next_step
                            run_elapsed_skip = time.time() - te
                            eta_skip = (run_elapsed_skip / max(done_skip, 1)
                                        * (n_steps - step))
                            progress.update(step,
                                            f" | L=NaN(skip) skips={n_skip}"
                                            f" avg={tot / max(nb, 1):.4f}"
                                            f" N={len(y)}"
                                            f" | {_format_duration(epoch_elapsed)}"
                                            f" ETA {_format_duration(eta_skip)}"
                                            f" |{_gpu_memory_compact(device)}")
                            continue
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg["train"]["clip"]
                        )
                        scaler.step(opt)
                        scaler.update()
                        tot += loss_value
                        nb += 1
                        epoch_elapsed = prior_epoch_elapsed + time.time() - te
                        active.update(next_step=step, tot=tot, nb=nb,
                                      epoch_elapsed=epoch_elapsed)
                        done_this_run = step - next_step
                        run_elapsed = time.time() - te
                        eta = (run_elapsed / max(done_this_run, 1)
                               * (n_steps - step))
                        progress.update(
                            step,
                            f" | L={loss_value:.4f} avg={tot / nb:.4f}"
                            f" N={len(y)}"
                            f" | {_format_duration(epoch_elapsed)}"
                            f" ETA {_format_duration(eta)}"
                            f" |{_gpu_memory_compact(device)}",
                        )
                        # The epoch-end state is saved immediately below as
                        # phase=validation, so do not write the same weights twice.
                        if step % checkpoint_every == 0 and step < n_steps:
                            _save_resume(ep, "train", order, step, tot, nb,
                                         epoch_elapsed, n_skip=n_skip)
                finally:
                    progress.close()

                sched.step()
                if ep >= swa_start:
                    swa_model.update_parameters(model)
                    swa_updated = True
                epoch_elapsed = prior_epoch_elapsed + time.time() - te
                active.update(phase="validation", order=None,
                              epoch_elapsed=epoch_elapsed)
                _save_resume(ep, "validation", tot=tot, nb=nb,
                             epoch_elapsed=epoch_elapsed, n_skip=n_skip)
            else:
                tot, nb, n_skip = resume_tot, resume_nb, resume_n_skip
                skip_causes = {}
                epoch_elapsed = resume_epoch_elapsed
                active.update(epoch=ep, phase="validation", order=None,
                              next_step=0, tot=tot, nb=nb, n_skip=n_skip,
                              epoch_elapsed=epoch_elapsed)

            print(f"[epoch {ep + 1}/{cfg['train']['epochs']}] validation start "
                  f"days={len(val_days)}", flush=True)
            m = _evaluate(model, index, val_days, torch_normalizer, device, metric_w,
                          progress_label=f"val {ep + 1}/{cfg['train']['epochs']}",
                          log_every=log_every, exposure_lookup=exposure_lookup)
            full_epoch_elapsed = epoch_elapsed
            print(f"[epoch {ep + 1}/{cfg['train']['epochs']}] complete "
                  f"loss={tot / max(nb, 1):.4f} "
                  f"ic={m['ic']:.4f} icir={m['icir']:.2f} "
                  f"sharpe={m['sharpe']:.2f} pred_std={m['pred_std']:.4f} "
                  f"score={m['score']:.4f} skips={n_skip} "
                  f"train_elapsed={_format_duration(full_epoch_elapsed)}",
                  flush=True)
            skip_rate = n_skip / max(nb + n_skip, 1)
            if n_skip:
                causes = " ".join(f"{k}={v}" for k, v
                                  in sorted(skip_causes.items()))
                print(f"[nan-summary] epoch {ep + 1} skips={n_skip} "
                      f"causes: {causes}", flush=True)
            if n_skip and skip_rate > 0.02:
                print(f"[warn] epoch {ep + 1} NaN skip rate "
                      f"{skip_rate:.1%} ({n_skip}/{nb + n_skip} days) > 2% — "
                      f"非偶发, 建议排查 fp16 溢出等结构性原因", flush=True)
            if m["score"] > best["score"]:
                best = {"score": m["score"], "epoch": ep,
                        "state": copy.deepcopy(model.state_dict()), "metrics": m}
                bad_epochs = 0
                _save_best(announce=True)
            else:
                bad_epochs += 1

            epoch_times.append(full_epoch_elapsed)
            if bad_epochs >= cfg["train"]["patience"]:
                stop_reason = "early_stop"
                print("[early stop]", flush=True)
                break
            if budget_s and _total_elapsed() + max(epoch_times) * 1.2 > budget_s:
                stop_reason = "time_budget"
                print(f"[time budget] elapsed={_format_duration(_total_elapsed())}, "
                      f"next epoch would exceed {_format_duration(budget_s)}",
                      flush=True)
                break

            _save_resume(ep + 1, "train", announce=True)
            resume_phase, resume_order = "train", None
            resume_next_step, resume_tot, resume_nb = 0, 0.0, 0
            resume_n_skip = 0
            resume_epoch_elapsed = 0.0
            active.update(epoch=ep + 1, phase="train", order=None,
                          next_step=0, tot=0.0, nb=0, n_skip=0,
                          epoch_elapsed=0.0)
    except KeyboardInterrupt:
        try:
            _save_resume(active["epoch"], active["phase"], active["order"],
                         active["next_step"], active["tot"], active["nb"],
                         active["epoch_elapsed"], n_skip=active["n_skip"])
            print(f"\n[interrupt] stopped safely; rerun python train.py to resume "
                  f"from {resume_path}", flush=True)
        except Exception as save_error:
            print(f"\n[interrupt] emergency checkpoint failed: {save_error!r}",
                  flush=True)
        return None

    # ---- SWA 候选与最优单点二选一 ----
    if swa_updated:
        print(f"[swa] evaluating days={len(val_days)}", flush=True)
        m_swa = _evaluate(swa_model.module, index, val_days, torch_normalizer, device,
                          metric_w, progress_label="swa", log_every=log_every,
                          exposure_lookup=exposure_lookup)
        if m_swa["score"] > best["score"]:
            print(f"[swa selected] score={m_swa['score']:.4f}")
            best = {"score": m_swa["score"], "epoch": -1,
                    "state": copy.deepcopy(swa_model.module.state_dict()),
                    "metrics": m_swa}
            _save_best(announce=True)

    ckpt = _save_best()
    _save_resume(last_epoch, "done", complete=True)
    print(f"[done] reason={stop_reason or 'epochs_complete'} "
          f"total={_format_duration(_total_elapsed())} best={best['metrics']}",
          flush=True)
    return ckpt


def train_main():
    """从平台训练表中训练并同时导出 JSON 训练产物。"""
    import dai

    def query_fn(sql, filters):
        return dai.query(sql, filters=filters, compression=True).df()

    def exposure_query_fn():
        return dai.query(
            f"SELECT * FROM {CONFIG['exposure_table']}",
            filters={"date": [TRAIN_START, TRAIN_END]},
            compression=True,
        ).df()

    def universe_query_fn():
        return dai.query(
            f"SELECT date, instrument FROM {CONFIG['instruments_table']}",
            filters={"date": [TRAIN_START, TRAIN_END]},
            compression=True,
        ).df()

    ckpt = run_training(CONFIG, query_fn, exposure_query_fn, universe_query_fn)
    export_model_json(ckpt, CONFIG["artifact_path"])
    return ckpt


# ============================================================
# §8 推理
# ============================================================

import numpy as np
import pandas as pd
import torch


def run_inference(ckpt, query_fn, infer_table, instruments_query_fn,
                  start_date, end_date, device=None):
    cfg = ckpt["config"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = StockScorer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if not cfg.get("skip_param_check"):
        assert_param_count(count_params(model))
    normalizer = Normalizer(ckpt["mean"], ckpt["std"])

    skeleton = instruments_query_fn(start_date, end_date)[["date", "instrument"]].copy()
    skeleton["date"] = pd.to_datetime(skeleton["date"]).dt.normalize()
    instruments = sorted(skeleton["instrument"].unique().tolist())

    buf = (pd.to_datetime(start_date)
           - pd.Timedelta(days=cfg["infer_buffer_natural_days"])).strftime("%Y-%m-%d 00:00:00")
    sd = pd.to_datetime(start_date).normalize()
    ed = pd.to_datetime(end_date).normalize()

    has_lowfreq = cfg.get("lowfreq_table") is not None
    lowfreq_table = cfg.get("lowfreq_table")

    tau = cfg["model"]["tau_cross"]
    per_day = {}  # date -> list[(instrument, H_tau (tau, d_day))]
    with torch.no_grad():
        for chunk in iter_chunks(instruments, cfg["chunk_size"]):
            arrays_high = load_stock_arrays(query_fn, infer_table, cfg,
                                            buf, str(end_date), list(chunk),
                                            bars_per_day=cfg["bars_per_day"])
            arrays_low = None
            if has_lowfreq and lowfreq_table:
                arrays_low = load_stock_arrays(query_fn, lowfreq_table, cfg,
                                               buf, str(end_date), list(chunk),
                                               bars_per_day=cfg["lowfreq_bars_per_day"])
            # 按 instrument 对齐双频数据
            for ins in arrays_high:
                dates_high, X_high, _ = arrays_high[ins]
                X_low = None
                if arrays_low and ins in arrays_low:
                    dates_low, X_low_raw, _ = arrays_low[ins]
                idx = [i for i, d in enumerate(dates_high)
                       if sd <= pd.Timestamp(d) <= ed]
                if not idx:
                    continue
                Xn = torch.from_numpy(
                    normalizer.apply(X_high.astype(np.float32)))[None].to(device)
                Xn_low = None
                if X_low is not None:
                    # 对齐低频日期：找到每个高频评估日对应的低频日位置
                    low_date_to_i = {pd.Timestamp(d): i for i, d in enumerate(dates_low)}
                    low_indices = []
                    for i in idx:
                        d_high = pd.Timestamp(dates_high[i])
                        li = low_date_to_i.get(d_high)
                        if li is not None:
                            low_indices.append(li)
                    if low_indices:
                        X_low_aligned = X_low_raw[low_indices]
                        Xn_low = torch.from_numpy(
                            normalizer.apply(X_low_aligned.astype(np.float32)))[None].to(device)
                if Xn_low is not None:
                    H = model.encode_days(Xn, Xn_low)[0].cpu()  # (D, d_day)
                else:
                    H = model.encode_days(Xn)[0].cpu()  # (D, d_day)
                for i in idx:
                    h = H[max(0, i + 1 - tau): i + 1]
                    if len(h) < tau:  # 历史不足: 重复首日补齐
                        h = torch.cat([h[:1].expand(tau - len(h), -1), h])
                    d_dt = pd.Timestamp(dates_high[i])
                    per_day.setdefault(d_dt, []).append((ins, h))
            del arrays_high
            if arrays_low:
                del arrays_low

        rows = []
        for d, items in per_day.items():
            Ht = torch.stack([h for _, h in items]).to(device)  # (N, tau, d_day)
            scores = model.score_day(Ht).cpu().numpy()
            rows += [(d, ins, float(s)) for (ins, _), s in zip(items, scores)]

    pred = pd.DataFrame(rows, columns=["date", "instrument", "score"])
    out = skeleton.merge(pred, on=["date", "instrument"], how="left")
    # 缺失分数填入当日截面中位数，避免固定值 0 偏离分布中心
    out["score"] = out["score"].replace([np.inf, -np.inf], np.nan)
    out["score"] = out.groupby("date")["score"].transform(
        lambda x: x.fillna(x.median()))
    # 极端情况：某天完全没有有效分（全部 NaN），退化为 0
    out["score"] = out["score"].fillna(0.0)
    return out[["date", "instrument", "score"]].reset_index(drop=True)


# ============================================================
# §9 JSON 产物导入/导出
# ============================================================

def export_model_json(ckpt, path):
    """把 PyTorch checkpoint 无损编码为官方要求的 JSON 训练产物。"""
    import base64
    import json

    encoded_state = {}
    for name, value in ckpt["state_dict"].items():
        array = value.detach().cpu().contiguous().numpy()
        encoded_state[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    artifact = {
        "format_version": 1,
        "artifact_type": "bigalpha_pytorch_state_dict",
        "state_dict": encoded_state,
        "mean": np.asarray(ckpt["mean"], np.float32).tolist(),
        "std": np.asarray(ckpt["std"], np.float32).tolist(),
        "config": ckpt["config"],
        "best_epoch": ckpt.get("best_epoch"),
        "val_metrics": ckpt.get("val_metrics"),
        "metric_version": ckpt.get("metric_version", 1),
    }
    target = os.fspath(path)
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, target)
    print(f"[artifact] saved {target}", flush=True)
    return target


def load_model_json(path):
    """读取 JSON 训练产物并还原为 run_inference 使用的 checkpoint。"""
    import base64
    import json

    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("format_version") != 1:
        raise ValueError(
            f"不支持的模型产物版本: {artifact.get('format_version')}"
        )
    state_dict = {}
    for name, item in artifact["state_dict"].items():
        raw = base64.b64decode(item["data_b64"])
        array = np.frombuffer(raw, dtype=np.dtype(item["dtype"])).copy()
        array = array.reshape(item["shape"])
        state_dict[name] = torch.from_numpy(array)
    return {
        "state_dict": state_dict,
        "mean": np.asarray(artifact["mean"], np.float32),
        "std": np.asarray(artifact["std"], np.float32),
        "config": artifact["config"],
        "best_epoch": artifact.get("best_epoch"),
        "val_metrics": artifact.get("val_metrics"),
        "metric_version": artifact.get("metric_version", 1),
    }


# ============================================================
# §10 CLI
# ============================================================

def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="不重新训练，把已有 checkpoint.pt 转为 JSON 训练产物",
    )
    parser.add_argument("--checkpoint", default=CONFIG["checkpoint_path"])
    parser.add_argument("--artifact", default=CONFIG["artifact_path"])
    args = parser.parse_args()
    if args.export_only:
        ckpt = _torch_load(args.checkpoint, "cpu")
        export_model_json(ckpt, args.artifact)
    else:
        train_main()


if __name__ == "__main__":
    _cli()
