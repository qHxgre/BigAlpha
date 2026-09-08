"""BigAlpha 2026 自包含训练脚本；运行后生成 transformer_model.json。"""

"""BigAlpha 端到端模型全局配置。提交要求的超参配置 + 随机种子集中于此。"""
import random

import numpy as np

# 字段清单依据 2026-07-27 平台探查结果 (bar15m 实际 schema, 盘口仅 5 档,
# 无 num_trades/avg_price/total_volume 系字段; 成交笔数为 deal_number)。
PRICE_COLS = ["open", "high", "low", "close", "pre_close",
              "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",
              "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5"]
VOL_COLS = ["volume", "amount", "deal_number",
            "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",
            "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5"]

CONFIG = {
    "table": "bigalpha_2026_stock_bar15m",
    "instruments_table": "bigalpha_2026_instruments",
    "exposure_table": "bigalpha_2026_exposure",
    "bars_per_day": 16,
    "price_cols": PRICE_COLS,
    "vol_cols": VOL_COLS,
    "feature_cols": PRICE_COLS + VOL_COLS,
    "min_time_coverage": 0.0,   # 每个股票日最少有效 (bar, 字段) 占比; 0=不丢弃 (缺失率仅 0.13%)
    # 复权因子仅用于标签收益计算 (close*adjust_factor), 不作为模型输入字段
    "adjust_col": "adjust_factor",
    "lookback_days": 60,
    "infer_buffer_natural_days": 130,
    "chunk_size": 250,  # 每块股票数; 内存紧张(<32G)时降回 100
    "train_frac": 0.8,
    "seed": 42,
    "cache_path": "train_cache.npz",
    "checkpoint_path": "checkpoint.pt",
    "resume_checkpoint_path": "last_checkpoint.pt",
    # 官方提交的训练产物必须为 JSON；checkpoint.pt 仅用于本地断点/最佳权重。
    "artifact_path": "transformer_model.json",
    "label": {"mode": "residual", "horizon": 1, "winsor_pct": 1.0},
    "model": {"d_intra": 64, "heads_intra": 4, "layers_intra": 2, "ffn_intra": 128,
              "intra_chunk_size": 1024,
              "d_day": 128, "tau_cross": 8, "heads_cross": 2, "ffn_cross": 512,
              "use_cross_attn": True, "gate_init": -5.0, "glu_bottleneck": 64,
              "dropout": 0.1,
              "filter_kernel_sizes": [3, 7, 15]},
    "loss": {"mse_w": 0.1, "listnet_w": 0.05, "listnet_temp": 1.0},
    "train": {"epochs": 40, "lr": 5e-4, "weight_decay_head": 1.0, "clip": 1.0,
              "patience": 3, "swa_frac": 0.25, "stock_sample_ratio": 1.0,
              "log_every": 5, "checkpoint_every": 100,
              "time_budget_min": None,  # 由用户手动停止; Ctrl+C 会保存完整断点
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


def unify_missing(df, price_cols):
    """把哨兵值统一为 NaN，保留原始缺失位置的掩码语义。

    - inf / -inf → NaN
    - 非数值列 (空字符串等) → 强制转数值，失败变 NaN
    - price_cols 中负值 → NaN (价格不可能为负)
    """
    out = df.copy()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    skip = {"date", "instrument"}
    for c in out.columns:
        if c in skip:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in price_cols:
        if c in out.columns:
            mask_neg = out[c] < 0
            if mask_neg.any():
                out.loc[mask_neg, c] = np.nan
    return out


"""数据层：合规预处理、流式统计、按日透视、分块取数与缓存。

平台查询通过可注入的 query_fn(sql, filters) -> DataFrame 隔离,
平台侧实现为 lambda sql, filters: dai.query(sql, filters=filters, compression=True).df()。
读取方式遵循官方 OOM 指引: 按股票分块、只 SELECT 所需列、逐块释放。
"""
import numpy as np
import pandas as pd


class RunningStats:
    """流式 mean/std（官方 OOM 指引方案），逐字段仅用有效值统计。"""

    def __init__(self, n_feat):
        self.n = np.zeros(n_feat, np.float64)  # 逐字段有效样本数
        self.s = np.zeros(n_feat, np.float64)
        self.ss = np.zeros(n_feat, np.float64)

    def update(self, x, mask=None):
        """x: (..., F); mask: (..., F) bool, None 时从 isfinite 推导。"""
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        if mask is None:
            mask = np.isfinite(x)
        else:
            mask = mask.reshape(-1, mask.shape[-1]).astype(bool)
        x = np.where(mask, x, 0.0)
        self.n += mask.sum(0).astype(np.float64)
        self.s += x.sum(0)
        self.ss += (x ** 2).sum(0)

    def finalize(self):
        mean = self.s / np.maximum(self.n, 1)
        var = self.ss / np.maximum(self.n, 1) - mean ** 2
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


def pivot_to_days(df, feature_cols, bars_per_day, close_col="close",
                  min_time_coverage=0.0):
    """单只股票的 bar 级 df → (dates, X, mask, day_close)。

    - ffill/bfill 填充缺失值 (局部合理, 优于全局均值填 0)
    - observed_mask 记录 ffill/bfill 前的真实缺失位置, 供 filter 做 mask-aware 加权
    - 缺 bar 的天以当日首 bar 左侧补齐; 多余 bar 取最后 bars_per_day 根
    """
    n_feat = len(feature_cols)
    empty = (np.array([], "datetime64[D]"),
             np.zeros((0, bars_per_day, n_feat), np.float32),
             np.zeros((0, bars_per_day, n_feat), bool),
             np.array([], np.float64))
    if len(df) == 0:
        return empty
    feats = df[feature_cols].to_numpy(np.float32)
    # 记录 ffill/bfill 前的真实缺失位置
    observed_before = np.isfinite(feats)
    # 跨日 ffill/bfill: 用相邻 bar 的局部值填充, 比全局均值更合理
    feats = pd.DataFrame(feats).ffill().bfill().to_numpy(np.float32)
    closes_all = df[close_col].to_numpy(np.float64)
    day = df["date"].dt.normalize().to_numpy().astype("datetime64[D]")
    starts = np.flatnonzero(np.concatenate([[True], day[1:] != day[:-1]]))
    ends = np.append(starts[1:], len(day))
    counts = ends - starts

    # 快速路径: 所有天都是标准 bar 数且无原始缺失
    if np.all(counts == bars_per_day) and observed_before.all():
        return (day[starts],
                feats.reshape(-1, bars_per_day, n_feat),
                np.ones((len(starts), bars_per_day, n_feat), bool),
                closes_all[ends - 1])

    dates, arrs, masks, closes = [], [], [], []
    for s, e in zip(starts, ends):
        a = feats[s:e]
        obs = observed_before[s:e]
        n_bars = len(a)

        # ffill/bfill 后仍有 NaN → 整列全空, 丢弃该日
        if not np.isfinite(a).all():
            continue

        if n_bars >= bars_per_day:
            a = a[-bars_per_day:]
            obs = obs[-bars_per_day:]
        else:
            pad_n = bars_per_day - n_bars
            a = np.concatenate([np.repeat(a[:1], pad_n, axis=0), a])
            obs = np.concatenate([np.zeros((pad_n, n_feat), dtype=bool), obs])

        dates.append(day[s])
        arrs.append(a)
        masks.append(obs)
        closes.append(closes_all[e - 1])
    if not dates:
        return empty
    return (np.array(dates), np.stack(arrs),
            np.stack(masks), np.array(closes, np.float64))


def iter_chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def list_instruments(query_fn, table, start, end):
    df = query_fn(f"SELECT DISTINCT instrument FROM {table}",
                  {"date": [start, end]})
    return sorted(df["instrument"].tolist())


def load_stock_arrays(query_fn, table, cfg, start, end, instruments):
    """分块查询 -> 缺失统一 -> 合规变换 -> 按日透视。每块用完即释放（官方 OOM 指引）。
    day_close 用于标签: 若表提供复权因子 (adjust_col), 用 close*adjust_factor
    计算复权收益, 避免分红送转污染标签; 复权因子不进入模型输入。"""
    import time as _time
    adj = cfg.get("adjust_col")
    extra = [adj] if adj and adj not in cfg["feature_cols"] else []
    cols = ", ".join(["date", "instrument"] + cfg["feature_cols"] + extra)
    sql = f"SELECT {cols} FROM {table} ORDER BY instrument, date"
    # Fallback SQL without adjust_col (inference table may lack it)
    cols_no_adj = ", ".join(["date", "instrument"] + cfg["feature_cols"])
    sql_no_adj = f"SELECT {cols_no_adj} FROM {table} ORDER BY instrument, date"
    out = {}
    min_cov = cfg.get("min_time_coverage", 0.95)
    chunks = list(iter_chunks(instruments, cfg["chunk_size"]))
    t_start = _time.time()
    for ci, chunk in enumerate(chunks):
        t_c = _time.time()
        try:
            df = query_fn(sql, {"date": [start, end], "instrument": list(chunk)})
        except Exception:
            if extra:
                # adjust_factor may not exist in inference table
                df = query_fn(sql_no_adj, {"date": [start, end], "instrument": list(chunk)})
                adj, extra, sql = None, [], sql_no_adj  # 后续 chunk 直接用无 adj 的 SQL
            else:
                raise
        n_rows = 0 if df is None else len(df)
        elapsed = _time.time() - t_start
        eta = elapsed / (ci + 1) * (len(chunks) - ci - 1)
        print(f"[data] chunk {ci + 1}/{len(chunks)} rows={n_rows} "
              f"query={_time.time() - t_c:.0f}s elapsed={elapsed:.0f}s "
              f"eta={eta:.0f}s", flush=True)
        if df is None or len(df) == 0:
            continue
        t_p = _time.time()
        if adj and adj in df.columns:
            raw_close = df["close"].astype(np.float64) * df[adj].astype(np.float64)
        else:
            raw_close = df["close"].copy()
        df = unify_missing(df, cfg["price_cols"])
        df = apply_field_transforms(df, cfg)
        df["_raw_close"] = raw_close
        for ins, sub in df.groupby("instrument", sort=False):
            dates, X, mask, close = pivot_to_days(
                sub, cfg["feature_cols"], cfg["bars_per_day"],
                close_col="_raw_close", min_time_coverage=min_cov)
            if len(dates):
                out[str(ins)] = (dates, X.astype(np.float16),
                                 mask, close)
        print(f"[data] chunk {ci + 1}/{len(chunks)} processed "
              f"proc={_time.time() - t_p:.0f}s stocks_total={len(out)}", flush=True)
        del df
    return out


def save_cache(path, arrays):
    import time as _time
    t0 = _time.time()
    print(f"[cache] saving {len(arrays)} stocks -> {path} ...", flush=True)
    flat = {}
    for ins, (dates, X, mask, close) in arrays.items():
        flat[f"{ins}::dates"] = dates
        flat[f"{ins}::X"] = X
        flat[f"{ins}::mask"] = mask
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
    return {n: (z[f"{n}::dates"], z[f"{n}::X"],
                z[f"{n}::mask"], z[f"{n}::close"]) for n in names}


"""标签：未来收益 -> (可选)风格残差 -> 逐日截面 winsorize + z-score。只在训练路径使用。

残差化依据: 平台评估会把分数对 BARRA 风格暴露截面回归取残差, 训练标签做同样处理
让模型直接学习被考核的目标（设计文档 v2 第 3 节）。exposure 仅用于标签, 不进模型输入。
"""
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


"""训练集组装：以交易日为 batch（截面损失与截面注意力的共同要求）。"""
import numpy as np


def split_days(days, train_frac):
    days = sorted(days)
    k = int(len(days) * train_frac)
    return days[:k], days[k:]


class DaySampleIndex:
    """arrays: {instrument: (dates, X (D,B,F) float16, mask (D,B,F) bool, day_close)};
    labels_df: DataFrame[date, instrument, label, residual_return]。

    batch(day) -> (X (N,L,B,F), observed_mask (N,L,B,F), time_mask (N,L),
                   y (N,), instruments)
    eval_batch(day) 额外返回未标准化 residual_return。

    observed_mask: 每个 (day, bar, feature) 是否真实有效
    time_mask:     每个 day 是否为真实交易日（True）vs 左侧 padding（False）
                   padded day 的 observed_mask 一定全 False，但反过来不成立
                   （真实日也可能部分字段缺失）。time_mask 是更粗粒度的信号，
                   专门防止 GRU / Attention 在 padding 位置上更新隐状态。

    保持 float16 到传入 GPU 后再转 float32，减少 CPU 内存复制和 PCIe 传输。"""

    def __init__(self, arrays, labels_df, cfg):
        self.arrays = arrays
        self.L = cfg["lookback_days"]
        self.n_feat = len(cfg["feature_cols"])
        self.bars = cfg["bars_per_day"]
        self.pos = {ins: {d: i for i, d in enumerate(dates)}
                    for ins, (dates, _, _, _) in arrays.items()}
        if "residual_return" not in labels_df.columns:
            raise ValueError("labels_df missing residual_return for validation Sharpe")
        by_day = {}
        columns = ["date", "instrument", "label", "residual_return"]
        for d, ins, v, raw_ret in labels_df[columns].itertuples(index=False):
            d = np.datetime64(d, "D")
            if ins in self.pos and d in self.pos[ins]:
                by_day.setdefault(d, []).append((ins, float(v), float(raw_ret)))
        # 截面太小的天丢弃（截面损失无意义）
        self.by_day = {d: v for d, v in by_day.items() if len(v) >= 2}
        self.days = sorted(self.by_day)

    def _window(self, ins, d):
        """返回 (X (L,B,F), observed_mask (L,B,F), time_mask (L,))。
        time_mask[i]=True 当第 i 天是真实交易日, False 当左侧 padding。"""
        _, X, obs_mask, _ = self.arrays[ins]
        i = self.pos[ins][d]
        lo = max(0, i + 1 - self.L)
        n_real = i + 1 - lo  # 真实交易日数
        w = X[lo:i + 1]
        m = obs_mask[lo:i + 1]
        tm = np.ones(n_real, dtype=bool)  # time_mask: 真实天 = True
        if len(w) < self.L:
            pad_n = self.L - len(w)
            pad_x = np.zeros((pad_n, self.bars, self.n_feat), dtype=X.dtype)
            pad_m = np.zeros((pad_n, self.bars, self.n_feat), dtype=bool)
            pad_tm = np.zeros(pad_n, dtype=bool)  # padding 天 = False
            w = np.concatenate([pad_x, w], axis=0)
            m = np.concatenate([pad_m, m], axis=0)
            tm = np.concatenate([pad_tm, tm], axis=0)
        return w, m, tm

    def batch(self, day):
        items = self.by_day[day]
        windows = [self._window(ins, day) for ins, _, _ in items]
        X = np.stack([w[0] for w in windows])
        obs_mask = np.stack([w[1] for w in windows])
        time_mask = np.stack([w[2] for w in windows])
        y = np.array([v for _, v, _ in items], np.float32)
        return X, obs_mask, time_mask, y, [ins for ins, _, _ in items]

    def eval_batch(self, day):
        items = self.by_day[day]
        windows = [self._window(ins, day) for ins, _, _ in items]
        X = np.stack([w[0] for w in windows])
        obs_mask = np.stack([w[1] for w in windows])
        time_mask = np.stack([w[2] for w in windows])
        y = np.array([v for _, v, _ in items], np.float32)
        residual_return = np.array([r for _, _, r in items], np.float64)
        return X, obs_mask, time_mask, y, residual_return, [ins for ins, _, _ in items]


"""模型：字段混合 -> 日内编码(ALiBi+global token) -> 日间GRU -> 逐时刻截面注意力(门控残差)
-> 末日query时序聚合 -> RankGLU打分头。

结构依据 docs/superpowers/specs/2026-07-26-bigalpha-e2e-model-design.md v2 第 1 节,
论文出处见 docs/papers/00-overview.md P0 清单。
"""
import torch
import torch.nn as nn


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def alibi_bias(n_heads, length, device):
    slopes = torch.tensor([2 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)],
                          device=device)
    pos = torch.arange(length, device=device)
    dist = (pos[None, :] - pos[:, None]).abs().float()
    return -slopes[:, None, None] * dist[None]


class MultiScaleCausalFilter(nn.Module):
    """多尺度因果逐字段滤波 (Depthwise Causal Conv1D)。

    对每个字段 f 和每个尺度 k，计算 mask-aware 因果加权平均 L^{(k)} 作为低频分量，
    通过可学习 gate 控制平滑/锐化程度:

      out = x + Σ_k tanh(g_k) · (L^{(k)} - x)

    - tanh(g) ∈ (-1, 1): 正=向平滑移动 (低通), 负=向锐化移动 (高通), 零=恒等
    - 初始化 g=0 → tanh(0)=0 → 滤波器起始为精确恒等变换
    - 逐字段 (groups=n_feat): 不同字段在这一层不混合
    - 因果 (left-pad k-1): t 时刻只使用 ≤t 的数据
    - mask-aware: 仅对真实观测加权, 缺失位置贡献 0
    - softmax 权重: w ≥ 0, Σw = 1 → 可解释为平滑滤波器
    """

    def __init__(self, n_feat, kernel_sizes=(3, 7, 15), eps=1e-5):
        super().__init__()
        self.n_feat = n_feat
        self.kernel_sizes = tuple(kernel_sizes)
        self.n_scales = len(kernel_sizes)
        self.max_k = max(kernel_sizes)
        self.eps = eps

        # 逐字段平滑核权重 → softmax → 非负、和为 1
        # 初始化为 0 → softmax 接近均匀分布 → 近似简单移动平均
        self.logits = nn.Parameter(torch.zeros(n_feat, self.max_k))

        # 逐字段逐尺度的 gate: tanh(g) ∈ (-1, 1) 控制平滑/锐化
        # 初始化为 0 → tanh(0)=0 → 精确恒等, 训练中逐步分化
        self.gate_raw = nn.Parameter(torch.zeros(n_feat, self.n_scales))

    def _causal_weighted_avg(self, x, mask, w, k):
        """Mask-aware 因果加权平均。

        x:    (N, B, F) float32 — 输入（NaN 已在 _normalize_to_device 填 0）
        mask: (N, B, F) bool    — observed_mask
        w:    (F, k) float32    — 非负权重, Σ_j w_j = 1
        返回: (N, B, F) — 过去 k 个有效值的加权平均
        """
        N, B, F_dim = x.shape
        # 因果 left-pad: k-1 个位置补零（沿 bar 维左侧）
        x_pad = F.pad(x, (0, 0, k - 1, 0))       # (N, B+k-1, F)
        m_pad = F.pad(mask.float(), (0, 0, k - 1, 0))

        # Unfold → (N, B, F, k) — kernel 维在最后
        x_win = x_pad.unfold(1, k, 1)
        m_win = m_pad.unfold(1, k, 1)

        # 广播逐字段权重: (F, k) → (1, 1, F, k)
        w_ = w.unsqueeze(0).unsqueeze(0)

        # 分子: Σ_j w_j · M_{t-j} · X_{t-j}  （沿 kernel 维求和）
        num = (x_win * w_ * m_win).sum(dim=-1)
        # 分母: Σ_j w_j · M_{t-j}, clamp(eps) 防除零
        den = (m_win * w_).sum(dim=-1).clamp(min=self.eps)

        return num / den

    def forward(self, x, observed_mask=None):
        """x: (N, B, F), observed_mask: (N, B, F) bool or None"""
        if observed_mask is None:
            observed_mask = torch.ones_like(x, dtype=torch.bool)

        # softmax → w_{f,j} ≥ 0, Σ_j w_{f,j} = 1
        w_all = F.softmax(self.logits, dim=-1)  # (F, max_k)
        # tanh gate ∈ (-1, 1), 0 = identity
        gate = torch.tanh(self.gate_raw)        # (F, n_scales)

        out = x  # 原始信号直通, gate 控制偏离程度

        for si, k in enumerate(self.kernel_sizes):
            wk = w_all[:, :k]  # (F, k)
            low = self._causal_weighted_avg(x, observed_mask, wk, k)
            g = gate[:, si].view(1, 1, -1)  # (1, 1, F)

            # g > 0 → 向平滑方向移动 (低通滤波)
            # g < 0 → 向锐化方向移动 (高通增强)
            # g = 0 → 精确恒等
            out = out + g * (low - x)

        return out


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
        # ALiBi is constant. Build the per-batch mask once and share it across
        # all encoder layers instead of rebuilding/repeating it per layer.
        mask = self.attn_bias.repeat(h.shape[0], 1, 1)
        for lyr in self.layers:
            h = lyr(h, mask)
        return self.out_norm(h[:, 0])


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
    """以最新一天为 query 的时序注意力加权 (MASTER/DTML)。"""

    def __init__(self, d):
        super().__init__()
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, h):  # (N, tau, d) -> (N, d)
        w = torch.softmax((self.q(h[:, -1:]) @ self.k(h).transpose(1, 2)) * self.scale, -1)
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
        k_sizes = m.get("filter_kernel_sizes")
        self.filter = MultiScaleCausalFilter(n_feat, k_sizes) if k_sizes else None
        self.fieldmix = FieldMix(n_feat)
        self.intra = IntradayEncoder(n_feat, m["d_intra"], m["heads_intra"],
                                     m["layers_intra"], m["ffn_intra"],
                                     cfg["bars_per_day"], m["dropout"])
        self.inter = InterDayEncoder(m["d_intra"], m["d_day"])
        self.cross = CrossSectionBlock(m["d_day"], m["heads_cross"], m["ffn_cross"],
                                       m["dropout"], m["gate_init"])
        self.agg = TemporalAggregator(m["d_day"])
        self.head = RankGLUHead(m["d_day"], m["glu_bottleneck"])

    def _intra_block(self, part, part_mask):
        """Filter + FieldMix + IntradayEncoder as one checkpointable unit."""
        if self.filter is not None and part_mask is not None:
            part = self.filter(part, part_mask)
        return self.intra(self.fieldmix(part))

    def encode_days(self, x, observed_mask=None):  # (N, L, B, F) -> (N, L, d_day)
        N, L, B, F_dim = x.shape
        flat = x.reshape(N * L, B, F_dim)
        if observed_mask is not None:
            flat_mask = observed_mask.reshape(N * L, B, F_dim)
        else:
            flat_mask = None
        chunks = []
        for start in range(0, len(flat), self.intra_chunk_size):
            part = flat[start:start + self.intra_chunk_size]
            part_mask = (flat_mask[start:start + self.intra_chunk_size]
                         if flat_mask is not None else None)
            # Gradient checkpointing: 512 stock-days → discard activations,
            # recompute during backward.  Drops peak memory ~6× for large N.
            chunks.append(torch.utils.checkpoint.checkpoint(
                self._intra_block, part, part_mask,
                use_reentrant=False))
        v = torch.cat(chunks, dim=0).reshape(N, L, -1)
        return self.inter(v)

    def score_day(self, H):  # (N, tau, d_day) -> (N,)
        if self.use_cross:
            H = self.cross(H.transpose(0, 1)).transpose(0, 1)
        return self.head(self.agg(H))

    def forward(self, x, observed_mask=None):
        H = self.encode_days(x, observed_mask)
        return self.score_day(H[:, -self.tau:])


"""损失与验证指标。

损失设计依据 docs/papers/00-overview.md 第三节 P0-6:
- 主项 1−dailyIC 直接对齐评分的 IC 均值 (预测/标签均截面标准化时 MSE≅2(1−IC), RankGLU 恒等式);
- 小权重 MSE 为尺度锚;
- 小权重 ListNet listwise 项聚焦截面头部, 实证改善多空组合 SR (论文8/ListNet)。
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as _sp_stats


def daily_ic_loss(pred, y, eps=1e-8):
    p = pred - pred.mean()
    t = y - y.mean()
    ic = (p * t).sum() / (p.norm() * t.norm() + eps)
    return 1.0 - ic


def listnet_loss(pred, y, temp):
    return -(torch.softmax(y / temp, 0) * torch.log_softmax(pred / temp, 0)).sum()


def total_loss(pred, y, cfg):
    l = cfg["loss"]
    out = daily_ic_loss(pred, y) + l["mse_w"] * F.mse_loss(pred, y)
    if l["listnet_w"] > 0:
        out = out + l["listnet_w"] * listnet_loss(pred, y, l["listnet_temp"])
    return out


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


"""训练入口。平台上直接 `python train.py`；不接收平台注入的 start/end（防泄漏）。

训练区间为开发表全部可用交易日, 前 80% 训练 / 后 20% 验证（相对切分, 私榜重训自动适配）。
选模/早停用验证集复合指标 (IC均值+IC_IR+多空夏普, WaveLSFormer 准则), 不用验证损失。
SWA 候选与最优单点权重在验证集上二选一 (TIPS)。
"""
import copy
import os
import random
import sys
import time

import numpy as np
import torch


# 开发表可用区间的宽松包络 (窄于哨兵区间, 利于引擎按日期裁剪);
# 实际训练/验证切分仍是"取表内全部可用交易日的前80%/后20%"的相对逻辑
TRAIN_START, TRAIN_END = "2019-01-01 00:00:00", "2025-12-31 23:59:59"
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
    out = out.sub_(mean).div_(std)
    # 标准化后缺失位置填 0 ≈ 训练集均值；±inf 同样填 0 防御
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _evaluate(model, index, days, torch_normalizer, device, metric_w,
              progress_label="val", log_every=20):
    model.eval()
    daily_labels = []
    daily_returns = []
    started = time.time()
    total = len(days)
    progress = _ProgressBar(progress_label, total, log_every)
    try:
        with torch.inference_mode():
            for step, d in enumerate(days, 1):
                X, obs_mask, time_mask, y, residual_return, _ = index.eval_batch(d)
                X = _normalize_to_device(X, torch_normalizer, device)
                pred = model(X, observed_mask=torch.from_numpy(obs_mask).to(device)).cpu().numpy()
                if np.std(pred) >= 1e-9 and np.std(y) >= 1e-9:
                    daily_labels.append((pred, y))
                    daily_returns.append((pred, residual_return))
                elapsed = time.time() - started
                eta = elapsed / step * (total - step)
                progress.update(
                    step,
                    f" | N={len(y)} valid={len(daily_labels)}"
                    f" | {_format_duration(elapsed)} ETA {_format_duration(eta)}"
                    f" |{_gpu_memory_compact(device)}",
                )
    finally:
        progress.close()
    ics = [rank_ic(p, y) for p, y in daily_labels]
    ics = [v for v in ics if np.isfinite(v)]
    if not ics:
        return {"ic": 0.0, "icir": 0.0, "sharpe": 0.0, "pred_std": 0.0, "score": -1e9}
    sharpe = long_short_sharpe(daily_returns)
    return {"ic": float(np.mean(ics)),
            "icir": float(np.mean(ics) / (np.std(ics) + 1e-9)),
            "sharpe": sharpe,
            "pred_std": float(np.mean([np.std(p) for p, _ in daily_labels])),
            "score": composite_score(ics, sharpe, metric_w)}


def run_training(cfg, query_fn, exposure_query_fn=None, universe_query_fn=None,
                 device=None):
    validate_config(cfg)
    set_seed(cfg["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---- 数据（缓存优先; 缓存损坏时自动重建） ----
    arrays = None
    if os.path.exists(cfg["cache_path"]):
        try:
            arrays = load_cache(cfg["cache_path"])
            print(f"[data] cache hit: {cfg['cache_path']}")
        except Exception as e:
            print(f"[data] cache corrupt ({e!r}), rebuilding ...", flush=True)
            arrays = None
    if arrays is None:
        instruments = list_instruments(query_fn, cfg["table"], TRAIN_START, TRAIN_END)
        print(f"[data] querying {len(instruments)} instruments in chunks of "
              f"{cfg['chunk_size']} ...")
        arrays = load_stock_arrays(query_fn, cfg["table"], cfg,
                                   TRAIN_START, TRAIN_END, instruments)
        save_cache(cfg["cache_path"], arrays)
    print(f"[data] stocks={len(arrays)} elapsed={time.time() - t0:.0f}s")

    # ---- 标签 ----
    exposure_df = None
    if exposure_query_fn is not None and cfg["label"]["mode"] == "residual":
        exposure_df = exposure_query_fn()
        print(f"[label] exposure rows={len(exposure_df)}", flush=True)
    universe_df = None
    if universe_query_fn is not None:
        universe_df = universe_query_fn()
        print(f"[universe] official rows={len(universe_df)}", flush=True)
    t_l = time.time()
    labels = build_label_panel({k: (v[0], v[3]) for k, v in arrays.items()},
                               exposure_df, cfg, universe_df)
    print(f"[label] panel rows={len(labels)} in {time.time() - t_l:.0f}s", flush=True)
    index = DaySampleIndex(arrays, labels, cfg)
    train_days, val_days = split_days(index.days, cfg["train_frac"])
    print(f"[data] days train={len(train_days)} val={len(val_days)}", flush=True)

    # ---- 标准化统计（仅训练区间的按股数组, 合规: 参数只来自训练集;
    #      每个交易日计入一次, 不因窗口重叠重复计权, 且比逐日组 batch 快两个量级） ----
    t_s = time.time()
    rs = RunningStats(len(cfg["feature_cols"]))
    val_start = val_days[0] if val_days else None
    for dates, X, mask, _ in arrays.values():
        Xm = X if val_start is None else X[dates < val_start]
        Mm = mask if val_start is None else mask[dates < val_start]
        if len(Xm):
            rs.update(Xm, Mm)
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
                "val_metrics": best["metrics"], "metric_version": 2,
                "fit_date_range": [str(train_days[0]), str(train_days[-1])]}
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
                     epoch_elapsed=0.0, complete=False, announce=True):
        state = {
            "format_version": 2,
            "complete": complete,
            "phase": phase,
            "epoch": int(epoch),
            "order": None if order is None else np.asarray(order),
            "next_step": int(next_step),
            "tot": float(tot),
            "nb": int(nb),
            "epoch_elapsed": float(epoch_elapsed),
            "elapsed_s": float(_total_elapsed()),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
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
            "metric_version": 2,
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
            swa_model.load_state_dict(saved["swa_state_dict"])
            swa_updated = bool(saved["swa_updated"])
            best = saved["best"]
            bad_epochs = int(saved["bad_epochs"])
            if saved.get("metric_version", 1) != 2 and best["state"] is not None:
                print("[resume] validation Sharpe upgraded to real residual returns; "
                      "old best will be re-evaluated before training resumes",
                      flush=True)
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
              "with unstandardized residual returns", flush=True)
        migrated_metrics = _evaluate(
            model, index, val_days, torch_normalizer, device, metric_w,
            progress_label=f"revalue best e{legacy_epoch + 1}",
            log_every=log_every,
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
        "epoch_elapsed": resume_epoch_elapsed,
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
                prior_epoch_elapsed = resume_epoch_elapsed if continuing else 0.0
                te = time.time()
                n_steps = len(order)
                active.update(epoch=ep, phase="train", order=order,
                              next_step=next_step, tot=tot, nb=nb,
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
                        X, obs_mask, time_mask, y, _ = index.batch(train_days[j])
                        if ratio < 1.0 and len(y) > 20:
                            keep = rng.choice(
                                len(y), max(20, int(len(y) * ratio)), replace=False
                            )
                            X, obs_mask, time_mask, y = X[keep], obs_mask[keep], time_mask[keep], y[keep]
                        X = _normalize_to_device(X, torch_normalizer, device)
                        y = torch.from_numpy(y).to(device, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        loss = total_loss(
                            model(X, observed_mask=torch.from_numpy(obs_mask).to(device)),
                            y, cfg)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg["train"]["clip"]
                        )
                        opt.step()
                        loss_value = loss.detach().item()
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
                                         epoch_elapsed)
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
                             epoch_elapsed=epoch_elapsed)
            else:
                tot, nb = resume_tot, resume_nb
                epoch_elapsed = resume_epoch_elapsed
                active.update(epoch=ep, phase="validation", order=None,
                              next_step=0, tot=tot, nb=nb,
                              epoch_elapsed=epoch_elapsed)

            print(f"[epoch {ep + 1}/{cfg['train']['epochs']}] validation start "
                  f"days={len(val_days)}", flush=True)
            m = _evaluate(model, index, val_days, torch_normalizer, device, metric_w,
                          progress_label=f"val {ep + 1}/{cfg['train']['epochs']}",
                          log_every=log_every)
            full_epoch_elapsed = epoch_elapsed
            print(f"[epoch {ep + 1}/{cfg['train']['epochs']}] complete "
                  f"loss={tot / max(nb, 1):.4f} "
                  f"ic={m['ic']:.4f} icir={m['icir']:.2f} "
                  f"sharpe={m['sharpe']:.2f} pred_std={m['pred_std']:.4f} "
                  f"score={m['score']:.4f} "
                  f"train_elapsed={_format_duration(full_epoch_elapsed)}",
                  flush=True)
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
            resume_epoch_elapsed = 0.0
            active.update(epoch=ep + 1, phase="train", order=None,
                          next_step=0, tot=0.0, nb=0, epoch_elapsed=0.0)
    except KeyboardInterrupt:
        try:
            _save_resume(active["epoch"], active["phase"], active["order"],
                         active["next_step"], active["tot"], active["nb"],
                         active["epoch_elapsed"])
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
                          metric_w, progress_label="swa", log_every=log_every)
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
    cfg = ckpt["config"]
    field_order = cfg["feature_cols"]
    transform_methods = {}
    for c in cfg["price_cols"]:
        if c in field_order:
            transform_methods[c] = "log"
    for c in cfg["vol_cols"]:
        if c in field_order:
            transform_methods[c] = "log1p"
    artifact = {
        "format_version": 1,
        "artifact_type": "bigalpha_pytorch_state_dict",
        "state_dict": encoded_state,
        "field_order": field_order,
        "transform": transform_methods,
        "mean": np.asarray(ckpt["mean"], np.float32).tolist(),
        "std": np.asarray(ckpt["std"], np.float32).tolist(),
        "epsilon": 1e-6,
        "fit_date_range": ckpt.get("fit_date_range", []),
        "config": cfg,
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
    """读取 JSON 训练产物并还原为 run_inference 使用的 checkpoint。

    向后兼容: 旧版产物缺少 field_order/transform/fit_date_range/epsilon 时
    不报错，从 config 推导 field_order 和 transform。"""
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
        "field_order": artifact.get("field_order",
                                    artifact["config"]["feature_cols"]),
        "transform": artifact.get("transform"),
        "fit_date_range": artifact.get("fit_date_range", []),
        "epsilon": artifact.get("epsilon", 1e-6),
    }


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
