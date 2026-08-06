"""未来函数（look-ahead bias）检测：切窗一致性复算比对。

原理
----
一个无未来函数的因子，factor(T) 只依赖 <= T 的数据。因此无论评估窗口的 end_date 取全窗 E
还是取某个中间截断日 cutoff（cutoff < E），只要保持 start_date 与 warmup 不变，两次运行在
date <= cutoff 的格子上，因子取值都应当**完全一致**。

反之，若因子在 T 日偷看了 T 之后 h 天的数据，把 end_date 砍到 cutoff 后，`[cutoff-h, cutoff]`
这段因子值会因为「未来数据被砍掉」而发生变化，于是在尾部出现差异，即为未来函数的直接证据；
差异最早出现的日期到 cutoff 的跨度还能反推出泄漏跨度 h。

本模块只做**纯比对**（不触碰 dai / 不做评估 / 不依赖 structlog / bigmodule），供评测主进程
（sfa.py）import：切窗子进程只负责跑 main 落盘全窗/截窗两份 raw factor，主进程读进来调
detect_lookahead 得判定。因此这段代码内聚在评测目录，不进对用户公开的 bigalpha_eval 包。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_KEY_COLS = ("date", "instrument")


def _factor_col(df: pd.DataFrame) -> str:
    """取 date/instrument 之外的唯一因子列名；用户输出恒为单因子（eval.run 里已约束）。"""
    cols = [c for c in df.columns if c not in _KEY_COLS]
    if not cols:
        raise ValueError("未找到因子列（date/instrument 之外）")
    return cols[0]


def _prep(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """规范化 date/instrument/factor 三列，供两份数据按键对齐。"""
    out = df[[*_KEY_COLS, factor_col]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str)
    out = out.rename(columns={factor_col: "factor"})
    out["factor"] = pd.to_numeric(out["factor"], errors="coerce")
    return out


def pick_cutoffs(dates, ratios=(0.5, 0.75)) -> list:
    """从全窗 raw factor 的交易日序列里，按分位取截断日（落在真实交易日上）。

    Args:
        dates: 全窗因子输出里出现过的 date 序列（可含重复，任意可转 datetime 的类型）。
        ratios: 截断分位，默认 50%/75%。

    Returns:
        list[str]: 去重升序后的截断日 'YYYY-MM-DD'，已剔除落在最后一个交易日的分位
        （截断日等于末日则无「被砍区间」，起不到检测作用）；交易日 < 2 时返回空。
    """
    uniq = pd.Index(
        pd.to_datetime(pd.Series(list(dates))).dt.normalize().unique()
    ).sort_values()
    n = len(uniq)
    if n < 2:
        return []
    picked = set()
    for r in ratios:
        idx = int(round(r * (n - 1)))
        idx = min(max(idx, 0), n - 2)  # 不取末日，保证 cutoff 后至少还有一个交易日被砍掉
        picked.add(pd.Timestamp(uniq[idx]))
    return [str(d.date()) for d in sorted(picked)]


def detect_lookahead(
    raw_full: pd.DataFrame,
    raw_cut: pd.DataFrame,
    cutoff: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    min_diff_ratio: float = 1e-4,
    sample_size: int = 50,
) -> dict:
    """比对全窗与截断窗运行结果，判定是否存在未来函数。

    在 date <= cutoff 的**全部 (date, instrument) 并集**上比对（两侧 outer join）：这段
    两次运行的输入起点与 warmup 完全相同，无未来函数时行集合与因子值都应逐格一致。以下任一
    情况计为「差异格」：
        - 两侧都是有限值但超出容差（|a-b| > atol + rtol*|b|）；
        - 一侧有值、另一侧为 NaN 或缺行（截断后取不到值，说明原本依赖了未来数据）。

    差异格占比超过 min_diff_ratio 即判为泄漏（min_diff_ratio 用于吸收极个别浮点末位抖动，
    不是给未来函数留口子——真正的未来函数差异比例远高于此）。

    Args:
        raw_full: 全窗 main(ds, start, end) 输出（含 date/instrument + 单因子列）。
        raw_cut:  截断 main(ds, start, cutoff) 输出，列结构同上。
        cutoff:   截断日 'YYYY-MM-DD'，只比对 date <= cutoff 的格子。
        rtol/atol: 浮点比对容差，语义同 numpy.isclose。
        min_diff_ratio: 判为泄漏所需的最小差异格占比。
        sample_size: 返回的差异样本条数上限，便于排查。

    Returns:
        dict: leaked / overlap_cells / diff_cells / diff_ratio / max_abs_dev /
              first_diff_date / leak_horizon_days / cutoff / sample。
    """
    cutoff_ts = pd.Timestamp(cutoff).normalize()

    full = _prep(raw_full, _factor_col(raw_full))
    cut = _prep(raw_cut, _factor_col(raw_cut))

    full = full[full["date"] <= cutoff_ts]
    cut = cut[cut["date"] <= cutoff_ts]
    merged = pd.merge(
        full, cut, how="outer", on=list(_KEY_COLS), suffixes=("_full", "_cut")
    )

    overlap_cells = int(len(merged))
    if overlap_cells == 0:
        # 无格可比无法判定：两侧在 cutoff 前都没有任何行，多半是数据异常，保守放行。
        return {
            "leaked": False,
            "overlap_cells": 0,
            "diff_cells": 0,
            "diff_ratio": 0.0,
            "max_abs_dev": 0.0,
            "first_diff_date": None,
            "leak_horizon_days": None,
            "cutoff": str(cutoff_ts.date()),
            "sample": [],
        }

    a = merged["factor_full"].to_numpy(dtype=float)
    b = merged["factor_cut"].to_numpy(dtype=float)

    a_nan, b_nan = np.isnan(a), np.isnan(b)
    both_nan = a_nan & b_nan
    both_finite = ~a_nan & ~b_nan

    # 差异格：两侧都有值但超容差，或一侧有值另一侧缺失（both_nan 视为一致）
    close = np.zeros(overlap_cells, dtype=bool)
    close[both_finite] = np.isclose(a[both_finite], b[both_finite], rtol=rtol, atol=atol)
    diff_mask = ~close & ~both_nan

    diff_cells = int(diff_mask.sum())
    diff_ratio = diff_cells / overlap_cells

    abs_dev = np.full(overlap_cells, np.nan, dtype=float)
    abs_dev[both_finite] = np.abs(a[both_finite] - b[both_finite])
    max_abs_dev = float(np.nanmax(abs_dev)) if both_finite.any() else 0.0

    leaked = diff_ratio > min_diff_ratio

    first_diff_date = None
    leak_horizon_days = None
    sample: list[dict] = []
    if diff_cells > 0:
        diff_df = merged.loc[diff_mask, ["date", "instrument", "factor_full", "factor_cut"]].copy()
        diff_df["abs_dev"] = np.abs(diff_df["factor_full"] - diff_df["factor_cut"])
        first_diff_ts = diff_df["date"].min()
        first_diff_date = str(first_diff_ts.date())
        leak_horizon_days = int((cutoff_ts - first_diff_ts).days)
        # 样本优先给最靠近 cutoff、偏差最大的差异格，便于人工核对泄漏位置
        top = diff_df.sort_values(["date", "abs_dev"], ascending=[False, False]).head(sample_size)
        sample = [
            {
                "date": str(pd.Timestamp(r.date).date()),
                "instrument": str(r.instrument),
                "full": None if pd.isna(r.factor_full) else float(r.factor_full),
                "cut": None if pd.isna(r.factor_cut) else float(r.factor_cut),
                "abs_dev": None if pd.isna(r.abs_dev) else float(r.abs_dev),
            }
            for r in top.itertuples(index=False)
        ]

    return {
        "leaked": bool(leaked),
        "overlap_cells": overlap_cells,
        "diff_cells": diff_cells,
        "diff_ratio": float(diff_ratio),
        "max_abs_dev": max_abs_dev,
        "first_diff_date": first_diff_date,
        "leak_horizon_days": leak_horizon_days,
        "cutoff": str(cutoff_ts.date()),
        "sample": sample,
    }
