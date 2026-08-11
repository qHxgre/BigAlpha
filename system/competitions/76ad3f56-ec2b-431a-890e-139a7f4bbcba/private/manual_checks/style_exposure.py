"""检查处理后因子是否仍残留 BARRA 风格暴露。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .common import show
from .config import CONFIG, EXPOSURE_COLUMNS, PATHS, STYLE_COLUMNS, CheckPaths
from .exposure import (
    calculate_residual,
    max_finite,
    query_exposures,
    safe_correlation,
)


@dataclass(frozen=True)
class StyleExposureCheckResult:
    """风格暴露检查结果。"""

    summary: pd.DataFrame
    exposure_summary: pd.DataFrame
    daily: pd.DataFrame


def analyze_style_exposure(
    paths: CheckPaths = PATHS,
    start_date: str = CONFIG.start_date,
    end_date: str = CONFIG.end_date,
    *,
    max_abs_style_corr: float = CONFIG.max_abs_style_correlation,
    max_regression_r2: float = CONFIG.max_style_regression_r2,
    sample_interval: int = 5,
    progress_every: int = 10,
    display: bool = True,
) -> StyleExposureCheckResult:
    """检查因子池的风格暴露残留。

    检查逻辑与正式预处理一致：将缺失暴露填为 0，并按交易日以十个 BARRA
    风格因子和行业哑变量做截面回归。结果同时给出原因子与各风格因子的
    Pearson 相关性、全暴露回归 R²，以及回归残差的最大绝对相关性。

    Parameters
    ----------
    paths:
        私榜人工复核路径。
    start_date, end_date:
        检查周期，格式为 ``YYYY-MM-DD``，首尾日期均包含。
    max_abs_style_corr:
        单日最大绝对风格相关性的告警阈值。
    max_regression_r2:
        单日全暴露回归 R² 的告警阈值。
    sample_interval:
        每隔多少个交易日抽取一个截面；同时始终保留首日、末日和每月最后一个
        交易日。告警因子也不会自动回退到全交易日检查。
    progress_every:
        每处理多少个抽样交易日输出一次进度；设为 0 时不输出逐日进度。
    display:
        是否在 Notebook 中展示汇总结果。
    """
    started_at = perf_counter()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(f"start_date 不能晚于 end_date: {start_date} > {end_date}")
    if sample_interval < 1:
        raise ValueError(f"sample_interval 必须大于等于 1，实际为 {sample_interval}")
    if progress_every < 0:
        raise ValueError(f"progress_every 不能小于 0，实际为 {progress_every}")

    stage_started_at = perf_counter()
    print(f"[风格暴露] 1/5 读取因子池: {paths.factor_pool_path}", flush=True)
    pool = pd.read_parquet(paths.factor_pool_path)
    required = {"date", "instrument"}
    missing = required.difference(pool.columns)
    if missing:
        raise ValueError(f"factor_pool 缺少必要列: {sorted(missing)}")

    factor_columns = [column for column in pool.columns if column not in required]
    if not factor_columns:
        raise ValueError("factor_pool 中没有可检查的因子列")

    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool = pool.loc[pool["date"].between(start, end)].copy()
    if pool.empty:
        raise ValueError(f"factor_pool 在 {start_date} ~ {end_date} 内没有数据")

    all_dates = pd.DatetimeIndex(pool["date"].drop_duplicates().sort_values())
    month_end_dates = (
        pd.Series(all_dates, index=all_dates)
        .groupby(all_dates.to_period("M"))
        .max()
    )
    sampled_dates = pd.DatetimeIndex(all_dates[::sample_interval]).union(
        pd.DatetimeIndex(month_end_dates)
    ).union(pd.DatetimeIndex([all_dates[0], all_dates[-1]])).sort_values()
    pool = pool.loc[pool["date"].isin(sampled_dates)].copy()
    print(
        f"[风格暴露] 因子池读取完成: {len(pool):,} 行，{len(factor_columns)} 个因子，"
        f"抽样 {len(sampled_dates)}/{len(all_dates)} 个交易日，"
        f"耗时 {perf_counter() - stage_started_at:.1f}s",
        flush=True,
    )

    stage_started_at = perf_counter()
    print(
        f"[风格暴露] 2/5 查询 BARRA 暴露: {start_date} ~ {end_date}",
        flush=True,
    )
    exposures = query_exposures(start_date, end_date)
    exposures["date"] = pd.to_datetime(exposures["date"]).dt.normalize()
    exposures = exposures.loc[exposures["date"].isin(sampled_dates)].copy()
    print(
        f"[风格暴露] 暴露查询完成: {len(exposures):,} 行，"
        f"耗时 {perf_counter() - stage_started_at:.1f}s",
        flush=True,
    )

    exposure_columns = list(EXPOSURE_COLUMNS)
    missing_exposures = set(EXPOSURE_COLUMNS).difference(exposures.columns)
    if missing_exposures:
        raise ValueError(f"暴露数据缺少必要列: {sorted(missing_exposures)}")

    stage_started_at = perf_counter()
    print("[风格暴露] 3/5 合并因子池与暴露数据", flush=True)
    merged = pool.merge(
        exposures[["date", "instrument", *exposure_columns]],
        how="left",
        on=["date", "instrument"],
        validate="many_to_one",
    )
    merged[exposure_columns] = merged[exposure_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    print(
        f"[风格暴露] 数据合并完成: {len(merged):,} 行，"
        f"耗时 {perf_counter() - stage_started_at:.1f}s",
        flush=True,
    )

    daily_rows: list[dict] = []
    correlation_rows: list[dict] = []
    grouped_dates = merged.groupby("date", sort=True)
    total_sampled_days = grouped_dates.ngroups
    compute_started_at = perf_counter()
    print(
        f"[风格暴露] 4/5 开始计算: {total_sampled_days} 个抽样交易日",
        flush=True,
    )
    for completed_days, (date, date_df) in enumerate(grouped_dates, start=1):
        x = date_df[exposure_columns].to_numpy(dtype=float)
        x_finite = np.isfinite(x).all(axis=1)

        for factor in factor_columns:
            y = pd.to_numeric(date_df[factor], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y) & x_finite
            sample_count = int(mask.sum())
            if sample_count < 5:
                daily_rows.append({
                    "date": date, "submission_id": str(factor), "sample_count": sample_count,
                    "regression_r2": np.nan, "max_abs_style_corr": np.nan,
                    "max_abs_residual_corr": np.nan,
                })
                continue

            valid_y = y[mask]
            valid_x = x[mask]
            residual = calculate_residual(valid_y, valid_x)
            total_ss = float(np.sum((valid_y - valid_y.mean()) ** 2))
            residual_ss = float(np.sum(residual ** 2))
            r2 = np.nan if total_ss == 0 else 1.0 - residual_ss / total_ss

            raw_correlations = []
            residual_correlations = []
            for index, exposure in enumerate(exposure_columns):
                raw_corr = (
                    safe_correlation(valid_y, valid_x[:, index])
                    if exposure in STYLE_COLUMNS
                    else np.nan
                )
                residual_corr = safe_correlation(residual, valid_x[:, index])
                if exposure in STYLE_COLUMNS:
                    raw_correlations.append(abs(raw_corr) if np.isfinite(raw_corr) else np.nan)
                    correlation_rows.append({
                        "date": date,
                        "submission_id": str(factor),
                        "style": exposure,
                        "correlation": raw_corr,
                        "abs_correlation": abs(raw_corr) if np.isfinite(raw_corr) else np.nan,
                    })
                residual_correlations.append(
                    abs(residual_corr) if np.isfinite(residual_corr) else np.nan
                )

            daily_rows.append({
                "date": date,
                "submission_id": str(factor),
                "sample_count": sample_count,
                "regression_r2": r2,
                "max_abs_style_corr": max_finite(raw_correlations),
                "max_abs_residual_corr": max_finite(residual_correlations),
            })

        if progress_every and (
            completed_days == 1
            or completed_days % progress_every == 0
            or completed_days == total_sampled_days
        ):
            elapsed = perf_counter() - compute_started_at
            remaining = (
                elapsed / completed_days * (total_sampled_days - completed_days)
            )
            print(
                f"[风格暴露] 计算进度 {completed_days}/{total_sampled_days} "
                f"({completed_days / total_sampled_days:.1%})，"
                f"当前日期 {date:%Y-%m-%d}，已用 {elapsed:.1f}s，"
                f"预计剩余 {remaining:.1f}s",
                flush=True,
            )

    print("[风格暴露] 5/5 汇总检查结果", flush=True)
    daily = pd.DataFrame(daily_rows)
    correlations = pd.DataFrame(correlation_rows)
    summary = daily.groupby("submission_id", as_index=False).agg(
        trading_days=("date", "nunique"),
        valid_days=("regression_r2", "count"),
        median_sample_count=("sample_count", "median"),
        mean_regression_r2=("regression_r2", "mean"),
        p95_regression_r2=("regression_r2", lambda value: value.quantile(CONFIG.style_quantile)),
        mean_max_abs_style_corr=("max_abs_style_corr", "mean"),
        p95_max_abs_style_corr=("max_abs_style_corr", lambda value: value.quantile(CONFIG.style_quantile)),
        max_abs_residual_corr=("max_abs_residual_corr", "max"),
    )
    summary["style_exposure_warning"] = (
        (summary["p95_max_abs_style_corr"] > max_abs_style_corr)
        | (summary["p95_regression_r2"] > max_regression_r2)
    )
    summary = summary.sort_values(
        ["style_exposure_warning", "p95_max_abs_style_corr", "p95_regression_r2"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    exposure_summary = correlations.groupby(
        ["submission_id", "style"], as_index=False
    ).agg(
        valid_days=("correlation", "count"),
        mean_correlation=("correlation", "mean"),
        mean_abs_correlation=("abs_correlation", "mean"),
        p95_abs_correlation=("abs_correlation", lambda value: value.quantile(CONFIG.style_quantile)),
        max_abs_correlation=("abs_correlation", "max"),
    ).sort_values(
        ["submission_id", "p95_abs_correlation"], ascending=[True, False]
    ).reset_index(drop=True)

    print(
        f"[风格暴露] 完成: 因子数 {len(factor_columns)}，"
        f"抽样交易日 {len(sampled_dates)}/{len(all_dates)}，"
        f"日期范围 {start_date} ~ {end_date}，"
        f"告警因子数 {int(summary['style_exposure_warning'].sum())}，"
        f"总耗时 {perf_counter() - started_at:.1f}s",
        flush=True,
    )
    if display:
        show(summary, exposure_summary)
    return StyleExposureCheckResult(summary, exposure_summary, daily)
