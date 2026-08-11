"""检查处理后因子是否仍残留 BARRA 风格暴露。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import show
from .config import CheckPaths


STYLE_COLUMNS = (
    "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
    "EARNYILD", "GROWTH", "LEVERAGE",
)

INDUSTRY_COLUMNS = (
    "AGRIFOREST", "MINING", "CHEM", "IRONSTEEL", "NONFERMETAL", "ELECTRONICS",
    "AUTO", "HOUSEAPP", "FOODBEVER", "TEXTILE", "LIGHTINDUS", "HEALTH",
    "UTILITIES", "TRANSPORTATION", "REALESTATE", "COMMETRADE", "LEISERVICE",
    "BANK", "NONBANKFINAN", "CONGLOMERATES", "CONMAT", "BUILDDECO", "ELECEQP",
    "AERODEF", "COMPUTER", "MEDIA", "TELECOM", "COAL", "PETRO", "ENVP", "BEAUTY",
)


@dataclass(frozen=True)
class StyleExposureCheckResult:
    """风格暴露检查结果。"""

    summary: pd.DataFrame
    exposure_summary: pd.DataFrame
    daily: pd.DataFrame


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """与评测预处理一致，加入截距后用最小二乘回归取残差。"""
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 5:
        return np.nan
    x = left[mask]
    y = right[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _query_exposures(start_date: str, end_date: str) -> pd.DataFrame:
    """查询与 ``DataProcess.neutralize`` 相同的风格和行业数据。"""
    try:
        import dai
    except ImportError as exc:  # pragma: no cover - 本地环境通常没有云端数据依赖
        raise RuntimeError("风格暴露检查需要在安装了 dai 的 BigQuant 云端环境运行") from exc

    columns = ",\n            ".join((*STYLE_COLUMNS, *INDUSTRY_COLUMNS))
    sql = f"""
        SELECT
            date,
            instrument,
            {columns}
        FROM bigalpha_2026_exposure
    """
    return dai.query(sql, filters={"date": [start_date, end_date]}).df()


def analyze_style_exposure(
    paths: CheckPaths,
    *,
    max_abs_style_corr: float = 0.10,
    max_regression_r2: float = 0.10,
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
    max_abs_style_corr:
        单日最大绝对风格相关性的告警阈值。
    max_regression_r2:
        单日全暴露回归 R² 的告警阈值。
    display:
        是否在 Notebook 中展示汇总结果。
    """
    pool = pd.read_parquet(paths.factor_pool_path).copy()
    required = {"date", "instrument"}
    missing = required.difference(pool.columns)
    if missing:
        raise ValueError(f"factor_pool 缺少必要列: {sorted(missing)}")

    factor_columns = [column for column in pool.columns if column not in required]
    if not factor_columns:
        raise ValueError("factor_pool 中没有可检查的因子列")

    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    start_date = pool["date"].min().strftime("%Y-%m-%d")
    end_date = pool["date"].max().strftime("%Y-%m-%d")
    exposures = _query_exposures(start_date, end_date)
    exposures["date"] = pd.to_datetime(exposures["date"]).dt.normalize()

    exposure_columns = [*STYLE_COLUMNS, *INDUSTRY_COLUMNS]
    missing_exposures = set(exposure_columns).difference(exposures.columns)
    if missing_exposures:
        raise ValueError(f"暴露数据缺少必要列: {sorted(missing_exposures)}")

    merged = pool.merge(
        exposures[["date", "instrument", *exposure_columns]],
        how="left",
        on=["date", "instrument"],
        validate="many_to_one",
    )
    merged[exposure_columns] = merged[exposure_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)

    daily_rows: list[dict] = []
    correlation_rows: list[dict] = []
    for date, date_df in merged.groupby("date", sort=True):
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
            residual = _residual(valid_y, valid_x)
            total_ss = float(np.sum((valid_y - valid_y.mean()) ** 2))
            residual_ss = float(np.sum(residual ** 2))
            r2 = np.nan if total_ss == 0 else 1.0 - residual_ss / total_ss

            raw_correlations = []
            residual_correlations = []
            for index, exposure in enumerate(exposure_columns):
                raw_corr = _safe_corr(valid_y, valid_x[:, index])
                residual_corr = _safe_corr(residual, valid_x[:, index])
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
                "max_abs_style_corr": np.nanmax(raw_correlations),
                "max_abs_residual_corr": np.nanmax(residual_correlations),
            })

    daily = pd.DataFrame(daily_rows)
    correlations = pd.DataFrame(correlation_rows)
    summary = daily.groupby("submission_id", as_index=False).agg(
        trading_days=("date", "nunique"),
        valid_days=("regression_r2", "count"),
        median_sample_count=("sample_count", "median"),
        mean_regression_r2=("regression_r2", "mean"),
        p95_regression_r2=("regression_r2", lambda value: value.quantile(0.95)),
        mean_max_abs_style_corr=("max_abs_style_corr", "mean"),
        p95_max_abs_style_corr=("max_abs_style_corr", lambda value: value.quantile(0.95)),
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
        p95_abs_correlation=("abs_correlation", lambda value: value.quantile(0.95)),
        max_abs_correlation=("abs_correlation", "max"),
    ).sort_values(
        ["submission_id", "p95_abs_correlation"], ascending=[True, False]
    ).reset_index(drop=True)

    print(
        f"因子数: {len(factor_columns)}，日期范围: {start_date} ~ {end_date}，"
        f"告警因子数: {int(summary['style_exposure_warning'].sum())}"
    )
    if display:
        show(summary, exposure_summary)
    return StyleExposureCheckResult(summary, exposure_summary, daily)
