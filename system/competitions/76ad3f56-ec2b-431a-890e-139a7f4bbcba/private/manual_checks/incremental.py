"""因子样本外增量贡献与相似因子组替代性分析。"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet

from .common import add_to_sys_path, show
from .config import (
    CONFIG,
    FACTOR_CLUSTERS_FILENAME,
    GROUP_INCREMENTAL_BY_WINDOW_FILENAME,
    GROUP_INCREMENTAL_SUMMARY_FILENAME,
    INCREMENTAL_BY_WINDOW_FILENAME,
    INCREMENTAL_METADATA_FILENAME,
    INCREMENTAL_SUMMARY_FILENAME,
    PATHS,
    CheckPaths,
)
from .factors import analyze_cross_sectional_similarity


def _cross_section_zscore(values: pd.Series) -> pd.Series:
    mean, std = values.mean(), values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mean) / std


def _daily_prediction_metrics(
    dates: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """计算测试窗口的每日截面 IC，以及整体 MSE/R²。"""
    daily_ic = []
    date_values = pd.to_datetime(dates)
    for date in pd.unique(date_values):
        mask = date_values == date
        left, right = actual[mask], prediction[mask]
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
            continue
        daily_ic.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
    residual = actual - prediction
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0 else np.nan
    ic = pd.Series(daily_ic, dtype=float)
    return {
        "ic_mean": float(ic.mean()) if not ic.empty else np.nan,
        "ic_median": float(ic.median()) if not ic.empty else np.nan,
        "ic_ir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else np.nan,
        "positive_ic_day_rate": float(ic.gt(0).mean()) if not ic.empty else np.nan,
        "mse": mse,
        "r2": r2,
        "test_days": int(len(ic)),
    }


def _fit_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_dates: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
) -> dict[str, float]:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=2000,
        tol=1e-4,
        precompute=True,
        selection="random",
        random_state=0,
    )
    model.fit(X_train, y_train)
    weights = np.asarray(model.coef_, dtype=float)
    prediction = X_test @ weights
    metrics = _daily_prediction_metrics(test_dates, y_test, prediction)
    metrics["nonzero_factors"] = int(np.count_nonzero(weights))
    metrics["weight_l1"] = float(np.abs(weights).sum())
    metrics["weight_l2"] = float(np.sqrt(np.square(weights).sum()))
    return metrics


def _delta_record(
    *,
    window_id: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    item_id: str,
    item_size: int,
    full: dict[str, float],
    reduced: dict[str, float],
) -> dict[str, object]:
    return {
        "window_id": window_id,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "item_id": item_id,
        "item_size": item_size,
        "full_ic": full["ic_mean"],
        "reduced_ic": reduced["ic_mean"],
        "delta_ic": full["ic_mean"] - reduced["ic_mean"],
        "full_mse": full["mse"],
        "reduced_mse": reduced["mse"],
        "delta_mse": reduced["mse"] - full["mse"],
        "full_r2": full["r2"],
        "reduced_r2": reduced["r2"],
        "delta_r2": full["r2"] - reduced["r2"],
        "full_positive_ic_day_rate": full["positive_ic_day_rate"],
        "reduced_positive_ic_day_rate": reduced["positive_ic_day_rate"],
        "delta_positive_ic_day_rate": (
            full["positive_ic_day_rate"] - reduced["positive_ic_day_rate"]
        ),
        "full_nonzero_factors": full["nonzero_factors"],
        "reduced_nonzero_factors": reduced["nonzero_factors"],
    }


def _summarize_incremental(
    by_window: pd.DataFrame, *, id_column: str
) -> pd.DataFrame:
    if by_window.empty:
        return pd.DataFrame()
    summary = by_window.groupby("item_id", as_index=False).agg(
        windows=("window_id", "nunique"),
        mean_delta_ic=("delta_ic", "mean"),
        median_delta_ic=("delta_ic", "median"),
        std_delta_ic=("delta_ic", "std"),
        min_delta_ic=("delta_ic", "min"),
        max_delta_ic=("delta_ic", "max"),
        positive_delta_rate=("delta_ic", lambda values: values.gt(0).mean()),
        mean_delta_mse=("delta_mse", "mean"),
        mean_delta_r2=("delta_r2", "mean"),
        mean_delta_positive_ic_day_rate=("delta_positive_ic_day_rate", "mean"),
        item_size=("item_size", "first"),
    )
    standard_error = summary["std_delta_ic"].div(np.sqrt(summary["windows"]))
    summary["delta_ic_t_stat"] = summary["mean_delta_ic"].div(standard_error.replace(0, np.nan))
    summary["incremental_score"] = summary["mean_delta_ic"] * summary["positive_delta_rate"]
    summary["incremental_rank"] = summary["incremental_score"].rank(
        ascending=False, method="min"
    )
    return summary.rename(columns={"item_id": id_column}).sort_values(
        ["incremental_score", "mean_delta_ic"], ascending=False
    )


def analyze_incremental_contribution(
    paths: CheckPaths = PATHS,
    *,
    train_window: int = CONFIG.incremental_train_window,
    test_window: int = CONFIG.incremental_test_window,
    step: int = CONFIG.incremental_step,
    alpha: float = CONFIG.incremental_alpha,
    l1_ratio: float = CONFIG.incremental_l1_ratio,
    returns_data: pd.DataFrame | None = None,
    n_jobs: int = -1,
    save: bool = True,
    display: bool = True,
) -> dict[str, pd.DataFrame]:
    """滚动执行完整模型、逐因子删除和逐相似组删除的样本外对照。"""
    if paths.bigalpha_eval_src is None:
        raise ValueError("CheckPaths.bigalpha_eval_src 未配置")
    pool = pd.read_parquet(paths.factor_pool_path)
    pool["date"] = pd.to_datetime(pool["date"])
    factor_cols = [str(column) for column in pool.columns if column not in {"date", "instrument"}]
    pool = pool.rename(columns={column: str(column) for column in pool.columns})
    if returns_data is None:
        add_to_sys_path(paths.bigalpha_eval_src)
        from bigalpha_eval.regmodel.data import get_next_day_return

        returns = get_next_day_return(
            pool["date"].min().strftime("%Y-%m-%d"),
            pool["date"].max().strftime("%Y-%m-%d"),
            instruments=pool["instrument"].dropna().astype(str).unique().tolist(),
        )
    else:
        returns = returns_data.copy()
        returns["date"] = pd.to_datetime(returns["date"])
    data = pool.merge(returns, on=["date", "instrument"], how="left")
    data = data.dropna(subset=[*factor_cols, "daily_ret"]).copy()
    data["daily_ret"] = data.groupby("date")["daily_ret"].transform(_cross_section_zscore)
    data = data.sort_values(["date", "instrument"]).reset_index(drop=True)

    cluster_path = paths.incremental_dir / FACTOR_CLUSTERS_FILENAME
    if not cluster_path.is_file():
        analyze_cross_sectional_similarity(paths, save=True, display=False)
    clusters = pd.read_csv(cluster_path, dtype={"submission_id": str})
    cluster_members = (
        clusters.groupby("cluster_id")["submission_id"].apply(list).to_dict()
    )

    all_days = pd.DatetimeIndex(sorted(data["date"].unique()))
    factor_records: list[dict[str, object]] = []
    group_records: list[dict[str, object]] = []
    window_starts = range(0, max(0, len(all_days) - train_window - test_window + 1), step)
    for window_id, start in enumerate(window_starts, start=1):
        train_days = all_days[start:start + train_window]
        test_days = all_days[start + train_window:start + train_window + test_window]
        if len(train_days) < train_window or len(test_days) < test_window:
            continue
        train = data.loc[data["date"].isin(train_days)]
        test = data.loc[data["date"].isin(test_days)]
        X_train = train[factor_cols].to_numpy(dtype=float)
        y_train = train["daily_ret"].to_numpy(dtype=float)
        X_test = test[factor_cols].to_numpy(dtype=float)
        y_test = test["daily_ret"].to_numpy(dtype=float)
        test_dates = test["date"].to_numpy()
        full = _fit_and_evaluate(
            X_train, y_train, X_test, y_test, test_dates,
            alpha=alpha, l1_ratio=l1_ratio,
        )

        def evaluate_factor(index: int, factor: str) -> dict[str, object]:
            keep = np.arange(len(factor_cols)) != index
            reduced = _fit_and_evaluate(
                X_train[:, keep], y_train, X_test[:, keep], y_test, test_dates,
                alpha=alpha, l1_ratio=l1_ratio,
            )
            return _delta_record(
                window_id=window_id,
                train_start=train_days[0], train_end=train_days[-1],
                test_start=test_days[0], test_end=test_days[-1],
                item_id=factor, item_size=1, full=full, reduced=reduced,
            )

        factor_records.extend(Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(evaluate_factor)(index, factor)
            for index, factor in enumerate(factor_cols)
        ))

        def evaluate_group(cluster_id: int, members: list[str]) -> dict[str, object]:
            member_set = set(members)
            keep = np.asarray([factor not in member_set for factor in factor_cols])
            reduced = _fit_and_evaluate(
                X_train[:, keep], y_train, X_test[:, keep], y_test, test_dates,
                alpha=alpha, l1_ratio=l1_ratio,
            )
            return _delta_record(
                window_id=window_id,
                train_start=train_days[0], train_end=train_days[-1],
                test_start=test_days[0], test_end=test_days[-1],
                item_id=str(cluster_id), item_size=len(members), full=full, reduced=reduced,
            )

        group_records.extend(Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(evaluate_group)(int(cluster_id), members)
            for cluster_id, members in cluster_members.items()
        ))
        print(
            f"增量贡献窗口 {window_id}: train {train_days[0].date()}~{train_days[-1].date()}, "
            f"test {test_days[0].date()}~{test_days[-1].date()}"
        )

    factor_by_window = pd.DataFrame(factor_records)
    group_by_window = pd.DataFrame(group_records)
    factor_summary = _summarize_incremental(factor_by_window, id_column="submission_id")
    group_summary = _summarize_incremental(group_by_window, id_column="cluster_id")
    if not group_summary.empty:
        group_summary["cluster_id"] = pd.to_numeric(group_summary["cluster_id"], errors="coerce").astype("Int64")
    if save:
        paths.incremental_dir.mkdir(parents=True, exist_ok=True)
        factor_by_window.to_parquet(paths.incremental_dir / INCREMENTAL_BY_WINDOW_FILENAME, index=False)
        factor_summary.to_csv(paths.incremental_dir / INCREMENTAL_SUMMARY_FILENAME, index=False)
        group_by_window.to_parquet(
            paths.incremental_dir / GROUP_INCREMENTAL_BY_WINDOW_FILENAME, index=False
        )
        group_summary.to_csv(paths.incremental_dir / GROUP_INCREMENTAL_SUMMARY_FILENAME, index=False)
        metadata = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "train_window": train_window,
            "test_window": test_window,
            "step": step,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "factor_count": len(factor_cols),
            "window_count": int(factor_by_window["window_id"].nunique()) if not factor_by_window.empty else 0,
            "date_start": str(all_days.min().date()) if len(all_days) else None,
            "date_end": str(all_days.max().date()) if len(all_days) else None,
        }
        (paths.incremental_dir / INCREMENTAL_METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"增量贡献完成: {len(factor_cols)} 个因子，"
        f"{factor_by_window['window_id'].nunique() if not factor_by_window.empty else 0} 个样本外窗口"
    )
    if display:
        show(factor_summary, group_summary)
    return {
        "factor_by_window": factor_by_window,
        "factor_summary": factor_summary,
        "group_by_window": group_by_window,
        "group_summary": group_summary,
        "clusters": clusters,
    }
