"""回归产物完整性、稳定性和替代口径检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import REGRESSION_METRICS, read_final, read_regression, read_summary, show
from .config import CheckPaths


def check_regression_integrity(paths: CheckPaths, *, display: bool = True) -> dict[str, pd.DataFrame]:
    """检查提交、因子池、回归结果的集合一致性和字段合法性。"""
    summary = read_summary(paths)
    regression = read_regression(paths)
    factor_pool = pd.read_parquet(paths.factor_pool_path)
    successful_ids = set(summary.loc[summary["status"].eq("success"), "submission_id"])
    pool_ids = {str(column) for column in factor_pool.columns if column not in {"date", "instrument"}}
    regression_ids = set(regression["factor"].dropna().astype(str))
    checks = {
        "成功提交未进入因子池": successful_ids - pool_ids,
        "因子池中没有对应成功提交": pool_ids - successful_ids,
        "因子池因子没有回归结果": pool_ids - regression_ids,
        "回归结果中没有对应因子池列": regression_ids - pool_ids,
    }
    set_problems = pd.DataFrame(
        ({"check": check, "factor": factor} for check, factors in checks.items() for factor in sorted(factors)),
        columns=["check", "factor"],
    )
    reasons = pd.DataFrame(index=regression.index)
    reasons["duplicate_factor"] = regression["factor"].duplicated(keep=False)
    reasons["non_finite_value"] = ~np.isfinite(regression[list(REGRESSION_METRICS)]).all(axis=1)
    reasons["negative_model_score"] = regression["model_score"] < 0
    reasons["negative_weight_stat"] = (regression["abs_weight_mean"] < 0) | (regression["abs_weight_std"] < 0)
    reasons["selection_rate_out_of_range"] = ~regression["selection_rate"].between(0, 1)
    reasons["unselected_but_nonzero"] = regression["selection_rate"].eq(0) & (
        regression["model_score"].ne(0) | regression["abs_weight_mean"].ne(0)
    )
    reasons["selected_but_zero_weight"] = regression["selection_rate"].gt(0) & regression["abs_weight_mean"].eq(0)
    field_problems = regression.loc[reasons.any(axis=1), ["factor", *REGRESSION_METRICS]].copy()
    if not field_problems.empty:
        field_problems["problems"] = reasons.loc[field_problems.index].apply(
            lambda row: "、".join(row.index[row]), axis=1
        )
    print(
        f"成功提交: {len(successful_ids)}，因子池因子: {len(pool_ids)}，回归结果: {len(regression)}，"
        f"集合差异: {len(set_problems)}，字段异常: {len(field_problems)}"
    )
    if display:
        show(set_problems, field_problems)
    return {"set_problems": set_problems, "field_problems": field_problems}


def analyze_regression_stability(paths: CheckPaths, *, display: bool = True) -> pd.DataFrame:
    """计算权重波动、稳定性得分并标记建议优先复核的因子。"""
    result = read_regression(paths)
    result["weight_cv"] = result["abs_weight_std"].div(result["abs_weight_mean"].replace(0, np.nan))
    result["stability_score"] = result["selection_rate"].div(1 + result["weight_cv"])
    rank_specs = {
        "model_score_rank": ("model_score", False), "weight_mean_rank": ("abs_weight_mean", False),
        "selection_rate_rank": ("selection_rate", False), "weight_cv_rank": ("weight_cv", True),
        "stability_rank": ("stability_score", False),
    }
    for rank_column, (metric, ascending) in rank_specs.items():
        result[rank_column] = result[metric].rank(ascending=ascending, method="min")
    result["model_stability_rank_gap"] = (result["model_score_rank"] - result["stability_rank"]).abs()
    high_model = result["model_score"] >= result["model_score"].quantile(0.80)
    low_selection = result["selection_rate"] <= result["selection_rate"].quantile(0.10)
    high_cv = result["weight_cv"] >= result["weight_cv"].quantile(0.90)
    large_gap = result["model_stability_rank_gap"] >= 0.30 * len(result)
    result["suspicious"] = high_model & (low_selection | high_cv | large_gap)
    result = result.sort_values(
        ["suspicious", "model_stability_rank_gap", "model_score"], ascending=[False, False, False]
    )
    print(f"回归因子: {len(result)}，建议优先复核: {int(result['suspicious'].sum())}")
    if display:
        show(result.style.background_gradient(subset=["model_stability_rank_gap"], cmap="YlOrRd"))
    return result


def analyze_b_score_robustness(paths: CheckPaths, *, display: bool = True) -> pd.DataFrame:
    """用三种替代 B 口径进行排名压力测试。"""
    final, regression = read_final(paths), read_regression(paths)
    data = final.loc[final["final_score"] >= 0].merge(
        regression, left_on="submission_id", right_on="factor", how="left", validate="one_to_one"
    )
    data["weight_cv"] = data["abs_weight_std"].div(data["abs_weight_mean"].replace(0, np.nan))
    alternatives = {
        "selection": data["model_score"] * data["selection_rate"],
        "stable": data["model_score"] / (1 + data["weight_cv"]),
        "robust": data["model_score"] * data["selection_rate"] / (1 + data["weight_cv"]),
    }
    result = data[["submission_id", "a_score", "b_score", "final_score"]].copy()
    result["base_b_rank"] = data["b_score"].rank(ascending=False, method="min")
    result["base_final_rank"] = data["final_score"].rank(ascending=False, method="min")
    valid_selected = data["model_score"].gt(0) & data["selection_rate"].gt(0)
    for name, raw_metric in alternatives.items():
        alternative_b = raw_metric.rank(pct=True).fillna(0.0).where(valid_selected, 0.0)
        alternative_final = 0.3 * data["a_score"] + 0.7 * alternative_b
        result[f"{name}_b_rank"] = alternative_b.rank(ascending=False, method="min")
        result[f"{name}_final_rank"] = alternative_final.rank(ascending=False, method="min")
        result[f"{name}_b_rank_change"] = result[f"{name}_b_rank"] - result["base_b_rank"]
        result[f"{name}_final_rank_change"] = result[f"{name}_final_rank"] - result["base_final_rank"]
    b_changes = [f"{name}_b_rank_change" for name in alternatives]
    final_changes = [f"{name}_final_rank_change" for name in alternatives]
    result["max_b_rank_change"] = result[b_changes].abs().max(axis=1)
    result["max_final_rank_change"] = result[final_changes].abs().max(axis=1)
    result = result.sort_values(
        ["max_final_rank_change", "max_b_rank_change", "base_final_rank"], ascending=[False, False, True]
    )
    print(
        f"成功提交: {len(result)}，最大 B 排名变化: {result['max_b_rank_change'].max():.0f}，"
        f"最大最终排名变化: {result['max_final_rank_change'].max():.0f}"
    )
    if display:
        show(result.style.background_gradient(subset=["max_b_rank_change", "max_final_rank_change"], cmap="YlOrRd"))
    return result
