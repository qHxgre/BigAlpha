"""正式评分复算及排名敏感性检查。"""

from __future__ import annotations

import pandas as pd

from .common import METRICS, add_to_sys_path, read_final, read_public_scores, read_summary, show
from .config import CONFIG, PATHS, CheckPaths


def check_score_consistency(paths: CheckPaths = PATHS, *, display: bool = True) -> pd.DataFrame:
    """复算正式成绩，返回与存储结果不一致的提交。"""
    add_to_sys_path(paths.private_code_dir)
    from scoring import compute_a_scores, compute_b_scores, compute_final_scores

    rows_df = read_summary(paths)
    rows = rows_df.where(pd.notna(rows_df), None).to_dict("records")
    stored = read_final(paths)
    b_scores = compute_b_scores(pd.read_csv(paths.regression_path)) if paths.regression_path.exists() else {}
    recomputed = compute_final_scores(rows, compute_a_scores(rows), b_scores)
    comparison = stored.merge(recomputed, on="submission_id", suffixes=("_stored", "_recomputed"))
    score_columns = ("a_score", "b_score", "final_score")
    for column in score_columns:
        comparison[f"{column}_delta"] = (
            comparison[f"{column}_stored"] - comparison[f"{column}_recomputed"]
        ).abs()
    problem_columns = ["submission_id", *(f"{column}_delta" for column in score_columns)]
    problems = comparison.loc[comparison[problem_columns[1:]].max(axis=1) > 1e-9, problem_columns]
    print(f"提交数: {len(stored)}，复算不一致数: {len(problems)}")
    if display:
        show(problems)
    return problems


def analyze_rank_conflicts(paths: CheckPaths = PATHS, *, display: bool = True) -> pd.DataFrame:
    """返回最终排名、各指标排名及排名冲突。"""
    summary, final = read_summary(paths), read_final(paths)
    table = final.merge(summary[["submission_id", "status", *METRICS]], on="submission_id", how="left")
    table["final_rank"] = table["final_score"].rank(ascending=False, method="min").astype(int)
    table["a_rank"] = table["a_score"].rank(ascending=False, method="min")
    table["b_rank"] = table["b_score"].rank(ascending=False, method="min")
    for metric in METRICS:
        table[f"{metric}_rank"] = pd.to_numeric(table[metric], errors="coerce").rank(
            ascending=False, method="min"
        )
    metric_ranks = [f"{metric}_rank" for metric in METRICS]
    table["a_b_rank_gap"] = (table["a_rank"] - table["b_rank"]).abs()
    table["metric_rank_spread"] = table[metric_ranks].max(axis=1) - table[metric_ranks].min(axis=1)
    columns = [
        "final_rank", "submission_id", *METRICS, "a_score", "a_rank", "b_score", "b_rank",
        "final_score", "a_b_rank_gap", "metric_rank_spread",
    ]
    result = table.sort_values("final_rank")[columns]
    if display:
        show(result.style.background_gradient(subset=["a_b_rank_gap", "metric_rank_spread"], cmap="YlOrRd"))
    return result


def analyze_ab_weight_sensitivity(
    paths: CheckPaths = PATHS, *, steps: int = CONFIG.ab_weight_steps, display: bool = True
) -> pd.DataFrame:
    """扫描 A/B 权重，返回每个提交可能出现的名次范围。"""
    successful = read_final(paths).loc[lambda x: x["final_score"] >= 0].copy().set_index("submission_id")
    ranks = {}
    for step in range(steps + 1):
        a_weight = step / steps
        score = a_weight * successful["a_score"] + (1 - a_weight) * successful["b_score"]
        ranks[f"A={a_weight:.2f}"] = score.rank(ascending=False, method="min").astype(int)
    result = pd.DataFrame(ranks)
    result["best_rank"] = result.min(axis=1)
    result["worst_rank"] = result.max(axis=1)
    result["rank_span"] = result["worst_rank"] - result["best_rank"]
    result = result.sort_values(["rank_span", "best_rank"], ascending=[False, True])
    if display:
        show(result.style.background_gradient(subset=["rank_span"], cmap="YlOrRd"))
    return result


def analyze_a_metric_sensitivity(paths: CheckPaths = PATHS, *, display: bool = True) -> pd.DataFrame:
    """逐项移除 A 指标，返回最终名次变化。"""
    summary, final = read_summary(paths), read_final(paths)
    data = summary.loc[summary["status"].eq("success"), ["submission_id", *METRICS]].merge(
        final[["submission_id", "b_score", "final_score"]], on="submission_id"
    )
    data["base_rank"] = data["final_score"].rank(ascending=False, method="min").astype(int)
    result = data[["submission_id", "base_rank"]].set_index("submission_id")
    for omitted in METRICS:
        kept = [metric for metric in METRICS if metric != omitted]
        alternative_a = sum(pd.to_numeric(data[m], errors="coerce").rank(pct=True) for m in kept) / len(kept)
        alternative_rank = (0.3 * alternative_a + 0.7 * data["b_score"]).rank(
            ascending=False, method="min"
        ).astype(int)
        result[f"without_{omitted}"] = alternative_rank.to_numpy()
        result[f"change_without_{omitted}"] = alternative_rank.to_numpy() - data["base_rank"].to_numpy()
    changes = [column for column in result if column.startswith("change_")]
    result["max_abs_rank_change"] = result[changes].abs().max(axis=1)
    result = result.sort_values(["max_abs_rank_change", "base_rank"], ascending=[False, True])
    if display:
        show(result.style.background_gradient(subset=["max_abs_rank_change"], cmap="YlOrRd"))
    return result


def compare_public_private_ranking(
    paths: CheckPaths = PATHS, *, display: bool = True
) -> pd.DataFrame:
    """以私榜执行清单为主，比较全部提交（含失败项）的公私榜得分和排名。"""
    summary_columns = ["submission_id", "status", "error"]
    summary = read_summary(paths)[summary_columns].copy()
    private = summary.merge(
        read_final(paths), on="submission_id", how="left", validate="one_to_one"
    ).rename(columns={"final_score": "private_score"})
    result = private.merge(
        read_public_scores(paths), on="submission_id", how="left",
        validate="one_to_one", indicator=True,
    )

    result["private_rank"] = result["private_score"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype("Int64")
    result["public_rank"] = result["public_score"].rank(
        ascending=False, method="min", na_option="keep"
    ).astype("Int64")
    result["score_delta"] = result["private_score"] - result["public_score"]
    result["rank_delta"] = result["private_rank"] - result["public_rank"]
    result["public_score_found"] = result["_merge"].eq("both") & result["public_score"].notna()
    result = result.drop(columns="_merge").sort_values(
        ["private_rank", "submission_id"], na_position="last"
    )[
        [
            "private_rank", "public_rank", "rank_delta", "submission_id", "status", "error",
            "a_score", "b_score", "private_score", "public_score", "score_delta",
            "public_score_found",
        ]
    ]
    if display:
        print(
            f"私榜提交数: {len(result)}，匹配到公榜分数: "
            f"{int(result['public_score_found'].sum())}，未匹配: "
            f"{int((~result['public_score_found']).sum())}"
        )
        show(result)
    return result
