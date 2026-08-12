"""正式评分复算、指标冲突和榜单敏感性检查。"""

from __future__ import annotations

import pandas as pd

from .common import add_to_sys_path, read_final, read_public_scores, read_score, read_summary, show
from .config import CONFIG, METRICS, PATHS, CheckPaths


def check_score_consistency(
    paths: CheckPaths = PATHS,
    *,
    tolerance: float = CONFIG.score_tolerance,
    display: bool = True,
) -> pd.DataFrame:
    """调用正式 ``compute_scores`` 复算分数，返回缺失或不一致记录。"""
    add_to_sys_path(paths.private_code_dir)
    from scoring import compute_scores

    summary = read_summary(paths)
    rows = summary.drop(columns=["score"], errors="ignore")
    rows = rows.where(pd.notna(rows), None).to_dict("records")
    recomputed = compute_scores(rows).rename(columns={"score": "score_recomputed"})
    stored = read_final(paths).rename(columns={"score": "score_stored"})
    comparison = stored.merge(
        recomputed, on="submission_id", how="outer", validate="one_to_one", indicator=True
    )
    comparison["score_delta"] = (
        comparison["score_stored"] - comparison["score_recomputed"]
    ).abs()
    comparison["problem"] = comparison["_merge"].ne("both") | comparison["score_delta"].gt(tolerance)
    problems = comparison.loc[comparison["problem"]].copy()
    print(f"提交数: {len(comparison)}，评分复算不一致数: {len(problems)}")
    if display:
        show(problems)
    return problems


def analyze_metric_rank_conflicts(
    paths: CheckPaths = PATHS, *, display: bool = True
) -> pd.DataFrame:
    """比较综合排名和四项原始指标排名，定位偏科及指标冲突。"""
    score = read_score(paths).copy()
    score["final_rank"] = score["score"].rank(ascending=False, method="min").astype(int)
    for metric in METRICS:
        score[f"{metric}_rank"] = score[metric].rank(ascending=False, method="min")
    rank_columns = [f"{metric}_rank" for metric in METRICS]
    score["best_metric_rank"] = score[rank_columns].min(axis=1)
    score["worst_metric_rank"] = score[rank_columns].max(axis=1)
    score["metric_rank_spread"] = score["worst_metric_rank"] - score["best_metric_rank"]
    score["max_metric_final_gap"] = score[rank_columns].sub(score["final_rank"], axis=0).abs().max(axis=1)
    result = score.sort_values("final_rank")[
        ["final_rank", "submission_id", "score", *METRICS, *rank_columns,
         "best_metric_rank", "worst_metric_rank", "metric_rank_spread",
         "max_metric_final_gap"]
    ]
    if display:
        show(result)
    return result


def analyze_metric_sensitivity(
    paths: CheckPaths = PATHS, *, display: bool = True
) -> pd.DataFrame:
    """逐项移除指标重新等权评分，检查榜单对单项指标的依赖。"""
    data = read_score(paths).copy()
    data["base_rank"] = data["score"].rank(ascending=False, method="min").astype(int)
    result = data[["submission_id", "base_rank"]].set_index("submission_id")
    for omitted in METRICS:
        kept = [metric for metric in METRICS if metric != omitted]
        alternative_score = sum(data[metric].rank(pct=True) for metric in kept) / len(kept)
        alternative_rank = alternative_score.rank(ascending=False, method="min").astype(int)
        result[f"score_without_{omitted}"] = alternative_score.to_numpy()
        result[f"rank_without_{omitted}"] = alternative_rank.to_numpy()
        result[f"change_without_{omitted}"] = alternative_rank.to_numpy() - data["base_rank"].to_numpy()
    changes = [column for column in result if column.startswith("change_without_")]
    result["max_abs_rank_change"] = result[changes].abs().max(axis=1)
    result["most_sensitive_metric"] = result[changes].abs().idxmax(axis=1).str.removeprefix(
        "change_without_"
    )
    result = result.sort_values(["max_abs_rank_change", "base_rank"], ascending=[False, True])
    if display:
        show(result)
    return result


def compare_public_private_ranking(
    paths: CheckPaths = PATHS, *, display: bool = True
) -> pd.DataFrame:
    """以私榜执行清单为主，比较全部提交（含失败项）的公私榜表现。"""
    summary = read_summary(paths)[
        ["submission_id", "status", "failure_type", "error"]
    ].copy()
    private = read_final(paths).rename(columns={"score": "private_score"})
    private = summary.merge(
        private, on="submission_id", how="left", validate="one_to_one"
    )
    public = read_public_scores(paths)
    result = private.merge(
        public, on="submission_id", how="left", validate="one_to_one", indicator=True
    )

    # 失败分数仍参与榜单排名，保持与成绩文件的实际排序口径一致。
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
            "private_rank", "public_rank", "rank_delta", "submission_id",
            "status", "failure_type", "error", "private_score", "public_score",
            "score_delta", "public_score_found",
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
