"""私榜 A 分、B 分和最终分的纯计算逻辑。"""
from __future__ import annotations

import pandas as pd


SFA_METRICS = ("ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir")


def compute_a_scores(rows: list[dict]) -> pd.DataFrame:
    successful = [row for row in rows if row.get("status") == "success"]
    if not successful:
        return pd.DataFrame()

    scores = pd.DataFrame(successful)
    for metric in SFA_METRICS:
        scores[metric] = pd.to_numeric(scores.get(metric), errors="coerce")
    scores["a_score"] = sum(scores[metric].rank(pct=True) for metric in SFA_METRICS) / len(SFA_METRICS)
    return scores


def compute_b_scores(regression: pd.DataFrame) -> dict[str, float]:
    if "factor" not in regression or "model_score" not in regression:
        return {}

    regression = regression.copy()
    regression["model_score"] = pd.to_numeric(regression["model_score"], errors="coerce")
    selected = regression["model_score"] > 0
    if "selection_rate" in regression:
        selected &= pd.to_numeric(regression["selection_rate"], errors="coerce") > 0
    regression["b_score"] = regression["model_score"].rank(pct=True).fillna(0.0)
    regression.loc[~selected, "b_score"] = 0.0
    return {
        str(factor): float(score)
        for factor, score in zip(regression["factor"], regression["b_score"])
    }


def compute_final_scores(
    rows: list[dict],
    a_scores: pd.DataFrame,
    b_scores: dict[str, float],
) -> pd.DataFrame:
    """失败提交记 -2；成功提交按 0.3*A + 0.7*B 合成最终分。"""
    a_by_id = {}
    if not a_scores.empty:
        a_by_id = {
            str(sid): float(score)
            for sid, score in zip(a_scores["submission_id"], a_scores["a_score"])
        }

    final_rows = []
    for row in rows:
        sid = str(row["submission_id"])
        if row.get("status") != "success":
            final_rows.append(
                {"submission_id": sid, "a_score": None, "b_score": None, "final_score": -2.0}
            )
            continue
        a_score = a_by_id.get(sid, 0.0)
        b_score = b_scores.get(sid, 0.0)
        final_rows.append(
            {
                "submission_id": sid,
                "a_score": a_score,
                "b_score": b_score,
                "final_score": 0.3 * a_score + 0.7 * b_score,
            }
        )
    return pd.DataFrame(final_rows).sort_values("final_score", ascending=False)
