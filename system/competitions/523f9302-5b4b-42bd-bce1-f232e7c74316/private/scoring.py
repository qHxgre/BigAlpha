"""端到端模型私榜的纯评分逻辑。"""
from __future__ import annotations

import pandas as pd

METRICS = ("ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir")


def compute_scores(rows: list[dict]) -> pd.DataFrame:
    """成功提交按四项指标百分位等权评分；失败提交记 -2。"""
    successful = [dict(row) for row in rows if row.get("status") == "success"]
    score_by_id: dict[str, float] = {}
    if successful:
        frame = pd.DataFrame(successful)
        for metric in METRICS:
            frame[metric] = pd.to_numeric(frame.get(metric), errors="coerce")
        frame["score"] = sum(frame[m].rank(pct=True) for m in METRICS) / len(METRICS)
        score_by_id = {
            str(sid): float(score)
            for sid, score in zip(frame["submission_id"], frame["score"])
            if pd.notna(score)
        }

    result = []
    for row in rows:
        sid = str(row["submission_id"])
        result.append({"submission_id": sid, "score": score_by_id.get(sid, -2.0)})
    if not result:
        return pd.DataFrame(columns=["submission_id", "score"])
    return pd.DataFrame(result).sort_values("score", ascending=False)
