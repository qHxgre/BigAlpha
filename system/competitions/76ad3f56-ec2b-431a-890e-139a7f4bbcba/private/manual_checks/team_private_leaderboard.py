"""按团队最高私榜分生成独立 CSV 排行榜。

本脚本不属于 Markdown 人工复核报告流程。可直接运行：

    python -m manual_checks.team_private_leaderboard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import METRICS, read_final, read_summary
from .config import PATHS, TEAM_PRIVATE_LEADERBOARD_FILENAME, CheckPaths


OUTPUT_COLUMNS = [
    "团队名",
    "私榜排名",
    "私榜得分",
    "公榜排名",
    "公榜得分",
    "成员名称及对应学校",
    "私榜细节得分",
    "公榜细节得分",
]


def _json_cell(values: dict[str, Any] | None) -> str:
    """将有值的得分字段序列化为适合 CSV 单元格的紧凑 JSON。"""
    if not isinstance(values, dict):
        return ""
    cleaned = {
        str(key): value
        for key, value in values.items()
        if value is not None and not pd.isna(value)
    }
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) if cleaned else ""


def _members_cell(members: list[dict[str, Any]]) -> str:
    return "；".join(
        f"{member.get('name') or member.get('user_id') or '未命名'}"
        f"（{member.get('school') or '未填写'}）"
        for member in members
    )


def _public_score_detail(
    participant: dict[str, Any], submissions_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """读取官方团队公榜细节，并兼容未直接保存该字段的旧 metadata。"""
    detail = participant.get("public_score_data")
    if not isinstance(detail, dict) or not detail:
        public_score = pd.to_numeric(participant.get("public_score"), errors="coerce")
        for submission_id in participant.get("private_submission_ids") or []:
            submission = submissions_by_id.get(str(submission_id), {})
            submission_score = pd.to_numeric(submission.get("public_score"), errors="coerce")
            if pd.notna(public_score) and pd.notna(submission_score) and abs(public_score - submission_score) < 1e-5:
                raw_submission = submission.get("submission") or {}
                detail = raw_submission.get("public_score_data")
                break
    if not isinstance(detail, dict) or not detail:
        return None
    return {
        key: detail.get(key)
        for key in ("a_score", "b_score", "final_score")
        if key in detail
    }


def build_team_private_leaderboard(paths: CheckPaths = PATHS) -> pd.DataFrame:
    """返回团队级排行榜，每个团队取私榜 ``final_score`` 最高的提交。"""
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    submissions_by_id = {
        str(item.get("submission_id")): item
        for item in metadata.get("submissions", [])
        if item.get("submission_id")
    }
    summary = read_summary(paths)
    final = read_final(paths)

    score_columns = ["submission_id", "a_score", "b_score", "final_score"]
    details = summary[["submission_id", *METRICS]].merge(
        final[score_columns], on="submission_id", how="right", validate="one_to_one"
    )
    for column in [*METRICS, "a_score", "b_score", "final_score"]:
        details[column] = pd.to_numeric(details[column], errors="coerce")
    details_by_id = details.set_index("submission_id", drop=False)

    rows: list[dict[str, Any]] = []
    for participant in metadata.get("participants", []):
        if participant.get("type") != "team":
            continue
        submission_ids = [
            str(value) for value in participant.get("private_submission_ids") or []
            if str(value) in details_by_id.index
        ]
        if not submission_ids:
            continue
        candidates = details_by_id.loc[submission_ids].reset_index(drop=True)
        if isinstance(candidates, pd.Series):
            candidates = candidates.to_frame().T
        best = candidates.sort_values(
            ["final_score", "submission_id"], ascending=[False, True], na_position="last"
        ).iloc[0]
        private_detail = {
            "a_score": best["a_score"],
            "b_score": best["b_score"],
            "final_score": best["final_score"],
            **{metric: best[metric] for metric in METRICS},
        }
        rows.append({
            "团队名": participant.get("team_name") or "（未命名队伍）",
            "私榜得分": best["final_score"],
            "公榜排名": participant.get("public_rank"),
            "公榜得分": participant.get("public_score"),
            "成员名称及对应学校": _members_cell(participant.get("members") or []),
            "私榜细节得分": _json_cell(private_detail),
            "公榜细节得分": _json_cell(_public_score_detail(participant, submissions_by_id)),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    result["私榜得分"] = pd.to_numeric(result["私榜得分"], errors="coerce")
    result["公榜得分"] = pd.to_numeric(result["公榜得分"], errors="coerce")
    result["公榜排名"] = pd.to_numeric(result["公榜排名"], errors="coerce").astype("Int64")
    result["私榜排名"] = result["私榜得分"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype("Int64")
    return result.sort_values(
        ["私榜排名", "团队名"], ascending=[True, True], na_position="last"
    )[OUTPUT_COLUMNS].reset_index(drop=True)


def export_team_private_leaderboard(
    paths: CheckPaths = PATHS, *, output: str | Path | None = None
) -> Path:
    """生成 UTF-8 BOM 编码的团队私榜 CSV，并返回输出路径。"""
    output_path = (
        Path(output).expanduser().resolve()
        if output
        else paths.artifacts_dir / TEAM_PRIVATE_LEADERBOARD_FILENAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = build_team_private_leaderboard(paths)
    leaderboard.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="CSV 输出路径；默认写入本次运行 artifacts 目录")
    args = parser.parse_args()
    output = export_team_private_leaderboard(output=args.output)
    print(f"团队私榜已生成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
