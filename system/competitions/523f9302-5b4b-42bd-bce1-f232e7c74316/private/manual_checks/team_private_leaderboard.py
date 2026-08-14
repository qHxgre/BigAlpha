"""生成团队/个人逐 submission 的公私榜细项 CSV、JSON 和 Markdown。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .common import read_final, read_public_scores, read_score, read_summary
from .config import METRICS, PATHS, CheckPaths


CSV_COLUMNS = [
    "参赛方", "参赛类型", "参赛方ID", "私榜排名", "私榜最高分", "公榜排名", "公榜得分",
    "提交数量", "成员名称及对应学校", "submission_id", "提交时间", "提交者ID",
    "私榜排名（全部提交）", "是否参赛方最佳私榜提交", "评测状态", "失败类型", "评测错误",
    "私榜得分", *[f"私榜{metric}" for metric in METRICS],
    "公榜得分（submission）", *[f"公榜{metric}" for metric in METRICS], "私榜减公榜",
]

TEAM_SUMMARY_COLUMNS = [
    "团队名", "团队成员及学校", "团队私榜提交数量", "团队私榜得分", "团队私榜排名",
    "团队公榜得分", "团队公榜排名",
]


def _clean(value: Any) -> Any:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _members(participant: dict[str, Any]) -> list[dict[str, Any]]:
    if participant.get("type") == "team":
        return participant.get("members") or []
    return [participant.get("user") or {}]


def _members_cell(members: list[dict[str, Any]]) -> str:
    return "；".join(
        f"{member.get('name') or member.get('user_id') or '未命名'}"
        f"（{member.get('school') or '未填写'}）" for member in members
    )


def build_team_private_report(paths: CheckPaths = PATHS) -> dict[str, Any]:
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    metadata_by_id = {
        str(item["submission_id"]): item for item in metadata.get("submissions", [])
        if item.get("submission_id")
    }
    detail = read_summary(paths).merge(
        read_score(paths), on="submission_id", how="outer", suffixes=("", "_score")
    ).merge(read_final(paths), on="submission_id", how="outer", suffixes=("", "_final"))
    if "score_final" in detail:
        detail["score"] = detail["score_final"].combine_first(detail.get("score"))
    detail_by_id = detail.set_index("submission_id", drop=False)
    public = read_public_scores(paths).set_index("submission_id", drop=False)
    ranks = detail["score"].rank(ascending=False, method="min", na_option="bottom")
    rank_by_id = dict(zip(detail["submission_id"].astype(str), ranks))

    participants = []
    for participant in metadata.get("participants", []):
        participant_type = participant.get("type") or "individual"
        members = _members(participant)
        participant_id = participant.get("team_id") or (members[0].get("user_id") if members else None)
        name = participant.get("team_name") if participant_type == "team" else (
            members[0].get("name") if members else None
        )
        submissions = []
        for submission_id in map(str, participant.get("private_submission_ids") or []):
            row = detail_by_id.loc[submission_id] if submission_id in detail_by_id.index else pd.Series(dtype=object)
            pub = public.loc[submission_id] if submission_id in public.index else pd.Series(dtype=object)
            meta = metadata_by_id.get(submission_id, {})
            private_score = _clean(pd.to_numeric(row.get("score"), errors="coerce"))
            public_score = _clean(pd.to_numeric(pub.get("public_score", meta.get("public_score")), errors="coerce"))
            submissions.append({
                "submission_id": submission_id,
                "created_at": meta.get("created_at") or (meta.get("submission") or {}).get("created_at"),
                "user_id": meta.get("user_id"), "status": _clean(row.get("status")),
                "failure_type": _clean(row.get("failure_type")), "error": _clean(row.get("error")),
                "private_rank_all_submissions": _clean(rank_by_id.get(submission_id)),
                "private_score": private_score,
                "private_metrics": {metric: _clean(row.get(metric)) for metric in METRICS},
                "public_score": public_score,
                "public_metrics": {metric: _clean(pub.get(metric)) for metric in METRICS},
                "public_score_found": submission_id in public.index,
                "score_delta": _clean(private_score - public_score)
                if private_score is not None and public_score is not None else None,
            })
        submissions.sort(key=lambda x: (x["private_score"] is None, -(x["private_score"] or 0)))
        best = next((item for item in submissions if item["private_score"] is not None), None)
        participants.append({
            "participant_type": participant_type, "participant_id": participant_id,
            "participant_name": name or "（未命名参赛方）", "members": members,
            "submission_count": len(submissions), "public_rank": _clean(participant.get("public_rank")),
            "public_score": _clean(pd.to_numeric(participant.get("public_score"), errors="coerce")),
            "best_private_submission_id": best["submission_id"] if best else None,
            "best_private_score": best["private_score"] if best else None, "submissions": submissions,
        })
    participant_ranks = pd.Series([p["best_private_score"] for p in participants], dtype=float).rank(
        ascending=False, method="min", na_option="bottom"
    )
    for participant, rank in zip(participants, participant_ranks):
        participant["private_rank"] = _clean(rank)
    participants.sort(key=lambda x: (x["private_rank"] or 10**9, x["participant_name"]))
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "competition_id": metadata.get("competition_id"), "batch_id": metadata.get("batch_id"),
        "participant_count": len(participants),
        "submission_count": sum(p["submission_count"] for p in participants),
        "scoring_note": "四项指标百分位排名等权；本比赛没有 A/B 或回归 B 项评分。",
        "participants": participants,
    }


def build_team_private_leaderboard(paths: CheckPaths = PATHS) -> pd.DataFrame:
    report = build_team_private_report(paths)
    rows = []
    for participant in report["participants"]:
        for sub in participant["submissions"]:
            rows.append({
                "参赛方": participant["participant_name"], "参赛类型": participant["participant_type"],
                "参赛方ID": participant["participant_id"], "私榜排名": participant["private_rank"],
                "私榜最高分": participant["best_private_score"], "公榜排名": participant["public_rank"],
                "公榜得分": participant["public_score"], "提交数量": participant["submission_count"],
                "成员名称及对应学校": _members_cell(participant["members"]),
                "submission_id": sub["submission_id"], "提交时间": sub["created_at"],
                "提交者ID": sub["user_id"], "私榜排名（全部提交）": sub["private_rank_all_submissions"],
                "是否参赛方最佳私榜提交": sub["submission_id"] == participant["best_private_submission_id"],
                "评测状态": sub["status"], "失败类型": sub["failure_type"], "评测错误": sub["error"],
                "私榜得分": sub["private_score"],
                **{f"私榜{m}": sub["private_metrics"].get(m) for m in METRICS},
                "公榜得分（submission）": sub["public_score"],
                **{f"公榜{m}": sub["public_metrics"].get(m) for m in METRICS},
                "私榜减公榜": sub["score_delta"],
            })
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def build_team_leaderboard_summary(paths: CheckPaths = PATHS) -> pd.DataFrame:
    """生成仅含团队的公私榜摘要；公榜分数取团队最佳私榜提交对应的公榜分。"""
    report = build_team_private_report(paths)
    rows = []
    for participant in report["participants"]:
        if participant["participant_type"] != "team":
            continue
        best_submission = next(
            (
                sub for sub in participant["submissions"]
                if sub["submission_id"] == participant["best_private_submission_id"]
            ),
            None,
        )
        rows.append({
            "团队名": participant["participant_name"],
            "团队成员及学校": _members_cell(participant["members"]),
            "团队私榜提交数量": participant["submission_count"],
            "团队私榜得分": participant["best_private_score"],
            "团队私榜排名": participant["private_rank"],
            "团队公榜得分": best_submission["public_score"] if best_submission else None,
            "团队公榜排名": None,
        })
    summary = pd.DataFrame(rows, columns=TEAM_SUMMARY_COLUMNS)
    if not summary.empty:
        summary["团队公榜排名"] = summary["团队公榜得分"].rank(
            ascending=False, method="min", na_option="bottom"
        )
        summary = summary.sort_values(
            ["团队私榜排名", "团队名"], na_position="last", kind="stable"
        ).reset_index(drop=True)
    return summary


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 参赛方私榜逐 submission 报告", "", f"- 参赛方数：**{report['participant_count']}**",
        f"- 提交数：**{report['submission_count']}**", f"- 评分说明：{report['scoring_note']}", "",
    ]
    for participant in report["participants"]:
        lines.extend([
            f"## {participant['private_rank']}. {participant['participant_name']}", "",
            f"成员：{_members_cell(participant['members']) or '—'}", "",
            "| 最佳 | submission_id | 状态 | 私榜排名 | 榜单 | IC均值 | ICIR | 夏普 | 压力ICIR | 总分 |",
            "|:---:|---|---|---:|---|---:|---:|---:|---:|---:|",
        ])
        for sub in participant["submissions"]:
            prefix = (
                f"| {'✓' if sub['submission_id'] == participant['best_private_submission_id'] else ''} | "
                f"{sub['submission_id']} | {sub['status'] or '—'} | {_fmt(sub['private_rank_all_submissions'])}"
            )
            for label, metrics, score in (
                ("私榜", sub["private_metrics"], sub["private_score"]),
                ("公榜", sub["public_metrics"], sub["public_score"]),
            ):
                lines.append(
                    f"{prefix} | {label} | " + " | ".join(_fmt(metrics.get(m)) for m in METRICS)
                    + f" | {_fmt(score)} |"
                )
                prefix = "|  |  |  | —"
        lines.append("")
    return "\n".join(lines)


def export_team_private_leaderboard(
    paths: CheckPaths = PATHS, *, output: str | Path | None = None,
    json_output: str | Path | None = None, markdown_output: str | Path | None = None,
    team_summary_output: str | Path | None = None,
) -> dict[str, Path]:
    csv_path = Path(output).expanduser().resolve() if output else paths.team_private_leaderboard_path
    json_path = Path(json_output).expanduser().resolve() if json_output else csv_path.with_suffix(".json")
    md_path = Path(markdown_output).expanduser().resolve() if markdown_output else csv_path.with_suffix(".md")
    team_summary_path = (
        Path(team_summary_output).expanduser().resolve()
        if team_summary_output else paths.team_leaderboard_summary_path
    )
    report = build_team_private_report(paths)
    for path in (csv_path, json_path, md_path, team_summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    build_team_private_leaderboard(paths).to_csv(csv_path, index=False, encoding="utf-8-sig")
    build_team_leaderboard_summary(paths).to_csv(
        team_summary_path, index=False, encoding="utf-8-sig"
    )
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "csv": csv_path, "json": json_path, "markdown": md_path,
        "team_summary_csv": team_summary_path,
    }
