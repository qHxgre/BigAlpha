"""按团队汇总私榜 submission，并导出 JSON、CSV 和 Markdown。"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from .common import read_final, read_public_scores, read_score, read_summary
from .config import METRICS, PATHS, CheckPaths


TEAM_SUMMARY_COLUMNS = [
    "团队名", "团队成员（学校/学历）", "团队私榜提交数量",
    "私榜得分", "私榜得分排名", "公榜得分", "公榜得分排名",
]


def _clean(value: Any) -> Any:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _clean_rank(value: Any) -> int | None:
    value = _clean(value)
    return int(value) if value is not None else None


def _clean_bool(value: Any) -> bool | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _members(participant: dict[str, Any]) -> list[dict[str, Any]]:
    if participant.get("type") == "team":
        return participant.get("members") or []
    return [participant.get("user") or {}]


def _members_cell(members: list[dict[str, Any]]) -> str:
    return "；".join(
        f"{member.get('name') or member.get('user_id') or '未命名'}"
        f"（{member.get('school') or '未填写'}/{member.get('education') or '未填写'}）"
        for member in members
    )


def _read_user_profiles(paths: CheckPaths) -> dict[str, dict[str, Any]]:
    if not paths.user_source_path.is_file():
        return {}
    users = pd.read_csv(paths.user_source_path, dtype=str)
    users = users.loc[users["competition_id"].eq(paths.run_dir.parents[2].name)].copy()
    profiles: dict[str, dict[str, Any]] = {}
    for row in users.to_dict("records"):
        try:
            data = json.loads(row.get("data") or "{}")
        except (TypeError, json.JSONDecodeError):
            data = {}
        profiles[str(row["user_id"])] = {
            "name": data.get("name"), "school": data.get("school"),
            "education": data.get("education"),
        }
    if paths.account_source_path.is_file():
        accounts = pd.read_csv(
            paths.account_source_path, dtype=str, usecols=["id", "username", "nickname"]
        )
        for row in accounts.to_dict("records"):
            profile = profiles.setdefault(str(row["id"]), {})
            # 榜单展示名优先使用用户设置的 nickname，其次才是登录 username。
            profile["display_name"] = _clean(row.get("nickname")) or _clean(row.get("username"))
    return profiles


def _enrich_members(members: list[dict[str, Any]], profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for member in members:
        user_id = str(member.get("user_id") or "")
        profile = profiles.get(user_id, {})
        enriched.append({
            "user_id": user_id or None,
            "name": profile.get("name") or member.get("name"),
            "display_name": profile.get("display_name") or member.get("display_name"),
            "school": profile.get("school") or member.get("school"),
            "education": profile.get("education"),
        })
    return enriched


def _read_teams(paths: CheckPaths) -> dict[str, dict[str, Any]]:
    if not paths.team_source_path.is_file():
        return {}
    teams = pd.read_csv(paths.team_source_path, dtype=str)
    competition_id = paths.run_dir.parents[2].name
    teams = teams.loc[teams["competition_id"].eq(competition_id)]
    result = {}
    for row in teams.to_dict("records"):
        try:
            member_ids = json.loads(row.get("members") or "[]")
        except (TypeError, json.JSONDecodeError):
            member_ids = []
        # ``members`` 不包含队长（creator），团队归属必须把两者合并。
        member_ids = list(dict.fromkeys([
            *([str(row["creator"])] if row.get("creator") else []),
            *[str(user_id) for user_id in member_ids],
        ]))
        result[str(row["id"])] = {
            "name": row.get("name"),
            "member_ids": member_ids,
        }
    return result


def _public_score_from_source(row: pd.Series) -> float | None:
    """读取 submission 的未截断公榜分数，失败时回退表中的公榜分数。"""
    try:
        score_data = json.loads(row.get("public_score_data") or "{}")
    except (TypeError, json.JSONDecodeError):
        score_data = {}
    score = score_data.get("score", score_data.get("final_score"))
    if score is None:
        score = row.get("public_score")
    return _clean(pd.to_numeric(score, errors="coerce"))


def build_team_private_report(paths: CheckPaths = PATHS) -> dict[str, Any]:
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    metadata_by_id = {
        str(item["submission_id"]): item for item in metadata.get("submissions", [])
        if item.get("submission_id")
    }
    profiles = _read_user_profiles(paths)
    teams_by_id = _read_teams(paths)
    source_submissions = pd.read_csv(paths.submission_source_path, dtype=str).set_index("id", drop=False)
    competition_id = str(metadata.get("competition_id") or paths.run_dir.parents[2].name)
    competition_submissions = source_submissions.loc[
        source_submissions["competition_id"].eq(competition_id)
    ]
    detail = read_summary(paths).merge(
        read_score(paths), on="submission_id", how="outer", suffixes=("", "_score")
    ).merge(read_final(paths), on="submission_id", how="outer", suffixes=("", "_final"))
    if "score_final" in detail:
        detail["score"] = detail["score_final"].combine_first(detail.get("score"))
    detail_by_id = detail.set_index("submission_id", drop=False)
    public = read_public_scores(paths).set_index("submission_id", drop=False)
    ranks = detail["score"].rank(ascending=False, method="min", na_option="bottom")
    rank_by_id = dict(zip(detail["submission_id"].astype(str), ranks))
    public_ranks = public["public_score"].rank(ascending=False, method="min", na_option="keep")
    public_rank_by_id = dict(zip(public.index.astype(str), public_ranks))

    participants = []
    for participant in metadata.get("participants", []):
        participant_type = participant.get("type") or "individual"
        participant_id = participant.get("team_id")
        team_source = teams_by_id.get(str(participant_id), {}) if participant_id else {}
        metadata_members = _members(participant)
        source_member_ids = team_source.get("member_ids", []) if participant_type == "team" else []
        member_ids = list(dict.fromkeys([
            *source_member_ids,
            *[str(member.get("user_id")) for member in metadata_members if member.get("user_id")],
        ]))
        source_members = ([{"user_id": user_id} for user_id in member_ids] if member_ids else metadata_members)
        members = _enrich_members(source_members, profiles)
        participant_id = participant_id or (members[0].get("user_id") if members else None)
        name = (team_source.get("name") or participant.get("team_name")) if participant_type == "team" else (
            (members[0].get("display_name") or members[0].get("user_id")) if members else None
        )
        submissions = []
        for submission_id in map(str, participant.get("private_submission_ids") or []):
            row = detail_by_id.loc[submission_id] if submission_id in detail_by_id.index else pd.Series(dtype=object)
            pub = public.loc[submission_id] if submission_id in public.index else pd.Series(dtype=object)
            meta = metadata_by_id.get(submission_id, {})
            source = source_submissions.loc[submission_id] if submission_id in source_submissions.index else pd.Series(dtype=object)
            private_score = _clean(pd.to_numeric(row.get("score"), errors="coerce"))
            public_score = _clean(pd.to_numeric(pub.get("public_score", meta.get("public_score")), errors="coerce"))
            submissions.append({
                "submission_id": submission_id,
                "created_at": _clean(source.get("created_at")) or meta.get("created_at")
                or (meta.get("submission") or {}).get("created_at"),
                "user_id": _clean(source.get("user_id")) or meta.get("user_id"), "status": _clean(row.get("status")),
                "selected_for_private": _clean_bool(source.get("selected_for_private")),
                "failure_type": _clean(row.get("failure_type")), "error": _clean(row.get("error")),
                "private_rank_all_submissions": _clean_rank(rank_by_id.get(submission_id)),
                "private_score": private_score,
                "private_metrics": {metric: _clean(row.get(metric)) for metric in METRICS},
                "public_score": public_score,
                "public_rank_all_submissions": _clean_rank(public_rank_by_id.get(submission_id)),
                "public_metrics": {metric: _clean(pub.get(metric)) for metric in METRICS},
                "public_score_found": submission_id in public.index,
                "score_delta": _clean(private_score - public_score)
                if private_score is not None and public_score is not None else None,
            })
        submissions.sort(key=lambda x: (x["private_score"] is None, -(x["private_score"] or 0)))
        best = next((item for item in submissions if item["private_score"] is not None), None)
        public_candidates = competition_submissions.loc[
            competition_submissions["user_id"].isin(member_ids)
        ].copy()
        public_candidates["_public_score"] = public_candidates.apply(_public_score_from_source, axis=1)
        public_candidates = public_candidates.loc[public_candidates["_public_score"].notna()]
        best_public = (
            public_candidates.sort_values(["_public_score", "created_at"], ascending=[False, True]).iloc[0]
            if not public_candidates.empty else None
        )
        participants.append({
            "participant_type": participant_type, "participant_id": participant_id,
            "participant_name": name or "（未命名参赛方）", "members": members,
            "submission_count": len(submissions),
            "best_private_submission_id": best["submission_id"] if best else None,
            "best_private_score": best["private_score"] if best else None,
            "best_public_submission_id": str(best_public["id"]) if best_public is not None else None,
            "team_public_score": round(float(best_public["_public_score"]), 5) if best_public is not None else None,
            "submissions": submissions,
        })
    participant_ranks = pd.Series([p["best_private_score"] for p in participants], dtype=float).rank(
        ascending=False, method="min", na_option="bottom"
    )
    for participant, rank in zip(participants, participant_ranks):
        participant["private_rank"] = _clean_rank(rank)
    public_team_ranks = pd.Series([p["team_public_score"] for p in participants], dtype=float).rank(
        ascending=False, method="min", na_option="bottom"
    )
    for participant, rank in zip(participants, public_team_ranks):
        participant["public_rank"] = _clean_rank(rank)
    participants.sort(key=lambda x: (x["private_rank"] or 10**9, x["participant_name"]))
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "competition_id": metadata.get("competition_id"), "batch_id": metadata.get("batch_id"),
        "participant_count": len(participants),
        "submission_count": sum(p["submission_count"] for p in participants),
        "scoring_note": "四项指标百分位排名等权；本比赛没有 A/B 或回归 B 项评分。",
        "participants": participants,
    }


def build_team_submission_detail(paths: CheckPaths = PATHS) -> pd.DataFrame:
    report = build_team_private_report(paths)
    rows = []
    for participant in report["participants"]:
        for sub in participant["submissions"]:
            rows.append({
                "参赛方": participant["participant_name"], "参赛类型": participant["participant_type"],
                "参赛方ID": participant["participant_id"], "私榜排名": participant["private_rank"],
                "私榜最高分": participant["best_private_score"], "公榜排名": participant["public_rank"],
                "公榜得分": participant["team_public_score"], "提交数量": participant["submission_count"],
                "成员名称及对应学校": _members_cell(participant["members"]),
                "submission_id": sub["submission_id"], "提交时间": sub["created_at"],
                "提交者ID": sub["user_id"], "私榜排名（全部提交）": sub["private_rank_all_submissions"],
                "人为选择私榜": sub["selected_for_private"],
                "是否参赛方最佳私榜提交": sub["submission_id"] == participant["best_private_submission_id"],
                "评测状态": sub["status"], "失败类型": sub["failure_type"], "评测错误": sub["error"],
                "私榜得分": sub["private_score"],
                **{f"私榜{m}": sub["private_metrics"].get(m) for m in METRICS},
                "公榜得分（submission）": sub["public_score"],
                **{f"公榜{m}": sub["public_metrics"].get(m) for m in METRICS},
                "私榜减公榜": sub["score_delta"],
            })
    return pd.DataFrame(rows)


def build_team_private_leaderboard(paths: CheckPaths = PATHS) -> pd.DataFrame:
    """生成用户要求的团队级 CSV 数据。"""
    report = build_team_private_report(paths)
    rows = [{
        "团队名": participant["participant_name"],
        "团队成员（学校/学历）": _members_cell(participant["members"]),
        "团队私榜提交数量": participant["submission_count"],
        "私榜得分": participant["best_private_score"],
        "私榜得分排名": participant["private_rank"],
        "公榜得分": participant["team_public_score"],
        "公榜得分排名": participant["public_rank"],
    } for participant in report["participants"]]
    return pd.DataFrame(rows, columns=TEAM_SUMMARY_COLUMNS)


def build_team_leaderboard_summary(paths: CheckPaths = PATHS) -> pd.DataFrame:
    return build_team_private_leaderboard(paths)


def _fmt(value: Any, *, decimal_places: int = 6) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.{decimal_places}f}" if isinstance(value, float) else str(value)


def _json_report(report: dict[str, Any]) -> dict[str, Any]:
    teams = []
    for participant in report["participants"]:
        submissions = []
        for sub in participant["submissions"]:
            submissions.append({
                "submission_id": sub["submission_id"],
                "提交时间": sub["created_at"],
                "提交者ID": sub["user_id"],
                "selected_for_private": sub["selected_for_private"],
                "评测状态": sub["status"],
                "失败类型": sub["failure_type"],
                "评测错误": sub["error"],
                "是否团队最佳私榜提交": sub["submission_id"] == participant["best_private_submission_id"],
                "私榜": {
                    "排名（全部submission）": sub["private_rank_all_submissions"],
                    "最终得分": sub["private_score"],
                    "细节得分": sub["private_metrics"],
                },
                "公榜": {
                    "排名（全部submission）": sub["public_rank_all_submissions"],
                    "最终得分": sub["public_score"],
                    "细节得分": sub["public_metrics"],
                },
                "私榜减公榜": sub["score_delta"],
            })
        teams.append({
            "团队": participant["participant_name"],
            "参赛类型": participant["participant_type"],
            "团队ID": participant["participant_id"],
            "成员": [{
                "用户ID": member["user_id"], "姓名": member["name"],
                "学校": member["school"], "学历": member["education"],
            } for member in participant["members"]],
            "团队私榜提交数量": participant["submission_count"],
            "团队最佳私榜submission_id": participant["best_private_submission_id"],
            "团队私榜得分": participant["best_private_score"],
            "团队私榜得分排名": participant["private_rank"],
            "团队公榜得分": participant["team_public_score"],
            "团队最佳公榜submission_id": participant["best_public_submission_id"],
            "团队公榜得分排名": participant["public_rank"],
            "submissions": submissions,
        })
    return {
        "生成时间": report["generated_at"],
        "比赛ID": report["competition_id"],
        "批次ID": report["batch_id"],
        "团队及个人参赛方数量": report["participant_count"],
        "私榜submission数量": report["submission_count"],
        "评分说明": report["scoring_note"],
        "团队": teams,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 团队公私榜报告", "", f"- 团队/个人参赛方数：**{report['participant_count']}**",
        f"- 私榜提交数：**{report['submission_count']}**", f"- 评分说明：{report['scoring_note']}", "",
        "## 团队排名", "",
        "| 私榜排名 | 公榜排名 | 团队名 | 团队成员（学校/学历） | 私榜提交数量 | 私榜得分 | 公榜得分 |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for participant in report["participants"]:
        lines.append(
            f"| {_fmt(participant['private_rank'])} | {_fmt(participant['public_rank'])} | "
            f"{participant['participant_name']} | {_members_cell(participant['members'])} | "
            f"{participant['submission_count']} | {_fmt(participant['best_private_score'])} | "
            f"{_fmt(participant['team_public_score'])} |"
        )
    lines.extend(["", "## 各团队 submission 详解", ""])
    for participant in report["participants"]:
        lines.extend([
            f"### {_fmt(participant['private_rank'])}. {participant['participant_name']}", "",
            f"成员：{_members_cell(participant['members']) or '—'}", "",
            "| 最佳 | 人为选择私榜 | submission_id | 状态 | 榜单 | 全榜排名 | IC均值 | ICIR | 夏普 | 压力ICIR | 总分 |",
            "|:---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for sub in participant["submissions"]:
            prefix = (
                f"| {'✓' if sub['submission_id'] == participant['best_private_submission_id'] else ''} | "
                f"{'True' if sub['selected_for_private'] is True else 'False' if sub['selected_for_private'] is False else '—'} | "
                f"{sub['submission_id']} | {sub['status'] or '—'}"
            )
            for label, rank, metrics, score in (
                ("私榜", sub["private_rank_all_submissions"], sub["private_metrics"], sub["private_score"]),
                ("公榜", sub["public_rank_all_submissions"], sub["public_metrics"], sub["public_score"]),
            ):
                lines.append(
                    f"{prefix} | {label} | {_fmt(rank)} | " + " | ".join(_fmt(metrics.get(m)) for m in METRICS)
                    + f" | {_fmt(score)} |"
                )
                prefix = "|  |  |  |  "
        lines.append("")
    return "\n".join(lines)


def _competition_notice_markdown(
    report: dict[str, Any], *, submission_id_length: int = 8, score_decimal_places: int = 4
) -> str:
    """生成赛事通知单表；用 Markdown 内嵌 HTML 实现二级表头及团队单元格合并。"""

    def cell(value: Any) -> str:
        return escape(
            _fmt(value, decimal_places=score_decimal_places), quote=True
        ).replace("\n", "<br>")

    lines = [
        "# 赛事排名通知", "",
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th colspan="2">团队排名</th>',
        '      <th colspan="2">团队信息</th>',
        '      <th colspan="2">团队最终得分</th>',
        '      <th>Submission</th>',
        '      <th colspan="2">Submission 最终得分</th>',
        '      <th colspan="4">私榜细分项</th>',
        '      <th colspan="4">公榜细分项</th>',
        "    </tr>",
        "    <tr>",
        "      <th>私榜</th><th>公榜</th>",
        "      <th>团队名</th><th>私榜提交数量</th>",
        "      <th>私榜</th><th>公榜</th>",
        "      <th>ID</th>",
        "      <th>私榜</th><th>公榜</th>",
        "      <th>IC均值</th><th>ICIR</th><th>夏普</th><th>压力ICIR</th>",
        "      <th>IC均值</th><th>ICIR</th><th>夏普</th><th>压力ICIR</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for participant in report["participants"]:
        # 没有私榜 submission 的团队也保留一行团队汇总信息。
        submissions = participant["submissions"] or [None]
        for submission_index, submission in enumerate(submissions):
            private_metrics = submission["private_metrics"] if submission else {}
            public_metrics = submission["public_metrics"] if submission else {}
            submission_id = submission["submission_id"] if submission else None
            lines.append("    <tr>")
            if submission_index == 0:
                rowspan = len(submissions)
                for value in (
                    participant["private_rank"], participant["public_rank"],
                    participant["participant_name"], participant["submission_count"],
                    participant["best_private_score"], participant["team_public_score"],
                ):
                    lines.append(f'      <td rowspan="{rowspan}">{cell(value)}</td>')
            values = [
                str(submission_id)[:submission_id_length] if submission_id else None,
                submission["private_score"] if submission else None,
                submission["public_score"] if submission else None,
                private_metrics.get("ic_mean"), private_metrics.get("ic_ir"),
                private_metrics.get("sharpe_ratio"), private_metrics.get("stress_ic_ir"),
                public_metrics.get("ic_mean"), public_metrics.get("ic_ir"),
                public_metrics.get("sharpe_ratio"), public_metrics.get("stress_ic_ir"),
            ]
            lines.extend(f"      <td>{cell(value)}</td>" for value in values)
            lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def export_team_private_leaderboard(
    paths: CheckPaths = PATHS, *, output: str | Path | None = None,
    json_output: str | Path | None = None, markdown_output: str | Path | None = None,
    team_summary_output: str | Path | None = None,
    competition_notice_output: str | Path | None = None,
) -> dict[str, Path]:
    csv_path = Path(output).expanduser().resolve() if output else paths.team_private_leaderboard_path
    json_path = Path(json_output).expanduser().resolve() if json_output else csv_path.with_suffix(".json")
    md_path = Path(markdown_output).expanduser().resolve() if markdown_output else csv_path.with_suffix(".md")
    team_summary_path = (
        Path(team_summary_output).expanduser().resolve()
        if team_summary_output else paths.team_leaderboard_summary_path
    )
    competition_notice_path = (
        Path(competition_notice_output).expanduser().resolve()
        if competition_notice_output else csv_path.with_name("competition_notice.md")
    )
    report = build_team_private_report(paths)
    for path in (csv_path, json_path, md_path, team_summary_path, competition_notice_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_team_private_leaderboard(paths)
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary.to_csv(team_summary_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(_json_report(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    competition_notice_path.write_text(
        _competition_notice_markdown(report), encoding="utf-8"
    )
    return {
        "csv": csv_path, "json": json_path, "markdown": md_path,
        "team_summary_csv": team_summary_path,
        "competition_notice_markdown": competition_notice_path,
    }
