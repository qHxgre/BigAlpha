"""生成团队和个人全部提交及公私榜得分明细。

可直接运行：

    python -m manual_checks.team_private_leaderboard

默认在本次运行的 artifacts 目录同时生成 JSON、Markdown 和 CSV。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    METRICS, REGRESSION_METRICS, read_final, read_public_regression,
    read_public_scores, read_regression, read_summary,
)
from .config import (
    CONFIG,
    FACTOR_SIMILARITY_SUMMARY_FILENAME,
    PATHS,
    PERIOD_COMPARISON_FILENAME,
    TEAM_PERIOD_COMPARISON_FILENAME,
    TEAM_PRIVATE_LEADERBOARD_FILENAME,
    CheckPaths,
)


CSV_COLUMNS = [
    "团队名", "团队ID", "私榜排名", "团队私榜最高分", "团队公榜排名", "团队公榜得分",
    "提交数量", "成员名称及对应学校", "submission_id", "提交时间",
    "提交者ID", "私榜排名（全部提交）", "是否团队最佳私榜提交", "评测状态", "评测错误",
    "私榜得分", "私榜A分", "私榜B分", *[f"私榜{metric}" for metric in METRICS],
    *[f"私榜B项{metric}" for metric in REGRESSION_METRICS],
    "公榜得分", "公榜A分", "公榜B分", *[f"公榜{metric}" for metric in METRICS],
    *[f"公榜B项{metric}" for metric in REGRESSION_METRICS],
    "私榜减公榜",
]

TEAM_SUMMARY_COLUMNS = [
    "参赛类型", "团队名", "团队成员及学校", "团队私榜提交数量", "团队私榜得分",
    "团队私榜排名", "私榜计分submission", "团队公榜得分", "团队公榜排名",
]


def _clean(value: Any) -> Any:
    """把 numpy/pandas 标量和缺失值转换为可序列化的原生值。"""
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _score_detail(values: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    return {key: _clean(values.get(key)) for key in keys if _clean(values.get(key)) is not None}


def _members_cell(members: list[dict[str, Any]]) -> str:
    return "；".join(
        f"{member.get('name') or member.get('user_id') or '未命名'}"
        f"（{member.get('school') or '未填写'}/{member.get('education') or '未填写'}）"
        for member in members
    )


def _user_profiles(paths: CheckPaths) -> dict[str, dict[str, Any]]:
    if not paths.user_source_path.is_file():
        return {}
    users = pd.read_csv(paths.user_source_path, dtype=str)
    users = users.loc[users["competition_id"].eq(paths.private_code_dir.parent.name)]
    result = {}
    for row in users.to_dict("records"):
        try:
            data = json.loads(row.get("data") or "{}")
        except (TypeError, json.JSONDecodeError):
            data = {}
        result[str(row["user_id"])] = data
    return result


def _public_score_from_source(row: pd.Series) -> float | None:
    """读取原始 submission 中未截断的公榜最终分。"""
    try:
        detail = json.loads(row.get("public_score_data") or "{}")
    except (TypeError, json.JSONDecodeError):
        detail = {}
    value = detail.get("score", detail.get("final_score", row.get("public_score")))
    return _clean(pd.to_numeric(value, errors="coerce"))


def _team_member_ids(paths: CheckPaths) -> dict[str, list[str]]:
    if not paths.team_source_path.is_file():
        return {}
    teams = pd.read_csv(paths.team_source_path, dtype=str)
    competition_id = paths.private_code_dir.parent.name
    teams = teams.loc[teams["competition_id"].eq(competition_id)]
    result: dict[str, list[str]] = {}
    for row in teams.to_dict("records"):
        try:
            members = json.loads(row.get("members") or "[]")
        except (TypeError, json.JSONDecodeError):
            members = []
        result[str(row["id"])] = list(dict.fromkeys([
            *([str(row["creator"])] if row.get("creator") else []),
            *[str(value) for value in members],
        ]))
    return result


def _load_similarity(paths: CheckPaths) -> pd.DataFrame:
    path = paths.incremental_dir / FACTOR_SIMILARITY_SUMMARY_FILENAME
    if not path.is_file():
        return pd.DataFrame(columns=["submission_id_1", "submission_id_2"])
    data = pd.read_csv(path, dtype={"submission_id_1": str, "submission_id_2": str})
    for column in data.columns:
        if column not in {"submission_id_1", "submission_id_2", "high_similarity"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _team_correlations(submission_ids: list[str], similarity: pd.DataFrame) -> dict[str, Any]:
    ids = set(submission_ids)
    pairs = similarity.loc[
        similarity["submission_id_1"].isin(ids) & similarity["submission_id_2"].isin(ids)
    ].copy()
    if pairs.empty:
        return {
            "pair_count": 0,
            "expected_pair_count": len(ids) * (len(ids) - 1) // 2,
            "mean_abs_correlation": None,
            "max_abs_correlation": None,
            "high_correlation_pair_count": 0,
            "high_correlation_threshold": CONFIG.high_correlation,
            "top_pair": None,
            "pairs": [],
        }
    abs_column = "mean_abs_correlation" if "mean_abs_correlation" in pairs else "abs_correlation"
    pairs = pairs.sort_values(abs_column, ascending=False, na_position="last")
    pair_records = [
        {str(key): _clean(value) for key, value in row.items()}
        for row in pairs.to_dict(orient="records")
    ]
    high_count = int(pd.to_numeric(pairs[abs_column], errors="coerce").ge(CONFIG.high_correlation).sum())
    return {
        "pair_count": len(pairs),
        "expected_pair_count": len(ids) * (len(ids) - 1) // 2,
        "mean_abs_correlation": _clean(pd.to_numeric(pairs[abs_column], errors="coerce").mean()),
        "max_abs_correlation": _clean(pd.to_numeric(pairs[abs_column], errors="coerce").max()),
        "high_correlation_pair_count": high_count,
        "high_correlation_threshold": CONFIG.high_correlation,
        "top_pair": pair_records[0],
        "pairs": pair_records,
    }


def build_team_private_report(paths: CheckPaths = PATHS) -> dict[str, Any]:
    """构建便于 JSON 保存的团队级完整私榜报告。"""
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    submissions_by_id = {
        str(item["submission_id"]): item
        for item in metadata.get("submissions", []) if item.get("submission_id")
    }
    summary = read_summary(paths)
    final = read_final(paths)
    detail = summary.merge(final, on="submission_id", how="outer", suffixes=("", "_final"))
    for column in [*METRICS, "a_score", "b_score", "final_score"]:
        if column in detail:
            detail[column] = pd.to_numeric(detail[column], errors="coerce")
    detail_by_id = detail.set_index("submission_id", drop=False)
    public_scores = read_public_scores(paths).set_index("submission_id", drop=False)
    private_regression = read_regression(paths).set_index("factor", drop=False)
    public_regression = read_public_regression(paths).set_index("factor", drop=False)
    similarity = _load_similarity(paths)
    user_profiles = _user_profiles(paths)
    team_members = _team_member_ids(paths)
    source_submissions = pd.read_csv(paths.submission_source_path, dtype=str)
    source_submissions = source_submissions.loc[
        source_submissions["competition_id"].eq(paths.private_code_dir.parent.name)
    ].copy()
    source_submissions["_public_score"] = source_submissions.apply(
        _public_score_from_source, axis=1
    )

    participants: list[dict[str, Any]] = []
    all_private_scores = pd.to_numeric(detail.get("final_score"), errors="coerce")
    submission_rank = all_private_scores.rank(ascending=False, method="min", na_option="bottom")
    rank_by_id = dict(zip(detail["submission_id"].astype(str), submission_rank))

    for participant in metadata.get("participants", []):
        participant_type = participant.get("type")
        if participant_type not in {"team", "individual"}:
            continue
        if participant_type == "team":
            participant_id = str(participant.get("team_id"))
            participant_name = participant.get("team_name") or "（未命名队伍）"
            members = participant.get("members") or []
            team_id = participant.get("team_id")
            member_ids = team_members.get(str(team_id)) or [
                str(member.get("user_id")) for member in members if member.get("user_id")
            ]
        else:
            user = participant.get("user") or {}
            participant_id = str(user.get("user_id") or "")
            participant_name = f"个人：{user.get('name') or participant_id or '未命名'}"
            members = [user]
            team_id = None
            member_ids = [str(user.get("user_id"))] if user.get("user_id") else []
        members = [
            {
                **member,
                **{
                    key: value
                    for key, value in user_profiles.get(str(member.get("user_id")), {}).items()
                    if key in {"name", "school", "education"} and value
                },
            }
            for member in members
        ]
        submission_ids = [str(value) for value in participant.get("private_submission_ids") or []]
        submission_rows: list[dict[str, Any]] = []
        for submission_id in submission_ids:
            meta = submissions_by_id.get(submission_id, {})
            raw_submission = meta.get("submission") or {}
            row = detail_by_id.loc[submission_id] if submission_id in detail_by_id.index else pd.Series(dtype=object)
            public_row = (
                public_scores.loc[submission_id]
                if submission_id in public_scores.index else pd.Series(dtype=object)
            )
            public_detail = _score_detail(
                {
                    **public_row.to_dict(),
                    "final_score": public_row.get("public_score"),
                },
                ("a_score", "b_score", "final_score", *METRICS),
            )
            private_detail = {
                **_score_detail(row.to_dict(), ("a_score", "b_score", "final_score", *METRICS)),
            }
            private_b_row = (
                private_regression.loc[submission_id]
                if submission_id in private_regression.index else pd.Series(dtype=object)
            )
            public_b_row = (
                public_regression.loc[submission_id]
                if submission_id in public_regression.index else pd.Series(dtype=object)
            )
            private_b_detail = _score_detail(private_b_row.to_dict(), REGRESSION_METRICS)
            public_b_detail = _score_detail(public_b_row.to_dict(), REGRESSION_METRICS)
            private_score = private_detail.get("final_score")
            public_score = _clean(pd.to_numeric(
                public_detail.get("final_score", meta.get("public_score")), errors="coerce"
            ))
            submission_rows.append({
                "submission_id": submission_id,
                "user_id": meta.get("user_id"),
                "created_at": meta.get("created_at"),
                "status": _clean(row.get("status")),
                "error": _clean(row.get("error")),
                "private_rank_all_submissions": _clean(rank_by_id.get(submission_id)),
                "private_score": private_score,
                "private_score_detail": private_detail,
                "private_b_detail": private_b_detail,
                "public_score": public_score,
                "public_score_detail": public_detail,
                "public_b_detail": public_b_detail,
                "public_score_found": submission_id in public_scores.index,
                "score_delta": _clean(private_score - public_score)
                if private_score is not None and public_score is not None else None,
            })
        submission_rows.sort(
            key=lambda item: (item["private_score"] is None, -(item["private_score"] or 0), item["submission_id"])
        )
        best = next((item for item in submission_rows if item["private_score"] is not None), None)
        correlations = _team_correlations(submission_ids, similarity)
        public_candidates = source_submissions.loc[
            source_submissions["user_id"].isin(member_ids)
            & source_submissions["_public_score"].notna()
        ].sort_values(["_public_score", "created_at"], ascending=[False, True])
        best_public = public_candidates.iloc[0] if not public_candidates.empty else None
        participants.append({
            "participant_type": participant_type,
            "participant_id": participant_id,
            "participant_name": participant_name,
            # 保留原字段，兼容现有 JSON/CSV/报告消费者。
            "team_id": team_id,
            "team_name": participant_name,
            "members": members,
            "submission_count": len(submission_ids),
            "metadata_submission_count": participant.get("private_submission_count"),
            "best_public_submission_id": (
                str(best_public["id"]) if best_public is not None else None
            ),
            "public_rank": None,
            "public_score": (
                _clean(best_public["_public_score"]) if best_public is not None else None
            ),
            "best_private_submission_id": best["submission_id"] if best else None,
            "best_private_score": best["private_score"] if best else None,
            "correlation_summary": {key: value for key, value in correlations.items() if key != "pairs"},
            "correlation_pairs": correlations["pairs"],
            "submissions": submission_rows,
        })

    scores = pd.Series([item["best_private_score"] for item in participants], dtype="float64")
    ranks = scores.rank(ascending=False, method="min", na_option="bottom")
    for participant, rank in zip(participants, ranks):
        participant["private_rank"] = _clean(rank)
    public_scores_by_participant = pd.Series(
        [item["public_score"] for item in participants], dtype="float64"
    )
    public_ranks = public_scores_by_participant.rank(
        ascending=False, method="min", na_option="bottom"
    )
    for participant, rank in zip(participants, public_ranks):
        participant["public_rank"] = _clean(rank)
    participants.sort(
        key=lambda item: (
            item["private_rank"] is None,
            item["private_rank"] or 10**9,
            item["participant_name"],
        )
    )
    team_count = sum(item["participant_type"] == "team" for item in participants)
    individual_count = sum(item["participant_type"] == "individual" for item in participants)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "competition_id": metadata.get("competition_id"),
        "competition_name": metadata.get("competition_name"),
        "batch_id": metadata.get("batch_id"),
        "participant_count": len(participants),
        "team_count": team_count,
        "individual_count": individual_count,
        "submission_count": sum(item["submission_count"] for item in participants),
        "correlation_method": "同一参赛主体内 submission 两两日截面 Pearson 相关的汇总",
        "participants": participants,
        # 保留 teams 键兼容旧调用；现在其中包含团队和个人两类参赛主体。
        "teams": participants,
    }


def build_compact_team_private_report(report: dict[str, Any]) -> dict[str, Any]:
    """压缩完整报告，保留最终报告和外部展示需要的核心字段。"""
    teams = []
    for team in report["teams"]:
        correlation = team["correlation_summary"]
        top_pair = correlation.get("top_pair") or {}
        teams.append({
            "participant_type": team["participant_type"],
            "participant_id": team["participant_id"],
            "participant_name": team["participant_name"],
            "private_rank": team["private_rank"],
            "team_id": team["team_id"],
            "team_name": team["team_name"],
            "members": [
                {
                    "name": member.get("name"),
                    "school": member.get("school"),
                }
                for member in team["members"]
            ],
            "submission_count": team["submission_count"],
            "public_rank": team["public_rank"],
            "public_score": team["public_score"],
            "best_private_submission_id": team["best_private_submission_id"],
            "best_private_score": team["best_private_score"],
            "correlation": {
                "pair_count": correlation["pair_count"],
                "expected_pair_count": correlation["expected_pair_count"],
                "mean_abs_correlation": correlation["mean_abs_correlation"],
                "max_abs_correlation": correlation["max_abs_correlation"],
                "high_correlation_pair_count": correlation["high_correlation_pair_count"],
                "high_correlation_threshold": correlation["high_correlation_threshold"],
                "top_pair": {
                    key: top_pair.get(key)
                    for key in (
                        "submission_id_1", "submission_id_2", "mean_correlation",
                        "mean_abs_correlation", "p95_abs_correlation", "valid_days",
                    )
                    if top_pair.get(key) is not None
                } or None,
            },
            "submissions": [
                {
                    "submission_id": submission["submission_id"],
                    "status": submission["status"],
                    "private_rank": submission["private_rank_all_submissions"],
                    "private_score": submission["private_score_detail"],
                    "private_b_detail": submission["private_b_detail"],
                    "public_score": submission["public_score_detail"],
                    "public_b_detail": submission["public_b_detail"],
                    "public_score_found": submission["public_score_found"],
                    "score_delta": submission["score_delta"],
                }
                for submission in team["submissions"]
            ],
        })
    compact = {
        key: report[key]
        for key in (
            "generated_at", "competition_id", "competition_name", "batch_id",
            "participant_count", "team_count", "individual_count",
            "submission_count", "correlation_method",
        )
    }
    compact["teams"] = teams
    return compact


def build_team_private_leaderboard(paths: CheckPaths = PATHS) -> pd.DataFrame:
    """返回一行一个 submission 的扁平表，兼容原 CSV 调用入口。"""
    report = build_team_private_report(paths)
    rows: list[dict[str, Any]] = []
    for team in report["teams"]:
        corr = team["correlation_summary"]
        top_pair = corr.get("top_pair") or {}
        top_pair_cell = " ↔ ".join(filter(None, [top_pair.get("submission_id_1"), top_pair.get("submission_id_2")]))
        for submission in team["submissions"]:
            private = submission["private_score_detail"]
            public = submission["public_score_detail"]
            private_b = submission["private_b_detail"]
            public_b = submission["public_b_detail"]
            rows.append({
                "团队名": team["team_name"], "团队ID": team["team_id"], "私榜排名": team["private_rank"],
                "团队私榜最高分": team["best_private_score"], "团队公榜排名": team["public_rank"],
                "团队公榜得分": team["public_score"], "提交数量": team["submission_count"],
                "团队内相关对数量": corr["pair_count"], "团队内平均绝对相关性": corr["mean_abs_correlation"],
                "团队内最大绝对相关性": corr["max_abs_correlation"],
                "高相关对数量": corr["high_correlation_pair_count"], "最高相关提交对": top_pair_cell,
                "成员名称及对应学校": _members_cell(team["members"]), "submission_id": submission["submission_id"],
                "提交时间": submission["created_at"], "提交者ID": submission["user_id"],
                "私榜排名（全部提交）": submission["private_rank_all_submissions"],
                "是否团队最佳私榜提交": submission["submission_id"] == team["best_private_submission_id"],
                "评测状态": submission["status"], "评测错误": submission["error"],
                "私榜得分": submission["private_score"], "私榜A分": private.get("a_score"),
                "私榜B分": private.get("b_score"), **{f"私榜{m}": private.get(m) for m in METRICS},
                **{f"私榜B项{m}": private_b.get(m) for m in REGRESSION_METRICS},
                "公榜得分": submission["public_score"], "公榜A分": public.get("a_score"),
                "公榜B分": public.get("b_score"),
                **{f"公榜{m}": public.get(m) for m in METRICS},
                **{f"公榜B项{m}": public_b.get(m) for m in REGRESSION_METRICS},
                "私榜减公榜": submission["score_delta"],
            })
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def build_team_leaderboard_summary(paths: CheckPaths = PATHS) -> pd.DataFrame:
    """生成团队和个人摘要；公榜分取主体全部公榜提交中的最高分。"""
    report = build_team_private_report(paths)
    rows: list[dict[str, Any]] = []
    for participant in report["participants"]:
        rows.append({
            "参赛类型": "团队" if participant["participant_type"] == "team" else "个人",
            "团队名": participant["participant_name"],
            "团队成员及学校": _members_cell(participant["members"]),
            "团队私榜提交数量": participant["submission_count"],
            "团队私榜得分": participant["best_private_score"],
            "团队私榜排名": participant["private_rank"],
            "私榜计分submission": participant["best_private_submission_id"],
            "团队公榜得分": participant["public_score"],
            "团队公榜排名": participant["public_rank"],
        })
    summary = pd.DataFrame(rows, columns=TEAM_SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(
            ["团队私榜排名", "参赛类型", "团队名"], na_position="last", kind="stable"
        ).reset_index(drop=True)
    return summary


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 参赛主体私榜完整报告", "",
        f"- 参赛主体数：**{report['participant_count']}**",
        f"- 团队数：**{report['team_count']}**",
        f"- 个人数：**{report['individual_count']}**",
        f"- 提交数：**{report['submission_count']}**",
        f"- 相关性口径：{report['correlation_method']}", "",
        "## 参赛主体概览", "",
        "| 私榜排名 | 类型 | 队伍/个人 | 提交数 | 私榜最高分 | 公榜排名 | 公榜分 | 平均绝对相关 | 最大绝对相关 | 高相关对 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for team in report["teams"]:
        corr = team["correlation_summary"]
        lines.append(
            f"| {_fmt(team['private_rank'], 0)} | {'团队' if team['participant_type'] == 'team' else '个人'} | "
            f"{team['participant_name']} | {team['submission_count']} | "
            f"{_fmt(team['best_private_score'])} | {_fmt(team['public_rank'], 0)} | {_fmt(team['public_score'])} | "
            f"{_fmt(corr['mean_abs_correlation'])} | {_fmt(corr['max_abs_correlation'])} | "
            f"{corr['high_correlation_pair_count']} |"
        )
    for team in report["teams"]:
        corr = team["correlation_summary"]
        lines.extend(["", f"## {team['private_rank']}. {team['team_name']}", "",
            f"成员：{_members_cell(team['members']) or '—'}  ",
            f"类型：{'团队' if team['participant_type'] == 'team' else '个人'}；"
            f"提交数：{team['submission_count']}；主体内相关对：{corr['pair_count']}/{corr['expected_pair_count']}；"
            f"平均绝对相关：{_fmt(corr['mean_abs_correlation'])}；最大绝对相关：{_fmt(corr['max_abs_correlation'])}。",
            "", "| 最佳 | submission_id | 状态 | 私榜排名 | 榜单 | IC均值 | ICIR | 夏普 | 压力ICIR | A分 | B分 | final |",
            "|:---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for sub in team["submissions"]:
            pri, pub = sub["private_score_detail"], sub["public_score_detail"]
            prefix = (
                f"| {'✓' if sub['submission_id'] == team['best_private_submission_id'] else ''} | "
                f"{sub['submission_id']} | {sub['status'] or '—'} | "
                f"{_fmt(sub['private_rank_all_submissions'], 0)}"
            )
            for label, scores in (("私榜", pri), ("公榜", pub)):
                lines.append(
                    f"{prefix} | {label} | {_fmt(scores.get('ic_mean'))} | {_fmt(scores.get('ic_ir'))} | "
                    f"{_fmt(scores.get('sharpe_ratio'))} | {_fmt(scores.get('stress_ic_ir'))} | "
                    f"{_fmt(scores.get('a_score'))} | {_fmt(scores.get('b_score'))} | "
                    f"{_fmt(scores.get('final_score'))} |"
                )
                prefix = "|  |  |  | —"
            lines.extend([
                "", "B 项回归指标：",
                "", "| 榜单 | model_score | 平均绝对权重 | 绝对权重标准差 | 入选率 |",
                "|---|---:|---:|---:|---:|",
                f"| 私榜 | {_fmt(sub['private_b_detail'].get('model_score'))} | "
                f"{_fmt(sub['private_b_detail'].get('abs_weight_mean'))} | "
                f"{_fmt(sub['private_b_detail'].get('abs_weight_std'))} | "
                f"{_fmt(sub['private_b_detail'].get('selection_rate'))} |",
                f"| 公榜 | {_fmt(sub['public_b_detail'].get('model_score'))} | "
                f"{_fmt(sub['public_b_detail'].get('abs_weight_mean'))} | "
                f"{_fmt(sub['public_b_detail'].get('abs_weight_std'))} | "
                f"{_fmt(sub['public_b_detail'].get('selection_rate'))} |",
            ])
        if team["correlation_pairs"]:
            lines.extend(["", "### 主体内 submission 相关性", "",
                "| submission 1 | submission 2 | 平均相关 | 平均绝对相关 | P95 绝对相关 | 有效天数 |",
                "|---|---|---:|---:|---:|---:|"])
            for pair in team["correlation_pairs"]:
                lines.append(
                    f"| {pair.get('submission_id_1')} | {pair.get('submission_id_2')} | "
                    f"{_fmt(pair.get('mean_correlation', pair.get('pearson')))} | "
                    f"{_fmt(pair.get('mean_abs_correlation', pair.get('abs_correlation')))} | "
                    f"{_fmt(pair.get('p95_abs_correlation'))} | {_fmt(pair.get('valid_days'), 0)} |"
                )
    return "\n".join(lines) + "\n"


def export_team_private_leaderboard(
    paths: CheckPaths = PATHS, *, output: str | Path | None = None,
    json_output: str | Path | None = None, markdown_output: str | Path | None = None,
    team_summary_output: str | Path | None = None,
) -> dict[str, Path]:
    """同时生成完整报告和团队公私榜摘要，返回各输出路径。"""
    csv_path = Path(output).expanduser().resolve() if output else paths.artifacts_dir / TEAM_PRIVATE_LEADERBOARD_FILENAME
    json_path = Path(json_output).expanduser().resolve() if json_output else csv_path.with_suffix(".json")
    md_path = Path(markdown_output).expanduser().resolve() if markdown_output else csv_path.with_suffix(".md")
    team_summary_path = (
        Path(team_summary_output).expanduser().resolve()
        if team_summary_output else paths.team_leaderboard_summary_path
    )
    for path in (csv_path, json_path, md_path, team_summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report = build_team_private_report(paths)
    compact_report = build_compact_team_private_report(report)
    json_path.write_text(json.dumps(compact_report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    # 避免重新读取和计算相关性，直接从已构建报告生成 CSV。
    rows = []
    for team in report["teams"]:
        corr = team["correlation_summary"]
        top = corr.get("top_pair") or {}
        pair_text = " ↔ ".join(filter(None, [top.get("submission_id_1"), top.get("submission_id_2")]))
        for sub in team["submissions"]:
            pri, pub = sub["private_score_detail"], sub["public_score_detail"]
            private_b, public_b = sub["private_b_detail"], sub["public_b_detail"]
            rows.append({
                "团队名": team["team_name"], "团队ID": team["team_id"], "私榜排名": team["private_rank"],
                "团队私榜最高分": team["best_private_score"], "团队公榜排名": team["public_rank"], "团队公榜得分": team["public_score"],
                "提交数量": team["submission_count"], "团队内相关对数量": corr["pair_count"], "团队内平均绝对相关性": corr["mean_abs_correlation"],
                "团队内最大绝对相关性": corr["max_abs_correlation"], "高相关对数量": corr["high_correlation_pair_count"], "最高相关提交对": pair_text,
                "成员名称及对应学校": _members_cell(team["members"]), "submission_id": sub["submission_id"], "提交时间": sub["created_at"],
                "提交者ID": sub["user_id"], "私榜排名（全部提交）": sub["private_rank_all_submissions"],
                "是否团队最佳私榜提交": sub["submission_id"] == team["best_private_submission_id"], "评测状态": sub["status"], "评测错误": sub["error"],
                "私榜得分": sub["private_score"], "私榜A分": pri.get("a_score"), "私榜B分": pri.get("b_score"),
                **{f"私榜{m}": pri.get(m) for m in METRICS}, "公榜得分": sub["public_score"], "公榜A分": pub.get("a_score"),
                **{f"私榜B项{m}": private_b.get(m) for m in REGRESSION_METRICS},
                "公榜B分": pub.get("b_score"), **{f"公榜{m}": pub.get(m) for m in METRICS},
                **{f"公榜B项{m}": public_b.get(m) for m in REGRESSION_METRICS},
                "私榜减公榜": sub["score_delta"],
            })
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8-sig")
    build_team_leaderboard_summary(paths).to_csv(
        team_summary_path, index=False, encoding="utf-8-sig"
    )
    return {
        "json": json_path, "markdown": md_path, "csv": csv_path,
        "team_summary_csv": team_summary_path,
    }


def export_period_leaderboards(
    period_paths: dict[str, CheckPaths] | None = None,
    *,
    refresh_similarity: bool = False,
) -> dict[str, Any]:
    """导出两个原始周期和合并周期，并生成逐 submission 对照表。"""
    period_paths = period_paths or CONFIG.period_paths
    outputs: dict[str, dict[str, Path]] = {}
    period_frames: dict[str, pd.DataFrame] = {}
    comparisons: list[pd.DataFrame] = []
    team_comparisons: list[pd.DataFrame] = []
    for period, paths in period_paths.items():
        outputs[period] = export_team_private_leaderboard(paths)
        frame = pd.read_csv(outputs[period]["csv"], dtype={"submission_id": str})
        period_frames[period] = frame
        keep = [
            "submission_id", "团队名", "评测状态", "私榜排名（全部提交）",
            "私榜得分", "私榜A分", "私榜B分",
        ]
        available = [column for column in keep if column in frame]
        frame = frame[available].drop_duplicates("submission_id")
        frame = frame.rename(columns={
            column: f"{period}_{column}"
            for column in frame.columns if column != "submission_id"
        })
        comparisons.append(frame)

        team_frame = pd.read_csv(
            outputs[period]["team_summary_csv"],
            dtype={"参赛类型": str, "团队名": str},
        )
        team_keep = [
            "参赛类型", "团队名", "团队成员及学校", "团队私榜提交数量",
            "团队私榜得分", "团队私榜排名", "团队公榜得分", "团队公榜排名",
        ]
        team_frame = team_frame[[column for column in team_keep if column in team_frame]]
        team_frame = team_frame.rename(columns={
            column: f"{period}_{column}"
            for column in team_frame.columns
            if column not in {"参赛类型", "团队名"}
        })
        team_comparisons.append(team_frame)

    comparison = comparisons[0]
    for frame in comparisons[1:]:
        comparison = comparison.merge(frame, on="submission_id", how="outer")
    for period in ("period_1", "period_2"):
        source = f"{period}_私榜得分"
        merged = "merged_私榜得分"
        if source in comparison and merged in comparison:
            comparison[f"merged_vs_{period}_score_delta"] = comparison[merged] - comparison[source]
    comparison_path = (
        period_paths["merged"].artifacts_dir / PERIOD_COMPARISON_FILENAME
    )
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    team_comparison = team_comparisons[0]
    for frame in team_comparisons[1:]:
        team_comparison = team_comparison.merge(
            frame, on=["参赛类型", "团队名"], how="outer"
        )
    team_comparison = team_comparison.sort_values(
        ["merged_团队私榜排名", "参赛类型", "团队名"],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    team_comparison_path = (
        period_paths["merged"].artifacts_dir / TEAM_PERIOD_COMPARISON_FILENAME
    )
    team_comparison.to_csv(team_comparison_path, index=False, encoding="utf-8-sig")

    from .team_similarity import analyze_team_submission_similarity

    similarity_outputs = analyze_team_submission_similarity(
        period_paths["merged"], force=refresh_similarity
    )
    similarity_summary_by_team = similarity_outputs["summary"].set_index("team_name")
    similarity_pairs_by_team = {
        team_name: frame
        for team_name, frame in similarity_outputs["pairs"].groupby("team_name")
    }

    report_path = outputs["merged"]["markdown"]
    merged_summary = pd.read_csv(outputs["merged"]["team_summary_csv"])
    lines = ["# 私榜人工复核报告", "", "## 团队排名", ""]
    overview_columns = {
        "团队私榜排名": "私榜排名",
        "团队公榜排名": "公榜排名",
        "团队名": "团队名",
        "团队成员及学校": "团队成员（学校/学历）",
        "团队私榜提交数量": "私榜提交数量",
        "团队私榜得分": "私榜得分",
        "私榜计分submission": "私榜计分 submission",
        "团队公榜得分": "公榜得分",
    }
    overview = merged_summary[list(overview_columns)].rename(columns=overview_columns)
    headers = list(overview.columns)
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ])
    for row in overview.itertuples(index=False, name=None):
        values = [
            _fmt(value, 0) if index in {0, 1, 4} else _fmt(value)
            for index, value in enumerate(row)
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "", "## 各团队 submission 详解", "",
        "- 私榜时间段 1：2025-03-01 至 2025-11-30",
        "- 私榜时间段 2：2025-12-01 至 2026-08-10",
        "- 私榜 merge：2025-03-01 至 2026-08-10",
        "",
    ])
    period_lookup = {
        period: frame.drop_duplicates("submission_id").set_index("submission_id")
        for period, frame in period_frames.items()
    }
    source_submissions = pd.read_csv(
        period_paths["merged"].submission_source_path,
        dtype={"id": str, "selected_for_private": str},
    ).drop_duplicates("id").set_index("id")
    metric_columns = [
        ("总分", "得分"), ("A分", "A分"), ("B分", "B分"),
        ("IC均值", "ic_mean"), ("ICIR", "ic_ir"),
        ("夏普", "sharpe_ratio"), ("压力ICIR", "stress_ic_ir"),
        ("model_score", "B项model_score"),
        ("平均绝对权重", "B项abs_weight_mean"),
        ("权重标准差", "B项abs_weight_std"),
        ("入选率", "B项selection_rate"),
    ]
    merged_frame = period_frames["merged"].sort_values(
        ["私榜排名", "团队名", "私榜排名（全部提交）"], kind="stable"
    )
    for team_name, team_rows in merged_frame.groupby("团队名", sort=False, dropna=False):
        first = team_rows.iloc[0]
        lines.extend([
            f"**{_fmt(first.get('私榜排名'), 0)}. {team_name}**  ",
            f"成员：{first.get('成员名称及对应学校') or '—'}",
            "",
        ])
        team_similarity_summary = (
            similarity_summary_by_team.loc[team_name]
            if team_name in similarity_summary_by_team.index else None
        )
        if team_similarity_summary is not None:
            lines.append(
                f"主体内 submission 相关性（process_factor 截面 Pearson）："
                f"计算对数 {_fmt(team_similarity_summary.get('computed_pair_count'), 0)}/"
                f"{_fmt(team_similarity_summary.get('expected_pair_count'), 0)}；"
                f"平均绝对相关 {_fmt(team_similarity_summary.get('mean_pair_abs_correlation'))}；"
                f"最大绝对相关 {_fmt(team_similarity_summary.get('max_pair_abs_correlation'))}；"
                f"高相关对（≥0.8）{_fmt(team_similarity_summary.get('high_correlation_pair_count_0_8'), 0)}；"
                f"近似重复对（≥0.95）{_fmt(team_similarity_summary.get('near_duplicate_pair_count_0_95'), 0)}"
            )
            lines.append("")
        for _, merged_row in team_rows.iterrows():
            submission_id = str(merged_row["submission_id"])
            selected_value = (
                source_submissions.at[submission_id, "selected_for_private"]
                if submission_id in source_submissions.index else None
            )
            selected_for_private = (
                "是" if str(selected_value).strip().lower() in {"true", "1", "yes"}
                else "否" if pd.notna(selected_value) else "—"
            )
            lines.extend([
                f"**submission：`{submission_id}`**",
                "",
                "| 评测口径 | 状态 | 人为选为私榜 | "
                + " | ".join(label for label, _ in metric_columns) + " |",
                "|---|---|---|" + "|".join("---:" for _ in metric_columns) + "|",
            ])
            rows_to_show = [
                ("公榜", merged_row, "公榜"),
                ("私榜 merge", merged_row, "私榜"),
                ("私榜时间段 1", period_lookup["period_1"].loc[submission_id], "私榜"),
                ("私榜时间段 2", period_lookup["period_2"].loc[submission_id], "私榜"),
            ]
            for label, row, prefix in rows_to_show:
                values = []
                for _, suffix in metric_columns:
                    values.append(_fmt(row.get(f"{prefix}{suffix}")))
                lines.append(
                    f"| {label} | {_fmt(row.get('评测状态'))} | {selected_for_private} | "
                    + " | ".join(values) + " |"
                )
            lines.append("")
        team_pairs = similarity_pairs_by_team.get(team_name)
        if team_pairs is not None and not team_pairs.empty:
            lines.extend([
                "主体内 submission 两两相关性明细：", "",
                "| submission 1 | submission 2 | 平均相关 | 平均绝对相关 | P95 绝对相关 | 有效天数 |",
                "|---|---|---:|---:|---:|---:|",
            ])
            for pair in team_pairs.itertuples(index=False):
                lines.append(
                    f"| {pair.submission_id_1} | {pair.submission_id_2} | "
                    f"{_fmt(pair.mean_correlation)} | {_fmt(pair.mean_abs_correlation)} | "
                    f"{_fmt(pair.p95_abs_correlation)} | {_fmt(pair.valid_days, 0)} |"
                )
            lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "periods": outputs,
        "comparison_csv": comparison_path,
        "team_period_comparison_csv": team_comparison_path,
        "team_similarity_summary_csv": similarity_outputs["summary_csv"],
        "team_similarity_pairs_csv": similarity_outputs["pairs_csv"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="CSV 输出路径；JSON/Markdown 默认使用同名后缀")
    parser.add_argument("--json-output", type=Path, help="JSON 输出路径")
    parser.add_argument("--markdown-output", type=Path, help="Markdown 输出路径")
    parser.add_argument("--team-summary-output", type=Path, help="团队公私榜摘要 CSV 输出路径")
    args = parser.parse_args()
    outputs = export_team_private_leaderboard(
        output=args.output, json_output=args.json_output, markdown_output=args.markdown_output,
        team_summary_output=args.team_summary_output,
    )
    for kind, path in outputs.items():
        print(f"参赛主体私榜 {kind} 已生成：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
