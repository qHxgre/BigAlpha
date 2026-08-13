"""生成包含团队和个人全部提交、得分细节和主体内相关性的私榜报告。

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
    TEAM_PRIVATE_LEADERBOARD_FILENAME,
    CheckPaths,
)


CSV_COLUMNS = [
    "团队名", "团队ID", "私榜排名", "团队私榜最高分", "团队公榜排名", "团队公榜得分",
    "提交数量", "团队内相关对数量", "团队内平均绝对相关性", "团队内最大绝对相关性",
    "高相关对数量", "最高相关提交对", "成员名称及对应学校", "submission_id", "提交时间",
    "提交者ID", "私榜排名（全部提交）", "是否团队最佳私榜提交", "评测状态", "评测错误",
    "私榜得分", "私榜A分", "私榜B分", *[f"私榜{metric}" for metric in METRICS],
    *[f"私榜B项{metric}" for metric in REGRESSION_METRICS],
    "公榜得分", "公榜A分", "公榜B分", *[f"公榜{metric}" for metric in METRICS],
    *[f"公榜B项{metric}" for metric in REGRESSION_METRICS],
    "私榜减公榜",
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
        f"（{member.get('school') or '未填写'}）"
        for member in members
    )


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
        else:
            user = participant.get("user") or {}
            participant_id = str(user.get("user_id") or "")
            participant_name = f"个人：{user.get('name') or participant_id or '未命名'}"
            members = [user]
            team_id = None
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
            "public_rank": _clean(participant.get("public_rank")),
            "public_score": _clean(pd.to_numeric(participant.get("public_score"), errors="coerce")),
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
) -> dict[str, Path]:
    """同时生成 JSON、Markdown、CSV，返回三个输出路径。"""
    csv_path = Path(output).expanduser().resolve() if output else paths.artifacts_dir / TEAM_PRIVATE_LEADERBOARD_FILENAME
    json_path = Path(json_output).expanduser().resolve() if json_output else csv_path.with_suffix(".json")
    md_path = Path(markdown_output).expanduser().resolve() if markdown_output else csv_path.with_suffix(".md")
    for path in (csv_path, json_path, md_path):
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
    return {"json": json_path, "markdown": md_path, "csv": csv_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="CSV 输出路径；JSON/Markdown 默认使用同名后缀")
    parser.add_argument("--json-output", type=Path, help="JSON 输出路径")
    parser.add_argument("--markdown-output", type=Path, help="Markdown 输出路径")
    args = parser.parse_args()
    outputs = export_team_private_leaderboard(
        output=args.output, json_output=args.json_output, markdown_output=args.markdown_output
    )
    for kind, path in outputs.items():
        print(f"参赛主体私榜 {kind} 已生成：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
