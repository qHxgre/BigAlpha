"""运行端到端模型赛道的人工复核并生成 Markdown 报告。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import read_final, read_score, read_summary
from .config import CONFIG, METRICS, PATHS, CheckPaths
from .ranking import (
    analyze_metric_rank_conflicts,
    analyze_metric_sensitivity,
    check_score_consistency,
    compare_public_private_ranking,
)
from .similarity import analyze_prediction_similarity
from .team_private_leaderboard import build_team_private_leaderboard


def _format(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(data: pd.DataFrame, limit: int | None = None) -> str:
    if data.empty:
        return "_无异常记录。_"
    frame = data.head(limit).copy() if limit else data.copy()
    frame = frame.reset_index() if frame.index.name else frame.reset_index(drop=True)
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _participants(paths: CheckPaths) -> pd.DataFrame:
    if not paths.metadata_path.is_file():
        return pd.DataFrame(columns=["submission_id", "participant", "schools"])
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    rows = []
    for participant in metadata.get("participants", []):
        if participant.get("type") == "team":
            name = participant.get("team_name") or "（未命名队伍）"
            members = participant.get("members") or []
        else:
            user = participant.get("user") or {}
            name, members = f"个人：{user.get('name') or '未命名'}", [user]
        schools = "<br>".join(dict.fromkeys(str(m.get("school") or "（未填写）") for m in members))
        for submission_id in participant.get("private_submission_ids") or []:
            rows.append({"submission_id": str(submission_id), "participant": name, "schools": schools})
    return pd.DataFrame(rows, columns=["submission_id", "participant", "schools"])


def _require_inputs(paths: CheckPaths) -> None:
    missing = [str(path) for path in (paths.summary_path, paths.score_path, paths.final_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("生成报告所需文件不存在：\n" + "\n".join(missing))


def generate_markdown_report(
    paths: CheckPaths = PATHS,
    *,
    output_path: str | Path | None = None,
    run_similarity: bool = True,
) -> Path:
    """执行全部适用检查并生成报告；明确不包含回归模型评价。"""
    _require_inputs(paths)
    output = Path(output_path).expanduser().resolve() if output_path else paths.report_path
    score_problems = check_score_consistency(paths, display=False)
    conflicts = analyze_metric_rank_conflicts(paths, display=False)
    sensitivity = analyze_metric_sensitivity(paths, display=False)
    public_private = compare_public_private_ranking(paths, display=False)
    similarity = analyze_prediction_similarity(paths, save=True, display=False) if run_similarity else pd.DataFrame()
    summary, score, final = read_summary(paths), read_score(paths), read_final(paths)
    participant = _participants(paths)
    participant_submission_detail = build_team_private_leaderboard(paths)

    ranking = conflicts.merge(participant, on="submission_id", how="left")
    public_private = public_private.merge(participant, on="submission_id", how="left")
    sensitive = sensitivity.loc[
        sensitivity["max_abs_rank_change"].ge(CONFIG.max_final_rank_change)
    ].reset_index()
    high_similarity = similarity.loc[
        similarity.get("abs_correlation", pd.Series(dtype=float)).ge(CONFIG.high_correlation)
    ] if not similarity.empty else similarity

    valid_scores = final.loc[final["score"].ge(0), "score"]
    lines = [
        "# 端到端模型私榜人工复核报告", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 比赛：`523f9302-5b4b-42bd-bce1-f232e7c74316`",
        f"- 批次：`{paths.run_dir.name}`",
        "- 评分口径：四项指标百分位排名等权，每项 25%；失败提交为 -2。",
        "- 本比赛没有回归模型评价，本报告不包含 A/B 分、回归复跑或回归解释性分析。", "",
        "## 1. 批次总览", "",
        f"- 总提交：{len(summary)}",
        f"- 成功：{int(summary['status'].eq('success').sum())}",
        f"- 失败：{int(summary['status'].ne('success').sum())}",
        f"- 有效最高分：{valid_scores.max():.6f}" if not valid_scores.empty else "- 有效最高分：无",
        f"- 有效最低分：{valid_scores.min():.6f}" if not valid_scores.empty else "- 有效最低分：无", "",
        "## 2. 正式成绩复算", "",
        f"复算不一致记录数：**{len(score_problems)}**。", "", _table(score_problems), "",
        "## 3. 公榜与私榜得分排名差异", "",
        "以私榜 submission id 为基准匹配公榜成绩；`score_delta = private_score - public_score`，"
        "`rank_delta = private_rank - public_rank`，因此排名差为正表示私榜名次下降。", "",
        "下表展示私榜执行清单中的全部 submission id，不过滤失败提交，也不限制展示条数。", "",
        f"匹配到公榜分数：**{int(public_private['public_score_found'].sum())}/{len(public_private)}**。", "",
        f"私榜执行成功：**{int(public_private['status'].eq('success').sum())}**；"
        f"私榜执行失败：**{int(public_private['status'].ne('success').sum())}**。", "",
        _table(public_private[["private_rank", "public_rank", "rank_delta", "submission_id",
                               "participant", "schools", "status", "failure_type", "error",
                               "private_score", "public_score", "score_delta",
                               "public_score_found"]]), "",
        "### 参赛方逐 submission 公榜/私榜指标明细", "",
        "本比赛没有 A/B 或回归 B 项评分；下表展示实际评分使用的四项指标及总分。", "",
        _table(participant_submission_detail), "",
        "## 4. 综合榜单与指标排名", "",
        _table(ranking[["final_rank", "submission_id", "participant", "schools", "score", *METRICS,
                        "metric_rank_spread", "max_metric_final_gap"]], CONFIG.report_top_n), "",
        "## 5. 去除单项指标的排名敏感性", "",
        f"名次最大变化达到 {CONFIG.max_final_rank_change} 的提交数：**{len(sensitive)}**。", "",
        _table(sensitive, CONFIG.report_top_n), "",
        "## 6. 预测结果相似度", "",
        f"绝对相关系数不低于 {CONFIG.high_correlation:.2f} 的提交对：**{len(high_similarity)}**。", "",
        _table(high_similarity, CONFIG.report_top_n), "",
        "## 7. 人工确认清单", "",
        "- [ ] 正式成绩复算无不一致记录。",
        "- [ ] 核查公榜与私榜得分、排名变化异常的提交。",
        "- [ ] 核查指标严重偏科、去除单项指标后名次大幅变化的头部提交。",
        "- [ ] 核查处理后或原始预测高度相似的提交对。",
        "- [ ] 确认无回归模型评价相关产物或发布字段。", "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"人工复核报告已生成: {output}")
    return output


if __name__ == "__main__":
    generate_markdown_report()
