#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"
INPUT_DIR = ROOT / "files" / "submissions" / "analyze_outputs" / COMPETITION_ID
OUTPUT = ROOT / "files" / "submissions" / "end_to_end_submission_review_20260805.md"


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def text(value: Any, default: str = "—") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, list):
        return "、".join(text(v, "") for v in value if v not in (None, "")) or default
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if v not in (None, "", [], {}):
                parts.append(f"{k}={text(v, '')}")
        return "；".join(parts) or default
    rendered = str(value).replace("\n", " ").replace("|", "\\|")
    rendered = rendered.replace("股票��面��序", "股票截面时序").replace("（�� ", "（约 ")
    return rendered


def score(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.5f}"


def compact(value: Any, limit: int = 220) -> str:
    value = text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def current_state(value: Any, limit: int = 520) -> str:
    """保留事实判断，去掉原始 AI 文本中夹带的处置或整改意见。"""
    raw = text(value)
    clauses = []
    action_words = (
        "建议", "应判", "不宜直接", "需整改", "应要求", "暂停成绩",
        "进入私榜", "不进入私榜", "私榜候选", "要求解释", "要求团队",
    )
    for sentence in raw.replace("；", "。").split("。"):
        sentence = sentence.strip(" ，；。")
        if not sentence:
            continue
        positions = [sentence.find(word) for word in action_words if word in sentence]
        if positions:
            sentence = sentence[: min(positions)].rstrip(" ，；。")
            if not sentence:
                continue
        clauses.append(sentence)
    cleaned = "；".join(clauses) if clauses else "—"
    return compact(cleaned, limit)


def get(obj: dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        cur: Any = obj
        ok = True
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok and cur not in (None, "", [], {}):
            return cur
    return default


def bullets(lines: list[str], values: Any, max_items: int = 6) -> None:
    if values in (None, "", [], {}):
        return
    if not isinstance(values, list):
        values = [values]
    for item in values[:max_items]:
        lines.append(f"  - {compact(item, 320)}")
    if len(values) > max_items:
        lines.append(f"  - 其余 {len(values) - max_items} 项详见原始分析 JSON。")


def available(sub: dict[str, Any]) -> bool:
    availability = sub.get("availability")
    if isinstance(availability, dict):
        return availability.get("locally_available") is True
    return sub.get("status") not in {"unavailable", "not_selected_and_not_available_locally"}


def parameter_count(sub: dict[str, Any]) -> Any:
    return get(
        sub,
        "model_construction.parameter_count_recomputed_from_state_dict",
        "model_construction.parameter_count",
        "model_scale_and_feature_audit.parameter_scale.trainable_parameter_count",
    )


def input_count(sub: dict[str, Any]) -> Any:
    return get(
        sub,
        "entry_and_inference.input_fields.declared_count",
        "model_scale_and_feature_audit.input_feature_scale.declared_raw_field_count",
    )


def lookback(sub: dict[str, Any]) -> Any:
    return get(
        sub,
        "entry_and_inference.lookback",
        "model_scale_and_feature_audit.temporal_scale.lookback",
    )


def model_name(sub: dict[str, Any]) -> str:
    return text(get(sub, "model_construction.name", "model_construction.model_name"))


def compliance(sub: dict[str, Any]) -> str:
    return current_state(get(sub, "compliance_review.overall_verdict", "compliance_review.verdict"))


def cheating_risk(sub: dict[str, Any]) -> str:
    return text(get(sub, "cheating_review.risk"))


def innovation_level(sub: dict[str, Any]) -> str:
    return text(get(sub, "innovation_review.originality_level", "innovation_review.verdict"))


def repro(sub: dict[str, Any]) -> str:
    return text(get(
        sub,
        "training_review.reproducibility_verdict",
        "training_review.reproducibility",
        "reproducibility.verdict",
    ))


def false_checks(sub: dict[str, Any]) -> list[str]:
    checks = get(sub, "compliance_review.checks", default={})
    if not isinstance(checks, dict):
        return []
    return [k for k, v in checks.items() if v is False or (isinstance(v, str) and v.lower().startswith("fail"))]


def render_submission(lines: list[str], sub: dict[str, Any], member_name: str, index: int) -> None:
    sid = text(sub.get("submission_id"))
    lines.append(f"#### {index}. `{sid}`")
    lines.append("")
    lines.append(
        f"- **提交人/结果**：{member_name}；分数 {score(sub.get('score'))}；状态 {text(sub.get('status'))}；"
        f"完成时间 {text(sub.get('finished_at'))}；耗时 {text(sub.get('elapsed_ms'))} ms。"
    )
    metrics = sub.get("metrics") or {}
    if isinstance(metrics, dict) and any(v is not None for v in metrics.values()):
        lines.append(
            "- **分项指标**："
            + "；".join(f"{k}={text(v)}" for k, v in metrics.items() if v is not None)
            + "。"
        )

    lines.append(
        f"- **模型与规模**：{model_name(sub)}；可训练参数量 {text(parameter_count(sub))}；"
        f"输入形状 {text(get(sub, 'model_construction.input_shape'))}。"
    )
    logic = get(sub, "model_construction.logic", "model_construction.architecture", default=[])
    lines.append("- **模型逻辑**：")
    bullets(lines, logic)
    target = get(sub, "model_construction.target")
    if target:
        lines.append(f"- **训练目标**：{compact(target, 420)}")

    fields = get(sub, "entry_and_inference.input_fields.fields", "model_scale_and_feature_audit.input_feature_scale.field_names", default=[])
    lines.append(
        f"- **输入数据**：声明原始字段 {text(input_count(sub))} 个；回看窗口 {compact(lookback(sub), 180)}；"
        f"入口 {text(get(sub, 'entry_and_inference.entry'), 'judge_runner.py::judge_runner_main()')}。"
    )
    if fields:
        lines.append(f"  - 字段：{compact(fields, 520)}")
    flow = get(sub, "entry_and_inference.flow", default=[])
    if flow:
        lines.append("  - 推理流程：")
        bullets(lines, flow, 5)
    future = get(sub, "entry_and_inference.future_data_at_inference")
    if future is not None:
        lines.append(f"  - 推理期未来数据检查：{text(future)}。")

    tr = sub.get("training_review") or {}
    lines.append(
        "- **训练与复现**："
        f"训练区间 {text(get(sub, 'training_review.actual_training_range_from_metadata', 'training_review.declared_training_range', 'training_review.training_range'))}；"
        f"seed {text(tr.get('seed'))}；epoch {text(tr.get('epochs'))}；"
        f"优化器 {compact(tr.get('optimizer'), 180)}；复现判断 {compact(repro(sub), 260)}。"
    )
    training_entry = get(sub, "training_review.training_entry")
    if training_entry:
        lines.append(f"  - 训练入口：{compact(training_entry, 360)}")
    evidence = get(sub, "training_review.reproduction_evidence", "training_review.evidence", "training_review.reproduction_blockers", default=[])
    if evidence:
        lines.append("  - 复现证据/限制：")
        bullets(lines, evidence, 5)

    cr = sub.get("cheating_review") or {}
    lines.append(f"- **作弊检查**：风险 {cheating_risk(sub)}；{current_state(cr.get('verdict'), 520)}")
    if cr.get("evidence"):
        lines.append("  - 检查依据：")
        bullets(lines, cr.get("evidence"), 5)

    lines.append(f"- **合规现状**：{compact(compliance(sub), 520)}")
    failed = false_checks(sub)
    if failed:
        lines.append(f"  - 未通过或高风险检查项：{text(failed)}。")

    ir = sub.get("innovation_review") or {}
    lines.append(f"- **创新性**：{compact(ir.get('verdict') or innovation_level(sub), 520)}")
    analysis = ir.get("analysis")
    if analysis:
        lines.append("  - 分析：")
        bullets(lines, analysis, 4)

    limitations = sub.get("analysis_limitations") or []
    if limitations:
        lines.append("- **本提交审查局限**：")
        bullets(lines, limitations, 3)
    lines.append("")


def main() -> None:
    teams: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for team_dir in sorted(p for p in INPUT_DIR.iterdir() if p.is_dir()):
        team_file = team_dir / f"{team_dir.name}.json"
        if not team_file.exists():
            continue
        team = load(team_file)
        members = []
        for path in sorted(team_dir.glob("*.json")):
            if path == team_file:
                continue
            data = load(path)
            if data.get("member"):
                members.append(data)
        teams.append((team, members))
    teams.sort(key=lambda pair: pair[0].get("rank_in_reference", 999))

    all_subs = [s for _, ms in teams for m in ms for s in (m.get("submissions") or [])]
    local_subs = [s for s in all_subs if available(s)]
    unavailable_subs = [s for s in all_subs if not available(s)]

    lines: list[str] = []
    lines.extend([
        "# BigAlpha 2026「端到端模型」团队 Submission 质量详报",
        "",
        "> 汇报日期：2026-08-05  ",
        f"> 比赛 ID：`{COMPETITION_ID}`  ",
        f"> 审查范围：{len(teams)} 个团队、{sum(len(ms) for _, ms in teams)} 名成员、{len(all_subs)} 条 submission 记录  ",
        f"> 其中：{len(local_subs)} 条有本地代码并完成详细静态分析，{len(unavailable_subs)} 条仅有榜单元数据或本地代码不可用。  ",
        "> 本报告只描述当前材料反映的提交质量与风险，不包含处置意见。代码未实际运行。",
        "",
        "## 一、整体情况",
        "",
        "### 当前值得关注的关键信息",
        "",
        "- **已发现明确未来数据或评估期标签使用证据的成员有 2 人，涉及 3 条本地可审查 submission。**",
        "  - 号大特子牛队周迥：`2c94b00b-e33f-4c21-a44a-1ccdf392c36d`，分数 0.98649。权重元数据记录训练区间截至 2025-10-27，并出现 `include_public_2025_train_in_sample`、`fold_2025_include_public_train` 等信息，与 2025-03-01 至 2025-11-30 的评估期重叠。",
        "  - DP 队倪志鹏：`1d3dedbd-1768-46be-809f-97aa15b0641f`，分数 0.88125；`8edaacb5-3095-4e8d-9011-c39c86eb79a2`，分数 0.54899。两条提交均直接使用未来收益或评估标签代理构造分数，且模型参数量为 0。",
        "- **另有 15 条 submission 被标记为 high 风险，涉及 6 个团队、7 名成员。** 这些提交目前未发现直接评估标签泄漏，主要问题是明确禁止的衍生特征、额外 exposure 数据、评估日统计量、输出非线性变换或预训练来源无法审计。涉及陈名驹、何相毅、王梓桂、刘锦松、张郡杰、池洪伟、王佳豪。",
        "- **按各团队最高分提交统计，风险分布为：critical 2 队、high 3 队、medium 11 队、low 4 队。** 这里的 high/medium 多数表示规则合规或复现风险，不等同于已经确认标签作弊。",
        f"- **审查覆盖仍有限。** {len(all_subs)} 条记录中只有 {len(local_subs)} 条具备本地代码和详细分析，另有 {len(unavailable_subs)} 条缺少代码包；因此当前能够明确判断的主要是本地可用提交，不能将结论自动扩展到所有历史版本。",
        "- **高分与提交质量并不完全一致。** 当前榜首提交存在评估期标签重叠证据；部分 0.84—0.94 分的模型存在跨字段特征或评估期自适应统计；相对低风险团队的最高分主要分布在 0.73629—0.89355。",
        "",
        "### 提交质量的共性表现",
        "",
        "- 当前团队提交的主要质量差异体现在：模型是否真正端到端、输入和预处理是否符合白名单、训练链路是否完整、是否存在未来数据/评估期统计使用、以及模型创新是否有明确增量。",
        f"- 现有详细审查覆盖 {len(local_subs)} 条本地可用 submission。大量历史提交未被复制到本地，因此只能看到分数、状态或 submission ID，无法判断其模型结构、输入、参数量和作弊风险。",
        "- 多数团队的创新属于 Transformer、GRU/LSTM、SSM/Mamba、HIST、MoE、RevIN 等公开组件在分钟量价场景中的组合与工程适配，原创基础架构较少。",
        "- 常见质量问题包括跨字段价格比值或复权、滚动统计、评估日横截面标准化、模型外 rank/融合、完整训练入口缺失、依赖未锁定及权重来源链不完整。",
        "",
        "## 二、团队总览",
        "",
        "| 排名 | 团队 | 成员数 | 最高分 | 计分记录 | 本地详细提交 | 最高分模型 | 最高分提交风险 |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ])
    for team, members in teams:
        bi = team.get("basic_info") or {}
        top = team.get("highest_scoring_submission") or {}
        local_count = sum(1 for m in members for s in (m.get("submissions") or []) if available(s))
        lines.append(
            f"| {text(team.get('rank_in_reference'))} | {text(team.get('team_name'))} | {len(bi.get('members') or members)} | "
            f"{score(bi.get('highest_score'))} | {text(bi.get('scored_submission_count'), '0')} | {local_count} | "
            f"{compact(get(top, 'model_construction.name', 'model_construction.model_name'), 80)} | "
            f"{text(get(top, 'cheating_review.risk'))} |"
        )

    lines.extend(["", "## 三、各团队及 Submission 详细情况", ""])

    for team, members in teams:
        rank = team.get("rank_in_reference")
        name = text(team.get("team_name"))
        bi = team.get("basic_info") or {}
        lines.append(f"## {rank}. {name}")
        lines.append("")
        lines.append(
            f"- **团队成绩**：最高分 {score(bi.get('highest_score'))}；平均分 {score(bi.get('average_score'))}；"
            f"最低分 {score(bi.get('lowest_score'))}；计分记录 {text(bi.get('scored_submission_count'), '0')}。"
        )
        if bi.get("score_statistics_note"):
            lines.append(f"- **统计口径**：{compact(bi.get('score_statistics_note'), 520)}")
        lines.append(f"- **团队总体合规判断**：{current_state(get(team, 'highest_scoring_submission.rule_compliance.overall_verdict'), 620)}")
        lines.append(f"- **团队最高分作弊检查**：{current_state(get(team, 'highest_scoring_submission.cheating_review.verdict'), 620)}")
        lines.append(f"- **团队最高分创新性**：{compact(get(team, 'highest_scoring_submission.innovation.verdict'), 620)}")
        lines.append("")
        lines.append("### 成员与提交覆盖")
        lines.append("")
        lines.append("| 成员 | 学校 | 最高分 | 平均分 | 计分数 | 参考/分析记录 | 本地详细提交 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for m in members:
            mi = m.get("member") or {}
            ss = m.get("score_summary") or {}
            subs = m.get("submissions") or []
            local_count = sum(1 for s in subs if available(s))
            lines.append(
                f"| {text(mi.get('name'))} | {text(mi.get('school'))} | {score(ss.get('highest_score'))} | "
                f"{score(ss.get('average_score'))} | {text(ss.get('scored_submission_count'), '0')} | {len(subs)} | {local_count} |"
            )
        if not members:
            lines.append("| — | — | — | — | — | — | — |")
        lines.append("")

        local_entries: list[tuple[dict[str, Any], str]] = []
        unavailable_count = 0
        for m in members:
            member_name = text((m.get("member") or {}).get("name"))
            for sub in m.get("submissions") or []:
                if available(sub):
                    local_entries.append((sub, member_name))
                else:
                    unavailable_count += 1
        local_entries.sort(key=lambda x: (x[0].get("score") is None, -(x[0].get("score") or -999)))

        lines.append("### 本地可审查 Submission 一览")
        lines.append("")
        if local_entries:
            lines.append("| 提交人 | Submission | 分数 | 模型 | 参数量 | 输入字段/窗口 | 合规现状 | 作弊风险 | 创新性 |")
            lines.append("|---|---|---:|---|---:|---|---|---|---|")
            for sub, member_name in local_entries:
                lines.append(
                    f"| {member_name} | `{text(sub.get('submission_id'))[:8]}` | {score(sub.get('score'))} | "
                    f"{compact(model_name(sub), 55)} | {text(parameter_count(sub))} | "
                    f"{text(input_count(sub))} / {compact(lookback(sub), 65)} | {compact(compliance(sub), 90)} | "
                    f"{cheating_risk(sub)} | {compact(innovation_level(sub), 70)} |"
                )
        else:
            lines.append("本团队当前没有可进行代码级详细审查的 submission。")
        lines.append("")
        if unavailable_count:
            lines.append(f"> 另有 {unavailable_count} 条 submission 本地代码不可用，只能保留榜单元数据，无法判断模型逻辑、输入、参数量、作弊风险和创新性。")
            lines.append("")

        lines.append("### Submission 逐项分析")
        lines.append("")
        for idx, (sub, member_name) in enumerate(local_entries, 1):
            render_submission(lines, sub, member_name, idx)
        if not local_entries:
            lines.append("无可展开条目。")
            lines.append("")

    lines.extend([
        "## 四、整体审查局限",
        "",
        "- 本报告来自 AI 静态分析结果汇总，未实际运行训练或推理，不能确认运行时数据访问、资源消耗、输出覆盖和数值复现。",
        "- 89 条本地可用提交可以做代码级分析；其余 980 条记录缺少本地代码，不能据其分数推断模型质量或合规性。",
        "- 权重元数据、README 和代码注释属于重要证据线索，但训练来源仍依赖平台访问日志、训练日志和文件哈希进一步证明。",
        "- 创新性判断基于公开方法对照，只能描述与既有技术的关系，不能单独认定抄袭、授权关系或主观作弊。",
        "- 各团队 JSON 对 `-2` 是否纳入均分存在口径差异，因此平均分主要用于描述原始分析现状。",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
