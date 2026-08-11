"""运行全部人工复核并生成可归档的 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import read_final, read_regression, read_summary
from .config import CheckPaths
from .factors import analyze_factor_similarity
from .ranking import (
    analyze_a_metric_sensitivity,
    analyze_ab_weight_sensitivity,
    analyze_rank_conflicts,
    check_score_consistency,
)
from .regression import (
    analyze_b_score_robustness,
    analyze_regression_stability,
    check_regression_integrity,
)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(data: pd.DataFrame, *, limit: int | None = None) -> str:
    """不依赖 tabulate，将 DataFrame 转为 Markdown 表格。"""
    if data.empty:
        return "_无异常记录。_"
    frame = data.head(limit).copy() if limit is not None else data.copy()
    frame = frame.reset_index() if frame.index.name else frame.reset_index(drop=True)
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    if limit is not None and len(data) > limit:
        lines.append(f"\n_仅展示前 {limit} 条，共 {len(data)} 条；完整结果请通过 Python 返回值查看。_")
    return "\n".join(lines)


def _require_inputs(paths: CheckPaths) -> None:
    required = [
        paths.summary_path,
        paths.final_path,
        paths.regression_path,
        paths.factor_pool_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("生成报告所需文件不存在：\n" + "\n".join(missing))


def generate_markdown_report(
    paths: CheckPaths,
    output_path: str | Path | None = None,
    *,
    high_correlation: float = 0.95,
    max_similarity_samples: int = 50_000,
    top_n: int = 20,
    regression_rerun_dir: str | Path | None = None,
) -> Path:
    """运行全部非绘图检查，将结果写入 Markdown，并返回报告路径。"""
    _require_inputs(paths)
    output = Path(output_path) if output_path else paths.artifacts_dir / "manual_check_report.md"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = read_summary(paths)
    final = read_final(paths)
    regression = read_regression(paths)

    score_problems = check_score_consistency(paths, display=False)
    rank_conflicts = analyze_rank_conflicts(paths, display=False)
    ab_sensitivity = analyze_ab_weight_sensitivity(paths, display=False)
    a_metric_sensitivity = analyze_a_metric_sensitivity(paths, display=False)
    regression_integrity = check_regression_integrity(paths, display=False)
    regression_stability = analyze_regression_stability(paths, display=False)
    b_robustness = analyze_b_score_robustness(paths, display=False)
    similarity = analyze_factor_similarity(
        paths,
        high_correlation=high_correlation,
        max_samples=max_similarity_samples,
        display=False,
    )
    rerun_dir = (
        Path(regression_rerun_dir).expanduser().resolve()
        if regression_rerun_dir is not None
        else paths.artifacts_dir / "regression_rerun"
    )
    rerun_summary_path = rerun_dir / "regression_rerun_summary.json"
    rerun_comparison_path = rerun_dir / "regression_rerun_comparison.csv"
    rerun_summary = None
    rerun_comparison = pd.DataFrame()
    if rerun_summary_path.is_file():
        rerun_summary = json.loads(rerun_summary_path.read_text(encoding="utf-8"))
        if rerun_comparison_path.is_file():
            rerun_comparison = pd.read_csv(rerun_comparison_path, encoding="utf-8-sig")

    success_count = int(summary["status"].eq("success").sum())
    set_problems = regression_integrity["set_problems"]
    field_problems = regression_integrity["field_problems"]
    high_similarity = similarity.loc[similarity["high_similarity"]].copy()
    suspicious = regression_stability.loc[regression_stability["suspicious"]].copy()
    rerun_blocking = int(rerun_summary is not None and rerun_summary.get("status") == "BLOCK")
    rerun_missing = int(rerun_summary is None)
    blocking_count = len(score_problems) + len(set_problems) + len(field_problems) + rerun_blocking
    review_count = len(high_similarity) + len(suspicious) + rerun_missing
    status = "BLOCK" if blocking_count else ("REVIEW" if review_count else "PASS")

    top_final = final.loc[final["final_score"].ge(0)].head(top_n).copy()
    top_final.insert(0, "rank", range(1, len(top_final) + 1))
    sensitive_ab = ab_sensitivity[["best_rank", "worst_rank", "rank_span"]].head(top_n)
    sensitive_a = a_metric_sensitivity.head(top_n)
    robust_b = b_robustness[
        ["submission_id", "base_final_rank", "max_b_rank_change", "max_final_rank_change"]
    ].head(top_n)
    similarity_top = similarity[
        ["submission_id_1", "submission_id_2", "pearson", "overlap", "abs_correlation", "high_similarity"]
    ].head(top_n)

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = paths.run_dir.name
    lines = [
        "# 私榜人工复核自动报告",
        "",
        f"> 本报告由 `manual_checks.generate_markdown_report` 自动生成。生成时间：`{generated_at}`。",
        "",
        "## 1. 总体状态",
        "",
        f"**{status}**",
        "",
        "| 项目 | 结果 |",
        "| --- | ---: |",
        f"| Run ID | `{run_id}` |",
        f"| 总提交数 | {len(summary)} |",
        f"| 成功提交数 | {success_count} |",
        f"| 成功率 | {success_count / len(summary):.2%} |",
        f"| 回归因子数 | {len(regression)} |",
        f"| 成绩复算异常 | {len(score_problems)} |",
        f"| 回归集合异常 | {len(set_problems)} |",
        f"| 回归字段异常 | {len(field_problems)} |",
        f"| 高相似因子对（≥ {high_correlation:.2f}） | {len(high_similarity)} |",
        f"| 稳定性可疑因子 | {len(suspicious)} |",
        f"| 回归复跑检查 | {rerun_summary.get('status') if rerun_summary else 'NOT_PROVIDED'} |",
        "",
        "状态规则：成绩、集合或字段异常时为 `BLOCK`；不存在阻断项但有高相似因子或稳定性可疑因子时为 `REVIEW`；否则为 `PASS`。",
        "",
        "## 2. 成绩复算",
        "",
        f"正式成绩复算异常数：**{len(score_problems)}**。",
        "",
        _table(score_problems, limit=top_n),
        "",
        "## 3. 最终榜单",
        "",
        _table(top_final),
        "",
        "## 4. 排名冲突与敏感性",
        "",
        f"- 最大 A/B 排名差：{rank_conflicts['a_b_rank_gap'].max():.0f} 名。",
        f"- A/B 排名差不低于 20 名：{int(rank_conflicts['a_b_rank_gap'].ge(20).sum())} 个提交。",
        f"- 最大 A/B 权重扫描名次跨度：{ab_sensitivity['rank_span'].max():.0f} 名。",
        f"- 权重扫描名次跨度不低于 20 名：{int(ab_sensitivity['rank_span'].ge(20).sum())} 个提交。",
        f"- 逐项移除 A 指标后的最大名次变化：{a_metric_sensitivity['max_abs_rank_change'].max():.0f} 名。",
        "",
        "### A/B 权重最敏感提交",
        "",
        _table(sensitive_ab),
        "",
        "### A 指标留一法最敏感提交",
        "",
        _table(sensitive_a),
        "",
        "## 5. 回归完整性与稳定性",
        "",
        "### 集合异常",
        "",
        _table(set_problems, limit=top_n),
        "",
        "### 字段异常",
        "",
        _table(field_problems, limit=top_n),
        "",
        f"当前规则标记稳定性可疑因子 **{len(suspicious)}** 个。",
        "",
        _table(suspicious, limit=top_n),
        "",
        "## 6. 云端回归复跑一致性",
        "",
    ]

    if rerun_summary is None:
        lines.extend([
            "**NOT_PROVIDED**：本地未找到云端回归复跑结果包，本项尚未验证。",
            "",
            f"预期目录：`{rerun_dir}`",
            "",
            "至少需要下载以下文件：",
            "",
            "- `regression_rerun_summary.json`",
            "- `regression_rerun_comparison.csv`",
            "",
        ])
    else:
        delta_rows = pd.DataFrame(
            [
                {"metric": metric, "max_delta": value}
                for metric, value in rerun_summary.get("max_delta_by_metric", {}).items()
            ]
        )
        mismatches = (
            rerun_comparison.loc[rerun_comparison["rerun_mismatch"].astype(str).str.lower().eq("true")]
            if "rerun_mismatch" in rerun_comparison
            else pd.DataFrame()
        )
        lines.extend([
            f"**{rerun_summary.get('status', 'UNKNOWN')}**",
            "",
            f"- 比较容差：`{rerun_summary.get('tolerance')}`",
            f"- 正式因子数：{rerun_summary.get('official_factor_count')}",
            f"- 复跑因子数：{rerun_summary.get('rerun_factor_count')}",
            f"- 滚动窗口数：{rerun_summary.get('rolling_window_count')}",
            f"- 差异因子数：{rerun_summary.get('mismatch_count')}",
            f"- 全字段最大绝对差异：{rerun_summary.get('max_delta')}",
            "",
            "### 各字段最大差异",
            "",
            _table(delta_rows),
            "",
            "### 差异因子",
            "",
            _table(mismatches, limit=top_n),
            "",
        ])

    lines.extend([
        "## 7. B 分稳健性",
        "",
        f"- 最大 B 排名变化：{b_robustness['max_b_rank_change'].max():.0f} 名。",
        f"- 最大最终排名变化：{b_robustness['max_final_rank_change'].max():.0f} 名。",
        f"- 最终名次变化不低于 10 名：{int(b_robustness['max_final_rank_change'].ge(10).sum())} 个提交。",
        "",
        _table(robust_b),
        "",
        "## 8. 因子相似度",
        "",
        f"从最多 {max_similarity_samples:,} 个样本计算两两 Pearson 相关。高相似阈值为 `{high_correlation:.2f}`，命中 **{len(high_similarity)}** 对。",
        "",
        _table(similarity_top),
        "",
        "## 9. 自动建议",
        "",
    ])

    recommendations = []
    if blocking_count:
        recommendations.append("停止放榜，先处理成绩复算、回归集合或字段异常。")
    if rerun_summary is None:
        recommendations.append("从云端下载回归复跑结果包后重新生成报告；当前回归可复现性尚未验证。")
    elif rerun_summary.get("status") == "BLOCK":
        recommendations.append("云端回归复跑与正式结果不一致，应检查环境、数据版本和回归确定性。")
    if len(high_similarity):
        recommendations.append(f"对 {len(high_similarity)} 组高相似因子执行全量相关、参赛方关系和代码人工核验。")
    if int(ab_sensitivity["rank_span"].ge(20).sum()):
        recommendations.append("确认 A/B 权重已经在赛事规则中固定，并专项查看获奖线附近的局部权重敏感性。")
    if int(b_robustness["max_final_rank_change"].ge(10).sum()):
        recommendations.append("复核 B 分只使用 model_score 百分位的口径是否符合赛事设计预期。")
    if not recommendations:
        recommendations.append("未发现需处置项，可以进入人工签署阶段。")
    lines.extend(f"{index}. {item}" for index, item in enumerate(recommendations, start=1))
    lines.extend([
        "",
        "## 10. 输入路径",
        "",
        f"- 运行目录：`{paths.run_dir}`",
        f"- 预处理目录：`{paths.prepared_dir}`",
        f"- 私榜代码目录：`{paths.private_code_dir}`",
        "",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown 报告已生成：{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成私榜人工复核 Markdown 报告")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--private-code-dir", required=True, type=Path)
    parser.add_argument("--bigalpha-eval-src", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--high-correlation", type=float, default=0.95)
    parser.add_argument("--max-similarity-samples", type=int, default=50_000)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--regression-rerun-dir", type=Path)
    args = parser.parse_args()
    paths = CheckPaths(
        run_dir=args.run_dir,
        prepared_dir=args.prepared_dir,
        private_code_dir=args.private_code_dir,
        bigalpha_eval_src=args.bigalpha_eval_src,
    )
    generate_markdown_report(
        paths,
        args.output,
        high_correlation=args.high_correlation,
        max_similarity_samples=args.max_similarity_samples,
        top_n=args.top_n,
        regression_rerun_dir=args.regression_rerun_dir,
    )


if __name__ == "__main__":
    main()
