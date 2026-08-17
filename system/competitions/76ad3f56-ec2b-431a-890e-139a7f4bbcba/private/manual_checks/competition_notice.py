"""独立生成最终赛事排名通知，不修改团队私榜报告的生成流程。"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

from .config import CONFIG, CheckPaths
from .team_private_leaderboard import _fmt, build_team_private_report


def competition_notice_markdown(
    report: dict[str, Any], *, submission_id_length: int = 8,
    score_decimal_places: int = 4,
) -> str:
    """生成最终赛事通知表；私榜列统一使用完整 merge 周期结果。"""

    def cell(value: Any, decimal_places: int = score_decimal_places) -> str:
        return escape(_fmt(value, decimal_places), quote=True).replace("\n", "<br>")

    lines = [
        "# 赛事排名通知", "",
        "> 私榜共包含两个评测阶段。下表中的私榜排名、私榜最终得分及私榜细分项，"
        "均以两个阶段合并后重新评分的 merge 结果为准。",
        "", "<table>", "  <thead>", "    <tr>",
        '      <th colspan="2">团队排名</th>',
        '      <th colspan="2">团队信息</th>',
        '      <th colspan="2">团队最终得分</th>',
        '      <th>Submission</th>',
        '      <th colspan="2">Submission 最终得分</th>',
        '      <th colspan="10">私榜 merge 细分项</th>',
        '      <th colspan="10">公榜细分项</th>',
        "    </tr>", "    <tr>",
        "      <th>私榜</th><th>公榜</th>",
        "      <th>团队名</th><th>私榜提交数量</th>",
        "      <th>私榜 merge</th><th>公榜</th>",
        "      <th>ID</th>",
        "      <th>私榜 merge</th><th>公榜</th>",
        "      <th>A分</th><th>B分</th><th>IC均值</th><th>ICIR</th><th>夏普</th><th>压力ICIR</th>"
        "<th>model_score</th><th>平均绝对权重</th><th>权重标准差</th><th>入选率</th>",
        "      <th>A分</th><th>B分</th><th>IC均值</th><th>ICIR</th><th>夏普</th><th>压力ICIR</th>"
        "<th>model_score</th><th>平均绝对权重</th><th>权重标准差</th><th>入选率</th>",
        "    </tr>", "  </thead>", "  <tbody>",
    ]
    for participant in report["participants"]:
        submissions = participant["submissions"] or [None]
        for submission_index, submission in enumerate(submissions):
            private = submission["private_score_detail"] if submission else {}
            public = submission["public_score_detail"] if submission else {}
            private_b = submission["private_b_detail"] if submission else {}
            public_b = submission["public_b_detail"] if submission else {}
            submission_id = submission["submission_id"] if submission else None
            lines.append("    <tr>")
            if submission_index == 0:
                rowspan = len(submissions)
                for value, decimal_places in (
                    (participant["private_rank"], 0),
                    (participant["public_rank"], 0),
                    (participant["participant_name"], score_decimal_places),
                    (participant["submission_count"], 0),
                    (participant["best_private_score"], score_decimal_places),
                    (participant["public_score"], score_decimal_places),
                ):
                    lines.append(
                        f'      <td rowspan="{rowspan}">{cell(value, decimal_places)}</td>'
                    )
            values = [
                str(submission_id)[:submission_id_length] if submission_id else None,
                submission["private_score"] if submission else None,
                submission["public_score"] if submission else None,
                private.get("a_score"), private.get("b_score"),
                private.get("ic_mean"), private.get("ic_ir"),
                private.get("sharpe_ratio"), private.get("stress_ic_ir"),
                private_b.get("model_score"), private_b.get("abs_weight_mean"),
                private_b.get("abs_weight_std"), private_b.get("selection_rate"),
                public.get("a_score"), public.get("b_score"),
                public.get("ic_mean"), public.get("ic_ir"),
                public.get("sharpe_ratio"), public.get("stress_ic_ir"),
                public_b.get("model_score"), public_b.get("abs_weight_mean"),
                public_b.get("abs_weight_std"), public_b.get("selection_rate"),
            ]
            lines.extend(f"      <td>{cell(value)}</td>" for value in values)
            lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>", ""])
    return "\n".join(lines)


def export_competition_notice(
    paths: CheckPaths | None = None, *, output: str | Path | None = None,
) -> Path:
    """使用 merged 周期数据独立生成 competition_notice.md。"""
    paths = paths or CONFIG.period_paths["merged"]
    output_path = (
        Path(output).expanduser().resolve()
        if output else paths.artifacts_dir / "competition_notice.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        competition_notice_markdown(build_team_private_report(paths)), encoding="utf-8"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="通知 Markdown 输出路径")
    args = parser.parse_args()
    print(f"赛事排名通知已生成：{export_competition_notice(output=args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
