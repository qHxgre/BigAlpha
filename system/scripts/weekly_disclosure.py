"""每周公示汇总脚本。

把两个需求合并成一份对外公示 Markdown：
    需求一（factor_portrait）：前 10 因子画像（BARRA 风格暴露 + 行业 IC）；
    需求三（index_strategy）  ：中证 1000 指数增强策略跟踪（相对超额指标 + 收益曲线）。

两个需求各自的图表 / CSV 与合并后的 Markdown 都落在 files/weekly_disclosure/ 下，
图片以相对文件名内嵌，便于整目录一起分发。

用法：
    python weekly_disclosure.py [competition_id]

    competition_id 默认：76ad3f56-ec2b-431a-890e-139a7f4bbcba

输出（files/weekly_disclosure/）：
    factor_style_exposure_<date>.csv / .png    需求一：风格暴露
    factor_industry_ic_<date>.csv / .png       需求一：行业 IC
    excess_curve_<date>.png                     需求三：收益曲线
    strategy_metrics_history.csv                需求三：历轮指标（逐轮追加）
    weekly_disclosure_<date>.md                 合并后的对外公示正文

注意：本脚本只能在云端评测环境运行（依赖 dai + M.bigtrader）。
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime

import factor_portrait
import index_strategy
from _disclosure_common import (
    DEFAULT_COMPETITION_ID,
    OUTPUT_DIR,
    resolve_leaderboard_dir,
)


def _run_section(name: str, fn) -> str:
    """执行单个需求的 run()，成功返回其 Markdown 片段；失败时打印堆栈并回退占位片段。

    一个需求失败不影响另一个——比如本地无 M.bigtrader 时需求三会抛错，此时仍产出需求一。
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 —— 汇总脚本需容忍单需求失败
        traceback.print_exc()
        print(f"  [错误] {name} 生成失败: {exc}", file=sys.stderr)
        return f"## {name}\n\n> 本需求生成失败：`{exc}`\n"


def build_markdown(competition_id: str, date_str: str, sections: list[str]) -> str:
    """把各需求的 Markdown 片段拼成完整的对外公示正文（统一标题 + 分节）。"""
    header = [
        f"# {date_str} BigAlpha 每周公示",
        "",
        f"比赛 `{competition_id}` 本周公示内容，含前 10 因子画像与中证 1000 指数增强策略跟踪。",
        "",
    ]
    return "\n".join(header) + "\n" + "\n".join(sections)


def main(argv: list[str]) -> int:
    competition_id = argv[0] if argv else DEFAULT_COMPETITION_ID
    leaderboard_dir = resolve_leaderboard_dir(competition_id)
    date_str = datetime.now().strftime("%Y%m%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] 榜单目录: {leaderboard_dir}", file=sys.stderr)

    sections = [
        _run_section(
            "一、前 10 因子画像",
            lambda: factor_portrait.run(competition_id, leaderboard_dir, OUTPUT_DIR, date_str),
        ),
        _run_section(
            "三、中证 1000 指数增强策略跟踪",
            lambda: index_strategy.run(competition_id, leaderboard_dir, OUTPUT_DIR, date_str),
        ),
    ]

    md_path = OUTPUT_DIR / f"weekly_disclosure_{date_str}.md"
    md_path.write_text(build_markdown(competition_id, date_str, sections), encoding="utf-8")
    print(f"[{datetime.now():%H:%M:%S}] 已写入合并公示: {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
