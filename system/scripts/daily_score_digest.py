"""每日得分摘要脚本。

查询某场比赛所有用户的 submission，按用户聚合，生成 Markdown 格式的站内信内容，
并将每个用户的 markdown 保存到 files/daily_reports/<user_id>.md。

分数数据来源：
  - leaderboard_final.csv     → a_score / b_score / final_score（以 id 对应 submission_id）
  - submissions_summary.csv   → ic_mean / ic_ir / sharpe_ratio / stress_ic_ir

用法:
    python daily_score_digest.py [competition_id]

    competition_id 默认: 76ad3f56-ec2b-431a-890e-139a7f4bbcba

环境变量:
    ALPHATHON_API_TOKEN      bigjwt token
    ALPHATHON_JWT_FILE       token 文件路径
    ALPHATHON_API_BASE_URL   API 地址

输出:
    每个用户的 markdown 写入 files/daily_reports/<user_id>.md
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from _client import AlphathonClient

DEFAULT_COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
DEFAULT_LEADERBOARD_BASE = (
    "/home/aiuser/work/workspace/BigAlpha/system/files"
    "/{competition_id}/leaderboard"
)
# 本地 fallback：脚本目录下的 files/leaderboard
LOCAL_LEADERBOARD_FALLBACK = Path(_scripts_dir) / "files" / "leaderboard"
DAILY_REPORTS_DIR = Path(_scripts_dir) / "files" / "daily_reports"


# ---- 格式工具 ----------------------------------------------------------------


def _safe(val: Any) -> Any:
    if isinstance(val, float) and not math.isfinite(val):
        return None
    return val


def _fmt(val: Any, fmt: str = ".4f", fallback: str = "N/A") -> str:
    v = _safe(val)
    if v is None:
        return fallback
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return fallback


def _status_label(sub: dict) -> str:
    status = sub.get("status") or ""
    mapping = {
        "pending":   "待评测",
        "running":   "评测中",
        "done":      "已完成",
        "failed":    "失败",
        "cancelled": "已取消",
    }
    return mapping.get(status, status)


# ---- CSV 加载 ----------------------------------------------------------------

# A项详细分的字段名 → 展示列名
_A_DETAIL_COLS: dict[str, str] = {
    "ic_mean":      "IC均值",
    "ic_ir":        "IC IR",
    "sharpe_ratio": "Sharpe",
    "stress_ic_ir": "压力IC IR",
}


def _load_score_tables(leaderboard_dir: str) -> tuple[
    dict[str, dict],   # sid → {a_score, b_score, final_score}
    dict[str, dict],   # sid → {ic_mean, ic_ir, sharpe_ratio, stress_ic_ir}
]:
    """从 leaderboard_dir 读取打分 CSV，返回两张以 submission_id 为键的查找表。"""

    def read_csv(name: str) -> pd.DataFrame | None:
        path = os.path.join(leaderboard_dir, name)
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"  [警告] 无法读取 {path}: {e}", file=sys.stderr)
            return None

    # leaderboard_final.csv — A/B/最终得分
    final_map: dict[str, dict] = {}
    final_df = read_csv("leaderboard_final.csv")
    if final_df is not None and "id" in final_df.columns:
        for _, row in final_df.iterrows():
            sid = str(row["id"])
            final_map[sid] = {
                "a_score":     _safe(row.get("a_score")),
                "b_score":     _safe(row.get("b_score")),
                "final_score": _safe(row.get("final_score")),
            }

    # submissions_summary.csv — A项详细分
    detail_map: dict[str, dict] = {}
    summary_df = read_csv("submissions_summary.csv")
    if summary_df is not None and "submission_id" in summary_df.columns:
        for _, row in summary_df.iterrows():
            sid = str(row["submission_id"])
            detail_map[sid] = {col: _safe(row.get(col)) for col in _A_DETAIL_COLS}

    return final_map, detail_map


# ---- 核心逻辑 ----------------------------------------------------------------


def group_submissions_by_user(
    submissions: list[dict],
) -> dict[str, list[dict]]:
    """按 user_id 对 submission 列表分组，每组内按 created_at 降序。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sub in submissions:
        uid = str(sub.get("user_id") or "")
        if uid:
            grouped[uid].append(sub)
    for uid in grouped:
        grouped[uid].sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return dict(grouped)


def build_user_markdown(
    user_id: str,
    submissions: list[dict],
    competition_id: str,
    now: datetime,
    final_map: dict[str, dict],
    detail_map: dict[str, dict],
) -> str:
    """为单个用户生成 Markdown 站内信正文。"""
    date_str = now.strftime("%Y-%m-%d")

    total = len(submissions)
    done_subs = [s for s in submissions if s.get("status") == "done"]

    lines: list[str] = [
        f"# {date_str} 每日得分日报",
        "",
        f"您好，以下是您在比赛 `{competition_id}` 中截至 **{date_str}** 的所有提交得分汇总。",
        "",
        "## 摘要",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 总提交数 | {total} |",
        f"| 成功运行 | {len(done_subs)} |",
        "",
        "## 提交明细",
        "",
    ]

    detail_headers = " | ".join(_A_DETAIL_COLS.values())
    lines.append(f"| 序号 | Submission ID | 状态 | A项得分 | B项得分 | 最终得分 | {detail_headers} | 提交时间 |")
    sep_detail = " | ".join(["---"] * len(_A_DETAIL_COLS))
    lines.append(f"|---|---|---|---|---|---|{sep_detail}|---|")

    for i, sub in enumerate(submissions, start=1):
        sid = str(sub.get("id") or "")
        status = _status_label(sub)
        created = (sub.get("created_at") or "")[:19].replace("T", " ")

        scores = final_map.get(sid, {})
        a = _fmt(scores.get("a_score"))
        b = _fmt(scores.get("b_score"))
        fs = _fmt(scores.get("final_score"))

        detail = detail_map.get(sid, {})
        detail_cells = " | ".join(_fmt(detail.get(col)) for col in _A_DETAIL_COLS)

        lines.append(f"| {i} | `{sid}` | {status} | {a} | {b} | {fs} | {detail_cells} | {created} |")

    lines += [
        "",
        "> A项得分基于单因子评测指标（IC均值、IC IR、Sharpe、压力IC IR等）加权合成；B项得分由回归系统决定；最终得分 = 0.3×A + 0.7×B。",
        "> 如有疑问请联系比赛管理员。",
        "",
    ]
    return "\n".join(lines)


# 站内信正文长度上限（字符数）。超出则精简模式会自适应减少明细行数。
NOTICE_MAX_CHARS = 1024


def build_user_markdown_concise(
    user_id: str,
    submissions: list[dict],
    competition_id: str,
    now: datetime,
    final_map: dict[str, dict],
    detail_map: dict[str, dict],
    max_chars: int = NOTICE_MAX_CHARS,
) -> str:
    """为单个用户生成精简版 Markdown，控制在 max_chars 字符以内，供站内信发送。

    相比完整版的压缩手段：
      - 去掉 4 个 IC 明细列（IC均值/IC IR/Sharpe/压力IC IR），引导用户到平台查看；
      - Submission ID 只显示前 8 位；
      - 提交时间精简为 MM-DD HH:MM；
      - 明细按最终得分降序，突出最优提交；
      - 自适应行数：若仍超限，整行删减（不破坏表格、不截断中文），
        并在脚注注明「仅显示分数最高的 N 条，共 M 条」。
    """
    date_str = now.strftime("%Y-%m-%d")
    total = len(submissions)
    done_subs = [s for s in submissions if s.get("status") == "done"]

    # 计算每条提交的最终得分，用于排序和摘要
    def final_score_of(sub: dict) -> float | None:
        sid = str(sub.get("id") or "")
        return _safe(final_map.get(sid, {}).get("final_score"))

    ranked = sorted(
        submissions,
        key=lambda s: (final_score_of(s) is not None, final_score_of(s) or 0.0),
        reverse=True,
    )
    best_score = next((final_score_of(s) for s in ranked if final_score_of(s) is not None), None)

    header = [
        f"# {date_str} 每日得分日报",
        "",
        f"您好，以下是您在比赛 `{competition_id}` 中截至 **{date_str}** 的提交得分汇总。",
        "",
        "## 摘要",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 总提交数 | {total} |",
        f"| 成功运行 | {len(done_subs)} |",
        f"| 最高最终得分 | {_fmt(best_score)} |",
        "",
        "## 提交明细（按最终得分排序）",
        "",
        "| 序号 | Submission | 状态 | A项 | B项 | 最终 | 提交时间 |",
        "|---|---|---|---|---|---|---|",
    ]

    def detail_row(idx: int, sub: dict) -> str:
        sid = str(sub.get("id") or "")
        status = _status_label(sub)
        created = (sub.get("created_at") or "")[:16].replace("T", " ")[5:]  # MM-DD HH:MM
        scores = final_map.get(sid, {})
        a = _fmt(scores.get("a_score"))
        b = _fmt(scores.get("b_score"))
        fs = _fmt(scores.get("final_score"))
        return f"| {idx} | `{sid[:8]}` | {status} | {a} | {b} | {fs} | {created} |"

    def footer(shown: int) -> list[str]:
        note = ""
        if shown < total:
            note = f"> 仅显示分数最高的 {shown} 条，共 {total} 条，完整明细请登录平台查看。\n"
        return [
            "",
            f"{note}> 最终得分 = 0.3×A + 0.7×B；IC 均值、Sharpe 等详细指标请登录平台查看。",
            "> 如有疑问请联系比赛管理员。",
            "",
        ]

    # 从全部明细开始，超限则每次砍掉分数最低的一行
    for shown in range(len(ranked), 0, -1):
        rows = [detail_row(i, sub) for i, sub in enumerate(ranked[:shown], start=1)]
        content = "\n".join(header + rows + footer(shown))
        if len(content) <= max_chars:
            return content

    # 一行明细都放不下（极端情况）：只保留摘要
    content = "\n".join(header[:-2] + footer(0))
    return content[:max_chars]


def save_user_reports(digest: dict[str, str], reports_dir: Path) -> None:
    """将每个用户的 markdown 保存到 reports_dir/<user_id>.md。"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for user_id, content in digest.items():
        out_path = reports_dir / f"{user_id}.md"
        out_path.write_text(content, encoding="utf-8")
    print(f"[{datetime.now():%H:%M:%S}] 报告已写入: {reports_dir}", file=sys.stderr)


def build_daily_digest(
    competition_id: str,
    leaderboard_dir: str,
    client: AlphathonClient,
    concise: bool = False,
) -> dict[str, str]:
    """拉取比赛所有 submission，返回 {user_id: markdown_content}。

    concise=True 时生成精简版（正文控制在 1024 字符内，供站内信发送）；
    默认 False 生成完整版报告。
    """
    now = datetime.now()

    submissions = client.list_submissions(
        competition_id,
        order_by=["-created_at"],
    )

    if not submissions:
        return {}

    final_map, detail_map = _load_score_tables(leaderboard_dir)
    grouped = group_submissions_by_user(submissions)

    builder = build_user_markdown_concise if concise else build_user_markdown
    return {
        uid: builder(uid, subs, competition_id, now, final_map, detail_map)
        for uid, subs in grouped.items()
    }


# ---- CLI 入口 ----------------------------------------------------------------


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="生成每日得分日报（每用户一份 Markdown）。")
    parser.add_argument(
        "competition_id",
        nargs="?",
        default=DEFAULT_COMPETITION_ID,
        help=f"比赛 ID（默认: {DEFAULT_COMPETITION_ID}）",
    )
    parser.add_argument(
        "--concise",
        action="store_true",
        help="生成精简版（正文压缩到 1024 字符内，供站内信发送）；不加则生成完整报告。",
    )
    args = parser.parse_args(argv)

    competition_id = args.competition_id
    leaderboard_dir = DEFAULT_LEADERBOARD_BASE.format(competition_id=competition_id)
    # 远端路径不存在时，fallback 到本地 files/leaderboard
    if not os.path.isdir(leaderboard_dir):
        print(
            f"  [提示] 远端目录不存在，使用本地目录: {LOCAL_LEADERBOARD_FALLBACK}",
            file=sys.stderr,
        )
        leaderboard_dir = str(LOCAL_LEADERBOARD_FALLBACK)

    client = AlphathonClient()

    mode = "精简版(≤1024字符)" if args.concise else "完整版"
    print(f"[{datetime.now():%H:%M:%S}] 拉取比赛 {competition_id} 的所有提交（{mode}）...", file=sys.stderr)
    digest = build_daily_digest(competition_id, leaderboard_dir, client, concise=args.concise)
    print(f"[{datetime.now():%H:%M:%S}] 共 {len(digest)} 位用户", file=sys.stderr)

    if args.concise:
        over = [uid for uid, c in digest.items() if len(c) > NOTICE_MAX_CHARS]
        if over:
            print(f"  [警告] 仍有 {len(over)} 份超过 {NOTICE_MAX_CHARS} 字符", file=sys.stderr)
        else:
            print(f"  [OK] 全部 {len(digest)} 份均在 {NOTICE_MAX_CHARS} 字符内", file=sys.stderr)

    save_user_reports(digest, DAILY_REPORTS_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
