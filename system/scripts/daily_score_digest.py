"""每日得分摘要脚本。

查询某场比赛所有用户的 submission，按用户聚合，生成 Markdown 格式的站内信内容。

明细字段：A项得分、B项得分、最终得分，以及 A项详细分（ic_mean / ic_ir / sharpe_ratio / stress_ic_ir）。

返回值：dict[user_id, markdown_content]

分数数据来源：
  - leaderboard_final.csv     → a_score / b_score / final_score（以 id 对应 submission_id）
  - submissions_summary.csv   → ic_mean / ic_ir / sharpe_ratio / stress_ic_ir

用法:
    python daily_score_digest.py <比赛ID> [--leaderboard-dir <路径>]
                                  [--base-url <url>] [--token <token>]
                                  [--out <输出JSON路径>]

默认 leaderboard_dir:
    /home/aiuser/work/workspace/BigAlpha/system/competitions/<competition_id>/leaderboard/
"""

from __future__ import annotations

import argparse
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
    "/home/aiuser/work/workspace/BigAlpha/system/competitions"
    "/{competition_id}/leaderboard"
)


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

    # 最高最终得分（有则用 final_score，否则看 a_score）
    final_scores = [
        float(final_map[str(s["id"])]["final_score"])
        for s in done_subs
        if str(s.get("id")) in final_map
        and final_map[str(s["id"])]["final_score"] is not None
    ]
    best_final = _fmt(max(final_scores)) if final_scores else "N/A"

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
        f"| 已完成评测 | {len(done_subs)} |",
        f"| 最高最终得分 | {best_final} |",
        "",
        "## 提交明细",
        "",
    ]

    # 表头：基础列 + A项详细列
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
        "> A项得分基于单因子评测指标（IC均值、IC IR、Sharpe、压力IC IR等）加权合成；B项得分由专家评审给出；最终得分 = 0.3×A + 0.7×B。",
        "> 如有疑问请联系比赛管理员。",
        "",
    ]
    return "\n".join(lines)


def build_daily_digest(
    competition_id: str,
    leaderboard_dir: str,
    client: AlphathonClient,
) -> dict[str, str]:
    """拉取比赛所有 submission，返回 {user_id: markdown_content}。"""
    now = datetime.now()

    submissions = client.list_submissions(
        competition_id,
        order_by=["-created_at"],
    )

    if not submissions:
        return {}

    final_map, detail_map = _load_score_tables(leaderboard_dir)
    grouped = group_submissions_by_user(submissions)

    return {
        uid: build_user_markdown(uid, subs, competition_id, now, final_map, detail_map)
        for uid, subs in grouped.items()
    }


# ---- CLI 入口 ----------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="生成每日得分摘要，返回 {user_id: markdown} 字典"
    )
    parser.add_argument(
        "competition_id",
        nargs="?",
        default=DEFAULT_COMPETITION_ID,
        help="比赛 ID",
    )
    parser.add_argument(
        "--leaderboard-dir",
        default=None,
        help="榜单目录路径（默认 competitions/<id>/leaderboard）",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--out", default=None, help="输出 JSON 文件路径（默认打印到 stdout）")
    args = parser.parse_args(argv)

    leaderboard_dir = args.leaderboard_dir or DEFAULT_LEADERBOARD_BASE.format(
        competition_id=args.competition_id
    )

    client = AlphathonClient(base_url=args.base_url, token=args.token)

    print(f"[{datetime.now():%H:%M:%S}] 拉取比赛 {args.competition_id} 的所有提交...", file=sys.stderr)
    digest = build_daily_digest(args.competition_id, leaderboard_dir, client)
    print(f"[{datetime.now():%H:%M:%S}] 共 {len(digest)} 位用户", file=sys.stderr)

    output = json.dumps(digest, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"结果已写入: {args.out}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
