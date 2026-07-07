"""爬取比赛排行榜信息。

从 GET /leaderboard/{competition_id} 拉取整场比赛的排行榜条目（团队 + 个人），
落两份文件到 files/leaderboard_crawl/：

    leaderboard_<competition_id>.json   完整原始条目（含团队成员明细）
    leaderboard_<competition_id>.csv    扁平化后的榜单（每行一个团队/个人）

用法:
    python crawl_leaderboad.py [competition_id ...]

    不带参数时，默认爬取三条赛道（见 DEFAULT_COMPETITION_IDS）。

环境变量:
    ALPHATHON_API_TOKEN      bigjwt token
    ALPHATHON_JWT_FILE       token 文件路径
    ALPHATHON_API_BASE_URL   API 地址
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from _client import AlphathonClient

DEFAULT_COMPETITION_IDS = [
    "76ad3f56-ec2b-431a-890e-139a7f4bbcba",   # AI 因子挖掘
    "523f9302-5b4b-42bd-bce1-f232e7c74316",   # 端到端大模型
    "63dd885c-2488-4efd-9c61-9e3a536f172c",   # AI 开放创新
]

OUT_DIR = Path(_scripts_dir) / "files" / "leaderboard_crawl"


def flatten_leaderboard(items: list[dict[str, Any]]) -> pd.DataFrame:
    """把排行榜条目扁平化成一行一个团队/个人，团队成员数放到 member_count 列。"""
    rows: list[dict[str, Any]] = []
    for it in items:
        members = it.get("members") or []
        rows.append(
            {
                "rank":                 it.get("rank"),
                "type":                 it.get("type"),
                "team_id":              it.get("team_id"),
                "team_name":            it.get("team_name"),
                "user_id":              it.get("user_id"),
                "public_score":         it.get("public_score"),
                "private_score":        it.get("private_score"),
                "submission_count":     it.get("submission_count"),
                "last_submission_time": it.get("last_submission_time"),
                "member_count":         len(members) if it.get("type") == "team" else None,
            }
        )
    return pd.DataFrame(rows)


def crawl_one(client: AlphathonClient, competition_id: str, out_dir: Path) -> int:
    """爬取单场比赛的排行榜，落 JSON + CSV，返回条目数。"""
    items = client.list_leaderboard(competition_id)

    json_path = out_dir / f"leaderboard_{competition_id}.json"
    json_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    df = flatten_leaderboard(items)
    csv_path = out_dir / f"leaderboard_{competition_id}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(
        f"  {competition_id}: {len(items)} 条 -> {json_path.name} / {csv_path.name}",
        file=sys.stderr,
    )
    return len(items)


def main(argv: list[str]) -> int:
    competition_ids = argv or DEFAULT_COMPETITION_IDS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AlphathonClient()

    total = 0
    for cid in competition_ids:
        print(f"[{datetime.now():%H:%M:%S}] 爬取比赛 {cid} 排行榜...", file=sys.stderr)
        total += crawl_one(client, cid, OUT_DIR)

    print(f"[{datetime.now():%H:%M:%S}] 完成，共 {total} 条榜单条目 -> {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
