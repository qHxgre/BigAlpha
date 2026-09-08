"""爬取比赛排行榜信息。

直接请求公开接口 GET https://bigquant.com/bigapis/alphathon/v1/leaderboard/{competition_id}
拉取整场比赛的排行榜条目（团队 + 个人），落两份文件到 leaderboard_crawl 目录
（common.paths.LEADERBOARD_CRAWL_DIR：system/files/scripts/leaderboard_crawl）：

    leaderboard_<competition_id>.json   完整原始条目（含团队成员明细）
    leaderboard_<competition_id>.csv    扁平化后的榜单（每行一个团队/个人，
                                         个人条目的 team_name 列用参赛者姓名填充）

默认行为（不带参数直接运行）：
    早上启动脚本后一直挂着，在每天 14:50–15:30 之间每隔 5 分钟爬取一次，
    每次落一份带时间戳的独立文件；15:30 那次爬完后自动退出。

用法:
    python crawl_leaderboad.py                 # 定时模式：等到时间窗口，按间隔爬取
    python crawl_leaderboad.py --once          # 立即爬一次就退出（默认三条赛道）
    python crawl_leaderboad.py <competition_id> [<id> ...]         # 定时模式，指定赛道
    python crawl_leaderboad.py --once <competition_id> [<id> ...]  # 立即爬指定赛道

环境变量:
    ALPHATHON_LEADERBOARD_BASE_URL   接口根地址（默认 https://bigquant.com/bigapis/alphathon/v1）
    ALPHATHON_API_TOKEN              可选 bigjwt token，设了就带成 Cookie: bigjwt=<token>
    ALPHATHON_API_TIMEOUT            单次请求超时秒数（默认 30）
    ALPHATHON_CRAWL_START            爬取窗口起点，HH:MM（默认 14:50）
    ALPHATHON_CRAWL_END              爬取窗口终点，HH:MM（默认 15:30）
    ALPHATHON_CRAWL_INTERVAL         间隔秒数（默认 300）
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient
from common.paths import LEADERBOARD_CRAWL_DIR

DEFAULT_BASE_URL = "https://bigquant.com/bigapis/alphathon/v1"

DEFAULT_COMPETITION_IDS = [
    "76ad3f56-ec2b-431a-890e-139a7f4bbcba",   # AI 因子挖掘
    "523f9302-5b4b-42bd-bce1-f232e7c74316",   # 端到端大模型
    "63dd885c-2488-4efd-9c61-9e3a536f172c",   # AI 开放创新
]

OUT_DIR = LEADERBOARD_CRAWL_DIR

# 定时爬取窗口，可用环境变量覆盖
CRAWL_START = os.getenv("ALPHATHON_CRAWL_START", "10:00")
CRAWL_END = os.getenv("ALPHATHON_CRAWL_END", "15:30")
CRAWL_INTERVAL = int(os.getenv("ALPHATHON_CRAWL_INTERVAL", "300"))


def fetch_leaderboard(competition_id: str, page_size: int = 1000, max_pages: int = 100000) -> list[dict[str, Any]]:
    """翻页拉取整场比赛的排行榜条目。

    接口返回 {code, data: {items, total, page, size}}，逐页翻完直到 items 不足一页。
    """
    base_url = os.getenv("ALPHATHON_LEADERBOARD_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv("ALPHATHON_API_TIMEOUT", "30"))
    token = (os.getenv("ALPHATHON_API_TOKEN") or "").strip()

    headers = {"accept": "application/json"}
    if token:
        headers["Cookie"] = f"bigjwt={token}"

    url = f"{base_url}/leaderboard/{competition_id}"
    results: list[dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        resp = requests.get(url, params={"page": page, "size": page_size}, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data") or {}
        items = data.get("items") or []
        if not items:
            break
        results.extend(items)
        if len(items) < page_size:
            break
        page += 1
    return results


def build_name_map(competition_id: str) -> dict[str, str]:
    """拉某场比赛的参赛者名单，返回 {user_id: 姓名}，查不到姓名时不放进去。"""
    name_map: dict[str, str] = {}
    try:
        client = AlphathonClient()
        for u in client.list_users(competition_id):
            user_id = str(u.get("user_id"))
            name = (u.get("data") or {}).get("name")
            if name:
                name_map[user_id] = name
    except Exception as exc:  # 名单查不到不应该影响排行榜落地
        print(f"  警告：拉取参赛者名单失败（{competition_id}）：{exc}", file=sys.stderr)
    return name_map


def flatten_leaderboard(items: list[dict[str, Any]], name_map: dict[str, str] | None = None) -> pd.DataFrame:
    """把排行榜条目扁平化成一行一个团队/个人，团队成员数放到 member_count 列。

    个人条目（type == "individual"）本身没有 team_name，这里用参赛者姓名（name_map）填充，
    查不到姓名时回退成 user_id。
    """
    name_map = name_map or {}
    rows: list[dict[str, Any]] = []
    for it in items:
        members = it.get("members") or []
        is_individual = it.get("type") == "individual"
        user_id = it.get("user_id")
        team_name = it.get("team_name")
        if is_individual:
            team_name = name_map.get(str(user_id), str(user_id))
        rows.append(
            {
                "rank":                 it.get("rank"),
                "type":                 it.get("type"),
                "team_id":              it.get("team_id"),
                "team_name":            team_name,
                "user_id":              user_id,
                "public_score":         it.get("public_score"),
                "private_score":        it.get("private_score"),
                "submission_count":     it.get("submission_count"),
                "last_submission_time": it.get("last_submission_time"),
                "member_count":         len(members) if it.get("type") == "team" else None,
            }
        )
    return pd.DataFrame(rows)


def crawl_one(competition_id: str, out_dir: Path, stamp: str) -> int:
    """爬取单场比赛的排行榜，落带时间戳的 JSON + CSV，返回条目数。

    每次运行用同一个 stamp（形如 20260708_1450），落到独立文件，方便按时间点回溯。
    """
    items = fetch_leaderboard(competition_id)

    json_path = out_dir / f"leaderboard_{competition_id}_{stamp}.json"
    json_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    name_map = build_name_map(competition_id)
    df = flatten_leaderboard(items, name_map)
    csv_path = out_dir / f"leaderboard_{competition_id}_{stamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(
        f"  {competition_id}: {len(items)} 条 -> {json_path.name} / {csv_path.name}",
        file=sys.stderr,
    )
    return len(items)


def crawl_all(competition_ids: list[str]) -> int:
    """爬取所有指定赛道一轮，同一 stamp 落文件，返回总条目数。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"

    total = 0
    for cid in competition_ids:
        print(f"[{datetime.now():%H:%M:%S}] 爬取比赛 {cid} 排行榜...", file=sys.stderr)
        total += crawl_one(cid, OUT_DIR, stamp)
    return total


def _parse_hhmm(value: str, base: datetime) -> datetime:
    """把 HH:MM 解析成 base 当天的具体时刻。"""
    hour, minute = (int(x) for x in value.split(":"))
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def run_scheduled(competition_ids: list[str]) -> int:
    """定时模式：等到窗口起点，之后每隔 CRAWL_INTERVAL 秒爬一次，直到过了窗口终点。"""
    now = datetime.now()
    start = _parse_hhmm(CRAWL_START, now)
    end = _parse_hhmm(CRAWL_END, now)

    if now < start:
        wait = (start - now).total_seconds()
        print(
            f"[{now:%H:%M:%S}] 等待爬取窗口开始（{CRAWL_START}，约 {wait/60:.1f} 分钟后）...",
            file=sys.stderr,
        )
        time.sleep(wait)
    elif now > end:
        print(
            f"[{now:%H:%M:%S}] 已过今天的爬取窗口（{CRAWL_START}-{CRAWL_END}），不爬取。",
            file=sys.stderr,
        )
        return 0

    print(
        f"[{datetime.now():%H:%M:%S}] 进入爬取窗口 {CRAWL_START}-{CRAWL_END}，间隔 {CRAWL_INTERVAL}s。",
        file=sys.stderr,
    )

    rounds = 0
    while datetime.now() <= end:
        crawl_all(competition_ids)
        rounds += 1

        next_run = datetime.now() + timedelta(seconds=CRAWL_INTERVAL)
        if next_run > end:
            break
        time.sleep((next_run - datetime.now()).total_seconds())

    print(f"[{datetime.now():%H:%M:%S}] 窗口结束，共爬取 {rounds} 轮 -> {OUT_DIR}", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    argv = list(argv)
    run_once = "--once" in argv
    if run_once:
        argv.remove("--once")

    competition_ids = argv or DEFAULT_COMPETITION_IDS

    if run_once:
        total = crawl_all(competition_ids)
        print(f"[{datetime.now():%H:%M:%S}] 完成，共 {total} 条榜单条目 -> {OUT_DIR}", file=sys.stderr)
        return 0

    return run_scheduled(competition_ids)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
