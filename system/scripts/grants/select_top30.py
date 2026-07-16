"""从爬取的排行榜快照筛选各赛道前 30%，输出可直接喂给 reward_coins.py 的候选 user_id 名单。

口径（与人工确认一致）：
    - 池 = 该赛道榜单里 submission_count>0 的条目（队伍/个人混合）。
    - 排序 = 按 rank 升序，取前 ceil(池 * TOP_RATIO) 条。
    - 展开 = team 行取 members[] 全体 user_id；individual 行取自身 user_id；去重保序。
    - 赛道三（开放创新）无 public_score，本次不参与，故不在 CIDS 内。

用法：
    python3 select_top30.py
    生成 reward_coins 目录下 candidates_top30_<key>_<date>.json，
    再在 reward_coins.py 的 TASKS 里用 candidates_file 引用它们。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.paths import LEADERBOARD_CRAWL_DIR, REWARD_COINS_DIR

# ===== 配置 =================================================================
SNAPSHOT_DATE = "20260715"     # 快照日期目录
SNAPSHOT_TIME = "1500"         # 以接近 15:00 的快照为准
TOP_RATIO = 0.30               # 取前 30%

# 参与本轮的赛道：cid -> (输出名 key, 该赛道单价, 中文名)
CIDS: dict[str, tuple[str, int, str]] = {
    "76ad3f56-ec2b-431a-890e-139a7f4bbcba": ("factor", 768, "赛道一·AI因子挖掘"),
    "523f9302-5b4b-42bd-bce1-f232e7c74316": ("e2e", 13920, "赛道二·端到端AI量化模型"),
}

CRAWL_DIR = LEADERBOARD_CRAWL_DIR / SNAPSHOT_DATE
OUT_DIR = REWARD_COINS_DIR
# ===========================================================================


def expand_uids(row: dict) -> list[str]:
    """把一条榜单记录展开成 user_id 列表：team 取全体成员，individual 取自身。"""
    members = row.get("members") or []
    if members:
        return [str(m.get("user_id")).strip() for m in members if m.get("user_id")]
    uid = row.get("user_id")
    return [str(uid).strip()] if uid else []


def select_one(cid: str) -> tuple[list[str], dict]:
    """筛选单个赛道，返回 (去重保序的 user_id 列表, 摘要 dict)。"""
    snap = CRAWL_DIR / f"leaderboard_{cid}_{SNAPSHOT_DATE}_{SNAPSHOT_TIME}.json"
    rows = json.loads(snap.read_text(encoding="utf-8"))

    pool = [r for r in rows if (r.get("submission_count") or 0) > 0]
    pool_sorted = sorted(
        pool, key=lambda r: (r.get("rank") is None, r.get("rank") or 10**9)
    )
    n_top = math.ceil(len(pool) * TOP_RATIO)
    top = pool_sorted[:n_top]

    uids: list[str] = []
    seen: set[str] = set()
    team_cnt = ind_cnt = 0
    for r in top:
        if r.get("type") == "team":
            team_cnt += 1
        else:
            ind_cnt += 1
        for u in expand_uids(r):
            if u not in seen:
                seen.add(u)
                uids.append(u)

    last = top[-1] if top else None
    last_name = (last.get("team_name") if last and last.get("type") == "team"
                 else last.get("user_id")) if last else None

    summary = {
        "snapshot": snap.name,
        "total_entries": len(rows),
        "pool_size": len(pool),
        "top_n_entries": n_top,
        "team_entries": team_cnt,
        "individual_entries": ind_cnt,
        "unique_users": len(uids),
        "cutoff_rank_score": last.get("public_score") if last else None,
        "cutoff_entry_name": last_name,
    }
    return uids, summary


def main() -> None:
    if not CRAWL_DIR.exists():
        raise SystemExit(f"未找到快照目录: {CRAWL_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cid, (key, amt, name) in CIDS.items():
        uids, s = select_one(cid)
        out = OUT_DIR / f"candidates_top30_{key}_{SNAPSHOT_DATE}.json"
        out.write_text(
            json.dumps(uids, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n=== {name} (单价 {amt}) ===")
        print(f"  快照       : {s['snapshot']}")
        print(f"  参赛总数   : {s['total_entries']} 条")
        print(f"  submission>0池: {s['pool_size']} 条")
        print(f"  前{int(TOP_RATIO*100)}% 取     : {s['top_n_entries']} 条"
              f"（team {s['team_entries']} / 个人 {s['individual_entries']}）")
        print(f"  末位队伍/用户: {s['cutoff_entry_name']}")
        print(f"  末位分数   : {s['cutoff_rank_score']}")
        print(f"  展开去重   : {s['unique_users']} 人")
        print(f"  预计发币   : {s['unique_users'] * amt}")
        print(f"  输出       : {out}")


if __name__ == "__main__":
    main()
