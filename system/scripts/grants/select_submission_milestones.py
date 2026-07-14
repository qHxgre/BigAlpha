"""从爬取的排行榜快照筛选"提交里程碑"用户，输出可直接喂给 reward_coins.py 的候选 user_id 名单。

里程碑（与运营奖励规则一致，见 docs/others/宽币赠送_*.md）：
    - 首次提交       submission_count >= 1
    - 累计第5次提交   submission_count >= 5
    - 累计第10次提交  submission_count >= 10

口径（与 select_top30.py 保持一致）：
    - 数据源 = 该赛道榜单快照里的条目（队伍/个人混合），字段 submission_count。
    - 达标 = submission_count >= 该里程碑阈值。
    - 展开 = team 行取 members[] 全体 user_id；individual 行取自身 user_id；去重保序。
    - 里程碑按赛道各自计数（队伍成员共享队伍的 submission_count）。

与 reward_coins.py 的对接（关键）：
    reward_coins.py 里一个 task 的 candidates_file 会作用于该 task amounts 的所有赛道，
    而提交里程碑是「按赛道分别计数」的，跨赛道共用一个名单会误发。因此本脚本沿用
    select_top30.py 的做法：按「里程碑 × 赛道」各出一个候选文件，并让每个赛道成为
    reward_coins.py 里独立的一个 task（单赛道 amounts + 对应 candidates_file）。

    脚本结束会打印可直接粘贴进 reward_coins.py TASKS 的任务片段。

用法：
    python3 select_submission_milestones.py
    生成 reward_coins 目录下 candidates_<里程碑>_<key>_<date>.json，
    再把打印出的 TASKS 片段粘进 reward_coins.py。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.paths import LEADERBOARD_CRAWL_DIR, REWARD_COINS_DIR

# ===== 配置 =================================================================
SNAPSHOT_DATE = "20260708"     # 快照日期目录
SNAPSHOT_TIME = "1500"         # 以接近 15:00 的快照为准

# 赛道：cid -> (输出名 key, 中文名)
CID_FACTOR = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"   # 赛道一 · AI 因子挖掘
CID_E2E = "523f9302-5b4b-42bd-bce1-f232e7c74316"       # 赛道二 · 端到端 AI 量化模型
CID_OPEN = "63dd885c-2488-4efd-9c61-9e3a536f172c"      # 赛道三 · AI 开放创新

CIDS: dict[str, tuple[str, str]] = {
    CID_FACTOR: ("factor", "赛道一·AI因子挖掘"),
    CID_E2E: ("e2e", "赛道二·端到端AI量化模型"),
    CID_OPEN: ("open", "赛道三·AI开放创新"),
}

# 里程碑：token -> (阈值, 中文名, reward_coins task_key, {cid: 该赛道单价})。
# amounts 里不含的赛道 = 该里程碑该赛道不发（不出文件、不进 TASKS）。
MILESTONES: dict[str, tuple[int, str, str, dict[str, int]]] = {
    "first_submit": (1, "首次提交", "首次提交", {CID_FACTOR: 288, CID_E2E: 5000, CID_OPEN: 288}),
    "cum5": (5, "累计第5次提交", "累计第5次提交", {CID_FACTOR: 480, CID_OPEN: 288}),
    "cum10": (10, "累计第10次提交", "累计第10次提交", {CID_FACTOR: 480}),
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


def load_snapshot(cid: str) -> tuple[list[dict], str]:
    """读某赛道的榜单快照，返回 (原始条目列表, 快照文件名)。"""
    snap = CRAWL_DIR / f"leaderboard_{cid}_{SNAPSHOT_DATE}_{SNAPSHOT_TIME}.json"
    return json.loads(snap.read_text(encoding="utf-8")), snap.name


def select_one(rows: list[dict], min_count: int) -> tuple[list[str], dict]:
    """从榜单条目里筛出 submission_count >= min_count 的用户，返回 (去重保序 user_id 列表, 摘要)。"""
    hit = [r for r in rows if (r.get("submission_count") or 0) >= min_count]

    uids: list[str] = []
    seen: set[str] = set()
    team_cnt = ind_cnt = 0
    for r in hit:
        if r.get("type") == "team":
            team_cnt += 1
        else:
            ind_cnt += 1
        for u in expand_uids(r):
            if u not in seen:
                seen.add(u)
                uids.append(u)

    summary = {
        "hit_entries": len(hit),
        "team_entries": team_cnt,
        "individual_entries": ind_cnt,
        "unique_users": len(uids),
    }
    return uids, summary


def cid_const_name(cid: str) -> str:
    """cid 字符串对应 reward_coins.py 里的常量名，便于打印可读的 TASKS 片段。"""
    return {CID_FACTOR: "CID_FACTOR", CID_E2E: "CID_E2E", CID_OPEN: "CID_OPEN"}[cid]


def main() -> None:
    if not CRAWL_DIR.exists():
        raise SystemExit(f"未找到快照目录: {CRAWL_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有生成的 task 片段，最后统一打印，方便粘进 reward_coins.py。
    task_snippets: list[str] = []

    # 先按赛道读一次快照，供各里程碑复用。
    snapshots: dict[str, tuple[list[dict], str]] = {
        cid: load_snapshot(cid) for cid in CIDS
    }

    for token, (min_count, ms_name, task_key, amounts) in MILESTONES.items():
        print(f"\n=== {ms_name}（submission_count>={min_count}）===")
        for cid, amount in amounts.items():
            key, name = CIDS[cid]
            rows, snap_name = snapshots[cid]
            uids, s = select_one(rows, min_count)

            out = OUT_DIR / f"candidates_{token}_{key}_{SNAPSHOT_DATE}.json"
            out.write_text(
                json.dumps(uids, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(
                f"  [{name}] 单价 {amount}："
                f"命中 {s['hit_entries']} 条"
                f"（team {s['team_entries']} / 个人 {s['individual_entries']}），"
                f"展开去重 {s['unique_users']} 人，预计发币 {s['unique_users'] * amount}"
                f" -> {out.name}"
            )

            task_snippets.append(
                "    {\n"
                f'        "label": "BigAlpha2026{ms_name}宽币赠送-{key}-{SNAPSHOT_DATE}",\n'
                f'        "task_key": "{task_key}",\n'
                f'        "amounts": {{{cid_const_name(cid)}: {amount}}},\n'
                f'        "candidates_file": str(_CAND_DIR / "{out.name}"),\n'
                "    },"
            )

    print("\n" + "=" * 74)
    print("把下面的片段粘进 reward_coins.py 的 TASKS（每个赛道一个独立 task）：")
    print("=" * 74)
    print("\n".join(task_snippets))


if __name__ == "__main__":
    main()
