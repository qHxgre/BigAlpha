"""从爬取的排行榜快照筛选"提交里程碑"用户，输出可直接喂给 reward_coins.py 的候选 user_id 名单。

与「周榜前30%」不同：周榜是每周固定评一次，提交里程碑是**每天**按最新快照滚动赠送——
用户哪天攒够提交数就在哪天发，且**每个里程碑每人只发一次**。因此本脚本：
    1. 默认自动取最新快照（最近日期目录里时间最晚的一份），无需每天手改日期；
    2. 读 charge_records.csv，把已经领过该里程碑赠送的用户剔掉，只输出「本次新达标」的人。

里程碑（与运营奖励规则一致，见 docs/others/宽币赠送_*.md）：
    - 首次提交       submission_count >= 1
    - 累计第5次提交   submission_count >= 5
    - 累计第10次提交  submission_count >= 10

口径（与 select_top30.py 保持一致）：
    - 数据源 = 该赛道榜单快照里的条目（队伍/个人混合），字段 submission_count。
    - 达标 = submission_count >= 该里程碑阈值。
    - 展开 = team 行取 members[] 全体 user_id；individual 行取自身 user_id；去重保序。
    - 里程碑按赛道各自计数（队伍成员共享队伍的 submission_count）。

剔重（关键，保证每天重跑不重复发）：
    - 每个里程碑有一个稳定的 base_label（不带日期），既是 reward_coins 里该任务的 label
      （会写进充值 notes.remark），也是本脚本在 charge_records.csv 里匹配已发用户的关键字。
    - 剔重按「里程碑」计（跨赛道共用同一 base_label）；因为用户只属于一个赛道，
      按里程碑剔重与按赛道剔重结果一致，但更稳、不受赛道 key 命名影响。
    - 筛选/剔重条件与 reward_coins.load_history 对齐：type∈{reward,bigquant_charge}、
      status=paid、space_id==SPACE_ID、notes 含 base_label。

与 reward_coins.py 的对接（关键）：
    reward_coins.py 里一个 task 的 candidates_file 会作用于该 task amounts 的所有赛道，
    而提交里程碑是「按赛道分别计数」的，跨赛道共用一个名单会误发。因此本脚本沿用
    select_top30.py 的做法：按「里程碑 × 赛道」各出一个候选文件，并让每个赛道成为
    reward_coins.py 里独立的一个 task（单赛道 amounts + 对应 candidates_file）。

    脚本结束会打印可直接粘贴进 reward_coins.py TASKS 的任务片段。

用法：
    python3 select_submission_milestones.py
    生成 reward_coins 目录下 candidates_<里程碑>_<key>_<date>.json（只含本次新达标的人），
    再把打印出的 TASKS 片段粘进 reward_coins.py。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.paths import LEADERBOARD_CRAWL_DIR, REWARD_COINS_DIR

# ===== 配置 =================================================================
# 快照选择：默认自动取最新（最近日期目录里时间最晚的一份），适配每天滚动赠送。
# 需要复现历史某次筛选时，把它们填成具体值即可（如 "20260708" / "1500"）。
SNAPSHOT_DATE: str | None = None   # None=自动取最近的日期目录
SNAPSHOT_TIME: str | None = None   # None=该日期目录里时间最晚的一份

# 主空间全零 UUID（与 reward_coins.py 对齐；发平台宽币用主空间）。
SPACE_ID = "00000000-0000-0000-0000-000000000000"

# 充值流水（用于剔除已发过的用户；与 reward_coins.py 同一份，导出方式见 generate_sql.ipynb）。
CHARGE_RECORDS_CSV = REWARD_COINS_DIR / "charge_records.csv"

# 赛道：cid -> (输出名 key, 中文名)
CID_FACTOR = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"   # 赛道一 · AI 因子挖掘
CID_E2E = "523f9302-5b4b-42bd-bce1-f232e7c74316"       # 赛道二 · 端到端 AI 量化模型
CID_OPEN = "63dd885c-2488-4efd-9c61-9e3a536f172c"      # 赛道三 · AI 开放创新

CIDS: dict[str, tuple[str, str]] = {
    CID_FACTOR: ("factor", "赛道一·AI因子挖掘"),
    CID_E2E: ("e2e", "赛道二·端到端AI量化模型"),
    CID_OPEN: ("open", "赛道三·AI开放创新"),
}

# 里程碑定义。base_label 是稳定的赠送标识（不带日期）：既作为 reward_coins 里该任务
# label 的前缀（会写进充值 notes.remark），也是本脚本在流水里剔重的匹配关键字。
# amounts 里不含的赛道 = 该里程碑该赛道不发（不出文件、不进 TASKS）。
MILESTONES: list[dict] = [
    {
        "token": "first_submit", "threshold": 1, "name": "首次提交", "task_key": "首次提交",
        "base_label": "BigAlpha2026首次提交宽币赠送",
        "amounts": {CID_FACTOR: 288, CID_E2E: 5000, CID_OPEN: 288},
    },
    {
        "token": "cum5", "threshold": 5, "name": "累计第5次提交", "task_key": "累计第5次提交",
        "base_label": "BigAlpha2026累计第5次提交宽币赠送",
        "amounts": {CID_FACTOR: 480, CID_OPEN: 288},
    },
    {
        "token": "cum10", "threshold": 10, "name": "累计第10次提交", "task_key": "累计第10次提交",
        "base_label": "BigAlpha2026累计第10次提交宽币赠送",
        "amounts": {CID_FACTOR: 480},
    },
]

OUT_DIR = REWARD_COINS_DIR
# ===========================================================================


def expand_uids(row: dict) -> list[str]:
    """把一条榜单记录展开成 user_id 列表：team 取全体成员，individual 取自身。"""
    members = row.get("members") or []
    if members:
        return [str(m.get("user_id")).strip() for m in members if m.get("user_id")]
    uid = row.get("user_id")
    return [str(uid).strip()] if uid else []


def latest_date_dir() -> Path:
    """最近的一个日期目录（目录名形如 20260708，按名字排序取最大）。"""
    dirs = sorted(p for p in LEADERBOARD_CRAWL_DIR.iterdir() if p.is_dir())
    if not dirs:
        raise SystemExit(f"{LEADERBOARD_CRAWL_DIR} 下没有任何日期目录，请先跑爬虫。")
    return dirs[-1]


def resolve_snapshot(cid: str) -> tuple[Path, str, str]:
    """定位某赛道要用的快照，返回 (文件路径, 日期, 时间)。

    SNAPSHOT_DATE/TIME 有值就用指定的；否则自动取最近日期目录里时间最晚的一份。
    """
    date_dir = LEADERBOARD_CRAWL_DIR / SNAPSHOT_DATE if SNAPSHOT_DATE else latest_date_dir()
    date = date_dir.name
    if not date_dir.exists():
        raise SystemExit(f"未找到快照目录: {date_dir}")

    if SNAPSHOT_TIME:
        snap = date_dir / f"leaderboard_{cid}_{date}_{SNAPSHOT_TIME}.json"
        if not snap.exists():
            raise SystemExit(f"未找到指定快照: {snap}")
        return snap, date, SNAPSHOT_TIME

    cands = sorted(date_dir.glob(f"leaderboard_{cid}_{date}_*.json"))
    if not cands:
        raise SystemExit(f"{date_dir} 下没有赛道 {cid} 的快照。")
    snap = cands[-1]  # 文件名内时间戳递增，最后一个即最新
    return snap, date, snap.stem.rsplit("_", 1)[-1]


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


def load_rewarded(remark: str) -> set[str]:
    """从 charge_records.csv 查已经领过该 remark 赠送的 user_id 集合（用于剔重）。

    过滤条件与 reward_coins.load_history 对齐：
    type∈{reward,bigquant_charge}、status 为空或 paid、space_id 匹配、notes 含 remark。
    """
    rewarded: set[str] = set()
    if not CHARGE_RECORDS_CSV.exists():
        print(
            f"⚠️  未找到充值流水: {CHARGE_RECORDS_CSV}\n"
            f"    无法剔除已发用户，候选名单可能包含已赠送过的人！\n"
            f"    请先用 generate_sql.ipynb 的 SQL 导出流水到该路径。",
            file=sys.stderr,
        )
        return rewarded

    with CHARGE_RECORDS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("type") not in ("bigquant_charge", "reward"):
                continue
            if row.get("status") not in (None, "", "paid"):
                continue
            if SPACE_ID and row.get("space_id") not in (None, "", SPACE_ID):
                continue
            if remark and remark in (row.get("notes") or ""):
                uid = (row.get("user_id") or "").strip()
                if uid:
                    rewarded.add(uid)
    return rewarded


def cid_const_name(cid: str) -> str:
    """cid 字符串对应 reward_coins.py 里的常量名，便于打印可读的 TASKS 片段。"""
    return {CID_FACTOR: "CID_FACTOR", CID_E2E: "CID_E2E", CID_OPEN: "CID_OPEN"}[cid]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 先按赛道定位并读一次快照，供各里程碑复用；顺便打印用的是哪一份。
    snapshots: dict[str, tuple[list[dict], str, str]] = {}
    print("=== 使用的快照 ===")
    for cid in CIDS:
        snap, date, time = resolve_snapshot(cid)
        snapshots[cid] = (json.loads(snap.read_text(encoding="utf-8")), date, time)
        print(f"  [{CIDS[cid][1]}] {snap.name}")

    # 收集所有生成的 task 片段，最后统一打印，方便粘进 reward_coins.py。
    task_snippets: list[str] = []

    for ms in MILESTONES:
        min_count, ms_name, base_label = ms["threshold"], ms["name"], ms["base_label"]
        print(f"\n=== {ms_name}（submission_count>={min_count}）===")

        # 剔重按里程碑计（跨赛道共用同一 base_label），只算一次。
        rewarded = load_rewarded(base_label)

        for cid, amount in ms["amounts"].items():
            key, name = CIDS[cid]
            rows, date, _time = snapshots[cid]
            uids, s = select_one(rows, min_count)

            new_uids = [u for u in uids if u not in rewarded]
            skipped = len(uids) - len(new_uids)

            out = OUT_DIR / f"candidates_{ms['token']}_{key}_{date}.json"
            out.write_text(
                json.dumps(new_uids, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            label = f"{base_label}-{key}"
            print(
                f"  [{name}] 单价 {amount}："
                f"命中 {s['hit_entries']} 条"
                f"（team {s['team_entries']} / 个人 {s['individual_entries']}），"
                f"展开去重 {s['unique_users']} 人，"
                f"已发过 {skipped} 人，本次新达标 {len(new_uids)} 人，"
                f"预计发币 {len(new_uids) * amount}"
                f" -> {out.name}"
            )

            if not new_uids:
                continue  # 本赛道没有新达标的人，就不必生成 task 片段了

            task_snippets.append(
                "    {\n"
                f'        "label": "{label}",\n'
                f'        "task_key": "{ms["task_key"]}",\n'
                f'        "amounts": {{{cid_const_name(cid)}: {amount}}},\n'
                f'        "candidates_file": str(_CAND_DIR / "{out.name}"),\n'
                "    },"
            )

    print("\n" + "=" * 74)
    if task_snippets:
        print("把下面的片段粘进 reward_coins.py 的 TASKS（每个赛道一个独立 task）：")
        print("=" * 74)
        print("\n".join(task_snippets))
    else:
        print("本次没有任何新达标的用户，无需在 reward_coins.py 里配置任务。")


if __name__ == "__main__":
    main()
