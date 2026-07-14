"""按接口查询每个 user_id 的累计提交次数，筛出"提交里程碑"用户，输出可直接喂给
reward_coins.py 的候选 user_id 名单。

与「周榜前30%」不同：周榜每周固定评一次，提交里程碑是**每天**滚动赠送——用户哪天
攒够提交数就在哪天发，且**每个里程碑每人只发一次**。因此本脚本每天跑一遍即可：
    1. 走 AlphathonClient.list_submissions 拉每个赛道的全部提交，按 user_id 计数，
       落一份 submission_counts_<cid>_<date>.json 存档（每天覆盖当天那份）；
    2. 读 charge_records.csv，把已领过该里程碑赠送的用户剔掉，只输出「本次新达标」的人。

提交次数口径（已与运营确认）：
    - 数据源 = 该赛道全部 submission 记录（走接口，不依赖榜单快照）。
    - 计数 = 按提交人 user_id 各自累计，即「谁点的提交算谁的」。
      团队不共享次数：同队里没亲自提交的成员，其提交次数为 0、不达标。

里程碑（与运营奖励规则一致，见 docs/others/宽币赠送_*.md）：
    - 首次提交       submission_count >= 1
    - 累计第5次提交   submission_count >= 5
    - 累计第10次提交  submission_count >= 10

剔重（关键，保证每天重跑不重复发）：
    - 每个里程碑有一个稳定的 base_label（不带日期），既是 reward_coins 里该任务的 label
      （会写进充值 notes.remark），也是本脚本在 charge_records.csv 里匹配已发用户的关键字。
    - 剔重按「里程碑」计（跨赛道共用同一 base_label）；因为用户只属于一个赛道，
      按里程碑剔重与按赛道剔重结果一致，但更稳、不受赛道 key 命名影响。
    - 筛选/剔重条件与 reward_coins.load_history 对齐：type∈{reward,bigquant_charge}、
      status=paid、space_id==SPACE_ID、notes 含 base_label。

与 reward_coins.py 的对接（关键）：
    reward_coins.py 里一个 task 的 candidates_file 会作用于该 task amounts 的所有赛道，
    而提交里程碑是「按赛道分别计数」的，跨赛道共用一个名单会误发。因此本脚本按
    「里程碑 × 赛道」各出一个候选文件，并让每个赛道成为 reward_coins.py 里独立的一个
    task（单赛道 amounts + 对应 candidates_file）。

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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient
from common.paths import REWARD_COINS_DIR

# ===== 配置 =================================================================
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


def count_submissions(client: AlphathonClient, cid: str) -> dict[str, int]:
    """走接口拉该赛道全部提交，按提交人 user_id 各自累计计数。

    口径：谁点的提交算谁的，团队不共享次数（同队没亲自提交的成员计数为 0）。
    """
    counts: dict[str, int] = defaultdict(int)
    for sub in client.list_submissions(cid):
        uid = str(sub.get("user_id") or "").strip()
        if uid:
            counts[uid] += 1
    return dict(counts)


def save_counts(counts: dict[str, int], cid: str, date: str) -> Path:
    """把某赛道的 {user_id: 提交次数} 存档，便于回溯与人工核对。"""
    out = OUT_DIR / f"submission_counts_{cid}_{date}.json"
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    out.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def select_one(counts: dict[str, int], min_count: int) -> list[str]:
    """从提交次数表里筛出 submission_count >= min_count 的 user_id（保序按次数降序）。"""
    hit = [(uid, c) for uid, c in counts.items() if c >= min_count]
    hit.sort(key=lambda kv: (-kv[1], kv[0]))
    return [uid for uid, _ in hit]


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
    date = f"{datetime.now():%Y%m%d}"
    client = AlphathonClient()

    # 先按赛道走接口拉一次提交次数，供各里程碑复用，并存档一份便于核对。
    counts_by_cid: dict[str, dict[str, int]] = {}
    print("=== 各赛道提交次数（走接口 list_submissions，按提交人 user_id 计数）===")
    for cid, (key, name) in CIDS.items():
        counts = count_submissions(client, cid)
        counts_by_cid[cid] = counts
        out = save_counts(counts, cid, date)
        total_subs = sum(counts.values())
        print(f"  [{name}] 提交人 {len(counts)} 人，累计提交 {total_subs} 次 -> {out.name}")

    # 收集所有生成的 task 片段，最后统一打印，方便粘进 reward_coins.py。
    task_snippets: list[str] = []

    for ms in MILESTONES:
        min_count, ms_name, base_label = ms["threshold"], ms["name"], ms["base_label"]
        print(f"\n=== {ms_name}（submission_count>={min_count}）===")

        # 剔重按里程碑计（跨赛道共用同一 base_label），只算一次。
        rewarded = load_rewarded(base_label)

        for cid, amount in ms["amounts"].items():
            key, name = CIDS[cid]
            uids = select_one(counts_by_cid[cid], min_count)

            new_uids = [u for u in uids if u not in rewarded]
            skipped = len(uids) - len(new_uids)

            out = OUT_DIR / f"candidates_{ms['token']}_{key}_{date}.json"
            out.write_text(
                json.dumps(new_uids, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            label = f"{base_label}-{key}"
            print(
                f"  [{name}] 单价 {amount}："
                f"达标 {len(uids)} 人，"
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
