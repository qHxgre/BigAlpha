"""按「任务」分批生成宽币赠送的 txt 名单，台账按任务分类去重。

方案见 docs/others/宽币赠送_20260630.md。每个赛道有多个赠送任务
（初始礼包 / 首次提交 / 提交里程碑 / 周榜前30% / 进入决赛 / 社媒…），
本脚本一次只处理**一个任务**，产物和去重台账都按任务隔离，互不影响。

规则（对齐文档「多赛道参与说明：按较大值发放，不叠加」）：
    * 同一参赛者报名多个赛道，同一个任务只按其报名赛道里的**最大**金额发一次。
    * 台账按任务分类：{任务key: {user_id: 已发金额}}。同一任务下已发过的人，
      后续再跑不会重复；不同任务各记各的，互不干扰。

产物：
    OUTPUT_DIR/<任务key>/<日期>/<金额>.txt
        文件名 = 该批用户本任务应发的宽币数（如 5220.txt）
        文件内容 = 需要发这么多宽币的 **user_id**，一行一个。

候选用户来源：
    * 默认（CANDIDATE_USER_IDS 为空）：用该任务涉及赛道的全部报名者，适用于
      「报名即发」类任务（初始礼包）。
    * 其它任务（首次提交 / 里程碑 / 周榜前30% / 决赛 / 社媒）：把满足条件的
      user_id 填进 CANDIDATE_USER_IDS（或指向一个 JSON 数组文件），脚本会用
      这些人和报名记录做交集，按其报名赛道取最大金额。

报名数据来源：
    直接读 participants.py 预生成的按赛道文件
    files/participants/user_id_<competition_id>.json（每份是一个 user_id 数组），
    不再调 API。名单过期就重跑 participants.py 刷新。

用法（沿用 grant_sdk_data.py 的 DRY_RUN 习惯）：
    1. 先跑一次 participants.py 生成/刷新按赛道的报名 json。
    2. 选好 TASK_KEY（周榜/滚动类任务记得带上周次后缀，见 WEEK_SUFFIX）。
    3. 先 DRY_RUN=True 跑一遍，看各档人数分布。
    4. 确认无误后把 DRY_RUN 改成 False 再跑，才会写 txt 并更新台账。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# ===== 赛道 ID ==============================================================
CID_FACTOR = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"   # 赛道一 · AI 因子挖掘
CID_E2E = "523f9302-5b4b-42bd-bce1-f232e7c74316"       # 赛道二 · 端到端 AI 量化模型
CID_OPEN = "63dd885c-2488-4efd-9c61-9e3a536f172c"      # 赛道三 · AI 开放创新

# ===== 各任务 × 各赛道的宽币金额（照抄方案表格） =============================
# 某赛道不设该任务时，就不在字典里出现（如端到端不设提交里程碑）。
TASKS: dict[str, dict[str, int]] = {
    "初始礼包":        {CID_FACTOR: 5000, CID_E2E: 5000,  CID_OPEN: 5000},
    "首次提交":        {CID_FACTOR: 288,  CID_E2E: 5000,  CID_OPEN: 288},
    "累计第5次提交":   {CID_FACTOR: 480,                  CID_OPEN: 288},
    "累计第10次提交":  {CID_FACTOR: 480},
    "周榜前30%":       {CID_FACTOR: 768,  CID_E2E: 13920, CID_OPEN: 288},
    "端到端按周滚动":  {CID_E2E: 10000},
    "进入决赛":        {CID_FACTOR: 960,  CID_E2E: 17400, CID_OPEN: 288},
    "社媒发帖":        {CID_FACTOR: 288,  CID_E2E: 5000,  CID_OPEN: 288},
}

# ===== 配置：改这里就行 =====================================================
# 本次要处理哪个任务（必须是 TASKS 的 key 之一）。
TASK_KEY = "初始礼包"

# 周榜/滚动类任务每周都发，需要区分周次；填如 "2026W27"，会拼进台账键和输出目录，
# 从而「同一周不重复、不同周各发一次」。一次性任务留空 ""。
WEEK_SUFFIX = ""

# 候选用户：留空则拉全部报名者（适用于报名即发）。其它任务把合格的 user_id
# 填进列表，或用 CANDIDATE_FILE 指向一个 JSON 数组文件。
CANDIDATE_USER_IDS: list[str] = []
CANDIDATE_FILE: Path | None = None   # 例：Path(__file__).with_name("first_submit_users.json")

# 所有产物统一放在 files/grant_coins 下。
FILES_DIR = Path(__file__).parent / "files"

# 报名数据目录：读 participants.py 生成的 user_id_<competition_id>.json。
PARTICIPANTS_DIR = FILES_DIR / "participants"

# 台账 & 输出目录
GRANT_DIR = FILES_DIR / "grant_coins"
LEDGER_FILE = GRANT_DIR / "coin_grant_ledger.json"
OUTPUT_DIR = GRANT_DIR

DRY_RUN = False   # True 只预览；确认后改 False 才真正写 txt 并更新台账
# ===========================================================================


def ledger_key() -> str:
    """台账/输出用的任务键；带周次后缀的周任务用 '任务@周次'。"""
    return f"{TASK_KEY}@{WEEK_SUFFIX}" if WEEK_SUFFIX else TASK_KEY


def load_ledger() -> dict[str, dict[str, int]]:
    """读台账 {任务key: {user_id: 已发金额}}，不存在则返回空。"""
    try:
        raw = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        print(f"{LEDGER_FILE} 应为 JSON 对象 {{任务key: {{user_id: amount}}}}", file=sys.stderr)
        sys.exit(1)
    # 兼容：值必须是 dict，规整类型
    out: dict[str, dict[str, int]] = {}
    for task, users in raw.items():
        if not isinstance(users, dict):
            print(f"台账里 {task} 的值应为 {{user_id: amount}} 对象", file=sys.stderr)
            sys.exit(1)
        out[str(task)] = {str(k): int(v) for k, v in users.items()}
    return out


def save_ledger(ledger: dict[str, dict[str, int]]) -> None:
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_FILE.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_candidates() -> set[str] | None:
    """返回本任务的候选 user_id 集合；None 表示不限制（用全部报名者）。"""
    raw: list = list(CANDIDATE_USER_IDS)
    if CANDIDATE_FILE is not None:
        try:
            file_ids = json.loads(CANDIDATE_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"未找到候选用户文件: {CANDIDATE_FILE}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(file_ids, list):
            print(f"{CANDIDATE_FILE} 应为 JSON 数组", file=sys.stderr)
            sys.exit(1)
        raw += file_ids
    if not raw:
        return None
    return {str(u).strip() for u in raw if str(u).strip()}


def load_participants(cid: str) -> list[str]:
    """读某赛道的报名 user_id 列表 user_id_<cid>.json，返回 [user_id, ...]。"""
    path = PARTICIPANTS_DIR / f"user_id_{cid}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"未找到报名文件: {path}\n请先跑 participants.py 生成按赛道的报名 json。", file=sys.stderr)
        sys.exit(1)
    if not isinstance(raw, list):
        print(f"{path} 应为 JSON 数组", file=sys.stderr)
        sys.exit(1)
    return [str(u).strip() for u in raw if str(u).strip()]


def compute_user_gifts(gifts_by_cid: dict[str, int], candidates: set[str] | None) -> dict[str, int]:
    """按 user_id 汇总本任务应发金额（取其报名赛道里的最大值）。

    只读本任务涉及赛道的 user_id_<cid>.json，多赛道取较大值、不叠加。
    """
    gifts: dict[str, int] = {}
    for cid, amount in gifts_by_cid.items():
        for uid in load_participants(cid):
            if candidates is not None and uid not in candidates:
                continue
            if amount > gifts.get(uid, 0):  # 多赛道取较大值，不叠加
                gifts[uid] = amount
    return gifts


def main() -> None:
    if TASK_KEY not in TASKS:
        print(f"TASK_KEY={TASK_KEY!r} 不在 TASKS 中，可选：{list(TASKS)}", file=sys.stderr)
        sys.exit(1)

    gifts_by_cid = TASKS[TASK_KEY]
    key = ledger_key()
    print(f"=== 任务：{key}（涉及 {len(gifts_by_cid)} 个赛道）===")
    for cid, amt in gifts_by_cid.items():
        print(f"  {cid} -> {amt} 宽币")

    candidates = load_candidates()
    if candidates is None:
        print("\n候选用户：本任务涉及赛道的全部报名者（报名即发）")
    else:
        print(f"\n候选用户：指定 {len(candidates)} 人（与报名记录取交集）")

    ledger = load_ledger()
    already = ledger.get(key, {})
    user_gifts = compute_user_gifts(gifts_by_cid, candidates)
    print(f"\n本任务符合发放条件：{len(user_gifts)} 人")
    print(f"台账中本任务已发过：{len(already)} 人")

    # 只保留本任务台账里没有的用户 —— 保证同一任务不重复赠送
    new_gifts = {uid: amt for uid, amt in user_gifts.items() if uid not in already}
    print(f"本次待赠送（新用户）：{len(new_gifts)} 人\n")

    amount_dist = Counter(new_gifts.values())
    print("本批分档（金额 -> 人数）：")
    for amt in sorted(amount_dist, reverse=True):
        print(f"  {amt} 宽币 : {amount_dist[amt]} 人")

    if not new_gifts:
        print("\n没有需要新赠送的用户，结束。")
        return

    if DRY_RUN:
        print(
            "\n当前为预览模式(DRY_RUN=True)，未写任何文件、未更新台账。"
            "确认无误后把 DRY_RUN 改成 False 再跑。"
        )
        return

    # 真正写 txt：OUTPUT_DIR/<任务key>/<日期>/<金额>.txt，内容为 user_id
    by_amount: dict[int, list[str]] = defaultdict(list)
    for uid, amt in new_gifts.items():
        by_amount[amt].append(uid)

    safe_key = key.replace("/", "_").replace("%", "pct")
    batch_dir = OUTPUT_DIR / safe_key / date.today().isoformat()
    batch_dir.mkdir(parents=True, exist_ok=True)
    for amt in sorted(by_amount, reverse=True):
        # 去重排序后写出 user_id，一行一个
        uids = sorted(set(by_amount[amt]))
        out_file = batch_dir / f"{amt}.txt"
        out_file.write_text("\n".join(uids) + "\n", encoding="utf-8")
        print(f"  已写出 {out_file}（{len(uids)} 行，去重后）")

    # 更新台账：把本批用户记进本任务名下
    ledger.setdefault(key, {}).update(new_gifts)
    save_ledger(ledger)
    print(f"\n=== 完成：任务 {key} 本批 {len(new_gifts)} 人已写入 txt 并记入台账 ===")
    print(f"台账文件：{LEDGER_FILE}")
    print(f"txt 目录：{batch_dir}")


if __name__ == "__main__":
    main()
