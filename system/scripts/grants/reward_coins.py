"""给一批用户赠送宽币（对接 /balance/reward），分两步走、中间留人工审核。

流程：
    第 1 步 STEP="generate"：
        遍历 TASKS，对每个任务生成独立的 charge_plan_<label>.csv。
        已按该 label 发过的用户 to_charge 自动置 0，人工核查后可手动改回。
    第 2 步 STEP="charge"：
        遍历 REWARD_DIR 下所有 charge_plan_*.csv，对每个文件里
        to_charge>0 且 charged_at 为空的行调接口发币，成功后回写 CSV。
        重跑不会重复充（CSV 既是计划也是结果台账）。

认证：优先环境变量 BIGQUANT_TOKEN / BIGQUANT_SERVER，其次 ~/.bigquant/auth.json。

用法：
    1. 先把最新的 charge_records.csv 放到 reward_coins 目录（common.paths.REWARD_COINS_DIR）。
    2. 在 TASKS 里配置好所有活动任务。
    3. STEP="generate" 跑一遍，生成各任务的 charge_plan_*.csv。
    4. 人工用表格软件审核各计划文件，history_same_label>0 的行已自动置 0。
    5. STEP="charge" + DRY_RUN=True 预览，确认无误后 DRY_RUN=False 真发。
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.auth import load_auth
from common.ids import read_id_list
from common.paths import PARTICIPANTS_DIR, REWARD_COINS_DIR

try:
    import requests
except ImportError:
    print("请先安装依赖: python3 -m pip install requests", file=sys.stderr)
    sys.exit(1)

# ===== 赛道 ID & 各任务×各赛道金额（原 grant_coins.py 内联，脱离依赖）=========
CID_FACTOR = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"   # 赛道一 · AI 因子挖掘
CID_E2E = "523f9302-5b4b-42bd-bce1-f232e7c74316"       # 赛道二 · 端到端 AI 量化模型
CID_OPEN = "63dd885c-2488-4efd-9c61-9e3a536f172c"      # 赛道三 · AI 开放创新

# 任务定义：label 用于防重和统计，amounts 是各赛道的发放金额。
# 不需要的任务直接注释掉即可。
_CAND_DIR = REWARD_COINS_DIR
TASKS: list[dict] = [
    # {
    #     # 周榜前30%（20260708 快照）· 因子赛道。
    #     # 候选来自 select_top30.py：submission>0 池按 rank 取前30%，team 全体展开。
    #     "label": "BigAlpha2026周榜前30%宽币赠送-因子-20260708",
    #     "task_key": "周榜前30%",
    #     "amounts": {CID_FACTOR: 768},
    #     "candidates_file": str(_CAND_DIR / "candidates_top30_factor_20260708.json"),
    # },
    # {
    #     # 周榜前30%（20260708 快照）· 端到端赛道。金额较高，charge 前务必人工核对。
    #     "label": "BigAlpha2026周榜前30%宽币赠送-端到端-20260708",
    #     "task_key": "周榜前30%",
    #     "amounts": {CID_E2E: 13920},
    #     "candidates_file": str(_CAND_DIR / "candidates_top30_e2e_20260708.json"),
    # },
    # ↓ 历史任务，需要时再放开 ↓
    {
        "label": "BigAlpha2026报名宽币赠送",
        "task_key": "初始礼包",
        "amounts": {CID_FACTOR: 5000, CID_E2E: 5000, CID_OPEN: 5000},
    },
    # ↓ 提交里程碑：每天跑 select_submission_milestones.py，把它打印的 TASKS 片段粘到这里 ↓
    #   —— 那个脚本会自动取最新快照、并按 charge_records.csv 剔除已发过的人，
    #      只输出「本次新达标」的候选文件（candidates_<里程碑>_<key>_<date>.json）。
    #   下面是各里程碑的模板（label 带 -<key> 后缀，与脚本产出对齐；candidates_file 每天更新）。
    # {
    #     "label": "BigAlpha2026首次提交宽币赠送-factor",
    #     "task_key": "首次提交",
    #     "amounts": {CID_FACTOR: 288},
    #     "candidates_file": str(_CAND_DIR / "candidates_first_submit_factor_20260708.json"),
    # },
    # {
    #     "label": "BigAlpha2026累计第5次提交宽币赠送-factor",
    #     "task_key": "累计第5次提交",
    #     "amounts": {CID_FACTOR: 480},
    #     "candidates_file": str(_CAND_DIR / "candidates_cum5_factor_20260708.json"),
    # },
    # {
    #     "label": "BigAlpha2026累计第10次提交宽币赠送-factor",
    #     "task_key": "累计第10次提交",
    #     "amounts": {CID_FACTOR: 480},
    #     "candidates_file": str(_CAND_DIR / "candidates_cum10_factor_20260708.json"),
    # },
    # {
    #     "label": "BigAlpha2026周榜前30%宽币赠送",
    #     "task_key": "周榜前30%",
    #     "amounts": {CID_FACTOR: 768, CID_E2E: 13920, CID_OPEN: 288},
    # },
    # {
    #     "label": "BigAlpha2026端到端按周滚动宽币赠送",
    #     "task_key": "端到端按周滚动",
    #     "amounts": {CID_E2E: 10000},
    # },
    # {
    #     "label": "BigAlpha2026进入决赛宽币赠送",
    #     "task_key": "进入决赛",
    #     "amounts": {CID_FACTOR: 960, CID_E2E: 17400, CID_OPEN: 288},
    # },
    # {
    #     "label": "BigAlpha2026社媒发帖宽币赠送",
    #     "task_key": "社媒发帖",
    #     "amounts": {CID_FACTOR: 288, CID_E2E: 5000, CID_OPEN: 288},
    # },
]
# ===== 配置：改这里就行 =====================================================
# 两步流程开关："generate"（生成待审核计划）或 "charge"（按审核后的计划发币）。
STEP = "charge"

# 主空间全零 UUID：发平台宽币用主空间；子空间发币换成对应 space_id。
SPACE_ID = "00000000-0000-0000-0000-000000000000"

REWARD_DIR = REWARD_COINS_DIR

# 真实充值流水 CSV（generate 步用来查历史；导出方式见 generate_sql.ipynb）。
CHARGE_RECORDS_CSV = REWARD_DIR / "charge_records.csv"

# CSV 列顺序（charge 步按这个头读写；人工只需关心 to_charge，删行也安全）。
# history_same_label：历史里 description 含 label 的笔数，>0 说明本次已发过，to_charge 自动置 0。
PLAN_COLUMNS = [
    "user_id", "to_charge",
    "history_same_label",
    "charged_at", "charge_ok",
]

# 每次调用之间的间隔（秒），0 表示不限速。
SLEEP_BETWEEN = 0.2

DRY_RUN = True             # True 只预览；确认后改 False 真正写入

# 测试模式：True 时只处理 TEST_USER_IDS，不读也不改任何 JSON 报名文件。
TEST_MODE = False
TEST_USER_IDS: list[str] = ["5dd35480-0f38-11ed-93bb-da75731aa77c"]
# ===========================================================================

# 本批次启动时间戳，写入每条 notes，便于后期按批次聚合统计。
_BATCH_TS = datetime.now(timezone.utc).astimezone().isoformat()


def load_participants(cid: str) -> list[str]:
    """读某赛道报名 user_id 列表 user_id_<cid>.json。"""
    if TEST_MODE:
        return list(TEST_USER_IDS)
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


def load_task_candidates(task: dict) -> set[str] | None:
    """task 的候选集合；None=不限制（用全部报名者）。"""
    if TEST_MODE:
        return set(TEST_USER_IDS)
    raw: list = list(task.get("candidates") or [])
    candidates_file = task.get("candidates_file")
    if candidates_file is not None:
        try:
            file_ids = json.loads(Path(candidates_file).read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"未找到候选用户文件: {candidates_file}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(file_ids, list):
            print(f"{candidates_file} 应为 JSON 数组", file=sys.stderr)
            sys.exit(1)
        raw += file_ids
    if not raw:
        return None
    return {str(u).strip() for u in raw if str(u).strip()}


def collect_targets(task: dict) -> dict[str, int]:
    """按 task 配置汇总本次要发的 {user_id: amount}。多赛道同一 user 取较大金额。"""
    targets: dict[str, int] = {}

    def put(uid: str, amt: int) -> None:
        if amt > targets.get(uid, 0):
            targets[uid] = amt

    amounts = task.get("amounts")
    if amounts:
        candidates = load_task_candidates(task)
        for cid, amount in amounts.items():
            for uid in load_participants(cid):
                if candidates is not None and uid not in candidates:
                    continue
                put(uid, amount)
        if not targets:
            print(f"任务 {task.get('task_key')!r} 没有符合条件的用户。", file=sys.stderr)
            sys.exit(1)
        return targets

    user_ids_file = task.get("user_ids_file")
    if user_ids_file is not None:
        for uid in read_id_list(Path(user_ids_file)):
            put(uid, int(task.get("amount", 0)))
        return targets

    for uid in (task.get("user_ids") or []):
        uid = str(uid).strip()
        if uid:
            put(uid, int(task.get("amount", 0)))
    if not targets:
        print(f"任务 {task.get('label')!r} 没有任何用户来源，请配置 amounts / user_ids_file / user_ids 之一。", file=sys.stderr)
        sys.exit(1)
    return targets


def load_history(uids: set[str], label: str) -> dict[str, int]:
    """从 charge_records.csv 查这些用户是否已有 label 匹配的充值记录。

    返回 {user_id: same_label_count}，count>0 说明已按本次 label 发过。
    只取 type=bigquant_charge、status=paid、space_id==SPACE_ID 的行，不限日期。
    """
    counts: dict[str, int] = {u: 0 for u in uids}
    if not CHARGE_RECORDS_CSV.exists():
        print(
            f"⚠️  未找到充值流水: {CHARGE_RECORDS_CSV}\n"
            f"    无法检查历史发放记录，生成的计划将缺少防重依据！\n"
            f"    请先用 generate_sql.ipynb 的 SQL 导出流水到该路径。",
            file=sys.stderr,
        )
        return counts

    with CHARGE_RECORDS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = (row.get("user_id") or "").strip()
            if uid not in counts:
                continue
            if row.get("type") not in ("bigquant_charge", "reward"):
                continue
            if row.get("status") not in (None, "", "paid"):
                continue
            if SPACE_ID and row.get("space_id") not in (None, "", SPACE_ID):
                continue
            if label and label in (row.get("notes") or ""):
                counts[uid] += 1
    return counts


def plan_csv_path(label: str) -> Path:
    """按 label 生成计划文件路径，去掉文件名不安全字符。"""
    safe = label.replace("/", "_").replace("\\", "_").replace(":", "_")
    return REWARD_DIR / f"charge_plan_{safe}.csv"


def step_generate() -> None:
    """第 1 步：遍历 TASKS，为每个任务生成独立的 charge_plan_<label>.csv。"""
    if not TASKS:
        print("TASKS 为空，请先配置任务列表。", file=sys.stderr)
        sys.exit(1)

    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        label = task.get("label", "").strip()
        if not label:
            print(f"跳过未设置 label 的任务: {task}", file=sys.stderr)
            continue

        targets = collect_targets(task)
        same_label_counts = load_history(set(targets), label)

        rows = []
        for uid in sorted(targets):
            amt = targets[uid]
            same_label = same_label_counts.get(uid, 0)
            rows.append({
                "user_id": uid,
                "to_charge": 0 if same_label > 0 else amt,
                "history_same_label": same_label,
                "charged_at": "",
                "charge_ok": "",
            })

        plan_csv = plan_csv_path(label)
        with plan_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS)
            w.writeheader()
            w.writerows(rows)

        already_sent = sum(1 for r in rows if r["history_same_label"] > 0)
        print(f"\n[task] {label}")
        print(f"  计划文件 : {plan_csv}")
        print(f"  候选人数 : {len(rows)}")
        print(f"  已发过   : {already_sent} 人（to_charge 已自动置 0）")
        print(f"  待发放   : {len(rows) - already_sent} 人")

    print("\n请用表格软件审核各 charge_plan_*.csv，如需强制补发把 to_charge 改回目标金额。")
    print("审核完成后，把 STEP 改成 \"charge\" 再跑（建议先 DRY_RUN=True 预览）。")


def reward_one(token: str, server: str, user_id: str, amount: int, task: dict) -> tuple[bool, str]:
    """给单个用户发币。返回 (是否成功, 说明信息)。"""
    url = f"{server}/bigapis/kbb2/v1/balance/reward"
    body = {
        "reward_space_id": SPACE_ID,
        "reward_user_id": user_id,
        "reward_kbb": str(amount),
        "notes": {"remark": task.get("label", "")},
    }
    try:
        resp = requests.post(
            url, headers={"Authorization": f"Bearer {token}"}, json=body, timeout=30
        )
    except requests.RequestException as e:
        return False, f"请求异常: {e}"
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        return False, f"HTTP {resp.status_code}: {detail}"
    return True, "OK"


def to_int(v) -> int:
    """CSV 里的 to_charge 容错转 int（'', '5000', '5000.0' 都能处理）。"""
    s = str(v or "").strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def load_plan_rows(plan_csv: Path) -> list[dict]:
    """读审核后的计划 CSV，返回行列表（保序，用于回写）。"""
    try:
        with plan_csv.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"未找到计划文件: {plan_csv}\n请先用 STEP=\"generate\" 生成并人工审核。", file=sys.stderr)
        sys.exit(1)


def save_plan_rows(plan_csv: Path, rows: list[dict]) -> None:
    """把（可能已回填 charged_at 的）行写回计划 CSV，保持列顺序。"""
    with plan_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def step_charge() -> None:
    """第 2 步：遍历所有 charge_plan_*.csv，按审核后的计划实际发币，结果回写 CSV。"""
    plan_files = sorted(REWARD_DIR.glob("charge_plan_*.csv"))
    if not plan_files:
        print(f"在 {REWARD_DIR} 下未找到任何 charge_plan_*.csv，请先跑 STEP=generate。", file=sys.stderr)
        sys.exit(1)

    # label → task 的映射，用于给 reward_one 传 notes 信息
    label_to_task: dict[str, dict] = {t["label"]: t for t in TASKS if t.get("label")}

    _, server = load_auth()
    print(f"=== STEP=charge · space_id={SPACE_ID} ===")
    print(f"接口: {server}/bigapis/kbb2/v1/balance/reward")
    if DRY_RUN:
        print("当前为预览模式(DRY_RUN=True)，未调任何接口、未回写 CSV。")

    total_ok = total_fail = 0
    for plan_csv in plan_files:
        # 从文件名还原 label（去掉前缀 "charge_plan_" 和后缀 ".csv"）
        label = plan_csv.stem[len("charge_plan_"):]
        task = label_to_task.get(label) or {"label": label, "task_key": ""}

        rows = load_plan_rows(plan_csv)
        pending = [
            r for r in rows
            if to_int(r.get("to_charge")) > 0 and not (r.get("charged_at") or "").strip()
        ]
        total_coins = sum(to_int(r["to_charge"]) for r in pending)

        print(f"\n[task] {label}")
        print(f"  计划文件 : {plan_csv}")
        print(f"  共 {len(rows)} 行，待发 {len(pending)} 人，合计 {total_coins} 宽币")

        if not pending:
            print("  → 无待发记录，跳过。")
            continue
        if DRY_RUN:
            continue

        token, server = load_auth()
        ok_count = fail_count = 0
        for i, r in enumerate(pending, 1):
            uid = (r.get("user_id") or "").strip()
            amt = to_int(r["to_charge"])
            ok, msg = reward_one(token, server, uid, amt, task)
            now = datetime.now(timezone.utc).astimezone().isoformat()
            r["charge_ok"] = str(ok)
            if ok:
                ok_count += 1
                r["charged_at"] = now
                print(f"  [{i}/{len(pending)}] ✓ {uid}  {amt} 宽币")
            else:
                fail_count += 1
                print(f"  [{i}/{len(pending)}] ✗ {uid}  {amt} 宽币  —— {msg}", file=sys.stderr)
            if ok and ok_count % 20 == 0:
                save_plan_rows(plan_csv, rows)
            if SLEEP_BETWEEN:
                time.sleep(SLEEP_BETWEEN)

        save_plan_rows(plan_csv, rows)
        print(f"  → 成功 {ok_count} 人，失败 {fail_count} 人。结果已回写。")
        total_ok += ok_count
        total_fail += fail_count

    if not DRY_RUN:
        print(f"\n=== 全部完成：成功 {total_ok} 人，失败 {total_fail} 人 ===")
        if total_fail:
            print("失败的行 charged_at 仍为空，修正后重跑本脚本(charge)会自动重试。")
    else:
        print("\n确认无误后把 DRY_RUN 改成 False 再跑。")


def main() -> None:
    if STEP == "generate":
        step_generate()
    elif STEP == "charge":
        step_charge()
    else:
        print(f'STEP={STEP!r} 无效，应为 "generate" 或 "charge"。', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

