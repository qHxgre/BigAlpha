"""给一批用户赠送宽币（对接 /balance/reward），分两步走、中间留人工审核。

流程：
    第 1 步 STEP="generate"：
        算出本次「谁该发、发多少」，再从 charge_records.csv 里查出每个用户
        自 SINCE（默认 2026-06-30）以来的真实充值记录，写出两份文件：
            * charge_plan.csv       —— 人工审核用（用 Excel/表格打开）。
                  一行一个用户，含 to_charge / history_total / history_count /
                  charged_at / charge_ok。审核时：
                      - 已经充过的人 → 把该行 to_charge 改成 0，或直接删掉整行；
                      - 名单/金额有误 → 直接改 to_charge。
            * charge_plan.meta.json —— 侧车，存本次元信息 + 每人历史充值明细，
                  只作备查，charge 步不依赖它，人工无需改。
    第 2 步 STEP="charge"：
        读审核后的 charge_plan.csv，只对 to_charge>0 且尚未充过（charged_at 为空）
        的行调接口发币，成功后就地把 charged_at/charge_ok 写回 CSV，
        重跑不会重复充（CSV 既是计划也是结果台账）。

认证：优先环境变量 BIGQUANT_TOKEN / BIGQUANT_SERVER，其次 ~/.bigquant/auth.json。

用法：
    1. 先把最新的 charge_records.csv 放到 files/reward_coins/。
    2. STEP="generate" 跑一遍，生成 charge_plan.csv（+ .meta.json）。
    3. 人工用表格软件审核 charge_plan.csv，删/改错误行。
    4. STEP="charge" + DRY_RUN=True 预览，确认无误后 DRY_RUN=False 真发。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("请先安装依赖: python3 -m pip install requests", file=sys.stderr)
    sys.exit(1)

# ===== 赛道 ID & 各任务×各赛道金额（原 grant_coins.py 内联，脱离依赖）=========
CID_FACTOR = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"   # 赛道一 · AI 因子挖掘
CID_E2E = "523f9302-5b4b-42bd-bce1-f232e7c74316"       # 赛道二 · 端到端 AI 量化模型
CID_OPEN = "63dd885c-2488-4efd-9c61-9e3a536f172c"      # 赛道三 · AI 开放创新

# 某赛道不设该任务时就不在字典里出现（如端到端不设提交里程碑）。
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
# 两步流程开关："generate"（生成待审核计划）或 "charge"（按审核后的计划发币）。
STEP = "generate"

# 主空间全零 UUID：发平台宽币用主空间；子空间发币换成对应 space_id。
SPACE_ID = "00000000-0000-0000-0000-000000000000"

# —— 用户来源（三选一，按优先级命中即用）——
# 方式 0：按任务发。填 TASKS 里的任务名，自动按报名名单算「谁该发多少」。
TASK_KEY: str | None = "初始礼包"
# 方式 1：单独的 user_id 名单文件（.txt 一行一个，或 .json 数组），配 AMOUNT。
USER_IDS_FILE: Path | None = None
# 方式 2：内联写死的 user_id 列表，配 AMOUNT。
USER_IDS: list[str] = []
# 方式 1/2 用到的统一发放金额（TASK_KEY 模式忽略此项）。
AMOUNT = 5000

# TASK_KEY 模式的候选用户：留空=该任务涉及赛道的全部报名者；否则与报名记录取交集。
CANDIDATE_USER_IDS: list[str] = []
CANDIDATE_FILE: Path | None = None

# 本次发放的业务原因，写进请求 notes，也写进计划文件。
REASON = "初始礼包"

FILES_DIR = Path(__file__).parent / "files"
PARTICIPANTS_DIR = FILES_DIR / "participants"
REWARD_DIR = FILES_DIR / "reward_coins"

# 真实充值流水 CSV（generate 步用来查历史；导出方式见 generate_sql.ipynb）。
CHARGE_RECORDS_CSV = REWARD_DIR / "charge_records.csv"

# 只统计这个日期（含）以来的真实充值记录，写进计划供人工审核。
SINCE = date(2026, 6, 30)

# 充值计划：CSV 供人工审核（generate 写、charge 读回写），meta 侧车存元信息+明细。
PLAN_CSV = REWARD_DIR / "charge_plan.csv"
PLAN_META = REWARD_DIR / "charge_plan.meta.json"

# CSV 列顺序（charge 步按这个头读写；人工只需关心 to_charge，删行也安全）。
# history_same_amount：历史里金额==本次 to_charge 的笔数，>0 即本档已充过=重复。
PLAN_COLUMNS = [
    "user_id", "to_charge", "reason",
    "history_total", "history_count", "history_same_amount",
    "last_charge_at",
    "charged_at", "charge_ok",
]

# 每次调用之间的间隔（秒），0 表示不限速。
SLEEP_BETWEEN = 0.2

DRY_RUN = False   # charge 步：True 只预览；确认后改 False 才真正调接口发币
# ===========================================================================


def load_auth() -> tuple[str, str]:
    """读认证：优先环境变量，其次 ~/.bigquant/auth.json。返回 (token, server)。"""
    token = os.environ.get("BIGQUANT_TOKEN")
    server = os.environ.get("BIGQUANT_SERVER", "").rstrip("/")
    if not token:
        auth_file = Path(
            os.environ.get("BIGQUANT_AUTH_FILE", Path.home() / ".bigquant" / "auth.json")
        )
        try:
            data = json.loads(auth_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"未找到认证文件: {auth_file}", file=sys.stderr)
            sys.exit(1)
        token = data.get("token")
        if not token:
            print("auth.json 中缺少 token 字段", file=sys.stderr)
            sys.exit(1)
        if not server:
            server = str(data.get("server", "https://bigquant.com")).rstrip("/")
    if not server:
        server = "https://bigquant.com"
    return token, server


def read_id_list(path: Path) -> list[str]:
    """读 user_id 名单：.json 数组或纯文本（一行一个），去空去重保序。"""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        arr = json.loads(text)
        if not isinstance(arr, list):
            print(f"{path} 应为 JSON 数组", file=sys.stderr)
            sys.exit(1)
        ids = [str(x).strip() for x in arr]
    else:
        ids = [ln.strip() for ln in text.splitlines()]
    return list(dict.fromkeys(u for u in ids if u))


def load_participants(cid: str) -> list[str]:
    """读某赛道报名 user_id 列表 user_id_<cid>.json。"""
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


def load_task_candidates() -> set[str] | None:
    """TASK_KEY 模式的候选集合；None=不限制（用全部报名者）。"""
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


def collect_targets() -> dict[str, int]:
    """按配置汇总本次要发的 {user_id: amount}。多赛道同一 user 取较大金额。"""
    targets: dict[str, int] = {}

    def put(uid: str, amt: int) -> None:
        if amt > targets.get(uid, 0):
            targets[uid] = amt

    if TASK_KEY:
        if TASK_KEY not in TASKS:
            print(f"TASK_KEY={TASK_KEY!r} 不在 TASKS 中，可选：{list(TASKS)}", file=sys.stderr)
            sys.exit(1)
        candidates = load_task_candidates()
        for cid, amount in TASKS[TASK_KEY].items():
            for uid in load_participants(cid):
                if candidates is not None and uid not in candidates:
                    continue
                put(uid, amount)  # 多赛道取较大值，不叠加
        if not targets:
            print(f"任务 {TASK_KEY} 没有符合条件的用户。", file=sys.stderr)
            sys.exit(1)
        return targets

    if USER_IDS_FILE is not None:
        for uid in read_id_list(USER_IDS_FILE):
            put(uid, AMOUNT)
        return targets

    for uid in USER_IDS:
        uid = str(uid).strip()
        if uid:
            put(uid, AMOUNT)
    if not targets:
        print("没有任何用户来源：请设置 TASK_KEY / USER_IDS_FILE / USER_IDS 之一。", file=sys.stderr)
        sys.exit(1)
    return targets


def parse_csv_date(s: str) -> date | None:
    """解析 CSV created_at，形如 '2026-7-1, 14:26'。失败返回 None。"""
    s = (s or "").strip().strip('"')
    if not s:
        return None
    head = s.split(",")[0].strip()  # '2026-7-1'
    try:
        y, m, d = (int(x) for x in head.split("-"))
        return date(y, m, d)
    except ValueError:
        return None


def load_history(uids: set[str]) -> dict[str, list[dict]]:
    """从 charge_records.csv 查这些用户 SINCE 以来、本 space 的真实充值记录。

    返回 {user_id: [{date, amount, description, id}, ...]}，按日期升序。
    只取 type=bigquant_charge、status=paid、space_id==SPACE_ID 的行。
    """
    hist: dict[str, list[dict]] = {u: [] for u in uids}
    if not CHARGE_RECORDS_CSV.exists():
        print(
            f"⚠️  未找到充值流水: {CHARGE_RECORDS_CSV}\n"
            f"    无法带出历史充值记录，生成的计划将缺少防重依据！"
            f"请先用 generate_sql.ipynb 的 SQL 导出流水到该路径。",
            file=sys.stderr,
        )
        return hist

    with CHARGE_RECORDS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = (row.get("user_id") or "").strip()
            if uid not in hist:
                continue
            if row.get("type") != "bigquant_charge":
                continue
            if row.get("status") not in (None, "", "paid"):
                continue
            if SPACE_ID and row.get("space_id") not in (None, "", SPACE_ID):
                continue
            d = parse_csv_date(row.get("created_at", ""))
            if d is None or d < SINCE:
                continue
            try:
                amt = float(row.get("amount") or 0)
            except ValueError:
                amt = 0.0
            hist[uid].append({
                "date": d.isoformat(),
                "amount": amt,
                "description": (row.get("description") or "").strip(),
                "id": (row.get("id") or "").strip(),
            })
    for recs in hist.values():
        recs.sort(key=lambda r: r["date"])
    return hist


def step_generate() -> None:
    """第 1 步：生成待人工审核的 charge_plan.csv（+ meta 侧车）。"""
    targets = collect_targets()
    hist = load_history(set(targets))

    rows = []
    meta_history = {}
    for uid in sorted(targets):
        past = hist.get(uid, [])
        total = round(sum(r["amount"] for r in past), 2)
        amt = targets[uid]
        # 历史里金额==本次拟充档位的笔数：>0 说明这个档已经发过，本次多半是重复。
        same_amount = sum(1 for r in past if float(r["amount"]) == float(amt))
        last_charge_at = max((r["date"] for r in past), default="")
        rows.append({
            "user_id": uid,
            "to_charge": amt,           # 本次拟充；人工可改，改成 0 或删行即不发
            "reason": REASON,
            "history_total": total,     # 该期间已充合计
            "history_count": len(past), # 该期间充值总笔数
            "history_same_amount": same_amount,  # 同档已充笔数，判断重复的关键
            "last_charge_at": last_charge_at,    # 最近一次充值日期
            "charged_at": "",           # charge 步成功后回填，防重复
            "charge_ok": "",
        })
        meta_history[uid] = past        # 明细放侧车备查，不进 CSV 免噪音

    rows.sort(key=lambda r: r["last_charge_at"] or "", reverse=True)

    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    with PLAN_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "reason": REASON,
        "space_id": SPACE_ID,
        "since": SINCE.isoformat(),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": (
            f"TASK_KEY={TASK_KEY}" if TASK_KEY
            else f"USER_IDS_FILE={USER_IDS_FILE}" if USER_IDS_FILE
            else "内联 USER_IDS"
        ),
        "plan_csv": PLAN_CSV.name,
        "history": meta_history,  # {user_id: [{date, amount, description, id}, ...]}
    }
    PLAN_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    any_hist = sum(1 for r in rows if r["history_total"] > 0)
    same_amt = sum(1 for r in rows if r["history_same_amount"] > 0)
    print(f"=== STEP=generate · REASON={REASON} ===")
    print(f"计划文件: {PLAN_CSV}")
    print(f"侧车文件: {PLAN_META}（元信息 + 历史明细，备查，无需改）")
    print(f"本次候选: {len(rows)} 人")
    print(f"{SINCE.isoformat()} 以来有过任意充值: {any_hist} 人")
    print(f"其中「本档金额已充过」(history_same_amount>0): {same_amt} 人（大概率重复，重点核对）")
    print("\n请用表格软件打开 charge_plan.csv 审核：")
    print("  · history_same_amount>0 → 本档已发过，把该行 to_charge 改成 0 或删整行；")
    print("  · 名单/金额有误 → 直接改 to_charge。")
    print("审核完成后，把 STEP 改成 \"charge\" 再跑（建议先 DRY_RUN=True 预览）。")


def reward_one(token: str, server: str, user_id: str, amount: int) -> tuple[bool, str]:
    """给单个用户发币。返回 (是否成功, 说明信息)。"""
    url = f"{server}/bigapis/kbb2/v1/balance/reward"
    body = {
        "reward_space_id": SPACE_ID,
        "reward_user_id": user_id,
        "reward_kbb": str(amount),
        "notes": {"reason": REASON, "source": "reward_coins.py"},
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


def load_plan_rows() -> list[dict]:
    """读审核后的 charge_plan.csv，返回行列表（保序，用于回写）。"""
    try:
        with PLAN_CSV.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"未找到计划文件: {PLAN_CSV}\n请先用 STEP=\"generate\" 生成并人工审核。", file=sys.stderr)
        sys.exit(1)


def save_plan_rows(rows: list[dict]) -> None:
    """把（可能已回填 charged_at 的）行写回 charge_plan.csv，保持列顺序。"""
    with PLAN_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def to_int(v) -> int:
    """CSV 里的 to_charge 容错转 int（'', '5000', '5000.0' 都能处理）。"""
    s = str(v or "").strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def step_charge() -> None:
    """第 2 步：按审核后的 charge_plan.csv 实际发币，结果回写 CSV。"""
    rows = load_plan_rows()
    # 待发：to_charge>0 且还没成功充过（charged_at 为空）。
    pending = [
        r for r in rows
        if to_int(r.get("to_charge")) > 0 and not (r.get("charged_at") or "").strip()
    ]
    total_coins = sum(to_int(r["to_charge"]) for r in pending)
    _, server = load_auth()

    print(f"=== STEP=charge · REASON={REASON} ===")
    print(f"接口: {server}/bigapis/auth/v1/balance/reward   space_id={SPACE_ID}")
    print(f"计划文件: {PLAN_CSV}")
    print(f"计划内共 {len(rows)} 行，本次待发（to_charge>0 且未充过）: {len(pending)} 人")
    print(f"本次合计发放: {total_coins} 宽币")

    if not pending:
        print("\n没有需要发放的记录，结束。")
        return

    if DRY_RUN:
        print("\n当前为预览模式(DRY_RUN=True)，未调任何接口、未回写 CSV。"
              "确认无误后把 DRY_RUN 改成 False 再跑。")
        return

    token, server = load_auth()
    ok_count = fail_count = 0
    for i, r in enumerate(pending, 1):
        uid = (r.get("user_id") or "").strip()
        amt = to_int(r["to_charge"])
        ok, msg = reward_one(token, server, uid, amt)
        now = datetime.now(timezone.utc).astimezone().isoformat()
        r["charge_ok"] = str(ok)
        if ok:
            ok_count += 1
            r["charged_at"] = now          # 回填，重跑不再重复
            print(f"[{i}/{len(pending)}] ✓ {uid}  {amt} 宽币")
        else:
            fail_count += 1
            print(f"[{i}/{len(pending)}] ✗ {uid}  {amt} 宽币  —— {msg}", file=sys.stderr)
        # 每成功 20 个回写一次 CSV，中断也不丢已充的账。
        if ok and ok_count % 20 == 0:
            save_plan_rows(rows)
        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)

    save_plan_rows(rows)
    print(f"\n=== 完成：成功 {ok_count} 人，失败 {fail_count} 人 ===")
    print(f"结果已回写: {PLAN_CSV}")
    if fail_count:
        print("失败的行 charged_at 仍为空，修正后重跑本脚本(charge)会自动重试。")


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

