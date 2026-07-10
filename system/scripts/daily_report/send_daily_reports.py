"""把 daily_reports 目录下每个 <user_id>.md 作为站内信发给对应用户。
（目录见 common.paths.DAILY_REPORTS_DIR：system/files/scripts/daily_reports）

对接消息通知接口 /bigapis/notify/v1/notice（POST），单用户 signal 通知：
文件名（去掉 .md）即接收者 user_id，文件内容即通知正文。

认证：优先环境变量 BIGQUANT_TOKEN / BIGQUANT_SERVER，其次 ~/.bigquant/auth.json，
与 reward_coins.py 一致。

防重：每次发送成功都会往 REPORTS_DIR/.sent_ledger.csv 追加一行
(user_id, report_sha1, sent_at)。重跑时，正文内容未变（sha1 相同）的用户会被
跳过，不会重复打扰；报告更新（sha1 变化）后会重新发送。

用法：
    1. 先跑 daily_score_digest.py 生成当天各用户的 <user_id>.md。
    2. DRY_RUN=True 跑一遍预览：确认接收人数、标题、正文长度都正常。
    3. 改 DRY_RUN=False 再跑，实际发送并回写 ledger。
"""

from __future__ import annotations

import csv
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.auth import load_auth
from common.paths import DAILY_REPORTS_DIR

try:
    import requests
except ImportError:
    print("请先安装依赖: python3 -m pip install requests", file=sys.stderr)
    sys.exit(1)

# ===== 配置：改这里就行 =====================================================
DRY_RUN = False                 # True 只预览，不发送、不回写 ledger；确认后改 False

# 通知标题。{date} 会被替换成从正文里解析出的日期（解析不到则留空）。
TITLE_TEMPLATE = "BigAlpha 得分日报 {date}"

# 通知分类：system(系统) / resource(资源算力) / community(社区) / papertrading(模拟任务)。
CHANNEL = "community"

# 目标空间 ID，不传则用当前登录账号所属空间。主空间全零 UUID。
SPACE_ID = "00000000-0000-0000-0000-000000000000"

# 每次调用之间的间隔（秒），0 表示不限速。
SLEEP_BETWEEN = 0.2

# 通知正文最大长度（接口限制 1024 字符），超长的报告会被跳过并告警。
MAX_CONTENT = 1024

# 测试模式：只发给 TEST_USER_IDS 里列出的用户（对应文件需存在）。
TEST_MODE = False
TEST_USER_IDS: list[str] = []
# ===========================================================================

REPORTS_DIR = DAILY_REPORTS_DIR
LEDGER_CSV = REPORTS_DIR / ".sent_ledger.csv"
LEDGER_COLUMNS = ["user_id", "report_sha1", "sent_at"]


def parse_date(content: str) -> str:
    """从报告首行 '### 得分日报 07-09' 里抠出日期串；抠不到返回空串。"""
    first = content.lstrip().splitlines()[0] if content.strip() else ""
    tail = first.lstrip("#").strip()
    for token in tail.split():
        if any(ch.isdigit() for ch in token) and ("-" in token or "/" in token):
            return token
    return ""


def make_title(content: str) -> str:
    date = parse_date(content)
    return TITLE_TEMPLATE.format(date=date).strip()


def collect_reports() -> list[tuple[str, str]]:
    """返回 [(user_id, content)]，按 user_id 排序。"""
    if not REPORTS_DIR.exists():
        print(f"未找到报告目录: {REPORTS_DIR}", file=sys.stderr)
        sys.exit(1)
    items: list[tuple[str, str]] = []
    for md in sorted(REPORTS_DIR.glob("*.md")):
        user_id = md.stem.strip()
        if not user_id:
            continue
        if TEST_MODE and user_id not in set(TEST_USER_IDS):
            continue
        items.append((user_id, md.read_text(encoding="utf-8")))
    return items


def load_ledger() -> set[tuple[str, str]]:
    """读已发台账，返回 {(user_id, report_sha1)} 集合。"""
    sent: set[tuple[str, str]] = set()
    if not LEDGER_CSV.exists():
        return sent
    with LEDGER_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = (row.get("user_id") or "").strip()
            sha = (row.get("report_sha1") or "").strip()
            if uid and sha:
                sent.add((uid, sha))
    return sent


def append_ledger(user_id: str, sha1: str) -> None:
    exists = LEDGER_CSV.exists()
    with LEDGER_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({
            "user_id": user_id,
            "report_sha1": sha1,
            "sent_at": datetime.now(timezone.utc).astimezone().isoformat(),
        })


def send_one(token: str, server: str, user_id: str, title: str, content: str) -> tuple[bool, str]:
    """给单个用户发一条 signal 站内信。返回 (是否成功, 说明信息)。"""
    url = f"{server}/bigapis/notify/v1/notice"
    body = {
        "notice": {
            "title": title,
            "content": content,
            "channel": CHANNEL,
            "notice_type": "signal",
            "recipient_id": user_id,
        },
    }
    if SPACE_ID:
        body["space_id"] = SPACE_ID
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


def main() -> None:
    reports = collect_reports()
    if not reports:
        print(f"在 {REPORTS_DIR} 下没有可发送的 <user_id>.md。", file=sys.stderr)
        sys.exit(1)

    sent = load_ledger()
    _, server = load_auth()
    print(f"=== 发送得分日报站内信 · channel={CHANNEL} · space_id={SPACE_ID or '(默认)'} ===")
    print(f"接口: {server}/bigapis/notify/v1/notice")
    print(f"报告目录: {REPORTS_DIR}（共 {len(reports)} 份）")
    if DRY_RUN:
        print("当前为预览模式(DRY_RUN=True)，未调任何接口、未回写 ledger。\n")

    token = None
    ok_count = skip_count = fail_count = toolong_count = 0
    for i, (uid, content) in enumerate(reports, 1):
        sha1 = hashlib.sha1(content.encode("utf-8")).hexdigest()
        title = make_title(content)

        if (uid, sha1) in sent:
            skip_count += 1
            print(f"[{i}/{len(reports)}] - {uid}  已发过（内容未变），跳过")
            continue

        if len(content) > MAX_CONTENT:
            toolong_count += 1
            print(
                f"[{i}/{len(reports)}] ! {uid}  正文 {len(content)} 字符超过 {MAX_CONTENT} 上限，跳过",
                file=sys.stderr,
            )
            continue

        if DRY_RUN:
            print(f"[{i}/{len(reports)}] ✓ {uid}  «{title}»  {len(content)} 字符")
            continue

        if token is None:
            token, server = load_auth()
        ok, msg = send_one(token, server, uid, title, content)
        if ok:
            ok_count += 1
            append_ledger(uid, sha1)
            print(f"[{i}/{len(reports)}] ✓ {uid}  «{title}»  {len(content)} 字符")
        else:
            fail_count += 1
            print(f"[{i}/{len(reports)}] ✗ {uid}  —— {msg}", file=sys.stderr)
        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)

    print("\n=== 汇总 ===")
    if DRY_RUN:
        pending = len(reports) - skip_count - toolong_count
        print(f"待发 {pending} 人 · 已发过跳过 {skip_count} 人 · 超长跳过 {toolong_count} 人")
        print("确认无误后把 DRY_RUN 改成 False 再跑。")
    else:
        print(
            f"成功 {ok_count} 人 · 失败 {fail_count} 人 · "
            f"已发过跳过 {skip_count} 人 · 超长跳过 {toolong_count} 人"
        )
        if fail_count:
            print("失败的用户未写入 ledger，修正后重跑本脚本会自动重试。")


if __name__ == "__main__":
    main()
