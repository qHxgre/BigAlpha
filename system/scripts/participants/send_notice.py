"""给某赛道全体参赛者发一条固定内容的站内信（对接 /bigapis/notify/v1/notice）。

场景：向报名文件 user_id_<competition_id>.json 里的所有 user 群发同一条通知
（如「模型本地化训练指南」压缩包下载密码）。参赛者每天新增，本脚本按人维护
一份「已发名单」台账，重跑时只补发新报名、还没发过的人，不会重复打扰老用户。

认证：优先环境变量 BIGQUANT_TOKEN / BIGQUANT_SERVER，其次 ~/.bigquant/auth.json，
与 send_daily_reports.py / reward_coins.py 一致。

防重：每次发送成功都会往「该批次台账」<NOTICE_DIR>/<NOTICE_KEY>.sent.csv 追加一行
(user_id, sent_at)。台账按 NOTICE_KEY 独立，互不影响。重跑时名单里已在台账中的
user 会被跳过，只发新报名、还没发过的人。改了正文想重发时，换一个 NOTICE_KEY 即可
（等于开一份新台账）。

用法：
    python send_notice.py [competition_id]

    competition_id 默认见下方 COMPETITION_ID；命令行传入可覆盖。

流程：
    1. 先跑 participants.py 刷新 user_id_<competition_id>.json（拉最新报名名单）。
    2. DRY_RUN=True 跑一遍预览：确认接收人数、标题、正文都正常。
    3. 改 DRY_RUN=False 再跑，实际发送并回写台账。
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.auth import load_auth
from common.ids import load_id_list_json
from common.paths import NOTICE_DIR, PARTICIPANTS_DIR

try:
    import requests
except ImportError:
    print("请先安装依赖: python3 -m pip install requests", file=sys.stderr)
    sys.exit(1)

# ===== 配置：改这里就行 =====================================================
# 目标比赛 ID：控制给哪个赛道的报名者发。命令行传入 argv[0] 可覆盖此默认值。
COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"

# 批次标识：台账文件名用它区分不同的群发批次（<NOTICE_KEY>.sent.csv）。
# 想重发/改内容时换一个 KEY，等于开一份新台账、对全体重新发放。
NOTICE_KEY = "e2e_local_train_download_pwd"

DRY_RUN = False                  # True 只预览，不发送、不回写台账；确认后改 False

# 通知标题与正文（固定内容，群发给名单里每个人）。
TITLE = "BigAlpha 端到端赛道 · 本地训练数据下载密码"
CONTENT = (
    "各位参赛选手好，端到端赛道「模型本地化训练指南」中训练数据压缩包的下载密码为：\n"
    "0qi9\n"
    "请查看指南文档中的分享链接下载数据，祝训练顺利。"
)

# 通知分类：system(系统) / resource(资源算力) / community(社区) / papertrading(模拟任务)。
CHANNEL = "community"

# 目标空间 ID，不传则用当前登录账号所属空间。主空间全零 UUID。
SPACE_ID = "00000000-0000-0000-0000-000000000000"

# 每次调用之间的间隔（秒），0 表示不限速。
SLEEP_BETWEEN = 0.2

# 通知正文最大长度（接口限制 1024 字符）。
MAX_CONTENT = 1024

# 测试模式：只发给 TEST_USER_IDS 里列出的用户（不读报名名单、不影响正式台账逻辑）。
TEST_MODE = False
TEST_USER_IDS: list[str] = [
    "5dd35480-0f38-11ed-93bb-da75731aa77c",
]
# ===========================================================================

LEDGER_COLUMNS = ["user_id", "sent_at"]


def ledger_path(notice_key: str) -> Path:
    """按批次 KEY 生成台账路径，去掉文件名不安全字符。"""
    safe = notice_key.replace("/", "_").replace("\\", "_").replace(":", "_")
    return NOTICE_DIR / f"{safe}.sent.csv"


def load_recipients(competition_id: str) -> list[str]:
    """读该赛道报名 user_id 名单 user_id_<competition_id>.json（去重保序）。"""
    if TEST_MODE:
        from common.ids import dedup_keep_order
        return dedup_keep_order(TEST_USER_IDS)
    path = PARTICIPANTS_DIR / f"user_id_{competition_id}.json"
    if not path.exists():
        print(
            f"未找到报名文件: {path}\n请先跑 participants.py 生成按赛道的报名 json。",
            file=sys.stderr,
        )
        sys.exit(1)
    return load_id_list_json(path)


def load_ledger(ledger_csv: Path) -> set[str]:
    """读已发台账，返回已发过的 {user_id} 集合。"""
    sent: set[str] = set()
    if not ledger_csv.exists():
        return sent
    with ledger_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = (row.get("user_id") or "").strip()
            if uid:
                sent.add(uid)
    return sent


def append_ledger(ledger_csv: Path, user_id: str) -> None:
    exists = ledger_csv.exists()
    with ledger_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({
            "user_id": user_id,
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


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    competition_id = argv[0] if argv else COMPETITION_ID

    if len(CONTENT) > MAX_CONTENT:
        print(f"正文 {len(CONTENT)} 字符超过 {MAX_CONTENT} 上限，请精简后再发。", file=sys.stderr)
        sys.exit(1)

    NOTICE_DIR.mkdir(parents=True, exist_ok=True)
    ledger_csv = ledger_path(NOTICE_KEY)

    recipients = load_recipients(competition_id)
    if not recipients:
        print(f"报名名单为空，无可发送对象（competition={competition_id}）。", file=sys.stderr)
        sys.exit(1)

    sent = load_ledger(ledger_csv)
    _, server = load_auth()
    print(f"=== 群发站内信 · competition={competition_id} · key={NOTICE_KEY} · channel={CHANNEL} · space_id={SPACE_ID or '(默认)'} ===")
    print(f"接口: {server}/bigapis/notify/v1/notice")
    print(f"标题: {TITLE}")
    print(f"名单: {len(recipients)} 人 · 台账已发 {len(sent)} 人 · 台账文件 {ledger_csv}")
    if DRY_RUN:
        print("当前为预览模式(DRY_RUN=True)，未调任何接口、未回写台账。\n")

    token = None
    ok_count = skip_count = fail_count = 0
    for i, uid in enumerate(recipients, 1):
        if uid in sent:
            skip_count += 1
            continue

        if DRY_RUN:
            ok_count += 1
            print(f"[{i}/{len(recipients)}] ✓ {uid}  待发")
            continue

        if token is None:
            token, server = load_auth()
        ok, msg = send_one(token, server, uid, TITLE, CONTENT)
        if ok:
            ok_count += 1
            append_ledger(ledger_csv, uid)
            print(f"[{i}/{len(recipients)}] ✓ {uid}")
        else:
            fail_count += 1
            print(f"[{i}/{len(recipients)}] ✗ {uid}  —— {msg}", file=sys.stderr)
        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)

    print("\n=== 汇总 ===")
    if DRY_RUN:
        print(f"待发 {ok_count} 人 · 已发过跳过 {skip_count} 人")
        print("确认无误后把 DRY_RUN 改成 False 再跑。")
    else:
        print(f"成功 {ok_count} 人 · 失败 {fail_count} 人 · 已发过跳过 {skip_count} 人")
        if fail_count:
            print("失败的用户未写入台账，修正后重跑本脚本会自动重试。")


if __name__ == "__main__":
    main()
