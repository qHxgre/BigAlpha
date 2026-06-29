"""批量给「已通过审核」的参赛者开通 spro（旗舰版 L1）权益。

不再依赖 user_id.json —— 直接通过 AlphathonClient 查询比赛报名记录，
筛出已审核通过的参赛者（默认 approved + approved_join_space），再逐个调用
后端 set_privilege 接口（POST /user_equity/{user_id}/set_privilege）写入。

space_id 用主空间（全零 UUID）—— 主空间会同步更新数据权益资产，即账号级别的
全局旗舰版权限，而不是只在某个子空间生效。

两套认证：
  查询参赛者走 AlphathonClient（见 _client.py，读 ALPHATHON_API_TOKEN /
    ALPHATHON_JWT_FILE）。
  写入权益走 bigauth，复用 bigquant-space skill 的方式：优先读环境变量
    BIGQUANT_TOKEN / BIGQUANT_SERVER，否则读 ~/.bigquant/auth.json 里的
    token / server。

先用 DRY_RUN=True 预览名单，确认无误后改成 False 再跑。
失败单条不影响其他人，最后打印汇总。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("请先安装依赖: python3 -m pip install requests", file=sys.stderr)
    sys.exit(1)

from _client import AlphathonClient

# ===== 配置：改这里就行 =====================================================
COMPETITION_IDS = [
    "76ad3f56-ec2b-431a-890e-139a7f4bbcba",
    "523f9302-5b4b-42bd-bce1-f232e7c74316",
    "63dd885c-2488-4efd-9c61-9e3a536f172c",
]
SOURCE_STATUSES = {"approved", "approved_join_space"}    # 给这些状态的参赛者开通（已通过审核）
SPACE_ID = "00000000-0000-0000-0000-000000000000"        # 主空间（全零 UUID）
PRO_TYPE = "spro"                                         # 旗舰版 L1
EXPIRE_AT = "2026-08-20"                                  # 权益到期日 YYYY-MM-DD
DRY_RUN = False                                            # True 只预览；确认后改 False 真正写入

# spro（旗舰版 L1）固定权益结构，见 skill references/users/management.md
EQUITY = {
    "aiflow_server": [
        {"cpu": 1, "memory": 6, "specification": "P1", "count": 3},
        {"cpu": 1, "memory": 6, "specification": "R1", "count": 2},
    ],
    "aistudio_server": {"cpu": 2, "memory": 8, "specification": "D1"},
    "datasource": {"limit": 2},
}
# ===========================================================================

API_BASE = "/bigapis/auth/v1"


def load_auth() -> tuple[str, str]:
    """返回 (token, server)。优先环境变量，否则读 ~/.bigquant/auth.json。"""
    token = os.environ.get("BIGQUANT_TOKEN")
    server = os.environ.get("BIGQUANT_SERVER", "").rstrip("/")

    if not token:
        auth_file = Path(
            os.environ.get("BIGQUANT_AUTH_FILE", Path.home() / ".bigquant" / "auth.json")
        )
        try:
            data = json.loads(auth_file.read_text())
        except FileNotFoundError:
            print(f"未找到认证文件: {auth_file}", file=sys.stderr)
            sys.exit(1)
        token = data.get("token")
        if not token:
            print("auth.json 中缺少 token 字段", file=sys.stderr)
            sys.exit(1)
        if not server:
            server = data.get("server", "https://bigquant.com").rstrip("/")

    return token, server or "https://bigquant.com"


def set_privilege(token: str, server: str, user_id: str) -> dict:
    """给单个用户写入 spro 权益。"""
    url = f"{server}{API_BASE}/user_equity/{user_id}/set_privilege"
    body = {
        "space_id": SPACE_ID,
        "equity": EQUITY,
        "expire_at": EXPIRE_AT,
        "pro_type": PRO_TYPE,
    }
    resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def collect_approved_user_ids(client: AlphathonClient) -> list[str]:
    """查询所有比赛，筛出已通过审核的参赛者账号 user_id，去重保序返回。"""
    seen: set[str] = set()
    result: list[str] = []
    for cid in COMPETITION_IDS:
        users = client.list_users(cid, order_by=["created_at"])
        hit = sum(1 for u in users if u.get("status") in SOURCE_STATUSES)
        print(f"  比赛 {cid}: 报名 {len(users)}，命中 {hit}")
        for u in users:
            if u.get("status") not in SOURCE_STATUSES:
                continue
            uid = str(u.get("user_id") or "").strip()
            if uid and uid not in seen:
                seen.add(uid)
                result.append(uid)
    return result


def main() -> None:
    print(f"=== 查询参赛者（状态命中 {sorted(SOURCE_STATUSES)}）===")
    client = AlphathonClient()
    user_ids = collect_approved_user_ids(client)

    token, server = load_auth()

    print(f"\n=== 批量开通 {PRO_TYPE}（旗舰版 L1）===")
    print(f"目标空间 : {SPACE_ID}（主空间，账号级全局权限）")
    print(f"到期日   : {EXPIRE_AT}")
    print(f"用户数量 : {len(user_ids)}")
    print(f"服务地址 : {server}\n")

    ok, fail = 0, 0
    failures: list[tuple[str, str]] = []

    for i, uid in enumerate(user_ids, 1):
        if DRY_RUN:
            print(f"  [dry-run] [{i}/{len(user_ids)}] {uid}  -> {PRO_TYPE} 至 {EXPIRE_AT}")
            continue
        try:
            set_privilege(token, server, uid)
            ok += 1
            print(f"  [ok] [{i}/{len(user_ids)}] {uid}")
        except requests.HTTPError as e:
            fail += 1
            detail = ""
            if e.response is not None:
                try:
                    detail = json.dumps(e.response.json(), ensure_ascii=False)
                except Exception:
                    detail = e.response.text
            failures.append((uid, detail or str(e)))
            print(f"  [失败] [{i}/{len(user_ids)}] {uid}: {detail or e}")
        except Exception as e:  # noqa: BLE001 — 单条失败不影响其他人
            fail += 1
            failures.append((uid, str(e)))
            print(f"  [失败] [{i}/{len(user_ids)}] {uid}: {e}")

    if DRY_RUN:
        print(f"\n当前为预览模式(DRY_RUN=True)，共 {len(user_ids)} 人。确认无误后把 DRY_RUN 改成 False 再跑。")
    else:
        print(f"\n=== 完成：成功 {ok}，失败 {fail} ===")
        if failures:
            print("失败明细：")
            for uid, reason in failures:
                print(f"  {uid}: {reason}")


if __name__ == "__main__":
    main()
