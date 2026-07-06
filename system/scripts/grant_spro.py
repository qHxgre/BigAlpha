"""批量给 user_id.json 里的用户开通 spro（旗舰版 L1）权益。

调用后端 set_privilege 接口（POST /user_equity/{user_id}/set_privilege）逐个写入。
space_id 用主空间（全零 UUID）—— 主空间会同步更新数据权益资产，即账号级别的
全局旗舰版权限，而不是只在某个子空间生效。

认证复用 bigquant-space skill 的方式：
    优先读环境变量 BIGQUANT_TOKEN / BIGQUANT_SERVER；
    否则读 ~/.bigquant/auth.json 里的 token / server。

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

# ===== 配置：改这里就行 =====================================================
USER_ID_FILE = Path(__file__).parent / "files" / "participants" / "user_id.json"  # 用户 ID 列表（JSON 数组）
SPACE_ID = "00000000-0000-0000-0000-000000000000"        # 主空间（全零 UUID）
PRO_TYPE = "spro"                                         # 旗舰版 L1
EXPIRE_AT = "2026-08-20"                                  # 权益到期日 YYYY-MM-DD
DRY_RUN = True                                            # True 只预览；确认后改 False 真正写入

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


def get_privilege(token: str, server: str, user_id: str) -> dict | None:
    """查询用户当前权益，返回权益 dict（items 第一条）；查不到或异常返回 None。"""
    url = f"{server}{API_BASE}/user_equity/@query"
    params = {"constraints": json.dumps({"user_id": user_id}), "page": 1, "size": 1}
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        items = data.get("data", {}).get("items") or data.get("items") or []
        return items[0] if items else None
    except Exception as e:
        print(f"    [warn] 查询权益失败 {user_id}: {e}")
        return None


def has_spro(privilege: dict | None) -> bool:
    """判断用户是否已持有 spro 权益（equity.specification == 'spro'，不校验过期）。"""
    if not privilege:
        return False
    return privilege.get("equity", {}).get("specification") == PRO_TYPE


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


def load_user_ids() -> list[str]:
    try:
        ids = json.loads(USER_ID_FILE.read_text())
    except FileNotFoundError:
        print(f"未找到用户列表: {USER_ID_FILE}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(ids, list):
        print(f"{USER_ID_FILE} 应为 JSON 数组", file=sys.stderr)
        sys.exit(1)
    # 去重 + 去空，保持顺序
    seen: set[str] = set()
    result: list[str] = []
    for uid in ids:
        uid = str(uid).strip()
        if uid and uid not in seen:
            seen.add(uid)
            result.append(uid)
    return result


def main() -> None:
    user_ids = load_user_ids()
    token, server = load_auth()

    print(f"=== 批量开通 {PRO_TYPE}（旗舰版 L1）===")
    print(f"目标空间 : {SPACE_ID}（主空间，账号级全局权限）")
    print(f"到期日   : {EXPIRE_AT}")
    print(f"用户数量 : {len(user_ids)}")
    print(f"服务地址 : {server}\n")

    ok, skip, fail = 0, 0, 0
    failures: list[tuple[str, str]] = []

    for i, uid in enumerate(user_ids, 1):
        if DRY_RUN:
            privilege = get_privilege(token, server, uid)
            if has_spro(privilege):
                print(f"  [dry-run][已有spro] [{i}/{len(user_ids)}] {uid}  -> 跳过")
            else:
                print(f"  [dry-run] [{i}/{len(user_ids)}] {uid}  -> {PRO_TYPE} 至 {EXPIRE_AT}")
            continue

        # 已有有效 spro 权益则跳过，不覆盖
        privilege = get_privilege(token, server, uid)
        if has_spro(privilege):
            skip += 1
            print(f"  [跳过][已有spro] [{i}/{len(user_ids)}] {uid}")
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
        print(f"\n=== 完成：成功 {ok}，跳过（已有spro） {skip}，失败 {fail} ===")
        if failures:
            print("失败明细：")
            for uid, reason in failures:
                print(f"  {uid}: {reason}")


if __name__ == "__main__":
    main()