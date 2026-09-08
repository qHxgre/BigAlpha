"""批量给指定用户配置数据 SDK 的只读权限。

沿用 grant_spro.py 的用法：顶部集中配置 + DRY_RUN 预览，先看名单确认无误
再把 DRY_RUN 改成 False 真正写入。单条失败不影响其他人，最后打印汇总。

权限走 BigQuant 数据 SDK 接口（同 bigquant-data-manager skill）：
    POST   /bigapis/data/v1/sdk/userconfig          新建配置
    PATCH  /bigapis/data/v1/sdk/userconfigs/{uid}    更新已有配置
    GET    /bigapis/data/v1/sdk/userconfigs          查询是否已有配置

字段说明：
    sdk_level    套餐等级，这里固定 custom（自定义表，须配合 data 列表）
    role         1=只读（本脚本固定只读），2=读写，0=无权限
    data         自定义数据表 ID 列表（datasource_id），仅 custom 生效
    weekly_quota 每周配额（行数），-1 表示不限量
    end_date     权限截止日期 YYYY-MM-DD，留空表示无限期

已有配置的用户：默认直接跳过，不改动其原有配置；但如果检查到其每周配额是 -1
（不限额度），则不跳过，重新按下面的配置修改（PATCH）其 SDK 权限。没有配置的
用户：按下面的数据表 / 配额 / 截止日期新建。

认证复用 ~/.bigquant/auth.json（或环境变量 BIGQUANT_TOKEN / BIGQUANT_SERVER）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.auth import load_auth
from common.ids import dedup_keep_order, load_id_list_json
from common.paths import PARTICIPANTS_DIR

try:
    import requests
except ImportError:
    print("请先安装依赖: python3 -m pip install requests", file=sys.stderr)
    sys.exit(1)

# ===== 配置：改这里就行 =====================================================
# 用户列表：默认读 participants 目录下按赛道的 user_id_<cid>.json（JSON 数组）。
# 也可以直接把用户 ID 写进下面的 USER_IDS 列表，非空时优先用它。
USER_ID_FILE = PARTICIPANTS_DIR / "user_id_523f9302-5b4b-42bd-bce1-f232e7c74316.json"
USER_IDS: list[str] = []  # 例如 ["uid1", "uid2"]；留空则读 USER_ID_FILE

# 数据表列表（datasource_id）。这些表将以只读权限开给上面的用户。
DATASOURCE_IDS: list[str] = [
    "bigalpha_2026_e2e_bar1m",
    "bigalpha_2026_e2e_bar5m",
    "bigalpha_2026_e2e_bar15m",
    "bigalpha_2026_e2e_bar30m",
]

WEEKLY_QUOTA = 50000000           # 每周配额（行数），-1 表示不限量（后端整数上限约 21 亿，勿填更大数字）
END_DATE = "2026-08-20"      # 权限截止日期 YYYY-MM-DD，留空 "" 表示无限期
DRY_RUN = False               # True 只预览；确认后改 False 真正写入
# ===========================================================================

SDK_BASE = "/bigapis/data/v1/sdk"
SDK_LEVEL = "custom"         # 自定义表套餐
ROLE_READONLY = 1            # 只读


def _api(method: str, path: str, token: str, server: str, **kwargs) -> dict | None:
    url = f"{server}{SDK_BASE}{path}"
    resp = requests.request(
        method, url, headers={"Authorization": f"Bearer {token}"}, timeout=30, **kwargs
    )
    resp.raise_for_status()
    if resp.status_code == 204 or not resp.content:
        return None
    return resp.json()


def get_config(token: str, server: str, user_id: str) -> dict | None:
    """查询用户现有 SDK 配置，没有则返回 None。"""
    result = _api(
        "GET", "/userconfigs", token, server,
        params={"constraints": json.dumps({"user_id": user_id}), "size": 1},
    )
    items = (result or {}).get("data", {}).get("items") or []
    return items[0] if items else None


def grant_readonly(token: str, server: str, user_id: str) -> str:
    """给单个用户配置只读数据权限。

    已有配置的用户：默认跳过，返回 'skipped'，不改动其原有配置；
        但若其 weekly_quota 为 -1（不限额度），则改用 PATCH 重新按当前配置
        修改权限，返回 'updated'。
    没有配置的用户：新建，返回 'created'。
    """
    existing = get_config(token, server, user_id)

    if existing:
        # 已有配置：仅当额度为 -1（不限量）时重新修改，否则保持原样跳过。
        if existing.get("weekly_quota") == -1:
            patch: dict = {
                "sdk_level": SDK_LEVEL,
                "role": ROLE_READONLY,
                "data": list(dict.fromkeys(DATASOURCE_IDS)),
                "weekly_quota": WEEKLY_QUOTA,
            }
            if END_DATE:
                patch["end_date"] = END_DATE
            _api("PATCH", f"/userconfigs/{user_id}", token, server, json=patch)
            return "updated"
        return "skipped"

    body: dict = {
        "user_id": user_id,
        "sdk_level": SDK_LEVEL,
        "role": ROLE_READONLY,
        "data": list(dict.fromkeys(DATASOURCE_IDS)),
        "weekly_quota": WEEKLY_QUOTA,
    }
    if END_DATE:
        body["end_date"] = END_DATE
    _api("POST", "/userconfig", token, server, json=body)
    return "created"


def load_user_ids() -> list[str]:
    if USER_IDS:
        return dedup_keep_order(USER_IDS)
    return load_id_list_json(USER_ID_FILE)


def main() -> None:
    if not DATASOURCE_IDS:
        print("请先在 DATASOURCE_IDS 里填入要开权限的数据表 ID（datasource_id）。", file=sys.stderr)
        sys.exit(1)

    user_ids = load_user_ids()
    token, server = load_auth()
    tables = list(dict.fromkeys(DATASOURCE_IDS))

    print("=== 批量配置数据 SDK 只读权限 ===")
    print(f"套餐等级 : {SDK_LEVEL}（自定义表）")
    print(f"角色     : 只读 (role={ROLE_READONLY})")
    print(f"数据表   : {len(tables)} 个")
    for t in tables:
        print(f"           - {t}")
    print(f"每周配额 : {'不限量' if WEEKLY_QUOTA == -1 else str(WEEKLY_QUOTA) + ' 行'}")
    print(f"截止日期 : {END_DATE or '无限期'}")
    print(f"用户数量 : {len(user_ids)}")
    print(f"服务地址 : {server}\n")

    created = updated = skipped = fail = 0
    failures: list[tuple[str, str]] = []

    for i, uid in enumerate(user_ids, 1):
        if DRY_RUN:
            existing = get_config(token, server, uid)
            if existing and existing.get("weekly_quota") == -1:
                print(f"  [dry-run] [{i}/{len(user_ids)}] {uid}  -> 已有配置但额度为 -1（不限量），将重新修改：只读 {len(tables)} 张表，配额 {WEEKLY_QUOTA}，至 {END_DATE or '无限期'}")
            elif existing:
                print(f"  [dry-run] [{i}/{len(user_ids)}] {uid}  -> 已有 SDK 配置，跳过")
            else:
                print(f"  [dry-run] [{i}/{len(user_ids)}] {uid}  -> 无配置，将新建：只读 {len(tables)} 张表，至 {END_DATE or '无限期'}")
            continue
        try:
            action = grant_readonly(token, server, uid)
            if action == "created":
                created += 1
                print(f"  [新建] [{i}/{len(user_ids)}] {uid}")
            elif action == "updated":
                updated += 1
                print(f"  [修改] [{i}/{len(user_ids)}] {uid}  原额度为 -1（不限量），已重新配置")
            else:  # skipped
                skipped += 1
                print(f"  [跳过] [{i}/{len(user_ids)}] {uid}  已有 SDK 配置，未改动")
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
        print(f"\n=== 完成：新建 {created}，修改 {updated}（原额度 -1），跳过 {skipped}（已有配置），失败 {fail} ===")
        if failures:
            print("失败明细：")
            for uid, reason in failures:
                print(f"  {uid}: {reason}")


if __name__ == "__main__":
    main()
