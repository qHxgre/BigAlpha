"""拉取各赛道参赛者的 user_id 名单，按赛道 + 全量分别落 JSON。

输出到 common.paths.PARTICIPANTS_DIR（system/files/scripts/participants）：
    user_id_<competition_id>.json   各赛道去重后的报名 user_id
    user_id.json                    全部赛道合并去重后的 user_id
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient
from common.ids import dedup_keep_order
from common.paths import PARTICIPANTS_DIR


def collect_user_ids(client: AlphathonClient, competition_id: str) -> list[str]:
    """拉这场比赛的参赛者，只返回 user_id 列表。"""
    users = client.list_users(competition_id, order_by=["-created_at"])
    return [u.get("user_id") for u in users if u.get("user_id")]


competition_ids = [
    '76ad3f56-ec2b-431a-890e-139a7f4bbcba',     # AI 因子挖掘
    '523f9302-5b4b-42bd-bce1-f232e7c74316',     # 端到端大模型
    '63dd885c-2488-4efd-9c61-9e3a536f172c'      # AI 开放创新
]


if __name__ == "__main__":
    client = AlphathonClient()

    PARTICIPANTS_DIR.mkdir(parents=True, exist_ok=True)

    all_user_ids: list[str] = []

    # 按赛道分开落一份 user_id_<competition_id>.json（去重）
    for cid in competition_ids:
        uniq = dedup_keep_order(collect_user_ids(client, cid))
        all_user_ids.extend(uniq)
        path = PARTICIPANTS_DIR / f"user_id_{cid}.json"
        path.write_text(json.dumps(uniq, ensure_ascii=False), encoding="utf-8")
        print(f"  {cid}: {len(uniq)} 位参赛者 -> {path.name}")

    # 合并去重的全量列表
    all_uniq = dedup_keep_order(all_user_ids)
    (PARTICIPANTS_DIR / "user_id.json").write_text(
        json.dumps(all_uniq, ensure_ascii=False), encoding="utf-8"
    )
    print(f"共拉到 {len(all_uniq)} 位参赛者（去重后）-> user_id.json")
