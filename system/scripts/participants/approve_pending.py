"""把比赛里"还没审核通过"的参赛者批量改成"通过审核并加入空间"。

报名记录的状态见 alphathonapiserver.constants.UserStatus：
    pending               待审核
    approved              通过审核
    approved_join_space   通过审核并加入空间（审批时会把用户拉进比赛空间）
    rejected              拒绝

做法：对每场比赛拉全部报名记录，把状态还不是目标状态的（默认 pending + approved，
不动 rejected）逐条调用 POST /users/{报名记录id} 改成 approved_join_space。
服务端在该状态下会把人加入空间，并发送审核通过通知 + 微信消息。

注意：这里用的是「报名记录 id」(u["id"])，不是 participants.py 里保存的账号
user_id (u["user_id"])，两者不是一回事，所以不能直接读那个 json，得按比赛查。

先用 DRY_RUN=True 预览名单，确认无误后改成 False 再跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient

# ===== 配置：改这里就行 =====================================================
COMPETITION_IDS = [
    "76ad3f56-ec2b-431a-890e-139a7f4bbcba",
    "523f9302-5b4b-42bd-bce1-f232e7c74316",
    "63dd885c-2488-4efd-9c61-9e3a536f172c",
]
TARGET_STATUS = "approved_join_space"      # 目标状态：通过并加入空间
SOURCE_STATUSES = {"pending", "approved"}  # 要处理的源状态（不含 rejected）
DRY_RUN = False                             # True 只预览；确认后改 False 真正写入
# ===========================================================================


def approve_competition(client: AlphathonClient, competition_id: str) -> None:
    """把一场比赛里源状态命中的参赛者改成 TARGET_STATUS。"""
    users = client.list_users(competition_id, order_by=["created_at"])
    todo = [u for u in users if u.get("status") in SOURCE_STATUSES]

    print(f"\n=== 比赛 {competition_id}  待处理 {len(todo)} 人 ===")
    for u in todo:
        record_id = str(u.get("id"))
        name = (u.get("data") or {}).get("name") or str(u.get("user_id"))
        cur = u.get("status")
        if DRY_RUN:
            print(f"  [dry-run] {name}  ({record_id})  {cur} -> {TARGET_STATUS}")
            continue
        try:
            client.update_user_status(record_id, TARGET_STATUS)
            print(f"  [ok] {name}  ({record_id})  {cur} -> {TARGET_STATUS}")
        except Exception as e:  # noqa: BLE001 — 单条失败不影响其他人
            print(f"  [失败] {name}  ({record_id})  {cur}: {e}")


if __name__ == "__main__":
    client = AlphathonClient()
    for cid in COMPETITION_IDS:
        approve_competition(client, cid)

    if DRY_RUN:
        print("\n当前为预览模式(DRY_RUN=True)，确认名单无误后把 DRY_RUN 改成 False 再跑。")
