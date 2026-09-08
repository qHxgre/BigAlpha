"""批量查询多个 submission（提交记录）对应的信息。

给一批 submission id，直接查出每条提交的比赛、提交人、公榜/私榜分数等信息，
不需要预先知道它们属于哪场比赛。提交人 user_id 会按比赛翻译成姓名。

走 AlphathonClient.get_submissions_by_ids（GET /submissions?constraints={"id__in":[...]}），
需要 competition_manage 权限（cptjudge token 即可），否则只能查到自己的提交。

把要查的 id 填到下面的 SUBMISSION_IDS 里，直接跑就行。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient

# ===== 配置：把要查的 submission id 填这里 ==================================
SUBMISSION_IDS = [
    # "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
]
# ===========================================================================


def build_name_maps(client: AlphathonClient, competition_ids: set[str]) -> dict[str, dict[str, str]]:
    """对涉及到的每场比赛拉参赛者，返回 {competition_id: {user_id: 姓名}}。"""
    maps: dict[str, dict[str, str]] = {}
    for cid in competition_ids:
        name_map: dict[str, str] = {}
        for u in client.list_users(cid):
            user_id = str(u.get("user_id"))
            name_map[user_id] = (u.get("data") or {}).get("name") or user_id
        maps[cid] = name_map
    return maps


def main() -> None:
    if not SUBMISSION_IDS:
        print("请先把要查的 submission id 填到 SUBMISSION_IDS 里。")
        return

    client = AlphathonClient()
    subs = client.get_submissions_by_ids(SUBMISSION_IDS, order_by=["-created_at"])

    competition_ids = {str(s.get("competition_id")) for s in subs}
    name_maps = build_name_maps(client, competition_ids)

    print(f"=== 查询 {len(SUBMISSION_IDS)} 个 submission，命中 {len(subs)} 个 ===")
    for s in subs:
        cid = str(s.get("competition_id"))
        uid = str(s.get("user_id"))
        name = name_maps.get(cid, {}).get(uid, uid)
        pub = s.get("public_score") if s.get("public_score") is not None else "-"
        pri = s.get("private_score") if s.get("private_score") is not None else "-"
        sel = "✓" if s.get("selected_for_private") else " "
        print(f"  [{sel}] {s.get('id')}  {name}  公榜:{pub}  私榜:{pri}  比赛:{cid}")

    found = {str(s.get("id")) for s in subs}
    missing = [sid for sid in SUBMISSION_IDS if sid not in found]
    if missing:
        print(f"\n未查到 {len(missing)} 个 id（不存在或无权限）:")
        for sid in missing:
            print(f"  {sid}")


if __name__ == "__main__":
    main()
