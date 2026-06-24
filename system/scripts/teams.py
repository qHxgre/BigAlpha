"""统计每个队伍的人员信息。

队伍表里成员存的是 user_id，要显示人名得先拉参赛者列表，
做一个 user_id -> 姓名 的映射，再去翻译每个队伍的队长/队员。

用法:
    python teams.py <比赛ID> [<比赛ID> ...]
    python teams.py --out teams.json <比赛ID>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from _client import AlphathonClient


def build_name_map(client: AlphathonClient, competition_id: str) -> dict[str, str]:
    """拉这场比赛的参赛者，返回 {user_id: 姓名}。查不到名字就用 user_id 兜底。"""
    name_map = {}
    for u in client.list_users(competition_id):
        user_id = str(u.get("user_id"))
        name = (u.get("data") or {}).get("name") or user_id
        name_map[user_id] = name
    return name_map


def collect_teams(client: AlphathonClient, competition_id: str) -> dict:
    """拉这场比赛的所有队伍，把队长/队员的 user_id 翻译成姓名。"""
    name_map = build_name_map(client, competition_id)
    teams_raw = client.list_teams(competition_id, order_by=["-created_at"])

    teams = []
    for t in teams_raw:
        creator_id = str(t.get("creator"))
        member_ids = [str(m) for m in (t.get("members") or [])]
        pending_ids = [str(m) for m in (t.get("pending_users") or [])]

        teams.append(
            {
                "team_id": t.get("id"),
                "name": t.get("name"),
                "captain": name_map.get(creator_id, creator_id),
                "members": [name_map.get(m, m) for m in member_ids],
                # 队长 + 队员，正式在队人数
                "member_count": 1 + len(member_ids),
                "pending": [name_map.get(m, m) for m in pending_ids],
            }
        )

    return {
        "competition_id": competition_id,
        "team_count": len(teams),
        "teams": teams,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="统计每个队伍的人员信息")
    parser.add_argument("competition_ids", nargs="+", help="一个或多个比赛ID")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径")
    parser.add_argument("--base-url", default=None, help="覆盖 ALPHATHON_API_BASE_URL")
    parser.add_argument("--token", default=None, help="覆盖 ALPHATHON_API_TOKEN")
    args = parser.parse_args(argv)

    client = AlphathonClient(base_url=args.base_url, token=args.token)

    results = []
    for cid in args.competition_ids:
        result = collect_teams(client, cid)
        results.append(result)

        print(f"\n=== 比赛 {cid}：共 {result['team_count']} 支队伍 ===")
        for team in result["teams"]:
            members = "、".join(team["members"]) or "(无队员)"
            print(f"  [{team['member_count']}人] {team['name']}  队长: {team['captain']}  队员: {members}")

    out_path = args.out or f"teams_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
