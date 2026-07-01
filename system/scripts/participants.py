from __future__ import annotations


from collections import Counter

from _client import AlphathonClient


def collect_participants(client: AlphathonClient, competition_id: str) -> dict:
    """拉这场比赛的参赛者，按报名状态统计。"""
    users = client.list_users(competition_id, order_by=["-created_at"])
    status_counts = Counter(u.get("status") for u in users)

    participants = [
        {
            "user_id": u.get("user_id"),
            "name": (u.get("data") or {}).get("name"),
            "status": u.get("status"),
            "created_at": u.get("created_at"),
        }
        for u in users
    ]

    return {
        "competition_id": competition_id,
        "total": len(users),
        "status_counts": dict(status_counts),
        "participants": participants,
    }

def fetch_participants_df(client: AlphathonClient, competition_ids: list[str]):
    """查询多场比赛的所有报名列表，合并成一个 pandas DataFrame。

    每行是一条报名记录，带上所属 competition_id，方便后续按比赛分组解析。
    """
    import pandas as pd

    rows: list[dict] = []
    for cid in competition_ids:
        result = collect_participants(client, cid)
        for p in result["participants"]:
            rows.append({"competition_id": cid, **p})

    return pd.DataFrame(rows)


competition_ids = [
    '76ad3f56-ec2b-431a-890e-139a7f4bbcba',     # AI 因子挖掘
    '523f9302-5b4b-42bd-bce1-f232e7c74316',     # 端到端大模型
    '63dd885c-2488-4efd-9c61-9e3a536f172c'      # AI 开放创新
]


if __name__ == "__main__":
    import json
    from pathlib import Path

    client = AlphathonClient()
    df = fetch_participants_df(client, competition_ids)
    print(f"共拉到 {len(df)} 条报名记录")
    print(df.groupby(["competition_id", "status"]).size())

    out_dir = Path(__file__).parent

    # 1) 合并去重的全量列表（老产物，保持兼容）
    user_id = df["user_id"].unique().tolist()
    print(f"共拉到 {len(user_id)} 位参赛者")
    (out_dir / "user_id.json").write_text(json.dumps(user_id, ensure_ascii=False))

    # 2) 按赛道分开落一份 participants_<competition_id>.json
    #    每条含 user_id / status，供 grant_coins.py 按赛道取较大金额、按状态过滤。
    for cid, sub in df.groupby("competition_id"):
        records = sub[["user_id", "status"]].to_dict("records")
        path = out_dir / f"participants_{cid}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"  {cid}: {len(records)} 条 -> {path.name}")
