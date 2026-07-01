from __future__ import annotations


from _client import AlphathonClient


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
    import json
    from pathlib import Path

    client = AlphathonClient()

    out_dir = Path(__file__).parent / "files" / "participants"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_user_ids: list[str] = []

    # 按赛道分开落一份 user_id_<competition_id>.json（去重）
    for cid in competition_ids:
        user_ids = collect_user_ids(client, cid)
        uniq = list(dict.fromkeys(user_ids))
        all_user_ids.extend(uniq)
        path = out_dir / f"user_id_{cid}.json"
        path.write_text(json.dumps(uniq, ensure_ascii=False), encoding="utf-8")
        print(f"  {cid}: {len(uniq)} 位参赛者 -> {path.name}")

    # 合并去重的全量列表
    all_uniq = list(dict.fromkeys(all_user_ids))
    (out_dir / "user_id.json").write_text(
        json.dumps(all_uniq, ensure_ascii=False), encoding="utf-8"
    )
    print(f"共拉到 {len(all_uniq)} 位参赛者（去重后）-> user_id.json")
