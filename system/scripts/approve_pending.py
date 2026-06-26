"""把比赛里"待审核"的参赛者批量改成"通过审核并加入空间"。

报名记录的状态见 alphathonapiserver.constants.UserStatus：
    pending               待审核
    approved              通过审核
    approved_join_space   通过审核并加入空间（审批时会把用户拉进比赛空间）
    rejected              拒绝

本脚本默认只处理 pending 的记录，逐条调用 POST /users/{user_id} 把状态
改成 approved_join_space。服务端在该状态下会调用 join_space 把人加入空间，
并发送审核通过通知 + 微信消息。

默认是 dry-run（只打印将要处理的人），加 --apply 才真正写入。

用法:
    python approve_pending.py <比赛ID> [<比赛ID> ...]            # 预览
    python approve_pending.py <比赛ID> --apply                  # 执行
    python approve_pending.py <比赛ID> --status approved --apply # 改成只通过、不加空间
"""

from __future__ import annotations

import argparse
import sys

from _client import AlphathonClient

PENDING = "pending"
TARGET_DEFAULT = "approved_join_space"
VALID_TARGETS = ("approved", "approved_join_space")


def collect_pending(client: AlphathonClient, competition_id: str) -> list[dict]:
    """拉这场比赛里所有 pending 的报名记录。"""
    users = client.list_users(
        competition_id,
        constraints={"status": PENDING},
        order_by=["created_at"],
    )
    return users


def approve_competition(
    client: AlphathonClient,
    competition_id: str,
    *,
    target_status: str,
    apply: bool,
) -> dict:
    """把一场比赛里 pending 的参赛者改成 target_status。

    apply=False 时只预览，不写入。返回处理结果汇总。
    """
    pending = collect_pending(client, competition_id)
    results = {"competition_id": competition_id, "pending_count": len(pending), "ok": [], "failed": []}

    for u in pending:
        user_id = str(u.get("id"))
        name = (u.get("data") or {}).get("name") or str(u.get("user_id"))
        if not apply:
            print(f"  [dry-run] {name}  ({user_id}) -> {target_status}")
            results["ok"].append(user_id)
            continue
        try:
            client.update_user_status(user_id, target_status)
            print(f"  [ok] {name}  ({user_id}) -> {target_status}")
            results["ok"].append(user_id)
        except Exception as e:  # noqa: BLE001 — 单条失败不影响其他人
            print(f"  [失败] {name}  ({user_id}): {e}")
            results["failed"].append({"user_id": user_id, "error": str(e)})

    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="把比赛里待审核的参赛者批量改成通过并加入空间")
    parser.add_argument("competition_ids", nargs="+", help="一个或多个比赛ID")
    parser.add_argument(
        "--status",
        default=TARGET_DEFAULT,
        choices=VALID_TARGETS,
        help=f"目标状态，默认 {TARGET_DEFAULT}（通过并加入空间）",
    )
    parser.add_argument("--apply", action="store_true", help="真正写入；不加则只预览 (dry-run)")
    parser.add_argument("--base-url", default=None, help="覆盖 ALPHATHON_API_BASE_URL")
    parser.add_argument("--token", default=None, help="覆盖 ALPHATHON_API_TOKEN")
    args = parser.parse_args(argv)

    client = AlphathonClient(base_url=args.base_url, token=args.token)

    mode = "执行" if args.apply else "预览(dry-run)"
    total_ok = total_failed = total_pending = 0
    for cid in args.competition_ids:
        print(f"\n=== 比赛 {cid}  [{mode}] 目标状态: {args.status} ===")
        result = approve_competition(client, cid, target_status=args.status, apply=args.apply)
        if not result["pending_count"]:
            print("  (没有待审核的报名记录)")
        total_pending += result["pending_count"]
        total_ok += len(result["ok"])
        total_failed += len(result["failed"])

    print(f"\n汇总：待审核 {total_pending} 条，成功 {total_ok} 条，失败 {total_failed} 条。")
    if not args.apply and total_pending:
        print("当前为预览模式，加 --apply 才会真正写入。")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

