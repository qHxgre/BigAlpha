"""查询指定比赛中公榜没有分数（即运行失败）的 submission。

走 AlphathonClient.list_submissions（GET /submissions?competition_id=...），
把 public_score 为空的提交视为运行失败，导出它们的 submission_id 到
fail_submission.json，方便后续重跑或排查。

需要 competition_manage 权限（cptjudge token 即可），否则只能看到自己的提交。

把要查的比赛填到下面的 COMPETITION_IDS 里，直接跑就行。
"""

from __future__ import annotations

import json
import os

from _client import AlphathonClient

# ===== 配置：把要查的 competition id 填这里 ================================
COMPETITION_IDS = [
    # "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
]
# ===========================================================================

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fail_submission.json")


def collect_failed(client: AlphathonClient, competition_id: str) -> list[dict]:
    """拉这场比赛的全部提交，返回公榜没有分数（运行失败）的记录。"""
    subs = client.list_submissions(competition_id, order_by=["-created_at"])
    failed = [s for s in subs if s.get("public_score") is None]
    print(f"  比赛 {competition_id}: 共 {len(subs)} 条提交，失败（公榜无分数）{len(failed)} 条")
    return failed


def main() -> None:
    if not COMPETITION_IDS:
        print("请先把要查的 competition id 填到 COMPETITION_IDS 里。")
        return

    client = AlphathonClient()

    fail_ids: list[str] = []
    for cid in COMPETITION_IDS:
        for s in collect_failed(client, cid):
            fail_ids.append(str(s.get("id")))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(fail_ids, f, ensure_ascii=False, indent=2)

    print(f"\n共 {len(fail_ids)} 个失败 submission_id，已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
