"""按接口统计每个 user_id 的累计提交次数，只落一个文件供 reward_coins.py 使用。

提交里程碑是**每天**滚动赠送——用户哪天攒够提交数就在哪天发，且每个里程碑每人只发
一次。本脚本只负责把"事实"查清楚并存下来，不做里程碑筛选、不做剔重（那些交给
reward_coins.py）：

    走 AlphathonClient.list_submissions 拉每个赛道的全部提交，按提交人 user_id 各自
    累计计数，把三条赛道合并落成一个 submission_counts.json（每天覆盖）。

提交次数口径（已与运营确认）：
    - 数据源 = 该赛道全部 submission 记录（走接口，不依赖榜单快照）。
    - 计数 = 按提交人 user_id 各自累计，即「谁点的提交算谁的」。
      团队不共享次数：同队里没亲自提交的成员，其提交次数为 0。

输出（common.paths.REWARD_COINS_DIR 下，就一个文件）：
    submission_counts.json  形如 {competition_id: {user_id: 提交次数}}

对接 reward_coins.py：
    reward_coins.py 的里程碑任务用 counts_file 指向这份文件、用 threshold 定阈值
    （首次=1 / 累计第5次=5 / 累计第10次=10），自行筛人并生成各赛道的发币 CSV。
    每人只发一次由 reward_coins.load_history 按 label 剔重保证（本脚本不管剔重）。

用法：
    python3 select_submission_milestones.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient
from common.paths import REWARD_COINS_DIR

# ===== 配置 =================================================================
# 赛道：cid -> 中文名（只用于打印）。三条赛道统一查、合并落一个文件。
CID_FACTOR = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"   # 赛道一 · AI 因子挖掘
CID_E2E = "523f9302-5b4b-42bd-bce1-f232e7c74316"       # 赛道二 · 端到端 AI 量化模型
CID_OPEN = "63dd885c-2488-4efd-9c61-9e3a536f172c"      # 赛道三 · AI 开放创新

CIDS: dict[str, str] = {
    CID_FACTOR: "赛道一·AI因子挖掘",
    CID_E2E: "赛道二·端到端AI量化模型",
    CID_OPEN: "赛道三·AI开放创新",
}

# 唯一的输出文件（每天覆盖）；reward_coins.py 用 counts_file 指向它。
OUT_FILE = REWARD_COINS_DIR / "submission_counts.json"
# ===========================================================================


def count_submissions(client: AlphathonClient, cid: str) -> dict[str, int]:
    """走接口拉该赛道全部提交，按提交人 user_id 各自累计计数。

    口径：谁点的提交算谁的，团队不共享次数（同队没亲自提交的成员计数为 0）。
    """
    counts: dict[str, int] = defaultdict(int)
    for sub in client.list_submissions(cid):
        uid = str(sub.get("user_id") or "").strip()
        if uid:
            counts[uid] += 1
    # 按次数降序、user_id 次之排一下，文件更易人工核对
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> None:
    REWARD_COINS_DIR.mkdir(parents=True, exist_ok=True)
    client = AlphathonClient()

    all_counts: dict[str, dict[str, int]] = {}
    print("=== 各赛道提交次数（走接口 list_submissions，按提交人 user_id 计数）===")
    for cid, name in CIDS.items():
        counts = count_submissions(client, cid)
        all_counts[cid] = counts
        total_subs = sum(counts.values())
        print(f"  [{name}] 提交人 {len(counts)} 人，累计提交 {total_subs} 次")

    OUT_FILE.write_text(
        json.dumps(all_counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写出: {OUT_FILE}")
    print("接下来在 reward_coins.py 里用里程碑任务（counts_file + threshold）生成发币 CSV。")


if __name__ == "__main__":
    main()
