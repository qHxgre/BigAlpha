"""私榜评测入口：使用当前配置执行新批次、断点续跑或指定 submission 重跑。"""
from __future__ import annotations

import os
import sys
from datetime import datetime

# private.py 作为脚本直接运行；与 public.py 一样，把评测框架目录和比赛目录加入
# sys.path，保证既能 import judge.*，也能 import 同目录的私榜模块。
HERE = os.path.dirname(os.path.abspath(__file__))
paths = [
    "/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver")),
    HERE,
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from private_judge import PRIVATE_FILES_DIR, PrivateJudge


# ---------------------------------------------------------------------------
# 私榜运行配置：所有运行参数都在本文件中显式设置，不读取命令行或环境变量。
# ---------------------------------------------------------------------------

# 新批次默认使用启动时间创建目录。
# 断点续跑/指定重跑时，必须改成目标批次目录名，例如 "20260806_172301"。
BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
# BATCH_ID = "20260811_102014"

# False：创建新批次；True：断点续跑或指定重跑 BATCH_ID 对应的已有批次。
RESUME = False

# 仅在 RESUME = True 时生效。程序会删除这些 submission 的旧运行目录并强制重跑；
# 其他 submission 复用原 result.json。重跑完成后会基于全批次最新结果重新计算
# leaderboard_sfa.csv、回归 B 分、leaderboard_final.csv 和 pending_publish.jsonl。
RERUN_SUBMISSION_IDS: list[str] = [
    "993c4dbd-5035-4c6c-8543-2b20cbfd51e2", 
    "4e504ccc-765b-43b0-90f4-2df2c58c83c8", 
    "b8f6af17-daf0-4c44-ad48-9bd6ef3d7313", 
    "dad2b2fd-8e90-430f-ac6e-2f84498e91e5", 
    "ded7a92a-a8a2-43b8-bdc9-8ffa260a9781"
    "bc2d7876-1e07-4846-960d-b3c456b04e8e",
    "e1267c29-aab4-449b-80f1-8dd7fa28699a"
]

# 同时运行的 submission 数量，根据机器 CPU、内存和数据查询承载能力调整。
MAX_WORKERS = 5


class Judge(PrivateJudge):
    """私榜评测：读取已固化的提交，并生成待人工发布的评分批次。"""

    INPUT_DIR = os.path.join(PRIVATE_FILES_DIR, "prepared")

    def __init__(self) -> None:
        self.DATE_START = "2025-12-01 00:00:00"
        self.DATE_END = "2026-08-10 23:59:59"
        super().__init__(
            input_dir=self.INPUT_DIR,
            batch_id=BATCH_ID,
            resume=RESUME,
            rerun_submission_ids=RERUN_SUBMISSION_IDS,
            max_workers=MAX_WORKERS,
        )

    def run(self) -> None:
        self.run_once()


if __name__ == "__main__":
    Judge().run()
