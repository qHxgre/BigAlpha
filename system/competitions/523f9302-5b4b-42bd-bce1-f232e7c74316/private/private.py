"""端到端模型赛道私榜的单批次入口。"""
from __future__ import annotations

import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.abspath(os.path.join(HERE, "..", "public"))
API_SERVER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver"))
# private 模块优先，public 次之（复用其 MemoryLimitedUserRunner），评测框架最后。
# 使用明确顺序可避免误导入 alphathonapiserver 中同名的 runner 模块。
for path in reversed((HERE, PUBLIC_DIR, API_SERVER)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from private_judge import PRIVATE_FILES_DIR, PrivateJudge

# BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BATCH_ID = "20260812_180115"
RESUME = True
# RERUN_SUBMISSION_IDS: list[str] = []
RERUN_SUBMISSION_IDS: list[str] = [
    "3f0339e2-8b62-4016-8fb3-253ab184517e"
]
# 单机一张 GPU；端到端模型默认串行，避免两个模型同时抢占显存。
MAX_WORKERS = 1


class Judge(PrivateJudge):
    INPUT_DIR = os.path.join(PRIVATE_FILES_DIR, "prepared")
    DATASETS = {
        "bar1m": "bigalpha_2026_stock_bar1m_private",
        "bar5m": "bigalpha_2026_stock_bar5m_private",
        "bar15m": "bigalpha_2026_stock_bar15m_private",
        "bar30m": "bigalpha_2026_stock_bar30m_private",
    }
    DATE_START = "2025-01-01 00:00:00"

    def __init__(self) -> None:
        # self.DATE_END = datetime.now().strftime("%Y-%m-%d 23:59:59")

        self.DATE_START = "2025-12-01 00:00:00"
        self.DATE_END = "2026-08-10 23:59:59"
        super().__init__(self.INPUT_DIR, batch_id=BATCH_ID, resume=RESUME,
                         rerun_submission_ids=RERUN_SUBMISSION_IDS,
                         max_workers=MAX_WORKERS)


if __name__ == "__main__":
    Judge().run_once()
