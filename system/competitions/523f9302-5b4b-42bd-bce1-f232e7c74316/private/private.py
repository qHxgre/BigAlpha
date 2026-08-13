"""端到端模型赛道私榜的单批次入口。"""
from __future__ import annotations

import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
API_SERVER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver"))
# 私榜只加载 private 目录内的比赛模块；评测框架从 alphathonapiserver 加载。
# 不把 public 加入 sys.path，避免私榜配置和公榜配置互相影响。
for path in reversed((HERE, API_SERVER)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from private_judge import PRIVATE_FILES_DIR, PrivateJudge
from runner import MemoryLimitedUserRunner

# BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BATCH_ID = "20260812_180115"
RESUME = True
# 同一 BATCH_ID 续跑时，已有完整 result.json 的 submission 会自动跳过，
# 只评测当前 prepared 输入包中尚未产出完整结果的 submission。
# 如需强制重跑个别 submission，再把 ID 填入此列表。
RERUN_SUBMISSION_IDS: list[str] = []
# 单机一张 GPU；端到端模型默认串行，避免两个模型同时抢占显存。
MAX_WORKERS = 1
# 512 GiB 主机、单 worker：给用户子进程 400 GiB 虚拟地址空间，
# 余下约 112 GiB 留给 judge、系统、CUDA 驱动及其它常驻进程。
RUNNER_MEM_LIMIT_GIB = 400
MemoryLimitedUserRunner.MEM_LIMIT = RUNNER_MEM_LIMIT_GIB * 1024**3


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
