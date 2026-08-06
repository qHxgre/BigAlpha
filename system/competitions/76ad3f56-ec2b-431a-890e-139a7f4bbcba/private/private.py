"""私榜评测入口：使用当前日期作为评估结束日并执行一次完整批次。"""
from __future__ import annotations

import os
import sys
from datetime import datetime

# private.py 作为脚本直接运行；与 public.py 一样，把评测框架目录和比赛目录加入
# sys.path，保证既能 import judge.*，也能 import 同目录的私榜模块。
paths = [
    "/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver",
    os.path.dirname(os.path.abspath(__file__)),
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from private_judge import PRIVATE_FILES_DIR, PrivateJudge


# ---------------------------------------------------------------------------
# 私榜运行配置：所有运行参数都在本文件中显式设置，不读取环境变量。
# ---------------------------------------------------------------------------

# 新批次默认使用启动时间创建目录。需要断点续跑时，将这里改为原批次目录名，
# 例如 "20260806_172301"，并把 RESUME 改为 True。
BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# False：创建新批次；True：断点续跑 BATCH_ID 指定的已有批次。
RESUME = False

# 同时运行的 submission 数量，根据机器 CPU、内存和数据查询承载能力调整。
MAX_WORKERS = 5


class Judge(PrivateJudge):
    """私榜评测：读取已固化的提交，并生成待人工发布的评分批次。"""

    # 与 prepare_submissions.py 的默认输出目录保持一致：
    # system/files/<competition_id>/private/prepared。
    INPUT_DIR = os.path.join(PRIVATE_FILES_DIR, "prepared")

    def __init__(self) -> None:
        # 每次启动时取当天作为评估结束日，避免模块长期驻留时日期过期。
        self.DATE_END = datetime.now().strftime("%Y-%m-%d 23:59:59")
        super().__init__(
            input_dir=self.INPUT_DIR,
            batch_id=BATCH_ID,
            resume=RESUME,
            max_workers=MAX_WORKERS,
        )

    def run(self) -> None:
        self.run_once()


if __name__ == "__main__":
    Judge().run()
