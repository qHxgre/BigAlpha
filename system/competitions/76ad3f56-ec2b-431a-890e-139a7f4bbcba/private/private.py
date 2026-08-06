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


class Judge(PrivateJudge):
    """私榜评测：读取已固化的提交，并生成待人工发布的评分批次。"""

    # 与 prepare_submissions.py 的默认输出目录保持一致：
    # system/files/<competition_id>/private/prepared。
    INPUT_DIR = os.path.join(PRIVATE_FILES_DIR, "prepared")

    def __init__(self) -> None:
        # 每次启动时取当天作为评估结束日，避免模块长期驻留时日期过期。
        self.DATE_END = datetime.now().strftime("%Y-%m-%d 23:59:59")
        super().__init__(input_dir=self.INPUT_DIR)

    def run(self) -> None:
        self.run_once()


if __name__ == "__main__":
    Judge().run()
