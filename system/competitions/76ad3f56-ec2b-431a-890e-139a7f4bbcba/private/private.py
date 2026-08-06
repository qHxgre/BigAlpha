"""私榜评测入口：使用当前日期作为评估结束日并执行一次完整批次。"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
API_SERVER = os.getenv(
    "ALPHATHON_API_SERVER_DIR",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver")),
)
for path in (API_SERVER, HERE):
    if path not in sys.path:
        sys.path.append(path)

from private_judge import PrivateJudge


# 保留原来的类名，避免外部脚本通过 ``from private import Judge`` 使用时失效。
Judge = PrivateJudge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估已经由 prepare_submissions.py 固化的私榜提交")
    parser.add_argument("--input", required=True, help="prepared/<batch_id> 输入包目录")
    args = parser.parse_args()
    Judge.DATE_END = datetime.now().strftime("%Y-%m-%d 23:59:59")
    Judge(input_dir=args.input).run_once()
