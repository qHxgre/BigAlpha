"""私榜评测入口：解析评估结束日并执行一次完整批次。"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
API_SERVER = os.getenv(
    "ALPHATHON_API_SERVER_DIR",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver")),
)
for path in (API_SERVER, HERE):
    if path not in sys.path:
        sys.path.append(path)

from private_judge import PrivateJudge
from trading_day import resolve_latest_trading_day


# 保留原来的类名，避免外部脚本通过 ``from private import Judge`` 使用时失效。
Judge = PrivateJudge


if __name__ == "__main__":
    Judge.DATE_END = resolve_latest_trading_day(Judge.DATASETS["bar1m"])
    Judge().run_once()
