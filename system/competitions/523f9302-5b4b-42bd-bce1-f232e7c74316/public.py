from __future__ import annotations

import os
import sys
# public.py 作为脚本直接运行；把评测框架目录与本比赛目录都加进 sys.path，
# 保证既能 import judge.*，也能 import 同目录的 base/score/final_scoring 等模块。
paths = [
    '/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver',
    os.path.dirname(os.path.abspath(__file__)),
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from endtoend_judge import EndToEndJudge


class Judge(EndToEndJudge):
    """公榜评测：用公开数据集与验证集区间跑模型推理打分。"""

    mode = "public"

    # 公榜推理所用数据集与验证集时间区间。
    # DATASETS：{逻辑名: 物理表名}，用户代码用逻辑名（"bar1m"/"bar5m"/"bar15m"/"bar30m"/"snapshot"）
    # 取物理表名拼 SQL；公榜验证集为 2025 全年（训练集 2019~2023 由用户代码内部写死）。
    DATASETS = {
        "bar1m": "bigalpha_2026_stock_bar1m_test",
        "bar5m": "bigalpha_2026_stock_bar5m_test",
        "bar15m": "bigalpha_2026_stock_bar15m_test",
        "bar30m": "bigalpha_2026_stock_bar30m_test",
    }
    DATE_START = "2025-03-01 00:00:00"
    DATE_END = "2025-11-30 23:59:59"


if __name__ == "__main__":
    Judge().run()
