from __future__ import annotations

import os
import sys
# private.py 作为脚本直接运行；把评测框架目录与本比赛目录都加进 sys.path，
# 保证既能 import judge.*，也能 import 同目录的 base/sfa/regression/final_scoring 等模块。
paths = [
    '/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver',
    os.path.dirname(os.path.abspath(__file__)),
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from bigalpha_judge import BigAlphaJudge


class Judge(BigAlphaJudge):
    """私榜评测：与公榜逻辑完全一致，只换数据集与时间区间（产物文件名自动带 -private 后缀隔离）。"""

    mode = "private"

    # TODO: 换成私榜数据集与对应的数据时间区间
    DATASET = "bigalpha_factor_2026_stock_bar1m_private"
    DATE_START = "2025-01-01 00:00:00"
    DATE_END = "2025-12-31 23:59:59"


if __name__ == "__main__":
    Judge().run()
