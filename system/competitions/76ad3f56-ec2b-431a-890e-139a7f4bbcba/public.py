from __future__ import annotations

import os
import sys
# public.py 作为脚本直接运行；把评测框架目录与本比赛目录都加进 sys.path，
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
    """公榜评测：用公开数据集与时间区间跑因子挖掘。"""

    mode = "public"

    # 公榜数据集与数据时间区间
    # DATASETS：{逻辑名: 物理表名}，用户代码用逻辑名（"bar1m"/"financial"）取物理表名拼 SQL。
    DATASETS = {
        "bar1m": "bigalpha_2026_stock_bar1m_test",
        "financial": "bigalpha_2026_financial_test",
    }
    DATE_START = "2025-03-01 00:00:00"
    DATE_END = "2025-11-30 23:59:59"

    # 自适应评估间隔：Elastic Net 计算量随全局因子数增长，间隔按上一轮实测耗时自调。
    #     t_next = max(k * t_last_run, t_min)，k = 1.5，t_min = 1 小时。
    # 比赛初期因子少、间隔短；后期因子池扩大、间隔自动拉长，无需人工干预。
    adaptive_interval = True
    tick_safety_factor = 1.5
    tick_min_interval = 60

    # 并行跑用户代码的线程数上限（同时评测的提交数）。默认继承基类的 5，这里按需覆盖。
    max_workers = 5

    # 只跑部分提交（调试 / 复测用）：填上后整条流水线（跑用户代码 + 排名 + 回归 + 汇总）
    # 只处理这几个 id，留空则跑全量。MAX_PAGES 可选，限制拉取页数。
    # SUBMISSION_IDS = [
    #     "4ec02a39-de56-4aa7-8c19-b195f212b3cd",
    # ]
    # MAX_PAGES = 1


if __name__ == "__main__":
    Judge().run()
