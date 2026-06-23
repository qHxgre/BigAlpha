from __future__ import annotations

import os
import sys
# private.py 作为脚本直接运行；把评测框架目录与本比赛目录都加进 sys.path，
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
    """私榜评测：与公榜逻辑完全一致，只换数据集与时间区间（产物文件名自动带 _private 后缀隔离）。

    注意：私榜阶段平台会用提交的训练脚本在隔离环境中【从零重训】候选模型，再用重训权重在私榜
    验证区间推理打分。本评测器负责的是「拿到推理产出的分数后做评分」这一段，与公榜完全一致；
    重训环节由平台另行编排，这里只需把 DATASETS / 日期区间换成私榜版本。
    """

    mode = "private"

    # TODO: 换成私榜各表的物理表名与对应的验证集时间区间（私榜含样本外数据，不公开）。
    # DATASETS：{逻辑名: 物理表名}，逻辑名须与公榜一致，物理表名换成私榜带后缀的版本，
    # 使用户代码无需改动即可在私榜运行。
    DATASETS = {
        "bar1m": "bigalpha_2026_stock_bar1m_private",
        "bar5m": "bigalpha_2026_stock_bar5m_private",
        "bar15m": "bigalpha_2026_stock_bar15m_private",
        "bar30m": "bigalpha_2026_stock_bar30m_private",
        "snapshot": "bigalpha_2026_stock_snapshot_private",
    }
    DATE_START = "2025-01-01 00:00:00"
    DATE_END = "2025-12-31 23:59:59"


if __name__ == "__main__":
    Judge().run()
