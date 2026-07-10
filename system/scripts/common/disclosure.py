"""每周公示脚本的共享配置与工具函数。

被 factor_portrait / index_strategy / weekly_disclosure 复用：matplotlib 中文字体、
截面标准化、公用常量。榜单目录定位与输出目录见 common.paths。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 无显示环境（云端）后端，import 后立刻设定
import numpy as np
import pandas as pd

# 让 matplotlib 能渲染中文标签，找不到中文字体时静默退回默认字体
for _cn_font in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]:
    if _cn_font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.sans-serif"] = [_cn_font]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
BENCHMARK = "000852.SH"  # 中证 1000


def zscore(s: pd.Series) -> pd.Series:
    """截面标准化，std 为 0 或非有限时返回全 0。"""
    std = s.std()
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std
