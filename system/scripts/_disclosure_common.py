"""每周公示脚本的共享配置与工具函数。

被 factor_portrait / index_strategy / weekly_disclosure 复用：matplotlib 中文字体、
榜单目录定位、截面标准化、以及两个需求共用的输出目录等常量。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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

_scripts_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
BENCHMARK = "000852.SH"  # 中证 1000

# 云端评测榜单目录；不存在时回退到脚本目录下的 files/leaderboard
REMOTE_LEADERBOARD_BASE = (
    "/home/aiuser/work/workspace/BigAlpha/system/files"
    "/{competition_id}/leaderboard"
)
LOCAL_LEADERBOARD_FALLBACK = Path(_scripts_dir) / "files" / "leaderboard"

# 两个需求共用同一输出目录，图片 / CSV / 合并 Markdown 都落在这里
OUTPUT_DIR = Path(_scripts_dir) / "files" / "weekly_disclosure"


def resolve_leaderboard_dir(competition_id: str) -> str:
    """定位榜单目录：优先云端路径，缺失时回退到本地 files/leaderboard。"""
    remote = REMOTE_LEADERBOARD_BASE.format(competition_id=competition_id)
    if os.path.isdir(remote):
        return remote
    print(
        f"  [提示] 云端目录不存在，回退到本地目录: {LOCAL_LEADERBOARD_FALLBACK}",
        file=sys.stderr,
    )
    return str(LOCAL_LEADERBOARD_FALLBACK)


def zscore(s: pd.Series) -> pd.Series:
    """截面标准化，std 为 0 或非有限时返回全 0。"""
    std = s.std()
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std
