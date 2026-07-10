"""数据文件路径的统一入口。

约定：脚本产出的数据一律落在 <repo>/system/files/scripts/ 下，按用途分子目录。
路径以本模块自身位置反推 repo 根，因此本地（/Users/...）与云端（/home/aiuser/...）
两套仓库根目录都能自动适配，无需硬编码。

云端评测系统产出的「榜单输入」是另一套独立目录（system/files/{competition_id}/leaderboard），
只读、不属于 scripts 数据，由 REMOTE_LEADERBOARD_BASE 指向；它同时充当「是否在云端」的探测信号。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# common/ -> scripts/ -> system/ ；数据根固定在 system/files/scripts。
_SYSTEM_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = _SYSTEM_DIR / "files" / "scripts"

# 各用途子目录（脚本产出/读取的本地数据都在这里）
PARTICIPANTS_DIR = DATA_ROOT / "participants"
REWARD_COINS_DIR = DATA_ROOT / "reward_coins"
DAILY_REPORTS_DIR = DATA_ROOT / "daily_reports"
LEADERBOARD_CRAWL_DIR = DATA_ROOT / "leaderboard_crawl"
SQL_DIR = DATA_ROOT / "sql"

# 每周公示的输出目录（OUTPUT_DIR 为兼容旧名的别名）
WEEKLY_DISCLOSURE_DIR = DATA_ROOT / "weekly_disclosure"
OUTPUT_DIR = WEEKLY_DISCLOSURE_DIR

# 云端评测榜单目录（只读输入，由评测系统产出）；不在云端时回退到本地 DATA_ROOT/leaderboard。
REMOTE_LEADERBOARD_BASE = (
    "/home/aiuser/work/workspace/BigAlpha/system/files"
    "/{competition_id}/leaderboard"
)
LOCAL_LEADERBOARD_FALLBACK = DATA_ROOT / "leaderboard"


def resolve_leaderboard_dir(competition_id: str) -> str:
    """定位榜单目录：优先云端路径，缺失时回退到本地 DATA_ROOT/leaderboard。"""
    remote = REMOTE_LEADERBOARD_BASE.format(competition_id=competition_id)
    if os.path.isdir(remote):
        return remote
    print(
        f"  [提示] 云端目录不存在，回退到本地目录: {LOCAL_LEADERBOARD_FALLBACK}",
        file=sys.stderr,
    )
    return str(LOCAL_LEADERBOARD_FALLBACK)
