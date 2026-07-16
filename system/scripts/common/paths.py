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

# 评测系统产出的比赛文件根（与 scripts 数据平级）：system/files/{competition_id}。
FILES_ROOT = _SYSTEM_DIR / "files"

# 默认比赛 ID（放在无重依赖的 paths 里，disclosure 再从此处 re-export）。
DEFAULT_COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"

# 赛道标识：不同赛道的榜单文件名/列名不同（factor 用 leaderboard_reg.csv + factor/model_score，
# e2e 用 leaderboard_score.csv + id/score），需要 track 区分处理逻辑。
TRACK_BY_COMPETITION = {
    "76ad3f56-ec2b-431a-890e-139a7f4bbcba": "factor",   # 赛道一 · AI 因子挖掘
    "523f9302-5b4b-42bd-bce1-f232e7c74316": "e2e",       # 赛道二 · 端到端 AI 量化模型
}


def resolve_track(competition_id: str) -> str:
    """把 competition_id 映射到赛道标识（"factor" / "e2e"），未注册的 ID 直接报错。"""
    track = TRACK_BY_COMPETITION.get(competition_id)
    if track is None:
        raise ValueError(
            f"未知的 competition_id: {competition_id}，请先在 TRACK_BY_COMPETITION 中注册赛道"
        )
    return track


# 各赛道的榜单文件名 / 列名映射（weekly_disclosure 用它读取排名与因子池/得分池）：
#   factor（赛道一）：leaderboard_reg.csv 的 factor/model_score，因子池宽表按因子 id 分列；
#   e2e（赛道二）    ：leaderboard_score.csv 的 id/score，得分池宽表按 submission id 分列。
TRACK_LEADERBOARD_FILES: dict[str, dict[str, str]] = {
    "factor": {
        "rank_csv": "leaderboard_reg.csv",
        "id_col": "factor",
        "score_col": "model_score",
        "raw_pool": "factor_pool_raw.parquet",
        "pool": "factor_pool.parquet",
    },
    "e2e": {
        "rank_csv": "leaderboard_score.csv",
        "id_col": "id",
        "score_col": "score",
        "raw_pool": "score_pool_raw.parquet",
        "pool": "score_pool.parquet",
    },
}


def resolve_leaderboard_files(competition_id: str) -> dict[str, str]:
    """按赛道返回该比赛榜单排名 CSV 与因子池/得分池 parquet 的文件名/列名映射。"""
    return TRACK_LEADERBOARD_FILES[resolve_track(competition_id)]

# 各用途子目录（脚本产出/读取的本地数据都在这里）
PARTICIPANTS_DIR = DATA_ROOT / "participants"
REWARD_COINS_DIR = DATA_ROOT / "reward_coins"
NOTICE_DIR = DATA_ROOT / "notices"
DAILY_REPORTS_DIR = DATA_ROOT / "daily_reports"
LEADERBOARD_CRAWL_DIR = DATA_ROOT / "leaderboard_crawl"
SQL_DIR = DATA_ROOT / "sql"

# 每周公示的输出目录（OUTPUT_DIR 为兼容旧名的别名）
WEEKLY_DISCLOSURE_DIR = DATA_ROOT / "weekly_disclosure"
OUTPUT_DIR = WEEKLY_DISCLOSURE_DIR

# 云端评测系统产出的榜单目录：system/files/<competition_id>/leaderboard（只读，与 scripts 数据平级）。
REMOTE_LEADERBOARD_BASE = FILES_ROOT

# 榜单目录（本地回归测试用，落在 DATA_ROOT/leaderboard/<competition_id>）。
LOCAL_LEADERBOARD_BASE = DATA_ROOT / "leaderboard"
# 兼容旧名
LOCAL_LEADERBOARD_FALLBACK = LOCAL_LEADERBOARD_BASE


def resolve_leaderboard_dir(competition_id: str) -> str:
    """定位榜单目录，按优先级：
    1) 云端评测系统产出：system/files/<competition_id>/leaderboard；
    2) 本地回归测试：DATA_ROOT/leaderboard/<competition_id>
       （子目录内若还有一层 leaderboard/ 则进入，兼容新评测系统布局）；
    3) 均缺失时回退到 DATA_ROOT/leaderboard（兼容旧布局）。"""
    remote_dir = REMOTE_LEADERBOARD_BASE / competition_id / "leaderboard"
    if os.path.isdir(remote_dir):
        return str(remote_dir)

    local_by_id = LOCAL_LEADERBOARD_BASE / competition_id
    if os.path.isdir(local_by_id):
        nested = local_by_id / "leaderboard"
        if os.path.isdir(nested):
            return str(nested)
        return str(local_by_id)
    print(
        f"  [提示] 云端/本地按ID目录均不存在，回退到: {LOCAL_LEADERBOARD_BASE}",
        file=sys.stderr,
    )
    return str(LOCAL_LEADERBOARD_BASE)


def resolve_daily_reports_dir(competition_id: str) -> Path:
    """每个比赛的报告输出到独立子目录 DATA_ROOT/daily_reports/<competition_id>/。"""
    return DATA_ROOT / "daily_reports" / competition_id


def resolve_submissions_dir(competition_id: str) -> Path:
    """评测系统产出的提交目录：system/files/{competition_id}/submissions。"""
    return FILES_ROOT / competition_id / "submissions"
