"""各检查模块共用的数据读取和展示工具。"""

from __future__ import annotations

import sys
from collections.abc import Iterable

import pandas as pd

from .config import CheckPaths, METRICS, REGRESSION_METRICS


def read_summary(paths: CheckPaths) -> pd.DataFrame:
    return pd.read_csv(paths.summary_path, dtype={"submission_id": str, "user_id": str})


def read_final(paths: CheckPaths) -> pd.DataFrame:
    return pd.read_csv(paths.final_path, dtype={"submission_id": str})


def read_public_scores(paths: CheckPaths) -> pd.DataFrame:
    """从公榜 leaderboard 读取每个 submission 的完整得分明细。"""
    path = paths.public_summary_path
    if not path.is_file():
        raise FileNotFoundError(f"公榜得分明细不存在: {path}")
    columns = ["submission_id", *METRICS, "a_score", "b_score", "score"]
    data = pd.read_csv(path, dtype={"submission_id": str}, usecols=columns).rename(
        columns={"score": "public_score"}
    )
    for column in [*METRICS, "a_score", "b_score", "public_score"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    duplicates = data.loc[data["submission_id"].duplicated(keep=False), "submission_id"].unique()
    if len(duplicates):
        raise ValueError(f"公榜 submissions_summary.csv 中存在重复 submission_id：{', '.join(duplicates)}")
    return data


def read_regression(paths: CheckPaths) -> pd.DataFrame:
    data = pd.read_csv(paths.regression_path, dtype={"factor": str}, encoding="utf-8-sig")
    for metric in REGRESSION_METRICS:
        data[metric] = pd.to_numeric(data.get(metric), errors="coerce")
    return data


def read_public_regression(paths: CheckPaths) -> pd.DataFrame:
    """读取公榜 B 项逐 submission 回归指标。"""
    path = paths.public_regression_path
    if not path.is_file():
        raise FileNotFoundError(f"公榜 B 项明细不存在: {path}")
    data = pd.read_csv(path, dtype={"factor": str}, encoding="utf-8-sig")
    for metric in REGRESSION_METRICS:
        data[metric] = pd.to_numeric(data.get(metric), errors="coerce")
    duplicates = data.loc[data["factor"].duplicated(keep=False), "factor"].unique()
    if len(duplicates):
        raise ValueError(f"公榜 leaderboard_reg.csv 中存在重复 factor：{', '.join(duplicates)}")
    return data


def add_to_sys_path(path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def show(*objects: Iterable[object]) -> None:
    """在 Notebook 中展示对象；普通 Python 环境则打印。"""
    try:
        from IPython.display import display
    except ImportError:
        for obj in objects:
            print(obj)
    else:
        for obj in objects:
            display(obj)
