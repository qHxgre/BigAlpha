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


def read_regression(paths: CheckPaths) -> pd.DataFrame:
    data = pd.read_csv(paths.regression_path, dtype={"factor": str}, encoding="utf-8-sig")
    for metric in REGRESSION_METRICS:
        data[metric] = pd.to_numeric(data.get(metric), errors="coerce")
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
