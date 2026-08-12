"""人工复核模块共用的数据读取和展示工具。"""

from __future__ import annotations

import json
import sys

import pandas as pd

from .config import METRICS, CheckPaths


def _read_csv(path, **dtypes) -> pd.DataFrame:
    return pd.read_csv(path, dtype={key: str for key in dtypes})


def read_summary(paths: CheckPaths) -> pd.DataFrame:
    return _read_csv(paths.summary_path, submission_id=str, user_id=str)


def read_score(paths: CheckPaths) -> pd.DataFrame:
    data = _read_csv(paths.score_path, submission_id=str)
    for column in (*METRICS, "score"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    return data


def read_final(paths: CheckPaths) -> pd.DataFrame:
    data = _read_csv(paths.final_path, submission_id=str)
    data["score"] = pd.to_numeric(data.get("score"), errors="coerce")
    return data


def read_public_scores(paths: CheckPaths) -> pd.DataFrame:
    """从准备阶段元数据读取所选私榜提交对应的公榜分数。"""
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    rows = [
        {
            "submission_id": str(item.get("submission_id") or ""),
            "public_score": item.get("public_score"),
        }
        for item in metadata.get("submissions", [])
        if item.get("submission_id")
    ]
    data = pd.DataFrame(rows, columns=["submission_id", "public_score"])
    data["public_score"] = pd.to_numeric(data["public_score"], errors="coerce")
    duplicates = data.loc[data["submission_id"].duplicated(keep=False), "submission_id"].unique()
    if len(duplicates):
        raise ValueError(f"metadata.json 中存在重复 submission_id：{', '.join(duplicates)}")
    return data


def add_to_sys_path(path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def show(*objects) -> None:
    try:
        from IPython.display import display
    except ImportError:
        for obj in objects:
            print(obj)
    else:
        for obj in objects:
            display(obj)
