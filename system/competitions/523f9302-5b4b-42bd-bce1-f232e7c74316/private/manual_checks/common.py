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
    """读取公榜逐 submission 总分和四项指标；旧数据缺文件时回退 metadata。"""
    if paths.public_summary_path.is_file():
        data = pd.read_csv(paths.public_summary_path, dtype={"submission_id": str})
        score_column = "score" if "score" in data else "final_score"
        keep = ["submission_id", *[m for m in METRICS if m in data], score_column]
        data = data[keep].rename(columns={score_column: "public_score"})
        for column in [*METRICS, "public_score"]:
            if column not in data:
                data[column] = pd.NA
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data[["submission_id", *METRICS, "public_score"]]
    else:
        metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
        rows = []
        for item in metadata.get("submissions", []):
            if not item.get("submission_id"):
                continue
            score_data = (item.get("submission") or {}).get("public_score_data") or {}
            rows.append({
                "submission_id": str(item["submission_id"]),
                **{metric: score_data.get(metric) for metric in METRICS},
                "public_score": score_data.get("score", item.get("public_score")),
            })
        data = pd.DataFrame(rows, columns=["submission_id", *METRICS, "public_score"])
        for column in [*METRICS, "public_score"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    duplicates = data.loc[data["submission_id"].duplicated(keep=False), "submission_id"].unique()
    if len(duplicates):
        raise ValueError(f"公榜数据中存在重复 submission_id：{', '.join(duplicates)}")
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
