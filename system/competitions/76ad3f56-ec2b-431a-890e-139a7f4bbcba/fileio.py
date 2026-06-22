"""通用文件读取工具（本比赛内部使用）。

把「文件不存在 / 解析失败时返回 None 并记一行日志」这类样板收敛到一处，
让评测主流程（public.py）只表达业务意图，不再重复 exists + try/except。
logger 可选，未传入时退回到模块级 structlog logger。
"""
from __future__ import annotations

import json
import os

import pandas as pd
import structlog

_logger = structlog.get_logger()


def read_json(path: str, logger=None) -> dict | None:
    """读取 json 文件，不存在或解析失败时返回 None（吞异常只记一行日志）。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as reader:
            return json.load(reader)
    except Exception as e:
        (logger or _logger).error("read.json_failed", path=path, error=str(e), msg="读取 json 文件失败")
        return None


def read_csv(path: str, logger=None) -> pd.DataFrame | None:
    """读取 csv 文件，不存在或解析失败时返回 None（吞异常只记一行日志）。"""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        (logger or _logger).error("read.csv_failed", path=path, error=str(e), msg="读取 csv 文件失败")
        return None


def csv_to_map(df: pd.DataFrame | None, key: str, val: str) -> dict:
    """把 DataFrame 的两列转成 {str(key): val} 映射；df 为空或缺列时返回空 dict。"""
    if df is None or key not in df.columns or val not in df.columns:
        return {}
    return {str(k): v for k, v in zip(df[key], df[val])}
