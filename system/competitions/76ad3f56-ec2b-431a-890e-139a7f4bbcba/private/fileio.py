"""私榜批次的 JSON、manifest 和待发布文件写入工具。"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable


def jsonable(value: Any) -> Any:
    """递归转换为严格 JSON 可序列化的数据，并把 NaN/Infinity 转成 null。"""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(jsonable(value), writer, ensure_ascii=False, indent=2, allow_nan=False)


def update_manifest(path: str, **updates: Any) -> None:
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as reader:
            data = json.load(reader)
    data.update(updates)
    write_json(path, data)


def write_pending_publish(path: str, records: Iterable[dict]) -> None:
    """写入人工核验后才会发布的后台更新载荷。"""
    with open(path, "w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
