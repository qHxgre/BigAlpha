"""私榜批次的安全 JSON 写入工具。"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as writer:
        json.dump(jsonable(value), writer, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def update_manifest(path: str, **updates: Any) -> None:
    data = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as reader:
            data = json.load(reader)
    data.update(updates)
    write_json(path, data)


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)
