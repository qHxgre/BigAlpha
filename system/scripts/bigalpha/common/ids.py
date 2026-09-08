"""user_id 名单的读取与去重工具。

原先 grant_sdk_data / grant_spro / reward_coins / participants 各自手写「去空去重保序」，
这里合并为 dedup_keep_order()，并提供两个常用的名单读取函数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


def dedup_keep_order(ids: Iterable) -> list[str]:
    """去空 + 去重 + 保持首次出现顺序，元素统一 str 化并 strip。"""
    return list(dict.fromkeys(s for s in (str(x).strip() for x in ids) if s))


def load_id_list_json(path: Path) -> list[str]:
    """读 JSON 数组形式的 user_id 名单，去空去重保序。文件缺失或格式错误则退出。"""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"未找到用户列表: {path}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(raw, list):
        print(f"{path} 应为 JSON 数组", file=sys.stderr)
        sys.exit(1)
    return dedup_keep_order(raw)


def read_id_list(path: Path) -> list[str]:
    """读 user_id 名单：.json 数组或纯文本（一行一个），去空去重保序。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        arr = json.loads(text)
        if not isinstance(arr, list):
            print(f"{path} 应为 JSON 数组", file=sys.stderr)
            sys.exit(1)
        return dedup_keep_order(arr)
    return dedup_keep_order(text.splitlines())
