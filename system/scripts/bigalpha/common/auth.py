"""BigQuant 认证：读取 token / server。

原先 reward_coins / send_daily_reports / grant_spro / grant_sdk_data 各有一份几乎相同的
load_auth()，这里合并为一处，统一带 encoding="utf-8"。

优先环境变量 BIGQUANT_TOKEN / BIGQUANT_SERVER，其次 ~/.bigquant/auth.json
（可用 BIGQUANT_AUTH_FILE 覆盖路径）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_SERVER = "https://bigquant.com"


def load_auth() -> tuple[str, str]:
    """读认证：优先环境变量，其次 ~/.bigquant/auth.json。返回 (token, server)。"""
    token = os.environ.get("BIGQUANT_TOKEN")
    server = os.environ.get("BIGQUANT_SERVER", "").rstrip("/")

    if not token:
        auth_file = Path(
            os.environ.get("BIGQUANT_AUTH_FILE", Path.home() / ".bigquant" / "auth.json")
        )
        try:
            data = json.loads(auth_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"未找到认证文件: {auth_file}", file=sys.stderr)
            sys.exit(1)
        token = data.get("token")
        if not token:
            print("auth.json 中缺少 token 字段", file=sys.stderr)
            sys.exit(1)
        if not server:
            server = str(data.get("server", DEFAULT_SERVER)).rstrip("/")

    return token, server or DEFAULT_SERVER
