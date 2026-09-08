"""scripts/ 下各功能簇脚本复用的共享包。

模块划分：
    client      Alphathon API 客户端（AlphathonClient）
    auth        BigQuant 认证（load_auth：token/server）
    ids         user_id 名单读取与去重
    paths       数据根目录与各子目录常量、榜单目录定位
    disclosure  每周公示共享配置（matplotlib 中文字体、zscore 等）

各簇脚本直接运行时，先把 scripts/ 根加入 sys.path 再 `from common.xxx import ...`，
见 common.bootstrap。
"""
from __future__ import annotations

import sys
from pathlib import Path

# scripts/ 根目录（common 的父目录）。各脚本 import 前调用 bootstrap() 即可无痛
# 以 `python scripts/<簇>/xxx.py` 直接运行。
SCRIPTS_ROOT = Path(__file__).resolve().parent.parent


def bootstrap() -> None:
    """把 scripts/ 根加入 sys.path，使 `from common.xxx import ...` 在直接运行脚本时可用。"""
    root = str(SCRIPTS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
