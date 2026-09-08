# -*- coding: utf-8 -*-
"""比赛训练侧入口（薄封装）。

全部训练 / 存盘 / 路径逻辑在 Model.Framework 中。
本文件只负责：
  1. 把 PonyOracle 根目录与 Model/ 注入 sys.path
  2. re-export 平台需要的符号：train_and_save / load_model / MODEL_PATH
  3. __main__ 本地一键训练

用法:
    python transformer_train.py
    from transformer_train import train_and_save
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 平台 / notebook 从此处 import —— 实现全部在 Framework
from ModelSet_Framework import (  # noqa: E402
    MODEL_PATH,
    TRAIN_START,
    TRAIN_END,
    load_model,
    save_model,
    train_and_save,
    predict_scores,
    default_config,
    PonyFramework,
)

__all__ = [
    "MODEL_PATH",
    "TRAIN_START",
    "TRAIN_END",
    "load_model",
    "save_model",
    "train_and_save",
    "predict_scores",
    "default_config",
    "PonyFramework",
]


if __name__ == "__main__":
    
    datasources = {
        "bar1m": "bigalpha_2026_stock_bar1m",
        "bar5m": "bigalpha_2026_stock_bar5m",
        "bar15m": "bigalpha_2026_stock_bar15m",
        "bar30m": "bigalpha_2026_stock_bar30m"
    }
    
    train_and_save(datasources)
