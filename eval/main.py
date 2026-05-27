"""绘图工具 —— 输出 base64 PNG 字符串。

绘制项：
- 滚动 IC 曲线（含均值线）
- 分组累计收益（含多空 ls）
- Elastic Net 滚动权重曲线（多因子时）
"""

import base64
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fig_to_base64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img


def plot_ic(daily_ic: pd.Series) -> str:
    """绘制日度 IC 曲线 + 累计 IC。"""
    if daily_ic is None or daily_ic.empty:
        plt.figure(figsize=(12, 4))
        plt.title("Daily IC (empty)")
        return _fig_to_base64()

    s = daily_ic.copy().sort_index()
    s.index = pd.to_datetime(s.index)
    cum_ic = s.cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(s.index, s.values, color="#4ECDC4", alpha=0.6, width=1.0, label="Daily IC")
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_ylabel("Daily IC")
    ax1.set_xlabel("Date")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(cum_ic.index, cum_ic.values, color="#FF6B6B", linewidth=1.6, label="Cumulative IC")
    ax2.set_ylabel("Cumulative IC")

    plt.title(f"Daily IC (mean={s.mean():.4f}, IR={s.mean()/s.std(ddof=1)*np.sqrt(252):.4f})")
    fig.tight_layout()
    return _fig_to_base64()


def plot_group_cumret(group_cum: pd.DataFrame, group_num: int) -> str:
    """分组累计收益曲线（含多空 ls）。"""
    if group_cum is None or group_cum.empty:
        plt.figure(figsize=(12, 4))
        plt.title("Group Cumulative Returns (empty)")
        return _fig_to_base64()

    df = group_cum.copy().sort_index()
    df.index = pd.to_datetime(df.index)

    plt.figure(figsize=(12, 6))
    blues = plt.cm.Blues(np.linspace(0.25, 0.90, group_num))
    for i in range(group_num):
        col = str(i)
        if col in df.columns:
            plt.plot(df.index, df[col].values, color=blues[i], linewidth=1.5, label=f"G{i}")
    if "ls" in df.columns:
        plt.plot(df.index, df["ls"].values, linestyle="--", color="red", label="Long-Short")

    plt.title("Group Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _fig_to_base64()


def plot_weights_history(weights_history: pd.DataFrame, factor_cols: list) -> str:
    """Elastic Net 滚动权重曲线。"""
    if weights_history is None or weights_history.empty:
        plt.figure(figsize=(12, 4))
        plt.title("Elastic Net Rolling Weights (empty)")
        return _fig_to_base64()

    df = weights_history.copy()
    df["window_end"] = pd.to_datetime(df["window_end"])
    df = df.sort_values("window_end")

    plt.figure(figsize=(12, 5))
    for col in factor_cols:
        if col in df.columns:
            plt.plot(df["window_end"], df[col].values, linewidth=1.4, label=col)
    plt.axhline(0, color="black", linewidth=0.6)

    plt.title("Elastic Net Rolling Weights")
    plt.xlabel("Window End")
    plt.ylabel("Weight")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _fig_to_base64()
