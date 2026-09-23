"""30 分钟单因子分析报告的 notebook 渲染工具。"""

import base64
import io
from html import escape
from typing import Iterable

import numpy as np
import pandas as pd


def _to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    fig.clf()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _ticks(length: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, length - 1, min(12, length), dtype=int))


def _prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().sort_index()
    frame.index = pd.to_datetime(frame.index)
    return frame


def plot_group_cumret(group_cumret: pd.DataFrame, factor_name: str) -> str:
    """绘制各分组和多空组合的累计超额收益。"""
    import matplotlib.pyplot as plt

    frame = _prepare_frame(group_cumret)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    group_columns = [column for column in frame.columns if column != "ls"]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, max(len(group_columns), 1)))
    x = np.arange(len(frame))
    for color, column in zip(colors, group_columns):
        ax.plot(x, frame[column], color=color, linewidth=1.2, label=f"G{column}")
    if "ls" in frame:
        ax.plot(x, frame["ls"], color="#D62728", linestyle="--", linewidth=1.8, label="Long-Short")
    tick_positions = _ticks(len(frame))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        frame.index[tick_positions].strftime("%m-%d %H:%M"), rotation=40, ha="right"
    )
    ax.set_title(f"Group Cumulative Excess Returns - {factor_name}")
    ax.set_ylabel("Cumulative excess return")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    encoded = _to_base64(fig)
    plt.close(fig)
    return encoded


def plot_ic_series(section_ic: pd.Series, factor_name: str) -> str:
    """绘制截面 RankIC、滚动均值与累计 RankIC。"""
    import matplotlib.pyplot as plt

    ic = section_ic.copy().sort_index()
    ic.index = pd.to_datetime(ic.index)
    # 30 分钟频率下，8 个截面约等于一个交易日。
    rolling = ic.rolling(8, min_periods=1).mean()
    cumulative = ic.cumsum()
    x = np.arange(len(ic))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, ic.values, color="#9CC3E6", width=0.8, label="Section RankIC")
    ax.plot(x, rolling.values, color="#1F4E79", linewidth=1.5, label="Rolling mean (8 sections)")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, cumulative.values, color="#D62728", linewidth=1.5, label="Cumulative RankIC")
    tick_positions = _ticks(len(ic))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(ic.index[tick_positions].strftime("%m-%d %H:%M"), rotation=40, ha="right")
    ax.set_title(f"RankIC Series - {factor_name}")
    ax.set_ylabel("RankIC")
    ax2.set_ylabel("Cumulative RankIC")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    encoded = _to_base64(fig)
    plt.close(fig)
    return encoded


def plot_long_short(group_cumret: pd.DataFrame, factor_name: str) -> str:
    """绘制最高组、最低组和多空组合累计超额收益。"""
    import matplotlib.pyplot as plt

    frame = _prepare_frame(group_cumret)
    groups: Iterable = [column for column in frame.columns if column != "ls"]
    groups = list(groups)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(frame))
    if groups:
        low, high = min(groups), max(groups)
        ax.plot(x, frame[high], color="#D62728", linewidth=1.7, label=f"High group (G{high})")
        ax.plot(x, frame[low], color="#4C78A8", linewidth=1.7, label=f"Low group (G{low})")
    if "ls" in frame:
        ax.plot(x, frame["ls"], color="#F2A104", linestyle="--", linewidth=1.7, label="Long-Short")
    tick_positions = _ticks(len(frame))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        frame.index[tick_positions].strftime("%m-%d %H:%M"), rotation=40, ha="right"
    )
    ax.set_title(f"High / Low / Long-Short - {factor_name}")
    ax.set_ylabel("Cumulative excess return")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    encoded = _to_base64(fig)
    plt.close(fig)
    return encoded


def render_report(
    group_cumret: pd.DataFrame,
    section_ic: pd.Series,
    factor_name: str,
    score: dict,
) -> None:
    """生成三张绩效图和指标卡，并在 notebook 中 inline 展示。"""
    from IPython.display import HTML, display

    group_chart = plot_group_cumret(group_cumret, factor_name)
    ic_chart = plot_ic_series(section_ic, factor_name)
    long_short_chart = plot_long_short(group_cumret, factor_name)
    cards = "".join(
        f'<div class="metric"><span>{escape(label)}</span><strong>{float(score[key]):.4f}</strong></div>'
        for key, label in (
            ("ic_mean", "IC Mean"),
            ("ic_ir", "IC IR"),
            ("sharpe_ratio", "Long-Short Sharpe"),
            ("stress_stability", "Stress Stability"),
            ("turnover", "Turnover"),
        )
    )
    html = f"""
    <style>
      .jyc-factor-report {{ font-family: Arial, sans-serif; color: #1f2937; }}
      .jyc-factor-report h1 {{ font-size: 22px; border-left: 4px solid #4C78A8; padding-left: 10px; }}
      .jyc-factor-report h2 {{ font-size: 17px; margin-top: 22px; }}
      .jyc-factor-report .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; }}
      .jyc-factor-report .metric {{ background: #f8fafc; border: 1px solid #e2e8f0;
          border-radius: 7px; padding: 10px 14px; min-width: 145px; }}
      .jyc-factor-report .metric span {{ display: block; color: #64748b; font-size: 12px; }}
      .jyc-factor-report .metric strong {{ font-size: 20px; }}
      .jyc-factor-report img {{ max-width: 100%; }}
    </style>
    <div class="jyc-factor-report">
      <h1>30 分钟单因子分析 · {escape(str(factor_name))}</h1>
      <div class="metrics">{cards}</div>
      <h2>分组累计超额收益</h2><img src="data:image/png;base64,{group_chart}">
      <h2>截面 RankIC</h2><img src="data:image/png;base64,{ic_chart}">
      <h2>最高组 / 最低组 / 多空组合</h2><img src="data:image/png;base64,{long_short_chart}">
    </div>
    """
    display(HTML(html))
