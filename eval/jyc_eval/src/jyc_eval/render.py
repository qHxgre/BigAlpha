"""30 分钟单因子分析报告的 notebook 渲染工具。"""

import base64
import io
from html import escape
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


def _safe_ratio(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2 or not np.isfinite(values.std(ddof=1)) or values.std(ddof=1) <= 0:
        return 0.0
    return float(values.mean() / values.std(ddof=1))


def plot_intraday_effectiveness(
    section_ic: pd.Series, group_ret: pd.DataFrame, factor_name: str
) -> str:
    """按 30 分钟时点展示 IC、IC IR 和多空平均收益。"""
    import matplotlib.pyplot as plt

    data = pd.concat(
        [section_ic.rename("ic"), group_ret.get("ls", pd.Series(dtype=float)).rename("ls")],
        axis=1,
    )
    data.index = pd.to_datetime(data.index)
    data["time"] = data.index.strftime("%H:%M")
    summary = data.groupby("time", sort=True).agg(ic_mean=("ic", "mean"), ls_mean=("ls", "mean"))
    summary["ic_ir"] = data.groupby("time")["ic"].apply(_safe_ratio)
    x = np.arange(len(summary))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.bar(x, summary["ic_mean"], color="#4C78A8", label="IC Mean")
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1b = ax1.twinx()
    ax1b.plot(x, summary["ic_ir"], color="#D62728", marker="o", label="IC IR")
    ax1.set_ylabel("IC Mean")
    ax1b.set_ylabel("IC IR")
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="best")
    ax1.grid(alpha=0.25, axis="y")

    ax2.bar(x, summary["ls_mean"] * 10000, color="#F2A104", label="Mean Long-Short")
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_ylabel("Mean return (bp)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary.index)
    ax2.grid(alpha=0.25, axis="y")
    ax2.legend(loc="best")
    fig.suptitle(f"Intraday Effectiveness - {factor_name}")
    fig.tight_layout()
    encoded = _to_base64(fig)
    plt.close(fig)
    return encoded


def plot_turnover_series(turnover_series: pd.DataFrame, factor_name: str) -> str:
    """绘制多头、空头及平均单边换手率时序。"""
    import matplotlib.pyplot as plt

    frame = _prepare_frame(turnover_series)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, frame["high_turnover"], color="#D62728", alpha=0.55, label="High group")
    ax.plot(x, frame["low_turnover"], color="#4C78A8", alpha=0.55, label="Low group")
    ax.plot(x, frame["turnover"], color="#374151", linewidth=1.5, label="Average")
    ax.plot(
        x,
        frame["turnover"].rolling(8, min_periods=1).mean(),
        color="#F2A104",
        linewidth=2,
        label="Rolling mean (8 sections)",
    )
    tick_positions = _ticks(len(frame))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(frame.index[tick_positions].strftime("%m-%d %H:%M"), rotation=40, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("One-way turnover")
    ax.set_title(f"Turnover Series - {factor_name}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    encoded = _to_base64(fig)
    plt.close(fig)
    return encoded


def plot_market_regime(
    section_ic: pd.Series,
    group_ret: pd.DataFrame,
    section_volatility: pd.Series,
    factor_name: str,
) -> str:
    """按截面收益离散度中位数划分高低波动环境并对比因子表现。"""
    import matplotlib.pyplot as plt

    data = pd.concat(
        [
            section_ic.rename("ic"),
            group_ret.get("ls", pd.Series(dtype=float)).rename("ls"),
            section_volatility.rename("volatility"),
        ],
        axis=1,
    ).dropna(subset=["volatility"])
    median = data["volatility"].median()
    data["regime"] = np.where(data["volatility"] <= median, "Low volatility", "High volatility")
    order = ["Low volatility", "High volatility"]
    summary = data.groupby("regime").agg(ic_mean=("ic", "mean"), ls_mean=("ls", "mean")).reindex(order)
    summary["ic_ir"] = data.groupby("regime")["ic"].apply(_safe_ratio).reindex(order)
    summary["ls_sharpe"] = (
        data.groupby("regime")["ls"].apply(_safe_ratio).reindex(order) * np.sqrt(8 * 242)
    )

    x = np.arange(2)
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(x - width / 2, summary["ic_mean"], width, label="IC Mean", color="#4C78A8")
    ax1.bar(x + width / 2, summary["ic_ir"], width, label="IC IR", color="#72B7B2")
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_xticks(x, order)
    ax1.set_title("RankIC by market regime")
    ax1.grid(alpha=0.25, axis="y")
    ax1.legend()

    ax2.bar(x - width / 2, summary["ls_mean"] * 10000, width, label="Mean return (bp)", color="#F2A104")
    ax2.bar(x + width / 2, summary["ls_sharpe"], width, label="Annualized Sharpe", color="#D62728")
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_xticks(x, order)
    ax2.set_title("Long-short by market regime")
    ax2.grid(alpha=0.25, axis="y")
    ax2.legend()
    fig.suptitle(f"Market Regime Analysis - {factor_name}")
    fig.tight_layout()
    encoded = _to_base64(fig)
    plt.close(fig)
    return encoded


def render_report(
    group_cumret: pd.DataFrame,
    group_ret: pd.DataFrame,
    section_ic: pd.Series,
    section_volatility: pd.Series,
    turnover_series: pd.DataFrame,
    factor_name: str,
    score: dict,
) -> None:
    """生成完整绩效图和指标卡，并在 notebook 中 inline 展示。"""
    from IPython.display import HTML, display

    group_chart = plot_group_cumret(group_cumret, factor_name)
    ic_chart = plot_ic_series(section_ic, factor_name)
    intraday_chart = plot_intraday_effectiveness(section_ic, group_ret, factor_name)
    turnover_chart = plot_turnover_series(turnover_series, factor_name)
    regime_chart = plot_market_regime(section_ic, group_ret, section_volatility, factor_name)
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
      <h2>日内时段有效性</h2><img src="data:image/png;base64,{intraday_chart}">
      <h2>换手率时序</h2><img src="data:image/png;base64,{turnover_chart}">
      <h2>市场环境分层</h2><img src="data:image/png;base64,{regime_chart}">
    </div>
    """
    display(HTML(html))
