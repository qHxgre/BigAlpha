import io
import base64
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _date_ticks(n: int, lo: int = 5, hi: int = 15, step: int = 10) -> np.ndarray:
    num_ticks = min(hi, max(lo, n // step))
    return np.linspace(0, max(n - 1, 0), num_ticks, dtype=int)


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_group_cumret(
    group_cumret: pd.DataFrame,
    group_num: int,
    factor_name: str = "factor",
) -> str:
    """分组累计收益曲线，返回 base64。"""
    df = group_cumret.reset_index().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    blues = plt.cm.Blues(np.linspace(0.25, 0.90, group_num))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = df.index
    for i in range(group_num):
        col = str(i)
        if col in df.columns:
            ax.plot(x, df[col].values, color=blues[i], linewidth=1.4, label=f"G{i}")
    if "ls" in df.columns:
        ax.plot(x, df["ls"].values, label="ls", linestyle="--", color="red", linewidth=1.6)
    if "bm" in df.columns:
        ax.plot(x, df["bm"].values, label="bm", linestyle=":", color="black", linewidth=1.2)

    ticks = _date_ticks(len(df))
    ax.set_xticks(ticks)
    ax.set_xticklabels(df.loc[ticks, "date"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right")
    ax.set_title(f"Group Cumulative Returns - {factor_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_ic_series(
    daily_ic: pd.Series,
    factor_name: str = "factor",
    stress_periods: List[Tuple[str, str, str]] = None,
) -> str:
    """日 IC 柱状 + 22 日滚动均值 + 累计 IC（双轴），返回 base64。

    若提供 stress_periods，会在背景用 axvspan 高亮各压力时段。
    """
    ic = daily_ic.copy()
    ic.index = pd.to_datetime(ic.index)
    ic = ic.sort_index()
    cum_ic = ic.cumsum()
    roll_ic = ic.rolling(22).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if stress_periods:
        ic_start, ic_end = ic.index.min(), ic.index.max()
        for label, s, e in stress_periods:
            ps = max(pd.Timestamp(s), ic_start)
            pe = min(pd.Timestamp(e), ic_end)
            if pe < ps:
                continue
            ax1.axvspan(ps, pe, color="grey", alpha=0.18, zorder=0)
            ax1.text(
                ps, 0,
                label, fontsize=8, color="#555",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )

    ax1.bar(ic.index, ic.values, color="#9CC3E6", width=1.0, label="Daily IC")
    ax1.plot(roll_ic.index, roll_ic.values, color="#1F4E79", linewidth=1.4, label="Rolling IC (22d)")
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("IC")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(cum_ic.index, cum_ic.values, color="red", linewidth=1.6, label="Cumulative IC")
    ax2.set_ylabel("Cumulative IC")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax1.set_title(f"IC Series - {factor_name}")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_long_short(
    group_cumret: pd.DataFrame,
    group_num: int,
    factor_name: str = "factor",
) -> str:
    """多头组 / 基准 / 多空 累计收益对比，返回 base64。"""
    df = group_cumret.reset_index().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    top_col = str(group_num - 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = df.index
    if top_col in df.columns:
        ax.plot(x, df[top_col].values, color="#FF6B6B", linewidth=1.8, label="Long (top group)")
    if "bm" in df.columns:
        ax.plot(x, df["bm"].values, color="#4ECDC4", linewidth=1.6, label="Benchmark")
    if "ls" in df.columns:
        ax.plot(x, df["ls"].values, color="#FFD166", linewidth=1.6, linestyle="--", label="Long-Short")

    ticks = _date_ticks(len(df), lo=6, hi=12, step=20)
    ax.set_xticks(ticks)
    ax.set_xticklabels(df.loc[ticks, "date"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right")
    ax.set_title(f"Long / Benchmark / Long-Short - {factor_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_stress_ic(
    stress: Dict[str, float],
    stress_periods: List[Tuple[str, str, str]],
    factor_name: str = "factor",
) -> str:
    """各压力时段 IC 条形图，并用横线标注综合 stress_ic_ir，返回 base64。"""
    names, ic_vals = [], []
    for label, _, _ in stress_periods:
        names.append(label)
        ic_vals.append(stress.get(f"{label}_ic", np.nan))

    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(len(names))
    ax.bar(idx, ic_vals, 0.6, color="#4ECDC4", label="IC")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=20, ha="right")
    stress_ir = stress.get("stress_ic_ir", np.nan)
    title_ir = f"{stress_ir:.4f}" if isinstance(stress_ir, (int, float)) and not np.isnan(stress_ir) else "nan"
    ax.set_title(f"Stress-Period IC - {factor_name} (pooled IC IR = {title_ir})")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_stress_long_short(
    group_cumret: pd.DataFrame,
    stress_periods: List[Tuple[str, str, str]],
    group_num: int,
    factor_name: str = "factor",
) -> str:
    """每个压力期单独画一张多头/基准/多空累计收益子图（窗口内归零起步），返回 base64。"""
    df_ret = group_cumret.copy()
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret = df_ret.sort_index().diff().fillna(0.0)

    top_col = str(group_num - 1)
    n = len(stress_periods)
    if n == 0:
        return ""
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4 * nrows), squeeze=False)

    for i, (label, s, e) in enumerate(stress_periods):
        ax = axes[i // ncols][i % ncols]
        ps, pe = pd.Timestamp(s), pd.Timestamp(e)
        window = df_ret.loc[(df_ret.index >= ps) & (df_ret.index <= pe)]
        if window.empty:
            ax.set_title(f"{label} (no data)")
            ax.axis("off")
            continue
        cum = window.cumsum()
        x = cum.index
        if top_col in cum.columns:
            ax.plot(x, cum[top_col].values, color="#FF6B6B", linewidth=1.6, label="Long (top)")
        if "bm" in cum.columns:
            ax.plot(x, cum["bm"].values, color="#4ECDC4", linewidth=1.4, label="Benchmark")
        if "ls" in cum.columns:
            ax.plot(x, cum["ls"].values, color="#FFD166", linewidth=1.6, linestyle="--", label="Long-Short")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_title(f"{label}  ({len(window)}d)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(loc="best", fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Stress-Period Cumulative Return - {factor_name}", fontsize=12)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _fmt(v) -> str:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "nan"
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def render_report(
    group_cumret: pd.DataFrame,
    daily_ic: pd.Series,
    stress: Dict[str, float],
    stress_periods: List[Tuple[str, str, str]],
    group_num: int,
    factor_name: str,
    score: Dict[str, float],
) -> None:
    """组装四张图 + 核心指标的 HTML，并通过 IPython.display 渲染。"""
    from IPython.display import HTML, display

    c1 = plot_group_cumret(group_cumret, group_num, factor_name)
    c2 = plot_ic_series(daily_ic, factor_name, stress_periods=stress_periods)
    c3 = plot_long_short(group_cumret, group_num, factor_name)

    if stress_periods:
        c4 = plot_stress_ic(stress, stress_periods, factor_name)
        c5 = plot_stress_long_short(group_cumret, stress_periods, group_num, factor_name)
        stress_rows = "".join(
            f"<tr><td>{label}</td>"
            f"<td>{_fmt(stress.get(f'{label}_ic'))}</td></tr>"
            for label, _s, _e in stress_periods
        )
        stress_section = f"""
        <h2>压力时段 IC</h2>
        <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse;">
            <tr><th>时段</th><th>IC</th></tr>
            {stress_rows}
        </table>
        <br>
        <img src="data:image/png;base64,{c4}" alt="stress ic">
        <br>
        <h2>压力时段多空累计收益</h2>
        <img src="data:image/png;base64,{c5}" alt="stress long-short">
        <br>
        """
    else:
        stress_section = """
        <h2>压力时段 IC</h2>
        <p style="color: #B7791F;">⚠ 因子数据时间范围与所有预设压力时段均无交集，已跳过本节。</p>
        """

    html_content = f"""
    <div style="font-family: Arial, sans-serif;">
        <h1>单因子分析 - {factor_name}</h1>
        <h2>核心指标</h2>
        <ul>
            <li>IC mean = {_fmt(score.get('ic_mean'))}</li>
            <li>IC IR = {_fmt(score.get('ic_ir'))}</li>
            <li>多空 Sharpe = {_fmt(score.get('sharpe_ratio'))}</li>
            <li>压力期 IC IR = {_fmt(score.get('stress_ic_ir'))}</li>
        </ul>

        <h2>分组累计收益</h2>
        <img src="data:image/png;base64,{c1}" alt="group cumret">
        <br>

        <h2>IC 序列</h2>
        <img src="data:image/png;base64,{c2}" alt="ic series">
        <br>

        <h2>多头 / 基准 / 多空</h2>
        <img src="data:image/png;base64,{c3}" alt="long short">
        <br>

        {stress_section}
    </div>
    """
    display(HTML(html_content))