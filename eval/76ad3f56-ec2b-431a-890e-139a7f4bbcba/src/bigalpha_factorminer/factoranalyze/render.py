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
    stress_periods: List[Tuple[str, str, str]] = None,
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

    # 叠加压力时段背景
    if stress_periods:
        ymax = ax.get_ylim()[1]
        for label, s, e in stress_periods:
            ps, pe = pd.Timestamp(s), pd.Timestamp(e)
            sub_s = df[df["date"] >= ps]
            sub_e = df[df["date"] <= pe]
            if sub_s.empty or sub_e.empty:
                continue
            idx_s = sub_s.index.min()
            idx_e = sub_e.index.max()
            if idx_e < idx_s:
                continue
            ax.axvspan(idx_s, idx_e, color="grey", alpha=0.15, zorder=0)
            ax.text(
                idx_s, ymax, label, fontsize=7, color="#555",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )

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
    """日 IC 柱状 + 22 日滚动均值 + 累计 IC（双轴），返回 base64。"""
    ic = daily_ic.copy()
    ic.index = pd.to_datetime(ic.index)
    ic = ic.sort_index()
    cum_ic = ic.cumsum()
    roll_ic = ic.rolling(22).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))

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
    stress_periods: List[Tuple[str, str, str]] = None,
) -> str:
    """多头组 / 基准 / 多空 累计收益对比，并标注压力时段，返回 base64。"""
    df = group_cumret.reset_index().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    top_col = str(group_num - 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = df.index

    # 先画曲线，让 ylim 稳定
    if top_col in df.columns:
        ax.plot(x, df[top_col].values, color="#FF6B6B", linewidth=1.8, label="Long (top group)")
    if "bm" in df.columns:
        ax.plot(x, df["bm"].values, color="#4ECDC4", linewidth=1.6, label="Benchmark")
    if "ls" in df.columns:
        ax.plot(x, df["ls"].values, color="#FFD166", linewidth=1.6, linestyle="--", label="Long-Short")

    # 再叠加压力时段背景
    if stress_periods:
        ymax = ax.get_ylim()[1]
        for label, s, e in stress_periods:
            ps, pe = pd.Timestamp(s), pd.Timestamp(e)
            sub_s = df[df["date"] >= ps]
            sub_e = df[df["date"] <= pe]
            if sub_s.empty or sub_e.empty:
                continue
            idx_s = sub_s.index.min()
            idx_e = sub_e.index.max()
            if idx_e < idx_s:
                continue
            ax.axvspan(idx_s, idx_e, color="grey", alpha=0.15, zorder=0)
            ax.text(
                idx_s, ymax, label, fontsize=7, color="#555",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )

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
    """每个压力期单独画一张多头/基准累计收益子图（窗口内归零起步），返回 base64。"""
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


def _signal(value: float, good_thresh: float, ok_thresh: float, higher_is_better: bool = True) -> str:
    """根据阈值返回颜色信号灯 HTML 标签。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if higher_is_better:
        color = "#22C55E" if v >= good_thresh else ("#F59E0B" if v >= ok_thresh else "#EF4444")
    else:
        color = "#22C55E" if v <= good_thresh else ("#F59E0B" if v <= ok_thresh else "#EF4444")
    return f'<span style="color:{color};font-size:16px;line-height:1;">●</span>'


_METRIC_CSS = """
<style>
.fa-report { font-family: "Helvetica Neue", Arial, sans-serif; color: #1a1a2e; }
.fa-report h1 { font-size: 1.5em; margin: 0 0 6px 0; color: #1a1a2e; border-left: 4px solid #4C78A8; padding-left: 10px; }
.fa-report h2 { font-size: 1.15em; margin: 18px 0 8px 0; color: #2d3748; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
.metric-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px 18px; min-width: 180px; flex: 1 1 180px;
}
.metric-card .label {
    font-size: 0.78em; color: #64748b; margin-bottom: 4px;
    display: flex; align-items: center; gap: 5px;
}
.metric-card .label .tip {
    cursor: help; font-size: 0.85em; color: #94a3b8;
    border-bottom: 1px dashed #94a3b8;
}
.metric-card .value { font-size: 1.4em; font-weight: 600; color: #1e293b; }
.metric-card .signal { display: inline-block; margin-left: 6px; }
.fa-table { border-collapse: collapse; font-size: 0.88em; }
.fa-table th {
    background: #f1f5f9; color: #475569;
    padding: 7px 12px; border: 1px solid #e2e8f0;
    font-weight: 600; white-space: nowrap;
}
.fa-table td { padding: 6px 12px; border: 1px solid #e2e8f0; }
.fa-table tr:nth-child(even) td { background: #f8fafc; }
.fa-table .tip-th { cursor: help; border-bottom: 1px dashed #94a3b8; }
.warn-box {
    background: #FFFBEB; border: 1px solid #F59E0B; border-radius: 6px;
    padding: 10px 14px; color: #92400E; font-size: 0.9em;
}
</style>
"""


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

    c1 = plot_group_cumret(group_cumret, group_num, factor_name, stress_periods=stress_periods)
    c2 = plot_ic_series(daily_ic, factor_name)
    c3 = plot_long_short(group_cumret, group_num, factor_name, stress_periods=stress_periods)

    # ── 核心指标卡片 ─────────────────────────────────────────────────────────
    ic_mean = score.get("ic_mean")
    ic_ir = score.get("ic_ir")
    sharpe = score.get("sharpe_ratio")
    stress_ir = score.get("stress_ic_ir")

    def _card(label: str, value: float, tip: str, sig_html: str) -> str:
        return (
            f'<div class="metric-card">'
            f'  <div class="label"><span class="tip" title="{tip}">{label}</span></div>'
            f'  <div class="value">{_fmt(value)}<span class="signal">{sig_html}</span></div>'
            f'</div>'
        )

    metric_cards = "".join([
        _card(
            "IC Mean", ic_mean,
            "因子值与下期收益的 Spearman 相关系数均值。|IC| > 0.03 表示因子有效，> 0.05 较强。",
            _signal(ic_mean, 0.05, 0.03) if ic_mean is not None else "",
        ),
        _card(
            "IC IR（信息比率）", ic_ir,
            "IC Mean / IC Std，衡量 IC 的稳定性。|IC IR| > 0.5 为合格，> 1.0 为优秀。",
            _signal(ic_ir, 1.0, 0.5) if ic_ir is not None else "",
        ),
        _card(
            "多空 Sharpe", sharpe,
            "多头组 - 空头组的日收益序列年化 Sharpe 比率。> 1.0 为合格，> 2.0 为优秀。",
            _signal(sharpe, 2.0, 1.0) if sharpe is not None else "",
        ),
        _card(
            "压力期 IC IR", stress_ir,
            "仅在预设压力时段（市场极端行情）内计算的汇总 IC IR，反映因子的抗压能力。> 0.3 为合格。",
            _signal(stress_ir, 0.5, 0.3) if stress_ir is not None else "",
        ),
    ])

    # ── 压力时段区块 ─────────────────────────────────────────────────────────
    if stress_periods:
        c4 = plot_stress_ic(stress, stress_periods, factor_name)
        c5 = plot_stress_long_short(group_cumret, stress_periods, group_num, factor_name)
        stress_rows = "".join(
            f"<tr><td>{label}</td>"
            f"<td style='text-align:right'>{_fmt(stress.get(f'{label}_ic'))}</td></tr>"
            for label, _s, _e in stress_periods
        )
        stress_section = f"""
        <h2>压力时段 IC</h2>
        <p style="font-size:0.85em;color:#64748b;margin:0 0 8px 0;">
          在以下极端行情时段分别统计因子 IC，检验因子在市场压力下的稳定性。
        </p>
        <table class="fa-table" style="margin-bottom:12px;">
            <tr>
              <th><span class="tip-th" title="预设的极端行情时段名称">时段</span></th>
              <th><span class="tip-th" title="该时段内因子值与下期收益的 Spearman 相关系数均值">IC</span></th>
            </tr>
            {stress_rows}
        </table>
        <img src="data:image/png;base64,{c4}" alt="stress ic" style="max-width:100%">
        <br><br>
        <h2>压力时段多空累计收益</h2>
        <p style="font-size:0.85em;color:#64748b;margin:0 0 8px 0;">
          各极端行情区间内，多头组（最高分位）与基准各自从零起步的累计收益对比。
        </p>
        <img src="data:image/png;base64,{c5}" alt="stress long-short" style="max-width:100%">
        <br>
        """
    else:
        stress_section = """
        <h2>压力时段 IC</h2>
        <div class="warn-box">⚠ 因子数据时间范围与所有预设压力时段均无交集，已跳过本节。</div>
        """

    # ── 最终 HTML ─────────────────────────────────────────────────────────────
    html_content = f"""
    {_METRIC_CSS}
    <div class="fa-report">
        <h1>单因子分析 &nbsp;·&nbsp; {factor_name}</h1>

        <h2>核心指标
          <span style="font-size:0.75em;font-weight:400;color:#64748b;margin-left:8px;">
            ● 优秀 &nbsp; ● 合格 &nbsp; ● 待改进（信号灯以绝对值判断）
          </span>
        </h2>
        <div class="metric-grid">
          {metric_cards}
        </div>

        <h2>分组累计收益</h2>
        <p style="font-size:0.85em;color:#64748b;margin:0 0 8px 0;">
          将全部股票按因子值从低到高等分为 {group_num} 组（G0 最低，G{group_num-1} 最高），
          各组累计收益应呈现单调递增/递减趋势以验证因子有效性。
          灰色背景为预设压力时段，红色虚线为多空组合（G{group_num-1} − G0）。
        </p>
        <img src="data:image/png;base64,{c1}" alt="group cumret" style="max-width:100%">
        <br><br>

        <h2>IC 序列</h2>
        <p style="font-size:0.85em;color:#64748b;margin:0 0 8px 0;">
          浅蓝柱：每日 IC（因子与下期收益的 Spearman 相关系数）；
          深蓝线：22 日滚动 IC 均值，反映因子的近期稳定性；
          右轴红线：累计 IC，持续上行说明因子长期有效。
        </p>
        <img src="data:image/png;base64,{c2}" alt="ic series" style="max-width:100%">
        <br><br>

        <h2>多头 / 基准 / 多空</h2>
        <p style="font-size:0.85em;color:#64748b;margin:0 0 8px 0;">
          多头组（G{group_num-1}，最高分位）、基准（全市场等权）及多空组合（G{group_num-1} − G0）
          的累计收益对比。灰色背景为压力时段标注。
        </p>
        <img src="data:image/png;base64,{c3}" alt="long short" style="max-width:100%">
        <br><br>

        {stress_section}
    </div>
    """
    display(HTML(html_content))