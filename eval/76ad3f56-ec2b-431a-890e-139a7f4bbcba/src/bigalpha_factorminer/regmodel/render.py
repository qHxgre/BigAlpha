import io
import base64
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .constants import TOP_N_DISPLAY


def _date_ticks(n: int, lo: int = 5, hi: int = 15, step: int = 10) -> np.ndarray:
    num_ticks = min(hi, max(lo, n // step))
    return np.linspace(0, max(n - 1, 0), num_ticks, dtype=int)


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_weights_history(
    weights_history: pd.DataFrame,
    factor_cols: List[str],
    title_suffix: str = "",
) -> str:
    """每个滚动窗口的权重曲线，返回 base64。"""
    df = weights_history.copy()
    df["window_end"] = pd.to_datetime(df["window_end"])
    df = df.sort_values("window_end").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(factor_cols), 1)))
    x = df.index
    for i, col in enumerate(factor_cols):
        if col in df.columns:
            ax.plot(x, df[col].values, color=cmap[i % len(cmap)], linewidth=1.3, label=col)
    ax.axhline(0, color="grey", linewidth=0.8)

    ticks = _date_ticks(len(df))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        df.loc[ticks, "window_end"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right"
    )
    ax.set_title(f"Rolling Elastic Net Weights{title_suffix}")
    ax.set_xlabel("Window End")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.3)
    if len(factor_cols) <= 12:
        ax.legend(loc="best", ncol=2, fontsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_abs_weight_distribution(
    weights_history: pd.DataFrame,
    factor_cols: List[str],
    title_suffix: str = "",
) -> str:
    """每个因子 |w| 的箱线图，直观展示稳定性，返回 base64。"""
    df = weights_history[factor_cols].abs()

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(factor_cols) + 4), 5))
    ax.boxplot([df[col].dropna().values for col in factor_cols], labels=factor_cols)
    ax.set_title(f"Per-Factor |Weight| Distribution Across Windows{title_suffix}")
    ax.set_ylabel("|w|")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_model_score_bar(per_factor_scores: pd.DataFrame, title_suffix: str = "") -> str:
    """ModelScore 横向条形图，返回 base64。"""
    df = per_factor_scores.sort_values("model_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(df) + 1)))
    ax.barh(df["factor"], df["model_score"], color="#4C78A8")
    ax.set_title(f"ModelScore = mean(|w|) / (std(|w|) + eps){title_suffix}")
    ax.set_xlabel("ModelScore")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_factor_corr_heatmap(
    factor_panel: pd.DataFrame,
    factor_cols: List[str],
    title_suffix: str = "",
) -> Optional[str]:
    """因子之间的相关性热力图，返回 base64。仅在因子数 >= 2 时绘制。"""
    if len(factor_cols) < 2:
        return None

    corr = factor_panel[factor_cols].corr()
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(factor_cols) + 2),
                                    max(5, 0.6 * len(factor_cols) + 1)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(factor_cols)))
    ax.set_yticks(np.arange(len(factor_cols)))
    ax.set_xticklabels(factor_cols, rotation=45, ha="right")
    ax.set_yticklabels(factor_cols)
    if len(factor_cols) <= 25:
        for i in range(len(factor_cols)):
            for j in range(len(factor_cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    ax.set_title(f"Factor Correlation Matrix{title_suffix}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    per_factor_scores: pd.DataFrame,
    weights_history: pd.DataFrame,
    factor_panel: pd.DataFrame,
    factor_cols: List[str],
    top_n: int = TOP_N_DISPLAY,
) -> None:
    """组装 ModelScore 条形图 + 滚动权重曲线 + |w| 分布 + 相关性热力图 + 汇总表的 HTML，并通过 IPython.display 渲染。

    因子库可能上千，仅展示按 ModelScore 排名前 ``top_n`` 个因子的图表与表格。
    """
    from IPython.display import HTML, display

    total = len(factor_cols)
    top_n_eff = max(1, min(top_n, total))

    top_scores = per_factor_scores.head(top_n_eff).reset_index(drop=True)
    top_factors = [f for f in top_scores["factor"].tolist() if f in factor_cols]
    suffix = f" (Top {len(top_factors)} of {total})" if len(top_factors) < total else ""

    c_bar = plot_model_score_bar(top_scores, title_suffix=suffix)
    c_w = plot_weights_history(weights_history, top_factors, title_suffix=suffix)
    c_box = plot_abs_weight_distribution(weights_history, top_factors, title_suffix=suffix)
    c_corr = plot_factor_corr_heatmap(factor_panel, top_factors, title_suffix=suffix)

    score_rows = "".join(
        f"<tr><td>{row['factor']}</td>"
        f"<td>{_fmt(row['model_score'])}</td>"
        f"<td>{_fmt(row['abs_weight_mean'])}</td>"
        f"<td>{_fmt(row['abs_weight_std'])}</td>"
        f"<td>{_fmt(row['selection_rate'])}</td></tr>"
        for _, row in top_scores.iterrows()
    )

    corr_block = (
        f'<h2>因子相关性{suffix}</h2><img src="data:image/png;base64,{c_corr}" alt="corr heatmap"><br>'
        if c_corr is not None
        else ""
    )

    html_content = f"""
    <div style="font-family: Arial, sans-serif;">
        <h1>Elastic Net 滚动回归</h1>
        <p style="color:#666;">共 {total} 个因子参与回归，仅展示 ModelScore 排名前 {len(top_factors)} 的结果。</p>

        <h2>单因子 ModelScore{suffix}</h2>
        <img src="data:image/png;base64,{c_bar}" alt="model score bar"><br>
        <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse;">
            <tr>
                <th>factor</th>
                <th>ModelScore</th>
                <th>mean(|w|)</th>
                <th>std(|w|)</th>
                <th>selection_rate</th>
            </tr>
            {score_rows}
        </table>
        <br>

        <h2>滚动权重曲线{suffix}</h2>
        <img src="data:image/png;base64,{c_w}" alt="weights history"><br>

        <h2>|w| 跨窗口分布{suffix}</h2>
        <img src="data:image/png;base64,{c_box}" alt="abs weight distribution"><br>

        {corr_block}
    </div>
    """
    display(HTML(html_content))
