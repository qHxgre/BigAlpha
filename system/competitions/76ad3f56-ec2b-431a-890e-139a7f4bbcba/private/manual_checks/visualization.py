"""回归结果的静态图和解释性复跑。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import add_to_sys_path, read_regression, show
from .config import CheckPaths


def plot_regression_overview(paths: CheckPaths, *, top_n: int = 30) -> pd.DataFrame:
    """仅依赖现有 leaderboard_reg.csv 绘制四组回归概览图。"""
    import matplotlib.pyplot as plt

    data = read_regression(paths)
    data["weight_cv"] = data["abs_weight_std"].div(data["abs_weight_mean"].replace(0, np.nan))
    data["stability_score"] = data["selection_rate"].div(1 + data["weight_cv"])
    data = data.sort_values("model_score", ascending=False).reset_index(drop=True)
    top = data.head(min(top_n, len(data))).copy()
    top["short_factor"] = top["factor"].str[:8]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    bars = top.sort_values("model_score")
    axes[0, 0].barh(
        bars["short_factor"], bars["model_score"],
        color=plt.cm.Blues(0.25 + 0.75 * bars["selection_rate"].clip(0, 1)),
    )
    axes[0, 0].set_title(f"Top {len(top)} Factors by ModelScore")
    axes[0, 0].grid(alpha=0.25, axis="x")
    max_weight = data["abs_weight_mean"].max()
    sizes = 35 + 750 * data["abs_weight_mean"].div(max_weight) if max_weight > 0 else 35
    scatter = axes[0, 1].scatter(
        data["selection_rate"], data["model_score"], s=sizes, c=data["weight_cv"],
        cmap="viridis_r", alpha=0.75,
    )
    axes[0, 1].set_title("ModelScore vs. Selection Rate")
    fig.colorbar(scatter, ax=axes[0, 1], label="Weight CV")
    weight_scatter = axes[1, 0].scatter(
        data["abs_weight_mean"], data["abs_weight_std"], s=35 + 250 * data["selection_rate"],
        c=data["model_score"], cmap="plasma", alpha=0.75,
    )
    limit = max(data["abs_weight_mean"].max(), data["abs_weight_std"].max())
    axes[1, 0].plot([0, limit], [0, limit], "--", color="grey", linewidth=1)
    axes[1, 0].set_title("Mean Absolute Weight vs. Volatility")
    fig.colorbar(weight_scatter, ax=axes[1, 0], label="ModelScore")
    metrics = ["model_score", "abs_weight_mean", "abs_weight_std", "selection_rate", "stability_score"]
    heatmap = top.set_index("short_factor")[metrics]
    zscore = heatmap.apply(lambda col: (col - col.mean()) / col.std(ddof=0) if col.std(ddof=0) > 0 else 0)
    image = axes[1, 1].imshow(zscore, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    axes[1, 1].set_xticks(np.arange(len(metrics)), labels=metrics, rotation=30, ha="right")
    axes[1, 1].set_yticks(np.arange(len(zscore)), labels=zscore.index, fontsize=8)
    axes[1, 1].set_title("Standardized Regression Profile")
    fig.colorbar(image, ax=axes[1, 1], label="Z-Score")
    fig.suptitle("Factor Pool Regression Overview", fontsize=17)
    fig.tight_layout()
    plt.show()
    result = data[["factor", *metrics, "weight_cv"]]
    show(result)
    return result


def rerun_regression_explanation(paths: CheckPaths, *, top_n: int = 20) -> dict[str, pd.DataFrame]:
    """复跑正式滚动回归并绘制解释图；需要 BigQuant 评测环境。"""
    if paths.bigalpha_eval_src is None:
        raise ValueError("CheckPaths.bigalpha_eval_src 未配置")
    add_to_sys_path(paths.bigalpha_eval_src)
    from bigalpha_eval.regmodel import ElasticNetRegress

    pool = pd.read_parquet(paths.factor_pool_path)
    pool["date"] = pd.to_datetime(pool["date"])
    analyzer = ElasticNetRegress(pool["date"].min().strftime("%Y-%m-%d"), pool["date"].max().strftime("%Y-%m-%d"))
    rerun = analyzer.score(pool, plot=False)
    scores, weights = rerun.per_factor_scores.copy(), rerun.weights_history.copy()
    official = read_regression(paths)
    check = official.merge(scores, on="factor", suffixes=("_official", "_rerun"), validate="one_to_one")
    for metric in ("model_score", "abs_weight_mean", "abs_weight_std", "selection_rate"):
        check[f"{metric}_delta"] = (check[f"{metric}_official"] - check[f"{metric}_rerun"]).abs()
    delta_columns = [column for column in check if column.endswith("_delta")]
    print(f"解释性复跑与正式回归最大字段差异: {check[delta_columns].max().max():.3e}；滚动窗口: {len(weights)}")
    analyzer.plot()
    # 库内置图保留正式解释口径；返回底层数据，便于 notebook 继续自定义绘图。
    result = {"score_check": check, "scores": scores, "weights_history": weights}
    show(scores.head(top_n), weights.head())
    return result
