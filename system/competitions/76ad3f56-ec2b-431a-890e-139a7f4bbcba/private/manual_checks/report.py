"""运行全部人工复核并生成可归档的 Markdown 报告。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_final, read_summary
from .config import (
    CONFIG,
    FACTOR_CLUSTERS_FILENAME,
    FACTOR_SIMILARITY_SUMMARY_FILENAME,
    GROUP_INCREMENTAL_SUMMARY_FILENAME,
    INCREMENTAL_BY_WINDOW_FILENAME,
    INCREMENTAL_SUMMARY_FILENAME,
    MAX_AB_RANK_GAP,
    PATHS,
    STYLE_EXPOSURE_FIGURE_FILENAME,
    STYLE_EXPOSURE_DETAIL_FILENAME,
    STYLE_EXPOSURE_METADATA_FILENAME,
    STYLE_EXPOSURE_SUMMARY_FILENAME,
    RERUN_WEIGHTS_FILENAME,
    CheckPaths,
)
from .factors import analyze_factor_similarity
from .regression import analyze_regression_explainability
from .team_private_leaderboard import build_team_private_report
from .ranking import (
    analyze_a_metric_sensitivity,
    analyze_ab_weight_sensitivity,
    analyze_rank_conflicts,
    check_score_consistency,
)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(data: pd.DataFrame, *, limit: int | None = None) -> str:
    """不依赖 tabulate，将 DataFrame 转为 Markdown 表格。"""
    if data.empty:
        return "_无异常记录。_"
    frame = data.head(limit).copy() if limit is not None else data.copy()
    frame = frame.reset_index() if frame.index.name else frame.reset_index(drop=True)
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    if limit is not None and len(data) > limit:
        lines.append(f"\n_仅展示前 {limit} 条，共 {len(data)} 条；完整结果请通过 Python 返回值查看。_")
    return "\n".join(lines)


def _dominant_style_classification(
    style_summary: pd.DataFrame,
    style_detail: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """将每个因子唯一归入最强风格；未超过阈值的归入 OTHER。"""
    factor_ids = style_summary["submission_id"].astype(str)
    if style_detail.empty:
        return pd.DataFrame({
            "submission_id": factor_ids,
            "dominant_style": "OTHER",
            "dominant_mean_abs_correlation": np.nan,
        })
    strongest = (
        style_detail.sort_values("mean_abs_correlation", ascending=False)
        .drop_duplicates("submission_id")
        [["submission_id", "style", "mean_abs_correlation"]]
        .rename(columns={
            "style": "dominant_style",
            "mean_abs_correlation": "dominant_mean_abs_correlation",
        })
    )
    classified = pd.DataFrame({"submission_id": factor_ids}).merge(
        strongest, on="submission_id", how="left"
    )
    classified.loc[
        classified["dominant_mean_abs_correlation"].isna()
        | classified["dominant_mean_abs_correlation"].le(threshold),
        "dominant_style",
    ] = "OTHER"
    return classified


def _require_inputs(paths: CheckPaths) -> None:
    required = [
        paths.summary_path,
        paths.final_path,
        paths.regression_path,
        paths.factor_pool_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("生成报告所需文件不存在：\n" + "\n".join(missing))


def _figure_reference(path: Path, report_path: Path) -> str:
    """返回适合写入 Markdown 的图片路径。"""
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _chart(path: Path, output: Path, title: str) -> list[str]:
    return [f"![{title}]({_figure_reference(path, output)})", ""]


def _participant_lookup(paths: CheckPaths) -> pd.DataFrame:
    """从准备阶段元数据构造 submission 到队伍及学校的映射。"""
    if not paths.metadata_path.is_file():
        return pd.DataFrame(columns=["submission_id", "participant", "schools"])
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    rows = []
    for participant in metadata.get("participants", []):
        if participant.get("type") == "team":
            participant_name = participant.get("team_name") or "（未命名队伍）"
            members = participant.get("members") or []
        else:
            user = participant.get("user") or {}
            participant_name = f"个人：{user.get('name') or '未命名'}"
            members = [user]
        schools = list(dict.fromkeys(
            str(member.get("school") or "（未填写）") for member in members
        ))
        school_text = "<br>".join(schools) if schools else "（未填写）"
        for submission_id in participant.get("private_submission_ids") or []:
            rows.append({
                "submission_id": str(submission_id),
                "participant": participant_name,
                "schools": school_text,
            })
    return pd.DataFrame(rows, columns=["submission_id", "participant", "schools"])


def _generate_report_figures(
    *,
    paths: CheckPaths,
    output: Path,
    summary: pd.DataFrame,
    final: pd.DataFrame,
    score_problems: pd.DataFrame,
    rank_conflicts: pd.DataFrame,
    ab_sensitivity: pd.DataFrame,
    a_metric_sensitivity: pd.DataFrame,
    style_summary: pd.DataFrame,
    style_detail: pd.DataFrame,
    similarity: pd.DataFrame,
    regression_explanation: pd.DataFrame,
    incremental_summary: pd.DataFrame,
    incremental_by_window: pd.DataFrame,
    factor_clusters: pd.DataFrame,
    group_incremental_summary: pd.DataFrame,
    high_correlation: float,
    top_ids: list[str],
    rank_map: dict[str, int],
) -> dict[str, Path]:
    """用全量检查数据生成报告图；异常为空时仍保留正常分布。"""
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    figure_dir = output.parent / f"{output.stem}_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.formatter.useoffset": False,
    })
    figures: dict[str, Path] = {}

    def save(name: str, fig) -> None:
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                formatter = axis.get_major_formatter()
                if isinstance(formatter, ScalarFormatter):
                    formatter.set_scientific(False)
                    formatter.set_useOffset(False)
        path = figure_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        figures[name] = path

    # 总览：水平展示 success/failed 的绝对数量和占总提交比例。
    normalized_status = summary["status"].fillna("failed").astype(str).str.lower()
    status_counts = pd.Series(
        {
            "success": int(normalized_status.eq("success").sum()),
            "failed": int(normalized_status.ne("success").sum()),
        }
    )
    total_submissions = int(status_counts.sum())
    status_rates = status_counts.div(total_submissions) if total_submissions else status_counts.astype(float)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(status_counts))
    bars = ax.barh(y, status_counts.values, color=["#54A24B", "#E45756"])
    ax.set_yticks(y, status_counts.index)
    ax.invert_yaxis()
    ax.set(title="Submission success / failed distribution", xlabel="submission count", ylabel="status")
    ax.set_xlim(0, max(status_counts.max() * 1.25, 1))
    for bar, count, rate in zip(bars, status_counts.values, status_rates.values):
        ax.text(
            bar.get_width() + max(total_submissions * .01, .2),
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({rate:.1%})",
            ha="left",
            va="center",
            fontweight="bold",
        )
    save("01_submission_status", fig)

    # 成绩复算：展示所有有效提交的三项正式分数，而非只展示异常行。
    score_data = final.loc[final["final_score"].ge(0)].sort_values("final_score", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(1, len(score_data) + 1)
    for column, color in (("a_score", "#4C78A8"), ("b_score", "#F58518"), ("final_score", "#54A24B")):
        ax.plot(x, score_data[column], label=column, linewidth=1.5, color=color)
    ax.set(title=f"Stored scores for all valid submissions (recompute issues: {len(score_problems)})",
           xlabel="official final rank", ylabel="score")
    ax.legend()
    save("02_score_consistency", fig)

    # 头部榜单：用一张哑铃图同时表达 A、B 和最终分，避免堆叠多张敏感性图。
    top_ranking = rank_conflicts.loc[rank_conflicts["submission_id"].isin(top_ids)].sort_values("final_rank")
    fig, ax = plt.subplots(figsize=(11, max(6, len(top_ranking) * .38)))
    y = np.arange(len(top_ranking))
    low = top_ranking[["a_score", "b_score"]].min(axis=1)
    high = top_ranking[["a_score", "b_score"]].max(axis=1)
    ax.hlines(y, low, high, color="#B8B8B8", linewidth=2.2, zorder=1)
    ax.scatter(top_ranking["a_score"], y, color="#4C78A8", s=48, label="A score", zorder=3)
    ax.scatter(top_ranking["b_score"], y, color="#F58518", s=48, label="B score", zorder=3)
    ax.scatter(
        top_ranking["final_score"], y, color="#54A24B", marker="D", s=42,
        label="final score", zorder=4,
    )
    ax.set_yticks(
        y,
        [f"#{int(rank)} {submission_id[:8]}" for rank, submission_id in zip(
            top_ranking["final_rank"], top_ranking["submission_id"]
        )],
    )
    ax.invert_yaxis()
    ax.set(
        title="Top finalists: A / B score range and weighted final score",
        xlabel="score",
        ylabel="final rank / submission",
    )
    ax.legend(ncol=3, loc="lower right")
    save("03_top_ranking_comparison", fig)

    # 联合回归解释：头部因子的相对重要性、持续入选情况和逐期权重方向。
    regression_top = regression_explanation.loc[
        regression_explanation["submission_id"].isin(top_ids)
    ].sort_values("final_rank")
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    labels = [f"#{int(rank)} {sid[:8]}" for rank, sid in zip(
        regression_top["final_rank"], regression_top["submission_id"]
    )]
    y = np.arange(len(regression_top))
    bars = axes[0, 0].barh(
        y, regression_top["model_score"],
        color=plt.cm.Blues(0.25 + 0.75 * regression_top["selection_rate"].clip(0, 1)),
    )
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(
        title="Top finalists: joint-model importance",
        xlabel="model score (color intensity = selection rate)",
        ylabel="final rank / submission",
    )
    for bar, rate in zip(bars, regression_top["selection_rate"]):
        axes[0, 0].text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f"  {rate:.0%}", va="center", fontsize=8,
        )

    axes[0, 1].scatter(
        regression_explanation["final_rank"], regression_explanation["model_rank"],
        color="#B8B8B8", alpha=.55, label="all factors",
    )
    axes[0, 1].scatter(
        regression_top["final_rank"], regression_top["model_rank"],
        s=70, color="#E45756", label="top finalists", zorder=3,
    )
    limit = max(len(regression_explanation), 1)
    axes[0, 1].plot([1, limit], [1, limit], "--", color="grey", linewidth=1)
    axes[0, 1].invert_xaxis()
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(
        title="Official final rank vs. regression importance rank",
        xlabel="final rank (better to the right)",
        ylabel="model-score rank (better at the top)",
    )
    axes[0, 1].legend()

    weights_path = paths.regression_rerun_dir / RERUN_WEIGHTS_FILENAME
    if weights_path.is_file():
        weights = pd.read_parquet(weights_path)
        available = [sid for sid in regression_top["submission_id"] if sid in weights]
        heatmap = weights.set_index("window_end")[available].T
        max_abs_weight = np.nanmax(np.abs(heatmap.values))
        max_abs_weight = max_abs_weight if np.isfinite(max_abs_weight) and max_abs_weight > 0 else 1.0
        image = axes[1, 0].imshow(
            heatmap, aspect="auto", cmap="RdBu_r",
            vmin=-max_abs_weight, vmax=max_abs_weight,
        )
        axes[1, 0].set_yticks(
            np.arange(len(available)),
            [f"#{rank_map.get(sid, '-')} {sid[:8]}" for sid in available],
        )
        tick_positions = np.linspace(0, max(len(heatmap.columns) - 1, 0), min(6, len(heatmap.columns))).astype(int)
        dates = pd.to_datetime(heatmap.columns)
        axes[1, 0].set_xticks(tick_positions, [dates[i].strftime("%Y-%m") for i in tick_positions])
        axes[1, 0].set(title="Rolling regression weights of top finalists", xlabel="window end")
        fig.colorbar(image, ax=axes[1, 0], label="signed coefficient")
    else:
        axes[1, 0].text(.5, .5, "NOT PROVIDED\nNo rolling weight history", ha="center", va="center")
        axes[1, 0].set(xticks=[], yticks=[])

    axes[1, 1].errorbar(
        regression_top["abs_weight_mean"], y,
        xerr=regression_top["abs_weight_std"], fmt="o", color="#4C78A8",
        ecolor="#A0A0A0", capsize=3,
    )
    axes[1, 1].set_yticks(y, labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(
        title="Typical coefficient magnitude and variation",
        xlabel="mean absolute weight ± one standard deviation",
        ylabel="final rank / submission",
    )
    save("04_regression_explainability", fig)

    # 头部因子逐截面权重趋势：每 5 个排名一组，避免全因子折线相互遮挡。
    if weights_path.is_file():
        weights = pd.read_parquet(weights_path)
        weights["window_end"] = pd.to_datetime(weights["window_end"])
        factor_columns = [column for column in weights.columns if column != "window_end"]
        ranked_top_ids = [submission_id for submission_id in top_ids if submission_id in factor_columns][:20]
        group_colors = ["#B2182B", "#EF8A62", "#FDDDBC", "#67A9CF", "#2166AC"]
        fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharex=True)
        axes = axes.ravel()
        for group_index, ax in enumerate(axes):
            start = group_index * 5
            group_ids = ranked_top_ids[start:start + 5]
            for offset, factor in enumerate(group_ids):
                color = group_colors[offset]
                final_rank = rank_map.get(factor, start + offset + 1)
                ax.plot(
                    weights["window_end"], weights[factor],
                    color=color, linewidth=2.2, marker="o", markersize=3.5,
                    label=f"#{final_rank} {factor[:8]}",
                )
            ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
            ax.set(
                title=f"Final ranks {start + 1}–{start + len(group_ids)}",
                xlabel="regression window end",
                ylabel="signed coefficient",
            )
            ax.legend(fontsize=8, loc="best")
            ax.tick_params(axis="x", rotation=25)
        fig.suptitle("Rolling regression weights of top 20 finalists", fontsize=17)
        save("04b_top20_weight_trends", fig)
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(
            .5, .5, "NOT PROVIDED\nNo rolling weight history",
            ha="center", va="center", fontsize=15,
        )
        ax.set(xticks=[], yticks=[], title="Rolling regression weights of top 20 finalists")
        save("04b_top20_weight_trends", fig)

    # 样本外增量贡献与信息独特性。
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    if not incremental_summary.empty and not factor_clusters.empty:
        contribution = incremental_summary.merge(
            factor_clusters, on="submission_id", how="left", validate="one_to_one"
        ).merge(
            regression_explanation[["submission_id", "final_rank", "abs_weight_mean"]],
            on="submission_id", how="left", validate="one_to_one",
        )
        is_top = contribution["submission_id"].isin(top_ids)
        sizes = 30 + 900 * contribution["abs_weight_mean"].div(
            contribution["abs_weight_mean"].max()
        ).fillna(0)
        axes[0, 0].scatter(
            contribution.loc[~is_top, "uniqueness"],
            contribution.loc[~is_top, "mean_delta_ic"],
            s=sizes.loc[~is_top], color="#B8B8B8", alpha=.55, label="all other factors",
        )
        top_scatter = axes[0, 0].scatter(
            contribution.loc[is_top, "uniqueness"],
            contribution.loc[is_top, "mean_delta_ic"],
            s=sizes.loc[is_top], c=contribution.loc[is_top, "final_rank"],
            cmap="RdYlBu", alpha=.9, edgecolor="black", linewidth=.4, label="top finalists",
        )
        axes[0, 0].axhline(0, color="grey", linestyle="--", linewidth=.8)
        axes[0, 0].axvline(contribution["uniqueness"].median(), color="grey", linestyle="--", linewidth=.8)
        axes[0, 0].set(
            title="Out-of-sample incremental value vs. uniqueness",
            xlabel="uniqueness (1 - mean top-peer similarity)",
            ylabel="mean delta IC when factor is removed",
        )
        axes[0, 0].legend()
        fig.colorbar(top_scatter, ax=axes[0, 0], label="official final rank")

        top_contribution = contribution.loc[is_top].sort_values("final_rank")
        labels = [f"#{int(rank)} {sid[:8]}" for rank, sid in zip(
            top_contribution["final_rank"], top_contribution["submission_id"]
        )]
        y = np.arange(len(top_contribution))
        axes[0, 1].errorbar(
            top_contribution["mean_delta_ic"], y,
            xerr=top_contribution["std_delta_ic"], fmt="o", capsize=3,
            color="#4C78A8", ecolor="#A0A0A0",
        )
        axes[0, 1].axvline(0, color="grey", linestyle="--", linewidth=.8)
        axes[0, 1].set_yticks(y, labels)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set(
            title="Top finalists: incremental IC and window variation",
            xlabel="mean delta IC ± one standard deviation", ylabel="final rank / submission",
        )

        group_lookup = group_incremental_summary.set_index("cluster_id")["mean_delta_ic"] \
            if not group_incremental_summary.empty else pd.Series(dtype=float)
        top_contribution["group_delta_ic"] = top_contribution["cluster_id"].map(group_lookup)
        axes[1, 0].scatter(
            top_contribution["mean_delta_ic"], top_contribution["group_delta_ic"],
            c=top_contribution["cluster_size"], cmap="viridis", s=75, alpha=.85,
        )
        axes[1, 0].axvline(0, color="grey", linestyle="--", linewidth=.8)
        axes[1, 0].axhline(0, color="grey", linestyle="--", linewidth=.8)
        axes[1, 0].set(
            title="Individual removal vs. similarity-group removal",
            xlabel="individual mean delta IC", ylabel="group mean delta IC",
        )
        for row in top_contribution.itertuples(index=False):
            axes[1, 0].annotate(
                f"#{int(row.final_rank)}", (row.mean_delta_ic, row.group_delta_ic),
                xytext=(3, 3), textcoords="offset points", fontsize=8,
            )

        axes[1, 1].scatter(
            contribution["max_peer_similarity"], contribution["positive_delta_rate"],
            c=contribution["mean_delta_ic"], cmap="RdBu_r", s=sizes, alpha=.7,
        )
        axes[1, 1].set(
            title="Similarity and persistence of incremental contribution",
            xlabel="maximum peer similarity", ylabel="positive delta-IC window rate",
        )
    else:
        for ax in axes.ravel():
            ax.text(.5, .5, "NOT PROVIDED\nRun incremental contribution analysis", ha="center", va="center")
            ax.set(xticks=[], yticks=[])
    save("05_incremental_contribution", fig)

    # 前20名逐窗口增量 IC，每5个最终排名一组。
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharex=True)
    axes = axes.ravel()
    if not incremental_by_window.empty:
        group_colors = ["#B2182B", "#EF8A62", "#FDDDBC", "#67A9CF", "#2166AC"]
        for group_index, ax in enumerate(axes):
            start = group_index * 5
            group_ids = top_ids[start:start + 5]
            for offset, factor in enumerate(group_ids):
                values = incremental_by_window.loc[
                    incremental_by_window["item_id"].astype(str).eq(factor)
                ].sort_values("test_end")
                ax.plot(
                    pd.to_datetime(values["test_end"]), values["delta_ic"],
                    color=group_colors[offset], linewidth=2, marker="o", markersize=3.5,
                    label=f"#{rank_map.get(factor, start + offset + 1)} {factor[:8]}",
                )
            ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
            ax.set(
                title=f"Final ranks {start + 1}–{start + len(group_ids)}",
                xlabel="out-of-sample window end", ylabel="delta IC",
            )
            ax.legend(fontsize=8, loc="best")
            ax.tick_params(axis="x", rotation=25)
    else:
        for ax in axes:
            ax.text(.5, .5, "NOT PROVIDED", ha="center", va="center")
            ax.set(xticks=[], yticks=[])
    fig.suptitle("Rolling out-of-sample incremental IC of top 20 finalists", fontsize=17)
    save("05b_top20_incremental_trends", fig)

    def plot_style_group(group_summary: pd.DataFrame, title_prefix: str, figure_name: str) -> None:
        """按相同口径绘制主风格数量与平均暴露强度。"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        if not group_summary.empty and not style_detail.empty:
            classified = _dominant_style_classification(
                group_summary,
                style_detail,
                threshold=CONFIG.max_abs_style_correlation,
            )
            style_counts = classified["dominant_style"].value_counts().sort_values(ascending=False)
            sx = np.arange(len(style_counts))
            bars = axes[0].bar(sx, style_counts.values, color="#4C78A8")
            axes[0].set(
                title=f"{title_prefix}: dominant style count (total={int(style_counts.sum())})",
                ylabel="factor count",
                xticks=sx,
                xticklabels=style_counts.index,
                xlabel=f"OTHER if strongest mean abs correlation <= {CONFIG.max_abs_style_correlation:.2f}",
            )
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].bar_label(bars, padding=2)

            axes[1].scatter(
                group_summary["mean_max_abs_style_corr"],
                group_summary["mean_regression_r2"],
                color="#E45756",
                alpha=.75,
            )
            axes[1].axvline(CONFIG.max_abs_style_correlation, color="grey", linestyle="--")
            axes[1].axhline(CONFIG.max_style_regression_r2, color="grey", linestyle="--")
            axes[1].set(
                title=f"{title_prefix}: average exposure strength",
                xlabel="average daily maximum style correlation",
                ylabel="average daily exposure regression R²",
            )
        else:
            for ax in axes:
                ax.text(.5, .5, "NOT PROVIDED\nNo BARRA exposure detail", ha="center", va="center", fontsize=15)
                ax.set(xticks=[], yticks=[])
        save(figure_name, fig)

    plot_style_group(style_summary, "Full factor pool", "07_style_exposure")
    top_style_summary = style_summary.loc[style_summary["submission_id"].isin(top_ids)].copy()
    plot_style_group(top_style_summary, f"Top {len(top_ids)} finalists", "07b_top_style_exposure")

    # 相似度：保留全量背景，右侧专门展示涉及头部选手的最高相似关系。
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    if similarity.empty:
        for ax in axes:
            ax.text(.5, .5, "No comparable factor pairs", ha="center", va="center", fontsize=15)
            ax.set(xticks=[], yticks=[])
    else:
        axes[0].hist(similarity["abs_correlation"], bins=30, color="#4C78A8")
        axes[0].axvline(high_correlation, linestyle="--", color="#E45756", label="threshold")
        axes[0].legend()
        top_pairs = similarity.loc[
            similarity["submission_id_1"].isin(top_ids) | similarity["submission_id_2"].isin(top_ids)
        ].head(max(20, len(top_ids)))
        labels = [
            f"#{rank_map.get(a, '-')}:{a[:5]} / #{rank_map.get(b, '-')}:{b[:5]}"
            for a, b in zip(top_pairs["submission_id_1"], top_pairs["submission_id_2"])
        ]
        y = np.arange(len(top_pairs))
        axes[1].barh(y, top_pairs["abs_correlation"], color="#4C78A8")
        axes[1].set_yticks(y, labels, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].axvline(high_correlation, linestyle="--", color="#E45756")
        axes[1].set(title="Highest similarities involving top finalists", xlabel="absolute correlation")
    axes[0].set(title="All factor pairs", xlabel="absolute correlation", ylabel="pairs")
    save("08_factor_similarity", fig)
    return figures


def generate_markdown_report(
    paths: CheckPaths = PATHS,
    output_path: str | Path | None = None,
    *,
    high_correlation: float = CONFIG.high_correlation,
    max_similarity_samples: int = CONFIG.max_similarity_samples,
    top_n: int = CONFIG.report_top_n,
    style_exposure_dir: str | Path | None = None,
) -> Path:
    """运行全部检查，生成图表和 Markdown，并返回报告路径。"""
    _require_inputs(paths)
    output = Path(output_path) if output_path else paths.report_path
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = read_summary(paths)
    final = read_final(paths)
    participants = _participant_lookup(paths)
    score_problems = check_score_consistency(paths, display=False)
    rank_conflicts = analyze_rank_conflicts(paths, display=False)
    team_private_report = build_team_private_report(paths)
    ab_sensitivity = analyze_ab_weight_sensitivity(paths, display=False)
    a_metric_sensitivity = analyze_a_metric_sensitivity(paths, display=False)
    cross_similarity_path = paths.incremental_dir / FACTOR_SIMILARITY_SUMMARY_FILENAME
    using_cross_similarity = cross_similarity_path.is_file()
    if using_cross_similarity:
        similarity = pd.read_csv(
            cross_similarity_path,
            dtype={"submission_id_1": str, "submission_id_2": str},
        ).rename(columns={
            "mean_correlation": "pearson",
            "mean_abs_correlation": "abs_correlation",
            "mean_overlap": "overlap",
        })
        similarity["high_similarity"] = similarity["abs_correlation"].ge(high_correlation)
    else:
        similarity = analyze_factor_similarity(
            paths,
            high_correlation=high_correlation,
            max_samples=max_similarity_samples,
            display=False,
        )
    regression_explanation = analyze_regression_explainability(paths, display=False)
    incremental_summary_path = paths.incremental_dir / INCREMENTAL_SUMMARY_FILENAME
    incremental_by_window_path = paths.incremental_dir / INCREMENTAL_BY_WINDOW_FILENAME
    factor_clusters_path = paths.incremental_dir / FACTOR_CLUSTERS_FILENAME
    group_incremental_summary_path = paths.incremental_dir / GROUP_INCREMENTAL_SUMMARY_FILENAME
    incremental_summary = (
        pd.read_csv(incremental_summary_path, dtype={"submission_id": str})
        if incremental_summary_path.is_file() else pd.DataFrame()
    )
    incremental_by_window = (
        pd.read_parquet(incremental_by_window_path)
        if incremental_by_window_path.is_file() else pd.DataFrame()
    )
    factor_clusters = (
        pd.read_csv(factor_clusters_path, dtype={"submission_id": str})
        if factor_clusters_path.is_file() else pd.DataFrame()
    )
    group_incremental_summary = (
        pd.read_csv(group_incremental_summary_path)
        if group_incremental_summary_path.is_file() else pd.DataFrame()
    )
    style_dir = (
        Path(style_exposure_dir).expanduser().resolve()
        if style_exposure_dir is not None
        else paths.style_exposure_dir
    )
    style_summary_path = style_dir / STYLE_EXPOSURE_SUMMARY_FILENAME
    style_detail_path = style_dir / STYLE_EXPOSURE_DETAIL_FILENAME
    style_metadata_path = style_dir / STYLE_EXPOSURE_METADATA_FILENAME
    style_figure_path = style_dir / STYLE_EXPOSURE_FIGURE_FILENAME
    style_summary = pd.DataFrame()
    style_detail = pd.DataFrame()
    style_metadata = None
    if style_summary_path.is_file():
        style_summary = pd.read_csv(style_summary_path, dtype={"submission_id": str})
        if "style_exposure_warning" in style_summary:
            style_summary["style_exposure_warning"] = (
                style_summary["style_exposure_warning"].astype(str).str.lower().eq("true")
            )
    if style_metadata_path.is_file():
        style_metadata = json.loads(style_metadata_path.read_text(encoding="utf-8"))
    if style_detail_path.is_file():
        style_detail = pd.read_csv(style_detail_path, dtype={"submission_id": str})

    success_count = int(summary["status"].eq("success").sum())
    high_similarity = similarity.loc[similarity["high_similarity"]].copy()
    style_missing = int(style_summary.empty)
    style_warnings = (
        int(style_summary["style_exposure_warning"].sum())
        if "style_exposure_warning" in style_summary else 0
    )
    blocking_count = len(score_problems)
    review_count = len(high_similarity) + style_missing + style_warnings
    status = "BLOCK" if blocking_count else ("REVIEW" if review_count else "PASS")

    ranked_final = final.loc[final["final_score"].ge(0)].sort_values(
        "final_score", ascending=False
    ).reset_index(drop=True)
    ranked_final.insert(0, "rank", range(1, len(ranked_final) + 1))
    top_final = ranked_final.head(top_n).copy()
    top_ids = top_final["submission_id"].astype(str).tolist()
    rank_map = dict(zip(ranked_final["submission_id"].astype(str), ranked_final["rank"]))

    figures = _generate_report_figures(
        paths=paths, output=output, summary=summary, final=final, score_problems=score_problems,
        rank_conflicts=rank_conflicts, ab_sensitivity=ab_sensitivity,
        a_metric_sensitivity=a_metric_sensitivity,
        style_summary=style_summary, style_detail=style_detail,
        regression_explanation=regression_explanation,
        incremental_summary=incremental_summary,
        incremental_by_window=incremental_by_window,
        factor_clusters=factor_clusters,
        group_incremental_summary=group_incremental_summary,
        similarity=similarity, high_correlation=high_correlation,
        top_ids=top_ids, rank_map=rank_map,
    )

    top_comparison = rank_conflicts.loc[
        rank_conflicts["submission_id"].isin(top_ids),
        ["final_rank", "submission_id", "a_score", "a_rank", "b_score", "b_rank", "final_score"],
    ].sort_values("final_rank")
    top_comparison = top_comparison.merge(
        ab_sensitivity[["rank_span"]],
        left_on="submission_id", right_index=True, how="left",
    ).merge(
        a_metric_sensitivity[["max_abs_rank_change"]],
        left_on="submission_id", right_index=True, how="left",
    )
    score_by_rank = ranked_final.set_index("rank")["final_score"]
    top_comparison["b_minus_a"] = top_comparison["b_score"] - top_comparison["a_score"]
    top_comparison["score_driver"] = np.where(
        top_comparison["b_minus_a"].gt(0), "B 高于 A",
        np.where(top_comparison["b_minus_a"].lt(0), "A 高于 B", "A/B 持平"),
    )
    top_comparison["b_rank_minus_a_rank"] = top_comparison["b_rank"] - top_comparison["a_rank"]
    top_comparison["score_gap_to_prev"] = top_comparison.apply(
        lambda row: score_by_rank.get(int(row["final_rank"]) - 1, np.nan) - row["final_score"], axis=1
    )
    top_comparison["score_gap_to_next"] = top_comparison.apply(
        lambda row: row["final_score"] - score_by_rank.get(int(row["final_rank"]) + 1, np.nan), axis=1
    )
    high_sensitivity = (
        top_comparison["rank_span"].ge(MAX_AB_RANK_GAP)
        | top_comparison["max_abs_rank_change"].ge(5)
    )
    medium_sensitivity = (
        top_comparison["rank_span"].ge(MAX_AB_RANK_GAP / 2)
        | top_comparison["max_abs_rank_change"].ge(3)
    )
    top_comparison["sensitivity"] = np.select(
        [high_sensitivity, medium_sensitivity], ["高", "中"], default="低"
    )
    top_comparison = top_comparison[
        [
            "final_rank", "submission_id", "a_score", "b_score", "final_score",
            "b_minus_a", "score_driver", "a_rank", "b_rank", "b_rank_minus_a_rank",
            "score_gap_to_prev", "score_gap_to_next", "rank_span",
            "max_abs_rank_change", "sensitivity",
        ]
    ]
    top_comparison = top_comparison.merge(participants, on="submission_id", how="left")

    top_comparison["participant"] = top_comparison["participant"].fillna("（未匹配）")
    top_comparison["schools"] = top_comparison["schools"].fillna("（未匹配）")
    largest_score_divergence = top_comparison.loc[top_comparison["b_minus_a"].abs().idxmax()]
    closest_competition = top_comparison.dropna(subset=["score_gap_to_next"]).loc[
        top_comparison.dropna(subset=["score_gap_to_next"])["score_gap_to_next"].idxmin()
    ]
    b_driven_count = int(top_comparison["b_minus_a"].gt(0).sum())
    high_sensitivity_count = int(top_comparison["sensitivity"].eq("高").sum())
    participant_rows = []
    for participant in team_private_report["participants"]:
        correlation = participant["correlation_summary"]
        top_pair = correlation.get("top_pair") or {}
        participant_rows.append({
            "私榜排名": int(participant["private_rank"]) if participant["private_rank"] is not None else pd.NA,
            "类型": "团队" if participant["participant_type"] == "team" else "个人",
            "队伍/个人": participant["participant_name"],
            "提交数": participant["submission_count"],
            "私榜最高分": participant["best_private_score"],
            "公榜排名": participant["public_rank"],
            "公榜分": participant["public_score"],
            "平均绝对相关": correlation["mean_abs_correlation"],
            "最大绝对相关": correlation["max_abs_correlation"],
            "高相关对": correlation["high_correlation_pair_count"],
            "最高相关提交对": (
                f"{str(top_pair.get('submission_id_1', ''))[:8]} ↔ "
                f"{str(top_pair.get('submission_id_2', ''))[:8]}"
                if top_pair else ""
            ),
        })
    participant_diagnostic = pd.DataFrame(participant_rows)
    participants_with_multiple_submissions = participant_diagnostic.loc[
        participant_diagnostic["提交数"].gt(1)
    ]
    participants_with_high_correlation = participant_diagnostic.loc[
        participant_diagnostic["高相关对"].gt(0)
    ]
    valid_participant_correlations = participants_with_multiple_submissions.dropna(
        subset=["最大绝对相关"]
    )
    highest_participant_correlation = (
        valid_participant_correlations.loc[valid_participant_correlations["最大绝对相关"].idxmax()]
        if not valid_participant_correlations.empty else None
    )
    participant_display = participant_diagnostic.head(top_n)
    submission_score_rows = []
    for participant in team_private_report["participants"]:
        for submission in participant["submissions"]:
            private = submission["private_score_detail"]
            public = submission["public_score_detail"]
            private_b = submission["private_b_detail"]
            public_b = submission["public_b_detail"]
            submission_score_rows.append({
                "队伍/个人": participant["participant_name"],
                "submission_id": submission["submission_id"],
                "榜单": "私榜",
                "IC均值": private.get("ic_mean"),
                "ICIR": private.get("ic_ir"),
                "夏普": private.get("sharpe_ratio"),
                "压力ICIR": private.get("stress_ic_ir"),
                "A分": private.get("a_score"),
                "B分": private.get("b_score"),
                "B模型得分": private_b.get("model_score"),
                "B平均绝对权重": private_b.get("abs_weight_mean"),
                "B权重标准差": private_b.get("abs_weight_std"),
                "B入选率": private_b.get("selection_rate"),
                "最终分": private.get("final_score"),
            })
            submission_score_rows.append({
                "队伍/个人": participant["participant_name"],
                "submission_id": submission["submission_id"],
                "榜单": "公榜",
                "IC均值": public.get("ic_mean"),
                "ICIR": public.get("ic_ir"),
                "夏普": public.get("sharpe_ratio"),
                "压力ICIR": public.get("stress_ic_ir"),
                "A分": public.get("a_score"),
                "B分": public.get("b_score"),
                "B模型得分": public_b.get("model_score"),
                "B平均绝对权重": public_b.get("abs_weight_mean"),
                "B权重标准差": public_b.get("abs_weight_std"),
                "B入选率": public_b.get("selection_rate"),
                "最终分": public.get("final_score"),
            })
    submission_score_detail = pd.DataFrame(submission_score_rows)
    top_diagnostic = pd.DataFrame({
        "排名": top_comparison["final_rank"].astype(int),
        "队伍/个人": top_comparison["participant"],
        "涉及学校": top_comparison["schools"],
        "submission": top_comparison["submission_id"].str[:8],
        "A分": top_comparison["a_score"].map(lambda value: f"{value:.4f}"),
        "B分": top_comparison["b_score"].map(lambda value: f"{value:.4f}"),
        "最终分": top_comparison["final_score"].map(lambda value: f"{value:.4f}"),
        "B-A": top_comparison["b_minus_a"].map(lambda value: f"{value:+.4f}"),
        "A/B名次": top_comparison.apply(
            lambda row: f"{int(row['a_rank'])}/{int(row['b_rank'])}", axis=1
        ),
        "最近相邻分差": top_comparison[["score_gap_to_prev", "score_gap_to_next"]].min(axis=1).map(
            lambda value: f"{value:.4f}"
        ),
        "敏感性": top_comparison["sensitivity"],
    })
    similarity_top = similarity.loc[
        similarity["submission_id_1"].isin(top_ids) | similarity["submission_id_2"].isin(top_ids)
    ].copy()
    similarity_top.insert(0, "rank_1", similarity_top["submission_id_1"].map(rank_map))
    similarity_top.insert(2, "rank_2", similarity_top["submission_id_2"].map(rank_map))
    similarity_top = similarity_top[
        ["rank_1", "submission_id_1", "rank_2", "submission_id_2", "pearson", "overlap",
         "abs_correlation", "high_similarity"]
    ].head(top_n)

    regression_top = regression_explanation.loc[
        regression_explanation["submission_id"].isin(top_ids)
    ].sort_values("final_rank").merge(participants, on="submission_id", how="left")
    regression_top["participant"] = regression_top["participant"].fillna("（未匹配）")
    regression_diagnostic = pd.DataFrame({
        "最终排名": regression_top["final_rank"].astype(int),
        "队伍/个人": regression_top["participant"],
        "submission": regression_top["submission_id"].str[:8],
        "回归重要性排名": regression_top["model_rank"].astype("Int64"),
        "模型得分": regression_top["model_score"],
        "重要性占比": regression_top["importance_share"],
        "入选率": regression_top["selection_rate"],
        "平均绝对权重": regression_top["abs_weight_mean"],
        "权重波动": regression_top["abs_weight_std"],
    })
    if "dominant_direction" in regression_top:
        regression_diagnostic["主要方向"] = regression_top["dominant_direction"]
        regression_diagnostic["方向一致率"] = regression_top["direction_consistency"]
    strongest_regression = regression_top.loc[regression_top["model_score"].idxmax()]
    most_persistent = regression_top.loc[regression_top["selection_rate"].idxmax()]
    top_importance_share = regression_top["importance_share"].sum()

    incremental_available = not incremental_summary.empty and not factor_clusters.empty
    incremental_diagnostic = pd.DataFrame()
    incremental_observations: list[str] = []
    if incremental_available:
        incremental_top = (
            incremental_summary.merge(
                factor_clusters, on="submission_id", how="left", validate="one_to_one"
            )
            .loc[lambda frame: frame["submission_id"].isin(top_ids)]
            .merge(
                ranked_final[["rank", "submission_id"]], on="submission_id", how="left"
            )
            .merge(participants, on="submission_id", how="left")
        )
        if not group_incremental_summary.empty:
            group_values = group_incremental_summary[["cluster_id", "mean_delta_ic"]].rename(
                columns={"mean_delta_ic": "group_mean_delta_ic"}
            )
            incremental_top = incremental_top.merge(group_values, on="cluster_id", how="left")
        else:
            incremental_top["group_mean_delta_ic"] = np.nan
        incremental_top["category"] = np.select(
            [
                incremental_top["mean_delta_ic"].gt(0)
                & incremental_top["uniqueness"].ge(incremental_top["uniqueness"].median()),
                incremental_top["mean_delta_ic"].gt(0),
                incremental_top["uniqueness"].ge(incremental_top["uniqueness"].median()),
            ],
            ["独特且有增量", "有增量但可替代", "独特但增量有限"],
            default="相似且增量有限",
        )
        incremental_top = incremental_top.sort_values("rank")
        incremental_diagnostic = pd.DataFrame({
            "最终排名": incremental_top["rank"].astype(int),
            "队伍/个人": incremental_top["participant"].fillna("（未匹配）"),
            "submission": incremental_top["submission_id"].str[:8],
            "平均增量IC": incremental_top["mean_delta_ic"],
            "正贡献窗口占比": incremental_top["positive_delta_rate"],
            "增量IC波动": incremental_top["std_delta_ic"],
            "独特性": incremental_top["uniqueness"],
            "最大同类相似度": incremental_top["max_peer_similarity"],
            "相似组": incremental_top["cluster_id"],
            "组规模": incremental_top["cluster_size"],
            "整组增量IC": incremental_top["group_mean_delta_ic"],
            "特征分类": incremental_top["category"],
        })
        best_incremental = incremental_top.loc[incremental_top["mean_delta_ic"].idxmax()]
        most_unique = incremental_top.loc[incremental_top["uniqueness"].idxmax()]
        incremental_observations = [
            f"- 前 {len(incremental_top)} 名中，平均样本外增量 IC 为正的有 **{int(incremental_top['mean_delta_ic'].gt(0).sum())}** 个。",
            f"- 头部中增量贡献最高的是最终第 **{int(best_incremental['rank'])}** 名，平均 ΔIC 为 **{best_incremental['mean_delta_ic']:.6f}**，正贡献窗口占比为 **{best_incremental['positive_delta_rate']:.2%}**。",
            f"- 头部中信息独特性最高的是最终第 **{int(most_unique['rank'])}** 名，独特性为 **{most_unique['uniqueness']:.4f}**，最相似同类相关为 **{most_unique['max_peer_similarity']:.4f}**。",
        ]

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = paths.run_dir.name
    lines = [
        "# 私榜人工复核自动报告",
        "",
        f"> 本报告由 `manual_checks.generate_markdown_report` 自动生成。生成时间：`{generated_at}`。",
        "",
        "## 1. 总体状态",
        "",
        f"**{status}**",
        "",
        "| 项目 | 结果 |",
        "| --- | ---: |",
        f"| Run ID | `{run_id}` |",
        f"| 总提交数 | {len(summary)} |",
        f"| 成功提交数 | {success_count} |",
        f"| 成功率 | {success_count / len(summary):.2%} |",
        f"| 成绩复算异常 | {len(score_problems)} |",
        f"| 高相似因子对（≥ {high_correlation:.2f}） | {len(high_similarity)} |",
        f"| BARRA 风格暴露告警 | {style_warnings if not style_summary.empty else 'NOT_PROVIDED'} |",
        "",
        "状态规则：成绩复算异常时为 `BLOCK`；不存在阻断项但有高相似、风格暴露告警或风格结果缺失时为 `REVIEW`；否则为 `PASS`。联合回归章节用于解释因子贡献，不作为问题判定项。",
        "",
        "### 怎么分析",
        "",
        "先确认成功率和输入规模是否符合本次运行预期，再按 `BLOCK → REVIEW → PASS` 的顺序处理。总状态只用于分流，最终结论必须结合后续各节的分布图、阈值命中和异常明细。",
        "",
        *_chart(figures["01_submission_status"], output, "提交状态分布"),
        "## 2. 成绩复算",
        "",
        "### 怎么分析",
        "",
        "将已落盘的 A 分、B 分和最终分按正式评分代码重新计算，逐提交比较绝对差值。任一分数字段差值超过 `0.000000001` 都应阻断放榜；即使异常数为 0，也要查看全量分数曲线是否存在断层、异常聚集、负分或 A/B 分尺度明显不一致。",
        "",
        f"正式成绩复算异常数：**{len(score_problems)}**。",
        "",
        *_chart(figures["02_score_consistency"], output, "全量正式分数分布"),
        _table(score_problems, limit=top_n),
        "",
        "## 3. 参赛主体私榜提交与内部相关性",
        "",
        "### 怎么分析",
        "",
        "每个团队或个人按其所有入围私榜 submission 统计提交数，并以主体内最高 `final_score` 生成私榜排名。"
        "相关性先在每个交易日对两个 submission 的股票截面因子值计算 Pearson 相关，"
        "再跨交易日计算平均绝对相关；主体平均值和最大值均基于同一主体内部所有 submission 两两组合。"
        f"绝对相关达到 `{high_correlation:.2f}` 的组合记为高相关对。正负相关都代表较强线性关系，"
        "因此主体汇总使用绝对值；相关方向仍保留在私榜 JSON 的最高相关提交对中。",
        "",
        "### 自动观察",
        "",
        f"- 共 **{team_private_report['participant_count']}** 个参赛主体（"
        f"**{team_private_report['team_count']}** 个团队、**{team_private_report['individual_count']}** 个个人），"
        f"共 **{team_private_report['submission_count']}** 份提交；"
        f"其中 **{len(participants_with_multiple_submissions)}** 个主体有至少 2 个 submission，可计算主体内相关性。",
        f"- 有 **{len(participants_with_high_correlation)}** 个主体至少存在一对平均绝对相关不低于 "
        f"**{high_correlation:.2f}** 的 submission。",
        (
            f"- 主体内最大相关性最高的是 **{highest_participant_correlation['队伍/个人']}**，最大平均绝对相关为 "
            f"**{highest_participant_correlation['最大绝对相关']:.6f}**，对应提交对 "
            f"`{highest_participant_correlation['最高相关提交对']}`。"
            if highest_participant_correlation is not None else
            "- 当前没有可用的主体内 submission 相关性结果。"
        ),
        "",
        f"### 私榜前 {len(participant_display)} 个参赛主体概览",
        "",
        _table(participant_display),
        "",
        "### 全部参赛主体 submission 公榜/私榜逐项得分",
        "",
        _table(submission_score_detail),
        "",
        "公榜 A 项明细读取自 `public_leaderboard_dir/submissions_summary.csv`，"
        "B 项明细读取自 `public_leaderboard_dir/leaderboard_reg.csv`。"
        "完整逐 submission 数据也保存在 `team_private_leaderboard.csv`；"
        "精简后的团队摘要和 submission 得分保存在 `team_private_leaderboard.json`。",
        "",
        "## 4. 头部榜单得分结构与竞争区间",
        "",
        "### 怎么分析",
        "",
        f"本节聚焦最终榜前 {len(top_final)} 名。核心图将 A 分和 B 分作为区间两端，将最终分标在加权落点上，用一张图判断每位选手更依赖 A 还是 B。随后从相邻名次分差识别紧密竞争区间，并用排名敏感性字段辅助判断结果是否稳健。",
        "",
        "### 自动观察",
        "",
        f"- 前 {len(top_final)} 名中，B 分高于 A 分的有 **{b_driven_count}** 人，A 分高于 B 分的有 **{len(top_final) - b_driven_count}** 人。",
        f"- A/B 得分差最大的是第 **{int(largest_score_divergence['final_rank'])}** 名，B-A 为 **{largest_score_divergence['b_minus_a']:.6f}**。",
        f"- 最紧密的相邻竞争发生在第 **{int(closest_competition['final_rank'])}** 名与下一名之间，最终分差为 **{closest_competition['score_gap_to_next']:.6f}**。",
        f"- 按当前规则，头部选手中排名敏感性标记为“高”的有 **{high_sensitivity_count}** 人。",
        "",
        *_chart(figures["03_top_ranking_comparison"], output, "头部选手 A/B 得分区间与最终分落点"),
        "### 头部选手诊断表",
        "",
        _table(top_diagnostic),
        "",
        "字段说明：`submission` 展示完整 ID 的前 8 位；`B-A` 为正表示 B 分更高，为负表示 A 分更高；`A/B名次` 依次为 A 排名和 B 排名；`最近相邻分差` 取与前后名次分差中的较小值；`敏感性` 综合 A/B 权重扫描和 A 指标留一法标记为高、中、低。",
        "",
        "## 5. 联合回归中的因子解释性",
        "",
        "### 怎么分析",
        "",
        f"本节聚焦最终榜前 {len(top_final)} 名，解释每个因子进入全因子联合 ElasticNet 后的模型内重要性和权重稳定性。`模型得分`表示平均绝对权重相对其波动的大小；`入选率`表示滚动窗口中非零系数出现的比例；`平均绝对权重`表示典型影响强度；逐期有符号权重展示影响方向是否持续一致。严格的独立增量信息由下一节的样本外删除实验判断。",
        "",
        "### 自动观察",
        "",
        f"- 头部因子合计占全部非负模型得分的 **{top_importance_share:.2%}**。",
        f"- 头部中联合回归重要性最高的是最终第 **{int(strongest_regression['final_rank'])}** 名，模型得分为 **{strongest_regression['model_score']:.6f}**，全池回归重要性排名第 **{int(strongest_regression['model_rank'])}**。",
        f"- 头部中入选最持续的是最终第 **{int(most_persistent['final_rank'])}** 名，滚动窗口入选率为 **{most_persistent['selection_rate']:.2%}**。",
        "",
        *_chart(figures["04_regression_explainability"], output, "头部因子的联合回归重要性、排名对应与逐期权重"),
        "### 前 20 名因子逐截面回归权重趋势",
        "",
        "按最终排名每 5 个因子分为一组，分别展示第 1–5、6–10、11–15 和 16–20 名。每组统一使用从深红、橙红、浅色到浅蓝、深蓝的五级渐变，对应组内排名由高到低。每条折线代表一个因子在各滚动回归截面上的有符号权重；权重跨越零线表示回归方向发生改变，曲线长期接近零表示该因子在相应阶段贡献较弱或未被模型选择。",
        "",
        *_chart(figures["04b_top20_weight_trends"], output, "最终榜前20名因子的分组回归权重趋势"),
        "### 头部因子回归解释表",
        "",
        _table(regression_diagnostic),
        "",
        "阅读顺序建议：先看模型得分和重要性排名，了解因子在联合模型中的权重表现；再看入选率和权重波动；最后看有符号权重的方向切换。是否具有不可替代的预测信息，应结合下一节的样本外增量贡献和相似因子组结果判断。",
        "",
        "## 6. 样本外增量贡献与信息独特性",
        "",
        "### 怎么分析",
        "",
        f"使用过去 {CONFIG.incremental_train_window} 个交易日训练完整 ElasticNet，并在随后 {CONFIG.incremental_test_window} 个交易日测试；随后分别删除单个因子和整个相似因子组重新拟合。`平均增量IC`为完整模型样本外 IC 减去删除后的样本外 IC，正值表示该因子或因子组提供了不可被剩余因子完全替代的未来预测信息。独特性定义为 `1 - 最相似若干同类因子的平均截面绝对相关`。",
        "",
        "### 自动观察",
        "",
        *(incremental_observations if incremental_observations else [
            "- **NOT_PROVIDED**：尚未生成样本外增量贡献结果，需要在可查询未来收益数据的环境运行 `analyze_incremental_contribution`。"
        ]),
        "",
        *_chart(figures["05_incremental_contribution"], output, "因子样本外增量贡献、独特性与相似组替代关系"),
        "### 前20名逐窗口增量贡献",
        "",
        "每条曲线表示删除该因子后，完整模型相对于删除模型的样本外 IC 改善。曲线持续高于零表示因子在多个未来窗口中稳定提供增量信息；围绕零波动表示容易被其他因子替代；持续低于零则意味着删除该因子后模型反而更好。",
        "",
        *_chart(figures["05b_top20_incremental_trends"], output, "最终榜前20名的滚动样本外增量IC"),
        "### 前20名增量贡献与独特性明细",
        "",
        _table(incremental_diagnostic) if incremental_available else "_尚未生成增量贡献结果。_",
        "",
        "## 7. BARRA 风格暴露",
        "",
        "本节统一使用更直观的**抽样交易日平均值**：风格相关表示各抽样日截面绝对相关的平均值，"
        "回归 R² 表示各抽样日暴露解释度的平均值。每个因子只归入平均绝对相关最高的主风格；"
        f"最高值不超过 `{CONFIG.max_abs_style_correlation:.2f}` 时归为 `OTHER`。",
        "",
        "### 全部因子池",
        "",
        *_chart(figures["07_style_exposure"], output, "全部因子池主风格分布与平均暴露强度"),
        f"### 最终榜前 {len(top_final)} 名",
        "",
        *_chart(figures["07b_top_style_exposure"], output, f"最终榜前 {len(top_final)} 名主风格分布与平均暴露强度"),
    ]

    if style_summary.empty:
        lines.extend([
            "**NOT_PROVIDED**：本地未找到云端风格暴露结果包，本项尚未验证。",
            "",
            f"预期目录：`{style_dir}`",
            "",
        ])
    else:
        style_top = style_summary.loc[style_summary["submission_id"].isin(top_ids), [
            "submission_id", "mean_regression_r2", "mean_max_abs_style_corr",
        ]].copy()
        final_score_lookup = ranked_final.set_index("submission_id")["final_score"]
        style_top.insert(0, "最终得分排名", style_top["submission_id"].map(rank_map))
        style_top.insert(2, "最终得分", style_top["submission_id"].map(final_score_lookup))
        top_style_rows = []
        for submission_id, group in style_detail.loc[
            style_detail["submission_id"].isin(top_ids)
        ].groupby("submission_id"):
            ranked = group.sort_values("mean_abs_correlation", ascending=False).head(3)
            labels = [
                f"{row.style}({'+' if row.mean_correlation >= 0 else '-'}) {row.mean_abs_correlation:.3f}"
                for row in ranked.itertuples(index=False)
            ]
            dominant = (
                labels[0]
                if labels and ranked.iloc[0]["mean_abs_correlation"] > CONFIG.max_abs_style_correlation
                else "OTHER"
            )
            top_style_rows.append({
                "submission_id": submission_id,
                "主要风格": dominant,
                "前三风格": "；".join(labels),
            })
        style_top = style_top.merge(
            pd.DataFrame(top_style_rows, columns=["submission_id", "主要风格", "前三风格"]),
            on="submission_id",
            how="left",
        )
        style_top["明显风格"] = style_top["主要风格"].ne("OTHER")
        style_top = style_top.rename(columns={
            "mean_regression_r2": "平均暴露回归R²",
            "mean_max_abs_style_corr": "平均最大风格相关",
        }).sort_values(["最终得分", "最终得分排名"], ascending=[False, True])
        lines.extend([
            _table(style_top),
            "",
            f"全部因子池共 {len(style_summary)} 个因子；前 {len(top_final)} 名图使用完全相同的分类与阈值口径。",
            "",
        ])

    lines.extend([
        "## 8. 因子相似度",
        "",
        "### 怎么分析",
        "",
        f"全量绝对相关分布用于提供背景，明细只展示至少一端属于最终榜前 {len(top_final)} 名的因子对，并按绝对相关性从高到低排列。重点看头部选手彼此之间、以及头部选手与全池其他因子的最高相似关系；高相关仍需结合参赛方关系、代码实现和经济含义判断。",
        "",
        (
            f"相似度先在每个交易日做截面 Pearson 相关，再对每日绝对相关取均值；高相似阈值为 `{high_correlation:.2f}`，命中 **{len(high_similarity)}** 对。"
            if using_cross_similarity else
            f"从最多 {max_similarity_samples:,} 个样本计算两两 Pearson 相关。高相似阈值为 `{high_correlation:.2f}`，命中 **{len(high_similarity)}** 对。"
        ),
        "",
        *_chart(figures["08_factor_similarity"], output, "全量相似度背景与头部选手最高相似关系"),
        f"### 涉及最终榜前 {len(top_final)} 名的最高相似因子对",
        "",
        _table(similarity_top),
        "",
        "## 9. 自动建议",
        "",
    ])

    recommendations = []
    if blocking_count:
        recommendations.append("停止放榜，先处理成绩复算异常。")
    if len(high_similarity):
        recommendations.append(f"对 {len(high_similarity)} 组高相似因子执行全量相关、参赛方关系和代码人工核验。")
    if style_summary.empty:
        recommendations.append("从云端下载风格暴露结果包后重新生成报告；当前 BARRA 中性化效果尚未验证。")
    elif style_warnings:
        recommendations.append(f"复核 {style_warnings} 个风格暴露告警因子的中性化过程和样本覆盖。")
    if int(ab_sensitivity["rank_span"].ge(MAX_AB_RANK_GAP).sum()):
        recommendations.append("确认 A/B 权重已经在赛事规则中固定，并专项查看获奖线附近的局部权重敏感性。")
    if not recommendations:
        recommendations.append("未发现需处置项，可以进入人工签署阶段。")
    lines.extend(f"{index}. {item}" for index, item in enumerate(recommendations, start=1))
    lines.extend([
        "",
        "## 10. 输入路径",
        "",
        f"- 运行目录：`{paths.run_dir}`",
        f"- 预处理目录：`{paths.prepared_dir}`",
        f"- 私榜代码目录：`{paths.private_code_dir}`",
        "",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown 报告已生成：{output}")
    return output


def main() -> None:
    """直接使用 ``config.py`` 的统一配置生成报告。"""
    CONFIG.activate()
    generate_markdown_report(CONFIG.paths)


if __name__ == "__main__":
    main()
