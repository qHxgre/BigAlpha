"""处理后因子的相似度检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import show
from .config import (
    CONFIG,
    FACTOR_CLUSTERS_FILENAME,
    FACTOR_SIMILARITY_DAILY_FILENAME,
    FACTOR_SIMILARITY_SUMMARY_FILENAME,
    PATHS,
    CheckPaths,
)


def _build_similarity_clusters(
    factors: list[str], pair_summary: pd.DataFrame, *, threshold: float, top_peers: int
) -> pd.DataFrame:
    """按相似度阈值连接因子，并生成连通分量及独特性指标。"""
    parent = {factor: factor for factor in factors}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    connected = pair_summary.loc[pair_summary["mean_abs_correlation"].ge(threshold)]
    for row in connected.itertuples(index=False):
        union(str(row.submission_id_1), str(row.submission_id_2))

    groups: dict[str, list[str]] = {}
    for factor in factors:
        groups.setdefault(find(factor), []).append(factor)
    ordered_groups = sorted(groups.values(), key=lambda values: (-len(values), values[0]))
    cluster_by_factor = {
        factor: cluster_id
        for cluster_id, members in enumerate(ordered_groups, start=1)
        for factor in members
    }
    peer_values: dict[str, list[float]] = {factor: [] for factor in factors}
    best_peer: dict[str, tuple[str | None, float]] = {
        factor: (None, np.nan) for factor in factors
    }
    for row in pair_summary.itertuples(index=False):
        left, right = str(row.submission_id_1), str(row.submission_id_2)
        value = float(row.mean_abs_correlation)
        peer_values[left].append(value)
        peer_values[right].append(value)
        if pd.isna(best_peer[left][1]) or value > best_peer[left][1]:
            best_peer[left] = (right, value)
        if pd.isna(best_peer[right][1]) or value > best_peer[right][1]:
            best_peer[right] = (left, value)

    rows = []
    for factor in factors:
        peers = sorted(peer_values[factor], reverse=True)
        top_mean = float(np.mean(peers[:top_peers])) if peers else np.nan
        peer, maximum = best_peer[factor]
        cluster_id = cluster_by_factor[factor]
        members = ordered_groups[cluster_id - 1]
        rows.append({
            "submission_id": factor,
            "cluster_id": cluster_id,
            "cluster_size": len(members),
            "cluster_members": ",".join(members),
            "most_similar_factor": peer,
            "max_peer_similarity": maximum,
            "mean_top_peer_similarity": top_mean,
            "uniqueness": 1.0 - top_mean if np.isfinite(top_mean) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["cluster_size", "cluster_id", "max_peer_similarity"],
        ascending=[False, True, False],
    )


def analyze_cross_sectional_similarity(
    paths: CheckPaths = PATHS,
    *,
    cluster_threshold: float = CONFIG.similarity_cluster_threshold,
    high_threshold: float = CONFIG.similarity_high_threshold,
    top_peers: int = CONFIG.similarity_top_peers,
    save: bool = True,
    display: bool = True,
) -> dict[str, pd.DataFrame]:
    """按交易日计算因子截面相关，汇总相似性并划分可替代因子组。"""
    pool = pd.read_parquet(paths.factor_pool_path)
    pool["date"] = pd.to_datetime(pool["date"])
    factor_cols = [str(column) for column in pool.columns if column not in {"date", "instrument"}]
    pool = pool.rename(columns={column: str(column) for column in pool.columns})
    factor_ids = np.asarray(factor_cols)
    left, right = np.triu_indices(len(factor_ids), k=1)
    daily_frames = []
    for date, frame in pool.groupby("date", sort=True):
        values = frame[factor_cols].apply(pd.to_numeric, errors="coerce")
        correlation = values.corr().to_numpy()[left, right]
        valid = values.notna().to_numpy(dtype=np.int32)
        overlap = (valid.T @ valid)[left, right]
        valid_pair = np.isfinite(correlation)
        if not valid_pair.any():
            continue
        daily_frames.append(pd.DataFrame({
            "date": date,
            "submission_id_1": factor_ids[left][valid_pair],
            "submission_id_2": factor_ids[right][valid_pair],
            "pearson": correlation[valid_pair],
            "overlap": overlap[valid_pair],
        }))
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(
        columns=["date", "submission_id_1", "submission_id_2", "pearson", "overlap"]
    )
    daily["abs_correlation"] = daily["pearson"].abs()
    if daily.empty:
        summary = pd.DataFrame()
    else:
        summary = daily.groupby(
            ["submission_id_1", "submission_id_2"], as_index=False
        ).agg(
            mean_correlation=("pearson", "mean"),
            mean_abs_correlation=("abs_correlation", "mean"),
            median_abs_correlation=("abs_correlation", "median"),
            p90_abs_correlation=("abs_correlation", lambda values: values.quantile(.90)),
            p95_abs_correlation=("abs_correlation", lambda values: values.quantile(.95)),
            high_corr_day_rate=("abs_correlation", lambda values: values.ge(high_threshold).mean()),
            positive_corr_day_rate=("pearson", lambda values: values.gt(0).mean()),
            valid_days=("pearson", "count"),
            mean_overlap=("overlap", "mean"),
        )
        summary["sign_consistency"] = summary[
            ["positive_corr_day_rate"]
        ].assign(negative=1 - summary["positive_corr_day_rate"]).max(axis=1)
        summary["high_similarity"] = summary["mean_abs_correlation"].ge(high_threshold)
        summary = summary.sort_values("mean_abs_correlation", ascending=False)
    clusters = _build_similarity_clusters(
        factor_cols, summary, threshold=cluster_threshold, top_peers=top_peers
    )
    if save:
        paths.incremental_dir.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(paths.incremental_dir / FACTOR_SIMILARITY_DAILY_FILENAME, index=False)
        summary.to_csv(paths.incremental_dir / FACTOR_SIMILARITY_SUMMARY_FILENAME, index=False)
        clusters.to_csv(paths.incremental_dir / FACTOR_CLUSTERS_FILENAME, index=False)
        print(f"截面相似度数据已保存: {paths.incremental_dir}")
    print(
        f"截面相似度: {len(factor_cols)} 个因子，{daily['date'].nunique()} 个交易日，"
        f"{len(summary)} 个因子对，{int(summary.get('high_similarity', pd.Series(dtype=bool)).sum())} 对高相似"
    )
    if display:
        show(summary, clusters)
    return {"daily": daily, "summary": summary, "clusters": clusters}


def analyze_factor_similarity(
    paths: CheckPaths = PATHS,
    *,
    high_correlation: float = CONFIG.high_correlation,
    max_samples: int = CONFIG.max_similarity_samples,
    display: bool = True,
) -> pd.DataFrame:
    """基于合并后的因子池快速计算两两 Pearson 相关性。"""
    factor_matrix = pd.read_parquet(paths.factor_pool_path).drop(columns=["date", "instrument"])
    factor_matrix.columns = factor_matrix.columns.astype(str)
    factor_matrix.columns.name = "submission_id"
    total_samples = len(factor_matrix)
    if total_samples > max_samples:
        positions = np.linspace(0, total_samples - 1, max_samples, dtype=np.int64)
        factor_matrix = factor_matrix.iloc[positions]
    factor_ids = factor_matrix.columns.to_numpy()
    correlation = factor_matrix.corr(method="pearson")
    valid = factor_matrix.notna().to_numpy(dtype=np.int32)
    overlap = valid.T @ valid
    left, right = np.triu_indices(len(factor_ids), k=1)
    result = pd.DataFrame({
        "submission_id_1": factor_ids[left], "submission_id_2": factor_ids[right],
        "pearson": correlation.to_numpy()[left, right], "overlap": overlap[left, right],
    }).dropna(subset=["pearson"])
    result["abs_correlation"] = result["pearson"].abs()
    result["high_similarity"] = result["abs_correlation"] >= high_correlation
    result = result.sort_values("abs_correlation", ascending=False)
    print(
        f"因子数: {len(factor_ids)}，总样本数: {total_samples}，分析样本数: {len(factor_matrix)}，"
        f"因子对: {len(result)}，高相似因子对: {int(result['high_similarity'].sum())}"
    )
    if display:
        show(result)
    return result
