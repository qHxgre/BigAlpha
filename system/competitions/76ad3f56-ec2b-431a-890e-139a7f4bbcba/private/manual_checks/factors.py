"""处理后因子的相似度检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import show
from .config import CheckPaths


def analyze_factor_similarity(
    paths: CheckPaths,
    *,
    high_correlation: float = 0.95,
    max_samples: int = 50_000,
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
