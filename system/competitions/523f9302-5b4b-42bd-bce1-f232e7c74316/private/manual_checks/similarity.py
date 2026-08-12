"""端到端模型预测值的两两相似度检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import show
from .config import CONFIG, PATHS, CheckPaths


def _pairwise_summary(pool: pd.DataFrame, pool_type: str) -> pd.DataFrame:
    factor_columns = [column for column in pool.columns if column not in {"date", "instrument"}]
    numeric = pool[factor_columns].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(min_periods=5)
    rows = []
    for left_index, left in enumerate(factor_columns):
        for right in factor_columns[left_index + 1:]:
            overlap = int(numeric[[left, right]].notna().all(axis=1).sum())
            value = correlation.at[left, right]
            rows.append({
                "pool_type": pool_type,
                "submission_id_1": str(left),
                "submission_id_2": str(right),
                "correlation": value,
                "abs_correlation": abs(value) if pd.notna(value) else np.nan,
                "overlap_count": overlap,
            })
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False, na_position="last")


def analyze_prediction_similarity(
    paths: CheckPaths = PATHS,
    *,
    max_samples: int = CONFIG.max_similarity_samples,
    random_state: int = 20260812,
    save: bool = True,
    display: bool = True,
) -> pd.DataFrame:
    """抽样检查处理后及原始 score pool 的 Pearson 相关性。

    该检查只评估不同提交预测结果是否高度相似，不涉及回归模型评分。
    """
    results = []
    for pool_type, path in (("processed", paths.process_pool_path), ("raw", paths.raw_pool_path)):
        if not path.is_file():
            print(f"跳过不存在的预测池: {path}")
            continue
        pool = pd.read_parquet(path)
        if max_samples > 0 and len(pool) > max_samples:
            pool = pool.sample(max_samples, random_state=random_state)
        result = _pairwise_summary(pool, pool_type)
        results.append(result)
        print(f"{pool_type} 预测池: {len(pool):,} 个样本，{len(result):,} 个提交对")
    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["pool_type", "submission_id_1", "submission_id_2", "correlation",
                 "abs_correlation", "overlap_count"]
    )
    combined = combined.sort_values("abs_correlation", ascending=False, na_position="last")
    if save:
        paths.similarity_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(paths.similarity_path, index=False, encoding="utf-8-sig")
        print(f"相似度结果已保存: {paths.similarity_path}")
    if display:
        show(combined.head(CONFIG.report_top_n))
    return combined
