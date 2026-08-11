"""BARRA 风格暴露检查的统一字段定义和计算工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXPOSURE_COLUMNS, EXPOSURE_DATASOURCE


def query_exposures(start_date: str, end_date: str) -> pd.DataFrame:
    """查询与 ``DataProcess.neutralize`` 相同的风格和行业暴露数据。"""
    try:
        import dai
    except ImportError as exc:  # pragma: no cover - 本地环境通常没有云端数据依赖
        raise RuntimeError("风格暴露检查需要在安装了 dai 的 BigQuant 云端环境运行") from exc

    columns = ",\n            ".join(EXPOSURE_COLUMNS)
    sql = f"""
        SELECT
            date,
            instrument,
            {columns}
        FROM {EXPOSURE_DATASOURCE}
    """
    return dai.query(sql, filters={"date": [start_date, end_date]}).df()


def calculate_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """加入截距后做最小二乘回归并返回残差。"""
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """计算至少包含五个有效样本且两侧非恒定时的 Pearson 相关性。"""
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 5:
        return np.nan
    x = left[mask]
    y = right[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def max_finite(values: list[float]) -> float:
    """返回有限值中的最大值；没有有限值时返回 NaN。"""
    finite = [value for value in values if np.isfinite(value)]
    return max(finite) if finite else np.nan
