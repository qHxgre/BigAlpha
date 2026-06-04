import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from .constants import EPS


def cross_section_zscore(x: pd.Series) -> pd.Series:
    """截面 z-score 标准化。

    输入是单一截面（同一交易日）的序列，输出按截面 mean/std 归一化的结果。
    截面 std 为 0 或非有限值时返回全 0，避免污染下游回归。
    """
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def fit_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
) -> np.ndarray:
    """单窗口 Elastic Net 拟合，返回权重向量 w。

    损失：L = ||y - F w||^2 + lambda1 ||w||_1 + lambda2 ||w||^2
    """
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=5000,
        tol=1e-4,
    )
    model.fit(X, y)
    return np.asarray(model.coef_, dtype=float)


def cal_model_score(abs_weight_series: pd.Series) -> dict:
    """单因子 ModelScore 及附属指标。

    ModelScore = mean(|w|) / (std(|w|) + eps)
    selection_rate 为该因子在所有滚动窗口中权重非零的占比。
    """
    if len(abs_weight_series) == 0:
        return {
            "model_score": 0.0,
            "abs_weight_mean": 0.0,
            "abs_weight_std": 0.0,
            "selection_rate": 0.0,
        }

    mean_abs = float(abs_weight_series.mean())
    std_abs = (
        float(abs_weight_series.std(ddof=1)) if len(abs_weight_series) > 1 else 0.0
    )
    model_score = mean_abs / (std_abs + EPS) if mean_abs > 0 else 0.0
    selection_rate = float((abs_weight_series > 0).mean())
    return {
        "model_score": model_score,
        "abs_weight_mean": mean_abs,
        "abs_weight_std": std_abs,
        "selection_rate": selection_rate,
    }
