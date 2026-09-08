"""与比赛目标一致的本地截面评估指标。"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def score_from_output(output: np.ndarray, task: str) -> np.ndarray:
    if task == "regression":
        return output.reshape(-1).astype(np.float64)
    logits = output.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability @ np.arange(logits.shape[1], dtype=np.float64)


def daily_ic(
    score: np.ndarray, raw_return: np.ndarray, date_ids: np.ndarray
) -> np.ndarray:
    values = []
    for date_id in np.unique(date_ids):
        mask = (
            (date_ids == date_id)
            & np.isfinite(score)
            & np.isfinite(raw_return)
        )
        if mask.sum() < 2:
            continue
        section_score = score[mask]
        section_return = raw_return[mask]
        if section_score.std() < 1e-12 or section_return.std() < 1e-12:
            continue
        value = spearmanr(section_score, section_return).statistic
        if np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def long_short_sharpe(
    score: np.ndarray,
    raw_return: np.ndarray,
    date_ids: np.ndarray,
    fraction: float = 0.1,
) -> float:
    returns = []
    for date_id in np.unique(date_ids):
        mask = (
            (date_ids == date_id)
            & np.isfinite(score)
            & np.isfinite(raw_return)
        )
        section_score = score[mask]
        section_return = raw_return[mask]
        count = max(1, int(len(section_score) * fraction))
        if len(section_score) < 2 * count:
            continue
        order = np.argsort(section_score)
        returns.append(
            float(
                section_return[order[-count:]].mean()
                - section_return[order[:count]].mean()
            )
        )
    if len(returns) < 2:
        return 0.0
    values = np.asarray(returns)
    std = values.std(ddof=1)
    return 0.0 if std < 1e-12 else float(values.mean() / std * np.sqrt(252))


def factor_metrics(
    output: np.ndarray,
    raw_return: np.ndarray,
    date_ids: np.ndarray,
    task: str,
) -> dict[str, float]:
    score = score_from_output(output, task)
    ics = daily_ic(score, raw_return, date_ids)
    ic_mean = float(ics.mean()) if len(ics) else 0.0
    ic_std = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    result = {
        "ic": ic_mean,
        "ic_std": ic_std,
        "ic_ir": 0.0 if ic_std < 1e-12 else ic_mean / ic_std,
        "long_short_sharpe": long_short_sharpe(
            score, raw_return, date_ids
        ),
        "evaluated_days": int(len(ics)),
    }
    return result
