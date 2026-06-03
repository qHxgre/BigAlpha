"""Elastic Net 回归得分（B 项）

按 docs/因子挖掘_介绍_20260525.md "Elastic Net 回归得分（B 项）" 实现：

- 目标：截面 z-score 标准化后的下期收益率
        y_{i,t} = (r_{i,t+1} - mu_t) / sigma_t
- 损失：L = ||y - F w||^2 + lambda1 ||w||_1 + lambda2 ||w||^2
- 滚动窗口：窗口长度 60 个交易日，步长 20 个交易日
- ModelScore_i = mean(|w_i|) / (std(|w_i|) + eps)

本地工具用法（与 docs "本地调试支持" 描述一致）：
- 输入因子库（单列或多列因子），输出每个因子的 ModelScore、滚动权重曲线。
- 单因子时退化为单列回归，仍输出权重稳定性。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import dai
import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import ElasticNet

logger = structlog.get_logger()


# 滚动窗口长度（交易日）
WINDOW = 60
# 滚动步长（交易日）
STEP = 20
# 数值稳定项
EPS = 1e-8


def _zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


class ElasticNetEvaluator:
    """滚动 Elastic Net 评估器。

    输入：
        factor_panel: long DataFrame，至少包含 (trading_day, instrument, <factor cols...>)
        factor_cols : 参与回归的因子列名

    输出：
        per_factor_scores: 每个因子的 ModelScore、平均 |w|、入选率（被选中窗口占比）
        weights_history : 每个滚动窗口的权重，DataFrame(index=window_end, columns=factor_cols)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        alpha: float = 1e-3,
        l1_ratio: float = 0.5,
        window: int = WINDOW,
        step: int = STEP,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.window = window
        self.step = step

    # ---------- 收益数据 ----------

    def _load_next_day_return(self, instruments: List[str]) -> pd.DataFrame:
        """读取 T+1 日频收益（用于回归 y）。"""
        after_end_date = (
            datetime.strptime(self.end_date, "%Y-%m-%d") + timedelta(days=15)
        ).strftime("%Y-%m-%d")
        sql = """
        WITH cte_status as (
            SELECT date as trading_day, instrument
            FROM cn_stock_status
            WHERE price_limit_status = 2
        ),
        cte_bar1d as (
            SELECT date as trading_day, instrument, close
            FROM cn_stock_bar1d
            WHERE volume > 0
        )
        SELECT
            trading_day,
            instrument,
            m_lead(close, 1) / close - 1 AS ret
        FROM cte_bar1d
        SEMI JOIN cte_status USING (trading_day, instrument)
        ORDER BY trading_day, instrument
        """
        df = dai.query(
            sql,
            filters={
                "date": [f"{self.start_date} 00:00:00", f"{after_end_date} 23:59:59"],
                "instrument": instruments,
            },
        ).df()
        df["trading_day"] = pd.to_datetime(df["trading_day"])
        return df[["trading_day", "instrument", "ret"]]

    # ---------- 主流程 ----------

    def run(self, factor_panel: pd.DataFrame, factor_cols: List[str]) -> Dict:
        if not factor_cols:
            raise ValueError("factor_cols 不能为空")

        df = factor_panel.copy()
        df["trading_day"] = pd.to_datetime(df["trading_day"])
        df["instrument"] = df["instrument"].astype(str)

        ret_df = self._load_next_day_return(df["instrument"].unique().tolist())
        df = pd.merge(df, ret_df, how="left", on=["trading_day", "instrument"])
        df = df.dropna(subset=factor_cols + ["ret"])

        # 截面 z-score：因子和收益都按交易日截面归一化
        for col in factor_cols + ["ret"]:
            df[col] = df.groupby("trading_day")[col].transform(_zscore)

        df = df.sort_values(["trading_day", "instrument"]).reset_index(drop=True)

        all_days = sorted(df["trading_day"].unique())
        if len(all_days) < self.window:
            logger.warning(
                "可用交易日不足以填满一个窗口",
                available=len(all_days),
                window=self.window,
            )

        # 滚动窗口：[i, i+window) 训练，window_end 标记
        weights_records = []
        for start_idx in range(0, max(1, len(all_days) - self.window + 1), self.step):
            window_days = all_days[start_idx:start_idx + self.window]
            if len(window_days) < self.window:
                break
            window_end = window_days[-1]

            mask = df["trading_day"].isin(window_days)
            sub = df.loc[mask, factor_cols + ["ret"]]
            if len(sub) < max(50, len(factor_cols) * 5):
                logger.debug("窗口样本过少，跳过", window_end=str(window_end))
                continue

            X = sub[factor_cols].to_numpy(dtype=float)
            y = sub["ret"].to_numpy(dtype=float)

            try:
                model = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    fit_intercept=False,
                    max_iter=5000,
                    tol=1e-4,
                )
                model.fit(X, y)
                w = np.asarray(model.coef_, dtype=float)
            except Exception:
                logger.exception("Elastic Net 拟合失败", window_end=str(window_end))
                continue

            weights_records.append({"window_end": window_end, **dict(zip(factor_cols, w))})

        if not weights_records:
            logger.warning("没有可用的滚动窗口结果")
            empty_scores = pd.DataFrame(
                {
                    "factor": factor_cols,
                    "model_score": np.nan,
                    "abs_weight_mean": np.nan,
                    "selection_rate": 0.0,
                }
            )
            return {
                "per_factor_scores": empty_scores,
                "weights_history": pd.DataFrame(columns=["window_end"] + factor_cols),
            }

        weights_history = pd.DataFrame(weights_records).set_index("window_end").sort_index()
        abs_w = weights_history[factor_cols].abs()

        scores = []
        for col in factor_cols:
            series = abs_w[col]
            mean_abs = float(series.mean())
            std_abs = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            model_score = mean_abs / (std_abs + EPS) if mean_abs > 0 else 0.0
            selection_rate = float((series > 0).mean())
            scores.append(
                {
                    "factor": col,
                    "model_score": model_score,
                    "abs_weight_mean": mean_abs,
                    "selection_rate": selection_rate,
                }
            )

        per_factor_scores = pd.DataFrame(scores)

        return {
            "per_factor_scores": per_factor_scores,
            "weights_history": weights_history.reset_index(),
        }
