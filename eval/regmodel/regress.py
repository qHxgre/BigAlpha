from typing import List

import numpy as np
import pandas as pd
import structlog

from . import render
from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_L1_RATIO,
    MIN_SAMPLES_PER_FACTOR,
    MIN_WINDOW_SAMPLES,
    STEP,
    WINDOW,
)
from .data import get_next_day_return
from .metrics import (
    cal_model_score,
    cross_section_zscore,
    fit_elastic_net,
)
from .schemas import ElasticNetResult

logger = structlog.get_logger()


class ElasticNetRegress:
    """Elastic Net 滚动回归评估。

    - 目标：截面 z-score 标准化后的下期收益率
            y_{i,t} = (r_{i,t+1} - mu_t) / sigma_t
    - 损失：L = ||y - F w||^2 + lambda1 ||w||_1 + lambda2 ||w||^2
    - 滚动窗口：默认 60 个交易日，步长 20 个交易日
    - ModelScore_i = mean(|w_i|) / (std(|w_i|) + eps)

    使用方式：
        ana = ElasticNetAnalyze(start_date, end_date)
        result = ana.score(factor_panel)

    其中 factor_panel 至少包含列：
        - date / instrument
        - 其余列均视为参与回归的因子
    """

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date

        self.alpha = DEFAULT_ALPHA
        self.l1_ratio = DEFAULT_L1_RATIO
        self.window = WINDOW
        self.step = STEP

        self.factor_cols: List[str] = []
        self.merge_data = pd.DataFrame()
        self.weights_history = pd.DataFrame()
        self.per_factor_scores = pd.DataFrame()

    @staticmethod
    def _resolve_factor_cols(factor_panel: pd.DataFrame) -> List[str]:
        """date / instrument 之外的所有列视为因子列。"""
        reserved = {"date", "instrument"}
        factor_cols = [c for c in factor_panel.columns if c not in reserved]
        if not factor_cols:
            raise ValueError("factor_panel 中找不到任何因子列（date/instrument 之外）")
        return factor_cols

    def merge_related_data(
        self, factor_panel: pd.DataFrame, factor_cols: List[str]
    ) -> pd.DataFrame:
        """合并因子面板与 T+1 收益，并在每个交易日做 z-score 标准化。"""
        df = factor_panel.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["instrument"] = df["instrument"].astype(str)

        ret_df = get_next_day_return(
            self.start_date,
            self.end_date,
            instruments=df["instrument"].unique().tolist(),
        )
        df = pd.merge(df, ret_df, how="left", on=["date", "instrument"])
        df = df.dropna(subset=factor_cols + ["daily_ret"])

        for col in factor_cols + ["daily_ret"]:
            df[col] = df.groupby("date")[col].transform(cross_section_zscore)

        df = df.sort_values(["date", "instrument"]).reset_index(drop=True)
        return df

    def get_weights_history(
        self, merge_data: pd.DataFrame, factor_cols: List[str]
    ) -> pd.DataFrame:
        """滚动窗口拟合 Elastic Net，返回每个窗口的权重。"""
        all_days = sorted(merge_data["date"].unique())
        if len(all_days) < self.window:
            logger.warning(
                "可用交易日不足以填满一个窗口",
                available=len(all_days),
                window=self.window,
            )

        records = []
        end_idx = max(1, len(all_days) - self.window + 1)
        for start_idx in range(0, end_idx, self.step):
            window_days = all_days[start_idx:start_idx + self.window]
            if len(window_days) < self.window:
                break
            window_end = window_days[-1]

            mask = merge_data["date"].isin(window_days)
            sub = merge_data.loc[mask, factor_cols + ["daily_ret"]]
            min_samples = max(
                MIN_WINDOW_SAMPLES, len(factor_cols) * MIN_SAMPLES_PER_FACTOR
            )
            if len(sub) < min_samples:
                logger.debug("窗口样本过少，跳过", window_end=str(window_end))
                continue

            X = sub[factor_cols].to_numpy(dtype=float)
            y = sub["daily_ret"].to_numpy(dtype=float)

            try:
                w = fit_elastic_net(X, y, alpha=self.alpha, l1_ratio=self.l1_ratio)
            except Exception:
                logger.exception("Elastic Net 拟合失败", window_end=str(window_end))
                continue

            records.append({"window_end": window_end, **dict(zip(factor_cols, w))})

        if not records:
            return pd.DataFrame(columns=["window_end"] + factor_cols)

        return pd.DataFrame(records).sort_values("window_end").reset_index(drop=True)

    def get_per_factor_scores(
        self, weights_history: pd.DataFrame, factor_cols: List[str]
    ) -> pd.DataFrame:
        """根据滚动权重历史计算每个因子的 ModelScore。"""
        if weights_history.empty:
            return pd.DataFrame(
                {
                    "factor": factor_cols,
                    "model_score": np.nan,
                    "abs_weight_mean": np.nan,
                    "abs_weight_std": np.nan,
                    "selection_rate": 0.0,
                }
            )

        abs_w = weights_history[factor_cols].abs()
        rows = [
            {"factor": col, **cal_model_score(abs_w[col])}
            for col in factor_cols
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("model_score", ascending=False)
            .reset_index(drop=True)
        )

    def plot(self) -> None:
        """渲染滚动权重曲线、|w| 分布、相关性热力图与 ModelScore 表格的 HTML 报告。"""
        if self.weights_history.empty or self.per_factor_scores.empty:
            raise RuntimeError("请先调用 score(factor_panel) 计算中间结果，再调用 plot()。")

        render.render_report(
            per_factor_scores=self.per_factor_scores,
            weights_history=self.weights_history,
            factor_panel=self.merge_data,
            factor_cols=self.factor_cols,
        )

    def score(
        self,
        factor_panel: pd.DataFrame,
        plot: bool = True,
    ) -> ElasticNetResult:
        """执行滚动 Elastic Net 回归，输出每个因子的 ModelScore。

        Args:
            factor_panel: long DataFrame，至少包含 (date, instrument, factor1, factor2, ...)。
                date / instrument 之外的所有列均参与回归。
            plot: 是否在 notebook 中渲染 HTML 报告，默认 True。

        Returns:
            ElasticNetResult: 单因子得分表（按 model_score 倒序）与滚动权重历史。
        """
        factor_cols = self._resolve_factor_cols(factor_panel)
        self.factor_cols = factor_cols

        self.merge_data = self.merge_related_data(factor_panel, factor_cols)
        self.weights_history = self.get_weights_history(self.merge_data, factor_cols)
        self.per_factor_scores = self.get_per_factor_scores(
            self.weights_history, factor_cols
        )

        if plot:
            self.plot()

        return ElasticNetResult(
            per_factor_scores=self.per_factor_scores,
            weights_history=self.weights_history,
        )
