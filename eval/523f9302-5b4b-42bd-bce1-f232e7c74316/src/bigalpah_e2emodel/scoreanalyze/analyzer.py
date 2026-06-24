from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List, Optional, Tuple

import empyrical
import numpy as np
import pandas as pd
import structlog

from . import render
from .constants import STRESS_PERIODS, PortfolioCode
from .data import get_bm_ret, get_daily_ret
from .metrics import cal_ic, cut
from .schemas import ScoreMetrics

logger = structlog.get_logger()

# 无风险利率（年化）/ 年化交易日数
RISK_FREE_RATE = 0.035
TRADING_DAYS_PER_YEAR = 242


class ScoreAnalyze:
    """端到端模型分数评估。

    分数经风格剔除后等价于一个每日更新的单因子，因此评估口径与单因子一致：
    计算 ic_mean / ic_ir / 多空 Sharpe / 压力期 IC IR 四项核心指标。
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        score_name: str = "score",
        benchmark: str = "中证1000",
        group_number: int = 10,
    ) -> None:
        self.start_date = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
        self.end_date = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
        self.score_name = score_name
        self.benchmark = benchmark
        self.group_num = group_number

        self.merge_data = pd.DataFrame()
        self.group_data = pd.DataFrame()
        self.group_ret = pd.DataFrame()
        self.group_cumret = pd.DataFrame()

    def merge_related_data(self, score_data: pd.DataFrame) -> pd.DataFrame:
        """合并分数与每日收益数据。"""
        bsd = (datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        aed = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        instruments = score_data["instrument"].unique().tolist()
        daily_ret_data = get_daily_ret(bsd, aed, instruments)
        merge_data = pd.merge(
            score_data, daily_ret_data, on=["date", "instrument"], how="left"
        )
        merge_data.sort_values(["date", "instrument"], inplace=True)
        miss = int(merge_data["daily_ret"].isna().sum())
        if miss:
            logger.warning(
                "合并分数与每日收益后存在缺失",
                missing_rows=miss,
                total_rows=len(merge_data),
            )
        return merge_data

    def get_group_data(self, score_data: pd.DataFrame) -> pd.DataFrame:
        """按分数对每个截面分组。"""
        cut_func = partial(cut, score_name=self.score_name, group_num=self.group_num)
        group_data = score_data.groupby("date", group_keys=False).apply(cut_func)
        return group_data

    def get_group_cumret(
        self, group_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """计算分组日收益与累计收益（含多空 ls 与基准 bm）。"""
        bm_ret = get_bm_ret(self.start_date, self.end_date, self.benchmark)
        bm_ret = bm_ret.set_index("date")
        groupret_data = (
            group_data[["date", "group", "daily_ret"]]
            .groupby(["date", "group"], group_keys=False)
            .apply(lambda x: np.nanmean(x))
            .reset_index()
        )
        groupret_data.rename(columns={0: "g_ret"}, inplace=True)
        groupret_pivotdata = groupret_data.pivot(
            index="date", values="g_ret", columns="group"
        )
        groupret_pivotdata["ls"] = (
            groupret_pivotdata[str(self.group_num - 1)]
            - groupret_pivotdata["0"]
        )
        groupret_pivotdata["bm"] = bm_ret["benchmark_ret"]
        groupret_pivotdata = groupret_pivotdata.shift(1)
        groupcumret_pivotdata = groupret_pivotdata.cumsum()
        return groupret_pivotdata, groupcumret_pivotdata

    def _active_stress_periods(self) -> List[Tuple[str, str, str]]:
        """筛选与 daily_ic 实际时间范围有交集的压力时段。"""
        if not hasattr(self, "daily_ic") or self.daily_ic.empty:
            return []
        ic_start = self.daily_ic.index.min()
        ic_end = self.daily_ic.index.max()
        active: List[Tuple[str, str, str]] = []
        for name, s, e in STRESS_PERIODS:
            ps, pe = pd.Timestamp(s), pd.Timestamp(e)
            if pe < ic_start or ps > ic_end:
                continue
            active.append((name, s, e))
        return active

    def stress_ic(self) -> Dict[str, float]:
        """计算压力时段（分 regime）的 IC / IR，用于稳健性评估。

        仅统计与评估区间有交集的压力时段。返回各时段 IC 均值，外加综合
        stress_ic_ir（所有压力时段 IC 拼接后的均值 / 标准差，单一数值）。

        Returns:
            Dict[str, float]: 各压力时段 ic 均值，外加 stress_ic_ir。
        """
        if not hasattr(self, "daily_ic") or self.daily_ic.empty:
            return {"stress_ic_ir": np.nan}

        ic = self.daily_ic
        result: Dict[str, float] = {}
        pooled = []
        for name, s, e in self._active_stress_periods():
            window = ic.loc[(ic.index >= pd.Timestamp(s)) & (ic.index <= pd.Timestamp(e))]
            if window.empty:
                result[f"{name}_ic"] = np.nan
                continue
            result[f"{name}_ic"] = float(window.mean())
            pooled.append(window)

        if pooled:
            pooled_ic = pd.concat(pooled)
            mean = float(pooled_ic.mean())
            std = float(pooled_ic.std())
            result["stress_ic_ir"] = mean / std if std and not np.isnan(std) else np.nan
        else:
            result["stress_ic_ir"] = np.nan
        return result

    def plot(self, score: Optional["ScoreMetrics"] = None) -> None:
        """在 notebook 中渲染含图表与核心指标的 HTML 报告。

        需要先调用 ``score`` 准备好 ``daily_ic`` / ``group_cumret`` 等中间数据。
        """
        if not hasattr(self, "daily_ic") or not hasattr(self, "group_cumret"):
            raise RuntimeError("请先调用 score(score_data) 计算中间结果，再调用 plot()。")
        score_dict = score.to_dict() if score is not None else getattr(self, "_score_dict", {})
        active_periods = self._active_stress_periods()
        if not active_periods:
            logger.warning(
                "分数时间范围与所有压力时段均无交集，将跳过压力期 IC 图与表格",
                score_name=self.score_name,
                ic_start=str(self.daily_ic.index.min()) if not self.daily_ic.empty else None,
                ic_end=str(self.daily_ic.index.max()) if not self.daily_ic.empty else None,
            )
        render.render_report(
            group_cumret=self.group_cumret,
            daily_ic=self.daily_ic,
            stress=self.stress_ic(),
            stress_periods=active_periods,
            group_num=self.group_num,
            score_name=self.score_name,
            score=score_dict,
        )

    def score(
        self,
        score_data: pd.DataFrame,
        plot: bool = True,
    ) -> ScoreMetrics:
        """计算端到端模型分数的四个核心指标，并可选 inline 渲染 HTML 报告。

        Args:
            score_data: 预处理后的分数数据（含 date/instrument/score）。
            plot: 是否在 notebook 中渲染 HTML 报告，默认 True。

        Returns:
            ScoreMetrics: ic_mean / ic_ir / sharpe_ratio / stress_ic_ir。
        """
        t0 = datetime.now()
        self.merge_data = self.merge_related_data(score_data)
        t1 = datetime.now()
        logger.info(f"合并相关数据, 耗时: {round((t1 - t0).total_seconds(), 4)} 秒")

        self.group_data = self.get_group_data(self.merge_data)
        t2 = datetime.now()
        logger.info(f"分数分组, 耗时: {round((t2 - t1).total_seconds(), 4)} 秒")

        self.group_ret, self.group_cumret = self.get_group_cumret(self.group_data)
        t3 = datetime.now()
        logger.info(f"分组收益与累计收益, 耗时: {round((t3 - t2).total_seconds(), 4)} 秒")

        daily_ic = (
            self.group_data.groupby("date", group_keys=False)
            .apply(lambda x: cal_ic(x, self.score_name, method="spearman"))
            .rename("ic")
            .dropna()
        )
        daily_ic.index = pd.to_datetime(daily_ic.index)
        self.daily_ic = daily_ic
        t4 = datetime.now()
        logger.info(f"日 IC 序列计算, 耗时: {round((t4 - t3).total_seconds(), 4)} 秒")

        ic_mean = float(daily_ic.mean())
        ic_std = float(daily_ic.std())
        ic_ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan

        ls_ret = self.group_ret[PortfolioCode.ls_pos].dropna()
        sharpe_ratio = float(
            empyrical.sharpe_ratio(ls_ret, RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)  # type: ignore
        )

        stress = self.stress_ic()
        stress_ic_ir = stress.get("stress_ic_ir", np.nan)
        t5 = datetime.now()
        logger.info(f"压力期指标计算, 耗时: {round((t5 - t4).total_seconds(), 4)} 秒")

        result = ScoreMetrics(
            ic_mean=ic_mean,
            ic_ir=ic_ir,
            sharpe_ratio=sharpe_ratio,
            stress_ic_ir=stress_ic_ir,
        )
        self._score_dict = result.to_dict()

        logger.info(
            "端到端模型分数评估完成",
            score_name=self.score_name,
            ic_mean=round(ic_mean, 6),
            ic_ir=round(ic_ir, 6) if np.isfinite(ic_ir) else None,
            sharpe_ratio=round(sharpe_ratio, 6) if np.isfinite(sharpe_ratio) else None,
            stress_ic_ir=round(stress_ic_ir, 6) if np.isfinite(stress_ic_ir) else None,
            total_seconds=round((t5 - t0).total_seconds(), 4),
        )

        if plot:
            self.plot(result)

        return result
