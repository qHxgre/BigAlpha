from functools import partial
from typing import Dict, Optional, Tuple

import empyrical
import numpy as np
import pandas as pd

from . import render
from .constants import STRESS_PERIODS, DataType, PortfolioCode
from .data import get_bm_ret, get_daily_ret
from .metrics import (
    cal_ic,
    cal_ic_stats,
    cal_Performance,
    cal_stats,
    cal_turnover,
    cut,
)
from .schemas import (
    BasicPerf,
    FactorScore,
    ICPerf,
    SummaryPerf,
    TurnoverPerf,
)


class FactorAnalyze:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        factor_name: str = 'factor',
        benchmark: str = '中证1000',
        group_number: int = 10,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        self.benchmark = benchmark
        self.group_num = group_number

        self.merge_data = pd.DataFrame()
        self.group_data = pd.DataFrame()
        self.group_ret = pd.DataFrame()
        self.group_cumret = pd.DataFrame()
        self.whole_perf = pd.DataFrame()
        self.yearly_perf = pd.DataFrame()
        self.ic_perf = pd.DataFrame()

    def merge_related_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        合并因子分析需要的相关数据，包括每日收益数据。

        Args:
            factor_data (pd.DataFrame): 因子数据。

        Returns:
            pd.DataFrame: 合并后的因子数据。
        """
        daily_ret_data = get_daily_ret(self.start_date, self.end_date)
        merge_data = pd.merge(
            factor_data, daily_ret_data, on=["date", "instrument"], how="left"
        )
        merge_data.sort_values(["date", "instrument"], inplace=True)
        return merge_data

    def get_group_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        因子数据分组。

        Args:
            factor_data (pd.DataFrame): 因子数据。

        Returns:
            pd.DataFrame: 分组后的因子数据。
        """
        cut_func = partial(
            cut, factor_name=self.factor_name, group_num=self.group_num
        )
        group_data = factor_data.groupby("date", group_keys=False).apply(cut_func)
        return group_data

    def get_group_cumret(
        self, group_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        计算因子分组收益率。

        Args:
            group_data (pd.DataFrame): 因子分组数据。

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 因子分组收益率与因子分组累计收益率。
        """
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

    def get_whole_perf(
        self, group_data: pd.DataFrame, group_ret: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算因子分析整体绩效指标。

        Args:
            group_data (pd.DataFrame): 因子分组数据。
            group_ret (pd.DataFrame): 因子分组收益率数据。

        Returns:
            pd.DataFrame: 因子分析整体绩效指标。
        """

        def get_basic_perf(data_type: str, group_ret: pd.DataFrame) -> BasicPerf:
            if data_type == DataType.LONG:
                perf = cal_stats(
                    group_ret[PortfolioCode.ll_pos],
                    group_ret[PortfolioCode.bm_pos],
                )
            elif data_type == DataType.SHORT:
                perf = cal_stats(
                    group_ret[PortfolioCode.ss_pos],
                    group_ret[PortfolioCode.bm_pos],
                )
            else:
                perf = cal_stats(
                    group_ret[PortfolioCode.ls_pos],
                    group_ret[PortfolioCode.bm_pos],
                )
            return perf

        def get_ic(data_type: str, group_data: pd.DataFrame) -> ICPerf:
            if data_type == DataType.LONG:
                ic = cal_ic_stats(
                    group_data[group_data["group"] == PortfolioCode.ll_pos][
                        ["date", "daily_ret", self.factor_name]
                    ],
                    self.factor_name,
                )
            elif data_type == DataType.SHORT:
                ic = cal_ic_stats(
                    group_data[group_data["group"] == PortfolioCode.ss_pos][
                        ["date", "daily_ret", self.factor_name]
                    ],
                    self.factor_name,
                )
            else:
                ic = cal_ic_stats(
                    group_data[
                        group_data["group"].isin(
                            [PortfolioCode.ll_pos, PortfolioCode.ss_pos]
                        )
                    ][["date", "daily_ret", self.factor_name]],
                    self.factor_name,
                )
            return ic

        def get_turnover(
            data_type: str, group_data: pd.DataFrame
        ) -> TurnoverPerf:
            if data_type == DataType.LONG:
                turnover = cal_turnover(
                    group_data[group_data["group"] == PortfolioCode.ll_pos]
                )
            elif data_type == DataType.SHORT:
                turnover = cal_turnover(
                    group_data[group_data["group"] == PortfolioCode.ss_pos]
                )
            else:
                turnover = cal_turnover(
                    group_data[group_data["group"] == PortfolioCode.ll_pos]
                ) + cal_turnover(
                    group_data[group_data["group"] == PortfolioCode.ss_pos]
                )
            return TurnoverPerf(turnover=turnover)  # type: ignore

        summary_df = pd.DataFrame()
        for data_type in DataType:
            ic_perf = get_ic(data_type.value, group_data)
            basic_perf = get_basic_perf(data_type.value, group_ret)
            turnover_perf = get_turnover(data_type.value, group_data)
            summary_perf = SummaryPerf(
                portfolio=data_type.value,
                basic_perf=basic_perf,
                ic_perf=ic_perf,
                turnover_perf=turnover_perf,
            )
            summary_df = pd.concat(
                [summary_df, summary_perf.to_dataframe()], axis=0
            )
        summary_df.reset_index(drop=True, inplace=True)
        return summary_df

    def get_yearly_perf(
        self, group_data: pd.DataFrame, group_ret: pd.DataFrame
    ):
        year_df = group_ret.reset_index("date")
        year_df["year"] = year_df["date"].apply(lambda x: x.year)
        cal_Performance_func = partial(
            cal_Performance,
            ll_pos=PortfolioCode.ll_pos.value,
            bm_pos=PortfolioCode.bm_pos.value,
        )
        yearly_perf = year_df.groupby(["year"], group_keys=True).apply(
            cal_Performance_func
        )
        yearly_perf = yearly_perf.droplevel(1)
        group_ic_data = (
            (
                group_data[group_data["group"] == PortfolioCode.ll_pos][
                    ["date", "daily_ret", self.factor_name]
                ]
            )
            .groupby("date", group_keys=False)
            .apply(lambda x: cal_ic(x, self.factor_name))
            .reset_index()
        )
        ic_data = group_ic_data.rename(columns={0: "g_ic"}).dropna()
        ic_data["year"] = ic_data["date"].apply(lambda x: x.year)
        yearly_ic = ic_data.groupby("year").apply(
            lambda x: np.nanmean(x["g_ic"])
        )
        yearly_perf["ic"] = yearly_ic
        yearly_perf = yearly_perf.reset_index()
        yearly_perf["year"] = yearly_perf["year"].apply(str)
        return yearly_perf

    def get_all_ic(self, group_data: pd.DataFrame) -> pd.DataFrame:
        group_ic_data = (
            group_data[["date", "daily_ret", self.factor_name]]
            .groupby("date", group_keys=False)
            .apply(lambda x: pd.Series({
                "g_ic": x[self.factor_name].corr(x["daily_ret"]),
                "g_rank_ic": x[self.factor_name].rank().corr(x["daily_ret"].rank())
            }))
            .reset_index()
        )
        group_ic_data.rename(columns={0: "g_ic"}, inplace=True)
        group_ic_data = group_ic_data.shift(1)
        group_ic_data["ic_cumsum"] = group_ic_data["g_ic"].cumsum()
        group_ic_data["ic_roll_ma"] = group_ic_data["g_ic"].rolling(22).mean()
        group_ic_data = group_ic_data.dropna()
        return group_ic_data

    def stress_ic(self) -> Dict[str, float]:
        """计算压力时段的 IC / IR。

        特殊压力时间段：
        1. 2020-02-03 ~ 2020-03-31：新冠疫情爆发，节后开盘千股跌停 + 全球流动性危机；考验因子在系统性下跌中的稳健性。
        2. 2024-01-15 ~ 2024-02-08：小微盘流动性危机（雪球敲入 + DMA 去杠杆）。
        3. 2024-09-24 ~ 2024-10-08：政策"组合拳"驱动的暴力反转行情。
        4. 2025-04-07 ~ 2025-04-30：特朗普对等关税冲击，外生事件冲击。

        Returns:
            Dict[str, float]: 各压力时段的 ic 均值，外加一个综合 stress_ic_ir
            （所有压力时段 IC 拼接后的均值/标准差，单一数值）。
        """
        if not hasattr(self, "daily_ic") or self.daily_ic.empty:
            return {"stress_ic_ir": np.nan}

        ic = self.daily_ic
        result: Dict[str, float] = {}
        pooled = []
        for name, s, e in STRESS_PERIODS:
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

    def plot(self, score: Optional["FactorScore"] = None) -> None:
        """在 notebook 中渲染含四张图与核心指标的 HTML 报告。

        需要先调用 ``score`` 准备好 ``daily_ic`` / ``group_cumret`` 等中间数据。
        """
        if not hasattr(self, "daily_ic") or not hasattr(self, "group_cumret"):
            raise RuntimeError("请先调用 score(factor_data) 计算中间结果，再调用 plot()。")
        score_dict = score.to_dict() if score is not None else getattr(self, "_score_dict", {})
        render.render_report(
            group_cumret=self.group_cumret,
            daily_ic=self.daily_ic,
            stress=self.stress_ic(),
            stress_periods=STRESS_PERIODS,
            group_num=self.group_num,
            factor_name=self.factor_name,
            score=score_dict,
        )

    def score(
        self,
        factor_data: pd.DataFrame,
        plot: bool = True,
    ) -> FactorScore:
        """计算单因子 A 项核心得分所需的四个指标，并可选 inline 渲染 HTML 报告。

        Args:
            factor_data: 因子数据。
            plot: 是否在 notebook 中渲染 HTML 报告，默认 True。

        Returns:
            FactorScore: ic_mean / ic_ir / sharpe_ratio / stress_ic_ir。
        """
        self.merge_data = self.merge_related_data(factor_data)
        self.group_data = self.get_group_data(self.merge_data)
        self.group_ret, self.group_cumret = self.get_group_cumret(self.group_data)

        daily_ic = (
            self.group_data.groupby("date", group_keys=False)
            .apply(lambda x: cal_ic(x, self.factor_name, method="spearman"))
            .rename("ic")
            .dropna()
        )
        daily_ic.index = pd.to_datetime(daily_ic.index)
        self.daily_ic = daily_ic

        ic_mean = float(daily_ic.mean())
        ic_std = float(daily_ic.std())
        ic_ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan

        ls_ret = self.group_ret[PortfolioCode.ls_pos].dropna()
        sharpe_ratio = float(empyrical.sharpe_ratio(ls_ret, 0.035 / 242))  # type: ignore

        stress = self.stress_ic()
        stress_ic_ir = stress.get("stress_ic_ir", np.nan)

        result = FactorScore(
            ic_mean=ic_mean,
            ic_ir=ic_ir,
            sharpe_ratio=sharpe_ratio,
            stress_ic_ir=stress_ic_ir,
        )
        self._score_dict = result.to_dict()

        if plot:
            self.plot(result)

        return result
