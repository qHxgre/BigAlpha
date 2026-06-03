from functools import partial
from typing import Tuple, List, Literal, Optional, Dict
from dataclasses import dataclass, fields, asdict
from enum import Enum

import dai
import numpy as np
import pandas as pd
import empyrical

try:
    from . import render
except ImportError:
    import render  # type: ignore


@dataclass
class FactorScore:
    ic_mean: float
    ic_ir: float
    sharpe_ratio: float
    stress_ic_ir: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnoverPerf:
    turnover: float

BM_DICT = {
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "沪深300": "000300.SH",
}

STRESS_PERIODS: List[Tuple[str, str, str]] = [
    ("covid_2020",        "2020-02-03", "2020-03-31"),
    ("micro_cap_2024",    "2024-01-15", "2024-02-08"),
    ("policy_rally_2024", "2024-09-24", "2024-10-08"),
    ("tariff_2025",       "2025-04-07", "2025-04-30"),
]

class DataType(str, Enum):
    LONG = "long"
    SHORT = "short"
    LONG_SHORT = "long_short"

    def __str__(self):
        return self.value

class PortfolioCode(str, Enum):
    ll_pos = "9"
    ss_pos = "0"
    ls_pos = "ls"
    bm_pos = "bm"

    def __str__(self):
        return self.value


@dataclass
class ICPerf:
    ic: float
    ir: float
    ic_3: float
    ic_10: float
    ic_21: float
    ic_63: float
    ic_126: float
    ic_252: float



@dataclass
class BasicPerf:

    return_ratio: float
    annual_return_ratio: float
    ex_return_ratio: float
    ex_annual_return_ratio: float
    sharp_ratio: float
    return_volatility: float
    information_ratio: float
    max_drawdown: float
    win_percent: float
    trading_days: float
    ret_3: float
    ret_10: float
    ret_21: float
    ret_63: float
    ret_126: float
    ret_252: float


@dataclass
class Performance:
    return_ratio: list
    annual_return_ratio: list
    ex_return_ratio: list
    ex_annual_return_ratio: list
    sharp_ratio: list
    return_volatility: list
    max_drawdown: list
    win_percent: list
    trading_days: list

    def to_dataframe(self):
        data_dict = asdict(self)
        df = pd.DataFrame(data_dict)
        return df


@dataclass
class SummaryPerf:
    portfolio: str
    basic_perf: BasicPerf
    ic_perf: ICPerf
    turnover_perf: TurnoverPerf

    def __post_init__(self):
        for field in fields(BasicPerf):
            setattr(self, field.name, getattr(self.basic_perf, field.name))
        for field in fields(ICPerf):
            setattr(self, field.name, getattr(self.ic_perf, field.name))
        for field in fields(TurnoverPerf):
            setattr(self, field.name, getattr(self.turnover_perf, field.name))

    def to_dataframe(self):
        flat_data = {"portfolio": self.portfolio}
        flat_data.update(
            {
                f"{field.name}": getattr(self.basic_perf, field.name)
                for field in fields(BasicPerf)
            }
        )
        flat_data.update(
            {
                f"{field.name}": getattr(self.ic_perf, field.name)
                for field in fields(ICPerf)
            }
        )
        flat_data.update(
            {
                f"{field.name}": getattr(self.turnover_perf, field.name)
                for field in fields(TurnoverPerf)
            }
        )
        df = pd.DataFrame([flat_data])
        return df



def get_daily_ret(start_date: str, end_date: str) -> pd.DataFrame:
    """
    计算A股每日收益。

    Args:
        start_date (str): 开始日期，格式为 'YYYY-MM-DD'。
        end_date (str): 结束日期，格式为 'YYYY-MM-DD'。

    Returns:
        pd.DataFrame: A股每日收益数据。
    """
    sql = f"""
        SELECT 
            date, 
            instrument, 
            (m_lead(open, 2)/ m_lead(open, 1) - 1) AS daily_ret 
        FROM cn_stock_bar1d 
        WHERE date BETWEEN DATE '{start_date}' - INTERVAL 10 DAY AND '{end_date}'
        ORDER BY date, instrument
    """
    daily_ret_data = dai.query(sql).df()
    return daily_ret_data



def cal_Performance(
    df: pd.DataFrame, ll_pos: str, bm_pos: str
) -> pd.DataFrame:
    ll_series = df[ll_pos]
    bm_series = df[bm_pos]
    trading_days = len(ll_series)
    return_ratio = ll_series.sum()
    annual_return_ratio = ll_series.sum() * 242 / trading_days
    ex_return_ratio = (ll_series - bm_series).sum()
    ex_annual_return_ratio = (ll_series - bm_series).sum() * 242 / trading_days
    sharp_ratio = empyrical.sharpe_ratio(ll_series, 0.035 / 242)  # type: ignore
    return_volatility = empyrical.annual_volatility(ll_series)
    max_drawdown = empyrical.max_drawdown(ll_series)
    information_ratio = ll_series.mean() / ll_series.std()
    win_percent = len(ll_series[ll_series > 0]) / trading_days
    perf = Performance(
        return_ratio=[return_ratio],
        annual_return_ratio=[annual_return_ratio],
        ex_return_ratio=[ex_return_ratio],
        ex_annual_return_ratio=[ex_annual_return_ratio],
        sharp_ratio=[sharp_ratio],
        return_volatility=[return_volatility],
        max_drawdown=[max_drawdown],
        win_percent=[win_percent],
        trading_days=[int(trading_days)],
    )
    return perf.to_dataframe()

def cal_stats(series: pd.Series, bm_series: pd.Series) -> BasicPerf:
    """
    计算因子基础指标。

    Args:
        series (pd.Series): 因子的收益率序列。
        bm_series (pd.Series): 基准指数的收益率序列。

    Returns:
        BasicPerf: 包含各项基础指标的命名元组。
    """
    series = series.fillna(0)
    trading_days = len(series)
    return_ratio = series.sum()
    annual_return_ratio = series.sum() * 242 / trading_days
    ex_return_ratio = (series - bm_series).sum()
    ex_annual_return_ratio = (series - bm_series).sum() * 242 / trading_days
    sharp_ratio = empyrical.sharpe_ratio(series, 0.035 / 242)  # type: ignore
    return_volatility = empyrical.annual_volatility(series)
    max_drawdown = empyrical.max_drawdown(series)
    information_ratio = series.mean() / series.std()
    win_percent = len(series[series > 0]) / trading_days
    ret_3 = series.tail(3).sum()
    ret_10 = series.tail(10).sum()
    ret_21 = series.tail(21).sum()
    ret_63 = series.tail(63).sum()
    ret_126 = series.tail(126).sum()
    ret_252 = series.tail(252).sum()
    return BasicPerf(
        return_ratio=return_ratio,
        annual_return_ratio=annual_return_ratio,
        ex_return_ratio=ex_return_ratio,
        ex_annual_return_ratio=ex_annual_return_ratio,
        sharp_ratio=sharp_ratio,  # type: ignore
        return_volatility=return_volatility,  # type: ignore
        information_ratio=information_ratio,
        max_drawdown=max_drawdown,  # type: ignore
        win_percent=win_percent,
        trading_days=trading_days,
        ret_3=ret_3,
        ret_10=ret_10,
        ret_21=ret_21,
        ret_63=ret_63,
        ret_126=ret_126,
        ret_252=ret_252,
    )


def cal_ic(
    df: pd.DataFrame,
    factor_name: str,
    method: Literal["pearson", "kendall", "spearman"] = "spearman",
):
    return df["daily_ret"].corr(df[factor_name], method=method)


def cal_ic_stats(df: pd.DataFrame, factor_name) -> ICPerf:
    group_ic_data = (
        df.groupby("date", group_keys=False)
        .apply(lambda x: cal_ic(x, factor_name))
        .reset_index()
    )
    ic_data = group_ic_data.rename(columns={0: "g_ic"}).dropna()
    ic_mean = np.nanmean(ic_data["g_ic"])
    ir = np.nanmean(ic_data["g_ic"]) / np.nanstd(ic_data["g_ic"])
    ic_3 = ic_data["g_ic"].tail(3).mean()
    ic_10 = ic_data["g_ic"].tail(10).mean()
    ic_21 = ic_data["g_ic"].tail(21).mean()
    ic_63 = ic_data["g_ic"].tail(63).mean()
    ic_126 = ic_data["g_ic"].tail(126).mean()
    ic_252 = ic_data["g_ic"].tail(252).mean()
    return ICPerf(
        ic=ic_mean,  # type: ignore
        ir=ir,  # type: ignore
        ic_3=ic_3,
        ic_10=ic_10,
        ic_21=ic_21,
        ic_63=ic_63,
        ic_126=ic_126,
        ic_252=ic_252,
    )

def cut(
    df: pd.DataFrame,
    factor_name: str,
    group_num: int,
) -> pd.DataFrame:
    """
    数据分组。

    Args:
        df (pd.DataFrame): 需要分组的数据。
        factor_name (str): 需要分组的因子名称。
        group_num (int): 分组的数量。

    Returns:
        pd.DataFrame: 经过分组后的数据，新增一列 "group" 表示每个数据所在的分组编号。
    """
    df = df.drop_duplicates(factor_name)
    df.loc[:, "group"] = pd.qcut(
        df[factor_name], q=group_num, labels=False, duplicates="drop"
    )
    df = df.dropna(subset=["group"], how="any")
    df["group"] = df["group"].apply(int).apply(str)
    return df

def count_repeat(dfs: pd.DataFrame) -> int:
    if dfs.name > 0:
        return len(set(dfs["instrument"]) & set(dfs["instrument_lag"]))
    else:
        return 0




def cal_turnover(df: pd.DataFrame):

    df_ins = pd.DataFrame(
        df.groupby("date").apply(lambda x: x.instrument.tolist()),
        columns=["instrument"],
    ).reset_index()
    df_ins["instrument_lag"] = df_ins["instrument"].shift(1)
    df_ins["instrument_count"] = df_ins["instrument"].apply(len)
    df_ins["repeat_count"] = df_ins.apply(count_repeat, axis=1)
    df_ins["turnover"] = (
        1 - df_ins["repeat_count"] / df_ins["instrument_count"]
    )
    mean_turnover = np.nanmean(df_ins["turnover"])
    return mean_turnover

def get_bm_ret(start_date: str, end_date: str, benchmark: str) -> pd.DataFrame:
    """
    获取指定时间段内基准指数的日收益率数据。

    Args:
        start_date (str): 开始日期，格式为 'YYYY-MM-DD'。
        end_date (str): 结束日期，格式为 'YYYY-MM-DD'。
        benchmark (str): 基准指数的名称。

    Returns:
        pd.DataFrame: 包含日期、标的代码和基准指数日收益率的数据框。
    """
    # 构造SQL查询语句
    sql = f"""
    SELECT 
        date, instrument,
        (close - m_Lag(close,1)) / m_LAG(close, 1) as benchmark_ret
    FROM cn_stock_index_bar1d
    WHERE date BETWEEN DATE '{start_date}' - INTERVAL 10 DAY AND '{end_date}'
    AND instrument = '{BM_DICT[benchmark]}'
    """
    bm_ret = dai.query(sql).df()
    return bm_ret


class FactorAnalyze:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        factor_name: str='factor',
        benchmark: str='中证1000',
        group_number: int=10,
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

    # @timing_decorator
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

    # @timing_decorator
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
        group_data = factor_data.groupby("date", group_keys=False).apply(
            cut_func
        )
        return group_data

    # @timing_decorator
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
        # 基准收益率
        bm_ret = get_bm_ret(self.start_date, self.end_date, self.benchmark)
        bm_ret = bm_ret.set_index("date")
        # 分组收益率
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
        # 分组累计收益率
        groupcumret_pivotdata = groupret_pivotdata.cumsum()
        # 返回分组收益率与分组累计收益率
        return groupret_pivotdata, groupcumret_pivotdata

    # @timing_decorator
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

        def get_basic_perf(
            data_type: str, group_ret: pd.DataFrame
        ) -> BasicPerf:
            """_summary_

            Args:
                data_type (str): _description_
                group_ret (pd.DataFrame): _description_

            Returns:
                BasicPerf: _description_
            """
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

        # 三种绩效综合一下
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

    # @timing_decorator
    def get_yearly_perf(
        self, group_data: pd.DataFrame, group_ret: pd.DataFrame
    ):
        # 计算ic

        # 计算年度综合收益
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
        # 计算年度IC
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
        # 返回年度收益
        return yearly_perf

    # @timing_decorator
    def get_all_ic(self, group_data: pd.DataFrame) -> pd.DataFrame:
        group_ic_data = (
            group_data[["date", "daily_ret", self.factor_name]]
            .groupby("date", group_keys=False)
            .apply(lambda x: pd.Series({
                "g_ic": x[self.factor_name].corr(x["daily_ret"]),  # Pearson IC
                "g_rank_ic": x[self.factor_name].rank().corr(x["daily_ret"].rank())  # Spearman Rank IC
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
            Dict[str, float]: 各压力时段的 ic / ir，加上一个综合 stress_ic_ir。
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
                result[f"{name}_ir"] = np.nan
                continue
            mean = float(window.mean())
            std = float(window.std())
            result[f"{name}_ic"] = mean
            result[f"{name}_ir"] = mean / std if std and not np.isnan(std) else np.nan
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


