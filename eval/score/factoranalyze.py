import dai
import pandas as pd
from functools import partial
from typing import Tuple
from dataclasses import dataclass, fields, asdict
from enum import Enum
import numpy as np
import empyrical
from typing import List, Literal


@dataclass
class TurnoverPerf:
    turnover: float

BM_DICT = {
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "沪深300": "000300.SH",
}

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
        factor_name: str,
        group_number: int,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        self.benchmark = '中证1000'
        self.group_num = group_number

    # @timing_decorator
    def merge_related_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        合并因子分析需要的相关数据，包括每日收益数据，计算相关性的因子数据。

        Args:
            factor_data (pd.DataFrame): 因子数据。

        Returns:
            pd.DataFrame: 合并后的因子数据。
        """
        daily_ret_data = get_daily_ret(self.start_date, self.end_date)
        # correlation_factor = dai.DataSource("cal_correlation_factor").read()
        # self.corr_factor_names = get_factor_names(correlation_factor)[0]
        merge_data = pd.merge(
            factor_data, daily_ret_data, on=["date", "instrument"], how="left"
        )
        # merge_data = pd.merge(
        #     merge_data,
        #     correlation_factor,
        #     on=["date", "instrument"],
        #     how="left",
        # )
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

    def validate(self, factor_data: pd.DataFrame):
        """
        执行所有因子分析流程。
        """
        merge_data = self.merge_related_data(factor_data)
        group_data = self.get_group_data(merge_data)
        group_ret, group_cumret = self.get_group_cumret(group_data)
        whole_perf = self.get_whole_perf(group_data, group_ret)
        yearly_perf = self.get_yearly_perf(group_data, group_ret)
        ic_perf = self.get_all_ic(group_data)
        # corr_perf = self.get_correlation(group_data)
        return whole_perf, yearly_perf, ic_perf, group_cumret
