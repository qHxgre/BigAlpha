from typing import Literal

import empyrical
import numpy as np
import pandas as pd

from .schemas import BasicPerf, ICPerf, Performance


def cal_Performance(df: pd.DataFrame, ll_pos: str, bm_pos: str) -> pd.DataFrame:
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


def cut(df: pd.DataFrame, factor_name: str, group_num: int) -> pd.DataFrame:
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
