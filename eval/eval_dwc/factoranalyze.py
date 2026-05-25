import dai
import empyrical
import structlog
import numpy as np
import pandas as pd
from functools import partial
from datetime import datetime, timedelta

logger = structlog.get_logger()

class FactorAnalyze:
    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.group_num = 5

        self.groupret_pivotdata = pd.DataFrame()
        self.groupcumret_pivotdata = pd.DataFrame()

    def merge_return_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """合并收益率数据"""
        # 多取n天数据计算收益率
        after_end_date = (datetime.strptime(self.end_date, "%Y-%m-%d") + timedelta(days=15)).strftime("%Y-%m-%d")
        # 股票池
        instruments = factor_data["instrument"].unique().tolist()

        # 获取收益率数据
        """获取收益率数据，以未来一日的收益率"""
        sql = """
        WITH cte_stock_pool as (
            SELECT date as trading_day, instrument
            FROM cn_stock_status
            WHERE 
                -- 剔除当日涨跌停股票
                price_limit_status=2
        ),
        cte_bar15m as (
            SELECT date, DATE_TRUNC('day', date) as trading_day, instrument, volume, close
            FROM cn_stock_bar15m_c
        )
        SELECT date, instrument, m_lead(close, 16) / close - 1 as ret
        FROM cte_stock_pool
        LEFT JOIN cte_bar15m USING (trading_day, instrument)
        WHERE volume > 0
        ORDER BY date, instrument
        """
        ret_df = dai.query(sql, filters={'date': [f"{self.start_date} 00:00:00", f"{after_end_date} 23:59:59"], "instrument": instruments}).df()
        
        merge_df = pd.merge(factor_data, ret_df, how="left", on=["date", "instrument"])
        merge_df = merge_df[["date", "instrument", "factor", "ret"]]
        return merge_df

    def cut_group_data(self, merge_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """分组数据"""

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
            
        cut_func = partial(
            cut, factor_name=factor_name, group_num=self.group_num
        )
        group_data = merge_data.groupby("date", group_keys=False).apply(
            cut_func
        )
        return group_data

    def cpt_group_ret(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """计算分组收益率"""
        # 分组收益率
        groupret_data = (
            group_data[["date", "group", "ret"]]
            .groupby(["date", "group"], group_keys=False)
            .apply(lambda x: np.nanmean(x))
            .reset_index()
            .rename(columns={0: "g_ret"})
        )
        groupret_pivotdata = groupret_data.pivot(
            index="date", values="g_ret", columns="group"
        )
        groupret_pivotdata["ls"] = (
            groupret_pivotdata[str(self.group_num - 1)]
            - groupret_pivotdata["0"]
        )
        groupret_pivotdata = groupret_pivotdata.shift(1)
        # 分组累计收益率
        groupcumret_pivotdata = groupret_pivotdata.cumsum()
        return groupret_pivotdata, groupcumret_pivotdata

    def cpt_ls_sharp(self, groupret_pivotdata: pd.DataFrame) -> float:
        """计算多空夏普"""
        series = groupret_pivotdata["ls"].fillna(0)
        sharp_ratio = empyrical.sharpe_ratio(series, 0.035 / 242)
        return sharp_ratio

    def validate(self, factor_data: pd.DataFrame, factor_name: str):
        """
        执行所有因子分析流程。
        """
        t0 = datetime.now()
        merge_data = self.merge_return_data(factor_data)
        t1 = datetime.now()
        logger.info(f"[单因子分析] 按照截断时间点获取未来一日收益率数据, 耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        group_data = self.cut_group_data(merge_data, factor_name)
        t2 = datetime.now()
        logger.info(f"[单因子分析] 因子数据分组, 耗时: {round((t2-t1).total_seconds(), 4)} 秒")

        groupret_pivotdata, groupcumret_pivotdata = self.cpt_group_ret(group_data)
        self.groupret_pivotdata = groupret_pivotdata
        self.groupcumret_pivotdata = groupcumret_pivotdata
        sharp_ratio = self.cpt_ls_sharp(groupret_pivotdata)
        t3 = datetime.now()
        logger.info(f"[单因子分析] 计算分组收益率&多空夏普比例, 耗时: {round((t3-t2).total_seconds(), 4)} 秒")
        return sharp_ratio
