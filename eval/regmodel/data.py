from datetime import datetime, timedelta
from typing import List, Optional

import dai
import pandas as pd


def get_next_day_return(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """读取 T+1 日频收益（用于 Elastic Net 回归 y）。

    与 factoranalyze.data.get_daily_ret 风格保持一致：直接走 dai.query。
    仅保留 (volume>0) 且未触及涨跌停（price_limit_status=2）的交易日。

    Args:
        start_date (str): 开始日期，格式为 'YYYY-MM-DD'。
        end_date (str): 结束日期，格式为 'YYYY-MM-DD'。
        instruments (Optional[List[str]]): 股票代码列表，为 None 时不限制。

    Returns:
        pd.DataFrame: 列为 (date, instrument, daily_ret)。
    """
    after_end_date = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=15)
    ).strftime("%Y-%m-%d")

    sql = """
    WITH cte_status as (
        SELECT date, instrument
        FROM cn_stock_status
        WHERE price_limit_status = 2
    ),
    cte_bar1d as (
        SELECT date, instrument, close
        FROM cn_stock_bar1d
        WHERE volume > 0
    )
    SELECT
        date,
        instrument,
        m_lead(close, 1) / close - 1 AS daily_ret
    FROM cte_bar1d
    SEMI JOIN cte_status USING (date, instrument)
    ORDER BY date, instrument
    """

    filters = {"date": [f"{start_date} 00:00:00", f"{after_end_date} 23:59:59"]}
    if instruments:
        filters["instrument"] = instruments

    df = dai.query(sql, filters=filters).df()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "instrument", "daily_ret"]]
