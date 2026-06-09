import dai
import pandas as pd
import structlog

from .constants import BM_DICT

logger = structlog.get_logger()


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
    if daily_ret_data is None or daily_ret_data.empty:
        logger.warning("每日收益数据为空", start_date=start_date, end_date=end_date)
    return daily_ret_data


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
    sql = f"""
    SELECT
        date, instrument,
        (close - m_Lag(close,1)) / m_LAG(close, 1) as benchmark_ret
    FROM cn_stock_index_bar1d
    WHERE date BETWEEN DATE '{start_date}' - INTERVAL 10 DAY AND '{end_date}'
    AND instrument = '{BM_DICT[benchmark]}'
    """
    bm_ret = dai.query(sql).df()
    if bm_ret is None or bm_ret.empty:
        logger.warning(
            "基准指数收益数据为空",
            benchmark=benchmark,
            instrument=BM_DICT.get(benchmark),
            start_date=start_date,
            end_date=end_date,
        )
    return bm_ret
