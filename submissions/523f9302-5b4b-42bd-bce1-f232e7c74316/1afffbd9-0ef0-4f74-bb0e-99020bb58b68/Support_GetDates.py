import bisect
from datetime import time, date as date_cls
from typing import List, Optional, Literal

import pandas as pd
import polars as pl

from Support_Calendars import (
    TRADING_DAYS,
    CALENDAR_START,
    CALENDAR_END,
    FIRST_PREV_DATE,
    LAST_NEXT_DATE,
)

_CALENDAR_DF: Optional[pl.DataFrame] = None
_CALENDAR_DATES: Optional[List[str]] = None
_CALENDAR_RANGE: Optional[tuple[str, str]] = None


def get_calendar(
    start_date: str = "2018-01-01",
    end_date: str = "2026-12-30",
) -> pl.DataFrame:
    global _CALENDAR_DF, _CALENDAR_DATES, _CALENDAR_RANGE

    need_build = _CALENDAR_DF is None or _CALENDAR_RANGE is None
    if not need_build:
        cached_start, cached_end = _CALENDAR_RANGE
        if start_date < cached_start or end_date > cached_end:
            need_build = True

    if need_build:
        if _CALENDAR_RANGE is not None:
            cached_start, cached_end = _CALENDAR_RANGE
            start_date = min(start_date, cached_start)
            end_date = max(end_date, cached_end)

        # 超出内嵌范围时给出明确错误（替代 xcals 扩表）
        if start_date < CALENDAR_START or end_date > CALENDAR_END:
            raise ValueError(
                f"请求区间 [{start_date}, {end_date}] 超出内嵌日历 "
                f"[{CALENDAR_START}, {CALENDAR_END}]，请重新生成 Support_Calendars.py"
            )

        i = bisect.bisect_left(TRADING_DAYS, start_date)
        j = bisect.bisect_right(TRADING_DAYS, end_date)
        dates_str = TRADING_DAYS[i:j]
        dates = [date_cls.fromisoformat(d) for d in dates_str]

        df = pd.DataFrame({"date": dates})
        df["last_date"] = df["date"].shift(1)
        df["next_date"] = df["date"].shift(-1)

        if len(df) > 0:
            # 与原先 previous_session / next_session 对齐
            if dates_str[0] == TRADING_DAYS[0]:
                df.iloc[0, df.columns.get_loc("last_date")] = date_cls.fromisoformat(
                    FIRST_PREV_DATE
                )
            else:
                # 切片中间：上一交易日在 TRADING_DAYS 内
                prev = TRADING_DAYS[i - 1]
                df.iloc[0, df.columns.get_loc("last_date")] = date_cls.fromisoformat(prev)

            if dates_str[-1] == TRADING_DAYS[-1]:
                df.iloc[-1, df.columns.get_loc("next_date")] = date_cls.fromisoformat(
                    LAST_NEXT_DATE
                )
            else:
                nxt = TRADING_DAYS[j]
                df.iloc[-1, df.columns.get_loc("next_date")] = date_cls.fromisoformat(nxt)

        _CALENDAR_DF = pl.from_pandas(df).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("last_date").cast(pl.Date),
            pl.col("next_date").cast(pl.Date),
        )
        _CALENDAR_DATES = dates_str  # 已是 "YYYY-MM-DD"
        _CALENDAR_RANGE = (start_date, end_date)

    return _CALENDAR_DF


def get_calendar_dates(
    start_date: str = "2018-01-01",
    end_date: str = "2026-12-30",
) -> List[str]:
    """返回缓存的升序交易日字符串列表（确保日历已构建）。"""
    
    get_calendar(start_date=start_date, end_date=end_date)
    
    assert _CALENDAR_DATES is not None
    return _CALENDAR_DATES


def get_last_days(date: str, N: int) -> List[str]:
    """
    获取不超过 date 的最近 N 个交易日（升序，含 date 若其为交易日）。
    """
    if N <= 0:
        return []

    dates = get_calendar_dates()
    # dates[i] <= date 的右边界
    idx = bisect.bisect_right(dates, date)
    if idx == 0:
        return []
    start = max(0, idx - N)
    return dates[start:idx]


def get_next_days(date: str, N: int) -> List[str]:
    """
    获取不早于 date 的最近 N 个交易日（升序，含 date 若其为交易日）。
    """
    if N <= 0:
        return []

    dates = get_calendar_dates()
    # dates[i] >= date 的左边界
    idx = bisect.bisect_left(dates, date)
    return dates[idx : idx + N]

def trading_days_between(start: str, end: str) -> List[str]:
    """
    返回 [start, end] 内的交易日（升序，两端均含，若其为交易日）。
    start > end 时返回 []。
    """
    if start > end:
        return []
    dates = get_calendar_dates()
    i = bisect.bisect_left(dates, start)
    j = bisect.bisect_right(dates, end)
    return dates[i:j]


def next_trading_day_after(date: str) -> Optional[str]:
    """严格晚于 date 的下一个交易日；若无则 None。"""
    dates = get_calendar_dates()
    idx = bisect.bisect_right(dates, date)
    if idx >= len(dates):
        return None
    return dates[idx]

def get_basetime(
    date: str, 
    delta: Literal["1m", "5m", "15m", "30m"]
):
    """ 生成指定间隔的标准时间 """

    return pl.select(
        pl.time_range(
            start=time(9, 30),
            end=time(15, 00),
            interval=delta
        ).alias("Time")
    ).filter(
        # 过滤出 A 股正常交易时间：09:30-11:30 以及 13:00-15:00
        ((pl.col("Time") > time(9, 30)) & (pl.col("Time") <= time(11, 30))) |
        ((pl.col("Time") > time(13, 0)) & (pl.col("Time") <= time(15, 0)))
    ).select(
        pl.lit(date).str.to_date("%Y-%m-%d").dt.combine(pl.col("Time")).cast(pl.Datetime).alias("date")
    )
