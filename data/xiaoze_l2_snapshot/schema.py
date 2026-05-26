import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class XiaozeL2SnapshotSchema(BaseSchema):
    # 主键
    date: np.datetime64 = Field(description="快照时间戳（含日期与时分秒，毫秒级）", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan, primary=True)

    # 基础时间字段
    trading_day: np.int32 = Field(description="交易日期（YYYYMMDD）", default=0)
    time: np.int32 = Field(description="当日时间（HHMMSSmmm，毫秒级）", default=0)

    # 基础行情
    pre_close: np.float32 = Field(description="前收盘价", default=np.nan)
    open: np.float32 = Field(description="开盘价", default=np.nan)
    high: np.float32 = Field(description="当日最高价", default=np.nan)
    low: np.float32 = Field(description="当日最低价", default=np.nan)
    price: np.float32 = Field(description="最新成交价", default=np.nan)

    # 5 档委卖价
    ask_price1: np.float32 = Field(description="1档委卖价", default=np.nan)
    ask_price2: np.float32 = Field(description="2档委卖价", default=np.nan)
    ask_price3: np.float32 = Field(description="3档委卖价", default=np.nan)
    ask_price4: np.float32 = Field(description="4档委卖价", default=np.nan)
    ask_price5: np.float32 = Field(description="5档委卖价", default=np.nan)

    # 5 档委买价
    bid_price1: np.float32 = Field(description="1档委买价", default=np.nan)
    bid_price2: np.float32 = Field(description="2档委买价", default=np.nan)
    bid_price3: np.float32 = Field(description="3档委买价", default=np.nan)
    bid_price4: np.float32 = Field(description="4档委买价", default=np.nan)
    bid_price5: np.float32 = Field(description="5档委买价", default=np.nan)

    # 5 档委卖量
    ask_volume1: np.int64 = Field(description="1档委卖量（股）", default=0)
    ask_volume2: np.int32 = Field(description="2档委卖量（股）", default=0)
    ask_volume3: np.int32 = Field(description="3档委卖量（股）", default=0)
    ask_volume4: np.int32 = Field(description="4档委卖量（股）", default=0)
    ask_volume5: np.int32 = Field(description="5档委卖量（股）", default=0)

    # 5 档委买量
    bid_volume1: np.int64 = Field(description="1档委买量（股）", default=0)
    bid_volume2: np.int32 = Field(description="2档委买量（股）", default=0)
    bid_volume3: np.int32 = Field(description="3档委买量（股）", default=0)
    bid_volume4: np.int32 = Field(description="4档委买量（股）", default=0)
    bid_volume5: np.int32 = Field(description="5档委买量（股）", default=0)

    # 累计成交
    num_trades: np.int32 = Field(description="当日累计成交笔数", default=0)
    volume: np.int64 = Field(description="当日累计成交量（股）", default=0)
    amount: np.float64 = Field(description="当日累计成交额（元）", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
