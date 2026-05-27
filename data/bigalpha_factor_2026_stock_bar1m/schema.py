import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class BigalphaFactor2026StockBar1mSchema(BaseSchema):
    date: np.datetime64 = Field(description="日期", default=0)
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan)

    adjust_factor: np.float32 = Field(description="累计后复权因子", default=np.nan)
    pre_close: np.float32 = Field(description="前收盘价", default=np.nan)
    high: np.float32 = Field(description="最高价", default=np.nan)
    open: np.float32 = Field(description="开盘价", default=np.nan)
    low: np.float32 = Field(description="最低价", default=np.nan)
    close: np.float32 = Field(description="收盘价", default=np.nan)
    deal_number: np.int32 = Field(description="成交笔数", default=0)
    volume: np.int32 = Field(description="最新总成交量", default=0)
    amount: np.float32 = Field(description="最新成交金额", default=np.nan)

    # 委托价格
    ask_price1: np.float32 = Field(description="1档委卖价", default=0)
    ask_price2: np.float32 = Field(description="2档委卖价", default=0)
    ask_price3: np.float32 = Field(description="3档委卖价", default=0)
    ask_price4: np.float32 = Field(description="4档委卖价", default=0)
    ask_price5: np.float32 = Field(description="5档委卖价", default=0)
    bid_price1: np.float32 = Field(description="1档委买价", default=0)
    bid_price2: np.float32 = Field(description="2档委买价", default=0)
    bid_price3: np.float32 = Field(description="3档委买价", default=0)
    bid_price4: np.float32 = Field(description="4档委买价", default=0)
    bid_price5: np.float32 = Field(description="5档委买价", default=0)

    # 委托量
    ask_volume1: np.int32 = Field(description="1档委卖量", default=0)
    ask_volume2: np.int32 = Field(description="2档委卖量", default=0)
    ask_volume3: np.int32 = Field(description="3档委卖量", default=0)
    ask_volume4: np.int32 = Field(description="4档委卖量", default=0)
    ask_volume5: np.int32 = Field(description="5档委卖量", default=0)
    bid_volume1: np.int32 = Field(description="1档委买量", default=0)
    bid_volume2: np.int32 = Field(description="2档委买量", default=0)
    bid_volume3: np.int32 = Field(description="3档委买量", default=0)
    bid_volume4: np.int32 = Field(description="4档委买量", default=0)
    bid_volume5: np.int32 = Field(description="5档委买量", default=0)

    # 委托比数
    ask_num_orders1: np.int32 = Field(description="卖1档委托笔数", default=0)
    ask_num_orders2: np.int32 = Field(description="卖2档委托笔数", default=0)
    ask_num_orders3: np.int32 = Field(description="卖3档委托笔数", default=0)
    ask_num_orders4: np.int32 = Field(description="卖4档委托笔数", default=0)
    ask_num_orders5: np.int32 = Field(description="卖5档委托笔数", default=0)
    bid_num_orders1: np.int32 = Field(description="买1档委托笔数", default=0)
    bid_num_orders2: np.int32 = Field(description="买2档委托笔数", default=0)
    bid_num_orders3: np.int32 = Field(description="买3档委托笔数", default=0)
    bid_num_orders4: np.int32 = Field(description="买4档委托笔数", default=0)
    bid_num_orders5: np.int32 = Field(description="买5档委托笔数", default=0)

    class Config:
        arbitrary_types_allowed = True
