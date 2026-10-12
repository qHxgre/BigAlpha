import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class CptGifp2026StockBar1mSchema(BaseSchema):
    """由 Level-2 快照构建的 1 分钟 K 线及十档盘口。"""

    date: np.datetime64 = Field(description="日期（Bar 结束时刻）", default=0)
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan)
    instrument_id: np.int16 = Field(description="股票代码 ID", default=np.nan)

    adjust_factor: np.float32 = Field(description="复权因子", default=np.nan)
    pre_close: np.float32 = Field(description="前收盘价", default=np.nan)
    high: np.float32 = Field(description="最高价", default=np.nan)
    open: np.float32 = Field(description="开盘价", default=np.nan)
    low: np.float32 = Field(description="最低价", default=np.nan)
    close: np.float32 = Field(description="收盘价", default=np.nan)
    deal_number: np.int64 = Field(description="区间成交笔数", default=0)
    volume: np.int64 = Field(description="区间成交量", default=0)
    amount: np.float64 = Field(description="区间成交额", default=np.nan)

    ask_price1: np.float32 = Field(description="1档委卖价", default=0)
    ask_price2: np.float32 = Field(description="2档委卖价", default=0)
    ask_price3: np.float32 = Field(description="3档委卖价", default=0)
    ask_price4: np.float32 = Field(description="4档委卖价", default=0)
    ask_price5: np.float32 = Field(description="5档委卖价", default=0)
    ask_price6: np.float32 = Field(description="6档委卖价", default=0)
    ask_price7: np.float32 = Field(description="7档委卖价", default=0)
    ask_price8: np.float32 = Field(description="8档委卖价", default=0)
    ask_price9: np.float32 = Field(description="9档委卖价", default=0)
    ask_price10: np.float32 = Field(description="10档委卖价", default=0)
    bid_price1: np.float32 = Field(description="1档委买价", default=0)
    bid_price2: np.float32 = Field(description="2档委买价", default=0)
    bid_price3: np.float32 = Field(description="3档委买价", default=0)
    bid_price4: np.float32 = Field(description="4档委买价", default=0)
    bid_price5: np.float32 = Field(description="5档委买价", default=0)
    bid_price6: np.float32 = Field(description="6档委买价", default=0)
    bid_price7: np.float32 = Field(description="7档委买价", default=0)
    bid_price8: np.float32 = Field(description="8档委买价", default=0)
    bid_price9: np.float32 = Field(description="9档委买价", default=0)
    bid_price10: np.float32 = Field(description="10档委买价", default=0)

    ask_volume1: np.int64 = Field(description="1档委卖量", default=0)
    ask_volume2: np.int64 = Field(description="2档委卖量", default=0)
    ask_volume3: np.int64 = Field(description="3档委卖量", default=0)
    ask_volume4: np.int64 = Field(description="4档委卖量", default=0)
    ask_volume5: np.int64 = Field(description="5档委卖量", default=0)
    ask_volume6: np.int64 = Field(description="6档委卖量", default=0)
    ask_volume7: np.int64 = Field(description="7档委卖量", default=0)
    ask_volume8: np.int64 = Field(description="8档委卖量", default=0)
    ask_volume9: np.int64 = Field(description="9档委卖量", default=0)
    ask_volume10: np.int64 = Field(description="10档委卖量", default=0)
    bid_volume1: np.int64 = Field(description="1档委买量", default=0)
    bid_volume2: np.int64 = Field(description="2档委买量", default=0)
    bid_volume3: np.int64 = Field(description="3档委买量", default=0)
    bid_volume4: np.int64 = Field(description="4档委买量", default=0)
    bid_volume5: np.int64 = Field(description="5档委买量", default=0)
    bid_volume6: np.int64 = Field(description="6档委买量", default=0)
    bid_volume7: np.int64 = Field(description="7档委买量", default=0)
    bid_volume8: np.int64 = Field(description="8档委买量", default=0)
    bid_volume9: np.int64 = Field(description="9档委买量", default=0)
    bid_volume10: np.int64 = Field(description="10档委买量", default=0)

    ask_num_orders1: np.int32 = Field(description="卖1档委托笔数", default=0)
    ask_num_orders2: np.int32 = Field(description="卖2档委托笔数", default=0)
    ask_num_orders3: np.int32 = Field(description="卖3档委托笔数", default=0)
    ask_num_orders4: np.int32 = Field(description="卖4档委托笔数", default=0)
    ask_num_orders5: np.int32 = Field(description="卖5档委托笔数", default=0)
    ask_num_orders6: np.int32 = Field(description="卖6档委托笔数", default=0)
    ask_num_orders7: np.int32 = Field(description="卖7档委托笔数", default=0)
    ask_num_orders8: np.int32 = Field(description="卖8档委托笔数", default=0)
    ask_num_orders9: np.int32 = Field(description="卖9档委托笔数", default=0)
    ask_num_orders10: np.int32 = Field(description="卖10档委托笔数", default=0)
    bid_num_orders1: np.int32 = Field(description="买1档委托笔数", default=0)
    bid_num_orders2: np.int32 = Field(description="买2档委托笔数", default=0)
    bid_num_orders3: np.int32 = Field(description="买3档委托笔数", default=0)
    bid_num_orders4: np.int32 = Field(description="买4档委托笔数", default=0)
    bid_num_orders5: np.int32 = Field(description="买5档委托笔数", default=0)
    bid_num_orders6: np.int32 = Field(description="买6档委托笔数", default=0)
    bid_num_orders7: np.int32 = Field(description="买7档委托笔数", default=0)
    bid_num_orders8: np.int32 = Field(description="买8档委托笔数", default=0)
    bid_num_orders9: np.int32 = Field(description="买9档委托笔数", default=0)
    bid_num_orders10: np.int32 = Field(description="买10档委托笔数", default=0)

    class Config:
        arbitrary_types_allowed = True
