import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class Bigalpha2026StockBarKmSchema(BaseSchema):
    """K 分钟 K 线 + 盘口快照

    字段与 bigalpha_2026_stock_bar1m 保持一致, date 为 K 分钟 bar 的
    结束时刻(收盘时刻), 例如 5 分钟频率的早盘首个 bar 标注为 09:35,
    早盘收盘 bar 标注为 11:30, 尾盘收盘 bar 标注为 15:00。
    """

    date: np.datetime64 = Field(description="日期(bar结束时刻)", default=0)
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan)
    instrument_id: np.int16 = Field(description="股票代码ID", default=np.nan)

    adjust_factor: np.float32 = Field(description="复权因子", default=np.nan)
    pre_close: np.float32 = Field(description="前收盘价", default=np.nan)
    high: np.float32 = Field(description="最高价", default=np.nan)
    open: np.float32 = Field(description="开盘价", default=np.nan)
    low: np.float32 = Field(description="最低价", default=np.nan)
    close: np.float32 = Field(description="收盘价", default=np.nan)
    deal_number: np.int32 = Field(description="成交笔数(当日累计)", default=0)
    volume: np.int32 = Field(description="最新总成交量(当日累计)", default=0)
    amount: np.float32 = Field(description="最新成交金额(当日累计)", default=np.nan)

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
