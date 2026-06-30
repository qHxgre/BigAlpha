import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class Bigalpha2026StockBarKmSchema(BaseSchema):
    """K 分钟 K 线 + 盘口快照

    字段与 bigalpha_2026_stock_bar1m 保持一致, date 为 K 分钟 bar 的
    结束时刻(收盘时刻), 例如 5 分钟频率的早盘首个 bar 标注为 09:35,
    早盘收盘 bar 标注为 11:30, 尾盘收盘 bar 标注为 15:00。

    存储优化约定(下游读取务必知晓):
      - 所有价格(OHLC + 三档盘口价)以"分"存储, 即 = 元 × 100 的 int32,
        读取时需 / 100 还原为元。整数 delta 压缩远优于 float, 且无浮点误差。
      - amount(成交金额)同样以"分"存储, 用 int64 避免 float32 在累计到
        亿元量级后的精度退化(float32 仅 ~7 位有效数字)。
      - OHLC 缺失(停牌/无成交)以 -1 表示("分"下不可能为负, 可与真实价区分);
        三档盘口价沿用原约定以 0 表示缺失。
      - 委托笔数降为 int16(单档笔数通常 < 1000), 若校验发现 > 32767 需上调。
    """

    date: np.datetime64 = Field(description="日期(bar结束时刻)", default=0)
    instrument_id: np.int16 = Field(description="股票代码ID", default=np.nan)

    adjust_factor: np.float32 = Field(description="复权因子", default=np.nan)
    high: np.int32 = Field(description="最高价(单位:分=元×100, 缺失=-1)", default=-1)
    open: np.int32 = Field(description="开盘价(单位:分=元×100, 缺失=-1)", default=-1)
    low: np.int32 = Field(description="最低价(单位:分=元×100, 缺失=-1)", default=-1)
    close: np.int32 = Field(description="收盘价(单位:分=元×100, 缺失=-1)", default=-1)
    deal_number: np.int32 = Field(description="成交笔数(当日累计)", default=0)
    volume: np.int64 = Field(description="最新总成交量(当日累计, 单位:股)", default=0)
    amount: np.int64 = Field(description="最新成交金额(当日累计, 单位:分=元×100)", default=0)

    # 委托价格(单位:分=元×100, 缺失=0)
    ask_price1: np.int32 = Field(description="1档委卖价(单位:分=元×100)", default=0)
    ask_price2: np.int32 = Field(description="2档委卖价(单位:分=元×100)", default=0)
    ask_price3: np.int32 = Field(description="3档委卖价(单位:分=元×100)", default=0)
    bid_price1: np.int32 = Field(description="1档委买价(单位:分=元×100)", default=0)
    bid_price2: np.int32 = Field(description="2档委买价(单位:分=元×100)", default=0)
    bid_price3: np.int32 = Field(description="3档委买价(单位:分=元×100)", default=0)

    # 委托量
    ask_volume1: np.int32 = Field(description="1档委卖量", default=0)
    ask_volume2: np.int32 = Field(description="2档委卖量", default=0)
    ask_volume3: np.int32 = Field(description="3档委卖量", default=0)
    bid_volume1: np.int32 = Field(description="1档委买量", default=0)
    bid_volume2: np.int32 = Field(description="2档委买量", default=0)
    bid_volume3: np.int32 = Field(description="3档委买量", default=0)

    # 委托比数(降为 int16)
    ask_num_orders1: np.int16 = Field(description="卖1档委托笔数", default=0)
    ask_num_orders2: np.int16 = Field(description="卖2档委托笔数", default=0)
    ask_num_orders3: np.int16 = Field(description="卖3档委托笔数", default=0)
    bid_num_orders1: np.int16 = Field(description="买1档委托笔数", default=0)
    bid_num_orders2: np.int16 = Field(description="买2档委托笔数", default=0)
    bid_num_orders3: np.int16 = Field(description="买3档委托笔数", default=0)

    class Config:
        arbitrary_types_allowed = True
