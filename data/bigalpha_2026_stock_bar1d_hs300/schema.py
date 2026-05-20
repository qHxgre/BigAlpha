import numpy as np
import pandas as pd
from pydantic import Field
from base import BaseSchema


class Bigalpha2026StockBar1dHs300Schema(BaseSchema):
    date: np.datetime64 = Field(description="日期", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan, primary=True)
    name: pd.StringDtype = Field(description="股票名称", default=np.nan)
    open: np.float32 = Field(description="开盘价（后复权）", default=np.nan)
    high: np.float32 = Field(description="最高价（后复权）", default=np.nan)
    low: np.float32 = Field(description="最低价（后复权）", default=np.nan)
    close: np.float32 = Field(description="收盘价（后复权）", default=np.nan)
    pre_close: np.float32 = Field(description="前收盘价（后复权）", default=np.nan)
    volume: np.float32 = Field(description="成交量（股）", default=np.nan)
    amount: np.float32 = Field(description="成交额（元）", default=np.nan)
    deal_number: np.int32 = Field(description="成交笔数", default=np.nan)
    change_ratio: np.float32 = Field(description="涨跌幅", default=np.nan)
    turn: np.float32 = Field(description="换手率", default=np.nan)
    adjust_factor: np.float32 = Field(description="复权因子", default=np.nan)
    upper_limit: np.float32 = Field(description="涨停价", default=np.nan)
    lower_limit: np.float32 = Field(description="跌停价", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
