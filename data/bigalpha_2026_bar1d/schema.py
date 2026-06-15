import numpy as np
import pandas as pd
from pydantic import Field
from base import BaseSchema


class Bigalpha2026Bar1dSchema(BaseSchema):

    date: np.datetime64 = Field(description="日期", default=np.nan)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan)
    name: pd.StringDtype = Field(description="证券简称", default=np.nan)
    adjust_factor: np.double = Field(description="累计后复权因子", default=np.nan)
    pre_close: np.double = Field(description="昨收盘价（后复权）", default=np.nan)
    open: np.double = Field(description="开盘价（后复权）", default=np.nan)
    close: np.double = Field(description="收盘价（后复权）", default=np.nan)
    high: np.double = Field(description="最高价（后复权）", default=np.nan)
    low: np.double = Field(description="最低价（后复权）", default=np.nan)
    volume: np.int64 = Field(description="成交量", default=0)
    deal_number: np.int32 = Field(description="成交笔数", default=0)
    amount: np.double = Field(description="成交金额", default=np.nan)
    change_ratio: np.double = Field(description="涨跌幅（后复权）", default=0)
    turn: np.double = Field(description="换手率", default=0)
    upper_limit: np.double = Field(description="涨停价", default=np.nan)
    lower_limit: np.double = Field(description="跌停价", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
