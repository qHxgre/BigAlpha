import numpy as np
import pandas as pd
from pydantic import Field
from base import BaseSchema


class Bigalpha2026StockRoeSchema(BaseSchema):
    date: np.datetime64 = Field(description="交易日", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan, primary=True)
    report_date: np.datetime64 = Field(description="当日所用财务数据的报告期截止日", default=np.nan)
    ann_date: np.datetime64 = Field(description="当日所用财务数据的公告日", default=np.nan)
    roe_lf: np.float32 = Field(description="ROE（最新一期）= 归母净利润_lf / 归母所有者权益_lf，按最近公告日前向填充", default=np.nan)
    roe_ttm: np.float32 = Field(description="ROE（TTM）= 归母净利润_ttm / 归母所有者权益_lf，按最近公告日前向填充", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
