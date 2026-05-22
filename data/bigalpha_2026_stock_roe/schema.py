import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class Bigalpha2026StockRoeSchema(BaseSchema):
    # 主键
    date: np.datetime64 = Field(description="公告日（PIT基准日）", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan, primary=True)

    # 报告期信息
    report_date: np.datetime64 = Field(description="最新报告期截止日", default=np.nan)

    # ROE 因子（三口径）
    roe_lf: np.float64 = Field(description="ROE（最新一期）= 归母净利润_lf / 归母净资产_lf", default=np.nan)
    roe_ttm: np.float64 = Field(description="ROE（TTM）= 归母净利润_ttm / 归母净资产_lf", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
