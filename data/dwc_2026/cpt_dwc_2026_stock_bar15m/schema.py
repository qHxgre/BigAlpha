import numpy as np
import pandas as pd
from pydantic import Field
from warehouse.builder.base import BaseSchema


class CptDeepwinBar15mSchema(BaseSchema):
    """15min 价格数据"""

    date: np.datetime64 = Field(description="时间", default=np.nan)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan)
    cutoff_time: pd.StringDtype = Field(description="截断时间", default=np.nan)
    vwap: np.float32 = Field(description="下个15分钟的vwap价格", default=np.nan)
    close: np.float32 = Field(description="当前分钟的收盘价", default=np.nan)
    time_range: pd.StringDtype = Field(description="时间范围", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
