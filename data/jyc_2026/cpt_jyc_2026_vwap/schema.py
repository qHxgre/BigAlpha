import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class CptJyc2026VwapSchema(BaseSchema):
    """未来 30 分钟 VWAP 收益标签。"""

    date: np.datetime64 = Field(
        description="采样时点（未来30分钟窗口起点）", default=0
    )
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan)
    vwap_return: np.float32 = Field(
        description="窗口终点价格相对未来30分钟VWAP的收益率",
        default=np.nan,
    )
    vwap: np.float32 = Field(
        description="未来30分钟成交量加权平均价", default=np.nan
    )
    end_price: np.float32 = Field(
        description="未来30分钟窗口终点最新有效成交价", default=np.nan
    )
    volume: np.int64 = Field(description="未来30分钟成交量", default=0)
    amount: np.float64 = Field(
        description="未来30分钟成交额（元）", default=np.nan
    )
    num_trades: np.int32 = Field(description="未来30分钟成交笔数", default=0)

    class Config:
        arbitrary_types_allowed = True
