import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class CptGifp2026ReturnSchema(BaseSchema):
    """沪深 300 历史成分股的下一交易日同分钟前向收益。"""

    date: np.datetime64 = Field(description="收益起点（15分钟采样时点）", default=0)
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan)
    return_rate: np.float32 = Field(
        description="下一交易日同分钟复权收盘价收益率",
        default=np.nan,
    )
    close: np.float32 = Field(description="起点分钟收盘价", default=np.nan)
    adjust_factor: np.float32 = Field(
        description="起点日累计后复权因子", default=np.nan
    )
    future_date: np.datetime64 = Field(description="收益终点", default=0)
    future_close: np.float32 = Field(description="终点分钟收盘价", default=np.nan)
    future_adjust_factor: np.float32 = Field(
        description="终点日累计后复权因子", default=np.nan
    )

    class Config:
        arbitrary_types_allowed = True
