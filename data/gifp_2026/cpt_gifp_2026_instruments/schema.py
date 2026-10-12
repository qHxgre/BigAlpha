import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class CptGifp2026InstrumentsSchema(BaseSchema):
    """沪深 300 每日历史成分股。"""

    date: np.datetime64 = Field(description="交易日", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(
        description="证券代码", default=np.nan, primary=True
    )
    name: pd.StringDtype = Field(description="证券简称", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
