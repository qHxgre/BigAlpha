import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class CptGifp2026InstrumentsSchema(BaseSchema):
    """沪深 300 历史成分股的 15 分钟因子截面。"""

    date: np.datetime64 = Field(
        description="15分钟因子计算时点", default=np.nan, primary=True
    )
    instrument: pd.StringDtype = Field(
        description="证券代码", default=np.nan, primary=True
    )
    name: pd.StringDtype = Field(description="证券简称", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
