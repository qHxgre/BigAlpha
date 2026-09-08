import numpy as np
import pandas as pd
from pydantic import Field
from base import BaseSchema


class Bigalpha2026InstrumentsSchema(BaseSchema):
    date: np.datetime64 = Field(description="日期", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan, primary=True)
    name: pd.StringDtype = Field(description="证券简称", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
