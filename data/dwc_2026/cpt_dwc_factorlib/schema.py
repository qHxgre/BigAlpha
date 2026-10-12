import numpy as np
import pandas as pd
from pydantic import Field
from warehouse.builder.base import BaseSchema

class DeepWinFactorsSchema(BaseSchema):
    """DeepWin 因子库"""
    date: np.datetime64 = Field(description="日期", default=np.nan)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan)
    
    fac1: np.float32 = Field(description="因子1", default=np.nan)
    fac2: np.float32 = Field(description="因子2", default=np.nan)
    fac3: np.float32 = Field(description="因子3", default=np.nan)
    fac4: np.float32 = Field(description="因子4", default=np.nan)
    fac5: np.float32 = Field(description="因子5", default=np.nan)
    fac6: np.float32 = Field(description="因子6", default=np.nan)
    fac7: np.float32 = Field(description="因子7", default=np.nan)
    fac8: np.float32 = Field(description="因子8", default=np.nan)
    fac9: np.float32 = Field(description="因子9", default=np.nan)
    beta: np.float32 = Field(description="风格因子-beta", default=np.nan)
    book_value_to_price: np.float32 = Field(description="风格因子-book_value_to_price", default=np.nan)
    earnings_yield: np.float32 = Field(description="风格因子-earnings_yield", default=np.nan)
    growth: np.float32 = Field(description="风格因子-growth", default=np.nan)
    leverage: np.float32 = Field(description="风格因子-leverage", default=np.nan)
    liquidity: np.float32 = Field(description="风格因子-liquidity", default=np.nan)
    momentum_rstr: np.float32 = Field(description="风格因子-momentum_rstr", default=np.nan)
    size: np.float32 = Field(description="风格因子-size", default=np.nan)
    size_nonlinearity: np.float32 = Field(description="风格因子-size_nonlinearity", default=np.nan)
    volatility: np.float32 = Field(description="风格因子-volatility", default=np.nan)
    
    class Config:
        arbitrary_types_allowed = True
