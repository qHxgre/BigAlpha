import numpy as np
import pandas as pd
from pydantic import Field
from warehouse.builder.base import BaseSchema

class CptDeepwinIndustryComponentSchema(BaseSchema):
    """行业分类 schema"""

    date: np.datetime64 = Field(description="日期", default=pd.NaT)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan)
    industry_level1_code: pd.StringDtype = Field(description="一级行业代码", default=np.nan)
    industry_level1_name: pd.StringDtype = Field(description="一级行业名称", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
