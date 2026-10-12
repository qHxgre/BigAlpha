import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema


class CptGifp2026ExposureSchema(BaseSchema):
    """沪深 300 成分股的 BARRA 风格及行业暴露。"""

    date: np.datetime64 = Field(description="交易日", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(
        description="证券代码", default=np.nan, primary=True
    )

    SIZE: np.double = Field(description="风格因子: 市值", default=np.nan)
    BETA: np.double = Field(description="风格因子: 贝塔", default=np.nan)
    MOMENTUM: np.double = Field(description="风格因子: 传统动量", default=np.nan)
    RESVOL: np.double = Field(description="风格因子: 残差波动率", default=np.nan)
    SIZENL: np.double = Field(description="风格因子: 非线性市值", default=np.nan)
    BTOP: np.double = Field(description="风格因子: 账面市值比", default=np.nan)
    LIQUIDTY: np.double = Field(description="风格因子: 流动性", default=np.nan)
    EARNYILD: np.double = Field(description="风格因子: 盈利能力", default=np.nan)
    GROWTH: np.double = Field(description="风格因子: 成长", default=np.nan)
    LEVERAGE: np.double = Field(description="风格因子: 杠杆", default=np.nan)

    industry_level1_code: pd.StringDtype = Field(
        description="一级行业代码", default=np.nan
    )
    float_market_cap: np.double = Field(description="流通市值", default=np.nan)
    weights: np.double = Field(description="市值权重（截面归一化）", default=np.nan)
    ret: np.double = Field(description="当期收益率", default=np.nan)

    AGRIFOREST: np.int8 = Field(description="行业哑变量: 农林牧渔", default=0)
    MINING: np.int8 = Field(description="行业哑变量: 采掘", default=0)
    CHEM: np.int8 = Field(description="行业哑变量: 化工", default=0)
    IRONSTEEL: np.int8 = Field(description="行业哑变量: 钢铁", default=0)
    NONFERMETAL: np.int8 = Field(description="行业哑变量: 有色金属", default=0)
    ELECTRONICS: np.int8 = Field(description="行业哑变量: 电子", default=0)
    AUTO: np.int8 = Field(description="行业哑变量: 汽车", default=0)
    HOUSEAPP: np.int8 = Field(description="行业哑变量: 家用电气", default=0)
    FOODBEVER: np.int8 = Field(description="行业哑变量: 食品饮料", default=0)
    TEXTILE: np.int8 = Field(description="行业哑变量: 纺织服饰", default=0)
    LIGHTINDUS: np.int8 = Field(description="行业哑变量: 轻工制造", default=0)
    HEALTH: np.int8 = Field(description="行业哑变量: 医药生物", default=0)
    UTILITIES: np.int8 = Field(description="行业哑变量: 公用事业", default=0)
    TRANSPORTATION: np.int8 = Field(description="行业哑变量: 交通运输", default=0)
    REALESTATE: np.int8 = Field(description="行业哑变量: 房地产", default=0)
    COMMETRADE: np.int8 = Field(description="行业哑变量: 商业贸易", default=0)
    LEISERVICE: np.int8 = Field(description="行业哑变量: 休闲服务", default=0)
    BANK: np.int8 = Field(description="行业哑变量: 银行", default=0)
    NONBANKFINAN: np.int8 = Field(description="行业哑变量: 非银金融", default=0)
    CONGLOMERATES: np.int8 = Field(description="行业哑变量: 综合", default=0)
    CONMAT: np.int8 = Field(description="行业哑变量: 建筑材料", default=0)
    BUILDDECO: np.int8 = Field(description="行业哑变量: 建筑装饰", default=0)
    ELECEQP: np.int8 = Field(description="行业哑变量: 电气设备", default=0)
    MACHIEQUIP: np.int8 = Field(description="行业哑变量: 机械设备", default=0)
    AERODEF: np.int8 = Field(description="行业哑变量: 国防军工", default=0)
    COMPUTER: np.int8 = Field(description="行业哑变量: 计算机", default=0)
    MEDIA: np.int8 = Field(description="行业哑变量: 传媒", default=0)
    TELECOM: np.int8 = Field(description="行业哑变量: 通信", default=0)
    COAL: np.int8 = Field(description="行业哑变量: 煤炭", default=0)
    PETRO: np.int8 = Field(description="行业哑变量: 石油石化", default=0)
    ENVP: np.int8 = Field(description="行业哑变量: 环保", default=0)
    BEAUTY: np.int8 = Field(description="行业哑变量: 美容护理", default=0)

    class Config:
        arbitrary_types_allowed = True
