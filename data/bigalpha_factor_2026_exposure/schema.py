import numpy as np
import pandas as pd
from pydantic import Field

from base import BaseSchema

class BigalphaFactor2026ExposureSchema(BaseSchema):
    """中证1000成分股的 Barra 风格因子暴露表（扩展自 bq_exposure / jq_style_factor）"""

    date: np.datetime64 = Field(description="日期", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan, primary=True)

    # 风格因子（Barra 10 因子，截面标准化后的暴露值）
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

    # 行业、市值与权重
    industry_level1_code: pd.StringDtype = Field(description="一级行业代码", default=np.nan)
    float_market_cap: np.double = Field(description="流通市值", default=np.nan)
    weights: np.double = Field(description="市值权重（截面归一化）", default=np.nan)
    ret: np.double = Field(description="当期收益率", default=np.nan)

    # 行业哑变量（31 个，0=不属于，1=属于）
    AGRIFOREST: np.int8 = Field(description="行业哑变量: 是否属于农林牧渔. 0: 不属于; 1: 属于", default=0)
    MINING: np.int8 = Field(description="行业哑变量: 是否属于采掘. 0: 不属于; 1: 属于", default=0)
    CHEM: np.int8 = Field(description="行业哑变量: 是否属于化工. 0: 不属于; 1: 属于", default=0)
    IRONSTEEL: np.int8 = Field(description="行业哑变量: 是否属于钢铁. 0: 不属于; 1: 属于", default=0)
    NONFERMETAL: np.int8 = Field(description="行业哑变量: 是否属于有色金属. 0: 不属于; 1: 属于", default=0)
    ELECTRONICS: np.int8 = Field(description="行业哑变量: 是否属于电子. 0: 不属于; 1: 属于", default=0)
    AUTO: np.int8 = Field(description="行业哑变量: 是否属于汽车. 0: 不属于; 1: 属于", default=0)
    HOUSEAPP: np.int8 = Field(description="行业哑变量: 是否属于家用电气. 0: 不属于; 1: 属于", default=0)
    FOODBEVER: np.int8 = Field(description="行业哑变量: 是否属于食品饮料. 0: 不属于; 1: 属于", default=0)
    TEXTILE: np.int8 = Field(description="行业哑变量: 是否属于纺织服饰. 0: 不属于; 1: 属于", default=0)
    LIGHTINDUS: np.int8 = Field(description="行业哑变量: 是否属于轻工制造. 0: 不属于; 1: 属于", default=0)
    HEALTH: np.int8 = Field(description="行业哑变量: 是否属于医药生物. 0: 不属于; 1: 属于", default=0)
    UTILITIES: np.int8 = Field(description="行业哑变量: 是否属于公用事业. 0: 不属于; 1: 属于", default=0)
    TRANSPORTATION: np.int8 = Field(description="行业哑变量: 是否属于交通运输. 0: 不属于; 1: 属于", default=0)
    REALESTATE: np.int8 = Field(description="行业哑变量: 是否属于房地产. 0: 不属于; 1: 属于", default=0)
    COMMETRADE: np.int8 = Field(description="行业哑变量: 是否属于商业贸易. 0: 不属于; 1: 属于", default=0)
    LEISERVICE: np.int8 = Field(description="行业哑变量: 是否属于休闲服务. 0: 不属于; 1: 属于", default=0)
    BANK: np.int8 = Field(description="行业哑变量: 是否属于银行. 0: 不属于; 1: 属于", default=0)
    NONBANKFINAN: np.int8 = Field(description="行业哑变量: 是否属于非银金融. 0: 不属于; 1: 属于", default=0)
    CONGLOMERATES: np.int8 = Field(description="行业哑变量: 是否属于综合. 0: 不属于; 1: 属于", default=0)
    CONMAT: np.int8 = Field(description="行业哑变量: 是否属于建筑材料. 0: 不属于; 1: 属于", default=0)
    BUILDDECO: np.int8 = Field(description="行业哑变量: 是否属于建筑装饰. 0: 不属于; 1: 属于", default=0)
    ELECEQP: np.int8 = Field(description="行业哑变量: 是否属于电气设备. 0: 不属于; 1: 属于", default=0)
    MACHIEQUIP: np.int8 = Field(description="行业哑变量: 是否属于机械设备. 0: 不属于; 1: 属于", default=0)
    AERODEF: np.int8 = Field(description="行业哑变量: 是否属于国防军工. 0: 不属于; 1: 属于", default=0)
    COMPUTER: np.int8 = Field(description="行业哑变量: 是否属于计算机. 0: 不属于; 1: 属于", default=0)
    MEDIA: np.int8 = Field(description="行业哑变量: 是否属于传媒. 0: 不属于; 1: 属于", default=0)
    TELECOM: np.int8 = Field(description="行业哑变量: 是否属于通信. 0: 不属于; 1: 属于", default=0)
    COAL: np.int8 = Field(description="行业哑变量: 是否属于煤炭. 0: 不属于; 1: 属于", default=0)
    PETRO: np.int8 = Field(description="行业哑变量: 是否属于石油石化. 0: 不属于; 1: 属于", default=0)
    ENVP: np.int8 = Field(description="行业哑变量: 是否属于环保. 0: 不属于; 1: 属于", default=0)
    BEAUTY: np.int8 = Field(description="行业哑变量: 是否属于美容护理. 0: 不属于; 1: 属于", default=0)

    class Config:
        arbitrary_types_allowed = True

# fmt: on
