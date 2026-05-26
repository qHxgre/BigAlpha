import numpy as np
import pandas as pd
from pydantic import Field

from warehouse.builder.base import BaseSchema


# fmt: off
class BqExposureSchema(BaseSchema):
    """因子暴露表: 扩展自 jq_style_factor"""

    date: np.datetime64 = Field(description="日期", default=np.nan)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan)
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

    industry_level1_code: pd.StringDtype = Field(description="一级行业", default=np.nan)
    float_market_cap: np.double = Field(description="市值", default=np.nan)
    weights: np.double = Field(description="市值权重", default=np.nan)
    ret: np.double = Field(description="收益率", default=np.nan)

    AGRIFOREST: np.int8 = Field(description="行业哑变量: 是否属于农林牧渔. 0: 不属于; 1: 属于", default=0)  # 农林牧渔 (sw2014, sw2021)
    MINING: np.int8 = Field(description="行业哑变量: 是否属于采掘. 0: 不属于; 1: 属于", default=0)  # 采掘 (sw2014)
    CHEM: np.int8 = Field(description="行业哑变量: 是否属于化工. 0: 不属于; 1: 属于", default=0)  # 化工 (sw2014), 基础化工 (sw2021)
    IRONSTEEL: np.int8 = Field(description="行业哑变量: 是否属于钢铁. 0: 不属于; 1: 属于", default=0)  # 钢铁 (sw2014, sw2021)
    NONFERMETAL: np.int8 = Field(description="行业哑变量: 是否属于有色金属. 0: 不属于; 1: 属于", default=0)  # 有色金属 (sw2014, sw2021)
    ELECTRONICS: np.int8 = Field(description="行业哑变量: 是否属于电子. 0: 不属于; 1: 属于", default=0)  # 电子 (sw2014, sw2021)
    AUTO: np.int8 = Field(description="行业哑变量: 是否属于汽车. 0: 不属于; 1: 属于", default=0)  # 汽车 (sw2014, sw2021)
    HOUSEAPP: np.int8 = Field(description="行业哑变量: 是否属于家用电气. 0: 不属于; 1: 属于", default=0)  # 家用电器 (sw2014, sw2021)
    FOODBEVER: np.int8 = Field(description="行业哑变量: 是否属于食品饮料. 0: 不属于; 1: 属于", default=0)  # 食品饮料 (sw2014, sw2021)
    TEXTILE: np.int8 = Field(description="行业哑变量: 是否属于纺织服饰. 0: 不属于; 1: 属于", default=0)  # 纺织服饰 (sw2014, sw2021)
    LIGHTINDUS: np.int8 = Field(description="行业哑变量: 是否属于轻工制造. 0: 不属于; 1: 属于", default=0)  # 轻工制造 (sw2014, sw2021)
    HEALTH: np.int8 = Field(description="行业哑变量: 是否属于医药生物. 0: 不属于; 1: 属于", default=0)  # 医药生物 (sw2014, sw2021)
    UTILITIES: np.int8 = Field(description="行业哑变量: 是否属于公用事业. 0: 不属于; 1: 属于", default=0)  # 公用事业 (sw2014, sw2021)
    TRANSPORTATION: np.int8 = Field(description="行业哑变量: 是否属于交通运输. 0: 不属于; 1: 属于", default=0)  # 交通运输 (sw2014, sw2021)
    REALESTATE: np.int8 = Field(description="行业哑变量: 是否属于房地产. 0: 不属于; 1: 属于", default=0)  # 房地产 (sw2014, sw2021)
    COMMETRADE: np.int8 = Field(description="行业哑变量: 是否属于商业贸易. 0: 不属于; 1: 属于", default=0)  # 商业贸易 (sw2014), 商贸零售 (sw2021)
    LEISERVICE: np.int8 = Field(description="行业哑变量: 是否属于休闲服务. 0: 不属于; 1: 属于", default=0)  # 休闲服务 (sw2014), 社会服务 (sw2021)
    BANK: np.int8 = Field(description="行业哑变量: 是否属于银行. 0: 不属于; 1: 属于", default=0)  # 银行 (sw2014, sw2021)
    NONBANKFINAN: np.int8 = Field(description="行业哑变量: 是否属于非银金融. 0: 不属于; 1: 属于", default=0)  # 非银金融 (sw2014, sw2021)
    CONGLOMERATES: np.int8 = Field(description="行业哑变量: 是否属于综合. 0: 不属于; 1: 属于", default=0)  # 综合 (sw2014, sw2021)
    CONMAT: np.int8 = Field(description="行业哑变量: 是否属于建筑材料. 0: 不属于; 1: 属于", default=0)  # 建筑材料 (sw2014, sw2021)
    BUILDDECO: np.int8 = Field(description="行业哑变量: 是否属于建筑装饰. 0: 不属于; 1: 属于", default=0)  # 建筑装饰 (sw2014, sw2021)
    ELECEQP: np.int8 = Field(description="行业哑变量: 是否属于电气设备. 0: 不属于; 1: 属于", default=0)  # 电气设备 (sw2014), 电力设备 (sw2021)
    MACHIEQUIP: np.int8 = Field(description="行业哑变量: 是否属于机械设备. 0: 不属于; 1: 属于", default=0)  # 机械设备 (sw2014, sw2021)
    AERODEF: np.int8 = Field(description="行业哑变量: 是否属于国防军工. 0: 不属于; 1: 属于", default=0)  # 国防军工 (sw2014, sw2021)
    COMPUTER: np.int8 = Field(description="行业哑变量: 是否属于计算机. 0: 不属于; 1: 属于", default=0)  # 计算机 (sw2014, sw2021)
    MEDIA: np.int8 = Field(description="行业哑变量: 是否属于传媒. 0: 不属于; 1: 属于", default=0)  # 传媒 (sw2014, sw2021)
    TELECOM: np.int8 = Field(description="行业哑变量: 是否属于通信. 0: 不属于; 1: 属于", default=0)  # 通信 (sw2014, sw2021)
    COAL: np.int8 = Field(description="行业哑变量: 是否属于煤炭. 0: 不属于; 1: 属于", default=0)  # 煤炭 (sw2021)
    PETRO: np.int8 = Field(description="行业哑变量: 是否属于石油石化. 0: 不属于; 1: 属于", default=0)  # 石油石化 (sw2021)
    ENVP: np.int8 = Field(description="行业哑变量: 是否属于环保. 0: 不属于; 1: 属于", default=0)  # 环保 (sw2021)
    BEAUTY: np.int8 = Field(description="行业哑变量: 是否属于美容护理. 0: 不属于; 1: 属于", default=0)  # 美容护理 (sw2021)

    class Config:
        arbitrary_types_allowed = True

# fmt: on
