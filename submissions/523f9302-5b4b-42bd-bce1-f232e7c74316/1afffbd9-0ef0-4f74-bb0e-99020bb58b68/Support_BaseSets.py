from typing import TypedDict, Union, List
import numpy as np

StandardMethod ={
    "adjust_factor": "direct",
    "pre_close": "direct",
    "open": "direct",
    "high": "direct",
    "low": "direct",
    "close": "direct",
    "ask_price1": "direct",
    "ask_price2": "direct",
    "ask_price3": "direct",
    "ask_price4": "direct",
    "ask_price5": "direct",
    "bid_price1": "direct",
    "bid_price2": "direct",
    "bid_price3": "direct",
    "bid_price4": "direct",
    "bid_price5": "direct",
    "deal_number": "log1p",
    "volume": "log1p",
    "amount": "log1p",
    "ask_volume1": "log1p",
    "ask_volume2": "log1p",
    "ask_volume3": "log1p",
    "ask_volume4": "log1p",
    "ask_volume5": "log1p",
    "bid_volume1": "log1p",
    "bid_volume2": "log1p",
    "bid_volume3": "log1p",
    "bid_volume4": "log1p",
    "bid_volume5": "log1p"
}

class SingleScale(TypedDict):
    """ 单尺度特征细节 """
    vector: Union[np.ndarray, List[np.ndarray]]     # (batch_size, seq_len, 17): 17 个一维向量
    matrix: Union[np.ndarray, List[np.ndarray]]     # (batch_size, seq_len, 4, 5): 4 个 5 档盘口矩阵

class Feature(TypedDict):
    """ 多尺度特征框架 """
    m01: SingleScale
    m05: SingleScale
    m15: SingleScale
    m30: SingleScale
    
class Label(TypedDict):
    """ 双标签框架细节 """
    rank:  Union[np.ndarray, List[np.ndarray]]      # (batch_size, ): 截面排序收益率
    return_:  Union[np.ndarray, List[np.ndarray]]   # (batch_size, ): 截面收益率

class BatchFeature(TypedDict):
    """ 单 batch 的特征组织 """
    date: str                                       # 日期
    code: list[str]                                 # 股票池
    usable: Union[np.ndarray, List[bool]]           # 表示样本是否可用（无缺失）
    feature: Feature                                # 包含四个时间尺度的多维度特征矩阵

class BatchLabel(TypedDict):
    """ 单 batch 的标签组织 """
    date: str                                       # 日期
    code: list[str]                                 # 股票池
    usable: Union[np.ndarray, List[bool]]           # 表示样本是否可用（无缺失）
    label: Label                                    # 包含两个标签的矩阵
