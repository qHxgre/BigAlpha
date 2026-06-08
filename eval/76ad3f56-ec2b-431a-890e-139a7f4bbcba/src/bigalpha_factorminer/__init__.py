"""bigalpha_factorminer package.

BigAlpha AI因子挖掘评估函数
"""

import numpy as np
import pandas as pd
import empyrical as ep
import structlog
from bigmodule import I
from pandas.api.types import is_integer_dtype, is_datetime64_any_dtype

logger = structlog.get_logger()

# 需要安装的第三方依赖包
# from bigmodule import R
# R.require("requests>=2.0", "isort==5.13.2")

# metadata
# 模块作者
author = "BigQuant"
# 模块分类
category = "BigAlpha"
# 模块显示名
friendly_name = "AI因子挖掘"
# 文档地址, optional
doc_url = "https://bigquant.com/wiki/"
# 是否自动缓存结果
cacheable = True


def run(
    factor_data: I.port("因子数据: 包含列 (date, instrument, factor) 的DataFrame"),
    factor_pool: I.port("因子池，包含因子池的的DataFrame，不能包含 factor 列"),
    show: I.bool("画出绩效图") = True,
)->[
    I.port("输出数据", "data")
]:
    # 检查date列
    if is_integer_dtype(factor_data['date']):
        factor_data['date'] = pd.to_datetime(factor_data['date'], format='%Y%m%d')

    # 确定因子列名
    candidate_cols = [col for col in factor_data.columns if col not in {'date', 'instrument'}]
    if len(candidate_cols) == 0:
        raise ValueError("未找到因子列")
    if len(candidate_cols) > 1:
        raise ValueError(f"factor_data 只能有一列因子！请检查： {candidate_cols}")
    if candidate_cols[0] != 'factor':
        factor_data = factor_data.rename(columns={candidate_cols[0]: 'factor'})
        logger.info('因子列名不为 factor, 自动重命名')
    
    candidate_cols = [col for col in factor_pool.columns if col not in {'date', 'instrument'}]
    if len(candidate_cols) < 2:
        raise ValueError("因子池数量不能少于2个")
    if 'factor' in factor_pool.columns:
        raise ValueError(f"因子池中不能包含列名为factor的因子！请检查： {candidate_cols}")

    # 合并因子数据
    merge_df = pd.merge(factor_data, factor_pool, how='inner', on=['date', 'instrument'])

    sd = merge_df['date'].min().strftime("%Y-%m-%d")
    ed = merge_df['date'].max().strftime("%Y-%m-%d")

    from .dataprocess.datachecker import DataCheck
    logger.info('========== 数据检查 ==========')
    dc = DataCheck(sd, ed)
    dc.validate(merge_df)


    from .dataprocess.dataprocess import DataProcess
    logger.info('========== 数据预处理 ==========')
    dp = DataProcess(sd, ed)
    pdf = dp.validate(factor_data)

    logger.info('========== 单因子分析 ==========')
    from .factoranalyze import FactorAnalyze
    fa = FactorAnalyze(sd, ed)
    fa_res = fa.score(pdf, plot=show)

    return {
        'ic_mean': fa_res.ic_mean,
        'ic_ir': fa_res.ic_ir,
        'sharpe_ratio': fa_res.sharpe_ratio,
        'stress_ic_ir': fa_res.stress_ic_ir
    }


def post_run(outputs):
    """后置运行函数"""
    return outputs
