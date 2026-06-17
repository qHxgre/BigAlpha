"""bigalpha_factorminer package.

BigAlpha AI因子挖掘评估函数
"""

import pandas as pd
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


_KEY_COLS = {'date', 'instrument'}


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    if is_datetime64_any_dtype(df['date']):
        return df
    if is_integer_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    else:
        df['date'] = pd.to_datetime(df['date'])
    return df


def _non_key_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _KEY_COLS]


def run(
    factor_data: I.port("因子数据: 包含列 (date, instrument, factor) 的DataFrame；为 None 时只进行因子池回归") = None,
    factor_pool: I.port("因子池，包含因子池的的DataFrame，不能包含 factor 列；为 None 时只进行单因子分析") = None,
    process_pools: I.bool("是否对因子池的数据进行预处理") = True,
    show: I.bool("画出绩效图") = True,
)->[
    I.port("输出数据", "data")
]:
    result = {}

    has_factor = factor_data is not None
    has_pool = factor_pool is not None
    if not has_factor and not has_pool:
        raise ValueError("factor_data 与 factor_pool 不能同时为空，至少提供其一")

    # ---------- 规范化单因子 ----------
    if has_factor:
        factor_data = _normalize_date(factor_data)
        factor_data['instrument'] = factor_data['instrument'].astype(str)

        # 确定因子列名
        candidate_cols = _non_key_columns(factor_data)
        if len(candidate_cols) == 0:
            raise ValueError("未找到因子列")
        if len(candidate_cols) > 1:
            raise ValueError(f"factor_data 只能有一列因子！请检查： {candidate_cols}")
        if candidate_cols[0] != 'factor':
            factor_data = factor_data.rename(columns={candidate_cols[0]: 'factor'})
            logger.info('因子列名不为 factor, 自动重命名')

        result['raw_factor'] = factor_data.copy()

    # ---------- 规范化因子池 ----------
    if has_pool:
        factor_pool = _normalize_date(factor_pool)
        factor_pool['instrument'] = factor_pool['instrument'].astype(str)
        pool_cols = _non_key_columns(factor_pool)
        if len(pool_cols) < 2:
            raise ValueError("因子池数量不能少于2个")
        if 'factor' in factor_pool.columns:
            raise ValueError(f"因子池中不能包含列名为factor的因子！请检查： {pool_cols}")

    # ---------- 组装待检查/回归的面板 ----------
    if has_factor and has_pool:
        merge_df = pd.merge(factor_data, factor_pool, how='inner', on=['date', 'instrument'])
        mode_desc = "合并后"
    elif has_factor:
        logger.info('未提供因子池，只进行单因子分析')
        merge_df = factor_data
        mode_desc = "单因子"
    else:
        logger.info('未提供单因子，只进行因子池回归')
        merge_df = factor_pool
        mode_desc = "因子池"

    sd = merge_df['date'].min().strftime("%Y-%m-%d")
    ed = merge_df['date'].max().strftime("%Y-%m-%d")
    logger.info(f'{mode_desc}时间范围: {sd} 至 {ed}')

    from .datachecker import DataCheck
    logger.info('========== 数据检查 ==========')
    DataCheck(sd, ed).validate(merge_df)

    from .dataprocess import DataProcess
    logger.info('========== 数据预处理 ==========')
    dp = DataProcess(sd, ed)
    if process_pools:
        # 对面板内全部因子（单因子和/或因子池）一并预处理
        pdf = dp.validate(merge_df)
    elif has_factor and has_pool:
        # 仅对单因子预处理，再与未处理的因子池合并
        process_factor = dp.validate(factor_data)
        pdf = pd.merge(process_factor, factor_pool, how='inner', on=['date', 'instrument'])
    elif has_factor:
        # 只有单因子时只需预处理单因子
        pdf = dp.validate(factor_data)
    else:
        # 只有因子池且不预处理，直接使用原始因子池
        pdf = factor_pool

    # ---------- 单因子分析（仅在提供单因子时执行） ----------
    if has_factor:
        result['process_factor'] = pdf[['date', 'instrument', 'factor']]

        from .factoranalyze import FactorAnalyze
        logger.info('========== 单因子分析 ==========')
        fa_res = FactorAnalyze(sd, ed).score(pdf[['date', 'instrument', 'factor']], plot=show)
        result['factor_analyze'] = fa_res.to_dict()
    else:
        logger.warning('未传入单因子，跳过单因子分析')

    # ---------- 因子池回归（仅在提供因子池时执行） ----------
    if has_pool:
        from .regmodel import ElasticNetRegress
        logger.info('========== 因子池回归 ==========')
        reg_res = ElasticNetRegress(sd, ed).score(pdf, plot=show)
        result['factor_regression'] = reg_res.to_dict()
    else:
        logger.warning('未传入因子池，跳过因子池回归检验')

    return result


def post_run(outputs):
    """后置运行函数"""
    return outputs
