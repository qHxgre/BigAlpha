"""jyc_eval package.

BigAlpha 评估函数
"""

import pandas as pd
import structlog
from bigmodule import I
from pandas.api.types import is_integer_dtype, is_datetime64_any_dtype
from .data import load_pool_pairs

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
friendly_name = "评估打分"
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


def _align_to_pool(pool_pairs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """把单因子数据对齐到官方股票池面板，统一 universe 与交易日。"""
    aligned = pd.merge(df, pool_pairs, how='right', on=['date', 'instrument'])
    return aligned.sort_values(["date", "instrument"]).reset_index(drop=True)


def run(
    factor_data: I.port("因子数据: 包含列 (date, instrument, factor) 的DataFrame") = None,
    start_date: I.str("评估窗口起始日 YYYY-MM-DD；官方权威窗口，用于裁定 instruments 与对齐面板。为空则回退到数据自身范围") = None,
    end_date: I.str("评估窗口结束日 YYYY-MM-DD；官方权威窗口。为空则回退到数据自身范围") = None,
    show: I.bool("画出绩效图") = True,
)->[
    I.port("输出数据", "data")
]:
    result = {}

    if factor_data is None:
        raise ValueError("factor_data 不能为空")

    # ---------- 规范化单因子 ----------
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

    # ---------- 确定官方权威窗口并加载股票池面板 ----------
    # 评估窗口与股票池以 bigalpha_2026_instruments 为准：start_date/end_date 为官方配置窗口，
    # 据此查官方面板，sd/ed 取面板真实交易日，保证所有提交的时间跨度与 universe 一致。
    if not start_date or not end_date:
        logger.warning(
            '未传入官方评估窗口 start_date/end_date，回退到数据自身范围（仅建议本地调试时使用）'
        )
        win_start = factor_data['date'].min().strftime("%Y-%m-%d")
        win_end = factor_data['date'].max().strftime("%Y-%m-%d")
    else:
        win_start, win_end = start_date, end_date
    pool_pairs = load_pool_pairs(win_start, win_end)

    # ---------- 对齐官方面板 ----------
    # 把因子 left-join 到官方面板：超窗 / 非成分股行被丢弃，未覆盖格补 NaN。
    # 此后所有口径都基于对齐面板（universe / 交易日跨提交一致）。
    aligned = _align_to_pool(pool_pairs, factor_data)
    sd = aligned['date'].min().strftime("%Y-%m-%d")
    ed = aligned['date'].max().strftime("%Y-%m-%d")
    logger.info(f'对齐中证1000历史成分后，官方评估窗口: {sd} 至 {ed}')

    logger.info('========== 数据检查 ==========')
    from .datachecker import DataCheck
    DataCheck(sd, ed).validate(aligned)

    from .dataprocess import DataProcess
    logger.info('========== 数据预处理 ==========')
    dp = DataProcess(sd, ed)
    pdf = dp.validate(aligned)

    logger.info('========== 预处理后数据检查 ==========')
    # 预处理（去极值/标准化/风格剔除取残差）可能引入新的 NaN 或异常值，
    # 打分前再对 pdf 走一遍同样的校验，确保参与打分的数据仍然合规。
    DataCheck(sd, ed).validate(pdf)

    result['process_factor'] = pdf[['date', 'instrument', 'factor']]

    from .factoranalyze import FactorAnalyze
    logger.info('========== 单因子分析 ==========')
    fa_res = FactorAnalyze(sd, ed).score(result['process_factor'], plot=show)
    result['factor_analyze'] = fa_res.to_dict()

    return result


def post_run(outputs):
    """后置运行函数"""
    return outputs
