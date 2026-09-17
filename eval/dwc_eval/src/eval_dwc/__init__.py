"""eval_dwc package.

蝶威量化2026年因子大赛评估系统
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
category = "量化比赛"
# 模块显示名
friendly_name = "蝶威量2026年因子大赛评估系统"
# 文档地址, optional
doc_url = "https://bigquant.com/wiki/"
# 是否自动缓存结果
cacheable = True


def run(
    data: I.port("因子数据: 包含列 (date, instrument, factor) 的DataFrame/DataSource"),
    check_data: I.port("因子检查数据: 包含列 (date, instrument, factor) 的DataFrame/DataSource") = None,
    factor_col: I.str("因子列名,如果不指定则优先使用'factor'列,否则使用date和instrument外的列") = None,
    show: I.bool("画出绩效图") = True,
)->[
    I.port(desc="输出数据", name="data")
]:

    if hasattr(data, 'read'):
        data = data.read()

    data['instrument'] = data['instrument'].astype('string')

    if not factor_col:
        if 'factor' in data.columns:
            factor_col = 'factor'
        else:
            candidate_cols = [col for col in data.columns if col not in {'date', 'instrument'}]
            if len(candidate_cols) == 0:
                raise ValueError("no factor column found")
            if len(candidate_cols) > 1:
                raise ValueError(f"multi factor candidate columns found {candidate_cols}, please set factor_col")
            factor_col = candidate_cols[0]

    if factor_col not in data.columns:
        raise ValueError(f"{factor_col=} not found in data")

    from .todo import run as todo_run
    results = todo_run(
        data=data,
        check_data=check_data,
        factor_field=factor_col,
        show=show,
    )
    return {"_result": results["result"], "_detail": results["details"]}

def post_run(outputs):
    """后置运行函数"""
    return outputs
