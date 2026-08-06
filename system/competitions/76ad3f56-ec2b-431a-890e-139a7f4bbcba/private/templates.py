"""注入隔离子进程执行的单因子和因子池回归代码模板。"""
from __future__ import annotations


_SFA_TEMPLATE = '''
__USER_CODE__

def judge_runner_main():
    import json
    result = main(__DATASETS__, "__DATE_START__", "__DATE_END__")
    from bigmodule import M
    result = M.bigalpha_eval._latest(
        factor_data=result,
        start_date="__DATE_START__",
        end_date="__DATE_END__",
        show=True,
    )
    result["raw_factor"].to_parquet("raw_factor.parquet")
    result["process_factor"].to_parquet("process_factor.parquet")
    with open("factor_analyze.json", "w", encoding="utf-8") as f:
        json.dump(result["factor_analyze"], f, ensure_ascii=False, default=str)
'''

_REGRESSION_TEMPLATE = '''
def judge_runner_main():
    import pandas as pd
    from bigmodule import M
    pool = pd.read_parquet("__FACTOR_POOL__")
    result = M.bigalpha_eval._latest(
        factor_pool=pool,
        start_date="__DATE_START__",
        end_date="__DATE_END__",
        process_pools=False,
        show=True,
    )
    result["factor_regression"]["per_factor_scores"].to_csv(
        "__REGRESSION_CSV__", index=False, encoding="utf-8-sig"
    )
'''


def build_sfa_runner(user_code: str, datasets: dict[str, str], date_start: str, date_end: str) -> str:
    """渲染单个提交的评测 runner。"""
    return (
        _SFA_TEMPLATE
        .replace("__DATASETS__", repr(datasets))
        .replace("__DATE_START__", date_start)
        .replace("__DATE_END__", date_end)
        .replace("__USER_CODE__", user_code)
    )


def build_regression_runner(
    factor_pool: str,
    regression_csv: str,
    date_start: str,
    date_end: str,
) -> str:
    """渲染因子池回归 runner。"""
    return (
        _REGRESSION_TEMPLATE
        .replace("__FACTOR_POOL__", factor_pool)
        .replace("__REGRESSION_CSV__", regression_csv)
        .replace("__DATE_START__", date_start)
        .replace("__DATE_END__", date_end)
    )
