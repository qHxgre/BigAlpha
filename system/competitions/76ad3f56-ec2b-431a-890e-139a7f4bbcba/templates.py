"""注入到隔离子进程里运行的 runner 代码模板（参数化版本）。

评测分两步：先对每个提交跑「单因子分析」，再用入选的优质因子拼成「因子池」做回归。
public / private 两套评测除了数据集、日期区间、产物文件名不同，模板逻辑完全一致，
因此这里只保留原始模板字符串，由 build_sfa_runner / build_reg_runner 在运行时注入具体值，
public.py / private.py 不必再各写一份模板。

占位符约定（注入时被替换）：
    __USER_CODE__          用户提交的代码
    __DATASET__            因子计算所用数据集名
    __DATE_START__ / __DATE_END__   数据时间区间
    __RAW_FACTOR_FILE__ / __PROCESS_FACTOR_FILE__ / __FACTOR_ANALYZE_FILE__   单因子分析产物文件名
    __FACTOR_POOL_FILE__ / __FACTOR_REGRESSION_SCORE__                        回归阶段读入/产出文件
"""
from __future__ import annotations

# 第一步：跑单因子分析，并把原始因子、处理后因子、单因子得分分别落盘
_SFA_TEMPLATE = '''
__USER_CODE__

def judge_runner_main():
    import json
    import pandas as pd

    factor_data = main("__DATASET__", "__DATE_START__", "__DATE_END__")

    from bigmodule import M
    result = M.bigalpha_factorminer._latest(
        factor_data=factor_data,
        show=True,
    )

    # 把原始因子数据 raw_factor 落盘为 parquet 文件
    result["raw_factor"].to_parquet("__RAW_FACTOR_FILE__")

    # 把处理后的因子数据 process_factor 落盘为 parquet 文件
    result["process_factor"].to_parquet("__PROCESS_FACTOR_FILE__")

    # 把单因子得分 factor_analyze（dict）落盘为 json 文件
    with open("__FACTOR_ANALYZE_FILE__", "w", encoding="utf-8") as writer:
        json.dump(result["factor_analyze"], writer, ensure_ascii=False, default=str)
'''


# 第二步：跑因子池回归。因子池由评测系统汇总所有提交的优质因子后落盘（__FACTOR_POOL_FILE__），
# 这里只读取它并交给 bigalpha_factorminer 做回归，不调用任何用户代码，
# 因此模板里不注入 __USER_CODE__，可作为独立脚本运行（不依赖任何提交）。
_REG_TEMPLATE = '''
def judge_runner_main():
    import json
    import pandas as pd

    # 读取评测系统落盘的因子池（所有提交的优质因子按 date/instrument 合并而成）
    factor_pool = pd.read_parquet("__FACTOR_POOL_FILE__")

    from bigmodule import M
    result = M.bigalpha_factorminer._latest(
        factor_pool=factor_pool,
        process_pools=False,
        show=True,
    )

    # 将 per_factor_scores 数据落盘，
    result['factor_regression']['per_factor_scores'].to_parquet("__FACTOR_REGRESSION_SCORE__")
'''


def build_sfa_runner(
    *,
    dataset: str,
    date_start: str,
    date_end: str,
    raw_factor_file: str,
    process_factor_file: str,
    factor_analyze_file: str,
) -> str:
    """注入数据集/日期/产物文件名，返回单因子分析 runner 模板。

    注入完仍保留 __USER_CODE__ 占位符，由调用方在拿到用户代码后再替换。
    """
    return (
        _SFA_TEMPLATE
        .replace("__DATASET__", dataset)
        .replace("__DATE_START__", date_start)
        .replace("__DATE_END__", date_end)
        .replace("__RAW_FACTOR_FILE__", raw_factor_file)
        .replace("__PROCESS_FACTOR_FILE__", process_factor_file)
        .replace("__FACTOR_ANALYZE_FILE__", factor_analyze_file)
    )


def build_reg_runner(*, factor_pool_file: str, factor_regression_score: str) -> str:
    """注入因子池读入路径与回归得分产出路径，返回可独立运行的回归 runner 模板。"""
    return (
        _REG_TEMPLATE
        .replace("__FACTOR_POOL_FILE__", factor_pool_file)
        .replace("__FACTOR_REGRESSION_SCORE__", factor_regression_score)
    )
