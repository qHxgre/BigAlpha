"""注入到隔离子进程里运行的 runner 代码模板（参数化版本）。

评测分两步：先对每个提交跑「单因子分析」，再用入选的优质因子拼成「因子池」做回归。
public / private 两套评测除了数据集、日期区间、产物文件名不同，模板逻辑完全一致，
因此这里只保留原始模板字符串，由 build_sfa_runner / build_reg_runner 在运行时注入具体值，
public.py / private.py 不必再各写一份模板。

占位符约定（注入时被替换）：
    __USER_CODE__          用户提交的代码
    __DATASETS__           因子计算所用数据集表名映射（dict，注入成 Python 字面量，非字符串）
    __DATE_START__ / __DATE_END__   数据时间区间
    __RAW_FACTOR_FILE__ / __PROCESS_FACTOR_FILE__ / __FACTOR_ANALYZE_FILE__   单因子分析产物文件名
    __FACTOR_POOL_FILE__ / __FACTOR_REGRESSION_SCORE__                        回归阶段读入/产出文件

关于 __DATASETS__：
    一个因子可能要同时用到多张表（如分钟 K 线 + 财务数据合成），且各表在 public / private
    阶段是不同的物理表（带不同后缀）。因此不再注入单个表名，而是注入一个 {逻辑名: 物理表名}
    的 dict，由评测系统按阶段填好物理表名，用户代码只认逻辑名、从 dict 里取实际表名拼 SQL。
    注入时用 repr() 渲染成 Python 字面量（如 {'bar1m': '..._test', 'financial': '..._test'}），
    所以模板里 main(__DATASETS__, ...) 不加引号。
"""
from __future__ import annotations

# 第一步：跑单因子分析，并把原始因子、处理后因子、单因子得分分别落盘
_SFA_TEMPLATE = '''
__USER_CODE__

def judge_runner_main():
    import json
    import pandas as pd

    # 日期区间是「评估输出区间」：要求该区间内每个官方交易日都产出因子值（缺整天会被判不通过）。
    # 时序因子（滚动窗口/动量等）须在 __DATE_START__ 之前自行向前多取 warmup 历史，
    # 使输出覆盖到区间第一个交易日，否则会因缺交易日而被拒。
    factor_data = main(__DATASETS__, "__DATE_START__", "__DATE_END__")

    from bigmodule import M
    result = M.bigalpha_factorminer._latest(
        factor_data=factor_data,
        start_date="__DATE_START__",
        end_date="__DATE_END__",
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
        start_date="__DATE_START__",
        end_date="__DATE_END__",
        process_pools=False,
        show=True,
    )

    # 将 per_factor_scores 落盘为 CSV（utf-8-sig 带 BOM，Excel 打开中文不乱码）
    result['factor_regression']['per_factor_scores'].to_csv(
        "__FACTOR_REGRESSION_SCORE__", index=False, encoding="utf-8-sig"
    )
'''


def build_sfa_runner(
    *,
    datasets: dict[str, str],
    date_start: str,
    date_end: str,
    raw_factor_file: str,
    process_factor_file: str,
    factor_analyze_file: str,
) -> str:
    """注入数据集映射/日期/产物文件名，返回单因子分析 runner 模板。

    datasets 是 {逻辑名: 物理表名} 的 dict（如 {"bar1m": "..._test", "financial": "..._test"}），
    用 repr() 渲染成 Python 字面量注入到 main(__DATASETS__, ...) 处（不带引号），
    用户代码以单个 dict 入参收下、按逻辑名取物理表名。

    注入完仍保留 __USER_CODE__ 占位符，由调用方在拿到用户代码后再替换。
    """
    return (
        _SFA_TEMPLATE
        .replace("__DATASETS__", repr(datasets))
        .replace("__DATE_START__", date_start)
        .replace("__DATE_END__", date_end)
        .replace("__RAW_FACTOR_FILE__", raw_factor_file)
        .replace("__PROCESS_FACTOR_FILE__", process_factor_file)
        .replace("__FACTOR_ANALYZE_FILE__", factor_analyze_file)
    )


def build_reg_runner(
    *,
    date_start: str,
    date_end: str,
    factor_pool_file: str,
    factor_regression_score: str,
) -> str:
    """注入日期区间与因子池读入/回归得分产出路径，返回可独立运行的回归 runner 模板。

    date_start/date_end 传给 _latest()，使回归阶段也用官方配置窗口裁 instruments 定 sd/ed
    并把因子池对齐到官方面板，保证回归窗口与单因子分析一致、跨提交可比。
    """
    return (
        _REG_TEMPLATE
        .replace("__DATE_START__", date_start)
        .replace("__DATE_END__", date_end)
        .replace("__FACTOR_POOL_FILE__", factor_pool_file)
        .replace("__FACTOR_REGRESSION_SCORE__", factor_regression_score)
    )
