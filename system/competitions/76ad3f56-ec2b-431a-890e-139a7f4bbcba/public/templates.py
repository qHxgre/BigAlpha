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
    import time
    import pandas as pd

    # 日期区间是「评估输出区间」：要求该区间内每个官方交易日都产出因子值（缺整天会被判不通过）。
    # 时序因子（滚动窗口/动量等）须在 __DATE_START__ 之前自行向前多取 warmup 历史，
    # 使输出覆盖到区间第一个交易日，否则会因缺交易日而被拒。
    _t0 = time.perf_counter()
    factor_data = main(__DATASETS__, "__DATE_START__", "__DATE_END__")
    print(f"[judge_runner] main() 全窗耗时 {time.perf_counter() - _t0:.2f}s", flush=True)

    from bigmodule import M
    result = M.bigalpha_eval._latest(
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
# 这里只读取它并交给 bigalpha_eval 做回归，不调用任何用户代码，
# 因此模板里不注入 __USER_CODE__，可作为独立脚本运行（不依赖任何提交）。
_REG_TEMPLATE = '''
def judge_runner_main():
    import json
    import pandas as pd

    # 读取评测系统落盘的因子池（所有提交的优质因子按 date/instrument 合并而成）
    factor_pool = pd.read_parquet("__FACTOR_POOL_FILE__")

    from bigmodule import M
    result = M.bigalpha_eval._latest(
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


# 未来函数切窗检测：只把用户代码在「截断窗」上再跑一次并落盘 raw factor，不做任何评估/比对。
# 起点与全窗一致（同一 __DATE_START__、同样的 warmup），只把 end_date 换成截断日 __CUTOFF__。
# 与 _SFA_TEMPLATE 复用同一份 __USER_CODE__，保证「同一段用户代码、只改评估结束日」这个唯一变量。
# 比对（detect_lookahead）由评测主进程 sfa.py 读全窗/截窗两份 parquet 完成，子进程只产数据，
# 因此这里不 import 任何检测代码，也不依赖 bigalpha_eval。
_LOOKAHEAD_TEMPLATE = '''
__USER_CODE__

def judge_runner_main():
    import time
    import pandas as pd

    # 只改 end_date 为截断日，其余入参与全窗完全一致；不跑 eval，直接把因子输出落盘。
    _t0 = time.perf_counter()
    raw_cut = main(__DATASETS__, "__DATE_START__", "__CUTOFF__")
    print(f"[judge_runner] main() 截断窗耗时 {time.perf_counter() - _t0:.2f}s", flush=True)

    # 强制把返回数据裁到目标检验窗口 [__DATE_START__, __CUTOFF__]：
    # 用户 main 可能没按 end_date 严格过滤输出（如把查询上界放宽了 buffer、或压根没裁），
    # 导致 raw_cut 含窗口外的行。检测只比对 date <= cutoff 的格子，窗口外的行是噪声，
    # 且会让存档的校验数据超出「用截断表跑 March」的语义，故在此按 date 硬裁到目标窗口。
    _date = pd.to_datetime(raw_cut["date"]).dt.normalize()
    _lo = pd.Timestamp("__DATE_START__").normalize()
    _hi = pd.Timestamp("__CUTOFF__").normalize()
    raw_cut = raw_cut[(_date >= _lo) & (_date <= _hi)]

    raw_cut.to_parquet("__RAW_FACTOR_CUT_FILE__")
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


def build_lookahead_runner(
    *,
    datasets: dict[str, str],
    date_start: str,
    cutoff: str,
    raw_factor_cut_file: str,
) -> str:
    """注入数据集映射/起始日/截断日/产物路径，返回切窗复算 runner 模板。

    与 build_sfa_runner 用同一份 __USER_CODE__（由调用方替换），差别只在评估结束日改成 cutoff
    且不跑 eval，只落盘截断窗 raw factor，供评测主进程与全窗 raw_factor 比对。
    """
    return (
        _LOOKAHEAD_TEMPLATE
        .replace("__DATASETS__", repr(datasets))
        .replace("__DATE_START__", date_start)
        .replace("__CUTOFF__", cutoff)
        .replace("__RAW_FACTOR_CUT_FILE__", raw_factor_cut_file)
    )
