"""注入到隔离子进程里运行的 runner 代码模板（参数化版本）。

端到端模型赛道：用户提交训练/推理代码，推理产出一个截面分数文件（date/instrument/score）。
评测时平台只替换 datasources / start_date / end_date 三个入参，由用户代码端到端生成分数；
评测系统拿到分数后，做平台预处理（去极值 + 标准化 + BARRA 风格剔除取残差）并算单因子四指标。

public / private 两套评测除了数据集、日期区间、产物文件名不同，模板逻辑完全一致，
因此这里只保留原始模板字符串，由 build_score_runner 在运行时注入具体值，
public.py / private.py 不必再各写一份模板。

占位符约定（注入时被替换）：
    __USER_CODE__          用户提交的代码（须定义 main(datasources, start_date, end_date)）
    __DATASETS__           推理所用数据集表名映射（dict，注入成 Python 字面量，非字符串）
    __DATE_START__ / __DATE_END__   评估（验证集）时间区间
    __RAW_SCORE_FILE__ / __PROCESS_SCORE_FILE__ / __SCORE_ANALYZE_FILE__   产物文件名

关于 __DATASETS__：
    模型可能要同时用到多频率行情表（1m/5m/15m/30m K 线 + 盘口快照），且各表在
    public / private 阶段是不同的物理表（带不同后缀）。因此注入一个 {逻辑名: 物理表名} 的
    dict，由评测系统按阶段填好物理表名，用户代码只认逻辑名、从 dict 里取实际表名拼 SQL。
    注入时用 repr() 渲染成 Python 字面量，所以模板里 main(__DATASETS__, ...) 不加引号。
"""
from __future__ import annotations

# 跑用户模型推理拿到截面分数，做平台预处理 + 单因子分析，并把原始分数、处理后分数、分析结果分别落盘。
_SCORE_TEMPLATE = '''
__USER_CODE__

def judge_runner_main():
    import json
    import pandas as pd

    # 用户代码端到端推理：返回 date/instrument/score 三列的截面分数
    score_data = main(__DATASETS__, "__DATE_START__", "__DATE_END__")

    # 分数经风格剔除后等价于一个每日更新的单因子，复用因子分析模块：
    # 模块以 factor 列为评估对象，这里把 score 重命名为 factor 后送入。
    if "score" in score_data.columns:
        score_data = score_data.rename(columns={"score": "factor"})

    from bigmodule import M
    result = M.bigalpha_eval._latest(
        factor_data=score_data,
        start_date="__DATE_START__",
        end_date="__DATE_END__",
        show=True,
    )

    # 把原始分数 raw_factor 落盘为 parquet 文件
    result["raw_factor"].to_parquet("__RAW_SCORE_FILE__")

    # 把平台预处理（去极值 + 标准化 + 风格剔除残差）后的分数落盘为 parquet 文件
    result["process_factor"].to_parquet("__PROCESS_SCORE_FILE__")

    # 把单因子分析结果 factor_analyze（dict）落盘为 json 文件
    with open("__SCORE_ANALYZE_FILE__", "w", encoding="utf-8") as writer:
        json.dump(result["factor_analyze"], writer, ensure_ascii=False, default=str)
'''


def build_score_runner(
    *,
    datasets: dict[str, str],
    date_start: str,
    date_end: str,
    raw_score_file: str,
    process_score_file: str,
    score_analyze_file: str,
) -> str:
    """注入数据集映射/日期/产物文件名，返回模型评分 runner 模板。

    datasets 是 {逻辑名: 物理表名} 的 dict，用 repr() 渲染成 Python 字面量注入到
    main(__DATASETS__, ...) 处（不带引号），用户代码以单个 dict 入参收下、按逻辑名取物理表名。

    注入完仍保留 __USER_CODE__ 占位符，由调用方在拿到用户代码后再替换。
    """
    return (
        _SCORE_TEMPLATE
        .replace("__DATASETS__", repr(datasets))
        .replace("__DATE_START__", date_start)
        .replace("__DATE_END__", date_end)
        .replace("__RAW_SCORE_FILE__", raw_score_file)
        .replace("__PROCESS_SCORE_FILE__", process_score_file)
        .replace("__SCORE_ANALYZE_FILE__", score_analyze_file)
    )
