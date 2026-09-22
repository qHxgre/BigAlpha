"""进益资本 2026 分钟级单因子评估入口。"""

import pandas as pd

try:
    from bigmodule import I
except ImportError:  # 本地测试环境不依赖 BigQuant UI 类型系统
    class _I:
        @staticmethod
        def port(*args, **kwargs):
            return object
        str = bool = port
    I = _I()

author = "BigQuant"
category = "BigAlpha"
friendly_name = "进益资本评估函数"
doc_url = "https://bigquant.com/wiki/"
cacheable = True

def run(
    factor_data: I.port("因子数据: date, instrument, factor") = None,
    start_date: I.str("评估窗口起始日") = None,
    end_date: I.str("评估窗口结束日") = None,
    show: I.bool("画出绩效图") = False,
):
    """评估单个分钟级 submission，返回原始因子、处理后因子和五项指标。"""
    if factor_data is None:
        raise ValueError("factor_data 不能为空")

    if "date" not in factor_data.columns:
        raise ValueError("factor_data 必须包含 date 列")

    window_start = start_date or pd.to_datetime(factor_data["date"]).min().strftime("%Y-%m-%d %H:%M:%S")
    window_end = end_date or pd.to_datetime(factor_data["date"]).max().strftime("%Y-%m-%d %H:%M:%S")
    from .datachecker import DataCheck
    DataCheck(window_start, window_end).validate(factor_data)

    from .dataprocess import DataProcess
    processor = DataProcess(window_start, window_end)
    processed_data = processor.run(factor_data)

    from .factoranalyze import FactorAnalyze
    score = FactorAnalyze(window_start, window_end).score(processed_data, plot=show)


    return {
        "raw_factor": processor.raw_factor,
        "process_factor": processed_data[["date", "instrument", "factor"]].reset_index(drop=True),
        "factor_analyze": score.to_dict(),
    }


def post_run(outputs):
    return outputs
