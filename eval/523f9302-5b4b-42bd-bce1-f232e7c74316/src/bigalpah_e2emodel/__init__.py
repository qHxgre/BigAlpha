"""bigalpah_e2emodel package.

BigAlpha 端到端模型评估函数。

端到端模型的分数经风格剔除后等价于一个每日更新的单因子，因此评估流程为：
    数据校验 -> 平台预处理（去极值 / 标准化 / 风格剔除取残差） -> 单因子式打分。
最终团队得分由 ic_mean / ic_ir / 多空 Sharpe / 压力期 IC IR 四项的全场百分位排名等权相加。
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
friendly_name = "端到端模型"
# 文档地址, optional
doc_url = "https://bigquant.com/wiki/"
# 是否自动缓存结果
cacheable = True


_KEY_COLS = {"date", "instrument"}


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    if is_datetime64_any_dtype(df["date"]):
        return df
    if is_integer_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    else:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _non_key_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _KEY_COLS]


def run(
    score_data: I.port("分数数据: 包含列 (date, instrument, score) 的DataFrame/DataSource") = None,
    show: I.bool("画出绩效图") = True,
) -> [
    I.port("输出数据", "data")
]:
    result = {}

    if score_data is None:
        raise ValueError("score_data 不能为空")

    # ---------- 规范化分数 ----------
    score_data = _normalize_date(score_data)
    score_data["instrument"] = score_data["instrument"].astype(str)

    # 确定分数列名（date/instrument 之外只能有一列，且统一命名为 score）
    candidate_cols = _non_key_columns(score_data)
    if len(candidate_cols) == 0:
        raise ValueError("未找到分数列")
    if len(candidate_cols) > 1:
        raise ValueError(f"score_data 只能有一列分数！请检查： {candidate_cols}")
    if candidate_cols[0] != "score":
        score_data = score_data.rename(columns={candidate_cols[0]: "score"})
        logger.info("分数列名不为 score, 自动重命名")

    result["raw_score"] = score_data.copy()

    sd = score_data["date"].min().strftime("%Y-%m-%d")
    ed = score_data["date"].max().strftime("%Y-%m-%d")
    logger.info(f"分数时间范围: {sd} 至 {ed}")

    from .datachecker import DataCheck
    logger.info("========== 数据检查 ==========")
    DataCheck(sd, ed).validate(score_data)

    from .dataprocess import DataProcess
    logger.info("========== 数据预处理 ==========")
    pdf = DataProcess(sd, ed).validate(score_data)
    result["process_score"] = pdf[["date", "instrument", "score"]]

    from .scoreanalyze import ScoreAnalyze
    logger.info("========== 分数评估 ==========")
    sa_res = ScoreAnalyze(sd, ed).score(pdf[["date", "instrument", "score"]], plot=show)
    result["score_analyze"] = sa_res.to_dict()

    return result


def post_run(outputs):
    """后置运行函数"""
    return outputs
