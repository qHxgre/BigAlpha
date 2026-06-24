from typing import Literal

import pandas as pd


def cal_ic(
    df: pd.DataFrame,
    score_name: str,
    method: Literal["pearson", "kendall", "spearman"] = "spearman",
) -> float:
    """单日截面 IC：分数与下期收益的相关系数。"""
    return df["daily_ret"].corr(df[score_name], method=method)


def cut(df: pd.DataFrame, score_name: str, group_num: int) -> pd.DataFrame:
    """
    按分数分组（用于多空组合构建）。

    Args:
        df (pd.DataFrame): 需要分组的数据。
        score_name (str): 用于分组的分数列名。
        group_num (int): 分组的数量。

    Returns:
        pd.DataFrame: 新增一列 "group" 表示每个标的所在的分组编号（字符串）。
    """
    df = df.drop_duplicates(score_name)
    df.loc[:, "group"] = pd.qcut(
        df[score_name], q=group_num, labels=False, duplicates="drop"
    )
    df = df.dropna(subset=["group"], how="any")
    df["group"] = df["group"].apply(int).apply(str)
    return df
