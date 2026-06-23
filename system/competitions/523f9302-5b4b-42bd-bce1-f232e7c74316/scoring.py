"""评分相关的纯计算逻辑（不依赖 Judge 实例状态，便于单测）。

对应端到端模型赛道的评分规则：
    模型推理产出的截面分数经平台预处理（去极值 + 标准化 + BARRA 风格剔除取残差）后，
    等价于一个每日更新的单因子，直接以单因子四指标的截面排名加权作为团队最终得分：

        Score_final = 0.25*Rank_IC_mean + 0.25*Rank_IC_IR + 0.25*Rank_SR + 0.25*Rank_Stress

    与因子挖掘赛道不同：本赛道没有因子池回归（B 项），最终得分即此处的 A 项。
"""
from __future__ import annotations

import pandas as pd


def group_key(submission: dict) -> str:
    """队伍分组键。当前 API 未暴露队伍列表，按 user_id 分组（一个用户视作一个队伍）。"""
    return str(submission.get("user_id") or submission.get("id"))


def compute_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """计算最终得分：四个指标各占 25%，按截面 rank 百分位加权。

    对应评分规则：
        Score_final = 0.25*Rank_IC_mean + 0.25*Rank_IC_IR + 0.25*Rank_SR + 0.25*Rank_Stress
    四项均为在全体提交分数上做截面百分位排名（pct rank）后的结果，落在 [0, 1]。
    返回带 score 列的同一个 DataFrame。

    - ic_mean      ：评估区间内截面 IC 均值；
    - ic_ir        ：IC 序列的 IR（IC 均值 / IC 标准差）；
    - sharpe_ratio ：多空 10 分组组合的年化夏普比率；
    - stress_ic_ir ：分 regime 评估的稳健性指标。
    """
    df["score"] = (
        df["ic_mean"].rank(pct=True) * 0.25
        + df["ic_ir"].rank(pct=True) * 0.25
        + df["sharpe_ratio"].rank(pct=True) * 0.25
        + df["stress_ic_ir"].rank(pct=True) * 0.25
    )
    return df
