"""评分相关的纯计算逻辑（不依赖 Judge 实例状态，便于单测）。

对应评分规则：最终得分 Score_i = 0.3 * A_i + 0.7 * B_i。
    A_i：单因子分析四个指标各占 25%，截面 rank 百分位加权；
    B_i：因子池 Elastic Net 回归 ModelScore 的百分位归一化，未被选中则记 0。
"""
from __future__ import annotations

import pandas as pd


def group_key(submission: dict) -> str:
    """队伍分组键。当前 API 未暴露队伍列表，按 user_id 分组（一个用户视作一个队伍）。"""
    return str(submission.get("user_id") or submission.get("id"))


def compute_sfa_score(df: pd.DataFrame) -> pd.DataFrame:
    """计算单因子得分 A 项：四个指标各占 25%，按截面 rank 百分位加权。

    对应评分规则中的：
        A_i = 0.25*Rank_IC_mean + 0.25*Rank_IC_IR + 0.25*Rank_SR + 0.25*Rank_Stress
    四项均为在全体提交因子上做截面百分位排名（pct rank）后的结果，落在 [0, 1]。
    返回带 score 列的同一个 DataFrame。
    """
    df["score"] = (
        df["ic_mean"].rank(pct=True) * 0.25
        + df["ic_ir"].rank(pct=True) * 0.25
        + df["sharpe_ratio"].rank(pct=True) * 0.25
        + df["stress_ic_ir"].rank(pct=True) * 0.25
    )
    return df


def compute_b_scores(reg: pd.DataFrame) -> dict[str, float]:
    """从因子池回归产物（per_factor_scores）计算每个因子的 B 项得分。

    评分规则：
        B_i = 在全体被回归因子上做百分位归一化后的 ModelScore，落在 [0, 1]；
        若该因子未被 Elastic Net 选中（权重恒为 0），则 B_i = 0。

    入参 reg 含 factor / model_score（可选 selection_rate）列，factor 即提交 id。
    缺少必要列时返回空 dict（调用方据此把 B_i 视为 0）。
    """
    if "factor" not in reg.columns or "model_score" not in reg.columns:
        return {}

    reg = reg.copy()
    reg["model_score"] = pd.to_numeric(reg["model_score"], errors="coerce")

    # 被选中的判定：ModelScore 为正即说明跨窗口存在非零权重；
    # 若有 selection_rate 列则进一步要求其 > 0（权重并非恒为 0）。
    selected = reg["model_score"] > 0
    if "selection_rate" in reg.columns:
        sel_rate = pd.to_numeric(reg["selection_rate"], errors="coerce")
        selected = selected & (sel_rate > 0)

    # 在全体被回归因子上做百分位归一化，未被选中的因子强制置 0。
    reg["b_score"] = reg["model_score"].rank(pct=True)
    reg.loc[~selected, "b_score"] = 0.0
    reg["b_score"] = reg["b_score"].fillna(0.0)

    return {str(f): float(b) for f, b in zip(reg["factor"], reg["b_score"])}
