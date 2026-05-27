"""评估主流程

串联：
1. 数据校验（DataCheck）
2. 数据预处理：去极值 + 标准化 + 风格剔除（DataProcess）
3. 单因子 A 项：IC_mean / IC_IR / 多空 SR / Stress IC_IR（FactorAnalyze）
4. Elastic Net 滚动 B 项：ModelScore（ElasticNetEvaluator）
5. 输出指标 + 可视化

A 项 / B 项的"团队 Rank 归一化"由官方全局回归完成，本地工具仅给出
原始指标和单因子归一化得分，便于参赛者横向对比自己的因子。
"""

from datetime import datetime

import dai
import numpy as np
import pandas as pd
import structlog
from pandas.api.types import is_integer_dtype

from bigmodule import I

logger = structlog.get_logger()


def run(
    data: I.port("因子数据: 包含列 (date, instrument, factor) 的DataFrame/DataSource"),
    show: I.bool("画出绩效图") = True,
):
    start_date = data["date"].min().strftime("%Y-%m-%d")
    end_date = data["date"].max().strftime("%Y-%m-%d")

    # 1. 数据校验 
    # 2. 数据预处理（去极值 / 标准化 / 风格剔除）
    from .dataprocess import DataProcess
    logger.info("========== 2. 数据预处理 ==========")
    processed = DataProcess().validate(data, "factor")

    # 3. A 项：单因子指标
    from .factoranalyze import FactorAnalyze
    logger.info("========== 3. 单因子 A 项指标 ==========")
    fa = FactorAnalyze(start_date, end_date)
    a_metrics = fa.validate(processed, "factor")

    # 单因子场景 Rank 退化：直接以原始指标作为代理。多因子团队评估时
    # 由官方全局回归再做 cross-team rank。
    factor_score = float(np.nanmean(
        [a_metrics["ic_mean"], a_metrics["ic_ir"], a_metrics["ls_sharpe"], a_metrics["stress_ic_ir"]]
    ))

    # 4. B 项：Elastic Net 滚动 ModelScore
    from .elasticnet import ElasticNetEvaluator
    logger.info("========== 4. Elastic Net B 项 ==========")
    panel = processed.rename(columns={"factor": "factor"}).copy()
    panel["trading_day"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel[["trading_day", "instrument", "factor"]].dropna(subset=["factor"])

    en = ElasticNetEvaluator(start_date, end_date)
    en_result = en.run(panel, ["factor"])
    per_factor = en_result["per_factor_scores"]
    weights_history = en_result["weights_history"]
    model_score = float(per_factor["model_score"].iloc[0]) if not per_factor.empty else np.nan

    # 5. 汇总得分
    score_data = pd.DataFrame({
        "ic_mean": [round(a_metrics["ic_mean"], 6)],
        "ic_ir": [round(a_metrics["ic_ir"], 6)],
        "ls_sharpe": [round(a_metrics["ls_sharpe"], 6)],
        "stress_ic_ir": [round(a_metrics["stress_ic_ir"], 6)],
        "factor_score_proxy": [round(factor_score, 6)],
        "model_score": [round(model_score, 6) if pd.notna(model_score) else np.nan],
        "selection_rate": [round(float(per_factor["selection_rate"].iloc[0]), 6)
                           if not per_factor.empty else np.nan],
    })

    details = {
        "a_metrics": a_metrics,
        "daily_ic": fa.daily_ic.reset_index().rename(columns={"index": "trading_day"}),
        "group_cumret": fa.group_cumret.reset_index(),
        "long_short_ret": fa.long_short_ret.reset_index().rename(columns={"index": "trading_day"}),
        "per_factor_scores": per_factor,
        "weights_history": weights_history,
    }

    # 6. 可视化
    if show:
        from .render import plot_ic, plot_group_cumret, plot_weights_history
        from IPython.display import HTML, display

        c1 = plot_ic(fa.daily_ic)
        c2 = plot_group_cumret(fa.group_cumret, fa.group_num)
        c3 = plot_weights_history(weights_history, ["factor"])

        html_content = f"""
        <div>
            <h1>A 项 单因子指标</h1>
            <ul>
                <li>IC_mean      = {a_metrics['ic_mean']:.4f}</li>
                <li>IC_IR        = {a_metrics['ic_ir']:.4f}</li>
                <li>多空 SR      = {a_metrics['ls_sharpe']:.4f}</li>
                <li>Stress IC_IR = {a_metrics['stress_ic_ir']:.4f}</li>
                <li>FACTOR(代理) = {factor_score:.4f}（单因子场景下取四指标均值）</li>
            </ul>
            <img src="data:image/png;base64,{c1}" alt="Daily IC"/>
            <br/>
            <img src="data:image/png;base64,{c2}" alt="Group Cumret"/>
            <br/>

            <h1>B 项 Elastic Net</h1>
            <ul>
                <li>ModelScore    = {model_score:.4f}</li>
                <li>SelectionRate = {(per_factor['selection_rate'].iloc[0] if not per_factor.empty else float('nan')):.4f}</li>
                <li>窗口长度=60 个交易日；步长=20 个交易日</li>
            </ul>
            <img src="data:image/png;base64,{c3}" alt="Rolling Weights"/>
        </div>
        """
        display(HTML(html_content))

    return dict(
        result=dai.DataSource.write_pickle(score_data),
        details=dai.DataSource.write_pickle(details),
    )


def post_run(outputs):
    return outputs
