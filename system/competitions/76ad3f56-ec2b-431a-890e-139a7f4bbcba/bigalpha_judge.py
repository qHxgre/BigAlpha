"""BigAlpha 因子挖掘比赛的完整评测器（组合各阶段 mixin）。

把三个阶段 mixin 与共享基类组装成一个可运行的 Judge：
    - SFAMixin          单因子分析（A 项）：跑用户代码、状态记录、截面排名；
    - RegressionMixin   因子池回归（B 项来源）：构建因子池、回归、算 B 项；
    - ScoringMixin      最终打分与汇总：合成 0.3*A + 0.7*B、汇总运行结果；
    - BigAlphaJudgeBase 共享配置与 mode 感知的产物路径。

on_tick 在这里编排各阶段的执行顺序。public.py / private.py 继承本类，只填 mode 与数据配置。
"""
from __future__ import annotations

import os

from base import BigAlphaJudgeBase
from final_scoring import ScoringMixin
from regression import RegressionMixin
from sfa import SFAMixin


class BigAlphaJudge(SFAMixin, RegressionMixin, ScoringMixin, BigAlphaJudgeBase):
    """因子挖掘比赛评测器：on_submission 跑单因子分析，on_tick 统一做排名/回归/打分。"""

    competition_id = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"

    def on_tick(self) -> None:
        """每个 tick 统一评分：截面排名(A) -> 构建因子池 -> 因子池回归(B) -> 合成最终得分。

        on_submission 只负责跑通用户代码并保留单因子分析结果，所有横向计算集中在此，
        保证每轮都用「全体已跑通提交」做一致的截面排名与回归。
        """
        # 第一步：单因子横向排名（A 项），刷新 leaderboard_sfa.csv
        try:
            self.score_sfa()
        except Exception as e:
            self.log.error("sfa.failed", error=str(e), msg="单因子排名失败")

        # 第二步：用排名靠前的因子拼出因子池，并跑一次因子池回归（产出 B 项所需的 ModelScore）
        try:
            self.save_factor_pool()
            if os.path.exists(self.factor_pool_path):
                self.run_regression()
        except Exception as e:
            self.log.error("regression.failed", error=str(e), msg="因子池回归失败")

        # 第三步：合成最终得分 0.3*A + 0.7*B 并回写
        try:
            self.score_final()
            self.log.info("tick.refreshed", msg="刷新单因子榜单并合成最终得分")
        except Exception as e:
            self.log.error("final.failed", error=str(e), msg="合成最终得分失败")

        # 第四步：汇总各提交运行结果
        try:
            self.summarize_submissions()
        except Exception as e:
            self.log.error("summary.failed", error=str(e), msg="汇总提交运行结果失败")
