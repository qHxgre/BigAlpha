"""端到端模型赛道的完整评测器（组合各阶段 mixin）。

把两个阶段 mixin 与共享基类组装成一个可运行的 Judge：
    - ScoreMixin        模型评分：跑用户推理、平台预处理 + 单因子分析、状态记录、截面排名；
    - ScoringMixin      最终打分与汇总：回写最终得分、汇总运行结果；
    - EndToEndJudgeBase 共享配置与 mode 感知的产物路径。

on_tick 在这里编排各阶段的执行顺序。public.py / private.py 继承本类，只填 mode 与数据配置。

与因子挖掘赛道的区别：本赛道分数经风格剔除后直接作为单因子评估，没有因子池回归（B 项），
最终得分即四指标（IC mean / IC_IR / SR / Stress）的截面排名等权加权。
"""
from __future__ import annotations

from base import EndToEndJudgeBase
from final_scoring import ScoringMixin
from score import ScoreMixin


class EndToEndJudge(ScoreMixin, ScoringMixin, EndToEndJudgeBase):
    """端到端模型赛道评测器：on_submission 跑模型推理 + 单因子分析，on_tick 统一做排名/打分。"""

    competition_id = "523f9302-5b4b-42bd-bce1-f232e7c74316"

    def on_tick(self) -> None:
        """每个 tick 统一评分：截面排名 -> 回写最终得分 -> 分数池存档 -> 汇总运行结果。

        on_submission 只负责跑通用户代码并保留单因子分析结果，所有横向计算集中在此，
        保证每轮都用「全体已跑通提交」做一致的截面排名。

        注意顺序：score_final 提前到 save_score_pool 之前。save_score_pool 要把
        全体已跑通提交的 process/raw score parquet 逐个读出来做 outer merge，是本
        流程里最吃内存/磁盘的一步，之前发生过把进程打挂（硬杀，非 Python 异常，
        try/except 拦不住）的情况；一旦在这一步中断，后面的 score_final 就没机会
        跑，导致 leaderboard_score.csv 已刷新但 leaderboard_final.csv 没跟上。
        save_score_pool 只是存档、不参与打分，往后挪没有副作用，但能保证最终得分
        优先落地。
        """
        # 第一步：模型分数单因子横向排名，刷新 leaderboard_score.csv
        try:
            self.score_models()
        except Exception as e:
            self.log.error("score.failed", error=str(e), msg="模型分数排名失败")

        # 第二步：回写最终得分（提前，保证即便下一步把进程打挂，分数也已落地）
        try:
            self.score_final()
            self.log.info("tick.refreshed", msg="刷新得分榜单并回写最终得分")
        except Exception as e:
            self.log.error("final.failed", error=str(e), msg="回写最终得分失败")

        # 第三步：把全体已跑通提交的分数汇总成分数池存档（不参与打分）
        try:
            self.save_score_pool()
        except Exception as e:
            self.log.error("pool.failed", error=str(e), msg="构建分数池失败")

        # 第四步：汇总各提交运行结果
        try:
            self.summarize_submissions()
        except Exception as e:
            self.log.error("summary.failed", error=str(e), msg="汇总提交运行结果失败")
