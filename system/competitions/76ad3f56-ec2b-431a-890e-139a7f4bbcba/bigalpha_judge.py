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

        各阶段明细（sfa.ranked / pool.saved / regression.done / final.scored / summary.saved）
        统一降为 debug 只进日志文件；本方法把各阶段关键数字收进 stats，最后汇总成一行
        INFO 输出到终端（见 _emit_tick_summary）。阶段失败仍按 error 单独打，便于定位。
        """
        stats: dict = {}

        # 第一步：单因子横向排名（A 项），刷新 leaderboard_sfa.csv
        try:
            stats["sfa"] = self.score_sfa()
        except Exception as e:
            self.log.error("sfa.failed", error=str(e), msg="单因子排名失败")

        # 第二步：用排名靠前的因子拼出因子池，并跑一次因子池回归（产出 B 项所需的 ModelScore）
        try:
            stats["pool"] = self.save_factor_pool()
            if os.path.exists(self.factor_pool_path):
                stats["reg_s"] = self.run_regression()
        except Exception as e:
            self.log.error("regression.failed", error=str(e), msg="因子池回归失败")

        # 第三步：合成最终得分 0.3*A + 0.7*B 并回写
        try:
            final = self.score_final()
            stats["final"] = final.get("count")
            stats["with_b"] = final.get("with_b")
        except Exception as e:
            self.log.error("final.failed", error=str(e), msg="合成最终得分失败")

        # 第四步：汇总各提交运行结果
        try:
            self.summarize_submissions()
        except Exception as e:
            self.log.error("summary.failed", error=str(e), msg="汇总提交运行结果失败")

        # 收尾：把本轮各阶段数字暂存，供汇总成一行日志。
        # 非自适应时由本方法直接输出；自适应时交给 base 的计时包装在算出下一轮间隔后输出，
        # 这样 next= 显示的就是本轮实测耗时算出的真实间隔，而不是上一轮的旧值。
        self._tick_stats = stats
        if not self.adaptive_interval:
            self._emit_tick_summary()

    def _emit_tick_summary(self) -> None:
        """把本轮各阶段关键数字汇总成一行 INFO 输出到终端。

        形如：tick ✓ sfa=42 pool=118 reg=7.9s final=42(b=30) → next=3600s
        各阶段明细仍在日志文件（debug）里可查。stats 字段缺失（该阶段未跑或失败）时跳过不显示。
        """
        stats = getattr(self, "_tick_stats", {}) or {}
        parts = []
        if stats.get("sfa") is not None:
            parts.append(f"sfa={stats['sfa']}")
        if stats.get("pool") is not None:
            parts.append(f"pool={stats['pool']}")
        if stats.get("reg_s") is not None:
            parts.append(f"reg={stats['reg_s']:.1f}s")
        if stats.get("final") is not None:
            parts.append(f"final={stats['final']}(b={stats.get('with_b', 0)})")
        self.log.info("tick.done", stats=" ".join(parts), next=f"{self.tick_interval}s", msg="本轮评分完成")
