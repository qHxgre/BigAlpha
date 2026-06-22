"""BigAlpha 因子挖掘比赛的共享评测基类。

public / private 两套评测只差三处：mode、因子数据集、数据时间区间；其余逻辑（单因子分析、
因子池回归、最终打分、汇总）完全一致。这些公共逻辑由 sfa / regression / final_scoring 三个
mixin 提供，本基类负责把它们需要的「配置位」「mode 感知的产物路径」收敛到一处：

    - DATASET / DATE_START / DATE_END：子类（public.py / private.py）填写的差异配置；
    - mode_suffix：public 为 ""，private 为 "-private"，拼进所有产物文件名，
      使两套评测共用同一个比赛目录也不会互相覆盖（factor_analyze.json vs factor_analyze-private.json）；
    - JUDGE_SFA / JUDGE_REG：把上面的配置注入 templates 的模板，得到本实例专用的 runner 代码。

子类只需声明 mode 与三项数据配置，不再重复任何逻辑。
"""
from __future__ import annotations

import os

from judge.judgebase import JudgeBase

import constants
import templates
from fileio import read_json


def _with_suffix(filename: str, suffix: str) -> str:
    """在扩展名之前插入 mode 后缀：foo.json + '-private' -> foo-private.json。

    suffix 为空（public）时原样返回，保证与历史产物文件名一致。
    """
    if not suffix:
        return filename
    root, ext = os.path.splitext(filename)
    return f"{root}{suffix}{ext}"


class BigAlphaJudgeBase(JudgeBase):
    """因子挖掘比赛评测器的公共基类（不含具体阶段逻辑，逻辑在各 mixin 中）。"""

    # ---- 子类必填的差异配置 ----------------------------------------------
    # 因子计算所用数据集名与数据时间区间，public / private 各不相同。
    DATASET: str = ""
    DATE_START: str = ""
    DATE_END: str = ""

    # 每个队伍最多入选因子池的因子数量
    FACTOR_POOL_TOP_N = 50

    # ---- mode 感知的产物文件名 -------------------------------------------
    # public/private 共用同一个比赛目录（同一个 competition_id），靠文件名后缀隔离产物。

    @property
    def mode_suffix(self) -> str:
        """public 为空串，private 为 '-private'，拼进所有产物文件名以隔离两套评测。"""
        return "" if self.mode == "public" else "-private"

    # 每个提交目录下的产物（按 mode 加后缀，避免 public/private 互相覆盖）
    @property
    def raw_factor_file(self) -> str:
        return _with_suffix(constants.RAW_FACTOR_FILE, self.mode_suffix)

    @property
    def process_factor_file(self) -> str:
        return _with_suffix(constants.PROCESS_FACTOR_FILE, self.mode_suffix)

    @property
    def factor_analyze_file(self) -> str:
        return _with_suffix(constants.FACTOR_ANALYZE_FILE, self.mode_suffix)

    @property
    def factor_regression_score_file(self) -> str:
        return _with_suffix(constants.FACTOR_REGRESSION_SCORE, self.mode_suffix)

    @property
    def sfa_status_file(self) -> str:
        return _with_suffix(constants.SFA_STATUS_FILE, self.mode_suffix)

    # ---- 榜单目录下的产物 -------------------------------------------------
    @property
    def leaderboard_reg_csv(self) -> str:
        """回归分析榜单（per_factor_scores 落盘处）。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("leaderboard_reg.csv", self.mode_suffix))

    @property
    def leaderboard_sfa_csv(self) -> str:
        """单因子分析榜单（截面 rank 后的 A 项得分快照）。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("leaderboard_sfa.csv", self.mode_suffix))

    @property
    def leaderboard_final_csv(self) -> str:
        """最终得分榜单（id / a_score / b_score / final_score）。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("leaderboard_final.csv", self.mode_suffix))

    @property
    def factor_pool_path(self) -> str:
        """因子池 parquet 的绝对路径（注入到回归模板里供子进程读取）。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("factor_pool.parquet", self.mode_suffix))

    @property
    def submissions_summary_csv(self) -> str:
        """所有提交运行结果的汇总统计文件。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("submissions_summary.csv", self.mode_suffix))

    @property
    def regression_runner_dir(self) -> str:
        """回归 runner 的运行目录（与提交目录互不干扰，按 mode 区分）。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("regression", self.mode_suffix))

    # ---- 注入了本实例配置的 runner 模板 -----------------------------------
    @property
    def JUDGE_SFA(self) -> str:
        """单因子分析 runner 模板：注入数据集/日期/产物文件名，仍保留 __USER_CODE__ 占位符。"""
        assert self.DATASET and self.DATE_START and self.DATE_END, "子类必须设置 DATASET / DATE_START / DATE_END"
        return templates.build_sfa_runner(
            dataset=self.DATASET,
            date_start=self.DATE_START,
            date_end=self.DATE_END,
            raw_factor_file=self.raw_factor_file,
            process_factor_file=self.process_factor_file,
            factor_analyze_file=self.factor_analyze_file,
        )

    @property
    def JUDGE_REG(self) -> str:
        """因子池回归 runner 模板：注入因子池读入路径与回归得分产出路径。"""
        return templates.build_reg_runner(
            factor_pool_file=self.factor_pool_path,
            factor_regression_score=self.leaderboard_reg_csv,
        )

    # ---- 共享小工具 -------------------------------------------------------
    def sfa_status_path(self, submission: dict) -> str:
        """该提交单因子分析状态文件的绝对路径。"""
        return os.path.join(self.submission_path(submission), self.sfa_status_file)

    def read_sfa_status(self, submission: dict) -> dict | None:
        """读取该提交的状态文件，不存在或读不出时返回 None。"""
        return read_json(self.sfa_status_path(submission), logger=self.log)

    def _iter_submission_dirs(self):
        """遍历 submissions 目录，逐个 yield (sid, submission, sub_dir)。

        只产出仍在本场比赛提交列表里的 sid，过滤掉目录残留的无效项；
        score_sfa / save_factor_pool / summarize_submissions 共用此遍历逻辑。
        """
        submissions = self.alphathon_api.query_submissions(competition_id=self.competition_id)
        sub_by_id = {str(s["id"]): s for s in submissions}
        if not os.path.isdir(self.submission_dir):
            return
        for sid in os.listdir(self.submission_dir):
            submission = sub_by_id.get(sid)
            if submission is None:
                continue
            yield sid, submission, os.path.join(self.submission_dir, sid)
