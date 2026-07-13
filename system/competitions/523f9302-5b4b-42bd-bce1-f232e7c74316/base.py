"""端到端模型赛道的共享评测基类。

public / private 两套评测只差三处：mode、推理数据集、数据时间区间；其余逻辑（跑模型推理、
平台预处理 + 单因子分析、最终打分、汇总）完全一致。这些公共逻辑由 score / final_scoring 两个
mixin 提供，本基类负责把它们需要的「配置位」「mode 感知的产物路径」收敛到一处：

    - DATASETS / DATE_START / DATE_END：子类（public.py / private.py）填写的差异配置；
      DATASETS 是 {逻辑名: 物理表名} 的 dict，注入给用户代码的 main()，使模型能跨多张表
      （如 1m/5m/15m/30m K 线 + 盘口快照）取数，且各表在 public / private 阶段切换成对应后缀的物理表；
    - mode_suffix：public 为 ""，private 为 "_private"，拼进所有产物文件名，
      使两套评测共用同一个比赛目录也不会互相覆盖；
    - JUDGE_SCORE：把上面的配置注入 templates 的模板，得到本实例专用的 runner 代码。

子类只需声明 mode 与三项数据配置，不再重复任何逻辑。
"""
from __future__ import annotations

import os

from judge.judgebase import JudgeBase

import constants
import templates
from fileio import read_csv, read_json


def _with_suffix(filename: str, suffix: str) -> str:
    """在扩展名之前插入 mode 后缀：foo.json + '_private' -> foo_private.json。

    suffix 为空（public）时原样返回，保证与历史产物文件名一致。
    """
    if not suffix:
        return filename
    root, ext = os.path.splitext(filename)
    return f"{root}{suffix}{ext}"


class EndToEndJudgeBase(JudgeBase):
    """端到端模型赛道评测器的公共基类（不含具体阶段逻辑，逻辑在各 mixin 中）。"""

    # 单机只有 1 张 A100：用户代码大概率上 GPU 推理，多个提交并行会抢同一张卡的 80G 显存导致
    # 相互 OOM（内存上限只管主机内存、管不到显存）。故串行执行，GPU 独占。若要提高吞吐改为并行，
    # 需同步下调 MemoryLimitedUserRunner.MEM_LIMIT，保证 max_workers * MEM_LIMIT 稳稳小于 256 GiB。
    max_workers: int = 1

    # ---- 子类必填的差异配置 ----------------------------------------------
    # 模型推理所用数据集表名映射与数据（验证集）时间区间，public / private 各不相同。
    # DATASETS：{逻辑名: 物理表名}，逻辑名是用户代码里约定的 key（如 "bar1m"/"bar5m"/"snapshot"）。
    DATASETS: dict[str, str] = {}
    DATE_START: str = ""
    DATE_END: str = ""

    # ---- 只跑部分提交（调试 / 复测用）------------------------------------
    # SUBMISSION_IDS 非空时，整条流水线只处理这些 submission id：主循环只拉取它们来跑用户
    # 代码，截面排名 / 打分 / 汇总也只遍历它们。留空（默认）则跑全量。子类（public/private）
    # 临时复测某几个提交时填上即可，无需改其它逻辑：
    #     SUBMISSION_IDS = ["fe0722a2-887c-4dbe-bb9b-6634c0b392bb", ...]
    # MAX_PAGES 限制拉取页数，配合调试用；None 表示用 API 默认（全量翻页）。
    SUBMISSION_IDS: list[str] = []
    MAX_PAGES: int | None = None

    # ---- mode 感知的产物文件名 -------------------------------------------
    # public/private 共用同一个比赛目录（同一个 competition_id），靠文件名后缀隔离产物。

    @property
    def mode_suffix(self) -> str:
        """public 为空串，private 为 '_private'，拼进所有产物文件名以隔离两套评测。"""
        return "" if self.mode == "public" else "_private"

    # 每个提交目录下的产物（按 mode 加后缀，避免 public/private 互相覆盖）
    @property
    def raw_score_file(self) -> str:
        return _with_suffix(constants.RAW_SCORE_FILE, self.mode_suffix)

    @property
    def process_score_file(self) -> str:
        return _with_suffix(constants.PROCESS_SCORE_FILE, self.mode_suffix)

    @property
    def score_analyze_file(self) -> str:
        return _with_suffix(constants.SCORE_ANALYZE_FILE, self.mode_suffix)

    @property
    def score_status_file(self) -> str:
        return _with_suffix(constants.SCORE_STATUS_FILE, self.mode_suffix)

    # ---- 榜单目录下的产物 -------------------------------------------------
    @property
    def leaderboard_score_csv(self) -> str:
        """单因子分析榜单（截面 rank 后的最终得分快照：id + 四指标 + score）。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("leaderboard_score.csv", self.mode_suffix))

    @property
    def leaderboard_final_csv(self) -> str:
        """最终得分榜单（id / score，按 score 降序）。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("leaderboard_final.csv", self.mode_suffix))

    @property
    def score_pool_path(self) -> str:
        """分数池 parquet 的绝对路径：全体已跑通提交的处理后分数按 date/instrument 合并成宽表。

        本赛道没有回归环节，分数池纯作存档，不参与打分。
        """
        return os.path.join(self.leaderboard_dir, _with_suffix(constants.SCORE_POOL_FILE, self.mode_suffix))

    @property
    def score_pool_raw_path(self) -> str:
        """分数池「原始分数」parquet 的绝对路径。

        与 score_pool_path 入池的提交集合一致，只是取每个提交的 raw_score（未处理）
        而非 process_score，作为存档保留一份。
        """
        return os.path.join(self.leaderboard_dir, _with_suffix(constants.SCORE_POOL_RAW_FILE, self.mode_suffix))

    @property
    def submissions_summary_csv(self) -> str:
        """所有提交运行结果的汇总统计文件。"""
        return os.path.join(self.leaderboard_dir, _with_suffix("submissions_summary.csv", self.mode_suffix))

    # ---- 注入了本实例配置的 runner 模板 -----------------------------------
    @property
    def JUDGE_SCORE(self) -> str:
        """模型评分 runner 模板：注入数据集映射/日期/产物文件名，仍保留 __USER_CODE__ 占位符。"""
        assert self.DATASETS and self.DATE_START and self.DATE_END, "子类必须设置 DATASETS / DATE_START / DATE_END"
        return templates.build_score_runner(
            datasets=self.DATASETS,
            date_start=self.DATE_START,
            date_end=self.DATE_END,
            raw_score_file=self.raw_score_file,
            process_score_file=self.process_score_file,
            score_analyze_file=self.score_analyze_file,
        )

    # ---- 只跑部分提交时统一收敛的拉取逻辑 --------------------------------
    def query_constraints(self) -> dict:
        """限定拉取提交的范围：在父类约束（private 默认只跑入围者）之上，叠加
        SUBMISSION_IDS 子集过滤。两处拉取提交的入口（主循环与 _iter_submission_dirs）
        都走这里，保证「只跑部分提交」在跑用户代码、排名、打分、汇总各环节口径一致。
        """
        constraints = dict(super().query_constraints())
        if self.SUBMISSION_IDS:
            constraints["id__in"] = list(self.SUBMISSION_IDS)
        return constraints

    def _query_submissions_kwargs(self) -> dict:
        """统一组装 query_submissions 的关键字参数（约束 + 可选页数上限）。"""
        kwargs: dict = {
            "competition_id": self.competition_id,
            "constraints": self.query_constraints(),
        }
        if self.MAX_PAGES is not None:
            kwargs["max_pages"] = self.MAX_PAGES
        return kwargs

    def query_submissions(self) -> list[dict]:
        """按本比赛的约束（含 SUBMISSION_IDS 子集）拉取提交列表。"""
        return self.alphathon_api.query_submissions(**self._query_submissions_kwargs())

    # ---- 共享小工具 -------------------------------------------------------
    def heartbeat_fields(self) -> dict:
        """给心跳日志附加运行进度：total / done / remaining / ok / failed。

        - total：本场比赛当前提交总数，复用主循环每 tick 缓存的 _submission_total（不额外打 API）；
        - done / ok / failed：取自上一个 tick 落盘的 submissions_summary.csv 的 status 分布，
          done = 终态数（success/user_error/oom/timeout/file_error），ok = success，failed = done - ok；
        - remaining = total - done，涵盖「刚提交还没建目录」与 env_error（会重试）的提交。

        两个来源都最多滞后一个 tick；csv 尚未生成时只回退显示 total。
        """
        total = getattr(self, "_submission_total", None)
        df = read_csv(self.submissions_summary_csv, logger=self.log)
        if df is None or "status" not in df.columns:
            return {"total": total} if total is not None else {}
        counts = df["status"].value_counts().to_dict()
        done = sum(counts.get(s, 0) for s in constants.TERMINAL_STATUSES)
        ok = counts.get(constants.STATUS_SUCCESS, 0)
        fields = {"total": total, "done": done, "ok": ok, "failed": done - ok}
        if total is not None:
            fields["remaining"] = total - done
        return fields

    def score_status_path(self, submission: dict) -> str:
        """该提交评分状态文件的绝对路径。"""
        return os.path.join(self.submission_path(submission), self.score_status_file)

    def read_score_status(self, submission: dict) -> dict | None:
        """读取该提交的状态文件，不存在或读不出时返回 None。"""
        return read_json(self.score_status_path(submission), logger=self.log)

    def _iter_submission_dirs(self):
        """遍历 submissions 目录，逐个 yield (sid, submission, sub_dir)。

        只产出仍在本场比赛提交列表里的 sid，过滤掉目录残留的无效项；
        score_models / summarize_submissions 共用此遍历逻辑。
        设置 SUBMISSION_IDS 时只产出该子集，与主循环口径一致。
        """
        submissions = self.query_submissions()
        sub_by_id = {str(s["id"]): s for s in submissions}
        if not os.path.isdir(self.submission_dir):
            return
        for sid in os.listdir(self.submission_dir):
            submission = sub_by_id.get(sid)
            if submission is None:
                continue
            yield sid, submission, os.path.join(self.submission_dir, sid)
