"""BigAlpha 因子挖掘比赛的共享评测基类。

public / private 两套评测只差三处：mode、因子数据集、数据时间区间；其余逻辑（单因子分析、
因子池回归、最终打分、汇总）完全一致。这些公共逻辑由 sfa / regression / final_scoring 三个
mixin 提供，本基类负责把它们需要的「配置位」「mode 感知的产物路径」收敛到一处：

    - DATASETS / DATE_START / DATE_END：子类（public.py / private.py）填写的差异配置；
      DATASETS 是 {逻辑名: 物理表名} 的 dict，注入给用户代码的 main()，使一个因子能跨多张表
      （如分钟 K 线 + 财务数据）合成，且各表在 public / private 阶段切换成对应后缀的物理表；
    - mode_suffix：public 为 ""，private 为 "-private"，拼进所有产物文件名，
      使两套评测共用同一个比赛目录也不会互相覆盖（factor_analyze.json vs factor_analyze-private.json）；
    - JUDGE_SFA / JUDGE_REG：把上面的配置注入 templates 的模板，得到本实例专用的 runner 代码。

子类只需声明 mode 与三项数据配置，不再重复任何逻辑。
"""
from __future__ import annotations

import os
import time

from judge.judgebase import JudgeBase

import constants
import templates
from constants import STATUS_SUCCESS, TERMINAL_STATUSES
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
    # 因子计算所用数据集表名映射与数据时间区间，public / private 各不相同。
    # DATASETS：{逻辑名: 物理表名}，逻辑名是用户代码里约定的 key（如 "bar1m"/"financial"），
    DATASETS: dict[str, str] = {}
    DATE_START: str = ""
    DATE_END: str = ""

    # 每个队伍最多入选因子池的因子数量
    FACTOR_POOL_TOP_N = 50

    # ---- 未来函数截断表复算检测 -------------------------------------------
    # 单因子分析跑通后，用「只截到 cutoff 的物理表」把用户代码再复算一次，与全窗输出在
    # date<=cutoff 上逐格比对：无未来函数的因子两次结果应完全一致；偷看 cutoff 之后数据的
    # 因子，因截断表取不到未来行、尾部因子值变化（NaN/缺行），即被检出。命中直接判 -2 并剔除
    # 产物（不进 A 项排名、不进因子池），前端提示「疑似有未来函数」。
    #
    # 为什么用截断物理表而非只改 end_date：用户代码常自行把查询上界放宽 buffer（如 end_date+7d）
    # 再去查物理表，只改 end_date 参数根本砍不掉它偷看的未来行。把物理表本身截到 cutoff，
    # 才是数据可见性的硬边界，buffer 再大也取不到未来数据。
    LOOKAHEAD_ENABLED = True
    # 固定截断日（cutoff）：须落在 [DATE_START, DATE_END] 内的真实交易日，且 < DATE_END。
    # 检测复算时 end_date 设为它，并把数据表换成截断到该日的物理表（见 LOOKAHEAD_DATASETS）。
    LOOKAHEAD_CUTOFF: str = ""
    # 截断数据表映射：{逻辑名: 截断物理表名}，逻辑名须是 DATASETS 的子集。
    # 截断表须与原表「同下界」（保留一致的 warmup 历史，避免正常时序因子头部因缺 warmup 被误判）、
    # 「上界砍到 cutoff」。未在此列出的逻辑名沿用 DATASETS 原表（该表维度不做截断，其未来函数无法检出）。
    LOOKAHEAD_DATASETS: dict[str, str] = {}
    # 浮点比对容差（语义同 numpy.isclose）与判为泄漏所需的最小差异格占比。
    LOOKAHEAD_RTOL = 1e-5
    LOOKAHEAD_ATOL = 1e-8
    LOOKAHEAD_MIN_DIFF_RATIO = 1e-4

    # ---- 只跑部分提交（调试 / 复测用）------------------------------------
    # SUBMISSION_IDS 非空时，整条流水线只处理这些 submission id：主循环只拉取它们来跑用户
    # 代码，截面排名 / 回归 / 汇总也只遍历它们。留空（默认）则跑全量。子类（public/private）
    # 临时复测某几个提交时填上即可，无需改其它逻辑：
    #     SUBMISSION_IDS = ["00c316c9-b866-40a6-ad8d-1034865d24c5", ...]
    # MAX_PAGES 限制拉取页数，配合调试用；None 表示用 API 默认（全量翻页）。
    SUBMISSION_IDS: list[str] = []
    MAX_PAGES: int | None = None

    # ---- 自适应评估间隔 ---------------------------------------------------
    # 本比赛特有：on_tick 里的 Elastic Net 回归计算量随全局因子数增长，固定间隔会在因子池
    # 变大后频繁空转或排队堆积。开启 adaptive_interval 后，下一轮间隔按上一轮实际评估耗时自调：
    #     t_next = max(k * t_last_run, t_min)，k = tick_safety_factor，t_min = tick_min_interval
    # 比赛初期因子少、间隔短；后期因子池扩大、间隔自动拉长，无需人工干预。
    # 关闭时（默认）退化为父类的固定 tick_interval —— 私榜逐日增量构建用固定节奏即可。
    #
    # 实现上不改动父类主循环：父类每轮先调 on_tick() 再 sleep(self.tick_interval)。这里给
    # 本实例的 on_tick 包一层计时，在每轮跑完后按公式回写 self.tick_interval（普通属性），
    # 下一次 sleep 就会用上新间隔。
    adaptive_interval: bool = False
    tick_safety_factor: float = 1.5   # k：安全系数
    tick_min_interval: int = 3600     # t_min：最小间隔（秒），默认 1 小时

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 给本实例的 on_tick（无论由哪个 mixin 提供）包一层计时，跑完后据耗时回写 tick_interval。
        if self.adaptive_interval:
            self._wrap_on_tick_with_adaptive_interval()

    def next_tick_interval(self, last_run_seconds: float) -> int:
        """据上一轮评估实测耗时按 t_next = max(k * t_last_run, t_min) 算下一轮 sleep 秒数。"""
        return int(max(self.tick_safety_factor * last_run_seconds, self.tick_min_interval))

    def _wrap_on_tick_with_adaptive_interval(self) -> None:
        """把实例上的 on_tick 包成「计时 + 回写 tick_interval」版本。

        父类主循环在 on_tick 之后读取 self.tick_interval 决定 sleep，所以在 on_tick 末尾
        把下一轮间隔写进 self.tick_interval 即可生效，无需改动父类。
        """
        original_on_tick = self.on_tick

        def timed_on_tick(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original_on_tick(*args, **kwargs)
            finally:
                last_run_seconds = time.perf_counter() - start
                self.tick_interval = self.next_tick_interval(last_run_seconds)
                self.log.debug(
                    "tick.adaptive_interval",
                    tick_interval=self.tick_interval,
                    last_run_seconds=round(last_run_seconds, 1),
                    msg="按上一轮评估耗时自适应下一轮间隔",
                )
                # 间隔已按本轮实测耗时算好，此时输出汇总行，next= 才是真实的下一轮间隔。
                emit = getattr(self, "_emit_tick_summary", None)
                if emit is not None:
                    emit()

        self.on_tick = timed_on_tick  # type: ignore[assignment]


    # ---- mode 感知的产物文件名 -------------------------------------------
    # public/private 共用同一个比赛目录（同一个 competition_id），靠文件名后缀隔离产物。

    @property
    def mode_suffix(self) -> str:
        """public 为空串，private 为 '-private'，拼进所有产物文件名以隔离两套评测。"""
        return "" if self.mode == "public" else "_private"

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
    def sfa_status_file(self) -> str:
        return _with_suffix(constants.SFA_STATUS_FILE, self.mode_suffix)

    def raw_factor_cut_file(self, cutoff: str) -> str:
        """某个截断日对应的截断窗 raw factor 产物文件名（带 cutoff 与 mode 后缀）。

        多个截断日各落一份、互不覆盖：raw_factor_cut.parquet -> raw_factor_cut_2025-06-30.parquet
        （private 再叠加 _private 后缀）。
        """
        tag = str(cutoff).replace(":", "").replace(" ", "_")
        base_name = _with_suffix(constants.RAW_FACTOR_CUT_FILE, f"_{tag}")
        return _with_suffix(base_name, self.mode_suffix)

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
    def factor_pool_raw_path(self) -> str:
        """因子池「原始因子数据」parquet 的绝对路径。

        与 factor_pool_path 入池的因子集合一致，只是取每个提交的 raw_factor（未处理）
        而非 process_factor，作为存档保留一份，不参与回归。
        """
        return os.path.join(self.leaderboard_dir, _with_suffix("factor_pool_raw.parquet", self.mode_suffix))

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
        """单因子分析 runner 模板：注入数据集映射/日期/产物文件名，仍保留 __USER_CODE__ 占位符。"""
        assert self.DATASETS and self.DATE_START and self.DATE_END, "子类必须设置 DATASETS / DATE_START / DATE_END"
        return templates.build_sfa_runner(
            datasets=self.DATASETS,
            date_start=self.DATE_START,
            date_end=self.DATE_END,
            raw_factor_file=self.raw_factor_file,
            process_factor_file=self.process_factor_file,
            factor_analyze_file=self.factor_analyze_file,
        )

    def lookahead_datasets(self) -> dict[str, str]:
        """检测复算用的数据集映射：在 DATASETS 基础上，把 LOOKAHEAD_DATASETS 里声明了截断表的
        逻辑名替换成截断物理表名，其余逻辑名沿用原表。用户代码只认逻辑名，无需改动即可在截断
        数据上复算。"""
        merged = dict(self.DATASETS)
        merged.update(self.LOOKAHEAD_DATASETS)
        return merged

    def JUDGE_LOOKAHEAD(self, cutoff: str) -> str:
        """截断表复算 runner 模板：与 JUDGE_SFA 同一份用户代码，把 end_date 换成 cutoff、
        数据表换成截断到 cutoff 的物理表，不跑 eval，只落盘截断窗 raw factor（比对由主进程
        check_lookahead 完成）。cutoff 参数化，故为方法而非 property。
        仍保留 __USER_CODE__ 占位符，由调用方替换。"""
        assert self.DATASETS and self.DATE_START, "子类必须设置 DATASETS / DATE_START"
        return templates.build_lookahead_runner(
            datasets=self.lookahead_datasets(),
            date_start=self.DATE_START,
            cutoff=cutoff,
            raw_factor_cut_file=self.raw_factor_cut_file(cutoff),
        )

    @property
    def JUDGE_REG(self) -> str:
        """因子池回归 runner 模板：注入日期区间与因子池读入路径、回归得分产出路径。"""
        assert self.DATE_START and self.DATE_END, "子类必须设置 DATE_START / DATE_END"
        return templates.build_reg_runner(
            date_start=self.DATE_START,
            date_end=self.DATE_END,
            factor_pool_file=self.factor_pool_path,
            factor_regression_score=self.leaderboard_reg_csv,
        )

    # ---- 共享小工具 -------------------------------------------------------
    def heartbeat_fields(self) -> dict:
        """给心跳日志附加运行进度，分两块：

        单因子分析（提交维度，实时）：
          - total：本场比赛当前提交总数，复用主循环每 tick 缓存的 _submission_total（不额外打 API）；
          - done / ok / failed：直接扫各提交目录的 sfa_status.json 现场统计。这些状态文件是
            on_submission 里每跑完一个立刻写的，故随每个提交完成实时前进——不像旧实现读
            submissions_summary.csv（每个 tick 才刷新一次，会滞后一整个 tick）。
            done = 终态数（success/user_error/timeout/file_error/lookahead），ok = success，
            failed = done - ok；
          - remaining = total - done，涵盖「刚提交还没建目录」与 env_error（会重试）的提交。

        tick 阶段（on_tick 维度，实时）：
          - stage：主循环此刻所处阶段，用于区分「在跑单因子分析」还是「在跑 on_tick 里的
            因子池回归」。取值：
                sfa      —— on_tick 已跑完，主循环 sleep + 等线程池里的单因子分析任务
                sfa_rank —— on_tick 正在做单因子截面排名（A 项）
                pool     —— on_tick 正在构建因子池
                regression—— on_tick 正在跑因子池回归（B 项来源）
                final    —— on_tick 正在合成最终得分
                summary  —— on_tick 正在汇总运行结果；
          - tick：已进入 on_tick 的次数（第几个 tick）；
          - pool：最近一次因子池入池因子数；
          - reg：回归耗时。进行中显示 f"{已跑秒数}s"，空闲显示上次耗时 f"{_reg_last}s"。

        字段均用 getattr 兜底，兼容进程刚启动、on_tick / 回归尚未跑过的情况。
        """
        total = getattr(self, "_submission_total", None)

        # done / ok / failed：现场扫 sfa_status.json（实时，不依赖每 tick 才刷新的 csv）
        done = ok = 0
        if os.path.isdir(self.submission_dir):
            for sid in os.listdir(self.submission_dir):
                status = read_json(
                    os.path.join(self.submission_dir, sid, self.sfa_status_file),
                    logger=self.log,
                )
                st = (status or {}).get("status")
                if st in TERMINAL_STATUSES:
                    done += 1
                    if st == STATUS_SUCCESS:
                        ok += 1

        fields: dict = {
            "total": total,
            "done": done,
            "ok": ok,
            "failed": done - ok,
        }
        if total is not None:
            fields["remaining"] = total - done

        # tick 阶段块
        fields["stage"] = getattr(self, "_stage", "init")
        fields["tick"] = getattr(self, "_tick_seq", 0)
        pool = getattr(self, "_pool_size", None)
        if pool is not None:
            fields["pool"] = pool
        reg_start = getattr(self, "_reg_start", None)
        if reg_start is not None:
            fields["reg"] = f"{int(time.time() - reg_start)}s"  # 回归进行中，实时秒数
        else:
            reg_last = getattr(self, "_reg_last", None)
            if reg_last is not None:
                fields["reg"] = f"{reg_last}s"                  # 上次回归耗时
        return fields

    def query_constraints(self) -> dict:
        """限定拉取提交的范围：在父类约束（private 默认只跑入围者）之上，叠加
        SUBMISSION_IDS 子集过滤。两处拉取提交的入口（主循环与 _iter_submission_dirs）
        都走这里，保证「只跑部分提交」在跑用户代码、排名、回归、汇总各环节口径一致。
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
