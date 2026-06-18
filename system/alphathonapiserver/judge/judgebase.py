import concurrent.futures
import logging
import os
import threading
import time
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, Optional

import dai
import pandas as pd
import structlog

from .api import AlphathonAPI
from .paths import FILE_DIR
from .runner import K8SPodUserRunner, LocalProcessUserRunner, UserCodeRunner

logger = structlog.get_logger()

__all__ = [
    "AlphathonAPI",
    "JudgeBase",
    "K8SPodUserRunner",
    "LocalProcessUserRunner",
    "UserCodeRunner",
]


def setup_judge_logging(log_file: str) -> None:
    """把 judge 评估系统的日志同时输出到终端和文件。

    终端用彩色渲染方便实时观察，文件里落盘成结构化文本方便后续查看。

    文件采用「按天滚动 + 追加 + 永不删除」策略：
      - 当天写入 judge-{mode}.log，每天零点自动切到新文件；
      - 昨天的日志归档成 judge-{mode}.log.YYYY-MM-DD，历史全部保留不删除；
      - 追加模式：手动中断后重启，当天日志接着写，不会覆盖、不会丢失。

    只配置一次即可（重复调用会清掉旧 handler 再重建，避免重复输出）。
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 终端：彩色 console 渲染
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )
    # 文件：无颜色 console 渲染，纯文本方便 grep/查看
    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # 按天滚动：when="midnight" 每天零点切文件；backupCount=0 表示不删除任何历史归档。
    # 默认 mode="a"（追加），中断重启后当天日志接着写。归档文件名形如 judge-public.log.2026-06-14。
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=0,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    # 重复调用时先清掉旧 handler，避免日志被打印多份
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # 第三方 HTTP 库默认会在 INFO 级别打印每一次正常请求（200 OK），噪音很大。
    # 抬到 WARNING，只有请求失败（4xx/5xx/超时）时才输出。
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)



class JudgeBase:
    """所有评测器的基类，只负责通用机制：拉取提交 -> 跑用户代码 -> 取结果 -> 横向排名 -> 回写分数。

    每个比赛的差异都通过子类声明/重写来表达，子类放在
    system/competitions/{cid}/{public,private}.py 下。

    必填：
        - competition_id: 比赛 id
        - mode: "public" 或 "private"
        - JUDGE_RUNNER_CODE: 包裹用户代码的运行模板，__USER_CODE__ 会被替换成用户代码
        - compute_score(df): 输入所有原始结果的 DataFrame，返回带 score 列的 DataFrame

    可选 hook（按需重写）：
        - preprocess_user_code(submission, code): 对个别提交做特殊处理
        - extract_result(submission, runner): 从 runner 产物里解出 raw_result，返回 None 表示本场不排名
        - query_constraints(): 限定拉取提交的范围
        - on_tick(): 每个 tick 的额外动作（默认重排榜单）
    """

    competition_id: str = ""
    mode: str = "public"  # public / private
    tick_interval: int = 60
    heartbeat_interval: int = 15  # 心跳间隔（秒），证明评测器仍在运行
    max_workers: int = 5
    JUDGE_RUNNER_CODE: str = ""

    def __init__(self, competition_id: Optional[str] = None, tick_interval: Optional[int] = None) -> None:
        if competition_id:
            self.competition_id = competition_id
        if tick_interval is not None:
            self.tick_interval = tick_interval
        assert self.competition_id, "competition_id 必须在子类或构造时指定"
        assert self.mode in ("public", "private"), f"mode 必须是 public 或 private, got {self.mode}"

        # 比赛目录
        self.compeition_dir = os.path.join(FILE_DIR, self.competition_id)
        # 提交文件存放目录
        self.submission_dir = os.path.join(self.compeition_dir, 'submissions')
        # 排行榜 csv / 已完成 id 持久化目录
        self.leaderboard_dir = os.path.join(self.compeition_dir, "leaderboard")
        # judge 评估系统自身的日志目录与文件
        self.log_dir = os.path.join(self.compeition_dir, "logs")
        self.judge_log_file = os.path.join(self.log_dir, f"judge_{self.mode}.log")

        # 配置评估系统日志：终端 + 文件双输出（用户任务日志不会进到这里）
        setup_judge_logging(self.judge_log_file)

        self.alphathon_api = AlphathonAPI()
        self.log = logger.bind()
        self.log.info("judge.init", tick_interval=self.tick_interval, log_file=self.judge_log_file)

    # ---- 字段 / 路径 ------------------------------------------------------

    @property
    def score_field(self) -> str:
        return f"{self.mode}_score"

    @property
    def score_data_field(self) -> str:
        return f"{self.mode}_score_data"

    @property
    def leaderboard_csv(self) -> str:
        os.makedirs(self.leaderboard_csv, exist_ok=True)
        # 私榜文件名加 -private 后缀，与公榜区分；同一个 competition_id 在磁盘上保留两份榜单
        suffix = "" if self.mode == "public" else "-private"
        return os.path.join(self.leaderboard_dir, f"{self.competition_id}{suffix}.csv")

    # ---- 子类可重写的 hook -------------------------------------------------

    def query_constraints(self) -> Dict[str, Any]:
        """限定拉取提交的范围。private 模式默认只跑入围私榜的提交。"""
        if self.mode == "private":
            return {"selected_for_private": True}
        return {}

    def preprocess_user_code(self, submission: dict, code: str) -> str:
        """对个别提交做特殊处理（例如修复已知错误），默认原样返回。"""
        return code

    def extract_result(self, submission: dict, runner: LocalProcessUserRunner) -> Optional[dict]:
        """从 runner 产物里解出一行 raw_result（dict）参与排名。

        默认读取 output.data（dai DataSource 序列化的结果 id）。返回 None 表示
        本场比赛只跑用户代码、不产出榜单分数（此时不会回写 score，也不会触发重排）。
        """
        with open(os.path.join(runner.runner_dir, "output.data")) as reader:
            return dai.DataSource(reader.read()).read().iloc[0].to_dict()

    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """子类必须重写：基于 raw_result 各列计算 score 列。"""
        raise NotImplementedError

    def on_tick(self) -> None:
        """每个 tick 的额外动作，默认重排一次榜单。"""
        self.rank_score()

    # ---- 通用机制 ----------------------------------------------------------

    def submission_path(self, submission: dict) -> str:
        """该提交的统一目录：FILE_DIR/{competition_id}/submissions/{sid}。

        原始文件、注入的 judge_runner.py、stdout 运行日志、output.data 产物都收在这里。
        """
        return os.path.join(self.submission_dir, str(submission["id"]))

    def save_submission_files(self, submission: dict) -> str:
        """把该提交的所有原始文件落盘到 submission_dir/{submission_id}/{原文件名}。

        返回该提交的文件目录。提交可能包含多个文件，逐个按原始文件名保存。
        """
        sid = str(submission["id"])
        dst_dir = self.submission_path(submission)
        os.makedirs(dst_dir, exist_ok=True)

        files = (submission.get("data") or {}).get("files") or {}
        for file_id, file_info in files.items():
            # 用原始文件名落盘；缺失时回退到 file_id，避免覆盖/丢失
            file_name = (file_info or {}).get("name") or file_id
            self.alphathon_api.get_submission_file(
                sid, file_id, file_info, save_to=os.path.join(dst_dir, file_name)
            )
        self.log.info("[submission] 下载文件", submission_id=sid, count=len(files))
        return dst_dir

    def run_user_code(self, submission: dict) -> LocalProcessUserRunner:
        """拉取用户代码、注入 JUDGE_RUNNER_CODE 模板，并在隔离子进程中执行。"""
        sid = submission["id"]
        # ipynb 会被转成 .py 字符串，便于注入到 runner 模板中
        user_code = self.alphathon_api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)
        if isinstance(user_code, bytes):
            user_code = user_code.decode("utf-8")
        user_code = self.preprocess_user_code(submission, user_code)

        runner = LocalProcessUserRunner(
            submission_id=sid,
            files={"judge_runner.py": self.JUDGE_RUNNER_CODE.replace("__USER_CODE__", user_code)},
            cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
            # 运行目录与原始文件同目录，所有产物（含 stdout 日志）都收在该提交的文件夹下
            runner_dir=self.submission_path(submission),
        )
        runner.run(_raise=True)
        self.log.info("[submission] 代码运行成功", submission_id=sid)
        return runner

    def on_submission(self, submission: dict) -> None:
        sid = submission["id"]
        self.log.info("[submission] 处理提交文件", submission_id=sid)
        try:
            # 先把用户提交的所有原始文件落盘留档
            self.save_submission_files(submission)
            runner = self.run_user_code(submission)
            raw_result = self.extract_result(submission, runner)
            # 单条提交跑通时先占位 -1，最终分数等 rank_score 横向排序后再写入
            score = -1
            score_data = {"raw_result": raw_result}
        except Exception as e:
            # -2 表示用户代码运行失败；err_msg 会回显在前端的提交详情里
            score = -2
            score_data = {"err_msg": "run error: check your code / get code templates in [code] tab"}
            self.log.error("[submission] 代码运行失败", submission_id=sid, error=str(e))

        self.alphathon_api.update_submission_score(
            submission_id=sid,
            **{self.score_field: score, self.score_data_field: score_data},
        )
        # 单条跑完立即触发一次全量重排，这样榜单可以增量刷新
        self.rank_score()

    def rank_score(self) -> None:
        # 拉取本场比赛的所有提交，按 raw_result 横向计算名次/分数
        all_submissions = self.alphathon_api.query_submissions(competition_id=self.competition_id)
        self.log.info("rank.fetched", count=len(all_submissions))

        raw_results = []
        for x in all_submissions:
            # 只对已经成功跑出 raw_result 的提交参与排名；失败/未跑的跳过
            score_data = x.get(self.score_data_field) or {}
            raw_result = score_data.get("raw_result")
            if not raw_result:
                continue
            raw_result["id"] = x["id"]
            raw_results.append(raw_result)

        if not raw_results:
            self.log.warning("[rank] 没有分数数据")
            return

        df = pd.DataFrame(raw_results)
        # compute_score 由子类实现，必须返回带 score 列的 DataFrame
        df = self.compute_score(df)
        df.to_csv(self.leaderboard_csv, index=False)

        for _, row in df.iterrows():
            score = row.score
            # NaN != NaN 是 pandas 里识别空分数最稳妥的写法；空分数统一记为 -2 失败
            if score != score:
                score = -2
            self.alphathon_api.update_submission_score(
                submission_id=row.id,
                **{self.score_field: score},
            )

    def run(self) -> None:
        # 评测器主循环：每隔 tick_interval 秒拉一次新提交、回收已完成的 future、重排榜单
        dispatched: set[str] = set()  # 本进程已派发过的提交，避免重复入队
        futures: dict[str, concurrent.futures.Future] = {}  # 仍在运行/未回收的 future

        # 心跳线程：独立于主循环，按 heartbeat_interval 周期性打印一条 alive 日志，
        # 证明评测系统仍在运行（即便某个 tick 正卡在拉取/排名上也能看到心跳）。
        stop_heartbeat = threading.Event()
        beat = {"n": 0}

        def _heartbeat() -> None:
            while not stop_heartbeat.wait(self.heartbeat_interval):
                beat["n"] += 1
                self.log.info(
                    "judge.heartbeat",
                    seq=beat["n"],
                    running=len(futures),
                    dispatched=len(dispatched),
                )

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            while True:
                self.log.info("tick.start")
                submissions = self.alphathon_api.query_submissions(
                    competition_id=self.competition_id,
                    constraints=self.query_constraints(),
                )
                # 过滤掉本进程已派发过的，剩下的入队执行
                pending = [s for s in submissions if s.get("id") not in dispatched]
                for submission in pending:
                    sid = str(submission.get("id"))
                    dispatched.add(sid)
                    futures[sid] = executor.submit(self.on_submission, submission)

                # 回收本轮已经跑完的任务
                for sid in [sid for sid, f in futures.items() if f.done()]:
                    futures.pop(sid, None)
                self.log.info("tick.status", pending=len(pending), running=len(futures))

                # 即使本轮没有新提交，也要走一次 on_tick（默认重排榜单）
                self.on_tick()
                self.log.info("tick.sleep", seconds=self.tick_interval)
                time.sleep(self.tick_interval)
        finally:
            # 进程退出时停掉心跳线程，并不等正在跑的子任务，直接尝试取消未开始的 future
            stop_heartbeat.set()
            executor.shutdown(wait=False, cancel_futures=True)
