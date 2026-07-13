"""模型评分阶段（跑用户模型推理 + 平台预处理 + 单因子分析 + 截面排名）。

职责：对每个提交跑通用户代码（端到端模型推理产出截面分数）、做平台预处理与单因子分析并把
产物落盘，记录运行状态（成功/各类失败），以及在 on_tick 中把全体已跑通提交的单因子指标做
截面排名得到最终得分。

作为 mixin 混入 EndToEndJudge，依赖 EndToEndJudgeBase 提供的 mode 感知路径与 JUDGE_SCORE 模板。
"""
from __future__ import annotations

import datetime
import json
import os

import pandas as pd

from judge.judgebase import UserCodeRunError, log_context, log_timer

import scoring
from constants import (
    STATUS_ENV_ERROR,
    STATUS_ERR_MSG,
    STATUS_FILE_ERROR,
    STATUS_OOM,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    STATUS_USER_ERROR,
    TERMINAL_STATUSES,
    SubmissionFileError,
)
from fileio import read_json
from runner import MemoryLimitedUserRunner


class ScoreMixin:
    """模型评分：跑用户代码 + 状态记录 + 截面排名（最终得分）。"""

    # ---- 运行用户/注入代码 -------------------------------------------------

    def run_user_code(self, submission: dict, runner_code: str) -> MemoryLimitedUserRunner:
        """拉取用户代码、注入 runner 模板，并在隔离子进程中执行。

        runner_code 由调用方传入（当前为 JUDGE_SCORE，跑模型推理 + 单因子分析）。
        """
        sid = submission["id"]
        # ipynb 会被转成 .py 字符串，便于注入到 runner 模板中。
        # 文件缺失/数量不对/无法解析属于用户文件问题，单独抛 SubmissionFileError（终态，不重试）。
        try:
            user_code = self.alphathon_api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)
            if isinstance(user_code, bytes):
                user_code = user_code.decode("utf-8")
        except Exception as e:
            raise SubmissionFileError(str(e)) from e
        user_code = self.preprocess_user_code(submission, user_code)

        injected = runner_code.replace("__USER_CODE__", user_code)

        # 用带内存上限的 runner：给用户子进程设 RLIMIT_AS，超限只杀该子进程、不波及 judge；
        # 内存溢出会以 reason="oom" 抛出，并带出 stdout 日志尾部。
        runner = MemoryLimitedUserRunner(
            submission_id=sid,
            files={"judge_runner.py": injected},
            cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
            # 运行目录与原始文件同目录，所有产物（含 stdout 日志）都收在该提交的文件夹下
            runner_dir=self.submission_path(submission),
        )
        runner.run(_raise=True)
        return runner

    # ---- 运行状态记录 -----------------------------------------------------

    def write_score_status(self, submission: dict, status: str, **extra) -> None:
        """记录该提交评分的运行结果（成功/失败都写）。

        status 取 STATUS_* 之一（success / user_error / timeout / file_error / env_error）。
        额外字段（如 elapsed_ms、error、return_code）一并落盘，方便排查；
        写文件失败不应影响主流程，故吞掉异常只记一行日志。
        """
        record = {
            "submission_id": str(submission["id"]),
            "status": status,
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        try:
            os.makedirs(self.submission_path(submission), exist_ok=True)
            with open(self.score_status_path(submission), "w", encoding="utf-8") as writer:
                json.dump(record, writer, ensure_ascii=False, default=str)
        except Exception as e:
            self.log.error("submission.status_write_failed", error=str(e), msg="写入运行状态文件失败")

    def is_done(self, submission: dict) -> bool:
        """判断该提交是否已到达终态，无需再跑。

        以状态文件 score_status.json 的 status 为准：
            - success / user_error / timeout / file_error 属于终态（重跑也是同样结果），跳过；
            - env_error（评测环境自身问题）不算终态，留待重试。
        进程重启后内存里的 dispatched 集合会清空，靠这个文件判断哪些提交不必重跑。

        兼容旧数据：状态文件机制之前跑成功的提交只有 score_analyze.json、没有状态文件，
        这类也视为已完成，避免上线后把历史成功提交全部重跑一遍。

        例外：显式列在 SUBMISSION_IDS 里的提交是「指定复测」，无论此前是否已到终态都强制重跑，
        否则填了 SUBMISSION_IDS 也只会被这里的终态判断跳过、达不到复测的目的。
        """
        if self.SUBMISSION_IDS and str(submission["id"]) in {str(s) for s in self.SUBMISSION_IDS}:
            return False
        status = self.read_score_status(submission)
        if status is not None:
            return status.get("status") in TERMINAL_STATUSES
        legacy = os.path.join(self.submission_path(submission), self.score_analyze_file)
        return os.path.exists(legacy)

    def _fail_submission(self, submission: dict, status: str, **extra) -> None:
        """记录一次失败：落盘状态文件 + 回写 -2 分与对应提示语。"""
        self.log.error("submission.score_failed", status=status, msg="模型评分失败", **extra)
        self.write_score_status(submission, status, **extra)
        self.alphathon_api.update_submission_score(
            submission_id=submission["id"],
            **{
                self.score_field: -2,
                self.score_data_field: {"err_msg": STATUS_ERR_MSG.get(status, STATUS_ERR_MSG[STATUS_ENV_ERROR])},
            },
        )

    # ---- 单条提交处理 -----------------------------------------------------

    def on_submission(self, submission: dict) -> None:
        sid = submission["id"]
        # 绑定一次 submission_id，作用域内所有 self.log 自动带上
        with log_context(submission_id=sid):
            # 已经跑过的提交（产物已落盘）直接跳过，避免重启后重复执行用户代码。
            # 截面排名/最终评分统一由 on_tick 刷新，跳过这里不影响榜单。
            if self.is_done(submission):
                self.log.info("submission.skip", msg="已跑过，跳过重复执行")
                return

            self.log.info("submission.start", msg="开始处理提交")

            # on_submission 只负责「跑通用户模型推理 + 平台预处理 + 单因子分析」并把结果落盘保留；
            # 截面排名与最终评分统一放到 on_tick 里做。
            # 失败按类型记录：user_error / timeout / file_error 是终态，env_error 会重试。
            try:
                with log_timer() as elapsed:
                    self.save_submission_files(submission)
                    self.run_user_code(submission, self.JUDGE_SCORE)
                self.log.info("submission.score_done", elapsed_ms=elapsed(), msg="模型评分完成")
                self.write_score_status(submission, STATUS_SUCCESS, elapsed_ms=elapsed())
            except UserCodeRunError as e:
                # 用户代码子进程异常退出，reason 区分「超时」「内存溢出」「用户报错」，三者都是终态。
                # OOM 单独记状态并把子进程日志尾部（error 里带的 MemoryError 回溯）一并落盘，便于排查。
                status = {
                    "timeout": STATUS_TIMEOUT,
                    "oom": STATUS_OOM,
                }.get(e.reason, STATUS_USER_ERROR)
                if status == STATUS_OOM:
                    self.log.warning("submission.oom", error=str(e), return_code=e.return_code, msg="用户代码内存溢出")
                self._fail_submission(submission, status, error=str(e), return_code=e.return_code)
                return
            except SubmissionFileError as e:
                # 用户提交的文件本身有问题（缺失/数量不对/无法解析）。属于用户侧终态，不重试。
                self._fail_submission(submission, STATUS_FILE_ERROR, error=str(e))
                return
            except Exception as e:
                # 其余异常（拉取 ipynb 失败、落盘失败、注入失败等）归类为评测环境问题。
                # 这类多半是临时性的，状态记为 env_error（非终态），下个 tick / 重启后会重试。
                self._fail_submission(submission, STATUS_ENV_ERROR, error=str(e))
                return

            # 跑通即可，等待 on_tick 统一做截面排名与最终评分。
            self.log.info("submission.ready", msg="单因子分析结果已保留，等待 on_tick 统一评分")

    # ---- 截面排名（最终得分）---------------------------------------------

    def score_models(self) -> None:
        """模型分数单因子横向排名。

        遍历 submissions 目录读取 score_analyze 文件，把全体已跑通提交的四指标做截面 rank 加权，
        得到最终得分并落盘到 leaderboard_score.csv。

        注意：这里只计算并落盘得分快照，不直接回写 public_score。
        最终分数由 score_final() 统一读取后回写，保持与因子挖掘赛道一致的分层结构。
        """
        rows = []
        for sid, _submission, sub_dir in self._iter_submission_dirs():
            analyze = read_json(os.path.join(sub_dir, self.score_analyze_file), logger=self.log)
            if analyze is None:
                continue
            analyze = dict(analyze)
            analyze["id"] = sid
            rows.append(analyze)

        if not rows:
            self.log.warning("score.empty", msg="没有任何单因子分数结果")
            return

        df = pd.DataFrame(rows)
        # 指标可能因 json 序列化变成字符串/缺失，统一转数值，rank 会自动忽略 NaN
        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        df = scoring.compute_final_score(df)

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        df.to_csv(self.leaderboard_score_csv, index=False)
        self.log.info("score.ranked", count=len(df), msg="模型分数截面排名完成")
