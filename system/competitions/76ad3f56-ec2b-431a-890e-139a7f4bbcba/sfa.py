"""单因子分析阶段（A 项）。

职责：对每个提交跑通用户代码、做单因子分析并把产物落盘，记录运行状态（成功/各类失败），
以及在 on_tick 中把全体已跑通提交的单因子指标做截面排名得到 A 项得分。

作为 mixin 混入 BigAlphaJudge，依赖 BigAlphaJudgeBase 提供的 mode 感知路径与 JUDGE_SFA 模板。
"""
from __future__ import annotations

import datetime
import json
import os

import pandas as pd

from judge.judgebase import LocalProcessUserRunner, UserCodeRunError, log_context, log_timer

import scoring
from constants import (
    STATUS_ENV_ERROR,
    STATUS_ERR_MSG,
    STATUS_FILE_ERROR,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    STATUS_USER_ERROR,
    TERMINAL_STATUSES,
    SubmissionFileError,
)
from fileio import read_json


class SFAMixin:
    """单因子分析：跑用户代码 + 状态记录 + 截面排名（A 项）。"""

    # ---- 运行用户/注入代码 -------------------------------------------------

    def run_user_code(self, submission: dict, runner_code: str) -> LocalProcessUserRunner:
        """拉取用户代码、注入 runner 模板，并在隔离子进程中执行。

        与基类不同：runner_code 由调用方传入（当前为 JUDGE_SFA，跑单因子分析）。
        因子池回归不依赖用户代码，已独立到 run_regression()，不再走这里。
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

        runner = LocalProcessUserRunner(
            submission_id=sid,
            files={"judge_runner.py": injected},
            cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
            # 运行目录与原始文件同目录，所有产物（含 stdout 日志）都收在该提交的文件夹下
            runner_dir=self.submission_path(submission),
        )
        runner.run(_raise=True)
        return runner

    def extract_sfa_score(self, runner) -> dict:
        """从 runner 产物里读出单因子分析结果（dict）。"""
        factor_analyze_path = os.path.join(runner.runner_dir, self.factor_analyze_file)
        with open(factor_analyze_path, encoding="utf-8") as reader:
            return json.load(reader)

    # ---- 运行状态记录 -----------------------------------------------------

    def write_sfa_status(self, submission: dict, status: str, **extra) -> None:
        """记录该提交单因子分析的运行结果（成功/失败都写）。

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
            with open(self.sfa_status_path(submission), "w", encoding="utf-8") as writer:
                json.dump(record, writer, ensure_ascii=False, default=str)
        except Exception as e:
            self.log.error("submission.status_write_failed", error=str(e), msg="写入运行状态文件失败")

    def is_done(self, submission: dict) -> bool:
        """判断该提交是否已到达终态，无需再跑。

        以状态文件 sfa_status.json 的 status 为准：
            - success / user_error / timeout / file_error 属于终态（重跑也是同样结果），跳过；
            - env_error（评测环境自身问题）不算终态，留待重试。
        进程重启后内存里的 dispatched 集合会清空，靠这个文件判断哪些提交不必重跑。

        兼容旧数据：状态文件机制之前跑成功的提交只有 factor_analyze.json、没有状态文件，
        这类也视为已完成，避免上线后把历史成功提交全部重跑一遍。
        """
        status = self.read_sfa_status(submission)
        if status is not None:
            return status.get("status") in TERMINAL_STATUSES
        legacy_fa = os.path.join(self.submission_path(submission), self.factor_analyze_file)
        return os.path.exists(legacy_fa)

    def _fail_submission(self, submission: dict, status: str, **extra) -> None:
        """记录一次失败：落盘状态文件 + 回写 -2 分与对应提示语。

        status 取 STATUS_* 之一；提示语查 STATUS_ERR_MSG。extra（error/return_code 等）
        一并写入状态文件，方便排查。
        """
        self.log.error("submission.sfa_failed", status=status, msg="单因子分析失败", **extra)
        self.write_sfa_status(submission, status, **extra)
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
            # 排名/回归/最终评分统一由 on_tick 刷新，跳过这里不影响榜单。
            if self.is_done(submission):
                self.log.debug("submission.skip", msg="已跑过，跳过重复执行")
                return

            self.log.info("submission.start", msg="开始处理提交")

            # on_submission 只负责「跑通用户代码 + 单因子分析」并把结果落盘保留；
            # 截面排名（A 项）、因子池回归（B 项）与最终评分统一放到 on_tick 里做。
            # 失败按类型记录：user_error / timeout / file_error 是终态，env_error 会重试。
            try:
                with log_timer() as elapsed:
                    self.save_submission_files(submission)
                    self.run_user_code(submission, self.JUDGE_SFA)
                self.log.info("submission.sfa_done", elapsed_ms=elapsed(), msg="单因子分析完成")
                self.write_sfa_status(submission, STATUS_SUCCESS, elapsed_ms=elapsed())
            except UserCodeRunError as e:
                # 用户代码子进程异常退出，reason 区分「用户报错」与「超时」。两者都是终态。
                status = STATUS_TIMEOUT if e.reason == "timeout" else STATUS_USER_ERROR
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

            # 跑通即可，等待 on_tick 统一做截面排名、因子池回归与最终评分。
            # submission.sfa_done 已记录成功，这里不再重复一行。

    # ---- 单因子排名 (A) ---------------------------------------------------

    def score_sfa(self) -> int:
        """单因子分析横向排名。

        参考基类 rank_score()，区别：
        1. 不从 alphathon_api 拉结果，而是遍历 submissions 目录读取 factor_analyze 文件；
        2. 截面 rank 后的得分快照另存到 leaderboard_sfa.csv。

        注意：这里只计算并落盘 A 项（单因子得分），不直接回写 public_score。
        最终分数 = 0.3*A + 0.7*B 由 score_final() 统一合成后回写，避免 A 覆盖最终分。

        返回参与排名的因子数（供 on_tick 汇总成一行日志）；无数据时返回 0。
        """
        rows = []
        for sid, _submission, sub_dir in self._iter_submission_dirs():
            fa = read_json(os.path.join(sub_dir, self.factor_analyze_file), logger=self.log)
            if fa is None:
                continue
            fa = dict(fa)
            fa["id"] = sid
            rows.append(fa)

        if not rows:
            self.log.warning("sfa.empty", msg="没有任何单因子分数结果")
            return 0

        df = pd.DataFrame(rows)
        # 指标可能因 json 序列化变成字符串/缺失，统一转数值，rank 会自动忽略 NaN
        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        df = scoring.compute_sfa_score(df)

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        df.to_csv(self.leaderboard_sfa_csv, index=False)
        self.log.debug("sfa.ranked", count=len(df), msg="单因子分数截面排名完成")
        return len(df)
