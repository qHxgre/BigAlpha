import os
import sys
paths = ['/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

import datetime
import json
import pandas as pd

from judge.judgebase import JudgeBase, LocalProcessUserRunner, UserCodeRunError, log_context, log_timer

RAW_FACTOR_FILE = "raw_factor.parquet"
PROCESS_FACTOR_FILE = "process_factor.parquet"
FACTOR_ANALYZE_FILE = "factor_analyze.json"
FACTOR_REGRESSION_SCORE = "factor_regression_score.parquet"
# 每个提交单因子分析的运行状态记录文件：无论成功/失败都会落盘一份，
# is_done() 据此判断该提交是否已跑过，避免进程重启后重复执行（尤其是注定失败的提交）。
SFA_STATUS_FILE = "sfa_status.json"

# 运行状态取值。失败再细分四类，便于排查与决定是否重试：
STATUS_SUCCESS = "success"          # 跑通
STATUS_USER_ERROR = "user_error"    # 用户代码本身报错（子进程非 0 退出）
STATUS_TIMEOUT = "timeout"          # 运行超时（被 judge kill）
STATUS_FILE_ERROR = "file_error"    # 用户提交的文件本身有问题（缺失/数量不对/无法解析）
STATUS_ENV_ERROR = "env_error"      # 评测环境自身问题（拉取/落盘/注入失败等）

# 这些终态视为「已完成、不再重跑」：成功是真完成，用户报错/超时/文件错误再跑也是同样结果。
# 唯独 env_error 不在此列——多半是临时性问题，重启/下个 tick 应当重试。
TERMINAL_STATUSES = {STATUS_SUCCESS, STATUS_USER_ERROR, STATUS_TIMEOUT, STATUS_FILE_ERROR}


class SubmissionFileError(Exception):
    """用户提交的文件本身有问题：缺失、notebook 数量不对、ipynb 无法解析等。

    属于用户侧错误（重试也是同样结果），与「评测环境异常」区分开，单独记为终态。
    """

# 评测分两步：先对每个提交跑「单因子分析」，再用入选的优质因子拼成「因子池」做回归。
# 占位符（__USER_CODE__ / __XXX_FILE__）在注入时被替换成真实内容/路径。

# 第一步：跑单因子分析，并把原始因子、处理后因子、单因子得分分别落盘
JUDGE_RUNNER_CODE_1 = '''
__USER_CODE__

def judge_runner_main():
    import json
    import pandas as pd

    factor_data = main("bigalpha_factor_2026_stock_bar1m", "2025-01-01 00:00:00", "2025-12-31 23:59:59")

    from bigmodule import M
    result = M.bigalpha_factorminer._latest(
        factor_data=factor_data,
        show=True,
    )

    # 把原始因子数据 raw_factor 落盘为 parquet 文件
    result["raw_factor"].to_parquet("__RAW_FACTOR_FILE__")

    # 把处理后的因子数据 process_factor 落盘为 parquet 文件
    result["process_factor"].to_parquet("__PROCESS_FACTOR_FILE__")

    # 把单因子得分 factor_analyze（dict）落盘为 json 文件
    with open("__FACTOR_ANALYZE_FILE__", "w", encoding="utf-8") as writer:
        json.dump(result["factor_analyze"], writer, ensure_ascii=False, default=str)
'''
JUDGE_SFA = (
    JUDGE_RUNNER_CODE_1
    .replace("__RAW_FACTOR_FILE__", RAW_FACTOR_FILE)
    .replace("__PROCESS_FACTOR_FILE__", PROCESS_FACTOR_FILE)
    .replace("__FACTOR_ANALYZE_FILE__", FACTOR_ANALYZE_FILE)
)


# 第二步：跑因子池回归。因子池由评测系统汇总所有提交的优质因子后落盘（__FACTOR_POOL_FILE__），
# 这里只读取它并交给 bigalpha_factorminer 做回归，不再调用用户的 main()。
JUDGE_RUNNER_CODE_2 = '''
__USER_CODE__

def judge_runner_main():
    import json
    import pandas as pd

    # 读取评测系统落盘的因子池（所有提交的优质因子按 date/instrument 合并而成）
    factor_pool = pd.read_parquet("__FACTOR_POOL_FILE__")

    from bigmodule import M
    result = M.bigalpha_factorminer._latest(
        factor_pool=factor_pool,
        process_pools=False,
        show=True,
    )

    # 将 per_factor_scores 数据落盘，
    result['factor_regression']['per_factor_scores'].to_parquet("__FACTOR_REGRESSION_SCORE__")
'''
JUDGE_REG = JUDGE_RUNNER_CODE_2


class Judge(JudgeBase):
    competition_id = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
    mode = "public"
    JUDGE_SFA = JUDGE_SFA
    JUDGE_REG = JUDGE_REG

    # 每个队伍最多入选因子池的因子数量
    FACTOR_POOL_TOP_N = 50

    # ---- 路径 -------------------------------------------------------------
    @property
    def leaderboard_reg_csv(self) -> str:
        """回归分析榜单"""
        return os.path.join(self.leaderboard_dir, "leaderboard_reg.csv")

    @property
    def leaderboard_sfa_csv(self) -> str:
        """单因子分析榜单（截面 rank 后的得分快照）。"""
        return os.path.join(self.leaderboard_dir, "leaderboard_sfa.csv")

    @property
    def factor_pool_path(self) -> str:
        """因子池 parquet 的绝对路径（注入到 JUDGE_REG 模板里供子进程读取）。"""
        return os.path.join(self.leaderboard_dir, "factor_pool.parquet")

    @property
    def submissions_summary_csv(self) -> str:
        """所有提交运行结果的汇总统计文件。"""
        return os.path.join(self.leaderboard_dir, "submissions_summary.csv")

    # ---- 运行用户/注入代码 -------------------------------------------------

    def run_user_code(self, submission: dict, runner_code: str) -> LocalProcessUserRunner:
        """拉取用户代码、注入 runner 模板，并在隔离子进程中执行。

        与基类不同：runner_code 由调用方传入（JUDGE_SFA / JUDGE_REG），
        并额外把 __FACTOR_POOL_FILE__ 替换成因子池的绝对路径。
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

        injected = (
            runner_code
            .replace("__USER_CODE__", user_code)
            .replace("__FACTOR_POOL_FILE__", self.factor_pool_path)
            .replace("__FACTOR_REGRESSION_SCORE__", self.leaderboard_reg_csv)
        )

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
        factor_analyze_path = os.path.join(runner.runner_dir, FACTOR_ANALYZE_FILE)
        with open(factor_analyze_path, encoding="utf-8") as reader:
            return json.load(reader)

    # ---- 主流程 -----------------------------------------------------------

    def sfa_status_path(self, submission: dict) -> str:
        """该提交单因子分析状态文件的绝对路径。"""
        return os.path.join(self.submission_path(submission), SFA_STATUS_FILE)

    def write_sfa_status(self, submission: dict, status: str, **extra) -> None:
        """记录该提交单因子分析的运行结果（成功/失败都写）。

        status 取 STATUS_* 之一（success / user_error / timeout / env_error）。
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

    def read_sfa_status(self, submission: dict) -> dict | None:
        """读取该提交的状态文件，不存在或读不出时返回 None。"""
        path = self.sfa_status_path(submission)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as reader:
                return json.load(reader)
        except Exception as e:
            self.log.error("submission.status_read_failed", error=str(e), msg="读取运行状态文件失败")
            return None

    def is_done(self, submission: dict) -> bool:
        """判断该提交是否已到达终态，无需再跑。

        以状态文件 sfa_status.json 的 status 为准：
            - success / user_error / timeout 属于终态（重跑也是同样结果），跳过；
            - env_error（评测环境自身问题）不算终态，留待重试。
        进程重启后内存里的 dispatched 集合会清空，靠这个文件判断哪些提交不必重跑。

        兼容旧数据：状态文件机制之前跑成功的提交只有 factor_analyze.json、没有状态文件，
        这类也视为已完成，避免上线后把历史成功提交全部重跑一遍。
        """
        status = self.read_sfa_status(submission)
        if status is not None:
            return status.get("status") in TERMINAL_STATUSES
        legacy_fa = os.path.join(self.submission_path(submission), FACTOR_ANALYZE_FILE)
        return os.path.exists(legacy_fa)

    def on_submission(self, submission: dict) -> None:
        sid = submission["id"]
        # 绑定一次 submission_id，作用域内所有 self.log 自动带上
        with log_context(submission_id=sid):
            # 已经跑过的提交（产物已落盘）直接跳过，避免重启后重复执行用户代码。
            # 排名由 on_tick -> score_sfa 统一刷新，跳过这里不影响榜单。
            if self.is_done(submission):
                self.log.info("submission.skip", msg="已跑过，跳过重复执行")
                return

            self.log.info("submission.start", msg="开始处理提交")

            # 第一步：落盘原始文件 + 跑单因子分析。
            # 失败按类型记录：user_error / timeout 是终态，env_error 会重试。
            try:
                with log_timer() as elapsed:
                    self.save_submission_files(submission)
                    self.run_user_code(submission, self.JUDGE_SFA)
                self.log.info("submission.sfa_done", elapsed_ms=elapsed(), msg="单因子分析完成")
                self.write_sfa_status(submission, STATUS_SUCCESS, elapsed_ms=elapsed())
            except UserCodeRunError as e:
                # 用户代码子进程异常退出，reason 区分「用户报错」与「超时」。两者都是终态。
                status = STATUS_TIMEOUT if e.reason == "timeout" else STATUS_USER_ERROR
                err_msg = (
                    "timeout: your code exceeded the time limit"
                    if status == STATUS_TIMEOUT
                    else "run error: check your code / get code templates in [code] tab"
                )
                self.log.error("submission.sfa_failed", status=status, error=str(e), msg="单因子分析运行失败")
                self.write_sfa_status(submission, status, error=str(e), return_code=e.return_code)
                self.alphathon_api.update_submission_score(
                    submission_id=sid,
                    **{
                        self.score_field: -2,
                        self.score_data_field: {"err_msg": err_msg},
                    },
                )
                return
            except SubmissionFileError as e:
                # 用户提交的文件本身有问题（缺失/数量不对/无法解析）。属于用户侧终态，不重试。
                self.log.error("submission.sfa_failed", status=STATUS_FILE_ERROR, error=str(e), msg="提交文件有问题")
                self.write_sfa_status(submission, STATUS_FILE_ERROR, error=str(e))
                self.alphathon_api.update_submission_score(
                    submission_id=sid,
                    **{
                        self.score_field: -2,
                        self.score_data_field: {"err_msg": "file error: check your submission file (exactly 1 valid .ipynb expected)"},
                    },
                )
                return
            except Exception as e:
                # 其余异常（拉取 ipynb 失败、落盘失败、注入失败等）归类为评测环境问题。
                # 这类多半是临时性的，状态记为 env_error（非终态），下个 tick / 重启后会重试。
                self.log.error("submission.sfa_failed", status=STATUS_ENV_ERROR, error=str(e), msg="评测环境异常，稍后重试")
                self.write_sfa_status(submission, STATUS_ENV_ERROR, error=str(e))
                self.alphathon_api.update_submission_score(
                    submission_id=sid,
                    **{
                        self.score_field: -2,
                        self.score_data_field: {"err_msg": "evaluation system error, will retry automatically"},
                    },
                )
                return

            # 第二步：单因子横向排名，刷新公榜
            try:
                self.score_sfa()
            except Exception as e:
                self.log.error("sfa.failed", error=str(e), msg="单因子排名失败")

            # 第三步：用排名靠前的因子拼出因子池，并对该提交跑因子池回归（产物落盘，供后续分析）。
            try:
                self.save_factor_pool()
                if os.path.exists(self.factor_pool_path):
                    with log_timer() as elapsed:
                        self.run_user_code(submission, self.JUDGE_REG)
                    self.log.info("regression.done", elapsed_ms=elapsed(), msg="因子池回归完成")
            except Exception as e:
                self.log.error("regression.failed", error=str(e), msg="因子池回归失败")

    def on_tick(self) -> None:
        """每个 tick 重排一次单因子公榜（增量刷新），并汇总各提交运行结果。"""
        try:
            self.score_sfa()
            self.log.info("tick.refreshed", msg="刷新单因子榜单")
        except Exception as e:
            self.log.error("tick.failed", error=str(e), msg="刷新榜单失败")

        try:
            self.summarize_submissions()
        except Exception as e:
            self.log.error("summary.failed", error=str(e), msg="汇总提交运行结果失败")

    # ---- 运行结果汇总 -----------------------------------------------------

    def summarize_submissions(self) -> None:
        """把所有提交的运行结果汇总到一个统计文件 submissions_summary.csv。

        逐个提交收集：
            - 运行状态（sfa_status.json：status / finished_at / elapsed_ms / error）；
            - 单因子分析指标（factor_analyze.json：ic_mean / ic_ir / sharpe_ratio / stress_ic_ir 等）；
            - 截面排名得分（leaderboard_sfa.csv 里的 score）；
            - 各产物是否落盘（raw/process factor、回归得分）。
        汇总后按 score 倒序落盘，方便整体观察各提交的成败与表现。
        """
        submissions = self.alphathon_api.query_submissions(competition_id=self.competition_id)
        sub_by_id = {str(s["id"]): s for s in submissions}

        # 截面排名得分，用于补充每个提交的最终单因子得分
        sfa_scores: dict[str, float] = {}
        if os.path.exists(self.leaderboard_sfa_csv):
            try:
                sfa_df = pd.read_csv(self.leaderboard_sfa_csv)
                sfa_scores = {str(r["id"]): r["score"] for _, r in sfa_df.iterrows()}
            except Exception as e:
                self.log.error("summary.sfa_read_failed", error=str(e), msg="读取单因子榜单失败")

        rows = []
        if os.path.isdir(self.submission_dir):
            for sid in os.listdir(self.submission_dir):
                submission = sub_by_id.get(sid)
                if submission is None:
                    continue
                sub_dir = os.path.join(self.submission_dir, sid)

                row: dict = {
                    "submission_id": sid,
                    "user_id": submission.get("user_id"),
                    "group": self._group_key(submission),
                }

                # 运行状态
                status = self.read_sfa_status(submission) or {}
                row["status"] = status.get("status")
                row["finished_at"] = status.get("finished_at")
                row["elapsed_ms"] = status.get("elapsed_ms")
                row["error"] = status.get("error")

                # 单因子分析指标
                fa_path = os.path.join(sub_dir, FACTOR_ANALYZE_FILE)
                if os.path.exists(fa_path):
                    try:
                        with open(fa_path, encoding="utf-8") as reader:
                            fa = json.load(reader)
                        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir"]:
                            row[col] = fa.get(col)
                    except Exception as e:
                        self.log.error("summary.fa_read_failed", submission_id=sid, error=str(e), msg="读取单因子分数结果失败")

                # 截面排名得分
                row["score"] = sfa_scores.get(sid)

                # 产物是否落盘
                row["has_raw_factor"] = os.path.exists(os.path.join(sub_dir, RAW_FACTOR_FILE))
                row["has_process_factor"] = os.path.exists(os.path.join(sub_dir, PROCESS_FACTOR_FILE))
                row["has_regression_score"] = os.path.exists(os.path.join(sub_dir, FACTOR_REGRESSION_SCORE))

                rows.append(row)

        if not rows:
            self.log.warning("summary.empty", msg="没有任何提交运行结果可汇总")
            return

        df = pd.DataFrame(rows)
        # 指标统一转数值，便于排序与后续分析
        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir", "score", "elapsed_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("score", ascending=False, na_position="last")

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        df.to_csv(self.submissions_summary_csv, index=False)
        self.log.info("summary.saved", count=len(df), path=self.submissions_summary_csv, msg="汇总提交运行结果完成")

    # ---- 单因子排名 -------------------------------------------------------

    def score_sfa(self) -> None:
        """单因子分析横向排名。

        参考基类 rank_score()，区别：
        1. 不从 alphathon_api 拉结果，而是遍历 submissions 目录读取 FACTOR_ANALYZE_FILE；
        2. 截面 rank 后的得分快照另存到 leaderboard_sfa.csv。
        """
        # sid -> submission，用于把分数写回对应提交
        submissions = self.alphathon_api.query_submissions(competition_id=self.competition_id)
        valid_ids = {str(s["id"]) for s in submissions}

        rows = []
        if os.path.isdir(self.submission_dir):
            for sid in os.listdir(self.submission_dir):
                if sid not in valid_ids:
                    continue
                fa_path = os.path.join(self.submission_dir, sid, FACTOR_ANALYZE_FILE)
                if not os.path.exists(fa_path):
                    continue
                try:
                    with open(fa_path, encoding="utf-8") as reader:
                        fa = json.load(reader)
                except Exception as e:
                    self.log.error("sfa.read_failed", submission_id=sid, error=str(e), msg="无法读取单因子分数结果")
                    continue
                fa = dict(fa)
                fa["id"] = sid
                rows.append(fa)

        if not rows:
            self.log.warning("sfa.empty", msg="没有任何单因子分数结果")
            return

        df = pd.DataFrame(rows)
        # 指标可能因 json 序列化变成字符串/缺失，统一转数值，rank 会自动忽略 NaN
        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        df = self.compute_sfa_score(df)

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        df.to_csv(self.leaderboard_sfa_csv, index=False)
        self.log.info("sfa.ranked", count=len(df), msg="单因子分数截面排名完成")

        for _, row in df.iterrows():
            score = row["score"]
            # NaN != NaN：空分数统一记为 -2 失败
            if score != score:
                score = -2
            self.alphathon_api.update_submission_score(
                submission_id=row["id"],
                **{self.score_field: float(score)},
            )

    def compute_sfa_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算单因子得分：四个指标各占 25%，按截面 rank 百分位加权。"""
        df["score"] = (
            df["ic_mean"].rank(pct=True) * 0.25
            + df["ic_ir"].rank(pct=True) * 0.25
            + df["sharpe_ratio"].rank(pct=True) * 0.25
            + df["stress_ic_ir"].rank(pct=True) * 0.25
        )
        return df

    # ---- 因子池构建 -------------------------------------------------------

    def save_factor_pool(self) -> None:
        """汇总优质因子，构建因子池 parquet。

        1. 遍历 submissions 目录读取 PROCESS_FACTOR_FILE，因子名取该提交的 ipynb 文件名；
        2. 按队伍分组，每队保留单因子得分排名前 FACTOR_POOL_TOP_N 的因子；
        3. 全部因子按 date / instrument 做 outer merge，落盘为 parquet。
        """
        submissions = self.alphathon_api.query_submissions(competition_id=self.competition_id)
        sub_by_id = {str(s["id"]): s for s in submissions}

        # 读取单因子得分，用于每个队伍内部的因子取舍
        sfa_scores: dict[str, float] = {}
        if os.path.exists(self.leaderboard_sfa_csv):
            sfa_df = pd.read_csv(self.leaderboard_sfa_csv)
            sfa_scores = {str(r["id"]): r["score"] for _, r in sfa_df.iterrows()}

        records = []
        if os.path.isdir(self.submission_dir):
            for sid in os.listdir(self.submission_dir):
                submission = sub_by_id.get(sid)
                if submission is None:
                    continue
                pf_path = os.path.join(self.submission_dir, sid, PROCESS_FACTOR_FILE)
                if not os.path.exists(pf_path):
                    continue
                records.append({
                    "sid": sid,
                    "group": self._group_key(submission),
                    "factor_name": sid,
                    "score": sfa_scores.get(sid, float("nan")),
                    "path": pf_path,
                })

        if not records:
            self.log.warning("pool.empty", msg="没有任何因子数据")
            return

        meta = pd.DataFrame(records)
        # 每个队伍内按单因子得分倒序，保留前 N（缺失得分排最后）
        meta["rank_in_group"] = (
            meta.groupby("group")["score"].rank(method="first", ascending=False, na_option="bottom")
        )
        kept = meta[meta["rank_in_group"] <= self.FACTOR_POOL_TOP_N]

        pool = None
        used_names: set[str] = set()
        for _, r in kept.iterrows():
            try:
                fdf = pd.read_parquet(r["path"])
            except Exception as e:
                self.log.error("pool.read_failed", submission_id=r["sid"], error=str(e), msg="无法读取因子数据")
                continue
            if not {"date", "instrument", "factor"}.issubset(fdf.columns):
                continue

            name = r["factor_name"]
            used_names.add(name)

            fdf = fdf[["date", "instrument", "factor"]].rename(columns={"factor": name})
            pool = fdf if pool is None else pool.merge(fdf, how="outer", on=["date", "instrument"])

        factor_cols = [] if pool is None else [c for c in pool.columns if c not in ("date", "instrument")]
        # 因子池回归要求至少 2 个因子，否则没有意义
        if pool is None or len(factor_cols) < 2:
            self.log.warning("pool.too_few", count=len(factor_cols), msg="因子池数量太少，无法进行回归")
            return

        os.makedirs(os.path.dirname(self.factor_pool_path), exist_ok=True)
        pool.to_parquet(self.factor_pool_path)
        self.log.info("pool.saved", factors=len(factor_cols), msg="保存因子池数据")

    # ---- 辅助 -------------------------------------------------------------

    def _group_key(self, submission: dict) -> str:
        """队伍分组键。当前 API 未暴露队伍列表，按 user_id 分组（一个用户视作一个队伍）
        TODO: 这里有问题，需要解决
        """
        return str(submission.get("user_id") or submission.get("id"))


if __name__ == "__main__":
    Judge().run()
