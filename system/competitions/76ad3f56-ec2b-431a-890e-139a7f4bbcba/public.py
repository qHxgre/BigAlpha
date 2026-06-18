import os
import sys
paths = ['/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

import json
import pandas as pd

from judge.judgebase import JudgeBase, LocalProcessUserRunner

RAW_FACTOR_FILE = "raw_factor.parquet"
PROCESS_FACTOR_FILE = "process_factor.parquet"
FACTOR_ANALYZE_FILE = "factor_analyze.json"
FACTOR_REGRESSION_SCORE = "factor_regression_score.parquet"

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
JUDGE_REG = (
    JUDGE_RUNNER_CODE_2
    .replace("__FACTOR_REGRESSION_SCORE__", FACTOR_REGRESSION_SCORE)
)


class Judge(JudgeBase):
    competition_id = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
    mode = "public"
    JUDGE_SFA = JUDGE_SFA
    JUDGE_REG = JUDGE_REG

    # 每个队伍最多入选因子池的因子数量
    FACTOR_POOL_TOP_N = 50

    # ---- 路径 -------------------------------------------------------------

    @property
    def leaderboard_sfa_csv(self) -> str:
        """单因子分析榜单（截面 rank 后的得分快照）。"""
        return os.path.join(self.leaderboard_dir, "leaderboard_sfa.csv")

    @property
    def factor_pool_path(self) -> str:
        """因子池 parquet 的绝对路径（注入到 JUDGE_REG 模板里供子进程读取）。"""
        return os.path.join(self.leaderboard_dir, "factor_pool.parquet")

    # ---- 运行用户/注入代码 -------------------------------------------------

    def run_user_code(self, submission: dict, runner_code: str) -> LocalProcessUserRunner:
        """拉取用户代码、注入 runner 模板，并在隔离子进程中执行。

        与基类不同：runner_code 由调用方传入（JUDGE_SFA / JUDGE_REG），
        并额外把 __FACTOR_POOL_FILE__ 替换成因子池的绝对路径。
        """
        sid = submission["id"]
        # ipynb 会被转成 .py 字符串，便于注入到 runner 模板中
        user_code = self.alphathon_api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)
        if isinstance(user_code, bytes):
            user_code = user_code.decode("utf-8")
        user_code = self.preprocess_user_code(submission, user_code)

        injected = (
            runner_code
            .replace("__USER_CODE__", user_code)
            .replace("__FACTOR_POOL_FILE__", self.factor_pool_path)
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

    def on_submission(self, submission: dict) -> None:
        sid = submission["id"]
        self.log.info("[submission] 开始处理提交文件", submission_id=sid)

        # 第一步：落盘原始文件 + 跑单因子分析。跑不通直接记 -2 并返回。
        try:
            self.save_submission_files(submission)
            self.run_user_code(submission, self.JUDGE_SFA)
            self.log.info("[submission] 代码运行成功", submission_id=sid)
        except Exception as e:
            # -2 表示用户代码运行失败；err_msg 会回显在前端的提交详情里
            self.log.exception("[submission] 代码运行失败", submission_id=sid, error=str(e))
            self.alphathon_api.update_submission_score(
                submission_id=sid,
                **{
                    self.score_field: -2,
                    self.score_data_field: {"err_msg": "run error: check your code / get code templates in [code] tab"},
                },
            )
            return

        # 第二步：单因子横向排名，刷新公榜
        try:
            self.score_sfa()
        except Exception as e:
            self.log.exception("[sfa] 计算得分失败", submission_id=sid, error=str(e))

        # 第三步：用排名靠前的因子拼出因子池，并对该提交跑因子池回归（产物落盘，供后续分析）。
        try:
            self.save_factor_pool()
            if os.path.exists(self.factor_pool_path):
                self.run_user_code(submission, self.JUDGE_REG)
        except Exception as e:
            self.log.exception("factor_pool_regression.failed", submission_id=sid, error=str(e))

    def on_tick(self) -> None:
        """每个 tick 重排一次单因子公榜（增量刷新）。"""
        try:
            self.score_sfa()
            self.log.exception("[on_tick] 刷新榜单", error=str(e))
        except Exception as e:
            self.log.exception("[on_tick] 刷新榜单失败", error=str(e))

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
                except Exception:
                    self.log.exception("[sfa] 无法读取 sfa 分数结果", submission_id=sid)
                    continue
                fa = dict(fa)
                fa["id"] = sid
                rows.append(fa)

        if not rows:
            self.log.exception("[sfa] 没有任何 sfa 分数结果")
            return

        df = pd.DataFrame(rows)
        # 指标可能因 json 序列化变成字符串/缺失，统一转数值，rank 会自动忽略 NaN
        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        df = self.compute_sfa_score(df)

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        df.to_csv(self.leaderboard_sfa_csv, index=False)
        self.log.info("[sfa] 单因子分数截面排名成功", count=len(df))

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
            self.log.exception("[regression] 没有任何因子数据")
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
            except Exception:
                self.log.exception("[regression] 无法读取因子数据", submission_id=r["sid"])
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
            self.log.exception("[regression] 因子池数量太少，无法进行回归", count=len(factor_cols))
            return

        os.makedirs(os.path.dirname(self.factor_pool_path), exist_ok=True)
        pool.to_parquet(self.factor_pool_path)
        self.log.info("[regression] 保存因子池数据", factors=len(factor_cols))

    # ---- 辅助 -------------------------------------------------------------

    def _group_key(self, submission: dict) -> str:
        """队伍分组键。当前 API 未暴露队伍列表，按 user_id 分组（一个用户视作一个队伍）
        TODO: 这里有问题，需要解决
        """
        return str(submission.get("user_id") or submission.get("id"))


if __name__ == "__main__":
    Judge().run()
