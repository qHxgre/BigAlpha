import concurrent.futures
import json
import os
import time
from typing import Any, Dict, List, Optional

import structlog

from api import AlphathonAPI
from paths import COMPLETE_IDS_DIR, LEADERBOARD_DIR
from runner import K8SPodUserRunner, LocalProcessUserRunner, UserCodeRunner

logger = structlog.get_logger()

__all__ = [
    "AlphathonAPI",
    "JudgeBase",
    "K8SPodUserRunner",
    "LocalProcessUserRunner",
    "UserCodeRunner",
]


class JudgeBase:
    """所有评测器的基类。子类放在 system/competitions/{cid}/{public,private}.py 下，
    只需要声明：
        - competition_id: 比赛 id
        - mode: "public" 或 "private"
        - JUDGE_RUNNER_CODE: 包裹用户代码的运行模板
        - compute_score(df): 输入所有原始结果的 DataFrame，返回带 score 列的 DataFrame
    可选 hook：preprocess_user_code(submission, code), query_constraints(), max_workers
    """

    competition_id: str = ""
    mode: str = "public"  # public / private
    tick_interval: int = 60
    max_workers: int = 5
    JUDGE_RUNNER_CODE: str = ""

    def __init__(self, competition_id: Optional[str] = None, tick_interval: Optional[int] = None) -> None:
        if competition_id:
            self.competition_id = competition_id
        if tick_interval is not None:
            self.tick_interval = tick_interval
        assert self.competition_id, "competition_id 必须在子类或构造时指定"
        assert self.mode in ("public", "private"), f"mode 必须是 public 或 private, got {self.mode}"

        self.alphathon_api = AlphathonAPI()
        self.log = logger.bind(competition_id=self.competition_id, mode=self.mode)
        self.log.info("judge.init", tick_interval=self.tick_interval)

    @property
    def score_field(self) -> str:
        return f"{self.mode}_score"

    @property
    def score_data_field(self) -> str:
        return f"{self.mode}_score_data"

    @property
    def leaderboard_csv(self) -> str:
        os.makedirs(LEADERBOARD_DIR, exist_ok=True)
        suffix = "" if self.mode == "public" else "-private"
        return os.path.join(LEADERBOARD_DIR, f"{self.competition_id}{suffix}.csv")

    @property
    def complete_ids_file(self) -> str:
        os.makedirs(COMPLETE_IDS_DIR, exist_ok=True)
        suffix = "" if self.mode == "public" else "-private"
        return os.path.join(COMPLETE_IDS_DIR, f"{self.competition_id}{suffix}-ids.json")

    def query_constraints(self) -> Dict[str, Any]:
        """子类可重写以限定查询范围。private 模式默认只跑入围私榜的提交。"""
        if self.mode == "private":
            return {"selected_for_private": True}
        return {}

    def preprocess_user_code(self, submission: dict, code: str) -> str:
        """子类可重写以对个别提交做特殊处理（例如修复已知错误）。"""
        return code

    def compute_score(self, df):  # df: pandas.DataFrame
        """子类必须重写：基于 raw_result 计算 score 列。"""
        raise NotImplementedError

    def on_submission(self, submission: dict) -> None:
        sid = submission["id"]
        self.log.info("submission.start", submission_id=sid)
        try:
            user_code = self.alphathon_api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)
            if isinstance(user_code, bytes):
                user_code = user_code.decode("utf-8")
            user_code = self.preprocess_user_code(submission, user_code)

            runner = LocalProcessUserRunner(
                submission_id=sid,
                files={"judge_runner.py": self.JUDGE_RUNNER_CODE.replace("__USER_CODE__", user_code)},
                cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
            )
            runner.run(_raise=True)

            import dai
            with open(os.path.join(runner.runner_dir, "output.data")) as reader:
                score_data = {"raw_result": dai.DataSource(reader.read()).read().iloc[0].to_dict()}
            score = -1
            self.log.info("submission.scored", submission_id=sid, score=score)
        except Exception as e:
            score = -2
            score_data = {"err_msg": "run error: check your code / get code templates in [code] tab"}
            self.log.exception("submission.failed", submission_id=sid, error=str(e))

        self.alphathon_api.update_submission_score(
            submission_id=sid,
            **{self.score_field: score, self.score_data_field: score_data},
        )
        self.rank_score()

    def rank_score(self) -> None:
        all_submissions = self.alphathon_api.query_submissions(competition_id=self.competition_id)
        self.log.info("rank.fetched", count=len(all_submissions))

        import pandas as pd
        raw_results = []
        for x in all_submissions:
            score_data = x.get(self.score_data_field)
            if not score_data:
                continue
            raw_result = score_data.get("raw_result")
            if not raw_result:
                continue
            raw_result["id"] = x["id"]
            raw_results.append(raw_result)

        if not raw_results:
            self.log.info("rank.empty")
            return

        df = pd.DataFrame(raw_results)
        df = self.compute_score(df)
        df.to_csv(self.leaderboard_csv, index=False)

        for _, row in df.iterrows():
            score = row.score
            if score != score:  # NaN
                score = -2
            self.alphathon_api.update_submission_score(
                submission_id=row.id,
                **{self.score_field: score},
            )

    def on_tick(self) -> None:
        self.rank_score()

    def _load_complete_ids(self) -> List[str]:
        try:
            with open(self.complete_ids_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_complete_ids(self, complete_ids: List[str]) -> None:
        with open(self.complete_ids_file, "w", encoding="utf-8") as f:
            json.dump(complete_ids, f, ensure_ascii=False, indent=4)

    def run(self) -> None:
        submitted_ids: set[str] = set()
        futures_by_id: dict[str, concurrent.futures.Future] = {}
        completed_total = 0
        complete_ids = self._load_complete_ids()
        self.log.info("complete_ids.loaded", total=len(complete_ids))

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            while True:
                self.log.info("tick.start")
                added = 0
                new_submissions = self.alphathon_api.query_submissions(
                    competition_id=self.competition_id,
                    constraints=self.query_constraints(),
                )
                pending = [s for s in new_submissions if s.get("id") not in submitted_ids]
                for submission in pending:
                    sid = submission.get("id")
                    if sid is None or sid in complete_ids:
                        continue
                    submitted_ids.add(sid)
                    fut = executor.submit(self.on_submission, submission)
                    futures_by_id[str(sid)] = fut
                    added += 1
                self.log.info("tick.dispatch", pending=len(pending), complete=len(complete_ids), tracked=len(futures_by_id))

                if futures_by_id:
                    done_ids = [sid for sid, f in futures_by_id.items() if f.done()]
                    if done_ids:
                        completed_total += len(done_ids)
                        for sid in done_ids:
                            futures_by_id.pop(sid, None)
                        complete_ids = list(set(complete_ids + done_ids))
                running = sum(1 for f in futures_by_id.values() if not f.done())
                self.log.info("tick.status", added=added, running=running, completed_total=completed_total, tracked=len(futures_by_id))

                self.on_tick()
                self._save_complete_ids(complete_ids)
                self.log.info("tick.sleep", seconds=self.tick_interval)
                time.sleep(self.tick_interval)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
