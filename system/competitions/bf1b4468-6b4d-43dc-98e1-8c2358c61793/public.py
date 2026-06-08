import os
from alphathonapiserver.judge.judgebase import JudgeBase


JUDGE_RUNNER_CODE = '''
__USER_CODE__

def judge_runner_main():
    data = main("cpt_dwc_2026_stock_hs300_snapshot", "2023-01-01 00:00:00", "2024-12-31 23:59:00")
    data.to_parquet("factor_data.parquet")
'''


class Judge(JudgeBase):
    competition_id = "bf1b4468-6b4d-43dc-98e1-8c2358c61793"
    mode = "public"
    JUDGE_RUNNER_CODE = JUDGE_RUNNER_CODE
    max_workers = 5

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
            self.log.info("submission.ok", submission_id=sid)
        except Exception as e:
            self.log.exception("submission.failed", submission_id=sid, error=str(e))

    def on_tick(self) -> None:
        return

    def compute_score(self, df):
        df["score"] = (
            df["sfa_sharp"].rank(pct=True) * 0.3
            + df["backtest_excess"].rank(pct=True) * 0.21
            + df["bacttest_sharp"].rank(pct=True) * 0.49
        )
        return df


if __name__ == "__main__":
    Judge().run()
