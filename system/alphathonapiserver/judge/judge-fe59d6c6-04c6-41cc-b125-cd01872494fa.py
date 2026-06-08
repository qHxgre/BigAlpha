import os
import sys
import structlog

import dai

from judgebase import JudgeBase, LocalProcessUserRunner

logger = structlog.get_logger()


JUDGE_RUNNER_CODE = """
__USER_CODE__

def judge_runner_main():
    # main from user code
    data = main("cn_stock_prefactors", "2025-04-01", "2025-11-30")

    # 特殊 fix: date=20230301
    import pandas as pd
    if data["date"].dtype == "int32":
        data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
    if data["date"].max().year == 1970:
        data["date"] = pd.to_datetime(data["date"].astype("int"), format="%Y%m%d")

    from bigmodule import M
    result = M.factorlens._latest(
        data=data,
        m_cached=False,
    )

    with open("output.data", "w") as writer:
        writer.write(result._result.id)
"""

class Judge(JudgeBase):
    def on_submission(self, submission: dict):
        if True or submission["id"] in {"bd089090-5c69-42ac-8636-0a6e0683f09a"}:
            print("New submission:", submission["id"])
            try:
                runner = LocalProcessUserRunner(
                    submission_id=submission['id'],
                    files={
                        "judge_runner.py": JUDGE_RUNNER_CODE.replace(
                            "__USER_CODE__", self.alphathon_api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)),
                    },
                    cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
                )
                runner.run(_raise=True)

                with open(os.path.join(runner.runner_dir, "output.data")) as reader:
                    score_data = {
                        "raw_result": dai.DataSource(reader.read()).read().iloc[0].to_dict()
                    }

                score = -1
                print(submission["id"], score, score_data)
            except Exception as e:
                score = -2
                score_data = {
                    "err_msg": "run error: check your code / get code templates in [code] tab",
                }
                logger.exception(e)

            # 如果评分成功，更新 submission 的 public_score 和 score_data
            self.alphathon_api.update_submission_score(
                submission_id=submission["id"],
                public_score=score,
                public_score_data=score_data
            )
            self.c_rank_score()

    def on_tick(self):
        self.c_rank_score()

    def c_rank_score(self):
        # TODO: 2
        all_submissions = self.alphathon_api.query_submissions(
            competition_id=self.competition_id,
        )
        logger.info(f"Found {len(all_submissions)} submissions")
        import pandas as pd
        raw_results = []
        for x in all_submissions:
            public_score_data = x.get("public_score_data")
            if not public_score_data:
                continue
            raw_result = public_score_data.get("raw_result")
            if not raw_result:
                continue
            raw_result["id"] = x["id"]
            raw_results.append(raw_result)
        # print(raw_results)

        df = pd.DataFrame(raw_results)
        df["score"] = df["rank_ic"].rank(pct=True) * 0.4 + df["rank_ir"].rank(pct=True) * 0.3 + df["sharp_ratio"].rank(pct=True) * 0.2 + df["turnover"].rank(pct=True, ascending=False) * 0.1
        df.to_csv("/home/aiuser/work/data/alphathon/fe59d6c6-04c6-41cc-b125-cd01872494fa.csv")

        # print(df)
        for i, row in df.iterrows():
            # print(i, row)
            self.alphathon_api.update_submission_score(
                submission_id=row.id,
                public_score=row.score,
                # public_score_data=score_data
            )


if __name__ == "__main__":
    Judge(
        "fe59d6c6-04c6-41cc-b125-cd01872494fa",
        60,
    ).run()
