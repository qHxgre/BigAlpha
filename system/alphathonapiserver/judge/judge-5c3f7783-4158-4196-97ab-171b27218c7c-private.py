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
    data = main("cpt_jyc_2025_stock_csi1000_bar1m_private", "2025-01-01 08:00:00", "2025-11-16 23:59:59")

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

            # raise

            # 如果评分成功，更新 submission 的 private_score 和 score_data
            self.alphathon_api.update_submission_score(
                submission_id=submission["id"],
                private_score=score,
                private_score_data=score_data
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
            private_score_data = x.get("private_score_data")
            if not private_score_data:
                continue
            raw_result = private_score_data.get("raw_result")
            if not raw_result:
                continue
            raw_result["id"] = x["id"]
            raw_results.append(raw_result)
        print(raw_results)

        df = pd.DataFrame(raw_results)
        df["score"] = df["rank_ic"].rank(pct=True) * 0.4 + df["rank_ir"].rank(pct=True) * 0.3 + df["sharp_ratio"].rank(pct=True) * 0.2 + df["turnover"].rank(pct=True, ascending=False) * 0.1
        df.to_csv("/home/aiuser/work/data/alphathon/5c3f7783-4158-4196-97ab-171b27218c7c-private.csv")

        # print(df)
        for i, row in df.iterrows():
            # print(i, row)
            self.alphathon_api.update_submission_score(
                submission_id=row.id,
                private_score=row.score,
                # private_score_data=score_data
            )



    def run(self) -> None:
        import time
        import concurrent.futures
        submitted_ids: set[str] = set()
        futures_by_id: dict[str, concurrent.futures.Future] = {}
        completed_total = 0
        try:
            while True:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
                logger.info("tick ..")
                added = 0
                if hasattr(self, "on_submission"):
                    new_submissions = self.alphathon_api.query_submissions(
                        competition_id=self.competition_id,
                        # constraints={"private_score": None, "selected_for_private": True,},
                        constraints={"selected_for_private": True,},        # 重跑 私榜 的所有代码
                        max_pages=1,
                        # constraints={
                        #     "selected_for_private": True,
                        #     "id__in": [
                        #     "00c316c9-b866-40a6-ad8d-1034865d24c5"
                        #     # "e48c7e3d-1426-439e-a1ab-b5e53ff8b999",
                        #     # "95d40fa4-20a3-4c1a-a2d8-aff336d1466c",
                        #     # "50c6c719-159b-4899-b93f-0489b9adce21",
                        #     # "5310f38b-f7cb-47a6-b6c9-f57848a50a4b",
                        #     # "04c569ab-9004-42aa-b9f8-5eb237897917",
                        #     # "c9757c8b-c8d0-4c7f-ad68-7bbd0c90d0b8",
                        #     # "18b3c334-e348-45d9-b501-16f11aaa1e91",
                        #     # "56a126f5-16cd-416f-bf80-31c7b96c1030",
                        #     # "e07436f2-b343-4984-a745-fb918ed739c5",
                        #     # "a85ba7d8-b57e-48a2-b0b2-5cf0e45687f6",
                        #     # "faf8b26a-2122-435f-a974-711939ed54d5",
                        #     # "3ef98415-0552-4580-a170-3e9a6dff51b6",
                        #     # "807d399f-505a-4b28-bc7e-6db06b0b1bd1",
                        #     # "b018acac-bfb3-45b2-a92c-e4176eb43982",
                        #     # "d52252f8-e4fc-44a8-af3d-65d27e652c38",
                        # ]},
                    )
                    pending = [s for s in new_submissions if s.get("id") not in submitted_ids]
                    for submission in pending:
                        sid = submission.get("id")
                        if sid is None:
                            continue
                        submitted_ids.add(sid)
                        fut = executor.submit(self.on_submission, submission)
                        futures_by_id[str(sid)] = fut
                        added += 1
                if futures_by_id:
                    done_ids = [sid for sid, f in futures_by_id.items() if f.done()]
                    if done_ids:
                        completed_total += len(done_ids)
                        for sid in done_ids:
                            futures_by_id.pop(sid, None)
                running = sum(1 for f in futures_by_id.values() if not f.done())
                logger.info("judge.status", added=added, running=running, completed_total=completed_total, total_tracked=len(futures_by_id))
                if hasattr(self, "on_tick"):
                    self.on_tick()
                break
                logger.info(f"sleep {self.tick_interval}s ..")
                time.sleep(self.tick_interval)
        finally:
            pass
        #     executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    Judge(
        "5c3f7783-4158-4196-97ab-171b27218c7c",
        60,
    ).run()
