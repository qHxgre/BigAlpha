import os
import sys
import structlog

import dai
import json

from judgebase import JudgeBase, LocalProcessUserRunner

logger = structlog.get_logger()


JUDGE_RUNNER_CODE = """
__USER_CODE__

def judge_runner_main():
    # main from user code
    data = main("cpt_dwc_2026_stock_hs300_snapshot_private", "2026-01-01 00:00:00", "2026-03-19 23:59:59")

    def build_check_data(datasource, start_date, end_date):
        import dai
        import pandas as pd
        from datetime import datetime, timedelta

        sql = "SELECT date FROM all_trading_days WHERE market_code='CN'"
        df = dai.query(sql, filters={'date': [start_date, end_date]}).df()
        # 扩展到15分钟时间点
        time_segment = ['0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130', 
                        '1315', '1330', '1345', '1400', '1415', '1430', '1445', '1500']
        all_timestamps = [
            pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {ts[:2]}:{ts[2:]}:00")
            for date in df['date']
            for ts in time_segment
        ]
        expanded_df = pd.DataFrame({'date': sorted(all_timestamps)})
        cutoff_times = sorted(pd.to_datetime(expanded_df['date']).dt.to_pydatetime())

        df_list = []
        for time in cutoff_times:
            ed = (time+timedelta(seconds=30)).strftime('%Y-%m-%d %H:%M:%S')
            sd = (datetime.strptime(ed, '%Y-%m-%d %H:%M:%S') - timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S')
            temp_df = main(datasource, sd, ed)
            df_list.append(temp_df[temp_df['date']==time])
            # print(time, sd, ed)

        check_data = pd.concat(df_list).drop_duplicates(subset=['date', 'instrument'])
        return check_data.reset_index(drop=True)

    check_data = build_check_data('cpt_dwc_2026_stock_hs300_snapshot_private', '2026-03-12', '2026-03-12')

    from bigmodule import M
    result = M.eval_dwc._latest(
        data=data,
        check_data=check_data,
    )
    with open("output.data", "w") as writer:
        writer.write(result._result.id)
"""

class Judge(JudgeBase):
    def on_submission(self, submission: dict):
        if True or submission["id"] in {"bd089090-5c69-42ac-8636-0a6e0683f09a"}:
            print("New submission:", submission["id"])
            try:
                user_code = self.alphathon_api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)
                if isinstance(user_code, bytes):
                    user_code = user_code.decode("utf-8")
                
                # 特殊处理某位参赛者的代码
                special_ids = [
                        "b270365c-875b-4108-9fbb-cca22283a4f7",
                        "1e3692e9-901e-4589-b45b-328cf076466b",
                        "0d3856aa-6e6d-4b12-8d82-5004ddb7c388",
                        "a8b3a84e-cb81-41cb-ad55-0a21052d92eb"
                ]
                if submission["id"] in special_ids:
                    user_code = user_code.replace(
                        "LAST(mid_price) as close_price",
                        "LAST(mid_price order by date) as close_price"
                    )
                runner = LocalProcessUserRunner( 
                    submission_id=submission['id'],
                    files={
                        "judge_runner.py": JUDGE_RUNNER_CODE.replace(
                            "__USER_CODE__", user_code),
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
        # print(raw_results)

        df = pd.DataFrame(raw_results)
        df["score"] = df["sfa_sharp"].rank(pct=True) * 0.3 + df["backtest_excess"].rank(pct=True) * 0.21 + df["bacttest_sharp"].rank(pct=True) * 0.49
        df.to_csv("/home/aiuser/work/data/alphathon/bf1b4468-6b4d-43dc-98e1-8c2358c61793-private.csv")

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

        complete_ids_file = '/home/aiuser/work/data/alphathon/bf1b4468-6b4d-43dc-98e1-8c2358c61793-private-ids-20260320.json'
        try:
            with open(complete_ids_file, 'r', encoding='utf-8') as f:
                complete_ids = json.load(f)
        except:
            complete_ids = []
        logger.info(f"find complete_ids, total= {len(complete_ids)} ..")

        try:
            while True:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
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
                        #         "b270365c-875b-4108-9fbb-cca22283a4f7",
                        #         "1e3692e9-901e-4589-b45b-328cf076466b",
                        #         "0d3856aa-6e6d-4b12-8d82-5004ddb7c388",
                        #         "a8b3a84e-cb81-41cb-ad55-0a21052d92eb"
                        # ]},
                    )
                    pending = [s for s in new_submissions if s.get("id") not in submitted_ids]
                    for submission in pending:
                        sid = submission.get("id")
                        if sid is None:
                            continue
                        if sid in complete_ids:
                            continue
                        submitted_ids.add(sid)
                        fut = executor.submit(self.on_submission, submission)
                        futures_by_id[str(sid)] = fut
                        added += 1
                    logger.info("check complete, total={total}, complete={complete}, left={left}".format(
                        total=len(pending),
                        complete=len(complete_ids),
                        left=len(list(futures_by_id.keys()))
                    ))
                if futures_by_id:
                    done_ids = [sid for sid, f in futures_by_id.items() if f.done()]
                    if done_ids:
                        completed_total += len(done_ids)
                        for sid in done_ids:
                            futures_by_id.pop(sid, None)
                    complete_ids = list(set(complete_ids + done_ids))
                running = sum(1 for f in futures_by_id.values() if not f.done())
                logger.info("judge.status", added=added, running=running, completed_total=completed_total, total_tracked=len(futures_by_id))
                if hasattr(self, "on_tick"):
                    self.on_tick()
                # break

                with open(complete_ids_file, 'w', encoding='utf-8') as f:
                    json.dump(complete_ids, f, ensure_ascii=False, indent=4)
                logger.info(f"save complete_ids, total= {len(complete_ids)} ..")

                logger.info(f"sleep {self.tick_interval}s ..")
                time.sleep(self.tick_interval)
        finally:
            pass
        #     executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    Judge(
        "bf1b4468-6b4d-43dc-98e1-8c2358c61793",
        60,
    ).run()
