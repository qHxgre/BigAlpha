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
    
    data = main("cpt_dwc_2026_stock_hs300_snapshot", "2023-01-01 00:00:00", "2024-12-31 23:59:00")
    data.to_parquet('factor_data.parquet')
    # def build_check_data(datasource, start_date, end_date):
    #     import dai
    #     import pandas as pd
    #     from datetime import datetime, timedelta

    #     sql = "SELECT date FROM all_trading_days WHERE market_code='CN'"
    #     df = dai.query(sql, filters={'date': [start_date, end_date]}).df()
    #     # 扩展到15分钟时间点
    #     time_segment = ['0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130', 
    #                     '1315', '1330', '1345', '1400', '1415', '1430', '1445', '1500']
    #     all_timestamps = [
    #         pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {ts[:2]}:{ts[2:]}:00")
    #         for date in df['date']
    #         for ts in time_segment
    #     ]
    #     expanded_df = pd.DataFrame({'date': sorted(all_timestamps)})
    #     cutoff_times = sorted(pd.to_datetime(expanded_df['date']).dt.to_pydatetime())

    #     df_list = []
    #     for time in cutoff_times:
    #         ed = (time+timedelta(seconds=30)).strftime('%Y-%m-%d %H:%M:%S')
    #         sd = (datetime.strptime(ed, '%Y-%m-%d %H:%M:%S') - timedelta(days=22)).strftime('%Y-%m-%d %H:%M:%S')
    #         temp_df = main(datasource, sd, ed)
    #         df_list.append(temp_df[temp_df['date']==time])
    #         # print(time, sd, ed)

    #     check_data = pd.concat(df_list).drop_duplicates(subset=['date', 'instrument'])
    #     return check_data.reset_index(drop=True)

    # check_data = build_check_data('cpt_dwc_2026_stock_hs300_snapshot_test', '2025-04-14', '2025-04-15')

    # from bigmodule import M
    # result = M.eval_dwc._latest(
    #     data=data,
    #     check_data=check_data,
    # )
    
    # with open("output.data", "w") as writer:
    #     writer.write(result._result.id)
"""

class Judge(JudgeBase):
    def on_submission(self, submission: dict):
        if True:
            print("New submission:", submission["id"])
            try:
                user_code = self.alphathon_api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)
                if isinstance(user_code, bytes):
                    user_code = user_code.decode("utf-8")
                runner = LocalProcessUserRunner( 
                    submission_id=submission['id'],
                    files={
                        "judge_runner.py": JUDGE_RUNNER_CODE.replace(
                            "__USER_CODE__", user_code),
                    },
                    cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
                )
                runner.run(_raise=True)
                logger.info(f"{submission['id']} 运行成功")

                # with open(os.path.join(runner.runner_dir, "output.data")) as reader:
                #     score_data = {
                #         "raw_result": dai.DataSource(reader.read()).read().iloc[0].to_dict()
                #     }

                # score = -1
                # print(submission["id"], score, score_data)
            except Exception as e:
                logger.exception(f"{submission['id']} 运行失败")

                # score = -2
                # score_data = {
                #     "err_msg": "run error: check your code / get code templates in [code] tab",
                # }
                # logger.exception(e)

            # # 如果评分成功，更新 submission 的 public_score 和 score_data
            # self.alphathon_api.update_submission_score(
            #     submission_id=submission["id"],
            #     public_score=score,
            #     public_score_data=score_data
            # )
            # self.c_rank_score()

    def on_tick(self):
        self.c_rank_score()

    def c_rank_score(self):
        return
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
        df["score"] = df["sfa_sharp"].rank(pct=True) * 0.3 + df["backtest_excess"].rank(pct=True) * 0.21 + df["bacttest_sharp"].rank(pct=True) * 0.49
        df.to_csv("/home/aiuser/work/data/alphathon/bf1b4468-6b4d-43dc-98e1-8c2358c61793.csv")

        for i, row in df.iterrows():
            if row.score != row.score:
                row.score = -2
            # print(i, row)
            self.alphathon_api.update_submission_score(
                submission_id=row.id,
                public_score=row.score,
                # public_score_data=score_data
            )


if __name__ == "__main__":
    Judge(
        "bf1b4468-6b4d-43dc-98e1-8c2358c61793",
        60,
    ).run()

