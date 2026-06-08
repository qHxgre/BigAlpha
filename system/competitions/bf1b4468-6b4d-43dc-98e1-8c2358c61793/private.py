import _bootstrap  # noqa: F401

from judgebase import JudgeBase


JUDGE_RUNNER_CODE = '''
__USER_CODE__

def judge_runner_main():
    data = main("cpt_dwc_2026_stock_hs300_snapshot_private", "2026-01-01 00:00:00", "2026-03-19 23:59:59")

    def build_check_data(datasource, start_date, end_date):
        import dai
        import pandas as pd
        from datetime import datetime, timedelta

        sql = "SELECT date FROM all_trading_days WHERE market_code='CN'"
        df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
        time_segment = ["0945", "1000", "1015", "1030", "1045", "1100", "1115", "1130",
                        "1315", "1330", "1345", "1400", "1415", "1430", "1445", "1500"]
        all_timestamps = [
            pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {ts[:2]}:{ts[2:]}:00")
            for date in df["date"]
            for ts in time_segment
        ]
        expanded_df = pd.DataFrame({"date": sorted(all_timestamps)})
        cutoff_times = sorted(pd.to_datetime(expanded_df["date"]).dt.to_pydatetime())

        df_list = []
        for time in cutoff_times:
            ed = (time + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
            sd = (datetime.strptime(ed, "%Y-%m-%d %H:%M:%S") - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
            temp_df = main(datasource, sd, ed)
            df_list.append(temp_df[temp_df["date"] == time])

        check_data = pd.concat(df_list).drop_duplicates(subset=["date", "instrument"])
        return check_data.reset_index(drop=True)

    check_data = build_check_data("cpt_dwc_2026_stock_hs300_snapshot_private", "2026-03-12", "2026-03-12")

    from bigmodule import M
    result = M.eval_dwc._latest(data=data, check_data=check_data)
    with open("output.data", "w") as writer:
        writer.write(result._result.id)
'''


# 这几个提交里用的是 LAST(mid_price)，未指定 order，需要打补丁
SPECIAL_ORDER_FIX_IDS = {
    "b270365c-875b-4108-9fbb-cca22283a4f7",
    "1e3692e9-901e-4589-b45b-328cf076466b",
    "0d3856aa-6e6d-4b12-8d82-5004ddb7c388",
    "a8b3a84e-cb81-41cb-ad55-0a21052d92eb",
}


class Judge(JudgeBase):
    competition_id = "bf1b4468-6b4d-43dc-98e1-8c2358c61793"
    mode = "private"
    JUDGE_RUNNER_CODE = JUDGE_RUNNER_CODE
    max_workers = 5

    def preprocess_user_code(self, submission: dict, code: str) -> str:
        if submission["id"] in SPECIAL_ORDER_FIX_IDS:
            code = code.replace(
                "LAST(mid_price) as close_price",
                "LAST(mid_price order by date) as close_price",
            )
        return code

    def compute_score(self, df):
        df["score"] = (
            df["sfa_sharp"].rank(pct=True) * 0.3
            + df["backtest_excess"].rank(pct=True) * 0.21
            + df["bacttest_sharp"].rank(pct=True) * 0.49
        )
        return df


if __name__ == "__main__":
    Judge().run()
