import _bootstrap  # noqa: F401

from judgebase import JudgeBase


JUDGE_RUNNER_CODE = '''
__USER_CODE__

def judge_runner_main():
    data = main("cpt_jyc_2025_stock_csi1000_bar1m_private", "2025-01-01 08:00:00", "2025-11-16 23:59:59")

    import pandas as pd
    if data["date"].dtype == "int32":
        data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
    if data["date"].max().year == 1970:
        data["date"] = pd.to_datetime(data["date"].astype("int"), format="%Y%m%d")

    from bigmodule import M
    result = M.factorlens._latest(data=data, m_cached=False)

    with open("output.data", "w") as writer:
        writer.write(result._result.id)
'''


class Judge(JudgeBase):
    competition_id = "5c3f7783-4158-4196-97ab-171b27218c7c"
    mode = "private"
    JUDGE_RUNNER_CODE = JUDGE_RUNNER_CODE

    def compute_score(self, df):
        df["score"] = (
            df["rank_ic"].rank(pct=True) * 0.4
            + df["rank_ir"].rank(pct=True) * 0.3
            + df["sharp_ratio"].rank(pct=True) * 0.2
            + df["turnover"].rank(pct=True, ascending=False) * 0.1
        )
        return df


if __name__ == "__main__":
    Judge().run()
