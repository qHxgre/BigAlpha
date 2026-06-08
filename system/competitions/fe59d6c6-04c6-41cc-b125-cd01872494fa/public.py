import _bootstrap  # noqa: F401

from judgebase import JudgeBase


JUDGE_RUNNER_CODE = '''
__USER_CODE__

def judge_runner_main():
    data = main("cn_stock_prefactors", "2025-04-01", "2025-11-30")

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
    competition_id = "fe59d6c6-04c6-41cc-b125-cd01872494fa"
    mode = "public"
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
