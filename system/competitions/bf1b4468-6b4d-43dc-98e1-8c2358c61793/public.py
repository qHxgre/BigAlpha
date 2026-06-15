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

    # 公榜只跑用户代码、落盘 factor_data.parquet，不产出榜单分数：
    # extract_result 返回 None 即可跳过排名（on_submission 跑通后记 submission.ok）。
    def extract_result(self, submission, runner):
        return None

    def on_tick(self) -> None:
        return


if __name__ == "__main__":
    Judge().run()
