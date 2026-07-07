
def judge_runner_main():
    import json
    import pandas as pd

    # 读取评测系统落盘的因子池（所有提交的优质因子按 date/instrument 合并而成）
    factor_pool = pd.read_parquet("/home/aiuser/work/workspace/BigAlpha/system/files/76ad3f56-ec2b-431a-890e-139a7f4bbcba/leaderboard/factor_pool.parquet")

    from bigmodule import M
    result = M.bigalpha_eval._latest(
        factor_pool=factor_pool,
        start_date="2025-03-01 00:00:00",
        end_date="2025-11-30 23:59:59",
        process_pools=False,
        show=True,
    )

    # 将 per_factor_scores 落盘为 CSV（utf-8-sig 带 BOM，Excel 打开中文不乱码）
    result['factor_regression']['per_factor_scores'].to_csv(
        "/home/aiuser/work/workspace/BigAlpha/system/files/76ad3f56-ec2b-431a-890e-139a7f4bbcba/leaderboard/leaderboard_reg.csv", index=False, encoding="utf-8-sig"
    )
