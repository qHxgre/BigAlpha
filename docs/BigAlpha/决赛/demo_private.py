"""私榜如何运行一个 submission：教学精简版。"""

import pandas as pd

from bigmodule import M
from demo_submission import main


DATASETS = {
    "bar1m": "bigalpha_2026_stock_bar1m_222",
    "financial": "bigalpha_2026_financial_222",
}
FULL_START = "2025-03-01 00:00:00"
FULL_END = "2026-08-10 23:59:59"


def run_period(start, end):
    """平台注入私榜数据表和当前分段日期，运行同一个 main()。"""
    factor = main(DATASETS, start, end)
    return M.bigalpha_eval._latest(
        factor_data=factor,
        start_date=start,
        end_date=end,
        show=False,
    )["raw_factor"]


# 1. 实际后台分两段运行，避免一次计算超长区间。
first = run_period("2025-03-01 00:00:00", "2025-11-30 23:59:59")
second = run_period("2025-12-01 00:00:00", "2026-08-10 23:59:59")

# 2. 拼接两段原始因子，再按完整区间统一重算 A 项。
merged_factor = pd.concat([first, second], ignore_index=True)
single = M.bigalpha_eval._latest(
    factor_data=merged_factor,
    start_date=FULL_START,
    end_date=FULL_END,
    show=False,
)

# 3. 假设 parquet 中已有其他私榜候选因子，将本因子合并进去。
factor_pool = pd.read_parquet("private_factor_pool.parquet")
my_factor = single["process_factor"].rename(columns={"factor": "demo_submission"})
factor_pool = factor_pool.merge(my_factor, on=["date", "instrument"], how="outer")

# 4. 对完整私榜因子池统一回归，得到 B 项 ModelScore。
regression = M.bigalpha_eval._latest(
    factor_pool=factor_pool,
    start_date=FULL_START,
    end_date=FULL_END,
    process_pools=False,
    show=False,
)

print("A 项原始指标：", single["factor_analyze"])
print("B 项回归结果：")
print(regression["factor_regression"]["per_factor_scores"])

# 平台最后在私榜候选因子中计算百分位：final_score = 0.3 * A + 0.7 * B。
# 所有候选 submission 处理完成并人工核验后，才发布最终私榜。
