"""公榜如何运行一个 submission：教学精简版。"""

import pandas as pd

from bigmodule import M
from demo_submission import main


DATASETS = {
    "bar1m": "bigalpha_2026_stock_bar1m_111",
    "financial": "bigalpha_2026_financial_111",
}
START = "2025-03-01 00:00:00"
END = "2025-11-30 23:59:59"


# 1. 平台注入公榜数据表和日期，调用参赛者的同一个 main()。
factor_data = main(DATASETS, START, END)

# 2. 单因子评估，得到 A 项所需指标和预处理后的因子。
single = M.bigalpha_eval._latest(
    factor_data=factor_data,
    start_date=START,
    end_date=END,
    show=False,
)

# 3. 假设 parquet 中已有其他提交的处理后因子，将本因子合并进去。
factor_pool = pd.read_parquet("public_factor_pool.parquet")
my_factor = single["process_factor"].rename(columns={"factor": "demo_submission"})
factor_pool = factor_pool.merge(my_factor, on=["date", "instrument"], how="outer")

# 4. 对完整公榜因子池统一回归，得到 B 项 ModelScore。
regression = M.bigalpha_eval._latest(
    factor_pool=factor_pool,
    start_date=START,
    end_date=END,
    process_pools=False,
    show=False,
)

print("A 项原始指标：", single["factor_analyze"])
print("B 项回归结果：")
print(regression["factor_regression"]["per_factor_scores"])

# 平台最后在全体因子中计算百分位：final_score = 0.3 * A + 0.7 * B。
# 每批新 submission 跑通后，公榜会重新建池、重新回归和更新排名。
# 正式评测还会使用截断数据表复跑一次，用于检测未来函数；此处省略。
