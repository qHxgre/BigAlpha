"""ElasticNet 滚动回归相关常量。

参考 docs/因子挖掘_介绍_20260603.md "Elastic Net 回归得分"：
- 滚动窗口长度 60 个交易日
- 步长 20 个交易日
- ModelScore_i = mean(|w_i|) / (std(|w_i|) + eps)
"""

WINDOW = 60
STEP = 20
EPS = 1e-8

DEFAULT_ALPHA = 1e-3
DEFAULT_L1_RATIO = 0.5

MIN_WINDOW_SAMPLES = 50
MIN_SAMPLES_PER_FACTOR = 5

TOP_N_DISPLAY = 20
