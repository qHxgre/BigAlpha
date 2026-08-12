from enum import Enum
from typing import List, Tuple


BM_DICT = {
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "沪深300": "000300.SH",
}

STRESS_PERIODS: List[Tuple[str, str, str]] = [
    ("2020-02-03~2020-03-31", "2020-02-03", "2020-03-31"),  # COVID crash
    ("2024-01-15~2024-02-08", "2024-01-15", "2024-02-08"),  # 小市值流动性危机
    ("2024-09-24~2024-10-08", "2024-09-24", "2024-10-08"),  # 政策驱动急涨
    ("2025-04-07~2025-04-30", "2025-04-07", "2025-04-30"),  # 关税冲击
    ("2025-12-08~2025-12-18", "2025-12-08", "2025-12-18"),  # 年末市场回调
    ("2026-03-19~2026-04-03", "2026-03-19", "2026-04-03"),  # 地缘冲突与风险重定价
    ("2026-05-20~2026-06-10", "2026-05-20", "2026-06-10"),  # 科技成长板块集中调整
]


class DataType(str, Enum):
    LONG = "long"
    SHORT = "short"
    LONG_SHORT = "long_short"

    def __str__(self):
        return self.value


class PortfolioCode(str, Enum):
    ll_pos = "9"
    ss_pos = "0"
    ls_pos = "ls"
    bm_pos = "bm"

    def __str__(self):
        return self.value
