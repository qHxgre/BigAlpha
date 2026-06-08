from enum import Enum
from typing import List, Tuple


BM_DICT = {
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "沪深300": "000300.SH",
}

STRESS_PERIODS: List[Tuple[str, str, str]] = [
    ("covid_2020",        "2020-02-03", "2020-03-31"),
    ("micro_cap_2024",    "2024-01-15", "2024-02-08"),
    ("policy_rally_2024", "2024-09-24", "2024-10-08"),
    ("tariff_2025",       "2025-04-07", "2025-04-30"),
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
