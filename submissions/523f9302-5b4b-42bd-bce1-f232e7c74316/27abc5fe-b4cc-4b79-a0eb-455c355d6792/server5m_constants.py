"""本地实验使用的唯一数据口径。"""

from __future__ import annotations

LEVELS = 5
SEQ_LEN = 48

# 按档位交错排列，让卷积依次完成“同档价量配对 → 买卖盘合并 → 五档聚合”。
FEATURE_COLUMNS = tuple(
    column
    for level in range(1, LEVELS + 1)
    for column in (
        f"ask_price{level}",
        f"ask_volume{level}",
        f"bid_price{level}",
        f"bid_volume{level}",
    )
)
PRICE_INDICES = tuple(i for i, name in enumerate(FEATURE_COLUMNS) if "price" in name)
VOLUME_INDICES = tuple(i for i, name in enumerate(FEATURE_COLUMNS) if "volume" in name)

TARGETS = {
    **{f"ret_{h}": ("regression", h, 1) for h in (1, 2, 3)},
    **{f"cls{bins}_{h}": ("classification", h, bins)
       for h in (1, 2, 3) for bins in (5, 10)},
}

SPLIT_RANGES = {
    "train": ("2022-01-01", "2023-12-31"),
    "val": ("2024-01-01", "2024-06-30"),
    "test": ("2024-07-01", "2024-12-31"),
}
