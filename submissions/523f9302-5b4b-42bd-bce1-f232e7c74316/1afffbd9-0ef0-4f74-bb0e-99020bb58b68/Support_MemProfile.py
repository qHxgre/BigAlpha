"""内存/拉取策略：一个 mem_mode 展开为块大小、水位、预取深度、超前帽。"""

from dataclasses import dataclass
from typing import Literal

MemMode = Literal["low", "mid", "high"]

@dataclass(frozen=True)
class MemProfile:
    name: MemMode
    # 一次 DAI 最多覆盖的交易日数
    fetch_chunk_days: int
    # 前方剩余天数 < 该值时主动再拉一块；0 = 不主动超前
    low_water_days: int
    # 已组装 batch 的有界队列深度（GPU ↔ 组装）
    prefetch_size: int
    # Base 相对组装前沿最多超前交易日数（内存背压）；0 = 不限制
    prefetch_max_ahead_days: int


PROFILES: dict[MemMode, MemProfile] = {
    "low": MemProfile(
        name="low",
        fetch_chunk_days=1,
        low_water_days=0,
        prefetch_size=1,
        prefetch_max_ahead_days=2,
    ),
    "mid": MemProfile(
        name="mid",
        fetch_chunk_days=30,
        low_water_days=15,
        prefetch_size=4,
        prefetch_max_ahead_days=45,
    ),
    "high": MemProfile(
        name="high",
        fetch_chunk_days=120,
        low_water_days=60,
        prefetch_size=16,
        prefetch_max_ahead_days=180,
    ),
}


def resolve_profile(mem_mode: MemMode = "low") -> MemProfile:
    if mem_mode not in PROFILES:
        raise ValueError(f"unknown mem_mode={mem_mode!r}, expect {list(PROFILES)}")
    return PROFILES[mem_mode]
