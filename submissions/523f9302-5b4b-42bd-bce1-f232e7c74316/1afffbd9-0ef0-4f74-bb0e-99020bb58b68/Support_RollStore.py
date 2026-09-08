"""
多尺度滚动基础数据缓存。

- 按日索引：_days[scale][date_day] = 当日全市场 DataFrame
- get_window 只取 lookback 各日，且每日先滤 pools 再 concat
- DAI 仍按日期块拉全市场（pools 不参与拉取）
- 双线程：BasePrefetch 调 prefetch_for；BatchAssemble 调 wait_ready / get_window / advance_assemble
- DAI 在锁外；ingest / 读日表在锁内
- 覆盖语义：_fetched[scale] = 已向 DAI 查询过的交易日
- 完整性：在 get_window 对回看窗执行 _check_base_frame
"""

import math
import threading
from typing import Dict, List, Optional, Set, Tuple

import polars as pl

from Support_BaseData import (
    VALUES_COLS,
    _apply_forward_fill,
    _check_base_frame,
    get_base_data,
)
from Support_GetDates import (
    get_last_days,
    get_next_days,
    next_trading_day_after,
    trading_days_between,
)
from Support_MemProfile import MemProfile


class RollingBaseStore:
    """多尺度滚动基础数据缓存（按日热窗 + 全市场块拉取 + 双线程协议）。"""

    BARS_PER_DAY = {"1m": 240, "5m": 48, "15m": 16, "30m": 8}
    SCALES = ("1m", "5m", "15m", "30m")

    def __init__(
        self,
        seq_len: int,
        scale_details: dict,
        profile: MemProfile,
        horizon_end: Optional[str] = None,
    ):
        self.seq_len = int(seq_len)
        self.scale_details = scale_details
        self.profile = profile
        self.horizon_end = horizon_end

        self.lookback: Dict[str, int] = {
            d: math.ceil(self.seq_len / self.BARS_PER_DAY[d]) for d in self.SCALES
        }

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        # scale -> { date_day -> 当日全市场 frame }
        self._days: Dict[str, Dict[str, pl.DataFrame]] = {d: {} for d in self.SCALES}
        self._watermark: Dict[str, Optional[str]] = {d: None for d in self.SCALES}
        self._fetched: Dict[str, Set[str]] = {d: set() for d in self.SCALES}

        self._assemble_keep_from: Optional[str] = None
        self._assemble_date: Optional[str] = None
        self._error: Optional[BaseException] = None
        self._stopped = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._cv:
            for d in self.SCALES:
                self._days[d].clear()
                self._watermark[d] = None
                self._fetched[d].clear()
            self._assemble_keep_from = None
            self._assemble_date = None
            self._error = None
            self._stopped = False
            self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()

    def set_error(self, exc: BaseException) -> None:
        with self._cv:
            self._error = exc
            self._cv.notify_all()

    # ------------------------------------------------------------------
    # 组装线程 API
    # ------------------------------------------------------------------

    def set_assemble_date(self, date: str) -> None:
        with self._cv:
            self._assemble_date = date
            self._cv.notify_all()

    def wait_ready(self, date: str, poll_s: float = 0.5) -> None:
        with self._cv:
            while True:
                if self._error is not None:
                    raise self._error
                if self._stopped:
                    raise RuntimeError("RollingBaseStore stopped while wait_ready")
                if self._is_ready_unlocked(date):
                    return
                self._cv.wait(timeout=poll_s)

    def advance_assemble(self, date: str) -> None:
        """组装完成：推进 trim 前沿（只跟组装日期走，不跟 prefetch 光标走）。"""
        with self._cv:
            keep_candidates: List[str] = []
            for delta in self.SCALES:
                window = get_last_days(date=date, N=self.lookback[delta])
                if window:
                    keep_candidates.append(window[0])
            if keep_candidates:
                self._assemble_keep_from = min(keep_candidates)
                for delta in self.SCALES:
                    self._trim(delta, keep_from=self._assemble_keep_from)
            self._cv.notify_all()

    def get_window(self, delta: str, date: str, pools: List[str]) -> pl.DataFrame:
        """
        只读 lookback 各日，且每日先滤 pools 再拼接。
        锁内拷贝小块；check / ffill 在锁外。
        """
        L = self.lookback[delta]
        window = get_last_days(date=date, N=L)
        if not window:
            return pl.DataFrame()

        parts: List[pl.DataFrame] = []
        with self._lock:
            day_map = self._days[delta]
            for day in window:
                day_df = day_map.get(day)
                if day_df is None or day_df.is_empty():
                    continue
                part = day_df.filter(pl.col("instrument").is_in(pools)).select(
                    ["date", "instrument", "date_day", *VALUES_COLS]
                )
                if not part.is_empty():
                    parts.append(part)

        if not parts:
            return pl.DataFrame()

        df = pl.concat(parts, how="vertical_relaxed")
        df = _check_base_frame(df, delta, window[0], date)
        if df.is_empty():
            return df
        return _apply_forward_fill(df)

    # ------------------------------------------------------------------
    # Base 预取线程 API
    # ------------------------------------------------------------------

    def wait_can_prefetch(self, prefetch_date: str, poll_s: float = 0.2) -> bool:
        max_ahead = int(getattr(self.profile, "prefetch_max_ahead_days", 0) or 0)
        if max_ahead <= 0:
            return not self._stopped

        with self._cv:
            while True:
                if self._stopped:
                    return False
                if self._error is not None:
                    raise self._error
                ad = self._assemble_date
                if ad is None:
                    return True
                if self._ahead_days(ad, prefetch_date) <= max_ahead:
                    return True
                self._cv.wait(timeout=poll_s)

    def prefetch_for(self, date: str) -> None:
        """补 lookback 缺口 + 水位超前；DAI 在锁外。"""
        for delta in self.SCALES:
            if self._stopped:
                return
            self._prefetch_scale(delta, date)

        with self._cv:
            if self._assemble_keep_from:
                for delta in self.SCALES:
                    self._trim(delta, keep_from=self._assemble_keep_from)
            self._cv.notify_all()

    def ensure(self, date: str, pools: Optional[List[str]] = None) -> None:
        """兼容旧单线程：先拉再等。"""
        self.prefetch_for(date)
        self.wait_ready(date)

    # ------------------------------------------------------------------
    # 就绪 / 超前
    # ------------------------------------------------------------------

    def _is_ready_unlocked(self, date: str) -> bool:
        for delta in self.SCALES:
            window = get_last_days(date=date, N=self.lookback[delta])
            fetched = self._fetched[delta]
            if any(d not in fetched for d in window):
                return False
        return True

    @staticmethod
    def _ahead_days(target: str, watermark: Optional[str]) -> int:
        if watermark is None or watermark < target:
            return 0
        days = trading_days_between(target, watermark)
        return max(0, len(days) - 1)

    # ------------------------------------------------------------------
    # 按尺度预取
    # ------------------------------------------------------------------

    def _prefetch_scale(self, delta: str, target_date: str) -> None:
        L = self.lookback[delta]
        window = get_last_days(date=target_date, N=L)
        if not window:
            return

        with self._lock:
            missing_days = [d for d in window if d not in self._fetched[delta]]

        if missing_days:
            for start, end in self._merge_day_ranges(missing_days):
                pull_end = self._extend_pull_end(start, end)
                self._pull_range(delta, start, pull_end)

        if self.profile.low_water_days > 0:
            with self._lock:
                ahead = self._ahead_days(target_date, self._watermark[delta])
                need = ahead < self.profile.low_water_days
                wm = self._watermark[delta]
            if need:
                pull_start = self._proactive_pull_start(target_date, wm)
                if pull_start is not None:
                    pull_end = self._chunk_end(pull_start)
                    if pull_end >= pull_start:
                        self._pull_range(delta, pull_start, pull_end)

    @staticmethod
    def _merge_day_ranges(days: List[str]) -> List[Tuple[str, str]]:
        if not days:
            return []
        ranges: List[Tuple[str, str]] = []
        start = prev = days[0]
        for d in days[1:]:
            between = trading_days_between(prev, d)
            if len(between) == 2:
                prev = d
            else:
                ranges.append((start, prev))
                start = prev = d
        ranges.append((start, prev))
        return ranges

    def _extend_pull_end(self, start: str, end: str) -> str:
        days = get_next_days(start, self.profile.fetch_chunk_days)
        cand = days[-1] if days else end
        cand = max(cand, end)
        if self.horizon_end:
            cand = min(cand, self.horizon_end)
        return cand

    def _chunk_end(self, pull_start: str) -> str:
        days = get_next_days(pull_start, self.profile.fetch_chunk_days)
        cand = days[-1] if days else pull_start
        if self.horizon_end:
            cand = min(cand, self.horizon_end)
        return cand

    def _proactive_pull_start(
        self, target_date: str, watermark: Optional[str]
    ) -> Optional[str]:
        if watermark is None or watermark < target_date:
            start = target_date
        else:
            start = next_trading_day_after(watermark)
            if start is None:
                return None
        if self.horizon_end and start > self.horizon_end:
            return None
        return start

    def _pull_range(self, delta: str, start: str, end: str) -> None:
        if start > end:
            return
        days = trading_days_between(start, end)
        if not days:
            return

        with self._lock:
            if all(d in self._fetched[delta] for d in days):
                return

        # ---- 锁外 DAI ----
        chunk = get_base_data(
            target_delta=delta,
            start_date=start,
            end_date=end,
            ScaleDetails=self.scale_details,
            pools=None,
            fill=False,
            check=False,
        )

        with self._cv:
            self._ingest(delta, chunk)
            self._fetched[delta].update(days)
            self._sync_watermark(delta)
            if self._assemble_keep_from:
                self._trim(delta, keep_from=self._assemble_keep_from)
            self._cv.notify_all()

    # ------------------------------------------------------------------
    # 按日存储：ingest / trim
    # ------------------------------------------------------------------

    def _ingest(self, delta: str, chunk: pl.DataFrame) -> None:
        if chunk is None or chunk.is_empty():
            return

        need = ["date", "instrument", "date_day", *VALUES_COLS]
        chunk = chunk.select([c for c in need if c in chunk.columns])
        if "date_day" not in chunk.columns:
            chunk = chunk.with_columns(
                pl.col("date").dt.date().cast(pl.String).alias("date_day")
            )
        else:
            chunk = chunk.with_columns(pl.col("date_day").cast(pl.String))

        day_map = self._days[delta]
        for day_df in chunk.partition_by("date_day", maintain_order=True):
            day = day_df["date_day"][0]
            if not isinstance(day, str):
                day = str(day)

            old = day_map.get(day)
            if old is None or old.is_empty():
                day_map[day] = day_df
            else:
                day_map[day] = (
                    pl.concat([old, day_df], how="diagonal_relaxed")
                    .unique(subset=["instrument", "date"], keep="last")
                    .sort("instrument", "date")
                )

    def _trim(self, delta: str, keep_from: str) -> None:
        day_map = self._days[delta]
        for d in [k for k in day_map.keys() if k < keep_from]:
            del day_map[d]
        self._fetched[delta] = {d for d in self._fetched[delta] if d >= keep_from}
        self._sync_watermark(delta)

    def _sync_watermark(self, delta: str) -> None:
        fetched = self._fetched[delta]
        self._watermark[delta] = max(fetched) if fetched else None
