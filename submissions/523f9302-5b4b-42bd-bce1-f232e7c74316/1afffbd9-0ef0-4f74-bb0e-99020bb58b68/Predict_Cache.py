# -*- coding: utf-8 -*-
"""预测侧单线程、分块预取的热窗基础数据缓存。"""

import logging
import math
import time
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
    trading_days_between,
)

logger = logging.getLogger("pony_predict")


class PredictBaseCache:
    """预测侧多尺度滚动缓存。

    特点
    ----
    1. DAI 查询严格单线程、顺序执行；
    2. 支持多个预测日期分块预取；
    3. 使用 _fetched 按交易日防重复查询（不做按股覆盖率检查/重提）；
    4. 仅保留当前预测所需的滚动窗口；
    5. get_window 只读缓存；缺数由上层 usable 标记处理。
    """

    BARS_PER_DAY = {
        "1m": 240,
        "5m": 48,
        "15m": 16,
        "30m": 8,
    }
    SCALES = ("1m", "5m", "15m", "30m")

    def __init__(
        self,
        seq_len: int,
        scale_details: dict,
    ):
        self.seq_len = int(seq_len)
        self.scale_details = scale_details

        self.lookback: Dict[str, int] = {
            delta: math.ceil(
                self.seq_len / self.BARS_PER_DAY[delta]
            )
            for delta in self.SCALES
        }

        # scale -> {date_day -> 当日标准化数据}
        self._days: Dict[
            str, Dict[str, pl.DataFrame]
        ] = {
            delta: {}
            for delta in self.SCALES
        }

        # scale -> 已向 DAI 查询过的交易日集合
        self._fetched: Dict[str, Set[str]] = {
            delta: set()
            for delta in self.SCALES
        }

        self._dai_query_seq = 0
        self._dai_stats = self._empty_dai_stats()

    @staticmethod
    def _empty_dai_stats() -> Dict[str, dict]:
        return {
            delta: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "cache_hits": 0,
                "rows": 0,
                "seconds": 0.0,
            }
            for delta in PredictBaseCache.SCALES
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> None:
        for delta in self.SCALES:
            self._days[delta].clear()
            self._fetched[delta].clear()
        self._dai_query_seq = 0
        self._dai_stats = self._empty_dai_stats()

    def get_dai_stats(self) -> dict:
        """返回预测期间的 DAI bar 查询统计。"""
        return {
            delta: dict(values)
            for delta, values in self._dai_stats.items()
        }

    # ------------------------------------------------------------------
    # DAI 查询包装（计时 / 日志）
    # ------------------------------------------------------------------

    def _query_base_data(
        self,
        delta: str,
        start_date: str,
        end_date: str,
        pools: List[str],
        *,
        reason: str,
    ) -> pl.DataFrame:
        """统一包装 get_base_data，记录单次 DAI 耗时。"""
        self._dai_query_seq += 1
        query_id = f"BAR-{self._dai_query_seq:04d}"
        stats = self._dai_stats[delta]
        stats["attempts"] += 1

        # logger.debug(
        #     "  [DAI][BAR][BEGIN] id=%s reason=%s scale=%s "
        #     "range=%s~%s pools=%s",
        #     query_id, reason, delta, start_date, end_date, len(pools),
        # )

        t0 = time.perf_counter()
        try:
            chunk = get_base_data(
                target_delta=delta,
                start_date=start_date,
                end_date=end_date,
                ScaleDetails=self.scale_details,
                pools=pools,
                fill=False,
                check=False,
            )
        except Exception as exc:
            sec = time.perf_counter() - t0
            stats["failures"] += 1
            stats["seconds"] += sec
            err = (
                str(exc)
                .replace("\n", " ")
                .replace("\r", " ")
            )[:300]
            logger.error(
                "  [DAI][BAR][ERROR] id=%s reason=%s scale=%s "
                "range=%s~%s seconds=%.3f error_type=%s error=%s",
                query_id,
                reason,
                delta,
                start_date,
                end_date,
                sec,
                type(exc).__name__,
                err,
            )
            raise

        sec = time.perf_counter() - t0
        rows = 0 if chunk is None else int(chunk.height)
        codes = 0
        if (
            chunk is not None
            and not chunk.is_empty()
            and "instrument" in chunk.columns
        ):
            codes = int(chunk["instrument"].n_unique())

        stats["successes"] += 1
        stats["rows"] += rows
        stats["seconds"] += sec

        logger.info(
            "  [DAI][BAR][FINISH] id=%s reason=%s scale=%s "
            "range=%s~%s rows=%s returned_pools=%s seconds=%.3f",
            query_id,
            reason,
            delta,
            start_date,
            end_date,
            rows,
            codes,
            sec,
        )
        return chunk

    # ------------------------------------------------------------------
    # 分块预取
    # ------------------------------------------------------------------

    def prefetch_dates(
        self,
        dates: List[str],
        pools: List[str],
    ) -> None:
        """顺序预取一组预测日期所需的四尺度数据。

        每个尺度仅对尚未查询过的交易日发起 DAI；
        四个尺度严格串行。不做按股覆盖率补提。
        """
        dates = sorted(set(dates))
        pools = list(pools)

        if not dates or not pools:
            return

        for delta in self.SCALES:
            self._prefetch_delta(
                delta=delta,
                dates=dates,
                pools=pools,
            )

    def _prefetch_delta(
        self,
        delta: str,
        dates: List[str],
        pools: List[str],
    ) -> None:
        """预取单个尺度：只补尚未 _fetched 的交易日。"""
        required_days: Set[str] = set()

        for date in dates:
            required_days.update(
                get_last_days(
                    date=date,
                    N=self.lookback[delta],
                )
            )

        if not required_days:
            return

        required_start = min(required_days)
        required_end = max(required_days)

        queried_days = trading_days_between(
            required_start,
            required_end,
        )

        if not queried_days:
            return

        self._trim(
            delta=delta,
            keep_from=required_start,
        )

        missing_days = [
            day
            for day in queried_days
            if day not in self._fetched[delta]
        ]

        if not missing_days:
            self._dai_stats[delta]["cache_hits"] += 1
            logger.info(
                "  [DAI][BAR][CACHE-HIT] reason=prefetch scale=%s "
                "range=%s~%s days=%s requested_pools=%s",
                delta,
                required_start,
                required_end,
                len(queried_days),
                len(pools),
            )
            return

        for start, end in self._merge_day_ranges(
            missing_days
        ):
            days = trading_days_between(start, end)
            if not days:
                continue

            chunk = self._query_base_data(
                delta=delta,
                start_date=start,
                end_date=end,
                pools=pools,
                reason="prefetch",
            )

            self._ingest(
                delta=delta,
                chunk=chunk,
                days=days,
            )

    # ------------------------------------------------------------------
    # FeatureCal Store 接口
    # ------------------------------------------------------------------

    def wait_ready(self, date: str) -> None:
        """兼容 RollingBaseStore 接口。"""
        return

    def ensure_all(
        self,
        date: str,
        pools: List[str],
    ) -> None:
        """兼容入口：等价于对该日做一次 prefetch（不按股补提）。"""
        self.prefetch_dates(dates=[date], pools=pools)

    def get_window(
        self,
        delta: str,
        date: str,
        pools: List[str],
    ) -> pl.DataFrame:
        """取得指定日期和尺度的完整热窗（只读缓存，不触发 DAI）。"""
        if delta not in self.SCALES:
            raise KeyError(
                f"unsupported delta={delta!r}, "
                f"expect {self.SCALES}"
            )

        window = get_last_days(
            date=date,
            N=self.lookback[delta],
        )

        if not window:
            return pl.DataFrame()

        parts: List[pl.DataFrame] = []
        day_map = self._days[delta]

        for day in window:
            day_df = day_map.get(day)

            if day_df is None or day_df.is_empty():
                continue

            part = (
                day_df
                .filter(
                    pl.col("instrument").is_in(pools)
                )
                .select(
                    [
                        "date",
                        "instrument",
                        "date_day",
                        *VALUES_COLS,
                    ]
                )
            )

            if not part.is_empty():
                parts.append(part)

        if not parts:
            return pl.DataFrame()

        df = pl.concat(
            parts,
            how="vertical_relaxed",
        )

        df = _check_base_frame(
            df,
            delta,
            window[0],
            date,
        )

        if df.is_empty():
            return df

        return _apply_forward_fill(df)

    # ------------------------------------------------------------------
    # 数据写入
    # ------------------------------------------------------------------

    def _ingest(
        self,
        delta: str,
        chunk: Optional[pl.DataFrame],
        days: List[str],
    ) -> None:
        """写入完整日期范围，并标记这些交易日已查询。"""
        self._fetched[delta].update(days)

        for day in days:
            self._days[delta].setdefault(
                day, pl.DataFrame()
            )

        if chunk is None or chunk.is_empty():
            return

        self._merge_chunk(
            delta=delta,
            chunk=chunk,
        )

    def _merge_chunk(
        self,
        delta: str,
        chunk: pl.DataFrame,
    ) -> None:
        required_columns = [
            "date",
            "instrument",
            "date_day",
            *VALUES_COLS,
        ]

        chunk = chunk.select(
            [
                column
                for column in required_columns
                if column in chunk.columns
            ]
        )

        if "date_day" not in chunk.columns:
            chunk = chunk.with_columns(
                pl.col("date")
                .dt.date()
                .cast(pl.String)
                .alias("date_day")
            )
        else:
            chunk = chunk.with_columns(
                pl.col("date_day").cast(pl.String)
            )

        day_map = self._days[delta]

        for day_df in chunk.partition_by(
            "date_day",
            maintain_order=True,
        ):
            day = day_df["date_day"][0]

            if not isinstance(day, str):
                day = str(day)

            old = day_map.get(day)

            if old is None or old.is_empty():
                day_map[day] = day_df
                continue

            day_map[day] = (
                pl.concat(
                    [old, day_df],
                    how="diagonal_relaxed",
                )
                .unique(
                    subset=["instrument", "date"],
                    keep="last",
                )
                .sort(["instrument", "date"])
            )

    # ------------------------------------------------------------------
    # 缓存清理
    # ------------------------------------------------------------------

    def _trim(
        self,
        delta: str,
        keep_from: str,
    ) -> None:
        day_map = self._days[delta]

        expired_days = [
            day
            for day in day_map
            if day < keep_from
        ]

        for day in expired_days:
            del day_map[day]

        self._fetched[delta] = {
            day
            for day in self._fetched[delta]
            if day >= keep_from
        }

    # ------------------------------------------------------------------
    # 日期范围工具
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_day_ranges(
        days: List[str],
    ) -> List[Tuple[str, str]]:
        """将相邻交易日合并成连续范围。"""
        if not days:
            return []

        days = sorted(set(days))
        ranges: List[Tuple[str, str]] = []

        start = days[0]
        previous = days[0]

        for day in days[1:]:
            between = trading_days_between(
                previous,
                day,
            )

            if len(between) == 2:
                previous = day
                continue

            ranges.append(
                (start, previous)
            )
            start = day
            previous = day

        ranges.append(
            (start, previous)
        )

        return ranges
