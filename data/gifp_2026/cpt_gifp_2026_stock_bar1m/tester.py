"""独立读取 cpt_gifp_2026_stock_bar1m 的分钟行情质量检查。"""

from __future__ import annotations

import io
import logging
import unittest

import dai
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StockBar1mDataTestError(AssertionError):
    """分钟行情未通过数据质量检查。"""


class _StockBar1mDataTests(unittest.TestCase):
    """一分钟行情的数据质量测试用例。"""

    test_data: pd.DataFrame = pd.DataFrame()
    require_complete_day = False
    snapshot_totals: pd.DataFrame | None = None
    total_rtol = 1e-2
    price_rtol = 1e-3
    price_atol = 2e-2

    price_fields = ["open", "high", "low", "close"]
    trade_fields = ["volume", "amount", "deal_number"]
    order_count_fields = [
        f"{side}_num_orders{level}"
        for side in ("ask", "bid")
        for level in range(1, 11)
    ]
    required_fields = [
        "date",
        "instrument",
        *price_fields,
        *trade_fields,
        *order_count_fields,
    ]

    @staticmethod
    def _continuous_minute_labels(day: pd.Timestamp) -> pd.DatetimeIndex:
        morning = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=31),
            day + pd.Timedelta(hours=11, minutes=30),
            freq="min",
        )
        afternoon = pd.date_range(
            day + pd.Timedelta(hours=13, minutes=1),
            day + pd.Timedelta(hours=15),
            freq="min",
        )
        return morning.append(afternoon)

    @classmethod
    def check(
        cls,
        data: pd.DataFrame,
        require_complete_day: bool = False,
        snapshot_totals: pd.DataFrame | None = None,
    ) -> None:
        """运行全部检查；失败时汇总异常并抛错。"""
        if data.empty:
            raise StockBar1mDataTestError("待测试分钟行情为空")

        missing = sorted(set(cls.required_fields).difference(data.columns))
        if missing:
            raise StockBar1mDataTestError(f"分钟行情缺少字段: {missing}")

        cls.test_data = data.copy()
        cls.test_data["date"] = pd.to_datetime(
            cls.test_data["date"], errors="coerce"
        )
        cls.require_complete_day = require_complete_day
        cls.snapshot_totals = snapshot_totals

        stream = io.StringIO()
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        if not result.wasSuccessful():
            issues = []
            for test_case, traceback in result.failures + result.errors:
                method_name = getattr(
                    test_case, "_testMethodName", str(test_case)
                )
                issues.append(
                    f"{method_name}: {traceback.strip().splitlines()[-1]}"
                )
            raise StockBar1mDataTestError(
                "cpt_gifp_2026_stock_bar1m 数据测试失败:\n"
                + "\n".join(issues)
            )

    def test_unique_keys_and_valid_dates(self):
        data = self.test_data
        self.assertEqual(int(data["date"].isna().sum()), 0, "存在非法日期")
        duplicated = data.duplicated(["date", "instrument"], keep=False)
        self.assertEqual(
            int(duplicated.sum()),
            0,
            f"存在重复分钟记录: {data.loc[duplicated].head().to_dict('records')}",
        )

    def test_prices_are_valid_ohlc(self):
        data = self.test_data
        prices = data[self.price_fields].apply(pd.to_numeric, errors="coerce")
        invalid_value = prices.isna().any(axis=1) | prices.le(0).any(axis=1)
        tolerance = np.maximum(self.price_atol, self.price_rtol * prices["high"])
        lower = prices["low"] - tolerance
        upper = prices["high"] + tolerance
        invalid_order = (
            prices["low"].gt(prices["high"])
            | prices["open"].lt(lower)
            | prices["open"].gt(upper)
            | prices["close"].lt(lower)
            | prices["close"].gt(upper)
        )
        invalid = invalid_value | invalid_order
        columns = ["date", "instrument", *self.price_fields]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "存在非正价格、空值或 OHLC 次序异常: "
            f"{data.loc[invalid, columns].head(20).to_dict('records')}",
        )

    def test_trade_statistics_are_consistent(self):
        data = self.test_data
        stats = data[self.trade_fields].apply(pd.to_numeric, errors="coerce")
        negative = stats.lt(0).any(axis=1)
        mismatched = stats["volume"].gt(0) ^ stats["amount"].gt(0)
        invalid = stats.isna().any(axis=1) | negative | mismatched
        columns = ["date", "instrument", *self.trade_fields]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "存在负成交统计或量额一正一零: "
            f"{data.loc[invalid, columns].head(20).to_dict('records')}",
        )

    def test_daily_totals_match_snapshot(self):
        snap = self.snapshot_totals
        if snap is None:
            return
        minute = self.test_data.groupby("instrument")[["volume", "amount"]].sum()
        minute = minute.reindex(snap.index.union(minute.index)).fillna(0)
        snap = snap.reindex(minute.index).fillna(0)
        difference = (minute - snap).abs()
        invalid = (difference > self.total_rtol * snap).any(axis=1)
        sample = minute.join(
            snap, lsuffix="_minute", rsuffix="_snapshot"
        ).loc[invalid]
        self.assertEqual(
            int(invalid.sum()),
            0,
            f"{int(invalid.sum())} 只股票分钟量额之和与快照累计最大值相差超过 "
            f"{self.total_rtol:.1%}: {sample.head(20).reset_index().to_dict('records')}",
        )

    def test_order_counts_are_nonnegative(self):
        data = self.test_data
        counts = data[self.order_count_fields].apply(
            pd.to_numeric, errors="coerce"
        )
        invalid = counts.isna().any(axis=1) | counts.lt(0).any(axis=1)
        self.assertEqual(
            int(invalid.sum()),
            0,
            "十档委托笔数存在负数或空值: "
            f"{data.loc[invalid, ['date', 'instrument', *self.order_count_fields]].head(20).to_dict('records')}",
        )

    def test_market_has_every_continuous_minute(self):
        if not self.require_complete_day:
            return
        missing_records = []
        data = self.test_data
        for day, daily in data.groupby(data["date"].dt.normalize()):
            actual = pd.DatetimeIndex(daily["date"].dropna().unique())
            missing = self._continuous_minute_labels(day).difference(actual)
            if len(missing):
                missing_records.append(
                    {
                        "trading_day": day.strftime("%Y-%m-%d"),
                        "missing_count": len(missing),
                        "missing_minutes": missing.strftime("%H:%M").tolist()[:20],
                    }
                )
        self.assertEqual(
            len(missing_records),
            0,
            f"全市场集中缺分钟: {missing_records[:20]}",
        )


class CptGifp2026StockBar1mTester:
    """按照沪深300交易日逐日读取并测试一分钟行情。"""

    index_instrument = "000300.SH"

    def __init__(
        self,
        start_date: str,
        end_date: str,
        datasource_id: str = "cpt_gifp_2026_stock_bar1m",
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.datasource_id = datasource_id

    def check_day(
        self, day_str: str, instruments: list[str] | None = None
    ) -> pd.DataFrame:
        day_filter = {
            "date": [f"{day_str} 00:00:00", f"{day_str} 23:59:59"]
        }
        data = dai.query(
            f"SELECT * FROM {self.datasource_id}",
            filters=day_filter,
            compression=True,
        ).df()
        if data.empty:
            raise StockBar1mDataTestError(f"{day_str} 整日缺少分钟行情")

        instruments = (
            instruments
            if instruments is not None
            else data["instrument"].unique().tolist()
        )
        snapshot_totals = dai.query(
            """
            SELECT instrument, max(volume) AS volume, max(amount) AS amount
            FROM cn_stock_level2_snapshot
            GROUP BY instrument
            """,
            filters={**day_filter, "instrument": instruments},
        ).df()
        if snapshot_totals.empty:
            raise StockBar1mDataTestError(f"{day_str} 未查询到原始快照累计值")
        snapshot_totals = snapshot_totals.set_index("instrument")

        _StockBar1mDataTests.check(
            data,
            require_complete_day=True,
            snapshot_totals=snapshot_totals,
        )
        return data

    def run(self) -> pd.DataFrame:
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        if start > end:
            raise ValueError("start_date 不能晚于 end_date")

        calendar = dai.query(
            f"""
            SELECT DISTINCT CAST(date AS DATE) AS trading_day
            FROM cn_stock_index_component
            WHERE instrument = '{self.index_instrument}'
            ORDER BY trading_day
            """,
            filters={"date": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]},
        ).df()
        if calendar.empty:
            raise StockBar1mDataTestError(
                f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d} 未查询到沪深300交易日"
            )

        reports = []
        trading_days = pd.to_datetime(calendar["trading_day"]).dt.normalize()
        for day in trading_days:
            day_str = day.strftime("%Y-%m-%d")
            instruments = dai.query(
                f"""
                SELECT member_code AS instrument
                FROM cn_stock_index_component
                WHERE instrument = '{self.index_instrument}'
                """,
                filters={"date": [day_str, day_str]},
            ).df()["instrument"].dropna().unique().tolist()
            data = self.check_day(day_str, instruments)
            reports.append(
                {
                    "trading_day": day,
                    "row_count": len(data),
                    "stock_count": data["instrument"].nunique(),
                    "passed": True,
                }
            )
            logger.info("%s: 检查通过，%s 行", day_str, len(data))
        return pd.DataFrame(reports)
