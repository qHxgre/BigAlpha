"""独立读取 cpt_jyc_2026_stock_bar1m 的分钟行情质量检查。"""

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

    datasource_id = "cpt_jyc_2026_stock_bar1m"
    test_data: pd.DataFrame = pd.DataFrame()
    require_complete_day = False
    # 快照并不覆盖每一笔成交，分钟量额隐含均价可能比采样到的 OHLC
    # 多出一两个最小价位；保留 0.1% + 0.02 元容差，仍足以拦截累计值。
    price_rtol = 1e-3
    price_atol = 2e-2

    price_fields = ["open", "high", "low", "close"]
    trade_fields = ["volume", "amount", "deal_number"]
    order_count_fields = [f"{side}_num_orders{level}" for side in ("ask", "bid") for level in range(1, 6)]
    required_fields = ["date", "instrument", *price_fields, *trade_fields, *order_count_fields]

    @staticmethod
    def _continuous_minute_labels(day: pd.Timestamp) -> pd.DatetimeIndex:
        morning = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), day + pd.Timedelta(hours=11, minutes=30), freq="min")
        afternoon = pd.date_range(day + pd.Timedelta(hours=13, minutes=1), day + pd.Timedelta(hours=15), freq="min")
        return morning.append(afternoon)

    @classmethod
    def check(cls, data: pd.DataFrame, require_complete_day: bool = False) -> None:
        """运行全部检查；失败时汇总异常并抛错。"""
        if data.empty:
            raise StockBar1mDataTestError("待测试分钟行情为空")

        missing = sorted(set(cls.required_fields).difference(data.columns))
        if missing:
            raise StockBar1mDataTestError(f"分钟行情缺少字段: {missing}")

        cls.test_data = data.copy()
        cls.test_data["date"] = pd.to_datetime(cls.test_data["date"], errors="coerce")
        cls.require_complete_day = require_complete_day

        stream = io.StringIO()
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        if not result.wasSuccessful():
            issues = []
            for test_case, traceback in result.failures + result.errors:
                method_name = getattr(test_case, "_testMethodName", str(test_case))
                issues.append(f"{method_name}: {traceback.strip().splitlines()[-1]}")
            raise StockBar1mDataTestError("cpt_jyc_2026_stock_bar1m 数据测试失败:\n" + "\n".join(issues))

    def test_unique_keys_and_valid_dates(self):
        """日期必须有效，且 date、instrument 联合键不能重复。"""
        data = self.test_data
        self.assertEqual(int(data["date"].isna().sum()), 0, "存在非法日期")
        duplicated = data.duplicated(["date", "instrument"], keep=False)
        self.assertEqual(
            int(duplicated.sum()),
            0,
            f"存在重复分钟记录: {data.loc[duplicated].head().to_dict('records')}",
        )

    def test_numeric_values_are_finite(self):
        """核心价格、量额及委托笔数不能出现无穷值。"""
        columns = self.price_fields + self.trade_fields + self.order_count_fields
        numeric = self.test_data[columns].apply(pd.to_numeric, errors="coerce")
        invalid = np.isinf(numeric).any(axis=1)
        self.assertEqual(
            int(invalid.sum()),
            0,
            f"存在无穷值: {self.test_data.loc[invalid].head().to_dict('records')}",
        )

    def test_prices_are_valid_ohlc(self):
        """OHLC 必须为正，且满足 low <= open/close <= high。"""
        data = self.test_data
        prices = data[self.price_fields].apply(pd.to_numeric, errors="coerce")
        invalid_value = prices.isna().any(axis=1) | prices.le(0).any(axis=1)
        invalid_order = (
            prices["low"].gt(prices["high"])
            | prices["open"].lt(prices["low"])
            | prices["open"].gt(prices["high"])
            | prices["close"].lt(prices["low"])
            | prices["close"].gt(prices["high"])
        )
        invalid = invalid_value | invalid_order
        columns = ["date", "instrument", *self.price_fields]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "存在 -1、0、空值或 OHLC 次序异常: "
            f"{data.loc[invalid, columns].head(20).to_dict('records')}",
        )

    def test_trade_statistics_are_consistent(self):
        """量、额、笔数非负，且成交量与成交额应同时为零或同时为正。"""
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

    def test_implied_trade_price_is_inside_bar(self):
        """有成交时 amount / volume 应落在该分钟最低价和最高价之间。"""
        data = self.test_data
        valid = data["volume"].gt(0) & data["amount"].gt(0)
        implied = data["amount"] / data["volume"]
        lower = data["low"] * (1 - self.price_rtol) - self.price_atol
        upper = data["high"] * (1 + self.price_rtol) + self.price_atol
        invalid = valid & ~implied.between(lower, upper, inclusive="both")
        sample = data.loc[
            invalid,
            ["date", "instrument", "low", "high", "volume", "amount"],
        ].copy()
        sample["amount_div_volume"] = implied.loc[invalid]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "成交均价超出分钟价格区间，疑似混入累计量额: "
            f"{sample.head(20).to_dict('records')}",
        )

    def test_order_counts_are_nonnegative(self):
        """买卖五档委托笔数不能为负，负数通常表示上游整数溢出。"""
        data = self.test_data
        counts = data[self.order_count_fields].apply(pd.to_numeric, errors="coerce")
        invalid = counts.isna().any(axis=1) | counts.lt(0).any(axis=1)
        self.assertEqual(
            int(invalid.sum()),
            0,
            "存在负委托笔数或空值，疑似源字段溢出: "
            f"{data.loc[invalid, ['date', 'instrument', *self.order_count_fields]].head(20).to_dict('records')}",
        )

    def test_market_has_every_continuous_minute(self):
        """完整交易日的全市场数据不能集中缺少连续竞价分钟。"""
        if not self.require_complete_day:
            return
        data = self.test_data
        missing_records = []
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


class CptJyc2026StockBar1mTester:
    """按照中证1000交易日逐日读取并测试一分钟行情。"""

    def __init__(self, start_date: str, end_date: str, datasource_id: str = "cpt_jyc_2026_stock_bar1m") -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.datasource_id = datasource_id

    def run(self) -> pd.DataFrame:
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        if start > end:
            raise ValueError("start_date 不能晚于 end_date")

        calendar = dai.query(
            """
            SELECT DISTINCT CAST(date AS DATE) AS trading_day
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
            ORDER BY trading_day
            """,
            filters={"date": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]},
        ).df()
        if calendar.empty:
            raise StockBar1mDataTestError(f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d} 未查询到中证1000交易日")

        trading_days = pd.to_datetime(calendar["trading_day"]).dt.normalize()
        actual_start, actual_end = trading_days.min(), trading_days.max()
        logger.info("开始检查 %s: start_date=%s, end_date=%s, 交易日数=%s", self.datasource_id, actual_start.strftime("%Y-%m-%d"), actual_end.strftime("%Y-%m-%d"), len(trading_days))

        reports = []
        for day in trading_days:
            day_str = day.strftime("%Y-%m-%d")
            data = dai.query(f"SELECT * FROM {self.datasource_id}", filters={"date": [f"{day_str} 00:00:00", f"{day_str} 23:59:59"]}, compression=True).df()
            if data.empty:
                raise StockBar1mDataTestError(f"{day_str} 整日缺少分钟行情")
            _StockBar1mDataTests.check(data, require_complete_day=True)
            reports.append({"trading_day": day, "start_date": actual_start, "end_date": actual_end, "row_count": len(data), "stock_count": data["instrument"].nunique(), "passed": True})
            logger.info("%s: 检查通过，%s 行", day_str, len(data))

        report = pd.DataFrame(reports)
        logger.info("检查完成: start_date=%s, end_date=%s, 通过交易日数=%s", actual_start.strftime("%Y-%m-%d"), actual_end.strftime("%Y-%m-%d"), len(report))
        return report
