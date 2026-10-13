"""cpt_gifp_2026_stock_bar1m 入库后数据完整性检查。"""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Optional
import unittest

import dai
import pandas as pd

from gifp_2026.cpt_gifp_2026_stock_bar1m.schema import (
    CptGifp2026StockBar1mSchema,
)


class StockBar1mDataTestError(AssertionError):
    """分钟行情未通过完整性检查。"""


@dataclass(frozen=True)
class _MonthRange:
    start: pd.Timestamp
    end: pd.Timestamp


class CptGifp2026StockBar1mTester(unittest.TestCase):
    """按月查询并检查沪深 300 成分股分钟行情。"""

    test_data: pd.DataFrame = pd.DataFrame()
    datasource_id = "cpt_gifp_2026_stock_bar1m"
    index_instrument = "000300.SH"
    min_stock_ratio = 0.90
    max_stock_ratio = 1.10
    numeric_fields = [
        field
        for field in CptGifp2026StockBar1mSchema.columns()
        if field not in {"date", "instrument"}
    ]

    @staticmethod
    def _validate_dates(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError("start_date 不能晚于 end_date")
        return start, end

    @staticmethod
    def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[_MonthRange]:
        ranges = []
        for period in pd.period_range(start, end, freq="M"):
            ranges.append(
                _MonthRange(
                    start=max(start, period.start_time.normalize()),
                    end=min(end, period.end_time.normalize()),
                )
            )
        return ranges

    @classmethod
    def _query_expected_stocks(cls, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        data = dai.query(
            f"""
            SELECT
                CAST(date AS DATE) AS trading_day,
                COUNT(DISTINCT member_code) AS expected_stock_count
            FROM cn_stock_index_component
            WHERE instrument = '{cls.index_instrument}'
            GROUP BY CAST(date AS DATE)
            ORDER BY trading_day
            """,
            filters={"date": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]},
        ).df()
        if not data.empty:
            data["trading_day"] = pd.to_datetime(data["trading_day"]).dt.normalize()
        return data

    @classmethod
    def _query_month_stats(
        cls,
        start: pd.Timestamp,
        end: pd.Timestamp,
        datasource_id: str,
    ) -> pd.DataFrame:
        non_null_counts = ",\n".join(
            f'COUNT("{field}") AS "non_null__{field}"'
            for field in cls.numeric_fields
        )
        data = dai.query(
            f"""
            SELECT
                CAST(date AS DATE) AS trading_day,
                COUNT(*) AS row_count,
                COUNT(DISTINCT instrument) AS stock_count,
                {non_null_counts}
            FROM {datasource_id}
            GROUP BY CAST(date AS DATE)
            ORDER BY trading_day
            """,
            filters={
                "date": [
                    f"{start:%Y-%m-%d} 00:00:00",
                    f"{end:%Y-%m-%d} 23:59:59",
                ]
            },
        ).df()
        if not data.empty:
            data["trading_day"] = pd.to_datetime(data["trading_day"]).dt.normalize()
        return data

    @classmethod
    def run_date_range_tests(
        cls,
        start_date: str,
        end_date: str,
        datasource_id: Optional[str] = None,
        min_stock_ratio: float = 0.90,
        max_stock_ratio: float = 1.10,
    ) -> pd.DataFrame:
        """按月查询数据、运行单元测试并返回逐交易日汇总。

        检查项：
        1. 沪深 300 成分数据存在的每个交易日都有分钟行情；
        2. 实际股票数占当日成分股数的比例在给定范围内；
        3. schema 中每个数值字段在当日都至少有一个非空值。
        """
        start, end = cls._validate_dates(start_date, end_date)
        if not 0 < min_stock_ratio <= max_stock_ratio:
            raise ValueError("股票数量比例参数不合法")

        source = datasource_id or cls.datasource_id
        expected = cls._query_expected_stocks(start, end)
        if expected.empty:
            raise StockBar1mDataTestError(
                f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d} 未查询到沪深300交易日"
            )

        month_stats = []
        for month_range in cls._month_ranges(start, end):
            print(
                f"检查 {source}: {month_range.start:%Y-%m-%d} "
                f"至 {month_range.end:%Y-%m-%d}"
            )
            month_stats.append(
                cls._query_month_stats(month_range.start, month_range.end, source)
            )

        actual = pd.concat(month_stats, ignore_index=True) if month_stats else pd.DataFrame()
        report = expected.merge(actual, how="left", on="trading_day", validate="one_to_one")
        report["day_exists"] = report["row_count"].fillna(0).gt(0)
        report["stock_ratio"] = (
            report["stock_count"] / report["expected_stock_count"]
        )

        non_null_columns = [f"non_null__{field}" for field in cls.numeric_fields]
        for column in non_null_columns:
            if column not in report:
                report[column] = 0
        report["empty_numeric_fields"] = report[non_null_columns].apply(
            lambda row: [
                column.removeprefix("non_null__")
                for column, count in row.items()
                if pd.isna(count) or count <= 0
            ],
            axis=1,
        )

        cls.min_stock_ratio = min_stock_ratio
        cls.max_stock_ratio = max_stock_ratio
        cls.run_tests(report)
        report["passed"] = True

        print(f"检查通过：{len(report)} 个交易日全部正常")
        return report[
            [
                "trading_day",
                "row_count",
                "stock_count",
                "expected_stock_count",
                "stock_ratio",
                "passed",
            ]
        ]

    @classmethod
    def run_tests(cls, data: pd.DataFrame) -> bool:
        """运行全部单元测试，任何失败都会汇总后抛出异常。"""
        if data.empty:
            raise StockBar1mDataTestError("待测试分钟行情统计为空")

        cls.test_data = data.copy()
        cls.test_data["trading_day"] = pd.to_datetime(
            cls.test_data["trading_day"], errors="coerce"
        )

        stream = io.StringIO()
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        if not result.wasSuccessful():
            issues = []
            for test_case, traceback in result.failures + result.errors:
                method_name = getattr(test_case, "_testMethodName", str(test_case))
                detail = traceback.strip().splitlines()[-1]
                issues.append(f"{method_name}: {detail}")
            raise StockBar1mDataTestError(
                "cpt_gifp_2026_stock_bar1m 数据测试失败:\n" + "\n".join(issues)
            )
        return True

    def test_every_trading_day_exists(self):
        """沪深 300 成分数据存在的每个交易日都必须有分钟行情。"""
        data = self.test_data
        invalid = ~data["day_exists"]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "交易日缺少分钟行情: "
            f"{data.loc[invalid, 'trading_day'].dt.strftime('%Y-%m-%d').tolist()}",
        )

    def test_daily_stock_count_is_reasonable(self):
        """每日股票数应接近当日沪深 300 成分股数量。"""
        data = self.test_data
        valid_day = data["day_exists"]
        invalid = valid_day & ~data["stock_ratio"].between(
            self.min_stock_ratio,
            self.max_stock_ratio,
            inclusive="both",
        )
        columns = [
            "trading_day",
            "stock_count",
            "expected_stock_count",
            "stock_ratio",
        ]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "每日股票数量不合理: "
            f"{data.loc[invalid, columns].head(20).to_dict('records')}",
        )

    def test_numeric_fields_are_not_all_null(self):
        """每个数值字段在每个有数据的交易日至少存在一个非空值。"""
        data = self.test_data
        invalid = data["day_exists"] & data["empty_numeric_fields"].map(bool)
        columns = ["trading_day", "empty_numeric_fields"]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "交易日存在全空数值字段: "
            f"{data.loc[invalid, columns].head(20).to_dict('records')}",
        )
