"""cpt_gifp_2026_return 入库前数据校验。"""

from __future__ import annotations

import io
import unittest

import numpy as np
import pandas as pd


class ReturnDataTestError(AssertionError):
    """收益标签未通过入库前校验。"""


class CptGifp2026ReturnTester(unittest.TestCase):
    test_data: pd.DataFrame = pd.DataFrame()
    sample_minutes = {
        9 * 60 + 45,
        10 * 60,
        10 * 60 + 15,
        10 * 60 + 30,
        10 * 60 + 45,
        11 * 60,
        11 * 60 + 15,
        11 * 60 + 30,
        13 * 60 + 15,
        13 * 60 + 30,
        13 * 60 + 45,
        14 * 60,
        14 * 60 + 15,
        14 * 60 + 30,
        14 * 60 + 45,
        15 * 60,
    }

    @classmethod
    def run_tests(cls, data: pd.DataFrame) -> bool:
        if data.empty:
            raise ReturnDataTestError("待入库收益标签为空")
        cls.test_data = data.copy()
        for column in ("date", "future_date"):
            cls.test_data[column] = pd.to_datetime(cls.test_data[column])

        stream = io.StringIO()
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        if not result.wasSuccessful():
            issues = []
            for test_case, traceback in result.failures + result.errors:
                name = getattr(test_case, "_testMethodName", str(test_case))
                issues.append(f"{name}: {traceback.strip().splitlines()[-1]}")
            raise ReturnDataTestError(
                "cpt_gifp_2026_return 入库前测试失败:\n" + "\n".join(issues)
            )
        return True

    def test_unique_keys(self):
        duplicated = self.test_data.duplicated(["date", "instrument"])
        self.assertEqual(int(duplicated.sum()), 0, "存在重复 date/instrument")

    def test_sample_times(self):
        dates = self.test_data["date"]
        minutes = dates.dt.hour * 60 + dates.dt.minute
        invalid = ~minutes.isin(self.sample_minutes) | dates.dt.second.ne(0)
        self.assertEqual(
            int(invalid.sum()),
            0,
            f"存在非法15分钟采样点: {dates.loc[invalid].head().tolist()}",
        )

    def test_same_clock_time_on_next_trading_day(self):
        data = self.test_data
        same_time = data["date"].dt.time == data["future_date"].dt.time
        later_day = data["future_date"].dt.normalize() > data["date"].dt.normalize()
        invalid = ~(same_time & later_day)
        self.assertEqual(
            int(invalid.sum()),
            0,
            "终点必须是下一交易日的同一分钟",
        )

    def test_return_formula(self):
        data = self.test_data
        expected = (
            data["future_close"]
            * data["future_adjust_factor"]
            / (data["close"] * data["adjust_factor"])
            - 1.0
        )
        matches = np.isclose(
            data["return_rate"], expected, rtol=1e-6, atol=2e-7, equal_nan=False
        )
        self.assertEqual(
            int((~matches).sum()),
            0,
            f"收益公式不正确: {data.loc[~matches].head().to_dict('records')}",
        )

    def test_prices_and_factors_are_positive(self):
        columns = [
            "close",
            "adjust_factor",
            "future_close",
            "future_adjust_factor",
        ]
        invalid = self.test_data[columns].le(0).any(axis=1)
        self.assertEqual(int(invalid.sum()), 0, "价格或复权因子非正")

    def test_no_infinite_values(self):
        numeric = self.test_data[
            [
                "return_rate",
                "close",
                "adjust_factor",
                "future_close",
                "future_adjust_factor",
            ]
        ]
        self.assertEqual(int(np.isinf(numeric).any(axis=1).sum()), 0, "存在无穷值")
