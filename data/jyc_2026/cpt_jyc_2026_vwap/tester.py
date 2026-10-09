"""cpt_jyc_2026_vwap 单元测试。"""

from __future__ import annotations

import io
import unittest

import numpy as np
import pandas as pd


class VwapDataTestError(AssertionError):
    """VWAP 数据未通过入库前单元测试。"""


class CptJyc2026VwapTester(unittest.TestCase):
    """未来 30 分钟 VWAP 标签测试。"""

    test_data: pd.DataFrame = pd.DataFrame()
    index_instruments = {"000852.SH"}
    formula_rtol = 1e-6
    # float32 执行 ``end_price / vwap - 1`` 时会产生约 1.2e-7 的
    # 消减误差，绝对误差阈值略高于 float32 epsilon。
    formula_atol = 2e-7

    @classmethod
    def run_tests(cls, data: pd.DataFrame) -> bool:
        """运行全部测试；任何失败都会抛错并阻止构建器入库。"""
        if data.empty:
            raise VwapDataTestError("待入库 VWAP 数据为空")

        cls.test_data = data.copy()
        if "date" in cls.test_data:
            cls.test_data["date"] = pd.to_datetime(
                cls.test_data["date"], errors="coerce"
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
            raise VwapDataTestError(
                "cpt_jyc_2026_vwap 入库前测试失败:\n" + "\n".join(issues)
            )
        return True

    def test_nonnegative_trade_statistics(self):
        """窗口成交量、成交额和成交笔数不能为负。"""
        data = self.test_data
        invalid = data[["volume", "amount", "num_trades"]].lt(0).any(axis=1)
        self.assertEqual(
            int(invalid.sum()),
            0,
            "存在负数成交统计: "
            f"{data.loc[invalid].head().to_dict('records')}",
        )

    def test_no_infinite_values(self):
        """核心数值字段不能出现正负无穷。"""
        columns = ["vwap_return", "vwap", "end_price", "volume", "amount"]
        numeric = self.test_data[columns].apply(pd.to_numeric, errors="coerce")
        invalid = np.isinf(numeric).any(axis=1)
        self.assertEqual(
            int(invalid.sum()),
            0,
            f"存在无穷值: {self.test_data.loc[invalid].head().to_dict('records')}",
        )

    def test_vwap_price_is_reasonable(self):
        """重点拦截累计成交额重复计入导致的数量级异常 VWAP。"""
        data = self.test_data
        valid = data["vwap"].gt(0) & data["end_price"].gt(0)
        ratio = data["vwap"] / data["end_price"]
        invalid = valid & ~ratio.between(0.5, 2.0)
        sample_columns = [
            "date",
            "instrument",
            "vwap",
            "end_price",
            "volume",
            "amount",
            "vwap_return",
        ]
        self.assertEqual(
            int(invalid.sum()),
            0,
            "VWAP 与终点价格相差超过两倍，可能重复计入累计成交额: "
            f"{data.loc[invalid, sample_columns].head(10).to_dict('records')}",
        )

    def test_vwap_formulas(self):
        """收益公式正确；股票 VWAP 等于窗口成交额除以成交量。"""
        data = self.test_data
        valid_price = data["vwap"].gt(0) & data["end_price"].gt(0)
        expected_return = data["end_price"] / data["vwap"] - 1.0
        return_matches = np.isclose(
            data["vwap_return"],
            expected_return,
            rtol=self.formula_rtol,
            atol=self.formula_atol,
            equal_nan=False,
        )
        invalid_return = valid_price & ~return_matches
        self.assertEqual(
            int(invalid_return.sum()),
            0,
            "vwap_return 公式不正确: "
            f"{data.loc[invalid_return].head().to_dict('records')}",
        )

        is_stock = ~data["instrument"].isin(self.index_instruments)
        valid_stock = is_stock & data["volume"].gt(0) & data["amount"].notna()
        expected_vwap = data["amount"] / data["volume"]
        vwap_matches = np.isclose(
            data["vwap"],
            expected_vwap,
            rtol=self.formula_rtol,
            atol=self.formula_atol,
            equal_nan=False,
        )
        invalid_vwap = valid_stock & ~vwap_matches
        self.assertEqual(
            int(invalid_vwap.sum()),
            0,
            "股票 VWAP 不等于 amount / volume: "
            f"{data.loc[invalid_vwap].head().to_dict('records')}",
        )
