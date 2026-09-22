import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jyc_eval import run
from jyc_eval.datachecker import DataCheck, DataValidationError


class JycEvalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        section_times = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
        self.times = pd.DatetimeIndex(
            [pd.Timestamp(f"2026-01-05 {t}") for t in section_times]
            + [pd.Timestamp(f"2026-01-06 {t}") for t in section_times]
        )
        self.instruments = [f"{i:06d}.SZ" for i in range(30)]
        factor_rows = []
        evaluation_rows = []
        exposure_rows = []
        for time_index, dt in enumerate(self.times):
            for instrument_index, instrument in enumerate(self.instruments):
                factor = instrument_index + rng.normal(0, 0.2) + time_index * 0.01
                forward_return = factor * 0.001 + rng.normal(0, 0.003)
                factor_rows.append((dt, instrument, factor))
                evaluation_rows.append(
                    (
                        dt,
                        instrument,
                        forward_return,
                        "stress" if time_index < 6 else None,
                    )
                )
                exposure_rows.append((dt.normalize(), instrument, rng.normal(), rng.normal()))

        self.factor_data = pd.DataFrame(
            factor_rows, columns=["date", "instrument", "factor"]
        )
        self.evaluation_data = pd.DataFrame(
            evaluation_rows,
            columns=["date", "instrument", "forward_return", "scenario"],
        )
        # JYC 修改：测试日频 BARRA 暴露可映射到同一天的所有分钟截面。
        self.exposure_data = pd.DataFrame(
            exposure_rows, columns=["date", "instrument", "SIZE", "BETA"]
        ).drop_duplicates(["date", "instrument"])

    def evaluate(self, factor_data):
        pool_pairs = self.evaluation_data[["date", "instrument"]]
        with patch("jyc_eval.datachecker.load_pool_pairs", return_value=pool_pairs), patch(
            "jyc_eval.factoranalyze.analyzer.load_evaluation_data",
            return_value=self.evaluation_data,
        ), patch(
            "jyc_eval.dataprocess.get_exposure", return_value=self.exposure_data
        ):
            return run(factor_data, "2026-01-05", "2026-01-06", False)

    def test_run_returns_single_factor_analysis(self):
        result = self.evaluate(self.factor_data)
        self.assertEqual(
            set(result), {"raw_factor", "process_factor", "factor_analyze"}
        )
        self.assertEqual(
            set(result["factor_analyze"]),
            {"ic_mean", "ic_ir", "sharpe_ratio", "stress_stability", "turnover"},
        )
        self.assertTrue(
            all(np.isfinite(value) for value in result["factor_analyze"].values())
        )

    def test_score_column_is_rejected(self):
        score_data = self.factor_data.rename(columns={"factor": "score"})
        with self.assertRaisesRegex(DataValidationError, "必须且只能包含"):
            self.evaluate(score_data)

    def test_more_than_40_percent_missing_is_invalid(self):
        one_time = self.factor_data[self.factor_data["date"] == self.times[0]].copy()
        one_time.loc[one_time.index[:13], "factor"] = np.nan
        pool_pairs = one_time[["date", "instrument"]]
        with patch("jyc_eval.datachecker.load_pool_pairs", return_value=pool_pairs):
            with self.assertRaises(DataValidationError):
                DataCheck(self.times[0], self.times[0]).validate(one_time)

    def test_illegal_key_is_rejected(self):
        invalid = self.factor_data.copy()
        invalid.loc[0, "date"] = pd.Timestamp("2030-01-01 10:00:00")
        with self.assertRaisesRegex(ValueError, "非法采样时点"):
            self.evaluate(invalid)

    def test_invalid_numeric_factor_is_rejected(self):
        invalid = self.factor_data.astype({"factor": object})
        invalid.loc[0, "factor"] = "not-a-number"
        with self.assertRaisesRegex(DataValidationError, "无法转换为数值"):
            self.evaluate(invalid)

    def test_missing_whole_section_fails_coverage(self):
        incomplete = self.factor_data[self.factor_data["date"] != self.times[0]]
        with self.assertRaisesRegex(DataValidationError, "截面因子缺失率"):
            self.evaluate(incomplete)


if __name__ == "__main__":
    unittest.main()
