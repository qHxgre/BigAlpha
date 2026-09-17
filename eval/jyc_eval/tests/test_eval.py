import unittest

import numpy as np
import pandas as pd

from jyc_eval import run
from jyc_eval.datachecker import DataCheck, DataValidationError


class JycEvalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.times = pd.date_range("2026-01-05 10:00:00", periods=12, freq="30min")
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
            factor_rows, columns=["datetime", "instrument", "factor"]
        )
        self.evaluation_data = pd.DataFrame(
            evaluation_rows,
            columns=["datetime", "instrument", "forward_return", "scenario"],
        )
        # JYC 修改：测试日频 BARRA 暴露可映射到同一天的所有分钟截面。
        self.exposure_data = pd.DataFrame(
            exposure_rows, columns=["date", "instrument", "SIZE", "BETA"]
        ).drop_duplicates(["date", "instrument"])

    def test_run_returns_single_factor_analysis(self):
        result = run(
            self.factor_data,
            evaluation_data=self.evaluation_data,
            exposure_data=self.exposure_data,
            group_number=5,
        )
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

    def test_score_column_is_accepted(self):
        score_data = self.factor_data.rename(columns={"factor": "score"})
        result = run(
            score_data,
            evaluation_data=self.evaluation_data,
            exposure_data=self.exposure_data,
            group_number=5,
        )
        self.assertIn("factor_analyze", result)
        self.assertEqual(list(result["raw_factor"].columns), ["datetime", "instrument", "factor"])

    def test_more_than_40_percent_missing_is_invalid(self):
        one_time = self.factor_data[self.factor_data["datetime"] == self.times[0]].copy()
        one_time.loc[one_time.index[:13], "factor"] = np.nan
        with self.assertRaises(DataValidationError):
            DataCheck(self.times[0], self.times[0]).validate(one_time)

    def test_illegal_key_is_rejected(self):
        invalid = self.factor_data.copy()
        invalid.loc[0, "datetime"] = pd.Timestamp("2030-01-01 10:00:00")
        with self.assertRaisesRegex(ValueError, "非法采样时点"):
            run(
                invalid,
                evaluation_data=self.evaluation_data,
                exposure_data=self.exposure_data,
                group_number=5,
            )


if __name__ == "__main__":
    unittest.main()
