import numpy as np
import pandas as pd
import structlog

from .data import load_pool_pairs
logger = structlog.get_logger()

KEY_COLUMNS = ("date", "instrument")
MAX_MISSING_RATE = 0.4


class DataValidationError(ValueError):
    """提交数据不符合评估口径。"""


class DataCheck:
    """加载官方评估面板，并按独立测试点校验提交数据。"""

    def __init__(self, start_date=None, end_date=None) -> None:
        self.window_start = start_date
        self.window_end = end_date
        self.pool_pairs = load_pool_pairs(start_date, end_date)

    def _fail(self, message: str, **details) -> None:
        logger.error(message, **details)
        suffix = f"；详情={details}" if details else ""
        raise DataValidationError(message + suffix)

    def check_columns(self, df: pd.DataFrame) -> None:
        """测试点：输出必须且只能包含标准三列。"""
        required = {*KEY_COLUMNS, "factor"}
        actual = set(df.columns)
        if actual != required or len(df.columns) != len(required):
            self._fail(
                "输出必须且只能包含 date、instrument、factor",
                missing=sorted(required - actual),
                extra=sorted(actual - required),
                columns=list(df.columns),
            )

    def check_date_window(self, df: pd.DataFrame) -> None:
        """测试点：提交数据的日期范围不得超出评估窗口。"""
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().any():
            sample = df.loc[dates.isna(), ["date"]].head(20)
            self._fail("date 包含无法转换为日期的值", sample=sample.to_dict("records"))

        actual_start = dates.min().normalize()
        actual_end = dates.max().normalize()
        window_start = pd.to_datetime(self.window_start).normalize()
        window_end = pd.to_datetime(self.window_end).normalize()
        if actual_start < window_start or actual_end > window_end:
            self._fail(
                "factor_data 的时间范围不得超出评估窗口",
                actual_start=str(actual_start),
                actual_end=str(actual_end),
                window_start=str(window_start),
                window_end=str(window_end),
            )

    def check_uniqueness(self, df: pd.DataFrame) -> None:
        """测试点：(date, instrument) 必须唯一。"""
        duplicated = df.duplicated(list(KEY_COLUMNS), keep=False)
        if duplicated.any():
            sample = df.loc[duplicated, list(KEY_COLUMNS)].head(20)
            self._fail("存在重复 (date, instrument)", sample=sample.to_dict("records"))

    def check_factor_numeric(self, df: pd.DataFrame) -> pd.Series:
        """测试点：factor 必须可转为数值，且不允许正负无穷。"""
        original_missing = df["factor"].isna()
        numeric = pd.to_numeric(df["factor"], errors="coerce")
        invalid = numeric.isna() & ~original_missing
        if invalid.any():
            sample = df.loc[invalid, list(KEY_COLUMNS) + ["factor"]].head(20)
            self._fail("factor 包含无法转换为数值的值", sample=sample.to_dict("records"))
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            self._fail("factor 不允许出现正负无穷")
        return numeric

    def check_official_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验提交记录合法性，并返回对齐官方面板后的数据。"""
        submitted_keys = df[list(KEY_COLUMNS)].drop_duplicates()
        key_check = submitted_keys.merge(
            self.pool_pairs.assign(__official=True),
            on=list(KEY_COLUMNS),
            how="left",
        )
        illegal = key_check["__official"].isna()
        if illegal.any():
            sample = key_check.loc[illegal, list(KEY_COLUMNS)].head(20).to_dict("records")
            self._fail("存在非法采样时点或非股票池记录", sample=sample)
        return self._align_to_official_panel(df)

    def _align_to_official_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """将合法提交对齐到官方面板，缺少的股票行以 NaN 补齐。"""
        return (
            self.pool_pairs.merge(
                df, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
            )
            .sort_values(list(KEY_COLUMNS))
            .reset_index(drop=True)
        )

    def check_factor_coverage(self, df: pd.DataFrame) -> None:
        """测试点：合并后的每个截面因子缺失率不得高于阈值。"""
        numeric = pd.to_numeric(df["factor"], errors="coerce")
        miss_rate = numeric.isna().groupby(df["date"]).mean()
        bad = miss_rate[miss_rate > MAX_MISSING_RATE]
        if not bad.empty:
            self._fail(
                f"截面因子缺失率不得高于 {MAX_MISSING_RATE:.0%}",
                sample={str(k): float(v) for k, v in bad.head(20).items()},
            )

    def validate(self, factor_data: pd.DataFrame) -> None:
        """统一校验入口；任一测试点不通过时抛出 DataValidationError。"""
        self.check_columns(factor_data)
        self.check_date_window(factor_data)

        df = factor_data[[*KEY_COLUMNS, "factor"]].copy()
        self.check_uniqueness(df)
        self.check_factor_numeric(df)
        merged_df = self.check_official_panel(df)
        self.check_factor_coverage(merged_df)
