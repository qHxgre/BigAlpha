import re
import dai
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class DataValidationError(ValueError):
    """分数数据校验失败时抛出（用户数据问题）。"""


# 中证 1000 指数代码
INDEX_CODE = "000852.SH"
# 提交分数文件唯一允许的列
REQUIRED_COLUMNS = ["date", "instrument", "score"]
# 单日分数缺失率上限
MAX_MISSING_RATE = 0.4
# instrument 合法格式
INSTRUMENT_PATTERN = re.compile(r"^\d{6}\.(SZ|SH)$")


class DataCheck:
    """端到端模型分数文件校验。

    分数经风格剔除后等价于一个每日更新的单因子，因此校验口径与单因子一致，
    但列名要求更严格：必须且仅包含 date / instrument / score。
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
    ) -> None:
        # 统一为 Timestamp，避免后续字符串比较脆弱
        self.start_date = pd.to_datetime(start_date).normalize()
        self.end_date = pd.to_datetime(end_date).normalize()

    @property
    def pool_pairs(self) -> pd.DataFrame:
        """中证 1000 历史成分股 (date, instrument)，惰性加载。"""
        return self._load_pool_pairs()

    @property
    def pool_days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.pool_pairs["date"].unique())

    def _load_pool_pairs(self) -> pd.DataFrame:
        sql = "SELECT date, instrument FROM bigalpha_2026_instruments"
        df = dai.query(
            sql,
            filters={"date": [
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d'),
            ]},
        ).df()

        if df is None or df.empty:
            error_msg = "无法获取中证 1000 股票池数据，无法进行数据校验，请联系官方解决"
            logger.error(error_msg)
            raise ValueError(error_msg)

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[["date", "instrument"]].drop_duplicates()

    def _fail(self, msg: str, **fields) -> None:
        logger.error(msg, **fields)
        if fields:
            raise DataValidationError(f"{msg}；详情={fields}")
        raise DataValidationError(msg)

    def check_required_columns(self, df: pd.DataFrame) -> None:
        """列检查：列名必须严格匹配 date / instrument / score，无缺失、无多余列。"""
        cols = list(df.columns)
        required = set(REQUIRED_COLUMNS)
        actual = set(cols)

        missing = required - actual
        if missing:
            self._fail("列检查失败：缺少必需列", missing=sorted(missing))

        extra = actual - required
        if extra:
            self._fail("列检查失败：存在多余列（只允许 date/instrument/score）", extra=sorted(extra))

        if len(cols) != len(actual):
            dup = [c for c in actual if cols.count(c) > 1]
            self._fail("列检查失败：存在重复列名", duplicated=sorted(dup))

    def check_instrument_format(self, df: pd.DataFrame) -> None:
        """格式检查：instrument 必须形如 6 位数字 + .SZ/.SH。"""
        ok = df["instrument"].astype(str).str.match(INSTRUMENT_PATTERN)
        if (~ok).any():
            bad = df.loc[~ok, ["date", "instrument"]].head(50)
            self._fail(
                "instrument 格式不符合 ^\\d{6}\\.(SZ|SH)$",
                sample=bad.to_dict(orient="records"),
            )

    def check_score_finite(self, df: pd.DataFrame) -> None:
        """有限值检查：score 须可数值化且不允许出现 inf/-inf（NaN 由覆盖度规则约束）。"""
        x = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
        inf_mask = np.isinf(x)
        if inf_mask.any():
            bad = df.loc[inf_mask, ["date", "instrument", "score"]].head(50)
            self._fail(
                "score 列不允许出现 inf/-inf",
                sample=bad.to_dict(orient="records"),
            )

    def check_time_period(self, df: pd.DataFrame) -> None:
        """时间范围：date 须落在 [start_date, end_date] 之内。"""
        min_day = df["date"].min()
        max_day = df["date"].max()
        if min_day < self.start_date:
            self._fail(
                "分数最早日期超出规定时间周期",
                min=min_day.strftime("%Y-%m-%d"),
                start=self.start_date.strftime("%Y-%m-%d"),
            )
        if max_day > self.end_date:
            self._fail(
                "分数最晚日期超出规定时间周期",
                max=max_day.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
            )

    def check_trading_days_complete(self, df: pd.DataFrame) -> None:
        """交易日完整性：评估期内每个股票池交易日都必须有提交记录。"""
        submitted = pd.DatetimeIndex(df["date"].unique())
        missing = self.pool_days.difference(submitted).sort_values()
        if len(missing) > 0:
            sample = [d.strftime("%Y-%m-%d") for d in missing[:50]]
            self._fail(
                "交易日完整性检查失败：存在缺失交易日",
                missing_count=len(missing),
                sample=sample,
            )

    def check_uniqueness(self, df: pd.DataFrame) -> None:
        """唯一性：同一 (date, instrument) 组合不允许出现多条记录。"""
        dup_mask = df.duplicated(subset=["date", "instrument"], keep=False)
        if dup_mask.any():
            sample = df.loc[dup_mask, ["date", "instrument", "score"]].head(50)
            self._fail(
                "唯一性检查失败：存在重复 (date, instrument)",
                sample=sample.to_dict(orient="records"),
            )

    def check_stock_pool(self, df: pd.DataFrame) -> None:
        """股票池：所有 (date, instrument) 都必须是当日的中证 1000 成分股。"""
        score_pairs = df[["date", "instrument"]].drop_duplicates()
        merged = score_pairs.merge(
            self.pool_pairs, on=["date", "instrument"], how="left", indicator=True
        )
        out_of_pool = merged[merged["_merge"] == "left_only"]
        if not out_of_pool.empty:
            sample = out_of_pool.head(50)
            details = sample.groupby("date")["instrument"].apply(list).to_dict()
            self._fail(
                "股票池检查失败：存在非当日中证1000成分股",
                sample=details,
                total=len(out_of_pool),
            )

    def check_score_coverage(self, df: pd.DataFrame) -> None:
        """覆盖度：每交易日 score 缺失率（按股票池口径，未提交视为缺失）须 <= MAX_MISSING_RATE。"""
        full = self.pool_pairs.merge(
            df[["date", "instrument", "score"]],
            on=["date", "instrument"],
            how="left",
        )
        miss_rate = full["score"].isna().groupby(full["date"]).mean()
        high_miss = miss_rate[miss_rate > MAX_MISSING_RATE]
        if not high_miss.empty:
            top = high_miss.sort_values(ascending=False).head(50)
            details = {d.strftime("%Y-%m-%d"): float(r) for d, r in top.items()}
            self._fail(
                f"覆盖度检查失败：score 单日缺失率 > {MAX_MISSING_RATE:.0%}",
                sample=details,
                total_days=len(high_miss),
            )

    def validate(self, score_data: pd.DataFrame) -> None:
        """完整校验流程。校验失败抛 DataValidationError。"""
        if score_data is None or len(score_data) == 0:
            self._fail("提交的分数数据为空")

        df = score_data.copy()

        self.check_required_columns(df)
        logger.info("通过：列名检查（严格匹配 date/instrument/score）")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            self._fail("date 列存在无法解析的值")
        df["date"] = df["date"].dt.normalize()

        self.check_instrument_format(df)
        logger.info("通过：instrument 格式检查")

        self.check_score_finite(df)
        logger.info("通过：score 有限值检查（禁止 inf/-inf）")

        self.check_time_period(df)
        logger.info("通过：时间范围检查")

        self.check_trading_days_complete(df)
        logger.info("通过：交易日完整性检查")

        self.check_uniqueness(df)
        logger.info("通过：唯一性检查")

        self.check_stock_pool(df)
        logger.info("通过：股票池范围检查（中证1000历史成分）")

        self.check_score_coverage(df)
        logger.info(f"通过：覆盖度检查（每交易日缺失率<={MAX_MISSING_RATE:.0%}）")
