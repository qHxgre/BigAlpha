"""数据校验

按 docs/因子挖掘_介绍_20260525.md "数据校验与预处理" 章节实现：
- 列检查：必须且仅包含 date / instrument / factor 三列
- instrument 格式：^\\d{6}\\.(SZ|SH)$
- factor：可数值化、不允许 inf；NaN 受覆盖度规则约束
- 时间范围：[start_date, end_date]
- 交易日完整性：评估期内不得缺失任何交易日
- 唯一性：(date, instrument) 不重复
- 股票池：中证 1000（000852.SH）历史成分；不允许包含非当日成分股
- 覆盖度：每个交易日 factor 缺失率 <= 40%
- 数据泄露：与 check_data 比对，验证不存在未来函数
"""

import re
import dai
import numpy as np
import pandas as pd
import structlog
from typing import Optional

logger = structlog.get_logger()


class DataValidationError(ValueError):
    """因子数据校验失败时抛出（用户数据问题）。"""


# 中证 1000 指数代码
INDEX_CODE = "000852.SH"
# 单日因子缺失率上限
MAX_MISSING_RATE = 0.4
# 数据泄露：相对误差阈值与超阈样本占比上限
DATA_BREACH_RTOL = 1e-5
DATA_BREACH_ATOL = 1e-8
DATA_BREACH_RATIO_LIMIT = 0.05
# instrument 合法格式
INSTRUMENT_PATTERN = re.compile(r"^\d{6}\.(SZ|SH)$")


class DataCheck:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        check_data: Optional[pd.DataFrame] = None,
        pool_pairs: Optional[pd.DataFrame] = None,
    ) -> None:
        # 统一为 Timestamp，避免后续字符串比较脆弱
        self.start_date = pd.to_datetime(start_date).normalize()
        self.end_date = pd.to_datetime(end_date).normalize()
        self.check_data = check_data
        self._pool_pairs = pool_pairs  # 允许外部注入，便于单测

    @property
    def pool_pairs(self) -> pd.DataFrame:
        """中证 1000 历史成分股 (date, instrument)，惰性加载。"""
        if self._pool_pairs is None:
            self._pool_pairs = self._load_pool_pairs()
        return self._pool_pairs

    @property
    def pool_days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.pool_pairs["date"].unique())

    def _load_pool_pairs(self) -> pd.DataFrame:
        sql = f"""
            SELECT date, member_code as instrument
            FROM cn_stock_index_component
            WHERE instrument = '{INDEX_CODE}'
        """
        df = dai.query(
            sql,
            filters={"date": [
                f"{self.start_date.strftime('%Y-%m-%d')} 00:00:00",
                f"{self.end_date.strftime('%Y-%m-%d')} 23:59:59",
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

    def check_columns_exactly_three(self, df: pd.DataFrame) -> None:
        """列检查：必须且仅包含 date/instrument/factor 三列。"""
        required = {"date", "instrument", "factor"}
        cols = set(df.columns)
        if cols != required:
            self._fail(
                "列检查失败：必须且仅包含 date/instrument/factor",
                missing=sorted(required - cols),
                extra=sorted(cols - required),
            )

    def check_instrument_format(self, df: pd.DataFrame) -> None:
        """格式检查：instrument 必须形如 6 位数字 + .SZ/.SH。"""
        ok = df["instrument"].astype(str).str.match(INSTRUMENT_PATTERN)
        if (~ok).any():
            bad = df.loc[~ok, ["date", "instrument"]].head(50)
            self._fail(
                "instrument 格式不符合 ^\\d{6}\\.(SZ|SH)$",
                sample=bad.to_dict(orient="records"),
            )

    def check_factor_finite(self, df: pd.DataFrame) -> None:
        """有限值检查：factor 须可数值化且不允许出现 inf/-inf（NaN 由覆盖度规则约束）。"""
        x = pd.to_numeric(df["factor"], errors="coerce").to_numpy(dtype=float)
        inf_mask = np.isinf(x)
        if inf_mask.any():
            bad = df.loc[inf_mask, ["date", "instrument", "factor"]].head(50)
            self._fail(
                "factor 不允许出现 inf/-inf",
                sample=bad.to_dict(orient="records"),
            )

    def check_time_period(self, df: pd.DataFrame) -> None:
        """时间范围：date 须落在 [start_date, end_date] 之内。"""
        min_day = df["date"].min()
        max_day = df["date"].max()
        if min_day < self.start_date:
            self._fail(
                "因子最早日期超出规定时间周期",
                min=min_day.strftime("%Y-%m-%d"),
                start=self.start_date.strftime("%Y-%m-%d"),
            )
        if max_day > self.end_date:
            self._fail(
                "因子最晚日期超出规定时间周期",
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
            sample = df.loc[dup_mask, ["date", "instrument", "factor"]].head(50)
            self._fail(
                "唯一性检查失败：存在重复 (date, instrument)",
                sample=sample.to_dict(orient="records"),
            )

    def check_stock_pool(self, df: pd.DataFrame) -> None:
        """股票池：所有 (date, instrument) 都必须是当日的中证 1000 成分股。"""
        factor_pairs = df[["date", "instrument"]].drop_duplicates()
        merged = factor_pairs.merge(
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

    def check_factor_coverage(self, df: pd.DataFrame) -> None:
        """覆盖度：每交易日 factor 缺失率（按股票池口径，未提交视为缺失）须 <= MAX_MISSING_RATE。"""
        factor_min = df[["date", "instrument", "factor"]]
        full = self.pool_pairs.merge(factor_min, on=["date", "instrument"], how="left")
        miss_rate = full["factor"].isna().groupby(full["date"]).mean()
        high_miss = miss_rate[miss_rate > MAX_MISSING_RATE]
        if not high_miss.empty:
            top = high_miss.sort_values(ascending=False).head(50)
            details = {d.strftime("%Y-%m-%d"): float(r) for d, r in top.items()}
            self._fail(
                f"覆盖度检查失败：单日因子缺失率 > {MAX_MISSING_RATE:.0%}",
                sample=details,
                total_days=len(high_miss),
            )

    def check_data_breach(self, factor_data: pd.DataFrame, check_data: pd.DataFrame) -> None:
        """数据泄露：与 check_data 在 (date, instrument) 上对齐比对，超阈值样本占比过高视为存在未来函数。"""
        check_data = check_data.copy()
        check_data["date"] = pd.to_datetime(check_data["date"]).dt.normalize()
        merged = pd.merge(
            check_data, factor_data, how="left",
            on=["date", "instrument"], suffixes=["_check", ""],
        ).dropna(subset=["factor_check", "factor"])

        if merged.empty:
            logger.warning("数据泄露检查：无法对齐，跳过")
            return

        # 用相对误差 + 绝对误差的组合，避免 factor==0 导致除零
        diff = (merged["factor_check"] - merged["factor"]).abs()
        tol = DATA_BREACH_ATOL + DATA_BREACH_RTOL * merged["factor"].abs()
        invalid_ratio = (diff > tol).mean()
        if invalid_ratio > DATA_BREACH_RATIO_LIMIT:
            self._fail(
                "数据泄露检查不通过，代码可能存在未来函数",
                invalid_ratio=float(invalid_ratio),
                limit=DATA_BREACH_RATIO_LIMIT,
            )

    def validate(self, factor_data: pd.DataFrame) -> None:
        """完整校验流程。校验失败抛 DataValidationError。"""
        if factor_data is None or len(factor_data) == 0:
            self._fail("提交的因子数据为空")

        df = factor_data.copy()

        self.check_columns_exactly_three(df)
        logger.info("通过：列名检查（必须且仅三列）")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            self._fail("date 列存在无法解析的值")
        df["date"] = df["date"].dt.normalize()

        self.check_instrument_format(df)
        logger.info("通过：instrument 格式检查")

        self.check_factor_finite(df)
        logger.info("通过：factor 有限值检查（禁止 inf/-inf）")

        self.check_time_period(df)
        logger.info("通过：时间范围检查")

        self.check_trading_days_complete(df)
        logger.info("通过：交易日完整性检查")

        self.check_uniqueness(df)
        logger.info("通过：唯一性检查")

        self.check_stock_pool(df)
        logger.info("通过：股票池范围检查（中证1000历史成分）")

        self.check_factor_coverage(df)
        logger.info(f"通过：覆盖度检查（每交易日缺失率<={MAX_MISSING_RATE:.0%}）")

        if self.check_data is not None:
            self.check_data_breach(df, self.check_data)
            logger.info("通过：数据泄露检查")
        else:
            logger.info("跳过：数据泄露检查")

