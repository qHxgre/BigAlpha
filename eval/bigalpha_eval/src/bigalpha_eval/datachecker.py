import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class DataValidationError(ValueError):
    """因子数据校验失败时抛出（用户数据问题）。"""


# 单日因子缺失率上限
MAX_MISSING_RATE = 0.4


class DataCheck:
    """因子数据校验器（不触碰 dai / 不加载股票池）。

    股票池加载与对齐由调用方（__init__.py）在校验前完成：把提交因子 left-join 到官方面板后，
    再交给本类 validate 一次性走完全部检查。sd/ed 为对齐所用的官方评估窗口（面板真实交易日），
    仅用于日志标注。任一检查失败抛 DataValidationError。
    """

    def __init__(self, start_date: str, end_date: str) -> None:
        # 统一为 Timestamp，避免后续字符串比较脆弱
        self.start_date = pd.to_datetime(start_date).normalize()
        self.end_date = pd.to_datetime(end_date).normalize()

    def _fail(self, msg: str, **fields) -> None:
        logger.error(msg, **fields)
        if fields:
            raise DataValidationError(f"{msg}；详情={fields}")
        raise DataValidationError(msg)

    @staticmethod
    def _factor_cols(df: pd.DataFrame) -> list:
        """date/instrument 之外的列均视为待校验的因子列。"""
        return [c for c in df.columns if c not in {"date", "instrument"}]

    def check_required_columns(self, df: pd.DataFrame) -> None:
        """列检查：必须包含 date/instrument，且至少存在 1 个因子列。"""
        required = {"date", "instrument"}
        cols = set(df.columns)
        missing = required - cols
        if missing:
            self._fail("列检查失败：缺少必需列", missing=sorted(missing))
        if not self._factor_cols(df):
            self._fail("列检查失败：未发现任何因子列（date/instrument 之外）")

    def check_factor_finite(self, df: pd.DataFrame) -> None:
        """有限值检查：所有因子列须可数值化且不允许出现 inf/-inf（NaN 由覆盖度规则约束）。"""
        for col in self._factor_cols(df):
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            inf_mask = np.isinf(x)
            if inf_mask.any():
                bad = df.loc[inf_mask, ["date", "instrument", col]].head(50)
                self._fail(
                    f"因子列 {col} 不允许出现 inf/-inf",
                    factor=col,
                    sample=bad.to_dict(orient="records"),
                )

    def check_uniqueness(self, df: pd.DataFrame) -> None:
        """唯一性：同一 (date, instrument) 组合不允许出现多条记录。

        与官方面板 left-join 不会消除提交里原有的重复键（一行面板会被放大成多行），
        因此对齐后仍可在此检出提交存在的重复 (date, instrument)。
        """
        dup_mask = df.duplicated(subset=["date", "instrument"], keep=False)
        if dup_mask.any():
            cols = ["date", "instrument"] + self._factor_cols(df)
            sample = df.loc[dup_mask, cols].head(50)
            self._fail(
                "唯一性检查失败：存在重复 (date, instrument)",
                sample=sample.to_dict(orient="records"),
            )

    def check_missing_days(self, df: pd.DataFrame) -> None:
        """缺日检查：对齐面板后，不允许存在「整日所有因子值全缺」的交易日。

        时序因子（滚动窗口 / 动量等）若未在窗口前向前多取 warmup 历史，会在评估区间
        开头缺若干交易日；对齐到官方面板后表现为这些交易日因子值全为 NaN。这里把这种
        整日全缺单列出来报错，给出明确的缺失日清单，比混在覆盖度里更直观。
        """
        factor_cols = self._factor_cols(df)
        all_nan = df.groupby("date")[factor_cols].apply(lambda g: g.isna().all().all())
        missing = all_nan[all_nan].index.sort_values()
        if len(missing) > 0:
            sample = [d.strftime("%Y-%m-%d") for d in missing[:50]]
            self._fail(
                "缺日检查失败：存在整日因子值全缺的交易日（时序因子需向前多取 warmup 历史）",
                missing_count=len(missing),
                sample=sample,
            )

    def check_factor_coverage(self, df: pd.DataFrame) -> None:
        """覆盖度：对齐面板后，每交易日各因子列缺失率须 <= MAX_MISSING_RATE。

        df 已是对齐到官方面板的结果（未提交的格子为 NaN），直接按 date 统计缺失率即可。
        """
        factor_cols = self._factor_cols(df)
        for col in factor_cols:
            miss_rate = df[col].isna().groupby(df["date"]).mean()
            high_miss = miss_rate[miss_rate > MAX_MISSING_RATE]
            if not high_miss.empty:
                top = high_miss.sort_values(ascending=False).head(50)
                details = {d.strftime("%Y-%m-%d"): float(r) for d, r in top.items()}
                self._fail(
                    f"覆盖度检查失败：因子 {col} 单日缺失率 > {MAX_MISSING_RATE:.0%}",
                    factor=col,
                    sample=details,
                    total_days=len(high_miss),
                )

    def validate(self, df: pd.DataFrame) -> None:
        """完整校验流程，对「已对齐官方面板的因子数据」依次检查，不通过抛 DataValidationError。

        df 是调用方把提交因子 left-join 到官方面板后的结果（越界 / 非成分股行已丢弃，
        未覆盖格为 NaN），即真正参与打分的口径。检查顺序：
            列 → 唯一性 → 有限值 → 缺日 → 覆盖度。
        """
        if df is None or len(df) == 0:
            self._fail("对齐官方面板后的因子数据为空")

        self.check_required_columns(df)
        logger.info("通过：列名检查（date/instrument + 至少 1 个因子列）", factor_cols=self._factor_cols(df))

        self.check_uniqueness(df)
        logger.info("通过：唯一性检查")

        self.check_factor_finite(df)
        logger.info("通过：factor 有限值检查（禁止 inf/-inf）")

        self.check_missing_days(df)
        logger.info("通过：缺日检查（无整日全缺的交易日）")

        self.check_factor_coverage(df)
        logger.info(f"通过：覆盖度检查（每交易日缺失率<={MAX_MISSING_RATE:.0%}）")




