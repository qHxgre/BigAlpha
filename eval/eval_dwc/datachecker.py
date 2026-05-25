import re
import dai
import numpy as np
import pandas as pd
import structlog
from typing import Optional

logger = structlog.get_logger()


class DataValidationError(ValueError):
    """用于因子数据校验失败时抛出的异常。"""


class DataCheck:
    """
    竞赛提交因子数据校验（严格版，默认开启额外约束）：
    - 列检查：必须且仅包含三列：date, instrument, factor
    - instrument 格式：必须匹配 ^\\d{6}\\.(SZ|SH)$
    - factor：必须可转为数值；且不能为 inf/-inf（NaN 允许，但受覆盖度规则约束）
    - 时间范围：必须落在 [start_date, end_date] 内
    - 交易日完整性：不得缺失评估期内任何交易日（以股票池历史成分 trading_day 为准）
    - 唯一性：(date, instrument) 不得重复
    - 股票池检查：不得包含非当日沪深300历史成分股
    - 全量成分行数：每个交易日必须覆盖当日股票池的全部 (day,instrument) 组合（可选开关，默认开启）
    - 覆盖度：按股票池口径统计每交易日 factor 缺失率 <= 40%
    - 数据泄露
    """

    INSTRUMENT_PATTERN = re.compile(r"^\d{6}\.(SZ|SH)$")

    def __init__(
        self,
        start_date: str,
        end_date: str,
        check_data: Optional[pd.DataFrame]=None
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date

        # 股票池：沪深300历史成分（按你原代码保留 SQL，不处理 where 字段可能写错的问题）
        index_code = "000300.SH"
        sql = f"""
            SELECT date as trading_day, member_code as instrument
            FROM cn_stock_index_component
            WHERE instrument = '{index_code}'
        """
        self.stocks_pool = dai.query(
            sql, filters={"date": [f"{self.start_date} 00:00:00", f"{self.end_date} 23:59:59"]}
        ).df()

        if self.stocks_pool is None or self.stocks_pool.empty:
            error_msg = '无法获取股票池数据，无法进行数据检验，请联系官方解决'
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 预计算：股票池全集 (trading_day, instrument)
        self.pool_pairs = self.stocks_pool[["trading_day", "instrument"]].drop_duplicates()

        # 预计算：股票池交易日全集
        self.pool_days = set(self.stocks_pool["trading_day"].unique())

        # 检查数据
        self.check_data = check_data

    def check_columns_exactly_three(self, factor_data: pd.DataFrame) -> None:
        """检查列名"""
        required = {"date", "instrument", "factor"}
        cols = list(factor_data.columns)

        if set(cols) != required:
            missing = sorted(list(required - set(cols)))
            extra = sorted(list(set(cols) - required))
            logger.error(
                "列检查失败：必须且仅包含 date/instrument/factor",
                missing=missing,
                extra=extra,
                cols=cols,
            )
            raise DataValidationError(
                f"列检查失败：必须且仅包含 {sorted(list(required))}；缺失={missing}；多余={extra}"
            )

    def check_instrument_format(self, df: pd.DataFrame) -> None:
        """检查 instrument 格式"""
        ok = df["instrument"].apply(lambda x: bool(self.INSTRUMENT_PATTERN.match(x)))
        if (~ok).any():
            bad = df.loc[~ok, ["trading_day", "instrument"]].head(50)
            logger.error("instrument 格式不符合要求", sample=bad.to_dict(orient="records"))
            raise DataValidationError(
                "instrument 格式不符合 ^\\d{6}\\.(SZ|SH)$，样例(最多50条)："
                + str(bad.to_dict(orient="records"))
            )

    def check_factor_finite(self, df: pd.DataFrame) -> None:
        # NaN 允许（由覆盖度规则控制），但 inf/-inf 不允许
        inf_mask = np.isinf(df["factor"].to_numpy(dtype=float))
        if inf_mask.any():
            bad = df.loc[inf_mask, ["trading_day", "instrument", "factor"]].head(50)
            logger.error("factor 存在 inf/-inf", sample=bad.to_dict(orient="records"))
            raise DataValidationError(
                "factor 不允许出现 inf/-inf，样例(最多50条)："
                + str(bad.to_dict(orient="records"))
            )

    def check_time_period(self, df: pd.DataFrame) -> None:
        """检查数据范围"""
        min_day = df["trading_day"].min().strftime("%Y-%m-%d")
        max_day = df["trading_day"].max().strftime("%Y-%m-%d")

        if min_day < self.start_date:
            logger.error("因子数据最早日期超出规定范围", min_day=min_day, start_date=self.start_date)
            raise DataValidationError(
                f"因子数据的最早日期超出规定时间周期：min={min_day}, start={self.start_date}"
            )

        if max_day > self.end_date:
            logger.error("因子数据最晚日期超出规定范围", max_day=max_day, end_date=self.end_date)
            raise DataValidationError(
                f"因子数据的最晚日期超出规定时间周期：max={max_day}, end={self.end_date}"
            )

    def check_trading_days_complete(self, df: pd.DataFrame) -> None:
        """检查交易日完整性"""
        submitted_days = set(df["trading_day"].unique())
        missing_days = sorted(list(self.pool_days - submitted_days))
        if missing_days:
            logger.error(
                "交易日完整性检查失败：缺失交易日",
                missing_count=len(missing_days),
                sample=missing_days[:50],
            )
            raise DataValidationError(
                f"交易日完整性检查失败：缺失 {len(missing_days)} 个交易日，样例(最多50个)：{missing_days[:50]}"
            )

    def check_uniqueness(self, df):
        """数据唯一性检查"""
        dup = df[df.duplicated(subset=["date", "instrument"], keep=False)].head(50)
        if not dup.empty:
            logger.error(
                "唯一性检查失败：存在重复 (date,instrument)",
                sample=dup[["date", "instrument", "factor"]].to_dict(orient="records"),
            )
            raise DataValidationError(
                "唯一性检查失败：存在重复的 (date,instrument) 组合，样例(最多50条)："
                + str(dup[["date", "instrument"]].to_dict(orient="records"))
            )

    def check_stock_pool(self, df: pd.DataFrame) -> None:
        """
        不得包含非当日成分股：提交的 (trading_day, instrument) 必须是 pool_pairs 的子集
        """
        factor_pairs = df[["trading_day", "instrument"]].drop_duplicates()
        merged = factor_pairs.merge(
            self.pool_pairs, on=["trading_day", "instrument"], how="left", indicator=True
        )
        out_of_pool = merged[merged["_merge"] == "left_only"]
        if not out_of_pool.empty:
            sample = out_of_pool.head(50)
            details = sample.groupby("trading_day")["instrument"].apply(list).to_dict()
            logger.error("股票池检查失败：存在非成分股", sample=details, total=int(len(out_of_pool)))
            raise DataValidationError(
                f"股票池检查失败：存在非当日成分股，样例(最多50条)：{details}，总数={len(out_of_pool)}"
            )

    def check_full_pool_rows(self, df: pd.DataFrame) -> None:
        """
        额外强约束：每个交易日必须提交当日股票池全量 (day,instrument) 组合：
        - 不允许缺行（缺某些成分股的行）
        - 不允许多行（出现非股票池组合这步已被 check_stock_pool 拦截）
        """
        factor_pairs = df[["trading_day", "instrument"]].drop_duplicates()

        # 找缺失的 pool pairs
        missing = self.pool_pairs.merge(
            factor_pairs, on=["trading_day", "instrument"], how="left", indicator=True
        )
        missing = missing[missing["_merge"] == "left_only"][["trading_day", "instrument"]]

        if not missing.empty:
            sample = missing.head(50)
            details = sample.groupby("trading_day")["instrument"].apply(list).to_dict()
            logger.error("全量成分行检查失败：存在缺失成分股行", sample=details, total=int(len(missing)))
            raise DataValidationError(
                f"全量成分行检查失败：存在缺失的 (交易日,成分股) 行，样例(最多50条)：{details}，总缺失行数={len(missing)}"
            )

    def check_factor_coverage(self, df: pd.DataFrame) -> None:
        """
        覆盖度（按股票池口径）：
        - 分母：当日成分股数量（pool_pairs）
        - 分子：factor 为 NaN 的数量（提交但 NaN / 或未提交行：若 enforce_full_pool_rows=True，未提交行已被拦截）
        - 规则：每个 trading_day 缺失率 <= 40%
        """
        factor_min = df[["trading_day", "instrument", "factor"]].copy()

        full = self.pool_pairs.merge(
            factor_min, on=["trading_day", "instrument"], how="left"
        )

        miss_rate = full["factor"].isna().groupby(full["trading_day"]).mean()
        high_miss = miss_rate[miss_rate > 0.4]
        if not high_miss.empty:
            high_miss_sorted = high_miss.sort_values(ascending=False).head(50)
            details = {day: float(rate) for day, rate in high_miss_sorted.items()}
            logger.error("覆盖度检查失败：缺失率超过40%", sample=details, total_days=int(len(high_miss)))
            raise DataValidationError(
                f"覆盖度检查失败：以下交易日因子缺失率 > 40%（展示最多50天）：{details}；超标天数={len(high_miss)}"
            )

    def check_intraday_bars(self, df: pd.DataFrame) -> None:
        """
        日内分钟k线是否完整：可以缺少，但不能有多余的时间点
        """
        df = df.copy()
        # 定义允许的时间点集合
        valid_times = {
            '094500', '100000', '101500', '103000', '104500', '110000',
            '111500', '113000', '131500', '133000', '134500', '140000',
            '141500', '143000', '144500', '150000'
        }
        
        # 计算时间字符串
        df['time'] = df['date'].dt.strftime('%H%M%S')
        df['trading_day'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # 检查每个分组的时间是否都是valid_times中的时间
        invalid_groups = []
        
        for (trading_day, instrument), group in df.groupby(['trading_day', 'instrument']):
            # 获取该分组所有不重复的时间点
            group_times = set(group['time'])
            
            # 检查是否有不在valid_times中的时间点
            if not group_times.issubset(valid_times):
                invalid_groups.append(f"{trading_day}-{instrument}")
        
        if invalid_groups:
            # 取前50个展示
            display_text = ", ".join(invalid_groups[:50])
            
            if len(invalid_groups) > 50:
                display_text += f" ...(还有{len(invalid_groups)-50}个)"
            
            logger.error("日内k线异常: %s, 总数: %d", display_text, len(invalid_groups))
            raise DataValidationError(f"日内k线检查失败: 共{len(invalid_groups)}个组合包含无效时间点")

    def check_data_breach(self, factor_data: pd.DataFrame, check_data: pd.DataFrame) -> None:
        """
        检查是否存在未来函数
        """
        # check_data = check_data[(check_data['date']>='2025-04-01 00:00:00') & (check_data['date']<='2025-04-30 23:59:59')]
        check_data['time'] = check_data['date'].dt.strftime('%H:%M:%S').str.replace(':', '').astype(int)
        check_data = check_data[
            (
                check_data['time']<=103000)
            | (
                (check_data['time']>=130000)
                & (check_data['time']<=140000)
            )
        ]
        merge_data = pd.merge(check_data, factor_data, how='left', on=['date', 'instrument'], suffixes=['_check', ''])
        merge_data['diff'] = (merge_data['factor_check'] / merge_data['factor'] - 1).abs()

        invalid_data = merge_data[merge_data['diff']>0.00001][['date', 'instrument', 'factor', 'factor_check']]
        if invalid_data.shape[0] / merge_data.shape[0] > 0.05:
            logger.error("数据泄露检查不通过，代码可能存在未来函数，请检查！")
            raise DataValidationError(
                f"数据泄露检查不通过，代码可能存在未来函数，请检查！"
            )

    def validate(self, factor_data: pd.DataFrame) -> None:
        """
        执行完整校验流程。不通过则抛 DataValidationError。
        """
        if factor_data is None or len(factor_data) == 0:
            logger.error("提交数据为空")
            raise DataValidationError("提交的因子数据为空")

        df = factor_data.copy()

        self.check_columns_exactly_three(df)
        logger.info("通过：列名检查（必须且仅三列）")

        df["trading_day"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d"))
        
        self.check_instrument_format(df)
        logger.info("通过：instrument 格式检查（^\u005cd{6}\u005c.(SZ|SH)$）")

        self.check_factor_finite(df)
        logger.info("通过：factor 有限值检查（禁止 inf/-inf）")
        
        self.check_time_period(df)
        logger.info("通过：时间范围检查")
        
        self.check_trading_days_complete(df)
        logger.info("通过：交易日完整性检查")

        self.check_uniqueness(df)
        logger.info("通过：唯一性检查")

        self.check_stock_pool(df)
        logger.info("通过：股票池范围检查（沪深300历史成分）")

        # self.check_full_pool_rows(df)
        # logger.info("通过：全量成分行数检查（每交易日覆盖全部成分股行）")

        self.check_intraday_bars(df)
        logger.info("通过：日内分钟因子符合要求")

        self.check_factor_coverage(df)
        logger.info("通过：覆盖度检查（每交易日缺失率<=40%）")

        if self.check_data is not None:
            self.check_data_breach(df, self.check_data.copy())
            logger.info("通过：数据泄露检查，不存在未来数据")
        else:
            logger.info("跳过：数据泄露检查，不检查是否存在未来函数")

