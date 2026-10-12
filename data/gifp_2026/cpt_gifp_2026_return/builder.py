from datetime import datetime

import dai
import numpy as np
import pandas as pd

from base import BaseBuilder
from gifp_2026.cpt_gifp_2026_return.schema import CptGifp2026ReturnSchema
from gifp_2026.cpt_gifp_2026_return.tester import CptGifp2026ReturnTester


class CptGifp2026ReturnBuilder(BaseBuilder):
    """构建下一交易日同一分钟的 15 分钟频率前向收益。

    对任一采样点 ``(t, HH:MM)``，标签终点是下一交易日 ``t+1`` 的
    ``HH:MM`` 分钟 Bar 收盘价。这里的 ``t+1`` 指下一个实际交易日，
    会自然跨过周末和节假日。收益使用累计后复权因子修正：

    ``future_close * future_adjust_factor / (close * adjust_factor) - 1``。

    采样点为 09:45 至 11:30、13:15 至 15:00 的所有 15 分钟端点。
    区间含义为 ``(t日采样时点, t+1交易日同一时点]``。上游 1 分钟表
    已将 ``[14:59:00, 15:01:00)`` 的快照统一归入 15:00 Bar，所以
    收盘后数秒才落库的快照仍会参与 15:00 收盘价计算。
    """

    datasource_id = "cpt_gifp_2026_return"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptGifp2026ReturnSchema
    source_datasource_id = "cpt_gifp_2026_stock_bar1m"
    lookahead_calendar_days = 31
    output_columns = [
        "date",
        "instrument",
        "return_rate",
        "close",
        "adjust_factor",
        "future_date",
        "future_close",
        "future_adjust_factor",
    ]

    def __init__(self, start_date: str, end_date: str, suffix: str = None) -> None:
        self.start_date = start_date
        self.end_date = end_date
        name = "cpt_gifp_2026_return"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        print(
            f"初始化！{self.datasource_id}, 频率: 15分钟, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    @staticmethod
    def _is_sample_time(date: pd.Series) -> pd.Series:
        minute = date.dt.hour * 60 + date.dt.minute
        morning = minute.between(9 * 60 + 45, 11 * 60 + 30)
        afternoon = minute.between(13 * 60 + 15, 15 * 60)
        return (morning | afternoon) & minute.mod(15).eq(0) & date.dt.second.eq(0)

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        query_end = (
            pd.Timestamp(end_date) + pd.Timedelta(days=self.lookahead_calendar_days)
        ).strftime("%Y-%m-%d")
        return dai.query(
            f"""
            SELECT date, instrument, close, adjust_factor
            FROM {self.source_datasource_id}
            """,
            filters={
                "date": [f"{start_date} 00:00:00", f"{query_end} 23:59:59"]
            },
            compression=True,
        ).df()

    @classmethod
    def aggregate(
        cls, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=cls.output_columns)

        required = {"date", "instrument", "close", "adjust_factor"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"分钟行情缺少字段: {sorted(missing)}")

        data = df[list(required)].copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date", "instrument"])
        data = data.loc[cls._is_sample_time(data["date"])].copy()
        data = data.sort_values(["date", "instrument"])
        data = data.drop_duplicates(["date", "instrument"], keep="last")
        if data.empty:
            return pd.DataFrame(columns=cls.output_columns)

        data["trading_day"] = data["date"].dt.normalize()
        # 用全市场实际出现的日期定义“下一交易日”，不能按自然日加一天，
        # 也不能按单只股票的下一条记录推断，否则停牌会被错误跨越。
        trading_days = pd.Index(data["trading_day"].drop_duplicates().sort_values())
        next_day = pd.Series(trading_days[1:], index=trading_days[:-1])
        data["future_trading_day"] = data["trading_day"].map(next_day)
        data["minute_of_day"] = data["date"].dt.hour * 60 + data["date"].dt.minute

        future = data[
            ["trading_day", "minute_of_day", "instrument", "date", "close", "adjust_factor"]
        ].rename(
            columns={
                "trading_day": "future_trading_day",
                "date": "future_date",
                "close": "future_close",
                "adjust_factor": "future_adjust_factor",
            }
        )
        labels = data.merge(
            future,
            how="left",
            on=["future_trading_day", "minute_of_day", "instrument"],
            validate="one_to_one",
        )

        valid = (
            labels["close"].gt(0)
            & labels["adjust_factor"].gt(0)
            & labels["future_close"].gt(0)
            & labels["future_adjust_factor"].gt(0)
        )
        labels = labels.loc[valid].copy()
        labels["return_rate"] = (
            labels["future_close"]
            * labels["future_adjust_factor"]
            / (labels["close"] * labels["adjust_factor"])
            - 1.0
        )

        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        labels = labels.loc[labels["date"].ge(start) & labels["date"].lt(end)]
        labels = labels.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["return_rate"]
        )
        return (
            labels[cls.output_columns]
            .sort_values(["date", "instrument"])
            .reset_index(drop=True)
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        return df.fillna(self.schema.field_default_mapping())

    def dai_write(self, df: pd.DataFrame) -> None:
        df[dai.DEFAULT_PARTITION_FIELD] = (
            df["date"].dt.strftime("%Y%m").astype("int64")
        )
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=self.schema.default_docs(),
        )

    def build(self) -> pd.DataFrame:
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        if start > end:
            raise ValueError("start_date 不能晚于 end_date")

        t0 = datetime.now()
        data = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取分钟行情耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(data)}"
        )

        labels = self.aggregate(data, self.start_date, self.end_date)
        t2 = datetime.now()
        print(
            f"下一交易日同分钟收益计算耗时: "
            f"{round((t2 - t1).total_seconds(), 4)} 秒, 行数: {len(labels)}"
        )
        if labels.empty:
            print("收益标签为空，跳过入库")
            return pd.DataFrame(columns=self.schema.columns())

        labels = self.normalize(labels)
        CptGifp2026ReturnTester.run_tests(labels)
        self.dai_write(labels)
        t3 = datetime.now()
        print(f"数据存储耗时: {round((t3 - t2).total_seconds(), 4)} 秒")
        return labels
