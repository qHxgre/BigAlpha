from datetime import datetime

import dai
import numpy as np
import pandas as pd

from base import BaseBuilder
from jyc_2026.cpt_jyc_2026_vwap.schema import CptJyc2026VwapSchema


class CptJyc2026VwapBuilder(BaseBuilder):
    """构建股票及指数的未来 30 分钟 VWAP 收益标签。

    标签时点为 09:30、10:00、10:30、11:00、11:30、13:30、14:00、
    14:30。每个截面同时输出三种收益口径：窗口终点相对未来 VWAP、
    未来 VWAP 相对信号价格、窗口终点相对信号价格。11:30 信号跳过午休，
    对应下午 ``(13:00, 13:30]``；14:30 信号通常使用
    ``(14:30, 15:00]``，并额外接纳 15:00 后一分钟内到达的收盘快照。

    Level-2 的 volume、amount、num_trades 是日内累计字段。这里先在逐笔
    快照层面做差，再汇总区间增量，避免用区间首末快照直接相减时
    遗漏区间开始后到首条快照之间的成交。
    """

    datasource_id = "cpt_jyc_2026_vwap"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptJyc2026VwapSchema

    _CUMULATIVE_COLUMNS = ["volume", "amount", "num_trades"]
    _INDEX_INSTRUMENTS = ("000852.SH",)
    _CLOSE_GRACE_MS = 60_000
    _OUTPUT_COLUMNS = [
        "date",
        "instrument",
        "vwap_return",
        "signal_to_vwap_return",
        "signal_to_end_return",
        "signal_price",
        "vwap",
        "end_price",
        "volume",
        "amount",
        "num_trades",
    ]

    def __init__(self, start_date: str, end_date: str, suffix: str = None) -> None:
        self.start_date = start_date
        self.end_date = end_date
        if suffix:
            self.datasource_id = f"cpt_jyc_2026_vwap_{suffix}"
        print(
            f"初始化！{self.datasource_id}, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        return df.fillna(self.schema.field_default_mapping())

    def dai_write(self, df: pd.DataFrame) -> None:
        default_docs = self.schema.default_docs()
        df[dai.DEFAULT_PARTITION_FIELD] = (
            df["date"].dt.strftime("%Y%m").astype("int64")
        )
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """读取中证 1000 历史成分股及指数本身的快照字段。"""
        stock_sql = """
        WITH cte_index AS (
            SELECT
                CAST(strftime(date, '%Y%m%d') AS INT32) AS trading_day,
                member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
        )
        SELECT
            s.date,
            s.trading_day,
            s.instrument,
            s.price,
            s.volume,
            s.amount,
            s.num_trades
        FROM cn_stock_level2_snapshot s
        INNER JOIN cte_index i
            ON s.trading_day = i.trading_day
            AND s.instrument = i.instrument
        """
        date_filter = {
            "date": [
                f"{start_date} 00:00:00",
                f"{end_date} 23:59:59",
            ]
        }
        stock_df = dai.query(
            stock_sql,
            filters=date_filter,
            compression=True,
        ).df()

        index_df = dai.query(
            "SELECT * FROM cn_stock_index_snapshot ORDER BY date",
            filters={
                **date_filter,
                "instrument": list(self._INDEX_INSTRUMENTS),
            },
            compression=True,
        ).df()
        snapshot_columns = [
            "date",
            "trading_day",
            "instrument",
            "price",
            "volume",
            "amount",
            "num_trades",
        ]
        return pd.concat(
            [stock_df[snapshot_columns], index_df[snapshot_columns]],
            ignore_index=True,
        )

    @classmethod
    def _assign_window_start(cls, date: pd.Series) -> pd.Series:
        """把快照映射到未来 30 分钟窗口的采样起点。

        窗口均为左开右闭。于是 10:00:00 属于 09:30 标签，而
        10:00:00.001 属于 10:00 标签；午休与集合竞价返回 NaT。
        参考分钟线构建逻辑，收盘后一分钟内到达的快照仍归入收盘窗口，
        即下午 ``(14:30:00, 15:01:00)`` 均归入 14:30 标签。
        """
        date = pd.to_datetime(date)
        day = date.dt.normalize()
        elapsed_ms = (
            (date.dt.hour * 3600 + date.dt.minute * 60 + date.dt.second) * 1000
            + date.dt.microsecond // 1000
        )

        morning_open = (9 * 3600 + 30 * 60) * 1000
        morning_close = (11 * 3600 + 30 * 60) * 1000
        afternoon_open = 13 * 3600 * 1000
        afternoon_close = 15 * 3600 * 1000
        window_ms = 30 * 60 * 1000

        start_minute = pd.Series(np.nan, index=date.index, dtype="float64")
        sessions = (
            (morning_open, morning_close, [570, 600, 630, 660]),
            # 下午首个窗口映射到午间 11:30 信号。
            (afternoon_open, afternoon_close, [690, 810, 840, 870]),
        )
        for session_open, session_close, label_minutes in sessions:
            in_session = (elapsed_ms > session_open) & (elapsed_ms <= session_close)
            # 减 1 使精确落在右端点的快照仍属于前一个窗口。
            slot = ((elapsed_ms - session_open - 1) // window_ms).astype("int64")
            labels = slot.map(dict(enumerate(label_minutes)))
            start_minute.loc[in_session] = labels.loc[in_session]

            # 部分标的的收盘快照会在收盘后数秒才落库。与 stock_barkm 的
            # 收盘口径保持一致，将收盘后一分钟内的快照归入最后一个窗口；
            # 右端保持开放，避免把下一分钟边界也算入交易时段。
            close_grace = (
                (elapsed_ms > session_close)
                & (elapsed_ms < session_close + cls._CLOSE_GRACE_MS)
            )
            start_minute.loc[close_grace] = label_minutes[-1]

        return day + pd.to_timedelta(start_minute, unit="m")

    @classmethod
    def aggregate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """由累计快照生成 8 个日内截面的未来 30 分钟 VWAP 收益。"""
        if df.empty:
            return pd.DataFrame(columns=cls._OUTPUT_COLUMNS)

        required = {
            "date",
            "instrument",
            "price",
            *cls._CUMULATIVE_COLUMNS,
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"快照数据缺少字段: {sorted(missing)}")

        data = df[list(required)].copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.dropna(subset=["date", "instrument"])
        data = data.sort_values(["instrument", "date"]).reset_index(drop=True)
        data["__trading_day"] = data["date"].dt.normalize()
        groups = data.groupby(
            ["instrument", "__trading_day"], sort=False, observed=True
        )

        # price <= 0 不作为窗口终点价格；组内最后一个有效成交价即终点价。
        data["__valid_price"] = data["price"].where(data["price"] > 0)

        # 09:30 是每日第一个标签，无法从上一窗口的 end_price 推导信号价；
        # 使用 09:30 当时已可获得的最后一个有效价格（通常为集合竞价价格）。
        signal_cutoff = data["__trading_day"] + pd.Timedelta(hours=9, minutes=30)
        opening_signal_prices = (
            data.loc[data["date"] <= signal_cutoff]
            .dropna(subset=["__valid_price"])
            .groupby(["instrument", "__trading_day"], sort=False, observed=True)[
                "__valid_price"
            ]
            .last()
            .rename("__opening_signal_price")
            .reset_index()
        )

        for column in cls._CUMULATIVE_COLUMNS:
            delta = groups[column].diff()
            # 盘中重置时，当前累计值就是重置后的有效增量。
            data[f"__{column}_delta"] = delta.where(delta >= 0, data[column])

        # 股票的 amount / volume 与股票价格处于同一量纲，可以直接用于
        # VWAP。指数快照的 amount、volume 是成分证券的汇总成交额和成交量，
        # amount / volume 并不是指数点位，因此指数改用快照点位按成交量增量
        # 加权：sum(price * volume_delta) / sum(volume_delta)。
        data["__vwap_numerator_delta"] = data["__amount_delta"]
        is_index = data["instrument"].isin(cls._INDEX_INSTRUMENTS)
        data.loc[is_index, "__vwap_numerator_delta"] = (
            data.loc[is_index, "__valid_price"]
            * data.loc[is_index, "__volume_delta"]
        )

        data["__window_start"] = cls._assign_window_start(data["date"])
        data = data.dropna(subset=["__window_start"])
        if data.empty:
            return pd.DataFrame(columns=cls._OUTPUT_COLUMNS)

        labels = (
            data.groupby(["instrument", "__window_start"], sort=True, observed=True)
            .agg(
                end_price=("__valid_price", "last"),
                **{
                    column: (f"__{column}_delta", "sum")
                    for column in cls._CUMULATIVE_COLUMNS
                },
                __vwap_numerator=("__vwap_numerator_delta", "sum"),
            )
            .reset_index()
            .rename(columns={"__window_start": "date"})
        )

        has_volume = labels["volume"] > 0
        labels["vwap"] = np.divide(
            labels["__vwap_numerator"],
            labels["volume"],
            out=np.full(len(labels), np.nan, dtype="float64"),
            where=has_volume & labels["__vwap_numerator"].notna(),
        )
        valid_vwap = labels["vwap"] > 0
        labels["vwap_return"] = np.divide(
            labels["end_price"],
            labels["vwap"],
            out=np.full(len(labels), np.nan, dtype="float64"),
            where=valid_vwap & labels["end_price"].notna(),
        ) - 1.0

        labels["__trading_day"] = labels["date"].dt.normalize()
        labels = labels.sort_values(["instrument", "date"]).reset_index(drop=True)
        # 除每日首个截面外，当前信号时刻价格等于上一个非重叠窗口的终点价。
        # 该方式也正确处理 11:30 -> 13:30 的午休映射。
        labels["signal_price"] = labels.groupby(
            ["instrument", "__trading_day"], sort=False, observed=True
        )["end_price"].shift(1)
        labels = labels.merge(
            opening_signal_prices,
            how="left",
            on=["instrument", "__trading_day"],
            validate="many_to_one",
        )
        first_section = labels.groupby(
            ["instrument", "__trading_day"], sort=False, observed=True
        ).cumcount().eq(0)
        labels.loc[first_section, "signal_price"] = labels.loc[
            first_section, "__opening_signal_price"
        ]

        valid_signal = labels["signal_price"] > 0
        labels["signal_to_vwap_return"] = np.divide(
            labels["vwap"],
            labels["signal_price"],
            out=np.full(len(labels), np.nan, dtype="float64"),
            where=valid_signal & valid_vwap,
        ) - 1.0
        labels["signal_to_end_return"] = np.divide(
            labels["end_price"],
            labels["signal_price"],
            out=np.full(len(labels), np.nan, dtype="float64"),
            where=valid_signal & labels["end_price"].notna(),
        ) - 1.0

        return (
            labels[cls._OUTPUT_COLUMNS]
            .sort_values(["date", "instrument"])
            .reset_index(drop=True)
        )

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取数据耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(df)}"
        )

        df = self.aggregate(df)
        t2 = datetime.now()
        print(
            f"未来30分钟VWAP收益聚合耗时: "
            f"{round((t2 - t1).total_seconds(), 4)} 秒, "
            f"行数: {len(df)}"
        )

        df = self.normalize(df)
        self.dai_write(df)
        t3 = datetime.now()
        print(f"数据存储耗时: {round((t3 - t2).total_seconds(), 4)} 秒")
        return df
