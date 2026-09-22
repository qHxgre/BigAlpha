from datetime import datetime

import dai
import numpy as np
import pandas as pd

from base import BaseBuilder
from jyc_2026.cpt_jyc_2026_stock_barkm.constant import TIME_SETS
from jyc_2026.cpt_jyc_2026_stock_barkm.schema import CptJyc2026StockBarKmSchema


class CptJyc2026StockBarKmBuilder(BaseBuilder):
    """由 Level-2 快照构建中证 1000 成分股 K 分钟行情。"""

    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptJyc2026StockBarKmSchema

    _CUMULATIVE_FIELDS = ["volume", "amount", "num_trades"]
    _PRICE_FIELDS = ["open", "high", "low", "close"]
    _IDENTITY_FIELDS = ["instrument_id", "adjust_factor"]
    _MIN_DATE = "2020-01-01"

    def __init__(
        self,
        start_date: str,
        end_date: str,
        K: int = 1,
        suffix: str = None,
    ) -> None:
        self.start_date = max(start_date, self._MIN_DATE)
        self.end_date = end_date
        self.K = int(K)
        if self.K not in TIME_SETS:
            raise ValueError(
                f"不支持的频率 K={self.K}, 可选: {sorted(TIME_SETS)} "
                "(在 constant.py 中定义)"
            )

        name = f"cpt_jyc_2026_stock_bar{self.K}m"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        print(
            f"初始化！{self.datasource_id}, 频率: {self.K}分钟, "
            f"时间周期: {self.start_date}, {self.end_date}"
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

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """按交易日成分关系读取 2020 年以来的中证 1000 快照。"""
        sql = """
        WITH index_members AS (
            SELECT
                CAST(strftime(date, '%Y%m%d') AS INT32) AS trading_day,
                member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
              AND date >= '2020-01-01'
        )
        SELECT s.*
        FROM cn_stock_level2_snapshot s
        INNER JOIN index_members m
            ON s.trading_day = m.trading_day
           AND s.instrument = m.instrument
        """
        return dai.query(
            sql,
            filters={
                "date": [
                    f"{max(start_date, self._MIN_DATE)} 00:00:00",
                    f"{end_date} 23:59:59",
                ]
            },
            compression=True,
        ).df()

    @staticmethod
    def _elapsed_ms(date: pd.Series) -> pd.Series:
        return (
            (date.dt.hour * 3600 + date.dt.minute * 60 + date.dt.second) * 1000
            + date.dt.microsecond // 1000
        )

    @classmethod
    def _assign_minute_end(cls, date: pd.Series) -> pd.Series:
        """按交易所边界把快照映射到 09:25 或连续竞价分钟末端。"""
        date = pd.to_datetime(date)
        day = date.dt.normalize()
        elapsed_ms = cls._elapsed_ms(date)
        minute_ms = 60_000

        auction_start = (9 * 3600 + 15 * 60) * 1000
        auction_end = (9 * 3600 + 25 * 60) * 1000
        morning_open = (9 * 3600 + 30 * 60) * 1000
        morning_close = (11 * 3600 + 30 * 60) * 1000
        afternoon_open = 13 * 3600 * 1000
        afternoon_close = 15 * 3600 * 1000

        end_minute = pd.Series(np.nan, index=date.index, dtype="float64")
        auction = (elapsed_ms >= auction_start) & (elapsed_ms <= auction_end)
        end_minute.loc[auction] = 9 * 60 + 25

        # 减 1ms 让精确的分钟端点仍落在前一个左开右闭区间。
        for session_open, session_close in (
            (morning_open, morning_close),
            (afternoon_open, afternoon_close),
        ):
            in_session = (elapsed_ms >= session_open) & (elapsed_ms <= session_close)
            labels = ((elapsed_ms - 1) // minute_ms + 1).astype("int64")
            # 开盘时刻本身属于开盘第一分钟，而不是单独生成 09:30/13:00 bar。
            labels = labels.where(elapsed_ms != session_open, session_open // minute_ms + 1)
            end_minute.loc[in_session] = labels.loc[in_session]

        return day + pd.to_timedelta(end_minute, unit="m")

    @classmethod
    def build_one_minute(cls, df: pd.DataFrame) -> pd.DataFrame:
        """将原始快照转换为 09:25 截面及右闭口径的一分钟行情。"""
        output_columns = [
            c for c in cls.schema.columns() if c not in cls._IDENTITY_FIELDS
        ]
        if df.empty:
            return pd.DataFrame(columns=output_columns)

        required = {
            "date",
            "instrument",
            "price",
            "pre_close",
            *cls._CUMULATIVE_FIELDS,
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"快照数据缺少字段: {sorted(missing)}")

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.dropna(subset=["date", "instrument"])
        data = data.sort_values(["instrument", "date"]).reset_index(drop=True)
        data["__trading_day"] = data["date"].dt.normalize()
        groups = data.groupby(
            ["instrument", "__trading_day"], sort=False, observed=True
        )

        for field in cls._CUMULATIVE_FIELDS:
            delta = groups[field].diff()
            # 首条记录或盘中累计值重置时，当前累计值就是有效增量。
            data[f"__{field}_delta"] = delta.where(
                delta.notna() & (delta >= 0), data[field]
            )

        data["__valid_price"] = data["price"].where(data["price"] > 0)
        data["__minute_end"] = cls._assign_minute_end(data["date"])
        data = data.dropna(subset=["__minute_end"])
        if data.empty:
            return pd.DataFrame(columns=output_columns)

        passthrough = [
            c
            for c in cls.schema.columns()
            if c
            not in {
                "date",
                "instrument",
                *cls._IDENTITY_FIELDS,
                *cls._PRICE_FIELDS,
                "deal_number",
                "volume",
                "amount",
            }
            and c in data.columns
        ]
        agg_spec = {
            "open": ("__valid_price", "first"),
            "high": ("__valid_price", "max"),
            "low": ("__valid_price", "min"),
            "close": ("__valid_price", "last"),
            "deal_number": ("__num_trades_delta", "sum"),
            "volume": ("__volume_delta", "sum"),
            "amount": ("__amount_delta", "sum"),
            **{field: (field, "last") for field in passthrough},
        }
        out = (
            data.groupby(["instrument", "__minute_end"], sort=True, observed=True)
            .agg(**agg_spec)
            .reset_index()
            .rename(columns={"__minute_end": "date"})
        )
        return out.reindex(columns=output_columns)

    def enrich_one_minute(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """一分钟行情生成后，再补充 instrument_id 和复权因子。"""
        if df.empty:
            return df.assign(instrument_id=pd.Series(dtype="int64"), adjust_factor=np.nan)

        instruments = df["instrument"].dropna().unique().tolist()
        instrument_ids = dai.query(
            "SELECT instrument, instrument_id FROM all_instruments",
            filters={"instrument": instruments},
        ).df()
        factors = dai.query(
            "SELECT date, instrument, adjust_factor FROM cn_stock_real_bar1d",
            filters={
                "date": [start_date, end_date],
                "instrument": instruments,
            },
        ).df()
        factors["trading_day"] = pd.to_datetime(factors.pop("date")).dt.normalize()

        out = df.copy()
        out["trading_day"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.merge(
            instrument_ids,
            how="left",
            on="instrument",
            validate="many_to_one",
        )
        out = out.merge(
            factors,
            how="left",
            on=["trading_day", "instrument"],
            validate="many_to_one",
        )
        return out.drop(columns="trading_day")

    @staticmethod
    def _hms_to_minute(hms: int) -> int:
        return (hms // 10000) * 60 + (hms // 100 % 100)

    def _assign_bar_end(self, df: pd.DataFrame) -> pd.Series:
        hms = df["date"].dt.strftime("%H%M%S").astype(int)
        day = df["date"].dt.normalize()
        end_minute = pd.Series(np.nan, index=df.index, dtype="float64")

        for series in TIME_SETS[self.K].values():
            for i in range(1, len(series)):
                start_time, end_time = series[i - 1], series[i]
                mask = (hms > start_time) & (hms <= end_time)
                end_minute.loc[mask] = self._hms_to_minute(end_time)
        return day + pd.to_timedelta(end_minute, unit="m")

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """将一分钟行情聚合为 K 分钟；09:25 集合竞价独立保留。"""
        if df.empty or self.K == 1:
            return df

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        auction = data.loc[data["date"].dt.strftime("%H%M%S") == "092500"].copy()
        data["__bar_end"] = self._assign_bar_end(data)
        data = data.dropna(subset=["__bar_end"]).sort_values(["instrument", "date"])

        excluded = {
            "date",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "deal_number",
        }
        last_fields = [c for c in data.columns if c not in excluded and c != "__bar_end"]
        agg_spec = {
            "open": ("open", "first"),
            "high": ("high", "max"),
            "low": ("low", "min"),
            "close": ("close", "last"),
            "volume": ("volume", "sum"),
            "amount": ("amount", "sum"),
            "deal_number": ("deal_number", "sum"),
            **{field: (field, "last") for field in last_fields},
        }
        out = (
            data.groupby(["instrument", "__bar_end"], sort=True, observed=True)
            .agg(**agg_spec)
            .reset_index()
            .rename(columns={"__bar_end": "date"})
        )
        if not auction.empty:
            out = pd.concat(
                [auction.reindex(columns=out.columns), out], ignore_index=True
            )
        return out.sort_values(["date", "instrument"]).reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        snapshots = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取快照耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(snapshots)}"
        )

        df = self.build_one_minute(snapshots)
        df = self.enrich_one_minute(df, self.start_date, self.end_date)
        t2 = datetime.now()
        print(
            f"一分钟构建及信息合并耗时: "
            f"{round((t2 - t1).total_seconds(), 4)} 秒, 行数: {len(df)}"
        )

        df = self.aggregate(df)
        t3 = datetime.now()
        print(f"K分钟聚合耗时: {round((t3 - t2).total_seconds(), 4)} 秒, 行数: {len(df)}")

        df = self.normalize(df)
        self.dai_write(df)
        t4 = datetime.now()
        print(f"数据存储耗时: {round((t4 - t3).total_seconds(), 4)} 秒")
        return df
