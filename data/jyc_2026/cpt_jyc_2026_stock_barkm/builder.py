from datetime import datetime

import dai
import numpy as np
import pandas as pd

from base import BaseBuilder
from jyc_2026.cpt_jyc_2026_stock_barkm.constant import TIME_SETS
from jyc_2026.cpt_jyc_2026_stock_barkm.kminute_builder import KMinuteBarBuilder
from jyc_2026.cpt_jyc_2026_stock_barkm.minute_builder import OneMinuteBarBuilder
from jyc_2026.cpt_jyc_2026_stock_barkm.schema import CptJyc2026StockBarKmSchema


class _CptJyc2026StockBarBaseBuilder(BaseBuilder):
    """股票分钟行情构建器的公共读写配置。"""

    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptJyc2026StockBarKmSchema

    _MIN_DATE = "2020-01-01"

    def _set_date_range(self, start_date: str, end_date: str) -> None:
        self.start_date = max(start_date, self._MIN_DATE)
        self.end_date = end_date

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


class CptJyc2026StockBar1mBuilder(_CptJyc2026StockBarBaseBuilder):
    """从 Level-2 snapshot 构建并写入 1 分钟行情。"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        suffix: str = None,
    ) -> None:
        self._set_date_range(start_date, end_date)
        name = "cpt_jyc_2026_stock_bar1m"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        print(
            f"初始化！{self.datasource_id}, 频率: 1分钟, "
            f"时间周期: {self.start_date}, {self.end_date}"
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
    def build_one_minute(df: pd.DataFrame) -> pd.DataFrame:
        return OneMinuteBarBuilder.build(df)

    @staticmethod
    def enrich_one_minute(
        df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """补充 instrument_id 和复权因子。"""
        if df.empty:
            return df.assign(
                instrument_id=pd.Series(dtype="int64"), adjust_factor=np.nan
            )

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

        df = self.normalize(df)
        self.dai_write(df)
        t3 = datetime.now()
        print(f"数据存储耗时: {round((t3 - t2).total_seconds(), 4)} 秒")
        return df


class CptJyc2026StockBarKmBuilder(_CptJyc2026StockBarBaseBuilder):
    """从已构建的 1 分钟数据聚合并写入 K 分钟行情。"""

    source_datasource_id = "cpt_jyc_2026_stock_bar1m"

    def __init__(
        self,
        start_date: str,
        end_date: str,
        K: int,
        suffix: str = None,
    ) -> None:
        self._set_date_range(start_date, end_date)
        self.K = int(K)
        supported = sorted(k for k in TIME_SETS if k != 1)
        if self.K not in supported:
            raise ValueError(
                f"不支持的聚合频率 K={self.K}, 可选: {supported}; "
                "1 分钟数据请使用 CptJyc2026StockBar1mBuilder"
            )

        name = f"cpt_jyc_2026_stock_bar{self.K}m"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        print(
            f"初始化！{self.datasource_id}, 频率: {self.K}分钟, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """读取第一阶段已经构建并落库的 1 分钟行情。"""
        sql = f"SELECT * FROM {self.source_datasource_id}"
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

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return KMinuteBarBuilder.build(df, self.K)

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取1分钟数据耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(df)}"
        )

        df = self.aggregate(df)
        t2 = datetime.now()
        print(
            f"{self.K}分钟聚合耗时: "
            f"{round((t2 - t1).total_seconds(), 4)} 秒, 行数: {len(df)}"
        )

        df = self.normalize(df)
        self.dai_write(df)
        t3 = datetime.now()
        print(f"数据存储耗时: {round((t3 - t2).total_seconds(), 4)} 秒")
        return df
