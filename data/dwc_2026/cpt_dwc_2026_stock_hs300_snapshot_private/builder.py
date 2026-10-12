import dai
import numpy as np
import pandas as pd
from datetime import datetime

from warehouse.builder.base import BaseBuilder
from snapshot_private.schema import CompetitionSnapshotSchema


class CompetitionSnapshotPrivateBuilder(BaseBuilder):
    datasource_id = "cpt_dwc_2026_stock_hs300_snapshot_private"
    unique_together = ["date", "instrument_id"]
    sort_by = [("date", "ascending"), ("instrument_id", "ascending")]
    indexes = ["date"]
    schema = CompetitionSnapshotSchema

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        print(f"初始化！{self.datasource_id}, 时间周期: {self.start_date}, {self.end_date}")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        df = df.fillna(self.schema.field_default_mapping())
        return df

    def dai_write(self, df: pd.DataFrame):
        default_docs = self.schema.default_docs()
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y%m%d").astype("int64")
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        sql = """
        WITH cte_index AS (
            SELECT CAST(strftime(date, '%Y%m%d') AS INT32) AS trading_day, member_code as instrument
            FROM cn_stock_index_component
            WHERE instrument = '000300.SH'
        ),
        cte_snapshot AS (
            SELECT s.*
            FROM cn_stock_level2_snapshot s
            INNER JOIN cte_index
                ON s.trading_day = cte_index.trading_day
                AND s.instrument = cte_index.instrument
        )
        SELECT
            cte_snapshot.* exclude(instrument, trading_day, time),
            all_instruments.instrument_id,
        FROM all_instruments
        JOIN cte_snapshot USING (instrument)
        """

        df = dai.query(sql, filters={"date": [f"{start_date} 00:00:00", f"{end_date} 23:59:59"]}, compression=True).df()
        use_meomory = df.memory_usage(deep=True).sum()
        print(f"股票数量: {len(df['instrument_id'].unique().tolist())} 原始数据内存占比: {use_meomory / 1024 ** 3:.2f} GB")
        return df

    def build(self) -> pd.DataFrame:
        """读取 snapshot 数据"""
        # 读取数据
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        # 存储数据
        df = self.normalize(df)
        use_meomory = df.memory_usage(deep=True).sum()
        print(f"待存储数据内存占比: {use_meomory / 1024 ** 3:.2f} GB")
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2-t1).total_seconds(), 4)} 秒")
