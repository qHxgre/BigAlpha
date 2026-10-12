from datetime import datetime

import dai
import pandas as pd

from base import BaseBuilder
from gifp_2026.cpt_gifp_2026_instruments.schema import (
    CptGifp2026InstrumentsSchema,
)


class CptGifp2026InstrumentsBuilder(BaseBuilder):
    """构建沪深 300 每日历史成分股列表。

    收益标签口径尚未确定，因此这里只保留日频股票池，不提前展开为日内
    采样截面。标签时点确定后，可在下游按交易日与本表连接。
    """

    datasource_id = "cpt_gifp_2026_instruments"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptGifp2026InstrumentsSchema
    index_instrument = "000300.SH"

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        print(
            f"初始化！{self.datasource_id}, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        return df.fillna(self.schema.field_default_mapping())

    def dai_write(self, df: pd.DataFrame) -> None:
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y").astype(
            "int64"
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
        sql = f"""
        WITH index_components AS (
            SELECT date, member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '{self.index_instrument}'
        )
        SELECT c.date, c.instrument, i.name
        FROM index_components c
        LEFT JOIN cn_stock_instruments i
            ON c.date = i.date
            AND c.instrument = i.instrument
        ORDER BY c.date, c.instrument
        """
        return dai.query(sql, filters={"date": [start_date, end_date]}).df()

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取股票池耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(df)}"
        )

        if df.empty:
            print("股票池为空，跳过入库")
            return pd.DataFrame(columns=self.schema.columns())

        df = self.normalize(df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2 - t1).total_seconds(), 4)} 秒")
        return df
