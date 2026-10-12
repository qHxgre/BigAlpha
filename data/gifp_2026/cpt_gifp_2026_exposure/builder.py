from datetime import datetime

import dai
import pandas as pd

from base import BaseBuilder
from gifp_2026.cpt_gifp_2026_exposure.schema import CptGifp2026ExposureSchema


class CptGifp2026ExposureBuilder(BaseBuilder):
    """构建沪深 300 每日 BARRA 风格与行业暴露。"""

    datasource_id = "cpt_gifp_2026_exposure"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptGifp2026ExposureSchema
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
        SELECT e.*
        FROM bq_exposure e
        INNER JOIN index_components c
            ON e.date = c.date
            AND e.instrument = c.instrument
        ORDER BY e.date, e.instrument
        """
        return dai.query(sql, filters={"date": [start_date, end_date]}).df()

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取风险暴露耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(df)}"
        )

        if df.empty:
            print("风险暴露为空，跳过入库")
            return pd.DataFrame(columns=self.schema.columns())

        df = self.normalize(df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2 - t1).total_seconds(), 4)} 秒")
        return df
