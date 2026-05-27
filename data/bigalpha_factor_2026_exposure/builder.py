import dai
import pandas as pd
from datetime import datetime

from base import BaseBuilder
from bigalpha_factor_2026_exposure.schema import BigalphaFactor2026ExposureSchema


class BigalphaFactor2026ExposureBuilder(BaseBuilder):
    datasource_id = "bigalpha_factor_2026_exposure"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = BigalphaFactor2026ExposureSchema

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
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y").astype("int64")
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
            SELECT date, member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
        )
        SELECT e.*
        FROM bq_exposure e
        INNER JOIN cte_index c
            ON e.date = c.date
            AND e.instrument = c.instrument
        """

        df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
        return df

    def build(self) -> pd.DataFrame:
        # 读取数据
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        # 存储数据
        df = self.normalize(df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2-t1).total_seconds(), 4)} 秒")
        return df
