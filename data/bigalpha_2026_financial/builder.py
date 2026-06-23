import dai
import pandas as pd
from datetime import datetime

from base import BaseBuilder
from bigalpha_2026_financial.schema import BigAlpha2026FinancialSchema


class Bigalpha2026FinancialBuilder(BaseBuilder):
    datasource_id = "bigalpha_2026_financial"
    unique_together = ["date", "instrument", "report_date", "shift", "category"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = BigAlpha2026FinancialSchema

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date

        # 股票池：2019年至今的中证1000指数成分
        self.instruments = dai.query("""
        SELECT date, member_code
        FROM cn_stock_index_component
        WHERE instrument='000852.SH'
        AND date>'2019-01-01'
        """).df()['member_code'].unique().tolist()

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

    def get_data(self) -> pd.DataFrame:
        sql = "SELECT * FROM cn_stock_financial_cleaned"

        df = dai.query(sql, filters={
            "date": [self.start_date, self.end_date],
            "instrument": self.instruments
        }).df()
        return df

    def build(self) -> pd.DataFrame:
        # 读取数据
        t0 = datetime.now()
        df = self.get_data()
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        # 存储数据
        df = self.normalize(df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2-t1).total_seconds(), 4)} 秒")
        return df
