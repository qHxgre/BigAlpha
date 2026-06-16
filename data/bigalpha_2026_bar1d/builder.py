import dai
import pandas as pd
from datetime import datetime

from base import BaseBuilder
from bigalpha_2026_bar1d.schema import Bigalpha2026Bar1dSchema


class Bigalpha2026Bar1dBuilder(BaseBuilder):
    """包括指数和个股"""
    datasource_id = "bigalpha_2026_bar1d"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = Bigalpha2026Bar1dSchema

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
        """该数据主要是算收益率，因此要成分股全时段的数据"""
        # 获取股票池
        instruments = dai.query("SELECT date, member_code FROM cn_stock_index_component WHERE instrument = '000852.SH'").df()['member_code'].unique().tolist()
        # 获取个股后复权价格
        sql = """
        SELECT
            date, instrument, name, adjust_factor,
            pre_close, high, open, low, close,
            volume, amount, change_ratio, turn
        FROM cn_stock_bar1d
        """
        stk_df = dai.query(sql, filters={"date": [start_date, end_date], instruments: instruments}).df()

        # 获取指数价格
        sql = """
        SELECT
            date, instrument, name, adjust_factor,
            pre_close, high, open, low, close,
            volume, amount, change_ratio, turn
        FROM cn_stock_index_bar1d
        """
        index_df = dai.query(sql, filters={
            "date": [start_date, end_date],
            instruments: ["000905.SH", "000852.SH", "000300.SH"]
        }).df()

        df = pd.concat([stk_df, index_df], axis=0)
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
