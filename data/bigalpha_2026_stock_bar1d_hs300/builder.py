import dai
import pandas as pd
from datetime import datetime
from base import BaseBuilder
from bigalpha_2026_stock_bar1d_hs300.schema import Bigalpha2026StockBar1dHs300Schema


class Bigalpha2026StockBar1dHs300Builder(BaseBuilder):
    datasource_id = "bigalpha_2026_stock_bar1d_hs300"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = Bigalpha2026StockBar1dHs300Schema

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        print(f"初始化构建器: {self.datasource_id} | 周期: {self.start_date} 至 {self.end_date}")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        df = df.fillna(self.schema.field_default_mapping())
        return df

    def dai_write(self, df: pd.DataFrame):
        default_docs = self.schema.default_docs()
        # 日频数据按年分区
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y").astype("int64")

        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )
        print(f"数据成功写入 BDB 数据源: {self.datasource_id}")

    def build(self) -> pd.DataFrame:
        print("开始构建数据...")
        t0 = datetime.now()

        # 1. 获取沪深300成分股列表
        hs300_sql = """
            SELECT DISTINCT date, member_code AS instrument, member_name AS name
            FROM cn_stock_index_component
            WHERE instrument = '000300.SH'
              AND date >= '{start_date}'
              AND date <= '{end_date}'
        """.format(start_date=self.start_date, end_date=self.end_date)

        df_components = dai.query(
            hs300_sql,
            filters={"date": [self.start_date, self.end_date]},
        ).df()

        # 2. 获取日行情数据
        bar1d_sql = """
            SELECT date, instrument, name,
                   open, high, low, close, pre_close,
                   volume, amount, deal_number,
                   change_ratio, turn, adjust_factor,
                   upper_limit, lower_limit
            FROM cn_stock_bar1d
            WHERE date >= '{start_date}'
              AND date <= '{end_date}'
        """.format(start_date=self.start_date, end_date=self.end_date)

        df_bar = dai.query(
            bar1d_sql,
            filters={"date": [self.start_date, self.end_date]},
        ).df()

        # 3. 按日期 + 成分股过滤，保留沪深300成分股当日行情
        df_components["date"] = pd.to_datetime(df_components["date"])
        df_bar["date"] = pd.to_datetime(df_bar["date"])

        df = pd.merge(
            df_components[["date", "instrument"]],
            df_bar,
            on=["date", "instrument"],
            how="inner",
        )

        t1 = datetime.now()
        print(f"获取与处理数据耗时: {round((t1 - t0).total_seconds(), 4)} 秒，共 {len(df)} 条记录")

        df_normalized = self.normalize(df)
        self.dai_write(df_normalized)

        t2 = datetime.now()
        print(f"数据落库存储耗时: {round((t2 - t1).total_seconds(), 4)} 秒")
        return df_normalized
