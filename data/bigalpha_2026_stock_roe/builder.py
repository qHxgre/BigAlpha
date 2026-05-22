import dai
import pandas as pd
from datetime import datetime

from base import BaseBuilder
from bigalpha_2026_stock_roe.schema import Bigalpha2026StockRoeSchema


class Bigalpha2026StockRoeBuilder(BaseBuilder):
    datasource_id = "bigalpha_2026_stock_roe"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = Bigalpha2026StockRoeSchema

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
        # 交易日历：取全市场每个交易日的 instrument 列表
        sql_bar = """
        SELECT date, instrument
        FROM cn_stock_bar1d
        """
        df_bar = dai.query(sql_bar, filters={"date": [start_date, end_date]}).df()

        # lf：最新一期归母净利润、归母净资产（shift=0）
        sql_lf = """
        SELECT
            date,
            instrument,
            report_date,
            net_profit_to_parent_shareholders_lf AS net_profit_lf,
            total_equity_to_parent_shareholders_lf AS total_equity_lf
        FROM cn_stock_financial_lf_shift
        WHERE shift = 0
        """
        df_lf = dai.query(sql_lf, filters={"date": [start_date, end_date]}).df()

        # ttm：归母净利润 TTM（shift=0）
        sql_ttm = """
        SELECT
            date,
            instrument,
            net_profit_to_parent_shareholders_ttm AS net_profit_ttm
        FROM cn_stock_financial_ttm_shift
        WHERE shift = 0
        """
        df_ttm = dai.query(sql_ttm, filters={"date": [start_date, end_date]}).df()

        # 合并财务数据
        df_fin = df_lf.merge(df_ttm, on=["date", "instrument"], how="left")

        # 以交易日历为基准，将财务数据 forward-fill 到每个交易日
        df_bar = df_bar.sort_values(["instrument", "date"])
        df_fin = df_fin.sort_values(["instrument", "date"])

        df = df_bar.merge(df_fin, on=["date", "instrument"], how="left")
        df = df.sort_values(["instrument", "date"])
        fin_cols = ["report_date", "net_profit_lf", "total_equity_lf", "net_profit_ttm"]
        df[fin_cols] = df.groupby("instrument")[fin_cols].ffill()

        return df

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        # 计算 ROE，分母为零时置 NaN 避免 inf
        df["roe_lf"] = df["net_profit_lf"] / df["total_equity_lf"].replace(0, float("nan"))
        df["roe_ttm"] = df["net_profit_ttm"] / df["total_equity_lf"].replace(0, float("nan"))

        df = self.normalize(df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2-t1).total_seconds(), 4)} 秒")
        return df
