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
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.year.astype("int64")
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

    def get_financial_data(self) -> pd.DataFrame:
        """从财务表取 shift=0 的最新一期数据，计算 ROE

        向前多取 1 年，确保 start_date 附近的交易日有足够的历史公告可供 forward-fill。
        """
        fin_start = (pd.Timestamp(self.start_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        sql = f"""
        SELECT
            lf.date AS ann_date,
            lf.instrument,
            lf.report_date,
            lf.net_profit_to_parent_shareholders_lf,
            lf.total_equity_to_parent_shareholders_lf,
            ttm.net_profit_to_parent_shareholders_ttm
        FROM cn_stock_financial_lf_shift lf
        LEFT JOIN cn_stock_financial_ttm_shift ttm
            ON lf.date = ttm.date
            AND lf.instrument = ttm.instrument
            AND ttm.shift = 0
        WHERE lf.shift = 0
          AND lf.date >= '{fin_start}'
          AND lf.date <= '{self.end_date}'
        """
        df = dai.query(sql, filters={"date": [fin_start, self.end_date]}).df()

        df["roe_lf"] = (
            df["net_profit_to_parent_shareholders_lf"]
            / df["total_equity_to_parent_shareholders_lf"]
        )
        df["roe_ttm"] = (
            df["net_profit_to_parent_shareholders_ttm"]
            / df["total_equity_to_parent_shareholders_lf"]
        )
        return df[["ann_date", "instrument", "report_date", "roe_lf", "roe_ttm"]]

    def get_trading_days(self) -> pd.DataFrame:
        """获取指定区间内的全市场交易日"""
        sql = f"""
        SELECT DISTINCT date
        FROM cn_stock_bar1d
        WHERE date >= '{self.start_date}' AND date <= '{self.end_date}'
        ORDER BY date
        """
        return dai.query(sql, filters={"date": [self.start_date, self.end_date]}).df()

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()

        # 1. 获取财务数据并计算 ROE（全历史，供 forward-fill 使用）
        df_fin = self.get_financial_data()
        df_fin["ann_date"] = pd.to_datetime(df_fin["ann_date"])
        df_fin["report_date"] = pd.to_datetime(df_fin["report_date"])
        df_fin = df_fin.sort_values(["instrument", "ann_date"]).reset_index(drop=True)

        # 2. 获取目标区间交易日
        df_td = self.get_trading_days()
        df_td["date"] = pd.to_datetime(df_td["date"])

        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1 - t0).total_seconds(), 4)} 秒")

        # 3. 对每只股票，将公告日数据 merge_asof 到交易日（forward-fill）
        instruments = df_fin["instrument"].unique()
        pieces = []
        for inst in instruments:
            fin_inst = df_fin[df_fin["instrument"] == inst].copy()
            # merge_asof: 每个交易日匹配 <= 该日的最近公告日
            merged = pd.merge_asof(
                df_td.rename(columns={"date": "date"}),
                fin_inst.rename(columns={"ann_date": "ann_date"}),
                left_on="date",
                right_on="ann_date",
                direction="backward",
            )
            merged["instrument"] = inst
            pieces.append(merged)

        df = pd.concat(pieces, ignore_index=True)
        df = df[["date", "instrument", "report_date", "ann_date", "roe_lf", "roe_ttm"]]

        t2 = datetime.now()
        print(f"forward-fill 耗时: {round((t2 - t1).total_seconds(), 4)} 秒")

        # 4. normalize & 落库
        df = self.normalize(df)
        self.dai_write(df)

        t3 = datetime.now()
        print(f"数据存储耗时: {round((t3 - t2).total_seconds(), 4)} 秒")
        return df
