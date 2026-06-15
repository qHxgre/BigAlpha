import dai
import pandas as pd
from datetime import datetime, timedelta

from base import BaseBuilder
from bigalpha_2026_factorlib.schema import Bigalpha2026FactorlibSchema


class Bigalpha2026FactorlibBuilder(BaseBuilder):
    datasource_id = "bigalpha_2026_factorlib"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = Bigalpha2026FactorlibSchema

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
        before_start_date = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        sql = """
        WITH cte_index AS (
            SELECT date, member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
        )
        SELECT 
            i.date, i.instrument,
            i.close,
            i.volume,
            i.amount,
            i.turn,
            i.change_ratio,
            i.daily_return,
            i.momentum_5,
            i.reversal_5,
            i.volatility_5,
            i.total_market_cap,
            i.float_market_cap,
            i.pe_ttm,
            i.pb,
            i.ps_ttm,
            i.sma_20,
            i.ema_20,
            i.macd_diff_12_26_9,
            i.macd_dea_12_26_9,
            i.macd_hist_12_26_9,
            i.rsi_12,
            i.kdj_k_9_3_3,
            i.kdj_d_9_3_3,
            i.bias_20,
            i.cci_14,
            i.atr_14,
            i.roe_avg_ttm,
            i.roa_avg_ttm,
            i.gross_profit_rate_ttm,
            i.net_profit_rate_ttm,
            i.debt_to_asset_lf,
            i.current_ratio_lf,
            i.netflow_amount_main,
            i.netflow_amount_rate_main,
            i.net_active_buy_amount_main,
            i.beta_000300SH_22,
            i.list_days,
        FROM cn_stock_prefactors i
        INNER JOIN cte_index c
            ON i.date = c.date
            AND i.instrument = c.instrument
        """

        df = dai.query(sql, filters={"date": [before_start_date, end_date]}).df()
        df = df[(df['date']>=self.start_date) & (df['date']<=self.end_date)]
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
