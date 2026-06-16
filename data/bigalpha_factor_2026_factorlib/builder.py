import dai
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from base import BaseBuilder
from bigalpha_factor_2026_factorlib.schema import Bigalpha2026FactorlibSchema


class BigalphaFactor2026FactorlibBuilder(BaseBuilder):
    datasource_id = "bigalpha_factor_2026_factorlib"
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


    def process_data(self, df: pd.DataFrame):
        """对因子值进行预处理"""
        def _build_normalize_sql(col: str, group_by: str = "date") -> str:
            avg_expr  = f"c_avg({col}, pb:={group_by})"
            std_expr  = f"c_std({col}, pb:={group_by})"
            clip_expr = (
                f"clip({col}, "
                f"{avg_expr} - 3*{std_expr}, "
                f"{avg_expr} + 3*{std_expr}"
                f") as _{col}"
            )
            norm_expr = f"c_normalize(_{col}, pb:={group_by}) as {col}"
            return clip_expr + ",\n" + norm_expr

        NORMALIZE_COLS = [
            # 行情基础
            "close", "volume", "amount", "turn", "change_ratio", "daily_return",
            # 动量/波动
            "momentum_5", "reversal_5", "volatility_5",
            # 市值/估值
            "total_market_cap", "float_market_cap", "pe_ttm", "pb", "ps_ttm",
            # 均线/MACD
            "sma_20", "ema_20",
            "macd_diff_12_26_9", "macd_dea_12_26_9", "macd_hist_12_26_9", "rsi_12",
            # 其他技术指标
            "kdj_k_9_3_3", "kdj_d_9_3_3", "bias_20", "cci_14", "atr_14",
            # 基本面
            "roe_avg_ttm", "roa_avg_ttm", "gross_profit_rate_ttm", "net_profit_rate_ttm",
            "debt_to_asset_lf", "current_ratio_lf",
            # 资金流/其他
            "netflow_amount_main", "netflow_amount_rate_main",
            "net_active_buy_amount_main", "beta_000300SH_22", "list_days",
        ]

        for col in NORMALIZE_COLS:
            # 替换 inf 为 nan，再填充或丢弃
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        factor_sql = ",\n".join(_build_normalize_sql(col) for col in NORMALIZE_COLS)
        sql = f"""
        SELECT 
            date, instrument,
            {factor_sql}
        FROM factor_data
        """
        process_df = dai.query(sql, bind_relations={"factor_data": df}).df()
        return process_df

    def build(self) -> pd.DataFrame:
        # 读取数据
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        process_df = self.process_data(df)
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        # 存储数据
        df = self.normalize(process_df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2-t1).total_seconds(), 4)} 秒")
        return df
