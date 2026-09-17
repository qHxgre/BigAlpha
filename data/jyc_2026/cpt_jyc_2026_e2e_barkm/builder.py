import dai
import pandas as pd

from cpt_jyc_2026_stock_barkm.builder import BarKmBuilder
from cpt_jyc_2026_e2e_barkm.schema import CptJyc2026E2EBarKmSchema


class CptJyc2026E2EBarKmBuilder(BarKmBuilder):
    """K 分钟 K 线构建器

    按日成分股筛选分钟数据, 共用 stock_barkm 中的分箱和构建流程,
    保留三档盘口及以分存储的整数价格。
    """

    datasource_prefix = "cpt_jyc_2026_e2e"
    group_key = "instrument_id"
    unique_together = ["date", "instrument_id"]
    sort_by = [("date", "ascending"), ("instrument_id", "ascending")]
    indexes = ["date"]
    schema = CptJyc2026E2EBarKmSchema

    # 以"分"(元×100)存储的价格/金额列: 入库前 ×100 取整为 int, 读取时 /100 还原
    SCALE_FIELDS = [
        "open", "high", "low", "close", "amount",
        "ask_price1", "ask_price2", "ask_price3",
        "bid_price1", "bid_price2", "bid_price3",
    ]
    PRICE_SCALE = 100

    def __init__(self, start_date: str, end_date: str, K: int = 1, suffix: str=None) -> None:
        super().__init__(start_date, end_date, K, suffix)
        
        # 股票池：2019年至今的中证1000指数成分
        self.instruments_df = dai.query("SELECT date, member_code FROM cn_stock_index_component", 
            filters={'date': [self.start_date, self.end_date], 'instrument': ['932000.CSI']}).df()
        self.instruments_df = self.instruments_df.rename(columns={'member_code': 'instrument', 'date': 'trading_day'})
        self.instruments_df['trading_day'] = self.instruments_df['trading_day'].dt.strftime('%Y-%m-%d')
        self.instruments_df['trading_day'] = self.instruments_df['trading_day'].str.replace('-', '').astype(int)
        self.instruments = self.instruments_df['instrument'].unique().tolist()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # 按 schema 确定列名和顺序
        df = df.reindex(columns=self.schema.columns())

        # 价格/金额列 ×100 取整为"分"。此时仍是 float, NaN 保持 NaN,
        # 不会污染后续填充的哨兵值(round 对 NaN 返回 NaN)。
        for c in self.SCALE_FIELDS:
            if c in df.columns:
                df[c] = (df[c] * self.PRICE_SCALE).round()

        # 必须先 fillna 再 astype: 整型列不接受 NaN, 若先 astype 会在
        # 含缺失的列上直接抛 "Cannot convert non-finite values to integer"。
        df = df.fillna(self.schema.field_default_mapping())
        df = df.astype(self.schema.field_type_mapping())
        return df

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        # 复权因子
        sql = """
        SELECT date as trading_day, instrument, adjust_factor
        FROM cn_stock_real_bar1d
        """
        dff = dai.query(sql, filters={
            "date": [start_date, end_date],
            "instrument": self.instruments
        }).df()
        dff['trading_day'] = dff['trading_day'].dt.strftime('%Y-%m-%d')
        dff['trading_day'] = dff['trading_day'].str.replace('-', '').astype(int)

        # 分钟数据
        sql = """
        SELECT
            b.date, b.instrument, b.trading_day,
            all_instruments.instrument_id,
            b.high, b.open, b.low, b.close, b.deal_number, b.volume, b.amount,
            b.ask_price1, b.ask_price2, b.ask_price3,
            b.bid_price1, b.bid_price2, b.bid_price3,
            b.ask_volume1, b.ask_volume2, b.ask_volume3,
            b.bid_volume1, b.bid_volume2, b.bid_volume3,
            b.ask_num_orders1, b.ask_num_orders2, b.ask_num_orders3,
            b.bid_num_orders1, b.bid_num_orders2, b.bid_num_orders3
        FROM cn_stock_bar1m_derived_c b
        LEFT JOIN all_instruments USING (instrument)
        """
        df = dai.query(sql, filters={
            "date": [f"{start_date} 00:00:00", f"{end_date} 23:59:59"],
            "instrument": self.instruments
        }).df()

        # 合并数据
        result = pd.merge(self.instruments_df, df, how='left', on=['trading_day', 'instrument'])
        result = pd.merge(result, dff, how='left', on=['trading_day', 'instrument'])

        # 去掉列
        result = result.drop(['trading_day', 'instrument'], axis=1)
        return result
