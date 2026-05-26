import dai
import pandas as pd
from datetime import datetime

from base import BaseBuilder
from xiaoze_l2_snapshot.schema import XiaozeL2SnapshotSchema


class XiaozeL2SnapshotBuilder(BaseBuilder):
    datasource_id = "xiaoze_l2_snapshot"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = XiaozeL2SnapshotSchema

    # 股票池（10 只，已补全交易所后缀）
    INSTRUMENTS = [
        "000333.SZ",  # 美的集团
        "300048.SZ",  # 合康新能
        "003042.SZ",  # 中农联合
        "300383.SZ",  # 光环新网
        "002121.SZ",  # 科陆电子
        "600055.SH",  # 华润双鹤
        "603929.SH",  # 亚翔集成
        "301611.SZ",  # 矽电股份
        "688668.SH",  # 鼎通科技
        "600578.SH",  # 京能电力
    ]

    # 抽取字段：基础行情 + 5 档盘口 + 累计成交
    SELECT_COLUMNS = """
        date, instrument, trading_day, time,
        pre_close, open, high, low, price,
        ask_price1, ask_price2, ask_price3, ask_price4, ask_price5,
        bid_price1, bid_price2, bid_price3, bid_price4, bid_price5,
        ask_volume1, ask_volume2, ask_volume3, ask_volume4, ask_volume5,
        bid_volume1, bid_volume2, bid_volume3, bid_volume4, bid_volume5,
        num_trades, volume, amount
    """

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
        # 按日分区：%Y%m%d
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y%m%d").astype("int64")
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

    def fetch_one_day(self, trading_day: int) -> pd.DataFrame:
        """按单个交易日抽取该股票池的 Level2 快照。"""
        sql = f"""
        SELECT {self.SELECT_COLUMNS}
        FROM cn_stock_level2_snapshot
        """
        d = pd.to_datetime(str(trading_day), format="%Y%m%d")
        next_d = d + pd.Timedelta(days=1)
        df = dai.query(
            sql,
            filters={
                "date": [d.strftime("%Y-%m-%d 00:00:00"), next_d.strftime("%Y-%m-%d 00:00:00")],
                "instrument": self.INSTRUMENTS,
            },
            compression=True,
        ).df()
        return df

    def build(self) -> None:
        t0 = datetime.now()
        date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        total_rows = 0
        wrote_days = 0

        for d in date_range:
            trading_day = int(d.strftime("%Y%m%d"))
            t_start = datetime.now()
            df = self.fetch_one_day(trading_day)
            if df.empty:
                print(f"[{trading_day}] 非交易日或无数据，跳过")
                continue

            df = self.normalize(df)
            self.dai_write(df)

            total_rows += len(df)
            wrote_days += 1
            cost = round((datetime.now() - t_start).total_seconds(), 2)
            print(f"[{trading_day}] 写入 {len(df):>7} 行，耗时 {cost} 秒")

            # 主动释放，控制内存
            del df

        t1 = datetime.now()
        print(
            f"全部完成：{wrote_days} 个交易日，累计 {total_rows} 行，"
            f"总耗时 {round((t1 - t0).total_seconds(), 2)} 秒"
        )
