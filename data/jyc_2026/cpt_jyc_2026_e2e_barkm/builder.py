from datetime import datetime

import dai
import pandas as pd

from base import BaseBuilder
from jyc_2026.cpt_jyc_2026_e2e_barkm.schema import CptJyc2026E2EBarKmSchema


class CptJyc2026E2EBarKmBuilder(BaseBuilder):
    """直接从 stock K 分钟行情生成整数化的 E2E 行情。"""

    unique_together = ["date", "instrument_id"]
    sort_by = [("date", "ascending"), ("instrument_id", "ascending")]
    indexes = ["date"]
    schema = CptJyc2026E2EBarKmSchema

    SUPPORTED_FREQUENCIES = {1, 5, 15, 30}
    SCALE_FIELDS = [
        "open",
        "high",
        "low",
        "close",
        "amount",
        "ask_price1",
        "ask_price2",
        "ask_price3",
        "bid_price1",
        "bid_price2",
        "bid_price3",
    ]
    PRICE_SCALE = 100

    def __init__(
        self,
        start_date: str,
        end_date: str,
        K: int = 1,
        suffix: str = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.K = int(K)
        if self.K not in self.SUPPORTED_FREQUENCIES:
            raise ValueError(
                f"不支持的频率 K={self.K}, "
                f"可选: {sorted(self.SUPPORTED_FREQUENCIES)}"
            )

        self.source_datasource_id = f"cpt_jyc_2026_stock_bar{self.K}m"
        name = f"cpt_jyc_2026_e2e_bar{self.K}m"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        print(
            f"初始化！{self.datasource_id}, 源数据: {self.source_datasource_id}, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    @staticmethod
    def get_daily_instruments(trading_day: str) -> list:
        """获取中证 1000 当日成分股。"""
        components = dai.query(
            """
            SELECT member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
            """,
            filters={"date": [trading_day, trading_day]},
        ).df()
        if components.empty:
            return []
        return components["instrument"].dropna().unique().tolist()

    def get_data(
        self,
        trading_day: str,
        instruments: list,
    ) -> pd.DataFrame:
        """读取指定交易日、指定成分股的目标字段。"""
        if not instruments:
            return pd.DataFrame(columns=self.schema.columns())

        fields = ", ".join(self.schema.columns())
        return dai.query(
            f"SELECT {fields} FROM {self.source_datasource_id}",
            filters={
                "date": [
                    f"{trading_day} 00:00:00",
                    f"{trading_day} 23:59:59",
                ],
                "instrument": instruments,
            },
            compression=True,
        ).df()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns()).copy()
        for field in self.SCALE_FIELDS:
            df[field] = (df[field] * self.PRICE_SCALE).round()

        # 先补默认值再转整数，避免 NaN 无法转换为整型。
        df = df.fillna(self.schema.field_default_mapping())
        return df.astype(self.schema.field_type_mapping())

    def dai_write(self, df: pd.DataFrame) -> None:
        df[dai.DEFAULT_PARTITION_FIELD] = (
            df["date"].dt.strftime("%Y%m").astype("int64")
        )
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=self.schema.default_docs(),
        )

    def build(self) -> pd.DataFrame:
        start_date = pd.Timestamp(self.start_date).normalize()
        end_date = pd.Timestamp(self.end_date).normalize()
        if start_date > end_date:
            raise ValueError("start_date 不能晚于 end_date")

        all_daily_data = []
        built_days = 0
        total_start = datetime.now()

        for day in pd.date_range(start_date, end_date, freq="D"):
            day_str = day.strftime("%Y-%m-%d")
            day_start = datetime.now()
            instruments = self.get_daily_instruments(day_str)
            component_end = datetime.now()

            if not instruments:
                print(f"[{day_str}] 无中证1000成分股数据，跳过")
                continue

            df = self.get_data(day_str, instruments)
            read_end = datetime.now()
            print(
                f"[{day_str}] 成分股数: {len(instruments)}, "
                f"查询成分耗时: {round((component_end - day_start).total_seconds(), 4)} 秒, "
                f"获取{self.K}分钟数据耗时: "
                f"{round((read_end - component_end).total_seconds(), 4)} 秒, "
                f"行数: {len(df)}"
            )

            if df.empty:
                print(f"[{day_str}] 成分股源行情为空，跳过转换和入库")
                continue

            df = self.normalize(df)
            normalize_end = datetime.now()
            self.dai_write(df)
            write_end = datetime.now()
            print(
                f"[{day_str}] 字段整数化耗时: "
                f"{round((normalize_end - read_end).total_seconds(), 4)} 秒, "
                f"数据存储耗时: "
                f"{round((write_end - normalize_end).total_seconds(), 4)} 秒"
            )
            all_daily_data.append(df)
            built_days += 1

        total_end = datetime.now()
        print(
            f"构建完成，成功入库交易日数: {built_days}, "
            f"总耗时: {round((total_end - total_start).total_seconds(), 4)} 秒"
        )
        if not all_daily_data:
            return pd.DataFrame(columns=self.schema.columns())
        return pd.concat(all_daily_data, ignore_index=True)
