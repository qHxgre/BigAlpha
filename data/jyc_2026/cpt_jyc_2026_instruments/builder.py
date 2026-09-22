import dai
import pandas as pd
from datetime import datetime

from base import BaseBuilder
from jyc_2026.cpt_jyc_2026_instruments.schema import CptJyc2026InstrumentsSchema


class CptJyc2026InstrumentsBuilder(BaseBuilder):
    """构建与 30 分钟 VWAP 标签一致的股票池截面。"""

    datasource_id = "cpt_jyc_2026_instruments"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptJyc2026InstrumentsSchema

    # 与 cpt_jyc_2026_vwap 的标签时点保持一致。11:30 标签的未来窗口
    # 跨过午休，因此下一个标签直接从 13:30 开始。
    _SECTION_TIMES = (
        "09:30:00",
        "10:00:00",
        "10:30:00",
        "11:00:00",
        "11:30:00",
        "13:30:00",
        "14:00:00",
        "14:30:00",
    )

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
        df[dai.DEFAULT_PARTITION_FIELD] = (
            df["date"].dt.strftime("%Y%m").astype("int64")
        )
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        sql = """
        WITH cte_index AS (
            SELECT date, member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
        )
        SELECT i.date, i.instrument, i.name
        FROM cn_stock_instruments i
        INNER JOIN cte_index c
            ON i.date = c.date
            AND i.instrument = c.instrument
        """

        df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
        return df

    @classmethod
    def expand_sections(cls, df: pd.DataFrame) -> pd.DataFrame:
        """将每日股票池展开到 VWAP 使用的 8 个日内截面。"""
        if df.empty:
            return df.copy()

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"]).dt.normalize()
        sections = pd.DataFrame(
            {"__section_offset": pd.to_timedelta(cls._SECTION_TIMES)}
        )
        data = data.merge(sections, how="cross")
        data["date"] = data["date"] + data.pop("__section_offset")
        return data.sort_values(["date", "instrument"]).reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        # 读取数据
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        # 展开为与未来 30 分钟 VWAP 标签相同的日内截面
        df = self.expand_sections(df)

        # 存储数据
        df = self.normalize(df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2-t1).total_seconds(), 4)} 秒")
        return df
