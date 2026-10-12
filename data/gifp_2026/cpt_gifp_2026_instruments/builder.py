from datetime import datetime

import dai
import pandas as pd

from base import BaseBuilder
from gifp_2026.cpt_gifp_2026_instruments.schema import (
    CptGifp2026InstrumentsSchema,
)


class CptGifp2026InstrumentsBuilder(BaseBuilder):
    """构建与 15 分钟收益标签一致的沪深 300 历史成分股截面。"""

    datasource_id = "cpt_gifp_2026_instruments"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptGifp2026InstrumentsSchema
    index_instrument = "000300.SH"
    # 必须与 cpt_gifp_2026_return 的标签时点完全一致。
    section_times = (
        "09:45:00",
        "10:00:00",
        "10:15:00",
        "10:30:00",
        "10:45:00",
        "11:00:00",
        "11:15:00",
        "11:30:00",
        "13:15:00",
        "13:30:00",
        "13:45:00",
        "14:00:00",
        "14:15:00",
        "14:30:00",
        "14:45:00",
        "15:00:00",
    )

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        print(
            f"初始化！{self.datasource_id}, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        return df.fillna(self.schema.field_default_mapping())

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

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        sql = f"""
        WITH index_components AS (
            SELECT date, member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '{self.index_instrument}'
        )
        SELECT c.date, c.instrument, i.name
        FROM index_components c
        LEFT JOIN cn_stock_instruments i
            ON c.date = i.date
            AND c.instrument = i.instrument
        ORDER BY c.date, c.instrument
        """
        return dai.query(sql, filters={"date": [start_date, end_date]}).df()

    @classmethod
    def expand_sections(cls, df: pd.DataFrame) -> pd.DataFrame:
        """将每日成分股展开到 16 个合法的 15 分钟因子截面。"""
        if df.empty:
            return df.copy()

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        data = data.dropna(subset=["date", "instrument"])
        data = data.drop_duplicates(["date", "instrument"], keep="last")
        sections = pd.DataFrame(
            {"__section_offset": pd.to_timedelta(cls.section_times)}
        )
        data = data.merge(sections, how="cross")
        data["date"] = data["date"] + data.pop("__section_offset")
        return data.sort_values(["date", "instrument"]).reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取股票池耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(df)}"
        )

        if df.empty:
            print("股票池为空，跳过入库")
            return pd.DataFrame(columns=self.schema.columns())

        df = self.expand_sections(df)
        if df.empty:
            print("股票池为空，跳过入库")
            return pd.DataFrame(columns=self.schema.columns())

        df = self.normalize(df)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2 - t1).total_seconds(), 4)} 秒")
        return df
