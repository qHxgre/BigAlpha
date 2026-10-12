import dai
import pandas as pd
from pathlib import Path

from warehouse.builder.base import BaseBuilder
from factorslib.schema import DeepWinFactorsSchema

class DeepWinFactorsBuilder(BaseBuilder):
    datasource_id = "cpt_dwc_factorlib"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = DeepWinFactorsSchema

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        print(f"初始化！{self.datasource_id}")

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

    def get_history_data(self) -> pd.DataFrame:
        """获取历史数据"""
        folder_path = Path('/home/aiuser/work/项目支持/量化比赛/蝶威资产/data/factorslib/facs')
        parquet_files = list(folder_path.glob('*.pq'))

        data_frames = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                df = df.T.stack().reset_index()
                df.columns = ["date", "instrument", file_path.stem]
                data_frames.append(df)
            except Exception as e:
                print(f"读取文件 {file_path.name} 时出错: {e}")
                continue

        # 合并所有数据
        if len(data_frames) == 1:
            data = data_frames[0]
        else:
            from functools import reduce
            data = reduce(
                lambda left, right: pd.merge(left, right, how="left", on=["date", "instrument"]),
                data_frames
            )
        return data

    def build(self) -> pd.DataFrame:
        data = self.get_history_data()
        normalized_df = self.normalize(data)
        normalized_df = normalized_df[(normalized_df['date']>=self.start_date)&(normalized_df['date']<=self.end_date)]
        self.dai_write(normalized_df)
