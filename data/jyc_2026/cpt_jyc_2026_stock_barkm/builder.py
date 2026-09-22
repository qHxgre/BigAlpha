from datetime import datetime

import dai
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from base import BaseBuilder
from jyc_2026.cpt_jyc_2026_stock_barkm.constant import TIME_SETS
from jyc_2026.cpt_jyc_2026_stock_barkm.kminute_builder import KMinuteBarBuilder
from jyc_2026.cpt_jyc_2026_stock_barkm.minute_builder import OneMinuteBarBuilder
from jyc_2026.cpt_jyc_2026_stock_barkm.schema import CptJyc2026StockBarKmSchema


class _CptJyc2026StockBarBaseBuilder(BaseBuilder):
    """股票分钟行情构建器的公共读写配置。"""

    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptJyc2026StockBarKmSchema

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


class CptJyc2026StockBar1mBuilder(_CptJyc2026StockBarBaseBuilder):
    """从 Level-2 snapshot 构建并写入 1 分钟行情。"""

    instrument_batch_size = 200

    snapshot_fields = [
        "date",
        "instrument",
        "pre_close",
        "price",
        "num_trades",
        "volume",
        "amount",
        *[f"ask_price{i}" for i in range(1, 6)],
        *[f"bid_price{i}" for i in range(1, 6)],
        *[f"ask_volume{i}" for i in range(1, 6)],
        *[f"bid_volume{i}" for i in range(1, 6)],
        *[f"ask_num_orders{i}" for i in range(1, 6)],
        *[f"bid_num_orders{i}" for i in range(1, 6)],
    ]

    def __init__(
        self,
        start_date: str,
        end_date: str,
        suffix: str = None,
        n_jobs: int = 8,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.n_jobs = int(n_jobs)
        if self.n_jobs == 0:
            raise ValueError("n_jobs 不能为 0")
        self.stock_pool = dai.query(
            """
            SELECT date, member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '000852.SH'
            """,
            filters={"date": ["2020-01-01", "2024-12-31"]},
        ).df()
        self.instruments = self.stock_pool["instrument"].unique().tolist()

        name = "cpt_jyc_2026_stock_bar1m"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        print(
            f"初始化！{self.datasource_id}, 频率: 1分钟, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """读取快照，并按 2020 至 2024 年中证 1000 成分股并集过滤。"""
        if not self.instruments:
            return pd.DataFrame()

        fields = ", ".join(self.snapshot_fields)
        return dai.query(
            f"SELECT {fields} FROM cn_stock_level2_snapshot",
            filters={
                "date": [
                    f"{start_date} 00:00:00",
                    f"{end_date} 23:59:59",
                ],
                "instrument": self.instruments,
            },
            compression=True,
        ).df()

    @staticmethod
    def build_one_minute(df: pd.DataFrame) -> pd.DataFrame:
        return OneMinuteBarBuilder.build(df)

    @staticmethod
    def enrich_one_minute(
        df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """补充 instrument_id 和复权因子。"""
        if df.empty:
            return df.assign(
                instrument_id=pd.Series(dtype="int64"), adjust_factor=np.nan
            )

        instruments = df["instrument"].dropna().unique().tolist()
        instrument_ids = dai.query(
            "SELECT instrument, instrument_id FROM all_instruments",
            filters={"instrument": instruments},
        ).df()
        factors = dai.query(
            "SELECT date, instrument, adjust_factor FROM cn_stock_real_bar1d",
            filters={
                "date": [start_date, end_date],
                "instrument": instruments,
            },
        ).df()
        factors["trading_day"] = pd.to_datetime(factors.pop("date")).dt.normalize()

        out = df.copy()
        out["trading_day"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.merge(
            instrument_ids,
            how="left",
            on="instrument",
            validate="many_to_one",
        )
        out = out.merge(
            factors,
            how="left",
            on=["trading_day", "instrument"],
            validate="many_to_one",
        )
        return out.drop(columns="trading_day")

    def build(self) -> pd.DataFrame:
        start_date = pd.Timestamp(self.start_date).normalize()
        end_date = pd.Timestamp(self.end_date).normalize()
        if start_date > end_date:
            raise ValueError("start_date 不能晚于 end_date")

        all_daily_bars = []
        built_days = 0
        total_start = datetime.now()

        for trading_day in pd.date_range(start_date, end_date, freq="D"):
            day_str = trading_day.strftime("%Y-%m-%d")
            day_start = datetime.now()
            snapshots = self.get_data(day_str, day_str)
            read_end = datetime.now()
            print(
                f"[{day_str}] 获取快照耗时: "
                f"{round((read_end - day_start).total_seconds(), 4)} 秒, "
                f"行数: {len(snapshots)}"
            )

            if snapshots.empty:
                print(f"[{day_str}] 源数据为空，跳过计算和入库")
                continue

            instruments = snapshots["instrument"].dropna().unique().tolist()
            instrument_batches = [
                instruments[i : i + self.instrument_batch_size]
                for i in range(0, len(instruments), self.instrument_batch_size)
            ]
            snapshot_batches = [
                snapshots.loc[snapshots["instrument"].isin(batch)].copy()
                for batch in instrument_batches
            ]
            del snapshots

            parallel_result = Parallel(backend="loky", n_jobs=self.n_jobs)(
                delayed(OneMinuteBarBuilder.build)(snapshot_batch)
                for snapshot_batch in snapshot_batches
            )
            parallel_result = [df for df in parallel_result if not df.empty]
            if not parallel_result:
                print(f"[{day_str}] 分钟行情构建结果为空，跳过入库")
                continue

            daily_bars = pd.concat(parallel_result, ignore_index=True)
            daily_bars = self.enrich_one_minute(daily_bars, day_str, day_str)
            daily_bars = self.normalize(daily_bars)
            build_end = datetime.now()
            print(
                f"[{day_str}] 分钟行情构建耗时: "
                f"{round((build_end - read_end).total_seconds(), 4)} 秒, "
                f"股票数: {len(instruments)}, 批次数: {len(instrument_batches)}, "
                f"行数: {len(daily_bars)}"
            )

            self.dai_write(daily_bars)
            write_end = datetime.now()
            print(
                f"[{day_str}] 数据存储耗时: "
                f"{round((write_end - build_end).total_seconds(), 4)} 秒"
            )
            all_daily_bars.append(daily_bars)
            built_days += 1

        total_end = datetime.now()
        print(
            f"构建完成，成功入库交易日数: {built_days}, "
            f"总耗时: {round((total_end - total_start).total_seconds(), 4)} 秒"
        )
        if not all_daily_bars:
            return pd.DataFrame(columns=self.schema.columns())
        return pd.concat(all_daily_bars, ignore_index=True)


class CptJyc2026StockBarKmBuilder(_CptJyc2026StockBarBaseBuilder):
    """从已构建的 1 分钟数据聚合并写入 K 分钟行情。"""

    source_datasource_id = "cpt_jyc_2026_stock_bar1m"

    def __init__(
        self,
        start_date: str,
        end_date: str,
        K: int,
        suffix: str = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.K = int(K)
        supported = sorted(k for k in TIME_SETS if k != 1)
        if self.K not in supported:
            raise ValueError(
                f"不支持的聚合频率 K={self.K}, 可选: {supported}; "
                "1 分钟数据请使用 CptJyc2026StockBar1mBuilder"
            )

        name = f"cpt_jyc_2026_stock_bar{self.K}m"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        print(
            f"初始化！{self.datasource_id}, 频率: {self.K}分钟, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """读取第一阶段已经构建并落库的 1 分钟行情。"""
        sql = f"SELECT * FROM {self.source_datasource_id}"
        return dai.query(
            sql,
            filters={
                "date": [
                    f"{start_date} 00:00:00",
                    f"{end_date} 23:59:59",
                ]
            },
            compression=True,
        ).df()

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return KMinuteBarBuilder.build(df, self.K)

    def build(self) -> pd.DataFrame:
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(
            f"获取1分钟数据耗时: {round((t1 - t0).total_seconds(), 4)} 秒, "
            f"行数: {len(df)}"
        )

        df = self.aggregate(df)
        t2 = datetime.now()
        print(
            f"{self.K}分钟聚合耗时: "
            f"{round((t2 - t1).total_seconds(), 4)} 秒, 行数: {len(df)}"
        )

        df = self.normalize(df)
        self.dai_write(df)
        t3 = datetime.now()
        print(f"数据存储耗时: {round((t3 - t2).total_seconds(), 4)} 秒")
        return df
