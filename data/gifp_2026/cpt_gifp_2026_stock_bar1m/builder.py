from datetime import datetime

import dai
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from base import BaseBuilder
from gifp_2026.cpt_gifp_2026_stock_bar1m.minute_builder import (
    OneMinuteBarBuilder,
)
from gifp_2026.cpt_gifp_2026_stock_bar1m.schema import (
    CptGifp2026StockBar1mSchema,
)
from gifp_2026.cpt_gifp_2026_stock_bar1m.tester import (
    CptGifp2026StockBar1mTester,
    StockBar1mDataTestError,
)


class CptGifp2026StockBar1mBuilder(BaseBuilder):
    """构建沪深 300 历史成分股的 1 分钟 K 线及十档盘口。"""

    datasource_id = "cpt_gifp_2026_stock_bar1m"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptGifp2026StockBar1mSchema
    index_instrument = "000300.SH"
    instrument_batch_size = 100

    snapshot_fields = [
        "date",
        "instrument",
        "pre_close",
        "price",
        "num_trades",
        "volume",
        "amount",
        *[f"ask_price{i}" for i in range(1, 11)],
        *[f"bid_price{i}" for i in range(1, 11)],
        *[f"ask_volume{i}" for i in range(1, 11)],
        *[f"bid_volume{i}" for i in range(1, 11)],
        *[f"ask_num_orders{i}" for i in range(1, 11)],
        *[f"bid_num_orders{i}" for i in range(1, 11)],
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

        name = "cpt_gifp_2026_stock_bar1m"
        self.datasource_id = f"{name}_{suffix}" if suffix else name
        self.tester = CptGifp2026StockBar1mTester(
            start_date,
            end_date,
            datasource_id=self.datasource_id,
        )
        print(
            f"初始化！{self.datasource_id}, 频率: 1分钟, "
            f"时间周期: {self.start_date}, {self.end_date}"
        )

    @classmethod
    def get_daily_instruments(cls, trading_day: str) -> list[str]:
        """读取交易日当天有效的沪深 300 历史成分股。"""
        components = dai.query(
            f"""
            SELECT member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '{cls.index_instrument}'
            """,
            filters={"date": [trading_day, trading_day]},
        ).df()
        if components.empty:
            return []
        return components["instrument"].dropna().unique().tolist()

    def get_data(
        self, start_date: str, end_date: str, instruments: list[str]
    ) -> pd.DataFrame:
        if not instruments:
            return pd.DataFrame()

        fields = ", ".join(self.snapshot_fields)
        return dai.query(
            f"SELECT {fields} FROM cn_stock_level2_snapshot",
            filters={
                "date": [
                    f"{start_date} 00:00:00",
                    f"{end_date} 23:59:59",
                ],
                "instrument": instruments,
            },
            compression=True,
        ).df()

    @staticmethod
    def enrich_one_minute(
        df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """补充 instrument_id 和日复权因子。"""
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
            instruments = self.get_daily_instruments(day_str)
            component_end = datetime.now()

            if not instruments:
                print(f"[{day_str}] 无沪深300成分股数据，跳过")
                continue

            snapshots = self.get_data(day_str, day_str, instruments)
            read_end = datetime.now()
            print(
                f"[{day_str}] 成分股数: {len(instruments)}, "
                f"查询成分耗时: "
                f"{round((component_end - day_start).total_seconds(), 4)} 秒, "
                f"获取快照耗时: "
                f"{round((read_end - component_end).total_seconds(), 4)} 秒, "
                f"行数: {len(snapshots)}"
            )

            if snapshots.empty:
                print(f"[{day_str}] 源快照为空，跳过计算和入库")
                continue

            present_instruments = snapshots["instrument"].dropna().unique().tolist()
            instrument_batches = [
                present_instruments[i : i + self.instrument_batch_size]
                for i in range(0, len(present_instruments), self.instrument_batch_size)
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
            parallel_result = [result for result in parallel_result if not result.empty]
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
                f"实际股票数: {len(present_instruments)}, "
                f"批次数: {len(instrument_batches)}, 行数: {len(daily_bars)}"
            )

            self.dai_write(daily_bars)
            write_end = datetime.now()
            print(
                f"[{day_str}] 数据存储耗时: "
                f"{round((write_end - build_end).total_seconds(), 4)} 秒"
            )
            try:
                checked = self.tester.check_day(day_str, instruments)
            except StockBar1mDataTestError as exc:
                print(
                    f"[{day_str}] 数据测试未通过，当日数据已入库，"
                    "修复后需重建，构建中止"
                )
                raise StockBar1mDataTestError(f"[{day_str}] {exc}") from exc
            print(
                f"[{day_str}] 数据测试通过, "
                f"耗时: {round((datetime.now() - write_end).total_seconds(), 4)} 秒, "
                f"行数: {len(checked)}, "
                f"股票数: {checked['instrument'].nunique()}"
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
