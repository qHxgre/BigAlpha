import dai
import numpy as np
import pandas as pd
from datetime import datetime

from base import BaseBuilder
from bigalpha_2026_stock_barkm.constant import TIME_SETS
from bigalpha_2026_stock_barkm.schema import Bigalpha2026StockBarKmSchema


class Bigalpha2026StockBarKmBuilder(BaseBuilder):
    """K 分钟 K 线构建器

    基于已构建的 1 分钟数据源 bigalpha_2026_stock_bar1m, 按交易时段
    (上午 09:30-11:30, 下午 13:00-15:00) 自定义时间段聚合为 K 分钟 bar。

    为什么不用简单的 resample:
        df.resample('5min') 以 0 点为锚点对齐分箱, 对于"右标注"的分钟
        bar 会整体错位一格, 导致开盘/收盘 bar 被划入相邻分箱, 同时午休
        时段会产生跨越 11:30~13:00 的空箱或错误聚合。这里改为按 constant.py
        中写死的时间段端点(TIME_SETS)分箱, 并把 bar 标注在分段的"结束时刻"
        (收盘段封顶到 11:30 / 15:00), 从而保证开盘、收盘数据都不丢失。

    参数 K 控制频率: 1 / 5 / 15 / 30 等, 必须是 TIME_SETS 中已定义的频率。
    K=1 时即原始 1 分钟, 不做聚合, 仅过滤连续竞价时段。
    """

    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = Bigalpha2026StockBarKmSchema

    # 取分段最后一笔(快照/累计类字段) / 首笔 / 极值
    FIRST_FIELDS = ["open", "pre_close"]
    MAX_FIELDS = ["high"]
    MIN_FIELDS = ["low"]
    # 其余字段(close、累计 volume/amount/deal_number、盘口快照)均取分段末值

    def __init__(self, start_date: str, end_date: str, K: int = 5) -> None:
        self.start_date = start_date
        self.end_date = end_date
        
        # 股票池：2019年至今的中证1000指数成分
        self.instruments = dai.query("""
        SELECT date, member_code
        FROM cn_stock_index_component
        WHERE instrument='000852.SH'
        AND date>'2019-01-01'
        """).df()['member_code'].unique().tolist()

        self.K = int(K)
        if self.K not in TIME_SETS:
            raise ValueError(f"不支持的频率 K={self.K}, 可选: {sorted(TIME_SETS)} (在 constant.py 中定义)")
        self.datasource_id = f"bigalpha_2026_stock_bar{self.K}m"
        print(f"初始化！{self.datasource_id}, 频率: {self.K}分钟, 时间周期: {self.start_date}, {self.end_date}")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        df = df.fillna(self.schema.field_default_mapping())
        return df

    def dai_write(self, df: pd.DataFrame):
        default_docs = self.schema.default_docs()
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y%m").astype("int64")
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        sql = f"SELECT * FROM cn_stock_bar1m_derived_c"
        df = dai.query(sql, filters={
            "date": [f"{start_date} 00:00:00", f"{end_date} 23:59:59"],
            "instrument": self.instruments
        }).df()
        return df

    @staticmethod
    def _hms_to_minute(hms: int) -> int:
        """HHMMSS 整数 -> 当日分钟数(从 0 点起), 如 93000 -> 570。"""
        return (hms // 10000) * 60 + (hms // 100 % 100)

    def _assign_bar_end(self, df: pd.DataFrame) -> pd.Series:
        """按 constant.py 中写死的时间段端点, 把每条 1 分钟数据分配到所属 bar。

        遍历 TIME_SETS[K] 各时段的相邻端点 (start, end), 每对构成一个 bar:
          - 每个时段的第一段实行"左闭右闭", 含开盘集合竞价(09:30/13:00)那一笔
          - 其余段实行"左开右闭", 不含分段起始那一笔
        bar 标注在该段的结束端点(如 09:35 / 11:30 / 15:00)。
        集合竞价/午休等不落在任何段内的数据保持 NaT, 由调用方丢弃。
        """
        # 当日 HHMMSS 整数, 直接与时间段端点比较
        hms = df["date"].dt.strftime("%H%M%S").astype(int)
        day = df["date"].dt.normalize()

        # bar 结束端点的分钟数(从 0 点起), 默认 NaN(非连续竞价时段)
        end_minute = pd.Series(np.nan, index=df.index, dtype="float64")

        for series in TIME_SETS[self.K].values():
            for i in range(1, len(series)):
                start_time, end_time = series[i - 1], series[i]
                if i - 1 == 0:
                    # 时段第一段: 左闭右闭, 包含开盘那一分钟
                    mask = (hms >= start_time) & (hms <= end_time)
                else:
                    # 其余段: 左开右闭, 不含分段起始那一分钟
                    mask = (hms > start_time) & (hms <= end_time)
                end_minute[mask] = self._hms_to_minute(end_time)

        bar_end = day + pd.to_timedelta(end_minute, unit="m")
        return bar_end

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """按 (instrument, bar结束时刻) 聚合 1 分钟数据为 K 分钟 bar。

        K=1 时目标频率即为 1 分钟, 无需聚合, 仅过滤连续竞价时段后直接返回。
        """
        if df.empty:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if self.K == 1:
            return df

        # 用预定义时间段端点标注 bar 结束时刻; 非连续竞价时段为 NaT
        df["__bar_end"] = self._assign_bar_end(df)
        df = df.dropna(subset=["__bar_end"])
        # 时间先排序, 保证 first/last 取到真正的段首/段末
        df = df.sort_values(["instrument", "date"])

        cols = [c for c in self.schema.columns() if c not in ("date", "instrument")]
        agg_map = {}
        for c in cols:
            if c in self.FIRST_FIELDS:
                agg_map[c] = "first"
            elif c in self.MAX_FIELDS:
                agg_map[c] = "max"
            elif c in self.MIN_FIELDS:
                agg_map[c] = "min"
            else:
                # close、累计量(volume/amount/deal_number)、盘口快照 -> 段末值
                agg_map[c] = "last"

        out = (
            df.groupby(["instrument", "__bar_end"], sort=True)
            .agg(agg_map)
            .reset_index()
            .rename(columns={"__bar_end": "date"})
        )
        return out

    def build(self) -> pd.DataFrame:
        # 读取 1 分钟数据
        t0 = datetime.now()
        df = self.get_data(self.start_date, self.end_date)
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1-t0).total_seconds(), 4)} 秒, 行数: {len(df)}")

        # 聚合为 K 分钟
        df = self.aggregate(df)
        t2 = datetime.now()
        print(f"K分钟聚合耗时: {round((t2-t1).total_seconds(), 4)} 秒, 行数: {len(df)}")

        # 存储数据
        df = self.normalize(df)
        self.dai_write(df)
        t3 = datetime.now()
        print(f"数据存储耗时: {round((t3-t2).total_seconds(), 4)} 秒")
        return df
