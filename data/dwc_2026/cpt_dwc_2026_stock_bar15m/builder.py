import dai
import numpy as np
import pandas as pd
from datetime import time, datetime, timedelta

from warehouse.builder.base import BaseBuilder
from bar15m_price.schema import CptDeepwinBar15mSchema


class CptDeepwinBar15mBuilder(BaseBuilder):
    datasource_id = "cpt_dwc_2026_stock_bar15m"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = CptDeepwinBar15mSchema

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date

        # 定义截断时间列表
        self.cutoff_times = [
            '0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130',
            '1315', '1330', '1345', '1400', '1415', '1430', '1445', '1500'
        ]

        print(f"初始化！{self.datasource_id}, 时间周期: {self.start_date}, {self.end_date}")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        df = df.fillna(self.schema.field_default_mapping())
        return df

    def dai_write(self, df: pd.DataFrame):
        default_docs = self.schema.default_docs()
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y%m%d").astype("int64")
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

    def _get_next_trading_day(self, end_date: str) -> str:
        """获取 end_date 之后的下一个交易日（YYYY-MM-DD）。"""
        sql = f"""
        SELECT date(date) AS d
        FROM cn_stock_bar1m_c
        WHERE date > '{end_date} 23:59:59'
        GROUP BY d
        ORDER BY d
        LIMIT 1
        """
        try:
            df = dai.query(sql, full_db_scan=True).df()
            if df is not None and len(df) > 0:
                return str(df.iloc[0, 0])
        except Exception:
            pass

        # 兜底：多拉几天分钟数据，在 pandas 里找最小的下一交易日
        probe_end = (pd.to_datetime(end_date) + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
        bar = dai.query(
            "SELECT date FROM cn_stock_bar1m_c",
            filters={'date': [f'{end_date} 00:00:00', f'{probe_end} 23:59:59']}
        ).df()
        bar['date'] = pd.to_datetime(bar['date'])
        ds = sorted(bar['date'].dt.date.unique())
        for d in ds:
            if str(d) > end_date:
                return str(d)
        return None

    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取分钟数据

        为了保证最后一个交易日的 1500 (由次日 09:30-09:45 计算) 也能算出来，
        会额外拉取 end_date 后一个交易日的 09:45 之前分钟数据。
        """
        next_td = self._get_next_trading_day(end_date)

        if next_td is not None:
            # 拉取到 next_td 09:45:00（含）即可覆盖 1500 的窗口 (09:30,09:45]
            start_ts = f'{start_date} 00:00:00'
            end_ts = f'{next_td} 09:45:00'
        else:
            start_ts = f'{start_date} 00:00:00'
            end_ts = f'{end_date} 23:59:59'
            
        print(f'获取时间范围：{start_ts} 至 {end_ts}')

        # ★ 增加 amount 字段
        sql = 'SELECT date, instrument, close, volume, amount FROM cn_stock_bar1m_c'
        bar1m_df = dai.query(sql, filters={'date': [start_ts, end_ts]}).df()
        bar1m_df['date'] = pd.to_datetime(bar1m_df['date'])
        return bar1m_df

    @staticmethod
    def _vwap_agg(period_data: pd.DataFrame, group_cols: list, date_col: str) -> pd.DataFrame:
        """用 amount/volume 计算 VWAP：VWAP = sum(amount)/sum(volume)"""
        tmp = period_data.copy()
        agg = (
            tmp.groupby(group_cols, as_index=False)
            .agg(sum_amt=('amount', 'sum'), sum_vol=('volume', 'sum'))
        )
        agg['vwap'] = agg['sum_amt'] / agg['sum_vol']
        agg.loc[agg['sum_vol'] <= 0, 'vwap'] = np.nan

        out = agg.rename(columns={date_col: 'date'})[['instrument', 'date', 'vwap']]
        return out

    def cpt_vwap(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """计算 15min 的 vwap"""
        df = input_df.copy()
        df = df.set_index('date')
        df = df.sort_index()

        # 提取日期和时间
        df['date_only'] = df.index.date
        df['time_only'] = df.index.time

        # 获取交易日列表和前一交易日映射（用于 1500 的跨日计算）
        trading_dates = sorted(df['date_only'].unique())
        date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
        df['date_idx'] = df['date_only'].map(date_to_idx)
        df['prev_date_idx'] = df['date_idx'] - 1
        idx_to_date = {idx: date for idx, date in enumerate(trading_dates)}
        df['pre_date'] = df['prev_date_idx'].map(idx_to_date)

        df_list = []
        for cutoff in self.cutoff_times:
            print(f"处理截断时间: {cutoff}")

            if cutoff == '1500':
                # 1500截断：使用下一个交易日 (09:30,09:45] 的分钟数据，归因到前一交易日
                time_mask = ((df['time_only'] > time(9, 30)) & (df['time_only'] <= time(9, 45)))
                date_mask = df['pre_date'].notna()
                mask = time_mask & date_mask

                if mask.sum() == 0:
                    continue

                # ★ 用 amount/volume
                period_data = df[mask][['instrument', 'pre_date', 'volume', 'amount']].copy()
                vwap_df = self._vwap_agg(period_data, ['instrument', 'pre_date'], 'pre_date')
                vwap_df['cutoff_time'] = cutoff
                vwap_df['time_range'] = '次日(0930-0945]'

            else:
                # 1130 午休特殊处理：用 (13:00,13:15] 作为“截断后15分钟”
                if cutoff == '1130':
                    start_t = time(13, 0)
                    end_t = time(13, 15)
                    time_mask = ((df['time_only'] > start_t) & (df['time_only'] <= end_t))
                    mask = time_mask

                    if mask.sum() == 0:
                        continue

                    # ★ 用 amount/volume
                    period_data = df[mask][['instrument', 'date_only', 'volume', 'amount']].copy()
                    vwap_df = self._vwap_agg(period_data, ['instrument', 'date_only'], 'date_only')
                    vwap_df['cutoff_time'] = cutoff
                    vwap_df['time_range'] = '(1300-1315]'

                else:
                    cutoff_time = pd.to_datetime(cutoff, format='%H%M').time()
                    end_time = (pd.to_datetime(cutoff, format='%H%M') + pd.Timedelta(minutes=15)).time()

                    time_mask = ((df['time_only'] > cutoff_time) & (df['time_only'] <= end_time))
                    mask = time_mask

                    if mask.sum() == 0:
                        continue

                    # ★ 用 amount/volume
                    period_data = df[mask][['instrument', 'date_only', 'volume', 'amount']].copy()
                    vwap_df = self._vwap_agg(period_data, ['instrument', 'date_only'], 'date_only')
                    vwap_df['cutoff_time'] = cutoff
                    vwap_df['time_range'] = f"({cutoff}-{end_time.strftime('%H%M')}]"

            # 这里仍然把 date 写成“截断时刻”（回测侧你准备改成按该时刻取 vwap）
            vwap_df['date'] = pd.to_datetime(
                vwap_df['date'].astype(str) + ' ' + cutoff[:2] + ':' + cutoff[2:] + ':00'
            )

            df_list.append(vwap_df[['date', 'instrument', 'cutoff_time', 'vwap', 'time_range']])

        bar1m_vwap = pd.concat(df_list, ignore_index=True)

        # 只保留目标区间内的输出（避免为 next_td 额外拉取的分钟数据产出“多余交易日”的 vwap）
        start_ts = pd.to_datetime(f'{self.start_date} 00:00:00')
        end_ts = pd.to_datetime(f'{self.end_date} 23:59:59')
        bar1m_vwap = bar1m_vwap[(bar1m_vwap['date'] >= start_ts) & (bar1m_vwap['date'] <= end_ts)].copy()

        # 合并收盘价数据（按 cutoff timestamp 对齐）
        bar1m_vwap = pd.merge(
            bar1m_vwap,
            input_df[['date', 'instrument', 'close']],
            how='left',
            on=['date', 'instrument']
        )
        return bar1m_vwap

    def build(self) -> pd.DataFrame:
        """读取 snapshot 数据"""
        t0 = datetime.now()
        bar1m_df = self.get_data(self.start_date, self.end_date)
        bar15m_vwap = self.cpt_vwap(bar1m_df)
        t1 = datetime.now()
        print(f"获取数据耗时: {round((t1 - t0).total_seconds(), 4)} 秒")

        df = self.normalize(bar15m_vwap)
        self.dai_write(df)
        t2 = datetime.now()
        print(f"数据存储耗时: {round((t2 - t1).total_seconds(), 4)} 秒")
        return df
