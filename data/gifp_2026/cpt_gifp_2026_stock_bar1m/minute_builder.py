import numpy as np
import pandas as pd

from gifp_2026.cpt_gifp_2026_stock_bar1m.schema import CptGifp2026StockBar1mSchema


class OneMinuteBarBuilder:
    """将 Level-2 快照转换为一分钟行情：先清洗快照，再聚合分钟行情。"""

    _CUMULATIVE_FIELDS = ["volume", "amount", "num_trades"]
    _PRICE_FIELDS = ["open", "high", "low", "close"]
    _IDENTITY_FIELDS = ["instrument_id", "adjust_factor"]
    _REQUIRED_FIELDS = [
        "date",
        "instrument",
        "price",
        "bid_price1",
        "ask_price1",
        "pre_close",
        *_CUMULATIVE_FIELDS,
    ]
    _KEYS = ["instrument", "__trading_day"]

    _MINUTE_MS = 60_000
    _AUCTION_START = (9 * 3600 + 15 * 60) * 1000
    _AUCTION_END = (9 * 3600 + 25 * 60) * 1000
    _SESSIONS = [
        ((9 * 3600 + 30 * 60) * 1000, (11 * 3600 + 30 * 60) * 1000),
        (13 * 3600 * 1000, 15 * 3600 * 1000),
    ]

    @staticmethod
    def _elapsed_ms(date: pd.Series) -> pd.Series:
        return (
            (date.dt.hour * 3600 + date.dt.minute * 60 + date.dt.second) * 1000
            + date.dt.microsecond // 1000
        )

    @classmethod
    def assign_minute_end(cls, date: pd.Series) -> pd.Series:
        """按左闭右开规则映射到一分钟末端，无法归属交易分钟的快照返回 NaT。"""
        date = pd.to_datetime(date)
        ms = cls._elapsed_ms(date)
        label = (ms // cls._MINUTE_MS + 1).astype("int64")
        end_minute = pd.Series(np.nan, index=date.index, dtype="float64")

        auction = (ms >= cls._AUCTION_START) & (ms < cls._AUCTION_END)
        end_minute.loc[auction] = label.loc[auction]
        end_minute.loc[ms == cls._AUCTION_END] = (
            cls._AUCTION_END // cls._MINUTE_MS
        )

        for session_open, session_close in cls._SESSIONS:
            in_session = (ms >= session_open) & (ms < session_close)
            end_minute.loc[in_session] = label.loc[in_session]
            close_window = (ms >= session_close - cls._MINUTE_MS) & (
                ms < session_close + cls._MINUTE_MS
            )
            end_minute.loc[close_window] = session_close // cls._MINUTE_MS

        return date.dt.normalize() + pd.to_timedelta(end_minute, unit="m")

    @classmethod
    def clean_snapshots(cls, df: pd.DataFrame) -> pd.DataFrame:
        """剔除非交易时段快照，以及累计量额或笔数发生回退的源数据噪声。"""
        if df.empty:
            return df
        missing = set(cls._REQUIRED_FIELDS).difference(df.columns)
        if missing:
            raise ValueError(f"快照数据缺少字段: {sorted(missing)}")

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = (
            data.dropna(subset=["date", "instrument"])
            .sort_values(["instrument", "date"])
            .reset_index(drop=True)
        )
        data["__trading_day"] = data["date"].dt.normalize()
        data["__minute_end"] = cls.assign_minute_end(data["date"])
        data = data.dropna(subset=["__minute_end"]).reset_index(drop=True)
        data[cls._CUMULATIVE_FIELDS] = data[cls._CUMULATIVE_FIELDS].apply(
            pd.to_numeric, errors="coerce"
        )
        running_max = data.groupby(
            cls._KEYS, sort=False, observed=True
        )[cls._CUMULATIVE_FIELDS].cummax()
        regressed = (data[cls._CUMULATIVE_FIELDS] < running_max).any(axis=1)
        return data.loc[~regressed].reset_index(drop=True)

    @classmethod
    def _fill_unpriced_bars(cls, out: pd.DataFrame) -> pd.DataFrame:
        """给无价格且无成交的分钟填入最近成交价或前收盘价。"""
        unpriced = out[cls._PRICE_FIELDS].isna().all(axis=1) & out["volume"].eq(0)
        if not unpriced.any():
            return out

        keys = [out["instrument"], out["date"].dt.normalize()]
        last_traded_close = (
            out["close"].where(out["volume"] > 0).groupby(keys, sort=False).ffill()
        )
        fill_price = last_traded_close.fillna(
            out["pre_close"].where(out["pre_close"] > 0)
        )
        out = out.copy()
        for field in cls._PRICE_FIELDS:
            out.loc[unpriced, field] = fill_price.loc[unpriced]
        return out

    @classmethod
    def aggregate_minute_bars(cls, data: pd.DataFrame) -> pd.DataFrame:
        """用清洗后的快照聚合左闭右开的一分钟行情。"""
        output_columns = [
            column
            for column in CptGifp2026StockBar1mSchema.columns()
            if column not in cls._IDENTITY_FIELDS
        ]
        if data.empty:
            return pd.DataFrame(columns=output_columns)

        data = data.copy()
        groups = data.groupby(cls._KEYS, sort=False, observed=True)
        first_in_day = groups.cumcount().eq(0)
        for field in cls._CUMULATIVE_FIELDS:
            data[f"__{field}_delta"] = groups[field].diff().mask(
                first_in_day, data[field]
            )

        price = data["price"].where(data["price"] > 0)
        ms = cls._elapsed_ms(data["date"])
        in_auction = (ms >= cls._AUCTION_START) & (ms <= cls._AUCTION_END)
        bid1 = data["bid_price1"].where(data["bid_price1"] > 0)
        ask1 = data["ask_price1"].where(data["ask_price1"] > 0)
        level1_price = pd.concat([bid1, ask1], axis=1).mean(axis=1, skipna=True)
        data["__valid_price"] = price.mask(
            in_auction & price.isna(), level1_price
        )

        skip = {
            "date",
            "instrument",
            *cls._IDENTITY_FIELDS,
            *cls._PRICE_FIELDS,
            "deal_number",
            "volume",
            "amount",
        }
        passthrough = [
            column
            for column in CptGifp2026StockBar1mSchema.columns()
            if column not in skip and column in data.columns
        ]
        agg_spec = {
            "open": ("__valid_price", "first"),
            "high": ("__valid_price", "max"),
            "low": ("__valid_price", "min"),
            "close": ("__valid_price", "last"),
            "deal_number": ("__num_trades_delta", "sum"),
            "volume": ("__volume_delta", "sum"),
            "amount": ("__amount_delta", "sum"),
            **{field: (field, "last") for field in passthrough},
        }
        out = data.groupby(
            ["instrument", "__minute_end"], sort=True, observed=True
        ).agg(**agg_spec)
        out = out.reset_index().rename(columns={"__minute_end": "date"})
        return cls._fill_unpriced_bars(out).reindex(columns=output_columns)

    @classmethod
    def build(cls, df: pd.DataFrame) -> pd.DataFrame:
        """生成集合竞价和连续竞价的左闭右开一分钟行情。"""
        return cls.aggregate_minute_bars(cls.clean_snapshots(df))
