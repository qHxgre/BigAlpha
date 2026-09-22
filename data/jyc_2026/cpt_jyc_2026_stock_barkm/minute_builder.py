import numpy as np
import pandas as pd

from jyc_2026.cpt_jyc_2026_stock_barkm.schema import CptJyc2026StockBarKmSchema


class OneMinuteBarBuilder:
    """将 Level-2 快照转换为一分钟行情。"""

    _CUMULATIVE_FIELDS = ["volume", "amount", "num_trades"]
    _PRICE_FIELDS = ["open", "high", "low", "close"]
    _IDENTITY_FIELDS = ["instrument_id", "adjust_factor"]

    @staticmethod
    def _elapsed_ms(date: pd.Series) -> pd.Series:
        return (
            (date.dt.hour * 3600 + date.dt.minute * 60 + date.dt.second) * 1000
            + date.dt.microsecond // 1000
        )

    @classmethod
    def assign_minute_end(cls, date: pd.Series) -> pd.Series:
        """按左闭右开规则映射到一分钟末端。

        集合竞价也逐分钟切分，09:25:00 的撮合结果归入 09:25；常规分钟
        ``[HH:MM:00, HH:MM+1:00)`` 标记为下一分钟。收盘分钟额外接纳
        收盘后一段快照，即 ``[14:59:00, 15:01:00)`` 都标记为 15:00
        （上午收盘同理）。
        """
        date = pd.to_datetime(date)
        day = date.dt.normalize()
        elapsed_ms = cls._elapsed_ms(date)
        minute_ms = 60_000

        auction_start = (9 * 3600 + 15 * 60) * 1000
        auction_end = (9 * 3600 + 25 * 60) * 1000
        morning_open = (9 * 3600 + 30 * 60) * 1000
        morning_close = (11 * 3600 + 30 * 60) * 1000
        afternoon_open = 13 * 3600 * 1000
        afternoon_close = 15 * 3600 * 1000

        end_minute = pd.Series(np.nan, index=date.index, dtype="float64")
        auction = (elapsed_ms >= auction_start) & (elapsed_ms < auction_end)
        auction_labels = (elapsed_ms // minute_ms + 1).astype("int64")
        end_minute.loc[auction] = auction_labels.loc[auction]
        # 09:25:00 是集合竞价撮合结果，仍归入最后一个竞价分钟。
        end_minute.loc[elapsed_ms == auction_end] = auction_end // minute_ms

        for session_open, session_close in (
            (morning_open, morning_close),
            (afternoon_open, afternoon_close),
        ):
            in_session = (elapsed_ms >= session_open) & (elapsed_ms < session_close)
            labels = (elapsed_ms // minute_ms + 1).astype("int64")
            end_minute.loc[in_session] = labels.loc[in_session]

            close_window = (
                (elapsed_ms >= session_close - minute_ms)
                & (elapsed_ms < session_close + minute_ms)
            )
            end_minute.loc[close_window] = session_close // minute_ms

        return day + pd.to_timedelta(end_minute, unit="m")

    @classmethod
    def build(cls, df: pd.DataFrame) -> pd.DataFrame:
        """生成集合竞价和连续竞价的左闭右开一分钟行情。"""
        output_columns = [
            c
            for c in CptJyc2026StockBarKmSchema.columns()
            if c not in cls._IDENTITY_FIELDS
        ]
        if df.empty:
            return pd.DataFrame(columns=output_columns)

        required = {
            "date",
            "instrument",
            "price",
            "bid_price1",
            "ask_price1",
            "pre_close",
            *cls._CUMULATIVE_FIELDS,
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"快照数据缺少字段: {sorted(missing)}")

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.dropna(subset=["date", "instrument"])
        data = data.sort_values(["instrument", "date"]).reset_index(drop=True)
        data["__trading_day"] = data["date"].dt.normalize()
        groups = data.groupby(
            ["instrument", "__trading_day"], sort=False, observed=True
        )

        for field in cls._CUMULATIVE_FIELDS:
            delta = groups[field].diff()
            data[f"__{field}_delta"] = delta.where(
                delta.notna() & (delta >= 0), data[field]
            )

        data["__valid_price"] = data["price"].where(data["price"] > 0)
        elapsed_ms = cls._elapsed_ms(data["date"])
        auction = (
            (elapsed_ms >= (9 * 3600 + 15 * 60) * 1000)
            & (elapsed_ms <= (9 * 3600 + 25 * 60) * 1000)
        )
        bid1 = data["bid_price1"].where(data["bid_price1"] > 0)
        ask1 = data["ask_price1"].where(data["ask_price1"] > 0)
        level1_price = pd.concat([bid1, ask1], axis=1).mean(axis=1, skipna=True)
        auction_fallback = auction & data["__valid_price"].isna()
        data.loc[auction_fallback, "__valid_price"] = level1_price.loc[
            auction_fallback
        ]

        data["__minute_end"] = cls.assign_minute_end(data["date"])
        data = data.dropna(subset=["__minute_end"])
        if data.empty:
            return pd.DataFrame(columns=output_columns)

        passthrough = [
            c
            for c in CptJyc2026StockBarKmSchema.columns()
            if c
            not in {
                "date",
                "instrument",
                *cls._IDENTITY_FIELDS,
                *cls._PRICE_FIELDS,
                "deal_number",
                "volume",
                "amount",
            }
            and c in data.columns
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
        out = (
            data.groupby(["instrument", "__minute_end"], sort=True, observed=True)
            .agg(**agg_spec)
            .reset_index()
            .rename(columns={"__minute_end": "date"})
        )
        return out.reindex(columns=output_columns)
