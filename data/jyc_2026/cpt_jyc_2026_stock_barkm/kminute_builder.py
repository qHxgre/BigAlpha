import numpy as np
import pandas as pd

from jyc_2026.cpt_jyc_2026_stock_barkm.constant import TIME_SETS


class KMinuteBarBuilder:
    """从一分钟行情聚合 K 分钟行情。"""

    @staticmethod
    def _hms_to_minute(hms: int) -> int:
        return (hms // 10000) * 60 + (hms // 100 % 100)

    @classmethod
    def _assign_bar_end(cls, df: pd.DataFrame, k: int) -> pd.Series:
        hms = df["date"].dt.strftime("%H%M%S").astype(int)
        day = df["date"].dt.normalize()
        end_minute = pd.Series(np.nan, index=df.index, dtype="float64")

        for series in TIME_SETS[k].values():
            for i in range(1, len(series)):
                start_time, end_time = series[i - 1], series[i]
                mask = (hms > start_time) & (hms <= end_time)
                end_minute.loc[mask] = cls._hms_to_minute(end_time)
        return day + pd.to_timedelta(end_minute, unit="m")

    @classmethod
    def build(cls, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """聚合连续竞价；K != 1 时集合竞价仅保留 09:25。"""
        if df.empty or k == 1:
            return df
        if k not in TIME_SETS:
            raise ValueError(f"不支持的频率 K={k}, 可选: {sorted(TIME_SETS)}")

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        auction_close = data.loc[
            data["date"].dt.strftime("%H%M%S") == "092500"
        ].copy()

        data["__bar_end"] = cls._assign_bar_end(data, k)
        data = data.dropna(subset=["__bar_end"]).sort_values(
            ["instrument", "date"]
        )

        excluded = {
            "date",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "deal_number",
        }
        last_fields = [
            c for c in data.columns if c not in excluded and c != "__bar_end"
        ]
        agg_spec = {
            "open": ("open", "first"),
            "high": ("high", "max"),
            "low": ("low", "min"),
            "close": ("close", "last"),
            "volume": ("volume", "sum"),
            "amount": ("amount", "sum"),
            "deal_number": ("deal_number", "sum"),
            **{field: (field, "last") for field in last_fields},
        }
        out = (
            data.groupby(["instrument", "__bar_end"], sort=True, observed=True)
            .agg(**agg_spec)
            .reset_index()
            .rename(columns={"__bar_end": "date"})
        )
        if not auction_close.empty:
            out = pd.concat(
                [auction_close.reindex(columns=out.columns), out], ignore_index=True
            )
        return out.sort_values(["date", "instrument"]).reset_index(drop=True)
