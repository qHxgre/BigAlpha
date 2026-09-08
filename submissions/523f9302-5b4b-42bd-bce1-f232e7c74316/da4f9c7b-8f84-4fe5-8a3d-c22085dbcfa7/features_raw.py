"""Raw-only schema alignment for BigAlpha E2E models.

This module deliberately contains no derived feature construction. It only
normalizes local compressed e2e tables and cloud stock tables into the same raw
bar schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_COLS = ["open", "high", "low", "close"]
TRADE_COLS = ["volume", "amount", "deal_number"]
ASK_PRICE = ["ask_price1", "ask_price2", "ask_price3"]
BID_PRICE = ["bid_price1", "bid_price2", "bid_price3"]
ASK_VOL = ["ask_volume1", "ask_volume2", "ask_volume3"]
BID_VOL = ["bid_volume1", "bid_volume2", "bid_volume3"]
ASK_ORDERS = ["ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]
BID_ORDERS = ["bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]

RAW_COLS: list[str] = (
    PRICE_COLS
    + TRADE_COLS
    + ASK_PRICE
    + BID_PRICE
    + ASK_VOL
    + BID_VOL
)
RAW25_COLS: list[str] = RAW_COLS + ASK_ORDERS + BID_ORDERS
FEATURE_COLS: list[str] = RAW_COLS
N_FEAT = len(FEATURE_COLS)
LOG1P_COLS = [
    "volume",
    "amount",
    "deal_number",
    "ask_volume1",
    "ask_volume2",
    "ask_volume3",
    "bid_volume1",
    "bid_volume2",
    "bid_volume3",
    "ask_num_orders1",
    "ask_num_orders2",
    "ask_num_orders3",
    "bid_num_orders1",
    "bid_num_orders2",
    "bid_num_orders3",
]


def to_canonical(df: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    """Convert local/cloud raw bar frames to date / instrument / RAW25_COLS."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    if is_local:
        for c in PRICE_COLS:
            if c in df.columns:
                df.loc[df[c] == -1, c] = np.nan

        price_money_cols = (
            PRICE_COLS
            + ["amount"]
            + [
                c
                for c in df.columns
                if c.startswith("ask_price") or c.startswith("bid_price")
            ]
        )
        for c in price_money_cols:
            if c in df.columns:
                df[c] = df[c].astype("float64") / 100.0

        df["instrument"] = df["instrument_id"].astype(str)
    else:
        if "num_trades" in df.columns and "deal_number" not in df.columns:
            df = df.rename(columns={"num_trades": "deal_number"})

        drop_cols = [
            c
            for c in df.columns
            if any(
                c.startswith(prefix) and c[-1] in "456789"
                for prefix in (
                    "ask_price",
                    "bid_price",
                    "ask_volume",
                    "bid_volume",
                    "ask_num_orders",
                    "bid_num_orders",
                )
            )
            or any(
                c.startswith(prefix) and c.endswith("10")
                for prefix in (
                    "ask_price",
                    "bid_price",
                    "ask_volume",
                    "bid_volume",
                    "ask_num_orders",
                    "bid_num_orders",
                )
            )
        ]
        df = df.drop(columns=drop_cols, errors="ignore")
        df["instrument"] = df["instrument"].astype(str)

    for c in RAW25_COLS:
        if c not in df.columns:
            df[c] = np.nan

    keep = ["date", "instrument"] + RAW25_COLS
    df = df[keep]
    for c in RAW25_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    return df.reset_index(drop=True)


def build_features(df_canonical: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw columns only; no derived indicators are constructed."""
    df = df_canonical.copy()
    for c in LOG1P_COLS:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    df[RAW25_COLS] = df[RAW25_COLS].fillna(0.0)
    return df


def compute_stats(x: np.ndarray):
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def load_local_table(parquet_path: str, instruments=None) -> pd.DataFrame:
    from parquet_compat import read_parquet

    df_raw = read_parquet(parquet_path)
    df_canon = to_canonical(df_raw, is_local=True)
    if instruments is not None:
        df_canon = df_canon[df_canon["instrument"].isin(instruments)]
    return build_features(df_canon)


def load_cloud_table(df_raw: pd.DataFrame, instruments=None) -> pd.DataFrame:
    df_canon = to_canonical(df_raw, is_local=False)
    if instruments is not None:
        df_canon = df_canon[df_canon["instrument"].isin(instruments)]
    return build_features(df_canon)
