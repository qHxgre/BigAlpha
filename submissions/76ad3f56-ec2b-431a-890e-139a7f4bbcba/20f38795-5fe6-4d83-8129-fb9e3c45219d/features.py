"""
特征构建模块 —— 本地训练与云端推理的统一入口。

核心设计原则:
    "特征构建只有一套逻辑"。本地表（e2e_bar*）与云端表（stock_bar*）
    格式不同，但在 to_canonical() 之后对齐为同一表示，
    后续 build_features() 完全复用，杜绝训练/推理漂移。
"""

from __future__ import annotations

import numpy as np
import numpy_compat  # noqa: F401
import pandas as pd

from parquet_compat import read_parquet

PRICE_COLS = ["open", "high", "low", "close"]
TRADE_COLS = ["volume", "amount", "deal_number"]
ASK_PRICE  = ["ask_price1",  "ask_price2",  "ask_price3"]
BID_PRICE  = ["bid_price1",  "bid_price2",  "bid_price3"]
ASK_VOL    = ["ask_volume1", "ask_volume2", "ask_volume3"]
BID_VOL    = ["bid_volume1", "bid_volume2", "bid_volume3"]
ASK_ORDERS = ["ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]
BID_ORDERS = ["bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]

RAW_COLS: list[str] = (
    PRICE_COLS + TRADE_COLS
    + ASK_PRICE + BID_PRICE
    + ASK_VOL   + BID_VOL
)
RAW25_COLS: list[str] = RAW_COLS + ASK_ORDERS + BID_ORDERS
DERIVED_COLS: list[str] = [
    "f_amplitude",
    "f_body",
    "f_upper_shadow",
    "f_lower_shadow",
    "f_vwap_dev",
    "f_ob_imbalance",
    "f_spread",
    "f_trade_intensity",
]
FEATURE_COLS: list[str] = RAW_COLS + DERIVED_COLS
N_FEAT = len(FEATURE_COLS)  # 27

LOG1P_COLS = [
    "volume", "amount", "deal_number",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
    "f_trade_intensity",
]


def to_canonical(
    df: pd.DataFrame,
    *,
    is_local: bool,
) -> pd.DataFrame:
    """把本地表或云端表归一化为 date / instrument / FEATURE_COLS。"""
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
            + [c for c in df.columns
               if c.startswith("ask_price") or c.startswith("bid_price")]
        )
        for c in price_money_cols:
            if c in df.columns:
                df[c] = df[c].astype("float64") / 100.0

        df["instrument"] = df["instrument_id"].astype(str)

    else:
        if "num_trades" in df.columns and "deal_number" not in df.columns:
            df = df.rename(columns={"num_trades": "deal_number"})

        drop_cols = [
            c for c in df.columns
            if any(
                c.startswith(prefix) and c[-1] in "456789"
                for prefix in ("ask_price", "bid_price",
                               "ask_volume", "bid_volume",
                               "ask_num_orders", "bid_num_orders")
            ) or any(
                c.startswith(prefix) and c.endswith("10")
                for prefix in ("ask_price", "bid_price",
                               "ask_volume", "bid_volume",
                               "ask_num_orders", "bid_num_orders")
            )
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

    keep = ["date", "instrument"] + RAW25_COLS
    for c in RAW25_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep]

    for c in RAW25_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    return df.reset_index(drop=True)


def build_features(df_canonical: pd.DataFrame) -> pd.DataFrame:
    """构建派生量价特征，log1p 大量纲字段并填充缺失。"""
    df = df_canonical.copy()

    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    amt = df["amount"].to_numpy(np.float64)
    deals = df["deal_number"].to_numpy(np.float64)
    ask_p1 = df["ask_price1"].to_numpy(np.float64)
    bid_p1 = df["bid_price1"].to_numpy(np.float64)
    ask_v1 = df["ask_volume1"].to_numpy(np.float64)
    bid_v1 = df["bid_volume1"].to_numpy(np.float64)
    eps = 1e-8

    valid_close = close > eps
    df["f_amplitude"] = np.where(valid_close, (high - low) / close, 0.0).astype(np.float32)
    df["f_body"] = np.where(valid_close, np.abs(close - open_) / close, 0.0).astype(np.float32)

    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    df["f_upper_shadow"] = np.where(valid_close, np.maximum(upper, 0.0) / close, 0.0).astype(np.float32)
    df["f_lower_shadow"] = np.where(valid_close, np.maximum(lower, 0.0) / close, 0.0).astype(np.float32)

    vwap = np.where(vol > eps, amt / vol, close)
    df["f_vwap_dev"] = np.where(valid_close, (close - vwap) / close, 0.0).astype(np.float32)

    ob_sum = bid_v1 + ask_v1
    df["f_ob_imbalance"] = np.where(ob_sum > eps, (bid_v1 - ask_v1) / ob_sum, 0.0).astype(np.float32)

    valid_spread = valid_close & (ask_p1 > eps) & (bid_p1 > eps)
    df["f_spread"] = np.where(valid_spread, (ask_p1 - bid_p1) / close, 0.0).astype(np.float32)
    df["f_trade_intensity"] = np.where(deals > eps, vol / deals, 0.0).astype(np.float32)

    for c in LOG1P_COLS:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


def make_windows(
    df_feat: pd.DataFrame,
    seq_len: int,
    mode: str,
    sd,
    ed,
):
    sd_ts = pd.Timestamp(sd)
    ed_ts = pd.Timestamp(ed)
    wins, ys_list, keys = [], [], []

    for ins, sub in df_feat.groupby("instrument", sort=False):
        if len(sub) < seq_len + 1:
            continue
        sub = sub.sort_values("date").reset_index(drop=True)
        feats    = sub[FEATURE_COLS].to_numpy(np.float32)
        day_arr  = sub["date"].dt.normalize().to_numpy()
        close_np = sub["close"].to_numpy(np.float64)
        last_bar = np.flatnonzero(np.append(day_arr[1:] != day_arr[:-1], True))
        close_by_day = close_np[last_bar]
        dates_by_day = day_arr[last_bar]

        for k, p in enumerate(last_bar):
            d = pd.Timestamp(dates_by_day[k])
            if p + 1 < seq_len or d < sd_ts or d > ed_ts:
                continue
            label = None
            if k + 1 < len(last_bar) and close_by_day[k] > 0:
                r = close_by_day[k + 1] / close_by_day[k] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if mode == "train" and label is None:
                continue
            wins.append(feats[p - seq_len + 1 : p + 1])
            ys_list.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))

    if not keys:
        raise RuntimeError(
            "make_windows 无样本 (mode=%s, sd=%s, ed=%s)。请检查日期范围或数据完整性。"
            % (mode, sd, ed)
        )

    X   = np.stack(wins, axis=0).astype(np.float32)
    ys  = np.array(ys_list, np.float32) if mode == "train" else None
    idx = pd.DataFrame(keys, columns=["date", "instrument"])
    return X, ys, idx


def compute_stats(X: np.ndarray):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = (flat.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def load_local_table(parquet_path: str, instruments=None) -> pd.DataFrame:
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
