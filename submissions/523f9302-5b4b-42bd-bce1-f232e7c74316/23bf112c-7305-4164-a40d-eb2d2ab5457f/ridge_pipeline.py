"""Shared preprocessing and inference code for the 30-minute Ridge smoke test.

This is an engineering smoke-test baseline. Ridge does not meet the contest's
minimum trainable-parameter requirement and must not be used as the final entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
    "ask_num_orders1",
    "bid_num_orders1",
]
PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "amount",
    "ask_price1",
    "bid_price1",
]
OHLC_COLUMNS = ["open", "high", "low", "close"]
SEQUENCE_LENGTH = 40  # 5 trading days * 8 bars/day at 30m frequency


def to_canonical(df: pd.DataFrame, *, is_local: bool) -> pd.DataFrame:
    """Align compressed local bars and original cloud bars."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if is_local:
        for column in OHLC_COLUMNS:
            df.loc[df[column] == -1, column] = np.nan
        for column in PRICE_COLUMNS:
            if column in df.columns:
                df[column] = df[column].astype(np.float64) / 100.0
        df["key"] = df["instrument_id"]
    else:
        df["key"] = df["instrument"]

    return df.sort_values(["key", "date"], kind="stable").reset_index(drop=True)


def standardize_features(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    values = np.where(np.isfinite(values), values, mean)
    return ((values - mean) / std).astype(np.float32)


def make_inference_samples(
    frame: pd.DataFrame,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    sequence_length: int,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build one causal sample per instrument and trading day."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    samples: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for key, group in frame.groupby("key", sort=False):
        group = group.sort_values("date", kind="stable")
        values = standardize_features(
            group[FEATURE_COLUMNS].to_numpy(), feature_mean, feature_std
        )
        days = group["date"].dt.normalize().to_numpy()
        eod = np.flatnonzero(np.r_[days[1:] != days[:-1], True])

        for position in eod:
            day = pd.Timestamp(days[position])
            if day < start or day > end or position + 1 < sequence_length:
                continue
            window = values[position - sequence_length + 1 : position + 1]
            if window.shape[0] != sequence_length:
                continue
            samples.append(window.reshape(-1))
            rows.append({"date": day, "key": key})

    if not samples:
        width = sequence_length * len(FEATURE_COLUMNS)
        return np.empty((0, width), np.float32), pd.DataFrame(
            columns=["date", "key"]
        )
    return np.stack(samples), pd.DataFrame(rows)


def load_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Artifact feature order does not match inference code")
    return artifact


def predict_matrix(x: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    weights = np.asarray(artifact["weights"], dtype=np.float64)
    return (x.astype(np.float64) @ weights[:-1] + weights[-1]).astype(np.float32)


def platform_main(
    datasources: dict[str, str],
    start_date: Any,
    end_date: Any,
    *,
    artifact_path: str | Path = "ridge_model.json",
) -> pd.DataFrame:
    """Cloud entry point used by predict.ipynb."""
    import dai

    artifact = load_artifact(artifact_path)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    # Five trading days need more than five calendar days; 30 is conservative.
    buffer_start = (start - pd.Timedelta(days=30)).strftime("%Y-%m-%d 00:00:00")
    query_end = end.strftime("%Y-%m-%d 23:59:59")

    table = datasources.get("bar30m")
    if table is None:
        table = datasources.get("bigalpha_2026_stock_bar30m")
    if table is None:
        candidates = [value for key, value in datasources.items() if "30m" in key]
        if len(candidates) != 1:
            raise KeyError(f"Cannot identify bar30m datasource from: {datasources}")
        table = candidates[0]

    selected = ["date", "instrument", *FEATURE_COLUMNS]
    raw = dai.query(
        f"SELECT {','.join(selected)} FROM {table} ORDER BY instrument, date",
        filters={"date": [buffer_start, query_end]},
    ).df()
    canonical = to_canonical(raw, is_local=False)
    x, index = make_inference_samples(
        canonical,
        feature_mean=np.asarray(artifact["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(artifact["feature_std"], dtype=np.float64),
        sequence_length=int(artifact["sequence_length"]),
        start_date=start,
        end_date=end,
    )
    if x.shape[0] == 0:
        raise RuntimeError("No inference samples were constructed")

    index["score"] = predict_matrix(x, artifact)
    result = index.rename(columns={"key": "instrument"})[
        ["date", "instrument", "score"]
    ]
    result["score"] = result["score"].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["score"])
    return result.sort_values(["date", "instrument"]).reset_index(drop=True)
