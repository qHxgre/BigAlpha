"""Parquet helpers that avoid pandas/pyarrow compatibility issues."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

import numpy_compat  # noqa: F401


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(pl.read_parquet(path, columns=columns).to_dict(as_series=False))


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    pl.from_dict(df.to_dict(orient="list")).write_parquet(path)
