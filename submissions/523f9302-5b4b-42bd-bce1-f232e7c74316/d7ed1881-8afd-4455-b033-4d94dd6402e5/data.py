import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import Config
from utils import normalize_date, rank_to_score


def _query_with_dai(sql: str, filters: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    import dai  # type: ignore

    if filters is None:
        result = dai.query(sql)
    else:
        result = dai.query(sql, filters=filters)
    if hasattr(result, "df"):
        return result.df()
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    if isinstance(result, pd.DataFrame):
        return result
    raise TypeError(f"Unsupported dai.query result type: {type(result)!r}")


def _query_with_bigdatasource(
    sql: str, filters: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    from bigdatasource.api import DataSource  # type: ignore

    ds = DataSource()
    result = ds.read_sql(sql)
    if isinstance(result, pd.DataFrame):
        return result
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    raise TypeError(f"Unsupported DataSource result type: {type(result)!r}")


def run_sql(sql: str, filters: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    errors = []
    for fn in (_query_with_dai, _query_with_bigdatasource):
        try:
            return fn(sql, filters=filters)
        except Exception as exc:  # pragma: no cover - platform dependent
            errors.append(f"{fn.__name__}: {exc}")
    raise RuntimeError("No supported BigQuant SQL client worked. " + " | ".join(errors))


def read_local_table(data_dir: str, table: str) -> Optional[pd.DataFrame]:
    if not data_dir:
        return None
    for ext in ("parquet", "csv", "feather"):
        path = os.path.join(data_dir, f"{table}.{ext}")
        if os.path.exists(path):
            if ext == "parquet":
                return pd.read_parquet(path)
            if ext == "feather":
                return pd.read_feather(path)
            return pd.read_csv(path)
    return None


def get_table_sample(cfg: Config, data_dir: str = "") -> pd.DataFrame:
    candidates = cfg.table_candidates if cfg.table == "auto" else [cfg.table]
    errors = []
    for table in candidates:
        local = read_local_table(data_dir, table)
        if local is not None:
            cfg.table = table
            return local.head(10)
        try:
            sample = run_sql(
                f"SELECT * FROM {table} LIMIT 10",
                filters={"date": [cfg.train_start, cfg.train_end]},
            )
            cfg.table = table
            print(f"using_table={table}")
            return sample
        except Exception as exc:
            errors.append(f"{table}: {exc}")
    visible = discover_visible_tables()
    hint = ""
    if visible:
        hint = "\nVisible tables containing stock/bar/alpha:\n" + "\n".join(visible[:80])
    raise RuntimeError(
        "Could not find a usable bar table. "
        "Set CFG.table to the exact table name shown by diagnose_tables.py.\n"
        + "\n".join(errors[:12])
        + hint
    )


def discover_visible_tables() -> List[str]:
    queries = [
        "SHOW TABLES",
        "SELECT table_name FROM information_schema.tables",
        "SELECT table_name FROM duckdb_tables()",
    ]
    found: List[str] = []
    for sql in queries:
        try:
            df = run_sql(sql)
        except Exception:
            continue
        if df.empty:
            continue
        values = df.astype(str).stack().tolist()
        for value in values:
            low = value.lower()
            if any(key in low for key in ("alpha", "stock", "bar", "quote", "kline")):
                if value not in found:
                    found.append(value)
    return found


def discover_columns(cfg: Config, data_dir: str = "") -> Tuple[str, str, str, List[str]]:
    sample = get_table_sample(cfg, data_dir)
    columns = list(sample.columns)
    date_col = cfg.date_col if cfg.date_col in columns else None
    instrument_col = cfg.instrument_col if cfg.instrument_col in columns else None
    time_col = None
    for c in cfg.time_col_candidates:
        if c in columns:
            time_col = c
            break
    if date_col is None:
        for c in ("date", "trading_date", "trade_date"):
            if c in columns:
                date_col = c
                break
    if instrument_col is None:
        for c in ("instrument", "symbol", "order_book_id", "code"):
            if c in columns:
                instrument_col = c
                break
    if time_col is None:
        time_col = date_col
    if date_col is None or instrument_col is None:
        raise ValueError(f"Could not discover date/instrument columns from {columns}")
    feature_cols = [c for c in cfg.raw_feature_candidates if c in columns][: cfg.max_fields]
    if not feature_cols:
        numeric_cols = [
            c
            for c in columns
            if c not in {date_col, instrument_col, time_col}
            and pd.api.types.is_numeric_dtype(sample[c])
        ]
        feature_cols = numeric_cols[: cfg.max_fields]
    if not feature_cols:
        raise ValueError(f"No numeric raw feature columns found in {cfg.table}")
    return date_col, instrument_col, time_col, feature_cols


def load_bars(
    cfg: Config,
    start: str,
    end: str,
    data_dir: str = "",
    extra_start: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    date_col, instrument_col, time_col, feature_cols = discover_columns(cfg, data_dir)
    load_start = extra_start or start
    cols = sorted(set([date_col, instrument_col, time_col] + feature_cols))
    local = read_local_table(data_dir, cfg.table)
    if local is not None:
        df = local.loc[:, [c for c in cols if c in local.columns]].copy()
        df[date_col] = normalize_date(df[date_col])
        df = df[(df[date_col] >= load_start) & (df[date_col] <= end)]
    else:
        col_sql = ", ".join(cols)
        if cfg.fast_query and time_col != date_col:
            sql = (
                f"SELECT {col_sql} FROM ("
                f"SELECT {col_sql}, "
                f"ROW_NUMBER() OVER (PARTITION BY {date_col}, {instrument_col} "
                f"ORDER BY {time_col} DESC) AS _rn "
                f"FROM {cfg.table} "
                f"WHERE {date_col} >= '{load_start}' AND {date_col} <= '{end}'"
                f") AS sampled WHERE _rn <= {int(cfg.bars_per_day)}"
            )
        else:
            sql = (
                f"SELECT {col_sql} FROM {cfg.table} "
                f"WHERE {date_col} >= '{load_start}' AND {date_col} <= '{end}'"
            )
        df = run_sql(sql, filters={date_col: [load_start, end]})
    if df.empty:
        raise ValueError(f"No rows loaded from {cfg.table} between {load_start} and {end}")
    df[date_col] = normalize_date(df[date_col])
    df[instrument_col] = df[instrument_col].astype(str)
    if time_col in df.columns and time_col != date_col:
        df[time_col] = pd.to_datetime(df[time_col])
    else:
        time_col = "_time_for_sort"
        df[time_col] = pd.to_datetime(df[date_col])
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df = df.sort_values([instrument_col, time_col]).reset_index(drop=True)
    df[date_col] = normalize_date(df[date_col])
    meta = {
        "date_col": date_col,
        "instrument_col": instrument_col,
        "time_col": time_col,
        "feature_cols": feature_cols,
        "cfg": asdict(cfg),
    }
    return df, meta


def fit_scaler(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    scaler = {}
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mean = float(s.mean())
        std = float(s.std())
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        scaler[c] = {"mean": mean, "std": std}
    return scaler


def apply_scaler(
    df: pd.DataFrame, feature_cols: List[str], scaler: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        mean = scaler[c]["mean"]
        std = scaler[c]["std"]
        out[c] = ((pd.to_numeric(out[c], errors="coerce").fillna(mean) - mean) / std).clip(
            -8, 8
        )
    return out


def build_daily_labels(
    df: pd.DataFrame,
    date_col: str,
    instrument_col: str,
    time_col: str,
    close_col: str = "close",
) -> pd.DataFrame:
    if close_col not in df.columns:
        raise ValueError("A close column is required to build next-day return labels.")
    daily_close = (
        df.sort_values([instrument_col, time_col])
        .groupby([instrument_col, date_col], as_index=False)[close_col]
        .last()
    )
    daily_close["next_close"] = daily_close.groupby(instrument_col)[close_col].shift(-1)
    daily_close["label"] = daily_close["next_close"] / daily_close[close_col] - 1.0
    labels = daily_close[[date_col, instrument_col, "label"]].dropna()
    labels["label"] = labels.groupby(date_col)["label"].transform(
        lambda s: s - s.mean()
    )
    labels["label"] = labels["label"].clip(-0.2, 0.2).astype("float32")
    return labels


def build_samples(
    df: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    date_col: str,
    instrument_col: str,
    time_col: str,
    feature_cols: List[str],
    seq_len: int,
    start: str,
    end: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end = pd.to_datetime(end).strftime("%Y-%m-%d")
    arrays: List[np.ndarray] = []
    targets: List[float] = []
    index_rows: List[Tuple[str, str]] = []
    label_map = None
    if labels is not None:
        label_map = labels.set_index([date_col, instrument_col])["label"]
    for instrument, g in df.groupby(instrument_col, sort=False):
        g = g.sort_values(time_col)
        dates = normalize_date(g[date_col]).to_numpy()
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        unique_dates, end_positions = np.unique(dates, return_index=False, return_counts=False), None
        unique_dates, counts = np.unique(dates, return_counts=True)
        end_positions = np.cumsum(counts)
        for date, end_pos in zip(unique_dates, end_positions):
            if date < start or date > end:
                continue
            end_pos = int(end_pos)
            begin_pos = max(0, end_pos - seq_len)
            seq = feats[begin_pos:end_pos]
            if len(seq) == 0:
                continue
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), len(feature_cols)), dtype=np.float32)
                seq = np.vstack([pad, seq])
            if label_map is not None:
                key = (date, instrument)
                if key not in label_map.index:
                    continue
                targets.append(float(label_map.loc[key]))
            else:
                targets.append(np.nan)
            arrays.append(seq.astype(np.float32))
            index_rows.append((date, instrument))
    if not arrays:
        raise ValueError("No samples were built. Check dates, table, and columns.")
    x = np.stack(arrays)
    y = np.asarray(targets, dtype=np.float32)
    index = pd.DataFrame(index_rows, columns=[date_col, instrument_col])
    return x, y, index


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def scores_to_submission(
    index: pd.DataFrame,
    raw_scores: Iterable[float],
    date_col: str,
    instrument_col: str,
) -> pd.DataFrame:
    out = index.copy()
    out["score"] = np.asarray(list(raw_scores), dtype=np.float32)
    out["score"] = out.groupby(date_col)["score"].transform(rank_to_score)
    out = out.rename(columns={date_col: "date", instrument_col: "instrument"})
    out = out[["date", "instrument", "score"]].sort_values(["date", "instrument"])
    out["date"] = normalize_date(out["date"])
    out["instrument"] = out["instrument"].astype(str)
    return out.reset_index(drop=True)
