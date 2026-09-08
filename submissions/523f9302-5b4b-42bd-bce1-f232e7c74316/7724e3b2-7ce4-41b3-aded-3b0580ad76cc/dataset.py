from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from core import FeatureTransform
from core import dataset_for_table, date_ts, load_daily_frame, read_table_filtered
from core import Normalizer
from core import (
    metrics_platform_score_enabled,
    platform_alignment_enabled,
    style_exposure_cols,
)


class DailyPatchBuilder:
    """Build one cross-sectional daily batch at a time.

    Leakage contract:
    - Features use bars with timestamp <= signal_date close only.
    - Labels come from target_day = signal_date + horizon trading days.
    - Normalizers are supplied externally and should be fit on train dates only.
    """

    def __init__(self, config: dict, transform: FeatureTransform, normalizers: dict[str, Normalizer]):
        self.config = config
        self.data_root = Path(config["data_root"])
        self.transform = transform
        self.normalizers = normalizers
        self.freq_cfg = config["frequency"]
        self.freq_keys = list(self.freq_cfg.keys())
        self.tables = config["tables"]
        self.max_instruments = config["universe"].get("max_instruments_per_day")
        self.candidate_buffer_multiplier = int(config["universe"].get("candidate_buffer_multiplier", 4))
        self.label_col = config["label"]["column"]
        self.horizon = int(config["label"]["horizon_trading_days"])
        self.padding_enabled = bool(config.get("data", {}).get("padding", True))
        self.min_real_bars = int(config.get("data", {}).get("min_real_bars", 1))
        self.platform_alignment = platform_alignment_enabled(config)
        self.metrics_platform_score = metrics_platform_score_enabled(config)
        need_score_exposure = self.platform_alignment or self.metrics_platform_score
        if need_score_exposure:
            cols = style_exposure_cols(config)
            if len(cols) == 1 and cols[0] == "auto":
                cols = tuple(self._auto_exposure_cols())
            self.score_exposure_cols = cols
        else:
            self.score_exposure_cols = ()

        start = date_ts(config["dates"]["train_start"])
        date_values = [date_ts(value) for key, value in config["dates"].items() if key.endswith("_end")]
        end = max(date_values)
        # Preserve the realized return separately from the optimization label.
        # Residual labels may be used for training, while all reported ranking
        # metrics must remain comparable on raw return.
        self.raw_labels = self._load_raw_labels(start, end + pd.Timedelta(days=10))
        self.labels = self._transform_labels(self.raw_labels, start, end)
        self.universe = self._load_universe(start, end)
        self.score_exposure = self._load_score_exposure(start, end)
        self.trading_days = sorted(self.labels["day"].drop_duplicates().tolist())

    def _load_raw_labels(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        source = str(self.config.get("label", {}).get("source", "factorlib"))
        table = self.tables.get(source, source)
        frame = load_daily_frame(
            self.data_root,
            table,
            start,
            end + pd.Timedelta(days=1),
            ["date", "instrument", self.label_col],
        )
        return frame[["day", "instrument", self.label_col]].dropna().copy()

    def _transform_labels(
        self,
        raw_labels: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        labels = raw_labels.copy()
        mode = self.config.get("label", {}).get("mode", "raw_next_return")
        if mode == "raw_next_return":
            return labels
        if mode == "daily_demean_return":
            return self._daily_demean_labels(labels)
        if mode == "exposure_residual_return":
            return self._exposure_residual_labels(labels, start, end)
        raise ValueError(f"Unknown label mode {mode}")

    def _daily_demean_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        labels[self.label_col] = labels[self.label_col] - labels.groupby("day")[self.label_col].transform("mean")
        return labels.dropna()

    def _auto_exposure_cols(self) -> list[str]:
        dataset = dataset_for_table(self.data_root, self.tables["exposure"])
        skip = {"date", "instrument", "year", "month", "ret", "weights", "float_market_cap"}
        cols = []
        for field in dataset.schema:
            if field.name in skip:
                continue
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                cols.append(field.name)
        return cols

    def _exposure_residual_labels(
        self,
        labels: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        residual_cfg = self.config.get("residualization", {})
        cols = residual_cfg.get("exposure_cols", "auto")
        if cols == "auto":
            cols = self._auto_exposure_cols()
        cols = list(cols)
        if not cols:
            return self._daily_demean_labels(labels)

        exposure = load_daily_frame(
            self.data_root,
            self.tables["exposure"],
            start,
            end + pd.Timedelta(days=1),
            ["date", "instrument"] + cols,
        )
        merged = labels.merge(exposure[["day", "instrument"] + cols], on=["day", "instrument"], how="left")
        min_obs = int(residual_cfg.get("min_obs", 50))
        ridge = float(residual_cfg.get("ridge", 1e-6))
        pieces = []
        for day, sub in merged.groupby("day", sort=True):
            y = sub[self.label_col].astype("float64").to_numpy()
            x = sub[cols].astype("float64").replace([np.inf, -np.inf], np.nan)
            valid = np.isfinite(y) & x.notna().all(axis=1).to_numpy()
            if valid.sum() < max(min_obs, len(cols) + 2):
                out = sub.loc[np.isfinite(y), ["day", "instrument", self.label_col]].copy()
                out[self.label_col] = out[self.label_col] - out[self.label_col].mean()
                pieces.append(out)
                continue

            x_values = x.loc[valid].to_numpy()
            x_mean = x_values.mean(axis=0, keepdims=True)
            x_std = x_values.std(axis=0, keepdims=True) + 1e-12
            x_values = (x_values - x_mean) / x_std
            x_design = np.concatenate([np.ones((x_values.shape[0], 1)), x_values], axis=1)
            y_values = y[valid]
            lhs = x_design.T @ x_design + ridge * np.eye(x_design.shape[1])
            lhs[0, 0] -= ridge
            beta = np.linalg.solve(lhs, x_design.T @ y_values)
            resid = y_values - x_design @ beta

            out = sub.loc[valid, ["day", "instrument"]].copy()
            out[self.label_col] = resid.astype(np.float32)
            pieces.append(out)
        return pd.concat(pieces, ignore_index=True).dropna()

    def _load_score_exposure(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame | None:
        if not (self.platform_alignment or self.metrics_platform_score):
            return None
        frame = load_daily_frame(
            self.data_root,
            self.tables["exposure"],
            start,
            end + pd.Timedelta(days=1),
            ["date", "instrument", *self.score_exposure_cols],
        )
        frame["instrument"] = frame["instrument"].astype(str)
        frame = frame[
            ["day", "instrument", *self.score_exposure_cols]
        ].drop_duplicates(["day", "instrument"], keep="last")
        return frame.set_index(["day", "instrument"]).sort_index()

    def score_exposure_for_day(
        self,
        signal_day: pd.Timestamp,
    ) -> pd.DataFrame | None:
        if self.score_exposure is None:
            return None
        day = pd.Timestamp(signal_day).normalize()
        try:
            frame = self.score_exposure.xs(day, level="day")
        except KeyError:
            return pd.DataFrame(columns=self.score_exposure_cols)
        values = frame.loc[:, list(self.score_exposure_cols)].astype("float64")
        valid = np.isfinite(values.to_numpy()).all(axis=1)
        return values.loc[valid]

    def _load_universe(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = load_daily_frame(
            self.data_root,
            self.tables["instruments"],
            start,
            end + pd.Timedelta(days=1),
            ["date", "instrument"],
        )
        return frame[["day", "instrument"]].drop_duplicates()

    def signal_days(self, start: str, end: str) -> list[pd.Timestamp]:
        lo, hi = date_ts(start), date_ts(end)
        days = [d for d in self.trading_days if lo <= d <= hi]
        max_idx = len(self.trading_days) - self.horizon - 1
        min_idx = max(int(cfg["lookback_days"]) for cfg in self.freq_cfg.values()) - 1
        return [d for d in days if min_idx <= self.trading_days.index(d) <= max_idx]

    def _target_day(self, signal_day: pd.Timestamp) -> pd.Timestamp | None:
        try:
            idx = self.trading_days.index(signal_day)
        except ValueError:
            return None
        target_idx = idx + self.horizon
        if target_idx >= len(self.trading_days):
            return None
        return self.trading_days[target_idx]

    def _feature_start_day(self, signal_day: pd.Timestamp, lookback_days: int) -> pd.Timestamp:
        idx = self.trading_days.index(signal_day)
        start_idx = max(0, idx - lookback_days + 1)
        return self.trading_days[start_idx]

    def _candidate_instruments(self, signal_day: pd.Timestamp, target_day: pd.Timestamp) -> list[str]:
        universe = self.universe.loc[self.universe["day"] == signal_day, "instrument"].astype(str)
        labels = self.labels.loc[self.labels["day"] == target_day, "instrument"].astype(str)
        names = sorted(set(universe).intersection(set(labels)))
        if self.max_instruments:
            # Load a larger candidate pool, then truncate after history checks.
            # Truncating before the lookback check can make a day disappear just
            # because the first N codes have insufficient high-frequency history.
            names = names[: int(self.max_instruments) * max(1, self.candidate_buffer_multiplier)]
        return names

    def _lookback_trading_days(
        self,
        signal_day: pd.Timestamp,
        lookback_days: int,
    ) -> list[pd.Timestamp]:
        idx = self.trading_days.index(signal_day)
        days = self.trading_days[idx - lookback_days + 1 : idx + 1]
        if len(days) != lookback_days:
            raise ValueError(
                f"Expected {lookback_days} trading days ending at {signal_day}, got {len(days)}"
            )
        return days

    def _expected_bar_index(self, day: pd.Timestamp, cfg: dict) -> pd.DatetimeIndex | None:
        starts = cfg.get("session_starts")
        lengths = cfg.get("session_lengths")
        if starts is None or lengths is None:
            return None
        if len(starts) != len(lengths):
            raise ValueError("session_starts and session_lengths must have the same length")
        pieces = []
        for start, length in zip(starts, lengths):
            start_time = pd.Timestamp(str(start)).time()
            session_start = pd.Timestamp.combine(pd.Timestamp(day).date(), start_time)
            pieces.append(pd.date_range(session_start, periods=int(length), freq="min"))
        index = pieces[0]
        for piece in pieces[1:]:
            index = index.append(piece)
        if len(index) != int(cfg["bars_per_day"]):
            raise ValueError("Configured trading sessions must produce bars_per_day timestamps")
        return index

    def _align_instrument_sequence(
        self,
        sub: pd.DataFrame,
        freq_key: str,
        signal_day: pd.Timestamp,
    ) -> np.ndarray | None:
        cfg = self.freq_cfg[freq_key]
        lookback_days = int(cfg["lookback_days"])
        bars_per_day = int(cfg["bars_per_day"])
        expected_days = self._lookback_trading_days(signal_day, lookback_days)
        sub = sub.copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub["_day"] = sub["date"].dt.normalize()
        aligned_days = []
        for day in expected_days:
            rows = sub.loc[sub["_day"] == pd.Timestamp(day).normalize()].copy()
            rows = rows.sort_values("date").drop_duplicates("date", keep="last")
            expected_index = self._expected_bar_index(day, cfg)
            if expected_index is not None:
                rows = rows.drop(columns=["_day"]).set_index("date").reindex(expected_index)
                if not self.padding_enabled and rows.isna().all(axis=1).any():
                    return None
                rows = rows.reset_index().rename(columns={"index": "date"})
            elif len(rows) != bars_per_day:
                if not self.padding_enabled:
                    return None
                rows = rows.drop(columns=["_day"]).iloc[:bars_per_day]
                rows = rows.reindex(range(bars_per_day))
            else:
                rows = rows.drop(columns=["_day"])
            features = self.transform.frame_to_features(rows)
            aligned_days.append(self.normalizers[freq_key].apply(features))
        sequence = np.concatenate(aligned_days, axis=0).astype(np.float32, copy=False)
        if sequence.shape != (int(cfg["seq_len"]), self.transform.n_features):
            raise ValueError(f"Day-aligned sequence has unexpected shape {sequence.shape}")
        return sequence

    def _build_feature_day_from_frame(
        self,
        frame: pd.DataFrame,
        freq_key: str,
        day: pd.Timestamp,
    ) -> dict | None:
        cfg = self.freq_cfg[freq_key]
        expected_index = self._expected_bar_index(day, cfg)
        if expected_index is None:
            raise ValueError("Daily feature cache requires explicit trading sessions")
        if frame.empty:
            return None

        rows = frame.copy()
        rows["date"] = pd.to_datetime(rows["date"])
        rows["instrument"] = rows["instrument"].astype(str)
        rows = rows.loc[rows["date"].isin(expected_index)]
        rows = rows.sort_values(["instrument", "date"]).drop_duplicates(
            ["instrument", "date"],
            keep="last",
        )
        if rows.empty:
            return None

        counts = rows.groupby("instrument", sort=False).size()
        if self.padding_enabled:
            valid = counts[counts >= self.min_real_bars]
        else:
            valid = counts[counts == len(expected_index)]
        instruments = sorted(valid.index.astype(str).tolist())
        if not instruments:
            return None

        rows = rows.loc[rows["instrument"].isin(instruments)]
        full_index = pd.MultiIndex.from_product(
            [instruments, expected_index],
            names=["instrument", "date"],
        )
        aligned = (
            rows.set_index(["instrument", "date"])
            .reindex(full_index)
            .reset_index()
        )
        features = self.transform.frame_to_features(aligned)
        normalized = self.normalizers[freq_key].apply(features)
        x = normalized.reshape(
            len(instruments),
            int(cfg["bars_per_day"]),
            self.transform.n_features,
        )
        return {
            "instrument": np.asarray(instruments, dtype=str),
            "x": x.astype(np.float32, copy=False),
        }

    def build_feature_days(
        self,
        freq_key: str,
        days: list[pd.Timestamp],
    ) -> dict[pd.Timestamp, dict | None]:
        normalized_days = sorted({pd.Timestamp(day).normalize() for day in days})
        if not normalized_days:
            return {}
        frame = read_table_filtered(
            self.data_root,
            self.tables[freq_key],
            self.transform.required_cols,
            normalized_days[0],
            normalized_days[-1] + pd.Timedelta(days=1),
            instruments=None,
        )
        requested = set(normalized_days)
        out: dict[pd.Timestamp, dict | None] = {day: None for day in normalized_days}
        if frame.empty:
            return out
        frame["date"] = pd.to_datetime(frame["date"])
        frame["_day"] = frame["date"].dt.normalize()
        for day, rows in frame.groupby("_day", sort=False):
            normalized = pd.Timestamp(day).normalize()
            if normalized in requested:
                out[normalized] = self._build_feature_day_from_frame(
                    rows.drop(columns=["_day"]),
                    freq_key,
                    normalized,
                )
        return out


    def _load_freq_sequences(
        self,
        freq_key: str,
        signal_day: pd.Timestamp,
        instruments: list[str],
    ) -> dict[str, np.ndarray]:
        cfg = self.freq_cfg[freq_key]
        seq_len = int(cfg["seq_len"])
        start_day = self._feature_start_day(signal_day, int(cfg["lookback_days"]))
        if not instruments:
            return {}
        frame = read_table_filtered(
            self.data_root,
            self.tables[freq_key],
            self.transform.required_cols,
            start_day,
            signal_day + pd.Timedelta(days=1),
            instruments,
        )
        if frame.empty and not self.padding_enabled:
            return {}

        if not frame.empty:
            frame = frame.sort_values(["instrument", "date"])
        out: dict[str, np.ndarray] = {}
        instrument_set = set(instruments)
        for instrument, sub in frame.groupby("instrument", sort=False):
            instrument = str(instrument)
            if instrument not in instrument_set:
                continue
            if len(sub) < self.min_real_bars:
                continue
            x = self._align_instrument_sequence(sub, freq_key, signal_day)
            if x is None:
                continue
            out[instrument] = x

        if self.padding_enabled and self.min_real_bars <= 0:
            zeros = np.zeros((seq_len, self.transform.n_features), dtype=np.float32)
            for instrument in instruments:
                out.setdefault(instrument, zeros)
        return out

    def build_day(self, signal_day: pd.Timestamp):
        target_day = self._target_day(signal_day)
        if target_day is None:
            return None

        instruments = self._candidate_instruments(signal_day, target_day)
        if not instruments:
            return None

        freq_seqs = {}
        for freq_key in self.freq_keys:
            freq_seqs[freq_key] = self._load_freq_sequences(freq_key, signal_day, instruments)
        labels = self.labels[self.labels["day"] == target_day].set_index("instrument")[self.label_col]
        raw_labels = self.raw_labels[self.raw_labels["day"] == target_day].set_index(
            "instrument"
        )[self.label_col]
        score_exposure = self.score_exposure_for_day(signal_day)
        if self.platform_alignment and (score_exposure is None or score_exposure.empty):
            return None

        names, freq_arrays, y, raw_y = [], {freq_key: [] for freq_key in self.freq_keys}, [], []
        for instrument in instruments:
            if not all(instrument in freq_seqs[freq_key] for freq_key in self.freq_keys):
                continue
            if instrument not in labels.index:
                continue
            if instrument not in raw_labels.index:
                continue
            value = labels.loc[instrument]
            raw_value = raw_labels.loc[instrument]
            if not np.isfinite(value) or not np.isfinite(raw_value):
                continue
            if score_exposure is not None and instrument not in score_exposure.index:
                continue
            names.append(instrument)
            for freq_key in self.freq_keys:
                freq_arrays[freq_key].append(freq_seqs[freq_key][instrument])
            y.append(float(value))
            raw_y.append(float(raw_value))
            if self.max_instruments and len(names) >= int(self.max_instruments):
                break

        if not names:
            return None

        result = {
            "date": signal_day,
            "target_date": target_day,
            "instrument": names,
            "y": np.asarray(y, dtype=np.float32),
            "raw_y": np.asarray(raw_y, dtype=np.float32),
        }
        if score_exposure is not None:
            result["score_exposure"] = score_exposure.loc[
                names,
                list(self.score_exposure_cols),
            ].to_numpy(dtype=np.float32)
        for freq_key in self.freq_keys:
            result[f"x_{freq_key}"] = np.stack(freq_arrays[freq_key]).astype(np.float32)
        return result


import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


CACHE_VERSION = "daily-features-v2"


class CachedDailyFeatureBuilder:
    """Persist one normalized 1-minute tensor per trading day and assemble windows lazily."""

    def __init__(
        self,
        raw_builder,
        cache_dir: str | Path,
        write: bool = True,
        month_workers: int = 1,
        open_day_limit: int = 16,
    ):
        self.raw_builder = raw_builder
        self.cache_dir = Path(cache_dir)
        self.write = bool(write)
        self.month_workers = max(1, int(month_workers))
        self.open_day_limit = max(1, int(open_day_limit))
        self._freq_keys = list(raw_builder.freq_keys)
        self._open_days: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, int]]] = OrderedDict()
        if self.write:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._validate_manifest()

    @property
    def freq_keys(self):
        return self._freq_keys

    def signal_days(self, start: str, end: str) -> list[pd.Timestamp]:
        return self.raw_builder.signal_days(start, end)

    def _manifest_payload(self) -> dict:
        frequencies = {}
        for freq_key in self._freq_keys:
            cfg = self.raw_builder.freq_cfg[freq_key]
            normalizer = self.raw_builder.normalizers[freq_key]
            frequencies[freq_key] = {
                "table": self.raw_builder.tables[freq_key],
                "bars_per_day": int(cfg["bars_per_day"]),
                "session_starts": list(cfg.get("session_starts", [])),
                "session_lengths": [int(value) for value in cfg.get("session_lengths", [])],
                "normalizer_mean": normalizer.mean.tolist(),
                "normalizer_std": normalizer.std.tolist(),
            }
        return {
            "version": CACHE_VERSION,
            "dtype": "float32",
            "data_root": str(self.raw_builder.data_root),
            "feature_names": list(self.raw_builder.transform.feature_names),
            "frequencies": frequencies,
        }

    def _validate_manifest(self) -> None:
        path = self.cache_dir / "manifest.json"
        expected = self._manifest_payload()
        if path.exists():
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != expected:
                raise RuntimeError(
                    f"Cache manifest mismatch at {path}; use a new cache directory"
                )
            return
        if not self.write:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _day_paths(self, freq_key: str, day: pd.Timestamp) -> tuple[Path, Path]:
        day = pd.Timestamp(day).normalize()
        directory = self.cache_dir / freq_key / f"year={day.year:04d}" / f"month={day.month:02d}"
        stem = str(day.date())
        return directory / f"{stem}.x.npy", directory / f"{stem}.instrument.npy"

    def _has_day(self, freq_key: str, day: pd.Timestamp) -> bool:
        x_path, instrument_path = self._day_paths(freq_key, day)
        return x_path.exists() and instrument_path.exists()

    def _write_day(self, freq_key: str, day: pd.Timestamp, payload: dict) -> None:
        x_path, instrument_path = self._day_paths(freq_key, day)
        x_path.parent.mkdir(parents=True, exist_ok=True)
        x = np.asarray(payload["x"], dtype=np.float32)
        instruments = np.asarray(payload["instrument"], dtype=str)
        if x.ndim != 3 or x.shape[0] != instruments.shape[0]:
            raise ValueError(f"Invalid daily cache payload for {freq_key} {day}")

        # Unique temp suffix so concurrent writers (e.g. a parallel cache
        # prebuild running next to training) never clobber each other's
        # in-progress file. Content is deterministic, so last-writer-wins is
        # safe.
        tmp_suffix = f".tmp.{os.getpid()}"
        x_tmp = x_path.with_suffix(".x.npy" + tmp_suffix)
        instrument_tmp = instrument_path.with_suffix(
            ".instrument.npy" + tmp_suffix
        )
        with x_tmp.open("wb") as handle:
            np.save(handle, x, allow_pickle=False)
        with instrument_tmp.open("wb") as handle:
            np.save(handle, instruments, allow_pickle=False)
        instrument_tmp.replace(instrument_path)
        x_tmp.replace(x_path)

    def _load_day(self, freq_key: str, day: pd.Timestamp):
        day_key = str(pd.Timestamp(day).date())
        key = (freq_key, day_key)
        cached = self._open_days.pop(key, None)
        if cached is not None:
            self._open_days[key] = cached
            return cached

        x_path, instrument_path = self._day_paths(freq_key, day)
        x = np.load(x_path, mmap_mode="r", allow_pickle=False)
        instruments = np.load(instrument_path, allow_pickle=False).astype(str)
        if x.ndim != 3 or x.shape[0] != instruments.shape[0]:
            raise RuntimeError(f"Corrupt daily cache entry: {x_path}")
        index = {instrument: idx for idx, instrument in enumerate(instruments.tolist())}
        value = (x, instruments, index)
        self._open_days[key] = value
        while len(self._open_days) > self.open_day_limit:
            self._open_days.popitem(last=False)
        return value

    @staticmethod
    def _month_groups(days: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
        groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
        for day in sorted({pd.Timestamp(value).normalize() for value in days}):
            groups.setdefault((day.year, day.month), []).append(day)
        return list(groups.values())

    def _build_month(self, freq_key: str, days: list[pd.Timestamp]) -> tuple[int, int]:
        missing = [day for day in days if not self._has_day(freq_key, day)]
        if not missing:
            return 0, len(days)
        payloads = self.raw_builder.build_feature_days(freq_key, missing)
        built = 0
        for day in missing:
            payload = payloads.get(pd.Timestamp(day).normalize())
            if payload is None:
                continue
            if self.write:
                self._write_day(freq_key, day, payload)
            built += 1
        return built, len(days) - len(missing)

    def prebuild(self, split_days: dict[str, list[pd.Timestamp]]) -> None:
        if not self.write:
            raise RuntimeError("Cannot prebuild a read-only daily feature cache")
        signal_days = sorted(
            {pd.Timestamp(day).normalize() for days in split_days.values() for day in days}
        )
        for freq_key in self._freq_keys:
            lookback = int(self.raw_builder.freq_cfg[freq_key]["lookback_days"])
            feature_days = sorted(
                {
                    feature_day
                    for signal_day in signal_days
                    for feature_day in self.raw_builder._lookback_trading_days(signal_day, lookback)
                }
            )
            groups = self._month_groups(feature_days)
            existing = sum(self._has_day(freq_key, day) for day in feature_days)
            print(
                f"daily_cache freq={freq_key} required={len(feature_days)} "
                f"existing={existing} missing={len(feature_days) - existing} "
                f"month_workers={self.month_workers}",
                flush=True,
            )
            if self.month_workers == 1:
                results = [(days, self._build_month(freq_key, days)) for days in groups]
            else:
                results = []
                with ThreadPoolExecutor(max_workers=self.month_workers) as pool:
                    future_to_days = {
                        pool.submit(self._build_month, freq_key, days): days for days in groups
                    }
                    for future in as_completed(future_to_days):
                        results.append((future_to_days[future], future.result()))
            built_total = 0
            for days, (built, reused) in results:
                built_total += built
                first = days[0]
                print(
                    f"daily_cache freq={freq_key} month={first.year:04d}-{first.month:02d} "
                    f"built={built} reused={reused}",
                    flush=True,
                )
            print(
                f"daily_cache_complete freq={freq_key} built={built_total} "
                f"total={len(feature_days)}",
                flush=True,
            )

    def _entries_for_days(self, freq_key: str, days: list[pd.Timestamp]):
        entries = {}
        missing = [day for day in days if not self._has_day(freq_key, day)]
        if missing:
            for month_days in self._month_groups(missing):
                payloads = self.raw_builder.build_feature_days(freq_key, month_days)
                for day, payload in payloads.items():
                    if self.write and payload is not None:
                        self._write_day(freq_key, day, payload)
                    entries[pd.Timestamp(day).normalize()] = payload
        for day in days:
            normalized = pd.Timestamp(day).normalize()
            if self._has_day(freq_key, normalized):
                entries[normalized] = self._load_day(freq_key, normalized)
            elif normalized in entries and entries[normalized] is not None:
                payload = entries[normalized]
                instruments = np.asarray(payload["instrument"], dtype=str)
                index = {instrument: idx for idx, instrument in enumerate(instruments.tolist())}
                entries[normalized] = (np.asarray(payload["x"], dtype=np.float32), instruments, index)
            else:
                return None
        return entries

    def build_day(self, signal_day: pd.Timestamp):
        signal_day = pd.Timestamp(signal_day).normalize()
        target_day = self.raw_builder._target_day(signal_day)
        if target_day is None:
            return None
        candidates = self.raw_builder._candidate_instruments(signal_day, target_day)
        if not candidates:
            return None

        entries_by_freq = {}
        days_by_freq = {}
        for freq_key in self._freq_keys:
            lookback = int(self.raw_builder.freq_cfg[freq_key]["lookback_days"])
            days = self.raw_builder._lookback_trading_days(signal_day, lookback)
            entries = self._entries_for_days(freq_key, days)
            if entries is None:
                return None
            entries_by_freq[freq_key] = entries
            days_by_freq[freq_key] = days

        labels = self.raw_builder.labels[
            self.raw_builder.labels["day"] == target_day
        ].set_index("instrument")[self.raw_builder.label_col]
        raw_labels = self.raw_builder.raw_labels[
            self.raw_builder.raw_labels["day"] == target_day
        ].set_index("instrument")[self.raw_builder.label_col]
        platform_alignment = bool(
            getattr(self.raw_builder, "platform_alignment", False)
        )
        metrics_platform_score = bool(
            getattr(self.raw_builder, "metrics_platform_score", False)
        )
        need_score_exposure = platform_alignment or metrics_platform_score
        score_exposure = (
            self.raw_builder.score_exposure_for_day(signal_day)
            if need_score_exposure
            else None
        )
        if platform_alignment and (
            score_exposure is None or score_exposure.empty
        ):
            return None
        names = []
        for instrument in candidates:
            if instrument not in labels.index:
                continue
            value = labels.loc[instrument]
            if not np.isfinite(value):
                continue
            if instrument not in raw_labels.index:
                continue
            raw_value = raw_labels.loc[instrument]
            if not np.isfinite(raw_value):
                continue
            if score_exposure is not None and instrument not in score_exposure.index:
                continue
            available = all(
                instrument in entries_by_freq[freq_key][day][2]
                for freq_key in self._freq_keys
                for day in days_by_freq[freq_key]
            )
            if not available:
                continue
            names.append(instrument)
            if self.raw_builder.max_instruments and len(names) >= int(self.raw_builder.max_instruments):
                break
        if not names:
            return None

        result = {
            "date": signal_day,
            "target_date": target_day,
            "instrument": names,
            "y": labels.loc[names].to_numpy(dtype=np.float32),
            "raw_y": raw_labels.loc[names].to_numpy(dtype=np.float32),
        }
        if score_exposure is not None:
            result["score_exposure"] = score_exposure.loc[
                names,
                list(self.raw_builder.score_exposure_cols),
            ].to_numpy(dtype=np.float32)
        for freq_key in self._freq_keys:
            cfg = self.raw_builder.freq_cfg[freq_key]
            bars_per_day = int(cfg["bars_per_day"])
            sequence = np.empty(
                (
                    len(names),
                    len(days_by_freq[freq_key]) * bars_per_day,
                    len(self.raw_builder.transform.feature_names),
                ),
                dtype=np.float32,
            )
            for day_idx, day in enumerate(days_by_freq[freq_key]):
                x, _instruments, index = entries_by_freq[freq_key][day]
                rows = np.fromiter(
                    (index[instrument] for instrument in names),
                    dtype=np.int64,
                    count=len(names),
                )
                start = day_idx * bars_per_day
                sequence[:, start : start + bars_per_day] = x[rows]
            result[f"x_{freq_key}"] = sequence
        return result


__all__ = ["CACHE_VERSION", "CachedDailyFeatureBuilder"]


from dataset import CachedDailyFeatureBuilder


def maybe_enable_cache(config: dict, raw_builder, output_dir, split_days: dict):
    cache_cfg = config.get("cache", {})
    if not bool(cache_cfg.get("enabled", False)):
        return raw_builder

    mode = str(cache_cfg.get("mode", "daily_features"))
    if mode != "daily_features":
        raise ValueError(f"Unsupported cache mode {mode!r}")
    cache_dir = cache_cfg.get("dir", output_dir / "cache")
    builder = CachedDailyFeatureBuilder(
        raw_builder,
        cache_dir,
        write=bool(cache_cfg.get("write", True)),
        month_workers=int(cache_cfg.get("month_workers", 1)),
        open_day_limit=int(cache_cfg.get("open_day_limit", 16)),
    )
    print(f"cache_dir={cache_dir} mode={mode}", flush=True)
    if bool(cache_cfg.get("prebuild", True)):
        builder.prebuild(split_days)
    return builder


__all__ = ["maybe_enable_cache"]


from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dataset import DailyPatchBuilder
from core import read_table_filtered


@dataclass
class DayCoverage:
    signal_day: str
    target_day: str | None
    universe: int
    labels: int
    candidates: int
    has_all: int
    selected: int
    missing_all: int
    buildable: bool
    freq_counts: dict[str, int]
    freq_missing: dict[str, int]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class TrainingCoverageProfiler:
    """Profile whether daily training samples can be built before training.

    This checks coverage at the tensor-construction level, not just raw table
    row counts. A day is buildable only if enough instruments have every
    configured lookback window plus a target label.
    """

    def __init__(self, builder: DailyPatchBuilder):
        self.builder = builder
        self.config = builder.config
        self.data_root = Path(self.config["data_root"])

    def profile_days(self, days: list[pd.Timestamp], progress_every: int = 5) -> pd.DataFrame:
        records = []
        total = len(days)
        for step, day in enumerate(days, 1):
            records.append(self.profile_day(day).to_dict())
            if progress_every > 0 and (step % progress_every == 0 or step == total):
                print(f"coverage step={step}/{total} day={day.date()}", flush=True)
        return pd.DataFrame.from_records(records)

    def profile_day(self, signal_day: pd.Timestamp) -> DayCoverage:
        target_day = self.builder._target_day(signal_day)
        if target_day is None:
            return DayCoverage(
                signal_day=str(signal_day.date()),
                target_day=None,
                universe=0,
                labels=0,
                candidates=0,
                has_all=0,
                selected=0,
                missing_all=0,
                buildable=False,
                freq_counts={},
                freq_missing={},
            )

        universe = self.builder.universe.loc[
            self.builder.universe["day"] == signal_day, "instrument"
        ].astype(str)
        labels = self.builder.labels.loc[
            self.builder.labels["day"] == target_day, "instrument"
        ].astype(str)
        candidates = self.builder._candidate_instruments(signal_day, target_day)
        if not candidates:
            return DayCoverage(
                signal_day=str(signal_day.date()),
                target_day=str(target_day.date()),
                universe=int(universe.nunique()),
                labels=int(labels.nunique()),
                candidates=0,
                has_all=0,
                selected=0,
                missing_all=0,
                buildable=False,
                freq_counts={},
                freq_missing={},
            )

        ok_sets = {
            freq_key: self._instruments_with_full_window(freq_key, signal_day, candidates)
            for freq_key in self.builder.freq_keys
        }
        has_all = set(candidates).intersection(set(labels))
        for ok in ok_sets.values():
            has_all = has_all.intersection(ok)
        max_instruments = self.builder.max_instruments
        selected = len(has_all) if not max_instruments else min(len(has_all), int(max_instruments))
        candidate_set = set(candidates)
        union_ok = set()
        for ok in ok_sets.values():
            union_ok = union_ok.union(ok)
        freq_counts = {freq_key: len(ok) for freq_key, ok in ok_sets.items()}
        freq_missing = {freq_key: len(candidate_set.difference(ok)) for freq_key, ok in ok_sets.items()}
        return DayCoverage(
            signal_day=str(signal_day.date()),
            target_day=str(target_day.date()),
            universe=int(universe.nunique()),
            labels=int(labels.nunique()),
            candidates=len(candidates),
            has_all=len(has_all),
            selected=selected,
            missing_all=len(candidate_set.difference(union_ok)),
            buildable=selected > 0,
            freq_counts=freq_counts,
            freq_missing=freq_missing,
        )

    def _instruments_with_full_window(
        self,
        freq_key: str,
        signal_day: pd.Timestamp,
        instruments: list[str],
    ) -> set[str]:
        freq_cfg = self.builder.freq_cfg[freq_key]
        seq_len = int(freq_cfg["seq_len"])
        start_day = self.builder._feature_start_day(signal_day, int(freq_cfg["lookback_days"]))
        instruments_for_filter: list[int] | list[str] = instruments
        columns = ["date", "instrument"]
        if self.builder.tables[freq_key] == self.builder.tables.get("bar1m") and self.builder.instrument_to_id:
            instruments_for_filter = [
                self.builder.instrument_to_id[name]
                for name in instruments
                if name in self.builder.instrument_to_id
            ]
            columns = ["date", "instrument_id"]
        if not instruments_for_filter:
            return set()
        frame = read_table_filtered(
            self.data_root,
            self.builder.tables[freq_key],
            columns,
            start_day,
            signal_day + pd.Timedelta(days=1),
            instruments_for_filter,
        )
        if frame.empty:
            return set()
        if "instrument_id" in frame.columns:
            counts = frame.groupby("instrument_id", sort=False)["date"].size()
            ids = counts[counts >= seq_len].index.astype(int)
            return {
                self.builder.id_to_instrument[int(instrument_id)]
                for instrument_id in ids
                if int(instrument_id) in self.builder.id_to_instrument
            }
        counts = frame.groupby("instrument", sort=False)["date"].size()
        return set(counts[counts >= seq_len].index.astype(str))


def summarize_coverage(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"days": 0, "buildable_days": 0, "build_rate": 0.0}
    by_year = []
    for year, sub in frame.groupby(frame["signal_day"].str.slice(0, 4), sort=True):
        by_year.append(
            {
                "year": year,
                "days": int(len(sub)),
                "buildable_days": int(sub["buildable"].sum()),
                "build_rate": float(sub["buildable"].mean()),
                "mean_selected": float(sub["selected"].mean()),
                "mean_has_all": float(sub["has_all"].mean()),
            }
        )
    by_month = []
    for month, sub in frame.groupby(frame["signal_day"].str.slice(0, 7), sort=True):
        by_month.append(
            {
                "month": month,
                "days": int(len(sub)),
                "buildable_days": int(sub["buildable"].sum()),
                "build_rate": float(sub["buildable"].mean()),
                "mean_selected": float(sub["selected"].mean()),
                "mean_has_all": float(sub["has_all"].mean()),
            }
        )
    return {
        "days": int(len(frame)),
        "buildable_days": int(frame["buildable"].sum()),
        "build_rate": float(frame["buildable"].mean()),
        "mean_selected": float(frame["selected"].mean()),
        "mean_candidates": float(frame["candidates"].mean()),
        "mean_has_all": float(frame["has_all"].mean()),
        "by_year": by_year,
        "by_month": by_month,
    }
