"""Train and validate the 30-minute standardized Ridge smoke-test model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ridge_pipeline import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    make_inference_samples,
    predict_matrix,
    to_canonical,
)


EXTRA_COLUMNS = ["date", "instrument_id", "adjust_factor"]


def load_months(data_dir: Path, first: str, last: str) -> pd.DataFrame:
    paths = [
        path
        for path in sorted(data_dir.glob("20????.0.feather"))
        if first <= path.name[:6] <= last
    ]
    if not paths:
        raise FileNotFoundError(f"No monthly files in {data_dir} for {first}..{last}")
    columns = list(dict.fromkeys([*EXTRA_COLUMNS, *FEATURE_COLUMNS]))
    print(f"Reading {len(paths)} monthly partitions ({first}..{last})")
    return pd.concat(
        [pd.read_feather(path, columns=columns) for path in paths],
        ignore_index=True,
    )


def feature_statistics(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[FEATURE_COLUMNS].to_numpy(np.float64)
    values[~np.isfinite(values)] = np.nan
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return mean, std


def iter_supervised_by_instrument(
    frame: pd.DataFrame,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    sequence_length: int,
):
    for key, group in frame.groupby("key", sort=False):
        group = group.sort_values("date", kind="stable").reset_index(drop=True)
        days = group["date"].dt.normalize().to_numpy()
        eod = np.flatnonzero(np.r_[days[1:] != days[:-1], True])
        if len(eod) < 2:
            continue

        # The target is next-trading-day adjusted-close return.
        close = group["close"].to_numpy(np.float64)[eod]
        adjust = group["adjust_factor"].to_numpy(np.float64)[eod]
        adjusted_close = close * adjust
        targets = np.full(len(adjusted_close) - 1, np.nan, dtype=np.float64)
        valid_denominator = (
            np.isfinite(adjusted_close[:-1]) & (adjusted_close[:-1] > 0)
        )
        np.divide(
            adjusted_close[1:],
            adjusted_close[:-1],
            out=targets,
            where=valid_denominator,
        )
        targets -= 1.0

        values = group[FEATURE_COLUMNS].to_numpy(np.float64)
        values = np.where(np.isfinite(values), values, feature_mean)
        values = ((values - feature_mean) / feature_std).astype(np.float32)

        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        meta: list[dict[str, object]] = []
        for j, position in enumerate(eod[:-1]):
            target = targets[j]
            if position + 1 < sequence_length or not np.isfinite(target):
                continue
            window = values[position - sequence_length + 1 : position + 1]
            if window.shape[0] != sequence_length:
                continue
            x_rows.append(window.reshape(-1))
            y_rows.append(float(target))
            meta.append({"date": pd.Timestamp(days[position]), "key": key})

        if x_rows:
            yield np.stack(x_rows), np.asarray(y_rows, np.float64), pd.DataFrame(meta)


def target_statistics(iterator_factory) -> tuple[float, float, int]:
    total = 0.0
    total_sq = 0.0
    count = 0
    for _, y, _ in iterator_factory():
        total += float(y.sum())
        total_sq += float(y @ y)
        count += len(y)
    if count == 0:
        raise RuntimeError("No supervised training samples")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    return mean, variance**0.5, count


def fit_ridge(
    iterator_factory,
    *,
    y_mean: float,
    y_std: float,
    alpha: float,
) -> np.ndarray:
    width = SEQUENCE_LENGTH * len(FEATURE_COLUMNS)
    xtx = np.zeros((width + 1, width + 1), np.float64)
    xty = np.zeros(width + 1, np.float64)

    for x, y, _ in iterator_factory():
        x64 = x.astype(np.float64)
        yz = (y - y_mean) / y_std
        xtx[:-1, :-1] += x64.T @ x64
        sums = x64.sum(axis=0)
        xtx[:-1, -1] += sums
        xtx[-1, :-1] += sums
        xtx[-1, -1] += len(x64)
        xty[:-1] += x64.T @ yz
        xty[-1] += yz.sum()

    penalty = np.eye(width + 1, dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0  # Do not regularize the intercept.
    return np.linalg.solve(xtx + penalty, xty)


def save_artifact(
    path: Path,
    *,
    weights: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    y_mean: float,
    y_std: float,
    alpha: float,
    train_end: str,
) -> None:
    artifact = {
        "kind": "ridge_smoke_test_noncompliant",
        "frequency": "30m",
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": y_mean,
        "target_std": y_std,
        "alpha": alpha,
        "train_start": "2019-01-01",
        "train_end": train_end,
        "weights": weights.tolist(),
    }
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def daily_ic(scores: pd.DataFrame) -> pd.Series:
    def corr(group: pd.DataFrame) -> float:
        if group["score"].nunique() < 2 or group["target_z"].nunique() < 2:
            return np.nan
        return group["score"].corr(group["target_z"])

    return scores.groupby("date", sort=True).apply(corr, include_groups=False)


def validate(
    frame: pd.DataFrame,
    artifact: dict,
    output_path: Path,
) -> None:
    x, index = make_inference_samples(
        frame,
        feature_mean=np.asarray(artifact["feature_mean"]),
        feature_std=np.asarray(artifact["feature_std"]),
        sequence_length=artifact["sequence_length"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    # Attach next-day targets using the same causal EOD convention.
    target_parts = []
    iterator = iter_supervised_by_instrument(
        frame,
        feature_mean=np.asarray(artifact["feature_mean"]),
        feature_std=np.asarray(artifact["feature_std"]),
        sequence_length=artifact["sequence_length"],
    )
    for _, y, meta in iterator:
        meta["target"] = y
        target_parts.append(meta)
    targets = pd.concat(target_parts, ignore_index=True)

    index["score"] = predict_matrix(x, artifact)
    result = index.merge(targets, on=["date", "key"], how="inner")
    result["target_z"] = (
        result["target"] - artifact["target_mean"]
    ) / artifact["target_std"]
    result = result.rename(columns={"key": "instrument_id"})
    result.to_parquet(output_path, index=False)

    ic = daily_ic(result)
    print(f"Validation rows: {len(result):,}")
    print(f"Validation days: {ic.notna().sum():,}")
    print(f"Daily Pearson IC mean: {ic.mean():.6f}")
    print(f"Daily IC IR (unannualized): {ic.mean() / ic.std():.6f}")
    print(f"Saved validation scores: {output_path}")


def train_once(
    frame: pd.DataFrame,
    *,
    alpha: float,
    artifact_path: Path,
    train_end: str,
) -> dict:
    feature_mean, feature_std = feature_statistics(frame)

    def iterator_factory():
        return iter_supervised_by_instrument(
            frame,
            feature_mean=feature_mean,
            feature_std=feature_std,
            sequence_length=SEQUENCE_LENGTH,
        )

    y_mean, y_std, count = target_statistics(iterator_factory)
    print(f"Training samples: {count:,}; target mean={y_mean:.8f}, std={y_std:.8f}")
    weights = fit_ridge(
        iterator_factory, y_mean=y_mean, y_std=y_std, alpha=alpha
    )
    save_artifact(
        artifact_path,
        weights=weights,
        feature_mean=feature_mean,
        feature_std=feature_std,
        y_mean=y_mean,
        y_std=y_std,
        alpha=alpha,
        train_end=train_end,
    )
    print(f"Saved artifact: {artifact_path}")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../e2e_data/bigalpha_2026_e2e_bar30m"),
    )
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-full-refit", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_raw = load_months(args.data_dir, "201901", "202312")
    train = to_canonical(train_raw, is_local=True)
    del train_raw
    validation_artifact_path = root / "ridge_model_validation.json"
    validation_artifact = train_once(
        train,
        alpha=args.alpha,
        artifact_path=validation_artifact_path,
        train_end="2023-12-31",
    )

    validation_raw = load_months(args.data_dir, "202312", "202412")
    validation = to_canonical(validation_raw, is_local=True)
    del validation_raw
    if not args.skip_validation:
        validate(validation, validation_artifact, root / "local_scores_2024.parquet")

    if not args.skip_full_refit:
        full = pd.concat([train, validation[validation["date"] >= "2024-01-01"]])
        train_once(
            full,
            alpha=args.alpha,
            artifact_path=root / "ridge_model.json",
            train_end="2024-12-31",
        )


if __name__ == "__main__":
    main()
