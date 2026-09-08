"""Memory-bounded full-universe training and frozen 2024 validation.

This candidate keeps the selected model, target, optimizer, and loss frozen.
The only research change is training-universe coverage.  Raw bars are read in
instrument partitions so the full historical pool fits safely in 64 GB RAM.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import dai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from bigmodule import M
from torch.utils.data import DataLoader, TensorDataset

import e2e_microstructure_transformer as model


HERE = Path(__file__).resolve().parent
MODEL_PATH = str(HERE / "e2e_full_universe_streaming.json")
SCORES_PATH = HERE / "e2e_full_universe_streaming_scores_2024.parquet"
TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31 23:59:59"
PARTITION_SIZE = 200


def _partitions(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _raw_partition(
    table: str,
    instruments: list[str],
    labels: dict[tuple[pd.Timestamp, str], np.float32],
) -> tuple[np.ndarray, np.ndarray]:
    """Build unstandardized sequences for one bounded instrument partition."""
    buffer_start = (pd.Timestamp(TRAIN_START) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    sql = (
        f"SELECT date, instrument, {', '.join(model.FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    raw = dai.query(
        sql,
        filters={"date": [buffer_start, TRAIN_END], "instrument": instruments},
        compression=True,
    ).df()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["instrument"] = raw["instrument"].astype(str)
    start_ts = pd.Timestamp(TRAIN_START)
    end_ts = pd.Timestamp(TRAIN_END)

    windows: list[np.ndarray] = []
    targets: list[np.float32] = []
    for instrument, stock in raw.groupby("instrument", sort=False, observed=True):
        values = model._transform_raw(stock[model.FEATURE_COLS].to_numpy(copy=False))
        dates = stock["date"].dt.normalize().to_numpy()
        close_positions = np.flatnonzero(np.r_[dates[1:] != dates[:-1], True])
        for position in close_positions:
            date = pd.Timestamp(dates[position])
            if position + 1 < model.SEQ_LEN or date < start_ts or date > end_ts:
                continue
            target = labels.get((date, str(instrument)))
            if target is None or not np.isfinite(target):
                continue
            windows.append(values[position - model.SEQ_LEN + 1 : position + 1])
            targets.append(np.float32(target))
    del raw
    if not windows:
        raise RuntimeError(f"No samples for partition starting {instruments[:1]}")
    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(targets, dtype=np.float32),
    )


def _streaming_stats(
    table: str,
    chunks: list[list[str]],
    labels: dict[tuple[pd.Timestamp, str], np.float32],
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(len(model.FEATURE_COLS), dtype=np.float64)
    sum_squares = np.zeros(len(model.FEATURE_COLS), dtype=np.float64)
    counts = np.zeros(len(model.FEATURE_COLS), dtype=np.int64)
    for number, instruments in enumerate(chunks, start=1):
        started = time.time()
        x, _ = _raw_partition(table, instruments, labels)
        finite = np.isfinite(x)
        safe = np.where(finite, x, 0.0).astype(np.float64, copy=False)
        sums += safe.sum(axis=(0, 1))
        sum_squares += np.square(safe).sum(axis=(0, 1))
        counts += finite.sum(axis=(0, 1))
        print(
            "stats_partition",
            number,
            len(chunks),
            "samples",
            len(x),
            "seconds",
            round(time.time() - started, 2),
            flush=True,
        )
        del x, finite, safe
        gc.collect()
    mean = sums / np.maximum(counts, 1)
    variance = sum_squares / np.maximum(counts, 1) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(
    x: np.ndarray,
    stats: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, std = stats
    x -= mean.reshape(1, 1, -1)
    x /= std.reshape(1, 1, -1)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
    np.clip(x, -8.0, 8.0, out=x)
    return x


def train_streaming(table: str) -> str:
    model.set_deterministic(model.SEED)
    instruments = model._pool(model.INSTRUMENT_TABLE, TRAIN_START, TRAIN_END)
    chunks = _partitions(instruments, PARTITION_SIZE)
    print(
        "full_universe",
        "instruments",
        len(instruments),
        "partitions",
        len(chunks),
        "partition_size",
        PARTITION_SIZE,
        flush=True,
    )
    labels = model._pure_forward_labels(TRAIN_START, TRAIN_END, set(instruments))
    print("labels", len(labels), flush=True)
    stats = _streaming_stats(table, chunks, labels)
    print("streaming_stats_complete", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model.RawMicrostructureTransformer(model.MODEL_CONFIG).to(device)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    if not 100_000 <= parameter_count <= 100_000_000:
        raise RuntimeError(f"Non-compliant trainable parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=model.LEARNING_RATE,
        weight_decay=model.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=model.EPOCHS,
        eta_min=model.LEARNING_RATE * 0.1,
    )
    loss_function = nn.SmoothL1Loss(beta=0.5)

    for epoch in range(model.EPOCHS):
        network.train()
        epoch_started = time.time()
        total_loss = 0.0
        batches = 0
        order = np.random.default_rng(model.SEED + epoch).permutation(len(chunks))
        for sequence, chunk_index in enumerate(order, start=1):
            x, y = _raw_partition(table, chunks[int(chunk_index)], labels)
            _standardize(x, stats)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                batch_size=model.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                generator=torch.Generator().manual_seed(
                    model.SEED + epoch * len(chunks) + int(chunk_index)
                ),
            )
            partition_loss = 0.0
            partition_batches = 0
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(network(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                value = float(loss.detach().cpu())
                total_loss += value
                partition_loss += value
                batches += 1
                partition_batches += 1
            print(
                "train_partition",
                "epoch",
                epoch + 1,
                "sequence",
                sequence,
                len(chunks),
                "samples",
                len(y),
                "loss",
                round(partition_loss / max(partition_batches, 1), 8),
                flush=True,
            )
            del loader, x, y
            gc.collect()
        scheduler.step()
        print(
            "epoch_finished",
            epoch + 1,
            "loss",
            round(total_loss / max(batches, 1), 8),
            "lr",
            optimizer.param_groups[0]["lr"],
            "seconds",
            round(time.time() - epoch_started, 2),
            flush=True,
        )

    original_train_end = model.TRAIN_END
    model.TRAIN_END = TRAIN_END
    try:
        model._checkpoint_to_json(network, stats, MODEL_PATH)
    finally:
        model.TRAIN_END = original_train_end
    print("model_saved", MODEL_PATH, flush=True)
    return MODEL_PATH


if __name__ == "__main__":
    print("trainable_parameters", model.model_parameter_count(), flush=True)
    sources = {"bar5m": model.TRAIN_TABLE}
    train_streaming(model.TRAIN_TABLE)
    scores = model.predict(
        sources,
        "2024-01-01 00:00:00",
        "2024-12-31 23:59:59",
        MODEL_PATH,
    )
    scores.to_parquet(SCORES_PATH, index=False)
    coverage = scores.groupby("date", observed=True)["instrument"].size()
    print(
        "score_contract",
        len(scores),
        scores["date"].nunique(),
        int(coverage.min()),
        int(coverage.max()),
        flush=True,
    )
    M.bigalpha_eval._latest(factor_data=scores, show=False)
