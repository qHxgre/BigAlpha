"""Training code for the final 61-day Fama-structured E2E submission.

The epoch count is selected on a held-out validation period, then the selected
schedule is refit on the full 2019--2024 local E2E sample.  The model consumes
raw 30-minute source fields only; preprocessing is isolated in
``e2e_preprocess.py`` so training and cloud inference share one definition.

中文说明：先在独立验证期选择训练轮次，再按选定计划在 2019--2024 年全部本地
E2E 样本上重新训练。模型只使用 30 分钟原始字段；预处理集中在
``e2e_preprocess.py``，训练与云端推理共用同一套定义。
"""

from __future__ import annotations

import json
import os
import random
import time
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from e2e_structured_fama_regime_localattn61 import FamaE2EConfig, FamaStructuredRegimeE2E as FamaStructuredE2E, parameter_count

MODEL_CONFIG = FamaE2EConfig(use_local_attention=False, use_daily_context_attention=False)
# 按比赛规定，预处理仅进行数据格式对齐、缺失值填充和逐字段固定变换/标准化，
# 不引入滚动、跨字段或复合的人工特征。
from e2e_preprocess import (
    RAW_INPUT_COLS,
    PreprocessStats,
    fit_preprocessor,
    required_source_columns,
    to_canonical,
    transform_inputs,
)
from evaluation import evaluate_factor


warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path("data/e2e/bigalpha_2026_e2e_bar30m")
FULL_REFIT = os.environ.get("E2E_FULL_REFIT", "0") == "1"
DATA_START = pd.Timestamp("2019-01-01")
if FULL_REFIT:
    # Select the epoch count on 2019--2023 / 2024, then refit on all six years.
    OUTPUT_DIR = Path("models/e2e/regime_transformer17_61_full_halflr")
    TRAIN_START, TRAIN_END = DATA_START, pd.Timestamp("2024-12-31")
    SELECTION_TRAIN_END = pd.Timestamp("2023-12-31")
    VALIDATION_START, VALIDATION_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
    MODEL_VARIANT = "regime_transformer17_61_full_halflr"
else:
    OUTPUT_DIR = Path("models/e2e/regime_transformer17_61_halflr_fama_2020_2023_rank")
    TRAIN_START, TRAIN_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")
    SELECTION_TRAIN_END = pd.Timestamp("2022-12-31")
    VALIDATION_START, VALIDATION_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    MODEL_VARIANT = "regime_transformer17_61_halflr"
WINDOW_BARS = MODEL_CONFIG.history_bars
MAX_CROSS_SECTION_ROWS = 512
MAX_EPOCHS, PATIENCE, MIN_DELTA = 80, 10, 2e-5
MIN_SELECTION_EPOCHS = 6
SCHEDULER_PATIENCE, SCHEDULER_COOLDOWN = 5, 2
SCHEDULER_FACTOR = 0.3
BATCH_SIZE = 512
RANK_LOSS_WEIGHT = 0.15
SEED = 20260731
FINAL_REFIT_SEED = SEED + 1


class EndOfDayDataset(Dataset):
    """Lazy 84-bar views over one key-sorted feature matrix."""

    def __init__(self, features: np.ndarray, records: list[tuple[int, float]]):
        self.features, self.records = features, records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end_idx, target = self.records[index]
        window = self.features[end_idx - WINDOW_BARS + 1:end_idx + 1]
        return torch.from_numpy(window), torch.tensor(target, dtype=torch.float32)


class DailyCrossSectionDataset(Dataset):
    """One fixed-size, deterministic cross-section per trading day."""

    def __init__(self, features: np.ndarray, daily_records: list[list[tuple[int, float]]], seed: int = SEED):
        self.features, self.daily_records, self.seed = features, daily_records, seed

    def __len__(self) -> int:
        return len(self.daily_records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        records = self.daily_records[index]
        if len(records) > MAX_CROSS_SECTION_ROWS:
            rng = np.random.default_rng(self.seed + index)
            records = [records[i] for i in rng.choice(len(records), MAX_CROSS_SECTION_ROWS, replace=False)]
        end_indices = np.fromiter((row[0] for row in records), dtype=np.int64)
        targets = np.fromiter((row[1] for row in records), dtype=np.float32)
        windows = np.stack([self.features[end - WINDOW_BARS + 1:end + 1] for end in end_indices])
        return torch.from_numpy(windows), torch.from_numpy(targets)


def month_paths(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    paths = [DATA_ROOT / f"{month.strftime('%Y%m')}.0.feather" for month in months]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing local E2E partitions: {missing[:3]}")
    return paths


def load_canonical(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cols = required_source_columns(is_local=True)
    parts = [pd.read_feather(path, columns=cols) for path in month_paths(start, end)]
    raw = pd.concat(parts, ignore_index=True)
    return to_canonical(raw, is_local=True, adjust_prices=True).sort_values(["key", "date"]).reset_index(drop=True)


def build_records(
    canonical: pd.DataFrame,
    stats: PreprocessStats,
    sample_start: pd.Timestamp,
    sample_end: pd.Timestamp,
) -> tuple[np.ndarray, pd.DataFrame, list[tuple[int, float]]]:
    """Build EOD windows and raw n1->n2 return labels from adjusted prices."""

    x = transform_inputs(canonical, stats)
    metadata: list[dict[str, object]] = []

    for key, positions in canonical.groupby("key", sort=False).indices.items():
        pos = np.asarray(positions)
        dates = canonical.iloc[pos]["date"].to_numpy(dtype="datetime64[ns]")
        opens = canonical.iloc[pos]["open"].to_numpy(dtype=np.float64)
        days = dates.astype("datetime64[D]")
        eod = np.flatnonzero(np.r_[days[1:] != days[:-1], True])
        first = np.r_[0, eod[:-1] + 1]
        if len(eod) < 3:
            continue

        for day_idx in range(len(eod) - 2):
            end_idx = int(eod[day_idx])
            day = pd.Timestamp(days[eod[day_idx]])
            buy_open, sell_open = opens[first[day_idx + 1]], opens[first[day_idx + 2]]
            if not (sample_start <= day <= sample_end and end_idx >= WINDOW_BARS - 1):
                continue
            if not (np.isfinite(buy_open) and np.isfinite(sell_open) and buy_open > 0):
                continue
            raw_return = float(sell_open / buy_open - 1.0)
            metadata.append({"trading_day": day, "instrument": int(key), "raw_return": raw_return,
                             "end_idx": int(pos[end_idx])})

    meta = pd.DataFrame(metadata)
    if meta.empty:
        raise ValueError("No valid end-of-day samples were created.")
    daily_mean = meta.groupby("trading_day")["raw_return"].transform("mean")
    daily_std = meta.groupby("trading_day")["raw_return"].transform("std", ddof=0)
    meta["target_zscore"] = (meta["raw_return"] - daily_mean) / daily_std.where(daily_std > 0)
    meta = meta.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_zscore"]).reset_index(drop=True)
    raw_records = [(int(row.end_idx), float(row.target_zscore)) for row in meta.itertuples()]
    return x, meta, raw_records


def group_records_by_day(meta: pd.DataFrame) -> list[list[tuple[int, float]]]:
    """Keep each day's ranking problem intact for the pairwise objective."""

    return [
        [(int(row.end_idx), float(row.target_zscore)) for row in day.itertuples()]
        for _, day in meta.groupby("trading_day", sort=True)
    ]


def loss_components(scores: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Huber fits standardized returns; pairwise logistic loss directly rewards rank order."""

    huber = loss_fn(scores, targets)
    row_i, row_j = torch.triu_indices(len(targets), len(targets), offset=1, device=targets.device)
    target_direction = torch.sign(targets[row_i] - targets[row_j])
    valid = target_direction != 0
    if not torch.any(valid):
        rank = torch.zeros((), device=targets.device)
    else:
        score_difference = scores[row_i[valid]] - scores[row_j[valid]]
        rank = torch.nn.functional.softplus(-target_direction[valid] * score_difference).mean()
    return huber + RANK_LOSS_WEIGHT * rank, huber, rank


def mean_loss(model: FamaStructuredE2E, features: np.ndarray, daily_records: list[list[tuple[int, float]]],
              loss_fn: nn.Module, device: torch.device) -> tuple[float, float, float]:
    loader = DataLoader(DailyCrossSectionDataset(features, daily_records), batch_size=None, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    model.eval()
    total_loss, total_huber, total_rank, total_days = 0.0, 0.0, 0.0, 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss, huber, rank = loss_components(model(xb), yb, loss_fn)
            total_loss += float(loss)
            total_huber += float(huber)
            total_rank += float(rank)
            total_days += 1
    divisor = max(total_days, 1)
    return total_loss / divisor, total_huber / divisor, total_rank / divisor


def train_model(
    features: np.ndarray,
    daily_records: list[list[tuple[int, float]]],
    *,
    validation_daily_records: list[list[tuple[int, float]]] | None = None,
    learning_rate_schedule: list[float] | None = None,
    max_epochs: int,
    phase: str,
) -> tuple[FamaStructuredE2E, int, float | None, list[float]]:
    """Train until validation Huber loss stops improving, or for a fixed epoch count."""

    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FamaStructuredE2E(MODEL_CONFIG).to(device)
    loader = DataLoader(DailyCrossSectionDataset(features, daily_records), batch_size=None,
                        shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, threshold=MIN_DELTA,
        threshold_mode="abs", cooldown=SCHEDULER_COOLDOWN, min_lr=2e-6,
    )
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print(f"{phase}: device={device}; parameters={parameter_count(model):,}; "
          f"training days={len(loader.dataset):,}; max cross-section rows={MAX_CROSS_SECTION_ROWS}")

    best_state, best_epoch, best_validation_loss, stale_epochs = None, 0, float("inf"), 0
    learning_rates: list[float] = []
    for epoch in range(1, max_epochs + 1):
        if learning_rate_schedule is not None:
            if epoch > len(learning_rate_schedule):
                raise ValueError(f"{phase}: learning-rate schedule is shorter than max_epochs")
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_schedule[epoch - 1]
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        model.train()
        total_loss, total_huber, total_rank, batches = 0.0, 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss, huber, rank = loss_components(model(xb), yb, loss_fn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            total_huber += float(huber.detach())
            total_rank += float(rank.detach())
            batches += 1
        train_loss = total_loss / max(batches, 1)
        train_huber, train_rank = total_huber / max(batches, 1), total_rank / max(batches, 1)
        if validation_daily_records is None:
            print(f"{phase}: epoch={epoch}/{max_epochs}; total_loss={train_loss:.6f}; "
                  f"huber={train_huber:.6f}; rank={train_rank:.6f}; "
                  f"lr={learning_rates[-1]:.1e}; elapsed={time.monotonic() - started:.1f}s")
            continue

        validation_loss, validation_huber, validation_rank = mean_loss(
            model, features, validation_daily_records, loss_fn, device,
        )
        print(f"{phase}: epoch={epoch}/{max_epochs}; total_loss={train_loss:.6f}; huber={train_huber:.6f}; "
              f"rank={train_rank:.6f}; validation_loss={validation_loss:.6f}; "
              f"validation_huber={validation_huber:.6f}; validation_rank={validation_rank:.6f}; "
              f"lr={optimizer.param_groups[0]['lr']:.1e}; elapsed={time.monotonic() - started:.1f}s")
        # Six unconditional warm-up epochs prevent a noisy early minimum from
        # controlling either the scheduler or final refit length.
        if epoch <= MIN_SELECTION_EPOCHS:
            continue
        scheduler.step(validation_loss)
        if validation_loss < best_validation_loss - MIN_DELTA:
            best_state, best_epoch, best_validation_loss, stale_epochs = deepcopy(model.state_dict()), epoch, validation_loss, 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"{phase}: early stop at epoch {epoch}; best epoch={best_epoch}")
                break


    if best_state is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_validation_loss, learning_rates
    return model, max_epochs, None, learning_rates


@torch.inference_mode()
def predict(model: FamaStructuredE2E, features: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    device = next(model.parameters()).device
    model.eval()
    dataset = EndOfDayDataset(features, [(int(row.end_idx), 0.0) for row in meta.itertuples()])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    scores = []
    for xb, _ in loader:
        scores.append(model(xb.to(device, non_blocking=True)).cpu().numpy())
    result = meta[["trading_day", "instrument", "raw_return"]].copy()
    result["factor"] = np.concatenate(scores).astype(np.float32)
    return result


def run_period(model: FamaStructuredE2E, stats: PreprocessStats, name: str, load_start: str, load_end: str,
               sample_start: str, sample_end: str) -> None:
    print(f"loading {name}")
    canonical = load_canonical(pd.Timestamp(load_start), pd.Timestamp(load_end))
    features, meta, _ = build_records(canonical, stats, pd.Timestamp(sample_start), pd.Timestamp(sample_end))
    predictions = predict(model, features, meta)
    output = OUTPUT_DIR / name
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics = evaluate_factor(predictions, "factor", "raw_return", output, output_prefix="e2e")
    print(f"{name}: samples={len(predictions):,}; metrics:\n{metrics.to_string(index=False)}")


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    # Full refit has no 2018 source data. Its earliest 2019 samples naturally
    # wait until each stock has accumulated a complete 61-day window.
    load_start = max(TRAIN_START - pd.Timedelta(days=150), DATA_START)
    load_end = TRAIN_END if FULL_REFIT else pd.Timestamp("2024-01-31")
    train_all = load_canonical(load_start, load_end)

    # Phase 1: choose the epoch count without using either out-of-sample period.
    selection_rows = train_all.loc[train_all["date"].dt.normalize().between(TRAIN_START, SELECTION_TRAIN_END)]
    selection_stats = fit_preprocessor(selection_rows, adjust_prices=True)
    selection_features, selection_meta, selection_records = build_records(
        train_all, selection_stats, TRAIN_START, SELECTION_TRAIN_END,
    )
    _, validation_meta, validation_records = build_records(
        train_all, selection_stats, VALIDATION_START, VALIDATION_END,
    )
    selection_daily_records = group_records_by_day(selection_meta)
    validation_daily_records = group_records_by_day(validation_meta)
    print(f"selection train samples={len(selection_records):,}; validation samples={len(validation_records):,}; "
          f"elapsed={time.monotonic() - started:.1f}s")
    _, best_epoch, best_validation_loss, selection_learning_rates = train_model(
        selection_features, selection_daily_records, validation_daily_records=validation_daily_records,
        max_epochs=MAX_EPOCHS, phase="epoch_selection",
    )
    del selection_rows, selection_features, selection_meta, selection_records, validation_meta, validation_records
    del selection_daily_records, validation_daily_records

    # Phase 2: refit on the requested training range for the selected epoch count.
    train_rows = train_all.loc[train_all["date"].dt.normalize().between(TRAIN_START, TRAIN_END)]
    stats = fit_preprocessor(train_rows, adjust_prices=True)
    (OUTPUT_DIR / "preprocess_stats.json").write_text(json.dumps(stats.to_dict()), encoding="utf-8")
    features, train_meta, train_records = build_records(train_all, stats, TRAIN_START, TRAIN_END)
    train_daily_records = group_records_by_day(train_meta)
    print(f"final train EOD samples={len(train_records):,}; selected_epochs={best_epoch}; elapsed={time.monotonic() - started:.1f}s")
    del selection_stats, train_rows, train_all

    # Epoch-selection length must not change the random initialisation of the
    # final model; otherwise patience becomes an accidental model parameter.
    random.seed(FINAL_REFIT_SEED); np.random.seed(FINAL_REFIT_SEED); torch.manual_seed(FINAL_REFIT_SEED)
    refit_learning_rates = selection_learning_rates[:best_epoch]
    model, _, _, _ = train_model(
        features,
        train_daily_records,
        learning_rate_schedule=refit_learning_rates,
        max_epochs=best_epoch,
        phase="final_refit",
    )
    checkpoint = {"model_config": asdict(model.cfg), "state_dict": model.state_dict(), "preprocess": stats.to_dict()}
    torch.save(checkpoint, OUTPUT_DIR / "model.pt")
    del features, train_meta, train_records, train_daily_records

    if not FULL_REFIT:
        # 2019 has no earlier local history; its first lookback window is naturally omitted.
        run_period(model, stats, "test_2019", "2019-01-01", "2020-01-31", "2019-01-01", "2019-12-31")
        run_period(model, stats, "test_2024", "2023-08-01", "2024-12-31", "2024-01-01", "2024-12-31")
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps({
        "model_variant": MODEL_VARIANT, "full_refit": FULL_REFIT,
        "train_period": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "window_bars": WINDOW_BARS,
        "max_cross_section_rows": MAX_CROSS_SECTION_ROWS, "rank_loss_weight": RANK_LOSS_WEIGHT,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "min_selection_epochs": MIN_SELECTION_EPOCHS,
        "min_delta": MIN_DELTA, "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_cooldown": SCHEDULER_COOLDOWN, "scheduler_factor": SCHEDULER_FACTOR,
        "final_refit_seed": FINAL_REFIT_SEED,
        "selected_epochs": best_epoch, "validation_loss": best_validation_loss,
        "refit_learning_rates": refit_learning_rates,
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
