"""Platform training entrypoint for the BigAlpha 5-minute Transformer."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset


DATASOURCE_KEY = "bar5m"
TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31 23:59:59"
SEQ_LEN = 64
EPOCHS = 4
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
STAGE2_EPOCHS = 4
STAGE2_LEARNING_RATE = 1e-5
STAGE2_TEMPERATURE = 1.0
MAX_TRAIN_INSTRUMENTS = 800
SEED = 2026
TARGET_SCALE = 100.0
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_model.json")

PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOLUME_COLS = ["volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_COLS + VOLUME_COLS
OHLC_COLS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class ModelConfig:
    n_feat: int = len(FEATURE_COLS)
    seq_len: int = SEQ_LEN
    d_model: int = 128
    nhead: int = 8
    nlayers: int = 3
    dim_ff: int = 256
    dropout: float = 0.1
    pooling: str = "mean"


class StockTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.pooling != "mean":
            raise ValueError(f"expected mean pooling, got {config.pooling!r}")
        self.proj = nn.Linear(config.n_feat, config.d_model)
        self.pos = nn.Parameter(torch.zeros(1, config.seq_len, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.nlayers, norm=nn.LayerNorm(config.d_model)
        )
        self.head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.proj(x) + self.pos)
        return self.head(hidden.mean(dim=1)).squeeze(-1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DateBatchSampler(Sampler):
    """Yield one complete trading-day cross section per batch."""

    def __init__(self, dates, seed: int, min_size: int = 10) -> None:
        normalized = pd.to_datetime(pd.Series(dates)).dt.normalize()
        self.groups = [
            np.asarray(indices, dtype=np.int64).tolist()
            for indices in normalized.groupby(normalized, sort=True).groups.values()
            if len(indices) >= min_size
        ]
        if not self.groups:
            raise ValueError("no trading day has enough samples for stage-2 batching")
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for position in self.rng.permutation(len(self.groups)):
            yield self.groups[int(position)]

    def __len__(self) -> int:
        return len(self.groups)


def soft_portfolio_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = STAGE2_TEMPERATURE,
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(scores) & torch.isfinite(targets)
    if int(finite.sum()) < 2:
        zero = scores.float().sum() * 0.0
        return zero, -zero
    scores = scores[finite].float()
    targets = targets[finite].float()
    weights = torch.softmax(scores / temperature, dim=0)
    benchmark = torch.full_like(weights, 1.0 / len(weights))
    portfolio_return = ((weights - benchmark) * targets).sum()
    return -portfolio_return, portfolio_return


def freeze_except_head(model: nn.Module) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def select_instruments(start: str, end: str, maximum: int = MAX_TRAIN_INSTRUMENTS) -> list[str]:
    import dai

    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [start, end]},
        compression=True,
    ).df()
    instruments = sorted(frame["instrument"].dropna().astype(str).unique().tolist())
    if maximum <= 0 or len(instruments) <= maximum:
        return instruments
    positions = np.linspace(0, len(instruments) - 1, maximum, dtype=int)
    return [instruments[position] for position in positions]


def query_bars(table: str, start: str, end: str, instruments: list[str]) -> pd.DataFrame:
    import dai

    buffer_start = (pd.Timestamp(start) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    sql = (
        f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    frame = dai.query(
        sql,
        filters={"date": [buffer_start, end], "instrument": instruments},
        compression=True,
    ).df()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype(str)
    for column in OHLC_COLS:
        frame.loc[frame[column] == -1, column] = np.nan
    for column in VOLUME_COLS:
        frame[column] = np.log1p(frame[column].clip(lower=0))
    return frame.sort_values(["instrument", "date"], kind="mergesort")


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    flat[~np.isfinite(flat)] = np.nan
    mean = np.nanmean(flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    scale = np.nanstd(flat, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    return mean, scale


def apply_standardizer(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, mean.reshape(1, 1, -1))
    return ((x - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)).astype(np.float32)


def build_training_dataset(
    table: str,
    start: str,
    end: str,
    instruments: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    started = time.time()
    frame = query_bars(table, start, end, instruments)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    windows: list[np.ndarray] = []
    targets: list[np.float32] = []
    dates: list[pd.Timestamp] = []

    for _, sub in frame.groupby("instrument", sort=False, observed=True):
        sub = sub.reset_index(drop=True)
        values = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
        days = sub["date"].dt.normalize().to_numpy()
        endpoints = np.flatnonzero(np.r_[days[1:] != days[:-1], True])
        close = sub["close"].to_numpy(dtype=np.float64)
        for position, endpoint in enumerate(endpoints):
            current_day = pd.Timestamp(days[endpoint])
            window_start = endpoint - seq_len + 1
            if (
                current_day < start_ts
                or current_day > end_ts
                or window_start < 0
                or position + 1 >= len(endpoints)
            ):
                continue
            next_day = pd.Timestamp(days[endpoints[position + 1]]).normalize()
            if next_day > end_ts:
                continue
            current_close = close[endpoint]
            next_close = close[endpoints[position + 1]]
            if (
                not np.isfinite(current_close)
                or current_close <= 0
                or not np.isfinite(next_close)
                or next_close <= 0
            ):
                continue
            target = next_close / current_close - 1.0
            if not np.isfinite(target):
                continue
            windows.append(values[window_start : endpoint + 1])
            targets.append(np.float32(target))
            dates.append(current_day)

    if not windows:
        raise RuntimeError(f"no training samples built for {start}~{end}")
    x = np.stack(windows).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    print(f"built {len(y):,} training samples in {time.time() - started:.1f}s")
    return x, y, np.asarray(dates, dtype="datetime64[ns]")


def tensor_to_json(tensor: torch.Tensor) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(cpu.dtype).replace("torch.", ""),
        "shape": list(cpu.shape),
        "data": cpu.reshape(-1).tolist(),
    }


def save_model(payload: dict[str, object], model_path: str = MODEL_PATH) -> str:
    serializable = dict(payload)
    serializable["state_dict"] = {
        name: tensor_to_json(tensor) for name, tensor in payload["state_dict"].items()
    }
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False)
    return model_path


def train_and_save(datasources, model_path: str = MODEL_PATH) -> str:
    """Train from scratch on the fixed public-training interval and save JSON weights."""
    if DATASOURCE_KEY not in datasources:
        raise KeyError(
            f"missing datasource {DATASOURCE_KEY!r}; available={list(datasources)}"
        )
    set_seed()
    table = datasources[DATASOURCE_KEY]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instruments = select_instruments(TRAIN_START, TRAIN_END)
    print(f"device={device}, table={table}, instruments={len(instruments)}")

    x_train, y_train, train_dates = build_training_dataset(
        table, TRAIN_START, TRAIN_END, instruments, SEQ_LEN
    )
    low, high = np.percentile(y_train, [1, 99])
    y_train = np.clip(y_train, low, high).astype(np.float32)
    mean, scale = fit_standardizer(x_train)
    x_train = apply_standardizer(x_train, mean, scale)

    config = ModelConfig()
    model = StockTransformer(config).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if not 100_000 <= parameter_count <= 100_000_000:
        raise RuntimeError(f"parameter count outside competition limits: {parameter_count}")
    print(f"trainable_parameters={parameter_count:,}")

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train * TARGET_SCALE),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    loss_fn = nn.HuberLoss(delta=1.0)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(EPOCHS):
        model.train()
        started = time.time()
        total = 0.0
        batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = loss_fn(model(batch_x), batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            total += float(loss.item())
            batches += 1
        print(
            f"epoch={epoch + 1}/{EPOCHS}, loss={total / max(batches, 1):.6f}, "
            f"final_fit=true, elapsed={time.time() - started:.1f}s"
        )

    fine_tune_history = []
    if STAGE2_EPOCHS > 0:
        trainable_head_parameters = freeze_except_head(model)
        print(f"stage2_trainable_parameters={trainable_head_parameters:,}")
        stage2_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train),
                torch.from_numpy(y_train * TARGET_SCALE),
            ),
            batch_sampler=DateBatchSampler(train_dates, seed=SEED + 10_000),
            pin_memory=device.type == "cuda",
            num_workers=0,
        )
        stage2_optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=STAGE2_LEARNING_RATE,
            weight_decay=0.0,
        )
        stage2_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for epoch in range(STAGE2_EPOCHS):
            model.eval()
            model.head.train()
            started = time.time()
            total, total_return, batches = 0.0, 0.0, 0
            for batch_x, batch_y in stage2_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                stage2_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    scores = model(batch_x)
                loss, portfolio_return = soft_portfolio_loss(scores, batch_y)
                stage2_scaler.scale(loss).backward()
                stage2_scaler.unscale_(stage2_optimizer)
                nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                stage2_scaler.step(stage2_optimizer)
                stage2_scaler.update()
                total += float(loss.item())
                total_return += float(portfolio_return.item())
                batches += 1
            record = {
                "epoch": epoch + 1,
                "loss": total / max(batches, 1),
                "soft_portfolio_return": total_return / max(batches, 1),
            }
            fine_tune_history.append(record)
            print(
                f"stage2_epoch={epoch + 1}/{STAGE2_EPOCHS}, "
                f"loss={record['loss']:.6f}, "
                f"soft_return={record['soft_portfolio_return']:.6f}, "
                f"final_fit=true, elapsed={time.time() - started:.1f}s"
            )

    payload = {
        "format_version": 1,
        "datasource_key": DATASOURCE_KEY,
        "final_fit": True,
        "model_cfg": asdict(config),
        "feature_cols": FEATURE_COLS,
        "volume_cols": VOLUME_COLS,
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "target_scale": TARGET_SCALE,
        "parameter_count": parameter_count,
        "train_config": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "seq_len": SEQ_LEN,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "stage2_epochs": STAGE2_EPOCHS,
            "stage2_learning_rate": STAGE2_LEARNING_RATE,
            "stage2_temperature": STAGE2_TEMPERATURE,
            "max_instruments": MAX_TRAIN_INSTRUMENTS,
            "seed": SEED,
            "datasource_key": DATASOURCE_KEY,
            "final_fit": True,
        },
        "validation": {
            "mode": "final_fit",
            "train_end": TRAIN_END,
            "stage2_epochs": STAGE2_EPOCHS,
        },
        "fine_tune": {
            "enabled": STAGE2_EPOCHS > 0,
            "method": "softmax_market_neutral_return",
            "head_only": True,
            "history": fine_tune_history,
            "selected_stage2_epoch": STAGE2_EPOCHS,
        },
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    }
    destination = save_model(payload, model_path)
    print(f"saved checkpoint: {destination}")
    return destination


if __name__ == "__main__":
    train_and_save({DATASOURCE_KEY: "bigalpha_2026_stock_bar5m"})
