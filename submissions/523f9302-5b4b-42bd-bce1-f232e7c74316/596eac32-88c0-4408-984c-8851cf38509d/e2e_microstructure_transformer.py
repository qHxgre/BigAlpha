# -*- coding: utf-8 -*-
"""BigAlpha 2026 end-to-end raw microstructure Transformer.

Only organizer-provided raw 5-minute fields enter the network.  The only
preprocessing is a per-field log1p transform, missing-value filling, and a
per-field StandardScaler fitted on the training sample.  The target is the
next-trading-day return residual after removing the current cross-section's
official style and industry exposures.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import dai
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


logger = structlog.get_logger()

TRAIN_TABLE = "bigalpha_2026_stock_bar5m"
EXPOSURE_TABLE = "bigalpha_2026_exposure"
INSTRUMENT_TABLE = "bigalpha_2026_instruments"
TRAIN_START = "2019-01-01"
# Final public build uses the full organizer-authorized training interval.
TRAIN_END = "2024-12-31 23:59:59"

SEQ_LEN = 48
EPOCHS = 4
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 20260805
MAX_TRAIN_INSTRUMENTS = 800

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "e2e_microstructure_transformer.json",
)

PRICE_COLS = [
    "pre_close", "open", "high", "low", "close",
    *[f"ask_price{i}" for i in range(1, 6)],
    *[f"bid_price{i}" for i in range(1, 6)],
]
ACTIVITY_COLS = ["deal_number", "volume", "amount"]
BOOK_SIZE_COLS = [
    *[f"ask_volume{i}" for i in range(1, 6)],
    *[f"bid_volume{i}" for i in range(1, 6)],
    *[f"ask_num_orders{i}" for i in range(1, 6)],
    *[f"bid_num_orders{i}" for i in range(1, 6)],
]
FEATURE_COLS = PRICE_COLS + ACTIVITY_COLS + BOOK_SIZE_COLS

STYLE_COLS = [
    "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL",
    "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE",
]
INDUSTRY_COLS = [
    "AGRIFOREST", "MINING", "CHEM", "IRONSTEEL", "NONFERMETAL",
    "ELECTRONICS", "AUTO", "HOUSEAPP", "FOODBEVER", "TEXTILE",
    "LIGHTINDUS", "HEALTH", "UTILITIES", "TRANSPORTATION",
    "REALESTATE", "COMMETRADE", "LEISERVICE", "BANK",
    "NONBANKFINAN", "CONGLOMERATES", "CONMAT", "BUILDDECO",
    "ELECEQP", "MACHIEQUIP", "AERODEF", "COMPUTER", "MEDIA",
    "TELECOM", "COAL", "PETRO", "ENVP", "BEAUTY",
]


@dataclass(frozen=True)
class ModelConfig:
    n_feat: int = len(FEATURE_COLS)
    seq_len: int = SEQ_LEN
    d_model: int = 128
    nhead: int = 8
    nlayers: int = 3
    dim_ff: int = 512
    dropout: float = 0.10


MODEL_CONFIG = ModelConfig()


def set_deterministic(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RawMicrostructureTransformer(nn.Module):
    """Learned field mixing followed by temporal self-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.n_feat, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.position = nn.Parameter(
            torch.zeros(1, cfg.seq_len + 1, cfg.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.nlayers)
        self.output = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        cls = self.cls_token.expand(len(x), -1, -1)
        h = torch.cat([cls, h], dim=1) + self.position
        h = self.encoder(h)
        return self.output(h[:, 0]).squeeze(-1)


def model_parameter_count(cfg: ModelConfig = MODEL_CONFIG) -> int:
    return sum(p.numel() for p in RawMicrostructureTransformer(cfg).parameters())


def _pool(table: str, start_date: str, end_date: str) -> list[str]:
    data = dai.query(
        f"SELECT DISTINCT instrument FROM {table}",
        filters={"date": [start_date, end_date]},
    ).df()
    return sorted(data["instrument"].astype(str).tolist())


def _training_instruments() -> list[str]:
    universe = _pool(INSTRUMENT_TABLE, TRAIN_START, TRAIN_END)
    if len(universe) <= MAX_TRAIN_INSTRUMENTS:
        return universe
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(
        np.asarray(universe, dtype=object),
        size=MAX_TRAIN_INSTRUMENTS,
        replace=False,
    )
    return sorted(chosen.tolist())


def _pure_forward_labels(
    start_date: str,
    end_date: str,
    selected_instruments: set[str],
) -> dict[tuple[pd.Timestamp, str], np.float32]:
    """Build T -> T+1 residual-return labels on the full daily cross-section."""
    cols = ["date", "instrument", "ret", *STYLE_COLS, *INDUSTRY_COLS]
    raw = dai.query(
        f"SELECT {', '.join(cols)} FROM {EXPOSURE_TABLE}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw["instrument"] = raw["instrument"].astype(str)
    raw = raw.sort_values(["instrument", "date"]).reset_index(drop=True)
    raw["future_ret"] = raw.groupby("instrument", observed=True)["ret"].shift(-1)

    residual_parts: list[pd.DataFrame] = []
    x_cols = STYLE_COLS + INDUSTRY_COLS[1:]
    for date, day in raw.groupby("date", sort=True, observed=True):
        y = pd.to_numeric(day["future_ret"], errors="coerce").to_numpy(float)
        x = day[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(y)
        if valid.sum() < 100:
            continue
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        design = np.column_stack([np.ones(len(day), dtype=float), x])
        xv = design[valid]
        yv = y[valid]
        gram = xv.T @ xv
        ridge = np.eye(gram.shape[0], dtype=float) * 1e-5
        ridge[0, 0] = 0.0
        try:
            beta = np.linalg.solve(gram + ridge, xv.T @ yv)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(xv, yv, rcond=None)[0]
        resid = yv - xv @ beta
        scale = resid.std(ddof=0)
        if not np.isfinite(scale) or scale < 1e-8:
            continue
        resid = np.clip(resid / scale, -5.0, 5.0).astype(np.float32)
        keep = day.loc[valid, ["instrument"]].copy()
        keep["date"] = date
        keep["label"] = resid
        residual_parts.append(keep)

    if not residual_parts:
        raise RuntimeError("No residual-return labels were built")
    labels = pd.concat(residual_parts, ignore_index=True)
    labels = labels[labels["instrument"].isin(selected_instruments)]
    return {
        (pd.Timestamp(row.date), str(row.instrument)): np.float32(row.label)
        for row in labels.itertuples(index=False)
    }


def _transform_raw(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, np.nan)
    values = np.where(values >= 0.0, np.log1p(values), np.nan)
    return values.astype(np.float32, copy=False)


def build_dataset(
    table: str,
    start_date: str,
    end_date: str,
    mode: str,
    instruments: list[str],
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame, tuple[np.ndarray, np.ndarray]]:
    """Convert raw 5-minute rows to one fixed-length end-of-day sequence."""
    started = time.time()
    buffer_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%d")
    sql = (
        f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    raw = dai.query(
        sql,
        filters={"date": [buffer_start, end_date], "instrument": instruments},
        compression=True,
    ).df()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["instrument"] = raw["instrument"].astype(str)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    labels = None
    if mode == "train":
        labels = _pure_forward_labels(
            start_date,
            end_date,
            set(instruments),
        )

    windows: list[np.ndarray] = []
    targets: list[np.float32] = []
    keys: list[tuple[pd.Timestamp, str]] = []
    for instrument, stock in raw.groupby("instrument", sort=False, observed=True):
        values = _transform_raw(stock[FEATURE_COLS].to_numpy(copy=False))
        dates = stock["date"].dt.normalize().to_numpy()
        close_positions = np.flatnonzero(np.r_[dates[1:] != dates[:-1], True])
        for position in close_positions:
            date = pd.Timestamp(dates[position])
            if position + 1 < SEQ_LEN or date < start_ts or date > end_ts:
                continue
            key = (date, str(instrument))
            if mode == "train":
                target = labels.get(key) if labels is not None else None
                if target is None or not np.isfinite(target):
                    continue
                targets.append(np.float32(target))
            windows.append(values[position - SEQ_LEN + 1 : position + 1])
            keys.append(key)

    if not windows:
        raise RuntimeError(
            f"No samples for mode={mode}, table={table}, {start_date}..{end_date}"
        )
    x = np.stack(windows).astype(np.float32, copy=False)
    del windows, raw
    if stats is None:
        mean = np.nanmean(x, axis=(0, 1)).astype(np.float32)
        std = np.nanstd(x, axis=(0, 1)).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        stats = (mean, std)
    mean, std = stats
    x -= mean.reshape(1, 1, -1)
    x /= std.reshape(1, 1, -1)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
    np.clip(x, -8.0, 8.0, out=x)
    key_frame = pd.DataFrame(keys, columns=["date", "instrument"])
    y = np.asarray(targets, dtype=np.float32) if mode == "train" else None
    logger.info(
        "dataset built",
        mode=mode,
        samples=len(key_frame),
        instruments=key_frame["instrument"].nunique(),
        seconds=round(time.time() - started, 2),
    )
    return x, y, key_frame, stats


def _checkpoint_to_json(
    model: nn.Module,
    stats: tuple[np.ndarray, np.ndarray],
    model_path: str,
) -> None:
    tensors = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    mean, std = stats
    payload = {
        "state_dict": tensors,
        "model_config": asdict(MODEL_CONFIG),
        "feature_cols": FEATURE_COLS,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "seed": SEED,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def load_checkpoint(
    model_path: str = MODEL_PATH,
    device: torch.device | str = "cpu",
) -> tuple[RawMicrostructureTransformer, tuple[np.ndarray, np.ndarray], dict]:
    with open(model_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg = ModelConfig(**payload["model_config"])
    model = RawMicrostructureTransformer(cfg).to(device)
    state = {}
    for name, meta in payload["state_dict"].items():
        tensor = torch.tensor(
            meta["data"],
            dtype=getattr(torch, meta["dtype"]),
        ).reshape(meta["shape"])
        state[name] = tensor.to(device)
    model.load_state_dict(state)
    stats = (
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
    )
    return model, stats, payload


def train_and_save(
    datasources: dict[str, str] | None = None,
    model_path: str = MODEL_PATH,
) -> str:
    """Train from zero on the frozen training interval and save text weights."""
    set_deterministic()
    table = (datasources or {}).get("bar5m", TRAIN_TABLE)
    instruments = _training_instruments()
    x, y, _, stats = build_dataset(
        table,
        TRAIN_START,
        TRAIN_END,
        "train",
        instruments,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawMicrostructureTransformer(MODEL_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if not 100_000 <= n_params <= 100_000_000:
        raise RuntimeError(f"Non-compliant trainable parameter count: {n_params}")
    logger.info(
        "training start",
        device=str(device),
        n_params=n_params,
        samples=len(y),
        instruments=len(instruments),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.1,
    )
    loss_function = nn.SmoothL1Loss(beta=0.5)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        batches = 0
        started = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        logger.info(
            "epoch finished",
            epoch=epoch + 1,
            loss=round(total_loss / max(batches, 1), 8),
            lr=optimizer.param_groups[0]["lr"],
            seconds=round(time.time() - started, 2),
        )
    _checkpoint_to_json(model, stats, model_path)
    logger.info("model saved", path=model_path, bytes=os.path.getsize(model_path))
    return model_path


def predict(
    datasources: dict[str, str],
    start_date: str,
    end_date: str,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, _ = load_checkpoint(model_path, device)
    model.eval()
    table = datasources["bar5m"]
    instruments = _pool(table, start_date, end_date)
    x, _, keys, _ = build_dataset(
        table,
        start_date,
        end_date,
        "infer",
        instruments,
        stats,
    )
    scores = []
    tensor = torch.from_numpy(x)
    with torch.no_grad():
        for start in range(0, len(tensor), BATCH_SIZE):
            xb = tensor[start : start + BATCH_SIZE].to(device, non_blocking=True)
            scores.append(model(xb).cpu().numpy())
    keys["score"] = np.concatenate(scores).astype(np.float64)
    official = dai.query(
        f"SELECT date, instrument FROM {INSTRUMENT_TABLE}",
        filters={"date": [start_date, end_date]},
    ).df()
    official["date"] = pd.to_datetime(official["date"]).dt.normalize()
    official["instrument"] = official["instrument"].astype(str)
    result = keys.merge(official, on=["date", "instrument"], how="inner")
    result = (
        result.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    return result[["date", "instrument", "score"]]


if __name__ == "__main__":
    train_and_save({"bar5m": TRAIN_TABLE})
