"""Standalone BigAlpha F3 champion training and inference entrypoint.

This file is copied into the formal submission as ``train.py``.  It contains
the frozen 26-field, 64-minute CNN+Transformer geometry champion without
repository imports.  The organizer entrypoint is ``main`` and returns exactly
``date, instrument, score``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset


HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "weights.json"
TRAIN_START, TRAIN_END = "2019-01-01", "2024-12-31 23:59:59"
SEQ_LEN, EPOCHS, LR, SEED = 64, 15, 5e-4, 11
DATES_PER_STEP, EMA_DECAY, UNIVERSE_LIMIT = 24, 0.9415, 600
DATE_CHUNK_DAYS = 5

FEATURE_COLS = [
    "adjust_factor", "high", "open", "low", "close", "deal_number",
    "volume", "amount", "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3", "ask_volume1",
    "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2",
    "bid_volume3", "ask_num_orders1", "ask_num_orders2",
    "ask_num_orders3", "bid_num_orders1", "bid_num_orders2",
    "bid_num_orders3",
]
OHLC_COLS = ["open", "high", "low", "close"]
COUNT_COLS = [
    "deal_number", "volume", "amount", "ask_volume1", "ask_volume2",
    "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
PRICE_INDICES = tuple(range(8, 14))
VOLUME_INDICES = tuple(range(14, 20))
ORDER_INDICES = tuple(range(20, 26))
MODEL_CFG = {
    "n_feat": 26,
    "d_model": 80,
    "nhead": 4,
    "nlayers": 2,
    "dim_ff": 320,
    "seq_len": SEQ_LEN,
    "pooling": "last",
}
DATA_CONTRACT = {
    "frequency": "1min",
    "sequence_length": SEQ_LEN,
    "label": "strict next-trading-day raw close return",
    "feature_source": "26 organizer-provided numeric fields",
    "price_adjustment": False,
    "engineered_features": "same-minute same-instrument causal book geometry",
    "stock_axis_mixing": False,
    "training_universe": "sorted first 600 instruments",
    "inference_universe": "daily full dynamic",
}


def _index(features: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    index = torch.as_tensor(indices, dtype=torch.long, device=features.device)
    return features.index_select(-1, index)


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((length, length), float("-inf"), device=device), diagonal=1
    )


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, padding=0
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.pad(hidden.transpose(1, 2), (self.left_padding, 0))
        return self.depthwise(hidden).transpose(1, 2)


class CausalMultiScaleTransformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        seq_len: int,
        pooling: str,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.1, batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.local_branches = nn.ModuleList(
            CausalDepthwiseConv(d_model, kernel) for kernel in (3, 7, 15)
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.local_gate = nn.Parameter(torch.zeros(d_model))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        local = self.local_fusion(
            torch.cat([branch(projected) for branch in self.local_branches], dim=-1)
        )
        hidden = (
            projected
            + torch.sigmoid(self.local_gate) * local
            + self.pos[:, : features.shape[1]]
        )
        mask = causal_attention_mask(hidden.shape[1], hidden.device)
        return self.encoder(hidden, mask=mask)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        pooled = hidden[:, -1] if self.pooling == "last" else hidden.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class BaselineGeometryTransformer(CausalMultiScaleTransformer):
    geometry_dim = 24

    def __init__(
        self,
        *args,
        feature_mean: list[float] | None = None,
        feature_std: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        raw_features = self.proj.in_features
        if raw_features != 26:
            raise ValueError("geometry champion requires exactly 26 raw fields")
        mean = torch.zeros(26) if feature_mean is None else torch.tensor(feature_mean)
        std = torch.ones(26) if feature_std is None else torch.tensor(feature_std)
        self.register_buffer("feature_mean", mean.float(), persistent=True)
        self.register_buffer("feature_std", std.float(), persistent=True)
        self.proj = nn.Linear(raw_features + self.geometry_dim, self.proj.out_features)

    def geometry(self, features: torch.Tensor) -> torch.Tensor:
        raw = features * self.feature_std + self.feature_mean
        price = _index(raw, PRICE_INDICES).reshape(*raw.shape[:2], 2, 3)
        volume = torch.expm1(_index(raw, VOLUME_INDICES).clamp_min(0.0)).reshape(
            *raw.shape[:2], 2, 3
        )
        orders = torch.expm1(_index(raw, ORDER_INDICES).clamp_min(0.0)).reshape(
            *raw.shape[:2], 2, 3
        )
        mid = 0.5 * (price[..., 0, 0] + price[..., 1, 0])
        scale = mid.abs().clamp_min(1e-6)
        relative_price = (
            (price - mid[..., None, None]) / scale[..., None, None]
        ).flatten(-2)
        spread = (price[..., 0, :] - price[..., 1, :]) / scale[..., None]
        volume_imbalance = (volume[..., 1, :] - volume[..., 0, :]) / (
            volume.sum(dim=-2).clamp_min(1.0)
        )
        order_imbalance = (orders[..., 1, :] - orders[..., 0, :]) / (
            orders.sum(dim=-2).clamp_min(1.0)
        )
        average_order = torch.log1p(volume / orders.clamp_min(1.0)).flatten(-2)
        cumulative_depth = (
            (volume[..., 1, :].sum(dim=-1) - volume[..., 0, :].sum(dim=-1))
            / volume.sum(dim=(-2, -1)).clamp_min(1.0)
        ).unsqueeze(-1)
        slope = torch.stack(
            [
                (price[..., 0, 2] - price[..., 0, 0]) / scale,
                (price[..., 1, 0] - price[..., 1, 2]) / scale,
            ],
            dim=-1,
        )
        value = torch.cat(
            [relative_price, spread, volume_imbalance, order_imbalance,
             average_order, cumulative_depth, slope],
            dim=-1,
        )
        if value.shape[-1] != self.geometry_dim:
            raise AssertionError(value.shape)
        return torch.nan_to_num(
            value, nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return super().encode(torch.cat([features, self.geometry(features)], dim=-1))


def build_model(mean: np.ndarray, std: np.ndarray) -> BaselineGeometryTransformer:
    return BaselineGeometryTransformer(
        **MODEL_CFG, feature_mean=mean.tolist(), feature_std=std.tolist()
    )


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hard_rank(values: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(values, stable=True), stable=True).to(values.dtype)


def soft_rank(values: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    centered = values - values.mean()
    normalized = centered / centered.square().mean().sqrt().clamp_min(1e-6)
    differences = normalized[:, None] - normalized[None, :]
    return 1.0 + torch.sigmoid(differences / temperature).sum(dim=1)


def pearson(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.reshape(-1) - left.mean()
    right = right.reshape(-1) - right.mean()
    return (left * right).sum() / (left.norm() * right.norm() + 1e-8)


class L5Loss(nn.Module):
    """Epochs 1-8 Soft-Spearman, epochs 9-15 Rank-target IC."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        values = []
        for group in torch.unique(groups, sorted=True):
            mask = groups == group
            day_prediction, day_target = prediction[mask], target[mask]
            target_rank = hard_rank(day_target)
            if self.epoch <= 8:
                values.append(pearson(soft_rank(day_prediction), target_rank))
            else:
                values.append(pearson(day_prediction, target_rank))
        if not values:
            raise RuntimeError("L5 requires complete daily cross-sections")
        return 1.0 - torch.stack(values).mean()


class MultiDateBatchSampler(Sampler[list[int]]):
    def __init__(self, dates: pd.Series, seed: int = SEED) -> None:
        date_values = pd.to_datetime(dates).to_numpy()
        self.daily = [
            np.flatnonzero(date_values == date).tolist()
            for date in np.unique(date_values)
        ]
        if not self.daily or min(map(len, self.daily)) < 2:
            raise RuntimeError("24-date batches require at least two stocks per date")
        self.rng = random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.daily)))
        self.rng.shuffle(order)
        for begin in range(0, len(order), DATES_PER_STEP):
            yield [
                index
                for day in order[begin : begin + DATES_PER_STEP]
                for index in self.daily[day]
            ]

    def __len__(self) -> int:
        return math.ceil(len(self.daily) / DATES_PER_STEP)


def _canonicalize_cloud(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype(str)
    for column in FEATURE_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in OHLC_COLS:
        frame.loc[frame[column] == -1, column] = np.nan
    for column in COUNT_COLS:
        frame[column] = np.log1p(frame[column].clip(lower=0)).astype(np.float32)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame["day"] = frame["date"].dt.normalize()
    frame.sort_values(["day", "instrument", "date"], kind="stable", inplace=True)
    frame[FEATURE_COLS] = frame.groupby(
        ["day", "instrument"], sort=False
    )[FEATURE_COLS].ffill()
    return frame


def _normalize_universe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["date", "instrument"]].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return (
        result.dropna().drop_duplicates(["date", "instrument"])
        .sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    )


def _query_universe(start_date: str, end_date: str) -> pd.DataFrame:
    import dai

    frame = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    return _normalize_universe(frame)


def _date_chunks(universe: pd.DataFrame, size: int = DATE_CHUNK_DAYS):
    days = sorted(pd.to_datetime(universe["date"]).dt.normalize().unique())
    for begin in range(0, len(days), size):
        yield pd.DatetimeIndex(days[begin : begin + size])


def _query_bars(table: str, universe: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    import dai

    selected = universe[universe["date"].isin(set(days))]
    instruments = sorted(selected["instrument"].unique().tolist())
    start = pd.Timestamp(days[0]).strftime("%Y-%m-%d 00:00:00")
    end = pd.Timestamp(days[-1]).strftime("%Y-%m-%d 23:59:59")
    fields = ", ".join(["date", "instrument", *FEATURE_COLS])
    return dai.query(
        f"SELECT {fields} FROM {table} ORDER BY date, instrument",
        filters={"date": [start, end], "instrument": instruments},
    ).df()


def _chunk_records(raw: pd.DataFrame, universe: pd.DataFrame, days: pd.DatetimeIndex):
    frame = _canonicalize_cloud(raw)
    selected = universe[universe["date"].isin(set(days))].rename(columns={"date": "day"})
    frame = frame.merge(selected, on=["day", "instrument"], how="inner", sort=False)
    closes, windows = {}, {}
    if not frame.empty:
        daily_last = frame.groupby(["day", "instrument"], sort=False).tail(1)
        for day, sub in daily_last.groupby("day", sort=False):
            values = sub.set_index("instrument")["close"]
            closes[pd.Timestamp(day)] = values[np.isfinite(values) & values.gt(0)]
        trimmed = frame.groupby(["day", "instrument"], sort=False).tail(SEQ_LEN)
        sizes = trimmed.groupby(["day", "instrument"], sort=False)["date"].transform("size")
        trimmed = trimmed[sizes.eq(SEQ_LEN)]
        if not trimmed.empty:
            x = trimmed[FEATURE_COLS].to_numpy(np.float32, copy=True).reshape(
                -1, SEQ_LEN, len(FEATURE_COLS)
            )
            keys = trimmed.iloc[SEQ_LEN - 1 :: SEQ_LEN][["day", "instrument"]].reset_index(drop=True)
            valid = np.isfinite(x).all(axis=(1, 2))
            x, keys = x[valid], keys.loc[valid].reset_index(drop=True)
            for day, positions in keys.groupby("day", sort=False).groups.items():
                index = np.asarray(list(positions), dtype=np.int64)
                windows[pd.Timestamp(day)] = (
                    keys.loc[index, "instrument"].to_numpy(dtype=str), x[index]
                )
    return windows, closes


@dataclass
class PreparedTrainingData:
    x: np.memmap
    y: np.ndarray
    keys: pd.DataFrame
    mean: np.ndarray
    std: np.ndarray
    temporary_directory: tempfile.TemporaryDirectory

    def close(self) -> None:
        self.x.flush()
        mmap = getattr(self.x, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self.temporary_directory.cleanup()


def _prepare_training_data(table: str, universe: pd.DataFrame) -> PreparedTrainingData:
    temporary_directory = tempfile.TemporaryDirectory(prefix="f3_submit_")
    cache_path = Path(temporary_directory.name) / "windows.f32"
    x_store = np.memmap(
        cache_path, mode="w+", dtype=np.float32,
        shape=(len(universe), SEQ_LEN, len(FEATURE_COLS)),
    )
    y_store = np.empty(len(universe), dtype=np.float32)
    key_parts, offset, previous = [], 0, None
    feature_sum = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    feature_square_sum = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    try:
        for days in _date_chunks(universe):
            windows, closes = _chunk_records(_query_bars(table, universe, days), universe, days)
            for day in pd.DatetimeIndex(days):
                current_closes = closes.get(pd.Timestamp(day), pd.Series(dtype=float))
                if previous is not None:
                    prev_day, instruments, features, prev_closes = previous
                    next_closes = current_closes.reindex(instruments).to_numpy(np.float64)
                    targets = next_closes / prev_closes - 1.0
                    valid = np.isfinite(targets) & np.isfinite(prev_closes) & (prev_closes > 0)
                    features, targets, instruments = features[valid], targets[valid], instruments[valid]
                    count = len(instruments)
                    if count:
                        x_store[offset : offset + count] = features
                        y_store[offset : offset + count] = targets.astype(np.float32)
                        key_parts.append(pd.DataFrame({
                            "date": np.repeat(prev_day, count), "instrument": instruments
                        }))
                        flat = features.reshape(-1, len(FEATURE_COLS)).astype(np.float64, copy=False)
                        feature_sum += flat.sum(axis=0)
                        feature_square_sum += np.square(flat).sum(axis=0)
                        offset += count
                current = windows.get(pd.Timestamp(day))
                if current is None:
                    previous = None
                else:
                    instruments, features = current
                    prev_closes = current_closes.reindex(instruments).to_numpy(np.float64)
                    previous = (pd.Timestamp(day), instruments, features, prev_closes)
            print(f"prepared through {pd.Timestamp(days[-1]).date()}: {offset:,} samples", flush=True)
        if not offset:
            raise RuntimeError("no valid full-history training windows")
        observation_count = offset * SEQ_LEN
        mean = (feature_sum / observation_count).astype(np.float32)
        variance = feature_square_sum / observation_count - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32) + 1e-6
        for begin in range(0, offset, 4096):
            chunk = x_store[begin : min(offset, begin + 4096)]
            np.subtract(chunk, mean, out=chunk)
            np.divide(chunk, std, out=chunk)
        y = y_store[:offset]
        lower, upper = np.percentile(y, [1, 99])
        np.clip(y, lower, upper, out=y)
        return PreparedTrainingData(
            x_store[:offset], y, pd.concat(key_parts, ignore_index=True),
            mean, std, temporary_directory,
        )
    except Exception:
        mmap = getattr(x_store, "_mmap", None)
        if mmap is not None:
            mmap.close()
        temporary_directory.cleanup()
        raise


def _tensor_payload(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu()
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "data": tensor.reshape(-1).tolist(),
    }


def save_weights(
    model: nn.Module, mean: np.ndarray, std: np.ndarray, history: list[dict],
    model_path: Path = WEIGHTS_PATH,
) -> str:
    payload = {
        "name": "F3_baseline_geometry_L5_EMA_full_2019_2024",
        "model_type": "BaselineGeometryTransformer",
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
        "epochs": EPOCHS,
        "seed": SEED,
        "dates_per_step": DATES_PER_STEP,
        "ema_decay": EMA_DECAY,
        "training_range": [TRAIN_START, TRAIN_END],
        "data_contract": DATA_CONTRACT,
        "history": history,
        "state_dict": {name: _tensor_payload(value) for name, value in model.state_dict().items()},
    }
    text = json.dumps(payload, separators=(",", ":"))
    Path(model_path).write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def load_weights(model_path: Path = WEIGHTS_PATH, device: str | torch.device = "cpu"):
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if payload.get("feature_cols") != FEATURE_COLS or payload.get("data_contract") != DATA_CONTRACT:
        raise RuntimeError("checkpoint input contract differs from train.py")
    mean, std = np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32)
    model = build_model(mean, std).to(device)
    state = {
        name: torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"])).reshape(meta["shape"])
        for name, meta in payload["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mean, std


def train_and_save(datasources: dict, model_path: Path = WEIGHTS_PATH) -> str:
    if "bar1m" not in datasources:
        raise KeyError("F3 requires the bar1m datasource")
    _set_seed()
    universe = _query_universe(TRAIN_START, TRAIN_END)
    selected = sorted(universe["instrument"].unique().tolist())[:UNIVERSE_LIMIT]
    universe = universe[universe["instrument"].isin(selected)].reset_index(drop=True)
    prepared = _prepare_training_data(datasources["bar1m"], universe)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(prepared.mean, prepared.std).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        shadow = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        loss_fn = L5Loss().to(device)
        groups = pd.factorize(pd.to_datetime(prepared.keys["date"]), sort=True)[0].astype(np.int64)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(prepared.x), torch.from_numpy(prepared.y), torch.from_numpy(groups)),
            batch_sampler=MultiDateBatchSampler(prepared.keys["date"]),
            pin_memory=device.type == "cuda",
        )
        history = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_fn.set_epoch(epoch)
            losses, gradients = [], []
            for features, target, date_groups in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(features.to(device, non_blocking=True))
                loss = loss_fn(
                    prediction, target.to(device, non_blocking=True),
                    date_groups.to(device, non_blocking=True),
                )
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        shadow[name].mul_(EMA_DECAY).add_(parameter, alpha=1.0 - EMA_DECAY)
                losses.append(float(loss.detach().cpu()))
                gradients.append(float(gradient.detach().cpu()))
            row = {
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "gradient_norm_mean": float(np.mean(gradients)),
                "optimizer_steps": len(losses),
            }
            history.append(row)
            print(f"epoch {epoch}/{EPOCHS}: loss={row['train_loss']:.6f}", flush=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(shadow[name])
        save_weights(model, prepared.mean, prepared.std, history, Path(model_path))
        return str(model_path)
    finally:
        prepared.close()


def predict_scores(
    model: nn.Module, table: str, start_date: str, end_date: str,
    mean: np.ndarray, std: np.ndarray, device: torch.device,
):
    universe = _query_universe(start_date, end_date)
    output = []
    model.eval()
    with torch.inference_mode():
        for days in _date_chunks(universe):
            windows, _ = _chunk_records(_query_bars(table, universe, days), universe, days)
            for day in pd.DatetimeIndex(days):
                current = windows.get(pd.Timestamp(day))
                if current is None:
                    continue
                instruments, features = current
                np.subtract(features, mean, out=features)
                np.divide(features, std, out=features)
                values = []
                for begin in range(0, len(features), 2048):
                    tensor = torch.from_numpy(features[begin : begin + 2048]).to(device)
                    values.append(model(tensor).cpu().numpy())
                output.append(pd.DataFrame({
                    "date": np.repeat(pd.Timestamp(day), len(instruments)),
                    "instrument": instruments,
                    "score": np.concatenate(values).astype(np.float64),
                }))
    if not output:
        raise RuntimeError("no valid 64-minute inference windows")
    return pd.concat(output, ignore_index=True), universe


def main(datasources: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Organizer entrypoint returning exactly date, instrument, and score."""
    if "bar1m" not in datasources:
        raise KeyError("F3 requires the bar1m datasource")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_weights(WEIGHTS_PATH, device)
    scores, universe = predict_scores(
        model, datasources["bar1m"], start_date, end_date, mean, std, device
    )
    return (
        scores.merge(universe, on=["date", "instrument"], how="inner")
        .replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"], keep="last")
        [["date", "instrument", "score"]]
        .sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    )


if __name__ == "__main__":
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})
