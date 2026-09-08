"""Tier 1 F1 ensemble with inner-selected per-seed epoch counts.

Public evaluation loads ``weights.json``. Private evaluation calls
``train_and_save`` and retrains the identical fixed 2022-2023 configuration.
All 26 numeric raw fields from the permitted 28-column local bar1m schema are
model inputs; date and instrument remain grouping/identity keys.
"""

from __future__ import annotations

import json
import math
import os
import random
import time

try:
    import dai
except ImportError:  # Local packaging and checkpoint validation.
    dai = None
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "weights.json")
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31 23:59:59"
SEQ_LEN = 64
BATCH = 512
LR = 5e-4
SEEDS = (11, 29, 47)
MAX_EPOCHS = 15
INNER_TRAIN_END = "2023-06-30 23:59:59"
INNER_VALID_START = "2023-07-01"
INNER_VALID_END = TRAIN_END
DATES_PER_STEP = 4
MAX_TRAIN_INSTRUMENTS = 600
CHUNK_SIZE = 50
SOFT_RANK_TEMPERATURE = 0.5
FEATURE_COLS = [
    "adjust_factor",
    "high",
    "open",
    "low",
    "close",
    "deal_number",
    "volume",
    "amount",
    "ask_price1",
    "ask_price2",
    "ask_price3",
    "bid_price1",
    "bid_price2",
    "bid_price3",
    "ask_volume1",
    "ask_volume2",
    "ask_volume3",
    "bid_volume1",
    "bid_volume2",
    "bid_volume3",
    "ask_num_orders1",
    "ask_num_orders2",
    "ask_num_orders3",
    "bid_num_orders1",
    "bid_num_orders2",
    "bid_num_orders3",
]
LOG1P_COLS = [
    "deal_number",
    "volume",
    "amount",
    "ask_volume1",
    "ask_volume2",
    "ask_volume3",
    "bid_volume1",
    "bid_volume2",
    "bid_volume3",
    "ask_num_orders1",
    "ask_num_orders2",
    "ask_num_orders3",
    "bid_num_orders1",
    "bid_num_orders2",
    "bid_num_orders3",
]
MODEL_CFG = {
    "n_feat": len(FEATURE_COLS),
    "d_model": 80,
    "nhead": 4,
    "nlayers": 2,
    "dim_ff": 320,
    "seq_len": SEQ_LEN,
    "pooling": "last",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def causal_attention_mask(length, device):
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, padding=0
        )

    def forward(self, hidden):
        hidden = torch.nn.functional.pad(
            hidden.transpose(1, 2), (self.left_padding, 0)
        )
        return self.depthwise(hidden).transpose(1, 2)


class CausalMultiScaleTransformer(nn.Module):
    """Left-padded 3/7/15-minute filters plus causally masked attention."""

    def __init__(
        self,
        n_feat,
        d_model,
        nhead,
        nlayers,
        dim_ff,
        seq_len,
        pooling,
    ):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(n_feat, d_model)
        self.local_branches = nn.ModuleList(
            CausalDepthwiseConv(d_model, kernel) for kernel in (3, 7, 15)
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.local_gate = nn.Parameter(torch.zeros(d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, features):
        projected = self.proj(features)
        local = self.local_fusion(
            torch.cat(
                [branch(projected) for branch in self.local_branches], dim=-1
            )
        )
        hidden = (
            projected
            + torch.sigmoid(self.local_gate) * local
            + self.pos[:, : features.shape[1]]
        )
        hidden = self.encoder(
            hidden, mask=causal_attention_mask(hidden.shape[1], hidden.device)
        )
        pooled = hidden[:, -1] if self.pooling == "last" else hidden.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def _pearson(prediction, target):
    px = prediction - prediction.mean()
    py = target - target.mean()
    return (px * py).sum() / (px.norm() * py.norm() + 1e-8)


def _soft_rank(values, temperature=SOFT_RANK_TEMPERATURE):
    centered = values - values.mean()
    scale = centered.pow(2).mean().sqrt().clamp_min(1e-6)
    normalized = centered / scale
    differences = normalized[:, None] - normalized[None, :]
    return 1.0 + torch.sigmoid(differences / temperature).sum(dim=1)


class SoftSpearmanLoss(nn.Module):
    def forward(self, prediction, target, groups):
        daily = []
        for group in torch.unique(groups, sorted=True):
            mask = groups == group
            prediction_rank = _soft_rank(prediction[mask])
            target_rank = torch.argsort(torch.argsort(target[mask])).to(
                prediction.dtype
            )
            daily.append(1.0 - _pearson(prediction_rank, target_rank))
        return torch.stack(daily).mean()


class MultiDateBatchSampler(Sampler):
    def __init__(self, groups, dates_per_step, seed, shuffle=True):
        groups = np.asarray(groups, dtype=np.int64)
        self.daily_indices = [
            np.flatnonzero(groups == group).tolist() for group in np.unique(groups)
        ]
        self.dates_per_step = dates_per_step
        self.rng = random.Random(seed)
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.daily_indices)))
        if self.shuffle:
            self.rng.shuffle(order)
        for begin in range(0, len(order), self.dates_per_step):
            yield [
                index
                for day in order[begin : begin + self.dates_per_step]
                for index in self.daily_indices[day]
            ]

    def __len__(self):
        return math.ceil(len(self.daily_indices) / self.dates_per_step)


def pool(start_date, end_date):
    if dai is None:
        raise RuntimeError("dai is available only in BigQuant AIStudio")
    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    return frame["instrument"].tolist()


def _chunks(values, size):
    for begin in range(0, len(values), size):
        yield values[begin : begin + size]


def _query_chunk(table, buffer_start, end_date, instruments):
    if dai is None:
        raise RuntimeError("dai is available only in BigQuant AIStudio")
    sql = (
        f"SELECT date, instrument, {', '.join(FEATURE_COLS)} "
        f"FROM {table} ORDER BY instrument, date"
    )
    frame = dai.query(
        sql,
        filters={
            "date": [buffer_start, end_date],
            "instrument": instruments,
        },
    ).df()
    for column in LOG1P_COLS:
        frame[column] = np.log1p(frame[column].clip(lower=0))
    return frame


def _windows_from_one(frame, start_date, end_date, need_label):
    windows, targets, keys = [], [], []
    if len(frame) <= SEQ_LEN:
        return windows, targets, keys
    instrument = frame["instrument"].iloc[0]
    features = frame[FEATURE_COLS].ffill().to_numpy(np.float32, copy=True)
    dates = frame["date"].dt.normalize().to_numpy()
    close_positions = np.flatnonzero(np.append(dates[1:] != dates[:-1], True))
    close = frame["close"].to_numpy(np.float64)[close_positions]
    for day_index, position in enumerate(close_positions):
        date = pd.Timestamp(dates[position])
        if (
            position + 1 < SEQ_LEN
            or date < start_date
            or date > end_date
        ):
            continue
        target = None
        if day_index + 1 < len(close_positions) and close[day_index] > 0:
            value = close[day_index + 1] / close[day_index] - 1.0
            if np.isfinite(value):
                target = np.float32(value)
        if need_label and target is None:
            continue
        window = features[position - SEQ_LEN + 1 : position + 1]
        if not np.isfinite(window).all():
            continue
        windows.append(window)
        targets.append(target if target is not None else np.float32(0.0))
        keys.append((date, instrument))
    return windows, targets, keys


class RunningStats:
    def __init__(self, n_feat):
        self.count = 0
        self.total = np.zeros(n_feat, np.float64)
        self.square_total = np.zeros(n_feat, np.float64)

    def update(self, values):
        values64 = values.astype(np.float64, copy=False)
        self.count += len(values64)
        self.total += values64.sum(axis=0)
        self.square_total += (values64**2).sum(axis=0)

    def finalize(self):
        mean = self.total / self.count
        variance = self.square_total / self.count - mean**2
        std = np.sqrt(np.clip(variance, 0, None)) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)


def build_train_dataset(table, start_date, end_date, instruments):
    buffer_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=20)
    ).strftime("%Y-%m-%d")
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    all_windows, all_targets, all_keys = [], [], []
    stats = RunningStats(len(FEATURE_COLS))
    for chunk_number, chunk in enumerate(_chunks(instruments, CHUNK_SIZE), 1):
        frame = _query_chunk(table, buffer_start, end_date, chunk)
        for _, instrument_frame in frame.groupby("instrument", sort=False):
            windows, targets, keys = _windows_from_one(
                instrument_frame, start_ts, end_ts, need_label=True
            )
            for window in windows:
                stats.update(window)
            all_windows.extend(windows)
            all_targets.extend(targets)
            all_keys.extend(keys)
        del frame
        logger.info(
            "training chunk complete",
            chunk=chunk_number,
            samples=len(all_windows),
        )
    if not all_windows:
        raise RuntimeError("training dataset contains no samples")
    features = np.stack(all_windows).astype(np.float32)
    targets = np.asarray(all_targets, np.float32)
    mean, std = stats.finalize()
    features = ((features - mean) / std).astype(np.float32)
    lower, upper = np.percentile(targets, [1, 99])
    np.clip(targets, lower, upper, out=targets)
    key_frame = pd.DataFrame(all_keys, columns=["date", "instrument"])
    groups, _ = pd.factorize(pd.to_datetime(key_frame["date"]), sort=True)
    return features, targets, groups.astype(np.int64), (mean, std)


def _make_loader(features, targets, groups, seed, device):
    sampler = MultiDateBatchSampler(
        groups, DATES_PER_STEP, seed, shuffle=True
    )
    return DataLoader(
        TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.from_numpy(groups),
        ),
        batch_sampler=sampler,
        pin_memory=device.type == "cuda",
    )


def _run_training_epoch(model, loader, optimizer, objective, device):
    losses = []
    model.train()
    for batch_features, batch_targets, batch_groups in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        batch_groups = batch_groups.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(batch_features), batch_targets, batch_groups)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _predict_array(model, features, device):
    model.eval()
    tensor = torch.from_numpy(features)
    outputs = []
    with torch.no_grad():
        for begin in range(0, len(tensor), BATCH):
            outputs.append(
                model(tensor[begin : begin + BATCH].to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs)


def _rank_ic_mean(scores, targets, groups):
    values = []
    for group in np.unique(groups):
        mask = groups == group
        if int(mask.sum()) < 3:
            continue
        score_rank = pd.Series(scores[mask]).rank(
            method="average"
        ).to_numpy()
        target_rank = pd.Series(targets[mask]).rank(
            method="average"
        ).to_numpy()
        correlation = np.corrcoef(score_rank, target_rank)[0, 1]
        if np.isfinite(correlation):
            values.append(float(correlation))
    if not values:
        raise RuntimeError("validation Rank IC has no finite dates")
    return float(np.mean(values))


def select_best_epoch(
    train_features,
    train_targets,
    train_groups,
    valid_features,
    valid_targets,
    valid_groups,
    seed,
    device,
):
    set_seed(seed)
    model = CausalMultiScaleTransformer(**MODEL_CFG).to(device)
    loader = _make_loader(
        train_features, train_targets, train_groups, seed, device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    objective = SoftSpearmanLoss().to(device)
    best_epoch, best_ic = 1, float("-inf")
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time()
        loss = _run_training_epoch(
            model, loader, optimizer, objective, device
        )
        scores = _predict_array(model, valid_features, device)
        rank_ic = _rank_ic_mean(scores, valid_targets, valid_groups)
        history.append(
            {"epoch": epoch, "train_loss": loss, "valid_rank_ic": rank_ic}
        )
        if rank_ic > best_ic:
            best_ic = rank_ic
            best_epoch = epoch
        logger.info(
            "inner validation epoch complete",
            seed=seed,
            epoch=epoch,
            loss=loss,
            valid_rank_ic=rank_ic,
            elapsed_seconds=round(time.time() - started, 2),
        )
    return best_epoch, history


def train_one(features, targets, groups, seed, device, epochs):
    set_seed(seed)
    model = CausalMultiScaleTransformer(**MODEL_CFG).to(device)
    loader = _make_loader(features, targets, groups, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    objective = SoftSpearmanLoss().to(device)
    for epoch in range(1, epochs + 1):
        started = time.time()
        loss = _run_training_epoch(
            model, loader, optimizer, objective, device
        )
        logger.info(
            "refit epoch complete",
            seed=seed,
            epoch=epoch,
            selected_epochs=epochs,
            loss=loss,
            elapsed_seconds=round(time.time() - started, 2),
        )
    return model


def _serialize_state(state_dict):
    tensors = {}
    for name, value in state_dict.items():
        tensor = value.detach().cpu()
        tensors[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    return tensors


def _deserialize_state(tensors, map_location):
    state = {}
    for name, meta in tensors.items():
        tensor = torch.tensor(
            meta["data"], dtype=getattr(torch, meta["dtype"])
        ).reshape(meta["shape"])
        state[name] = tensor.to(map_location)
    return state


def save_ensemble(
    models,
    mean,
    std,
    model_path=MODEL_PATH,
    selected_epochs=None,
    selection_history=None,
):
    payload = {
        "name": (
            "F1_causal_multiscale_soft_spearman_28column_26raw_"
            "capacity_tier1_inner_selected_ensemble_s11_s29_s47"
        ),
        "model_kind": "causal_multiscale_transformer",
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
        "seeds": list(SEEDS),
        "epochs": list(selected_epochs or [MAX_EPOCHS] * len(SEEDS)),
        "max_selection_epochs": MAX_EPOCHS,
        "selection_metric": "validation_rank_ic_mean",
        "selection_history": selection_history or {},
        "dates_per_step": DATES_PER_STEP,
        "soft_rank_temperature": SOFT_RANK_TEMPERATURE,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "blend": "equal mean of within-date percentile ranks",
        "states": [
            {"seed": seed, "state_dict": _serialize_state(model.state_dict())}
            for seed, model in zip(SEEDS, models)
        ],
    }
    with open(model_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    return model_path


def load_ensemble(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    models = []
    for item in payload["states"]:
        model = CausalMultiScaleTransformer(**payload["model_cfg"]).to(
            map_location
        )
        model.load_state_dict(
            _deserialize_state(item["state_dict"], map_location)
        )
        model.eval()
        models.append(model)
    return payload, models


def predict_scores(models, table, start_date, end_date, instruments, stats, device):
    mean, std = stats
    buffer_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=20)
    ).strftime("%Y-%m-%d")
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    outputs = []
    for chunk_number, chunk in enumerate(_chunks(instruments, CHUNK_SIZE), 1):
        frame = _query_chunk(table, buffer_start, end_date, chunk)
        windows, keys = [], []
        for _, instrument_frame in frame.groupby("instrument", sort=False):
            local_windows, _, local_keys = _windows_from_one(
                instrument_frame, start_ts, end_ts, need_label=False
            )
            windows.extend(local_windows)
            keys.extend(local_keys)
        del frame
        if not windows:
            continue
        features = np.stack(windows).astype(np.float32)
        features = ((features - mean) / std).astype(np.float32)
        tensor = torch.from_numpy(features)
        chunk_output = pd.DataFrame(keys, columns=["date", "instrument"])
        for model_index, model in enumerate(models):
            values = []
            with torch.no_grad():
                for begin in range(0, len(tensor), BATCH):
                    values.append(
                        model(tensor[begin : begin + BATCH].to(device))
                        .cpu()
                        .numpy()
                    )
            chunk_output[f"model_{model_index}"] = np.concatenate(values)
        outputs.append(chunk_output)
        logger.info("inference chunk complete", chunk=chunk_number)
    if not outputs:
        raise RuntimeError("inference dataset contains no samples")
    result = pd.concat(outputs, ignore_index=True)
    rank_columns = []
    for model_index in range(len(models)):
        column = f"rank_{model_index}"
        result[column] = result.groupby("date")[f"model_{model_index}"].rank(
            method="average", pct=True
        )
        rank_columns.append(column)
    result["score"] = result[rank_columns].mean(axis=1)
    return result[["date", "instrument", "score"]]


def train_and_save(datasources, model_path=MODEL_PATH):
    table = datasources["bar1m"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instruments = sorted(pool(TRAIN_START, TRAIN_END))[:MAX_TRAIN_INSTRUMENTS]
    inner_train = build_train_dataset(
        table, TRAIN_START, INNER_TRAIN_END, instruments
    )
    inner_valid = build_train_dataset(
        table, INNER_VALID_START, INNER_VALID_END, instruments
    )
    train_features, train_targets, train_groups, train_stats = inner_train
    valid_features, valid_targets, valid_groups, valid_stats = inner_valid
    train_mean, train_std = train_stats
    valid_mean, valid_std = valid_stats
    valid_features = (
        (
            valid_features * valid_std
            + valid_mean
            - train_mean
        )
        / train_std
    ).astype(np.float32)
    selected_epochs = []
    selection_history = {}
    for seed in SEEDS:
        epoch, history = select_best_epoch(
            train_features,
            train_targets,
            train_groups,
            valid_features,
            valid_targets,
            valid_groups,
            seed,
            device,
        )
        selected_epochs.append(epoch)
        selection_history[str(seed)] = history

    features, targets, groups, stats = build_train_dataset(
        table, TRAIN_START, TRAIN_END, instruments
    )
    models = [
        train_one(
            features,
            targets,
            groups,
            seed,
            device,
            epoch,
        )
        for seed, epoch in zip(SEEDS, selected_epochs)
    ]
    mean, std = stats
    return save_ensemble(
        models,
        mean,
        std,
        model_path,
        selected_epochs=selected_epochs,
        selection_history=selection_history,
    )


if __name__ == "__main__":
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})
