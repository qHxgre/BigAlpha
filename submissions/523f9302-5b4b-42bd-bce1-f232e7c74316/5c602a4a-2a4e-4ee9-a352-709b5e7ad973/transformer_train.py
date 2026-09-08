"""BigAlpha 2026 V5 training/inference single source of truth.

The platform calls main(datasources, start_date, end_date). The function
returns exactly: date, instrument, score.

Inference inputs are only the six raw organizer-provided fields. For score
date t, the model reads the preceding 20 market trading days and never reads
t's feature values; t's rows are used only to identify that day's universe.
"""

from pathlib import Path
import base64
import gc
import json

import numpy as np
import pandas as pd
import torch
from torch import nn


FEATURES = ["open", "high", "low", "close", "volume", "amount"]
MINUTES_PER_DAY = 240
LOOKBACK_DAYS = 20
PATCH_SIZE = 10
PATCHES_PER_DAY = MINUTES_PER_DAY // PATCH_SIZE

D_MODEL = 64
NHEAD = 4
INTRADAY_LAYERS = 2
DAILY_LAYERS = 2
CROSS_LAYERS = 2
DIM_FEEDFORWARD = 128
MARKET_QUERY_COUNT = 4
DROPOUT = 0.10
SEED = 42
TRAIN_START = "2019-01-01"
TRAIN_END = "2024-12-31 23:59:59"
TRAIN_EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
INFERENCE_CHUNK_SIZE = 25

MODEL_PATH = Path("transformer_model.json")
SOURCE_CHECKPOINT_PATH = Path("best_model.pt")


class LearnedPosition(nn.Module):
    def __init__(self, length, d_model, dropout):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, length, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        return self.dropout(values + self.position[:, : values.size(1)])


class HierarchicalMarketTransformerV5(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.patch_embedding = nn.Conv1d(
            n_features,
            D_MODEL,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )
        self.intraday_cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.intraday_position = LearnedPosition(
            PATCHES_PER_DAY + 1, D_MODEL, DROPOUT
        )
        intraday_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.intraday_encoder = nn.TransformerEncoder(
            intraday_layer, INTRADAY_LAYERS
        )
        self.intraday_norm = nn.LayerNorm(D_MODEL)

        self.daily_position = LearnedPosition(
            LOOKBACK_DAYS, D_MODEL, DROPOUT
        )
        daily_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.daily_encoder = nn.TransformerEncoder(
            daily_layer, DAILY_LAYERS
        )
        self.daily_norm = nn.LayerNorm(D_MODEL)

        cross_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(
            cross_layer, CROSS_LAYERS
        )
        self.market_queries = nn.Parameter(
            torch.randn(1, MARKET_QUERY_COUNT, D_MODEL) * 0.02
        )
        self.market_read = nn.MultiheadAttention(
            D_MODEL, NHEAD, dropout=DROPOUT, batch_first=True
        )
        self.stock_read = nn.MultiheadAttention(
            D_MODEL, NHEAD, dropout=DROPOUT, batch_first=True
        )
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def encode_days(self, values, minute_padding_mask, day_padding_mask):
        n_stocks, n_days, n_minutes, n_features = values.shape
        flat_values = values.reshape(
            n_stocks * n_days, n_minutes, n_features
        ).transpose(1, 2)
        patches = self.patch_embedding(flat_values).transpose(1, 2)
        flat_minute_mask = minute_padding_mask.reshape(
            n_stocks * n_days, n_minutes
        )
        patch_padding = flat_minute_mask.reshape(
            n_stocks * n_days, PATCHES_PER_DAY, PATCH_SIZE
        ).all(-1)
        cls = self.intraday_cls.expand(len(patches), -1, -1)
        tokens = self.intraday_position(torch.cat([cls, patches], dim=1))
        token_mask = torch.cat(
            [
                torch.zeros(
                    len(patches),
                    1,
                    dtype=torch.bool,
                    device=values.device,
                ),
                patch_padding,
            ],
            dim=1,
        )
        tokens = self.intraday_encoder(
            tokens, src_key_padding_mask=token_mask
        )
        daily_tokens = self.intraday_norm(tokens[:, 0]).reshape(
            n_stocks, n_days, D_MODEL
        )
        return daily_tokens.masked_fill(
            day_padding_mask.unsqueeze(-1), 0.0
        )

    def encode_stock_state(
        self, values, minute_padding_mask, day_padding_mask
    ):
        """Encode raw histories independently; safe to call in stock chunks."""
        daily_tokens = self.encode_days(
            values, minute_padding_mask, day_padding_mask
        )
        daily_tokens = self.daily_position(daily_tokens)
        daily_hidden = self.daily_encoder(
            daily_tokens, src_key_padding_mask=day_padding_mask
        )
        latest_context = daily_hidden[:, -1]
        latest_intraday = daily_tokens[:, -1]
        return self.daily_norm(latest_context + latest_intraday)

    def score_cross_section(self, stock_state):
        """Score one complete daily cross-section without chunk boundaries."""
        cross_state = self.cross_encoder(stock_state.unsqueeze(0))
        market_queries = self.market_queries.expand(
            cross_state.size(0), -1, -1
        )
        market_tokens, _ = self.market_read(
            market_queries, cross_state, cross_state, need_weights=False
        )
        market_context, _ = self.stock_read(
            cross_state, market_tokens, market_tokens, need_weights=False
        )
        fused = self.output_norm(cross_state + market_context)
        return self.head(fused).squeeze(0).squeeze(-1)

    def forward(self, values, minute_padding_mask, day_padding_mask):
        stock_state = self.encode_stock_state(
            values, minute_padding_mask, day_padding_mask
        )
        return self.score_cross_section(stock_state)


def _resolve_table(datasources):
    table = datasources.get("bar1m")
    if table is None:
        table = datasources.get("bigalpha_2026_stock_bar1m")
    if table is None:
        table = next(iter(datasources.values()))
    return table


def _tensor_to_json(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _tensor_from_json(payload):
    array = np.frombuffer(
        base64.b64decode(payload["data"]), dtype=np.dtype(payload["dtype"])
    ).reshape(payload["shape"]).copy()
    return torch.from_numpy(array)


def save_model(checkpoint, path=MODEL_PATH):
    """Save weights and preprocessing statistics as a portable JSON file."""
    payload = {
        "format": "bigalpha-v5-json-v1",
        "model_version": checkpoint.get(
            "model_version", "bar1m_hierarchical_v5_optimized"
        ),
        "model_state": {
            name: _tensor_to_json(value)
            for name, value in checkpoint["model_state"].items()
        },
        "feature_mean": checkpoint["feature_mean"],
        "feature_std": checkpoint["feature_std"],
        "config": checkpoint.get("config", {}),
        "seed": int(checkpoint.get("seed", 42)),
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return Path(path)


def load_model(path=MODEL_PATH, map_location="cpu"):
    """Load the JSON checkpoint without pickle or executable objects."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["model_state"] = {
        name: _tensor_from_json(value).to(map_location)
        for name, value in payload["model_state"].items()
    }
    return payload


def convert_checkpoint(
    source_path=SOURCE_CHECKPOINT_PATH, output_path=MODEL_PATH
):
    """Mechanically convert the trained local .pt checkpoint to JSON."""
    checkpoint = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    return save_model(checkpoint, output_path)


def _load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH.resolve()}")
    checkpoint = load_model(MODEL_PATH, map_location="cpu")
    model = HierarchicalMarketTransformerV5(len(FEATURES))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    mean = pd.Series(checkpoint["feature_mean"], dtype="float32")[FEATURES]
    std = (
        pd.Series(checkpoint["feature_std"], dtype="float32")[FEATURES]
        .replace(0, 1)
        .fillna(1)
    )
    return model, mean, std, checkpoint


def to_canonical(frame):
    """Convert the cloud 1m table to local training units.

    Cloud prices and amount are already in yuan, so they must not be divided
    by 100. Only organizer-provided raw fields are retained.
    """
    required = {"date", "instrument", *FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cloud data missing fields: {sorted(missing)}")
    result = frame[["date", "instrument", *FEATURES]].copy()
    result["date"] = pd.to_datetime(result["date"])
    for column in FEATURES:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("float32")
    return result


def build_daily_tensors(frame, mean, std):
    work = frame.sort_values(
        ["instrument", "date"], kind="mergesort"
    ).reset_index(drop=True)
    work["trade_date"] = work["date"].dt.normalize()
    groups = work.groupby(
        ["instrument", "trade_date"], sort=True, observed=True
    )
    work["_group_id"] = groups.ngroup().astype("int32")
    if work.empty:
        raise ValueError("Cloud query returned no 1m rows")
    group_count = int(work["_group_id"].max()) + 1
    work["_from_end"] = (
        work.groupby("_group_id", sort=False)
        .cumcount(ascending=False)
        .astype("int16")
    )
    selected = work.loc[
        work["_from_end"] < MINUTES_PER_DAY
    ].copy()
    selected["_slot"] = (
        MINUTES_PER_DAY - 1 - selected["_from_end"]
    ).astype("int16")

    normalized = (
        selected[FEATURES]
        .sub(mean, axis="columns")
        .div(std, axis="columns")
        .clip(-10, 10)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    group_ids = selected["_group_id"].to_numpy(np.int64)
    slots = selected["_slot"].to_numpy(np.int64)
    values = np.zeros(
        (group_count, MINUTES_PER_DAY, len(FEATURES)), dtype=np.float32
    )
    valid = np.zeros((group_count, MINUTES_PER_DAY), dtype=bool)
    values[group_ids, slots] = normalized
    valid[group_ids, slots] = True
    daily = (
        work.groupby("_group_id", sort=True)
        .agg(
            instrument=("instrument", "first"),
            date=("trade_date", "first"),
        )
        .reset_index(drop=True)
    )
    return daily, values, valid


def _history_indices_for_date(daily, lookup, market_dates, score_date):
    score_date = pd.Timestamp(score_date).normalize()
    score_rows = daily.loc[daily["date"] == score_date]
    if score_rows.empty:
        return None, None
    instruments = np.sort(score_rows["instrument"].astype(str).unique())
    date_position = market_dates.get_indexer([score_date])[0]
    if date_position < 0:
        return None, None
    history_dates = market_dates[
        max(0, date_position - LOOKBACK_DAYS) : date_position
    ]
    indices = np.full(
        (len(instruments), LOOKBACK_DAYS), -1, dtype=np.int64
    )
    if len(history_dates):
        query_index = pd.MultiIndex.from_arrays(
            [
                np.tile(history_dates.to_numpy(), len(instruments)),
                np.repeat(instruments, len(history_dates)),
            ],
            names=["date", "instrument"],
        )
        found = lookup.reindex(query_index).fillna(-1).to_numpy(np.int64)
        indices[:, -len(history_dates) :] = found.reshape(
            len(instruments), len(history_dates)
        )
    return instruments, indices


def predict_dates(model, daily, values, valid, start_date, end_date, device):
    market_dates = pd.DatetimeIndex(np.sort(daily["date"].unique()))
    lookup = pd.Series(
        np.arange(len(daily), dtype=np.int64),
        index=pd.MultiIndex.from_frame(
            daily[["date", "instrument"]].assign(
                instrument=lambda x: x["instrument"].astype(str)
            )
        ),
    )
    requested_dates = market_dates[
        (market_dates >= pd.Timestamp(start_date).normalize())
        & (market_dates <= pd.Timestamp(end_date).normalize())
    ]
    outputs = []
    amp_enabled = device.type == "cuda"

    for score_date in requested_dates:
        instruments, history_indices = _history_indices_for_date(
            daily, lookup, market_dates, score_date
        )
        if instruments is None:
            continue

        # Training only scores stocks with data on the immediately preceding
        # market day. Other universe members receive neutral score 0.
        fresh = history_indices[:, -1] >= 0
        date_scores = np.zeros(len(instruments), dtype=np.float32)
        if fresh.any():
            selected_indices = history_indices[fresh]
            day_valid = selected_indices >= 0
            safe_indices = np.where(day_valid, selected_indices, 0)
            x = np.array(values[safe_indices], copy=True)
            minute_valid = np.array(valid[safe_indices], copy=True)
            minute_valid &= day_valid[..., None]
            x[~day_valid] = 0.0

            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                prediction = model(
                    torch.from_numpy(x).to(device, non_blocking=True),
                    torch.from_numpy(~minute_valid).to(
                        device, non_blocking=True
                    ),
                    torch.from_numpy(~day_valid).to(
                        device, non_blocking=True
                    ),
                )
            date_scores[fresh] = prediction.float().cpu().numpy()

        outputs.append(
            pd.DataFrame(
                {
                    "date": score_date,
                    "instrument": instruments,
                    "score": date_scores,
                }
            )
        )

    if not outputs:
        raise ValueError("No valid score dates were produced")
    return pd.concat(outputs, ignore_index=True)


def _chunks(sequence, size):
    for offset in range(0, len(sequence), size):
        yield sequence[offset : offset + size]


def _query_bar_chunk(table, start_date, end_date, instruments):
    """Materialize raw bars for only one small stock batch."""
    import dai

    columns = ", ".join(["date", "instrument", *FEATURES])
    return dai.query(
        f"SELECT {columns} FROM {table} ORDER BY instrument, date",
        filters={
            "date": [start_date, end_date],
            "instrument": list(instruments),
        },
    ).df()


def predict_streaming(
    model,
    table,
    mean,
    std,
    universe,
    start_date,
    end_date,
    device,
    chunk_size=INFERENCE_CHUNK_SIZE,
):
    """Low-memory inference while preserving full daily cross attention.

    Raw 1m histories are queried and temporally encoded by stock chunk. Only
    D_MODEL-dimensional stock states survive each chunk. Cross-sectional
    attention runs once per date after states from all chunks are collected.
    """
    universe = universe.copy()
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["instrument"] = universe["instrument"].astype(str)
    requested_dates = pd.DatetimeIndex(np.sort(universe["date"].unique()))
    instruments = np.sort(universe["instrument"].unique())
    allowed_by_date = {
        pd.Timestamp(date): set(frame["instrument"])
        for date, frame in universe.groupby("date", sort=True)
    }
    states_by_date = {
        pd.Timestamp(date): {"instrument": [], "state": []}
        for date in requested_dates
    }
    buffer_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=90)
    ).strftime("%Y-%m-%d")
    query_end = pd.Timestamp(end_date).strftime("%Y-%m-%d 23:59:59")
    amp_enabled = device.type == "cuda"

    for chunk_number, chunk in enumerate(
        _chunks(instruments, chunk_size), start=1
    ):
        raw = _query_bar_chunk(table, buffer_start, query_end, chunk)
        if raw.empty:
            del raw
            continue
        canonical = to_canonical(raw)
        del raw
        daily, values, valid = build_daily_tensors(canonical, mean, std)
        del canonical
        market_dates = pd.DatetimeIndex(np.sort(daily["date"].unique()))
        lookup = pd.Series(
            np.arange(len(daily), dtype=np.int64),
            index=pd.MultiIndex.from_frame(
                daily[["date", "instrument"]].assign(
                    instrument=lambda frame: frame["instrument"].astype(str)
                )
            ),
        )

        for score_date in requested_dates:
            chunk_instruments, history_indices = _history_indices_for_date(
                daily, lookup, market_dates, score_date
            )
            if chunk_instruments is None:
                continue
            in_universe = np.fromiter(
                (
                    instrument in allowed_by_date[pd.Timestamp(score_date)]
                    for instrument in chunk_instruments
                ),
                dtype=bool,
                count=len(chunk_instruments),
            )
            fresh = in_universe & (history_indices[:, -1] >= 0)
            if not fresh.any():
                continue
            selected = history_indices[fresh]
            day_valid = selected >= 0
            safe = np.where(day_valid, selected, 0)
            x = np.array(values[safe], copy=True)
            minute_valid = np.array(valid[safe], copy=True)
            minute_valid &= day_valid[..., None]
            x[~day_valid] = 0.0
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                state = model.encode_stock_state(
                    torch.from_numpy(x).to(device, non_blocking=True),
                    torch.from_numpy(~minute_valid).to(
                        device, non_blocking=True
                    ),
                    torch.from_numpy(~day_valid).to(
                        device, non_blocking=True
                    ),
                )
            bucket = states_by_date[pd.Timestamp(score_date)]
            bucket["instrument"].extend(chunk_instruments[fresh].tolist())
            bucket["state"].append(state.float().cpu().numpy())
            del x, minute_valid, state

        del daily, values, valid, lookup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"inference chunk {chunk_number}: "
            f"{min(chunk_number * chunk_size, len(instruments))}/"
            f"{len(instruments)} instruments"
        )

    outputs = []
    model.eval()
    for score_date in requested_dates:
        bucket = states_by_date[pd.Timestamp(score_date)]
        if not bucket["state"]:
            continue
        state = np.concatenate(bucket["state"], axis=0)
        order = np.argsort(np.asarray(bucket["instrument"], dtype=str))
        ordered_instruments = np.asarray(
            bucket["instrument"], dtype=str
        )[order]
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            scores = model.score_cross_section(
                torch.from_numpy(state[order]).to(device)
            )
        outputs.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp(score_date),
                    "instrument": ordered_instruments,
                    "score": scores.float().cpu().numpy(),
                }
            )
        )
        del state, scores
    if not outputs:
        raise ValueError("No valid score dates were produced")
    return pd.concat(outputs, ignore_index=True)


def _correlation_loss(scores, targets, eps=1e-8):
    scores = scores.float() - scores.float().mean()
    ranks = torch.argsort(torch.argsort(targets.float())).float()
    ranks = ranks - ranks.mean()
    denominator = (
        scores.square().sum().sqrt() * ranks.square().sum().sqrt()
    ).clamp_min(eps)
    return -(scores * ranks).sum() / denominator


def train_and_save(datasources, output_path=MODEL_PATH):
    """Train V5 from scratch for the platform's isolated private stage.

    Inputs are restricted to the six organizer-provided raw 1-minute fields.
    The next-day close return is used only as the training label.
    """
    import dai

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = _resolve_table(datasources)
    columns = ", ".join(["date", "instrument", *FEATURES])
    raw = dai.query(
        f"SELECT {columns} FROM {table} ORDER BY instrument, date",
        filters={"date": [TRAIN_START, TRAIN_END]},
    ).df()
    canonical = to_canonical(raw)
    mean = canonical[FEATURES].mean().astype("float32")
    std = (
        canonical[FEATURES].std().replace(0, 1).fillna(1).astype("float32")
    )

    close_daily = canonical.assign(
        trade_date=canonical["date"].dt.normalize()
    ).groupby(["instrument", "trade_date"], sort=True, observed=True).agg(
        raw_close=("close", "last")
    ).reset_index().rename(columns={"trade_date": "date"})
    daily, values, valid = build_daily_tensors(canonical, mean, std)
    daily = daily.merge(
        close_daily, on=["instrument", "date"], how="left", validate="one_to_one"
    ).sort_values(["date", "instrument"]).reset_index(drop=True)

    by_instrument = daily.groupby("instrument", sort=False)
    daily["target_date"] = by_instrument["date"].shift(-1)
    daily["target"] = (
        by_instrument["raw_close"].shift(-1) / daily["raw_close"] - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    target_lookup = daily.dropna(subset=["target", "target_date"]).set_index(
        ["target_date", "instrument"]
    )["target"]
    market_dates = pd.DatetimeIndex(np.sort(daily["date"].unique()))
    lookup = pd.Series(
        np.arange(len(daily), dtype=np.int64),
        index=pd.MultiIndex.from_frame(daily[["date", "instrument"]]),
    )

    model = HierarchicalMarketTransformerV5(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    amp_enabled = device.type == "cuda"
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for score_date in market_dates[1:]:
            instruments, history_indices = _history_indices_for_date(
                daily, lookup, market_dates, score_date
            )
            if instruments is None:
                continue
            target_index = pd.MultiIndex.from_arrays(
                [np.repeat(score_date, len(instruments)), instruments],
                names=["target_date", "instrument"],
            )
            target = target_lookup.reindex(target_index).to_numpy(np.float32)
            usable = np.isfinite(target) & (history_indices[:, -1] >= 0)
            if usable.sum() < 20:
                continue
            selected = history_indices[usable]
            day_valid = selected >= 0
            safe = np.where(day_valid, selected, 0)
            x = np.array(values[safe], copy=True)
            minute_valid = np.array(valid[safe], copy=True)
            minute_valid &= day_valid[..., None]
            x[~day_valid] = 0.0
            y = torch.from_numpy(target[usable]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                scores = model(
                    torch.from_numpy(x).to(device),
                    torch.from_numpy(~minute_valid).to(device),
                    torch.from_numpy(~day_valid).to(device),
                )
                huber = torch.nn.functional.huber_loss(
                    scores.float(), y.float(), delta=0.02
                ) / 0.02
                loss = 0.30 * huber + 0.50 * _correlation_loss(scores, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        print(
            f"epoch={epoch:02d} train_loss="
            f"{np.mean(epoch_losses) if epoch_losses else np.nan:.6f}"
        )

    checkpoint = {
        "model_version": "bar1m_hierarchical_v5_platform",
        "model_state": model.cpu().state_dict(),
        "feature_mean": mean.to_dict(),
        "feature_std": std.to_dict(),
        "seed": SEED,
        "config": {
            "features": FEATURES,
            "lookback_days": LOOKBACK_DAYS,
            "minutes_per_day": MINUTES_PER_DAY,
            "patch_size": PATCH_SIZE,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "intraday_layers": INTRADAY_LAYERS,
            "daily_layers": DAILY_LAYERS,
            "cross_layers": CROSS_LAYERS,
            "market_query_count": MARKET_QUERY_COUNT,
            "from_scratch": True,
            "handcrafted_input_features": False,
        },
    }
    return save_model(checkpoint, output_path)


def predict(datasources, start_date, end_date):
    """Inference implementation; returns date/instrument/score exactly."""
    import dai

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, _ = _load_model(device)
    table = _resolve_table(datasources)
    universe = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    result = predict_streaming(
        model=model,
        table=table,
        mean=mean,
        std=std,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        device=device,
    )
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    result["score"] = result["score"].astype("float32")
    result = result[["date", "instrument", "score"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)

    if list(result.columns) != ["date", "instrument", "score"]:
        raise AssertionError("Submission columns or order are invalid")
    if result.isna().any().any():
        raise AssertionError("Submission contains missing values")
    if result.duplicated(["date", "instrument"]).any():
        raise AssertionError("Submission contains duplicate date/instrument")
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    if result["date"].min() < requested_start or result["date"].max() > requested_end:
        raise AssertionError("Submission dates exceed the requested interval")
    return result
