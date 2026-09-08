# -*- coding: utf-8 -*-
"""BigAlpha v1 共享定义：模型、JSON 权重加载和线上流式推理。"""

import gc
import base64
import json
import math
import os

import dai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

BATCH = 64
INSTRUMENT_CHUNK_SIZE = 100
DATE_CHUNK_DAYS = 20


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    """读取 dtype/shape/Base64 data JSON，并兼容官方扁平 data 格式。"""
    with open(model_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tensor_metadata = payload.pop("state_dict")
    state_dict = {}
    while tensor_metadata:
        name, metadata = tensor_metadata.popitem()
        if "data_b64" in metadata:
            array = np.frombuffer(
                base64.b64decode(metadata["data_b64"]),
                dtype=np.dtype(metadata["dtype"]),
            ).copy()
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.tensor(
                metadata["data"], dtype=getattr(torch, metadata["dtype"])
            )
        state_dict[name] = tensor.reshape(metadata["shape"]).to(map_location)
    payload["state_dict"] = state_dict
    return payload


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, tensor):
        batch, length, hidden = tensor.shape
        return tensor.view(
            batch, length, self.num_heads, self.head_size
        ).transpose(1, 2)

    def forward(self, hidden, valid):
        query = self._split_heads(self.query(hidden))
        key = self._split_heads(self.key(hidden))
        value = self._split_heads(self.value(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_size
        )
        scores = scores.masked_fill(
            ~valid[:, None, None, :], torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        return context.view(hidden.shape)


class SelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden, residual):
        return self.LayerNorm(self.dense(hidden) + residual)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        setattr(self, "self", SelfAttention(hidden_size, num_heads))
        self.output = SelfOutput(hidden_size)

    def forward(self, hidden, valid):
        return self.output(getattr(self, "self")(hidden, valid), hidden)


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden):
        return F.gelu(self.dense(hidden))


class TransformerOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden, residual):
        return self.LayerNorm(self.dense(hidden) + residual)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = Attention(hidden_size, num_heads)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = TransformerOutput(hidden_size, intermediate_size)

    def forward(self, hidden, valid):
        attention = self.attention(hidden, valid)
        return self.output(self.intermediate(attention), attention)


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, num_layers):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerLayer(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden, valid):
        for layer in self.layer:
            hidden = layer(hidden, valid)
        return hidden


class BigAlphaV1Model(nn.Module):
    def __init__(
        self,
        num_variables=8,
        lookback_days=5,
        bars_per_day=240,
        patch_len=8,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=512,
        temporal_layers=4,
        dropout=0.1,
        **_,
    ):
        super().__init__()
        self.num_variables = int(num_variables)
        self.lookback_days = int(lookback_days)
        self.bars_per_day = int(bars_per_day)
        self.patch_len = int(patch_len)
        self.sequence_length = self.lookback_days * self.bars_per_day
        num_patches = self.sequence_length // self.patch_len

        self.patch_embedding = nn.Sequential(
            nn.Linear(self.patch_len * self.num_variables * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, num_patches, hidden_size)
        )
        self.score_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.encoder = Encoder(
            hidden_size,
            int(num_attention_heads),
            int(intermediate_size),
            int(temporal_layers),
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )

    def _patches(self, tensor):
        return (
            tensor.unfold(1, self.patch_len, self.patch_len)
            .permute(0, 1, 3, 2)
            .flatten(-2)
        )

    def forward(self, values, observed):
        value_patches = self._patches(values)
        mask_patches = self._patches(observed.to(values.dtype))
        hidden = self.patch_embedding(
            torch.cat((value_patches, mask_patches), dim=-1)
        )
        valid = mask_patches.bool().any(-1)
        hidden = hidden + self.position_embedding
        score_valid = torch.ones(
            len(hidden), 1, dtype=torch.bool, device=hidden.device
        )
        hidden = torch.cat(
            (self.score_token.expand(len(hidden), -1, -1), hidden), dim=1
        )
        hidden = self.encoder(hidden, torch.cat((score_valid, valid), dim=1))
        return self.score_head(self.final_norm(hidden[:, 0])).squeeze(-1).float()


def pool(sd, ed):
    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return frame["instrument"].astype(str).tolist()


def _resolve_bar1m_table(datasources):
    if datasources is None:
        return "bigalpha_2026_stock_bar1m"
    if "bar1m" not in datasources:
        raise KeyError("该模型要求 datasources['bar1m']")
    return datasources["bar1m"]


def _slot_index(dates):
    dates = pd.DatetimeIndex(dates)
    minutes = dates.hour.to_numpy() * 60 + dates.minute.to_numpy()
    slots = np.full(len(dates), -1, dtype=np.int16)
    morning = (minutes >= 571) & (minutes <= 690)
    afternoon = (minutes >= 781) & (minutes <= 900)
    slots[morning] = minutes[morning] - 571
    slots[afternoon] = 120 + minutes[afternoon] - 781
    return slots


def _query_members(start_date, end_date):
    frame = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [str(start_date), str(end_date)]},
    ).df()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    return frame.drop_duplicates(["date", "instrument"])


def _query_calendar(start_date, end_date):
    frame = dai.query(
        "SELECT DISTINCT date FROM bigalpha_2026_instruments",
        filters={"date": [str(start_date), str(end_date)]},
    ).df()
    dates = pd.to_datetime(frame["date"]).dt.normalize().unique()
    return pd.DatetimeIndex(dates).sort_values()


def _query_bars(
    table, start_date, end_date, instruments, feature_columns
):
    query_features = [
        column for column in feature_columns if column != "adjust_factor"
    ]
    columns = ["date", "instrument", *query_features]
    sql = f"SELECT {', '.join(columns)} FROM {table} ORDER BY instrument, date"
    frame = dai.query(
        sql,
        filters={
            "date": [str(start_date), str(end_date)],
            "instrument": list(instruments),
        },
        compression=True,
    ).df()
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["instrument"] = frame["instrument"].astype(str)
        if "adjust_factor" in feature_columns:
            frame["adjust_factor"] = np.nan
    return frame


def _prepare_cube(
    frame, instruments, calendar, feature_columns, mean, std, bars_per_day
):
    shape = (
        len(instruments),
        len(calendar),
        bars_per_day,
        len(feature_columns),
    )
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    if frame.empty:
        return values, observed

    raw = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    mask = np.isfinite(raw)
    transformed = np.sign(raw) * np.log1p(np.abs(raw))
    transformed = np.where(mask, (transformed - mean) / std, 0.0)
    instrument_map = {
        instrument: index for index, instrument in enumerate(instruments)
    }
    instrument_index = frame["instrument"].map(instrument_map).to_numpy()
    day_index = calendar.get_indexer(
        pd.DatetimeIndex(frame["date"]).normalize()
    )
    slots = _slot_index(frame["date"])
    valid = (
        pd.notna(instrument_index)
        & (day_index >= 0)
        & (slots >= 0)
        & (slots < bars_per_day)
    )
    ii = instrument_index[valid].astype(np.int64)
    values[ii, day_index[valid], slots[valid]] = transformed[valid]
    observed[ii, day_index[valid], slots[valid]] = mask[valid]
    return values, observed


def _predict_chunk(
    model,
    table,
    members,
    calendar,
    stats,
    feature_columns,
    device,
    batch_size,
):
    mean, std = stats
    instruments = members["instrument"].drop_duplicates().tolist()
    first_target = members["date"].min()
    last_target = members["date"].max()
    first_position = int(calendar.get_indexer([first_target])[0])
    history_start = max(0, first_position - model.lookback_days + 1)
    last_position = int(calendar.get_indexer([last_target])[0])
    local_calendar = calendar[history_start : last_position + 1]

    raw = _query_bars(
        table,
        local_calendar[0],
        local_calendar[-1] + pd.Timedelta(days=1),
        instruments,
        feature_columns,
    )
    values, observed = _prepare_cube(
        raw,
        instruments,
        local_calendar,
        feature_columns,
        mean,
        std,
        model.bars_per_day,
    )
    instrument_map = {
        instrument: index for index, instrument in enumerate(instruments)
    }
    target_instruments = (
        members["instrument"].map(instrument_map).to_numpy(np.int64)
    )
    target_days = local_calendar.get_indexer(
        pd.DatetimeIndex(members["date"])
    )
    offsets = np.arange(1 - model.lookback_days, 1, dtype=np.int64)
    scores = []
    with torch.no_grad():
        for start in range(0, len(members), batch_size):
            stop = min(start + batch_size, len(members))
            day_window = target_days[start:stop, None] + offsets[None, :]
            stock_window = target_instruments[start:stop, None]
            batch_values = values[stock_window, day_window].reshape(
                -1, model.sequence_length, len(feature_columns)
            )
            batch_observed = observed[stock_window, day_window].reshape(
                batch_values.shape
            )
            value_tensor = torch.from_numpy(batch_values).to(device)
            mask_tensor = torch.from_numpy(batch_observed).to(device)
            scores.append(model(value_tensor, mask_tensor).cpu().numpy())

    result = members[["date", "instrument"]].copy()
    result["score"] = np.concatenate(scores).astype(np.float32)
    del raw, values, observed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def predict_scores_streaming(
    model,
    table,
    sd,
    ed,
    instruments,
    stats,
    feature_cols,
    device,
    batch_size=BATCH,
    seq_len=None,
):
    if seq_len is not None and int(seq_len) != model.sequence_length:
        raise ValueError(
            f"权重 seq_len={seq_len} 与模型 {model.sequence_length} 不一致"
        )
    members = (
        _query_members(sd, ed)
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    members = members[
        members["instrument"].isin(set(instruments))
    ].reset_index(drop=True)
    if members.empty:
        return pd.DataFrame(columns=["date", "instrument", "score"])

    calendar = _query_calendar(pd.Timestamp(sd) - pd.Timedelta(days=45), ed)
    target_days = pd.DatetimeIndex(members["date"].unique()).sort_values()
    results = []
    for date_start in range(0, len(target_days), DATE_CHUNK_DAYS):
        block_days = target_days[
            date_start : date_start + DATE_CHUNK_DAYS
        ]
        block = members[members["date"].isin(block_days)]
        block_instruments = block["instrument"].drop_duplicates().tolist()
        for instrument_start in range(
            0, len(block_instruments), INSTRUMENT_CHUNK_SIZE
        ):
            selected = set(
                block_instruments[
                    instrument_start : instrument_start
                    + INSTRUMENT_CHUNK_SIZE
                ]
            )
            chunk_members = block[
                block["instrument"].isin(selected)
            ].reset_index(drop=True)
            results.append(
                _predict_chunk(
                    model,
                    table,
                    chunk_members,
                    calendar,
                    stats,
                    feature_cols,
                    device,
                    batch_size,
                )
            )
    result = pd.concat(results, ignore_index=True)
    return (
        result.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])[
            ["date", "instrument", "score"]
        ]
        .reset_index(drop=True)
    )
