"""BigAlpha 端到端赛道合规推理：20 个原始字段，无衍生特征。"""
import json
import os
from datetime import timedelta

import dai
import numpy as np
import pandas as pd
import torch
from torch import nn


RAW_FEATURES = [
    "adjust_factor", "open", "high", "low", "close", "volume", "amount", "deal_number",
    "ask_price1", "ask_price2", "ask_price3", "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3",
]


class RawSequenceGRU(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=0.10)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, sequence):
        output, _ = self.gru(sequence)
        return self.head(output[:, -1]).squeeze(1)


def _load_model():
    candidates = ["/home/aiuser/work/userlib/model.json", "/home/aiuser/work/model.json", "model.json"]
    path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
    if path is None:
        raise FileNotFoundError("未找到 model.json")
    with open(path, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)
    model = RawSequenceGRU()
    state_dict = {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, np.asarray(checkpoint["mean"], np.float32), np.asarray(checkpoint["std"], np.float32)


def _resolve_datasource(datasources):
    if isinstance(datasources, dict):
        for key in ("bar30m", "e2e_bar30m", "stock_bar30m"):
            if key in datasources:
                return datasources[key]
        return next(iter(datasources.values()))
    if isinstance(datasources, str):
        return datasources
    raise ValueError("无法解析 30 分钟数据源")


def _allowed_preprocess(raw, mean, std):
    transformed = np.sign(raw) * np.log1p(np.abs(raw))
    return np.clip((transformed - mean) / std, -10, 10).astype(np.float32)


def main(datasources, start_date, end_date):
    datasource = _resolve_datasource(datasources)
    start_timestamp = pd.Timestamp(start_date).normalize()
    end_timestamp = pd.Timestamp(end_date).normalize()
    query_start = (start_timestamp - timedelta(days=15)).strftime("%Y-%m-%d")
    query_end = end_timestamp.strftime("%Y-%m-%d 23:59:59")
    columns = ", ".join(["date", "instrument"] + RAW_FEATURES)
    data = dai.query(
        f"SELECT {columns} FROM {datasource} ORDER BY instrument, date",
        filters={"date": [query_start, query_end]},
    ).df()
    model, mean, std = _load_model()

    sequences, output_dates, output_instruments = [], [], []
    for instrument, group in data.groupby("instrument", sort=False):
        group = group.sort_values("date")
        raw = group[RAW_FEATURES].to_numpy(np.float32)
        features = _allowed_preprocess(raw, mean, std)
        trade_dates = pd.to_datetime(group["date"]).dt.normalize().to_numpy()
        last_indices = np.flatnonzero(np.r_[trade_dates[1:] != trade_dates[:-1], True])
        for end_index in last_indices:
            prediction_date = pd.Timestamp(trade_dates[end_index])
            if prediction_date < start_timestamp or prediction_date > end_timestamp or end_index + 1 < 40:
                continue
            sequences.append(features[end_index - 39 : end_index + 1])
            output_dates.append(prediction_date)
            output_instruments.append(instrument)

    if not sequences:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    predictions = []
    sequence_array = np.asarray(sequences, np.float32)
    with torch.no_grad():
        for start in range(0, len(sequence_array), 1024):
            predictions.append(model(torch.from_numpy(sequence_array[start : start + 1024])).numpy())
    result = pd.DataFrame(
        {"date": output_dates, "instrument": output_instruments, "score": np.concatenate(predictions)}
    )
    stock_pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    return (
        result.merge(stock_pool, on=["date", "instrument"], how="inner")
        .replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])[["date", "instrument", "score"]]
        .reset_index(drop=True)
    )
