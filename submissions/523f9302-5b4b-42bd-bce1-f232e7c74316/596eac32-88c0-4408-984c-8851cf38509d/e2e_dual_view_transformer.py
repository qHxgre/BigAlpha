# -*- coding: utf-8 -*-
"""Raw-field dual-view Transformer for persistent microstructure alpha.

The network receives the same 38 organizer-provided raw fields as the baseline.
Inside the network, one stream represents persistent book/trading state while a
second represents within-day displacement and first differences.  A learned
gate combines both views before temporal attention.  The prediction head joins
the closing state with learned full-day attention pooling.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import dai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import e2e_microstructure_transformer as base


TRAIN_TABLE = base.TRAIN_TABLE
EXPOSURE_TABLE = base.EXPOSURE_TABLE
INSTRUMENT_TABLE = base.INSTRUMENT_TABLE
TRAIN_START = base.TRAIN_START
TRAIN_END = base.TRAIN_END
SEQ_LEN = base.SEQ_LEN
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LEARNING_RATE = base.LEARNING_RATE
WEIGHT_DECAY = base.WEIGHT_DECAY
SEED = base.SEED
FEATURE_COLS = base.FEATURE_COLS
STYLE_COLS = base.STYLE_COLS
INDUSTRY_COLS = base.INDUSTRY_COLS
ModelConfig = base.ModelConfig
MODEL_CONFIG = ModelConfig()

set_deterministic = base.set_deterministic
_pool = base._pool
_pure_forward_labels = base._pure_forward_labels
_transform_raw = base._transform_raw
build_dataset = base.build_dataset


class RawMicrostructureTransformer(nn.Module):
    """Learned persistent-state and displacement views of raw intraday data."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.level_projection = nn.Linear(cfg.n_feat, cfg.d_model)
        self.path_projection = nn.Linear(cfg.n_feat * 2, cfg.d_model)
        self.view_gate = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.position = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
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
        self.pool_score = nn.Linear(cfg.d_model, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        delta = torch.diff(x, dim=1, prepend=x[:, :1])
        level = self.level_projection(x)
        path = self.path_projection(torch.cat([centered, delta], dim=-1))
        gate = torch.sigmoid(self.view_gate(torch.cat([level, path], dim=-1)))
        h = self.input_norm(gate * level + (1.0 - gate) * path)
        h = self.encoder(h + self.position)
        weights = torch.softmax(self.pool_score(h).squeeze(-1), dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        closing = h[:, -1]
        return self.output(torch.cat([closing, pooled], dim=-1)).squeeze(-1)


def model_parameter_count(cfg: ModelConfig = MODEL_CONFIG) -> int:
    return sum(p.numel() for p in RawMicrostructureTransformer(cfg).parameters())


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
        "architecture": "raw_dual_view_level_path_closing_attention",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def load_checkpoint(
    model_path: str,
    device: torch.device | str = "cpu",
) -> tuple[RawMicrostructureTransformer, tuple[np.ndarray, np.ndarray], dict]:
    with open(model_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg = ModelConfig(**payload["model_config"])
    network = RawMicrostructureTransformer(cfg).to(device)
    state = {}
    for name, meta in payload["state_dict"].items():
        state[name] = torch.tensor(
            meta["data"], dtype=getattr(torch, meta["dtype"])
        ).reshape(meta["shape"]).to(device)
    network.load_state_dict(state)
    stats = (
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
    )
    return network, stats, payload


def predict(
    datasources: dict[str, str],
    start_date: str,
    end_date: str,
    model_path: str,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network, stats, _ = load_checkpoint(model_path, device)
    network.eval()
    table = datasources["bar5m"]
    instruments = _pool(table, start_date, end_date)
    x, _, keys, _ = build_dataset(
        table, start_date, end_date, "infer", instruments, stats
    )
    scores = []
    tensor = torch.from_numpy(x)
    with torch.no_grad():
        for start in range(0, len(tensor), BATCH_SIZE):
            xb = tensor[start : start + BATCH_SIZE].to(device, non_blocking=True)
            scores.append(network(xb).cpu().numpy())
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
