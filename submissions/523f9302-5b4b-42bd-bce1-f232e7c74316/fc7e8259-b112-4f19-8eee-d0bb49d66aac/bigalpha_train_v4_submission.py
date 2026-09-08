# -*- coding: utf-8 -*-
"""BigAlpha 2026 proxy-BARRA-residual cross-sectional Transformer.

Only raw bar5m fields are used.  The only preprocessing is per-field
log1p for non-negative size fields plus a fixed StandardScaler fitted on the
training interval.  No rolling statistics, cross-field arithmetic, external
data, or external pretrained weights are used. V3 aligns optimization with
the competition: every optimizer batch is one trading-day cross section and
the target is next-day return residualized each day against causal style
proxies built from the same raw history. The public checkpoint is refit on
the complete 2019-2024 public training range.
"""
import argparse
import base64
import glob
import json
import os
import random
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, TensorDataset


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "bigalpha_weights_v4.json")
LOCAL_PATTERN = os.path.join(HERE, "local_data", "bigalpha_2026_e2e_bar5m_*_daily", "*.parquet")

TRAIN_START, TRAIN_END = "2019-01-01", "2023-12-31 23:59:59"
VAL_START, VAL_END = "2024-01-01", "2024-12-31 23:59:59"
FINAL_END = "2024-12-31 23:59:59"
SEQ_LEN = 128
EPOCHS, FINAL_EPOCHS, BATCH, LR, SEED = 5, 5, 512, 3e-4, 42
PATIENCE = 2
MAX_TRAIN_INSTRUMENTS = 1200
PROXY_COLS = [
    "log_price", "log_amount", "log_volume", "log_deals",
    "mom_5", "mom_20", "mom_60", "vol_20", "vol_60",
    "spread_proxy", "book_imbalance",
]

PRICE_COLS = [
    "open", "high", "low", "close",
    "bid_price1", "bid_price2", "bid_price3",
    "ask_price1", "ask_price2", "ask_price3",
]
SIZE_COLS = [
    "volume", "amount", "deal_number",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
]
FEATURE_COLS = PRICE_COLS + SIZE_COLS
MODEL_CFG = {
    "n_feat": len(FEATURE_COLS),
    "d_model": 128,
    "nhead": 8,
    "nlayers": 3,
    "dim_ff": 256,
    "seq_len": SEQ_LEN,
    "dropout": 0.12,
}


def _seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RawBarTransformer(nn.Module):
    def __init__(self, n_feat, d_model, nhead, nlayers, dim_ff, seq_len, dropout):
        super().__init__()
        self.patch_projection_fast = nn.Conv1d(
            n_feat, d_model, kernel_size=8, stride=4, padding=2
        )
        self.patch_projection_slow = nn.Conv1d(
            n_feat, d_model, kernel_size=32, stride=16, padding=8
        )
        fast_count = (seq_len + 2 * 2 - 8) // 4 + 1
        slow_count = (seq_len + 2 * 8 - 32) // 16 + 1
        self.patch_position = nn.Parameter(
            torch.zeros(1, fast_count + slow_count, d_model)
        )
        self.scale_embedding = nn.Parameter(torch.zeros(1, 2, d_model))
        self.variable_projection = nn.Linear(seq_len, d_model)
        self.variable_position = nn.Parameter(torch.zeros(1, n_feat, d_model))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        variable_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=nlayers)
        self.variable_encoder = nn.TransformerEncoder(variable_layer, num_layers=nlayers)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.patch_position, std=0.02)
        nn.init.normal_(self.scale_embedding, std=0.02)
        nn.init.normal_(self.variable_position, std=0.02)

    def forward(self, x):
        channels = x.transpose(1, 2)
        fast = self.patch_projection_fast(channels).transpose(1, 2)
        slow = self.patch_projection_slow(channels).transpose(1, 2)
        fast = fast + self.scale_embedding[:, :1]
        slow = slow + self.scale_embedding[:, 1:]
        temporal = torch.cat([fast, slow], dim=1)
        temporal = self.temporal_encoder(temporal + self.patch_position).mean(dim=1)
        variable = self.variable_projection(x.transpose(1, 2))
        variable = self.variable_encoder(variable + self.variable_position).mean(dim=1)
        weights = self.fusion_gate(torch.cat([temporal, variable], dim=-1))
        fused = weights[:, :1] * temporal + weights[:, 1:] * variable
        return self.head(fused).squeeze(-1)


class RawBarEnsemble(nn.Module):
    """Jointly trained small deep ensemble for ICIR and regime stability."""

    def __init__(self, model_cfg, members=3):
        super().__init__()
        self.models = nn.ModuleList(
            [RawBarTransformer(**model_cfg) for _ in range(members)]
        )

    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=0).mean(dim=0)


def _get_dai():
    try:
        import dai
        return dai
    except ImportError:
        from bigquant import dai
        return dai


def _table_from_datasources(datasources):
    return (
        datasources.get("bar5m")
        or datasources.get("data")
        or "bigalpha_2026_stock_bar5m"
    )


def pool(start_date, end_date):
    dai = _get_dai()
    frame = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments ORDER BY instrument",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    return sorted(frame["instrument"].astype(str).tolist())


def _date_only(value):
    return pd.Timestamp(value).normalize()


def _month_ranges(start_date, end_date):
    current = _date_only(start_date)
    final = _date_only(end_date)
    while current <= final:
        month_end = current + pd.offsets.MonthEnd(0)
        chunk_end = min(month_end, final)
        yield (
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d 23:59:59"),
        )
        current = chunk_end + pd.Timedelta(days=1)


def _load_local(start_date, end_date, instruments=None):
    start_day, end_day = _date_only(start_date), _date_only(end_date)
    frames = []
    selected = None if instruments is None else set(map(str, instruments))
    columns = ["date", "instrument_id"] + FEATURE_COLS
    for path in sorted(glob.glob(LOCAL_PATTERN)):
        stem = os.path.basename(path).replace(".parquet", "")
        try:
            day = pd.Timestamp(stem)
        except ValueError:
            continue
        if day < start_day or day > end_day:
            continue
        frame = pd.read_parquet(path, columns=columns)
        frame["instrument"] = frame["instrument_id"].astype(str).str.zfill(6)
        if selected is not None:
            frame = frame[frame["instrument"].isin(selected)]
        if frame.empty:
            continue
        for col in PRICE_COLS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame.loc[frame[col] <= 0, col] = np.nan
            frame[col] = frame[col] / 100.0
        frames.append(frame[["date", "instrument"] + FEATURE_COLS])
    if not frames:
        raise RuntimeError(f"no local rows for {start_date}..{end_date}")
    return pd.concat(frames, ignore_index=True)


def _load_dai(table, start_date, end_date, instruments=None):
    dai = _get_dai()
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table}"
    filters = {"date": [str(start_date), str(end_date)]}
    if instruments:
        filters["instrument"] = list(instruments)
    frame = dai.query(sql, filters=filters, compression=True).df()
    if frame.empty:
        raise RuntimeError(f"DAI returned no rows for {start_date}..{end_date}")
    return frame


def _clean_raw(frame):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype(str)
    for col in PRICE_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.loc[frame[col] <= 0, col] = np.nan
    for col in SIZE_COLS:
        raw = pd.to_numeric(frame[col], errors="coerce").clip(lower=0)
        frame[col] = np.log1p(raw)
    frame = frame.sort_values(["instrument", "date"]).reset_index(drop=True)
    frame["trade_day"] = frame["date"].dt.normalize()
    return frame


def _fit_stats(frame):
    values = frame[FEATURE_COLS].to_numpy(np.float64)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}


def _residual_rank_targets(target_frame):
    """Daily ridge residuals after robust standardization of style proxies."""
    outputs = np.zeros(len(target_frame), dtype=np.float32)
    for _, group in target_frame.groupby("date", sort=False):
        positions = group.index.to_numpy()
        y = group["raw_target"].to_numpy(np.float64)
        lo, hi = np.nanquantile(y, [0.01, 0.99])
        y = np.clip(y, lo, hi)
        features = []
        for col in PROXY_COLS:
            value = group[col].to_numpy(np.float64)
            finite = np.isfinite(value)
            fill = np.nanmedian(value[finite]) if finite.any() else 0.0
            value = np.where(finite, value, fill)
            qlo, qhi = np.quantile(value, [0.01, 0.99])
            value = np.clip(value, qlo, qhi)
            scale = np.std(value)
            features.append((value - np.mean(value)) / (scale + 1e-6))
        x = np.column_stack([np.ones(len(group)), *features])
        penalty = np.eye(x.shape[1], dtype=np.float64) * 0.05
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        residual = y - x @ beta
        # Spearman training target after the same daily residualization idea
        # used by the competition evaluator.
        ranks = pd.Series(residual).rank(pct=True).to_numpy(np.float64)
        outputs[positions] = ((ranks - 0.5) * np.sqrt(12.0)).astype(np.float32)
    return outputs


def _build_windows(frame, start_date, end_date, mode, stats):
    start_day, end_day = _date_only(start_date), _date_only(end_date)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    xs, raw_targets, proxy_rows, keys = [], [], [], []
    for instrument, sub in frame.groupby("instrument", sort=False):
        sub = sub.sort_values("date")
        raw = sub[FEATURE_COLS].to_numpy(np.float32)
        raw = np.where(np.isfinite(raw), raw, mean[None, :])
        values = (raw - mean[None, :]) / std[None, :]
        days = sub["trade_day"].to_numpy()
        closes = sub["close"].to_numpy(np.float64)
        day_ends = np.flatnonzero(np.append(days[1:] != days[:-1], True))
        day_close = closes[day_ends]
        day_values = days[day_ends]
        day_returns = np.full(len(day_close), np.nan, dtype=np.float64)
        day_returns[1:] = day_close[1:] / day_close[:-1] - 1.0
        for index, position in enumerate(day_ends):
            sample_day = pd.Timestamp(day_values[index]).normalize()
            if sample_day < start_day or sample_day > end_day:
                continue
            if position + 1 < SEQ_LEN:
                continue
            if mode == "train":
                if index + 1 >= len(day_ends):
                    continue
                current_close, next_close = day_close[index], day_close[index + 1]
                if not np.isfinite(current_close) or not np.isfinite(next_close) or current_close <= 0:
                    continue
                target = next_close / current_close - 1.0
                if not np.isfinite(target):
                    continue
                raw_targets.append(float(target))
                row = sub.iloc[position]

                def momentum(lookback):
                    if index < lookback or day_close[index - lookback] <= 0:
                        return np.nan
                    return np.log(day_close[index] / day_close[index - lookback])

                def volatility(lookback):
                    start = max(1, index - lookback + 1)
                    values_ = day_returns[start : index + 1]
                    return (
                        float(np.nanstd(values_, ddof=1))
                        if np.isfinite(values_).sum() >= 5 else np.nan
                    )

                close = float(row["close"])
                bid = float(row["bid_price1"])
                ask = float(row["ask_price1"])
                bid_depth = sum(float(row[col]) for col in [
                    "bid_volume1", "bid_volume2", "bid_volume3"
                ])
                ask_depth = sum(float(row[col]) for col in [
                    "ask_volume1", "ask_volume2", "ask_volume3"
                ])
                proxy_rows.append({
                    "log_price": np.log(max(close, 1e-6)),
                    "log_amount": float(row["amount"]),
                    "log_volume": float(row["volume"]),
                    "log_deals": float(row["deal_number"]),
                    "mom_5": momentum(5),
                    "mom_20": momentum(20),
                    "mom_60": momentum(60),
                    "vol_20": volatility(20),
                    "vol_60": volatility(60),
                    "spread_proxy": (ask - bid) / max(close, 1e-6),
                    "book_imbalance": (bid_depth - ask_depth)
                    / (abs(bid_depth) + abs(ask_depth) + 1e-6),
                })
            xs.append(values[position - SEQ_LEN + 1 : position + 1].astype(np.float16))
            keys.append((sample_day, instrument))
    if not xs:
        raise RuntimeError(f"no samples for {mode} {start_date}..{end_date}")
    x = np.stack(xs).astype(np.float16)
    index_frame = pd.DataFrame(keys, columns=["date", "instrument"])
    if mode != "train":
        return x, None, index_frame
    target_frame = index_frame.copy()
    target_frame["raw_target"] = np.asarray(raw_targets, dtype=np.float32)
    for col in PROXY_COLS:
        target_frame[col] = [row[col] for row in proxy_rows]
    target_frame["target"] = _residual_rank_targets(target_frame)
    return x, target_frame["target"].to_numpy(np.float32), index_frame


def build_dataset(table, start_date, end_date, mode, instruments, stats=None, prefer_local=False):
    t0 = time.time()
    buffer_start = (_date_only(start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    query_end = (_date_only(end_date) + pd.Timedelta(days=8)).strftime("%Y-%m-%d 23:59:59")
    if prefer_local:
        frame = _load_local(buffer_start, query_end, instruments)
    else:
        frame = _load_dai(table, buffer_start, query_end, instruments)
    frame = _clean_raw(frame)
    if stats is None:
        if mode != "train":
            raise ValueError("inference requires training-set statistics")
        stats = _fit_stats(frame[frame["trade_day"] <= _date_only(end_date)])
    x, y, index_frame = _build_windows(frame, start_date, end_date, mode, stats)
    print(f"{mode} samples={len(x)} elapsed={time.time() - t0:.1f}s", flush=True)
    return x, y, index_frame, stats


def save_model(checkpoint, path=MODEL_PATH):
    payload = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    tensors = {}
    for key, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu()
        array = tensor.contiguous().numpy()
        tensors[key] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "encoding": "base64",
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    payload["state_dict"] = tensors
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def load_model(path=MODEL_PATH, map_location="cpu"):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dict = {}
    for key, meta in payload["state_dict"].items():
        if meta.get("encoding") == "base64":
            raw = base64.b64decode(meta["data"])
            array = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).copy()
            tensor = torch.from_numpy(array.reshape(meta["shape"]))
        else:
            tensor = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
            tensor = tensor.reshape(meta["shape"])
        state_dict[key] = tensor.to(map_location)
    payload["state_dict"] = state_dict
    return payload


def _correlation(prediction, target):
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    return torch.sum(prediction * target) / torch.sqrt(
        torch.sum(prediction ** 2) * torch.sum(target ** 2) + 1e-8
    )


def _objective(prediction, target):
    return -_correlation(prediction, target) + 0.05 * torch.mean(
        (prediction - target) ** 2
    )


class DayBatchSampler(Sampler):
    """Yield one complete trading-day cross section per optimizer step."""

    def __init__(self, index_frame, shuffle, seed=SEED):
        self.groups = [
            group.index.to_numpy(np.int64).tolist()
            for _, group in index_frame.reset_index(drop=True).groupby("date", sort=True)
            if len(group) >= 100
        ]
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for index in order:
            yield self.groups[int(index)]

    def __len__(self):
        return len(self.groups)


def _day_loader(x, y, index_frame, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    )
    return DataLoader(
        dataset,
        batch_sampler=DayBatchSampler(index_frame, shuffle=shuffle),
        pin_memory=torch.cuda.is_available(),
    )


def _fit_model(x_train, y_train, train_index, x_val=None, y_val=None,
               val_index=None, epochs=EPOCHS, ensemble_members=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        RawBarTransformer(**MODEL_CFG)
        if ensemble_members == 1
        else RawBarEnsemble(MODEL_CFG, ensemble_members)
    ).to(device)
    print(
        f"device={device} parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    train_loader = _day_loader(x_train, y_train, train_index, True)
    val_loader = (
        _day_loader(x_val, y_val, val_index, False) if x_val is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    best_metric, best_state, stale = -float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        train_losses, train_ics = [], []
        for inputs, targets in train_loader:
            inputs = inputs.float().to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = _objective(prediction, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_ics.append(float(_correlation(prediction.detach(), targets).cpu()))

        if val_loader is None:
            metric = float(np.mean(train_ics))
            val_text = "refit"
        else:
            model.eval()
            val_ics = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    prediction = model(inputs.float().to(device, non_blocking=True))
                    val_ics.append(
                        float(_correlation(
                            prediction, targets.to(device, non_blocking=True)
                        ).cpu())
                    )
            val_mean = float(np.mean(val_ics))
            val_ir = val_mean / (float(np.std(val_ics, ddof=1)) + 1e-12)
            # Mean RankIC dominates; the small IR term favors stable checkpoints.
            metric = val_mean + 0.10 * val_ir
            val_text = f"val_ic={val_mean:.6f} val_ir={val_ir:.6f}"
        print(
            f"epoch={epoch + 1} loss={np.mean(train_losses):.6f} "
            f"train_ic={np.mean(train_ics):.6f} {val_text}",
            flush=True,
        )
        if val_loader is None:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metric = metric
        elif metric > best_metric:
            best_metric = metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, best_metric


def train_and_save(datasources=None, model_path=MODEL_PATH, prefer_local=False,
                   mode="final"):
    _seed_everything()
    datasources = datasources or {"bar5m": "bigalpha_2026_stock_bar5m"}
    table = _table_from_datasources(datasources)
    if prefer_local:
        # Local files already contain the historical CSI 1000 universe.  Do
        # not freeze membership to the first few days, otherwise later index
        # entrants disappear from training.
        instruments = None
    else:
        instruments = pool(TRAIN_START, TRAIN_END)
    if mode == "experiment":
        x_train, y_train, train_index, stats = build_dataset(
            table, TRAIN_START, TRAIN_END, "train", instruments, None, prefer_local
        )
        x_val, y_val, val_index, _ = build_dataset(
            table, VAL_START, VAL_END, "train", instruments, stats, prefer_local
        )
        best_state, best_metric = _fit_model(
            x_train, y_train, train_index, x_val, y_val, val_index, EPOCHS
        )
        ensemble_members = 1
        fit_end = TRAIN_END
    else:
        x_train, y_train, train_index, stats = build_dataset(
            table, TRAIN_START, FINAL_END, "train", instruments, None, prefer_local
        )
        best_state, best_metric = _fit_model(
            x_train, y_train, train_index, epochs=FINAL_EPOCHS,
            ensemble_members=1,
        )
        ensemble_members = 1
        fit_end = FINAL_END
    checkpoint = {
        "state_dict": best_state,
        "model_cfg": MODEL_CFG,
        "ensemble_members": ensemble_members,
        "feature_cols": FEATURE_COLS,
        "size_cols": SIZE_COLS,
        "seq_len": SEQ_LEN,
        "stats": stats,
        "target_mean": 0.0,
        "target_std": 1.0,
        "train_start": TRAIN_START,
        "train_end": fit_end,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "seed": SEED,
        "epochs": EPOCHS if mode == "experiment" else FINAL_EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "best_validation_metric": best_metric,
        "training_mode": mode,
        "source_table": table,
    }
    save_model(checkpoint, model_path)
    print(f"saved {model_path}", flush=True)
    return model_path


def predict(datasources, start_date, end_date, model_path=MODEL_PATH):
    table = _table_from_datasources(datasources)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_model(model_path, map_location=device)
    members = int(checkpoint.get("ensemble_members", 1))
    model = (
        RawBarTransformer(**checkpoint["model_cfg"])
        if members == 1
        else RawBarEnsemble(checkpoint["model_cfg"], members)
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    instruments = pool(start_date, end_date)
    monthly_frames = []
    for chunk_start, chunk_end in _month_ranges(start_date, end_date):
        x_test, _, index_frame, _ = build_dataset(
            table, chunk_start, chunk_end, "infer", instruments,
            checkpoint["stats"], False,
        )
        outputs = []
        with torch.no_grad():
            tensor = torch.from_numpy(x_test)
            for offset in range(0, len(tensor), BATCH):
                batch = tensor[offset : offset + BATCH].float().to(device)
                outputs.append(model(batch).cpu().numpy())
        index_frame["score"] = (
            np.concatenate(outputs).astype(np.float64) * checkpoint["target_std"]
            + checkpoint["target_mean"]
        )
        monthly_frames.append(index_frame)
        del x_test, tensor, outputs
    return pd.concat(monthly_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--mode", choices=["experiment", "final"], default="final")
    args = parser.parse_args()
    train_and_save(
        {"bar5m": "bigalpha_2026_stock_bar5m"},
        model_path=args.model_path,
        prefer_local=args.local,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
