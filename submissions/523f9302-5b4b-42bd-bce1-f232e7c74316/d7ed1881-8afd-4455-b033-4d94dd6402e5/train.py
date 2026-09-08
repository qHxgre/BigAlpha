import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import CFG, Config
from data import (
    ArrayDataset,
    apply_scaler,
    build_daily_labels,
    build_samples,
    fit_scaler,
    load_bars,
)
from model import TimeSeriesTransformer
from utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="", help="Optional local table directory.")
    parser.add_argument("--table", default=CFG.table)
    parser.add_argument("--train-start", default=CFG.train_start)
    parser.add_argument("--train-end", default=CFG.train_end)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--model-path", default=CFG.model_path)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.table = args.table
    cfg.train_start = args.train_start
    cfg.train_end = args.train_end
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.model_path = args.model_path
    return cfg


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        losses.append(float(criterion(pred, y).detach().cpu()))
    return float(np.mean(losses))


def main(cfg: Config = None, data_dir: str = "") -> None:
    cfg = cfg or CFG
    cfg.table = os.getenv("BQ_TABLE", cfg.table)
    cfg.train_start = os.getenv("TRAIN_START", cfg.train_start)
    cfg.train_end = os.getenv("TRAIN_END", cfg.train_end)
    cfg.model_path = os.getenv("MODEL_PATH", cfg.model_path)
    set_seed(cfg.seed)
    ensure_dir(cfg.artifacts_dir)

    bars, meta = load_bars(cfg, cfg.train_start, cfg.train_end, data_dir=data_dir)
    date_col = meta["date_col"]
    instrument_col = meta["instrument_col"]
    time_col = meta["time_col"]
    feature_cols = meta["feature_cols"]

    labels = build_daily_labels(bars, date_col, instrument_col, time_col)
    scaler = fit_scaler(bars, feature_cols)
    bars = apply_scaler(bars, feature_cols, scaler)
    x, y, index = build_samples(
        bars,
        labels,
        date_col,
        instrument_col,
        time_col,
        feature_cols,
        cfg.seq_len,
        cfg.train_start,
        cfg.train_end,
    )
    index.to_csv(cfg.fallback_index_path, index=False)

    dates = np.array(sorted(index[date_col].unique()))
    split_at = max(1, int(len(dates) * (1.0 - cfg.valid_ratio)))
    valid_dates = set(dates[split_at:])
    valid_mask = index[date_col].isin(valid_dates).to_numpy()
    train_mask = ~valid_mask

    train_ds = ArrayDataset(x[train_mask], y[train_mask])
    valid_ds = ArrayDataset(x[valid_mask], y[valid_mask])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        n_features=len(feature_cols),
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=0.01)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg.grad_clip
        )
        valid_loss = evaluate(model, valid_loader, criterion, device)
        print(f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "model_state": best_state or model.state_dict(),
        "cfg": asdict(cfg),
        "meta": meta,
        "scaler": scaler,
        "best_valid_loss": best_loss,
    }
    torch.save(checkpoint, cfg.model_path)
    save_json(scaler, cfg.scaler_path)
    save_json({"meta": meta, "best_valid_loss": best_loss}, cfg.meta_path)
    print(f"saved_model={cfg.model_path}")


if __name__ == "__main__":
    args = parse_args()
    main(make_config(args), data_dir=args.data_dir)
