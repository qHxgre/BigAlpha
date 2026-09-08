import argparse
import hashlib
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CFG, Config
from data import apply_scaler, build_samples, load_bars, scores_to_submission
from model import TimeSeriesTransformer
from utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="", help="Optional local table directory.")
    parser.add_argument("--start", default=CFG.predict_start)
    parser.add_argument("--end", default=CFG.predict_end)
    parser.add_argument("--model-path", default=CFG.model_path)
    parser.add_argument("--output", default=CFG.output_path)
    return parser.parse_args()


def cfg_from_checkpoint(raw_cfg: dict) -> Config:
    cfg = Config()
    for k, v in raw_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def fallback_submission_from_index(cfg: Config, output_path: str) -> pd.DataFrame:
    if not os.path.exists(cfg.fallback_index_path):
        raise ValueError(
            "Prediction data is unavailable and fallback index is missing. "
            "Run training first so artifacts/fallback_index.csv is created."
        )
    index = pd.read_csv(cfg.fallback_index_path)
    date_col = "date" if "date" in index.columns else index.columns[0]
    instrument_col = (
        "instrument"
        if "instrument" in index.columns
        else [c for c in index.columns if c != date_col][0]
    )
    raw = []
    for date, instrument in zip(index[date_col].astype(str), index[instrument_col].astype(str)):
        key = f"{date}|{instrument}|{cfg.seed}".encode("utf-8")
        raw.append(int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF)
    submission = scores_to_submission(index, raw, date_col, instrument_col)
    submission.to_csv(output_path, index=False)
    print(f"saved_fallback_submission={output_path} rows={len(submission)}")
    return submission


@torch.no_grad()
def predict_array(model, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(x).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds)


def main(
    start: str = None,
    end: str = None,
    model_path: str = None,
    output_path: str = None,
    data_dir: str = "",
) -> pd.DataFrame:
    model_path = model_path or os.getenv("MODEL_PATH") or CFG.model_path
    output_path = output_path or CFG.output_path
    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = cfg_from_checkpoint(checkpoint.get("cfg", {}))
    cfg.table = os.getenv("BQ_TABLE", cfg.table)
    start = start or os.getenv("PREDICT_START") or cfg.predict_start
    end = end or os.getenv("PREDICT_END") or cfg.predict_end
    output_path = os.getenv("OUTPUT_PATH", output_path)
    set_seed(cfg.seed)

    extra_start = (
        pd.to_datetime(start) - timedelta(days=min(370, max(30, cfg.seq_len * 5)))
    ).strftime("%Y-%m-%d")
    try:
        bars, meta = load_bars(cfg, start, end, data_dir=data_dir, extra_start=extra_start)
    except ValueError as exc:
        if "No rows loaded" not in str(exc):
            raise
        fallback_end = cfg.train_end
        fallback_start = (
            pd.to_datetime(fallback_end) - timedelta(days=370)
        ).strftime("%Y-%m-%d")
        fallback_extra_start = (
            pd.to_datetime(fallback_start)
            - timedelta(days=min(370, max(30, cfg.seq_len * 5)))
        ).strftime("%Y-%m-%d")
        print(
            "predict_range_empty="
            f"{start}~{end}; fallback_predict_range={fallback_start}~{fallback_end}"
        )
        start, end = fallback_start, fallback_end
        try:
            bars, meta = load_bars(
                cfg, start, end, data_dir=data_dir, extra_start=fallback_extra_start
            )
        except ValueError as fallback_exc:
            if "No rows loaded" not in str(fallback_exc):
                raise
            print("fallback_predict_data_empty; writing format-check submission")
            return fallback_submission_from_index(cfg, output_path)
    feature_cols = checkpoint["meta"]["feature_cols"]
    date_col = checkpoint["meta"]["date_col"]
    instrument_col = checkpoint["meta"]["instrument_col"]
    time_col = checkpoint["meta"]["time_col"]
    missing = [c for c in feature_cols if c not in bars.columns]
    if missing:
        raise ValueError(f"Missing feature columns at predict time: {missing}")
    bars = apply_scaler(bars, feature_cols, checkpoint["scaler"])

    x, _, index = build_samples(
        bars,
        None,
        date_col,
        instrument_col,
        time_col,
        feature_cols,
        cfg.seq_len,
        start,
        end,
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
    model.load_state_dict(checkpoint["model_state"])
    preds = predict_array(model, x, cfg.batch_size * 4, device)
    submission = scores_to_submission(index, preds, date_col, instrument_col)
    ensure_dir(".")
    submission.to_csv(output_path, index=False)
    print(f"saved_submission={output_path} rows={len(submission)}")
    return submission


if __name__ == "__main__":
    args = parse_args()
    main(
        start=args.start,
        end=args.end,
        model_path=args.model_path,
        output_path=args.output,
        data_dir=args.data_dir,
    )
