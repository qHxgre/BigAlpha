"""Reproducible training entry point included with the competition package."""

from __future__ import annotations

import json
import os
from pathlib import Path

from e2e_train import train_and_save, train_ensemble_and_save


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "model_config.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(datasources=None, start_date=None, end_date=None):
    """Train from injected datasources or from the local tensor cache.

    The platform can pass a private training interval through ``start_date`` and
    ``end_date``.  For local/public reproduction the dates in model_config.json
    are used.
    """
    cfg = load_config()
    train_start = str(start_date or cfg["train_start"])
    train_end = str(end_date or cfg["train_end"])
    tensor_dir = os.environ.get("BIGALPHA_TENSOR_DIR", str(BASE_DIR / "data" / "tensor_cache"))
    cache_dir = os.environ.get("BIGALPHA_CACHE_DIR", str(BASE_DIR / "data" / "e2e_cache"))
    common = {
        "datasources": datasources,
        "model_path": str(BASE_DIR / "e2e_model.json"),
        "cache_dir": cache_dir,
        "tensor_dir": tensor_dir,
        "train_start": train_start,
        "train_end": train_end,
        "preload_workers": int(cfg.get("preload_workers", 0)),
    }
    if cfg["arch"] == "ensemble":
        return train_ensemble_and_save(
            branch_configs=cfg["branches"],
            alpha_v1=float(cfg["alpha_v1"]),
            **common,
        )
    return train_and_save(
        **common,
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        seed=int(cfg["seed"]),
        arch=cfg["arch"],
        objective=cfg["objective"],
        scheduler_name=cfg.get("scheduler", "cosine"),
    )


if __name__ == "__main__":
    main()
