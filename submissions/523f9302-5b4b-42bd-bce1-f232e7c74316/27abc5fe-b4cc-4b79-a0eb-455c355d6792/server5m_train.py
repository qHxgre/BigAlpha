"""五档本地训练命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from server5m_constants import TARGETS
from server5m_data import LocalDataset, build_cache, load_manifest
from server5m_losses import build_loss
from server5m_metrics import factor_metrics, score_from_output
from server5m_model import AttentionLOB
from server5m_optim import build_optimizer


def setup_logger(result_dir: Path) -> logging.Logger:
    logger = logging.getLogger("bigalpha_local.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(result_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_result_dir(config: dict) -> Path:
    root = Path(config.get("output_root", "training_results"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = root / f"{config['run_name']}_{timestamp}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def make_loader(
    dataset: Dataset,
    config: dict,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.get("num_workers", 4)) > 0,
        generator=generator,
    )


def loss_function(task: str, name: str, num_classes: int):
    return build_loss(name, task, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    dataset: LocalDataset,
    loss_fn,
    device: torch.device,
    amp: bool,
) -> tuple[dict, np.ndarray]:
    model.eval()
    outputs = []
    total_loss = 0.0
    total_samples = 0
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            output = model(features)
            loss = loss_fn(
                output,
                target if dataset.task == "regression" else target.long(),
            )
        outputs.append(output.float().cpu().numpy())
        total_loss += float(loss.detach()) * len(features)
        total_samples += len(features)
    output_array = np.concatenate(outputs)
    metrics = factor_metrics(
        output_array,
        dataset.raw_return,
        dataset.date_ids,
        dataset.task,
    )
    metrics["loss"] = total_loss / max(total_samples, 1)
    if dataset.task == "classification":
        prediction = output_array.argmax(axis=1)
        metrics["accuracy"] = float((prediction == dataset.target).mean())
    return metrics, output_array


class InferenceDataset(Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.array[index], copy=True))


@torch.no_grad()
def export_factors(
    model: nn.Module,
    cache_dir: Path,
    result_dir: Path,
    config: dict,
    task: str,
    device: torch.device,
    amp: bool,
):
    features = np.load(cache_dir / "test_X.npy", mmap_mode="r")
    date_ids = np.load(cache_dir / "test_date_ids.npy", mmap_mode="r")
    stocks = np.load(cache_dir / "test_stocks.npy", mmap_mode="r")
    dataset = InferenceDataset(features)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    outputs = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            outputs.append(model(batch).float().cpu().numpy())
    score = score_from_output(np.concatenate(outputs), task)
    manifest = load_manifest(cache_dir)
    dates = np.asarray(manifest["dates"], dtype=object)
    factors = pd.DataFrame(
        {
            "date": dates[np.asarray(date_ids)],
            "instrument": np.asarray(stocks).astype(str),
            "score": score,
        }
    )
    factors.to_parquet(result_dir / "factors.parquet", index=False)


def train(config: dict) -> Path:
    result_dir = make_result_dir(config)
    logger = setup_logger(result_dir)
    (result_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_config = config["data"]
    cache_dir = build_cache(
        data_config["source"],
        data_config["cache_dir"],
        force=bool(data_config.get("force_rebuild", False)),
        feature_mode=data_config.get("feature_mode", "log_standard"),
    )
    target_name = config["target"]
    task, _, output_dim = TARGETS[target_name]
    smoke_samples = config.get("max_samples")
    purge_boundary = bool(config.get("purge_boundary", False))
    train_data = LocalDataset(
        cache_dir, "train", target_name, smoke_samples, purge_boundary
    )
    val_data = LocalDataset(
        cache_dir,
        "val",
        target_name,
        None if smoke_samples is None else max(1024, int(smoke_samples) // 2),
        purge_boundary,
    )
    test_data = LocalDataset(
        cache_dir,
        "test",
        target_name,
        None if smoke_samples is None else max(1024, int(smoke_samples) // 2),
        False,
    )
    if not len(train_data) or not len(val_data):
        raise ValueError("Training and validation datasets must be non-empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    model = AttentionLOB(
        output_dim=output_dim, dropout=float(config.get("dropout", 0.1))
    ).to(device)
    optimizer = build_optimizer(model, config)
    loss_name = config.get(
        "loss", "mse" if task == "regression" else "celoss"
    )
    loss_fn = loss_function(task, loss_name, output_dim).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = make_loader(train_data, config, True, seed)
    val_loader = make_loader(val_data, config, False, seed)
    test_loader = make_loader(test_data, config, False, seed)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("result_dir=%s", result_dir)
    logger.info(
        "device=%s target=%s task=%s loss=%s optimizer=%s parameters=%d",
        device,
        target_name,
        task,
        loss_name,
        config["optimizer"],
        parameter_count,
    )
    logger.info(
        "samples train=%d val=%d test=%d batch=%d",
        len(train_data),
        len(val_data),
        len(test_data),
        int(config["batch_size"]),
    )
    if not 100_000 <= parameter_count <= 100_000_000:
        raise ValueError(f"Model parameter count {parameter_count} violates competition limits")

    best_ic = -np.inf
    best_epoch = -1
    stale_epochs = 0
    history_path = result_dir / "metrics.jsonl"
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                output = model(features)
                loss = loss_fn(
                    output,
                    target if task == "regression" else target.long(),
                )
            scaler.scale(loss).backward()
            if config.get("grad_clip") is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["grad_clip"])
                )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(features)
            total_samples += len(features)

        val_metrics, _ = evaluate(
            model, val_loader, val_data, loss_fn, device, amp
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_ic=%.6f "
            "val_ic_ir=%.4f val_ls_sharpe=%.4f",
            epoch,
            row["train_loss"],
            val_metrics["loss"],
            val_metrics["ic"],
            val_metrics["ic_ir"],
            val_metrics["long_short_sharpe"],
        )

        if val_metrics["ic"] > best_ic:
            best_ic = val_metrics["ic"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": {"output_dim": output_dim, "dropout": config.get("dropout", 0.1)},
                    "target": target_name,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "feature_columns": load_manifest(cache_dir)["feature_columns"],
                },
                result_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
        patience = int(config.get("early_stop_patience", 10))
        if patience > 0 and stale_epochs >= patience:
            logger.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    checkpoint = torch.load(
        result_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, _ = evaluate(model, test_loader, test_data, loss_fn, device, amp)
    export_factors(model, Path(cache_dir), result_dir, config, task, device, amp)
    summary = {
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val_ic": best_ic,
        "test_metrics": test_metrics,
        "result_dir": str(result_dir.resolve()),
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("test_metrics=%s", json.dumps(test_metrics, ensure_ascii=False))
    logger.info("factors=%s", result_dir / "factors.parquet")
    logger.info("training_complete")
    return result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(train(config))


if __name__ == "__main__":
    main()
