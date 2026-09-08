"""共享 AttnLOB 主干的六头多目标严格训练。"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from server5m_data import MultiTargetDataset
from server5m_losses import OrdinalCELoss
from server5m_metrics import daily_ic
from server5m_model import (
    FusionMultiTaskAttentionLOB,
    MultiTaskAttentionLOB,
    SmallMultiTaskAttentionLOB,
    SmallTCNFusionMultiTaskLOB,
    TCNFusionMultiTaskLOB,
    TransformerFusionMultiTaskLOB,
    PureTransformerMultiTaskLOB,
)
from server5m_optim import build_optimizer
from server5m_train import setup_logger


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DateBatchSampler(Sampler[list[int]]):
    """每个 batch 是一个完整交易日，供截面相关损失使用。"""

    def __init__(self, dataset, shuffle: bool, seed: int, half_life=None):
        self.batches = [
            np.flatnonzero(dataset.date_ids == date_id).tolist()
            for date_id in np.unique(dataset.date_ids)
        ]
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.probability = None
        if half_life is not None:
            half_life = float(half_life)
            if half_life <= 0:
                raise ValueError("drift_half_life_days must be positive")
            age = np.arange(len(self.batches)-1, -1, -1, dtype=np.float64)
            weight = np.exp2(-age/half_life)
            self.probability = torch.as_tensor(
                weight/weight.sum(), dtype=torch.double
            )

    def __iter__(self):
        if self.probability is not None:
            order = torch.multinomial(
                self.probability, len(self.batches), replacement=True,
                generator=self.generator,
            ).tolist()
        elif self.shuffle:
            order = torch.randperm(len(self.batches), generator=self.generator).tolist()
        else:
            order = range(len(self.batches))
        for index in order:
            yield self.batches[index]

    def __len__(self):
        return len(self.batches)


def _loader(dataset, config: dict, shuffle: bool, seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 4))
    common = dict(
        dataset=dataset,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    half_life = config.get("drift_half_life_days") if shuffle else None
    if config.get("date_batches", False) or half_life is not None:
        return DataLoader(
            **common,
            batch_sampler=DateBatchSampler(
                dataset, shuffle, seed, half_life=half_life
            ),
        )
    return DataLoader(
        **common,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _expected(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability @ np.arange(logits.shape[1], dtype=np.float64)


def _daily_zscore(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for date_id in np.unique(dates):
        mask = dates == date_id
        section = values[mask]
        result[mask] = (
            section - section.mean(axis=0, keepdims=True)
        ) / np.maximum(section.std(axis=0, keepdims=True), 1e-8)
    return result


def candidate_scores(
    regression: np.ndarray,
    classification: np.ndarray,
    dates: np.ndarray,
) -> dict[str, np.ndarray]:
    cls = np.column_stack(
        [_expected(classification[:, index]) for index in range(3)]
    )
    normalized = _daily_zscore(np.column_stack((regression, cls)), dates)
    return {
        "ret_day1": regression[:, 0],
        "reg_blend_532": normalized[:, 0:3] @ np.array([0.5, 0.3, 0.2]),
        "cls_day1": cls[:, 0],
        "cls_blend_532": normalized[:, 3:6] @ np.array([0.5, 0.3, 0.2]),
        "reg_cls_blend_532": normalized
        @ np.array([0.30, 0.18, 0.12, 0.20, 0.12, 0.08]),
        "all_blend_weighted": normalized
        @ np.array([0.25, 0.15, 0.10, 0.25, 0.15, 0.10]),
        "all_blend_equal": normalized.mean(axis=1),
    }


def _metrics(scores, raw_return, dates):
    result = {}
    for name, score in scores.items():
        values = daily_ic(score, raw_return, dates)
        result[name] = {
            "ic": float(values.mean()),
            "ic_std": float(values.std(ddof=1)),
            "ic_ir": float(values.mean() / values.std(ddof=1)),
            "days": int(len(values)),
        }
    return result


def _correlation_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    score = score.float() - score.float().mean()
    target = target.float() - target.float().mean()
    denominator = score.square().sum().sqrt() * target.square().sum().sqrt()
    return 1.0 - (score * target).sum() / denominator.clamp_min(1e-8)


def _loss(
    output,
    ret_target,
    cls_target,
    rank_target,
    losses,
    weights,
    regression_target: str = "return_zscore",
):
    if regression_target == "return_zscore":
        continuous_target = ret_target
    elif regression_target == "rank":
        continuous_target = rank_target
    else:
        raise ValueError(
            "regression_target must be 'return_zscore' or 'rank', "
            f"got {regression_target!r}"
        )
    value = sum(
        weights["regression"][index]
        * losses["mse"](
            output["regression"][:, index], continuous_target[:, index]
        )
        for index in range(3)
    )
    value += sum(
        weights["classification"][index]
        * losses["ordinal"](
            output["classification"][:, index], cls_target[:, index]
        )
        for index in range(3)
    )
    value = value / weights["total"]
    if weights["rank_loss_weight"] > 0:
        class_values = torch.arange(
            5, device=output["classification"].device, dtype=torch.float32
        )
        rank_loss = sum(
            weights["rank"][index]
            * _correlation_loss(
                torch.softmax(
                    output["classification"][:, index].float(), dim=1
                ) @ class_values,
                rank_target[:, index],
            )
            for index in range(3)
        ) / weights["rank"].sum()
        value = value + weights["rank_loss_weight"] * rank_loss
    return value


@torch.no_grad()
def evaluate(
    model, loader, dataset, losses, weights, device, amp,
    regression_target: str = "return_zscore",
):
    model.eval()
    regressions, classifications = [], []
    total_loss = 0.0
    total = 0
    for features, ret_target, cls_target, rank_target in loader:
        features = features.to(device, non_blocking=True)
        ret_target = ret_target.to(device, non_blocking=True)
        cls_target = cls_target.to(device, non_blocking=True)
        rank_target = rank_target.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            output = model(features)
            loss = _loss(
                output, ret_target, cls_target, rank_target, losses, weights,
                regression_target,
            )
        regressions.append(output["regression"].float().cpu().numpy())
        classifications.append(output["classification"].float().cpu().numpy())
        total_loss += float(loss) * len(features)
        total += len(features)
    regression = np.concatenate(regressions)
    classification = np.concatenate(classifications)
    metrics = _metrics(
        candidate_scores(regression, classification, dataset.date_ids),
        dataset.raw_return,
        dataset.date_ids,
    )
    return total_loss / total, metrics, regression, classification


def train_multitask(config: dict) -> Path:
    root = Path(config.get("output_root", "training_results/multitask"))
    result_dir = root / (
        f"{config['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(result_dir)
    (result_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    seed = int(config.get("seed", 42))
    _seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    cache = Path(config["data"]["cache_dir"])
    maximum = config.get("max_samples")
    train_data = MultiTargetDataset(cache, "train", maximum, True,
                                     num_classes=int(config.get("num_classes", 5)))
    val_data = MultiTargetDataset(
        cache, "val",
        None if maximum is None else max(1024, int(maximum) // 2),
        True, num_classes=int(config.get("num_classes", 5)),
    )
    test_data = MultiTargetDataset(
        cache, "test",
        None if maximum is None else max(1024, int(maximum) // 2),
        False, num_classes=int(config.get("num_classes", 5)),
    )
    train_loader = _loader(train_data, config, True, seed)
    val_loader = _loader(val_data, config, False, seed)
    test_loader = _loader(test_data, config, False, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    model_type = config.get("model_type", "attention")
    model_kwargs = {
        "feature_dim": int(config.get("feature_dim", 64)),
        "head_dim": int(config.get("head_dim", 32)),
        "dropout": float(config.get("dropout", 0.1)),
        "num_classes": int(config.get("num_classes", 5)),
    }
    if model_type == "fusion_gru":
        model_kwargs.update(
            gru_hidden=int(config.get("gru_hidden", 48)),
            gru_layers=int(config.get("gru_layers", 2)),
        )
        model = FusionMultiTaskAttentionLOB(**model_kwargs).to(device)
    elif model_type == "fusion_tcn":
        model_kwargs.update(
            tcn_channels=int(config.get("tcn_channels", 64)),
            tcn_bins=int(config.get("tcn_bins", 1)),
        )
        _d = config.get("tcn_dilations")
        if _d is not None:
            model_kwargs["tcn_dilations"] = tuple(int(x) for x in _d)
        model = TCNFusionMultiTaskLOB(**model_kwargs).to(device)
    elif model_type == "fusion_tcn_short":
        model_kwargs.update(
            tcn_channels=int(config.get("tcn_channels", 48)),
            tcn_bins=int(config.get("tcn_bins", 1)),
        )
        _d = config.get("tcn_dilations") or (1, 2, 4, 8)
        model_kwargs["tcn_dilations"] = tuple(int(x) for x in _d)
        model = TCNFusionMultiTaskLOB(**model_kwargs).to(device)
    elif model_type == "fusion_transformer":
        model_kwargs.update(
            transformer_dim=int(config.get("transformer_dim", 64)),
            transformer_heads=int(config.get("transformer_heads", 4)),
            transformer_layers=int(config.get("transformer_layers", 3)),
            max_tokens=int(config.get("max_tokens", 128)),
        )
        model = TransformerFusionMultiTaskLOB(**model_kwargs).to(device)
    elif model_type == "attention":
        model = MultiTaskAttentionLOB(**model_kwargs).to(device)
    elif model_type == "attention_small":
        model = SmallMultiTaskAttentionLOB(**model_kwargs).to(device)
    elif model_type == "pure_transformer":
        model_kwargs.update(
            transformer_dim=int(config.get("transformer_dim", 96)),
            transformer_heads=int(config.get("transformer_heads", 4)),
            transformer_layers=int(config.get("transformer_layers", 2)),
            max_seq=int(config.get("max_seq", 64)),
        )
        model = PureTransformerMultiTaskLOB(**model_kwargs).to(device)
    elif model_type == "fusion_tcn_small":
        model_kwargs.update(
            tcn_channels=int(config.get("tcn_channels", 32)),
            tcn_bins=int(config.get("tcn_bins", 1)),
        )
        _d = config.get("tcn_dilations")
        if _d is not None:
            model_kwargs["tcn_dilations"] = tuple(int(x) for x in _d)
        model = SmallTCNFusionMultiTaskLOB(**model_kwargs).to(device)
    elif model_type == "fusion_tcn_short_small":
        model_kwargs.update(
            tcn_channels=int(config.get("tcn_channels", 24)),
            tcn_bins=int(config.get("tcn_bins", 1)),
        )
        _d = config.get("tcn_dilations") or (1, 2, 4, 8)
        model_kwargs["tcn_dilations"] = tuple(int(x) for x in _d)
        model = SmallTCNFusionMultiTaskLOB(**model_kwargs).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    optimizer = build_optimizer(model, config)
    scheduler = None
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["epochs"]),
            eta_min=float(config.get("min_lr", 1e-5)),
        )
    losses = {
        "mse": torch.nn.MSELoss().to(device),
        "ordinal": OrdinalCELoss(
            int(config.get("num_classes", 5)),
            sigma=float(config.get("ordinal_sigma", 0.7)),
        ).to(device),
    }
    weights = {
        "regression": np.asarray(
            config.get("regression_weights", [0.25, 0.15, 0.10])
        ),
        "classification": np.asarray(
            config.get("classification_weights", [1.0, 0.5, 0.25])
        ),
        "rank": torch.tensor(
            config.get("rank_weights", [1.0, 0.5, 0.25]),
            device=device,
            dtype=torch.float32,
        ),
        "rank_loss_weight": float(config.get("rank_loss_weight", 0.0)),
    }
    weights["total"] = float(
        weights["regression"].sum() + weights["classification"].sum()
    )
    regression_target = config.get("regression_target", "return_zscore")
    if regression_target not in ("return_zscore", "rank"):
        raise ValueError(
            "regression_target must be 'return_zscore' or 'rank', "
            f"got {regression_target!r}"
        )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    logger.info(
        "device=%s parameters=%d samples train=%d val=%d test=%d",
        device,
        sum(parameter.numel() for parameter in model.parameters()),
        len(train_data),
        len(val_data),
        len(test_data),
    )
    logger.info(
        "weights regression=%s classification=%s",
        weights["regression"].tolist(),
        weights["classification"].tolist(),
    )

    selection_score = config.get("selection_score", "cls_blend_532")
    selection_metric = config.get("early_stop_metric", "ic")
    if selection_metric not in {"ic", "loss"}:
        raise ValueError("early_stop_metric must be 'ic' or 'loss'")
    best_value = np.inf if selection_metric == "loss" else -np.inf
    best_ic = -np.inf
    best_epoch = -1
    stale = 0
    history = result_dir / "metrics.jsonl"
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for features, ret_target, cls_target, rank_target in train_loader:
            features = features.to(device, non_blocking=True)
            ret_target = ret_target.to(device, non_blocking=True)
            cls_target = cls_target.to(device, non_blocking=True)
            rank_target = rank_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                output = model(features)
                loss = _loss(
                    output, ret_target, cls_target, rank_target, losses, weights,
                    regression_target,
                )
            if not torch.isfinite(loss):
                logger.warning("skip_nonfinite_loss epoch=%d", epoch)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get("grad_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(features)
            total += len(features)
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, val_data, losses, weights, device, amp,
            regression_target,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_score": selection_score,
            "selection_ic": val_metrics[selection_score]["ic"],
            "val_metrics": val_metrics,
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "epoch=%03d loss=%.6f val_loss=%.6f select=%s ic=%.6f "
            "cls1=%.6f ret1=%.6f",
            epoch,
            row["train_loss"],
            val_loss,
            selection_score,
            row["selection_ic"],
            val_metrics["cls_day1"]["ic"],
            val_metrics["ret_day1"]["ic"],
        )
        current_value = val_loss if selection_metric == "loss" else row["selection_ic"]
        improved = (current_value < best_value if selection_metric == "loss"
                    else current_value > best_value)
        if improved:
            best_value = current_value
            best_ic = row["selection_ic"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": model_type,
                    "model": model_kwargs,
                    "epoch": epoch,
                    "selection_score": selection_score,
                    "val_metrics": val_metrics,
                },
                result_dir / "best_model.pt",
            )
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step()
        if stale >= int(config.get("early_stop_patience", 10)):
            logger.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    checkpoint = torch.load(
        result_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_metrics, test_reg, test_cls = evaluate(
        model, test_loader, test_data, losses, weights, device, amp,
        regression_target,
    )
    np.savez(
        result_dir / "test_outputs.npz",
        regression=test_reg.astype(np.float32),
        classification=test_cls.astype(np.float32),
        date_ids=test_data.date_ids,
        stocks=test_data.stocks,
        raw_return=test_data.raw_return,
    )
    summary = {
        "status": "complete",
        "early_stop_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_value) if selection_metric == "loss" else None,
        "selection_score": selection_score,
        "best_val_ic": best_ic,
        "val_metrics": checkpoint["val_metrics"],
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "result_dir": str(result_dir.resolve()),
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("test_metrics=%s", json.dumps(test_metrics, ensure_ascii=False))
    logger.info("training_complete")
    return result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(train_multitask(config))


if __name__ == "__main__":
    main()
