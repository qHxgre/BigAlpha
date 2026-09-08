"""Date-wise cross-sectional Alpha ranking fine-tuning."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import maybe_enable_cache
from dataset import DailyPatchBuilder
from core import FeatureTransform
from core import Normalizer
from model import ARCHITECTURE, AlphaModel
from core import (
    all_reduce_counts,
    barrier,
    cleanup_distributed,
    gather_cross_section,
    gather_cross_section_features,
    initialize_distributed,
    shard_cross_section,
    unwrap_model,
    wrap_ddp,
)
from losses import RankingLoss
from core import icir, pearson_corr, rank_corr
from pretrain import (
    amp_context,
    configure_torch_runtime,
)
from core import (
    metrics_platform_score_enabled,
    metrics_score_kwargs,
    platform_alignment_enabled,
    platform_score_kwargs,
    platform_score_numpy,
    platform_score_torch,
)
from core import ensure_dir, load_config
from core import set_seed

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def iter_days(days: list, desc: str, log_cfg: dict, enabled: bool):
    if enabled and bool(log_cfg.get("use_tqdm", True)) and tqdm is not None:
        return tqdm(
            days,
            desc=desc,
            total=len(days),
            dynamic_ncols=sys.stderr.isatty(),
            mininterval=float(log_cfg.get("tqdm_mininterval", 0.5)),
            ascii=bool(log_cfg.get("tqdm_ascii", not sys.stderr.isatty())),
            file=sys.stdout,
        )
    return days


def set_progress_postfix(progress, values: dict) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(values, refresh=True)


def progress_write(progress, message: str) -> None:
    if hasattr(progress, "write"):
        progress.write(message)
    else:
        print(message, flush=True)


def format_finetune_progress(
    epoch: int, step: int, total: int, day, loss: float, ic: float, rankic: float
) -> str:
    return (
        f"finetune epoch={epoch} step={step}/{total} day={day} "
        f"loss={loss:.6g} ic={ic:.6g} rankic={rankic:.6g}"
    )



def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ranking_loss(
    prediction,
    target,
    loss_module: RankingLoss,
    head_target=None,
) -> torch.Tensor:
    return loss_module(prediction, target, head_target)


def ranking_score(
    prediction: torch.Tensor,
    score_exposure: torch.Tensor | None,
    config: dict,
) -> torch.Tensor:
    if not platform_alignment_enabled(config):
        return prediction
    if score_exposure is None:
        raise ValueError("Platform-aligned training requires signal-date score_exposure")
    return platform_score_torch(
        prediction,
        score_exposure,
        **platform_score_kwargs(config),
    )


SEGMENT_BOUNDARIES = (0.0, 0.10, 0.20, 0.70, 0.80, 0.90, 1.0)


def metric_score_numpy(
    prediction: np.ndarray,
    score_exposure: np.ndarray | None,
    config: dict,
) -> np.ndarray:
    """Score used for reported IC/RankIC: style-neutralized when configured."""
    prediction = np.asarray(prediction, dtype=np.float64)
    if metrics_platform_score_enabled(config):
        if score_exposure is None:
            raise ValueError("metrics.platform_score requires score_exposure data")
        return platform_score_numpy(
            prediction,
            score_exposure,
            **metrics_score_kwargs(config),
        )
    return prediction


def segment_rank_metrics(
    score: np.ndarray,
    y: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Within-segment IC/RankIC on return-rank bands (top/middle/tail)."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = score.size
    if n < 2:
        return {}
    order = np.argsort(y, kind="stable")
    out: dict[str, dict[str, float | int]] = {}
    for lo, hi in zip(SEGMENT_BOUNDARIES[:-1], SEGMENT_BOUNDARIES[1:]):
        i0 = int(round(n * lo))
        i1 = int(round(n * hi))
        i1 = max(i0 + 1, min(i1, n))
        idx = order[i0:i1]
        out[f"{int(lo * 100):02d}-{int(hi * 100):02d}"] = {
            "rankic": rank_corr(score[idx], y[idx]),
            "ic": pearson_corr(score[idx], y[idx]),
            "n": int(idx.size),
        }
    return out


def resolve_nested_metric(mapping: dict, path: str) -> float:
    """Resolve a dotted path like 'segments.90-100.rankic' to a float."""
    value: object = mapping
    for key in path.split("."):
        value = value[key]  # type: ignore[index]
    return float(value)


def load_normalizers(config: dict, checkpoint: dict):
    return {
        key: Normalizer.from_json(value)
        for key, value in checkpoint["normalizer"].items()
    }


def build_cached_builder(config, raw_builder, output_dir, split_days, context):
    if context.is_main:
        builder = maybe_enable_cache(config, raw_builder, output_dir, split_days)
        barrier(context)
        return builder
    barrier(context)
    local_config = copy.deepcopy(config)
    local_config.setdefault("cache", {})["prebuild"] = False
    local_config["cache"]["write"] = False
    return maybe_enable_cache(local_config, raw_builder, output_dir, split_days)


@torch.no_grad()
def evaluate(model, loss_module, builder, days, device, x_name, config, max_days):
    model.eval()
    loss_module.eval()
    losses, daily_ic, daily_rankic = [], [], []
    segment_daily: dict[str, list[dict[str, float | int]]] = {}
    samples = 0
    for day in days[:max_days] if max_days > 0 else days:
        batch = builder.build_day(day)
        if batch is None:
            continue
        x = torch.from_numpy(batch[x_name]).to(device)
        y = torch.from_numpy(np.array(batch["y"], copy=True)).to(device)
        raw_y = torch.from_numpy(
            np.array(batch.get("raw_y", batch["y"]), copy=True)
        ).to(device)
        score_exposure = None
        if platform_alignment_enabled(config) or metrics_platform_score_enabled(config):
            score_exposure = torch.from_numpy(
                np.array(batch["score_exposure"], copy=True)
            ).to(device)
        with amp_context(config, device):
            prediction = model(x)
            processed_prediction = ranking_score(
                prediction,
                score_exposure,
                config,
            )
            loss = ranking_loss(
                processed_prediction,
                y,
                loss_module,
                head_target=raw_y,
            )
        metric_pred_np = metric_score_numpy(
            prediction.detach().float().cpu().numpy(),
            (
                score_exposure.detach().float().cpu().numpy()
                if score_exposure is not None
                else None
            ),
            config,
        )
        # Optimization follows the configured label; IC-family metrics always
        # use realized raw return so residual-label runs remain comparable.
        y_np = batch.get("raw_y", batch["y"])
        losses.append(float(loss.item()))
        daily_ic.append(pearson_corr(metric_pred_np, y_np))
        daily_rankic.append(rank_corr(metric_pred_np, y_np))
        for name, segment in segment_rank_metrics(metric_pred_np, y_np).items():
            segment_daily.setdefault(name, []).append(segment)
        samples += len(y_np)
    segments = {}
    for name, daily in segment_daily.items():
        rankics = [float(value["rankic"]) for value in daily]
        ics = [float(value["ic"]) for value in daily]
        segments[name] = {
            "rankic": float(np.nanmean(rankics)) if rankics else math.nan,
            "rankicir": icir(rankics),
            "ic": float(np.nanmean(ics)) if ics else math.nan,
            "days": len(daily),
        }
    return {
        "loss": float(np.nanmean(losses)) if losses else math.nan,
        "metric_target": "raw_return",
        "ic": float(np.nanmean(daily_ic)) if daily_ic else math.nan,
        "rankic": float(np.nanmean(daily_rankic)) if daily_rankic else math.nan,
        "rankicir": icir(daily_rankic),
        "rankicir_ann": icir(daily_rankic, annualization=252.0),
        "segments": segments,
        "days": len(daily_rankic),
        "samples": samples,
    }


def train(config: dict, dry_run: bool = False) -> int:
    configure_torch_runtime(config)
    context = initialize_distributed(config)
    seed = int(config["training"]["seed"])
    set_seed(seed)
    output_dir = Path(config["output_dir"])
    if context.is_main:
        ensure_dir(output_dir)
        (output_dir / "config_effective.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
    barrier(context)

    resume_checkpoint = config["training"].get("resume_checkpoint")
    initialize_from_pretrained = bool(
        config["training"].get("initialize_from_pretrained", True)
    )
    checkpoint_path = Path(
        resume_checkpoint or config["training"]["pretrained_checkpoint"]
    )
    is_resume = resume_checkpoint is not None
    checkpoint_payload = torch_load(checkpoint_path)
    if checkpoint_payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Fine-tuning requires a compatible Alpha checkpoint")

    transform = FeatureTransform(config)
    normalizers = load_normalizers(config, checkpoint_payload)
    raw_builder = DailyPatchBuilder(config, transform, normalizers)
    freq_key = list(raw_builder.freq_keys)[0]
    x_name = f"x_{freq_key}"
    train_days = raw_builder.signal_days(config["dates"]["train_start"], config["dates"]["train_end"])
    valid_days = raw_builder.signal_days(config["dates"]["valid_start"], config["dates"]["valid_end"])
    test_days = raw_builder.signal_days(config["dates"]["test_start"], config["dates"]["test_end"])
    builder = build_cached_builder(
        config,
        raw_builder,
        output_dir,
        {"train": train_days, "valid": valid_days, "test": test_days},
        context,
    )
    first = builder.build_day(train_days[0])
    if first is None:
        raise RuntimeError("Unable to build the first ranking day")
    if context.is_main:
        print(
            f"device={context.device} world_size={context.world_size} "
            f"first_batch={first[x_name].shape} y={first['y'].shape}",
            flush=True,
        )
    if dry_run:
        cleanup_distributed(context)
        return 0

    set_seed(seed + context.rank)
    base = AlphaModel(config, transform.n_features).to(context.device)
    if is_resume or initialize_from_pretrained:
        base.load_state_dict(checkpoint_payload["model_state"], strict=True)
    elif context.is_main:
        print(
            "initialization=random model_state=not_loaded "
            f"normalizer_checkpoint={checkpoint_path}",
            flush=True,
        )
    loss_cfg = config.get("loss", {})
    initial_weights = tuple(
        float(value) for value in loss_cfg.get("initial_weights", [0.15, 0.55, 0.30])
    )
    loss_module = RankingLoss(
        learnable=bool(loss_cfg.get("learnable_loss_weights", True)),
        initial_weights=initial_weights,
        min_weight=float(loss_cfg.get("min_weight", 0.05)),
        pairwise_max_pairs=int(loss_cfg.get("pairwise_max_pairs", 8192)),
        pairwise_temperature=float(loss_cfg.get("pairwise_temperature", 0.05)),
        pairwise_extreme_fraction=float(loss_cfg.get("pairwise_extreme_fraction", 0.25)),
        pairwise_local_fraction=float(loss_cfg.get("pairwise_local_fraction", 0.50)),
        pairwise_local_window_fraction=float(loss_cfg.get("pairwise_local_window_fraction", 0.10)),
        pairwise_mode=str(loss_cfg.get("pairwise_mode", "stratified")),
        pairwise_top_fraction=float(loss_cfg.get("pairwise_top_fraction", 0.10)),
        pairwise_boundary_fraction=float(loss_cfg.get("pairwise_boundary_fraction", 0.0)),
        pairwise_intra_top_fraction=float(loss_cfg.get("pairwise_intra_top_fraction", 0.0)),
        pairwise_false_positive_fraction=float(
            loss_cfg.get("pairwise_false_positive_fraction", 0.0)
        ),
        pairwise_head_on_raw=bool(loss_cfg.get("pairwise_head_on_raw", False)),
        rank_corr_mode=str(loss_cfg.get("rank_corr_mode", "full")),
        rank_corr_groups=int(loss_cfg.get("rank_corr_groups", 10)),
        rank_corr_head_on_raw=bool(
            loss_cfg.get("rank_corr_head_on_raw", False)
        ),
        quantile_pairwise_enabled=bool(
            loss_cfg.get("quantile_pairwise_enabled", False)
        ),
        quantile_tail_fraction=float(
            loss_cfg.get("quantile_tail_fraction", 0.10)
        ),
        quantile_top_pair_fraction=float(
            loss_cfg.get("quantile_top_pair_fraction", 0.40)
        ),
        quantile_bottom_pair_fraction=float(
            loss_cfg.get("quantile_bottom_pair_fraction", 0.20)
        ),
        quantile_alpha=float(loss_cfg.get("quantile_alpha", 1.0)),
        quantile_beta=float(loss_cfg.get("quantile_beta", 0.5)),
        quantile_gamma=float(loss_cfg.get("quantile_gamma", 3.0)),
    ).to(context.device)
    parameter_groups = base.finetune_parameter_groups(config["training"])
    if loss_module.learnable:
        parameter_groups.append(
            {
                "name": "loss_weights",
                "params": list(loss_module.parameters()),
                "lr": float(config["training"].get("loss_lr", config["training"]["head_lr"])),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    model = wrap_ddp(base, context)
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    total_epochs = int(config["training"]["epochs"])
    start_epoch = 0
    global_step = 0
    best_rankic = -math.inf
    metrics_path = output_dir / "metrics.jsonl"
    best_path = output_dir / "model_best.pt"
    head_selection_metric = config.get("training", {}).get(
        "head_selection_metric"
    )
    head_best_path = output_dir / "head_best.pt"
    best_head = -math.inf
    save_per_epoch = bool(
        config.get("training", {}).get("save_per_epoch_checkpoints", False)
    )
    log_cfg = config.get("logging", {})
    progress_every_days = max(1, int(log_cfg.get("progress_every_days", 5)))
    routing_every_days = max(1, int(log_cfg.get("routing_every_days", 25)))
    finetune_bias_update_speed = float(
        config["routing"].get("finetune_bias_update_speed", 0.0)
    )
    if finetune_bias_update_speed < 0.0:
        raise ValueError("routing.finetune_bias_update_speed must be non-negative")

    scheduler = None
    if config["training"].get("scheduler") == "cosine_warm_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(config["training"].get("scheduler_t0", 3)),
            T_mult=int(config["training"].get("scheduler_t_mult", 2)),
            eta_min=float(config["training"].get("scheduler_eta_min", 5e-6)),
        )
        if context.is_main:
            print(
                "scheduler=cosine_warm_restart "
                f"T_0={config['training'].get('scheduler_t0', 3)} "
                f"T_mult={config['training'].get('scheduler_t_mult', 2)} "
                f"eta_min={config['training'].get('scheduler_eta_min', 5e-6)}",
                flush=True,
            )

    if is_resume:
        required = {
            "optimizer_state",
            "loss_state",
            "epoch",
            "global_step",
            "valid",
        }
        missing = sorted(required.difference(checkpoint_payload))
        if missing:
            raise ValueError(
                f"Resume checkpoint is missing required state: {', '.join(missing)}"
            )
        if config["training"].get("resume_loss_state", True):
            loss_module.load_state_dict(
                checkpoint_payload["loss_state"], strict=True
            )
        optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
        scheduler_state = checkpoint_payload.get("scheduler_state")
        if scheduler is not None:
            if scheduler_state is None:
                raise ValueError("Resume checkpoint is missing scheduler_state")
            scheduler.load_state_dict(scheduler_state)
        elif scheduler_state is not None:
            raise ValueError("Resume checkpoint has scheduler state but config disables it")
        start_epoch = int(checkpoint_payload["epoch"])
        global_step = int(checkpoint_payload["global_step"])
        best_rankic = float(checkpoint_payload["valid"]["rankic"])
        if total_epochs <= start_epoch:
            raise ValueError(
                f"training.epochs={total_epochs} must exceed resumed epoch {start_epoch}"
            )
        if context.is_main:
            if checkpoint_path.resolve() != best_path.resolve():
                shutil.copy2(checkpoint_path, best_path)
            print(
                f"resume_checkpoint={checkpoint_path} start_epoch={start_epoch} "
                f"target_epoch={total_epochs} global_step={global_step} "
                f"best_valid_rankic={best_rankic:.8f}",
                flush=True,
            )
        barrier(context)

    if context.is_main:
        print(
            f"routing={base.balance_mode}_clean_gate_top{base.top_k} noise=off "
            f"finetune_bias_update_speed={finetune_bias_update_speed} "
            f"selection_ema_decay={base.selection_ema_decay} "
            f"aux_balance_weight={base.aux_balance_weight if base.balance_mode == 'auxiliary_loss' else 0.0} "
            f"expert_prior={base.expert_prior.detach().cpu().tolist()} "
            f"selection_band=[{base.minimum_selection_share}, "
            f"{base.maximum_selection_share}]",
            flush=True,
        )

    for epoch in range(start_epoch, total_epochs):
        model.train()
        loss_module.train()
        epoch_days = list(train_days)
        random.Random(seed + epoch).shuffle(epoch_days)
        daily_losses, daily_aux_losses, daily_ics, daily_rankics = [], [], [], []
        day_iterator = iter_days(
            epoch_days,
            f"finetune epoch {epoch + 1}",
            log_cfg,
            context.is_main,
        )
        for day_index, day in enumerate(day_iterator, start=1):
            batch = builder.build_day(day)
            if batch is None:
                continue
            full_x = torch.from_numpy(batch[x_name]).to(context.device)
            full_y = torch.from_numpy(np.array(batch["y"], copy=True)).to(context.device)
            full_raw_y = torch.from_numpy(
                np.array(batch.get("raw_y", batch["y"]), copy=True)
            ).to(context.device)
            full_score_exposure = None
            if platform_alignment_enabled(config) or metrics_platform_score_enabled(
                config
            ):
                full_score_exposure = torch.from_numpy(
                    np.array(batch["score_exposure"], copy=True)
                ).to(context.device)
            x = shard_cross_section(full_x, context)
            y = shard_cross_section(full_y, context)
            raw_y = shard_cross_section(full_raw_y, context)
            local_score_exposure = (
                shard_cross_section(full_score_exposure, context)
                if full_score_exposure is not None
                else None
            )
            with amp_context(config, context.device):
                local_prediction, routing = model(x, return_routing=True)
                prediction, target = gather_cross_section(local_prediction, y, context)
                raw_target = gather_cross_section_features(raw_y, context)
                score_exposure = (
                    gather_cross_section_features(local_score_exposure, context)
                    if local_score_exposure is not None
                    else None
                )
                processed_prediction = ranking_score(
                    prediction,
                    score_exposure,
                    config,
                )
                torch.manual_seed(seed + global_step)
                ranking_objective = ranking_loss(
                    processed_prediction,
                    target,
                    loss_module,
                    head_target=raw_target,
                )
                aux_loss = ranking_objective.new_zeros(())
                if base.balance_mode == "auxiliary_loss":
                    global_counts_for_aux = all_reduce_counts(
                        routing.local_counts,
                        context,
                    )
                    aux_loss = unwrap_model(model).auxiliary_load_balance_loss(
                        routing.clean_logits,
                        global_counts_for_aux,
                    )
                loss = ranking_objective + base.aux_balance_weight * aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            gate_mass = all_reduce_counts(routing.local_gate_mass, context)
            global_counts = all_reduce_counts(routing.local_counts, context)
            global_compute = all_reduce_counts(
                torch.stack(
                    (
                        routing.token_counts.detach().sum(),
                        torch.tensor(
                            routing.token_counts.numel(),
                            device=context.device,
                            dtype=torch.long,
                        ),
                        routing.native_attention_pairs.detach(),
                        routing.executed_attention_pairs.detach(),
                        routing.fixed_grid_attention_pairs.detach(),
                    )
                ).to(torch.float64),
                context,
            )
            routing_model = unwrap_model(model)
            local_load = routing_model.select_balance_load(
                routing.local_counts,
                routing.local_gate_mass,
            )
            if finetune_bias_update_speed > 0.0 and local_load is not None:
                global_load = all_reduce_counts(local_load, context)
                routing_model.update_load_balance(
                    global_load,
                    update_speed=finetune_bias_update_speed,
                )

            if context.is_main:
                metric_pred_np = metric_score_numpy(
                    prediction.detach().float().cpu().numpy(),
                    (
                        score_exposure.detach().float().cpu().numpy()
                        if score_exposure is not None
                        else None
                    ),
                    config,
                )
                target_np = raw_target.detach().float().cpu().numpy()
                loss_value = float(loss.item())
                ic_value = pearson_corr(metric_pred_np, target_np)
                rankic_value = rank_corr(metric_pred_np, target_np)
                daily_losses.append(loss_value)
                daily_aux_losses.append(float(aux_loss.detach().item()))
                daily_ics.append(ic_value)
                daily_rankics.append(rankic_value)
                if (
                    day_index % progress_every_days == 0
                    or day_index == len(epoch_days)
                ):
                    progress_write(
                        day_iterator,
                        format_finetune_progress(
                            epoch=epoch + 1,
                            step=day_index,
                            total=len(epoch_days),
                            day=day.date(),
                            loss=loss_value,
                            ic=ic_value,
                            rankic=rankic_value,
                        ),
                    )
                if (
                    day_index % routing_every_days == 0
                    or day_index == len(epoch_days)
                ):
                    selection_share = global_counts / global_counts.sum().clamp_min(1.0)
                    gate_share = gate_mass / gate_mass.sum().clamp_min(1.0)
                    mean_tokens = float(global_compute[0] / global_compute[1].clamp_min(1.0))
                    token_saving = 1.0 - mean_tokens / routing_model.max_dynamic_tokens
                    attention_saving = 1.0 - float(
                        global_compute[3] / global_compute[4].clamp_min(1.0)
                    )
                    packing_efficiency = float(
                        global_compute[2] / global_compute[3].clamp_min(1.0)
                    )
                    progress_write(
                        day_iterator,
                        "routing "
                        f"epoch={epoch + 1} step={day_index}/{len(epoch_days)} "
                        f"patch_lens={list(routing_model.patch_lens)} "
                        f"selection_share="
                        f"{[round(float(value), 4) for value in selection_share.detach().cpu()]} "
                        f"gate_share="
                        f"{[round(float(value), 4) for value in gate_share.detach().cpu()]} "
                        f"expert_prior="
                        f"{[round(float(value), 4) for value in routing_model.expert_prior.detach().cpu()]} "
                        f"selection_ema="
                        f"{[round(float(value), 4) for value in routing_model.selection_ema.detach().cpu()]} "
                        f"selection_band="
                        f"[{routing_model.minimum_selection_share:.4f}, "
                        f"{routing_model.maximum_selection_share:.4f}] "
                        f"underused="
                        f"{(routing_model.selection_ema < routing_model.minimum_selection_share).nonzero().flatten().detach().cpu().tolist()} "
                        f"overused="
                        f"{(routing_model.selection_ema > routing_model.maximum_selection_share).nonzero().flatten().detach().cpu().tolist()} "
                        f"expert_bias="
                        f"{[round(float(value), 4) for value in routing_model.expert_bias.detach().cpu()]} "
                        f"mean_tokens={mean_tokens:.2f}/{routing_model.max_dynamic_tokens} "
                        f"token_saving={token_saving:.2%} "
                        f"attention_saving={attention_saving:.2%} "
                        f"packing_efficiency={packing_efficiency:.2%}",
                    )
            global_step += 1

        if scheduler is not None:
            scheduler.step()
        barrier(context)
        if context.is_main:
            eval_model = unwrap_model(model)
            valid = evaluate(
                eval_model,
                loss_module,
                builder,
                valid_days,
                context.device,
                x_name,
                config,
                int(config["training"].get("max_valid_days", 60)),
            )
            test = evaluate(
                eval_model,
                loss_module,
                builder,
                test_days,
                context.device,
                x_name,
                config,
                int(config["training"].get("max_test_days", 0)),
            )
            loss_weights = loss_module.effective_weights()
            train_summary = {
                "loss": float(np.nanmean(daily_losses)) if daily_losses else math.nan,
                "metric_target": "raw_return",
                "aux_balance_loss": (
                    float(np.nanmean(daily_aux_losses))
                    if daily_aux_losses
                    else math.nan
                ),
                "ic": float(np.nanmean(daily_ics)) if daily_ics else math.nan,
                "rankic": float(np.nanmean(daily_rankics)) if daily_rankics else math.nan,
                "rankicir": icir(daily_rankics),
                "rankicir_ann": icir(daily_rankics, annualization=252.0),
            }
            row = {
                "epoch": epoch + 1,
                "train": train_summary,
                "valid": valid,
                "test": test,
                "loss_weights": loss_weights,
                "routing": {
                    key: value.tolist()
                    for key, value in eval_model.routing_state().items()
                },
            }
            progress_write(day_iterator, json.dumps(row, sort_keys=True))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            payload = {
                "architecture": ARCHITECTURE,
                "model_state": eval_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "loss_state": loss_module.state_dict(),
                "routing_state": eval_model.routing_state(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "config": config,
                "feature_names": transform.feature_names,
                "normalizer": {
                    key: value.to_json() for key, value in normalizers.items()
                },
                "valid": valid,
                "test": test,
                "loss_weights": loss_weights,
            }
            if save_per_epoch:
                torch.save(
                    payload,
                    output_dir / f"epoch_{epoch + 1:02d}.pt",
                )
            if valid["rankic"] > best_rankic:
                best_rankic = valid["rankic"]
                torch.save(payload, best_path)
            if head_selection_metric is not None:
                head_value = resolve_nested_metric(valid, head_selection_metric)
                if math.isfinite(head_value) and head_value > best_head:
                    best_head = head_value
                    torch.save(payload, head_best_path)
        barrier(context)

    if context.is_main:
        if not best_path.exists():
            raise RuntimeError("Best validation checkpoint was not created")
        best_payload = torch_load(best_path)
        best_row = {
            "event": "best_epoch_summary",
            "best_epoch": int(best_payload["epoch"]),
            "valid": best_payload["valid"],
            "test": best_payload["test"],
            "loss_weights": best_payload["loss_weights"],
        }
        print(json.dumps(best_row, sort_keys=True), flush=True)
    barrier(context)
    cleanup_distributed(context)
    return 0



MODEL_FORMAT = "bigalpha_model_json_v1"
EXPECTED_ARCHITECTURE = ARCHITECTURE


def tensor_to_json(tensor: torch.Tensor) -> dict:
    array = tensor.detach().cpu().contiguous().numpy()
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def json_value(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    return value


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def export_model(checkpoint: dict, output_path: Path) -> None:
    """最优权重 -> 推理 JSON（原 export_model.py 逻辑）。"""
    architecture = checkpoint.get("architecture")
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Unexpected architecture {architecture!r}; "
            f"expected {EXPECTED_ARCHITECTURE!r}"
        )
    required = {"config", "feature_names", "normalizer", "model_state"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {missing}")
    payload = {
        "format": MODEL_FORMAT,
        "architecture": architecture,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": checkpoint["config"],
        "feature_names": checkpoint["feature_names"],
        "normalizer": checkpoint["normalizer"],
        "state_dict": {
            key: tensor_to_json(value)
            for key, value in checkpoint["model_state"].items()
        },
        "routing_state": {
            key: json_value(value)
            for key, value in checkpoint.get("routing_state", {}).items()
        },
        "loss_weights": checkpoint.get("loss_weights"),
        "valid": checkpoint.get("valid"),
        "test": checkpoint.get("test"),
    }
    output_path.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "architecture": architecture,
                "epoch": checkpoint.get("epoch"),
                "tensors": len(checkpoint["model_state"]),
                "out": str(output_path),
            },
            ensure_ascii=False,
        )
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/finetune/model_best.pt"),
    )
    parser.add_argument("--out", type=Path, default=Path("model.json"))
    args = parser.parse_args()
    if args.export:
        export_model(load_checkpoint(args.checkpoint), args.out)
        return 0
    if args.config is None:
        parser.error("--config is required unless --export is used")
    return train(load_config(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
