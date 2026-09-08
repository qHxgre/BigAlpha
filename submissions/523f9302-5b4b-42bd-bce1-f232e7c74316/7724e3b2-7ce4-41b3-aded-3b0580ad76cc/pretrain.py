"""Two-stage Alpha reconstruction pretraining with DDP-safe balancing."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import maybe_enable_cache
from dataset import DailyPatchBuilder
from core import FeatureTransform
from core import Normalizer, fit_normalizer_for_frequency
from model import ARCHITECTURE, AlphaModel
from core import (
    all_reduce_counts,
    barrier,
    cleanup_distributed,
    initialize_distributed,
    unwrap_model,
    wrap_ddp,
)
from core import ensure_dir, load_config
from core import set_seed

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


class PretrainTask(nn.Module):
    def __init__(self, model: AlphaModel, mask_ratio: float):
        super().__init__()
        self.model = model
        self.mask_ratio = float(mask_ratio)

    def forward(self, x: torch.Tensor, dense_routing: bool = False):
        loss, routing = self.model.masked_reconstruction_loss(
            x,
            self.mask_ratio,
            dense_routing=dense_routing,
        )
        return (
            loss,
            routing.clean_logits,
            routing.local_counts.detach(),
            routing.patch_counts.detach(),
            routing.local_gate_mass.detach(),
            routing.local_regions.detach(),
            routing.token_counts.detach().sum(),
            torch.tensor(
                routing.token_counts.numel(),
                device=routing.token_counts.device,
                dtype=torch.long,
            ),
            routing.native_attention_pairs.detach(),
            routing.executed_attention_pairs.detach(),
            routing.fixed_grid_attention_pairs.detach(),
        )


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


def format_pretrain_progress(
    epoch: int, step: int, total: int, day, loss: float, mask_ratio: float
) -> str:
    return (
        f"pretrain epoch={epoch} step={step}/{total} day={day} "
        f"loss={loss:.6g} mask_ratio={mask_ratio:.3f}"
    )



def configure_torch_runtime(config: dict | None = None) -> None:
    if torch.cuda.is_available():
        allow_tf32 = bool((config or {}).get("training", {}).get("allow_tf32", False))
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def shuffled_day_blocks(days: list, seed: int, block_size: int) -> list:
    rng = random.Random(seed)
    size = max(1, int(block_size))
    blocks = [list(days[start : start + size]) for start in range(0, len(days), size)]
    rng.shuffle(blocks)
    for block in blocks:
        if rng.random() < 0.5:
            block.reverse()
    return [day for block in blocks for day in block]


def amp_context(config: dict, device: torch.device):
    amp = str(config["training"].get("amp_dtype", "bf16")).lower()
    if device.type != "cuda" or amp in {"", "none", "off", "fp32", "float32"}:
        return nullcontext()
    if amp in {"bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp in {"fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported amp_dtype={amp!r}")


def load_or_fit_normalizers(config: dict, transform: FeatureTransform, output_dir: Path):
    norm_path = output_dir / "normalizer.json"
    if norm_path.exists():
        payload = json.loads(norm_path.read_text(encoding="utf-8"))
        return {key: Normalizer.from_json(value) for key, value in payload.items()}

    from core import date_ts

    normalizers = {}
    for freq_key in config["frequency"]:
        normalizers[freq_key] = fit_normalizer_for_frequency(
            Path(config["data_root"]),
            config["tables"][freq_key],
            transform,
            date_ts(config["dates"]["train_start"]),
            date_ts(config["dates"]["train_end"]),
            int(config["normalization"]["max_rows_per_frequency"]),
            int(config["normalization"]["batch_size"]),
        )
    norm_path.write_text(
        json.dumps({key: value.to_json() for key, value in normalizers.items()}, indent=2),
        encoding="utf-8",
    )
    return normalizers


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
def evaluate_pretrain(model, builder, days, x_name, device, config, batch_size):
    model.eval()
    torch.manual_seed(int(config["training"].get("validation_seed", 2026)))
    losses = []
    max_days = int(config["training"].get("max_valid_days", 40))
    for day in days[:max_days]:
        batch = builder.build_day(day)
        if batch is None:
            continue
        x = torch.from_numpy(batch[x_name]).to(device)
        for start in range(0, x.shape[0], batch_size):
            part = x[start : start + batch_size]
            with amp_context(config, device):
                loss, _ = model.masked_reconstruction_loss(part, float(config["training"]["mask_ratio"]))
            losses.append(float(loss.item()))
    return {"days": min(len(days), max_days), "loss": float(np.mean(losses)) if losses else math.nan}


def pretrain(config: dict, dry_run: bool = False) -> int:
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

    transform = FeatureTransform(config)
    if context.is_main:
        normalizers = load_or_fit_normalizers(config, transform, output_dir)
    barrier(context)
    if not context.is_main:
        normalizers = load_or_fit_normalizers(config, transform, output_dir)

    raw_builder = DailyPatchBuilder(config, transform, normalizers)
    freq_key = list(raw_builder.freq_keys)[0]
    x_name = f"x_{freq_key}"
    train_days = raw_builder.signal_days(config["dates"]["train_start"], config["dates"]["train_end"])
    valid_days = raw_builder.signal_days(config["dates"]["valid_start"], config["dates"]["valid_end"])
    builder = build_cached_builder(
        config,
        raw_builder,
        output_dir,
        {"train": train_days, "valid": valid_days},
        context,
    )
    first = builder.build_day(train_days[0])
    if first is None:
        raise RuntimeError("Unable to build the first pretraining day")
    if context.is_main:
        print(
            f"device={context.device} world_size={context.world_size} first_batch={first[x_name].shape}",
            flush=True,
        )
    if dry_run:
        cleanup_distributed(context)
        return 0

    set_seed(seed + context.rank)
    base = AlphaModel(config, transform.n_features).to(context.device)
    resume_checkpoint = config["training"].get("resume_checkpoint")
    resume_payload = None
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        resume_payload = torch.load(
            checkpoint_path,
            map_location=context.device,
            weights_only=False,
        )
        if resume_payload.get("architecture") != ARCHITECTURE:
            raise ValueError(
                "Resume checkpoint architecture does not match the current model: "
                f"{resume_payload.get('architecture')!r} != {ARCHITECTURE!r}"
            )
        base.load_state_dict(resume_payload["model_state"])
    task = PretrainTask(base, float(config["training"]["mask_ratio"])).to(context.device)
    task = wrap_ddp(task, context)
    optimizer = torch.optim.AdamW(
        task.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = None
    if config["training"].get("scheduler") == "cosine_warm_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(config["training"].get("scheduler_t0", 4)),
            T_mult=int(config["training"].get("scheduler_t_mult", 2)),
            eta_min=float(config["training"].get("scheduler_eta_min", 1e-5)),
        )
        if context.is_main:
            print(
                "scheduler=cosine_warm_restart "
                f"T_0={config['training'].get('scheduler_t0', 4)} "
                f"T_mult={config['training'].get('scheduler_t_mult', 2)} "
                f"eta_min={config['training'].get('scheduler_eta_min', 1e-5)}",
                flush=True,
            )
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    global_batch_size = int(config["training"].get("batch_size", 192))
    accumulate_day = bool(config["training"].get("accumulate_day", False))
    total_epochs = int(config["training"]["epochs"])
    dense_warmup_epochs = int(config["routing"].get("dense_warmup_epochs", 0))
    if dense_warmup_epochs != 0:
        raise ValueError("dense_warmup_epochs must be 0 for fixed-prior sparse routing")
    bias_update_speed = float(config["routing"].get("bias_update_speed", 0.01))
    day_block_size = int(config["training"].get("shuffle_day_block_size", 20))
    log_cfg = config.get("logging", {})
    progress_every_days = max(1, int(log_cfg.get("progress_every_days", 1)))
    progress_every_steps = max(1, int(log_cfg.get("progress_every_steps", 1)))
    routing_every_days = max(1, int(log_cfg.get("routing_every_days", 25)))
    log_json_events = bool(log_cfg.get("json_events", False))
    metrics_path = output_dir / "pretrain_metrics.jsonl"
    best = math.inf
    last_valid: dict = {}
    global_step = 0
    start_epoch = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler_state = resume_payload.get("scheduler_state")
        if scheduler is not None:
            if scheduler_state is None:
                raise ValueError("Resume checkpoint is missing scheduler_state")
            scheduler.load_state_dict(scheduler_state)
        start_epoch = int(resume_payload["epoch"])
        global_step = int(resume_payload.get("global_step", 0))
        best = float(resume_payload.get("valid", {}).get("loss", math.inf))
        last_valid = dict(resume_payload.get("valid", {}))
        if context.is_main:
            print(
                f"resume_checkpoint={resume_checkpoint} start_epoch={start_epoch} "
                f"global_step={global_step} best_valid={best:.6g}",
                flush=True,
            )

    if context.is_main:
        print(
            f"routing={base.balance_mode}_clean_gate_top{base.top_k} noise=off "
            "dense_warmup_epochs=0 "
            f"bias_update_speed={bias_update_speed if base.uses_bias_control else 0.0} "
            f"aux_balance_weight={base.aux_balance_weight if base.balance_mode == 'auxiliary_loss' else 0.0} "
            f"selection_ema_decay={base.selection_ema_decay} bias_clip={base.bias_clip} "
            f"expert_prior={base.expert_prior.detach().cpu().tolist()} "
            f"selection_band=[{base.minimum_selection_share}, "
            f"{base.maximum_selection_share}]",
            flush=True,
        )
    for epoch in range(start_epoch, total_epochs):
        task.train()
        dense_routing = epoch < dense_warmup_epochs
        epoch_days = shuffled_day_blocks(train_days, seed + epoch, day_block_size)
        losses = []
        day_iterator = iter_days(
            epoch_days,
            f"pretrain epoch {epoch + 1}",
            log_cfg,
            context.is_main,
        )
        for day_index, day in enumerate(day_iterator, start=1):
            day_started = time.perf_counter()
            if context.is_main and log_json_events:
                progress_write(
                    day_iterator,
                    json.dumps(
                        {
                            "event": "pretrain_day_start",
                            "epoch": epoch + 1,
                            "day_index": day_index,
                            "days_total": len(epoch_days),
                            "date": str(day.date()),
                            "global_step": global_step,
                        },
                        sort_keys=True,
                    ),
                )
            data_started = time.perf_counter()
            batch = builder.build_day(day)
            if batch is None:
                continue
            data_seconds = time.perf_counter() - data_started
            transfer_started = time.perf_counter()
            x = torch.from_numpy(batch[x_name]).to(context.device)
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            transfer_seconds = time.perf_counter() - transfer_started
            day_counts = torch.zeros(base.n_experts, device=context.device)
            day_patch_counts = torch.zeros(len(base.patch_lens), device=context.device)
            day_gate_mass = torch.zeros(base.n_experts, device=context.device)
            day_regions = torch.zeros((), device=context.device)
            day_compute = torch.zeros(5, dtype=torch.float64, device=context.device)
            n_chunks = (x.shape[0] + global_batch_size - 1) // global_batch_size
            day_loss_sum = 0.0
            day_loss_count = 0
            train_started = time.perf_counter()
            if accumulate_day:
                optimizer.zero_grad(set_to_none=True)
            for chunk_index, start in enumerate(
                range(0, x.shape[0], global_batch_size),
                start=1,
            ):
                step_started = time.perf_counter()
                global_part = x[start : start + global_batch_size]
                local_part = global_part[context.rank :: context.world_size]
                if local_part.shape[0] == 0:
                    raise RuntimeError("Global batch size must be at least the DDP world size")
                with amp_context(config, context.device):
                    (
                        reconstruction_loss,
                        clean_logits,
                        local_counts,
                        local_patch_counts,
                        local_gate_mass,
                        local_regions,
                        local_token_sum,
                        local_sample_count,
                        local_native_pairs,
                        local_executed_pairs,
                        local_fixed_pairs,
                    ) = task(local_part, dense_routing=dense_routing)
                    aux_loss = reconstruction_loss.new_zeros(())
                    if base.balance_mode == "auxiliary_loss":
                        global_counts_for_aux = all_reduce_counts(local_counts, context)
                        aux_loss = unwrap_model(task).model.auxiliary_load_balance_loss(
                            clean_logits,
                            global_counts_for_aux,
                        )
                    loss = reconstruction_loss + base.aux_balance_weight * aux_loss
                    if accumulate_day:
                        loss = loss / n_chunks
                if not accumulate_day:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                day_counts.add_(local_counts)
                day_patch_counts.add_(local_patch_counts)
                day_gate_mass.add_(local_gate_mass)
                day_regions.add_(local_regions)
                local_compute = torch.stack(
                    (
                        local_token_sum,
                        local_sample_count,
                        local_native_pairs,
                        local_executed_pairs,
                        local_fixed_pairs,
                    )
                ).to(torch.float64)
                day_compute.add_(local_compute)
                if not accumulate_day:
                    torch.nn.utils.clip_grad_norm_(task.parameters(), grad_clip)
                    optimizer.step()
                    routing_model = unwrap_model(task).model
                    local_load = routing_model.select_balance_load(
                        local_counts,
                        local_gate_mass,
                    )
                    if not dense_routing and local_load is not None:
                        global_load = all_reduce_counts(local_load, context)
                        routing_model.update_load_balance(
                            global_load,
                            update_speed=bias_update_speed,
                        )
                loss_value = float(loss.detach().item()) * (
                    n_chunks if accumulate_day else 1.0
                )
                losses.append(loss_value)
                day_loss_sum += loss_value
                day_loss_count += 1
                global_step += 1
                if (
                    context.is_main
                    and (hasattr(day_iterator, "set_postfix") or log_json_events)
                    and global_step % progress_every_steps == 0
                ):
                    step_seconds = time.perf_counter() - step_started
                    set_progress_postfix(
                        day_iterator,
                        {
                            "date": str(day.date()),
                            "chunk": f"{chunk_index}/{n_chunks}",
                            "step": global_step,
                            "loss": f"{loss_value:.6f}",
                            "aux": f"{float(aux_loss.detach().item()):.4f}",
                            "bias_eta": f"{bias_update_speed if base.uses_bias_control and not dense_routing else 0.0:.4f}",
                            "patch": "/".join(str(int(v)) for v in local_patch_counts.tolist()),
                            "tokens": f"{float(local_token_sum) / max(1, int(local_sample_count)):.1f}",
                            "attn_save": f"{1.0 - float(local_executed_pairs) / max(1.0, float(local_fixed_pairs)):.1%}",
                            "sec": f"{step_seconds:.2f}",
                        },
                    )
                    if log_json_events:
                        progress_write(
                            day_iterator,
                            json.dumps(
                                {
                                    "event": "pretrain_step",
                                    "epoch": epoch + 1,
                                    "date": str(day.date()),
                                    "day_index": day_index,
                                    "chunk": chunk_index,
                                    "chunks": n_chunks,
                                    "global_step": global_step,
                                    "global_samples": int(global_part.shape[0]),
                                    "local_samples": int(local_part.shape[0]),
                                    "loss": loss_value,
                                    "reconstruction_loss": float(reconstruction_loss.detach().item()),
                                    "auxiliary_balance_loss": float(aux_loss.detach().item()),
                                    "dense_routing": dense_routing,
                                    "balance_mode": base.balance_mode,
                                    "bias_update_speed": (
                                        bias_update_speed
                                        if base.uses_bias_control and not dense_routing
                                        else 0.0
                                    ),
                                    "seconds": step_seconds,
                                },
                                sort_keys=True,
                            ),
                        )
            if accumulate_day:
                torch.nn.utils.clip_grad_norm_(task.parameters(), grad_clip)
                optimizer.step()
                routing_model = unwrap_model(task).model
                local_load = routing_model.select_balance_load(
                    day_counts,
                    day_gate_mass,
                )
                if not dense_routing and local_load is not None:
                    global_load = all_reduce_counts(local_load, context)
                    routing_model.update_load_balance(
                        global_load,
                        update_speed=bias_update_speed,
                    )
            train_seconds = time.perf_counter() - train_started
            loss_stats = all_reduce_counts(
                torch.tensor(
                    [day_loss_sum, float(day_loss_count)],
                    dtype=torch.float64,
                    device=context.device,
                ),
                context,
            )
            global_day_counts = all_reduce_counts(day_counts, context)
            global_patch_counts = all_reduce_counts(day_patch_counts, context)
            global_day_gate_mass = all_reduce_counts(day_gate_mass, context)
            global_day_regions = all_reduce_counts(day_regions, context)
            global_day_compute = all_reduce_counts(day_compute, context)
            if context.is_main and (
                day_index % progress_every_days == 0
                or day_index == len(epoch_days)
            ):
                gpu_memory_mb = 0.0
                if context.device.type == "cuda":
                    gpu_memory_mb = torch.cuda.max_memory_allocated(context.device) / (1024 ** 2)
                day_loss = float(loss_stats[0] / loss_stats[1].clamp_min(1.0))
                total_seconds = time.perf_counter() - day_started
                progress_write(
                    day_iterator,
                    format_pretrain_progress(
                        epoch + 1, day_index, len(epoch_days), day.date(),
                        day_loss, float(config["training"]["mask_ratio"]),
                    ),
                )
                if (
                    day_index % routing_every_days == 0
                    or day_index == len(epoch_days)
                ):
                    routing_model = unwrap_model(task).model
                    selection_share = (
                        global_day_counts / global_day_counts.sum().clamp_min(1.0)
                    )
                    gate_share = global_day_gate_mass / global_day_gate_mass.sum().clamp_min(1.0)
                    mean_tokens = float(
                        global_day_compute[0] / global_day_compute[1].clamp_min(1.0)
                    )
                    token_saving = 1.0 - mean_tokens / routing_model.max_dynamic_tokens
                    attention_saving = 1.0 - float(
                        global_day_compute[3] / global_day_compute[4].clamp_min(1.0)
                    )
                    packing_efficiency = float(
                        global_day_compute[2] / global_day_compute[3].clamp_min(1.0)
                    )
                    progress_write(
                        day_iterator,
                        "routing "
                        f"epoch={epoch + 1} step={day_index}/{len(epoch_days)} "
                        f"patch_lens={list(base.patch_lens)} "
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
                if log_json_events:
                    progress_write(
                        day_iterator,
                        json.dumps(
                            {
                                "event": "pretrain_day_end",
                                "epoch": epoch + 1,
                                "date": str(day.date()),
                                "day_index": day_index,
                                "days_total": len(epoch_days),
                                "global_step": global_step,
                                "samples": int(x.shape[0]),
                                "chunks": n_chunks,
                                "loss": day_loss,
                                "expert_counts": global_day_counts.cpu().tolist(),
                                "gate_mass": global_day_gate_mass.cpu().tolist(),
                                "regions": float(global_day_regions.cpu()),
                                "patch_lens": list(base.patch_lens),
                                "patch_counts": global_patch_counts.cpu().tolist(),
                                "mean_tokens": float(
                                    global_day_compute[0]
                                    / global_day_compute[1].clamp_min(1.0)
                                ),
                                "native_attention_pairs": float(global_day_compute[2]),
                                "executed_attention_pairs": float(global_day_compute[3]),
                                "fixed_grid_attention_pairs": float(global_day_compute[4]),
                                "data_seconds": data_seconds,
                                "transfer_seconds": transfer_seconds,
                                "train_seconds": train_seconds,
                                "total_seconds": total_seconds,
                                "gpu_memory_mb": gpu_memory_mb,
                            },
                            sort_keys=True,
                        ),
                    )

        if scheduler is not None:
            scheduler.step()
        barrier(context)
        if context.is_main:
            eval_model = unwrap_model(task).model
            valid = evaluate_pretrain(
                eval_model,
                builder,
                valid_days,
                x_name,
                context.device,
                config,
                global_batch_size,
            )
            last_valid = valid
            routing_control = eval_model.routing_state()
            summary = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "valid": valid,
                "routing": {
                    "expert_prior": routing_control["expert_prior"].tolist(),
                    "selection_ema": routing_control["selection_ema"].tolist(),
                    "minimum_selection_share": base.minimum_selection_share,
                    "maximum_selection_share": base.maximum_selection_share,
                    "expert_bias": routing_control["expert_bias"].tolist(),
                    "bias_updates": int(routing_control["bias_updates"]),
                },
            }
            progress_write(day_iterator, json.dumps(summary, sort_keys=True))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")
            checkpoint = {
                "architecture": ARCHITECTURE,
                "model_state": eval_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "routing_state": eval_model.routing_state(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "config": config,
                "feature_names": transform.feature_names,
                "normalizer": {key: value.to_json() for key, value in normalizers.items()},
                "valid": valid,
            }
            if valid["loss"] < best:
                best = valid["loss"]
                torch.save(checkpoint, output_dir / "pretrain_best.pt")
        barrier(context)

    cleanup_distributed(context)
    return 0


def build_cache(config: dict, splits: list[str]) -> int:
    """预构建日级特征缓存（原 build_cache.py 逻辑）。"""
    configure_torch_runtime(config)
    set_seed(int(config["training"]["seed"]))
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)
    transform = FeatureTransform(config)
    normalizers = load_or_fit_normalizers(config, transform, output_dir)
    raw_builder = DailyPatchBuilder(config, transform, normalizers)
    split_days = {}
    for split in splits:
        split_days[split] = raw_builder.signal_days(
            config["dates"][f"{split}_start"],
            config["dates"][f"{split}_end"],
        )
    local_config = copy.deepcopy(config)
    local_config.setdefault("cache", {})["enabled"] = True
    local_config["cache"]["prebuild"] = True
    local_config["cache"]["write"] = True
    maybe_enable_cache(local_config, raw_builder, output_dir, split_days)
    print(
        f"daily feature cache ready: splits={','.join(splits)} "
        f"dir={local_config['cache']['dir']}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.build_cache:
        return build_cache(config, args.splits)
    return pretrain(config, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
