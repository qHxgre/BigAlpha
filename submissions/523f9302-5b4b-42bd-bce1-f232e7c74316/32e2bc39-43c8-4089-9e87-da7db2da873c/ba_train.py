"""BigAlpha 2026 E5H0：共享 E1 日内编码器 + 单层日级 GRU。

相对 0.85092 版 E4A，只修改 5m 主干的跨日信息路径：
最近 5 个交易日分别复用同一套 E1 日内编码器，return/corr/std 在每天
48 根 bar 内重置；每天池化成 96 维向量，再由单层 GRU 学习 5 个日向量
的顺序。GRU 增量头零初始化，阶段1的 epoch0 与 E1 当日分数严格等价。

阶段1：加载并冻结 E1 日内编码器和原打分头，只训练单层日级 GRU 增量；
阶段2：冻结完整 5m 主干，原样训练 E4A 的同日 1m 残差分支。

数据、标签、stable_rank_v2、60 日验证、purge、1m 架构、FP32 得分与
JSON 严格保存/加载校验保持不变。

内存流程：阶段1只加载5m；随后预计算冻结5m分数并释放5m原始X，再按单月
加载1m进入阶段2。该调度避免15.7GB 5m与39.2GB 1m同时常驻。
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ba_common import (
    CONFIG, CombinedRankLoss, PanelTensor, ResidualFrequencyPanelTensor,
    align_panel_to_master, apply_datasources, apply_universe_mask,
    build_labels, build_model, build_panel, compute_rank_ic,
    config_for_frequency, cs_target_zscore, fit_field_norm,
    load_ckpt_json, load_exposure_panel, load_panel_cache, load_universe,
    now_s, predict_scores, release_unused_memory,
    save_ckpt_json, save_panel_cache, set_seed,
)

MODEL_PATH = "ba_model.json"
BASE_MODEL_PATH = "init_e4a_0850.json"
BASE_STAGE_PATH = "base_5m_shared_day_gru5.json"
LOOKBACK_DAYS = 5
TRAIN_START, TRAIN_END = "2019-01-01", "2024-12-31"
VAL_DAYS = 60


def _e4_config(base_config: dict, datasources=None,
               overrides: dict | None = None) -> dict:
    """以0.85092版E4A配置为唯一来源，只补齐当前脚本所需兼容键。"""
    cfg = copy.deepcopy(base_config)
    # 0.85092 checkpoint是单一事实来源。仅补齐其中可能缺少的运行兼容键，
    # 不覆盖已经存在的取数、训练或数值参数。
    required_defaults = {
        "bar_tables", "panel_dtypes", "intraday_relative_frequencies",
        "panel_cache_by_frequency", "query_universe_at_source",
        "preallocate_from_universe", "residual_frequency",
        "deterministic_training", "score_head_fp32",
        "trim_memory_after_query_chunk",
    }
    for k, v in CONFIG.items():
        if (k in required_defaults or k.startswith("res1m_")) and k not in cfg:
            cfg[k] = copy.deepcopy(v)
    cfg["model_version"] = CONFIG["model_version"]
    cfg["cache_version"] = CONFIG["cache_version"]
    cfg["memory_pipeline_version"] = CONFIG["memory_pipeline_version"]
    cfg["bar_table"] = CONFIG["bar_table"]
    cfg["panel_on_gpu"] = False
    if overrides:
        cfg.update(copy.deepcopy(overrides))
    # 初始化checkpoint保存的是旧的1m“两个月一批”取数配置。六年1m数据每批
    # 可达一千多万行；这里强制压到单月，只改变峰值内存，不改变任何样本。
    query_chunks = copy.deepcopy(
        cfg.get("query_chunk_months_by_frequency",
                CONFIG["query_chunk_months_by_frequency"]))
    query_chunks["1m"] = 1
    cfg["query_chunk_months_by_frequency"] = query_chunks
    # 1m只提供残差输入，标签始终由5m面板生成。
    cfg["daily_target_frequencies"] = ["5m"]
    return apply_datasources(cfg, datasources)


def extract_initializer_base(ckpt: dict) -> dict:
    """从完整0.85092版E4A checkpoint提取精确的 E1 5m 主干。"""
    c = copy.deepcopy(ckpt.get("config", {}))
    version = c.get("model_version")
    if version == "raw_hier_v2":
        base = copy.deepcopy(ckpt)
    elif version == "raw_hier_1m_res_v1":
        fm = ckpt.get("frequencies", {}).get("5m", {})
        base_state = {
            k[len("base."):]: v.detach().cpu().clone()
            for k, v in ckpt.get("state_dict", {}).items()
            if k.startswith("base.")
        }
        if not base_state:
            raise ValueError("E4A checkpoint 中没有 base.* 的 E1 权重")
        base = {
            "config": c,
            "fields": list(fm.get("fields", ckpt.get("fields", []))),
            "bar_slots": list(fm.get("bar_slots", ckpt.get("bar_slots", []))),
            "bars_per_day": int(fm.get("bars_per_day", ckpt.get("bars_per_day", 0))),
            "norm_mu": np.asarray(fm.get("norm_mu", ckpt.get("norm_mu")), np.float32),
            "norm_sd": np.asarray(fm.get("norm_sd", ckpt.get("norm_sd")), np.float32),
            "state_dict": base_state,
            "meta": {"source": "E4A", **copy.deepcopy(ckpt.get("meta", {}))},
        }
        base["config"]["model_version"] = "raw_hier_v2"
    else:
        raise ValueError(f"初始化 checkpoint 版本不支持: {version}")

    c = base.get("config", {})
    required = {
        "model_version": "raw_hier_v2",
        "lookback_days": 1,
        "v2_use_crossday": False,
        "v2_use_return": True,
        "v2_use_corr": True,
        "v2_use_std": True,
        "loss_version": "stable_rank_v2",
    }
    bad = {k: (c.get(k), v) for k, v in required.items() if c.get(k) != v}
    if bad:
        raise ValueError(f"基础 checkpoint 不是0.85092版E4A的5m配置: {bad}")
    if int(base.get("bars_per_day", 0)) != 48:
        raise ValueError("E4A基础5m主干必须是每天48根5分钟bar")
    if len(base.get("fields", [])) < 20:
        raise ValueError("初始化 checkpoint 字段元数据不完整")
    if not base.get("state_dict"):
        raise ValueError("初始化 checkpoint 缺少 E1 5m state_dict")
    return base


def validate_e5_config(config: dict, lookback_days: int):
    expected = {
        "model_version": "raw_hier_1m_res_v1",
        "memory_pipeline_version": "staged_5m_release_then_1m_monthly_v1",
        "experiment_id": "E5H0_SHARED_DAY_GRU_1M_RESIDUAL",
        "residual_frequency": "1m",
        "score_head_fp32": True,
        "res1m_zero_output_bias": True,
        "lookback_days": int(lookback_days),
        "v2_long_window_mode": "shared_day_gru_residual",
        "v2_long_position_mode": "shared_intraday_only",
        "v2_day_gru_hidden": 48,
        "v2_day_gru_layers": 1,
        "v2_train_context_only": True,
        "v2_use_crossday": False,
        "v2_use_return": True,
        "v2_use_corr": True,
        "v2_use_std": True,
        "loss_version": "stable_rank_v2",
    }
    bad = {k: (config.get(k), v) for k, v in expected.items()
           if config.get(k) != v}
    if bad:
        raise ValueError(f"E5H0 共享日内编码器受控配置被修改: {bad}")
    tables = config.get("bar_tables", {})
    if "bar5m" not in str(tables.get("5m", "")):
        raise ValueError(f"E5 缺少5分钟表: {tables}")
    if "bar1m" not in str(tables.get("1m", "")):
        raise ValueError(f"E5 缺少1分钟表: {tables}")
    if int(config.get("query_chunk_months_by_frequency", {}).get("1m", 0)) != 1:
        raise ValueError("E5H0 的1m数据必须按单月查询，避免峰值内存换页")
    if list(config.get("daily_target_frequencies", [])) != ["5m"]:
        raise ValueError("E5H0 只允许5m面板生成标签")


def validate_unchanged_e4a_contract(config: dict, source_config: dict):
    """禁止 E5H0 在共享日级路径之外悄悄改变 E4A。"""
    exact_keys = [
        "n_levels", "label_exec", "neutralize_label",
        "dropout", "score_head_fp32", "residual_frequency",
        "loss_version", "ic_weight", "huber_weight", "tail_weight",
        "tail_frac", "tail_temperature", "huber_beta",
        "tail_warmup_epochs", "target_mad_n", "val_purge_days",
        "learning_rate", "weight_decay", "grad_clip",
        "v2_d_model", "v2_group_dims", "v2_intra_dilations",
        "v2_operator_dilations", "v2_use_return", "v2_use_corr",
        "v2_use_std", "v2_return_fields", "v2_return_horizons",
        "v2_corr_pairs", "v2_corr_windows", "v2_std_fields",
        "v2_std_windows",
    ]
    exact_keys.extend(sorted(
        key for key in source_config if key.startswith("res1m_")))
    missing = [key for key in exact_keys if key not in source_config]
    drift = {
        key: (config.get(key), source_config.get(key))
        for key in exact_keys
        if key in source_config and config.get(key) != source_config.get(key)
    }
    if missing or drift:
        raise ValueError(
            "E5H0 必须保持 E4A 的字段、日内编码器、损失、优化器和1m分支不变: "
            f"source_missing={missing}, drift={drift}")


def _split_ids(panel: dict, y: np.ndarray, val_days: int, config: dict):
    first = int(config.get("lookback_days", 1)) - 1
    usable = [d for d in range(first, len(panel["days"]))
              if (panel["valid"][d] & np.isfinite(y[d])).sum()
              >= config["min_cs_size"]]
    if len(usable) < val_days + 20:
        raise RuntimeError(f"可用交易日 {len(usable)} 太少 (val_days={val_days})")
    val_ids = usable[-val_days:]
    val_start = val_ids[0]
    purge = int(config.get("val_purge_days", 0))
    train_ids = [d for d in usable[:-val_days] if d + purge < val_start]
    if len(train_ids) < 20:
        raise RuntimeError(f"purge={purge} 后训练交易日仅 {len(train_ids)} 天")
    print(f"[{now_s()}] 训练截面 {len(train_ids)} 天 "
          f"({panel['days'][train_ids[0]]}..{panel['days'][train_ids[-1]]}) | "
          f"验证 {len(val_ids)} 天 "
          f"({panel['days'][val_ids[0]]}..{panel['days'][val_ids[-1]]}) | "
          f"purge={purge} 天")
    return train_ids, val_ids, purge


def _summary(ics: list[float], spreads: list[float]) -> dict:
    ic = np.asarray(ics, dtype=np.float64)
    ls = np.asarray(spreads, dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    ls = ls[np.isfinite(ls)]
    ic_mean = float(ic.mean()) if len(ic) else np.nan
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 1e-12 else np.nan
    ls_mean = float(ls.mean()) if len(ls) else np.nan
    ls_std = float(ls.std(ddof=1)) if len(ls) > 1 else np.nan
    ls_sharpe = (ls_mean / ls_std * np.sqrt(252.0)
                 if np.isfinite(ls_std) and ls_std > 1e-12 else np.nan)
    blocks = [b for b in np.array_split(ic, 3) if len(b)]
    stress_ic = float(min(np.mean(b) for b in blocks)) if blocks else np.nan
    if len(ic) >= 20:
        worst20 = float(np.convolve(ic, np.ones(20) / 20, mode="valid").min())
    else:
        worst20 = float(ic.min()) if len(ic) else np.nan
    return {
        "rank_ic": ic_mean,
        "ic_std": ic_std,
        "icir": float(icir),
        "ls_mean": ls_mean,
        "ls_sharpe": float(ls_sharpe),
        "stress_ic": stress_ic,
        "worst20_ic": worst20,
        "n_days": int(len(ic)),
    }


def precompute_base_scores(model, pt, panel, day_ids, config, device) -> np.ndarray:
    """新5m主干已冻结；一次前向缓存，随后可彻底释放5m原始X。"""
    scores = np.full(panel["valid"].shape, np.nan, dtype=np.float32)
    base_model = model.base if hasattr(model, "base") else model
    base_tensor = pt.base if hasattr(pt, "base") else pt
    if base_tensor is None:
        raise RuntimeError("预计算5m分数时没有可用的5m PanelTensor")
    base_model.eval()
    with torch.no_grad():
        for j, d in enumerate(day_ids):
            idx = np.where(panel["valid"][d])[0]
            if len(idx) == 0:
                continue
            x = base_tensor.make(d, idx)
            with torch.autocast("cuda", enabled=config["use_amp"]
                                and device.type == "cuda"):
                s = predict_scores(base_model, x, config["predict_chunk"])
            scores[d, idx] = s.float().cpu().numpy()
            if (j + 1) % 100 == 0:
                print(f"[{now_s()}] 5m基础分数缓存 {j + 1}/{len(day_ids)} 天", flush=True)
    return scores


def prepare_frozen_base_cache(config, panel5, base_ckpt, day_ids, device):
    """只持有5m时生成阶段2所需分数；构造模型不扰动后续1m初始化RNG。"""
    if "X" not in panel5:
        raise RuntimeError("缓存5m基础分数前，panel5.X 已被释放")
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None)
    base_model = base_tensor = None
    try:
        base_config = copy.deepcopy(base_ckpt["config"])
        if base_config.get("model_version") != "raw_hier_v2":
            raise RuntimeError("阶段1 checkpoint 不是独立5m主干")
        base_model = build_model(
            base_config, panel5["fields"], 48,
            panel5["bar_slots"], device)
        base_model.load_state_dict(base_ckpt["state_dict"], strict=True)
        base_model.eval()
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        base_tensor = PanelTensor(
            panel5,
            np.asarray(base_ckpt["norm_mu"], np.float32),
            np.asarray(base_ckpt["norm_sd"], np.float32),
            int(config["lookback_days"]), device,
            keep_on_device=config.get("panel_on_gpu", False))
        cache = precompute_base_scores(
            base_model, base_tensor, panel5, day_ids, config, device)
    finally:
        del base_model, base_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    expected = panel5["valid"][np.asarray(day_ids)]
    cached = np.isfinite(cache[np.asarray(day_ids)])
    if np.any(expected & ~cached):
        raise RuntimeError("5m基础分数缓存存在有效股票缺失")
    print(f"[{now_s()}] 5m基础分数已缓存: "
          f"{cache.nbytes / 1e6:.1f}MB；可释放5m原始X", flush=True)
    return cache


def evaluate_days(model, pt, panel, y, day_ids, base_cache,
                  config, device, base_only: bool = False):
    model.eval()
    ics, spreads = [], []
    delta_stds, delta_corrs = [], []
    attempted = bad_nonfinite = near_constant = 0
    with torch.no_grad():
        for d in day_ids:
            mask = panel["valid"][d] & np.isfinite(y[d]) & np.isfinite(base_cache[d])
            if mask.sum() < config["min_cs_size"]:
                continue
            idx = np.where(mask)[0]
            base = base_cache[d, idx]
            if base_only:
                pred = base.astype(np.float64)
            else:
                batch = pt.make(d, idx, base_score=base, include_base=False)
                with torch.autocast("cuda", enabled=config["use_amp"]
                                    and device.type == "cuda"):
                    s = predict_scores(model, batch, config["predict_chunk"])
                pred = s.float().cpu().numpy().astype(np.float64)
                delta = pred - base
                delta_stds.append(float(np.std(delta)))
                if np.std(delta) > 1e-12 and np.std(base) > 1e-12:
                    delta_corrs.append(float(np.corrcoef(delta, base)[0, 1]))
            attempted += 1
            if not np.isfinite(pred).all():
                bad_nonfinite += 1
                continue
            if np.std(pred) < 1e-8:
                near_constant += 1
                continue
            yd = y[d][idx].astype(np.float64)
            ic = compute_rank_ic(pred, yd)
            if np.isfinite(ic):
                ics.append(float(ic))
            k = max(1, len(pred) // 10)
            order = np.argsort(pred)
            spreads.append(float(np.nanmean(yd[order[-k:]])
                                 - np.nanmean(yd[order[:k]])))
    metrics = _summary(ics, spreads)
    metrics.update({
        "attempted": attempted,
        "bad_nonfinite": bad_nonfinite,
        "near_constant": near_constant,
        "delta_std": float(np.mean(delta_stds)) if delta_stds else 0.0,
        "delta_base_corr": float(np.mean(delta_corrs)) if delta_corrs else 0.0,
    })
    return metrics


def _fmt(m: dict) -> str:
    return (f"IC={m['rank_ic']:.5f} ICIR={m['icir']:.4f} "
            f"LS_SR={m['ls_sharpe']:.3f} stress={m['stress_ic']:.5f} "
            f"worst20={m['worst20_ic']:.5f}")


def train_one_epoch(model, criterion, optimizer, scaler, pt, panel, y,
                    day_ids, base_cache, config, device, rng, epoch):
    model.train()
    assert not model.base.training, "冻结5m主干被意外切换到train模式"
    criterion.set_epoch(epoch)
    losses = []
    order = list(day_ids)
    rng.shuffle(order)
    trainable = [p for p in model.residual.parameters() if p.requires_grad]
    for d in order:
        mask = panel["valid"][d] & np.isfinite(y[d]) & np.isfinite(base_cache[d])
        idx = np.where(mask)[0]
        if len(idx) < config["min_cs_size"]:
            continue
        batch = pt.make(d, idx, base_score=base_cache[d, idx], include_base=False)
        yb = torch.from_numpy(cs_target_zscore(
            y[d][idx].astype(np.float64), config["target_mad_n"]
        ).astype(np.float32)).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=config["use_amp"]
                            and device.type == "cuda"):
            score = model(batch)
        loss = criterion(score.float(), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable, config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        # The rank objective cannot identify a common output bias.  Pin it to
        # zero after every step so AMP cannot waste score resolution on drift.
        model.enforce_zero_delta_bias()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else np.nan


def _state_cpu(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


_DAY_CONTEXT_PREFIXES = ("day_gru.", "day_delta_head.")


def _is_day_context_parameter(name: str) -> bool:
    return name.startswith(_DAY_CONTEXT_PREFIXES)


def _load_and_audit_shared_day_gru(
        model, config, initializer, fields, bar_slots, device):
    """加载 E1 权重并证明 E5H0 的 epoch0 与最近一天 E1 等价。"""
    if model.use_crossday:
        raise RuntimeError("E5H0 禁止启用旧 v2_use_crossday 路径")
    if model.long_window_mode != "shared_day_gru_residual":
        raise RuntimeError(
            f"E5H0 错误的长窗口模式: {model.long_window_mode}")
    if model.long_position_mode != "shared_intraday_only":
        raise RuntimeError(
            f"E5H0 错误的位置策略: {model.long_position_mode}")
    if not hasattr(model, "day_gru") or not hasattr(model, "day_delta_head"):
        raise RuntimeError("E5H0 缺少单层 day_gru 或零初始化增量头")
    if model.day_gru.num_layers != 1:
        raise RuntimeError("E5H0 day_gru 必须严格为单层")
    if int(model.day_gru.hidden_size) != int(config["v2_day_gru_hidden"]):
        raise RuntimeError("E5H0 day_gru hidden 与配置不一致")

    state_names = set(model.state_dict())
    forbidden = [
        name for name in state_names
        if name in {"day_pos", "fixed_day_sincos"}
        or name.startswith(("day_encoder.", "day_norm.", "crossday_pool."))
    ]
    if forbidden:
        raise RuntimeError(
            f"E5H0 混入旧位置/跨日注意力模块: {forbidden[:8]}")

    source_state = initializer["state_dict"]
    source_names = set(source_state)
    model_base_names = {
        name for name in state_names if not _is_day_context_parameter(name)}
    if source_names != model_base_names:
        missing = sorted(model_base_names - source_names)
        extra = sorted(source_names - model_base_names)
        shape_bad = sorted(
            name for name in source_names & model_base_names
            if tuple(source_state[name].shape)
            != tuple(model.state_dict()[name].shape))
        raise RuntimeError(
            "E5H0 与 E1 日内编码器签名不一致: "
            f"missing={missing[:8]}, extra={extra[:8]}, "
            f"shape_bad={shape_bad[:8]}")

    incompatible = model.load_state_dict(source_state, strict=False)
    expected_missing = sorted(
        name for name in state_names if _is_day_context_parameter(name))
    if (sorted(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys):
        raise RuntimeError(
            "E1 权重加载范围异常: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}")
    loaded_state = model.state_dict()
    for name, value in source_state.items():
        if not torch.equal(value.cpu(), loaded_state[name].detach().cpu()):
            raise RuntimeError(f"E1 权重未逐元素加载: {name}")

    final_weight = model.day_delta_head[-1].weight.detach()
    if torch.count_nonzero(final_weight).item() != 0:
        raise RuntimeError("day_delta_head 最后一层没有零初始化")

    # 构造严格的一日 E1 参考模型。保存并恢复 RNG，避免审计改变训练序列。
    ref_config = copy.deepcopy(initializer["config"])
    ref_config.update({
        "model_version": "raw_hier_v2",
        "lookback_days": 1,
        "v2_use_crossday": False,
        "v2_long_window_mode": "shared_intraday_concat",
        "v2_long_position_mode": "recent_day_only",
    })
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None)
    try:
        reference = build_model(
            ref_config, fields, 48, bar_slots, device=device)
        reference.load_state_dict(source_state, strict=True)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    model.eval()
    reference.eval()
    n = 3
    total = n * int(config["lookback_days"]) * 48 * len(fields)
    x_long = torch.linspace(
        -1.0, 1.0, total, dtype=torch.float32, device=device
    ).reshape(n, int(config["lookback_days"]) * 48, len(fields))
    with torch.no_grad():
        day_sequence = model.encode(
            x_long, return_day_sequence=True).float()
        flat_days = x_long.reshape(
            n, int(config["lookback_days"]), 48, len(fields)
        ).reshape(n * int(config["lookback_days"]), 48, len(fields))
        reference_days = reference.encode(flat_days).float().reshape(
            n, int(config["lookback_days"]), -1)
        score_long = model(x_long).float()
        score_e1 = reference(x_long[:, -48:, :]).float()
    day_embedding_max_abs_diff = float(
        (day_sequence - reference_days).abs().max().cpu())
    max_abs_diff = float((score_long - score_e1).abs().max().cpu())
    del (reference, x_long, flat_days, day_sequence, reference_days,
         score_long, score_e1)
    if day_embedding_max_abs_diff > 2e-5:
        raise RuntimeError(
            "5日共享编码结果不等于逐日独立E1编码: "
            f"max_abs_diff={day_embedding_max_abs_diff:.3e}")
    if max_abs_diff > 2e-5:
        raise RuntimeError(
            f"E5H0 epoch0 未与 E1 等价: max_abs_diff={max_abs_diff:.3e}")

    total_params = sum(p.numel() for p in model.parameters())
    context_params = sum(
        p.numel() for name, p in model.named_parameters()
        if _is_day_context_parameter(name))
    e1_params = total_params - context_params
    print(
        f"[{now_s()}] E5H0 架构审计通过: "
        f"共享E1日内编码器={e1_params:,}参数；"
        f"单层day-GRU残差={context_params:,}参数；"
        f"逐日E1编码max_abs_diff={day_embedding_max_abs_diff:.3e}；"
        f"epoch0/E1 max_abs_diff={max_abs_diff:.3e}；"
        "无day_pos、无跨日注意力、算子逐日重置")
    return {
        "mode": "shared_day_gru_residual",
        "position_mode": "shared_intraday_only",
        "lookback_days": int(config["lookback_days"]),
        "bars_per_day": 48,
        "encoded_days": int(config["lookback_days"]),
        "day_embedding_dim": int(config["v2_d_model"]),
        "operators_reset_each_day": True,
        "shared_intraday_encoder": True,
        "day_gru_layers": 1,
        "day_gru_hidden": int(config["v2_day_gru_hidden"]),
        "day_position_encoding": False,
        "crossday_attention": False,
        "legacy_crossday_modules_absent": True,
        "per_day_e1_embedding_max_abs_diff": day_embedding_max_abs_diff,
        "initial_e1_max_abs_diff": max_abs_diff,
        "e1_parameters": int(e1_params),
        "day_context_parameters": int(context_params),
        "base_parameters": int(total_params),
    }


def evaluate_base_days(model, pt, panel, y, day_ids, config, device):
    """阶段1验证；指标定义与阶段2完全一致，只是不计算1m增量诊断。"""
    model.eval()
    ics, spreads = [], []
    attempted = bad_nonfinite = near_constant = 0
    with torch.no_grad():
        for d in day_ids:
            mask = panel["valid"][d] & np.isfinite(y[d])
            if mask.sum() < config["min_cs_size"]:
                continue
            idx = np.where(mask)[0]
            x = pt.make(d, idx)
            with torch.autocast("cuda", enabled=config["use_amp"]
                                and device.type == "cuda"):
                s = predict_scores(model, x, config["predict_chunk"])
            pred = s.float().cpu().numpy().astype(np.float64)
            attempted += 1
            if not np.isfinite(pred).all():
                bad_nonfinite += 1
                continue
            if np.std(pred) < 1e-8:
                near_constant += 1
                continue
            yd = y[d][idx].astype(np.float64)
            ic = compute_rank_ic(pred, yd)
            if np.isfinite(ic):
                ics.append(float(ic))
            k = max(1, len(pred) // 10)
            order = np.argsort(pred)
            spreads.append(float(np.nanmean(yd[order[-k:]])
                                 - np.nanmean(yd[order[:k]])))
    metrics = _summary(ics, spreads)
    metrics.update({
        "attempted": attempted,
        "bad_nonfinite": bad_nonfinite,
        "near_constant": near_constant,
        "delta_std": 0.0,
        "delta_base_corr": 0.0,
    })
    return metrics


def train_base_one_epoch(model, criterion, optimizer, scaler, pt, panel, y,
                         day_ids, config, device, rng, epoch):
    """每日完整截面；E5H0 仅训练 day-GRU 与跨日增量头。"""
    if config.get("v2_train_context_only", False):
        # 冻结 E1 中的 Dropout 也必须保持 eval，否则“冻结权重”仍会改变基线分数。
        model.eval()
        model.day_gru.train()
        model.day_delta_head.train()
    else:
        model.train()
    criterion.set_epoch(epoch)
    losses = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("阶段1没有可训练参数")
    order = list(day_ids)
    rng.shuffle(order)
    for d in order:
        mask = panel["valid"][d] & np.isfinite(y[d])
        idx = np.where(mask)[0]
        if len(idx) < config["min_cs_size"]:
            continue
        x = pt.make(d, idx)
        yb = torch.from_numpy(cs_target_zscore(
            y[d][idx].astype(np.float64), config["target_mad_n"]
        ).astype(np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=config["use_amp"]
                            and device.type == "cuda"):
            score = model(x)
        loss = criterion(score.float(), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable, config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else np.nan


def train_long_5m_base(
        config, panel, y, val_days, out_path, device, initializer):
    """阶段1：冻结精确 E1 主干，只训练单层 day-GRU 的增量分数。"""
    base_config = copy.deepcopy(config)
    base_config["model_version"] = "raw_hier_v2"
    base_config["experiment_id"] = "E5H0_SHARED_DAY_GRU_5M_BASE"
    train_ids, val_ids, purge = _split_ids(panel, y, val_days, base_config)
    # 复用 E1 的归一化口径是 epoch0 严格等价的一部分。
    mu = np.asarray(initializer["norm_mu"], np.float32)
    sd = np.asarray(initializer["norm_sd"], np.float32)
    pt = PanelTensor(panel, mu, sd, base_config["lookback_days"], device,
                     keep_on_device=base_config.get("panel_on_gpu", False))

    set_seed(base_config["seed"], base_config.get("deterministic_training", False))
    model = build_model(base_config, panel["fields"], 48,
                        panel["bar_slots"], device)
    architecture_audit = _load_and_audit_shared_day_gru(
        model, base_config, initializer, panel["fields"],
        panel["bar_slots"], device)

    # E5H0 是“只增加日级上下文”的受控实验。E1 编码器、原 head 及所有
    # 日内算子完全冻结；只有 day_gru/day_delta_head 可以更新。
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_day_context_parameter(name))
    trainable = [
        p for name, p in model.named_parameters()
        if _is_day_context_parameter(name) and p.requires_grad
    ]
    trainable_names = [
        name for name, p in model.named_parameters() if p.requires_grad]
    bad_trainable = [
        name for name in trainable_names if not _is_day_context_parameter(name)]
    if bad_trainable or not trainable:
        raise RuntimeError(
            f"E5H0 可训练参数范围错误: bad={bad_trainable[:8]}, "
            f"count={len(trainable_names)}")
    architecture_audit["trainable_parameters"] = int(
        sum(p.numel() for p in trainable))
    architecture_audit["trainable_parameter_names"] = trainable_names

    criterion = CombinedRankLoss(base_config)
    optimizer = torch.optim.AdamW(
        trainable, lr=base_config["learning_rate"],
        weight_decay=base_config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=base_config["use_amp"] and device.type == "cuda")
    rng = np.random.default_rng(base_config["seed"])

    baseline_metrics = evaluate_base_days(
        model, pt, panel, y, val_ids, base_config, device)
    history = {
        "baseline_e1_metrics": baseline_metrics,
        "train_loss": [], "val_metrics": [], "val_rank_ic": [],
    }
    best_val = float(baseline_metrics["rank_ic"])
    best_state = _state_cpu(model)
    best_metrics = copy.deepcopy(baseline_metrics)
    best_epoch = 0
    patience_left = int(base_config["patience"])
    print(f"[{now_s()}] base epoch 00 = exact E1 | "
          f"{_fmt(baseline_metrics)}")
    for epoch in range(int(base_config["epochs"])):
        t0 = time.time()
        loss = train_base_one_epoch(
            model, criterion, optimizer, scaler, pt, panel, y,
            train_ids, base_config, device, rng, epoch)
        val = evaluate_base_days(
            model, pt, panel, y, val_ids, base_config, device)
        history["train_loss"].append(loss)
        history["val_metrics"].append(val)
        history["val_rank_ic"].append(val["rank_ic"])
        print(f"  base epoch {epoch + 1:02d} | loss={loss:.4f} | {_fmt(val)} "
              f"| ΔIC={val['rank_ic'] - baseline_metrics['rank_ic']:+.5f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        if np.isfinite(val["rank_ic"]) and val["rank_ic"] > best_val:
            best_val = float(val["rank_ic"])
            best_state = _state_cpu(model)
            best_metrics = copy.deepcopy(val)
            best_epoch = epoch + 1
            patience_left = int(base_config["patience"])
        elif epoch + 1 >= int(base_config["min_epochs"]):
            patience_left -= 1
        if patience_left <= 0:
            print(f"  base early stop at epoch {epoch + 1}", flush=True)
            break
    model.load_state_dict(best_state, strict=True)
    selection = {
        "train_days": len(train_ids),
        "train_start": str(panel["days"][train_ids[0]]),
        "train_end": str(panel["days"][train_ids[-1]]),
        "val_days": len(val_ids),
        "val_start": str(panel["days"][val_ids[0]]),
        "val_end": str(panel["days"][val_ids[-1]]),
        "purge_days": purge,
        "best_epoch": int(best_epoch),
        "baseline_e1_metrics": baseline_metrics,
        "selected_metrics": best_metrics,
    }
    save_ckpt_json(
        out_path, model, base_config, panel["fields"], panel["bar_slots"],
        48, mu, sd, selected_epochs=int(best_epoch),
        training_protocol=(
            "frozen_e1_shared_day_single_gru_residual"),
        architecture_audit=architecture_audit,
        selection=selection, history=history)
    reloaded = load_ckpt_json(out_path)
    for name, value in best_state.items():
        if not torch.equal(value, reloaded["state_dict"][name]):
            raise RuntimeError(f"阶段1 JSON 保存/加载不一致: {name}")
    print(f"[{now_s()}] 阶段1选择 epoch={best_epoch} | {_fmt(best_metrics)} | "
          f"相对E1 ΔIC={best_metrics['rank_ic'] - baseline_metrics['rank_ic']:+.5f}")
    return reloaded


def train_model(config, panels, y, val_days, out_path, device, base_ckpt,
                base_cache=None, split_ids=None):
    panel5, panel1 = panels["5m"], panels["1m"]
    if split_ids is None:
        train_ids, val_ids, purge = _split_ids(
            panel5, y, val_days, config)
    else:
        train_ids, val_ids, purge = split_ids

    stage_config = base_ckpt.get("config", {})
    stage_expected = {
        "model_version": "raw_hier_v2",
        "experiment_id": "E5H0_SHARED_DAY_GRU_5M_BASE",
        "lookback_days": int(config["lookback_days"]),
        "v2_long_window_mode": "shared_day_gru_residual",
        "v2_long_position_mode": "shared_intraday_only",
        "v2_day_gru_hidden": 48,
        "v2_day_gru_layers": 1,
        "v2_train_context_only": True,
        "v2_use_crossday": False,
    }
    stage_bad = {
        key: (stage_config.get(key), expected)
        for key, expected in stage_expected.items()
        if stage_config.get(key) != expected
    }
    if stage_bad:
        raise RuntimeError(
            f"阶段1 checkpoint 不是本次 E5H0 共享编码器模型: {stage_bad}")

    # 5m归一化和权重来自刚完成的阶段1；1m统计量仍只来自训练日。
    mu5 = np.asarray(base_ckpt["norm_mu"], np.float32)
    sd5 = np.asarray(base_ckpt["norm_sd"], np.float32)
    mu1, sd1 = fit_field_norm(panel1["X"], np.asarray(train_ids))
    norms = {"5m": (mu5, sd5), "1m": (mu1, sd1)}
    pt = ResidualFrequencyPanelTensor(
        panels, norms, config, device,
        keep_on_device=config.get("panel_on_gpu", False),
        include_base_tensor=base_cache is None)

    frequency_meta = {
        "5m": {
            "fields": panel5["fields"], "bar_slots": panel5["bar_slots"],
            "bars_per_day": 48, "norm_mu": mu5, "norm_sd": sd5,
        },
        "1m": {
            "fields": panel1["fields"], "bar_slots": panel1["bar_slots"],
            "bars_per_day": 240, "norm_mu": mu1, "norm_sd": sd1,
        },
    }
    model = build_model(config, panel5["fields"], 48, panel5["bar_slots"],
                        device, frequency_meta=frequency_meta)
    model.base.load_state_dict(base_ckpt["state_dict"], strict=True)
    model.freeze_base()
    if any(p.requires_grad for p in model.base.parameters()):
        raise RuntimeError("共享日内 + day-GRU 的5m主干冻结失败")

    base_state_names = set(model.base.state_dict())
    required_context_prefixes = ("day_gru.", "day_delta_head.")
    missing_context = [
        prefix for prefix in required_context_prefixes
        if not any(name.startswith(prefix) for name in base_state_names)
    ]
    forbidden_context = [
        name for name in base_state_names
        if name in {"day_pos", "fixed_day_sincos"}
        or name.startswith(("day_encoder.", "day_norm.", "crossday_pool."))
    ]
    if missing_context or forbidden_context:
        raise RuntimeError(
            "阶段2的5m主干结构异常: "
            f"missing={missing_context}, forbidden={forbidden_context[:8]}")
    if model.base.day_gru.num_layers != 1:
        raise RuntimeError("阶段2 day-GRU 不再是单层")
    if int(model.base.day_gru.hidden_size) != 48:
        raise RuntimeError("阶段2 day-GRU hidden 不再是48")
    residual_rnns = [
        name for name, module in model.residual.named_modules()
        if isinstance(module, (nn.GRU, nn.LSTM))
    ]
    if residual_rnns:
        raise RuntimeError(
            f"1m残差分支意外加入循环网络: {residual_rnns}")

    all_ids = sorted(set(train_ids) | set(val_ids))
    if base_cache is None:
        if "X" not in panel5:
            raise RuntimeError(
                "panel5.X 已释放，但没有传入预计算 base_cache")
        base_cache = precompute_base_scores(
            model, pt, panel5, all_ids, config, device)
    else:
        base_cache = np.asarray(base_cache, dtype=np.float32)
        if base_cache.shape != panel5["valid"].shape:
            raise RuntimeError(
                f"base_cache形状错误: {base_cache.shape} != "
                f"{panel5['valid'].shape}")
        required = (panel5["valid"][np.asarray(all_ids)]
                    & np.isfinite(y[np.asarray(all_ids)]))
        cached = np.isfinite(base_cache[np.asarray(all_ids)])
        if np.any(required & ~cached):
            raise RuntimeError("外部5m base_cache缺少训练/验证样本")
        print(f"[{now_s()}] 阶段2复用预计算5m分数；"
              "不再持有5m原始X", flush=True)
    base_metrics = evaluate_days(
        model, pt, panel5, y, val_ids, base_cache, config, device,
        base_only=True)
    initial_metrics = evaluate_days(
        model, pt, panel5, y, val_ids, base_cache, config, device)
    max_initial_diff = abs(initial_metrics["rank_ic"] - base_metrics["rank_ic"])
    if max_initial_diff > 1e-12 or initial_metrics["delta_std"] > 1e-8:
        raise RuntimeError(
            f"1m零初始化不满足5m基线等价性: ICdiff={max_initial_diff}, "
            f"delta_std={initial_metrics['delta_std']}")
    print(f"[{now_s()}] epoch00 frozen shared-day-GRU 5m | "
          f"{_fmt(base_metrics)}")

    criterion = CombinedRankLoss(config)
    optimizer = torch.optim.AdamW(
        (p for p in model.residual.parameters() if p.requires_grad),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=config["use_amp"] and device.type == "cuda")
    rng = np.random.default_rng(config["seed"])

    history = {
        "base_metrics": base_metrics,
        "train_loss": [], "val_metrics": [], "val_rank_ic": [],
    }
    # epoch0是合法候选：若1m没有验证增量，自动退回精确的长窗口5m基线。
    best_val = float(base_metrics["rank_ic"])
    best_state = _state_cpu(model)
    best_metrics = copy.deepcopy(base_metrics)
    best_epoch = 0
    patience_left = int(config["patience"])

    pareto_state = _state_cpu(model)
    pareto_metrics = copy.deepcopy(base_metrics)
    pareto_epoch = 0

    for epoch in range(int(config["epochs"])):
        t0 = time.time()
        loss = train_one_epoch(
            model, criterion, optimizer, scaler, pt, panel5, y,
            train_ids, base_cache, config, device, rng, epoch)
        val = evaluate_days(
            model, pt, panel5, y, val_ids, base_cache, config, device)
        history["train_loss"].append(loss)
        history["val_metrics"].append(val)
        history["val_rank_ic"].append(val["rank_ic"])
        print(f"  epoch {epoch + 1:02d} | loss={loss:.4f} | {_fmt(val)} "
              f"| ΔIC={val['rank_ic'] - base_metrics['rank_ic']:+.5f} "
              f"delta_sd={val['delta_std']:.4e} "
              f"corr(delta,5m)={val['delta_base_corr']:+.3f} "
              f"| {time.time() - t0:.0f}s", flush=True)

        if np.isfinite(val["rank_ic"]) and val["rank_ic"] > best_val:
            best_val = float(val["rank_ic"])
            best_state = _state_cpu(model)
            best_metrics = copy.deepcopy(val)
            best_epoch = epoch + 1
            patience_left = int(config["patience"])
        elif epoch + 1 >= int(config["min_epochs"]):
            patience_left -= 1

        keys = ("rank_ic", "icir", "ls_sharpe", "stress_ic")
        safe = all(np.isfinite(val[k]) and val[k] >= base_metrics[k] for k in keys)
        if safe and val["rank_ic"] >= pareto_metrics["rank_ic"]:
            pareto_state = _state_cpu(model)
            pareto_metrics = copy.deepcopy(val)
            pareto_epoch = epoch + 1

        if patience_left <= 0:
            print(f"  early stop at epoch {epoch + 1}", flush=True)
            break

    model.load_state_dict(best_state, strict=True)
    model.freeze_base()
    print(f"[{now_s()}] E5H0残差阶段选择 epoch={best_epoch} | "
          f"{_fmt(best_metrics)} | "
          f"相对5m基线 ΔIC={best_metrics['rank_ic'] - base_metrics['rank_ic']:+.5f}")
    print(f"[{now_s()}] 四项不劣Pareto epoch={pareto_epoch} | {_fmt(pareto_metrics)}")

    base_sha = hashlib.sha256(Path(config["base_checkpoint_path"]).read_bytes()).hexdigest()
    architecture_audit = copy.deepcopy(
        base_ckpt.get("meta", {}).get("architecture_audit", {}))
    architecture_audit.update({
        "mode": str(config["v2_long_window_mode"]),
        "position_mode": str(config["v2_long_position_mode"]),
        "lookback_days": int(config["lookback_days"]),
        "encoded_days": int(config["lookback_days"]),
        "bars_per_encoded_day": 48,
        "operator_reset_each_day": True,
        "shared_intraday_encoder": True,
        "day_gru_layers": int(model.base.day_gru.num_layers),
        "day_gru_hidden": int(model.base.day_gru.hidden_size),
        "day_position_encoding": False,
        "crossday_attention": False,
        "v2_use_crossday": False,
        "required_day_context_modules_present": not missing_context,
        "legacy_crossday_modules_absent": not forbidden_context,
        "one_minute_residual_rnn_absent": not residual_rnns,
        "base_parameters": int(sum(
            p.numel() for p in model.base.parameters())),
    })
    if architecture_audit["mode"] != "shared_day_gru_residual":
        raise RuntimeError("最终模型不再是共享日内编码器 + day-GRU")
    if architecture_audit["position_mode"] != "shared_intraday_only":
        raise RuntimeError("最终模型意外加入日级位置编码")
    if not architecture_audit["legacy_crossday_modules_absent"]:
        raise RuntimeError("最终模型混入旧 day_pos/跨日注意力模块")
    if not architecture_audit["one_minute_residual_rnn_absent"]:
        raise RuntimeError("最终模型的1m残差分支不再与E4A一致")

    selection = {
        "train_days": len(train_ids),
        "train_start": str(panel5["days"][train_ids[0]]),
        "train_end": str(panel5["days"][train_ids[-1]]),
        "val_days": len(val_ids),
        "val_start": str(panel5["days"][val_ids[0]]),
        "val_end": str(panel5["days"][val_ids[-1]]),
        "purge_days": purge,
        "best_epoch": int(best_epoch),
        "base_metrics": base_metrics,
        "selected_metrics": best_metrics,
        "pareto_epoch": int(pareto_epoch),
        "pareto_metrics": pareto_metrics,
    }
    save_ckpt_json(
        out_path, model, config, panel5["fields"], panel5["bar_slots"],
        48, mu5, sd5, frequency_meta=frequency_meta,
        selected_epochs=int(best_epoch),
        training_protocol=(
            "frozen_e1_shared_intraday_day_gru_context_then_"
            "frozen_5m_same_day_1m_residual"),
        base_checkpoint_sha256=base_sha,
        base_checkpoint_meta=base_ckpt.get("meta", {}),
        architecture_audit=architecture_audit,
        selection=selection, history=history)

    # 同时保存严格四项不劣版本；epoch0表示精确5m基线。
    pareto_path = str(Path(out_path).with_name(Path(out_path).stem + "_pareto.json"))
    if pareto_epoch != best_epoch:
        model.load_state_dict(pareto_state, strict=True)
        model.freeze_base()
        save_ckpt_json(
            pareto_path, model, config, panel5["fields"], panel5["bar_slots"],
            48, mu5, sd5, frequency_meta=frequency_meta,
            selected_epochs=int(pareto_epoch),
            training_protocol=(
                "frozen_e1_shared_intraday_day_gru_context_then_"
                "frozen_1m_residual_pareto4"),
            base_checkpoint_sha256=base_sha,
            architecture_audit=architecture_audit,
            selection=selection, history=history)
        model.load_state_dict(best_state, strict=True)
        model.freeze_base()
    return model


def _cache_path(config, freq, start, end, fields, slots=None):
    fcfg = config_for_frequency(config, freq)
    spec = {
        "frequency": freq, "table": fcfg["bar_table"],
        "fields": list(fields) if fields is not None else None,
        "bar_slots": list(slots) if slots is not None else None,
        "panel_dtype": fcfg.get("panel_dtype"),
        "intraday_relative": fcfg.get("intraday_relative", False),
        "cache_version": config.get("cache_version"),
    }
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    s, e = str(start)[:10].replace("-", ""), str(end)[:10].replace("-", "")
    return Path(f"panel_{freq}_{s}_{e}_{h}.npz")


def train_and_save(datasources=None, start: str = TRAIN_START,
                   end: str = TRAIN_END, val_days: int = VAL_DAYS,
                   out_path: str = MODEL_PATH,
                   epochs: int | None = None,
                   base_model_path: str = BASE_MODEL_PATH,
                   config_overrides: dict | None = None,
                   lookback_days: int = LOOKBACK_DAYS,
                   base_stage_path: str | None = None):
    init_path = Path(base_model_path)
    if not init_path.exists():
        raise FileNotFoundError(
            f"缺少初始化checkpoint: {init_path}. 请把当前0.85092版保存为 "
            f"{BASE_MODEL_PATH}，或显式传 base_model_path。")
    initializer_ckpt = load_ckpt_json(str(init_path))
    initializer = extract_initializer_base(initializer_ckpt)
    initializer_1m = initializer_ckpt.get("frequencies", {}).get("1m", {})
    lookback_days = int(lookback_days)
    if lookback_days <= 1:
        raise ValueError("E5H0必须包含至少2个交易日")
    config = _e4_config(initializer["config"], datasources, config_overrides)
    # 受控改动：E1 每日独立复用、固定算子在日界重置；只增加单层日级 GRU
    # 增量分数。E1 的全部权重与归一化保持冻结。
    config.update({
        "lookback_days": lookback_days,
        "v2_long_window_mode": "shared_day_gru_residual",
        "v2_long_position_mode": "shared_intraday_only",
        "v2_day_gru_hidden": 48,
        "v2_day_gru_layers": 1,
        "v2_train_context_only": True,
        "v2_use_crossday": False,
        "experiment_id": "E5H0_SHARED_DAY_GRU_1M_RESIDUAL",
    })
    if epochs is not None:
        config["epochs"] = int(epochs)
    validate_unchanged_e4a_contract(config, initializer["config"])
    validate_e5_config(config, lookback_days)
    set_seed(config["seed"], config.get("deterministic_training", False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"[{now_s()}] E5H0 shared E1 intraday + single day-GRU | "
          f"5m窗口={lookback_days}日，"
          f"每天48步独立复用E1（日内算子逐日重置），"
          f"5个96维日向量 -> 单层GRU(48)，无day_pos/跨日注意力，"
          f"冻结E1只训练零初始化日级增量 "
          f"| 初始化权重={init_path} | 然后冻结5m训练原同日1m残差 "
          f"| loss={config['loss_version']} | val={val_days}d "
          "| memory=5m→cache→release→1m(monthly)")

    s10, e10 = str(start)[:10], str(end)[:10]

    # 必须在 bar 之前读取股票池：把历史成分股并集前推为 DAI instrument 分区
    # 过滤，并用相同日期/股票轴预分配最终 panel。若读取失败则安全退回旧路径。
    universe = load_universe(config, s10, e10)
    source_days = source_codes = None
    if universe and config.get("query_universe_at_source", True):
        source_days = sorted(universe)
        source_codes = sorted(set().union(*universe.values()))
        print(f"[{now_s()}] 查询端股票池: {len(source_days)}天, "
              f"历史并集 {len(source_codes)} 股")
    use_master = bool(source_days and source_codes
                      and config.get("preallocate_from_universe", True))

    def load_frequency(freq: str, fields=None, slots=None):
        fcfg = config_for_frequency(config, freq)
        cache = _cache_path(config, freq, s10, e10, fields, slots)
        use_cache = bool(config.get("panel_cache_by_frequency", {}).get(freq, False))
        if use_cache and cache.exists():
            p = load_panel_cache(str(cache))
        else:
            p = build_panel(
                fcfg, s10, e10, fields=fields, bar_slots=slots,
                instruments=source_codes,
                master_days=source_days if use_master else None,
                master_codes=source_codes if use_master else None)
            if use_cache:
                save_panel_cache(p, str(cache))
        # 兼容优化前已经生成的5m缓存；即使旧缓存含额外股票，也在此裁成同一主轴。
        if use_master:
            p = align_panel_to_master(
                p, np.asarray(source_days), np.asarray(source_codes))
        print(f"[{now_s()}] {freq}: {len(p['days'])}天 x {len(p['codes'])}股 x "
              f"{p['X'].shape[2]}bar x {p['X'].shape[3]}字段 "
              f"({p['X'].nbytes / 1e9:.1f}GB {p['X'].dtype}, "
              f"relative={fcfg.get('intraday_relative', False)})")
        return p

    panel5 = load_frequency(
        "5m", fields=list(initializer["fields"]),
        slots=list(initializer["bar_slots"]))
    if panel5["X"].shape[2] != 48:
        raise RuntimeError(f"E5预期5m每天48根，实际 {panel5['X'].shape[2]}")

    # 阶段1只需要5m。先做股票池、标签和5m训练，绝不提前分配39GB的1m面板。
    member = apply_universe_mask(panel5, universe, mask_history=False)
    print(f"[{now_s()}] 5m universe 对齐: "
          f"{int((member.sum(axis=1) > 0).sum())}/{len(panel5['days'])} 天")

    expo = (load_exposure_panel(config, panel5)
            if config["neutralize_label"] else None)
    y = build_labels(config, panel5, expo)
    del expo
    stage_path = Path(base_stage_path or BASE_STAGE_PATH)
    print(f"[{now_s()}] ===== 阶段1/2：冻结E1，训练"
          f"{lookback_days}日日级GRU增量 =====")
    base_ckpt = train_long_5m_base(
        config, panel5, y, val_days, str(stage_path), device, initializer)
    config["base_checkpoint_path"] = str(stage_path.resolve())

    # 阶段2训练只使用冻结5m分数，不再需要5m原始bar。趁5m X仍在内存时
    # 一次性缓存全部训练/验证分数，然后释放15.7GB，再开始构建1m面板。
    train_ids, val_ids, purge = _split_ids(
        panel5, y, val_days, config)
    all_ids = sorted(set(train_ids) | set(val_ids))
    print(f"[{now_s()}] ===== 阶段1.5/2：缓存冻结5m分数，"
          "随后释放5m原始X =====")
    base_cache = prepare_frozen_base_cache(
        config, panel5, base_ckpt, all_ids, device)
    released_5m_gb = panel5["X"].nbytes / 1e9
    del panel5["X"]
    release_unused_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{now_s()}] 已释放5m原始X约 {released_5m_gb:.1f}GB；"
          "现在才开始加载1m", flush=True)

    # 官方1m/5m表字段定义相同；强制相同字段。优先复用E4A checkpoint
    # 保存的240个bar槽位，使build_panel可在查询前完成最终数组分配。
    one_minute_slots = initializer_1m.get("bar_slots")
    one_minute_slots = (list(one_minute_slots)
                        if one_minute_slots is not None else None)
    panel1 = load_frequency(
        "1m", fields=list(initializer["fields"]),
        slots=one_minute_slots)
    if panel1["X"].shape[2] != 240:
        raise RuntimeError(f"E5预期1m每天240根，实际 {panel1['X'].shape[2]}")
    if len(panel5["fields"]) + len(panel1["fields"]) > 100:
        raise RuntimeError("双频率原始字段审计总数超过100")
    panel1 = align_panel_to_master(
        panel1, panel5["days"], panel5["codes"])
    apply_universe_mask(panel1, universe, mask_history=False)
    panels = {"5m": panel5, "1m": panel1}

    print(f"[{now_s()}] ===== 阶段2/2：冻结完整5m主干，"
          "训练原E4A同日1m残差 =====")
    trained = train_model(
        config, panels, y, val_days, out_path, device, base_ckpt,
        base_cache=base_cache,
        split_ids=(train_ids, val_ids, purge))

    # 防止再次出现训练/提交JSON数值不一致。
    reloaded = load_ckpt_json(out_path)
    current = _state_cpu(trained)
    for name, value in current.items():
        if not torch.equal(value, reloaded["state_dict"][name]):
            raise RuntimeError(f"最终JSON保存/加载不一致: {name}")
    print(f"[{now_s()}] 最终JSON逐张量保存/加载校验通过")
    return trained


def _smoke_panels(config, n_days=34, n_codes=80):
    fields = ["open", "high", "low", "close", "pre_close", "volume", "amount"]
    for i in range(1, 6):
        fields += [f"bid_price{i}", f"ask_price{i}",
                   f"bid_volume{i}", f"ask_volume{i}"]
    rng = np.random.default_rng(7)
    days = np.array([f"2024-{1 + i // 20:02d}-{1 + i % 20:02d}" for i in range(n_days)])
    codes = np.array([f"S{i:04d}" for i in range(n_codes)])
    X5 = rng.normal(size=(n_days, n_codes, 48, len(fields))).astype(np.float32)
    X1 = rng.normal(size=(n_days, n_codes, 240, len(fields))).astype(np.float16)
    valid = np.ones((n_days, n_codes), dtype=bool)
    close = np.exp(X5[:, :, -1, fields.index("close")]).astype(np.float32)
    vwap = close * (1 + rng.normal(0, .001, close.shape)).astype(np.float32)
    panel5 = {"days": days, "codes": codes, "X": X5, "vwap": vwap,
              "close": close, "valid": valid.copy(), "fields": fields,
              "bar_slots": list(range(48))}
    panel1 = {"days": days.copy(), "codes": codes.copy(), "X": X1,
              "vwap": vwap.copy(), "close": close.copy(), "valid": valid.copy(),
              "fields": fields.copy(), "bar_slots": list(range(240))}
    y = np.full_like(vwap, np.nan, dtype=np.float32)
    signal = X1[:, :, -10:, fields.index("bid_volume1")].astype(np.float32).mean(-1)
    y[:-2] = .002 * signal[:-2] + rng.normal(0, .002, signal[:-2].shape)
    return {"5m": panel5, "1m": panel1}, y


def run_smoke(out_path="ba_model_e5h0_smoke.json"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_path = Path(BASE_MODEL_PATH)
    if not init_path.exists():
        raise FileNotFoundError(
            f"smoke test 缺少初始化模型: {init_path}")
    initializer = extract_initializer_base(load_ckpt_json(str(init_path)))
    cfg = _e4_config(initializer["config"])
    cfg.update({
        "model_version": "raw_hier_1m_res_v1",
        "experiment_id": "E5H0_SHARED_DAY_GRU_1M_RESIDUAL",
        "lookback_days": LOOKBACK_DAYS,
        "v2_long_window_mode": "shared_day_gru_residual",
        "v2_long_position_mode": "shared_intraday_only",
        "v2_day_gru_hidden": 48,
        "v2_day_gru_layers": 1,
        "v2_train_context_only": True,
        "v2_use_crossday": False,
        "min_cs_size": 32,
        "min_group_size": 16,
        "epochs": 2,
        "min_epochs": 1,
        "patience": 2,
        "predict_chunk": 40,
        "use_amp": device.type == "cuda",
    })
    validate_unchanged_e4a_contract(cfg, initializer["config"])
    validate_e5_config(cfg, LOOKBACK_DAYS)
    panels, y = _smoke_panels(cfg)
    panels["5m"]["bar_slots"] = list(initializer["bar_slots"])
    if panels["5m"]["fields"] != initializer["fields"]:
        raise RuntimeError("smoke字段与E1初始化字段不一致")
    if panels["5m"]["bar_slots"] != initializer["bar_slots"]:
        raise RuntimeError("smoke的5m bar_slots与E1初始化不一致")
    stage = Path("base_5m_shared_day_gru5_smoke.json")
    try:
        base_ckpt = train_long_5m_base(
            cfg, panels["5m"], y, val_days=6,
            out_path=str(stage), device=device, initializer=initializer)
        cfg["base_checkpoint_path"] = str(stage.resolve())
        split = _split_ids(panels["5m"], y, 6, cfg)
        all_ids = sorted(set(split[0]) | set(split[1]))
        base_cache = prepare_frozen_base_cache(
            cfg, panels["5m"], base_ckpt, all_ids, device)
        del panels["5m"]["X"]
        gc.collect()
        train_model(
            cfg, panels, y, val_days=6, out_path=out_path,
            device=device, base_ckpt=base_ckpt,
            base_cache=base_cache, split_ids=split)
    finally:
        stage.unlink(missing_ok=True)
    print("SMOKE TEST DONE: shared E1 intraday + single day-GRU context "
          "+ frozen-5m + unchanged 1m residual 已贯通")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "E5H0 shared E1 intraday + single day-GRU + 1m residual"))
    parser.add_argument("--start", default=TRAIN_START)
    parser.add_argument("--end", default=TRAIN_END)
    parser.add_argument("--val-days", type=int, default=VAL_DAYS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base", default=BASE_MODEL_PATH)
    parser.add_argument("--base-stage", default=None)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--out", default=MODEL_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_smoke(args.out)
    else:
        train_and_save(
            None, args.start, args.end, args.val_days,
            args.out, args.epochs, args.base,
            lookback_days=args.lookback_days,
            base_stage_path=args.base_stage)


if __name__ == "__main__":
    main()
