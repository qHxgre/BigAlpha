"""Utilities for reproducible training runs.

These helpers do not make cross-hardware floating point math magically
identical, but they remove the avoidable randomness in Python, NumPy, PyTorch
and LightGBM. Use ``BQ_REPRO_STRICT=1`` for official retraining.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CUBLAS_WORKSPACE = ":4096:8"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def seed_from_env(default: int = 42) -> int:
    return int(os.environ.get("BQ_REPRO_SEED", str(default)))


def strict_repro_enabled() -> bool:
    return env_flag("BQ_REPRO_STRICT", False)


def bootstrap_repro_env(seed: int | None = None, *, single_thread: bool | None = None) -> None:
    """Set process env vars that should exist before torch/lightgbm import."""
    if seed is None:
        seed = seed_from_env()
    if single_thread is None:
        single_thread = strict_repro_enabled()

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS_WORKSPACE)

    if single_thread:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(key, "1")


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed Python/NumPy/PyTorch and configure deterministic torch behavior."""
    random.seed(seed)
    np.random.seed(seed)

    info: dict[str, Any] = {
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        info.update({
            "torch": torch.__version__,
            "torch_cuda": getattr(torch.version, "cuda", None),
            "torch_deterministic": deterministic_torch,
            "cuda_available": torch.cuda.is_available(),
        })
    except Exception as exc:  # pragma: no cover - best effort metadata
        info["torch_seed_error"] = repr(exc)

    return info


def make_torch_generator(seed: int):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn."""
    try:
        import torch

        worker_seed = torch.initial_seed() % 2**32
    except Exception:
        worker_seed = (seed_from_env() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def lgb_repro_params(params: dict[str, Any], *, seed: int | None = None, num_threads: int | None = None) -> dict[str, Any]:
    """Return LightGBM params with deterministic switches added."""
    out = dict(params)
    if seed is None:
        seed = int(out.get("random_state", seed_from_env()))
    if num_threads is None:
        num_threads = int(os.environ.get("BQ_LGB_NUM_THREADS", "1" if strict_repro_enabled() else "32"))

    out.setdefault("random_state", seed)
    out.setdefault("seed", seed)
    out.setdefault("bagging_seed", seed)
    out.setdefault("feature_fraction_seed", seed)
    out.setdefault("data_random_seed", seed)
    out.setdefault("drop_seed", seed)
    out.setdefault("deterministic", True)
    out.setdefault("force_col_wise", True)
    out.setdefault("n_jobs", num_threads)
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_repro_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


bootstrap_repro_env()
