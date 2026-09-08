# ModelSet_Framework.py
# PonyOracle EVO：训练 / 存盘 / 推理
import json
import base64
import zlib
import math
import os
import random
import sys
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import dai
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn

from Support_BaseData import (
    set_datasources,
    instruments_table,
    get_standard_details,
    seed_scale_details_cache,
    peek_scale_details_cache,
    DEFAULT_MODEL_JSON,
    SCALE_DETAILS_KEY,
)

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from ModelSet_Config import Config
from ModelSet_Model import build_model
from ModelSet_Loss import (
    PonyOracleLoss,
    hard_long_short_return,
    proxy_score_from_series,
    daily_ic,
)
from ModelSet_DataLoader import PrefetchDataLoader
from Support_GetDates import get_calendar

MODEL_PATH = os.path.join(_PKG_ROOT, DEFAULT_MODEL_JSON)
LOG2 = math.log(2.0)  # 零技能参照 ≈ 0.693

_PAYLOAD_MEM: Dict[str, dict] = {}

TRAIN_START, TRAIN_END = "2019-06-01", "2024-12-25"
VALID_START, VALID_END = "2019-03-01", "2019-05-31"

# ======================================================================
# JSON / 路径
# ======================================================================

def _payload_key(model_path: str) -> str:
    return os.path.abspath(model_path)


def _ensure_details_path(path: str) -> str:
    default_name = DEFAULT_MODEL_JSON
    if path and os.path.isabs(path) and os.path.exists(path):
        return path
    if path and not os.path.isabs(path):
        cand = os.path.join(_PKG_ROOT, path)
        if os.path.exists(cand) or path == default_name:
            return cand if (path == default_name or os.path.exists(cand)) else os.path.join(
                _PKG_ROOT, default_name
            )
        if path.endswith(".json"):
            return cand
    name = os.path.basename(path) if path else default_name
    if not name:
        name = default_name
    if name in ("Support_ScaleDetails.json", "ScaleDetails.json"):
        name = default_name
    return os.path.join(_PKG_ROOT, name)


def _seed_details_from_payload(model_path: str, payload: dict) -> None:
    nested = payload.get(SCALE_DETAILS_KEY)
    if isinstance(nested, dict) and nested:
        seed_scale_details_cache(model_path, nested)


def _load_existing_payload(model_path: str) -> dict:
    key = _payload_key(model_path)
    hit = _PAYLOAD_MEM.get(key)
    if hit is not None:
        return hit
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return {}
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    _PAYLOAD_MEM[key] = raw
    _seed_details_from_payload(model_path, raw)
    return raw


def _model_has_weights(model_path: str) -> bool:
    key = _payload_key(model_path)
    cached = _PAYLOAD_MEM.get(key)
    if cached is not None:
        sd = cached.get("state_dict")
        return isinstance(sd, dict) and len(sd) > 0
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return False
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            prev = ""
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return False
                if '"state_dict"' in (prev + chunk):
                    return True
                prev = chunk[-64:]
    except OSError:
        return False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_date_list(start: str, end: str) -> List[str]:
    sd = pd.Timestamp(start).strftime("%Y-%m-%d")
    ed = pd.Timestamp(end).strftime("%Y-%m-%d")
    return (
        get_calendar()
        .filter(
            (pl.col("date") >= pl.lit(sd).str.to_date("%Y-%m-%d"))
            & (pl.col("date") <= pl.lit(ed).str.to_date("%Y-%m-%d"))
        )
        .select(pl.col("date").cast(pl.String).sort())["date"]
        .to_list()
    )


def config_to_dict(config: Config) -> dict:
    d = asdict(config)
    d["device"] = str(getattr(config, "device", "cpu"))
    d.pop("model_path", None)
    d.pop("model_name", None)
    d["details_path"] = DEFAULT_MODEL_JSON
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


def config_from_dict(d: dict, device: Optional[str] = None) -> Config:
    """只接受当前 Config 字段；旧 ckpt 多余键自动忽略。"""
    allowed = {f.name for f in fields(Config)}
    kwargs = {}
    for k, v in d.items():
        if k not in allowed:
            continue
        if k in ("conv_dilations", "adam_betas") and isinstance(v, list):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    cfg = Config(**kwargs)
    cfg.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.details_path = _ensure_details_path(cfg.details_path)
    return cfg


# ---- state_dict JSON 编解码：zlib + base64（仍是合法 JSON，体积约为 tolist 的 1/4~1/5）----

_TORCH_TO_NP = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def _encode_tensor(t: torch.Tensor) -> dict:
    """张量 -> JSON 可写结构：zlib 压缩后再 base64。"""
    t = t.detach().cpu().contiguous()
    dtype_name = str(t.dtype).replace("torch.", "")

    # numpy 无原生 bfloat16：落盘为 float32，加载时再转回 bfloat16（若原 dtype 如此）
    if t.dtype == torch.bfloat16:
        arr = t.float().numpy()
        np_dtype_name = "float32"
        torch_dtype_name = "bfloat16"
    else:
        np_dtype = _TORCH_TO_NP.get(dtype_name)
        if np_dtype is None:
            arr = t.numpy()
            np_dtype_name = str(arr.dtype)
        else:
            arr = t.numpy().astype(np_dtype, copy=False)
            np_dtype_name = dtype_name
        torch_dtype_name = dtype_name

    raw = np.ascontiguousarray(arr).tobytes()
    blob = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    return {
        "dtype": torch_dtype_name,
        "np_dtype": np_dtype_name,
        "shape": list(t.shape),
        "encoding": "zlib+base64",
        "data": blob,
    }


def _decode_tensor(meta: dict, map_location: str = "cpu") -> torch.Tensor:
    """兼容新旧两种格式：list（官方/旧稿）与 zlib+base64（新稿）。"""
    dtype_name = meta["dtype"]
    shape = meta["shape"]
    data = meta["data"]
    encoding = meta.get("encoding")

    # 新格式
    if isinstance(data, str) or encoding in ("zlib+base64", "base64"):
        raw = base64.b64decode(data)
        if encoding == "zlib+base64" or meta.get("compressed", True):
            # 无 encoding 字段但 data 是 str 时，默认按 zlib+base64
            if encoding in (None, "zlib+base64") or meta.get("compressed", False):
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    # 兼容仅 base64、未压缩
                    pass
        elif encoding == "base64":
            pass

        np_dtype_name = meta.get("np_dtype", dtype_name)
        if dtype_name == "bfloat16":
            np_dtype_name = meta.get("np_dtype", "float32")
        np_dtype = _TORCH_TO_NP.get(np_dtype_name, np.float32)
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
        t = torch.from_numpy(arr)
        if dtype_name == "bfloat16":
            t = t.to(torch.bfloat16)
        else:
            torch_dtype = getattr(torch, dtype_name, None)
            if torch_dtype is not None and t.dtype != torch_dtype:
                t = t.to(torch_dtype)
        return t.to(map_location)

    # 旧格式：扁平 list
    t = torch.tensor(data, dtype=getattr(torch, dtype_name))
    return t.reshape(shape).to(map_location)


def save_model(ckpt: dict, model_path: str = MODEL_PATH) -> str:
    mem_sd = peek_scale_details_cache(model_path)
    if mem_sd is not None:
        existing = {SCALE_DETAILS_KEY: mem_sd}
    else:
        existing = _load_existing_payload(model_path)
        nested = existing.get(SCALE_DETAILS_KEY)
        if isinstance(nested, dict) and nested:
            seed_scale_details_cache(model_path, nested)

    tensors = {}
    for k, v in ckpt["state_dict"].items():
        tensors[k] = _encode_tensor(v)

    payload = dict(existing)
    for k, v in ckpt.items():
        if k != "state_dict":
            payload[k] = v
    payload["state_dict"] = tensors

    if SCALE_DETAILS_KEY in ckpt:
        payload[SCALE_DETAILS_KEY] = ckpt[SCALE_DETAILS_KEY]
    elif SCALE_DETAILS_KEY not in payload and SCALE_DETAILS_KEY in existing:
        payload[SCALE_DETAILS_KEY] = existing[SCALE_DETAILS_KEY]

    os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    key = _payload_key(model_path)
    _PAYLOAD_MEM[key] = payload
    if isinstance(payload.get(SCALE_DETAILS_KEY), dict) and payload[SCALE_DETAILS_KEY]:
        seed_scale_details_cache(model_path, payload[SCALE_DETAILS_KEY])
    return model_path


def load_model(model_path: str = MODEL_PATH, map_location: str = "cpu") -> dict:
    payload = _load_existing_payload(model_path)
    if not payload:
        raise FileNotFoundError(f"无法读取模型终稿: {model_path}")
    sd_raw = payload.get("state_dict")
    if not isinstance(sd_raw, dict) or not sd_raw:
        raise FileNotFoundError(
            f"终稿无有效 state_dict: {model_path}; 请先 train_and_save(...) 并随 notebook 上传"
        )
    _seed_details_from_payload(model_path, payload)
    sd = {}
    for k, meta in sd_raw.items():
        sd[k] = _decode_tensor(meta, map_location=map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ======================================================================
# EMA / 优化器
# ======================================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow or not torch.is_floating_point(v):
                self.shadow[k] = v.detach().cpu().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach().cpu(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in self.shadow.items()}, strict=False)

    def state_dict(self) -> dict:
        return dict(self.shadow)


def build_param_groups(model: nn.Module, config: Config) -> list:
    """
    decay / level_proj 更大 wd / no_decay。
    exclude_head_wd=True 时，rank 头最后一层权重不加 decay（避免压扁自学尺度）。
    """
    exclude_head = bool(getattr(config, "exclude_head_wd", True))
    decay, level, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_out = exclude_head and name.endswith("rank_head.3.weight")
        if p.ndim == 1 or name.endswith(".bias") or is_out:
            no_decay.append(p)
        elif "level_proj" in name:
            level.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(config.weight_decay)},
        {"params": level, "weight_decay": float(config.level_branch_wd)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step: int):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _mean_info(infos: List[dict], key: str) -> float:
    vals = []
    for info in infos:
        v = info.get(key)
        if v is None:
            continue
        vals.append(float(v.detach().item()) if torch.is_tensor(v) else float(v))
    return float(sum(vals) / len(vals)) if vals else float("nan")


# ======================================================================
# Framework
# ======================================================================

class PonyFramework:
    def __init__(self, config: Config, model_path: str = MODEL_PATH):
        set_seed(getattr(config, "seed", 42))
        config.details_path = _ensure_details_path(config.details_path)
        self.config = config
        self.device = config.device
        self.model_path = model_path

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.ema = None
        self._use_bf16 = False
        self._autocast_dtype = torch.float32

        self.is_fitted = False
        self.best_params = None
        self.best_valid_loss = float("inf")
        self.best_valid_rankic = float("-inf")
        self.best_valid_proxy = float("-inf")
        self.start_epoch = 0

        self.train_result = {
            "train_loss": [],
            "valid_loss": [],
            "valid_rankic": [],
            "valid_proxy": [],
        }
        self.pred_result = {
            "pred_rank": [],
            "pred_ret": [],
            "score": [],
            "metadata": {"code": [], "date": [], "usable": []},
        }

    # ----- 构建 -----
    def _build_model(self, n_train_days: int = 0):
        if self.model is not None:
            return
        self.model = build_model(self.config).to(self.device)
        self.criterion = PonyOracleLoss(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, self.config),
            lr=self.config.learning_rate,
            betas=tuple(self.config.adam_betas),
        )

        K = max(int(self.config.meta_batch_k), 1)
        steps_per_epoch = max(n_train_days // K, 1) if n_train_days > 0 else 100
        total_steps = steps_per_epoch * max(int(self.config.epochs), 1)
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer, total_steps, float(self.config.warmup_ratio)
        )

        self.ema = (
            ModelEMA(self.model, decay=float(self.config.ema_decay))
            if bool(self.config.use_ema)
            else None
        )

        amp_ok = (
            bool(self.config.use_bf16)
            and str(self.device).startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        self._use_bf16 = amp_ok
        self._autocast_dtype = torch.bfloat16 if amp_ok else torch.float32

    def _autocast(self):
        return torch.autocast(
            device_type="cuda" if str(self.device).startswith("cuda") else "cpu",
            dtype=self._autocast_dtype,
            enabled=self._use_bf16,
        )

    @classmethod
    def from_json(cls, model_path: str = MODEL_PATH, device: str = None):
        map_loc = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_model(model_path, map_location=map_loc)
        config = config_from_dict(ckpt["config"], device=map_loc)
        config.details_path = os.path.abspath(model_path)
        inst = cls(config, model_path=model_path)
        inst._build_model()
        inst.model.load_state_dict(ckpt["state_dict"], strict=False)
        inst.is_fitted = True
        meta = ckpt.get("meta", {})
        inst.best_valid_loss = float(meta.get("best_valid_loss", float("inf")))
        inst.best_valid_rankic = float(meta.get("best_valid_rankic", float("-inf")))
        inst.best_valid_proxy = float(meta.get("best_valid_proxy", float("-inf")))
        print(f"------ 已从 JSON 加载模型: {model_path} ------")
        return inst

    def export_json(self, model_path: str = None, extra_meta: dict = None) -> str:
        if self.model is None:
            raise RuntimeError("模型未构建，无法导出")
        path = model_path or self.model_path
        if self.ema is not None:
            sd = self.ema.state_dict()
        elif self.best_params is not None:
            sd = self.best_params
        else:
            sd = self.model.state_dict()

        meta = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "framework": "PonyOracleEVO",
            "best_valid_loss": float(self.best_valid_loss),
            "best_valid_rankic": float(self.best_valid_rankic),
            "best_valid_proxy": float(self.best_valid_proxy),
            "n_params": int(self.model.count_trainable_params()),
            "pair_label": str(getattr(self.config, "pair_label", "rank")),
        }
        if extra_meta:
            meta.update(extra_meta)
        return save_model(
            {"state_dict": sd, "config": config_to_dict(self.config), "meta": meta},
            path,
        )

    def _save_best_json(self, epoch: int) -> None:
        if self.ema is not None:
            self.best_params = deepcopy(self.ema.state_dict())
        else:
            self.best_params = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
        path = self.export_json(extra_meta={"best_epoch": epoch + 1})
        print(
            f"  => 最优模型已更新: {path} "
            f"(proxy={self.best_valid_proxy:.6f}, "
            f"rankic={self.best_valid_rankic:.6f}, "
            f"loss={self.best_valid_loss:.6f})"
        )

    # ----- batch -----
    def _to_tensor_feature(self, feature: dict) -> dict:
        out = {}
        for k, v in feature["feature"].items():
            out[k] = {
                "vector": torch.as_tensor(v["vector"], dtype=torch.float32, device=self.device),
                "matrix": torch.as_tensor(v["matrix"], dtype=torch.float32, device=self.device),
            }
        return out

    @staticmethod
    def _as_bool(x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.bool)

    def _make_mask(self, feature: dict, label: dict = None) -> torch.Tensor:
        feat_u = self._as_bool(feature["usable"]).to(self.device)
        if label is None:
            return feat_u
        return feat_u & self._as_bool(label["usable"]).to(self.device)

    def _compose_score(
        self,
        pred_rank: torch.Tensor,
        pred_ret: torch.Tensor,
        aux: Optional[dict] = None,
    ) -> torch.Tensor:
        _ = aux
        beta = float(self.config.score_beta)
        return pred_rank + beta * pred_ret

    @staticmethod
    @torch.no_grad()
    def _cross_section_rank_ic(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        m = mask.bool()
        if int(m.sum().item()) < 2:
            return float("nan")
        p = pred[m].detach().float()
        t = target[m].detach().float()
        rp = p.argsort().argsort().to(torch.float32)
        rt = t.argsort().argsort().to(torch.float32)
        rp = rp - rp.mean()
        rt = rt - rt.mean()
        denom = rp.norm() * rt.norm()
        if float(denom.item()) < 1e-12:
            return float("nan")
        return float((rp * rt).sum().item() / denom.item())

    def _forward_batch(self, feature, label=None, return_aux: bool = False):
        feat_t = self._to_tensor_feature(feature)
        mask = self._make_mask(feature, label)
        if return_aux:
            pred_rank, pred_ret, aux = self.model(feat_t, mask, return_aux=True)
            return pred_rank, pred_ret, mask, aux
        pred_rank, pred_ret = self.model(feat_t, mask, return_aux=False)
        return pred_rank, pred_ret, mask

    def _pack_meta(self, feature, mask_np=None):
        codes = list(feature["code"])
        date = feature["date"]
        dates = [date] * len(codes)
        usable = list(feature["usable"]) if mask_np is None else mask_np.tolist()
        return codes, dates, usable

    def _build_loader(self, date_list, need_label=True) -> PrefetchDataLoader:
        return PrefetchDataLoader(
            date_list=date_list,
            seq_len=self.config.seq_len,
            details_path=self.config.details_path,
            mem_mode=self.config.mem_mode,
            prefetch_size=self.config.prefetch_size,
            shuffle=False,
            use_rolling=self.config.use_rolling,
            need_label=need_label,
        )

    def _swap_ema(self, enable: bool):
        """enable=True：备份当前权重并拷入 EMA；返回 backup。False 时还原。"""
        if self.ema is None:
            return None
        if enable:
            backup = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.ema.copy_to(self.model)
            return backup
        return None

    def _restore_backup(self, backup: Optional[dict]):
        if backup is None:
            return
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in backup.items()},
            strict=False,
        )

    # ----- 训练：K 天梯度累积（逐日反传，显存 O(1)）-----
    def train_epoch(self, loader: PrefetchDataLoader):
        self.model.train()
        K = max(int(self.config.meta_batch_k), 1)
        total_loss, n_steps = 0.0, 0
        sum_ic, sum_r = 0.0, 0.0
        buf, step_in_epoch = [], 0

        def _flush(batch_pairs):
            nonlocal total_loss, n_steps, sum_ic, sum_r, step_in_epoch
            if not batch_pairs:
                return
            self.optimizer.zero_grad(set_to_none=True)
            infos = []
            K_eff = len(batch_pairs)

            for feature, label in batch_pairs:
                y_rank = torch.as_tensor(
                    label["label"]["rank"], dtype=torch.float32, device=self.device
                )
                y_ret = torch.as_tensor(
                    label["label"]["return"], dtype=torch.float32, device=self.device
                )
                with self._autocast():
                    pred_rank, pred_ret, mask, aux = self._forward_batch(
                        feature, label, return_aux=True
                    )
                    loss_day, info = self.criterion.daily(
                        pred_rank.float(),
                        y_rank,
                        y_ret,
                        mask,
                        aux=aux,
                        pred_ret=pred_ret.float(),
                    )
                (loss_day / float(K_eff)).backward()
                infos.append(info)

            logs = {
                "loss": _mean_info(infos, "L_pair"),
                "ic_mean": _mean_info(infos, "ic"),
                "r_mean": _mean_info(infos, "r_ls"),
                "score_std": _mean_info(infos, "score_std"),
                "bound": _mean_info(infos, "bound"),
                "K": K_eff,
            }

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.criterion.step()

            total_loss += logs["loss"] if logs["loss"] == logs["loss"] else 0.0
            sum_ic += logs["ic_mean"] if logs["ic_mean"] == logs["ic_mean"] else 0.0
            sum_r += logs["r_mean"] if logs["r_mean"] == logs["r_mean"] else 0.0
            n_steps += 1
            step_in_epoch += 1

            if step_in_epoch % 10 == 0 or step_in_epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                L = logs["loss"]
                print(
                    f"    Step {str(step_in_epoch).zfill(4)}, "
                    f"Loss: {L:.6f} (ref={LOG2:.3f}), "
                    f"bound: {logs['bound']:.4f}, "
                    f"std(f): {logs['score_std']:.4f}, "
                    f"IC: {logs['ic_mean']:.6f}, "
                    f"LS: {logs['r_mean']:.6f}, "
                    f"K={K_eff}, lr={lr:.2e}"
                )

        for feature, label in loader:
            buf.append((feature, label))
            if len(buf) >= K:
                _flush(buf)
                buf = []
        if buf:
            _flush(buf)

        return (
            total_loss / max(n_steps, 1),
            sum_ic / max(n_steps, 1),
            sum_r / max(n_steps, 1),
        )

    # ----- 验证 -----
    def valid_epoch(self, loader: PrefetchDataLoader):
        backup = self._swap_ema(True)
        self.model.eval()
        total, n = 0.0, 0
        ics, rets, rank_ics = [], [], []

        with torch.no_grad():
            for idx, (feature, label) in enumerate(loader):
                y_rank = torch.as_tensor(
                    label["label"]["rank"], dtype=torch.float32, device=self.device
                )
                y_ret = torch.as_tensor(
                    label["label"]["return"], dtype=torch.float32, device=self.device
                )
                with self._autocast():
                    pred_rank, pred_ret, mask, aux = self._forward_batch(
                        feature, label, return_aux=True
                    )
                    loss, L_pair, bound = self.criterion(
                        pred_rank.float(), pred_ret.float(), y_rank, y_ret, mask, aux=aux
                    )

                score = self._compose_score(pred_rank, pred_ret, aux)
                ric = self._cross_section_rank_ic(score, y_ret, mask)
                ic_r = (
                    float(daily_ic(score.float(), y_rank, mask).item())
                    if int(mask.sum()) >= 2
                    else float("nan")
                )
                ls = hard_long_short_return(score, y_ret, mask, frac=0.1)

                if ric == ric:
                    rank_ics.append(ric)
                if ic_r == ic_r:
                    ics.append(ic_r)
                if ls == ls:
                    rets.append(ls)

                total += float(loss.item())
                n += 1
                if idx % 10 == 0:
                    ric_str = f"{ric:.6f}" if ric == ric else "nan"
                    print(
                        f"    Step {str(idx).zfill(4)}, "
                        f"Loss: {loss.item():.6f}, "
                        f"L_pair: {float(L_pair):.6f}, "
                        f"bound: {float(bound):.4f}, "
                        f"RankIC: {ric_str}"
                    )

        mean_rankic = float(np.nanmean(rank_ics)) if rank_ics else float("nan")
        proxy = proxy_score_from_series(ics, rets, regime_size=max(len(ics) // 4, 5))
        print(
            f"  [valid] RankIC={mean_rankic:.6f} | "
            f"IC_mean={proxy['IC_mean']:.6f} | IC_IR={proxy['IC_IR']:.6f} | "
            f"SR={proxy['SR']:.6f} | Stress={proxy['Stress']:.6f} | "
            f"Proxy={proxy['Proxy']:.6f} (n_days={len(ics)})"
        )
        self._restore_backup(backup)

        return (
            total / max(n, 1),
            mean_rankic,
            float(proxy["Proxy"]) if proxy["Proxy"] == proxy["Proxy"] else float("-inf"),
            proxy,
        )

    # ----- fit -----
    def fit(self, train_dates: list, valid_dates: list):
        get_standard_details(self.config.details_path)

        self.model = None
        self._build_model(n_train_days=len(train_dates))
        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        self.best_valid_rankic = float("-inf")
        self.best_valid_proxy = float("-inf")
        self.best_params = None
        stopping_count = 0

        print("\n------ 模型开始训练 (EVO) ------\n")
        print(f"参数更新至: {self.model_path}")
        print(
            f"pair_label={getattr(self.config, 'pair_label', 'rank')}, "
            f"meta_batch_k={self.config.meta_batch_k}, "
            f"revin={self.config.revin_mode}, "
            f"cs={self.config.cs_block}, "
            f"fusion={self.config.fusion_mode}, "
            f"scales={self.config.active_scales}, "
            f"bf16={self._use_bf16}, ema={self.ema is not None}\n"
        )

        for epoch in range(self.start_epoch, self.config.epochs):
            train_loader = self._build_loader(train_dates, need_label=True)
            valid_loader = self._build_loader(valid_dates, need_label=True)

            print("  ----> Train Process <----")
            train_loss, train_ic, train_r = self.train_epoch(train_loader)
            print("  ----> valid Process <----")
            valid_loss, valid_rankic, valid_proxy, _ = self.valid_epoch(valid_loader)

            self.train_result["train_loss"].append(train_loss)
            self.train_result["valid_loss"].append(valid_loss)
            self.train_result["valid_rankic"].append(valid_rankic)
            self.train_result["valid_proxy"].append(valid_proxy)
            print(
                f"Epoch {epoch + 1}/{self.config.epochs}, "
                f"Train Loss: {train_loss:.6f} (IC={train_ic:.6f}, LS={train_r:.6f}), "
                f"valid Loss: {valid_loss:.6f}, "
                f"valid RankIC: {valid_rankic:.6f}, "
                f"valid Proxy: {valid_proxy:.6f}"
            )

            metric = valid_proxy if valid_proxy == valid_proxy else valid_rankic
            best_metric = (
                self.best_valid_proxy
                if self.best_valid_proxy == self.best_valid_proxy
                else self.best_valid_rankic
            )
            if metric == metric and metric > best_metric:
                self.best_valid_proxy = valid_proxy
                self.best_valid_rankic = valid_rankic
                self.best_valid_loss = valid_loss
                self._save_best_json(epoch)
                stopping_count = 0
                print("早停计数重置：0\n")
            else:
                stopping_count += 1
                print(f"早停计数增加：{stopping_count}\n")

            if self.config.early_stopping and stopping_count >= self.config.early_patience:
                print("早停触发，停止训练。")
                break

        if self.best_params is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_params.items()},
                strict=False,
            )
            if self.ema is not None:
                self.ema.shadow = deepcopy(self.best_params)
        elif _model_has_weights(self.model_path):
            ckpt = load_model(self.model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            self.export_json(extra_meta={"best_epoch": -1, "note": "no_valid_improve"})

        self.is_fitted = True
        print(
            f"------ 模型训练就绪 | "
            f"best_proxy={self.best_valid_proxy:.6f} | "
            f"best_rankic={self.best_valid_rankic:.6f} | "
            f"best_loss={self.best_valid_loss:.6f} ------\n"
        )
        return (
            self.train_result["train_loss"],
            self.train_result["valid_loss"],
            self.train_result["valid_rankic"],
        )

    # ----- 预测 -----
    def predict(self, pred_dates: list):
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练完成，无法进行预测。")
        for k in ("pred_rank", "pred_ret", "score"):
            self.pred_result[k] = []
        self.pred_result["metadata"] = {"code": [], "date": [], "usable": []}

        backup = self._swap_ema(True)
        self.model.eval()
        loader = self._build_loader(pred_dates, need_label=False)

        with torch.no_grad():
            for feature, _label in loader:
                with self._autocast():
                    pred_rank, pred_ret, mask, aux = self._forward_batch(
                        feature, None, return_aux=True
                    )
                score = self._compose_score(pred_rank, pred_ret, aux)
                codes, dates, usable = self._pack_meta(feature, mask.cpu().numpy())
                self.pred_result["pred_rank"].append(pred_rank.float().cpu())
                self.pred_result["pred_ret"].append(pred_ret.float().cpu())
                self.pred_result["score"].append(score.float().cpu())
                self.pred_result["metadata"]["code"].extend(codes)
                self.pred_result["metadata"]["date"].extend(dates)
                self.pred_result["metadata"]["usable"].extend(usable)

        self._restore_backup(backup)

        def _cat(xs):
            return torch.cat(xs, dim=0).tolist()

        self.pred_result["pred_rank"] = _cat(self.pred_result["pred_rank"])
        self.pred_result["pred_ret"] = _cat(self.pred_result["pred_ret"])
        self.pred_result["score"] = _cat(self.pred_result["score"])
        return (
            self.pred_result["pred_rank"],
            self.pred_result["pred_ret"],
            self.pred_result["score"],
            self.pred_result["metadata"],
        )

    def predict_to_frame(self, start_date, end_date) -> pd.DataFrame:
        dates = build_date_list(start_date, end_date)
        if not dates:
            raise RuntimeError(f"预测区间无交易日: {start_date} ~ {end_date}")

        _, _, scores, meta = self.predict(dates)
        df = pd.DataFrame({
            "date": pd.to_datetime(meta["date"]),
            "instrument": meta["code"],
            "score": np.asarray(scores, dtype=np.float64),
            "usable": meta["usable"],
        })
        df = df[df["usable"]].drop(columns=["usable"])

        stk = dai.query(
            f"SELECT date, instrument FROM {instruments_table()}",
            filters={"date": [start_date, end_date]},
        ).df()
        stk["date"] = pd.to_datetime(stk["date"]).dt.normalize()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        return (
            pd.merge(df, stk, on=["date", "instrument"], how="inner")
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["score"])
            .drop_duplicates(["date", "instrument"])
            [["date", "instrument", "score"]]
            .reset_index(drop=True)
        )


# ======================================================================
# 入口
# ======================================================================

def default_config() -> Config:
    cfg = Config()
    cfg.details_path = _ensure_details_path(cfg.details_path)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def train_and_save(
    datasources,
    model_path: str = MODEL_PATH,
    force_retrain: bool = False,
) -> str:
    ds = set_datasources(datasources)
    print(f"datasources={ds}")

    if _model_has_weights(model_path) and not force_retrain:
        print(f"已发现完整模型权重，跳过训练: {model_path}")
        return model_path

    set_seed(42)
    config = default_config()
    config.details_path = os.path.abspath(model_path)
    fw = PonyFramework(config, model_path=model_path)

    train_dates = build_date_list(TRAIN_START, TRAIN_END)
    valid_dates = build_date_list(VALID_START, VALID_END)
    if not train_dates or not valid_dates:
        raise RuntimeError("训练/验证交易日为空，请检查 TRAIN_*/VALID_* 常量")

    print(f"train days={len(train_dates)} ({train_dates[0]}→{train_dates[-1]})")
    print(f"valid days={len(valid_dates)} ({valid_dates[0]}→{valid_dates[-1]})")

    fw.fit(train_dates, valid_dates)
    path = fw.export_json()
    print(f"模型已保存(JSON)，请随 notebook 一并上传: {path}")
    return path


def predict_scores(datasources, start_date, end_date, model_path: str = MODEL_PATH) -> pd.DataFrame:
    set_datasources(datasources)
    fw = PonyFramework.from_json(model_path)
    return fw.predict_to_frame(start_date, end_date)
