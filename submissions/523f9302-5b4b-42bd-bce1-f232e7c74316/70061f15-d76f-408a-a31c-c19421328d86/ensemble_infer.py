"""plan5_public_v3 ensemble: per-member daily cs-zscore → equal mean."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_MAE_FAMILIES = frozenset({"mae", "mae_size_ind"})


def cs_zscore_1d(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    mu = float(np.nanmean(s))
    sd = float(np.nanstd(s))
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(s, dtype=np.float64)
    return (s - mu) / sd


def load_members_manifest(root: Path | str | None = None) -> dict:
    base = Path(root) if root is not None else _HERE
    return json.loads((base / "members.json").read_text(encoding="utf-8"))


def list_member_json_paths(root: Path | str | None = None) -> list[Path]:
    base = Path(root) if root is not None else _HERE
    man = load_members_manifest(base)
    paths = []
    for m in man.get("kept", []):
        p = base / m["json_name"]
        if p.is_file():
            paths.append(p)
    return paths


def decode_state_dict(encoded: dict, map_location: str = "cpu") -> dict:
    sd = {}
    for k, spec in encoded.items():
        dtype = getattr(torch, spec["dtype"])
        t = torch.tensor(spec["data"], dtype=dtype).reshape(spec["shape"])
        sd[k] = t.to(map_location)
    return sd


def is_mae_family(family: str, model_type: str = "") -> bool:
    return family in _MAE_FAMILIES or model_type == "mae_gru_readout_full"


def build_model_from_payload(payload: dict, device: torch.device):
    """Build silu_res2 / fixedop (M3) or MAE / mae_size_ind from a member JSON."""
    from models_m3 import TemporalConvGRUModel
    from mae_train import MAEEncoderGRUReadoutFull

    cfg = dict(payload["model_cfg"])
    family = str(payload.get("family", ""))
    model_type = str(payload.get("model_type", ""))
    if is_mae_family(family, model_type):
        model = MAEEncoderGRUReadoutFull(**cfg)
    else:
        model = TemporalConvGRUModel(**cfg)
    model.load_state_dict(decode_state_dict(payload["state_dict"]), strict=True)
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta


def load_ensemble_members(
    root: Path | str | None = None,
    device: torch.device | None = None,
) -> list[tuple[object, dict]]:
    device = device or torch.device("cpu")
    base = Path(root) if root is not None else _HERE
    man = load_members_manifest(base)
    out: list[tuple[object, dict]] = []
    for rec in man.get("kept", []):
        path = base / rec["json_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("pack_mode", rec.get("pack_mode"))
        payload.setdefault("feat_norm", rec.get("feat_norm"))
        payload.setdefault("family", rec.get("family"))
        model, meta = build_model_from_payload(payload, device)
        meta["json_path"] = str(path)
        out.append((model, meta))
    if not out:
        raise FileNotFoundError(f"no kept members under {base}")
    return out


def ensemble_day_scores(member_scores: list[np.ndarray]) -> np.ndarray:
    """Each entry is 1d scores for the same instruments that day.

    Apply cs-zscore per member, then equal-weight mean.
    """
    if not member_scores:
        raise ValueError("empty member_scores")
    zs = [cs_zscore_1d(s) for s in member_scores]
    return np.mean(np.stack(zs, axis=0), axis=0)


def ensemble_score_frames(
    frames: list[pd.DataFrame],
    *,
    date_col: str = "date",
    instrument_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """frames: one DataFrame per member with date/instrument/score."""
    if not frames:
        raise ValueError("empty frames")
    keys = [date_col, instrument_col]
    out = frames[0][keys].copy()
    zcols = []
    for i, df in enumerate(frames):
        sub = df[keys + [score_col]].copy()
        sub[score_col] = sub.groupby(date_col, sort=False)[score_col].transform(
            lambda s: cs_zscore_1d(s.to_numpy())
        )
        col = f"z_{i}"
        sub = sub.rename(columns={score_col: col})
        out = out.merge(sub, on=keys, how="inner")
        zcols.append(col)
    out[score_col] = out[zcols].mean(axis=1)
    return out[keys + [score_col]]
