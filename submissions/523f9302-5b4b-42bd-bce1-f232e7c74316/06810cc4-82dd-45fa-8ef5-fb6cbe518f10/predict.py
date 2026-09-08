# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端赛道 — PatchTCN 推理入口。

只输出 date / instrument / score 三列（[介绍] L42-51）。
所有预处理、窗口构造与模型定义都从 train.py 导入，确保训练、本地验证与云端推理
走的是同一套实现（任务四.5）。
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import torch

import train as _train
from train import (
    DEFAULT_WEIGHTS,
    build_day_table,
    build_windows,
    create_model,
    dai,            # reuse train.py's import/fallback so the universe query works
    iter_date_chunks,
    load_weights,
    query_bars,
    to_canonical,
    transform_standardizer,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
INFER_BATCH = 256

# Contest universe is the CSI-1000 constituents at each point in time
# (比赛介绍: 股票池 = 中证 1000 指数在历史相应时间点上的成分股).
# The cloud bar table carries ~2180 names/day, so more than half of an unfiltered
# score file falls outside the graded universe. Those rows earn nothing and they
# also set the 1%/99% winsorize cutoffs the platform applies, which shifts the
# ranks of the names that are actually graded. Measured 2024-03..04: IC was
# 0.0931 over all scored names but 0.0794 inside the universe.
UNIVERSE_TABLE = "bigalpha_2026_instruments"


def _universe(start_date, end_date):
    """(date, instrument) pairs of the contest universe, or None if unavailable."""
    try:
        u = dai.query(
            f"SELECT date, instrument FROM {UNIVERSE_TABLE}",
            filters={"date": [f"{str(start_date)[:10]} 00:00:00",
                              f"{str(end_date)[:10]} 23:59:59"]},
        ).df()
    except Exception as e:  # pragma: no cover - keep scoring rather than fail hard
        print(f"[predict] universe table unavailable ({e}); emitting all names", flush=True)
        return None
    if u is None or u.empty:
        print("[predict] universe table empty; emitting all names", flush=True)
        return None
    u["date"] = pd.to_datetime(u["date"]).dt.normalize()
    u["instrument"] = u["instrument"].astype(str)
    return u.drop_duplicates(["date", "instrument"])


def _resolve_weights(model_path: str | None, table: str | None = None) -> str:
    """Find the weight file, or train one, and say clearly what happened.

    Two different situations produce the same missing-weights symptom on the
    private board and they need different fixes, so the search order and the
    outcome are both logged:

      * the platform hands us an environment and expects this package to train
        (public weights are explicitly excluded from private scoring), or
      * the platform ran the training itself and it failed, leaving nothing.

    The file is looked up in several plausible locations because the working
    directory when notebook code is extracted and executed need not match where
    the sources were placed.
    """
    name = os.path.basename(DEFAULT_WEIGHTS)
    candidates = []
    if model_path:
        candidates.append(model_path)
    candidates += [
        DEFAULT_WEIGHTS,
        os.path.join(_HERE, name),
        os.path.join(os.getcwd(), name),
        name,
    ]
    seen, ordered = set(), []
    for c in candidates:
        a = os.path.abspath(c)
        if a not in seen:
            seen.add(a)
            ordered.append(a)

    for c in ordered:
        if os.path.exists(c):
            print(f"[predict] weights found: {c}", flush=True)
            return c

    # Accept browser-style duplicate names. Downloading a file into a folder that
    # already holds one yields "model_weights(1).json"; uploading that verbatim
    # leaves the expected name missing even though the weights are present.
    stem, ext = os.path.splitext(name)
    dup = re.compile(rf"^{re.escape(stem)}\s*\(\d+\)\s*{re.escape(ext)}$", re.I)
    loose = re.compile(rf"^{re.escape(stem)}.*{re.escape(ext)}$", re.I)
    for d in (_HERE, os.getcwd()):
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            continue
        for pattern in (dup, loose):
            for f in entries:
                if pattern.match(f):
                    found = os.path.join(d, f)
                    print(f"[predict] using '{f}' (expected '{name}'); "
                          f"rename it before submitting to avoid relying on this fallback",
                          flush=True)
                    return found

    print("[predict] no weights in any candidate location:", flush=True)
    for c in ordered:
        print(f"[predict]   miss {c}", flush=True)
    print(f"[predict] cwd={os.getcwd()}  _HERE={_HERE}", flush=True)
    try:
        print(f"[predict] cwd contents: {sorted(os.listdir(os.getcwd()))[:20]}", flush=True)
        print(f"[predict] _HERE contents: {sorted(os.listdir(_HERE))[:20]}", flush=True)
    except OSError:
        pass

    target = ordered[0]
    # bar1m resolves to different physical tables on the public and private
    # boards, so the caller's datasources value is used; falling back to the
    # constant would read the wrong table during a private retrain.
    train_table = table or _train.TABLE_CLOUD
    print(f"[predict] training from scratch on table={train_table} -> {target}", flush=True)

    _train.train_and_save(
        table=train_table,
        save_path=target,
        start=_train.TRAIN_START,
        end=_train.TRAIN_END,
        is_local=False,
    )
    if not os.path.exists(target):
        raise FileNotFoundError(f"训练结束但未写出权重: {target}")
    print(f"[predict] training complete: {target}", flush=True)
    return target


def _load(model_path: str, device: torch.device):
    """Load weights and adopt the look-back they were trained with.

    The look-back is a property of the checkpoint, not of this file: a model
    trained on 60 sessions must be served with a 60-session window and a matching
    history buffer, otherwise the windows are wrong or empty. train.py's module
    constants are updated so build_windows sees the same values training used.
    """
    ckpt = load_weights(model_path, map_location=device)

    w = int(ckpt.get("window_days", _train.WINDOW_DAYS))
    if w != _train.WINDOW_DAYS:
        print(f"[predict] checkpoint look-back {w} sessions "
              f"(file default {_train.WINDOW_DAYS}); adopting it", flush=True)
    _train.WINDOW_DAYS = w
    _train.MAX_WINDOW_SPAN_DAYS = _train.span_cap_for(w)
    hist = _train.history_days_for(w)
    print(f"[predict] window={w} span_cap={_train.MAX_WINDOW_SPAN_DAYS} "
          f"history_days={hist}", flush=True)

    model = create_model(
        n_layers=int(ckpt.get("n_layers", _train.N_LAYERS)),
        d_model=int(ckpt.get("d_model", _train.D_MODEL)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = np.asarray(ckpt["mean"], np.float32)
    std = np.asarray(ckpt["std"], np.float32)
    return model, mean, std, hist


def _score(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), INFER_BATCH):
            xb = torch.from_numpy(X[i:i + INFER_BATCH]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty(0, np.float32)


def main(datasources, start_date, end_date, model_path=None):
    """平台入口：返回 date / instrument / score。

    datasources 为平台传入的数据源映射；缺省回退到云端 1m 表。
    按短日期块查询，每块额外取 history_days 的前置数据以凑满回看窗口。
    """
    # Resolve the table first: a retrain triggered below needs it, and bar1m
    # points at different physical tables on the public and private boards.
    if isinstance(datasources, dict):
        table = datasources.get("bar1m") or next(iter(datasources.values()))
    else:
        table = datasources
    print(f"[predict] table={table}", flush=True)

    model_path = _resolve_weights(model_path, table=table)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, history_days = _load(model_path, device)

    frames = []
    for q_start, q_end, target_start in iter_date_chunks(
        str(start_date)[:10], str(end_date)[:10], history_days=history_days
    ):
        raw = query_bars(table, q_start, q_end)
        if raw.empty:
            continue
        can = to_canonical(raw, is_local=False)
        can["date"] = pd.to_datetime(can["date"])
        day_table = build_day_table(can, "instrument")
        if day_table.empty:
            continue

        target_days = sorted(
            d for d in day_table["day"].unique()
            if pd.Timestamp(target_start) <= pd.Timestamp(d) <= pd.Timestamp(q_end)
        )
        if not target_days:
            continue
        try:
            X, keys = build_windows(day_table, "instrument", target_days=target_days)
        except RuntimeError:
            continue

        X = transform_standardizer(X, mean, std)
        keys = keys.assign(score=_score(model, X, device).astype("float64"))
        frames.append(keys)
        print(f"[predict] {target_start}~{q_end} rows={len(keys)}", flush=True)

    if not frames:
        return pd.DataFrame(columns=["date", "instrument", "score"])

    out = pd.concat(frames, ignore_index=True)
    out = (
        out.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"], keep="last")
        .loc[:, ["date", "instrument", "score"]]
    )
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str)

    universe = _universe(start_date, end_date)
    if universe is not None:
        before = len(out)
        kept = out.merge(universe, on=["date", "instrument"], how="inner")
        # Only accept the filter if it retains a plausible cross-section; a join
        # that collapses the file (date-type or calendar mismatch) must not be
        # allowed to produce an invalid submission.
        if len(kept) >= 0.2 * before and kept["date"].nunique() == out["date"].nunique():
            print(f"[predict] universe filter: {before} -> {len(kept)} rows "
                  f"({len(kept)/max(out['date'].nunique(),1):.0f}/day)", flush=True)
            out = kept
        else:
            print(f"[predict] universe filter rejected (would keep {len(kept)}/{before} rows, "
                  f"{kept['date'].nunique()}/{out['date'].nunique()} days); emitting all names",
                  flush=True)

    return out.sort_values(["date", "instrument"], ignore_index=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--table", default="bigalpha_2026_stock_bar1m")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    df = main({"bar1m": a.table}, a.start, a.end, model_path=a.weights)
    print(df.head())
    print("rows", len(df))
