"""Support/FeatureCal.py — 从 Store/DAI 组装单日 batch 特征。"""

import math
import time
from typing import Optional

import numpy as np
import polars as pl

from Support_BaseData import get_base_data, get_standard_details, DEFAULT_MODEL_JSON
from Support_BaseSets import BatchFeature
from Support_GetDates import get_last_days
from Support_RollStore import RollingBaseStore

DAYS_DICT = {
    "1m": 240,
    "5m": 48,
    "15m": 16,
    "30m": 8,
}

SCALE_MAP = {
    "m01": "1m",
    "m05": "5m",
    "m15": "15m",
    "m30": "30m",
}

SCALAR_COLS = [
    "pre_close", "high", "open", "low", "close",
    "adjust_factor", "deal_number", "volume", "amount",
    "ask_price1", "ask_price2", "ask_volume1", "ask_volume2",
    "bid_price1", "bid_price2", "bid_volume1", "bid_volume2",
]

ASK_PRICE_COLS = [f"ask_price{i}" for i in "12345"]
BID_PRICE_COLS = [f"bid_price{i}" for i in "12345"]
ASK_VOLUME_COLS = [f"ask_volume{i}" for i in "12345"]
BID_VOLUME_COLS = [f"bid_volume{i}" for i in "12345"]
MATRIX_COLS = ASK_PRICE_COLS + BID_PRICE_COLS + ASK_VOLUME_COLS + BID_VOLUME_COLS
NEED_COLS = ["date", "instrument"] + SCALAR_COLS + [
    c for c in MATRIX_COLS if c not in SCALAR_COLS
]


def get_batch_feature(
    date: str = "2020-01-02",
    seq_len: int = 240,
    stock_list: list[str] = [],
    code_to_idx: dict[str, int] = {},
    details_path: str = DEFAULT_MODEL_JSON,
    scale_details: Optional[dict] = None,
    store: Optional[RollingBaseStore] = None,
):
    """指定日期、序列长度，获取单 batch 的特征。

    store 非空时：wait_ready 后只从内存 get_window 拼接；不在此拉 DAI BASE。
    """
    ScaleDetails = scale_details or get_standard_details(details_path=details_path)

    # ---- 等 BASE 就绪（不计入 CAL FEATURE）----
    wait_s = 0.0
    if store is not None:
        t_wait = time.perf_counter()
        # 双流水线：wait_ready；若你仍是旧 API，改成 store.ensure(date, stock_list)
        if hasattr(store, "wait_ready"):
            store.wait_ready(date)
        else:
            store.ensure(date, stock_list)
        wait_s = time.perf_counter() - t_wait

    batch_feature: BatchFeature = {
        "date": date,
        "code": stock_list,
        "usable": np.ones(len(stock_list), dtype=bool),
        "feature": {
            k: {
                "vector": np.zeros((len(stock_list), seq_len, 17), dtype=np.float32),
                "matrix": np.zeros((len(stock_list), seq_len, 4, 5), dtype=np.float32),
            }
            for k in SCALE_MAP
        },
    }

    def _load_scale_df(target_delta: str) -> pl.DataFrame:
        if store is not None:
            return (
                store.get_window(target_delta, date, stock_list)
                .select(NEED_COLS)
                .sort("instrument", "date")
                .group_by("instrument", maintain_order=True)
                .tail(seq_len)
            )

        N_days = math.ceil(seq_len / DAYS_DICT[target_delta])
        date_list = get_last_days(date=date, N=N_days)
        return (
            get_base_data(
                target_delta=target_delta,
                start_date=date_list[0],
                end_date=date,
                ScaleDetails=ScaleDetails,
                pools=stock_list,
                fill=True,
            )
            .select(NEED_COLS)
            .sort("instrument", "date")
            .group_by("instrument", maintain_order=True)
            .tail(seq_len)
        )

    def build_one_scale(target_delta: str):
        vector = np.zeros((len(stock_list), seq_len, 17), dtype=np.float32)
        matrix = np.zeros((len(stock_list), seq_len, 4, 5), dtype=np.float32)
        usable = np.zeros(len(stock_list), dtype=bool)

        df = _load_scale_df(target_delta)
        if df.is_empty():
            return vector, matrix, usable, 0

        ok_codes = (
            df.group_by("instrument", maintain_order=True)
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") == seq_len)["instrument"]
        )
        df = df.filter(pl.col("instrument").is_in(ok_codes))
        codes = df["instrument"].unique(maintain_order=True).to_list()
        ncode = len(codes)
        if ncode == 0:
            return vector, matrix, usable, 0

        vec_arr = (
            df.select(SCALAR_COLS).to_numpy().astype(np.float32, copy=False)
            .reshape(ncode, seq_len, 17)
        )
        ask_p = df.select(ASK_PRICE_COLS).to_numpy().astype(np.float32, copy=False).reshape(ncode, seq_len, 5)
        bid_p = df.select(BID_PRICE_COLS).to_numpy().astype(np.float32, copy=False).reshape(ncode, seq_len, 5)
        ask_v = df.select(ASK_VOLUME_COLS).to_numpy().astype(np.float32, copy=False).reshape(ncode, seq_len, 5)
        bid_v = df.select(BID_VOLUME_COLS).to_numpy().astype(np.float32, copy=False).reshape(ncode, seq_len, 5)
        mat_arr = np.stack([ask_p, bid_p, ask_v, bid_v], axis=2)

        idx = np.fromiter((code_to_idx[c] for c in codes), dtype=np.int64, count=ncode)
        vector[idx] = vec_arr
        matrix[idx] = mat_arr
        usable[idx] = True
        return vector, matrix, usable, ncode

    # ---- 从内存提取并拼接 FEATURE（计入 CAL FEATURE）----
    t_feat = time.perf_counter()
    scale_secs: dict[str, float] = {}
    usable_n = 0

    for scale_key, delta in SCALE_MAP.items():
        t_s = time.perf_counter()
        vec, mat, usable_scale, ncode = build_one_scale(delta)
        scale_secs[delta] = time.perf_counter() - t_s
        batch_feature["feature"][scale_key]["vector"] = vec
        batch_feature["feature"][scale_key]["matrix"] = mat
        batch_feature["usable"] &= usable_scale
        usable_n = int(usable_scale.sum())  # 最后一尺度的 usable 数仅作参考；下面用最终 usable

    batch_feature["usable"] = batch_feature["usable"].tolist()
    feat_s = time.perf_counter() - t_feat
    n_usable = int(sum(batch_feature["usable"]))

    # # 手动注释：与 [DAI BASE] / [DAI LABEL] 同风格
    # print(
    #     f"  [CAL FEATURE] date={date} codes={len(stock_list)} "
    #     f"usable={n_usable} "
    #     f"wait_ready={wait_s:.3f}s "
    #     f"seconds={feat_s:.3f}s "
    #     f"(1m={scale_secs.get('1m', 0):.3f} "
    #     f"5m={scale_secs.get('5m', 0):.3f} "
    #     f"15m={scale_secs.get('15m', 0):.3f} "
    #     f"30m={scale_secs.get('30m', 0):.3f})"
    # )

    return batch_feature
