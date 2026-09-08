from collections import defaultdict
from itertools import product
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import time
import json
import dai

from Support_BaseSets import StandardMethod
from Support_GetDates import get_calendar, get_basetime

VALUES_COLS = list(StandardMethod.keys())
NEEDED_COLS_SQL = ["date", "instrument"] + VALUES_COLS

DEFAULT_MODEL_JSON = "transformer_model.json"
SCALE_DETAILS_KEY = "scale_details"
_LEGACY_DELTAS = ("1m", "5m", "15m", "30m")

# 进程内缓存：key = 终稿绝对路径，value = 已 normalize 的 scale_details
_SCALE_DETAILS_MEM: dict[str, dict] = {}

# ---------- datasources：平台注入表名，默认与官方公榜一致 ----------
DEFAULT_DATASOURCES = {
    "bar1m": "bigalpha_2026_stock_bar1m",
    "bar5m": "bigalpha_2026_stock_bar5m",
    "bar15m": "bigalpha_2026_stock_bar15m",
    "bar30m": "bigalpha_2026_stock_bar30m",
    "instruments": "bigalpha_2026_instruments",
}

_datasources: dict = dict(DEFAULT_DATASOURCES)


def set_datasources(datasources: dict | None = None) -> dict:
    """合并平台注入的表名；未提供的 key 保留默认。"""
    global _datasources
    merged = dict(DEFAULT_DATASOURCES)
    if datasources:
        merged.update({k: v for k, v in datasources.items() if v})
    _datasources = merged
    return dict(_datasources)


def get_datasources() -> dict:
    return dict(_datasources)


def bar_table(delta: str) -> str:
    """delta: '1m'/'5m'/'15m'/'30m' → datasources['bar1m'] 等。"""
    key = f"bar{delta}"
    if key not in _datasources:
        raise KeyError(f"datasources 缺少 {key}，当前 keys={list(_datasources)}")
    return _datasources[key]


def instruments_table() -> str:
    return _datasources.get("instruments", DEFAULT_DATASOURCES["instruments"])


def get_batch_codes(date: str) -> tuple[list[str], dict[str, int]]:
    """ 返回指定日期 / batch 的股票池及其索引映射 """

    pools = dai.query(f"SELECT * FROM {instruments_table()}", filters={
        "date": [date, date]
    }).pl().select(
        pl.col("instrument").unique()
    ).sort("instrument")['instrument'].to_list()

    return pools, {c: i for i, c in enumerate(pools)}


def _resolve_model_json_path(details_path: str | None = None) -> Path:
    """相对路径锚定到本包根目录；默认 transformer_model.json。"""
    name = details_path or DEFAULT_MODEL_JSON
    path = Path(name)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _cache_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _copy_scale_details(details: dict) -> dict:
    """浅拷贝，避免调用方原地改坏缓存。"""
    return {
        delta: {col: [float(m), float(s)] for col, (m, s) in cols.items()}
        for delta, cols in details.items()
    }


def seed_scale_details_cache(
    details_path: str | None,
    scale_details: dict,
) -> dict:
    """把已解析的 scale_details 写入进程缓存（供 load_model / 外部注入）。"""
    path = _resolve_model_json_path(details_path)
    norm = _normalize_scale_details(scale_details)
    _SCALE_DETAILS_MEM[_cache_key(path)] = norm
    return _copy_scale_details(norm)


def peek_scale_details_cache(details_path: str | None = None) -> dict | None:
    """查看进程缓存；未命中返回 None（不读盘）。"""
    path = _resolve_model_json_path(details_path)
    hit = _SCALE_DETAILS_MEM.get(_cache_key(path))
    return None if hit is None else _copy_scale_details(hit)


def clear_scale_details_cache(details_path: str | None = None) -> None:
    """清空全部或指定路径的进程缓存。"""
    if details_path is None:
        _SCALE_DETAILS_MEM.clear()
        return
    path = _resolve_model_json_path(details_path)
    _SCALE_DETAILS_MEM.pop(_cache_key(path), None)


def _load_json_payload(path: Path) -> dict | None:
    """读取 JSON；不存在 / 空白 / 非法 → None。"""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _normalize_scale_details(raw: dict) -> dict:
    return {
        delta: {col: [float(m), float(s)] for col, (m, s) in cols.items()}
        for delta, cols in raw.items()
    }


def _extract_scale_details(payload: dict) -> dict | None:
    """从终稿提取 scale_details；兼容旧独立 ScaleDetails 顶层结构。"""
    if not payload:
        return None
    nested = payload.get(SCALE_DETAILS_KEY)
    if isinstance(nested, dict) and nested:
        return nested
    # 兼容：整文件就是旧版 Support_ScaleDetails.json
    if all(d in payload for d in _LEGACY_DELTAS) and all(
        isinstance(payload[d], dict) for d in _LEGACY_DELTAS
    ):
        return {d: payload[d] for d in _LEGACY_DELTAS}
    return None


def _compute_scale_details() -> dict:
    """原逻辑：全量扫描生成标准化 mean/std。"""
    ScaleDetails: dict = defaultdict(dict)

    for delta in ["1m", "5m", "15m", "30m"]:
        table = bar_table(delta)
        for col, method in list(StandardMethod.items()):
            print(f"处理 {delta} 数据")
            v = f"(CASE WHEN {col} < 0 THEN 0.0 ELSE {col} END)"
            x = v if method == "direct" else f"ln(1.0 + {v})"

            row = (
                dai.query(
                    f"SELECT AVG({x}) AS mean, STDDEV_POP({x}) AS std "
                    f"FROM {table}",
                    filters={"date": ["2020-01-01 00:00:00", "2021-12-31 23:59:59"]},
                ).pl().to_dicts()[0]
            )

            ScaleDetails[delta][col] = [
                float(row["mean"]), max(float(row["std"]), 1e-12),
            ]

    return {k: dict(v) for k, v in ScaleDetails.items()}


def _dump_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # 终稿可能含巨大 state_dict，不加 indent
        json.dump(payload, f, ensure_ascii=False)


def get_standard_details(
    details_path: str = DEFAULT_MODEL_JSON,
):
    """获取标准化参数。

    写入目标为终稿 JSON（默认 transformer_model.json）顶层键 scale_details。
    - 进程内已缓存：直接返回（不读盘）
    - 文件不存在 / 空白 / 无有效 scale_details：重算并增量写回（保留其余字段）
    - 已存在有效 scale_details：读取后写入进程缓存再返回
    """
    path = _resolve_model_json_path(details_path)
    key = _cache_key(path)

    hit = _SCALE_DETAILS_MEM.get(key)
    if hit is not None:
        return _copy_scale_details(hit)

    payload = _load_json_payload(path) or {}
    cached = _extract_scale_details(payload)

    if cached is not None:
        norm = _normalize_scale_details(cached)
        _SCALE_DETAILS_MEM[key] = norm
        return _copy_scale_details(norm)

    print(f"无有效标准化缓存（{path} 缺失/空白/无 scale_details），正在生成...")
    scale = _compute_scale_details()
    payload[SCALE_DETAILS_KEY] = scale
    # 若整文件曾是旧版顶层 ScaleDetails，清掉冲突的顶层 delta 键
    for d in _LEGACY_DELTAS:
        if d in payload and d != SCALE_DETAILS_KEY:
            # 仅当该键看起来是旧 scale 列字典时移除
            if isinstance(payload.get(d), dict) and SCALE_DETAILS_KEY in payload:
                payload.pop(d, None)
    _dump_payload(path, payload)
    print(f"标准化参数已写入终稿: {path}  [{SCALE_DETAILS_KEY}]")

    norm = _normalize_scale_details(scale)
    _SCALE_DETAILS_MEM[key] = norm
    return _copy_scale_details(norm)


def _check_base_frame(
    df: pl.DataFrame,
    target_delta: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """区间内与标准交易时间轴严格对齐；不完整股票整段剔除。"""
    date_list = (
        get_calendar()
        .filter(
            (pl.col("date") >= pl.lit(start_date).str.to_date("%Y-%m-%d"))
            & (pl.col("date") <= pl.lit(end_date).str.to_date("%Y-%m-%d"))
        )
        .select(pl.col("date").cast(pl.String))["date"]
        .to_list()
    )

    if not date_list:
        return df.head(0)

    expected = pl.concat([get_basetime(d, target_delta) for d in date_list]).unique().sort("date")
    n_exp = expected.height
    if df.is_empty() or n_exp == 0:
        return df.head(0)

    ok = (
        df.group_by("instrument")
        .agg(pl.len().alias("n_all"))
        .join(
            df.join(expected, on="date", how="inner")
            .group_by("instrument").agg(pl.len().alias("n_hit")),
            on="instrument",
            how="inner",
        )
        .filter((pl.col("n_all") == n_exp) & (pl.col("n_hit") == n_exp))
        .select("instrument")
    )

    return (
        df.join(ok, on="instrument", how="inner")
        .join(expected, on="date", how="inner")
        .sort("instrument", "date")
    )


def _standardize_keepnull(
    df: pl.DataFrame,
    target_delta: str,
    ScaleDetails: dict,
) -> pl.DataFrame:
    """
    clip → log1p → zscore，保留 null（不在此处填充）
    """

    if df.is_empty():
        return df

    return (
        df.select(
            pl.col("date"),
            pl.col("instrument"),
            *[
                (
                    pl.col(col_name).clip(lower_bound=0)
                    .fill_nan(None).cast(pl.Float64)
                ).alias(col_name)
                for col_name in VALUES_COLS
            ],
        )
        .with_columns(
            [
                pl.col(col_name).log1p().alias(col_name)
                if method == "log1p" else pl.col(col_name)
                for col_name, method in StandardMethod.items()
            ]
        )
        .with_columns(
            [
                (
                    (pl.col(col_name) - ScaleDetails[target_delta][col_name][0])
                    / ScaleDetails[target_delta][col_name][1]
                ).alias(col_name)
                for col_name in VALUES_COLS
            ]
        )
        .with_columns(pl.col("date").dt.date().cast(pl.String).alias("date_day"))
    )


def _apply_forward_fill(df: pl.DataFrame) -> pl.DataFrame:
    """标准化之后：按 instrument 时间序前向填充，剩余填 0（标准化后均值）"""

    if df.is_empty():
        return df

    return (
        df.sort("instrument", "date")
        .with_columns([pl.col(c).forward_fill().over("instrument") for c in VALUES_COLS])
        .with_columns([pl.col(c).fill_null(0.0) for c in VALUES_COLS])
    )


def get_base_data(
    target_delta: str,
    start_date: str,
    end_date: str,
    ScaleDetails: dict,
    pools: list[str] | None = None,
    fill: bool = True,
    check: bool = True,
):
    """
    返回指定频率、区间内的基础数据。
    - pools=None/[] : 不按股票筛选（全市场）
    - check=True    : 走原有 _check_base_frame（n_all == n_hit == n_exp）
    - check=False   : 跳过完整性剔除（供 RollingBaseStore 大块预取；
                      完整性改在 get_window 对回看窗执行）
    - fill=True     : 标准化后 ffill -> 0
    - fill=False    : 仅标准化，保留 null
    """
    filters: dict = {
        "date": [f"{start_date} 00:00:00", f"{end_date} 23:59:59"],
    }
    if pools:
        filters["instrument"] = pools

    table = bar_table(target_delta)
    q0 = time.perf_counter()
    raw = (
        dai.query(
            f"SELECT * FROM {table}",
            filters=filters,
        ).pl().with_columns(pl.col("date").cast(pl.Datetime))
    )
    dai_s = time.perf_counter() - q0

    t1 = time.perf_counter()
    df = _check_base_frame(raw, target_delta, start_date, end_date) if check else raw
    df = _standardize_keepnull(df, target_delta, ScaleDetails)

    if fill:
        df = _apply_forward_fill(df)
    post_s = time.perf_counter() - t1

    n_codes = (
        df.select("instrument").n_unique()
        if not df.is_empty() and "instrument" in df.columns
        else 0
    )

    return df
