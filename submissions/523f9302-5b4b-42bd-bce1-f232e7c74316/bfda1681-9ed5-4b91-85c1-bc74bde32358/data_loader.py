# -*- coding: utf-8 -*-
"""本地数据加载模块 —— 从 parquet 分片文件中读取多频率数据。

本地数据与在线平台数据的差异：
  - instrument_id (int16) 替代 instrument (string)
  - 仅 3 档盘口 (在线为 5 档)
  - 无 pre_close 字段
  - 价格字段为 int32 (需除以 100 还原为元, 再乘 adjust_factor 后复权)
  - 数据按周分片存储为 parquet 文件

用法:
    from data_loader import load_local_data, get_instruments

    df = load_local_data("bigalpha_2026_e2e_bar1m", "2022-01-01", "2023-12-31")
    instruments = get_instruments(df)
"""
import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")

# 本地数据表名 -> 目录映射
TABLE_DIR_MAP = {
    "bigalpha_2026_e2e_bar1m":  "bigalpha_2026_e2e_bar1m",
    "bigalpha_2026_e2e_bar5m":  "bigalpha_2026_e2e_bar5m",
    "bigalpha_2026_e2e_bar15m": "bigalpha_2026_e2e_bar15m",
    "bigalpha_2026_e2e_bar30m": "bigalpha_2026_e2e_bar30m",
}

# 本地数据的字段定义 (与在线版本不同: 仅 3 档盘口, 无 pre_close)
LOCAL_PRICE_COLS = [
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
LOCAL_VOL_COLS = [
    "volume", "amount",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
]
LOCAL_ORDER_NUM_COLS = [
    "deal_number",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
LOCAL_FEATURE_COLS = LOCAL_PRICE_COLS + LOCAL_VOL_COLS + LOCAL_ORDER_NUM_COLS
N_LOCAL_FEAT = len(LOCAL_FEATURE_COLS)  # 25

# 价格缩放因子: 本地 int32 价格需除以该值得到实际元
PRICE_SCALE = 100.0


def _find_parquet_files(table_name: str) -> list:
    """查找某表对应的所有 parquet 分片文件, 按文件名排序。"""
    subdir = TABLE_DIR_MAP.get(table_name, table_name)
    pattern = os.path.join(DATA_DIR, subdir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到数据文件: {pattern}")
    return files


def _parse_date_range_from_filename(filename: str) -> tuple:
    """从文件名中解析日期范围, 如 '..._2022-01-01_2022-01-07.parquet'。"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("_")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    return None, None


def load_local_data(table_name: str, start_date: str, end_date: str,
                    columns: list = None) -> pd.DataFrame:
    """加载指定表在日期范围内的数据。

    优先查找已合并的大文件, 若存在则直接加载后筛选日期;
    否则按周分片加载。
    通过 columns 参数可仅加载指定列以节省内存。
    """
    subdir = TABLE_DIR_MAP.get(table_name, table_name)
    sd_ts = pd.Timestamp(start_date)
    ed_ts = pd.Timestamp(end_date)

    # 优先查找合并文件
    merged_pattern = os.path.join(DATA_DIR, subdir, f"{table_name}_*.parquet")
    merged_files = sorted(glob.glob(merged_pattern))
    for mf in merged_files:
        f_sd, f_ed = _parse_date_range_from_filename(mf)
        if f_sd is None:
            continue
        f_sd_ts = pd.Timestamp(f_sd)
        f_ed_ts = pd.Timestamp(f_ed)
        if f_sd_ts <= sd_ts and f_ed_ts >= ed_ts:
            df = pd.read_parquet(mf, columns=columns)
            mask = (df["date"] >= sd_ts) & (df["date"] <= ed_ts)
            df = df.loc[mask].sort_values(["instrument_id", "date"]).reset_index(drop=True)
            if len(df) > 0:
                return df

    # 回退到分片加载
    all_files = _find_parquet_files(table_name)
    selected = []
    for f in all_files:
        f_sd, f_ed = _parse_date_range_from_filename(f)
        if f_sd is None:
            selected.append(f)
            continue
        f_sd_ts = pd.Timestamp(f_sd)
        f_ed_ts = pd.Timestamp(f_ed)
        if f_sd_ts <= ed_ts and f_ed_ts >= sd_ts:
            selected.append(f)
    if not selected:
        selected = all_files

    frames = []
    for f in selected:
        df_chunk = pd.read_parquet(f, columns=columns)
        mask = (df_chunk["date"] >= sd_ts) & (df_chunk["date"] <= ed_ts)
        df_chunk = df_chunk.loc[mask]
        if len(df_chunk) > 0:
            frames.append(df_chunk)

    if not frames:
        raise RuntimeError(f"表 {table_name} 在 {start_date}~{end_date} 无数据")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return df


def get_instruments(df: pd.DataFrame) -> list:
    """从数据中提取所有 instrument_id。"""
    return sorted(df["instrument_id"].unique().tolist())


def instrument_id_to_str(instrument_id: int) -> str:
    """将 instrument_id (int) 转为 instrument 字符串格式 (如 6 -> '000006.SZ')。

    启发式规则: 补零到 6 位, 根据首位判断交易所后缀。
    """
    code = f"{instrument_id:06d}"
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def preprocess_data(df: pd.DataFrame, use_adjust: bool = True) -> pd.DataFrame:
    """对本地数据做预处理 (尽量原地修改, 避免拷贝以节省内存)。

    1. 缺失值填充: 价格类向前填充后填0; 量额/笔数类填0
    2. 价格字段: int32 -> float32 (除以 PRICE_SCALE)
       若 use_adjust=True, 再乘 adjust_factor 后复权, 最后 log1p 变换
    3. 成交量/成交额字段: log1p 变换 -> float32
    4. 委托笔数列: log1p 变换 -> float32
    5. Inf 替换为 NaN 并填0
    6. 删除 adjust_factor 列
    """
    # ---- 缺失值填充 (在数值转换前) ----
    # 价格类: 按股票分组向前填充 (防止跨股票泄漏), 残余填 0
    id_col = "instrument_id" if "instrument_id" in df.columns else "instrument"
    for c in LOCAL_PRICE_COLS:
        if c in df.columns:
            df[c] = df.groupby(id_col, sort=False)[c].ffill().fillna(0)
    # 量额类 / 笔数类: 填 0 (语义: 无成交/无委托)
    for c in LOCAL_VOL_COLS + LOCAL_ORDER_NUM_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ---- 价格: 缩放 + 后复权 + log1p ----
    adj = df["adjust_factor"].to_numpy(np.float32) if use_adjust else None
    for c in LOCAL_PRICE_COLS:
        if c in df.columns:
            vals = df[c].to_numpy(np.float32) / PRICE_SCALE
            if use_adjust and adj is not None:
                vals = vals * adj
            # 保存原始复权价用于标签计算 (close 列专用)
            if c == "close":
                df["_close_raw"] = vals.copy()
            # log1p 变换 (规则允许的按字段统一对数变换)
            # 使价格特征与对数收益率目标处于同一线性空间, 便于模型学习
            vals = np.log1p(np.maximum(vals, 0))
            df[c] = vals

    # 成交量/额列: log1p -> float32
    for c in LOCAL_VOL_COLS:
        if c in df.columns:
            df[c] = np.log1p(np.maximum(df[c].to_numpy(np.float32), 0))

    # 委托笔数列: log1p -> float32
    for c in LOCAL_ORDER_NUM_COLS:
        if c in df.columns:
            df[c] = np.log1p(np.maximum(df[c].to_numpy(np.float32), 0))

    # Inf 值替换为 NaN 并填 0 (防御性处理)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in LOCAL_FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 删除 adjust_factor
    df.drop(columns=["adjust_factor"], inplace=True, errors="ignore")

    return df
