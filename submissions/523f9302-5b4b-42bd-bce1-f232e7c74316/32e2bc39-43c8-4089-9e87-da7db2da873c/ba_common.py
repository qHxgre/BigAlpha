"""
ba_common.py -- BigAlpha 2026 端到端量价赛道: 公共模块 (配置 / 数据 / 模型 / 损失)

赛题: https://bigquant.com/square/competition/523f9302-5b4b-42bd-bce1-f232e7c74316
数据: bigalpha_2026_stock_bar5m + bigalpha_2026_stock_bar1m
任务: 每个交易日收盘后输出一个截面分数 (date, instrument, score),
      平台 winsorize + zscore + BARRA 风格中性化后按 IC/ICIR/多空SR/Stress 排名

约束对照 (赛题规范 -> 本方案):
  - 原始字段 <= 100          -> 默认 28 个 (OHLC/pre_close/量额笔数 + 五档价量), 见 desired_fields()
  - 禁任何衍生/人工特征工程   -> 模型输入只有原始字段; 预处理仅用赛题明确允许的三类:
                                按字段统一的 log/log1p 变换 + 按字段统一的 zscore
                                (mu/sd 仅由训练集统计) + 缺失填 0
  - 回看窗口 <= 240 交易日    -> lookback_days=5 (5 天 x 48 根 5m bar = 240 步)
  - 参数量 1e5 ~ 1e8         -> build_model() 内 assert
  - 禁外部预训练权重          -> 全部随机初始化, SEED 固定
  - 无网络环境               -> 只依赖 numpy/pandas/torch + 平台 dai

标签 (标签不是模型输入, 不受特征工程约束; 沿用本项目"可执行价"原则):
  y_t = day_vwap_{t+2} / day_vwap_{t+1} - 1
  即 T 日收盘后打分, T+1 日全天 vwap 建仓, T+2 日全天 vwap 换仓 (日频调仓可执行口径)。
  可选: 对 bigalpha_2026_exposure 的风格暴露做截面回归取残差再训练,
  直接对齐平台"风格剔除后评估"的口径。

模型: E5H0 shared-E1-intraday + single-day-GRU。
  以公开分 0.85092 的 E4A 为唯一基线。最近 5 日分别复用同一套 E1
  5m raw+return+corr+std、TCN 与 intraday_pool；固定算子在每个日界重置，
  每天得到一个 96 维向量。仅新增一层 hidden=48 的日级 GRU 和零初始化
  增量打分头；E1 权重与归一化全部冻结，epoch0 严格等于原 E1 分数。
  不使用 day_pos、跨日注意力或连续 240 步算子。E4A 的同日 1m 残差
  分支及 FP32/零偏置数值路径保持不变，1m 分支不增加 LSTM/GRU。
  训练取数采用分阶段内存调度：缓存冻结5m分数并释放5m原始X后，才按单月
  加载1m，避免两个六年全历史面板同时常驻。
"""
from __future__ import annotations

import base64
import gc
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 必须在第一次 CUDA 运算前设置；E3A 默认启用严格可复现训练。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SEED = 42
EPS = 1e-12


def set_seed(seed: int = SEED, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def release_unused_memory():
    """释放分块DataFrame/临时ndarray，并在Linux上把空闲堆页归还给系统。"""
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


# ============================================================
# 1. 配置
# ============================================================

CONFIG = {
    # -- 版本: 旧 checkpoint 未包含该键时 build_model 自动走 v1, 保证 0.54545 基线可加载 --
    "model_version": "raw_hier_1m_res_v1",
    "cache_version": "raw_hier_1m_res_long5_v1",
    "memory_pipeline_version": "staged_5m_release_then_1m_monthly_v1",
    # -- 数据源 (平台注入 datasources 时由 apply_datasources 覆盖) --
    "bar_table": "bigalpha_2026_stock_bar5m",
    "bar_tables": {
        "1m": "bigalpha_2026_stock_bar1m",
        "5m": "bigalpha_2026_stock_bar5m",
    },
    "instruments_table": "bigalpha_2026_instruments",
    "exposure_table": "bigalpha_2026_exposure",
    "n_levels": 5,                   # 盘口档位数 (表里有 10 档, 取前 5 档控制字段数)
    # -- 输入窗口 --
    # E5H0：最近5日分别走共享E1日内编码器，再由单层日级GRU读取5个日向量。
    "lookback_days": 5,
    "short_days": 1,                 # 短分支: 最近 1 天原始 5m bar
    "long_pool": 8,                  # 长分支: 8 根 5m bar 池化为 40min bar
    # -- 标签 --
    "label_exec": "vwap",            # vwap: t+1 vwap -> t+2 vwap; close: t -> t+1 close
    "neutralize_label": False,       # 与 0.54545 基线保持一致; 中性化作为独立实验
    # 价格 log 后若存 fp16, 量化步长约 10~40bp, 会破坏 5m return/corr; V2 保留 fp32
    "panel_dtype": "float32",
    # 1m 全历史若用 fp32 约 35~40GB；先做每个股票日内的允许标准化后用 fp16
    # 保存。价格统一减当日最后成交价的 log，量/额类减各自日内 log 均值；差分、
    # corr、std 都保持不变，并且把峰值内存压到可在 64~100GB 环境运行的范围。
    "panel_dtypes": {"5m": "float32", "1m": "float16"},
    "intraday_relative_frequencies": ["1m"],
    # 先读取历史成分股，再把成分股并集作为 dai.query 的 instrument 分区过滤。
    # 同时以 universe 的日期/股票轴预分配最终数组，月度 chunk 用完即释放，避免
    # 旧实现先保留全部 chunk、最后再复制一次所造成的约 2 倍峰值内存。
    "query_universe_at_source": True,
    "preallocate_from_universe": True,
    # 1m 每月一批以降低首次分配39GB总面板时的临时峰值；5m缓存通常直接复用。
    "query_chunk_months_by_frequency": {"5m": 3, "1m": 1},
    "trim_memory_after_query_chunk": True,
    # 训练标签只由5m生成；1m只需 X 与“当日是否有数据”的有效标记。
    "daily_target_frequencies": ["5m"],
    # 1m 六年面板约20GB，默认不再额外写一份npz，避免工作盘空间不足。
    "panel_cache_by_frequency": {"5m": True, "1m": False},
    "panel_on_gpu": False,            # fp32全历史面板留在CPU，逐日送GPU，避免挤占激活显存
    # -- 模型: 第一版骨架 (公榜最优) + 阶段1 inter 组 (EXP_PLAN.md) --
    "group_channels": {"price": 64, "depth": 64, "flow": 48, "inter": 32},
    "short_channel_scale": 0.5,
    "short_tcn_kernel": 3, "short_tcn_dilations": (1, 2, 4, 8),
    "long_tcn_kernel": 3, "long_tcn_dilations": (1, 2, 4),
    "fusion_channels": 128, "fusion_se": True,
    "progressive_tcn_channels": True,
    # -- inter 组: log 域幂积单元 (乘除幂交互序列; use_inter=False 即纯阶段0骨架) --
    "use_inter": False, "inter_K": 10, "inter_l1": 1e-3,
    # -- ts_corr 分支 (二期): AlphaNet 式零参数窗口交互算子 (两两 ts_corr + ts_std) --
    "use_tscorr": False,
    "corr_fields": ["price", "high", "low", "volume", "amount", "num_trades",
                    "bid_volume1", "ask_volume1"],
    "corr_window": 48,               # 滑窗 bar 数 (48 = 1 交易日 5m)
    # -- 时间算子组 (框架第3-6点, 差分型): 每字段 x[t]-x[t-k] = k-bar 对数收益/动量,
    #    零参数, 与电平去相关 (刻意不含 MA/平滑, 规避已证实的共线掉分); 默认关 --
    "use_temporal_ops": False,
    "temporal_diff_bars": [8, 48],   # 差分跨度 (40min / 1交易日)
    # -- GRN 融合 (框架第10点): 门控残差网络替代 1x1conv+SE; 默认关 --
    "use_grn": False,
    "lstm_hidden": 64, "lstm_layers": 2,
    "fc_hidden": 48, "dropout": 0.1,
    # -- V2: 原始数据编码；E5H0 在每个交易日边界重置固定算子 --
    "v2_d_model": 96,
    "v2_group_dims": {"price": 32, "depth": 32, "flow": 16},
    "v2_intra_dilations": (1, 2, 4, 8),
    "v2_operator_dilations": (1, 2),
    "v2_gru_layers": 2,              # 仅供旧 checkpoint 兼容
    "v2_day_gru_hidden": 48,
    "v2_day_gru_layers": 1,
    "v2_train_context_only": True,
    # E5H0：每天独立复用 E1 日内编码器，得到 5 个日向量后仅增加单层 GRU。
    # 日内位置参数在每天复用；不增加 day_pos、跨日注意力或连续 240 步算子。
    "v2_long_window_mode": "shared_day_gru_residual",
    "v2_long_position_mode": "shared_intraday_only",
    # 以下正余弦参数只为兼容读取旧实验配置；E5H0 不创建或使用该编码。
    "v2_day_sincos_base": 10000.0,
    "v2_day_sincos_scale": 0.02,
    "v2_use_crossday": False,
    "v2_use_return": True,
    "v2_use_corr": True,
    # E4A的5m主干包含对一阶变化做日内因果 rolling std。
    "v2_use_std": True,
    "v2_return_fields": ["price", "open", "high", "low", "volume", "amount",
                         "bid_volume1", "ask_volume1"],
    "v2_return_horizons": [1, 3, 6, 12],
    "v2_corr_pairs": [["price", "volume"], ["price", "amount"],
                      ["price", "bid_volume1"], ["price", "ask_volume1"],
                      ["high", "volume"], ["low", "volume"]],
    "v2_corr_windows": [12, 24, 48],
    "v2_std_fields": ["price", "volume", "amount", "bid_volume1", "ask_volume1"],
    "v2_std_windows": [12, 24, 48],
    # -- E4A：冻结5m主干，只训练同日1m增量分支 --
    # 240 根 1m bar 先在分钟轴编码，再每 5 根压成一个 token，最后在 48 个
    # 5m 对齐 token 上编码。输出层零初始化，因此 epoch0 与冻结5m主干严格等价。
    "score_head_fp32": True,
    "residual_frequency": "1m",
    "res1m_zero_output_bias": True,
    "res1m_d_model": 64,
    "res1m_group_dims": {"price": 24, "depth": 24, "flow": 16},
    "res1m_local_dilations": [1, 2, 4],
    "res1m_block_dilations": [1, 2, 4, 8],
    "res1m_operator_dilations": [1, 2],
    "res1m_use_return": True,
    "res1m_use_corr": True,
    "res1m_use_std": True,
    "res1m_return_fields": ["price", "open", "high", "low", "volume", "amount",
                             "bid_volume1", "ask_volume1"],
    "res1m_return_horizons": [1, 2, 5, 10, 30, 60],
    "res1m_corr_pairs": [["price", "volume"], ["price", "amount"],
                          ["price", "bid_volume1"], ["price", "ask_volume1"],
                          ["high", "volume"], ["low", "volume"]],
    "res1m_corr_windows": [5, 15, 30, 60],
    "res1m_std_fields": ["price", "volume", "amount", "bid_volume1", "ask_volume1"],
    "res1m_std_windows": [5, 15, 30, 60],
    # -- 损失 (沿用 train0703: GroupedListMLE + PearsonIC) --
    "listmle_weight": 1.0, "rankic_weight": 0.2,
    "top_frac": 0.5, "bottom_frac": 0.5, "min_group_size": 30,
    # V2 全截面 IC+Huber 锚定中间样本, 头尾 pairwise 强化可交易排序
    "loss_version": "stable_rank_v2",
    "ic_weight": 1.0, "huber_weight": 0.25, "tail_weight": 0.25,
    "tail_frac": 0.2, "tail_temperature": 1.0, "huber_beta": 1.0,
    "tail_warmup_epochs": 3,
    # -- 训练 --
    "epochs": 40, "min_epochs": 3, "patience": 6,
    "experiment_id": "E5H0_SHARED_DAY_GRU_1M_RESIDUAL",
    "learning_rate": 3e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
    "use_amp": True,
    "max_stocks_per_cs": None,       # V2 使用完整截面, 不设置股票数上限
    "min_cs_size": 200,              # 截面有效股票数下限
    "val_purge_days": 2,             # vwap[t+1]->vwap[t+2] 标签与验证期隔离
    "target_mad_n": 5.0,
    "predict_chunk": 1024,
    "seed": SEED,
    # 沿用0.85092版E4A的随机性设置；E1冻结，只训练日级上下文与1m残差。
    "deterministic_training": False,
}


def apply_datasources(config: dict, datasources) -> dict:
    """按官方模版契约解析平台注入的 datasources。
    公/私榜平台只替换 datasources 与起止时间, 形如 {"bar5m": "..._stock_bar5m"};
    也兼容 {"bar1m": ...} (自动换成同前缀的 bar5m 表)、表名字符串、None。"""
    import re
    cfg = dict(config)
    tbl = None
    tables = dict(cfg.get("bar_tables", {}))
    if isinstance(datasources, dict):
        provided = {}
        for freq in ("1m", "5m", "15m", "30m"):
            if datasources.get(f"bar{freq}"):
                provided[freq] = datasources[f"bar{freq}"]
            elif datasources.get(freq):
                provided[freq] = datasources[freq]
        tables.update(provided)
        tbl = provided.get("5m")
        if not tbl:
            source = (next(iter(provided.values()), None)
                      or next((v for k, v in datasources.items()
                               if "bar" in str(k).lower()
                               or "_stock_bar" in str(v)), None))
            tbl = (re.sub(r"bar\d+m", "bar5m", source)
                   if source else tables.get("5m"))
        if tbl and "bar5m" not in tbl:
            tbl = re.sub(r"bar\d+m", "bar5m", tbl)
        # 若平台只注入一个频率表，用相同私榜前缀推导其余官方频率表。
        if tbl:
            for freq in ("1m", "5m"):
                if freq not in provided:
                    tables[freq] = re.sub(r"bar\d+m", f"bar{freq}", tbl)
    elif isinstance(datasources, str) and datasources:
        tbl = (datasources if "bar" in datasources
               else f"{datasources}_stock_bar5m")
        tables["5m"] = re.sub(r"bar\d+m", "bar5m", tbl)
        tables["1m"] = re.sub(r"bar\d+m", "bar1m", tbl)
    if tbl:
        cfg["bar_table"] = tables.get("5m", tbl)
        cfg["bar_tables"] = tables
        prefix = tbl.split("_stock_bar")[0]      # 私榜表名前缀可能不同, 同步辅助表
        cfg["instruments_table"] = f"{prefix}_instruments"
        cfg["exposure_table"] = f"{prefix}_exposure"
    else:
        cfg["bar_tables"] = tables
    return cfg


# ============================================================
# 2. 字段定义 (只有"原始字段", 无任何跨字段运算)
# ============================================================

FIELD_ALIASES = {
    "price": ("price", "close"),
    "num_trades": ("num_trades", "num_trade", "trade_count"),
}


def desired_fields(n_levels: int = 5) -> list[str]:
    f = ["open", "high", "low", "price", "pre_close", "volume", "amount",
         "num_trades"]
    for i in range(1, n_levels + 1):
        f += [f"bid_price{i}", f"ask_price{i}", f"bid_volume{i}", f"ask_volume{i}"]
    return f


def resolve_fields(available: list[str], n_levels: int = 5) -> list[str]:
    """desired -> 实际表列名 (处理别名), 缺失的字段丢弃并告警。"""
    out, missing = [], []
    avail = set(available)
    for f in desired_fields(n_levels):
        cands = FIELD_ALIASES.get(f, (f,))
        hit = next((c for c in cands if c in avail), None)
        (out if hit else missing).append(hit or f)
    if missing:
        print(f"[warn] 表中缺失字段 (已跳过): {missing}")
    assert len(out) <= 100, f"字段数 {len(out)} 超过赛题上限 100"
    return out


def is_volume_like(field: str) -> bool:
    return ("_volume" in field
            or field in ("volume", "amount", "num_trades", "num_trade",
                         "trade_count"))


def make_groups(fields: list[str]) -> dict[str, list[int]]:
    """按字段语义分组 (对应模型的分组 TCN), 与 train0703 的 GROUP_IDX 同构。"""
    groups = {"price": [], "depth": [], "flow": []}
    for i, f in enumerate(fields):
        if is_volume_like(f):
            groups["depth" if "_volume" in f else "flow"].append(i)
        else:
            groups["price"].append(i)
    return {k: v for k, v in groups.items() if v}


# ============================================================
# 3. dai 数据加载 -> 日频 panel 张量
# ============================================================

def date_filter(start: str, end: str) -> dict:
    """dai filters 的日期格式带时间部分 (官方示例: '2019-01-01 00:00:00')"""
    return {"date": [f"{str(start)[:10]} 00:00:00", f"{str(end)[:10]} 23:59:59"]}


def dai_query(sql: str, filters: dict | None = None) -> pd.DataFrame:
    import dai   # 平台内置, 本地无法运行 (本地测试走 ba_train.py --smoke)
    # compression=True: 数据量大时降内存 (会把 instrument 转成 category 类型)
    res = dai.query(sql, filters=filters or {}, compression=True)
    df = res.df() if hasattr(res, "df") else pd.DataFrame(res)
    # 必须保留 compression=True 产生的 category。旧代码在这里立刻 astype(str)，
    # 会把每月数百万个 instrument 展开成 Python 对象，既慢又显著放大内存。
    # _chunk_to_arrays 直接使用 category codes，不再需要展开整列。
    return df


def probe_columns(cfg: dict, probe_date: str) -> list[str]:
    end = (pd.Timestamp(probe_date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    df = dai_query(f"SELECT * FROM {cfg['bar_table']} LIMIT 5",
                   filters=date_filter(probe_date, end))
    return list(df.columns)


def month_ranges(start: str, end: str, chunk_months: int = 1):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunk_months = max(1, int(chunk_months))
    cur = s
    while cur <= e:
        nxt = (cur + pd.offsets.MonthBegin(chunk_months)).normalize()
        yield cur.strftime("%Y-%m-%d"), min(nxt - pd.Timedelta(days=1), e).strftime("%Y-%m-%d")
        cur = nxt


def _instrument_axis(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """instrument 列 -> 排序后的股票轴和逐行位置；category 路径不展开字符串。"""
    if isinstance(series.dtype, pd.CategoricalDtype):
        raw = series.cat.codes.to_numpy(dtype=np.int64, copy=False)
        if (raw < 0).any():
            raise ValueError("instrument 存在空值")
        used = np.unique(raw)
        cats = np.asarray(series.cat.categories.astype(str), dtype=str)
        values = cats[used]
        order = np.argsort(values)
        codes = values[order]
        remap = np.full(len(cats), -1, dtype=np.int64)
        remap[used[order]] = np.arange(len(used), dtype=np.int64)
        return codes, remap[raw]
    values = series.astype(str).to_numpy(copy=False)
    codes, inverse = np.unique(values, return_inverse=True)
    return codes.astype(str, copy=False), inverse.astype(np.int64, copy=False)


def _extract_tod(df: pd.DataFrame) -> np.ndarray:
    """bar 时间戳 -> 当日 HHMM 整数。优先 date 列的时间部分, 其次 time 列。"""
    dt = pd.to_datetime(df["date"])
    tod = (dt.dt.hour * 100 + dt.dt.minute).to_numpy()
    if tod.max() == 0 and "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(np.int64)
        tod = np.where(t >= 100000, t // 100, t).to_numpy() if hasattr(t, "to_numpy") else t
        tod = np.asarray(tod)
        tod = np.where(tod >= 10000, tod // 100, tod)   # HHMMSS -> HHMM
    return tod.astype(np.int64)


def _extract_trading_day(df: pd.DataFrame) -> np.ndarray:
    col = "trading_day" if "trading_day" in df.columns else "date"
    return pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d").to_numpy()


def _chunk_to_arrays(df: pd.DataFrame, fields: list[str], slot_index: dict,
                     storage_dtype=np.float16, intraday_relative: bool = False,
                     progress: bool = False,
                     compute_daily_targets: bool = True):
    """单个月度 chunk -> (days, codes, X(D,N,B,F) 已做 log 变换, amt, vol, cls)。

    旧 V1 默认 fp16; V2 使用 fp32，避免 log-price 在 5m 尺度产生约 10~40bp 的量化误差。
    E4A 的 1m 面板先做每个股票日内的 level 标准化再存 fp16：所有价格字段
    减同日最后成交价 log，量/额类字段减各自日内 log 均值。它只去除常数 level，
    不改变 return/corr/std，并显著降低 fp16 的量化误差与内存占用。
    """
    B, Fn = len(slot_index), len(fields)
    storage_dtype = np.dtype(storage_dtype)
    td = _extract_trading_day(df)
    tod = _extract_tod(df)
    bi = pd.Series(tod).map(slot_index).fillna(-1).astype(np.int64).to_numpy()
    keep = bi >= 0
    if not keep.all():
        n_bad = int((~keep).sum())
        if n_bad > 0.01 * len(df):
            print(f"[warn] {n_bad}/{len(df)} 行的 bar 时间不在既定槽位, 已丢弃")
        df, td, bi = df[keep], td[keep], bi[keep]

    days = np.unique(td)
    codes, ci = _instrument_axis(df["instrument"])
    di = np.searchsorted(days, td)

    D0, N0 = len(days), len(codes)
    X = np.full((D0, N0, B, Fn), np.nan, dtype=storage_dtype)

    # 1m 相对化共用同一个最后成交价基准，因而保留 OHLC/盘口价格之间的价差。
    price_ref = None
    if intraday_relative:
        price_col = next((c for c in FIELD_ALIASES["price"] if c in df.columns), None)
        if price_col is None:
            raise ValueError("intraday_relative=True 但表中不存在 price/close 字段")
        raw_ref = pd.to_numeric(df[price_col], errors="coerce").to_numpy(np.float32)
        log_ref = np.where(raw_ref > 0, np.log(np.maximum(raw_ref, EPS)), np.nan)
        pmat = np.full((D0, N0, B), np.nan, dtype=np.float32)
        pmat[di, ci, bi] = log_ref
        valid_p = np.isfinite(pmat)
        any_p = valid_p.any(axis=2)
        last_idx = B - 1 - np.argmax(valid_p[:, :, ::-1], axis=2)
        price_ref = np.take_along_axis(pmat, last_idx[:, :, None], axis=2)[:, :, 0]
        price_ref[~any_p] = np.nan
        del pmat, valid_p

    transform_t0 = time.time()
    for k, f in enumerate(fields):
        v = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=np.float32)
        # 按字段统一的对数变换 (赛题明确允许); 不做任何跨字段运算
        if is_volume_like(f):
            v = np.log1p(np.maximum(v, 0.0))
        else:
            v = np.where(v > 0, np.log(np.maximum(v, EPS)), np.nan)
        if intraday_relative:
            if is_volume_like(f):
                # np.add.at 对每个字段逐行做原子累加，在两个月约两千万行上极慢。
                # 数据本来就是唯一的 (day, code, bar)，先散射到连续3D张量，再沿
                # bar 轴向量化求均值；数值口径与逐行累加相同。
                grid = np.full((D0, N0, B), np.nan, dtype=np.float32)
                grid[di, ci, bi] = v
                finite = np.isfinite(grid)
                counts = finite.sum(axis=2, dtype=np.int32)
                sums = np.nansum(grid, axis=2, dtype=np.float64)
                means = (sums / np.maximum(counts, 1)).astype(np.float32)
                grid -= means[:, :, None]
                grid[~finite] = np.nan
                X[:, :, :, k] = grid.astype(storage_dtype)
                del grid, finite, counts, sums, means, v
                if progress and (k + 1) % 5 == 0:
                    print(f"[{now_s()}] 1m转换 {k + 1}/{Fn} 字段 "
                          f"({time.time() - transform_t0:.0f}s)", flush=True)
                continue
            else:
                v = v - price_ref[di, ci]
        X[di, ci, bi, k] = v.astype(storage_dtype)
        if progress and (k + 1) % 5 == 0:
            print(f"[{now_s()}] 1m转换 {k + 1}/{Fn} 字段 "
                  f"({time.time() - transform_t0:.0f}s)", flush=True)

    if progress and Fn % 5:
        print(f"[{now_s()}] 1m转换 {Fn}/{Fn} 字段 "
              f"({time.time() - transform_t0:.0f}s)", flush=True)

    if compute_daily_targets:
        # 仅5m标签面板需要原始日量额和收盘。
        amt = np.zeros((len(days), len(codes)), dtype=np.float64)
        vol = np.zeros_like(amt)
        raw_amt = pd.to_numeric(
            df["amount"], errors="coerce").fillna(0).to_numpy(np.float64)
        raw_vol = pd.to_numeric(
            df["volume"], errors="coerce").fillna(0).to_numpy(np.float64)
        np.add.at(amt, (di, ci), raw_amt)
        np.add.at(vol, (di, ci), raw_vol)

        price_col = next(
            (c for c in FIELD_ALIASES["price"] if c in df.columns), None)
        cls = np.full((len(days), len(codes)), np.nan, dtype=np.float32)
        if price_col:
            raw_p = pd.to_numeric(
                df[price_col], errors="coerce").to_numpy(np.float32)
            order = np.lexsort((bi, ci, di))
            d_o, c_o, p_o = di[order], ci[order], raw_p[order]
            ok = np.isfinite(p_o) & (p_o > 0)
            cls[d_o[ok], c_o[ok]] = p_o[ok]
    else:
        # 1m分支的 vwap/close 从不参与标签或模型；只保留有数据标记，跳过
        # 两次逐行np.add.at和一次全量lexsort。
        observed = np.zeros((len(days), len(codes)), dtype=bool)
        observed[di, ci] = True
        amt = observed.astype(np.float64)
        vol = observed.astype(np.float64)
        cls = np.where(observed, 1.0, np.nan).astype(np.float32)
        if progress:
            print(f"[{now_s()}] 1m跳过无用的日VWAP/收盘聚合", flush=True)
    return {"days": days, "codes": codes, "X": X, "amt": amt, "vol": vol, "cls": cls}


def build_panel(cfg: dict, start: str, end: str,
                fields: list[str] | None = None,
                bar_slots: list[int] | None = None,
                query_fn=None, verbose: bool = True,
                instruments: list[str] | None = None,
                master_days: list[str] | np.ndarray | None = None,
                master_codes: list[str] | np.ndarray | None = None) -> dict:
    """分块查询 bar 表并拼成日频 panel。

    提供 master_days/master_codes 时直接预分配最终数组，查询 chunk 转换后立即
    合并并释放，避免旧实现保留全部 chunk 后再复制所造成的约 2 倍峰值内存。
    instruments 作为 dai.query 的底层分区过滤，不替代后续逐日成分股掩码。
    """
    query = query_fn or dai_query
    if fields is None:
        fields = resolve_fields(probe_columns(cfg, start), cfg["n_levels"])
    # 本赛题 date 已是含分钟的时间戳；无需再传输 trading_day/time 两整列，
    # 也避免为探测这两列额外发起一次 DAI 查询。
    base_cols = ["date", "instrument"]
    cols = list(dict.fromkeys(base_cols + fields))
    sql = f"SELECT {', '.join(cols)} FROM {cfg['bar_table']}"

    slot_index = ({t: i for i, t in enumerate(bar_slots)}
                  if bar_slots is not None else None)
    storage_dtype = np.dtype(cfg.get("panel_dtype", "float16"))
    instruments = (sorted({str(x) for x in instruments})
                   if instruments is not None else None)
    use_master = master_days is not None and master_codes is not None
    if use_master:
        days = np.asarray(sorted({str(x) for x in master_days}))
        codes_arr = np.asarray(sorted({str(x) for x in master_codes}))
        day_pos = {d: i for i, d in enumerate(days.tolist())}
        code_pos = {c: i for i, c in enumerate(codes_arr.tolist())}
    else:
        days = codes_arr = day_pos = code_pos = None

    X = amt = vol = cls = None

    def allocate_master():
        nonlocal X, amt, vol, cls
        if X is not None:
            return
        D, N, B, Fn = len(days), len(codes_arr), len(slot_index), len(fields)
        need_gb = D * N * B * Fn * storage_dtype.itemsize / 1e9
        if verbose:
            print(f"[{now_s()}] 开始预分配最终 panel ({need_gb:.1f}GB)，"
                  "此步骤可能短暂无输出", flush=True)
        X = np.full((D, N, B, Fn), np.nan, dtype=storage_dtype)
        amt = np.zeros((D, N), dtype=np.float64)
        vol = np.zeros((D, N), dtype=np.float64)
        cls = np.full((D, N), np.nan, dtype=np.float32)
        if verbose:
            print(f"[{now_s()}] 预分配 panel: {D}天 x {N}股 x {B}bar x "
                  f"{Fn}字段 ({X.nbytes / 1e9:.1f}GB {X.dtype})", flush=True)

    def merge_master(c: dict):
        allocate_master()
        src_di, dst_di = [], []
        for i, d in enumerate(c["days"].tolist()):
            j = day_pos.get(str(d))
            if j is not None:
                src_di.append(i); dst_di.append(j)
        src_ci, dst_ci = [], []
        for i, code in enumerate(c["codes"].tolist()):
            j = code_pos.get(str(code))
            if j is not None:
                src_ci.append(i); dst_ci.append(j)
        if not src_di or not src_ci:
            return
        src = np.ix_(np.asarray(src_di), np.asarray(src_ci))
        dst = np.ix_(np.asarray(dst_di), np.asarray(dst_ci))
        X[dst] = c["X"][src]
        amt[dst] += c["amt"][src]
        vol[dst] += c["vol"][src]
        old = cls[dst]
        cls[dst] = np.where(np.isfinite(c["cls"][src]), c["cls"][src], old)

    # 已由checkpoint给出bar_slots且主轴已知时，先完成最终数组分配，再发起
    # 大查询。旧顺序会同时持有首月DataFrame、首月chunk和39GB最终数组，
    # 正好在np.full触页时把进程推入swap。
    if use_master and slot_index is not None:
        allocate_master()

    chunks = []
    saw_data = False
    chunk_months = max(1, int(cfg.get("query_chunk_months", 1)))
    if verbose:
        source_msg = (f", instrument过滤={len(instruments)}股"
                      if instruments is not None else "")
        print(f"[{now_s()}] 读取 {cfg['bar_table']}: 每{chunk_months}个月一批"
              f"{source_msg}", flush=True)
    for s, e in month_ranges(start, end, chunk_months):
        t0 = time.time()
        filters = date_filter(s, e)
        if instruments is not None:
            filters["instrument"] = instruments
        df = query(sql, filters=filters)
        if df is None or len(df) == 0:
            continue
        saw_data = True
        if slot_index is None:
            slots = sorted(np.unique(_extract_tod(df)).tolist())
            slot_index = {t: i for i, t in enumerate(slots)}
            if verbose:
                print(f"[{now_s()}] bar 槽位 {len(slots)} 个/日: "
                      f"{slots[:3]}..{slots[-3:]}")
        c = _chunk_to_arrays(
            df, fields, slot_index, storage_dtype,
            intraday_relative=bool(cfg.get("intraday_relative", False)),
            progress=bool(verbose and cfg.get("intraday_relative", False)),
            compute_daily_targets=bool(cfg.get("compute_daily_targets", True)))
        if use_master:
            merge_master(c)
        else:
            chunks.append(c)
        if verbose:
            print(f"[{now_s()}] {s}..{e}: {len(df)} 行, "
                  f"{len(c['days'])} 天 x {len(c['codes'])} 股 "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del df, c
        if cfg.get("trim_memory_after_query_chunk", True):
            release_unused_memory()

    if use_master and not saw_data:
        raise RuntimeError(f"{cfg['bar_table']} 在 [{start}, {end}] 无数据")
    if not use_master and not chunks:
        raise RuntimeError(f"{cfg['bar_table']} 在 [{start}, {end}] 无数据")

    if not use_master:
        days = np.asarray(sorted(set().union(*[c["days"].tolist()
                                                for c in chunks])))
        codes_arr = np.asarray(sorted(set().union(*[c["codes"].tolist()
                                                    for c in chunks])))
        day_idx = {d: i for i, d in enumerate(days.tolist())}
        D, N, B, Fn = len(days), len(codes_arr), len(slot_index), len(fields)
        storage_dtype = chunks[0]["X"].dtype
        X = np.full((D, N, B, Fn), np.nan, dtype=storage_dtype)
        amt = np.zeros((D, N)); vol = np.zeros((D, N))
        cls = np.full((D, N), np.nan, dtype=np.float32)
        for c in chunks:
            di = np.asarray([day_idx[d] for d in c["days"].tolist()])
            ci = np.searchsorted(codes_arr, c["codes"])
            X[np.ix_(di, ci)] = c["X"]
            amt[np.ix_(di, ci)] += c["amt"]
            vol[np.ix_(di, ci)] += c["vol"]
            sub = cls[np.ix_(di, ci)]
            cls[np.ix_(di, ci)] = np.where(np.isfinite(c["cls"]), c["cls"], sub)
        del chunks

    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(vol > 0, amt / np.where(vol > 0, vol, 1.0), np.nan)
    valid = (vol > 0) & np.isfinite(cls)
    return {"days": np.asarray(days), "codes": codes_arr, "X": X,
            "vwap": vwap.astype(np.float32), "close": cls, "valid": valid,
            "fields": fields,
            "bar_slots": sorted(slot_index, key=slot_index.get)}


PANEL_CACHE_KEYS = ("X", "vwap", "close", "valid", "days", "codes")


def save_panel_cache(panel: dict, path: str):
    """panel 落盘 (未压缩 npz, 6GB 级写入约 1 分钟), 后续重跑/多种子训练直接加载"""
    np.savez(path, fields=np.asarray(panel["fields"]),
             bar_slots=np.asarray(panel["bar_slots"], dtype=np.int64),
             **{k: panel[k] for k in PANEL_CACHE_KEYS})
    print(f"[{now_s()}] panel 缓存 -> {path} "
          f"({Path(path).stat().st_size / 1e9:.1f} GB)")


def load_panel_cache(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    panel = {k: z[k] for k in PANEL_CACHE_KEYS}
    panel["fields"] = [str(x) for x in z["fields"]]
    panel["bar_slots"] = [int(x) for x in z["bar_slots"]]
    print(f"[{now_s()}] 从缓存加载 panel: {path} "
          f"({len(panel['days'])} 天 x {len(panel['codes'])} 股)")
    return panel


def config_for_frequency(config: dict, frequency: str) -> dict:
    """返回单频率配置副本，并应用该频率的存储精度/日内标准化口径。"""
    cfg = dict(config)
    tables = config.get("bar_tables", {})
    if frequency not in tables:
        raise KeyError(f"未配置 {frequency} 数据表；已有 {sorted(tables)}")
    cfg["bar_table"] = tables[frequency]
    cfg["panel_dtype"] = config.get("panel_dtypes", {}).get(
        frequency, config.get("panel_dtype", "float32"))
    cfg["intraday_relative"] = frequency in set(
        config.get("intraday_relative_frequencies", []))
    cfg["query_chunk_months"] = int(
        config.get("query_chunk_months_by_frequency", {}).get(frequency, 1))
    cfg["compute_daily_targets"] = frequency in set(
        config.get("daily_target_frequencies", ["5m"]))
    return cfg


def align_panel_to_master(panel: dict, master_days: np.ndarray,
                          master_codes: np.ndarray) -> dict:
    """把任意频率 panel 重索引到 5m 主 panel 的日期和股票轴。

    频率自身的 bar/field 轴保持不变；不存在的日期或股票保留 NaN/False。
    """
    master_days = np.asarray(master_days)
    master_codes = np.asarray(master_codes)
    if (np.array_equal(panel["days"], master_days)
            and np.array_equal(panel["codes"], master_codes)):
        return panel

    D, N = len(master_days), len(master_codes)
    B, Fn = panel["X"].shape[2:]
    out = {
        "days": master_days.copy(),
        "codes": master_codes.copy(),
        "X": np.full((D, N, B, Fn), np.nan, dtype=panel["X"].dtype),
        "vwap": np.full((D, N), np.nan, dtype=np.float32),
        "close": np.full((D, N), np.nan, dtype=np.float32),
        "valid": np.zeros((D, N), dtype=bool),
        "fields": list(panel["fields"]),
        "bar_slots": list(panel["bar_slots"]),
    }
    day_pos = {d: i for i, d in enumerate(master_days.tolist())}
    code_pos = {c: i for i, c in enumerate(master_codes.tolist())}
    src_di, dst_di = [], []
    for i, d in enumerate(panel["days"].tolist()):
        if d in day_pos:
            src_di.append(i)
            dst_di.append(day_pos[d])
    src_ci, dst_ci = [], []
    for i, c in enumerate(panel["codes"].tolist()):
        if c in code_pos:
            src_ci.append(i)
            dst_ci.append(code_pos[c])
    if src_di and src_ci:
        src = np.ix_(np.asarray(src_di), np.asarray(src_ci))
        dst = np.ix_(np.asarray(dst_di), np.asarray(dst_ci))
        out["X"][dst] = panel["X"][src]
        for key in ("vwap", "close", "valid"):
            out[key][dst] = panel[key][src]
    return out


def apply_universe_mask(panel: dict, universe: dict[str, set] | None,
                        mask_history: bool = False) -> np.ndarray:
    """按历史逐日成分股口径生成 member mask，并可清空历史输入。

    `mask_history=True` 用于跨日分支，防止当前成分股读取其入池前的盘口数据。
    universe 缺少某日时沿用 panel.valid，和原提交推理 fallback 一致。
    """
    member = np.zeros_like(panel["valid"], dtype=bool)
    code_pos = {c: i for i, c in enumerate(panel["codes"].tolist())}
    for d, day in enumerate(panel["days"].tolist()):
        if universe is None or day not in universe:
            member[d] = panel["valid"][d]
            continue
        idx = [code_pos[c] for c in universe.get(day, ()) if c in code_pos]
        if idx:
            member[d, np.asarray(idx, dtype=np.int64)] = True
    panel["valid"] &= member
    if mask_history:
        for d in range(len(panel["days"])):
            panel["X"][d, ~member[d]] = np.nan
    return member


def load_universe(cfg: dict, start: str, end: str,
                  query_fn=None) -> dict[str, set] | None:
    """instruments 表 -> {date: set(instrument)}; 失败时返回 None (退化为 valid 掩码)"""
    query = query_fn or dai_query
    try:
        df = query(f"SELECT date, instrument FROM {cfg['instruments_table']}",
                   filters=date_filter(start, end))
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        # instrument 可以保持 category；这里只在每天约1000行的小组上构造字符串集合。
        return {str(d): {str(x) for x in g.dropna().tolist()}
                for d, g in df.groupby("date", observed=True)["instrument"]}
    except Exception as e:
        print(f"[warn] 读取 {cfg['instruments_table']} 失败, 退化为全 panel: {e}")
        return None


def load_exposure_panel(cfg: dict, panel: dict, query_fn=None) -> np.ndarray | None:
    """exposure 表 -> (D,N,K) float32, 与 panel 的 days/codes 对齐; 失败返回 None"""
    query = query_fn or dai_query
    try:
        df = query(f"SELECT * FROM {cfg['exposure_table']}",
                   filters=date_filter(panel["days"][0], panel["days"][-1]))
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        fac_cols = [c for c in df.columns
                    if c not in ("date", "instrument")
                    and pd.api.types.is_numeric_dtype(df[c])]
        if not fac_cols:
            return None
        day_idx = {d: i for i, d in enumerate(panel["days"].tolist())}
        di = df["date"].map(day_idx)
        inst = df["instrument"].astype(str).to_numpy(copy=False)
        ci = pd.Series(np.searchsorted(panel["codes"], inst))
        in_panel = (inst
                    == panel["codes"][np.clip(ci, 0, len(panel["codes"]) - 1)])
        m = di.notna().to_numpy() & in_panel
        E = np.full((len(panel["days"]), len(panel["codes"]), len(fac_cols)),
                    np.nan, dtype=np.float32)
        E[di[m].astype(int).to_numpy(), ci[m].to_numpy()] = \
            df.loc[m, fac_cols].to_numpy(dtype=np.float32)
        print(f"[{now_s()}] exposure: {len(fac_cols)} 个风格因子 {fac_cols}")
        return E
    except Exception as e:
        print(f"[warn] 读取 {cfg['exposure_table']} 失败, 标签不做风格中性化: {e}")
        return None


# ============================================================
# 4. 标签 / 标准化
# ============================================================

def neutralize_cs(y: np.ndarray, expo: np.ndarray) -> np.ndarray:
    """单截面: y 对 [1, expo] 回归取残差 (对齐平台风格剔除口径)"""
    m = np.isfinite(y) & np.isfinite(expo).all(axis=1)
    if m.sum() < expo.shape[1] + 5:
        return y
    Xd = np.column_stack([np.ones(int(m.sum())), expo[m]])
    beta, *_ = np.linalg.lstsq(Xd, y[m], rcond=None)
    out = y.copy()
    out[m] = y[m] - Xd @ beta
    return out


def build_labels(cfg: dict, panel: dict, expo: np.ndarray | None = None) -> np.ndarray:
    """y (D,N): 第 d 行是 d 日收盘后打分对应的前瞻收益 (最后 2 天为 NaN)"""
    D, N = panel["vwap"].shape
    y = np.full((D, N), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        if cfg["label_exec"] == "vwap":
            px = panel["vwap"]
            if D > 2:
                y[:-2] = px[2:] / px[1:-1] - 1.0
        else:                                    # close: t -> t+1 收盘
            px = panel["close"]
            if D > 1:
                y[:-1] = px[1:] / px[:-1] - 1.0
    y = np.where(panel["valid"], y, np.nan)
    if expo is not None and cfg.get("neutralize_label", True):
        for d in range(D):
            if np.isfinite(y[d]).sum() >= cfg["min_cs_size"]:
                y[d] = neutralize_cs(y[d], expo[d])
    return y


def fit_field_norm(X: np.ndarray, day_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按字段统一的 zscore 参数, 仅用训练日统计 (流式, 避免 fp32 全量拷贝)"""
    Fn = X.shape[-1]
    s = np.zeros(Fn, dtype=np.float64)
    ss = np.zeros(Fn, dtype=np.float64)
    n = np.zeros(Fn, dtype=np.float64)
    for d in day_ids:
        a = X[d].astype(np.float32).reshape(-1, Fn)
        m = np.isfinite(a)
        a0 = np.where(m, a, 0.0)
        s += a0.sum(axis=0)
        ss += (a0.astype(np.float64) ** 2).sum(axis=0)
        n += m.sum(axis=0)
    n = np.maximum(n, 1.0)
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


def make_input(X: np.ndarray, d: int, lookback_days: int, code_idx: np.ndarray,
               mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(D,N,B,F) -> (n, lookback_days*B, F) float32, zscore 后缺失填 0。
    窗口不足 lookback_days 时前端补 0 (仅推理期用, 训练期跳过这类样本)。"""
    q = d - lookback_days + 1
    win = X[max(q, 0): d + 1][:, code_idx].astype(np.float32)   # (L',n,B,F)
    z = (win - mu) / (sd + 1e-8)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if q < 0:
        pad = np.zeros((-q, *z.shape[1:]), dtype=np.float32)
        z = np.concatenate([pad, z], axis=0)
    Ld, n, B, Fn = z.shape
    return z.transpose(1, 0, 2, 3).reshape(n, Ld * B, Fn)


class PanelTensor:
    """panel X 常驻 device 的切窗器: 训练主瓶颈是每 batch 在 CPU 用 numpy 组装
    (切窗+标准化+reshape+拷贝, (1000,384,28) 每天一次), GPU 大部分时间在等数据。
    整块 fp16 面板 (~6GB) 放上 GPU 后, 切窗/标准化全在 GPU 上做; 显存不足时
    自动回退 CPU 张量 (仍省去 float32 中间拷贝)。语义与 make_input 一致。"""

    def __init__(self, panel: dict, mu: np.ndarray, sd: np.ndarray,
                 lookback_days: int, device, keep_on_device: bool = True):
        X = torch.from_numpy(panel["X"])                 # fp16, 零拷贝视图
        self.device = device
        if device.type == "cuda" and keep_on_device:
            try:
                X = X.to(device)
            except RuntimeError:                          # 显存不足回退
                torch.cuda.empty_cache()
                print(f"[{now_s()}] [warn] panel ({X.nbytes / 1e9:.1f} GB) "
                      f"放不进显存, 回退 CPU 常驻 + 逐批拷贝")
        self.X = X
        self.mu = torch.from_numpy(np.asarray(mu, np.float32)).to(device)
        self.sd = torch.from_numpy(np.asarray(sd, np.float32)).to(device)
        self.lookback = int(lookback_days)

    def make(self, d: int, code_idx: np.ndarray) -> torch.Tensor:
        """-> (n, lookback_days*B, F) float32 on device, zscore 后缺失填 0"""
        idx = torch.as_tensor(code_idx, dtype=torch.long, device=self.X.device)
        q = d - self.lookback + 1
        win = self.X[max(q, 0): d + 1].index_select(1, idx)   # (L',n,B,F) fp16
        z = win.to(self.device, non_blocking=True).float()
        z = (z - self.mu) / (self.sd + 1e-8)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if q < 0:
            pad = torch.zeros((-q, *z.shape[1:]), device=self.device)
            z = torch.cat([pad, z], dim=0)
        Ld, n, B, Fn = z.shape
        return z.permute(1, 0, 2, 3).reshape(n, Ld * B, Fn)


class MultiFrequencyPanelTensor:
    """E3A 双频率切窗器。

    base 是第 d 日完整 5m；coarse 默认是 [d-10, d) 的 30m 原始数据。
    coarse 同时返回 bar/day mask，避免补零历史被网络当成真实观测。
    """
    def __init__(self, panels: dict, norms: dict, config: dict, device,
                 keep_on_device: bool = False):
        self.device = device
        self.base = PanelTensor(
            panels["5m"], norms["5m"][0], norms["5m"][1],
            int(config.get("lookback_days", 1)), device,
            keep_on_device=keep_on_device)
        self.coarse_frequency = str(config.get("mf_coarse_frequency", "30m"))
        coarse = panels[self.coarse_frequency]
        X = torch.from_numpy(coarse["X"])
        # 只预计算一次 bar 有效性；避免每个截面反复扫描10日×全部字段。
        bar_valid = torch.isfinite(X).any(dim=-1)
        if device.type == "cuda" and keep_on_device:
            try:
                X_device = X.to(device)
                valid_device = bar_valid.to(device)
                X, bar_valid = X_device, valid_device
            except RuntimeError:
                torch.cuda.empty_cache()
                print(f"[{now_s()}] [warn] {self.coarse_frequency} panel "
                      f"({X.nbytes / 1e9:.1f} GB) 放不进显存，回退CPU")
        self.coarse_X = X
        self.coarse_bar_valid = bar_valid
        cmu, csd = norms[self.coarse_frequency]
        self.coarse_mu = torch.from_numpy(np.asarray(cmu, np.float32)).to(device)
        self.coarse_sd = torch.from_numpy(np.asarray(csd, np.float32)).to(device)
        self.coarse_days = int(config.get("mf_coarse_days", 10))
        self.include_today = bool(config.get("mf_coarse_include_today", False))

    def make(self, d: int, code_idx: np.ndarray) -> dict[str, torch.Tensor]:
        base = self.base.make(d, code_idx)
        idx = torch.as_tensor(code_idx, dtype=torch.long,
                              device=self.coarse_X.device)
        end = d + 1 if self.include_today else d
        start = end - self.coarse_days
        lo, hi = max(start, 0), max(end, 0)
        win = self.coarse_X[lo:hi].index_select(1, idx)  # (D',n,B,F)
        bar_mask = self.coarse_bar_valid[lo:hi].index_select(1, idx)
        z = win.to(self.device, non_blocking=True).float()
        z = (z - self.coarse_mu) / (self.coarse_sd + 1e-8)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        pad_days = self.coarse_days - z.size(0)
        if pad_days > 0:
            zpad = torch.zeros((pad_days, *z.shape[1:]), device=self.device,
                               dtype=z.dtype)
            mpad = torch.zeros((pad_days, *bar_mask.shape[1:]),
                               device=bar_mask.device, dtype=torch.bool)
            z = torch.cat([zpad, z], dim=0)
            bar_mask = torch.cat([mpad, bar_mask], dim=0)
        bar_mask = bar_mask.to(self.device, non_blocking=True)
        coarse = z.permute(1, 0, 2, 3).contiguous()      # (n,D,B,F)
        bar_mask = bar_mask.permute(1, 0, 2).contiguous()
        return {
            "base": base,
            "coarse": coarse,
            "coarse_bar_mask": bar_mask,
            "coarse_day_mask": bar_mask.any(dim=-1),
        }


class ResidualFrequencyPanelTensor:
    """两阶段双频率切片器：K日5m主干 + 同日240根1m残差分支。

    两个频率使用独立训练期统计量。若某只股票当日缺少 1m 数据，fine_valid=False，
    模型会令其增量严格为 0，仍保留冻结5m主干的基础分数。
    """
    def __init__(self, panels: dict, norms: dict, config: dict, device,
                 keep_on_device: bool = False,
                 include_base_tensor: bool = True):
        self.device = device
        self.base = None
        if include_base_tensor:
            if "X" not in panels["5m"]:
                raise ValueError(
                    "include_base_tensor=True 但5m panel已经释放X")
            self.base = PanelTensor(
                panels["5m"], norms["5m"][0], norms["5m"][1],
                int(config.get("lookback_days", 1)), device,
                keep_on_device=keep_on_device)
        self.fine_frequency = str(config.get("residual_frequency", "1m"))
        self.fine = PanelTensor(
            panels[self.fine_frequency],
            norms[self.fine_frequency][0], norms[self.fine_frequency][1],
            1, device, keep_on_device=keep_on_device)
        valid = torch.from_numpy(
            np.asarray(panels[self.fine_frequency]["valid"], dtype=bool))
        if device.type == "cuda" and keep_on_device:
            try:
                valid = valid.to(device)
            except RuntimeError:
                torch.cuda.empty_cache()
        self.fine_valid = valid

    def make(self, d: int, code_idx: np.ndarray,
             base_score: np.ndarray | torch.Tensor | None = None,
             include_base: bool = True) -> dict[str, torch.Tensor]:
        idx = torch.as_tensor(code_idx, dtype=torch.long,
                              device=self.fine_valid.device)
        out = {
            "fine": self.fine.make(d, code_idx),
            "fine_valid": self.fine_valid[d].index_select(0, idx).to(
                self.device, non_blocking=True),
        }
        if include_base:
            if self.base is None:
                raise RuntimeError(
                    "当前训练切片器不持有5m原始X；请传入预计算base_score并"
                    "设置include_base=False")
            out["base"] = self.base.make(d, code_idx)
        if base_score is not None:
            if not torch.is_tensor(base_score):
                base_score = torch.from_numpy(np.asarray(base_score, np.float32))
            out["base_score"] = base_score.to(self.device, non_blocking=True).float()
        return out


def cs_target_zscore(y: np.ndarray, mad_n: float) -> np.ndarray:
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826
    if mad > 0:
        y = np.clip(y, med - mad_n * mad, med + mad_n * mad)
    sd = y.std()
    return (y - y.mean()) / (sd + 1e-8)


# ============================================================
# 5. 模型: 第一版骨架 (双尺度分组 TCN, 移植自 train0703_snapshot_1m)
#    + 阶段1 inter 组 (log 域幂积单元, 见 EXP_PLAN.md; 交互版存 archive_interact/)
# ============================================================

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        res = self.residual(x)
        out = self.conv1(x)[:, :, :x.size(2)]
        out = self.dropout(self.act(self.bn1(out)))
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.dropout(self.act(self.bn2(out)))
        return self.act(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3,
                 dilation_list=(1, 2, 4), dropout=0.2, progressive=True):
        super().__init__()
        n = len(dilation_list)
        if progressive and n > 1:
            widths = [max(hidden_channels * (i + 1) // n,
                          min(hidden_channels, 8)) for i in range(n)]
        else:
            widths = [hidden_channels] * n
        layers = []
        for i, dilation in enumerate(dilation_list):
            inc = in_channels if i == 0 else widths[i - 1]
            layers.append(TCNResidualBlock(inc, widths[i], kernel_size,
                                           dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x).squeeze(-1)).unsqueeze(-1)
        return x * w


class CrossFieldMixer(nn.Module):
    """inter 组: log 域幂积单元 (EXP_PLAN 阶段1)。
    输入字段已在 log 域, 故逐时点 1x1 线性 (零时间混合) 等价于原始域的乘除幂:
        sum_i w_i * log(x_i) = log( prod_i x_i^{w_i} )
    可端到端学出 log(bv1/av1)(~imb1)、log(amount/volume)(~vwap)、
    log(ask1/bid1)(~相对价差) 等交互序列; 输出保持 log 域 (单调等价),
    其时序结构交由下游 TCN 提取 (mixer 只扩字段轴, 时间轴仍归 TCN)。
    容量三重约束: K 小 + 损失端 L1 稀疏 (inter_l1) + BN 定标。"""

    def __init__(self, n_fields: int, k: int):
        super().__init__()
        self.conv = nn.Conv1d(n_fields, k, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(k)

    def forward(self, x):                    # (N, F, L) -> (N, K, L)
        return self.bn(self.conv(x))


class TemporalOperatorBank(nn.Module):
    """差分型时间算子组 (框架第3-6点): 每字段产出若干阶差分 x[t]-x[t-k]
    (= k-bar 对数收益/动量), 零参数。刻意只用差分 (与电平去相关), 不含 MA/平滑 --
    sim_verify_claims 已证 raw 与 MA/frac-diff 共线 0.92-0.98、拼接后 OOS 掉 40%。
    输出按字段展开 (字段 i 的 n_ops 条通道连续), 与所属组的原始通道 concat 后进 TCN。"""

    def __init__(self, diffs):
        super().__init__()
        self.diffs = [int(k) for k in diffs]
        self.n_ops = len(self.diffs)

    def forward(self, x):                        # (N, F, L) -> (N, F*n_ops, L)
        outs = []
        for k in self.diffs:
            xk = torch.zeros_like(x)
            if 0 < k < x.size(2):
                xk[:, :, k:] = x[:, :, :-k]
            outs.append(x - xk)
        o = torch.stack(outs, dim=2)             # (N, F, n_ops, L)
        N, Fn, n, L = o.shape
        return o.reshape(N, Fn * n, L)


class GRNFusion(nn.Module):
    """门控残差网络融合 (框架第10点, TFT 的 GRN): ELU 投影 + GLU 门控 + 残差 + BN,
    替代原 1x1conv+BN+SE。门控让融合层按样本自适应地抑制无用通道。"""

    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.skip = (nn.Conv1d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())
        self.fc1 = nn.Conv1d(in_ch, out_ch, 1)
        self.fc2 = nn.Conv1d(out_ch, out_ch, 1)
        self.gate = nn.Conv1d(out_ch, out_ch * 2, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                        # (N, in_ch, T) -> (N, out_ch, T)
        h = self.drop(self.fc2(F.elu(self.fc1(x))))
        a, b = self.gate(h).chunk(2, dim=1)
        return self.bn(self.skip(x) + a * torch.sigmoid(b))    # GLU + 残差


class TSCorrOperators(nn.Module):
    """AlphaNet 式零参数窗口交互算子 (二期): 对 curated 字段集在长度 W 的滑窗上
    算两两 ts_corr (量价协同变化) + 单字段 ts_std (窗口波动), 步长 = long_pool
    对齐长分支时间栅格。算子本身无可学参数 (仅末端 BN 定标); 学习发生在下游
    fusion 的加权。corr 对 zscore 尺度不变, 故与 inter 组的乘除幂正交互补。"""

    def __init__(self, corr_idx, long_pool, window, long_steps):
        super().__init__()
        self.register_buffer("corr_idx",
                             torch.as_tensor(corr_idx, dtype=torch.long))
        n = len(corr_idx)
        ii, jj = torch.triu_indices(n, n, offset=1)
        self.register_buffer("pi", ii)
        self.register_buffer("pj", jj)
        self.step = int(long_pool)
        self.window = int(window)
        self.long_steps = int(long_steps)
        self.n_ops = int(ii.numel()) + n         # 两两 corr + 每字段 std
        self.bn = nn.BatchNorm1d(self.n_ops)

    def forward(self, x):                        # x: (N, F, L) -> (N, n_ops, long_steps)
        xs = x.index_select(1, self.corr_idx)                 # (N, n, L)
        W = min(self.window, xs.size(2))
        win = xs.unfold(2, W, self.step)                      # (N, n, nwin, W)
        c = win - win.mean(-1, keepdim=True)
        ss = (c * c).sum(-1)                                  # (N, n, nwin)
        std = torch.sqrt(ss / max(W - 1, 1) + EPS)
        ca, cb = c.index_select(1, self.pi), c.index_select(1, self.pj)
        cov = (ca * cb).sum(-1)                               # (N, P, nwin)
        denom = torch.sqrt(ss.index_select(1, self.pi)
                           * ss.index_select(1, self.pj)) + EPS
        out = torch.cat([cov / denom, std], dim=1)            # (N, n_ops, nwin)
        nwin = out.size(2)
        if nwin < self.long_steps:                            # 前端补零对齐栅格
            pad = out.new_zeros(out.size(0), out.size(1),
                                self.long_steps - nwin)
            out = torch.cat([pad, out], dim=2)
        elif nwin > self.long_steps:
            out = out[:, :, -self.long_steps:]
        return self.bn(out)


def _resolve_field_index(fields: list[str], name: str) -> int | None:
    """把配置中的规范名映射到实际表字段；price/close 双向兼容。"""
    if name in fields:
        return fields.index(name)
    if name == "price":
        for cand in FIELD_ALIASES["price"]:
            if cand in fields:
                return fields.index(cand)
    if name == "close" and "price" in fields:
        return fields.index("price")
    return None


class ChannelLayerNorm(nn.Module):
    """对每个时间点的通道做 LayerNorm；不依赖当日股票数或推理 chunk。"""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalDSResidualBlock(nn.Module):
    """轻量因果 depthwise-separable TCN + GLU + 残差。"""
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                   dilation=dilation, padding=pad,
                                   groups=channels)
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.project = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = ChannelLayerNorm(channels)

    def forward(self, x):
        h = self.depthwise(x)[:, :, :x.size(2)]
        a, b = self.expand(h).chunk(2, dim=1)
        h = self.project(a * torch.sigmoid(b))
        return self.norm(x + self.drop(h))


class ReturnOperatorBank(nn.Module):
    """模型内固定 ts_return：只在单日 B 根 bar 内做差分，绝不跨隔夜。"""
    def __init__(self, field_idx, horizons):
        super().__init__()
        self.register_buffer("field_idx", torch.as_tensor(field_idx, dtype=torch.long))
        self.horizons = [int(k) for k in horizons]
        self.n_ops = len(field_idx) * len(self.horizons)

    def forward(self, x):                       # (N*D,F,B) -> (N*D,n_ops,B)
        # 外层训练可开启 AMP；固定差分必须保留 float32 的细微价格变化。
        with torch.autocast(device_type=x.device.type, enabled=False):
            xs = x.float().index_select(1, self.field_idx)
            outs = []
            for k in self.horizons:
                d = torch.zeros_like(xs)
                if 0 < k < xs.size(2):
                    d[:, :, k:] = xs[:, :, k:] - xs[:, :, :-k]
                outs.append(d)
            return torch.cat(outs, dim=1)


class RollingReturnCorrBank(nn.Module):
    """白名单字段的一阶变化滚动相关；算子固定、单日内计算、前端不足窗口处补0。"""
    def __init__(self, pair_idx, windows):
        super().__init__()
        ai = [p[0] for p in pair_idx]
        bi = [p[1] for p in pair_idx]
        self.register_buffer("ai", torch.as_tensor(ai, dtype=torch.long))
        self.register_buffer("bi", torch.as_tensor(bi, dtype=torch.long))
        self.windows = [int(w) for w in windows]
        self.n_ops = len(pair_idx) * len(self.windows)

    def forward(self, x):                       # (N*D,F,B) -> (N*D,n_ops,B)
        # corr 的零方差窗口在 fp16 下会令 1e-12 下溢为0；强制 float32 并清理非有限值。
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            r = torch.zeros_like(xf)
            r[:, :, 1:] = xf[:, :, 1:] - xf[:, :, :-1]
            xa = r.index_select(1, self.ai)
            xb = r.index_select(1, self.bi)
            outs = []
            B = xf.size(2)
            for w in self.windows:
                W = min(max(w, 2), B)
                aw = xa.unfold(2, W, 1)
                bw = xb.unfold(2, W, 1)
                ac = aw - aw.mean(-1, keepdim=True)
                bc = bw - bw.mean(-1, keepdim=True)
                num = (ac * bc).sum(-1)
                den = torch.sqrt(ac.square().sum(-1) * bc.square().sum(-1))
                corr = num / den.clamp_min(1e-6)
                corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                corr = corr.clamp(-1.0, 1.0)
                if corr.size(2) < B:
                    corr = F.pad(corr, (B - corr.size(2), 0))
                outs.append(corr)
            return torch.cat(outs, dim=1)


class RollingReturnStdBank(nn.Module):
    """白名单字段一阶变化的因果 rolling std；只在单日内计算。

    时刻 t 的输出只使用 [t-W+1, t]，窗口不足处补 0。算子在 float32
    中计算，避免 AMP 下微小量价变化的方差下溢。
    """
    def __init__(self, field_idx, windows):
        super().__init__()
        self.register_buffer("field_idx", torch.as_tensor(field_idx, dtype=torch.long))
        self.windows = [int(w) for w in windows]
        self.n_ops = len(field_idx) * len(self.windows)

    def forward(self, x):                       # (N*D,F,B) -> (N*D,n_ops,B)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xs = x.float().index_select(1, self.field_idx)
            r = torch.zeros_like(xs)
            r[:, :, 1:] = xs[:, :, 1:] - xs[:, :, :-1]
            outs = []
            B = xs.size(2)
            for w in self.windows:
                W = min(max(w, 2), B)
                rw = r.unfold(2, W, 1)
                rc = rw - rw.mean(-1, keepdim=True)
                std = torch.sqrt(rc.square().mean(-1).clamp_min(1e-12))
                std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
                if std.size(2) < B:
                    std = F.pad(std, (B - std.size(2), 0))
                outs.append(std)
            return torch.cat(outs, dim=1)


class OperatorEncoder(nn.Module):
    def __init__(self, in_channels, d_model, dilations, dropout):
        super().__init__()
        layers = [nn.Conv1d(in_channels, d_model, 1),
                  ChannelLayerNorm(d_model), nn.GELU()]
        layers += [CausalDSResidualBlock(d_model, 3, d, dropout)
                   for d in dilations]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalLastAttentionPool(nn.Module):
    """保留最新状态，同时让模型从全窗口选择补充历史信息。"""
    def __init__(self, channels, dropout):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.score = nn.Sequential(nn.Linear(channels, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1, bias=False))
        self.proj = nn.Sequential(nn.Linear(channels * 2, channels),
                                  nn.LayerNorm(channels), nn.GELU(),
                                  nn.Dropout(dropout))

    def forward(self, x):                       # (N,C,T) -> (N,C)
        h = x.transpose(1, 2)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        attn = torch.sum(h * w.unsqueeze(-1), dim=1)
        return self.proj(torch.cat([h[:, -1], attn], dim=-1))


def _fixed_sinusoidal_day_encoding(length: int, channels: int,
                                   base: float = 10000.0) -> torch.Tensor:
    """Return centered, unit-scale fixed sin/cos encodings, shape (K, C).

    Vaswani et al. use deterministic sine/cosine functions at geometrically
    spaced frequencies. Here the positions are rolling day lags rather than
    word indices. Removing each channel's mean discards the common offset and
    preserves only position information; it also makes K=1 exactly zero so the
    one-day E4A path remains bitwise unchanged.
    """
    length, channels = int(length), int(channels)
    base = float(base)
    if length < 1 or channels < 1:
        raise ValueError(
            f"sinusoidal day encoding expects positive shape, got "
            f"({length}, {channels})")
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError(f"v2_day_sincos_base must be finite and >1, got {base}")

    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    even_dims = torch.arange(0, channels, 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(base) * even_dims / channels)
    angles = position * frequencies.unsqueeze(0)
    pe = torch.zeros(length, channels, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    if channels > 1:
        pe[:, 1::2] = torch.cos(angles[:, :pe[:, 1::2].shape[1]])

    pe = pe - pe.mean(dim=0, keepdim=True)
    scale = pe.std(unbiased=False)
    if length == 1 or not torch.isfinite(scale) or float(scale) <= EPS:
        return torch.zeros_like(pe)
    return pe / scale


class RawHierarchicalOperatorNet(nn.Module):
    """V2：纯原始字段主干 + 模型内固定 return/corr/std。

    ``v2_long_window_mode='shared_day_gru_residual'`` 是 E5H0 路径：
    K 天分别通过同一套 E1 日内编码器，固定算子在每个交易日边界重置；
    每天池化为一个 d_model 维向量，再由单层 GRU 产生跨日增量分数。
    增量头零初始化，因此 epoch 0 与最近一天的 E1 分数严格等价。

    ``v2_use_crossday=False`` 且 ``v2_long_window_mode='continuous_time_axis'``
    时，K 日输入被视为一条 K*48 步连续序列，原 raw/TCN/return/corr/std、
    ``intraday_pool`` 与 ``head`` 全部直接复用。每个交易日加入固定正余弦日
    编码并广播给当日48根bar；仅最近48步额外使用E4A原有 ``intraday_pos``。
    不复制日内位置表，也不增加参数。该路径有意不在日界重置。

    ``shared_intraday_concat`` 作为上一轮日界重置实验的兼容路径保留，但不是
    当前默认实验。

    旧的层级 GRU 路径仅为历史 checkpoint 兼容保留，只有显式设置
    ``v2_use_crossday=True`` 才会实例化。
    """
    def __init__(self, config, fields, groups, bars_per_day):
        super().__init__()
        c = config
        self.n_fields = len(fields)
        self.lookback_days = int(c["lookback_days"])
        self.bars_per_day = int(bars_per_day)
        self.lookback = self.lookback_days * self.bars_per_day
        d_model = int(c.get("v2_d_model", 96))
        dropout = float(c.get("dropout", 0.2))

        group_dims = c.get("v2_group_dims", {"price": 32, "depth": 32, "flow": 16})
        self.groups = groups
        self.group_proj = nn.ModuleDict()
        raw_dim = 0
        for name, idx in groups.items():
            width = int(group_dims.get(name, 16))
            self.group_proj[name] = nn.Conv1d(len(idx), width, 1)
            raw_dim += width
        self.raw_fuse = nn.Sequential(nn.Conv1d(raw_dim, d_model, 1),
                                      ChannelLayerNorm(d_model), nn.GELU())
        self.intraday_pos = nn.Parameter(
            torch.zeros(1, d_model, self.bars_per_day))
        nn.init.normal_(self.intraday_pos, std=0.02)
        self.raw_encoder = nn.Sequential(*[
            CausalDSResidualBlock(d_model, 3, int(d), dropout)
            for d in c.get("v2_intra_dilations", (1, 2, 4, 8))
        ])

        op_dil = c.get("v2_operator_dilations", (1, 2))
        self.use_return = bool(c.get("v2_use_return", True))
        if self.use_return:
            return_idx = []
            for name in c.get("v2_return_fields", []):
                i = _resolve_field_index(fields, name)
                if i is not None and i not in return_idx:
                    return_idx.append(i)
            if not return_idx:
                raise ValueError("V2 没有可用的 v2_return_fields")
            self.return_bank = ReturnOperatorBank(
                return_idx, c.get("v2_return_horizons", (1, 3, 6, 12)))
            self.return_encoder = OperatorEncoder(
                self.return_bank.n_ops, d_model, op_dil, dropout)
            self.return_gate = nn.Parameter(torch.zeros(1, d_model, 1))

        self.use_corr = bool(c.get("v2_use_corr", True))
        if self.use_corr:
            pair_idx = []
            for a, b in c.get("v2_corr_pairs", []):
                ia, ib = _resolve_field_index(fields, a), _resolve_field_index(fields, b)
                if ia is not None and ib is not None and ia != ib:
                    pair_idx.append((ia, ib))
            if not pair_idx:
                raise ValueError("V2 没有可用的 v2_corr_pairs")
            self.corr_bank = RollingReturnCorrBank(
                pair_idx, c.get("v2_corr_windows", (12, 24, 48)))
            self.corr_encoder = OperatorEncoder(
                self.corr_bank.n_ops, d_model, op_dil, dropout)
            self.corr_gate = nn.Parameter(torch.zeros(1, d_model, 1))

        self.fusion_norm = ChannelLayerNorm(d_model)

        self.intraday_pool = TemporalLastAttentionPool(d_model, dropout)
        self.long_window_mode = str(
            c.get("v2_long_window_mode", "shared_intraday_concat"))
        self.long_position_mode = str(
            c.get("v2_long_position_mode", "recent_day_only"))
        self.day_sincos_base = float(c.get("v2_day_sincos_base", 10000.0))
        self.day_sincos_scale = float(c.get("v2_day_sincos_scale", 0.02))
        self.use_crossday = bool(c.get("v2_use_crossday", True))
        if self.use_crossday:
            self.day_pos = nn.Parameter(torch.zeros(1, self.lookback_days, d_model))
            nn.init.normal_(self.day_pos, std=0.02)
            gru_layers = int(c.get("v2_gru_layers", 2))
            self.day_encoder = nn.GRU(d_model, d_model, num_layers=gru_layers,
                                      batch_first=True,
                                      dropout=dropout if gru_layers > 1 else 0.0)
            self.day_norm = nn.LayerNorm(d_model)
            self.crossday_pool = TemporalLastAttentionPool(d_model, dropout)
        elif self.long_window_mode not in {
                "shared_intraday_concat", "continuous_time_axis",
                "shared_day_gru_residual"}:
            raise ValueError(
                "v2_use_crossday=False 仅支持 v2_long_window_mode in "
                "{'shared_intraday_concat', 'continuous_time_axis', "
                "'shared_day_gru_residual'}")
        elif self.long_window_mode == "shared_day_gru_residual":
            if self.lookback_days < 2:
                raise ValueError(
                    "shared_day_gru_residual 至少需要 2 个交易日")
            self.day_gru_hidden = int(c.get("v2_day_gru_hidden", 48))
            day_gru_layers = int(c.get("v2_day_gru_layers", 1))
            if day_gru_layers != 1:
                raise ValueError(
                    "E5H0 为受控实验，只允许单层 day GRU")
            if self.day_gru_hidden < 8:
                raise ValueError("v2_day_gru_hidden 必须至少为 8")
            self.day_gru = nn.GRU(
                d_model, self.day_gru_hidden, num_layers=1,
                batch_first=True)
            delta_hidden = max(self.day_gru_hidden // 2, 16)
            self.day_delta_head = nn.Sequential(
                nn.Linear(self.day_gru_hidden, delta_hidden),
                nn.LayerNorm(delta_hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(delta_hidden, 1, bias=False),
            )
            # 训练开始时跨日增量严格为 0；原 E1 当日分数完整保留。
            nn.init.zeros_(self.day_delta_head[-1].weight)
        elif self.long_window_mode == "continuous_time_axis":
            allowed_position_modes = {
                "recent_day_only",
                "recent_day_plus_fixed_day_sincos",
            }
            if self.long_position_mode not in allowed_position_modes:
                raise ValueError(
                    "continuous_time_axis 的 v2_long_position_mode 仅允许 "
                    f"{sorted(allowed_position_modes)}，禁止复制日内位置编码")
            if (not math.isfinite(self.day_sincos_scale)
                    or self.day_sincos_scale < 0.0):
                raise ValueError(
                    "v2_day_sincos_scale must be finite and non-negative, got "
                    f"{self.day_sincos_scale}")
            day_sincos = _fixed_sinusoidal_day_encoding(
                self.lookback_days, d_model, self.day_sincos_base)
            day_sincos = day_sincos.transpose(0, 1).unsqueeze(0)
            day_sincos = day_sincos.repeat_interleave(
                self.bars_per_day, dim=-1)
            if self.long_position_mode == "recent_day_only":
                day_sincos.zero_()
            else:
                day_sincos.mul_(self.day_sincos_scale)
            # Deterministically reconstructed from checkpoint config. Keeping
            # it non-persistent preserves the exact E4A state-dict signature.
            self.register_buffer(
                "fixed_day_sincos", day_sincos, persistent=False)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                  nn.LayerNorm(d_model // 2), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model // 2, 1))

        # 必须在所有基线模块之后创建std分支：这样同一种子下，既有主干、池化
        # 和head的初始化逐元素不变。
        self.use_std = bool(c.get("v2_use_std", False))
        if self.use_std:
            std_idx = []
            for name in c.get("v2_std_fields", []):
                i = _resolve_field_index(fields, name)
                if i is not None and i not in std_idx:
                    std_idx.append(i)
            if not std_idx:
                raise ValueError("V2 没有可用的 v2_std_fields")
            self.std_bank = RollingReturnStdBank(
                std_idx, c.get("v2_std_windows", (12, 24, 48)))
            self.std_encoder = nn.Sequential(
                nn.Conv1d(self.std_bank.n_ops, d_model, 1),
                ChannelLayerNorm(d_model), nn.GELU())
            self.std_gate = nn.Parameter(torch.zeros(1, d_model, 1))

    def encode(self, x, return_day_sequence: bool = False):
        if x.dim() != 3 or x.size(1) != self.lookback or x.size(2) != self.n_fields:
            raise ValueError(
                f"Expected (N, {self.lookback}, {self.n_fields}), got {tuple(x.shape)}")
        N = x.size(0)
        if not self.use_crossday and self.long_window_mode == "continuous_time_axis":
            # 连续240步不在日界重置TCN或固定算子。E4A的intraday_pos仍仅
            # 用于最近一天；固定日正余弦编码让全部5日可区分，但不复制日内表。
            xd = x.permute(0, 2, 1)
            history_steps = self.lookback - self.bars_per_day
            position = (F.pad(self.intraday_pos, (history_steps, 0))
                        + self.fixed_day_sincos)
        else:
            # 旧层级路径与上一轮逐日重置实验保留兼容。
            xd = x.reshape(N, self.lookback_days, self.bars_per_day,
                           self.n_fields).permute(0, 1, 3, 2)
            xd = xd.reshape(
                N * self.lookback_days, self.n_fields, self.bars_per_day)
            position = self.intraday_pos

        raw_parts = [self.group_proj[name](xd[:, idx, :])
                     for name, idx in self.groups.items()]
        raw = self.raw_fuse(torch.cat(raw_parts, dim=1)) + position
        raw = self.raw_encoder(raw)
        h = raw
        if self.use_return:
            ret = self.return_encoder(self.return_bank(xd))
            h = h + torch.tanh(self.return_gate) * ret
        if self.use_corr:
            corr = self.corr_encoder(self.corr_bank(xd))
            h = h + torch.tanh(self.corr_gate) * corr
        if self.use_std:
            std = self.std_encoder(self.std_bank(xd))
            h = h + torch.tanh(self.std_gate) * std
        h = self.fusion_norm(h)

        if self.use_crossday:
            day = self.intraday_pool(h).reshape(N, self.lookback_days, -1)
            day = day + self.day_pos
            day, _ = self.day_encoder(day)
            day = self.day_norm(day)
            summary = self.crossday_pool(day.transpose(1, 2))
        elif self.long_window_mode == "shared_day_gru_residual":
            day = self.intraday_pool(h).reshape(N, self.lookback_days, -1)
            if return_day_sequence:
                return day
            # encode() 保持“返回主干表示”的既有接口；跨日增量只在 forward()
            # 中进入最终分数，方便其他代码读取最近一天的 E1 表示。
            summary = day[:, -1]
        elif self.long_window_mode == "shared_intraday_concat":
            # h 中每一天已经由同一套单日5m网络独立编码，固定算子也在
            # 日界处重置。这里只做无参数的时间轴重排：
            # (N*K,C,48) -> (N,C,K*48)，随后复用原有 intraday_pool。
            # K=1 时该兼容路径与单日5m模型逐元素等价。
            channels = h.size(1)
            h = h.reshape(
                N, self.lookback_days, channels, self.bars_per_day
            ).permute(0, 2, 1, 3).reshape(N, channels, self.lookback)
            summary = self.intraday_pool(h)
        else:
            # continuous_time_axis: h 已经是 (N,C,K*48)。
            summary = self.intraday_pool(h)
        return summary

    def forward(self, x):
        if (not self.use_crossday
                and self.long_window_mode == "shared_day_gru_residual"):
            day = self.encode(x, return_day_sequence=True)
            today = day[:, -1]
            day_hidden, _ = self.day_gru(day)
            context = day_hidden[:, -1]
            # 两个小打分头都固定在 FP32，避免 AMP 下截面分数分辨率下降。
            with torch.autocast(device_type=today.device.type, enabled=False):
                base_score = self.head(today.float()).squeeze(-1)
                delta_score = self.day_delta_head(
                    context.float()).squeeze(-1)
                return base_score + delta_score
        summary = self.encode(x)
        # Rank losses are invariant to a common score offset, but an FP16 final
        # head can turn that harmless offset into coarse quantisation.  Keep the
        # representation encoder under the caller's AMP policy and evaluate the
        # small score head in FP32.
        with torch.autocast(device_type=summary.device.type, enabled=False):
            return self.head(summary.float()).squeeze(-1)


class MaskedTemporalLastAttentionPool(nn.Module):
    """支持缺失 bar/day 的 last + attention pooling。"""
    def __init__(self, channels, dropout):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.score = nn.Sequential(nn.Linear(channels, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1, bias=False))
        self.proj = nn.Sequential(nn.Linear(channels * 2, channels),
                                  nn.LayerNorm(channels), nn.GELU(),
                                  nn.Dropout(dropout))

    def forward(self, x, mask):                 # x(N,C,T), mask(N,T)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        h = x.transpose(1, 2)
        valid_any = mask.any(dim=1)
        safe_mask = mask.clone()
        if (~valid_any).any():
            safe_mask[~valid_any, -1] = True
        logits = self.score(h).squeeze(-1).masked_fill(~safe_mask, -1e4)
        w = torch.softmax(logits, dim=1)
        attn = torch.sum(h * w.unsqueeze(-1), dim=1)
        positions = torch.arange(h.size(1), device=h.device).unsqueeze(0)
        last_idx = torch.where(safe_mask, positions, -1).max(dim=1).values
        last = h[torch.arange(h.size(0), device=h.device), last_idx]
        out = self.proj(torch.cat([last, attn], dim=-1))
        return torch.where(valid_any.unsqueeze(-1), out, torch.zeros_like(out))


class CoarseContextEncoder(nn.Module):
    """过去若干完整交易日的 30m 原始数据层级编码器。

    第一级在每天 8 根 bar 内共享参数；第二级只处理日级表示，明确保留日界。
    """
    def __init__(self, config, fields, groups, bars_per_day):
        super().__init__()
        c = config
        self.n_fields = len(fields)
        self.days = int(c.get("mf_coarse_days", 10))
        self.bars_per_day = int(bars_per_day)
        self.d_model = int(c.get("mf_coarse_d_model", 64))
        dropout = float(c.get("dropout", 0.1))
        dims = c.get("mf_coarse_group_dims",
                     {"price": 16, "depth": 16, "flow": 8})
        self.groups = groups
        self.group_proj = nn.ModuleDict()
        raw_dim = 0
        for name, idx in groups.items():
            width = int(dims.get(name, 8))
            self.group_proj[name] = nn.Conv1d(len(idx), width, 1)
            raw_dim += width
        self.raw_fuse = nn.Sequential(nn.Conv1d(raw_dim, self.d_model, 1),
                                      ChannelLayerNorm(self.d_model), nn.GELU())
        self.intraday_pos = nn.Parameter(
            torch.zeros(1, self.d_model, self.bars_per_day))
        nn.init.normal_(self.intraday_pos, std=0.02)
        self.intra_blocks = nn.ModuleList([
            CausalDSResidualBlock(self.d_model, 3, int(d), dropout)
            for d in c.get("mf_coarse_intra_dilations", (1, 2))
        ])
        self.intraday_pool = MaskedTemporalLastAttentionPool(self.d_model, dropout)

        self.day_pos = nn.Parameter(torch.zeros(1, self.days, self.d_model))
        nn.init.normal_(self.day_pos, std=0.02)
        self.day_blocks = nn.ModuleList([
            CausalDSResidualBlock(self.d_model, 3, int(d), dropout)
            for d in c.get("mf_coarse_day_dilations", (1, 2, 4))
        ])
        self.day_norm = nn.LayerNorm(self.d_model)
        self.day_pool = MaskedTemporalLastAttentionPool(self.d_model, dropout)

    def forward(self, x, bar_mask, day_mask):
        if (x.dim() != 4 or x.size(1) != self.days
                or x.size(2) != self.bars_per_day
                or x.size(3) != self.n_fields):
            raise ValueError(
                f"Expected coarse (N,{self.days},{self.bars_per_day},{self.n_fields}), "
                f"got {tuple(x.shape)}")
        N, D, B, Fn = x.shape
        xd = x.reshape(N * D, B, Fn).transpose(1, 2)
        bm = bar_mask.reshape(N * D, B)
        mbar = bm.unsqueeze(1).to(dtype=xd.dtype)
        parts = [self.group_proj[name](xd[:, idx, :])
                 for name, idx in self.groups.items()]
        h = self.raw_fuse(torch.cat(parts, dim=1))
        h = (h + self.intraday_pos) * mbar
        for block in self.intra_blocks:
            h = block(h) * mbar
        day = self.intraday_pool(h, bm).reshape(N, D, self.d_model)

        dm = day_mask.to(dtype=day.dtype).unsqueeze(-1)
        day = (day + self.day_pos) * dm
        hday = day.transpose(1, 2)
        md = day_mask.unsqueeze(1).to(dtype=hday.dtype)
        for block in self.day_blocks:
            hday = block(hday) * md
        hday = self.day_norm(hday.transpose(1, 2)) * dm
        return self.day_pool(hday.transpose(1, 2), day_mask)


class FiveMinuteBlockPool(nn.Module):
    """把连续5根1m隐状态压成一个与E4A 5m槽位对齐的token。

    last 保留块末状态，mean 汇总块内路径；二者拼接后再投影，不使用任何人工因子。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1),
            ChannelLayerNorm(channels), nn.GELU())

    def forward(self, x):                       # (N,C,240) -> (N,C,48)
        if x.size(2) % 5:
            raise ValueError(f"1m 序列长度 {x.size(2)} 不能被5整除")
        N, C, T = x.shape
        z = x.reshape(N, C, T // 5, 5)
        return self.proj(torch.cat([z[..., -1], z.mean(dim=-1)], dim=1))


class OneMinuteResidualEncoder(nn.Module):
    """E4A同日1m原始量价/盘口残差编码器。"""
    def __init__(self, config, fields, bars_per_day):
        super().__init__()
        c = config
        self.fields = list(fields)
        self.n_fields = len(fields)
        self.bars_per_day = int(bars_per_day)
        if self.bars_per_day != 240:
            raise ValueError(f"E4A 预期每天240根1m bar，实际 {self.bars_per_day}")
        self.blocks_per_day = self.bars_per_day // 5
        self.d_model = int(c.get("res1m_d_model", 64))
        dropout = float(c.get("dropout", 0.1))
        groups = make_groups(fields)
        dims = c.get("res1m_group_dims",
                     {"price": 24, "depth": 24, "flow": 16})

        self.groups = groups
        self.group_proj = nn.ModuleDict()
        raw_dim = 0
        for name, idx in groups.items():
            width = int(dims.get(name, 16))
            self.group_proj[name] = nn.Conv1d(len(idx), width, 1)
            raw_dim += width
        self.raw_fuse = nn.Sequential(
            nn.Conv1d(raw_dim, self.d_model, 1),
            ChannelLayerNorm(self.d_model), nn.GELU())
        self.minute_pos = nn.Parameter(
            torch.zeros(1, self.d_model, self.bars_per_day))
        nn.init.normal_(self.minute_pos, std=0.02)

        op_dil = c.get("res1m_operator_dilations", (1, 2))
        self.use_return = bool(c.get("res1m_use_return", True))
        if self.use_return:
            idx = []
            for name in c.get("res1m_return_fields", []):
                i = _resolve_field_index(fields, name)
                if i is not None and i not in idx:
                    idx.append(i)
            if not idx:
                raise ValueError("1m residual 没有可用的 return 字段")
            self.return_bank = ReturnOperatorBank(
                idx, c.get("res1m_return_horizons", (1, 2, 5, 10, 30, 60)))
            self.return_encoder = OperatorEncoder(
                self.return_bank.n_ops, self.d_model, op_dil, dropout)
            self.return_gate = nn.Parameter(torch.zeros(1, self.d_model, 1))

        self.use_corr = bool(c.get("res1m_use_corr", True))
        if self.use_corr:
            pairs = []
            for a, b in c.get("res1m_corr_pairs", []):
                ia, ib = _resolve_field_index(fields, a), _resolve_field_index(fields, b)
                if ia is not None and ib is not None and ia != ib:
                    pairs.append((ia, ib))
            if not pairs:
                raise ValueError("1m residual 没有可用的 corr 字段对")
            self.corr_bank = RollingReturnCorrBank(
                pairs, c.get("res1m_corr_windows", (5, 15, 30, 60)))
            self.corr_encoder = OperatorEncoder(
                self.corr_bank.n_ops, self.d_model, op_dil, dropout)
            self.corr_gate = nn.Parameter(torch.zeros(1, self.d_model, 1))

        self.use_std = bool(c.get("res1m_use_std", True))
        if self.use_std:
            idx = []
            for name in c.get("res1m_std_fields", []):
                i = _resolve_field_index(fields, name)
                if i is not None and i not in idx:
                    idx.append(i)
            if not idx:
                raise ValueError("1m residual 没有可用的 std 字段")
            self.std_bank = RollingReturnStdBank(
                idx, c.get("res1m_std_windows", (5, 15, 30, 60)))
            self.std_encoder = OperatorEncoder(
                self.std_bank.n_ops, self.d_model, op_dil, dropout)
            self.std_gate = nn.Parameter(torch.zeros(1, self.d_model, 1))

        self.fusion_norm = ChannelLayerNorm(self.d_model)
        self.local_encoder = nn.Sequential(*[
            CausalDSResidualBlock(self.d_model, 3, int(d), dropout)
            for d in c.get("res1m_local_dilations", (1, 2, 4))
        ])
        self.block_pool = FiveMinuteBlockPool(self.d_model)
        self.block_pos = nn.Parameter(
            torch.zeros(1, self.d_model, self.blocks_per_day))
        nn.init.normal_(self.block_pos, std=0.02)
        self.block_encoder = nn.Sequential(*[
            CausalDSResidualBlock(self.d_model, 3, int(d), dropout)
            for d in c.get("res1m_block_dilations", (1, 2, 4, 8))
        ])
        self.day_pool = TemporalLastAttentionPool(self.d_model, dropout)
        hidden = max(self.d_model // 2, 16)
        self.delta_head = nn.Sequential(
            nn.Linear(self.d_model, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        # 关键不变量：训练开始前组合输出逐元素等于冻结5m主干。
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        # Every loss used by E4A removes the cross-sectional score mean, so this
        # scalar bias is unidentified.  Letting Adam/AMP move it only reduces
        # FP16 resolution without changing the intended ranking.
        self.delta_head[-1].bias.requires_grad_(False)

    def forward(self, x):
        if (x.dim() != 3 or x.size(1) != self.bars_per_day
                or x.size(2) != self.n_fields):
            raise ValueError(
                f"Expected 1m (N,{self.bars_per_day},{self.n_fields}), got {tuple(x.shape)}")
        xd = x.transpose(1, 2)
        parts = [self.group_proj[name](xd[:, idx, :])
                 for name, idx in self.groups.items()]
        h = self.raw_fuse(torch.cat(parts, dim=1)) + self.minute_pos
        if self.use_return:
            h = h + torch.tanh(self.return_gate) * self.return_encoder(
                self.return_bank(xd))
        if self.use_corr:
            h = h + torch.tanh(self.corr_gate) * self.corr_encoder(
                self.corr_bank(xd))
        if self.use_std:
            h = h + torch.tanh(self.std_gate) * self.std_encoder(
                self.std_bank(xd))
        h = self.local_encoder(self.fusion_norm(h))
        h = self.block_encoder(self.block_pool(h) + self.block_pos)
        summary = self.day_pool(h)
        with torch.autocast(device_type=summary.device.type, enabled=False):
            return self.delta_head(summary.float()).squeeze(-1)


class FrozenE1OneMinuteResidualNet(nn.Module):
    """E4A两阶段模型；类名为兼容旧代码保留，不代表本实验以E1为基准。"""
    def __init__(self, config, base_fields, base_groups, base_bars,
                 fine_fields, fine_bars):
        super().__init__()
        self.base = RawHierarchicalOperatorNet(
            config, base_fields, base_groups, base_bars)
        self.residual = OneMinuteResidualEncoder(config, fine_fields, fine_bars)
        self.lookback = self.base.lookback
        self.n_fields = self.base.n_fields
        self.freeze_base()

    def enforce_zero_delta_bias(self):
        """Fix the additive-score gauge without changing any ranking in FP32."""
        bias = self.residual.delta_head[-1].bias
        with torch.no_grad():
            bias.zero_()
        bias.requires_grad_(False)
        return self

    def freeze_base(self):
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.enforce_zero_delta_bias()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # super().train会递归打开Dropout；立即把冻结5m主干恢复成推理状态。
        self.base.eval()
        return self

    def forward_base(self, batch):
        if "base_score" in batch:
            return batch["base_score"]
        with torch.no_grad():
            return self.base(batch["base"])

    def forward_delta(self, batch):
        delta = self.residual(batch["fine"])
        if "fine_valid" in batch:
            delta = delta * batch["fine_valid"].to(delta.dtype)
        return delta

    def forward_components(self, batch):
        # Force the additive score into FP32 even when the feature encoders run
        # under autocast.  This also makes chunked inference numerically stable.
        base = self.forward_base(batch).float()
        delta = self.forward_delta(batch).float()
        return {"base": base, "delta": delta, "final": base + delta}

    def forward(self, batch):
        return self.forward_components(batch)["final"]


class MultiFrequencyHierarchicalNet(nn.Module):
    """E3A：完整 E1 5m 主干 + 零门控 30m 跨日上下文。"""
    def __init__(self, config, base_fields, base_groups, base_bars,
                 coarse_fields, coarse_bars):
        super().__init__()
        # base 必须最先构造：相同 seed 下其初始化与独立 E1 逐元素一致。
        self.base = RawHierarchicalOperatorNet(
            config, base_fields, base_groups, base_bars)
        coarse_groups = make_groups(coarse_fields)
        self.coarse = CoarseContextEncoder(
            config, coarse_fields, coarse_groups, coarse_bars)
        base_dim = int(config.get("v2_d_model", 96))
        coarse_dim = int(config.get("mf_coarse_d_model", 64))
        self.coarse_proj = nn.Sequential(
            nn.Linear(coarse_dim, base_dim, bias=False),
            nn.LayerNorm(base_dim))
        gate_init = float(config.get("mf_gate_init", 0.0))
        self.coarse_gate = nn.Parameter(torch.full((1, base_dim), gate_init))
        self.lookback = self.base.lookback
        self.n_fields = self.base.n_fields
        self.use_return = self.base.use_return
        self.use_corr = self.base.use_corr
        self.use_std = self.base.use_std
        self.use_crossday = self.base.use_crossday

    def forward_components(self, batch):
        z5 = self.base.encode(batch["base"])
        z30 = self.coarse(
            batch["coarse"], batch["coarse_bar_mask"],
            batch["coarse_day_mask"])
        z30 = self.coarse_proj(z30)
        fused = z5 + torch.tanh(self.coarse_gate) * z30
        return {"base": z5, "coarse": z30, "fused": fused}

    def forward_base(self, batch):
        return self.base.head(self.base.encode(batch["base"])).squeeze(-1)

    def forward(self, batch):
        z = self.forward_components(batch)["fused"]
        return self.base.head(z).squeeze(-1)


class DailyMultiScaleTCN(nn.Module):
    """第一版骨架 (公榜最优) + 可选 inter 组 + 可选 ts_corr 分支:
    分组 TCN + SE -> concat(+算子分支) -> 1x1 融合(+SE) -> BiLSTM -> FC head。
    短分支 = 最近 short_days 天原始 bar; 长分支 = 全窗按 long_pool 池化。
    use_inter=True: CrossFieldMixer 的 K 条幂积序列作为第四组 "inter" 走双尺度 TCN;
    use_tscorr=True: TSCorrOperators 零参数窗口算子输出直接并入 fusion。
    bar_slots 仅为接口兼容保留, 本骨架不使用。"""

    def __init__(self, config, n_fields: int, groups: dict, bars_per_day: int,
                 bar_slots=None, fields=None):
        super().__init__()
        c = config
        self.n_fields = n_fields
        self.groups = groups
        self.lookback = int(c["lookback_days"]) * bars_per_day
        self.short_steps = int(c["short_days"]) * bars_per_day
        self.long_pool = int(c["long_pool"])
        if self.lookback % self.long_pool:
            raise ValueError("lookback_days*bars_per_day 必须能被 long_pool 整除")
        self.long_steps = self.lookback // self.long_pool
        if self.short_steps % self.long_steps:
            raise ValueError("short_steps 必须能被 long_steps 整除")
        self.short_pool = self.short_steps // self.long_steps

        gc = c["group_channels"]
        scale = float(c["short_channel_scale"])
        dropout = c["dropout"]

        # 时间算子组 (框架第3-6点): 每字段 concat n_ops 条差分通道后进 TCN
        self.use_temporal = bool(c.get("use_temporal_ops", False))
        self.tbank = (TemporalOperatorBank(c.get("temporal_diff_bars", [8, 48]))
                      if self.use_temporal else None)
        n_ops = self.tbank.n_ops if self.use_temporal else 0
        # 字段 i 的时间算子通道 = [i*n_ops : (i+1)*n_ops]
        self.temporal_exp = ({name: [i * n_ops + k for i in idx for k in range(n_ops)]
                              for name, idx in groups.items()}
                             if self.use_temporal else {})

        # inter 组 (阶段1): 幂积序列与原始字段组并列成第四组
        self.inter_k = (int(c.get("inter_K", 0))
                        if c.get("use_inter", False) else 0)
        if self.inter_k > 0:
            self.mixer = CrossFieldMixer(n_fields, self.inter_k)
        # 各组进 TCN 的输入通道数: 原始 + (可选)差分算子
        in_chs = {name: len(idx) * (1 + n_ops) for name, idx in groups.items()}
        if self.inter_k > 0:
            in_chs["inter"] = self.inter_k
        self.branch_names = list(in_chs)

        def make_branch(kernel, dilations, ch_scale):
            encoders = nn.ModuleDict()
            total = 0
            for name, cin in in_chs.items():
                ch = max(int(gc[name] * ch_scale), 8)
                encoders[name] = nn.Sequential(
                    TCNEncoder(cin, ch, kernel_size=kernel,
                               dilation_list=dilations, dropout=dropout,
                               progressive=c.get("progressive_tcn_channels", True)),
                    SEBlock1D(ch),
                )
                total += ch
            return encoders, total

        self.short_encoders, short_total = make_branch(
            c["short_tcn_kernel"], c["short_tcn_dilations"], scale)
        self.long_encoders, long_total = make_branch(
            c["long_tcn_kernel"], c["long_tcn_dilations"], 1.0)

        # ts_corr 分支 (二期): 零参数窗口算子, 输出直接并入 fusion
        self.use_tscorr = bool(c.get("use_tscorr", False))
        tscorr_total = 0
        if self.use_tscorr:
            if fields is None:
                raise ValueError("use_tscorr=True 需要向 build_model 传入 fields")
            corr_idx = [fields.index(f) for f in c["corr_fields"] if f in fields]
            if len(corr_idx) < 2:
                raise ValueError(f"corr_fields 在表中可用字段不足: {c['corr_fields']}")
            self.tscorr = TSCorrOperators(corr_idx, self.long_pool,
                                          c.get("corr_window", 48), self.long_steps)
            tscorr_total = self.tscorr.n_ops

        fusion_channels = c["fusion_channels"]
        fusion_in = short_total + long_total + tscorr_total
        if c.get("use_grn", False):              # 框架第10点: GRN 融合
            self.fusion = GRNFusion(fusion_in, fusion_channels, dropout)
        else:
            fusion = [
                nn.Conv1d(fusion_in, fusion_channels, kernel_size=1),
                nn.BatchNorm1d(fusion_channels), nn.ReLU(), nn.Dropout(dropout),
            ]
            if c["fusion_se"]:
                fusion.append(SEBlock1D(fusion_channels))
            self.fusion = nn.Sequential(*fusion)

        self.lstm = nn.LSTM(
            input_size=fusion_channels, hidden_size=c["lstm_hidden"],
            num_layers=c["lstm_layers"], batch_first=True, bidirectional=True,
            dropout=dropout if c["lstm_layers"] > 1 else 0,
        )
        fc_hidden = c["fc_hidden"]
        self.fc_head = nn.Sequential(
            nn.Linear(c["lstm_hidden"] * 2, fc_hidden),
            nn.LayerNorm(fc_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.LayerNorm(fc_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "lstm" in name and "bias" in name:
                nn.init.zeros_(param)
                hidden = param.shape[0] // 4
                param.data[hidden:2 * hidden].fill_(1.0)

    def forward(self, x):
        # x: (N, L, F)
        if x.dim() != 3 or x.size(1) != self.lookback or x.size(2) != self.n_fields:
            raise ValueError(
                f"Expected (N, {self.lookback}, {self.n_fields}), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1)                             # (N, F, L)
        feats = {name: x[:, idx, :] for name, idx in self.groups.items()}
        if self.use_temporal:                              # 差分算子 concat 到各组
            xt = self.tbank(x)                             # (N, F*n_ops, L)
            for name in self.groups:
                feats[name] = torch.cat(
                    [feats[name], xt[:, self.temporal_exp[name], :]], dim=1)
        if self.inter_k > 0:
            feats["inter"] = self.mixer(x)                 # 幂积交互序列

        parts = []
        for name in self.branch_names:
            s = self.short_encoders[name](feats[name][:, :, -self.short_steps:])
            parts.append(F.avg_pool1d(s, self.short_pool, self.short_pool))
        for name in self.branch_names:
            xl = F.avg_pool1d(feats[name], self.long_pool, self.long_pool)
            parts.append(self.long_encoders[name](xl))

        if self.use_tscorr:
            parts.append(self.tscorr(x))               # (N, n_ops, long_steps) 零参数算子

        z = self.fusion(torch.cat(parts, dim=1))
        z, _ = self.lstm(z.permute(0, 2, 1))
        return self.fc_head(z.mean(dim=1)).squeeze(-1)


def build_model(config: dict, fields: list[str], bars_per_day: int,
                bar_slots=None, device=None,
                frequency_meta: dict | None = None) -> nn.Module:
    groups = make_groups(fields)
    version = config.get("model_version", "v1")
    if version == "raw_hier_1m_res_v1":
        freq = str(config.get("residual_frequency", "1m"))
        if not frequency_meta or freq not in frequency_meta:
            raise ValueError(f"{version} 需要 frequency_meta['{freq}']")
        fm = frequency_meta[freq]
        model = FrozenE1OneMinuteResidualNet(
            config, fields, groups, bars_per_day,
            list(fm["fields"]), int(fm["bars_per_day"]))
    elif version == "raw_hier_mf_v1":
        freq = str(config.get("mf_coarse_frequency", "30m"))
        if not frequency_meta or freq not in frequency_meta:
            raise ValueError(f"{version} 需要 frequency_meta['{freq}']")
        cm = frequency_meta[freq]
        model = MultiFrequencyHierarchicalNet(
            config, fields, groups, bars_per_day,
            list(cm["fields"]), int(cm["bars_per_day"]))
    elif version == "raw_hier_v2":
        model = RawHierarchicalOperatorNet(config, fields, groups, bars_per_day)
    else:
        # 旧 checkpoint 没有 model_version，必须保持原 V1 构造路径可严格加载。
        model = DailyMultiScaleTCN(config, len(fields), groups, bars_per_day,
                                   bar_slots, fields)
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 100_000 <= n_params <= 100_000_000, \
        f"参数量 {n_params} 不满足赛题约束 [1e5, 1e8]"
    if version == "raw_hier_1m_res_v1":
        base_mode = ("crossday-gru"
                     if model.base.use_crossday
                     else f"{model.base.long_window_mode}"
                          f"({model.base.lookback_days}d)")
        if (not model.base.use_crossday
                and model.base.long_window_mode
                == "shared_day_gru_residual"):
            base_mode += f"-1x{model.base.day_gru_hidden}"
        if (not model.base.use_crossday
                and model.base.long_position_mode
                == "recent_day_plus_fixed_day_sincos"):
            base_mode += (
                f"+fixed-day-sincos(std={model.base.day_sincos_scale:g})")
        extra = (f" + frozen-5m-{base_mode}(return={model.base.return_bank.n_ops},"
                 f"corr={model.base.corr_bank.n_ops},std={model.base.std_bank.n_ops})"
                 f" + 1m-residual(return={model.residual.return_bank.n_ops},"
                 f"corr={model.residual.corr_bank.n_ops},"
                 f"std={model.residual.std_bank.n_ops}) + zero-head")
    elif version == "raw_hier_mf_v1":
        extra = (f" + E1-5m(return={model.base.return_bank.n_ops},"
                 f"corr={model.base.corr_bank.n_ops},std={model.base.std_bank.n_ops})"
                 f" + {config.get('mf_coarse_days', 10)}d-"
                 f"{config.get('mf_coarse_frequency', '30m')}-raw"
                 " + zero-gate")
    elif version == "raw_hier_v2":
        extra = (" + hierarchical"
                 if model.use_crossday else
                 f" + {model.long_window_mode}({model.lookback_days}d)")
        if (not model.use_crossday
                and model.long_window_mode == "shared_day_gru_residual"):
            extra += (
                f" + day-GRU(1x{model.day_gru_hidden})"
                " + zero-init-context-score")
        if (not model.use_crossday
                and model.long_position_mode
                == "recent_day_plus_fixed_day_sincos"):
            extra += f" + fixed-day-sincos(std={model.day_sincos_scale:g})"
        if model.use_return:
            extra += f" + return({model.return_bank.n_ops})"
        if model.use_corr:
            extra += f" + corr({model.corr_bank.n_ops})"
        if model.use_std:
            extra += f" + std({model.std_bank.n_ops})"
    else:
        extra = (f' + inter(K={model.inter_k})' if model.inter_k else '') \
            + (f' + tscorr({model.tscorr.n_ops}算子)' if model.use_tscorr else '')
    print(f"[{now_s()}] 模型参数量 {n_params:,} (可训练 {n_trainable:,}) | groups="
          f"{ {k: len(v) for k, v in groups.items()} }{extra} | "
          f"version={version} | lookback={model.lookback} steps")
    return model


# ============================================================
# 6. 权重存取: json 文本文件 (平台提交口只接受文本类文件)
# ============================================================

def _enc_array(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def _dec_array(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]),
                         dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def save_ckpt_json(path: str, model: nn.Module, config: dict, fields: list,
                   bar_slots: list, bars_per_day: int,
                   norm_mu: np.ndarray, norm_sd: np.ndarray,
                   frequency_meta: dict | None = None,
                   **meta) -> str:
    """模型权重 + 归一化参数 + 元数据 -> 单个 json 文本文件 (数组 base64 编码)"""
    obj = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in config.items()},
        "fields": list(fields),
        "bar_slots": [int(t) for t in bar_slots],
        "bars_per_day": int(bars_per_day),
        "norm_mu": _enc_array(np.asarray(norm_mu, dtype=np.float32)),
        "norm_sd": _enc_array(np.asarray(norm_sd, dtype=np.float32)),
        "state_dict": {k: _enc_array(v.detach().cpu().numpy())
                       for k, v in model.state_dict().items()},
        "meta": meta,
    }
    if frequency_meta:
        obj["frequencies"] = {}
        for freq, fm in frequency_meta.items():
            obj["frequencies"][freq] = {
                "fields": list(fm["fields"]),
                "bar_slots": [int(t) for t in fm["bar_slots"]],
                "bars_per_day": int(fm["bars_per_day"]),
                "norm_mu": _enc_array(np.asarray(fm["norm_mu"], np.float32)),
                "norm_sd": _enc_array(np.asarray(fm["norm_sd"], np.float32)),
            }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    print(f"[{now_s()}] 已保存 {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")
    return path


def load_ckpt_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    obj["state_dict"] = {k: torch.from_numpy(_dec_array(v))
                         for k, v in obj["state_dict"].items()}
    obj["norm_mu"] = _dec_array(obj["norm_mu"])
    obj["norm_sd"] = _dec_array(obj["norm_sd"])
    for fm in obj.get("frequencies", {}).values():
        fm["norm_mu"] = _dec_array(fm["norm_mu"])
        fm["norm_sd"] = _dec_array(fm["norm_sd"])
    return obj


# ============================================================
# 7. 损失 & 指标 (沿用 train0703)
# ============================================================

class GroupedListMLELoss(nn.Module):
    def __init__(self, top_frac=0.5, bottom_frac=0.5, min_group_size=30):
        super().__init__()
        self.top_frac, self.bottom_frac = top_frac, bottom_frac
        self.min_group_size = min_group_size

    def forward(self, scores, targets):
        n = scores.numel()
        if n < max(2, self.min_group_size):
            return scores.sum() * 0.0
        top_k = max(1, int(np.ceil(n * self.top_frac)))
        bottom_k = max(1, int(np.ceil(n * self.bottom_frac)))
        if top_k + bottom_k > n:
            top_k = max(1, n // 2)
            bottom_k = min(bottom_k, n - top_k)
        order = torch.argsort(targets, descending=True)
        sel = torch.cat([order[:top_k], order[-bottom_k:]])
        sort_idx = torch.argsort(targets[sel], descending=True)
        sorted_scores = scores[sel][sort_idx]
        log_denoms = torch.logcumsumexp(sorted_scores.flip(0), dim=0).flip(0)
        return (log_denoms - sorted_scores).mean()


class PearsonICLoss(nn.Module):
    def forward(self, scores, targets):
        if scores.numel() < 2:
            return scores.sum() * 0.0
        sc = scores - scores.mean()
        tc = targets - targets.mean()
        corr = (sc * tc).mean() / (torch.sqrt(sc.pow(2).mean() * tc.pow(2).mean()) + 1e-8)
        return -corr


class TailPairwiseLoss(nn.Module):
    """只强化头尾分离；中间样本由全截面 IC/Huber 持续约束。"""
    def __init__(self, frac=0.2, temperature=1.0, min_group_size=30):
        super().__init__()
        self.frac = float(frac)
        self.temperature = float(temperature)
        self.min_group_size = int(min_group_size)

    def forward(self, scores, targets):
        n = scores.numel()
        if n < max(2, self.min_group_size):
            return scores.sum() * 0.0
        k = max(1, min(n // 2, int(np.ceil(n * self.frac))))
        order = torch.argsort(targets)
        bottom = scores.index_select(0, order[:k])
        top = scores.index_select(0, order[-k:])
        margin = (top[:, None] - bottom[None, :]) / max(self.temperature, 1e-6)
        return F.softplus(-margin).mean()


class StablePearsonICLoss(nn.Module):
    """V2专用：把epsilon放进RMS，分数接近常数时不会产生1e8级梯度。"""
    def forward(self, scores, targets):
        if scores.numel() < 2:
            return scores.sum() * 0.0
        sc = scores - scores.mean()
        tc = targets - targets.mean()
        srms = torch.sqrt(sc.square().mean() + 1e-6)
        trms = torch.sqrt(tc.square().mean() + 1e-6)
        return -(sc * tc).mean() / (srms * trms)


class StableRankLossV2(nn.Module):
    """O(1)量级的全截面稳定损失 + 尾部交易目标。"""
    def __init__(self, config):
        super().__init__()
        self.ic = StablePearsonICLoss()
        self.tail = TailPairwiseLoss(config.get("tail_frac", 0.2),
                                     config.get("tail_temperature", 1.0),
                                     config.get("min_group_size", 30))
        self.ic_weight = float(config.get("ic_weight", 1.0))
        self.huber_weight = float(config.get("huber_weight", 0.25))
        self.tail_weight_target = float(config.get("tail_weight", 0.25))
        self.tail_weight = 0.0
        self.tail_warmup_epochs = int(config.get("tail_warmup_epochs", 3))
        self.huber_beta = float(config.get("huber_beta", 1.0))

    def set_epoch(self, epoch: int):
        if self.tail_warmup_epochs <= 0:
            self.tail_weight = self.tail_weight_target
        else:
            ratio = min(1.0, max(0.0, (epoch + 1) / self.tail_warmup_epochs))
            self.tail_weight = self.tail_weight_target * ratio

    def forward(self, scores, targets):
        # 截面标准化让 Huber/tail 的尺度稳定；IC 本身保持尺度不变。
        # scale不参与反向，避免标准差接近0时通过分母产生病态梯度；1e-3仅是安全下限。
        scale = scores.detach().std(unbiased=False).clamp_min(1e-3)
        sz = (scores - scores.mean()) / scale
        huber = F.smooth_l1_loss(sz, targets, beta=self.huber_beta)
        return (self.ic_weight * self.ic(scores, targets)
                + self.huber_weight * huber
                + self.tail_weight * self.tail(sz, targets))


class CombinedRankLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.v2 = (StableRankLossV2(config)
                   if config.get("loss_version") == "stable_rank_v2" else None)
        self.listmle = GroupedListMLELoss(config["top_frac"], config["bottom_frac"],
                                          config["min_group_size"])
        self.ic = PearsonICLoss()
        self.listmle_weight = config["listmle_weight"]
        self.rankic_weight = config["rankic_weight"]

    def set_epoch(self, epoch: int):
        if self.v2 is not None:
            self.v2.set_epoch(epoch)

    def forward(self, scores, targets):
        if self.v2 is not None:
            return self.v2(scores, targets)
        return (self.listmle_weight * self.listmle(scores, targets)
                + self.rankic_weight * self.ic(scores, targets))


def compute_rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    from scipy.stats import spearmanr
    valid = np.isfinite(pred) & np.isfinite(actual)
    if valid.sum() < 5:
        return np.nan
    return spearmanr(pred[valid], actual[valid])[0]


def predict_scores(model, x, chunk, base_only: bool = False):
    if isinstance(x, dict):
        n = next(iter(x.values())).size(0)
        slicer = lambda a, b: {k: v[a:b] for k, v in x.items()}
    else:
        n = x.size(0)
        slicer = lambda a, b: x[a:b]
    forward = (model.forward_base
               if base_only and hasattr(model, "forward_base") else model)
    outs = []
    for i in range(0, n, chunk):
        outs.append(forward(slicer(i, i + chunk)))
    return torch.cat(outs)
