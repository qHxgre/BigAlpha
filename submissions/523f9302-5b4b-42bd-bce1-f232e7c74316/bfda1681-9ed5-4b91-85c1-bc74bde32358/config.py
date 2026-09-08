# -*- coding: utf-8 -*-
"""统一配置模块 —— 通过 MODE 切换本地/在线环境。

MODE = "local"  → 本地 parquet 数据, 3 档盘口, 4 频率 (与在线架构一致)
MODE = "online" → BigQuant 平台 dai 数据, 3 档盘口, 4 频率

所有共享的模型超参、训练超参均在此定义, train.py 和 predict.ipynb 统一引用。
"""
import os

# ═══════════════════════════════════════════════════════════════════════════
# 模式切换: 改为 "online" 后在 BigQuant 平台使用
# ═══════════════════════════════════════════════════════════════════════════
MODE = "online"  # "local" | "online"

# ═══════════════════════════════════════════════════════════════════════════
# 路径 & 共享超参
# ═══════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")
STATS_PATH = os.path.join(_HERE, "transformer_stats.json")

# 模型检查点选择: ""(默认) | ".best" | ".ep01"~".ep10"
# 仅影响推理加载, 不影响训练保存逻辑; 用于灵活切换验证/推理时使用的检查点
MODEL_CHECKPOINT = ".ep05"

TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"
VAL_START,   VAL_END   = "2024-01-01", "2024-12-31 23:59:59"  # 与在线平台验证区间一致
SEQ_LEN = 96
EPOCHS, BATCH, LR, SEED = 10, 512, 3e-4, 42
WARMUP_EPOCHS = 6
WEIGHT_DECAY = 2e-4
GRAD_CLIP = 1.0

# 模型选择: "V1"=StockTransformer, "V2"=StockTransformerV2, "V3"=StockTransformerV3
MODEL_VERSION = "V3"

# 复权因子开关: True=后复权(乘adjust_factor), False=原始价格不复权
# 必须为 True: 不复权会导致除权除息日标签(收益率)被污染
USE_ADJUST_FACTOR = True

# ═══════════════════════════════════════════════════════════════════════════
# 特征列 & 频率表 (随模式切换)
# ═══════════════════════════════════════════════════════════════════════════
if MODE == "local":
    # ---- 本地数据: 3 档盘口, 无 pre_close, instrument_id (int) ----
    from data_loader import (
        LOCAL_PRICE_COLS as PRICE_COLS,
        LOCAL_VOL_COLS as VOL_COLS,
        LOCAL_ORDER_NUM_COLS as ORDER_NUM_COLS,
        LOCAL_FEATURE_COLS as FEATURE_COLS,
        N_LOCAL_FEAT as N_FEAT,
        instrument_id_to_str,
    )
    MAX_TRAIN_INSTRUMENTS = 300
    FREQ_TABLES = {
        "15m": "bigalpha_2026_e2e_bar15m",
        "30m": "bigalpha_2026_e2e_bar30m",
    }

    def pool(sd, ed):
        """本地模式: 从数据中提取 instrument_id 列表。"""
        from data_loader import load_local_data, get_instruments
        table = list(FREQ_TABLES.values())[0]
        df = load_local_data(table, sd, ed, columns=["date", "instrument_id"])
        return get_instruments(df)

    def resolve_instrument(inst):
        """将内部标识转为字符串 (本地为 instrument_id→str, 在线原样返回)。"""
        return instrument_id_to_str(inst)

    # 本地需要预处理 (价格 int32→float32 除 scale 乘 adjust_factor)
    NEEDS_PREPROCESS = True
    PRICE_SCALE = 100.0

else:
    # ---- 在线数据: 3 档盘口, instrument (string), 列顺序与本地一致 ----
    PRICE_COLS = [
        "open", "high", "low", "close",
        "ask_price1", "ask_price2", "ask_price3",
        "ask_price4", "ask_price5",
        "bid_price1", "bid_price2", "bid_price3",
        "bid_price4", "bid_price5"
    ]
    VOL_COLS = [
        "volume", "amount",
        "ask_volume1", "ask_volume2", "ask_volume3",
        "ask_volume4", "ask_volume5",
        "bid_volume1", "bid_volume2", "bid_volume3",
        "bid_volume4", "bid_volume5"
    ]
    ORDER_NUM_COLS = [
        "deal_number",
        "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
        "ask_num_orders4", "ask_num_orders5",
        "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
        "bid_num_orders4", "bid_num_orders5"
    ]
    FEATURE_COLS = PRICE_COLS + VOL_COLS + ORDER_NUM_COLS
    N_FEAT = len(FEATURE_COLS)  # 
    MAX_TRAIN_INSTRUMENTS = 1000
    FREQ_TABLES = {
        # "1m":  "bigalpha_2026_stock_bar1m",
        "5m":  "bigalpha_2026_stock_bar5m",
        # "15m": "bigalpha_2026_stock_bar15m",
        "30m": "bigalpha_2026_stock_bar30m",
    }

    def pool(sd, ed):
        """在线模式: 从 bigalpha_2026_instruments 查成分股。"""
        import dai
        df = dai.query(
            "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
            filters={"date": [sd, ed]},
        ).df()
        return df["instrument"].tolist()

    def resolve_instrument(inst):
        """在线模式: instrument 已是字符串, 原样返回。"""
        return inst

    NEEDS_PREPROCESS = False
    PRICE_SCALE = 1.0

# ═══════════════════════════════════════════════════════════════════════════
# 模型结构超参 (由以上配置推导, 训练与推理必须一致)
# ═══════════════════════════════════════════════════════════════════════════
N_FREQS = len(FREQ_TABLES)
N_FEAT_ALL = N_FEAT * N_FREQS
MODEL_CFG = dict(n_feat=N_FEAT_ALL, d_model=128, nhead=8, nlayers=8, dim_ff=256, seq_len=SEQ_LEN)

# V3 专用超参 (V3-Lite: CPU 优化版, 跳过 TimesBlock 2D 卷积以加速训练)
V3_CFG = dict(
    # TimesNet 周期发现 (V3.1: 禁用, 对短序列金融数据无效)
    top_k_periods=0,          # 0=跳过 TimesBlock
    # ModernTCN 大核卷积
    large_conv_kernel=13,     # 大核 depthwise conv 尺寸
    # Informer 稀疏注意力
    prob_sparse=False,        # 是否使用 ProbSparse 注意力 (False=RoPE, 更稳定)
    attn_distill=False,       # 是否使用注意力蒸馏 (金字塔)
    # 通用
    n_freqs=N_FREQS,
    n_feat=N_FEAT_ALL,
    d_model=128,
    nhead=8,
    nlayers=4,
    dim_ff=256,
    seq_len=SEQ_LEN,
    dropout=0.1,
)

# V3 训练专用超参 (不传给模型)
V3_TRAIN_CFG = dict(
    lambda_pearson=0.8,       # CombinedLoss Pearson 权重 (V3.1 最优)
)


def check_model_data_compat(model_cfg: dict, tables: dict) -> bool:
    """检查加载的模型与数据频率是否兼容。

    model_cfg: 模型保存时的 MODEL_CFG (含 n_feat)
    tables:    推理时使用的频率表 dict
    返回 True 表示兼容, False 表示不兼容。
    """
    expected_n_feat = N_FEAT * len(tables)
    actual_n_feat = model_cfg.get("n_feat", 0)
    if expected_n_feat != actual_n_feat:
        raise ValueError(
            f"模型与数据频率不兼容!\n"
            f"  模型期望特征数: {actual_n_feat} (对应 {actual_n_feat // N_FEAT} 个频率)\n"
            f"  数据提供特征数: {expected_n_feat} (对应 {len(tables)} 个频率: {list(tables.keys())})\n"
            f"  请确保训练和推理使用相同数量的频率。"
        )
    return True
