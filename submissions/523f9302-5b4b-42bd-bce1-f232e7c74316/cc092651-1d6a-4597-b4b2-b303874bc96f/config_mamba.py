"""
Mamba 模型配置 — BigAlpha 2026 端到端大模型赛道

约束检查:
  - 输入字段 ≤ 100
  - 回看窗口 ≤ 240 交易日
  - 参数量 ∈ [100K, 100M]
  - 无外部预训练权重
  - 无特征工程 (仅标准化 + log1p成交量)
"""
import random
import numpy as np

# ============================================================
# 全局种子 & 字段 (与 local_train/config_local.py 对齐)
# ============================================================
SEED = 42

# 25 原始字段 (3档盘口+委托笔数, 云端&本地均含)
SELECTED_FIELDS = [
    "open", "high", "low", "close", "volume", "amount",
    "deal_number",
    "ask_price1", "ask_price2", "ask_price3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_price1", "bid_price2", "bid_price3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]
NUM_FIELDS = len(SELECTED_FIELDS)  # 25

FREQ_TABLES = {
    "1min":  "bigalpha_2026_stock_bar1m",
    "5min":  "bigalpha_2026_stock_bar5m",
    "15min": "bigalpha_2026_stock_bar15m",
    "30min": "bigalpha_2026_stock_bar30m",
}
MAX_BARS = {"1min": 240, "5min": 48, "15min": 16, "30min": 8}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Mamba 模型配置
# ============================================================
class MambaConfig:
    """Mamba 端到端股票预测配置"""
    seed = SEED

    # ---- 频率 ----
    freq_names = ["5min"]     # 5分钟线, 查询量只有1min的1/5
    n_fields = NUM_FIELDS  # 24

    # ---- 时序窗口 (分钟级别) ----
    bars_per_day = 48         # 每天48根5分钟K线
    lookback_days = 20        # 回看20个交易日
    seq_len = bars_per_day * lookback_days  # 20 × 48 = 960
    forward_horizon = 1       # 预测 T+1

    # ---- 数据区间 ----
    # 训练: 2019-2024 全量 (公榜要求)
    train_start = "2019-01-01"
    train_end   = "2024-12-31"

    # ---- 模型超参 ----
    d_model = 128           # 隐藏维度
    n_layers = 4            # Mamba 层数
    d_state = 16            # SSM 状态维度
    d_conv = 4              # 卷积核大小

    # Patch 参数: 每16分钟一个patch
    patch_len = 16          # patch 长度 (分钟)
    patch_stride = 8        # stride (50% overlap)

    dropout = 0.1

    # ---- 训练超参 ----
    batch_size = 128        # 每批股票数
    learning_rate = 1e-3
    weight_decay = 0.01
    warmup_epochs = 2
    max_epochs = 50
    early_stopping_patience = 15

    # ---- 训练策略 ----
    epoch_days = 20         # 每次加载20天数据 (降内存)
    val_days = 15            # 验证时随机采15天
    max_stocks_per_day = 300  # 每天最多300只股票 (降内存)
    reuse = 0               # 不复用 (Mamba训练慢, 每次新数据)

    # ---- 预处理 ----
    normalization = "standard"   # StandardScaler
    use_log_volume = True        # log1p成交量

    # ---- 硬件 ----
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # ---- 损失 ----
    loss_type = "ic"  # "ic" | "mse" | "combined"
    loss_alpha = 0.5  # combined loss weight

    # ---- 推理 ----
    inference_batch_size = 500


def set_active_freqs(freqs):
    """设置频率 (兼容 data_pipeline.py)"""
    global FREQ_NAMES, NUM_FREQS
    FREQ_NAMES = list(freqs)
    NUM_FREQS = len(freqs)
    total_fields = NUM_FIELDS * NUM_FREQS
    assert total_fields <= 100, f"字段数超限: {total_fields} > 100"
    return total_fields


# 初始化
FREQ_NAMES = ["1min"]
NUM_FREQS = 1
