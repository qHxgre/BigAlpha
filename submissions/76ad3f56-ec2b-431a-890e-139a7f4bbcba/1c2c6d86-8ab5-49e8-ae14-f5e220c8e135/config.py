"""
DisFT-GNN 全局配置模块
======================
管理所有超参数与运行时配置，集中管理便于调参与合规审计。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """模型架构配置（适应Notebook内3小时训练，已精简）"""

    # --- 时序编码器 ---
    use_mamba: bool = False           # False=使用Bi-LSTM（更稳定，无需额外依赖）
    input_dim: int = 25               # M: 日频特征维度
    hidden_dim: int = 64              # Bi-LSTM隐藏维度
    num_layers: int = 1               # LSTM层数

    # --- 嵌入维度（精简版） ---
    d_p: int = 32                     # 历史时空嵌入维度 D_p
    d_f: int = 16                     # 未来趋势嵌入维度 D_f
    d_out: int = 4                    # 融合通道数 D
    fusion_rank: int = 8              # 低秩分解秩 r
    tau: float = 0.5                  # 注意力温度系数

    # --- GNN ---
    gnn_layers: int = 2               # GCN层数
    gnn_hidden: int = 32              # GCN隐藏维度

    # --- 时序窗口 ---
    lookback_short: int = 5           # 短窗口 L=5

    # --- 邻接矩阵融合权重（可学习，AI主导） ---
    adj_alpha_init: float = 0.4       # 行业权重初始值
    adj_beta_init: float = 0.4        # 收益相关性权重初始值
    adj_gamma_init: float = 0.2       # 市值相似性权重初始值

    # --- 投影层 ---
    projection_hidden: int = 16       # 投影MLP隐藏维度


@dataclass
class TrainConfig:
    """训练配置"""

    # --- 通用 ---
    seed: int = 42                    # 随机种子（确保审计可复现）
    device: str = "cpu"               # 设备（平台可能无GPU）
    learning_rate: float = 5e-4       # 教师学习率
    student_lr: float = 1e-4          # 学生学习率（更低防过拟合）
    batch_size: int = 16              # 每batch时间截面数

    # --- 教师训练 ---
    teacher_epochs: int = 30          # 教师训练轮数（精简）
    teacher_patience: int = 5         # 早停patience
    future_horizon: int = 1           # 未来标签天数 T=1
    future_threshold: float = 0.0     # 涨跌阈值 δ=0%

    # --- 蒸馏 ---
    use_distillation: bool = True     # 是否使用蒸馏（可蒸馏性验证后决定）
    lambda_distill_start: float = 0.1 # 蒸馏权重起始值
    lambda_distill_end: float = 0.5   # 蒸馏权重终值
    hsic_sigma_init: float = 1.0      # HSIC核带宽初始值
    hsic_eigenvalue_threshold: float = 1e-6  # 特征值稳定性阈值

    # --- 学生训练 ---
    student_epochs: int = 20          # 学生训练轮数
    student_patience: int = 5         # 早停patience
    alpha_rank: float = 1.0           # 排序损失权重 α
    beta_ce: float = 0.15             # CrossEntropy辅助权重 β

    # --- 可蒸馏性验证 ---
    distill_r2_threshold: float = 0.05  # R²阈值，低于此值放弃蒸馏
    distill_r2_cautious: float = 0.15   # R²谨慎阈值


@dataclass
class DataConfig:
    """数据配置"""

    # --- 时间划分（严格按时间轴，模拟公榜/私榜分布偏移） ---
    train_start: str = "2019-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: str = "2024-12-31"

    # --- 股票池 ---
    universe: str = "CSI1000"         # 中证1000历史时点成分股

    # --- 图构建 ---
    corr_window: int = 20             # 收益相关性滚动窗口（交易日）
    corr_threshold: float = 0.3       # 相关性连边阈值 |ρ| > 0.3
    size_threshold: float = 0.5       # 对数市值差阈值

    # --- 因子输出 ---
    max_missing_ratio: float = 0.4    # 最大缺失率（比赛要求<40%）
    winsorize_lower: float = 0.01     # 去极值下分位
    winsorize_upper: float = 0.99     # 去极值上分位
    barra_r2_threshold: float = 0.3   # BARRA残差R²阈值


@dataclass
class Config:
    """全局配置聚合"""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """初始化后验证"""
        assert self.model.d_p > self.model.fusion_rank, "D_p必须大于rank"
        assert self.model.d_f > self.model.fusion_rank, "D_f必须大于rank"
        assert 0 < self.train.alpha_rank, "alpha必须>0"
        assert 0 <= self.train.beta_ce <= 1, "beta必须在[0,1]"


# 全局配置实例
CONFIG = Config()
