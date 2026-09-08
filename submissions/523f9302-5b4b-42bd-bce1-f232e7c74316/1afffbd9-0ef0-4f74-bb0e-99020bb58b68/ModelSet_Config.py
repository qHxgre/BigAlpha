from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class Config:
    """多尺度轴向注意力 + 截面集合 Transformer（EVO）"""

    # =========================
    # 数据接口（与 PrefetchDataLoader 对齐）
    # =========================
    seq_len: int = 240
    n_fields: int = 17
    n_book_channels: int = 4
    n_book_levels: int = 5
    scale_keys: List[str] = field(
        default_factory=lambda: ["m01", "m05", "m15", "m30"]
    )
    active_scales: List[str] = field(
        default_factory=lambda: ["m01", "m05", "m15", "m30"]
    )
    fusion_order: List[str] = field(
        default_factory=lambda: ["m30", "m15", "m05", "m01"]  # 粗→细（GRU）
    )
    scale_id_map: dict = field(
        default_factory=lambda: {"m01": 0, "m05": 1, "m15": 2, "m30": 3}
    )
    scale_bar_minutes: dict = field(
        default_factory=lambda: {"m01": 1, "m05": 5, "m15": 15, "m30": 30}
    )
    details_path: str = "transformer_model.json"
    mem_mode: str = "high"
    prefetch_size: Optional[int] = None
    use_rolling: bool = True

    # =========================
    # 公共隐层
    # =========================
    d_model: int = 128
    d_book: int = 64
    nhead: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.05
    droppath: float = 0.05
    ffn_ratio: int = 2

    # =========================
    # B0 NormGate
    # =========================
    revin_mode: str = "dual"        # "off" | "dyn_only" | "dual"
    revin_affine: bool = True
    revin_eps: float = 1e-5
    level_branch_wd: float = 5e-2   # level_proj 独立 weight decay

    # =========================
    # B1 向量流 FT-Axial
    # =========================
    patch_len: int = 12
    patch_stride: int = 12
    conv_channels: int = 32
    conv_layers: int = 3
    conv_kernel: int = 5
    conv_dilations: Tuple[int, ...] = (1, 2, 4)
    n_blocks: int = 3
    time_pos: str = "alibi"         # "alibi" | "learned"
    use_field_attn: bool = True
    use_bigru: bool = True
    share_across_fields: bool = True

    # =========================
    # B2 盘口流 Book-Axial
    # =========================
    book_stream: bool = True
    book_conv_layers: int = 2
    level_attn_blocks: int = 2
    level_pool: str = "attn"        # "attn" | "mean"
    level_rel_bias: bool = True

    # =========================
    # B3 尺度内 Vector⇄Book
    # =========================
    cross_layers: int = 2
    bidirectional_cross: bool = True
    share_backbone_across_scales: bool = True
    scale_cond: str = "film"        # "film" | "embedding_only" | "separate_weights"

    # =========================
    # B4 跨尺度融合
    # =========================
    fusion_mode: str = "c2f+joint+moe"  # 子集可拼: joint / c2f / moe / gru
    joint_blocks: int = 2
    c2f_layers: int = 2
    moe_temp: float = 1.0
    time_offset_encoding: bool = True
    use_scale_gru: bool = False

    # =========================
    # B5 截面块
    # =========================
    cs_block: str = "ISAB"          # "off" | "ISAB" | "full-attn"
    n_inducing: int = 32
    cs_layers: int = 2
    stock_dropout: float = 0.1
    market_residual: bool = True
    cs_norm: bool = True

    # =========================
    # B6 预测头
    # =========================
    use_decile_head: bool = False   # 主损失不再用十分位 CE，默认关
    use_quantile_head: bool = False
    n_quantiles: int = 9
    score_beta: float = 0.0         # 推理: score = pred_rank + β·pred_ret

    # =========================
    # B7 损失（加权成对 logistic；零损失超参）
    # =========================
    rank_direction: int = -1        # -1: 标签越大越好
    pair_label: str = "vdw"        # "rank" | "return" | "vdw"
    meta_batch_k: int = 1           # K 天梯度累积（优化器设置）
    min_valid_stocks: int = 50
    ret_clamp: float = 4.0          # return / 变体 A 的 winsorize
    exclude_head_wd: bool = True    # rank 头最后一层不加 weight decay

    # =========================
    # 训练协议
    # =========================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 4
    learning_rate: float = 2.5e-4
    weight_decay: float = 5e-3
    adam_betas: Tuple[float, float] = (0.9, 0.98)
    warmup_ratio: float = 0.10
    grad_clip: float = 1.0
    early_stopping: bool = True
    early_patience: int = 2
    seed: int = 42

    use_ema: bool = True
    ema_decay: float = 0.999
    use_bf16: bool = True

    input_noise_std: float = 0.01
    time_crop_max: int = 0          # >0：训练时随机左裁剪上限
