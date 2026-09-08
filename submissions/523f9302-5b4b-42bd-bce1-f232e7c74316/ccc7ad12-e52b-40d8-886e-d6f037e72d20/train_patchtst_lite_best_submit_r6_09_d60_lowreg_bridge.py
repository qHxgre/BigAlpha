# ============================================================
# BigAlpha 2026 端到端量价时序预测 —— PatchTST-Lite 最优参数提交版本
# 数据源：bigalpha_2026_stock_bar30m
#
# 训练逻辑（配置 / 模型结构 / 数据构建 / train_and_save）
# 全部沉淀在本文件中，作为单一事实来源。
# 配套 .ipynb 在推理时 direct import 复用。
#
# 提交版约束：
# 1. 输入为当前探索文件中的 38 个原始字段；
# 2. 运行本 .py 只训练并输出 train_patchtst_lite_best_submit_r6_09_d60_lowreg_bridge.json；
# 3. 不额外保存训练日志、训练曲线、公榜 CSV、本地打分 CSV；
# 4. tqdm 被替换为静默迭代器，避免提交环境产生进度条刷屏；
# 5. main() 优先读取同目录下的 train_patchtst_lite_best_submit_r6_09_d60_lowreg_bridge.json。
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import time
import json
import base64
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Iterator, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

def tqdm(x, **kwargs):
    return x

try:
    import dai
except Exception as e:
    dai = None
    print("警告：dai 仅在 BigQuant AIStudio 平台内可用。", e)


# ============================================================
# 1. 全局配置
# ============================================================

@dataclass
class Config:
    # 已切换为 30 分钟附盘口原始行情表
    data_source: str = "bigalpha_2026_stock_bar30m"

    # 本训练文件方向
    variant_name: str = "r4_10_deep_stress_ic_bridge"
    output_dir: str = "runs_patchtst_lite_best_submit"

    # 公榜训练集
    train_start: str = "2019-01-01 00:00:00"
    train_end: str = "2023-12-31 23:59:59"

    # 公榜验证区间
    public_start: str = "2024-01-01 00:00:00"
    public_end: str = "2024-12-31 23:59:59"

    # 文件输出
    model_path: str = "train_patchtst_lite_best_submit_r6_09_d60_lowreg_bridge.json"
    submission_path: str = "runs_patchtst_lite_best_submit/submission_patchtst_lite.csv"

    train_log_path: str = "runs_patchtst_lite_best_submit/training_log_patchtst_lite.csv"
    train_curve_path: str = "runs_patchtst_lite_best_submit/training_curve_patchtst_lite.png"

    public_submission_path: str = "runs_patchtst_lite_best_submit/public_submission_patchtst_lite.csv"
    public_local_score_path: str = "runs_patchtst_lite_best_submit/public_local_score_summary_patchtst_lite.csv"
    public_daily_ic_path: str = "runs_patchtst_lite_best_submit/public_local_daily_ic_patchtst_lite.csv"
    public_long_short_path: str = "runs_patchtst_lite_best_submit/public_local_long_short_patchtst_lite.csv"
    public_local_leaderboard_score_path: str = "runs_patchtst_lite_best_submit/public_local_leaderboard_score_patchtst_lite.csv"

    seed: int = 42

    # 30m bar；轻量端到端版本优先控制容量与训练轮次，降低私榜过拟合风险。
    # 64 个交易日 × 每日约 8 根 30m bar = 512 steps
    seq_len: int = 512
    patch_size: int = 16
    patch_stride: int = 8
    architecture: str = "patchtst_finance_lite"

    # 训练资源控制
    max_train_samples: int = 2_000_000
    max_val_samples: int = 450000

    # 推理 batch 使用；训练 batch 由 DailyBatchSampler 控制。
    batch_size: int = 1024

    # 金融弱信号数据不做长轮次训练；依赖早停与完整日截面目标。
    epochs: int = 6
    min_epochs: int = 3
    early_stopping_patience: int = 2

    lr: float = 0.00030
    min_lr_ratio: float = 0.08
    weight_decay: float = 0.00020

    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    torch_num_threads: int = 6

    # Transformer 参数，满足 10 万至 1 亿参数限制。
    d_model: int = 80
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 160
    dropout: float = 0.10

    use_data_parallel: bool = False
    use_amp: bool = True
    allow_tf32: bool = False

    # 推理时往前取历史，构造 seq_len
    infer_history_days: int = 128

    # 本地评估最后一个公榜日需要下一期收益，多取一段 exposure
    eval_forward_extra_days: int = 40

    # 标签是否做风格中性化
    neutralize_label: bool = True

    # 排行榜导向训练目标权重：IC / Rank / SR / Stress 四项并行，减少只追平均 IC 的过拟合。
    loss_ic_weight: float = 0.54
    loss_point_weight: float = 0.035
    loss_longshort_weight: float = 0.03
    loss_rank_weight: float = 0.20
    loss_stress_weight: float = 0.14
    loss_disp_weight: float = 0.005
    min_pred_std: float = 0.020
    softmax_temperature: float = 2.60
    rank_pair_count: int = 4096
    negative_ic_margin: float = 0.0

    # ---- 轻量结构/正则化开关；默认关闭，具体变体在 Config 中打开 ----
    n_prototypes: int = 8
    prototype_residual_scale: float = 0.08
    conv_kernel_size: int = 5
    conv_residual_scale: float = 0.35
    group_residual_scale: float = 0.05

    # ---- PatchTST-Finance-Lite ----
    patch_feature_dim: int = 8
    mixer_rank: int = 16
    context_rank: int = 16
    context_residual_scale: float = 0.08
    feature_dropout: float = 0.02

    # 每个训练 batch 尽量使用完整交易日截面，贴近平台每日截面 IC / SR 评估。
    day_batch_size: int = 4096
    val_day_batch_size: int = 4096

    # checkpoint_top_k > 1 时保存 top-k epoch 权重平均，主要用于降低私榜方差。
    checkpoint_top_k: int = 1

    # 本地代理排行榜分数的校准参数。
    proxy_ic_scale: float = 0.020
    proxy_icir_scale: float = 0.350
    proxy_sr_scale: float = 2.000
    proxy_stress_scale: float = 0.015


# ============================================================
# 最优超参数提交入口
# ============================================================
#
# 只需要修改下面这个字典。
#
# 最方便的做法：
# 1. 从 LightExplore.csv 的“超参数JSON”列复制最优组合；
# 2. 粘贴替换 SUBMIT_HYPERPARAMS；
# 3. 可以保留 "name"，程序会自动忽略它，不会传给 Config；
# 4. 运行本 .py，生成 train_patchtst_lite_best_submit_r6_09_d60_lowreg_bridge.json；
# 5. 提交 .py、.ipynb 和训练好的 JSON 权重。
#
# data_source、训练区间、输出文件名和 architecture 被固定为比赛提交设置，
# 不允许在这个字典里覆盖，避免代码与权重不一致。
SUBMIT_HYPERPARAMS = {
    "name": "r6_09_d60_lowreg_bridge",
    "seq_len": 512,
    "patch_size": 16,
    "patch_stride": 8,
    "d_model": 60,
    "n_heads": 4,
    "n_layers": 4,
    "dim_feedforward": 88,
    "patch_feature_dim": 8,
    "mixer_rank": 8,
    "context_rank": 8,
    "context_residual_scale": 0.1,
    "dropout": 0.12,
    "feature_dropout": 0.025,
    "lr": 0.00021,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.00035,
    "epochs": 10,
    "min_epochs": 4,
    "early_stopping_patience": 3,
    "checkpoint_top_k": 1,
    "loss_ic_weight": 0.52,
    "loss_rank_weight": 0.19,
    "loss_stress_weight": 0.18,
    "loss_longshort_weight": 0.04,
    "loss_point_weight": 0.03,
    "loss_disp_weight": 0.01,
    "softmax_temperature": 2.6,
    "negative_ic_margin": 0.0,
    "seed": 42,
}


SUBMISSION_MODEL_FILENAME = "train_patchtst_lite_best_submit_r6_09_d60_lowreg_bridge.json"
SELECTED_EXPERIMENT_NAME = str(
    SUBMIT_HYPERPARAMS.get("name", "manual_best")
)

# 这些字段由提交代码固定，禁止在 SUBMIT_HYPERPARAMS 中覆盖。
_PROTECTED_SUBMIT_FIELDS = {
    "data_source",
    "variant_name",
    "output_dir",
    "model_path",
    "submission_path",
    "train_log_path",
    "train_curve_path",
    "public_submission_path",
    "public_local_score_path",
    "public_daily_ic_path",
    "public_long_short_path",
    "public_local_leaderboard_score_path",
    "train_start",
    "train_end",
    "public_start",
    "public_end",
    "architecture",
}


def make_submit_config(overrides: Optional[Dict] = None) -> Config:
    """
    根据 SUBMIT_HYPERPARAMS 构造唯一提交配置。

    支持直接粘贴探索文件中的单个参数字典：
    - "name" 会作为实验备注保留，但不会传给 Config；
    - 未写出的参数继续使用 Config 默认值；
    - 未知字段和受保护字段会立即报错，避免静默使用错误配置。
    """
    params = dict(SUBMIT_HYPERPARAMS if overrides is None else overrides)
    params.pop("name", None)

    default_values = asdict(Config())
    valid_fields = set(default_values)

    protected = sorted(set(params) & _PROTECTED_SUBMIT_FIELDS)
    if protected:
        raise KeyError(
            "以下提交固定字段不能在 SUBMIT_HYPERPARAMS 中覆盖："
            + ", ".join(protected)
        )

    unknown = sorted(set(params) - valid_fields)
    if unknown:
        raise KeyError(
            "SUBMIT_HYPERPARAMS 含有 Config 未定义字段："
            + ", ".join(unknown)
        )

    default_values.update(params)

    # 固定提交身份、数据源、结构类型和输出路径。
    default_values.update({
        "data_source": "bigalpha_2026_stock_bar30m",
        "variant_name": "patchtst_lite_best_submit",
        "output_dir": "runs_patchtst_lite_best_submit",
        "architecture": "patchtst_finance_lite",
        "model_path": SUBMISSION_MODEL_FILENAME,
        "submission_path": (
            "runs_patchtst_lite_best_submit/"
            "submission_patchtst_lite_best_submit.csv"
        ),
        "train_log_path": (
            "runs_patchtst_lite_best_submit/"
            "training_log_patchtst_lite_best_submit.csv"
        ),
        "train_curve_path": (
            "runs_patchtst_lite_best_submit/"
            "training_curve_patchtst_lite_best_submit.png"
        ),
        "public_submission_path": (
            "runs_patchtst_lite_best_submit/"
            "public_submission_patchtst_lite_best_submit.csv"
        ),
        "public_local_score_path": (
            "runs_patchtst_lite_best_submit/"
            "public_local_score_summary_patchtst_lite_best_submit.csv"
        ),
        "public_daily_ic_path": (
            "runs_patchtst_lite_best_submit/"
            "public_local_daily_ic_patchtst_lite_best_submit.csv"
        ),
        "public_long_short_path": (
            "runs_patchtst_lite_best_submit/"
            "public_local_long_short_patchtst_lite_best_submit.csv"
        ),
        "public_local_leaderboard_score_path": (
            "runs_patchtst_lite_best_submit/"
            "public_local_leaderboard_score_patchtst_lite_best_submit.csv"
        ),
    })

    cfg = Config(**default_values)

    # 轻量配置校验；参数总量在 build_model() 中继续检查。
    if int(cfg.seq_len) < int(cfg.patch_size):
        raise ValueError("seq_len 必须不小于 patch_size。")
    if int(cfg.patch_size) <= 0 or int(cfg.patch_stride) <= 0:
        raise ValueError("patch_size 和 patch_stride 必须为正整数。")
    if int(cfg.d_model) % int(cfg.n_heads) != 0:
        raise ValueError("d_model 必须能被 n_heads 整除。")
    if int(cfg.n_layers) <= 0:
        raise ValueError("n_layers 必须为正整数。")
    if not (0.0 <= float(cfg.dropout) < 1.0):
        raise ValueError("dropout 必须位于 [0, 1)。")
    if not (0.0 <= float(cfg.feature_dropout) < 1.0):
        raise ValueError("feature_dropout 必须位于 [0, 1)。")
    if int(cfg.epochs) < int(cfg.min_epochs):
        raise ValueError("epochs 必须不小于 min_epochs。")

    return cfg


CFG = make_submit_config()


# ============================================================
# 2. 原始输入字段
#    只用 stock_bar30m 直接提供的原始字段
#    38 个输入字段，低于比赛 100 原始字段上限
# ============================================================

FEATURE_COLS = [
    # OHLC + 前收
    "open", "high", "low", "close", "pre_close",

    # 五档卖价
    "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5",

    # 五档买价
    "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",

    # 五档卖量
    "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5",

    # 五档买量
    "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",

    # 五档卖委托笔数
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3", "ask_num_orders4", "ask_num_orders5",

    # 五档买委托笔数
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3", "bid_num_orders4", "bid_num_orders5",

    # 成交字段
    "amount", "volume", "deal_number",
]

PRICE_COLS = {
    "open", "high", "low", "close", "pre_close",
    "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5",
    "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",
}

RISK_COLS = [
    "BETA", "SIZE", "MOMENTUM", "RESVOL", "LIQUIDTY",
    "BTOP", "EARNYILD", "GROWTH", "LEVERAGE", "SIZENL"
]


# ============================================================
# 3. 基础工具函数
# ============================================================

def set_seed(seed: int = 20260704):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_output_dir(cfg: Config):
    # 提交版不创建输出目录；仅保存同目录模型 JSON。
    return None


def configure_runtime(cfg: Config):
    ensure_output_dir(cfg)
    try:
        torch.set_num_threads(int(cfg.torch_num_threads))
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def normalize_dt(x) -> pd.Timestamp:
    return pd.to_datetime(x).normalize()


def minus_days(x, days: int) -> str:
    return (pd.to_datetime(x) - pd.Timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")


def _ensure_dai():
    if dai is None:
        raise RuntimeError("dai 不可用。请在 BigQuant AIStudio 平台内运行此代码。")


def _query_sql(sql: str, start: str, end: str) -> pd.DataFrame:
    _ensure_dai()
    return dai.query(
        sql,
        filters={"date": [start, end]},
        compression=True,
    ).df()


def _datasource_to_dict(datasources) -> Dict:
    """
    将平台/本地数据源输入标准化。

    BigQuant 比赛模板将 datasources 字典传入 main()，而本地实验通常传入纯表名字符串。此辅助函数保持两种调用方式兼容，无需更改任何训练配置或输出路径。
    """
    if datasources is None:
        return {}
    if isinstance(datasources, dict):
        return datasources
    return {}


def _resolve_infer_data_source(datasources=None, cfg: Config = CFG) -> str:
    """
    解析用于预测的表名。

    优先级：
    1. 本地字符串表名；
    2. datasources 字典中平台注入的 30m 表；
    3. cfg.data_source 回退（用于本地公榜模拟）。
    """
    if isinstance(datasources, str) and datasources.strip():
        return datasources.strip()

    ds = _datasource_to_dict(datasources)
    preferred_keys = [
        "bar30m",
        "stock_bar30m",
        "bigalpha_2026_stock_bar30m",
        cfg.data_source,
    ]

    for k in preferred_keys:
        v = ds.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for k, v in ds.items():
        if isinstance(v, str) and v.strip():
            lk = str(k).lower()
            lv = v.lower()
            if "30m" in lk or "30m" in lv:
                return v.strip()

    for _, v in ds.items():
        if isinstance(v, str) and v.strip():
            return v.strip()

    return cfg.data_source


def _resolve_train_data_source(datasources=None, cfg: Config = CFG) -> str:
    """
    解析用于训练的表名。

    重要：当 main() 接收到平台 datasources 字典时，注入的表用于评测窗口，因此训练仍使用 cfg.data_source，遵循官方模板中固定训练数据与注入推理数据的分离。本地字符串表名仍可用于显式实验。
    """
    if isinstance(datasources, str) and datasources.strip():
        return datasources.strip()
    return cfg.data_source


def _sample_df(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_n:
        return df.reset_index(drop=True)
    return (
        df.sample(n=max_n, random_state=seed)
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_wrap_dataparallel(model: nn.Module, cfg: Config) -> nn.Module:
    """
    单卡调整：
    自动使用 DataParallel。
    保存 checkpoint 时会自动去掉 module. 前缀，推理时兼容单卡/双卡。
    """
    if (
        cfg.use_data_parallel
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    ):
        print(f"[GPU] using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model


def unwrap_state_dict(model: nn.Module) -> Dict:
    if isinstance(model, nn.DataParallel):
        return {
            k: v.detach().cpu().clone()
            for k, v in model.module.state_dict().items()
        }
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    对多个 epoch 的 state_dict 做简单算术平均。
    仅用于 checkpoint_top_k > 1 的轻量权重平均，推理入口不变。
    """
    if len(state_dicts) == 1:
        return {k: v.detach().cpu().clone() for k, v in state_dicts[0].items()}

    out = {}
    keys = state_dicts[0].keys()
    for k in keys:
        vals = [sd[k].detach().cpu().float() for sd in state_dicts]
        out[k] = torch.stack(vals, dim=0).mean(dim=0)
    return out


def make_loader_kwargs(cfg: Config) -> Dict:
    kwargs = {
        "num_workers": int(cfg.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }

    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    # 固定每个 DataLoader worker 的随机种子，保证可复现
    kwargs["worker_init_fn"] = lambda worker_id: np.random.seed(cfg.seed + worker_id)

    return kwargs


def save_training_curve(log_path: str, fig_path: str):
    # 提交版不输出 PNG 曲线。
    return None


def sigmoid_np(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def compute_local_proxy_score_from_metrics(
    ic_mean: float,
    ic_ir: float,
    long_short_sharpe: float,
    stress_min_ic_mean: float,
    cfg: Config = CFG
) -> Dict:
    """
    本地代理排行榜分数：
    官方最终是四项全场 rank 百分位等权，本地无法知道全场 rank。
    这里用单调 sigmoid 映射得到 0~1 的代理分，用于本地模型选择和调参。
    """
    ic_mean = 0.0 if not np.isfinite(ic_mean) else float(ic_mean)
    ic_ir = 0.0 if not np.isfinite(ic_ir) else float(ic_ir)
    long_short_sharpe = 0.0 if not np.isfinite(long_short_sharpe) else float(long_short_sharpe)
    stress_min_ic_mean = 0.0 if not np.isfinite(stress_min_ic_mean) else float(stress_min_ic_mean)

    proxy_rank_ic_mean = float(sigmoid_np(ic_mean / cfg.proxy_ic_scale))
    proxy_rank_ic_ir = float(sigmoid_np(ic_ir / cfg.proxy_icir_scale))
    proxy_rank_sr = float(sigmoid_np(long_short_sharpe / cfg.proxy_sr_scale))
    proxy_rank_stress = float(sigmoid_np(stress_min_ic_mean / cfg.proxy_stress_scale))

    local_proxy_score = float(
        0.25 * proxy_rank_ic_mean
        + 0.25 * proxy_rank_ic_ir
        + 0.25 * proxy_rank_sr
        + 0.25 * proxy_rank_stress
    )

    return {
        "local_proxy_score": local_proxy_score,
        "proxy_rank_ic_mean": proxy_rank_ic_mean,
        "proxy_rank_ic_ir": proxy_rank_ic_ir,
        "proxy_rank_sr": proxy_rank_sr,
        "proxy_rank_stress": proxy_rank_stress,
    }


def print_resource_plan(cfg: Config):
    print("=" * 80)
    print("[Resource Plan - Single GPU]")
    print(f"data_source           : {cfg.data_source}")
    print(f"seq_len               : {cfg.seq_len}")
    print(f"architecture          : {getattr(cfg, 'architecture', 'patch_transformer')}")
    print(f"patch_size            : {getattr(cfg, 'patch_size', 4)}")
    print(f"patch_stride          : {getattr(cfg, 'patch_stride', getattr(cfg, 'patch_size', 4))}")
    print(f"patch_feature_dim     : {getattr(cfg, 'patch_feature_dim', 8)}")
    print(f"mixer_rank            : {getattr(cfg, 'mixer_rank', 16)}")
    print(f"context_rank          : {getattr(cfg, 'context_rank', 16)}")
    print(f"batch_size            : {cfg.batch_size}")
    print(f"day_batch_size        : {cfg.day_batch_size}")
    print(f"epochs                : {cfg.epochs}")
    print(f"min_epochs            : {cfg.min_epochs}")
    print(f"early_stop_patience   : {cfg.early_stopping_patience}")
    print(f"max_train_samples     : {cfg.max_train_samples:,}")
    print(f"max_val_samples       : {cfg.max_val_samples:,}")
    print(f"num_workers           : {cfg.num_workers}")
    print(f"prefetch_factor       : {cfg.prefetch_factor}")
    print(f"persistent_workers    : {cfg.persistent_workers}")
    print(f"torch_num_threads     : {cfg.torch_num_threads}")
    print(f"use_amp               : {cfg.use_amp}")
    print(f"use_data_parallel     : {cfg.use_data_parallel}")
    print(f"allow_tf32            : {cfg.allow_tf32}")
    print(f"cuda_available        : {torch.cuda.is_available()}")
    print(f"gpu_count             : {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print("=" * 80)


# ============================================================
# 4. 读取 stock_bar30m 与 exposure
# ============================================================

def load_bar_data(
    data_source_name: str,
    start: str,
    end: str
) -> pd.DataFrame:
    """
    读取 bigalpha_2026_stock_bar30m。
    该表直接包含 instrument，因此不再需要 e2e 的 instrument_id 映射逻辑。
    """
    cols = ["date", "instrument", "instrument_id"] + FEATURE_COLS

    sql = f"""
        SELECT {", ".join(cols)}
        FROM {data_source_name}
        ORDER BY instrument, date
    """

    df = _query_sql(sql, start, end)

    if df is None or df.empty:
        raise RuntimeError(f"未能从 {data_source_name} 加载数据，范围={start} -> {end}")

    df["datetime"] = pd.to_datetime(df["date"])
    df["date"] = df["datetime"].dt.normalize()
    df["instrument"] = df["instrument"].astype(str)

    # G1 32G 内存调整：读取后立即压缩数值类型，降低 DataFrame 峰值内存
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    if "instrument_id" in df.columns:
        try:
            df["instrument_id"] = pd.to_numeric(df["instrument_id"], errors="coerce").astype(np.int32)
        except Exception:
            pass

    df = df.dropna(subset=["date", "instrument"])
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    print(
        f"[Data] loaded bars: rows={len(df):,}, "
        f"dates={df['date'].nunique():,}, "
        f"instruments={df['instrument'].nunique():,}"
    )
    return df


def load_exposure(start: str, end: str) -> pd.DataFrame:
    """
    读取 exposure。
    ret 仅用于训练标签和本地评估，不作为模型输入。
    """
    cols = ["date", "instrument", "ret"] + RISK_COLS

    sql = f"""
        SELECT {", ".join(cols)}
        FROM bigalpha_2026_exposure
        ORDER BY instrument, date
    """

    exp = _query_sql(sql, start, end)

    exp["date"] = pd.to_datetime(exp["date"]).dt.normalize()
    exp["instrument"] = exp["instrument"].astype(str)
    exp["ret"] = pd.to_numeric(exp["ret"], errors="coerce").astype(np.float32)

    for c in RISK_COLS:
        exp[c] = pd.to_numeric(exp[c], errors="coerce").astype(np.float32)

    print(
        f"[Data] loaded exposure: rows={len(exp):,}, "
        f"dates={exp['date'].nunique():,}, "
        f"instruments={exp['instrument'].nunique():,}"
    )
    return exp


# ============================================================
# 5. 标签构造
# ============================================================

def _neutralize_one_date(g: pd.DataFrame) -> pd.DataFrame:
    """
    单日截面标签处理：
    - ret 去极值
    - 对 BARRA 风险因子做截面回归取残差
    - 截面 z-score
    """
    y = g["fwd_ret"].astype(float).to_numpy()
    valid = np.isfinite(y)

    out = g[["date", "instrument"]].copy()

    if valid.sum() < 50:
        target = y.copy()
    else:
        lo, hi = np.nanpercentile(y[valid], [1, 99])
        y_clip = np.clip(y, lo, hi)

        X = g[RISK_COLS].astype(float).to_numpy()

        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(med, inds[1])

        mu = X[valid].mean(axis=0)
        sd = X[valid].std(axis=0) + 1e-6
        Xz = (X - mu) / sd
        Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)

        Xreg = np.concatenate([np.ones((len(Xz), 1)), Xz], axis=1)

        try:
            beta = np.linalg.lstsq(Xreg[valid], y_clip[valid], rcond=None)[0]
            target = y_clip - Xreg @ beta
        except Exception:
            target = y_clip

    valid2 = np.isfinite(target)

    if valid2.sum() >= 10:
        m = np.nanmean(target[valid2])
        s = np.nanstd(target[valid2]) + 1e-6
        target = (target - m) / s
    else:
        target = np.zeros_like(target)

    target = np.clip(target, -5, 5)

    out["target"] = target.astype(np.float32)
    return out


def make_forward_labels(
    start: str,
    end: str,
    neutralize: bool = True
) -> pd.DataFrame:
    """
    用 exposure.ret 构造下一期收益标签：
    对每只股票，把 date=t+1 的 ret shift 到 date=t。
    """
    exp = load_exposure(start, end)

    ret_next = exp[["date", "instrument", "ret"]].copy()
    ret_next = ret_next.sort_values(["instrument", "date"])
    ret_next["date"] = ret_next.groupby("instrument")["date"].shift(1)
    ret_next = ret_next.dropna(subset=["date"]).rename(columns={"ret": "fwd_ret"})

    if neutralize:
        risk_today = exp[["date", "instrument"] + RISK_COLS].copy()
        lab = ret_next.merge(risk_today, on=["date", "instrument"], how="left")

        parts = []
        for _, g in tqdm(
            lab.groupby("date", sort=True),
            desc="标签中性化",
            dynamic_ncols=True,
            leave=True,
        ):
            parts.append(_neutralize_one_date(g))

        lab = pd.concat(parts, ignore_index=True)

    else:
        lab = ret_next[["date", "instrument", "fwd_ret"]].rename(columns={"fwd_ret": "target"})
        lab["target"] = lab.groupby("date")["target"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-6)
        )
        lab["target"] = lab["target"].clip(-5, 5).astype(np.float32)

    del exp, ret_next
    gc.collect()

    lab = lab.dropna(subset=["date", "instrument", "target"])
    lab["date"] = pd.to_datetime(lab["date"]).dt.normalize()
    lab["instrument"] = lab["instrument"].astype(str)
    lab["target"] = lab["target"].astype(np.float32)

    print(f"[Label] labels: rows={len(lab):,}, dates={lab['date'].nunique():,}")
    return lab[["date", "instrument", "target"]]


# ============================================================
# 6. 原始字段预处理
# ============================================================

def transform_raw_features(
    df: pd.DataFrame,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    只做允许的按字段预处理：
    - 价格类字段：log1p
    - 量/额/笔数字段：signed log1p
    - 训练集均值方差标准化
    不做跨字段特征、不做滚动统计、不做人工因子。
    """
    arrs = []

    for c in FEATURE_COLS:
        v = pd.to_numeric(df[c], errors="coerce").astype("float32").to_numpy(copy=False)

        if c in PRICE_COLS:
            v = np.where(v > 0, np.log1p(v), np.nan)
        else:
            v = np.sign(v) * np.log1p(np.abs(v))

        arrs.append(v.astype(np.float32, copy=False))

    raw = np.stack(arrs, axis=1).astype(np.float32)

    if mean is None or std is None:
        mean = np.nanmean(raw, axis=0).astype(np.float32)
        std = np.nanstd(raw, axis=0).astype(np.float32)

        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = np.where((np.isfinite(std)) & (std > 1e-6), std, 1.0).astype(np.float32)

    x = (raw - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return x, mean.astype(np.float32), std.astype(np.float32)


def build_bar_cache(
    df: pd.DataFrame,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
):
    """
    生成：
    - features：所有 bar 的特征矩阵
    - meta：每个 date/instrument 对应当天最后一根 30m bar 的位置
    - mean/std：训练集 scaler
    """
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    features, mean, std = transform_raw_features(df, mean, std)

    df["_pos"] = np.arange(len(df), dtype=np.int64)

    meta = (
        df.groupby(["date", "instrument"], observed=True)["_pos"]
        .max()
        .reset_index()
        .rename(columns={"_pos": "end_pos"})
    )

    first_pos = (
        df.groupby("instrument", observed=True)["_pos"]
        .min()
        .to_dict()
    )

    meta["first_pos"] = meta["instrument"].map(first_pos).astype(np.int64)
    meta["end_pos"] = meta["end_pos"].astype(np.int64)
    meta["instrument"] = meta["instrument"].astype(str)

    print(
        f"[Cache] features shape={features.shape}, "
        f"feature_mem={features.nbytes / 1024**3:.2f} GB, "
        f"daily samples={len(meta):,}"
    )

    return features, meta, mean, std


# ============================================================
# 7. Dataset
# ============================================================

class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        end_pos: np.ndarray,
        first_pos: np.ndarray,
        y: Optional[np.ndarray],
        seq_len: int,
        date_ids: Optional[np.ndarray] = None
    ):
        self.features = features
        self.end_pos = end_pos.astype(np.int64)
        self.first_pos = first_pos.astype(np.int64)
        self.y = None if y is None else y.astype(np.float32)
        self.seq_len = int(seq_len)
        self.n_features = features.shape[1]
        self.date_ids = None if date_ids is None else date_ids.astype(np.int32)

    def __len__(self):
        return len(self.end_pos)

    def __getitem__(self, idx):
        end = int(self.end_pos[idx])
        first = int(self.first_pos[idx])
        start = max(first, end - self.seq_len + 1)

        x = self.features[start:end + 1]

        if len(x) < self.seq_len:
            pad = np.zeros((self.seq_len - len(x), self.n_features), dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)

        x = torch.from_numpy(x.astype(np.float32, copy=False))

        if self.y is None:
            return x, idx

        y = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.date_ids is None:
            return x, y

        date_id = torch.tensor(self.date_ids[idx], dtype=torch.int32)
        return x, y, date_id


class DailyBatchSampler(Sampler[List[int]]):
    """
    排行榜导向训练的关键：
    每个 batch 尽量是同一交易日完整股票截面，直接优化截面 IC / Rank / SR / Stress。
    """
    def __init__(
        self,
        date_ids: np.ndarray,
        max_batch_size: int,
        shuffle: bool = True,
        seed: int = 20260704
    ):
        self.date_ids = np.asarray(date_ids)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        self.groups = {}
        for i, d in enumerate(self.date_ids):
            self.groups.setdefault(int(d), []).append(i)

        self.date_keys = sorted(self.groups.keys())

        self._len = 0
        for d in self.date_keys:
            n = len(self.groups[d])
            self._len += int(np.ceil(n / self.max_batch_size))

    def set_epoch(self, epoch: int):
        # 每轮改变日顺序与日内顺序，避免固定 batch 顺序带来的隐性过拟合。
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 1009)

        batches = []
        date_keys = list(self.date_keys)

        if self.shuffle:
            rng.shuffle(date_keys)

        for d in date_keys:
            idxs = np.array(self.groups[d], dtype=np.int64)

            if self.shuffle:
                rng.shuffle(idxs)

            # day_batch_size 默认 4096，通常一个交易日只形成一个 batch；
            # 只有当日股票数异常超过 max_batch_size 时才拆分。
            for s in range(0, len(idxs), self.max_batch_size):
                batch = idxs[s:s + self.max_batch_size].tolist()
                if len(batch) > 0:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self) -> int:
        return self._len


# ============================================================
# 8. 模型
# ============================================================

class GatedResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.fc(x)
        g = self.gate(x)
        return x + g * z


class PatchTransformerLite(nn.Module):
    """
    轻量 Patch Transformer：
    将 30m bar 按 patch 聚合为更少 token，降低高频噪声与 attention 复杂度。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.20,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_size = max(1, int(patch_size))
        self.n_patches = int(np.ceil(self.seq_len / self.patch_size))
        self.pad_len = self.n_patches * self.patch_size - self.seq_len

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * self.patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gate = GatedResidualBlock(d_model, dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_patch_tokens(self, x):
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, self.pad_len, 0), mode="constant", value=0.0)
        b, t, c = x.shape
        x = x.reshape(b, self.n_patches, self.patch_size * c)
        return self.input_proj(x)

    def forward(self, x):
        h = self._make_patch_tokens(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)
        h0 = self.gate(h[:, 0])
        return self.head(h0).squeeze(-1)


class SiameseLOBTransformer(nn.Module):
    """
    盘口对称轻量模型：
    bid / ask 使用结构对称投影，再与 OHLCV/成交字段融合。
    不构造跨字段人工因子，只在网络内部学习盘口结构。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 160,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 640,
        dropout: float = 0.22,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_size = max(1, int(patch_size))
        self.n_patches = int(np.ceil(self.seq_len / self.patch_size))
        self.pad_len = self.n_patches * self.patch_size - self.seq_len

        # FEATURE_COLS 固定顺序：价格/盘口价/盘口量/订单数/成交字段。
        self.ask_idx = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 26, 27, 28, 29, 30]
        self.bid_idx = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35]
        self.misc_idx = [0, 1, 2, 3, 4, 5, 36, 37, 38]

        group_dim = self.patch_size * len(self.ask_idx)
        misc_dim = self.patch_size * len(self.misc_idx)

        self.ask_proj = nn.Sequential(nn.Linear(group_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.bid_proj = nn.Sequential(nn.Linear(group_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.misc_proj = nn.Sequential(nn.Linear(misc_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gate = GatedResidualBlock(d_model, dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _patch_group(self, x, idx):
        z = x[:, :, idx]
        if self.pad_len > 0:
            z = F.pad(z, (0, 0, self.pad_len, 0), mode="constant", value=0.0)
        b, t, c = z.shape
        return z.reshape(b, self.n_patches, self.patch_size * c)

    def forward(self, x):
        ask = self.ask_proj(self._patch_group(x, self.ask_idx))
        bid = self.bid_proj(self._patch_group(x, self.bid_idx))
        misc = self.misc_proj(self._patch_group(x, self.misc_idx))
        h = self.fuse(torch.cat([ask, bid, misc], dim=-1))
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)
        h0 = self.gate(h[:, 0])
        return self.head(h0).squeeze(-1)


class FieldTokenTransformer(nn.Module):
    """
    iTransformer-lite 风格：
    每个原始字段作为 token，先用线性层压缩时间维，再学习字段间关系。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.24,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.temporal_proj = nn.Sequential(
            nn.Linear(self.seq_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.field_pos = nn.Parameter(torch.zeros(1, input_dim, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.field_pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        h = x.transpose(1, 2)
        h = self.temporal_proj(h)
        h = h + self.field_pos[:, :h.size(1), :]
        h = self.encoder(h)
        w = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1)
        pooled = torch.sum(h * w.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(-1)


class TFTLiteTransformer(nn.Module):
    """
    TFT-lite：GRU 提取局部时序状态，门控残差抑制噪声，再用浅层 Transformer 做全局聚合。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 160,
        n_heads: int = 4,
        n_layers: int = 1,
        dim_feedforward: int = 640,
        dropout: float = 0.20,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.gate = GatedResidualBlock(d_model, dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.input_proj(x)
        h, _ = self.gru(h)
        h = self.gate(h)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)
        return self.head(h[:, 0]).squeeze(-1)



class PrototypePatchTransformerLite(nn.Module):
    """
    轻量 Prototype Patch 模型：
    借鉴 Falcon-X 的 latent prototype / positive-negative relation 思路，
    但仅作为 patch token 的小残差校正，避免增加过拟合风险。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.24,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_size = max(1, int(patch_size))
        self.n_patches = int(np.ceil(self.seq_len / self.patch_size))
        self.pad_len = self.n_patches * self.patch_size - self.seq_len
        self.n_prototypes = int(getattr(CFG, "n_prototypes", 8))
        self.prototype_residual_scale = float(getattr(CFG, "prototype_residual_scale", 0.10))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * self.patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.proto_pos = nn.Parameter(torch.zeros(self.n_prototypes, d_model))
        self.proto_neg = nn.Parameter(torch.zeros(self.n_prototypes, d_model))
        self.proto_norm = nn.LayerNorm(d_model)
        self.proto_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gate = GatedResidualBlock(d_model, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.proto_pos, std=0.02)
        nn.init.normal_(self.proto_neg, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_patch_tokens(self, x):
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, self.pad_len, 0), mode="constant", value=0.0)
        b, t, c = x.shape
        x = x.reshape(b, self.n_patches, self.patch_size * c)
        h = self.input_proj(x)

        q = self.proto_norm(h)
        scale = float(q.shape[-1]) ** -0.5
        pos_w = torch.softmax(torch.matmul(q, self.proto_pos.t()) * scale, dim=-1)
        neg_w = torch.softmax(torch.matmul(q, self.proto_neg.t()) * scale, dim=-1)
        proto_ctx = torch.matmul(pos_w, self.proto_pos) - torch.matmul(neg_w, self.proto_neg)
        h = h + self.prototype_residual_scale * self.proto_proj(proto_ctx)
        return h

    def forward(self, x):
        h = self._make_patch_tokens(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)
        h0 = self.gate(h[:, 0])
        return self.head(h0).squeeze(-1)


class WeakConvPatchTransformerLite(nn.Module):
    """
    弱 Conv residual stem：
    用极小残差系数让 depthwise conv 只补充局部微结构，不替代 Patch Transformer 主干。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.24,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_size = max(1, int(patch_size))
        self.n_patches = int(np.ceil(self.seq_len / self.patch_size))
        self.pad_len = self.n_patches * self.patch_size - self.seq_len
        self.conv_residual_scale = float(getattr(CFG, "conv_residual_scale", 0.06))
        kernel = int(getattr(CFG, "conv_kernel_size", 3))
        pad = kernel // 2

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * self.patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.conv_norm = nn.LayerNorm(d_model)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel, padding=pad, groups=d_model
        )
        self.conv_proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gate = GatedResidualBlock(d_model, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_patch_tokens(self, x):
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, self.pad_len, 0), mode="constant", value=0.0)
        b, t, c = x.shape
        x = x.reshape(b, self.n_patches, self.patch_size * c)
        h = self.input_proj(x)

        z = self.conv_norm(h).transpose(1, 2)
        z = self.depthwise_conv(z).transpose(1, 2)
        h = h + self.conv_residual_scale * self.conv_proj(z)
        return h

    def forward(self, x):
        h = self._make_patch_tokens(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)
        h0 = self.gate(h[:, 0])
        return self.head(h0).squeeze(-1)


class PatchGroupResidualTransformerLite(nn.Module):
    """
    Patch 主干 + 极轻 group residual 分支：
    不再让 group attention 替代 Patch 主干，而是只提供 0.05 左右的小残差信息。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.24,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_size = max(1, int(patch_size))
        self.n_patches = int(np.ceil(self.seq_len / self.patch_size))
        self.pad_len = self.n_patches * self.patch_size - self.seq_len
        self.group_residual_scale = float(getattr(CFG, "group_residual_scale", 0.05))

        self.groups = [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35],
            [36, 37, 38],
        ]
        self.n_groups = len(self.groups)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * self.patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.group_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(idx) * self.patch_size, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for idx in self.groups
        ])
        group_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.group_encoder = nn.TransformerEncoder(group_layer, num_layers=1)
        self.group_pos = nn.Parameter(torch.zeros(1, self.n_groups, d_model))
        self.group_to_patch = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gate = GatedResidualBlock(d_model, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.group_pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _patch_all(self, x):
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, self.pad_len, 0), mode="constant", value=0.0)
        b, t, c = x.shape
        return x.reshape(b, self.n_patches, self.patch_size * c)

    def _patch_group(self, x, idx):
        z = x[:, :, idx]
        if self.pad_len > 0:
            z = F.pad(z, (0, 0, self.pad_len, 0), mode="constant", value=0.0)
        b, t, c = z.shape
        return z.reshape(b, self.n_patches, self.patch_size * c)

    def _make_patch_tokens(self, x):
        h = self.input_proj(self._patch_all(x))

        group_tokens = []
        for proj, idx in zip(self.group_proj, self.groups):
            group_tokens.append(proj(self._patch_group(x, idx)))
        # [B, P, G, D] -> [B*P, G, D]
        g = torch.stack(group_tokens, dim=2)
        b, p, ng, d = g.shape
        g = g.reshape(b * p, ng, d)
        g = g + self.group_pos[:, :ng, :]
        g = self.group_encoder(g)
        g = g.mean(dim=1).reshape(b, p, d)

        h = h + self.group_residual_scale * self.group_to_patch(g)
        return h

    def forward(self, x):
        h = self._make_patch_tokens(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)
        h0 = self.gate(h[:, 0])
        return self.head(h0).squeeze(-1)


class PrototypeWeakConvPatchTransformerLite(nn.Module):
    """
    Prototype + WeakConv 双小残差：
    两个 residual scale 都保持很小，只用于测试二者是否互补。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.24,
        patch_size: int = 4,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_size = max(1, int(patch_size))
        self.n_patches = int(np.ceil(self.seq_len / self.patch_size))
        self.pad_len = self.n_patches * self.patch_size - self.seq_len
        self.n_prototypes = int(getattr(CFG, "n_prototypes", 8))
        self.prototype_residual_scale = float(getattr(CFG, "prototype_residual_scale", 0.06))
        self.conv_residual_scale = float(getattr(CFG, "conv_residual_scale", 0.04))
        kernel = int(getattr(CFG, "conv_kernel_size", 3))
        pad = kernel // 2

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * self.patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.proto_pos = nn.Parameter(torch.zeros(self.n_prototypes, d_model))
        self.proto_neg = nn.Parameter(torch.zeros(self.n_prototypes, d_model))
        self.proto_norm = nn.LayerNorm(d_model)
        self.proto_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.conv_norm = nn.LayerNorm(d_model)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad, groups=d_model)
        self.conv_proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gate = GatedResidualBlock(d_model, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.proto_pos, std=0.02)
        nn.init.normal_(self.proto_neg, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_patch_tokens(self, x):
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, self.pad_len, 0), mode="constant", value=0.0)
        b, t, c = x.shape
        x = x.reshape(b, self.n_patches, self.patch_size * c)
        h = self.input_proj(x)

        q = self.proto_norm(h)
        scale = float(q.shape[-1]) ** -0.5
        pos_w = torch.softmax(torch.matmul(q, self.proto_pos.t()) * scale, dim=-1)
        neg_w = torch.softmax(torch.matmul(q, self.proto_neg.t()) * scale, dim=-1)
        proto_ctx = torch.matmul(pos_w, self.proto_pos) - torch.matmul(neg_w, self.proto_neg)
        h = h + self.prototype_residual_scale * self.proto_proj(proto_ctx)

        z = self.conv_norm(h).transpose(1, 2)
        z = self.depthwise_conv(z).transpose(1, 2)
        h = h + self.conv_residual_scale * self.conv_proj(z)
        return h

    def forward(self, x):
        h = self._make_patch_tokens(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)
        h0 = self.gate(h[:, 0])
        return self.head(h0).squeeze(-1)



class RMSNorm(nn.Module):
    """轻量 RMSNorm，避免依赖较新版本 torch.nn.RMSNorm。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class LowRankResidualAdapter(nn.Module):
    """
    z' = z + scale * W2(GELU(W1(RMSNorm(z))))。
    rank 远小于 d_model，用很少参数补充跨变量/非线性交互。
    """
    def __init__(self, d_model: int, rank: int, dropout: float, scale: float = 1.0):
        super().__init__()
        rank = max(4, int(rank))
        self.scale = float(scale)
        self.norm = RMSNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up(self.dropout(F.gelu(self.down(self.norm(x)))))
        return x + self.scale * z


class CrossSectionContextAdapter(nn.Module):
    """
    同一交易日截面上下文适配器。

    DailyBatchSampler 保证训练/验证批次按交易日组织；推理也使用相同日批次。
    适配器只使用当前批次内部表示的均值与标准差，不引入额外数据字段。
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        dropout: float,
        residual_scale: float = 0.08,
    ):
        super().__init__()
        rank = max(4, int(rank))
        self.residual_scale = float(residual_scale)
        self.norm = RMSNorm(d_model)
        self.context_down = nn.Linear(d_model * 2, rank, bias=False)
        self.context_up = nn.Linear(rank, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.size(0) <= 1:
            return z

        mean = z.mean(dim=0, keepdim=True)
        var = (z - mean).pow(2).mean(dim=0, keepdim=True)
        std = torch.sqrt(var + 1e-6)
        context = torch.cat([mean, std], dim=-1)

        delta = self.context_up(
            self.dropout(F.gelu(self.context_down(context)))
        )
        delta = delta.expand_as(z)
        gate = torch.sigmoid(self.gate(self.norm(z)))
        return z + self.residual_scale * gate * delta


class FeatureGroupGatedFusion(nn.Module):
    """
    先在每个 patch 内对特征做低秩时间编码，再按原始字段组投影并门控融合。

    输入:  [B, N, F, R]
    输出:  [B, N, D]
    """
    def __init__(
        self,
        input_dim: int,
        patch_feature_dim: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        if input_dim != len(FEATURE_COLS):
            raise ValueError(
                f"PatchTSTFinanceLite 期望 {len(FEATURE_COLS)} 个字段，实际为 {input_dim}。"
            )

        self.groups = [
            [0, 1, 2, 3, 4],              # OHLC / pre_close
            [5, 6, 7, 8, 9],              # ask prices
            [10, 11, 12, 13, 14],         # bid prices
            [15, 16, 17, 18, 19],         # ask volumes
            [20, 21, 22, 23, 24],         # bid volumes
            [25, 26, 27, 28, 29],         # ask order counts
            [30, 31, 32, 33, 34],         # bid order counts
            [35, 36, 37],                 # amount / volume / deal_number
        ]
        self.n_groups = len(self.groups)

        self.group_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(idx) * patch_feature_dim, d_model),
                RMSNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for idx in self.groups
        ])

        gate_hidden = max(8, d_model // 4)
        self.gate_score = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.group_bias = nn.Parameter(torch.zeros(self.n_groups))
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, r = x.shape
        tokens = []
        for idx, proj in zip(self.groups, self.group_proj):
            g = x[:, :, idx, :].reshape(b, n, len(idx) * r)
            tokens.append(proj(g))

        g = torch.stack(tokens, dim=2)  # [B, N, G, D]
        logits = self.gate_score(g).squeeze(-1)
        logits = logits + self.group_bias.view(1, 1, -1)
        weights = torch.softmax(logits, dim=2)
        fused = (weights.unsqueeze(-1) * g).sum(dim=2)
        return self.out_norm(fused)


class PatchTSTFinanceLite(nn.Module):
    """
    面向高频量价截面预测的 PatchTST-Lite。

    结构:
    L×F 原始历史
      -> 按字段 InstanceNorm
      -> depthwise temporal convolution
      -> overlapping patchify
      -> 共享低秩 patch temporal projection
      -> 特征组 gated fusion
      -> Pre-LN Transformer over patch tokens
      -> gated temporal pooling
      -> 低秩变量 mixer
      -> 横截面 context adapter
      -> score head

    与直接 flatten(P×F) 不同，输入投影先把每个字段的 P 个时间点压到
    patch_feature_dim，再按字段组融合，因此参数量不会随 P×F 快速膨胀。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 80,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 160,
        dropout: float = 0.10,
        patch_size: int = 16,
        patch_stride: int = 8,
        patch_feature_dim: int = 8,
        mixer_rank: int = 16,
        context_rank: int = 16,
        context_residual_scale: float = 0.08,
        feature_dropout: float = 0.02,
        conv_kernel_size: int = 5,
        conv_residual_scale: float = 0.35,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.patch_feature_dim = int(patch_feature_dim)
        self.feature_dropout = float(feature_dropout)
        self.conv_residual_scale = float(conv_residual_scale)

        if self.patch_size <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_size 和 patch_stride 必须为正整数。")
        if self.seq_len < self.patch_size:
            raise ValueError("seq_len 必须不小于 patch_size。")
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除。")

        # 与原论文相同，在尾部重复最近值一个 stride，使 L=512/P=16/S=8 得到 64 tokens。
        self.pad_right = self.patch_stride
        self.n_patches = (
            (self.seq_len + self.pad_right - self.patch_size) // self.patch_stride + 1
        )

        self.instance_norm = nn.InstanceNorm1d(
            self.input_dim,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=False,
        )

        kernel = max(3, int(conv_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        self.depthwise_conv = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=self.input_dim,
            bias=True,
        )
        self.conv_gate = nn.Parameter(torch.tensor(0.0))

        # 对所有字段共享同一个 P -> R 投影，保留 PatchTST 的权重共享思想。
        self.patch_temporal_proj = nn.Sequential(
            nn.Linear(self.patch_size, self.patch_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.group_fusion = FeatureGroupGatedFusion(
            input_dim=self.input_dim,
            patch_feature_dim=self.patch_feature_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.token_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            norm=RMSNorm(d_model),
        )

        pool_hidden = max(16, d_model // 2)
        self.pool_score = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, pool_hidden),
            nn.Tanh(),
            nn.Linear(pool_hidden, 1),
        )
        self.pool_gate = nn.Sequential(
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.variable_mixer = LowRankResidualAdapter(
            d_model=d_model,
            rank=mixer_rank,
            dropout=dropout,
            scale=1.0,
        )
        self.context_adapter = CrossSectionContextAdapter(
            d_model=d_model,
            rank=context_rank,
            dropout=dropout,
            residual_scale=context_residual_scale,
        )

        self.head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos, std=0.02)
        nn.init.zeros_(self.depthwise_conv.bias)
        nn.init.kaiming_normal_(self.depthwise_conv.weight, nonlinearity="linear")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.feature_dropout <= 0:
            return x
        keep_prob = 1.0 - self.feature_dropout
        mask = torch.empty(
            x.size(0), x.size(1), 1,
            device=x.device,
            dtype=x.dtype,
        ).bernoulli_(keep_prob)
        return x * mask / max(keep_prob, 1e-6)

    def _make_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, F] -> [B, F, L]
        x = x.transpose(1, 2)
        x = self.instance_norm(x)
        x = self._feature_dropout(x)

        local = F.gelu(self.depthwise_conv(x))
        conv_scale = self.conv_residual_scale * torch.sigmoid(self.conv_gate)
        x = x + conv_scale * local

        # 重复最后一个观测值，不使用未来信息。
        x = F.pad(x, (0, self.pad_right), mode="replicate")
        patches = x.unfold(
            dimension=2,
            size=self.patch_size,
            step=self.patch_stride,
        )  # [B, F, N, P]

        if patches.size(2) != self.n_patches:
            patches = patches[:, :, :self.n_patches, :]

        patch_features = self.patch_temporal_proj(patches)
        patch_features = patch_features.permute(0, 2, 1, 3).contiguous()
        h = self.group_fusion(patch_features)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._make_patch_tokens(x)
        h = self.token_dropout(h + self.pos[:, :h.size(1), :])
        h = self.encoder(h)

        weights = torch.softmax(self.pool_score(h).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        recent = h[:, -1, :]
        gate = self.pool_gate(torch.cat([pooled, recent], dim=-1))
        z = gate * pooled + (1.0 - gate) * recent

        z = self.variable_mixer(z)
        z = self.context_adapter(z)
        return self.head(z).squeeze(-1)


class E2ETransformer(nn.Module):
    """
    保留原始类名 E2ETransformer，避免后续流水线依赖断裂。
    通过 cfg.architecture 选择 PatchTST-Finance-Lite 或保留的轻量基线架构。
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.20,
        architecture: str = "patch_transformer",
        patch_size: int = 4,
        patch_stride: int = 4,
        patch_feature_dim: int = 8,
        mixer_rank: int = 16,
        context_rank: int = 16,
        context_residual_scale: float = 0.08,
        feature_dropout: float = 0.02,
        conv_kernel_size: int = 5,
        conv_residual_scale: float = 0.35,
    ):
        super().__init__()
        architecture = str(architecture)
        if architecture == "patchtst_finance_lite":
            self.model = PatchTSTFinanceLite(
                input_dim=input_dim,
                seq_len=seq_len,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_feature_dim=patch_feature_dim,
                mixer_rank=mixer_rank,
                context_rank=context_rank,
                context_residual_scale=context_residual_scale,
                feature_dropout=feature_dropout,
                conv_kernel_size=conv_kernel_size,
                conv_residual_scale=conv_residual_scale,
            )
        elif architecture == "siamese_lob":
            self.model = SiameseLOBTransformer(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)
        elif architecture == "field_token":
            self.model = FieldTokenTransformer(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)
        elif architecture == "tft_lite":
            self.model = TFTLiteTransformer(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)
        elif architecture == "prototype_patch_lite":
            self.model = PrototypePatchTransformerLite(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)
        elif architecture == "weak_conv_patch":
            self.model = WeakConvPatchTransformerLite(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)
        elif architecture == "patch_group_residual_lite":
            self.model = PatchGroupResidualTransformerLite(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)
        elif architecture == "prototype_weakconv_patch":
            self.model = PrototypeWeakConvPatchTransformerLite(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)
        else:
            self.model = PatchTransformerLite(input_dim, seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, patch_size)

    def forward(self, x):
        return self.model(x)


def build_model(cfg: Config, input_dim: int) -> nn.Module:
    model = E2ETransformer(
        input_dim=input_dim,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        architecture=getattr(cfg, "architecture", "patch_transformer"),
        patch_size=getattr(cfg, "patch_size", 4),
        patch_stride=getattr(cfg, "patch_stride", getattr(cfg, "patch_size", 4)),
        patch_feature_dim=getattr(cfg, "patch_feature_dim", 8),
        mixer_rank=getattr(cfg, "mixer_rank", 16),
        context_rank=getattr(cfg, "context_rank", 16),
        context_residual_scale=getattr(cfg, "context_residual_scale", 0.08),
        feature_dropout=getattr(cfg, "feature_dropout", 0.02),
        conv_kernel_size=getattr(cfg, "conv_kernel_size", 5),
        conv_residual_scale=getattr(cfg, "conv_residual_scale", 0.35),
    )

    n = count_parameters(model)
    print(f"[Model] architecture: {getattr(cfg, 'architecture', 'patch_transformer')}")
    print(f"[Model] trainable parameters: {n:,}")

    if not (100_000 <= n <= 100_000_000):
        raise ValueError(f"参数数量 {n} 违反比赛限制。")

    return model

# ============================================================
# 9. 训练与验证
# ============================================================

def _torch_zscore(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (v - v.mean()) / (v.std(unbiased=False) + eps)


def _pairwise_rank_loss(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    pair_count: int = 4096,
) -> torch.Tensor:
    n = pred_z.numel()
    if n < 20:
        return pred_z.new_tensor(0.0)

    k = int(min(max(pair_count, 512), n * 4))
    i = torch.randint(0, n, (k,), device=pred_z.device)
    j = torch.randint(0, n, (k,), device=pred_z.device)

    dy = target_z[i] - target_z[j]
    valid = dy.abs() > 1e-6
    if valid.sum() < 10:
        return pred_z.new_tensor(0.0)

    sign = torch.sign(dy[valid])
    margin = (pred_z[i[valid]] - pred_z[j[valid]]) * sign
    return F.softplus(-margin).mean()


def composite_factor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: Config
) -> torch.Tensor:
    """
    排行榜导向训练目标：
    - IC loss：最大化同日截面相关性，服务 IC_mean。
    - pairwise rank loss：直接优化截面排序方向，减少 batch 内数值尺度依赖。
    - smooth long-short loss：用平滑多空权重贴近 10 分组多空 SR。
    - stress loss：惩罚负 IC / 低 IC 日，服务 IC_IR 与 Stress。
    - point loss / dispersion penalty：作为训练稳定项，权重较低。
    """
    pred = pred.float()
    target = target.float()

    if pred.numel() < 20:
        return F.smooth_l1_loss(pred, target)

    pred_z = _torch_zscore(pred)
    target_z = _torch_zscore(target)

    ic = torch.mean(pred_z * target_z)
    ic_loss = -ic

    point_loss = F.smooth_l1_loss(pred_z, target_z)

    temp = float(getattr(cfg, "softmax_temperature", 3.0))
    # temp 下调后不再只盯极少数股票，更接近 top/bottom 10% 的稳健多空。
    long_w = torch.softmax(temp * pred_z, dim=0)
    short_w = torch.softmax(-temp * pred_z, dim=0)
    soft_long_short = torch.sum((long_w - short_w) * target_z)
    longshort_loss = -soft_long_short

    rank_loss = _pairwise_rank_loss(
        pred_z,
        target_z,
        pair_count=int(getattr(cfg, "rank_pair_count", 4096)),
    )

    margin = float(getattr(cfg, "negative_ic_margin", 0.0))
    stress_loss = F.relu(margin - ic).pow(2)

    raw_std = pred.std(unbiased=False)
    disp_loss = F.relu(float(getattr(cfg, "min_pred_std", 0.02)) - raw_std).pow(2)

    loss = (
        cfg.loss_ic_weight * ic_loss
        + cfg.loss_point_weight * point_loss
        + cfg.loss_longshort_weight * longshort_loss
        + getattr(cfg, "loss_rank_weight", 0.0) * rank_loss
        + getattr(cfg, "loss_stress_weight", 0.0) * stress_loss
        + cfg.loss_disp_weight * disp_loss
    )

    return loss

def compute_factor_metrics_from_arrays(
    pred: np.ndarray,
    y: np.ndarray,
    date_ids: np.ndarray,
    cfg: Config = CFG
) -> Dict:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    date_ids = np.asarray(date_ids)

    daily_rows = []
    ls_rows = []

    for d in np.unique(date_ids):
        m = date_ids == d
        x = pred[m]
        yy = y[m]

        valid = np.isfinite(x) & np.isfinite(yy)
        x = x[valid]
        yy = yy[valid]

        if len(x) < 50:
            continue

        x_std = np.std(x)
        y_std = np.std(yy)

        if x_std < 1e-12 or y_std < 1e-12:
            continue

        ic = float(np.corrcoef(x, yy)[0, 1])

        x_rank = pd.Series(x).rank(method="first").to_numpy()
        y_rank = pd.Series(yy).rank(method="first").to_numpy()
        rank_ic = float(np.corrcoef(x_rank, y_rank)[0, 1])

        pct = pd.Series(x).rank(pct=True, method="first").to_numpy()
        long_ret = float(np.mean(yy[pct >= 0.90])) if np.any(pct >= 0.90) else np.nan
        short_ret = float(np.mean(yy[pct <= 0.10])) if np.any(pct <= 0.10) else np.nan
        ls_ret = long_ret - short_ret

        daily_rows.append({
            "date_id": int(d),
            "IC": ic,
            "RankIC": rank_ic,
        })

        ls_rows.append({
            "date_id": int(d),
            "long_short_ret": ls_ret,
        })

    if len(daily_rows) == 0:
        empty = {
            "loss": np.nan,
            "ic_mean": 0.0,
            "ic_std": np.nan,
            "ic_ir": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_ir": 0.0,
            "long_short_sharpe": 0.0,
            "stress_min_ic_mean": 0.0,
            "stress_avg_ic_mean": 0.0,
            "valid_days": 0,
            "local_proxy_score": 0.0,
        }
        empty.update({
            "proxy_rank_ic_mean": 0.0,
            "proxy_rank_ic_ir": 0.0,
            "proxy_rank_sr": 0.0,
            "proxy_rank_stress": 0.0,
        })
        return empty

    daily_df = pd.DataFrame(daily_rows).sort_values("date_id").reset_index(drop=True)
    ls_df = pd.DataFrame(ls_rows).sort_values("date_id").reset_index(drop=True)

    ic_series = daily_df["IC"].replace([np.inf, -np.inf], np.nan).dropna()
    rank_ic_series = daily_df["RankIC"].replace([np.inf, -np.inf], np.nan).dropna()
    ls_series = ls_df["long_short_ret"].replace([np.inf, -np.inf], np.nan).dropna()

    ic_mean = float(ic_series.mean()) if len(ic_series) else 0.0
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan
    ic_ir = float(ic_mean / (ic_std + 1e-12)) if np.isfinite(ic_std) else 0.0

    rank_ic_mean = float(rank_ic_series.mean()) if len(rank_ic_series) else 0.0
    rank_ic_std = float(rank_ic_series.std(ddof=1)) if len(rank_ic_series) > 1 else np.nan
    rank_ic_ir = float(rank_ic_mean / (rank_ic_std + 1e-12)) if np.isfinite(rank_ic_std) else 0.0

    ls_mean = float(ls_series.mean()) if len(ls_series) else 0.0
    ls_std = float(ls_series.std(ddof=1)) if len(ls_series) > 1 else np.nan
    long_short_sharpe = (
        float(np.sqrt(252) * ls_mean / (ls_std + 1e-12))
        if np.isfinite(ls_std) and ls_std > 0
        else 0.0
    )

    stress_values = []
    if len(daily_df) >= 20:
        chunks = np.array_split(daily_df, 4)
        for c in chunks:
            stress_values.append(float(c["IC"].mean()))
    stress_min_ic_mean = float(np.nanmin(stress_values)) if len(stress_values) else 0.0
    stress_avg_ic_mean = float(np.nanmean(stress_values)) if len(stress_values) else 0.0

    proxy = compute_local_proxy_score_from_metrics(
        ic_mean=ic_mean,
        ic_ir=ic_ir,
        long_short_sharpe=long_short_sharpe,
        stress_min_ic_mean=stress_min_ic_mean,
        cfg=cfg,
    )

    out = {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_ir": rank_ic_ir,
        "long_short_sharpe": long_short_sharpe,
        "stress_min_ic_mean": stress_min_ic_mean,
        "stress_avg_ic_mean": stress_avg_ic_mean,
        "valid_days": int(len(daily_df)),
    }
    out.update(proxy)
    return out


@torch.no_grad()
def evaluate_model(model, loader, device, cfg: Config) -> Tuple[float, Dict]:
    model.eval()

    preds, ys, dts = [], [], []
    total_loss, total_n = 0.0, 0

    for batch in loader:
        if len(batch) == 3:
            x, y, date_id = batch
        else:
            x, y = batch
            date_id = torch.zeros_like(y, dtype=torch.int32)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
            p = model(x)
            loss = composite_factor_loss(p, y, cfg)

        total_loss += float(loss.item()) * len(y)
        total_n += len(y)

        preds.append(p.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        dts.append(date_id.detach().cpu().numpy())

    pred = np.concatenate(preds) if preds else np.array([])
    yy = np.concatenate(ys) if ys else np.array([])
    dd = np.concatenate(dts) if dts else np.array([])

    metrics = compute_factor_metrics_from_arrays(pred, yy, dd, cfg=cfg)
    avg_loss = total_loss / max(total_n, 1)
    metrics["loss"] = avg_loss

    return avg_loss, metrics


def select_validation_score(metrics: Dict, cfg: Config) -> float:
    """
    用官方四分项思路做本地模型选择：
    不只看 local_proxy_score，而是提高 IC_IR / SR / Stress 的权重，降低单一公榜代理过拟合。
    """
    ic_mean = float(metrics.get("ic_mean", 0.0) or 0.0)
    ic_ir = float(metrics.get("ic_ir", 0.0) or 0.0)
    sr = float(metrics.get("long_short_sharpe", 0.0) or 0.0)
    stress = float(metrics.get("stress_min_ic_mean", 0.0) or 0.0)

    ic_part = np.tanh(ic_mean / max(float(cfg.proxy_ic_scale), 1e-6))
    icir_part = np.tanh(ic_ir / max(float(cfg.proxy_icir_scale), 1e-6))
    sr_part = np.tanh(sr / max(float(cfg.proxy_sr_scale), 1e-6))
    stress_part = np.tanh(stress / max(float(cfg.proxy_stress_scale), 1e-6))

    # 当前多轮提交结果显示，公榜更跟随 ICMean / ICIR；
    # 因此 checkpoint 选择更偏向截面排序稳定性，避免本地 SR 过度主导。
    score = (
        0.45 * ic_part
        + 0.35 * icir_part
        + 0.10 * sr_part
        + 0.10 * stress_part
    )
    return float(score)


def train_and_save(
    data_source_name: str = None,
    train_start: str = None,
    train_end: str = None,
    cfg: Config = CFG
) -> str:
    """
    训练模型并保存权重。
    """
    set_seed(cfg.seed)
    configure_runtime(cfg)
    print_resource_plan(cfg)

    data_source_name = _resolve_train_data_source(data_source_name, cfg)
    train_start = train_start or cfg.train_start
    train_end = train_end or cfg.train_end

    print("=" * 80)
    print("[Train]")
    print(f"data_source = {data_source_name}")
    print(f"range       = {train_start} -> {train_end}")
    print(f"model_path  = {cfg.model_path}")
    print("=" * 80)

    t0 = time.time()

    bars = load_bar_data(data_source_name, train_start, train_end)
    features, meta, mean, std = build_bar_cache(bars)

    # G1 32G 内存调整：features/meta 构造完成后立即释放原始 bar DataFrame
    del bars
    gc.collect()

    labels = make_forward_labels(
        train_start,
        train_end,
        neutralize=cfg.neutralize_label,
    )

    samples = meta.merge(labels, on=["date", "instrument"], how="inner")
    samples = samples.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])
    samples = samples.sort_values(["date", "instrument"]).reset_index(drop=True)

    del labels, meta
    gc.collect()

    if samples.empty:
        raise RuntimeError("合并 bar 和标签后没有训练样本。")

    unique_dates = np.array(sorted(samples["date"].unique()))
    split = max(1, int(len(unique_dates) * 0.85))

    train_dates = set(unique_dates[:split])
    val_dates = set(unique_dates[split:])

    train_df = samples[samples["date"].isin(train_dates)].reset_index(drop=True)
    val_df = samples[samples["date"].isin(val_dates)].reset_index(drop=True)

    del samples
    gc.collect()

    train_df = _sample_df(train_df, cfg.max_train_samples, cfg.seed)
    val_df = _sample_df(val_df, cfg.max_val_samples, cfg.seed + 1)

    train_date_codes, train_uniques = pd.factorize(train_df["date"], sort=True)
    val_date_codes, val_uniques = pd.factorize(val_df["date"], sort=True)

    print(f"[Train] train dates   = {min(train_dates)} -> {max(train_dates)}")
    print(f"[Train] val dates     = {min(val_dates)} -> {max(val_dates)}")
    print(f"[Train] train samples = {len(train_df):,}")
    print(f"[Train] val samples   = {len(val_df):,}")
    print(f"[Train] input fields  = {len(FEATURE_COLS)}")
    print(f"[Train] seq_len       = {cfg.seq_len}")
    print(f"[Train] epochs        = {cfg.epochs}")
    print(f"[Train] day_batch_size= {cfg.day_batch_size}")

    tr_ds = WindowDataset(
        features,
        train_df["end_pos"].values,
        train_df["first_pos"].values,
        train_df["target"].values,
        cfg.seq_len,
        date_ids=train_date_codes,
    )

    va_ds = WindowDataset(
        features,
        val_df["end_pos"].values,
        val_df["first_pos"].values,
        val_df["target"].values,
        cfg.seq_len,
        date_ids=val_date_codes,
    )

    del train_df, val_df
    gc.collect()

    train_sampler = DailyBatchSampler(
        date_ids=train_date_codes,
        max_batch_size=cfg.day_batch_size,
        shuffle=True,
        seed=cfg.seed,
    )

    val_sampler = DailyBatchSampler(
        date_ids=val_date_codes,
        max_batch_size=cfg.val_day_batch_size,
        shuffle=False,
        seed=cfg.seed,
    )

    loader_kwargs = make_loader_kwargs(cfg)

    tr_loader = DataLoader(
        tr_ds,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )

    va_loader = DataLoader(
        va_ds,
        batch_sampler=val_sampler,
        **loader_kwargs,
    )

    device = get_device()
    print(f"[Train] device        = {device}")

    model = build_model(cfg, input_dim=len(FEATURE_COLS)).to(device)
    model = maybe_wrap_dataparallel(model, cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = max(1, len(tr_loader) * cfg.epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.lr * cfg.min_lr_ratio,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_score = -1e9
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    history = []
    top_states = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        total_loss, total_n = 0.0, 0
        ema_loss = None

        pbar = tqdm(
            tr_loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(pbar, start=1):
            x, y, date_id = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                pred = model(x)
                loss = composite_factor_loss(pred, y, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = float(loss.item())
            ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value

            total_loss += loss_value * len(y)
            total_n += len(y)

            lr_now = optimizer.param_groups[0]["lr"]

            try:
                pbar.set_postfix(
                    loss=f"{loss_value:.5f}",
                    ema=f"{ema_loss:.5f}",
                    lr=f"{lr_now:.2e}",
                )
            except Exception:
                pass

        tr_loss = total_loss / max(total_n, 1)
        va_loss, val_metrics = evaluate_model(model, va_loader, device, cfg)

        val_score = select_validation_score(val_metrics, cfg)
        improved = val_score > best_score + 1e-6

        current_state = None
        if int(getattr(cfg, "checkpoint_top_k", 1)) > 1 or improved:
            current_state = unwrap_state_dict(model)

        if int(getattr(cfg, "checkpoint_top_k", 1)) > 1 and current_state is not None:
            top_states.append((val_score, current_state))
            top_states = sorted(top_states, key=lambda x: x[0], reverse=True)[:int(cfg.checkpoint_top_k)]

        if improved:
            best_score = val_score
            best_epoch = epoch
            bad_epochs = 0
            if int(getattr(cfg, "checkpoint_top_k", 1)) > 1 and len(top_states) > 0:
                best_state = average_state_dicts([sd for _, sd in top_states])
            else:
                best_state = current_state

            ckpt = {
                "model_state": best_state,
                "cfg": asdict(cfg),
                "feature_cols": FEATURE_COLS,
                "mean": mean,
                "std": std,
                "best_val_local_proxy_score": best_score,
                "best_epoch": best_epoch,
                "data_source_name": data_source_name,
                "train_start": train_start,
                "train_end": train_end,
            }

            save_model_json(ckpt, cfg.model_path)
            print(
                f"[Train] new best model saved: "
                f"epoch={best_epoch}, "
                f"val_local_proxy_score={best_score:.6f}"
            )
        else:
            if epoch >= cfg.min_epochs:
                bad_epochs += 1

        epoch_info = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_select_score": val_score,
            "val_local_proxy_score": val_metrics["local_proxy_score"],
            "val_ic_mean": val_metrics["ic_mean"],
            "val_ic_ir": val_metrics["ic_ir"],
            "val_rank_ic_mean": val_metrics["rank_ic_mean"],
            "val_rank_ic_ir": val_metrics["rank_ic_ir"],
            "val_long_short_sharpe": val_metrics["long_short_sharpe"],
            "val_stress_min_ic_mean": val_metrics["stress_min_ic_mean"],
            "val_proxy_rank_ic_mean": val_metrics["proxy_rank_ic_mean"],
            "val_proxy_rank_ic_ir": val_metrics["proxy_rank_ic_ir"],
            "val_proxy_rank_sr": val_metrics["proxy_rank_sr"],
            "val_proxy_rank_stress": val_metrics["proxy_rank_stress"],
            "lr": optimizer.param_groups[0]["lr"],
            "best_val_local_proxy_score_so_far": best_score,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "elapsed_min": (time.time() - t0) / 60,
        }

        history.append(epoch_info)

        print(
            f"[Epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={tr_loss:.6f} | "
            f"val_loss={va_loss:.6f} | "
            f"val_proxy={val_score:.6f} | "
            f"IC={val_metrics['ic_mean']:.6f} | "
            f"ICIR={val_metrics['ic_ir']:.4f} | "
            f"SR={val_metrics['long_short_sharpe']:.4f} | "
            f"Stress={val_metrics['stress_min_ic_mean']:.6f} | "
            f"best_proxy={best_score:.6f} | "
            f"best_epoch={best_epoch} | "
            f"bad_epochs={bad_epochs} | "
            f"elapsed={(time.time() - t0) / 60:.1f} min"
        )


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if epoch >= cfg.min_epochs and bad_epochs >= cfg.early_stopping_patience:
            print(
                f"[Train] early stopping triggered at epoch={epoch}, "
                f"best_epoch={best_epoch}, best_proxy={best_score:.6f}"
            )
            break

    if best_state is None:
        best_state = unwrap_state_dict(model)

    ckpt = {
        "model_state": best_state,
        "cfg": asdict(cfg),
        "feature_cols": FEATURE_COLS,
        "mean": mean,
        "std": std,
        "best_val_local_proxy_score": best_score,
        "best_epoch": best_epoch,
        "data_source_name": data_source_name,
        "train_start": train_start,
        "train_end": train_end,
    }

    save_model_json(ckpt, cfg.model_path)

    print("=" * 80)
    print(f"[Train] saved checkpoint : {cfg.model_path}")
    print(f"[Train] best epoch       : {best_epoch}")
    print(f"[Train] best proxy score : {best_score:.6f}")
    print(f"[Train] total elapsed    : {(time.time() - t0) / 60:.1f} min")
    print("=" * 80)

    del features, tr_ds, va_ds, tr_loader, va_loader, model, optimizer, scheduler
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cfg.model_path


# ============================================================
# 10. 推理
# ============================================================

# ==== 模型存/读: 一律用文本类文件 (JSON), 不使用 .pt 等二进制 ====
def save_model_json(ckpt, model_path):
    """把 checkpoint 存成 JSON 文本文件。

    state_dict 里每个张量转为 float16 → bytes → base64 字符串,
    大幅减小文件体积; 其余字段 (结构超参 / 标准化统计等) 原样写入;
    加载时用 load_model_json 按 dtype/shape 还原。"""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    sd = ckpt.get("state_dict", ckpt.get("model_state"))
    if sd is None:
        raise KeyError("ckpt 中缺少 'state_dict' 或 'model_state' 字段")
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        # float16 → bytes → base64, 比 JSON float 列表小 5-6 倍
        arr = t.numpy().astype(np.float16)
        b64 = base64.b64encode(arr.tobytes()).decode("ascii")
        tensors[k] = {
            "original_dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data": b64,
            "encoding": "base16",
        }
    payload = {k: v for k, v in ckpt.items() if k not in ("state_dict", "model_state")}
    payload["state_dict"] = tensors
    # 将 numpy 数组转为 Python list，确保 JSON 可序列化
    if "mean" in payload and hasattr(payload["mean"], "tolist"):
        payload["mean"] = payload["mean"].tolist()
    if "std" in payload and hasattr(payload["std"], "tolist"):
        payload["std"] = payload["std"].tolist()
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model_json(model_path, map_location="cpu"):
    """读取 save_model_json 写出的 JSON, 把 state_dict 还原为张量 dict。

    兼容 base16 编码新格式与旧版 JSON float 列表格式。
    返回结构与原 torch.load(...) 的 checkpoint 一致。"""
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        if meta.get("encoding") == "base16":
            # base64 → bytes → float16 numpy → 还原为原始 dtype
            arr = np.frombuffer(base64.b64decode(meta["data"]), dtype=np.float16)
            t = torch.from_numpy(arr.astype(np.float32)).reshape(meta["shape"])
            orig_dtype = meta.get("original_dtype", "float32")
            if orig_dtype != "float32":
                t = t.to(getattr(torch, orig_dtype))
        else:
            # 兼容旧格式: JSON float 列表
            t = torch.tensor(meta["data"], dtype=getattr(torch, meta.get("dtype", "float32")))
            t = t.reshape(meta["shape"])
        sd[k] = t.to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


def _load_checkpoint(model_path: str) -> Dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型检查点未找到：{model_path}")
    return load_model_json(model_path, map_location="cpu")


@torch.no_grad()
def predict_scores(
    data_source_name: str,
    start_date: str,
    end_date: str,
    model_path: str = None,
    cfg: Config = CFG
) -> pd.DataFrame:
    configure_runtime(cfg)
    data_source_name = _resolve_infer_data_source(data_source_name, cfg)

    model_path = model_path or cfg.model_path
    ckpt = _load_checkpoint(model_path)

    ckpt_cfg_dict = ckpt.get("cfg", asdict(cfg))
    pred_cfg = Config(**{**asdict(cfg), **ckpt_cfg_dict})

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    context_start = minus_days(start_date, pred_cfg.infer_history_days)

    print("=" * 80)
    print("[Predict]")
    print(f"data_source  = {data_source_name}")
    print(f"bar range    = {context_start} -> {end_date}")
    print(f"score range  = {start_date} -> {end_date}")
    print(f"model_path   = {model_path}")
    print("=" * 80)

    bars = load_bar_data(data_source_name, context_start, end_date)
    features, meta, _, _ = build_bar_cache(bars, mean=mean, std=std)

    del bars
    gc.collect()

    s0 = normalize_dt(start_date)
    s1 = normalize_dt(end_date)

    meta = meta[(meta["date"] >= s0) & (meta["date"] <= s1)].reset_index(drop=True)

    if meta.empty:
        raise RuntimeError("请求的日期范围内没有预测行。")

    infer_date_codes, _ = pd.factorize(meta["date"], sort=True)
    ds = WindowDataset(
        features,
        meta["end_pos"].values,
        meta["first_pos"].values,
        None,
        pred_cfg.seq_len,
        date_ids=infer_date_codes,
    )

    # 横截面上下文适配器要求同一批次来自同一交易日。
    # CSI1000 单日股票数通常低于 val_day_batch_size，因此每天形成一个完整批次。
    infer_sampler = DailyBatchSampler(
        date_ids=infer_date_codes,
        max_batch_size=pred_cfg.val_day_batch_size,
        shuffle=False,
        seed=pred_cfg.seed,
    )
    infer_loader_kwargs = {
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    loader = DataLoader(
        ds,
        batch_sampler=infer_sampler,
        **infer_loader_kwargs,
    )

    device = get_device()

    model = build_model(pred_cfg, input_dim=len(FEATURE_COLS)).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = maybe_wrap_dataparallel(model, pred_cfg)
    model.eval()

    preds = np.zeros(len(meta), dtype=np.float32)

    pbar = tqdm(loader, desc="推理中", dynamic_ncols=True, leave=True)

    for x, idx in pbar:
        x = x.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(pred_cfg.use_amp and device.type == "cuda")):
            p = model(x).detach().cpu().numpy().astype(np.float32)

        preds[np.asarray(idx)] = p

    out = meta[["date", "instrument"]].copy()
    out["score"] = preds

    def _cs_z(s):
        v = s.astype(float)
        sd = v.std()
        if not np.isfinite(sd) or sd < 1e-8:
            return v * 0.0
        return (v - v.mean()) / (sd + 1e-8)

    out["score"] = out.groupby("date")["score"].transform(_cs_z)
    out["score"] = out["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["instrument"] = out["instrument"].astype(str)

    out = out[["date", "instrument", "score"]]
    out = out.sort_values(["date", "instrument"]).reset_index(drop=True)

    print(f"[Predict] output rows = {len(out):,}")
    print(f"[Predict] dates       = {out['date'].nunique():,}")
    print(out.head())

    del features, meta, ds, loader, infer_sampler, model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


# ============================================================
# 11. 本地公榜近似评估
# ============================================================

def _safe_zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v, dtype=float)

    valid = np.isfinite(v)

    if valid.sum() < 3:
        return out

    mu = np.nanmean(v[valid])
    sd = np.nanstd(v[valid])

    if not np.isfinite(sd) or sd < 1e-12:
        return out

    out[valid] = (v[valid] - mu) / (sd + 1e-12)
    return out


def _platform_like_process_one_date(g: pd.DataFrame) -> pd.DataFrame:
    """
    本地近似复刻平台处理：
    1. 截面 1% / 99% winsorize
    2. 截面 z-score
    3. 对 BARRA 风险因子做截面回归，取残差
    """
    out = g.copy()

    s = pd.to_numeric(out["score"], errors="coerce").astype(float).to_numpy()
    valid_s = np.isfinite(s)

    if valid_s.sum() < 20:
        out["score_processed"] = 0.0
        return out

    lo, hi = np.nanpercentile(s[valid_s], [1, 99])
    s = np.clip(s, lo, hi)
    s = _safe_zscore(s)

    X = out[RISK_COLS].astype(float).to_numpy()

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])

    X_mu = np.nanmean(X, axis=0)
    X_sd = np.nanstd(X, axis=0) + 1e-6
    X = (X - X_mu) / X_sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = s.copy()
    valid = np.isfinite(y)

    if valid.sum() > len(RISK_COLS) + 30:
        Xreg = np.concatenate([np.ones((len(X), 1)), X], axis=1)

        try:
            beta = np.linalg.lstsq(Xreg[valid], y[valid], rcond=None)[0]
            resid = y - Xreg @ beta
        except Exception:
            resid = y
    else:
        resid = y

    resid = _safe_zscore(resid)
    out["score_processed"] = resid.astype(float)

    return out


def load_eval_forward_returns(
    start_date: str,
    end_date: str,
    cfg: Config = CFG
) -> pd.DataFrame:
    """
    为本地公榜评估构造下一期收益标签。
    """
    s0 = normalize_dt(start_date)
    s1 = normalize_dt(end_date)

    query_start = s0.strftime("%Y-%m-%d 00:00:00")
    query_end = (s1 + pd.Timedelta(days=cfg.eval_forward_extra_days)).strftime("%Y-%m-%d 23:59:59")

    exp = load_exposure(query_start, query_end)

    ret_next = exp[["date", "instrument", "ret"]].copy()
    ret_next = ret_next.sort_values(["instrument", "date"])

    ret_next["date"] = ret_next.groupby("instrument")["date"].shift(1)
    ret_next = ret_next.dropna(subset=["date"])
    ret_next = ret_next.rename(columns={"ret": "fwd_ret"})

    ret_next["date"] = pd.to_datetime(ret_next["date"]).dt.normalize()
    ret_next = ret_next[(ret_next["date"] >= s0) & (ret_next["date"] <= s1)]

    ret_next["instrument"] = ret_next["instrument"].astype(str)
    ret_next["fwd_ret"] = pd.to_numeric(ret_next["fwd_ret"], errors="coerce")
    ret_next = ret_next.dropna(subset=["fwd_ret"])

    del exp
    gc.collect()

    return ret_next[["date", "instrument", "fwd_ret"]]


def load_eval_risk_exposure(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    s0 = normalize_dt(start_date)
    s1 = normalize_dt(end_date)

    exp = load_exposure(
        s0.strftime("%Y-%m-%d 00:00:00"),
        s1.strftime("%Y-%m-%d 23:59:59"),
    )

    cols = ["date", "instrument"] + RISK_COLS

    exp = exp[cols].copy()
    exp["date"] = pd.to_datetime(exp["date"]).dt.normalize()
    exp["instrument"] = exp["instrument"].astype(str)

    return exp


def compute_local_public_score(
    submission: pd.DataFrame,
    start_date: str,
    end_date: str,
    cfg: Config = CFG
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    本地公榜近似打分。
    """
    required = ["date", "instrument", "score"]

    if list(submission.columns) != required:
        raise ValueError(f"submission 列必须严格为 {required}，当前为 {list(submission.columns)}")

    sub = submission.copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    sub["instrument"] = sub["instrument"].astype(str)
    sub["score"] = pd.to_numeric(sub["score"], errors="coerce")

    fwd = load_eval_forward_returns(start_date, end_date, cfg=cfg)
    risk = load_eval_risk_exposure(start_date, end_date)

    df = sub.merge(fwd, on=["date", "instrument"], how="inner")
    df = df.merge(risk, on=["date", "instrument"], how="left")

    del sub, fwd, risk
    gc.collect()

    df = df.dropna(subset=["score", "fwd_ret"])

    if df.empty:
        raise RuntimeError(
            "本地打分无有效行。"
            "可能原因：本地没有公榜区间 exposure/ret，或 submission 和 exposure 的日期/股票代码不匹配。"
        )

    processed_parts = []

    for _, g in tqdm(
        df.groupby("date", sort=True),
        desc="本地打分预处理",
        dynamic_ncols=True,
        leave=True,
    ):
        processed_parts.append(_platform_like_process_one_date(g))

    df = pd.concat(processed_parts, ignore_index=True)

    daily_rows = []
    ls_rows = []

    for dt, g in df.groupby("date", sort=True):
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["score_processed", "fwd_ret"])

        if len(g) < 50:
            continue

        x = g["score_processed"].astype(float)
        y = g["fwd_ret"].astype(float)

        if x.std() < 1e-12 or y.std() < 1e-12:
            continue

        ic = x.corr(y, method="pearson")
        rank_ic = x.corr(y, method="spearman")

        daily_rows.append({
            "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
            "n": int(len(g)),
            "IC": float(ic) if np.isfinite(ic) else np.nan,
            "RankIC": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
        })

        rank_pct = x.rank(pct=True, method="first")

        long_ret = y[rank_pct >= 0.90].mean()
        short_ret = y[rank_pct <= 0.10].mean()
        ls_ret = long_ret - short_ret

        ls_rows.append({
            "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
            "long_ret": float(long_ret),
            "short_ret": float(short_ret),
            "long_short_ret": float(ls_ret),
        })

    daily_ic_df = pd.DataFrame(daily_rows)
    long_short_df = pd.DataFrame(ls_rows)

    if daily_ic_df.empty:
        raise RuntimeError("本地打分无有效的日度 IC 行。")

    ic_series = daily_ic_df["IC"].dropna()
    rank_ic_series = daily_ic_df["RankIC"].dropna()
    ls_series = long_short_df["long_short_ret"].dropna()

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1))
    ic_ir = float(ic_mean / (ic_std + 1e-12))

    rank_ic_mean = float(rank_ic_series.mean())
    rank_ic_std = float(rank_ic_series.std(ddof=1))
    rank_ic_ir = float(rank_ic_mean / (rank_ic_std + 1e-12))

    ls_mean = float(ls_series.mean()) if len(ls_series) else np.nan
    ls_std = float(ls_series.std(ddof=1)) if len(ls_series) > 1 else np.nan
    long_short_sharpe = (
        float(np.sqrt(252) * ls_mean / (ls_std + 1e-12))
        if len(ls_series) > 1
        else np.nan
    )

    daily_ic_df["date_dt"] = pd.to_datetime(daily_ic_df["date"])
    daily_ic_df = daily_ic_df.sort_values("date_dt").reset_index(drop=True)

    stress_values = []

    if len(daily_ic_df) >= 20:
        chunks = np.array_split(daily_ic_df, 4)
        for c in chunks:
            if len(c) > 0:
                stress_values.append(float(c["IC"].mean()))

    stress_min_ic_mean = float(np.nanmin(stress_values)) if len(stress_values) else np.nan
    stress_avg_ic_mean = float(np.nanmean(stress_values)) if len(stress_values) else np.nan

    proxy = compute_local_proxy_score_from_metrics(
        ic_mean=ic_mean,
        ic_ir=ic_ir,
        long_short_sharpe=long_short_sharpe,
        stress_min_ic_mean=stress_min_ic_mean,
        cfg=cfg,
    )

    summary = {
        "start_date": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        "valid_days": int(len(daily_ic_df)),
        "valid_rows": int(len(df)),
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "IC_IR": ic_ir,
        "RankIC_mean": rank_ic_mean,
        "RankIC_std": rank_ic_std,
        "RankIC_IR": rank_ic_ir,
        "long_short_mean_daily_ret": ls_mean,
        "long_short_std_daily_ret": ls_std,
        "long_short_sharpe": long_short_sharpe,
        "stress_min_ic_mean": stress_min_ic_mean,
        "stress_avg_ic_mean": stress_avg_ic_mean,
    }

    summary.update(proxy)

    summary_df = pd.DataFrame([summary])
    daily_ic_df = daily_ic_df.drop(columns=["date_dt"])


    leaderboard_score_df = pd.DataFrame([{
        "local_proxy_score": summary["local_proxy_score"],
        "proxy_rank_ic_mean": summary["proxy_rank_ic_mean"],
        "proxy_rank_ic_ir": summary["proxy_rank_ic_ir"],
        "proxy_rank_sr": summary["proxy_rank_sr"],
        "proxy_rank_stress": summary["proxy_rank_stress"],
        "IC_mean": summary["IC_mean"],
        "IC_IR": summary["IC_IR"],
        "long_short_sharpe": summary["long_short_sharpe"],
        "stress_min_ic_mean": summary["stress_min_ic_mean"],
        "note": "仅为本地代理分数。官方分数需要全场排行榜百分位排名。"
    }])

    print("=" * 80)
    print("[Local Public Score Approximation]")

    for k, v in summary.items():
        print(f"{k:28s}: {v}")

    print("-" * 80)
    print(f"saved summary              : {cfg.public_local_score_path}")
    print(f"saved daily IC             : {cfg.public_daily_ic_path}")
    print(f"saved long-short           : {cfg.public_long_short_path}")
    print(f"saved local leaderboard    : {cfg.public_local_leaderboard_score_path}")
    print(f"LOCAL_PROXY_SCORE          : {summary['local_proxy_score']:.6f}")
    print("=" * 80)

    del df
    gc.collect()

    return summary_df, daily_ic_df, long_short_df


# ============================================================
# 12. 一键训练 + 本地公榜测试
# ============================================================

def run_public_test(
    data_source_name: str = None,
    train_start: str = None,
    train_end: str = None,
    public_start: str = None,
    public_end: str = None,
    force_train: bool = False,
    do_local_score: bool = True,
    cfg: Config = CFG
):
    """
    一键流程：
    1. 训练或加载权重
    2. 对公榜区间推理
    3. 保存 public_submission.csv
    4. 计算并保存本地近似公榜指标
    """
    data_source_arg = data_source_name or cfg.data_source
    train_data_source_name = _resolve_train_data_source(data_source_arg, cfg)
    infer_data_source_name = _resolve_infer_data_source(data_source_arg, cfg)
    train_start = train_start or cfg.train_start
    train_end = train_end or cfg.train_end
    public_start = public_start or cfg.public_start
    public_end = public_end or cfg.public_end

    print("=" * 80)
    print("[Run Public Test]")
    print(f"train_source : {train_data_source_name}")
    print(f"infer_source : {infer_data_source_name}")
    print(f"train range  : {train_start} -> {train_end}")
    print(f"public range : {public_start} -> {public_end}")
    print(f"force_train  : {force_train}")
    print(f"local score  : {do_local_score}")
    print("=" * 80)

    if force_train or (not os.path.exists(cfg.model_path)):
        train_and_save(
            data_source_name=train_data_source_name,
            train_start=train_start,
            train_end=train_end,
            cfg=cfg,
        )
    else:
        print(f"[Run Public Test] found checkpoint, skip training: {cfg.model_path}")

    sub = predict_scores(
        data_source_name=infer_data_source_name,
        start_date=public_start,
        end_date=public_end,
        model_path=cfg.model_path,
        cfg=cfg,
    )

    sub = sub[["date", "instrument", "score"]].copy()

    print("[Run Public Test] public submission generated in memory")
    print(sub.head())

    summary_df, daily_ic_df, long_short_df = None, None, None

    if do_local_score:
        try:
            summary_df, daily_ic_df, long_short_df = compute_local_public_score(
                submission=sub,
                start_date=public_start,
                end_date=public_end,
                cfg=cfg,
            )
        except Exception as e:
            print("[Run Public Test] local scoring failed, but submission file has been saved.")
            print(f"[Run Public Test] reason: {repr(e)}")

    return sub, summary_df, daily_ic_df, long_short_df


# ============================================================
# 13. 比赛评测入口
# ============================================================

def _here_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def _submission_model_path() -> str:
    return os.path.join(_here_dir(), SUBMISSION_MODEL_FILENAME)


def main(
    datasources=None,
    start_date: str = None,
    end_date: str = None,
    data_source_name: str = None,
) -> pd.DataFrame:
    """
    平台评测入口。返回 date/instrument/score 三列 DataFrame。

    提交版优先加载与本文件同目录的 train_patchtst_lite_best_submit_r6_09_d60_lowreg_bridge.json。
    若权重缺失，则使用 cfg 中的训练区间自动训练并保存同名 JSON。
    """
    cfg = CFG
    cfg.model_path = _submission_model_path()

    source_arg = datasources if datasources is not None else data_source_name
    train_source = _resolve_train_data_source(source_arg, cfg)
    infer_source = _resolve_infer_data_source(source_arg, cfg)

    start_date = start_date or cfg.public_start
    end_date = end_date or cfg.public_end

    print("=" * 80)
    print(f"[main] variant={cfg.variant_name}")
    print(f"[main] infer_source={infer_source}")
    print(f"[main] score range={start_date} -> {end_date}")
    print(f"[main] model_path={cfg.model_path}")
    print("=" * 80)

    if not os.path.exists(cfg.model_path):
        print(f"[main] checkpoint missing; auto-training: {cfg.model_path}")
        train_and_save(
            data_source_name=train_source,
            train_start=cfg.train_start,
            train_end=cfg.train_end,
            cfg=cfg,
        )

    sub = predict_scores(
        data_source_name=infer_source,
        start_date=start_date,
        end_date=end_date,
        model_path=cfg.model_path,
        cfg=cfg,
    )

    # ---------- 对齐中证 1000 + 规范输出 ----------
    if dai is not None:
        print("[main] 对齐中证 1000 (bigalpha_2026_instruments)...")
        stk = dai.query(
            "SELECT date, instrument FROM bigalpha_2026_instruments",
            filters={"date": [start_date, end_date]}
        ).df()

        sub["date"] = pd.to_datetime(sub["date"])
        stk["date"] = pd.to_datetime(stk["date"])
        sub["instrument"] = sub["instrument"].astype(str)
        stk["instrument"] = stk["instrument"].astype(str)

        sub = (
            pd.merge(sub, stk, on=["date", "instrument"], how="inner")
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["score"])
            .drop_duplicates(["date", "instrument"])
            .reset_index(drop=True)
        )

        sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
        print(f"[main] 对齐后输出行数: {len(sub)}")
    else:
        print("[main] 警告：未导入 dai 模块（可能在本地环境），跳过对齐中证 1000。")

    return sub[["date", "instrument", "score"]].copy()


# ============================================================
# 14. 本地训练入口
# ============================================================


# ============================================================
# 15. 本地训练入口
# ============================================================

if __name__ == "__main__":
    CFG.model_path = _submission_model_path()

    print("=" * 80)
    print("[Submit Config]")
    print(f"selected experiment : {SELECTED_EXPERIMENT_NAME}")
    print(f"feature count       : {len(FEATURE_COLS)}")
    print(f"model output        : {CFG.model_path}")
    print("hyperparameters     :")
    print(json.dumps(SUBMIT_HYPERPARAMS, ensure_ascii=False, indent=2))
    print("=" * 80)

    # 先实例化一次，立即校验输入维度、结构与比赛参数量限制。
    _check_model = build_model(CFG, input_dim=len(FEATURE_COLS))
    del _check_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_and_save(
        data_source_name=CFG.data_source,
        train_start=CFG.train_start,
        train_end=CFG.train_end,
        cfg=CFG,
    )
