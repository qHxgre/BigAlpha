"""
Mamba 端到端股票预测 — 训练脚本

训练产物统一存为 JSON 文本文件（平台只允许上传文本类文件），
state_dict 里每个张量转成 {dtype, shape, data(扁平 list)}。
推理端用 load_model_json 读回，按 dtype/shape 还原张量。
"""
import os
import json
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import pearsonr

from data_pipeline import DataPipeline, query_all_freqs
from config_mamba import (
    MambaConfig, set_seed, SEED,
    SELECTED_FIELDS, set_active_freqs, NUM_FIELDS
)
from mamba_model import MambaStockModel, build_model, rank_ic_loss, combined_loss

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "mamba_model.json")


# ============================================================
# 模型存/读: 一律用 JSON 文本文件, 不使用 .pt / .pkl 等二进制
# ============================================================
def save_model_json(ckpt, model_path=MODEL_PATH):
    """把 checkpoint 存成 JSON 文本文件。

    state_dict 里每个张量转成 {dtype, shape, data(扁平 list)},
    其余字段 (结构超参 / 归一化统计等) 原样写入。
    """
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),
        }
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model_json(model_path=MODEL_PATH, map_location="cpu"):
    """读取 save_model_json 写出的 JSON, 把 state_dict 还原为张量 dict。

    返回结构: {"state_dict": {name: Tensor}, "config": {...}, ...}
    """
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ============================================================
# 1. 训练数据构建
# ============================================================
def build_mamba_input(pipeline, indexed, instruments, date_idx, config):
    """
    从 indexed 数据构建 Mamba 模型输入

    输入: B 只股票, 回看 D 天, 每天最多 T 根 bar, 每 bar F 个字段
    输出: (B, D*T, F) 并填充缺失为 0
    """
    B = len(instruments)
    D = config.lookback_days
    T = config.bars_per_day
    F = config.n_fields
    S = D * T  # 总序列长度 = 3600

    X = np.zeros((B, S, F), dtype=np.float32)
    valid = np.zeros(B, dtype=np.bool_)

    # 确定回看日期范围
    lookback_start = max(0, date_idx - D + 1)
    lookback_dates = pipeline.all_trading_days[lookback_start:date_idx + 1]
    lookback_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in lookback_dates]
    offset = D - len(lookback_dates)

    freq = config.freq_names[0]
    store = indexed[freq]

    for i, inst in enumerate(instruments):
        inst_dict = store.get(inst)
        if inst_dict is None:
            continue
        has_data = False
        for d_idx, ds in enumerate(lookback_strs):
            out_day = offset + d_idx
            if out_day < 0 or out_day >= D:
                continue
            tup = inst_dict.get(ds)
            if tup is None:
                continue
            tensor, mask = tup  # tensor: (1, T_max, F), mask: (1, T_max)
            if tensor.shape[0] == 0:
                continue
            nb = min(tensor.shape[1], T)
            day_start = out_day * T
            X[i, day_start:day_start + nb, :] = tensor[0, :nb, :]
            has_data = True
        valid[i] = has_data

    return X, valid


def build_labels(pipeline, instruments, date_idx, horizon=1):
    """构建 T+N 日收益率标签 (复用 train.py 逻辑)"""
    import dai
    if date_idx + horizon >= len(pipeline.all_trading_days):
        return None, None

    td = pipeline.all_trading_days[date_idx + horizon]
    td_str = pd.to_datetime(td).strftime("%Y-%m-%d")
    insts = "', '".join(instruments)

    try:
        df = dai.query(
            f"""
            SELECT instrument,
                   (LAST(close) - FIRST(open)) / NULLIF(FIRST(open), 0) AS ret
            FROM bigalpha_2026_stock_bar1m
            WHERE instrument IN ('{insts}')
            GROUP BY instrument, date::DATE
            """,
            filters={"date": [f"{td_str} 09:00:00", f"{td_str} 17:00:00"]},
            compression=True,
        ).df()
        d = dict(zip(df["instrument"], df["ret"]))
        labels = np.array([d.get(inst, np.nan) for inst in instruments], dtype=np.float32)
        valid = ~np.isnan(labels)
        return labels, valid
    except Exception:
        return None, None


# ============================================================
# 2. 验证函数
# ============================================================
@torch.no_grad()
def validate(model, pipeline, all_instruments, val_di_sample, config):
    """在验证日上计算 IC"""
    device = config.device
    model.eval()
    daily_ics = []

    if not val_di_sample:
        model.train()
        return {"IC_mean": 0.0, "IC_IR": 0.0, "n": 0}

    # 加载验证数据
    min_di = min(val_di_sample)
    max_di = max(val_di_sample)
    load_start = pipeline.all_trading_days[max(0, min_di - config.lookback_days - 3)]
    load_end = pipeline.all_trading_days[max_di]

    try:
        raw = query_all_freqs(
            config.freq_names,
            load_start.strftime("%Y-%m-%d"),
            load_end.strftime("%Y-%m-%d"),
        )
        indexed = pipeline.index_data(raw)
        del raw; gc.collect()
    except Exception as e:
        print(f"  ⚠ val 查询失败: {e}")
        model.train()
        return {"IC_mean": 0.0, "IC_IR": 0.0, "n": 0}

    freq0 = config.freq_names[0]

    for di in val_di_sample:
        ds = pipeline.all_trading_days[di].strftime("%Y-%m-%d")

        available = [inst for inst in all_instruments
                     if inst in indexed[freq0] and ds in indexed[freq0][inst]]
        if len(available) > 500:
            available = np.random.choice(available, 500, replace=False).tolist()
        if len(available) < 30:
            continue

        try:
            X, feat_valid = build_mamba_input(pipeline, indexed, available, di, config)
            labels, label_valid = build_labels(pipeline, available, di, config.forward_horizon)
        except Exception:
            continue

        if labels is None or label_valid.sum() < 10:
            continue
        mask = feat_valid & label_valid
        if mask.sum() < 10:
            continue

        X_t = torch.from_numpy(X[mask]).to(device)
        scores = model(X_t).cpu().numpy()
        # 过滤 NaN/inf
        valid_pred = np.isfinite(scores) & np.isfinite(labels[mask])
        if valid_pred.sum() < 10:
            continue
        ic, _ = pearsonr(scores[valid_pred], labels[mask][valid_pred])
        if not np.isnan(ic):
            daily_ics.append(ic)

    del indexed; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    if not daily_ics:
        return {"IC_mean": 0.0, "IC_IR": 0.0, "n": 0}
    a = np.array(daily_ics)
    return {"IC_mean": float(a.mean()), "IC_IR": float(a.mean() / (a.std() + 1e-8)), "n": len(a)}


# ============================================================
# 3. 训练主函数
# ============================================================
def train(config=None):
    if config is None:
        config = MambaConfig()

    set_seed(config.seed)
    set_active_freqs(config.freq_names)
    device = config.device

    # CUDA 确定性 (确保可复现)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"设备: {device}")
    print(f"Mamba 模型: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"d_state={config.d_state}")
    print(f"序列长度: {config.seq_len} ({config.lookback_days}天 × {config.bars_per_day}分钟)")
    print(f"字段数: {config.n_fields}")

    # ---- 构建模型 ----
    model, n_params = build_model(config, verbose=True)
    model = model.to(device)
    print(f"设备确认: {next(model.parameters()).device}")

    # ---- 数据管道 ----
    print("\n" + "=" * 60)
    print("初始化数据管道")
    print("=" * 60)
    pipeline = DataPipeline(
        freq_names=config.freq_names,
        lookback_days=config.lookback_days,
        normalization=config.normalization,
        use_log_volume=config.use_log_volume,
    )
    pipeline.fit(config.train_start, config.train_end)
    print(f"交易日历: {len(pipeline.all_trading_days)} 天")

    # ---- 数据划分 ----
    # 全量训练集: 2019-2024, 最后60天做验证
    all_trading_days = pipeline.all_trading_days
    train_dates = [d for d in all_trading_days if d <= pd.Timestamp("2024-09-30")]
    val_dates = [d for d in all_trading_days if pd.Timestamp("2024-10-01") <= d <= pd.Timestamp("2024-12-31")]

    train_di = [pipeline.day_to_idx[d] for d in train_dates if d in pipeline.day_to_idx]
    val_di = [pipeline.day_to_idx[d] for d in val_dates if d in pipeline.day_to_idx]
    valid_train_di = [di for di in train_di if di >= pipeline.lookback_days]
    print(f"训练: {len(valid_train_di)} 天 | 验证: {len(val_di)} 天")

    # ---- 股票列表 ----
    import dai
    inst_df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_stock_bar1m ORDER BY instrument",
        filters={"date": [f"{config.train_start} 00:00:00",
                          f"{config.train_end} 23:59:59"]},
        compression=True,
    ).df()
    all_instruments = sorted(inst_df["instrument"].unique().tolist())
    print(f"股票: {len(all_instruments)} 只")

    # ---- 优化器 & 调度器 ----
    optimizer = AdamW(model.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ---- 损失函数 ----
    if config.loss_type == "ic":
        loss_fn = rank_ic_loss
    elif config.loss_type == "mse":
        loss_fn = lambda p, l, m: nn.functional.mse_loss(p[m], l[m])
    else:
        loss_fn = lambda p, l, m: combined_loss(p, l, m, config.loss_alpha)

    # ---- 训练循环 ----
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    best_ic = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"loss": [], "val_ic": [], "val_ir": []}

    n_val = min(config.val_days, len(val_di))

    for epoch in range(config.max_epochs):
        t0 = time.time()

        # 随机采样训练日
        max_start = len(valid_train_di) - config.epoch_days
        if max_start <= 0:
            sampled_di = valid_train_di
        else:
            start_pos = np.random.randint(0, max_start)
            sampled_di = valid_train_di[start_pos:start_pos + config.epoch_days]

        d0 = all_trading_days[sampled_di[0]].strftime('%Y-%m-%d')
        d1 = all_trading_days[sampled_di[-1]].strftime('%Y-%m-%d')
        print(f"\n[Epoch {epoch+1}] {len(sampled_di)}天 ({d0} ~ {d1})")

        # 加载数据
        raw = query_all_freqs(
            config.freq_names,
            (all_trading_days[max(0, sampled_di[0] - config.lookback_days - 3)]
             ).strftime("%Y-%m-%d"),
            all_trading_days[sampled_di[-1]].strftime("%Y-%m-%d"),
        )
        indexed = pipeline.index_data(raw)
        del raw; gc.collect()

        # 逐日训练
        epoch_losses = []
        freq0 = config.freq_names[0]
        n_days_trained = 0

        for di in sampled_di:
            ds = all_trading_days[di].strftime("%Y-%m-%d")

            day_insts = [inst for inst in all_instruments
                         if inst in indexed[freq0] and ds in indexed[freq0][inst]]
            if len(day_insts) < 50:
                continue
            if len(day_insts) > config.max_stocks_per_day:
                day_insts = np.random.choice(
                    day_insts, config.max_stocks_per_day, replace=False).tolist()

            X, feat_valid = build_mamba_input(pipeline, indexed, day_insts, di, config)
            labels, label_valid = build_labels(pipeline, day_insts, di, config.forward_horizon)
            if labels is None:
                continue

            mask = feat_valid & label_valid
            if mask.sum() < 30:
                continue

            X_t = torch.from_numpy(X[mask]).to(device)
            y_t = torch.from_numpy(labels[mask]).to(device)
            m_t = torch.ones(len(y_t), dtype=torch.bool, device=device)

            # 梯度累积
            micro_batch = config.batch_size
            optimizer.zero_grad()
            day_loss = 0.0
            n_valid_mb = 0

            for mb in range(0, len(X_t), micro_batch):
                x_mb = X_t[mb:mb + micro_batch]
                y_mb = y_t[mb:mb + micro_batch]
                m_mb = m_t[mb:mb + micro_batch]

                preds = model(x_mb)
                loss = loss_fn(preds, y_mb, m_mb)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                day_loss += loss.item()
                n_valid_mb += 1

            if n_valid_mb == 0:
                continue  # 全天 NaN, 跳过
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(day_loss / n_valid_mb)
            n_days_trained += 1

        del indexed; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()

        # ---- 验证 ----
        np.random.shuffle(val_di)
        val_sample = val_di[:n_val]
        val_metrics = validate(model, pipeline, all_instruments, val_sample, config)

        et = time.time() - t0
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0

        print(f"[Epoch {epoch+1}] {et:.0f}s ({et/60:.1f}min) | "
              f"Loss={avg_loss:.4f} | {n_days_trained}天训练")
        print(f"  Val IC_mean={val_metrics['IC_mean']:.4f} "
              f"IC_IR={val_metrics['IC_IR']:.4f} n={val_metrics['n']}")

        history["loss"].append(avg_loss)
        history["val_ic"].append(val_metrics["IC_mean"])
        history["val_ir"].append(val_metrics["IC_IR"])

        # ---- 早停 & 保存 ----
        if abs(val_metrics["IC_mean"]) > abs(best_ic):
            best_ic = val_metrics["IC_mean"]
            best_epoch = epoch + 1
            patience_counter = 0

            val_sign = 1 if val_metrics["IC_mean"] >= 0 else -1
            save_model_json({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": {
                    "d_model": config.d_model,
                    "n_layers": config.n_layers,
                    "d_state": config.d_state,
                    "n_fields": config.n_fields,
                    "seq_len": config.seq_len,
                    "lookback_days": config.lookback_days,
                    "bars_per_day": config.bars_per_day,
                    "patch_len": config.patch_len,
                    "patch_stride": config.patch_stride,
                    "freq_names": config.freq_names,
                    "forward_horizon": config.forward_horizon,
                    "seed": config.seed,
                },
                "val_metrics": val_metrics,
                "val_sign": val_sign,
            })
            print(f"  >> 最佳模型已保存 (IC_mean={best_ic:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config.early_stopping_patience} "
                  f"(best={best_ic:.4f} @ epoch {best_epoch})")
            if patience_counter >= config.early_stopping_patience:
                print(f"\n早停! 最佳 IC_mean={best_ic:.4f} @ epoch {best_epoch}")
                break

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳: epoch {best_epoch}, IC_mean={best_ic:.4f}")
    for i, (ic, ir) in enumerate(zip(history["val_ic"], history["val_ir"])):
        print(f"  Epoch {i+1}: Val IC={ic:.4f} IR={ir:.4f}")
    print(f">> {MODEL_PATH} 已保存 <<")
    return model, pipeline


if __name__ == "__main__":
    config = MambaConfig()
    print(f"Mamba 端到端模型 | {config.freq_names}")
    print(f"参数量预计 ~750K | 序列长度 {config.seq_len}")
    model, pipeline = train(config)
