import os
import sys
import gc
import json
import time
import math
import random

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    class _Logger:
        def info(self, msg, **kwargs):
            print(msg, kwargs)
    logger = _Logger()


# ============================================================
# 1. 路径与训练配置
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


def _resolve_existing_path(filename):
    candidates = [
        os.path.join(_HERE, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(_PROJECT_ROOT, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _normalize_date_arg(value):
    if value is None:
        return None
    return str(value)


TOKENIZER_JSON_PATH = _resolve_existing_path("kronos_tokenizer.json")
PREDICTOR_JSON_PATH = os.path.join(_HERE, "kronos_predictor.json")
SUMMARY_PATH = os.path.join(_HERE, "kronos_predictor_summary.json")

# 固定训练区间，不能使用平台注入的测试区间训练。
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31 23:59:59"
VAL_START, VAL_END = "2024-01-01", "2024-06-30 23:59:59"

# 内存敏感参数：先小规模跑通，再逐步放大。
EPOCHS = 5
BATCH = 512
ACCUMULATION_STEPS = 8

LR = 1e-3
WEIGHT_DECAY = 1e-2
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
SEED = 42

MAX_TRAIN_INSTRUMENTS = 50
INSTRUMENT_CHUNK_SIZE = 10
MAX_TRAIN_WINDOWS = 50000
MAX_INFER_INSTRUMENTS = None
MAX_INFER_WINDOWS = None
SAMPLE_STRIDE = 1        # 1 表示每个日末窗口都取；内存不够可改为 3、5、10
CLIP_VALUE = 5.0
LOG_INTERVAL = 20

# 是否使用 teacher forcing。
# False 更贴近原 train_predictor.py；
# True 训练更稳定，尤其是 s2 分支早期学习会更平滑。
TRAIN_USE_TEACHER_FORCING = False
EVAL_USE_TEACHER_FORCING = False

# Predictor 结构超参。
# s1_bits/s2_bits 会从 tokenizer json 中读取，这里只设置 predictor 自身规模。
PRED_D_MODEL = 32
PRED_N_HEADS = 4
PRED_N_LAYERS = 2
PRED_FF_DIM = 128
PRED_FFN_DROPOUT = 0.25
PRED_ATTN_DROPOUT = 0.1
PRED_RESID_DROPOUT = 0.25
PRED_TOKEN_DROPOUT = 0.1
PRED_LEARN_TE = True

DEFAULT_DATASOURCES = {"bar1m": "bigalpha_2026_stock_bar1m"}
SUBMISSION_COLUMNS = ["date", "instrument", "score"]


# ============================================================
# 2. 导入 Kronos 模型
# ============================================================

# 根据你的目录结构调整。
# 如果模型代码在 Kronos/model/kronos.py，则下面两行通常可用。
for _path in (_HERE, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.append(_path)

from kronos import KronosTokenizer, Kronos


# ============================================================
# 3. JSON 权重加载与保存
# ============================================================

def _dtype_to_name(dtype):
    return str(dtype).replace("torch.", "")


def _name_to_dtype(name):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype in JSON state_dict: {name}")
    return mapping[name]


def load_tokenizer_json(path, map_location="cpu"):
    """
    加载之前训练好的 kronos_tokenizer.json。
    该 JSON 应包含：
    - tokenizer_cfg
    - feature_cols
    - seq_len
    - stats: mean/std
    - state_dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model = KronosTokenizer(**payload["tokenizer_cfg"])

    state_dict = {}
    for k, obj in payload["state_dict"].items():
        dtype = _name_to_dtype(obj["dtype"])
        t = torch.tensor(obj["data"], dtype=dtype).reshape(obj["shape"])
        state_dict[k] = t.to(map_location)

    model.load_state_dict(state_dict, strict=True)
    model.to(map_location)

    mean = np.array(payload["stats"]["mean"], dtype=np.float32)
    std = np.array(payload["stats"]["std"], dtype=np.float32)

    return model, (mean, std), payload


def save_predictor_json(path, model, predictor_cfg, tokenizer_meta, best_val_loss, train_history):
    """
    保存 Kronos predictor 为 JSON。
    注意：JSON 比 .pt 更占空间。若平台允许，实际工程中 .pt 更合适。
    """
    state = {}
    for k, v in model.state_dict().items():
        vv = v.detach().cpu()
        state[k] = {
            "dtype": _dtype_to_name(vv.dtype),
            "shape": list(vv.shape),
            "data": vv.reshape(-1).tolist(),
        }

    payload = {
        "model_type": "KronosPredictor",
        "predictor_cfg": predictor_cfg,
        "tokenizer_json_path": os.path.basename(TOKENIZER_JSON_PATH),
        "feature_cols": tokenizer_meta["feature_cols"],
        "seq_len": tokenizer_meta["seq_len"],
        "tokenizer_cfg": tokenizer_meta["tokenizer_cfg"],
        "tokenizer_stats": tokenizer_meta["stats"],
        "best_val_loss": float(best_val_loss),
        "train_history": train_history,
        "state_dict": state,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Saved predictor JSON to: {path}")


def load_predictor_json(path, map_location="cpu"):
    """
    推理或继续训练时可用。
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model = Kronos(**payload["predictor_cfg"])

    sd = {}
    for k, obj in payload["state_dict"].items():
        dtype = _name_to_dtype(obj["dtype"])
        t = torch.tensor(obj["data"], dtype=dtype).reshape(obj["shape"])
        sd[k] = t.to(map_location)

    model.load_state_dict(sd, strict=True)
    model.to(map_location)
    return model, payload


def _to_float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def predicted_kline_to_score(pred_df, last_close, close_col="close", fallback=0.0):
    """
    把预测出的 K 线压成一个一日收益率分数。

    口径：score = pred_close_last / last_close - 1.0
    这里取预测 K 线最后一根 close 作为下一日价格代理。
    """
    if pred_df is None or len(pred_df) == 0 or close_col not in pred_df.columns:
        return float(fallback)

    base_close = _to_float_or_none(last_close)
    pred_close = _to_float_or_none(pred_df[close_col].iloc[-1])

    if base_close is None or base_close <= 0 or pred_close is None:
        return float(fallback)

    score = pred_close / base_close - 1.0
    return float(score) if np.isfinite(score) else float(fallback)


def build_score_dataframe(
    keys_df,
    scores,
    date_col=SUBMISSION_COLUMNS[0],
    instrument_col=SUBMISSION_COLUMNS[1],
    score_col=SUBMISSION_COLUMNS[2],
):
    """
    按官方要求组装提交结果表。

    输入 keys_df 推荐直接使用 build_predictor_dataset 返回的 keys，
    这样天然包含 date / instrument 两列。
    """
    if keys_df is None:
        raise ValueError("keys_df cannot be None")

    if isinstance(keys_df, pd.DataFrame):
        if date_col not in keys_df.columns or instrument_col not in keys_df.columns:
            raise ValueError(
                f"keys_df must contain columns [{date_col}, {instrument_col}]"
            )
        out_df = keys_df[[date_col, instrument_col]].copy()
    else:
        out_df = pd.DataFrame(keys_df, columns=[date_col, instrument_col])

    if np.isscalar(scores):
        score_list = [scores] * len(out_df)
    else:
        score_list = list(scores)

    if len(score_list) != len(out_df):
        raise ValueError(
            f"score length {len(score_list)} does not match keys length {len(out_df)}"
        )

    clean_scores = []
    for score in score_list:
        value = _to_float_or_none(score)
        clean_scores.append(float(value) if value is not None else float(0.0))

    out_df[score_col] = clean_scores
    return out_df[[date_col, instrument_col, score_col]]


# ============================================================
# 4. 数据构造：BigQuant 1m bar → X, X_stamp
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pool(sd, ed):
    """区间内中证1000成分股代码。"""
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return df["instrument"].tolist()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def calc_time_features(dt_series):
    """
    输出形状: [N, 5]
    对应 TemporalEmbedding 所需的:
    minute, hour, weekday, day, month
    """
    dt = pd.to_datetime(dt_series)

    if isinstance(dt, pd.Series):
        out = np.stack([
            dt.dt.minute.to_numpy(np.int64),
            dt.dt.hour.to_numpy(np.int64),
            dt.dt.weekday.to_numpy(np.int64),
            dt.dt.day.to_numpy(np.int64),
            dt.dt.month.to_numpy(np.int64),
        ], axis=1)
    else:
        idx = pd.DatetimeIndex(dt)
        out = np.stack([
            idx.minute.astype(np.int64),
            idx.hour.astype(np.int64),
            idx.weekday.astype(np.int64),
            idx.day.astype(np.int64),
            idx.month.astype(np.int64),
        ], axis=1)

    return out.astype(np.int64)


def build_predictor_dataset(
    table,
    sd,
    ed,
    instruments,
    feature_cols,
    seq_len,
    stats,
    max_windows,
    instrument_chunk_size=10,
    sample_stride=1,
    return_last_close=False,
):
    """
    构造 predictor 微调数据。

    返回：
    X:       [N, seq_len, n_feat]
    X_stamp: [N, seq_len, 5]

    训练逻辑：
    - tokenizer.encode(X, half=True)
    - token_in = token[:, :-1]
    - token_out = token[:, 1:]
    - stamp_in = X_stamp[:, :-1, :]
    """
    t0 = time.time()
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    buf = (sd_ts - pd.Timedelta(days=20)).strftime("%Y-%m-%d")

    mean, std = stats
    mean = mean.reshape(1, -1)
    std = std.reshape(1, -1)

    if len(feature_cols) != mean.shape[1]:
        raise ValueError(
            f"feature_cols length={len(feature_cols)}, stats dim={mean.shape[1]}，二者不一致。"
        )

    log_cols = [
        c for c in feature_cols
        if c in ["volume", "amount", "bid_volume1", "ask_volume1"]
    ]

    wins = []
    stamps = []
    keys = []
    last_closes = []
    close_idx = feature_cols.index("close") if "close" in feature_cols else None

    sql = (
        f"SELECT date, instrument, {', '.join(feature_cols)} "
        f"FROM {table} "
        f"ORDER BY instrument, date"
    )

    for ins_chunk in chunks(instruments, instrument_chunk_size):
        df = dai.query(
            sql,
            filters={"date": [buf, ed], "instrument": list(ins_chunk)},
        ).df()

        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"])

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in log_cols:
            df[c] = np.log1p(df[c].clip(lower=0))

        df = df.dropna(subset=feature_cols)

        for ins, sub in df.groupby("instrument", sort=False):
            sub = sub.sort_values("date")

            if len(sub) < seq_len:
                continue

            feats = sub[feature_cols].to_numpy(np.float32)
            time_feats = calc_time_features(sub["date"])

            day = sub["date"].dt.normalize().to_numpy()
            close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
            dates = day[close_pos]

            for k, p in enumerate(close_pos):
                if sample_stride > 1 and (k % sample_stride != 0):
                    continue

                d = pd.Timestamp(dates[k])

                if p + 1 < seq_len:
                    continue

                if d < sd_ts or d > ed_ts:
                    continue

                raw_x = feats[p - seq_len + 1: p + 1]
                raw_stamp = time_feats[p - seq_len + 1: p + 1]

                if not np.isfinite(raw_x).all():
                    continue

                if close_idx is None:
                    last_close = np.float32(np.nan)
                else:
                    last_close = np.float32(raw_x[-1, close_idx])

                x = ((raw_x - mean) / std).astype(np.float32)
                x = np.clip(x, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)

                wins.append(x)
                stamps.append(raw_stamp.astype(np.int64))
                keys.append((d, ins))
                last_closes.append(last_close)

                if max_windows is not None and len(wins) >= max_windows:
                    break

            if max_windows is not None and len(wins) >= max_windows:
                break

        del df
        gc.collect()

        if max_windows is not None and len(wins) >= max_windows:
            break

    if not wins:
        raise RuntimeError(f"build_predictor_dataset 无样本：{sd} ~ {ed}")

    X = np.stack(wins).astype(np.float32)
    X_stamp = np.stack(stamps).astype(np.int64)

    logger.info(
        "predictor 数据集构建完成",
        start=sd,
        end=ed,
        samples=len(wins),
        elapsed=round(time.time() - t0, 2),
    )

    # keys 可直接作为提交骨架，后续只需补 score 列即可。
    keys_df = pd.DataFrame(keys, columns=["date", "instrument"])
    if return_last_close:
        return X, X_stamp, keys_df, np.array(last_closes, dtype=np.float32)
    return X, X_stamp, keys_df


def make_dataloader(X, X_stamp, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(X_stamp),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=shuffle,
    )


def make_inference_dataloader(X, X_stamp, last_closes, batch_size):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(X_stamp),
        torch.from_numpy(np.asarray(last_closes, dtype=np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


# ============================================================
# 5. Predictor 训练与验证
# ============================================================

def build_predictor_cfg(tokenizer_payload):
    tok_cfg = tokenizer_payload["tokenizer_cfg"]

    return dict(
        s1_bits=int(tok_cfg["s1_bits"]),
        s2_bits=int(tok_cfg["s2_bits"]),
        n_layers=PRED_N_LAYERS,
        d_model=PRED_D_MODEL,
        n_heads=PRED_N_HEADS,
        ff_dim=PRED_FF_DIM,
        ffn_dropout_p=PRED_FFN_DROPOUT,
        attn_dropout_p=PRED_ATTN_DROPOUT,
        resid_dropout_p=PRED_RESID_DROPOUT,
        token_dropout_p=PRED_TOKEN_DROPOUT,
        learn_te=PRED_LEARN_TE,
    )


def train_one_epoch(
    model,
    tokenizer,
    loader,
    optimizer,
    scheduler,
    device,
    epoch_idx,
):
    model.train()
    tokenizer.eval()

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_loss_sum = 0.0
    total_s1_loss_sum = 0.0
    total_s2_loss_sum = 0.0
    total_count = 0

    optimizer.zero_grad(set_to_none=True)
    update_count = 0

    for step, (batch_x_cpu, batch_stamp_cpu) in enumerate(loader):
        batch_x = batch_x_cpu.to(device, non_blocking=False)
        batch_stamp = batch_stamp_cpu.to(device, non_blocking=False)

        # 1. 冻结 tokenizer，在线编码连续 K 线为 token。
        with torch.no_grad():
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

        # 2. 构造 next-token prediction。
        token_in = [
            token_seq_0[:, :-1],
            token_seq_1[:, :-1],
        ]
        token_out = [
            token_seq_0[:, 1:],
            token_seq_1[:, 1:],
        ]

        stamp_in = batch_stamp[:, :-1, :]

        # 3. Kronos predictor 前向。
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(
                token_in[0],
                token_in[1],
                stamp_in,
                use_teacher_forcing=TRAIN_USE_TEACHER_FORCING,
                s1_targets=token_out[0] if TRAIN_USE_TEACHER_FORCING else None,
            )

            loss, s1_loss, s2_loss = model.head.compute_loss(
                logits[0],
                logits[1],
                token_out[0],
                token_out[1],
            )

            loss_scaled = loss / ACCUMULATION_STEPS

        scaler.scale(loss_scaled).backward()

        n = batch_x.size(0)
        total_loss_sum += loss.item() * n
        total_s1_loss_sum += s1_loss.item() * n
        total_s2_loss_sum += s2_loss.item() * n
        total_count += n

        should_update = (
            ((step + 1) % ACCUMULATION_STEPS == 0)
            or ((step + 1) == len(loader))
        )

        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_count += 1

        if (step + 1) % LOG_INTERVAL == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch_idx + 1}/{EPOCHS}, Step {step + 1}/{len(loader)}] "
                f"lr={lr:.6g}, "
                f"loss={loss.item():.5f}, "
                f"s1={s1_loss.item():.5f}, "
                f"s2={s2_loss.item():.5f}"
            )

        del batch_x, batch_stamp
        del token_seq_0, token_seq_1, token_in, token_out, stamp_in, logits, loss, s1_loss, s2_loss, loss_scaled

        if device.type == "cuda" and (step + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return {
        "train_loss": total_loss_sum / max(total_count, 1),
        "train_s1_loss": total_s1_loss_sum / max(total_count, 1),
        "train_s2_loss": total_s2_loss_sum / max(total_count, 1),
        "optimizer_updates": update_count,
    }


@torch.no_grad()
def evaluate(model, tokenizer, loader, device):
    model.eval()
    tokenizer.eval()

    total_loss_sum = 0.0
    total_s1_loss_sum = 0.0
    total_s2_loss_sum = 0.0
    total_count = 0

    for batch_x_cpu, batch_stamp_cpu in loader:
        batch_x = batch_x_cpu.to(device, non_blocking=False)
        batch_stamp = batch_stamp_cpu.to(device, non_blocking=False)

        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

        token_in = [
            token_seq_0[:, :-1],
            token_seq_1[:, :-1],
        ]
        token_out = [
            token_seq_0[:, 1:],
            token_seq_1[:, 1:],
        ]

        stamp_in = batch_stamp[:, :-1, :]

        logits = model(
            token_in[0],
            token_in[1],
            stamp_in,
            use_teacher_forcing=EVAL_USE_TEACHER_FORCING,
            s1_targets=token_out[0] if EVAL_USE_TEACHER_FORCING else None,
        )

        loss, s1_loss, s2_loss = model.head.compute_loss(
            logits[0],
            logits[1],
            token_out[0],
            token_out[1],
        )

        n = batch_x.size(0)
        total_loss_sum += loss.item() * n
        total_s1_loss_sum += s1_loss.item() * n
        total_s2_loss_sum += s2_loss.item() * n
        total_count += n

        del batch_x, batch_stamp
        del token_seq_0, token_seq_1, token_in, token_out, stamp_in, logits, loss, s1_loss, s2_loss

    return {
        "val_loss": total_loss_sum / max(total_count, 1),
        "val_s1_loss": total_s1_loss_sum / max(total_count, 1),
        "val_s2_loss": total_s2_loss_sum / max(total_count, 1),
    }


@torch.no_grad()
def predict_score_dataframe(
    model,
    tokenizer,
    loader,
    keys_df,
    stats,
    feature_cols,
    device,
):
    """
    对推理/验证窗口产出官方要求的 date / instrument / score 表。

    score 使用下一根预测 K 线 close 相对窗口最后真实 close 的简单收益率。
    """
    if "close" not in feature_cols:
        raise ValueError("feature_cols must contain 'close' to build score")

    model.eval()
    tokenizer.eval()

    mean, std = stats
    close_idx = feature_cols.index("close")
    close_mean = float(mean[close_idx])
    close_std = float(std[close_idx])

    scores = []

    for batch_x_cpu, batch_stamp_cpu, last_close_cpu in loader:
        batch_x = batch_x_cpu.to(device, non_blocking=False)
        batch_stamp = batch_stamp_cpu.to(device, non_blocking=False)

        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

        s1_logits, context = model.decode_s1(
            token_seq_0,
            token_seq_1,
            batch_stamp,
        )
        s1_ids = torch.argmax(s1_logits, dim=-1)
        next_s1 = s1_ids[:, -1:]

        s2_logits = model.decode_s2(context, next_s1)
        next_s2 = torch.argmax(s2_logits[:, -1:, :], dim=-1)

        full_tokens = [
            torch.cat([token_seq_0, next_s1], dim=1),
            torch.cat([token_seq_1, next_s2], dim=1),
        ]
        decoded = tokenizer.decode(full_tokens, half=True)

        pred_close = decoded[:, -1, close_idx].detach().cpu().numpy()
        pred_close = pred_close * close_std + close_mean

        last_close = last_close_cpu.detach().cpu().numpy()
        for pred_value, base_value in zip(pred_close, last_close):
            base_value = _to_float_or_none(base_value)
            pred_value = _to_float_or_none(pred_value)
            if base_value is None or base_value <= 0 or pred_value is None:
                scores.append(float(0.0))
                continue

            score = pred_value / base_value - 1.0
            scores.append(float(score) if np.isfinite(score) else float(0.0))

        del batch_x, batch_stamp
        del token_seq_0, token_seq_1, s1_logits, context, s1_ids, s2_logits
        del next_s1, next_s2, full_tokens, decoded

    return build_score_dataframe(keys_df, scores)


# ============================================================
# 6. 训练入口
# ============================================================

def train_and_save(
    datasources,
    model_path=PREDICTOR_JSON_PATH,
    summary_path=SUMMARY_PATH,
):
    """
    在写死的训练区间上从零训练并保存模型。
    """
    if datasources is None:
        datasources = DEFAULT_DATASOURCES
    else:
        datasources = dict(datasources)

    set_seed(SEED)

    device = get_device()
    print(f"Using device: {device}")

    # 1. 加载之前训练好的 tokenizer。
    tokenizer, tokenizer_stats, tokenizer_payload = load_tokenizer_json(
        TOKENIZER_JSON_PATH,
        map_location=device,
    )
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    feature_cols = tokenizer_payload["feature_cols"]
    seq_len = int(tokenizer_payload["seq_len"])

    print(f"Loaded tokenizer from: {TOKENIZER_JSON_PATH}")
    print(f"feature_cols={feature_cols}")
    print(f"seq_len={seq_len}")

    # 2. 构造 predictor。
    predictor_cfg = build_predictor_cfg(tokenizer_payload)

    # 基本维度检查。
    if predictor_cfg["d_model"] % predictor_cfg["n_heads"] != 0:
        raise ValueError("PRED_D_MODEL 必须能被 PRED_N_HEADS 整除。")

    if (predictor_cfg["d_model"] // predictor_cfg["n_heads"]) % 2 != 0:
        raise ValueError("head_dim 必须是偶数，否则 RoPE 可能出错。")

    model = Kronos(**predictor_cfg)
    model.to(device)

    print(f"Predictor cfg: {predictor_cfg}")

    # 3. 构造 BigQuant 数据集。
    table = datasources["bar1m"]

    train_instruments = pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS]

    Xtr, Str, _ = build_predictor_dataset(
        table=table,
        sd=TRAIN_START,
        ed=TRAIN_END,
        instruments=train_instruments,
        feature_cols=feature_cols,
        seq_len=seq_len,
        stats=tokenizer_stats,
        max_windows=MAX_TRAIN_WINDOWS,
        instrument_chunk_size=INSTRUMENT_CHUNK_SIZE,
        sample_stride=SAMPLE_STRIDE,
    )

    val_instruments = pool(VAL_START, VAL_END)
    Xva, Sva, _ = build_predictor_dataset(
        table=table,
        sd=VAL_START,
        ed=VAL_END,
        instruments=val_instruments,
        feature_cols=feature_cols,
        seq_len=seq_len,
        stats=tokenizer_stats,
        max_windows=MAX_INFER_WINDOWS,
        instrument_chunk_size=INSTRUMENT_CHUNK_SIZE,
        sample_stride=SAMPLE_STRIDE,
    )

    print(f"Train samples: {len(Xtr)}, Val samples: {len(Xva)}")

    train_loader = make_dataloader(Xtr, Str, BATCH, shuffle=True)
    val_loader = make_dataloader(Xva, Sva, BATCH, shuffle=False)

    # 4. 优化器与学习率调度。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / ACCUMULATION_STEPS))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=0.03,
        div_factor=10,
    )

    # 5. 训练。
    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    start_time = time.time()

    for epoch_idx in range(EPOCHS):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch_idx=epoch_idx,
        )

        val_metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            loader=val_loader,
            device=device,
        )

        row = {
            "epoch": epoch_idx + 1,
            **train_metrics,
            **val_metrics,
            "epoch_elapsed": round(time.time() - epoch_start, 2),
        }
        history.append(row)

        print(
            f"\n--- Epoch {epoch_idx + 1}/{EPOCHS} Summary ---\n"
            f"train_loss={row['train_loss']:.6f}, "
            f"train_s1={row['train_s1_loss']:.6f}, "
            f"train_s2={row['train_s2_loss']:.6f}\n"
            f"val_loss={row['val_loss']:.6f}, "
            f"val_s1={row['val_s1_loss']:.6f}, "
            f"val_s2={row['val_s2_loss']:.6f}\n"
            f"elapsed={row['epoch_elapsed']}s\n"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch_idx + 1

            save_predictor_json(
                path=model_path,
                model=model,
                predictor_cfg=predictor_cfg,
                tokenizer_meta=tokenizer_payload,
                best_val_loss=best_val_loss,
                train_history=history,
            )

            print(
                f"Best predictor updated: epoch={best_epoch}, "
                f"val_loss={best_val_loss:.6f}"
            )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "predictor_json_path": model_path,
        "tokenizer_json_path": TOKENIZER_JSON_PATH,
        "total_elapsed": round(time.time() - start_time, 2),
        "history": history,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Training finished. best_epoch={best_epoch}, "
        f"best_val_loss={best_val_loss:.6f}"
    )
    print(f"Summary saved to: {summary_path}")

    return model_path


def main(datasources=None, start_date=None, end_date=None):
    if datasources is None:
        datasources = DEFAULT_DATASOURCES
    return train_and_save(datasources)


if __name__ == "__main__":
    main()
