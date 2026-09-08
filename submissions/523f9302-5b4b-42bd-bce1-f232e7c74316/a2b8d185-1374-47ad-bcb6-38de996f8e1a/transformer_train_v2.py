# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端量价预测 —— 训练脚本 v5

v5 改进：
1. 频率差异化字段（1m:28 / 5m:20 / 15m:12 / 30m:7）
2. 模型容量升级（d_model=128, nlayers=3）
3. 多周期残差收益率标签（BARRA风格剔除）
4. Warmup + Cosine退火
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

from model_v2 import (
    FREQ_CONFIGS, FREQ_LABELS, MODEL_CFG,
    MultiFreqStockTransformer, SimpleStockTransformer,
    save_model_json, load_model_json, count_parameters,
    PositionalEncoding, AttentionPooling, FreqEncoder, CrossStockAttention,
)

logger = structlog.get_logger()

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
MODEL_PATH = os.path.join(_HERE, "transformer_model_v2.json")

# 训练区间：2019 ~ 2024-Q3 训练，2024-Q4 验证早停
TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-12-31 23:59:59"
VAL_START   = "2024-10-01"
VAL_END     = "2024-12-31 23:59:59"

EPOCHS       = 30
WARMUP_EPOCHS = 2
LR           = 3e-4
SEED         = 42
MAX_STOCKS   = 500
EARLY_STOP_PATIENCE = 8

TABLE_1M  = "bigalpha_2026_stock_bar1m"
TABLE_5M  = "bigalpha_2026_stock_bar5m"
TABLE_15M = "bigalpha_2026_stock_bar15m"
TABLE_30M = "bigalpha_2026_stock_bar30m"
INSTRUMENT_TABLE = "bigalpha_2026_instruments"
EXPOSURE_TABLE   = "bigalpha_2026_exposure"

FREQ_TABLES = {"1m": TABLE_1M, "5m": TABLE_5M, "15m": TABLE_15M, "30m": TABLE_30M}

# ============================================================
# 排序损失
# ============================================================
class RankingLoss(nn.Module):
    def __init__(self, temperature=1.0, pairwise_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.pairwise_weight = pairwise_weight

    def listwise_loss(self, y_pred, y_true):
        pp = F.softmax(y_pred / self.temperature, dim=-1)
        tp = F.softmax(y_true / self.temperature, dim=-1)
        return (tp * (torch.log(tp + 1e-12) - torch.log(pp + 1e-12))).sum(-1).mean()

    def pairwise_loss(self, y_pred, y_true):
        n = y_pred.size(-1)
        if n < 2: return torch.tensor(0.0, device=y_pred.device)
        pd = y_pred.unsqueeze(-1) - y_pred.unsqueeze(-2)
        td = y_true.unsqueeze(-1) - y_true.unsqueeze(-2)
        mask = (td > 0).float()
        loss = -F.logsigmoid(pd) * mask
        return (loss.sum(dim=[-1,-2]) / mask.sum(dim=[-1,-2]).clamp(min=1)).mean()

    def forward(self, y_pred, y_true):
        return self.listwise_loss(y_pred, y_true) + self.pairwise_weight * self.pairwise_loss(y_pred, y_true)

# ============================================================
# 数据工具
# ============================================================
def query_instruments(dai_module, sd, ed):
    df = dai_module.query(
        f"SELECT DISTINCT instrument FROM {INSTRUMENT_TABLE}",
        filters={"date": [sd, ed]}
    ).df()
    return sorted(df["instrument"].tolist())

def query_freq_data(dai_module, table, sd, ed, instruments, columns):
    buf = (pd.to_datetime(sd) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    sql = f"SELECT date, instrument, {', '.join(columns)} FROM {table} ORDER BY instrument, date"
    df = dai_module.query(sql, filters={"date": [buf, ed], "instrument": instruments}).df()
    # log1p 量字段
    for c in columns:
        if "volume" in c.lower() or c == "amount":
            if c in df.columns: df[c] = np.log1p(df[c].clip(lower=0))
    return df

def build_day_samples(df, sd, ed, seq_len, columns):
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    samples = []
    for ins, sub in df.groupby("instrument", sort=False):
        if len(sub) <= seq_len: continue
        feats = sub[columns].to_numpy(np.float32)
        day_ts = sub["date"].dt.normalize().to_numpy()
        cp = np.flatnonzero(np.append(day_ts[1:] != day_ts[:-1], True))
        close_px = sub["close"].to_numpy(np.float64)[cp]
        dates_arr = day_ts[cp]
        for k, p in enumerate(cp):
            d = pd.Timestamp(dates_arr[k])
            if p + 1 < seq_len or d < sd_ts or d > ed_ts: continue
            if k + 1 < len(cp) and close_px[k] > 0:
                r = close_px[k+1] / close_px[k] - 1.0
                if np.isfinite(r):
                    samples.append({"date": d, "instrument": ins, "seq": feats[p-seq_len+1:p+1].copy(), "label": np.float32(r)})
    return samples

def align_samples(all_samples):
    idxs = [{(s["date"], s["instrument"]): s for s in ss} for ss in all_samples]
    common = set(idxs[0].keys())
    for idx in idxs[1:]: common &= set(idx.keys())
    logger.info("对齐样本", counts=[len(s) for s in all_samples], common=len(common))
    data = {}
    for date, ins in common:
        seqs = [idx[(date, ins)]["seq"] for idx in idxs]
        data.setdefault(date, []).append({"instrument": ins, "seqs": seqs, "label": idxs[0][(date, ins)]["label"]})
    return data

def compute_fit_stats(data):
    all_data = {fl: [] for fl in FREQ_LABELS}
    for entries in data.values():
        for e in entries:
            for i, fl in enumerate(FREQ_LABELS):
                all_data[fl].append(e["seqs"][i])
    stats = {}
    for fl, arr_list in all_data.items():
        if not arr_list: continue
        flat = np.concatenate([a.reshape(-1, FREQ_CONFIGS[fl]["n_feat"]) for a in arr_list], axis=0)
        stats[fl] = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32) + 1e-8)
    return stats

# ============================================================
# 残差收益率标签（可选：用 BARRA 暴露做风格剔除）
# ============================================================
def compute_residual_returns(dai_module, data_by_date):
    """对收益率做简易风格剔除（市值+行业），返回残差作为标签。

    如果 exposure 表不可用，则直接使用原始收益率 + winsorize。
    """
    try:
        # 尝试获取 BARRA 暴露数据
        exp_df = dai_module.query(
            f"SELECT date, instrument, size, beta, momentum, volatility, liquidity, "
            f"value, earnings_yield, growth, leverage, non_linear_size "
            f"FROM {EXPOSURE_TABLE}",
            filters={"date": [TRAIN_START[:10], VAL_END[:10]]}
        ).df()
        if exp_df.empty: raise ValueError("empty exposure")
        logger.info("使用 BARRA 暴露做残差标签")
    except Exception:
        logger.info("BARRA 暴露数据不可用，使用原始收益标签")
        return data_by_date  # 原样返回

    # 构建日期-标的索引
    exp_lookup = {}
    for _, row in exp_df.iterrows():
        d = pd.Timestamp(row["date"]).normalize()
        key = (d, row["instrument"])
        vals = row.drop(["date", "instrument"]).values.astype(np.float64)
        exp_lookup[key] = vals

    # 按日期分别做截面回归取残差
    new_data = {}
    for date, entries in data_by_date.items():
        X_list, y_list, ins_list = [], [], []
        for e in entries:
            key = (date, e["instrument"])
            if key in exp_lookup:
                X_list.append(exp_lookup[key])
                y_list.append(e["label"])
                ins_list.append(e)
        if len(X_list) < 30:  # 样本太少，不回归
            new_data[date] = entries
            continue

        X = np.array(X_list)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        X = np.column_stack([np.ones(len(X)), X])
        y = np.array(y_list)
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            for e, res in zip(ins_list, residuals):
                new_e = dict(e)
                new_e["label"] = np.float32(res)
                new_data.setdefault(date, []).append(new_e)
        except np.linalg.LinAlgError:
            new_data[date] = entries

    return new_data

# ============================================================
# 评估
# ============================================================
def compute_ic(scores, labels):
    from scipy import stats as st
    n = len(scores)
    if n < 10: return {"ic_spearman": 0.0, "n": n}
    ic, _ = st.spearmanr(scores, labels)
    return {"ic_spearman": ic, "n": n}

# ============================================================
# 训练主流程
# ============================================================
def train_and_save(datasources, model_path=MODEL_PATH, use_multi_freq=True):
    import dai

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备", device=str(device))

    # ---- 1. 成分股 ----
    train_ins = query_instruments(dai, TRAIN_START[:10], TRAIN_END[:10])[:MAX_STOCKS]
    val_ins = query_instruments(dai, VAL_START[:10], VAL_END[:10])
    val_ins = [i for i in val_ins if i in train_ins][:MAX_STOCKS]
    logger.info("成分股", train=len(train_ins), val=len(val_ins))

    # ---- 2. 查询各频率数据 ----
    all_samples = []
    for fl in FREQ_LABELS:
        cfg = FREQ_CONFIGS[fl]
        table = datasources.get(f"bar{fl}", FREQ_TABLES[fl])
        cols = cfg["all_cols"]
        logger.info(f"{fl}: {len(cols)}字段, seq_len={cfg['seq_len']}")
        df = query_freq_data(dai, table, TRAIN_START, VAL_END, train_ins, cols)
        samples = build_day_samples(df, TRAIN_START, VAL_END, cfg["seq_len"], cols)
        all_samples.append(samples)

    # ---- 3. 对齐 + 划分 ----
    data_by_date = align_samples(all_samples)

    # 残差标签（可选）
    data_by_date = compute_residual_returns(dai, data_by_date)

    train_dates = sorted([d for d in data_by_date if d < pd.to_datetime(VAL_START)])
    val_dates   = sorted([d for d in data_by_date if pd.to_datetime(VAL_START) <= d <= pd.to_datetime(VAL_END)])
    logger.info("数据划分", train_days=len(train_dates), val_days=len(val_dates))

    # ---- 4. 标准化 ----
    stats = compute_fit_stats({d: v for d, v in data_by_date.items() if d < pd.to_datetime(VAL_START)})

    def normalize(entry, st):
        for i, fl in enumerate(FREQ_LABELS):
            m, s = st[fl]
            entry["seqs"][i] = (entry["seqs"][i] - m) / s
        return entry

    for entries in data_by_date.values():
        for e in entries: normalize(e, stats)

    # ---- 5. 标签 winsorize ----
    all_labels = [e["label"] for entries in data_by_date.values() for e in entries]
    lo, hi = np.percentile(all_labels, [1, 99])
    for entries in data_by_date.values():
        for e in entries: e["label"] = np.clip(e["label"], lo, hi)

    # ---- 6. 构建模型 ----
    if use_multi_freq:
        model = MultiFreqStockTransformer(**MODEL_CFG).to(device)
    else:
        model = SimpleStockTransformer().to(device)
    n_params = count_parameters(model)
    logger.info("参数量", n_params=n_params)
    assert n_params >= 100_000

    loss_fn = RankingLoss(temperature=0.5, pairwise_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # 手动 warmup: 前 WARMUP_EPOCHS 个 epoch 线性增长 lr
    base_lr = LR

    best_val_ic = -float("inf")
    best_epoch = 0; patience = 0

    # ---- 7. 训练 ----
    for epoch in range(EPOCHS):
        # Warmup
        if epoch < WARMUP_EPOCHS:
            lr = base_lr * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups: pg["lr"] = lr

        model.train()
        total_loss, n_batches = 0.0, 0
        np.random.shuffle(train_dates)

        for date in train_dates:
            entries = data_by_date[date]
            if len(entries) < 10: continue

            xs = []
            for i, fl in enumerate(FREQ_LABELS):
                x = np.stack([e["seqs"][i] for e in entries]).astype(np.float32)
                xs.append(torch.from_numpy(x).to(device))

            labels = torch.from_numpy(np.array([e["label"] for e in entries], np.float32)).to(device)
            preds = model.encode_freqs(xs)
            preds = model.score_head(preds).squeeze(-1)
            loss = loss_fn(preds.unsqueeze(0), labels.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

        if epoch >= WARMUP_EPOCHS: scheduler.step()

        # ---- 验证 ----
        model.eval()
        val_ics = []
        with torch.no_grad():
            for date in val_dates:
                entries = data_by_date[date]
                if len(entries) < 10: continue
                xs, labels = [], np.array([e["label"] for e in entries], np.float32)
                for i, fl in enumerate(FREQ_LABELS):
                    x = np.stack([e["seqs"][i] for e in entries]).astype(np.float32)
                    xs.append(torch.from_numpy(x).to(device))
                h = model.encode_freqs(xs)
                preds = model.score_head(h).squeeze(-1).cpu().numpy()
                m = compute_ic(preds, labels)
                if m["n"] >= 10: val_ics.append(m)

        avg_ic = np.mean([m["ic_spearman"] for m in val_ics]) if val_ics else 0.0
        avg_loss = total_loss / max(n_batches, 1)

        logger.info(f"Epoch {epoch+1}/{EPOCHS}", loss=round(avg_loss, 6),
                    val_ic=round(avg_ic, 6), lr=round(optimizer.param_groups[0]["lr"], 8))

        if avg_ic > best_val_ic:
            best_val_ic = avg_ic; best_epoch = epoch + 1; patience = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "model_cfg": MODEL_CFG,
                "feature_cols": {fl: FREQ_CONFIGS[fl]["all_cols"] for fl in FREQ_LABELS},
                "vol_cols": {fl: FREQ_CONFIGS[fl]["vol_cols"] for fl in FREQ_LABELS},
                "freq_configs": {fl: FREQ_CONFIGS[fl]["seq_len"] for fl in FREQ_LABELS},
                "use_multi_freq": use_multi_freq,
                "stats": {k: (v[0].tolist(), v[1].tolist()) for k, v in stats.items()},
            }
            save_model_json(ckpt, model_path)
            logger.info("保存最佳", ic=round(avg_ic, 6))
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                logger.info("早停", best_epoch=best_epoch, best_ic=round(best_val_ic, 6))
                break

    logger.info("训练完成", best_epoch=best_epoch, best_val_ic=round(best_val_ic, 6))
    return model_path


if __name__ == "__main__":
    datasources = {"bar1m": TABLE_1M, "bar5m": TABLE_5M, "bar15m": TABLE_15M, "bar30m": TABLE_30M}
    train_and_save(datasources, use_multi_freq=True)
