# -*- coding: utf-8 -*-
"""IC3Net 残差3层 —— 训练侧脚本 (共享定义 + 从零训练并持久化)。

架构: 3层残差叠加 f = f0 + Delta1+ + Delta2+
  分层: 0-33-67-100 (3组等分, label quantile)
  训练: 逐层递进, 每层只在其对应分组及更高组上训练
  预测: 逐层剔除 cascading

本文件承担两件事:
  1. 沉淀训练与推理共用的定义, 作为单一事实来源
  2. 提供 train_and_save() 在写死的训练区间上从零训练, 保存模型到 JSON
"""
import os, json, time
import numpy as np
import pandas as pd
import dai
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger()

# ---------- 路径 (最先定义) ----------
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
MODEL_PATH = os.path.join(_HERE, "ic3net_res_up_model.json")

# ---------- 配置 ----------
TRAIN_START   = '2019-01-01 00:00:00'
TRAIN_END     = '2024-12-31 23:59:59'
N_GROUPS      = 3
N_CHAIN_BLOCKS = 2
MARGIN_CHAIN  = 0.1
MARGIN_CROSS  = 0.1
MARGIN_PAIR   = 0.2
LAMBDA_STRAT  = 0.5
LAMBDA_PAIR   = 0.1
LAMBDA_CROSS  = 0.3
N_PAIRS       = 50
N_EPOCHS      = 5
BATCH_DATES   = 10
LR            = 1e-3
WD            = 1e-5
SEED          = 42
BATCH_PRED    = 50000

price_cols = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'vwap', 'twap', 'vwap_twap_spread',
    'intraday_return', 'amplitude', 'close_position', 'real_body_ratio',
    'close_std', 'close_cv',
    'vol_concentration', 'vol_cv',
    'up_ratio', 'up_vol_ratio',
    'morning_vol_pct', 'afternoon_vol_pct',
]
FEATURE_COLS = price_cols
N_FEAT = len(FEATURE_COLS)

MODEL_CFG = dict(input_dim=N_FEAT, hidden_dims=[64, 32, 16])
RESIDUAL_CFG = dict(input_dim=N_FEAT, hidden_dims=[48, 24, 12])

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# ---------- 轻量 Scaler ----------
class SimpleScaler:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
    def transform(self, X):
        return (X - self.mean) / self.scale


# ---------- 模型 ----------
class FactorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [64, 32, 16]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


class ResidualHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [48, 24, 12]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


# ---------- 损失函数 ----------
def cross_sectional_ic_loss(predictions, labels, dates_tensor):
    unique_dates = torch.unique(dates_tensor)
    if len(unique_dates) == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
    date_ics = []
    for d in unique_dates:
        mask = (dates_tensor == d)
        if mask.sum().item() < 10: continue
        p = predictions[mask]; l = labels[mask]
        p_norm = (p - p.mean()) / (p.std() + 1e-8)
        l_norm = (l - l.mean()) / (l.std() + 1e-8)
        date_ics.append((p_norm * l_norm).mean())
    if len(date_ics) == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
    return -torch.stack(date_ics).mean()


def chain_stratification_loss(predictions, labels, dates_tensor,
                               n_blocks=N_CHAIN_BLOCKS, margin=MARGIN_CHAIN):
    ud = torch.unique(dates_tensor)
    losses = []
    for d in ud:
        m = (dates_tensor == d)
        n = m.sum().item()
        if n < n_blocks * 3: continue
        p = predictions[m]; l = labels[m]
        si = torch.argsort(l)
        block_means = []
        for i in range(n_blocks):
            start = int(n * i / n_blocks)
            end   = int(n * (i + 1) / n_blocks)
            block_means.append(p[si[start:end]].mean())
        for i in range(n_blocks - 1):
            losses.append(torch.relu(margin - (block_means[i+1] - block_means[i])))
    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
    return torch.stack(losses).mean()


def cross_block_strat_loss(predictions, group_ids, dates_tensor,
                            own_group, upper_groups, margin=MARGIN_CROSS):
    """单向向上: mean(f[upper]) > mean(f[own]) + margin"""
    ud = torch.unique(dates_tensor)
    losses = []
    for d in ud:
        date_mask = (dates_tensor == d)
        own_mask = date_mask & (group_ids == own_group)
        if own_mask.sum() < 3: continue
        own_mean = predictions[own_mask].mean()
        for ug in upper_groups:
            upper_mask = date_mask & (group_ids == ug)
            if upper_mask.sum() < 3: continue
            upper_mean = predictions[upper_mask].mean()
            losses.append(torch.relu(margin - (upper_mean - own_mean)))
    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
    return torch.stack(losses).mean()


def pairwise_ranking_loss(predictions, labels, dates_tensor,
                          top_pct=0.40, bottom_pct=0.40,
                          margin=MARGIN_PAIR, n_pairs=N_PAIRS):
    ud = torch.unique(dates_tensor)
    al = []
    for d in ud:
        m = (dates_tensor == d)
        n = m.sum().item()
        if n < 15: continue
        p = predictions[m]; l = labels[m]
        nt = max(1, int(n * top_pct)); nb = max(1, int(n * bottom_pct))
        si = torch.argsort(l)
        tf = p[si[-nt:]]; bf = p[si[:nb]]
        ns = min(n_pairs, nt * nb)
        if ns == 0: continue
        ti = torch.randint(0, nt, (ns,), device=p.device)
        bi = torch.randint(0, nb, (ns,), device=p.device)
        al.append(torch.relu(margin - (tf[ti] - bf[bi])).mean())
    if len(al) == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
    return torch.stack(al).mean()


# ---------- 数据集 ----------
class DateGroupedDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols, has_groups=True):
        self.feature_cols = feature_cols
        self.has_groups = has_groups
        self.dates = sorted(df['date'].unique())
        self.date_groups = {}
        for d in self.dates:
            mask = df['date'] == d
            gd = {
                'features': torch.tensor(df.loc[mask, feature_cols].values, dtype=torch.float32),
                'labels':   torch.tensor(df.loc[mask, 'label'].values, dtype=torch.float32),
            }
            if has_groups and 'group_id' in df.columns:
                gd['group_ids'] = torch.tensor(df.loc[mask, 'group_id'].values, dtype=torch.long)
            self.date_groups[d] = gd
    def __len__(self): return len(self.dates)
    def __getitem__(self, idx):
        g = self.date_groups[self.dates[idx]]
        item = {'features': g['features'], 'labels': g['labels'],
                'date_idx': idx, 'n_stocks': len(g['labels'])}
        if self.has_groups and 'group_ids' in g:
            item['group_ids'] = g['group_ids']
        return item


def collate_fn(batch):
    af, al, ad, ag = [], [], [], []
    has_groups = 'group_ids' in batch[0]
    for item in batch:
        n = item['n_stocks']
        af.append(item['features']); al.append(item['labels'])
        ad.append(torch.full((n,), item['date_idx'], dtype=torch.long))
        if has_groups: ag.append(item['group_ids'])
    result = {'features': torch.cat(af,0), 'labels': torch.cat(al,0),
               'date_indices': torch.cat(ad,0)}
    if has_groups: result['group_ids'] = torch.cat(ag,0)
    return result


# ---------- 训练函数 ----------
def _freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad = False


def train_net0(model, dataset, device, n_epochs=N_EPOCHS, batch_dates=BATCH_DATES,
               lr=LR, wd=WD, name='net0'):
    """训练 net0: 全截面 forward, IC/strat/pair 只在 Q0, cross_block 向上"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    model.train(); N = len(dataset)
    for ep in range(n_epochs):
        el = ei = es = epa = ecb = 0.0; nb = 0
        for s in range(0, N, batch_dates):
            e = min(s+batch_dates, N)
            b = collate_fn([dataset[i] for i in range(s, e)])
            feats = b['features'].to(device); labs = b['labels'].to(device)
            di = b['date_indices'].to(device); gids = b['group_ids'].to(device)
            preds = model(feats)
            g0 = (gids == 0)
            if g0.sum() < 10: continue
            ic = cross_sectional_ic_loss(preds[g0], labs[g0], di[g0])
            st = chain_stratification_loss(preds[g0], labs[g0], di[g0])
            pr = pairwise_ranking_loss(preds[g0], labs[g0], di[g0])
            cb = cross_block_strat_loss(preds, gids, di, own_group=0,
                                        upper_groups=list(range(1, N_GROUPS)))
            loss = ic + LAMBDA_STRAT*st + LAMBDA_PAIR*pr + LAMBDA_CROSS*cb
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item(); ei += ic.item(); es += st.item()
            epa += pr.item(); ecb += cb.item(); nb += 1
        avg = el/max(nb,1); sch.step(avg)
        logger.info(f"[{name}] E{ep+1}/{n_epochs} loss={avg:.4f} ic={ei/max(nb,1):.4f} "
                    f"st={es/max(nb,1):.4f} pr={epa/max(nb,1):.4f} cb={ecb/max(nb,1):.4f}")
    return model


def train_residual_net(model_res, upstream_specs, raw_feats_np, cur_scaler,
                       dataset, device, own_group, upper_groups=None,
                       has_cross_block=True,
                       n_epochs=N_EPOCHS, batch_dates=BATCH_DATES, lr=LR, wd=WD, name='res'):
    """训练残差模型。
    upstream_specs: [(model, scaler), ...] 上游模型和各自的 scaler
    raw_feats_np:   原始特征（供上游模型使用各自的 scaler）
    cur_scaler:     当前模型的 scaler
    own_group:      当前 net 的组号
    upper_groups:   跨块约束的上层组号列表
    """
    for um, _ in upstream_specs: _freeze(um)
    opt = torch.optim.Adam(model_res.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    model_res.train(); N = len(dataset)
    for ep in range(n_epochs):
        el = ei = es = epa = ecb = 0.0; nb = 0
        for s in range(0, N, batch_dates):
            e = min(s+batch_dates, N)
            batch_items = [dataset[i] for i in range(s, e)]
            b = collate_fn(batch_items)
            labs = b['labels'].to(device); di = b['date_indices'].to(device)
            gids = b['group_ids'].to(device)
            feats_for_res = b['features'].to(device)
            batch_start = sum(dataset[i]['n_stocks'] for i in range(s))
            batch_size  = labs.shape[0]
            with torch.no_grad():
                factor = torch.zeros(batch_size, device=device)
                for um, um_scaler in upstream_specs:
                    um_feats_np = um_scaler.transform(raw_feats_np[batch_start:batch_start+batch_size])
                    um_feats = torch.tensor(um_feats_np, dtype=torch.float32).to(device)
                    um_pred = um(um_feats)
                    factor = factor + (F.softplus(um_pred) if isinstance(um, ResidualHead) else um_pred)
            delta_raw = model_res(feats_for_res)
            factor = factor + F.softplus(delta_raw)
            g_own = (gids == own_group)
            if g_own.sum() < 10: continue
            ic = cross_sectional_ic_loss(factor[g_own], labs[g_own], di[g_own])
            st = chain_stratification_loss(factor[g_own], labs[g_own], di[g_own])
            pr = pairwise_ranking_loss(factor[g_own], labs[g_own], di[g_own])
            cb = torch.tensor(0.0, device=device)
            if has_cross_block and upper_groups:
                cb = cross_block_strat_loss(factor, gids, di, own_group=own_group, upper_groups=upper_groups)
            loss = ic + LAMBDA_STRAT*st + LAMBDA_PAIR*pr + LAMBDA_CROSS*cb
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_res.parameters(), 1.0)
            opt.step()
            el += loss.item(); ei += ic.item(); es += st.item()
            epa += pr.item(); ecb += cb.item(); nb += 1
        avg = el/max(nb,1); sch.step(avg)
        log = {"loss": round(avg,4), "ic": round(ei/max(nb,1),4),
               "st": round(es/max(nb,1),4), "pr": round(epa/max(nb,1),4)}
        if has_cross_block: log["cb"] = round(ecb/max(nb,1),4)
        logger.info(f"[{name}] E{ep+1}/{n_epochs} "
                    + " ".join(f"{k}={v:.4f}" for k, v in log.items()))
    return model_res


# ---------- 特征工程 ----------
def build_features(bar1m_table, sd, ed):
    t0 = time.time()
    logger.info("build_features 开始", start=str(sd), end=str(ed))
    price_start = pd.to_datetime(sd) - pd.Timedelta(days=7)
    price_sql = f"""
    SELECT date_trunc('day', date)::DATE AS trading_day, instrument,
        ARG_MIN(open, date) AS open, MAX(high) AS high, MIN(low) AS low,
        ARG_MAX(close, date) AS close, SUM(volume) AS volume, SUM(amount) AS amount,
        SUM(amount)/NULLIF(SUM(volume),0) AS vwap, AVG(close) AS twap,
        (SUM(amount)/NULLIF(SUM(volume),0)-AVG(close))/NULLIF(ABS(AVG(close)),0) AS vwap_twap_spread,
        (ARG_MAX(close,date)-ARG_MIN(open,date))/NULLIF(ARG_MIN(open,date),0) AS intraday_return,
        (MAX(high)-MIN(low))/NULLIF(ARG_MIN(open,date),0) AS amplitude,
        (ARG_MAX(close,date)-MIN(low))/NULLIF(MAX(high)-MIN(low),0) AS close_position,
        ABS(ARG_MAX(close,date)-ARG_MIN(open,date))/NULLIF(MAX(high)-MIN(low),0) AS real_body_ratio,
        STDDEV(close) AS close_std, STDDEV(close)/NULLIF(ABS(AVG(close)),0) AS close_cv,
        CAST(MAX(volume) AS DOUBLE)/NULLIF(SUM(volume),0) AS vol_concentration,
        STDDEV(volume)/NULLIF(AVG(volume),0) AS vol_cv,
        CAST(SUM(CASE WHEN close>=open THEN 1 ELSE 0 END) AS DOUBLE)/NULLIF(COUNT(*),0) AS up_ratio,
        CAST(SUM(CASE WHEN close>=open THEN volume ELSE 0 END) AS DOUBLE)/NULLIF(SUM(volume),0) AS up_vol_ratio,
        CAST(SUM(CASE WHEN EXTRACT(HOUR FROM date)<12 THEN volume ELSE 0 END) AS DOUBLE)/NULLIF(SUM(volume),0) AS morning_vol_pct,
        CAST(SUM(CASE WHEN EXTRACT(HOUR FROM date)>=12 THEN volume ELSE 0 END) AS DOUBLE)/NULLIF(SUM(volume),0) AS afternoon_vol_pct
    FROM {bar1m_table} GROUP BY trading_day, instrument ORDER BY trading_day, instrument"""
    price = dai.query(price_sql, filters={'date': [price_start, ed]}).df().rename(
        columns={'trading_day': 'date'})
    price['date'] = pd.to_datetime(price['date'])
    price = price.sort_values(['instrument','date']).reset_index(drop=True)
    logger.info("量价特征完成", rows=len(price), elapsed=round(time.time()-t0,2))
    g = price.groupby('instrument', group_keys=False)['close']
    price['label'] = g.shift(-1) / price['close'] - 1
    df = price.copy()
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df[(df['date']>=pd.to_datetime(sd))&(df['date']<=pd.to_datetime(ed))]
    logger.info("build_features 结束", rows=len(df), total_elapsed=round(time.time()-t0,2))
    return df.reset_index(drop=True)


# ---------- 预测函数（逐层剔除 cascading） ----------
def run_cascade_prediction(models, scalers, test_df, test_raw, device,
                           n_groups=N_GROUPS, batch_pred=BATCH_PRED):
    """逐层剔除 cascading 预测, 返回含 factor 列的 test_df

    分层: S0 排名 → 取底 1/N_GROUPS = Q0 → 剔除
          S1 排名剩余 → 取底 1/(N_GROUPS-1) = Q1 → 剔除
          剩余 = Q2
    """
    t0 = time.time()
    for m in models: m.eval()
    N_test = len(test_raw)
    chains = np.zeros((N_test, n_groups), dtype=np.float32)

    for start in range(0, N_test, batch_pred):
        end = min(start + batch_pred, N_test)
        batch_raw = test_raw[start:end]
        x0 = torch.tensor(scalers[0].transform(batch_raw), dtype=torch.float32).to(device)
        with torch.no_grad():
            accum = models[0](x0)
        chains[start:end, 0] = accum.cpu().numpy()
        for k in range(1, n_groups):
            xk = torch.tensor(scalers[k].transform(batch_raw), dtype=torch.float32).to(device)
            with torch.no_grad():
                accum = accum + F.softplus(models[k](xk))
            chains[start:end, k] = accum.cpu().numpy()

    test_df['factor'] = 0.0
    n_covered = [0] * n_groups

    def _cascade(group):
        n = len(group)
        gi = group.index.values
        remaining = np.ones(n, dtype=bool)
        for g in range(n_groups - 1):
            n_rem = remaining.sum()
            if n_rem == 0: break
            pool_local = np.where(remaining)[0]
            pool_global = gi[pool_local]
            score_g = chains[pool_global, g]
            order = np.argsort(score_g)
            frac = 1.0 / (n_groups - g)
            n_take = max(1, int(n_rem * frac))
            take_local = pool_local[order[:n_take]]
            take_global = gi[take_local]
            group.loc[group.index[take_local], 'factor'] = chains[take_global, g]
            n_covered[g] += n_take
            remaining[take_local] = False
        if remaining.sum() > 0:
            last_local = np.where(remaining)[0]
            last_global = gi[last_local]
            g_last = n_groups - 1
            group.loc[group.index[last_local], 'factor'] = chains[last_global, g_last]
            n_covered[g_last] += len(last_local)
        return group

    test_df = test_df.groupby('date', group_keys=False).apply(_cascade)
    logger.info("cascade预测完成", total=len(test_df), elapsed=round(time.time()-t0,2),
                **{f'Q{i}': n_covered[i] for i in range(n_groups)})
    return test_df


# ==== 模型存/读 (JSON, 3 models + 3 scalers) ====
def save_models(models, scalers, model_cfg, residual_cfg, feature_cols,
                n_groups, model_path=MODEL_PATH):
    def _serialize(sd):
        tensors = {}
        for k, v in sd.items():
            t = v.detach().cpu()
            tensors[k] = {"dtype": str(t.dtype).replace("torch.",""),
                          "shape": list(t.shape), "data": t.reshape(-1).tolist()}
        return tensors
    payload = {
        "models": {f"net{i}": _serialize(m.state_dict()) for i, m in enumerate(models)},
        "scalers": {str(i): {"mean": s.mean_.tolist(), "scale": s.scale_.tolist()}
                    for i, s in enumerate(scalers)},
        "model_cfg": model_cfg, "residual_cfg": residual_cfg,
        "feature_cols": feature_cols, "n_groups": n_groups,
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_models(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as f:
        p = json.load(f)
    def _deserialize(tensors):
        sd = {}
        for k, meta in tensors.items():
            t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
            sd[k] = t.reshape(meta["shape"]).to(map_location)
        return sd
    models = []
    models.append(FactorMLP(**p["model_cfg"]).to(map_location))
    models[0].load_state_dict(_deserialize(p["models"]["net0"]))
    models[0].eval()
    for i in range(1, p["n_groups"]):
        m = ResidualHead(**p["residual_cfg"]).to(map_location)
        m.load_state_dict(_deserialize(p["models"][f"net{i}"]))
        m.eval()
        models.append(m)
    scalers = [SimpleScaler(p["scalers"][str(i)]["mean"],
                            p["scalers"][str(i)]["scale"])
               for i in range(p["n_groups"])]
    return models, scalers, p["feature_cols"], p["n_groups"]


# ==== 训练并持久化 ====
def train_and_save(bar1m_table='bigalpha_2026_stock_bar1m',
                   train_start=TRAIN_START, train_end=TRAIN_END,
                   model_path=MODEL_PATH):
    global device
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logger.info("训练设备", device=str(device), table=bar1m_table)
    t_total = time.time()

    from sklearn.preprocessing import StandardScaler

    # --- 构建训练集 & 3等分拆分 ---
    logger.info("开始构建训练集", train_start=train_start, train_end=train_end)
    train_df = build_features(bar1m_table, train_start, train_end)
    train_df['label'] = train_df['label'].replace([np.inf,-np.inf], np.nan)
    train_df = train_df.dropna(subset=['label']).dropna(subset=FEATURE_COLS)

    quantiles = [1.0 / N_GROUPS * (i+1) for i in range(N_GROUPS - 1)]
    logger.info(f"按 label 分位{N_GROUPS}等分", quantiles=quantiles)

    def split_n(group):
        qs = [group['label'].quantile(p) for p in quantiles]
        masks = [group['label'] < qs[0]]
        for i in range(1, len(qs)):
            masks.append((group['label'] >= qs[i-1]) & (group['label'] < qs[i]))
        masks.append(group['label'] >= qs[-1])
        return [group[m] for m in masks]

    groups_lists = [[] for _ in range(N_GROUPS)]
    for d, grp in train_df.groupby('date'):
        for i, g in enumerate(split_n(grp)):
            groups_lists[i].append(g)
    train_groups = [pd.concat(gl, ignore_index=True) for gl in groups_lists]
    for i, tg in enumerate(train_groups): tg['group_id'] = i
    train_df = pd.concat(train_groups, ignore_index=True)
    logger.info("拆分完成", total=len(train_df),
                **{f'Q{i}': len(train_groups[i]) for i in range(N_GROUPS)})

    raw_all = train_df[FEATURE_COLS].values.astype(np.float64)
    raw_all = np.nan_to_num(raw_all, nan=0.0, posinf=0.0, neginf=0.0)
    g_masks = [train_df['group_id'] == i for i in range(N_GROUPS)]

    # --- 训练 net0 ---
    scalers = [None] * N_GROUPS
    models  = [None] * N_GROUPS
    scalers[0] = StandardScaler().fit(raw_all[g_masks[0]])
    df0 = train_df.copy()
    df0[FEATURE_COLS] = scalers[0].transform(raw_all)
    ds0 = DateGroupedDataset(df0, FEATURE_COLS, has_groups=True)
    models[0] = FactorMLP(**MODEL_CFG).to(device)
    t0 = time.time()
    models[0] = train_net0(models[0], ds0, device, name='net0')
    logger.info("net0 完成", elapsed=round(time.time()-t0,2))

    # --- 训练 net1~net2 ---
    for k in range(1, N_GROUPS):
        upper = list(range(k+1, N_GROUPS))
        logger.info(f"训练 net{k} (Q{k}, cross={upper})")
        train_mask = g_masks[k].copy()
        for j in range(k+1, N_GROUPS):
            train_mask = train_mask | g_masks[j]
        raw_k = raw_all[train_mask.values]
        scalers[k] = StandardScaler().fit(raw_all[g_masks[k]])
        df_k = train_df.loc[train_mask].copy()
        df_k[FEATURE_COLS] = scalers[k].transform(raw_k)
        ds_k = DateGroupedDataset(df_k, FEATURE_COLS, has_groups=True)
        upstream_specs = [(models[i], scalers[i]) for i in range(k)]
        models[k] = ResidualHead(**RESIDUAL_CFG).to(device)
        t0 = time.time()
        models[k] = train_residual_net(
            models[k], upstream_specs, raw_k, scalers[k], ds_k, device,
            own_group=k, upper_groups=upper if upper else None,
            has_cross_block=(len(upper) > 0), name=f'net{k}')
        logger.info(f"net{k} 完成", elapsed=round(time.time()-t0,2))

    # --- 持久化 ---
    save_models(models, scalers, MODEL_CFG, RESIDUAL_CFG, FEATURE_COLS, N_GROUPS, model_path)
    logger.info("模型已保存", path=model_path, total_elapsed=round(time.time()-t_total,2))
    return model_path


if __name__ == '__main__':
    train_and_save()
    logger.info("训练全部完成!")
