"""BigAlpha 2026 comp2 - local training script (wiki "模型本地化训练指南" format).

Trains the end-to-end Transformer LOCALLY on the downloaded 5m feather data and
saves `transformer_model.json` (state_dict + model_cfg + feature_cols + seq_len
+ z-score mean/std). The companion inference notebook loads this JSON at
predict time - the public leaderboard does NOT retrain.

Key points vs the earlier train_local.py:
  * `to_canonical(is_local=True)` aligns the LOCAL compressed data to the same
    canonical form the CLOUD inference will use (prices/amount /100 分->元,
    OHLC -1 -> NaN, 3-level book, instrument_id as key). This fixes the
    train/infer drift the wiki warns about (amount was log1p'd in 分 locally but
    in 元 on the cloud -> ~4.6 log1p offset).
  * v13 differentiable soft-rank IC loss (the v3-v12 argsort rankic was
    non-differentiable -> zero gradient).
  * full instrument pool, lr=5e-4, held-out 2023-10/11 val-IC model selection.

Run:  /home/user/miniconda3/envs/dm_mvp/bin/python transformer_train_local.py
"""
import os, sys, time, glob, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ======================================================================
# Config
# ======================================================================
LOCAL_DATA_ROOT = '/home/user/comp2_end2end/data'
LOCAL_TABLE = 'bigalpha_2026_e2e_bar5m'
TRAIN_START, TRAIN_END = '2019-01-01', '2023-12-31 23:59:59'
VAL_START, VAL_END = '2023-10-01', '2023-11-30'   # held out for val-IC selection
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'transformer_model.json')

SEQ_LEN   = 48      # 5m bars = one trading day
FWD_DAYS  = 20      # 20-trading-day forward return label
EPOCHS    = 15
LR        = 5e-4
SEED      = 42
MAX_INSTR = 0       # 0 = full pool
D_MODEL, NHEAD, DIM_FF, NLAYERS, DROPOUT = 128, 4, 512, 4, 0.1
BATCH_HOLDER = 1    # by-date cross-section batches

# Fixed 25-feature set (local 3-level book). Cloud inference drops 4/5档 via
# to_canonical so the same 25 columns exist on both sides.
_BASIC = ['open', 'high', 'low', 'close', 'volume', 'amount']           # 6
_ASK_P  = ['ask_price1', 'ask_price2', 'ask_price3']                    # 3
_BID_P  = ['bid_price1', 'bid_price2', 'bid_price3']                    # 3
_ASK_V  = ['ask_volume1', 'ask_volume2', 'ask_volume3']                 # 3
_BID_V  = ['bid_volume1', 'bid_volume2', 'bid_volume3']                 # 3
_ORDERS = ['ask_num_orders1', 'ask_num_orders2', 'ask_num_orders3',
           'bid_num_orders1', 'bid_num_orders2', 'bid_num_orders3']     # 6
_DEAL   = ['deal_number']                                               # 1
FEATURE_COLS = _BASIC + _ASK_P + _BID_P + _ASK_V + _BID_V + _ORDERS + _DEAL  # 25
PRICE_COLS = ['open', 'high', 'low', 'close'] + _ASK_P + _BID_P
# all non-price feats are non-negative counts -> log1p (allowed 'log')
VOL_COLS = [c for c in FEATURE_COLS if c not in PRICE_COLS]
SCALE_FIELDS = ['open', 'high', 'low', 'close', 'amount'] + _ASK_P + _BID_P  # /100 分->元
OHLC_COLS = ['open', 'high', 'low', 'close']
N_FEAT = len(FEATURE_COLS)
CLOSE_IX = FEATURE_COLS.index('close')
PRICE_SCALE = 100.0


def log(msg, **kw):
    ts = time.strftime('%H:%M:%S')
    parts = ' '.join(f'{k}={v}' for k, v in kw.items())
    print(f'[{ts}] {msg} {parts}'.strip(), flush=True)


# ======================================================================
# Canonical alignment (wiki section 三). Local vs cloud -> same representation.
# ======================================================================
def to_canonical(df, is_local):
    """Local e2e (compressed) or cloud stock (raw) -> canonical: prices/amount
    in 元 (float), OHLC missing as NaN, 3-level book, unified 'key' column."""
    df = df.copy()
    if is_local:
        for c in OHLC_COLS:
            if c in df.columns:
                df.loc[df[c] == -1, c] = np.nan          # -1 = halted/missing
        for c in SCALE_FIELDS:
            if c in df.columns:
                df[c] = df[c].astype('float64') / PRICE_SCALE   # 分 -> 元
        df['key'] = df['instrument_id']
    else:
        # cloud: prices already 元, OHLC already NaN; drop 4/5档 to match local
        drop = [c for c in df.columns
                if any(c.startswith(p) and c[-1] in '45'
                       for p in ('ask_price', 'bid_price', 'ask_volume',
                                 'bid_volume', 'ask_num_orders', 'bid_num_orders'))]
        df = df.drop(columns=drop, errors='ignore')
        df['key'] = df['instrument']
    return df


def preprocess(df):
    """log1p on count features (canonical form already in 元)."""
    df = df.copy()
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    for c in VOL_COLS:
        df[c] = np.log1p(np.clip(df[c].to_numpy(), 0, None))
    return df


# ======================================================================
# Data prep (vectorized; every (ins,day) has exactly 48 5m bars)
# ======================================================================
def load_raw(period):
    root = os.path.join(LOCAL_DATA_ROOT, LOCAL_TABLE)
    if period == 'train':
        files = sorted(glob.glob(os.path.join(root, '2019*.feather')) +
                       glob.glob(os.path.join(root, '2020*.feather')) +
                       glob.glob(os.path.join(root, '2021*.feather')) +
                       glob.glob(os.path.join(root, '2022*.feather')) +
                       glob.glob(os.path.join(root, '2023*.feather')))
        sd, ed = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    else:  # val (2024)
        files = sorted(glob.glob(os.path.join(root, '2024*.feather')))
        sd, ed = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31 23:59:59')
    cols = ['date', 'instrument_id'] + FEATURE_COLS
    dfs = [pd.read_feather(f, columns=cols) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    return df, sd, ed


def build_dataset(period, max_instr=0, seed=42):
    """Returns (X, y, meta[date,key]) with t+FWD_DAYS forward return labels."""
    df, sd, ed = load_raw(period)
    log(f'[{period}] raw', rows=len(df), ins=df.instrument_id.nunique())
    df = to_canonical(df, is_local=True)
    df = preprocess(df)
    df['day'] = df['date'].dt.normalize()
    df = df[(df['day'] >= sd.normalize()) & (df['day'] <= ed.normalize())].copy()
    # ffill OHLC per instrument (handles halted bars after -1->NaN)
    for c in OHLC_COLS:
        df[c] = df.groupby('key')[c].ffill()
    if max_instr and max_instr > 0:
        all_ins = sorted(df['key'].unique())
        if len(all_ins) > max_instr:
            keep = set(np.random.RandomState(seed).choice(all_ins, max_instr, replace=False))
            df = df[df['key'].isin(keep)].copy()
            log(f'[{period}] subsampled', kept=max_instr)
    df = df.sort_values(['key', 'date']).reset_index(drop=True)
    df['_sid'] = df.groupby(['key', 'day']).ngroup()
    df['_bix'] = df.groupby(['key', 'day']).cumcount()
    df = df[df._bix < SEQ_LEN].copy()
    cnt = df.groupby('_sid').size()
    df = df[df._sid.isin(set(cnt[cnt == SEQ_LEN].index))].copy()
    samp = df.groupby('_sid').agg(date=('day', 'first'), key=('key', 'first'),
                                  raw_first_close=('close', 'first'),
                                  raw_last_close=('close', 'last')).reset_index()
    samp['date'] = pd.to_datetime(samp['date'])
    n = len(samp)
    log(f'[{period}] samples', n=n, days=samp.date.nunique(), ins=samp.key.nunique())
    df = df.sort_values(['_sid', '_bix']).reset_index(drop=True)
    X = df[FEATURE_COLS].to_numpy(np.float32, copy=True).reshape(n, SEQ_LEN, N_FEAT)
    # per-day price normalization: prices / first close of the day
    ref = samp['raw_first_close'].to_numpy(np.float32)
    ref = np.where(ref > 0, ref, 1.0)
    pidx = [FEATURE_COLS.index(c) for c in PRICE_COLS]
    X[:, :, pidx] = X[:, :, pidx] / ref[:, None, None]
    # forward-return label from RAW daily closes (canonical 元)
    samp = samp.sort_values(['key', 'date']).reset_index(drop=True)
    fwd = samp.groupby('key')['raw_last_close'].shift(-FWD_DAYS)
    samp['y'] = fwd / samp['raw_last_close'] - 1.0
    samp = samp.dropna(subset=['y']).reset_index(drop=True)
    y = samp['y'].to_numpy(np.float32)
    lo, hi = np.percentile(y, [1, 99])
    y = np.clip(y, lo, hi).astype(np.float32)
    sid_sorted = np.array(sorted(df._sid.unique()))
    sid_pos = {s: i for i, s in enumerate(sid_sorted)}
    xpos = np.array([sid_pos[s] for s in samp['_sid'].to_numpy()])
    X = np.ascontiguousarray(X[xpos])
    meta = samp[['date', 'key']].rename(columns={'key': 'instrument_id'}).reset_index(drop=True)
    log(f'[{period}] done', samples=len(X), y_mean=round(float(y.mean()), 6),
        y_std=round(float(y.std()), 6), winsor=f'[{lo:.4f},{hi:.4f}]')
    return X, y, meta


def zscore(X, stats=None):
    if stats is None:
        flat = X.reshape(-1, N_FEAT)
        m = np.nanmean(flat, axis=0).astype(np.float32)
        s = np.nanstd(flat, axis=0).astype(np.float32) + 1e-6
        stats = (m, s)
    m, s = stats
    return ((X - m) / s).astype(np.float32), stats


# ======================================================================
# Model (v3 architecture + v11 deeper head)
# ======================================================================
class StockTransformer(nn.Module):
    def __init__(self, n_feat=N_FEAT, d_model=D_MODEL, nhead=NHEAD,
                 dim_ff=DIM_FF, nlayers=NLAYERS, dropout=DROPOUT, seq_len=SEQ_LEN):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, d_model // 2), nn.GELU(),
                                  nn.Linear(d_model // 2, 1))

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.size(1), :]
        return self.head(self.encoder(h).mean(dim=1)).squeeze(-1)


# ======================================================================
# v13 differentiable soft-rank IC loss
# ======================================================================
def soft_rank(x, temp=0.5):
    return torch.sigmoid((x.unsqueeze(-1) - x.unsqueeze(-2)) / temp).sum(-1)


def soft_rankic_loss(pred, target, temp=0.5):
    n = pred.size(0)
    if n < 5:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pn = (pred - pred.mean()) / (pred.std() + 1e-6)
    sp = soft_rank(pn, temp) - soft_rank(pn, temp).mean()
    rt = target.argsort().argsort().float()
    rt = rt - rt.mean()
    ic = (sp * rt).sum() / (sp.norm() * rt.norm() + 1e-10)
    return 1.0 - ic


def spread_loss(pred, target):
    n = pred.size(0)
    if n < 10:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    p = (pred - pred.mean()) / (pred.std() + 1e-8)
    w_top = torch.sigmoid(p * 2.0); w_bot = torch.sigmoid(-p * 2.0)
    top = (w_top * target).sum() / (w_top.sum() + 1e-8)
    bot = (w_bot * target).sum() / (w_bot.sum() + 1e-8)
    return torch.relu(0.005 - (top - bot))


def v13_loss(pred, target):
    l_ic = soft_rankic_loss(pred, target)
    l_sp = spread_loss(pred, target)
    l_huber = nn.functional.smooth_l1_loss(pred, target)
    return 0.70 * l_ic + 0.15 * l_sp + 0.15 * l_huber, {'ic': 1.0 - l_ic.detach().item()}


# ======================================================================
# By-date dataset + eval
# ======================================================================
class ByDateDataset(Dataset):
    def __init__(self, X, y, meta):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        dates = meta['date'].values
        self.groups = {}
        for d in np.unique(dates):
            idx = np.where(dates == d)[0]
            if len(idx) >= 5:
                self.groups[d] = torch.from_numpy(idx)
        self.day_list = list(self.groups.keys())

    def __len__(self):
        return len(self.day_list)

    def __getitem__(self, i):
        idx = self.groups[self.day_list[i]]
        return self.X[idx], self.y[idx]


def eval_val_ic(model, X, y, meta, device, batch=2048):
    model.eval()
    preds = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i+batch])).to(device)
            preds[i:i+batch] = model(xb).cpu().numpy()
    ics = []
    dates = meta['date'].values
    for d in np.unique(dates):
        idx = np.where(dates == d)[0]
        if len(idx) < 10:
            continue
        pr = pd.Series(preds[idx]).rank().to_numpy()
        yr = pd.Series(y[idx]).rank().to_numpy()
        pr = pr - pr.mean()
        yr = yr - yr.mean()
        ics.append((pr * yr).sum() / (np.linalg.norm(pr) * np.linalg.norm(yr) + 1e-12))
    model.train()
    return float(np.mean(ics)) if ics else 0.0


# ======================================================================
# Train + save model.json
# ======================================================================
def train_and_save():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('START', device=str(device), epochs=EPOCHS, lr=LR, max_instr=MAX_INSTR)

    Xtr, ytr, meta_tr = build_dataset('train', max_instr=MAX_INSTR, seed=SEED)
    Xtr_z, stats = zscore(Xtr)
    del Xtr
    # held-out val split (2023-10/11) for model selection
    vd = pd.to_datetime(meta_tr['date'].values)
    vmask = (vd >= pd.Timestamp(VAL_START)) & (vd <= pd.Timestamp(VAL_END))
    vidx = np.where(vmask)[0]; tidx = np.where(~vmask)[0]
    Xva, yva, meta_va = Xtr_z[vidx], ytr[vidx], meta_tr.iloc[vidx].reset_index(drop=True)
    Xtr_z, ytr, meta_tr = Xtr_z[tidx], ytr[tidx], meta_tr.iloc[tidx].reset_index(drop=True)
    log('split', train=len(Xtr_z), val=len(Xva), val_days=meta_va.date.nunique())

    # 2024 val for reporting (not used for selection)
    Xr, yr, meta_r = build_dataset('val')
    Xr_z, _ = zscore(Xr, stats)
    va_end = pd.Timestamp('2024-01-01') + pd.DateOffset(months=6)
    m = meta_r['date'] < va_end
    Xr_z, yr, meta_r = Xr_z[m.to_numpy()], yr[m.to_numpy()], meta_r[m].reset_index(drop=True)
    train_ins = set(meta_tr.instrument_id.unique())
    m2 = meta_r.instrument_id.isin(train_ins).to_numpy()
    Xr_z, yr, meta_r = Xr_z[m2], yr[m2], meta_r[m2].reset_index(drop=True)
    del Xr
    log('2024 report val', samples=len(Xr_z), days=meta_r.date.nunique())

    model = StockTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log('model', params=n_params)
    assert 1e5 <= n_params <= 1e8, f'params {n_params} out of bounds'

    ds = ByDateDataset(Xtr_z, ytr, meta_tr)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    # SWA: average the last SWA_N epochs' weights for a robust final model.
    # The held-out 2023-10/11 val IC is flat across epochs (doesn't track 2024),
    # so val-IC selection is unreliable; SWA over the cosine-annealed tail is a
    # leakage-free way to get a stable, well-converged model.
    SWA_N = 5
    swa_states = []
    log('training', loss='0.70*softRankIC+0.15*Spread+0.15*Huber',
        select=f'SWA last {SWA_N} epochs')
    for ep in range(EPOCHS):
        t0 = time.time(); model.train(); tot, cnt = 0.0, 0
        for Xb, yb in loader:
            Xb = Xb.squeeze(0).to(device); yb = yb.squeeze(0).to(device)
            if len(yb) < 5: continue
            opt.zero_grad()
            loss, _ = v13_loss(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); cnt += 1
        sched.step()
        if ep >= EPOCHS - SWA_N:
            swa_states.append({k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
        ric = eval_val_ic(model, Xr_z, yr, meta_r, device) if len(Xr_z) else 0.0
        log(f'ep {ep+1:2d}/{EPOCHS}', loss=round(tot/max(cnt,1),5),
            report_ic=round(ric,4), t=round(time.time()-t0,1),
            swa='*' if ep >= EPOCHS - SWA_N else '')
    # SWA: average the collected tail-epoch weights
    if swa_states:
        avg_state = {}
        for k in swa_states[0]:
            acc = swa_states[0][k].clone()
            for s in swa_states[1:]:
                acc += s[k]
            avg_state[k] = (acc / len(swa_states))
        model.load_state_dict({k: v.to(device) for k, v in avg_state.items()})
        log('SWA averaged', n=len(swa_states))
    final_ic = eval_val_ic(model, Xr_z, yr, meta_r, device) if len(Xr_z) else 0.0
    log('FINAL report(2024H1) ic', val=round(final_ic, 4))

    # ---- save transformer_model.json (wiki format) ----
    mean, std = stats
    sd_json = {k: {'dtype': str(v.dtype).replace('torch.', ''),
                   'shape': list(v.shape),
                   'data': v.cpu().reshape(-1).tolist()}
               for k, v in model.state_dict().items()}
    ckpt = {'state_dict': sd_json,
            'model_cfg': dict(n_feat=N_FEAT, d_model=D_MODEL, nhead=NHEAD,
                              dim_ff=DIM_FF, nlayers=NLAYERS, dropout=DROPOUT,
                              seq_len=SEQ_LEN),
            'feature_cols': FEATURE_COLS,
            'price_cols': PRICE_COLS,
            'vol_cols': VOL_COLS,
            'scale_fields': SCALE_FIELDS,
            'seq_len': SEQ_LEN,
            'mean': mean.tolist(),
            'std': std.tolist(),
            'n_params': n_params}
    with open(MODEL_PATH, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f)
    log('saved', path=MODEL_PATH, n_params=n_params, report_ic=round(final_ic,4),
        size_mb=round(os.path.getsize(MODEL_PATH)/1e6, 1))


if __name__ == '__main__':
    train_and_save()
