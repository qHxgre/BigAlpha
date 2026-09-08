# -*- coding: utf-8 -*-
"""
AlphaPortfolio v11 — 两阶段训练: IC基础 + SR精调

阶段1 (15 epochs): IC loss, 训练全部参数
阶段2 (4 epochs): 冻结SREM, 只训MLP头, 直接优化Sharpe
"""
import sys, time, glob, json, os, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, 'transformer_model.json')

# ===== 配置 =====
SEQ_LEN, EPOCHS, LR, SEED = 96, 19, 1e-3, 42  # 15+4=19
STAGE1_EPOCHS = 15
STAGE2_LR = 1e-4
DATES_PER_BATCH = 16
STOCKS_PER_DATE = 128
BATCH = DATES_PER_BATCH * STOCKS_PER_DATE
MAX_TRAIN_STOCKS = 0
DIM, N_LAYERS = 192, 2
ENSEMBLE_N = 5
CVAR_ALPHA = 0.2          # CVaR alpha (最低20%)
W_CVAR = 0.2              # CVaR loss权重
W_TURNOVER = 0.05         # 换手惩罚权重

PRICE_COLS = ['open','high','low','close',
              'ask_price1','ask_price2','ask_price3',
              'bid_price1','bid_price2','bid_price3']
BOOK_V_COLS = ['ask_volume1','ask_volume2','ask_volume3',
               'bid_volume1','bid_volume2','bid_volume3']
FEATURE_COLS = PRICE_COLS + BOOK_V_COLS + ['volume','amount','deal_number']
N_FEAT = len(FEATURE_COLS)


# ===== 模型 (同v3) =====
class FastTransformer(nn.Module):
    def __init__(self, n_feat=N_FEAT, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(n_feat, dim)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, dim))
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim*2, 0.2, 'gelu', True, norm_first=True),
            N_LAYERS)
        self.pool = nn.Parameter(torch.zeros(1, 1, dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//3),
                                  nn.GELU(), nn.Linear(dim//3, 1))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.pool, std=0.02)


    def forward(self, x):
        B, L, F = x.shape
        h = self.enc(self.proj(x) + self.pos[:, :L, :])
        q = self.pool.expand(B, -1, -1)
        a = torch.softmax(torch.bmm(q, h.transpose(1,2)) / (h.shape[-1]**0.5), dim=-1)
        return self.head(torch.bmm(a, h).squeeze(1)).squeeze(-1)


    def predict_with_ensemble(self, x, n=ENSEMBLE_N):
        self.train()
        preds = [self.forward(x) for _ in range(n)]
        self.eval()
        return torch.stack(preds).mean(0)


StockTransformer = FastTransformer


# ===== Loss函数 =====
def pearson_corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)

def compute_ic_loss(scores, returns):
    ics = [pearson_corr(s, r) for s, r in zip(scores, returns) if len(s)>=10]
    return 1.0 - torch.stack(ics).mean() if ics else torch.tensor(1.0)

def compute_ic_ir_loss(scores, returns):
    ics = [pearson_corr(s, r) for s, r in zip(scores, returns) if len(s)>=10]
    return torch.stack(ics).std() if len(ics)>=2 else torch.tensor(0.0)

def compute_spread_loss(scores, returns, top_frac=0.1):
    spreads = []
    for s, r in zip(scores, returns):
        n, k = len(s), max(1, int(len(s)*top_frac))
        if n < 3: continue
        _, ti = torch.topk(s, k); _, bi = torch.topk(s, k, largest=False)
        spreads.append(r[ti].mean() - r[bi].mean())
    if len(spreads) < 2: return torch.tensor(0.0)
    st = torch.stack(spreads)
    return -(st.mean() / (st.std() + 1e-8))

def compute_cvar_loss(scores, returns, alpha=CVAR_ALPHA):
    """CVaR: 最低alpha%组合收益的均值取负"""
    all_rets = []
    for s, r in zip(scores, returns):
        n, k = len(s), max(1, int(len(s)*0.1))
        if n < 3: continue
        _, ti = torch.topk(s, k); _, bi = torch.topk(s, k, largest=False)
        port_ret = r[ti].mean() - r[bi].mean()
        all_rets.append(port_ret)
    if len(all_rets) < 5: return torch.tensor(0.0)
    rets = torch.stack(all_rets)
    # 最低alpha%的均值
    k_cvar = max(1, int(len(rets) * alpha))
    worst = torch.topk(-rets, k_cvar).values
    return -worst.mean()  # 负均值=loss

def compute_turnover_loss(scores_by_date):
    """换手惩罚: 相邻日得分变化"""
    if len(scores_by_date) < 2: return torch.tensor(0.0)
    diffs = []
    for i in range(len(scores_by_date)-1):
        s0, s1 = scores_by_date[i], scores_by_date[i+1]
        # 取交集股票
        n = min(len(s0), len(s1))
        diffs.append((s0[:n] - s1[:n]).abs().mean())
    return torch.stack(diffs).mean()

def compute_total_loss(scores_by_date, returns_by_date):
    l_ic = compute_ic_loss(scores_by_date, returns_by_date)
    l_icir = compute_ic_ir_loss(scores_by_date, returns_by_date)
    l_sr = compute_spread_loss(scores_by_date, returns_by_date)
    l_cvar = compute_cvar_loss(scores_by_date, returns_by_date)
    l_to = compute_turnover_loss(scores_by_date)

    w_ic, w_icir, w_sr = 0.4, 0.15, 0.25
    total = w_ic*l_ic + w_icir*l_icir + w_sr*l_sr + W_CVAR*l_cvar + W_TURNOVER*l_to
    return total, l_ic.item(), l_icir.item(), l_sr.item(), l_cvar.item(), l_to.item()


# ===== 数据加载 (支持多期限标签) =====
def load_data(horizon=1):
    """加载数据, horizon=N → 标签 = T+N日收益率"""
    data_dir = os.path.join(_HERE, '..', 'outputs', 'competition')
    files = sorted(glob.glob(os.path.join(data_dir, 'e2e_bar5m_*.parquet')))
    if not files: raise RuntimeError(f'No data in {data_dir}')

    print(f'[DATA] Loading (horizon=T+{horizon})...', flush=True); t0 = time.time()
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        for c in PRICE_COLS:
            if c in df.columns: df[c] = df[c].astype('float64') / 100.0
        for c in ['open','high','low','close']:
            if c in df.columns: df.loc[df[c]==-1, c] = np.nan
        if 'amount' in df.columns: df['amount'] = df['amount'].astype('float64')/100.0
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).sort_values(['instrument_id','date']).reset_index(drop=True)

    instruments = sorted(df['instrument_id'].unique())
    if MAX_TRAIN_STOCKS > 0:
        instruments = instruments[:MAX_TRAIN_STOCKS]
        df = df[df['instrument_id'].isin(instruments)]
    print(f'[DATA] {len(instruments)} stocks, {len(df):,} rows ({time.time()-t0:.0f}s)', flush=True)

    for c in ['volume','amount']:
        if c in df.columns: df[c] = np.log1p(df[c].clip(lower=0))
    for c in FEATURE_COLS:
        if c not in df.columns: df[c] = 0.0

    print('[WINDOWS] Building...', flush=True); t0 = time.time()
    wins, ys, win_dates = [], [], []
    for ins, sub in df.groupby('instrument_id', sort=False):
        if len(sub) <= SEQ_LEN: continue
        sub = sub.sort_values('date')
        feats = pd.DataFrame(sub[FEATURE_COLS].to_numpy(np.float32)).ffill(limit=3).fillna(0.0).to_numpy(np.float32)
        day = sub['date'].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        cp = feats[:, 3].astype(np.float64)
        for k, p in enumerate(close_pos):
            if p + 1 < SEQ_LEN: continue
            # 标签: T+horizon日收益率
            label = None
            if k + horizon < len(close_pos) and cp[p] > 0:
                r = cp[close_pos[k+horizon]] / cp[p] - 1.0
                if np.isfinite(r):
                    label = np.float32(r)
            if label is None: continue
            wins.append(feats[p-SEQ_LEN+1:p+1])
            ys.append(label)
            win_dates.append(pd.Timestamp(day[p]).date())

    X = np.stack(wins).astype(np.float32); y = np.array(ys, np.float32)
    dates = np.array(win_dates)
    flat = X.reshape(-1, X.shape[-1])
    m, s = flat.mean(0), flat.std(0) + 1e-6
    X = ((X - m) / s).astype(np.float32)
    print(f'[WINDOWS] {len(X):,} samples, {len(np.unique(dates)):,} dates ({time.time()-t0:.0f}s)', flush=True)
    return X, y, dates, m, s


# ===== Date-aware Batcher =====
class DateBatcher:
    def __init__(self, X, y, dates, device='cpu'):
        self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
        self.device = device
        unique_dates = np.unique(dates)
        self.date_groups = {}
        for d in unique_dates:
            idx = np.where(dates == d)[0]
            if len(idx) >= 5: self.date_groups[d] = idx
        self.all_dates = sorted(self.date_groups.keys())

    def __len__(self):
        return max(1, len(self.all_dates) // DATES_PER_BATCH)

    def __iter__(self):
        date_order = np.random.permutation(self.all_dates)
        for i in range(0, len(date_order), DATES_PER_BATCH):
            bd = date_order[i:i+DATES_PER_BATCH]
            if len(bd) < 2: continue
            bs, br = [], []
            for d in bd:
                pool = self.date_groups[d]
                n_sample = min(STOCKS_PER_DATE, len(pool))
                sampled = np.random.choice(pool, n_sample, replace=False)
                bs.append(self.model(self.X[sampled].to(self.device)))
                br.append(self.y[sampled].to(self.device))
            yield bs, br

    def set_model(self, model): self.model = model


# ===== JSON save/load =====
def save_model(ckpt, path):
    sd = ckpt['state_dict']
    tensors = {k: {'dtype': str(v.detach().cpu().dtype).replace('torch.',''),
                    'shape': list(v.shape), 'data': v.detach().cpu().reshape(-1).tolist()}
               for k, v in sd.items()}
    payload = {k: v for k, v in ckpt.items() if k != 'state_dict'}
    payload['state_dict'] = tensors
    with open(path, 'w') as f: json.dump(payload, f, ensure_ascii=False)
    return path

def load_model(path=MODEL_PATH, map_location='cpu'):
    with open(path) as f: payload = json.load(f)
    sd = {}
    for k, meta in payload['state_dict'].items():
        sd[k] = torch.tensor(meta['data'], dtype=getattr(torch, meta['dtype'])).reshape(meta['shape']).to(map_location)
    return {k: v for k, v in payload.items() if k != 'state_dict'} | {'state_dict': sd}


def train_and_save(horizon=1):
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[DEVICE] {device} | v11 two-stage | T+{horizon}', flush=True)
    if device.type == 'cpu': torch.set_num_threads(8)

    X, y, dates, mean, std = load_data(horizon=horizon)
    y = np.clip(y, *np.percentile(y, [1, 99]))

    model = FastTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[MODEL] {n_params:,} params', flush=True)

    batcher = DateBatcher(X, y, dates, device=device)
    batcher.set_model(model)
    n_batches = len(batcher)
    print(f'[BATCH] {n_batches} batches/epoch ({len(batcher.all_dates):,} dates)', flush=True)

    # ===== 阶段1: IC基础训练 (15 epochs, 全参数) =====
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.2)
    model.train()
    best_ic = -1
    for ep in range(STAGE1_EPOCHS):
        t0 = time.time(); tl, tic, tir, tsr, tcv, tto, nb = 0,0,0,0,0,0,0
        for i, (scores_by_date, returns_by_date) in enumerate(batcher):
            opt.zero_grad()
            loss, l_ic, l_icir, l_sr, l_cvar, l_to = compute_total_loss(scores_by_date, returns_by_date)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl+=loss.item(); tic+=l_ic; tir+=l_icir; tsr+=l_sr; tcv+=l_cvar; tto+=l_to; nb+=1
        scheduler.step()
        ic = 1-tic/nb; sr = -tsr/nb
        if ic > best_ic: best_ic = ic
        print(f'[S1 EPOCH {ep+1}/{STAGE1_EPOCHS}] IC={ic:.4f} SR={sr:.2f} '
              f'TO={tto/nb:.4f} time={time.time()-t0:.0f}s', flush=True)

    print(f'[STAGE1 DONE] best_IC={best_ic:.4f}', flush=True)

    # ===== 阶段2: Soft Sharpe精调 (4 epochs, 只训head) =====
    # 用softmax权重替代hard top-k → 完全可导的Sharpe优化
    def soft_sharpe_loss(scores, returns, temp=0.1):
        port_rets = []
        for s, r in zip(scores, returns):
            w = F.softmax(s / temp, dim=0)        # softmax权重, sum=1
            w_zs = w - 1.0 / len(w)               # zero-sum
            port_rets.append((w_zs * r).sum())
        if len(port_rets) < 2: return torch.tensor(0.0, device=scores[0].device)
        pr = torch.stack(port_rets)
        return -(pr.mean() / (pr.std() + 1e-8))   # 负Sharpe=loss

    head_params = list(model.head.parameters())
    trainable = sum(p.numel() for p in head_params)
    print(f'[STAGE2] Only training head, trainable={trainable:,} params', flush=True)

    opt2 = torch.optim.Adam(head_params, lr=STAGE2_LR, weight_decay=1e-3)
    STAGE2_EPOCHS = 4
    for ep in range(STAGE2_EPOCHS):
        t0 = time.time(); tl, tic, tsr, nb = 0,0,0,0
        for i, (scores_by_date, returns_by_date) in enumerate(batcher):
            opt2.zero_grad()
            l_sr = soft_sharpe_loss(scores_by_date, returns_by_date)
            l_sr.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt2.step()
            tl+=l_sr.item(); nb+=1
            ics = [pearson_corr(s, r) for s, r in zip(scores_by_date, returns_by_date) if len(s)>=10]
            if ics: tic += (1 - torch.stack(ics).mean().item())
            tsr += (-l_sr.item())
        ic2 = 1-tic/nb; sr2 = tsr/nb
        print(f'[S2 EPOCH {ep+1}/{STAGE2_EPOCHS}] SR_loss={tl/nb:.4f} IC={ic2:.4f} '
              f'SoftSR={sr2:.2f} time={time.time()-t0:.0f}s', flush=True)
        if ic2 < best_ic * 0.75:
            print(f'[EARLY STOP] IC dropped {best_ic:.4f}→{ic2:.4f}', flush=True)
            break

    path = os.path.join(_HERE, f'transformer_model_H{horizon}.json')
    save_model({'state_dict': model.state_dict(),
                'model_cfg': dict(n_feat=N_FEAT, dim=DIM),
                'feature_cols': FEATURE_COLS, 'seq_len': SEQ_LEN,
                'horizon': horizon,
                'mean': mean.astype(np.float32).tolist(),
                'std': std.astype(np.float32).tolist()}, path)
    print(f'[DONE] {path}', flush=True)
    return path


# ===== 辅助函数 (供notebook导入) =====
def build_dataset(table, sd, ed, mode, instruments=None, stats=None, is_local=False, id_to_code=None):
    import structlog; logger = structlog.get_logger()
    t0 = time.time()
    sd_ts, ed_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    id_col = 'instrument_id' if is_local else 'instrument'
    try:
        import dai as dai_mod
        buf = (pd.to_datetime(sd) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        cols = ['date',id_col]+[c for c in FEATURE_COLS if c!='deal_number']
        result = dai_mod.query(f"SELECT {', '.join(cols)} FROM {table}", filters={'date':[buf,ed]})
        df = result.df()
    except Exception:
        data_dir = os.path.join(_HERE, '..', 'outputs', 'competition')
        files = sorted(glob.glob(os.path.join(data_dir, 'e2e_bar5m_*.parquet')))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if instruments is not None and id_col in df.columns: df = df[df[id_col].isin(instruments)]
    for c in ['volume','amount']:
        if c in df.columns: df[c] = np.log1p(df[c].clip(lower=0))
    for c in FEATURE_COLS:
        if c not in df.columns: df[c] = 0.0
    buf = pd.to_datetime(sd) - pd.Timedelta(days=60)
    df = df[(df['date']>=buf)&(df['date']<=pd.to_datetime(ed))].copy()
    wins, ys, keys = [], [], []
    for ins, sub in df.groupby(id_col, sort=False):
        if len(sub) <= SEQ_LEN: continue
        sub = sub.sort_values('date')
        feats = pd.DataFrame(sub[FEATURE_COLS].to_numpy(np.float32)).ffill(limit=3).fillna(0.0).to_numpy(np.float32)
        day = sub['date'].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:]!=day[:-1], True))
        cp = feats[:,3].astype(np.float64)
        for k, p in enumerate(close_pos):
            d = pd.Timestamp(day[p])
            if p+1<SEQ_LEN or d<sd_ts or d>ed_ts: continue
            label = None
            if k+1<len(close_pos) and cp[p]>0:
                r = cp[close_pos[k+1]]/cp[p]-1.0
                if np.isfinite(r): label=np.float32(r)
            if mode=='train' and label is None: continue
            wins.append(feats[p-SEQ_LEN+1:p+1])
            ys.append(label if label is not None else np.float32(0.0))
            keys.append((d, ins))
    if not keys: raise RuntimeError('build_dataset: no samples')
    X = np.stack(wins).astype(np.float32)
    if stats is None:
        flat = X.reshape(-1,X.shape[-1])
        stats = (np.nanmean(flat,0).astype(np.float32), np.nanstd(flat,0).astype(np.float32)+1e-6)
    m, s = stats; X = ((X-m)/s).astype(np.float32)
    return X, np.array(ys, np.float32) if mode=='train' else None, None if mode=='train' else pd.DataFrame(keys,columns=['date','instrument']), stats

def pool(sd, ed, table='bigalpha_2026_instruments'):
    try:
        import dai as dai_mod
        return dai_mod.query(f"SELECT DISTINCT instrument FROM {table}", filters={'date':[sd,ed]}).df()['instrument'].tolist()
    except Exception: return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=5)
    args = parser.parse_args()
    train_and_save(horizon=args.horizon)
