import gc
import json
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
TRAIN_FEATURE_TABLE = 'bigalpha_2026_stock_bar1m'

class _Cfg:
    pass

def make_config(smoke):
    c = _Cfg()
    c.SMOKE = smoke
    c.T, c.SEED = (240, 42)
    c.TRAIN_TABLE = TRAIN_FEATURE_TABLE
    c.MODE = 'multitask_patchtst'
    c.OUTPUT_HEAD = 'oo'
    c.CC_AUX_WEIGHT = 0.3
    c.TCN_KERNEL, c.TCN_DILATIONS, c.TCN_LEVELS = (3, [1, 2, 4, 8, 16], 5)
    if smoke:
        c.TRAIN_START, c.TRAIN_END = ('2024-01-02', '2024-02-09')
        c.VAL_START, c.VAL_END = ('2024-02-12', '2024-02-29')
        c.LABEL_DATA_END = '2024-03-15'
        c.DAY_SAMPLE, c.MIN_BARS, c.MIN_XSEC = (1, 30, 8)
        c.D_MODEL, c.N_HEADS, c.N_LAYERS, c.PATCH = (16, 2, 1, 16)
        c.DROPOUT, c.EPOCHS, c.PATIENCE, c.INDUSTRY_IN_X = (0.1, 2, 2, False)
    else:
        c.TRAIN_START, c.TRAIN_END = ('2019-01-01', '2023-12-31')
        c.VAL_START, c.VAL_END = ('2024-01-01', '2024-12-27')
        c.LABEL_DATA_END = '2024-12-31'
        c.DAY_SAMPLE, c.MIN_BARS, c.MIN_XSEC = (1, 60, 60)
        c.D_MODEL, c.N_HEADS, c.N_LAYERS, c.PATCH = (64, 4, 2, 16)
        c.DROPOUT, c.EPOCHS, c.PATIENCE, c.INDUSTRY_IN_X = (0.2, 15, 3, True)
    c.LR, c.WD, c.LAMBDA_MSE = (0.001, 0.01, 0.1)
    c.BEST_CC_IC = float('nan')
    return c
FEATURES = ['ret_1m', 'legacy_vol_share', 'legacy_amt_share', 'legacy_tn_share', 'legacy_rel_trade_size', 'increment_vol_share', 'increment_amt_share', 'increment_tn_share', 'increment_rel_trade_size', 'ofi_l1', 'ofi_l5', 'depth_imb', 'depth_imb_l1', 'ord_imb', 'rel_spread', 'microprice_dev', 'slope_bid', 'slope_ask', 'close_in_range', 'intraday_cumret']

def build_pool_sql(table, d0, d1):
    return f"\n        SELECT DISTINCT date::DATE AS pd, instrument\n        FROM bigalpha_2026_instruments\n        WHERE date::DATE BETWEEN DATE '{d0}' AND DATE '{d1}'\n        "

def build_feature_sql(table, d0, d1, T):
    pool_sql = build_pool_sql(table, d0, d1)
    return f"\n    WITH pool AS (\n        {pool_sql}\n    ),\n    base AS (\n        SELECT\n            t.instrument, t.date, t.date::DATE AS d,\n            close, high, low, open,\n            volume::DOUBLE AS volume, amount::DOUBLE AS amount,\n            deal_number::DOUBLE AS deal_number,\n            ask_price1, bid_price1, ask_price5, bid_price5,\n            ask_volume1::DOUBLE AS av1, bid_volume1::DOUBLE AS bv1,\n            (bid_volume1+bid_volume2+bid_volume3+bid_volume4+bid_volume5)::DOUBLE AS bidv5,\n            (ask_volume1+ask_volume2+ask_volume3+ask_volume4+ask_volume5)::DOUBLE AS askv5,\n            (bid_num_orders1+bid_num_orders2+bid_num_orders3+bid_num_orders4+bid_num_orders5)::DOUBLE AS bidn5,\n            (ask_num_orders1+ask_num_orders2+ask_num_orders3+ask_num_orders4+ask_num_orders5)::DOUBLE AS askn5\n        FROM {table} t\n        INNER JOIN pool p ON p.instrument = t.instrument AND p.pd = t.date::DATE\n        WHERE t.date::DATE BETWEEN DATE '{d0}' AND DATE '{d1}'\n    ),\n    lagged AS (\n        SELECT *,\n            LAG(close)       OVER w AS close_p,\n            LAG(volume)      OVER w AS volume_p,\n            LAG(amount)      OVER w AS amount_p,\n            LAG(deal_number) OVER w AS deal_number_p,\n            LAG(bid_price1)  OVER w AS bp1_p,\n            LAG(ask_price1)  OVER w AS ap1_p,\n            LAG(bv1)         OVER w AS bv1_p,\n            LAG(av1)         OVER w AS av1_p,\n            LAG(bidv5)       OVER w AS bidv5_p,\n            LAG(askv5)       OVER w AS askv5_p,\n            FIRST_VALUE(open) OVER w_day AS day_open,\n            MAX(volume)       OVER (PARTITION BY instrument, d) AS peak_volume,\n            MAX(amount)       OVER (PARTITION BY instrument, d) AS peak_amount,\n            MAX(deal_number)  OVER (PARTITION BY instrument, d) AS peak_deal_number,\n            SUM(GREATEST(volume, 0))      OVER (PARTITION BY instrument, d) AS sum_volume,\n            SUM(GREATEST(amount, 0))      OVER (PARTITION BY instrument, d) AS sum_amount,\n            SUM(GREATEST(deal_number, 0)) OVER (PARTITION BY instrument, d) AS sum_deal_number,\n            ROW_NUMBER() OVER (PARTITION BY instrument, d ORDER BY date DESC) AS rn_desc\n        FROM base\n        WINDOW\n            w AS (PARTITION BY instrument, d ORDER BY date),\n            w_day AS (PARTITION BY instrument, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n    ),\n    flow AS (\n        SELECT *,\n            GREATEST(volume - COALESCE(volume_p, 0), 0) AS legacy_volume,\n            GREATEST(amount - COALESCE(amount_p, 0), 0) AS legacy_amount,\n            GREATEST(deal_number - COALESCE(deal_number_p, 0), 0) AS legacy_deal_number,\n            GREATEST(volume, 0) AS increment_volume,\n            GREATEST(amount, 0) AS increment_amount,\n            GREATEST(deal_number, 0) AS increment_deal_number,\n            (ask_price1 + bid_price1) / 2.0 AS mid\n        FROM lagged\n        WHERE rn_desc <= {T}\n    ),\n    feat AS (\n        SELECT\n            instrument, d, ({T} - rn_desc)::INT AS pos,\n            COALESCE(ln(NULLIF(close, 0) / NULLIF(close_p, 0)), 0) AS ret_1m,\n            COALESCE(legacy_volume / NULLIF(peak_volume, 0), 0) AS legacy_vol_share,\n            COALESCE(legacy_amount / NULLIF(peak_amount, 0), 0) AS legacy_amt_share,\n            COALESCE(legacy_deal_number / NULLIF(peak_deal_number, 0), 0) AS legacy_tn_share,\n            CASE WHEN legacy_volume > 0 AND legacy_deal_number > 0\n                       AND peak_volume > 0 AND peak_deal_number > 0\n                 THEN ln((legacy_volume / legacy_deal_number) / (peak_volume / peak_deal_number))\n                 ELSE 0 END AS legacy_rel_trade_size,\n            COALESCE(increment_volume / NULLIF(sum_volume, 0), 0) AS increment_vol_share,\n            COALESCE(increment_amount / NULLIF(sum_amount, 0), 0) AS increment_amt_share,\n            COALESCE(increment_deal_number / NULLIF(sum_deal_number, 0), 0) AS increment_tn_share,\n            CASE WHEN increment_volume > 0 AND increment_deal_number > 0\n                       AND sum_volume > 0 AND sum_deal_number > 0\n                 THEN ln((increment_volume / increment_deal_number) / (sum_volume / sum_deal_number))\n                 ELSE 0 END AS increment_rel_trade_size,\n            COALESCE(((CASE WHEN bid_price1 >= bp1_p THEN bv1 ELSE 0 END)\n                    - (CASE WHEN bid_price1 <= bp1_p THEN bv1_p ELSE 0 END)\n                    - (CASE WHEN ask_price1 <= ap1_p THEN av1 ELSE 0 END)\n                    + (CASE WHEN ask_price1 >= ap1_p THEN av1_p ELSE 0 END))\n                    / NULLIF(bv1 + av1 + bv1_p + av1_p, 0), 0) AS ofi_l1,\n            COALESCE(((bidv5 - bidv5_p) - (askv5 - askv5_p))\n                    / NULLIF(bidv5 + askv5 + bidv5_p + askv5_p, 0), 0) AS ofi_l5,\n            COALESCE((bidv5 - askv5) / NULLIF(bidv5 + askv5, 0), 0) AS depth_imb,\n            COALESCE((bv1 - av1) / NULLIF(bv1 + av1, 0), 0) AS depth_imb_l1,\n            COALESCE((bidn5 - askn5) / NULLIF(bidn5 + askn5, 0), 0) AS ord_imb,\n            COALESCE((ask_price1 - bid_price1) / NULLIF(mid, 0), 0) AS rel_spread,\n            COALESCE((ask_price1 * bv1 + bid_price1 * av1)\n                    / NULLIF(bv1 + av1, 0) / NULLIF(mid, 0) - 1, 0) AS microprice_dev,\n            COALESCE((bid_price1 - bid_price5) / NULLIF(mid, 0), 0) AS slope_bid,\n            COALESCE((ask_price5 - ask_price1) / NULLIF(mid, 0), 0) AS slope_ask,\n            CASE WHEN high > low THEN (close - low) / (high - low) ELSE 0.5 END AS close_in_range,\n            COALESCE(ln(NULLIF(close, 0) / NULLIF(day_open, 0)), 0) AS intraday_cumret\n        FROM flow\n    )\n    SELECT instrument, d, pos,\n           {', '.join((f'{column}::FLOAT AS {column}' for column in FEATURES))}\n    FROM feat\n    "
STYLES = ['SIZE', 'BETA', 'MOMENTUM', 'RESVOL', 'SIZENL', 'BTOP', 'LIQUIDTY', 'EARNYILD', 'GROWTH', 'LEVERAGE']
INDUSTRIES = ['AGRIFOREST', 'MINING', 'CHEM', 'IRONSTEEL', 'NONFERMETAL', 'ELECTRONICS', 'AUTO', 'HOUSEAPP', 'FOODBEVER', 'TEXTILE', 'LIGHTINDUS', 'HEALTH', 'UTILITIES', 'TRANSPORTATION', 'REALESTATE', 'COMMETRADE', 'LEISERVICE', 'BANK', 'NONBANKFINAN', 'CONGLOMERATES', 'CONMAT', 'BUILDDECO', 'ELECEQP', 'MACHIEQUIP', 'AERODEF', 'COMPUTER', 'MEDIA', 'TELECOM', 'COAL', 'PETRO', 'ENVP', 'BEAUTY']

def _rank_gaussian(values):
    n = len(values)
    ranks = pd.Series(values).rank(method='average').to_numpy(np.float64)
    uniforms = (ranks - 0.5) / n
    return (torch.erfinv(torch.tensor(2.0 * uniforms - 1.0, dtype=torch.float64)) * math.sqrt(2.0)).numpy().astype(np.float32)

def build_targets(dai_mod, cfg):
    label_end = pd.Timestamp(cfg.LABEL_DATA_END).strftime('%Y-%m-%d')
    xcols = STYLES + (INDUSTRIES if cfg.INDUSTRY_IN_X else [])
    exposure = dai_mod.query('SELECT date, instrument, weights, ' + ', '.join(xcols) + ' FROM bigalpha_2026_exposure' + f" WHERE date::DATE BETWEEN DATE '{cfg.TRAIN_START}' AND DATE '{label_end}'", filters={'date': [cfg.TRAIN_START, label_end + ' 23:59:59']}, compression=True).df()
    exposure['d'] = pd.to_datetime(exposure['date']).dt.normalize()
    returns = _fetch_returns(dai_mod, cfg.TRAIN_START, label_end + ' 23:59:59', cfg.TRAIN_TABLE)
    merged = exposure.merge(returns, on=['d', 'instrument'], how='inner')
    merged = merged[merged['d'] <= pd.Timestamp(cfg.VAL_END)]
    output = []
    for day, group in merged.groupby('d'):
        group = group.dropna(subset=['ret_oo', 'ret_cc'])
        if len(group) < cfg.MIN_XSEC:
            continue
        n = len(group)
        x = np.column_stack([np.ones(n), group[xcols].to_numpy(np.float64)])
        weights = group['weights'].to_numpy(np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0 / n)
        root_weight = np.sqrt(weights)
        values = {'d': day, 'instrument': group['instrument'].to_numpy()}
        for label in ('oo', 'cc'):
            y = group[f'ret_{label}'].to_numpy(np.float64)
            beta = np.linalg.lstsq(x * root_weight[:, None], y * root_weight, rcond=None)[0]
            values[f'y_{label}'] = _rank_gaussian(y - x @ beta)
        output.append(pd.DataFrame(values))
    if not output:
        raise ValueError('no valid multitask targets')
    return pd.concat(output, ignore_index=True)

def month_chunks(d0, d1):
    s, e = (pd.Timestamp(d0), pd.Timestamp(d1))
    out = []
    cur = s
    while cur <= e:
        me = min(cur + pd.offsets.MonthEnd(0), e)
        if me < cur:
            me = min(cur + pd.offsets.MonthEnd(1), e)
        out.append((cur.strftime('%Y-%m-%d'), me.strftime('%Y-%m-%d')))
        cur = me + pd.Timedelta(days=1)
    return out

def frame_to_tensors(df, cfg):
    inst_codes, inst_uniq = pd.factorize(df['instrument'])
    day_codes, day_uniq = pd.factorize(pd.to_datetime(df['d']).values.astype('datetime64[D]'))
    combo = inst_codes.astype(np.int64) * len(day_uniq) + day_codes
    codes, uniq_combo = pd.factorize(combo)
    feats = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    np.clip(feats, -10.0, 10.0, out=feats)
    x = np.zeros((len(uniq_combo), cfg.T, len(FEATURES)), dtype=np.float16)
    x[codes, df['pos'].to_numpy(dtype=np.int64)] = feats.astype(np.float16)
    cnt = np.bincount(codes, minlength=len(uniq_combo))
    keep = cnt >= cfg.MIN_BARS
    ci = (np.asarray(uniq_combo) // len(day_uniq)).astype(np.int64)
    cd = (np.asarray(uniq_combo) % len(day_uniq)).astype(np.int64)
    keys = pd.DataFrame({'instrument': np.asarray(inst_uniq)[ci], 'd': pd.to_datetime(day_uniq[cd])})
    return (keys[keep].reset_index(drop=True), x[keep])

def fetch_features(dai_mod, table, d0, d1, cfg, day_sample):
    try:
        import psutil

        def _avail():
            return f'{psutil.virtual_memory().available / 2 ** 30:.1f}GB'
    except Exception:

        def _avail():
            return 'n/a'
    keys_all, xs = ([], [])
    chunks = month_chunks(d0, d1)
    for i, (cs, ce) in enumerate(chunks):
        sql = build_feature_sql(table, cs, ce, cfg.T)
        df = dai_mod.query(sql, filters={'date': [cs, ce + ' 23:59:59']}, compression=True).df()
        if day_sample > 1 and len(df):
            epoch_day = pd.to_datetime(df['d']).astype('int64') // 86400000000000
            df = df[epoch_day % day_sample == 0]
        if len(df) == 0:
            continue
        k, a = frame_to_tensors(df, cfg)
        del df
        gc.collect()
        k['chunk'] = len(xs)
        k['row'] = np.arange(len(k))
        keys_all.append(k)
        xs.append(a)
        print(f'[f002] 特征分块 {i + 1}/{len(chunks)} ({cs}..{ce}): {len(k)} stock-day | 剩余内存 {_avail()}')
    if not keys_all:
        raise ValueError(f'fetch_features: [{d0},{d1}] 无数据(表 {table})')
    return (pd.concat(keys_all, ignore_index=True), xs)

class PatchTSTEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.patch = cfg.PATCH
        self.n_patch = cfg.T // cfg.PATCH
        self.embed = nn.Linear(cfg.PATCH, cfg.D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, cfg.D_MODEL))
        layer = nn.TransformerEncoderLayer(cfg.D_MODEL, cfg.N_HEADS, cfg.D_MODEL * 2, cfg.DROPOUT, batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.N_LAYERS)
        self.mix = nn.Linear(len(FEATURES) * cfg.D_MODEL, cfg.D_MODEL)
        self.norm = nn.LayerNorm(cfg.D_MODEL)

    def forward(self, x):
        b, _t, f = x.shape
        h = x.permute(0, 2, 1).reshape(b * f, self.n_patch, self.patch)
        h = self.enc(self.embed(h) + self.pos).mean(dim=1).reshape(b, -1)
        return self.norm(torch.nn.functional.gelu(self.mix(h)))

class FactorModel(nn.Module):

    def __init__(self, name, cfg):
        super().__init__()
        if name != 'multitask_patchtst':
            raise ValueError(f'f002_multitask_oo_cc requires multitask_patchtst, got {name!r}')
        self.encoder = PatchTSTEncoder(cfg)
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.oo_head = nn.Linear(cfg.D_MODEL, 1)
        self.cc_head = nn.Linear(cfg.D_MODEL, 1)

    def forward_heads(self, x):
        hidden = self.dropout(self.encoder(x))
        return (self.oo_head(hidden).squeeze(-1), self.cc_head(hidden).squeeze(-1))

    def forward(self, x):
        oo, _cc = self.forward_heads(x)
        return oo

def make_day_batches(keys, targets, cfg):
    merged = keys.merge(targets, on=['instrument', 'd'], how='inner')
    batches = {}
    for day, group in merged.groupby('d'):
        if len(group) < cfg.MIN_XSEC:
            continue
        chunks = group['chunk'].unique()
        if len(chunks) != 1:
            raise ValueError(f'day {day} spans chunks: {chunks}')
        batches[day] = (int(chunks[0]), group['row'].to_numpy(), group[['y_oo', 'y_cc']].to_numpy(np.float32, copy=True))
    if not batches:
        raise ValueError('no multitask day batches')
    return batches

def rank_ic_loss(pred, target, lam):
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    ic = (pred_centered * target_centered).sum() / (pred_centered.norm() * target_centered.norm() + 1e-08)
    return -ic + lam * torch.mean((pred - target) ** 2)

def evaluate(model, arrays, batches, cfg, device):
    model.eval()
    oo_values, cc_values = ([], [])
    with torch.no_grad():
        for day in sorted(batches):
            chunk, rows, target = batches[day]
            x = torch.from_numpy(arrays[chunk][rows].astype(np.float32)).to(device)
            pred_oo, pred_cc = model.forward_heads(x)
            oo_values.append(pd.Series(pred_oo.cpu().numpy()).rank().corr(pd.Series(target[:, 0]).rank()))
            cc_values.append(pd.Series(pred_cc.cpu().numpy()).rank().corr(pd.Series(target[:, 1]).rank()))
    return (float(np.nanmean(oo_values)), float(np.nanmean(cc_values)))

def train_model(name, arrays, batches_train, batches_valid, cfg, device, ckpt_fn=None):
    torch.manual_seed(cfg.SEED)
    model = FactorModel(name, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    days = sorted(batches_train)
    rng = np.random.default_rng(cfg.SEED)
    best_oo, best_state, bad = (float('-inf'), None, 0)
    for epoch in range(cfg.EPOCHS):
        model.train()
        for index in rng.permutation(len(days)):
            chunk, rows, target = batches_train[days[index]]
            x = torch.from_numpy(arrays[chunk][rows].astype(np.float32)).to(device)
            y = torch.from_numpy(target).to(device)
            pred_oo, pred_cc = model.forward_heads(x)
            loss = rank_ic_loss(pred_oo, y[:, 0], cfg.LAMBDA_MSE)
            loss = loss + cfg.CC_AUX_WEIGHT * rank_ic_loss(pred_cc, y[:, 1], cfg.LAMBDA_MSE)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_oo, validation_cc = evaluate(model, arrays, batches_valid, cfg, device)
        print(f'[f002_multitask] epoch={epoch + 1} OO={validation_oo:.6f} CC={validation_cc:.6f}', flush=True)
        if np.isfinite(validation_oo) and validation_oo > best_oo:
            best_oo = validation_oo
            cfg.BEST_CC_IC = validation_cc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
            if ckpt_fn is not None:
                ckpt_fn(best_state, best_oo, epoch + 1)
        else:
            bad += 1
            if bad >= cfg.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, best_oo)

DEFAULT_TRAIN_TABLE = "bigalpha_2026_stock_bar1m"


def _validate_train_table(table):
    if table != DEFAULT_TRAIN_TABLE:
        raise ValueError(
            "this reference is frozen to the documented training minute table: "
            + DEFAULT_TRAIN_TABLE
        )
    return table


def _fetch_daily_prices(dai_mod, table, start, end):
    table = _validate_train_table(table)
    start_day = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_day = pd.Timestamp(end).strftime("%Y-%m-%d")
    sql = f"""
    WITH ranked AS (
        SELECT
            date::DATE AS d,
            instrument,
            open,
            close,
            pre_close,
            ROW_NUMBER() OVER (
                PARTITION BY instrument, date::DATE ORDER BY date
            ) AS first_row,
            ROW_NUMBER() OVER (
                PARTITION BY instrument, date::DATE ORDER BY date DESC
            ) AS last_row
        FROM {table}
        WHERE date::DATE BETWEEN DATE '{start_day}' AND DATE '{end_day}'
    )
    SELECT
        d,
        instrument,
        MAX(CASE WHEN first_row = 1 THEN open END)::DOUBLE AS open,
        MAX(CASE WHEN last_row = 1 THEN close END)::DOUBLE AS close,
        MAX(CASE WHEN first_row = 1 THEN pre_close END)::DOUBLE AS pre_close
    FROM ranked
    GROUP BY d, instrument
    """
    frame = dai_mod.query(
        sql,
        filters={"date": [start_day, end_day + " 23:59:59"]},
        compression=True,
    ).df()
    frame["d"] = pd.to_datetime(frame["d"]).dt.normalize()
    for column in ("open", "close", "pre_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["instrument", "d"]).reset_index(drop=True)


def fetch_fwd_cc(dai_mod, start, end, table=DEFAULT_TRAIN_TABLE):
    """Rebuild the adjacent adjusted close return from the minute table."""
    prices = _fetch_daily_prices(dai_mod, table, start, end)
    grouped = prices.groupby("instrument", sort=False)
    next_close = grouped["close"].shift(-1)
    next_pre_close = grouped["pre_close"].shift(-1)
    prices["fwd_ret"] = next_close / next_pre_close.replace(0.0, np.nan) - 1.0
    return prices[["d", "instrument", "fwd_ret"]].dropna(subset=["fwd_ret"])


def fetch_fwd_oo(dai_mod, start, end, table=DEFAULT_TRAIN_TABLE):
    """Rebuild the adjusted open-to-open return without a separate daily table."""
    prices = _fetch_daily_prices(dai_mod, table, start, end)
    grouped = prices.groupby("instrument", sort=False)
    next_open = grouped["open"].shift(-1)
    second_open = grouped["open"].shift(-2)
    next_close = grouped["close"].shift(-1)
    second_pre_close = grouped["pre_close"].shift(-2)
    adjustment = next_close / second_pre_close.replace(0.0, np.nan)
    prices["fwd_ret"] = (
        second_open / next_open.replace(0.0, np.nan) * adjustment - 1.0
    )
    return prices[["d", "instrument", "fwd_ret"]].dropna(subset=["fwd_ret"])


def _fetch_returns(dai_mod, start, end, table=DEFAULT_TRAIN_TABLE):
    oo = fetch_fwd_oo(dai_mod, start, end, table).rename(
        columns={"fwd_ret": "raw_oo"}
    )
    cc = fetch_fwd_cc(dai_mod, start, end, table).rename(
        columns={"fwd_ret": "raw_cc"}
    )
    return oo.merge(cc, on=["d", "instrument"], how="inner")


FACTOR_NAME = 'f002_multitask_oo_cc_cc_head'
DEFAULT_ARCH = 'multitask_patchtst'
OUTPUT_HEAD = 'cc'
DEFAULT_TRAIN_START = '2019-01-01'
DEFAULT_TRAIN_END = '2023-12-31'
DEFAULT_VALIDATION_START = '2024-01-01'
DEFAULT_VALIDATION_END = '2024-12-27'
DEFAULT_LABEL_DATA_END = '2024-12-31'
DEFAULT_RANDOM_SEED = 42


def _export_config(cfg):
    fields = ('T', 'D_MODEL', 'N_HEADS', 'N_LAYERS', 'PATCH', 'TCN_KERNEL', 'TCN_DILATIONS', 'TCN_LEVELS', 'DROPOUT', 'N_FEATURES', 'MIN_BARS', 'MIN_XSEC', 'SMOOTH_DAYS', 'CC_AUX_WEIGHT')
    payload = {}
    for name in fields:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if isinstance(value, tuple):
                value = list(value)
            payload[name] = value
    payload.setdefault("N_FEATURES", len(FEATURES))
    payload.setdefault("SMOOTH_DAYS", 1)
    return payload


def _write_trained_payload(arrays, keys, targets, cfg, output_path):
    if not isinstance(arrays, (list, tuple)) or not arrays:
        raise ValueError("arrays must be a non-empty sequence of feature tensors")
    required_keys = {"instrument", "d", "chunk", "row"}
    if not required_keys.issubset(keys.columns):
        raise ValueError("keys must contain instrument, d, chunk and row")
    batches = make_day_batches(keys, targets, cfg)
    train_left, train_right = pd.Timestamp(cfg.TRAIN_START), pd.Timestamp(cfg.TRAIN_END)
    valid_left, valid_right = pd.Timestamp(cfg.VAL_START), pd.Timestamp(cfg.VAL_END)
    train_batches = {day: value for day, value in batches.items() if train_left <= pd.Timestamp(day) <= train_right}
    valid_batches = {day: value for day, value in batches.items() if valid_left <= pd.Timestamp(day) <= valid_right}
    if not train_batches or not valid_batches:
        raise ValueError("training and validation batches must both be non-empty")
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, validation_score = train_model(
        DEFAULT_ARCH, arrays, train_batches, valid_batches, cfg, device
    )
    payload = {
        "schema_version": 1,
        "factor_name": FACTOR_NAME,
        "arch": DEFAULT_ARCH,
        "features": FEATURES,
        "config": _export_config(cfg),
        "state_dict": {
            name: tensor.detach().cpu().float().numpy().tolist()
            for name, tensor in model.state_dict().items()
        },
        "validation_score": float(validation_score),
        "_training": {
            "feature_table": cfg.TRAIN_TABLE,
            "train_start": cfg.TRAIN_START,
            "train_end": cfg.TRAIN_END,
            "validation_start": cfg.VAL_START,
            "validation_end": cfg.VAL_END,
            "label_data_end": getattr(cfg, "LABEL_DATA_END", None),
            "random_seed": int(cfg.SEED),
            "default_contract": True,
        },
    }
    if OUTPUT_HEAD:
        payload["output_head"] = OUTPUT_HEAD
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return payload


def _configured_contract(smoke=False, train_table=DEFAULT_TRAIN_TABLE,
                         train_start=None, train_end=None,
                         validation_start=None, validation_end=None,
                         label_data_end=None):
    _validate_train_table(train_table)
    cfg = make_config(smoke)
    if not smoke:
        observed = (
            cfg.TRAIN_START,
            cfg.TRAIN_END,
            cfg.VAL_START,
            cfg.VAL_END,
            getattr(cfg, "LABEL_DATA_END", None),
            int(cfg.SEED),
        )
        expected = (
            DEFAULT_TRAIN_START,
            DEFAULT_TRAIN_END,
            DEFAULT_VALIDATION_START,
            DEFAULT_VALIDATION_END,
            DEFAULT_LABEL_DATA_END,
            DEFAULT_RANDOM_SEED,
        )
        if observed != expected:
            raise RuntimeError(
                f"make_config drifted from the frozen default contract: {observed} != {expected}"
            )
    cfg.TRAIN_TABLE = train_table
    if train_start is not None:
        cfg.TRAIN_START = train_start
    if train_end is not None:
        cfg.TRAIN_END = train_end
    if validation_start is not None:
        cfg.VAL_START = validation_start
    if validation_end is not None:
        cfg.VAL_END = validation_end
    if label_data_end is not None:
        cfg.LABEL_DATA_END = label_data_end
    return cfg


def train_prepared(arrays, keys, targets,
                   train_start=DEFAULT_TRAIN_START,
                   train_end=DEFAULT_TRAIN_END,
                   validation_start=DEFAULT_VALIDATION_START,
                   validation_end=DEFAULT_VALIDATION_END,
                   output_path="f002_weights.json", smoke=False):
    """Train from prepared tensors while retaining the frozen model defaults."""
    cfg = _configured_contract(
        smoke=smoke,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    return _write_trained_payload(arrays, keys, targets, cfg, output_path)


def train_default(train_table=DEFAULT_TRAIN_TABLE,
                  train_start=None, train_end=None,
                  validation_start=None, validation_end=None,
                  label_data_end=None,
                  output_path="f002_weights.json", smoke=False):
    """Run the original frozen training contract unless an override is explicit."""
    import dai

    cfg = _configured_contract(
        smoke=smoke,
        train_table=train_table,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        label_data_end=label_data_end,
    )
    print(
        f"[{FACTOR_NAME}] table={cfg.TRAIN_TABLE} "
        f"train={cfg.TRAIN_START}..{cfg.TRAIN_END} "
        f"validation={cfg.VAL_START}..{cfg.VAL_END} seed={cfg.SEED}",
        flush=True,
    )
    keys, arrays = fetch_features(
        dai,
        cfg.TRAIN_TABLE,
        cfg.TRAIN_START,
        cfg.VAL_END,
        cfg,
        cfg.DAY_SAMPLE,
    )
    targets = build_targets(dai, cfg)
    return _write_trained_payload(arrays, keys, targets, cfg, output_path)


def main():
    return train_default()


if __name__ == "__main__":
    main()
