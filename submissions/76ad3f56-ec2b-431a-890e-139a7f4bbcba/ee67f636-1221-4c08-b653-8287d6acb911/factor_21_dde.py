"""
因子 21: DDE — 深度衰减指数 (Depth Decay Exponent)
=====================================================

经济逻辑:
bid/ask 各档深度沿档位指数的衰减率反映供需曲线的"厚度"形状。
衰减越快 → 深度主要集中在近档 → 市场缺乏远档保护 → 价格易受冲击。
买方衰减慢（深度均匀分布在远档）= 大单隐藏在远档等待成交 → 看涨。
卖方衰减慢 = 卖压隐藏在远档 → 看跌。

方法：
- 对每侧10档深度拟合指数衰减曲线 depth ~ exp(-λ × level)
- λ 越小 → 衰减越慢 → 深度越"厚实"
- DDE = bid_λ - ask_λ → bid衰减更慢(λ更小) → 买盘更厚实 → 看涨

方向: DDE 越大(即ask衰减快于bid) → 买盘相对厚实 → 预期正收益

AI应用说明:
大模型通过分析盘口微观结构文献中的指数衰减模型，
生成深度衰减因子公式，捕捉供需曲线形状信息。
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    levels = np.arange(1, 11, dtype=float)
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        dde_vals = []
        
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            if bid_vols.sum() < 100 or ask_vols.sum() < 100:
                continue
            
            bid_lambda = _exp_decay_rate(levels, bid_vols)
            ask_lambda = _exp_decay_rate(levels, ask_vols)
            
            if bid_lambda is not None and ask_lambda is not None:
                # bid衰减率小 → 买盘厚; ask衰减率大 → 卖盘薄 → DDE > 0 = 看涨
                dde_vals.append(ask_lambda - bid_lambda)
        
        factor = np.mean(dde_vals) if dde_vals else 0.0
        results.append({'date': dt, 'instrument': inst, 'factor': factor})
    
    result = pd.DataFrame(results)
    return _normalize(result)


def _exp_decay_rate(levels, volumes):
    """估计指数衰减率 λ (depth ~ exp(-λ × level))"""
    v = volumes.copy()
    v[v < 1] = 1e-8
    log_v = np.log(v)
    log_v0 = np.log(v[0])
    
    # 线性回归 log(v) ~ -λ × level + C
    # λ = -Cov(level, log(v)) / Var(level)
    valid = np.isfinite(log_v)
    if valid.sum() < 3:
        return None
    
    x = levels[valid] - 1  # 从0开始
    y = log_v[valid]
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    
    if den < 1e-8:
        return None
    
    # 斜率 ≈ -λ
    slope = num / den
    lam = -slope
    
    return max(lam, 0.0)


def _normalize(result):
    result = result.copy()
    for date in result['date'].unique():
        m = result['date'] == date
        vals = result.loc[m, 'factor'].values
        if len(vals) < 5:
            continue
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med)) * 1.4826
        if mad < 1e-12:
            continue
        vals = np.clip(vals, med - 5 * mad, med + 5 * mad)
        result.loc[m, 'factor'] = (vals - np.nanmean(vals)) / np.nanstd(vals)
    return result[['date', 'instrument', 'factor']]
