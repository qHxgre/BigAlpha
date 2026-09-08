"""
因子 45: PMV — 价格动量×成交量验证 (Price Momentum × Volume Validation)
==========================================================================

经济逻辑:
日内价格动量 × 成交量激增比率捕捉动量的可信度。
- 动量向上 + 成交量放大 → 买盘有量支撑 → 可信看涨
- 动量向上 + 成交量萎缩 → 无量空涨 → 不可信 → 可能反转
- 动量向下 + 成交量放大 → 真实卖压 → 看跌
- 动量向下 + 成交量萎缩 → 无量空跌 → 可能反弹

方法：
- intraday_mom = (mid_final - mid_initial) / mid_initial
- volume_surge = 后段平均成交量 / 前段平均成交量
- PMV = mom × (volume_surge - 1) 的正负号处理

方向: PMV 越大 → 价格涨且有量支撑 → 预期正收益

AI应用说明:
技术分析中"量价配合"原则的严格量化。
LLM提示: "涨跌有没有量的配合？"。
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        
        if len(grp) < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        mids = []
        volumes = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            if pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0:
                mids.append((ap1 + bp1) / 2)
            else:
                if mids:
                    mids.append(mids[-1])
                else:
                    mids.append(np.nan)
            
            # 成交量代理
            vol = row.get('volume', row.get('num_trades', 0)) or 0
            volumes.append(vol)
        
        mids = np.array(mids)
        volumes = np.array(volumes, dtype=float)
        
        valid = ~np.isnan(mids)
        if valid.sum() < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        mp = mids[valid]
        vol = volumes[valid]
        
        n = len(mp)
        n_half = n // 2
        
        # 价格动量
        mid_start = np.mean(mp[:max(n_half // 2, 2)])
        mid_end = np.mean(mp[-max(n_half // 2, 2):])
        mom = (mid_end - mid_start) / (mid_start + 1e-8)
        
        # 成交量激增
        vol_first = np.mean(vol[:n_half])
        vol_last = np.mean(vol[n_half:])
        vol_surge = vol_last / (vol_first + 1e-8)
        
        # PMV: 量价配合
        # 涨+量增 = 正; 跌+量增 = 负; 涨+量缩 = 中性偏负; 跌+量缩 = 中性偏正
        pmv = mom * (vol_surge - 1)
        
        results.append({'date': dt, 'instrument': inst, 'factor': pmv})
    
    result = pd.DataFrame(results)
    return _normalize(result)


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
