"""
因子 06: MSP — 微观结构压力 (Microstructure Pressure)
======================================================

经济逻辑:
对比加权平均委买价(bid_avg_price)与加权平均委卖价(ask_avg_price)
的变化速度差。委买价向上追赶委卖价 = 买方主动 = 看涨。

方向: MSP > 0 → 买价追卖价 → 买方主导 → 预期正收益

AI技术: LLM生成公式 → 硬编码提交
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        
        if 'bid_avg_price' not in grp.columns or 'ask_avg_price' not in grp.columns:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        bid_avg = grp['bid_avg_price'].dropna().values
        ask_avg = grp['ask_avg_price'].dropna().values
        
        if len(bid_avg) < 10 or len(ask_avg) < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 买价变化速度 vs 卖价变化速度
        bid_chg = np.diff(bid_avg) / (bid_avg[:-1] + 1e-8)
        ask_chg = np.diff(ask_avg) / (ask_avg[:-1] + 1e-8)
        
        # 累积: 买价累计涨幅 - 卖价累计涨幅
        bid_cum = np.sum(bid_chg)
        ask_cum = np.sum(ask_chg)
        
        # 正向 = 买价追卖价
        factor = bid_cum - ask_cum
        
        results.append({'date': dt, 'instrument': inst, 'factor': factor})
    
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
