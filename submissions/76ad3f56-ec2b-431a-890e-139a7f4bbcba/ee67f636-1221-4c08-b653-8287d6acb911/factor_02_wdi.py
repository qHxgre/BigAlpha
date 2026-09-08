"""
因子 02: WDI — 加权深度不平衡 (Weighted Depth Imbalance)
========================================================

经济逻辑:
10档盘口中，近档(1-3档)反映即时供需，远档(7-10档)反映隐藏意图。
大单藏在远档可能意味着知情交易者不愿暴露意图。
WDI 对各档按衰减权重加权，捕捉"隐藏的"供需失衡。

方向: WDI > 0 → 买盘整体力量 > 卖盘 → 预期正收益

AI技术: LLM生成公式 → 硬编码提交 (Prompt Engineering路线)
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    # 衰减权重: 近档权重高(即时性)，远档权重低但非零(隐藏意图)
    weights = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1])
    weights = weights / weights.sum()
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        wdis = []
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            bid_total = np.dot(weights, bid_vols)
            ask_total = np.dot(weights, ask_vols)
            total = bid_total + ask_total
            if total > 0:
                wdis.append((bid_total - ask_total) / total)
        
        factor = np.mean(wdis) if wdis else 0.0
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
