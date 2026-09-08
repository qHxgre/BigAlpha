"""
因子 24: DFL — 深度前置比 (Depth Front-Loading Ratio)
=======================================================

经济逻辑:
前3档深度占总深度的比例反映做市商的护盘意愿。
前档深度占比高 → 做市商在近端提供充裕流动性 → 护盘意愿强 → 市场稳定。
买方前档占比高 → 买方做市商积极护盘 → 看涨。
卖方前档占比高 → 卖方做市商积极护盘 → 看跌。

方法：
- DFL = bid前3档占比 - ask前3档占比
- 将bid深度近档集中度解释为买盘保护强度

方向: DFL 越大 → 买盘护盘更强 → 预期正收益

AI应用说明:
源自做市商行为研究，前档深度集中度反映做市商的风险偏好。
LLM提示词: "做市商的边际护盘成本"。
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
        dfl_vals = []
        
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            bid_total = bid_vols.sum()
            ask_total = ask_vols.sum()
            
            if bid_total < 50 or ask_total < 50:
                continue
            
            bid_front = bid_vols[:3].sum()
            ask_front = ask_vols[:3].sum()
            
            bid_ratio = bid_front / bid_total
            ask_ratio = ask_front / ask_total
            
            # 买盘前档占比高于卖盘 = 买盘护盘更强
            # 但也要考虑前档占比太高可能意味着远档没有支撑(脆弱)
            # 取差值+非线性修正
            raw = bid_ratio - ask_ratio
            
            # 加上bid总深度修正: 深度大+前档高 = 真护盘
            dfl_vals.append(raw * np.log1p(bid_total))
        
        factor = np.mean(dfl_vals) if dfl_vals else 0.0
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
