"""
因子 43: OIS — OBI×价差 (Order Book Imbalance × Spread)
=========================================================

经济逻辑:
order book imbalance(OBI)除以 spread 捕捉单位价差下的供需失衡效率。
OBI相同但价差不同 → 价差窄时OBI更可信(交易成本低，信号质量高)。
OBI大+价差窄 → 强供需方向+低交易成本 → 高置信度信号 → 看涨。
OBI大+价差宽 → 供需有方向但成本高 → 信号需打折。

方法：
- OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
- Spread = (ask_px - bid_px) / mid_px
- OIS = OBI / (spread + ε) = 单位价差的供需失衡

方向: OIS 越大 → 买盘失衡效率更高 → 预期正收益

AI应用说明:
经典OBI在宽spread时信号质量差，除以spread修正。
LLM提示: "OBI的信息效率"。
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
        ois_vals = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            
            if not (pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0):
                continue
            
            mid = (ap1 + bp1) / 2
            spread = (ap1 - bp1) / mid
            
            # 用所有10档计算OBI
            bid_vols = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            ask_vols = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            total = bid_vols + ask_vols
            
            if total < 10 or spread < 1e-8:
                continue
            
            obi = (bid_vols - ask_vols) / total
            
            # OBI / spread: 单位成本的信息效率
            # 额外加微小值避免除零但保持信号方向
            ois = obi / (spread + 1e-6)
            
            ois_vals.append(ois)
        
        if not ois_vals:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 取90分位(高信息效率时刻)和均值的加权
        ois_90 = np.percentile(ois_vals, 90)
        ois_mean = np.mean(ois_vals)
        factor = ois_mean * 0.7 + ois_90 * 0.3
        
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
