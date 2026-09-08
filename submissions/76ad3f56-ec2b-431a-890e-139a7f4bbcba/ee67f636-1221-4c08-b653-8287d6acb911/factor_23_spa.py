"""
因子 23: SPA — 价差加速度 (Spread Acceleration)
=================================================

经济逻辑:
买卖价差(spread)的二阶导(变化率的变化率)捕捉流动性急剧变化的前兆。
spread 加速收窄 → 流动性急剧改善 → 价格即将突破 → 看涨(取OBI方向)。
spread 加速扩大 → 流动性急剧恶化 → 风险上升 → 看跌。

方法：
- 计算每分钟 spread = (ask1-bid1)/mid
- 对日内spread序列做二阶差分
- SPA = -加速扩大程度 × OBI方向 → 负加速(spread加速收窄且买盘强) = 正因子

方向: SPA 越大 → spread加速收窄(流动性改善) → 预期正收益

AI应用说明:
借用物理学加速度概念，检测流动性二阶变化。
LLM提示：检测"spread的急转弯"时刻。
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
        
        spreads = []
        obi_vals = []
        for _, row in grp.iterrows():
            ap = row.get('ask_price1', np.nan)
            bp = row.get('bid_price1', np.nan)
            if not (pd.notna(ap) and pd.notna(bp) and ap > bp > 0):
                continue
            mid = (ap + bp) / 2
            spread = (ap - bp) / mid
            spreads.append(spread)
            
            bv = row.get('bid_volume1', 0) or 0
            av = row.get('ask_volume1', 0) or 0
            total = bv + av
            obi_vals.append((bv - av) / total if total > 0 else 0)
        
        if len(spreads) < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        spreads = np.array(spreads)
        # 一阶差: spread变化
        d1 = np.diff(spreads)
        # 二阶差: spread加速度 > 0 = 加速扩大; < 0 = 加速收窄
        d2 = np.diff(d1) if len(d1) >= 2 else np.array([0])
        
        # spread加速收窄 = 正信号(取负); spread加速扩大 = 负信号
        # 配合OBI方向
        obi_mean = np.mean(obi_vals) if obi_vals else 0
        
        # SPA = -d2_mean * sign(obi), spread加速收窄→d2<0→取负后为正
        spa = -np.mean(d2) * np.sign(obi_mean) if obi_mean != 0 else -np.mean(d2)
        
        # 用std归一化防止量纲影响
        if len(d2) > 1 and np.std(d2) > 1e-12:
            spa = spa / np.std(d2)
        
        results.append({'date': dt, 'instrument': inst, 'factor': spa})
    
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
