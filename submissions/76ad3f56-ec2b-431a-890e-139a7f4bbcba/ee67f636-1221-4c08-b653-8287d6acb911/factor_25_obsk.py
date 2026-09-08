"""
因子 25: OBSK — 订单簿稳定度 (Order Book Stability)
=====================================================

经济逻辑:
各档深度分钟级变异系数(CV)的负值反映盘口噪音水平。
深度波动大(CV高) → 市场参与者频繁挂撤单 → 噪音交易主导 → 信息效率低。
深度波动小(CV低) → 盘口稳定 → 流动性好 → 信息环境清晰 → 有利价格发现。

方法：
- 对每分钟10档深度计算 CV = std/mean
- 取各档CV的均值
- OBSK = -mean(CV) → 稳定度高(低CV) = 高因子值 = 看涨
- 配合买卖盘口稳定度差异: buy稳定 sell不稳定 → 看涨

方向: OBSK 越大 → 盘口稳定噪音小 → 预期正收益

AI应用说明:
市场微观结构中"噪音交易者"vs"知情交易者"框架。
LLM提示: "盘口深度的时间序列变异程度"。
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
        
        if len(grp) < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        bid_stability = []
        ask_stability = []
        
        for lvl in range(1, 11):
            bv = grp[f'bid_volume{lvl}'].values if f'bid_volume{lvl}' in grp.columns else np.zeros(len(grp))
            av = grp[f'ask_volume{lvl}'].values if f'ask_volume{lvl}' in grp.columns else np.zeros(len(grp))
            
            bv = np.array([v if pd.notna(v) and v > 0 else 0 for v in bv], dtype=float)
            av = np.array([v if pd.notna(v) and v > 0 else 0 for v in av], dtype=float)
            
            # CV = std/mean, 分母加小量
            b_mean = bv.mean()
            a_mean = av.mean()
            if b_mean > 1:
                bid_stability.append(bv.std() / b_mean)
            if a_mean > 1:
                ask_stability.append(av.std() / a_mean)
        
        bid_mean_cv = np.mean(bid_stability) if bid_stability else 0
        ask_mean_cv = np.mean(ask_stability) if ask_stability else 0
        
        # 稳定度 = -CV (取负号: 越不稳定→越小)
        # bid稳定(低CV)+ ask不稳定(高CV) → 看涨
        factor = -bid_mean_cv + ask_mean_cv  # -bid_CV + ask_CV = -(bid_CV - ask_CV)
        # 当bid稳定(小CV)且ask不稳定(大CV)时, factor > 0
        
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
