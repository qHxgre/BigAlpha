"""
因子 22: OBC — 订单簿曲率 (Order Book Curvature)
==================================================

经济逻辑:
10档深度的二阶差分（曲率）捕捉订单簿形状的凹凸性。
凸形(正曲率) → 近档薄远档厚 → 隐藏大单在远端 → 价格有支撑/阻力。
凹形(负曲率) → 近档厚远档薄 → 所有流动性在近端 → 冲击成本低。
买方正曲率 = 买盘远档有隐藏支撑 → 看涨。
卖方正曲率 = 卖盘远档有隐藏阻力 → 看跌。

方法：
- 对bid/ask各档深度计算二阶中心差分
- OBC = mean(bid曲率) - mean(ask曲率)
- bid侧凸 > ask侧凹 → 买盘有远档支撑 → 看涨

方向: OBC 越大 → 买盘远档支撑强 → 预期正收益

AI应用说明:
源自期权定价中的Gamma曲率概念，扩展到订单簿领域。
LLM通过"捕捉隐藏大单"的提示词生成。
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
        obc_vals = []
        
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            if bid_vols.sum() < 50 or ask_vols.sum() < 50:
                continue
            
            # 二阶中心差分 (曲率)
            bid_curv = _curvature(bid_vols)
            ask_curv = _curvature(ask_vols)
            
            if bid_curv is not None and ask_curv is not None:
                obc_vals.append(bid_curv - ask_curv)
        
        factor = np.mean(obc_vals) if obc_vals else 0.0
        results.append({'date': dt, 'instrument': inst, 'factor': factor})
    
    result = pd.DataFrame(results)
    return _normalize(result)


def _curvature(vols):
    """计算深度的平均二阶差分(曲率)"""
    if len(vols) < 5:
        return None
    # 二阶中心差分: f''(i) ≈ f(i+1) - 2*f(i) + f(i-1)
    d2 = vols[2:] - 2 * vols[1:-1] + vols[:-2]
    # 用深度加权平均曲率
    weights = vols[1:-1] + 1
    weights = weights / (weights.sum() + 1e-8)
    return np.dot(d2, weights)


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
