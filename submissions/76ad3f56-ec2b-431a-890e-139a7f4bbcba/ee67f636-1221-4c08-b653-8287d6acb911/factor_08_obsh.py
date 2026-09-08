"""
因子 08: OBSH — 订单簿形状 (Order Book Shape)
=============================================

经济逻辑:
盘口形状(凹/凸/线性)反映不同类型参与者的博弈。
凸形(近档厚远档薄) = 做市商主导 = 流动性好 = 低波动
凹形(近档薄远档厚) = 隐藏大单 = 知情交易 = 即将突破
用二阶差分来度量凹凸性。

方向: 凹形(知情交易信号) → 可能上行突破 → 正因子

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
        curvatures = []
        
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            # 二阶差分: curvature = vol[i-1] - 2*vol[i] + vol[i+1]
            # 正值 = 凹(convex) = 近档薄远档厚
            # 负值 = 凸(concave) = 近档厚远档薄
            def _curvature(vols):
                if vols.sum() < 50:
                    return 0.0
                vols_norm = vols / (vols.sum() + 1e-8)
                curv = 0
                for i in range(1, 9):
                    curv += vols_norm[i-1] - 2*vols_norm[i] + vols_norm[i+1]
                return curv
            
            bid_curv = _curvature(bid_vols)
            ask_curv = _curvature(ask_vols)
            
            # 买盘凹(隐藏买单) + 卖盘凸(卖单集中近档) = 买方隐藏意图 > 卖方
            curvatures.append(bid_curv - ask_curv)
        
        factor = np.mean(curvatures) if curvatures else 0.0
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
