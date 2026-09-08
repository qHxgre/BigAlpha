"""
因子 05: DSR — 深度扩散率 (Depth Spread Ratio)
================================================

经济逻辑:
盘口的"宽度"(深度分布在各档的离散程度)反映市场参与者分歧度。
深度集中在近档 = 共识强、方向明确
深度均匀分布在远档 = 分歧大、方向模糊
用各档量的变异系数(CV)来度量。

方向: DSR 小 = 深度集中(共识)→ 趋势延续 → 配合价格方向 → 正因子

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
        
        dsrs = []
        price_changes = []
        
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            # 变异系数 = std/mean
            bid_total = bid_vols.sum()
            ask_total = ask_vols.sum()
            
            if bid_total < 100 or ask_total < 100:
                continue
            
            bid_cv = np.std(bid_vols) / (np.mean(bid_vols) + 1e-8)
            ask_cv = np.std(ask_vols) / (np.mean(ask_vols) + 1e-8)
            
            # 平均 CV: 大 = 分散 = 分歧
            dsr = (bid_cv + ask_cv) / 2
            dsrs.append(dsr)
            
            # 记录价格变化方向
            price = row.get('price', np.nan)
            if pd.notna(price):
                price_changes.append(price)
        
        if not dsrs:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        avg_dsr = np.mean(dsrs)
        
        # 确定日内价格方向
        price_dir = 0
        if len(price_changes) >= 2:
            price_dir = np.sign(price_changes[-1] - price_changes[0])
        
        # 低分歧+上涨 = 强共识看涨 → 正向
        # 低分歧+下跌 = 强共识看跌 → 负向
        # 高分歧 → 中性/反转 → 信号弱
        # factor = (1 - DSR归一化) × 价格方向
        # 使用 tanh 归一化
        dsr_norm = np.tanh(avg_dsr)
        factor = (1 - dsr_norm) * price_dir
        
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
