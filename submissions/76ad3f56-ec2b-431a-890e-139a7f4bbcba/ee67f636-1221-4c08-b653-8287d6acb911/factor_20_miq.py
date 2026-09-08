"""
因子 20: MIQ — 微观信息质量 (Microstructure Information Quality)
================================================================

经济逻辑:
高质量的微观结构信号应该同时满足:
1. 买卖价差窄 (交易成本低 → 信号可信)
2. 盘口不平衡强 (供需差距大 → 信号强度高)
3. 深度充足 (流动性好 → 信号不易被操纵)

三个维度综合评分:
MIQ = OBI × (1/TWAS) × 深度因子
高 MIQ = 强信号+低成本+充足流动性 = 高质量α

方向: MIQ 大 → 高质量看涨信号 → 正因子

AI技术: LLM生成公式 → 硬编码提交 (多因子合成类)
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
        
        miq_vals = []
        
        for _, row in grp.iterrows():
            # 1. 订单簿不平衡 OBI (一档)
            bv1 = row.get('bid_volume1', 0) or 0
            av1 = row.get('ask_volume1', 0) or 0
            tot1 = bv1 + av1
            obi = (bv1 - av1) / tot1 if tot1 > 0 else 0
            
            # 2. 买卖价差倒数 (1/spread)
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            if pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0:
                spread = (ap1 - bp1) / ((ap1 + bp1) / 2)
                inv_spread = 1 / (spread + 0.0001)  # 价差窄=大值
            else:
                inv_spread = 0
            
            # 3. 深度充足度 (前5档总量归一化)
            depth = 0
            for i in range(1, 6):
                bv = row.get(f'bid_volume{i}', 0) or 0
                av = row.get(f'ask_volume{i}', 0) or 0
                depth += bv + av
            depth_score = np.log1p(depth) / 20  # 对数归一化
            
            # 4. 深度稳定性: 各档量偏离均值的程度
            vols = np.array([row.get(f'bid_volume{i}', 0) or 0 + (row.get(f'ask_volume{i}', 0) or 0) 
                           for i in range(1, 11)], dtype=float)
            vol_cv = np.std(vols) / (np.mean(vols) + 1e-8)
            stability = 1 / (1 + vol_cv)  # 低CV=稳定=高分数
            
            # MIQ = 方向 × 强度 × 质量
            signal_strength = abs(obi)
            signal_direction = np.sign(obi)
            quality = inv_spread * depth_score * stability
            
            miq = signal_strength * signal_direction * quality
            miq_vals.append(miq)
        
        factor = np.mean(miq_vals) if miq_vals else 0.0
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
