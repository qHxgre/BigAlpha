"""
因子 13: SPW — 价差楔形 (Spread Wedge)
======================================

经济逻辑:
买1卖1价差只反映即时流动性，但10档价差的"楔形"形状反映远档预期。
计算 ask_price<i> - bid_price<i> 在各档的变化率。
楔形收窄(远档价差相对变窄) = 市场预期价格向上
楔形扩大(远档价差相对变宽) = 市场预期价格向下

方向: 楔形收窄 → 看涨 → 正因子

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
        
        wedge_vals = []
        for _, row in grp.iterrows():
            spreads = []
            for i in range(1, 11):
                ap = row.get(f'ask_price{i}', np.nan)
                bp = row.get(f'bid_price{i}', np.nan)
                if pd.notna(ap) and pd.notna(bp) and ap > bp > 0:
                    spreads.append(ap - bp)
                else:
                    break
            
            if len(spreads) < 5:
                continue
            
            mid_prices = []
            for i in range(1, len(spreads) + 1):
                ap = row.get(f'ask_price{i}', 0) or 0
                bp = row.get(f'bid_price{i}', 0) or 0
                if ap > 0 and bp > 0:
                    mid_prices.append((ap + bp) / 2)
            
            if len(mid_prices) < 5:
                continue
            
            # 各档价差的增长率
            # spread_i = ask_i - bid_i
            # 如果远档 spread 相对窄 = ask价上涨慢 + bid价上涨快 = 看好
            spread_growth = (spreads[-1] - spreads[0]) / (spreads[0] + 1e-8)
            
            # 中点价增长率
            mid_growth = (mid_prices[-1] - mid_prices[0]) / (mid_prices[0] + 1e-8)
            
            # 楔形: 价差增长 vs 中点增长
            # spread增长慢 + mid增长快 = 看涨
            wedge = mid_growth - spread_growth
            wedge_vals.append(wedge)
        
        factor = np.mean(wedge_vals) if wedge_vals else 0.0
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
