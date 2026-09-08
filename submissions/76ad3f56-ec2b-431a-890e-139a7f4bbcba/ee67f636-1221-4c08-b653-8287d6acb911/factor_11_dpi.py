"""
因子 11: DPI — 深度价格冲击 (Depth-weighted Price Impact)
=========================================================

经济逻辑:
大单成交需要消耗多档深度。衡量"成交一定金额需要跨越多少档盘口"，
跨越档数越多 = 流动性越差 = 价格冲击越大。
DPI = 模拟买入100万的档位跨越数 − 模拟卖出100万的档位跨越数。
买盘更浅(买需跨更多档) = 买压 > 卖压 = 看涨。

方向: DPI > 0 → 买盘相对更浅 → 买方急于成交 → 正因子

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
    TARGET_NOTIONAL = 1_000_000  # 模拟成交100万元
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        
        dpi_vals = []
        for _, row in grp.iterrows():
            price = row.get('price', 0) or row.get('close', 0) or 0
            if price <= 0:
                continue
            
            # 卖盘(ask): 买方需要跨越的深度
            ask_cum = 0
            ask_levels = 0
            for i in range(1, 11):
                px = row.get(f'ask_price{i}', 0) or 0
                vol = row.get(f'ask_volume{i}', 0) or 0
                if px > 0 and vol > 0:
                    ask_cum += px * vol
                    ask_levels += 1
                    if ask_cum >= TARGET_NOTIONAL:
                        break
            
            # 买盘(bid): 卖方需要跨越的深度
            bid_cum = 0
            bid_levels = 0
            for i in range(1, 11):
                px = row.get(f'bid_price{i}', 0) or 0
                vol = row.get(f'bid_volume{i}', 0) or 0
                if px > 0 and vol > 0:
                    bid_cum += px * vol
                    bid_levels += 1
                    if bid_cum >= TARGET_NOTIONAL:
                        break
            
            # 买盘浅(需要更多档) = 买方压力 > 卖方
            if ask_levels > 0 and bid_levels > 0:
                dpi_vals.append(bid_levels - ask_levels)
        
        factor = np.mean(dpi_vals) if dpi_vals else 0.0
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
