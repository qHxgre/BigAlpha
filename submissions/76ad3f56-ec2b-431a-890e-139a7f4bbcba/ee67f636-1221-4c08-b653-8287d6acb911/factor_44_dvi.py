"""
因子 44: DVI — 深度×波动交互 (Depth-Volatility Interaction)
============================================================

经济逻辑:
总深度 / 日内波动率捕捉流动性风险溢价。
深度大+波动小 → 流动性充裕且市场平静 → 高质量环境 → 看涨。
深度小+波动大 → 流动性不足且市场动荡 → 高风险 → 看跌(取负后负值)。

方法：
- DVI = log(total_depth) - log(intraday_volatility)
- 实际上: DVI = total_depth / realized_vol
- 高DVI = 单位波动被大量深度支撑 = 市场质量好

方向: DVI 越大 → 深度/波动比越高 → 预期正收益

AI应用说明:
流动性风险溢价的代理变量，比单独深度或波动更全面。
LLM提示: "深度和波动的跷跷板关系"。
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
        
        total_depths = []
        mid_prices = []
        
        for _, row in grp.iterrows():
            bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            total_depths.append(bv + av)
            
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            if pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0:
                mid_prices.append((ap1 + bp1) / 2)
            else:
                mid_prices.append(np.nan)
        
        total_depths = np.array(total_depths)
        mid_prices = np.array(mid_prices)
        
        avg_depth = np.mean(total_depths)
        
        # 已实现波动率
        valid = ~np.isnan(mid_prices)
        if valid.sum() < 3:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        mp = mid_prices[valid]
        returns = np.diff(np.log(mp + 1e-12))
        real_vol = np.std(returns) if len(returns) > 0 else 1e-8
        
        if real_vol < 1e-12:
            real_vol = 1e-12
        
        # DVI = depth / vol, 取log
        dvi = np.log1p(avg_depth) / (real_vol * np.sqrt(len(returns)) + 1e-8)
        # 除以sqrt(n)做年化修正，使跨股票可比
        
        results.append({'date': dt, 'instrument': inst, 'factor': dvi})
    
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
