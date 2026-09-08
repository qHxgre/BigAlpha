"""
因子 35: DAV — 深度调整波动率 (Depth-Adjusted Volatility)
=========================================================

经济逻辑:
已实现波动率 / 平均深度 = 单位深度支撑的波动率。
波动率相同但深度不同 → 深度大的更稳定、冲击成本更低。
DAV 高 = 同样波动下深度不足 → 流动性风险溢价 → 看跌(取负)。
DAV 低 = 波动率被深度充分吸收 → 市场稳定 → 看涨。

方法：
- 计算日内已实现波动率: 价格收益率的标准差
- 计算日内平均总深度: mean(bid_total + ask_total)
- DAV = -realized_vol / log(avg_depth)

方向: DAV 越大(越负=波动被深度吸收) → 实际上DAV越小越好 → 取负：DAV=-vol/depth
  所以因子值越大表示越稳定 → 取负号使方向正确

AI应用说明:
深度和波动的联合评估，比单独波动率更全面。
LLM提示: "流动性如何缓冲波动？"。
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
        
        # 用中间价计算收益率
        mid_prices = []
        depths = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            
            if pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0:
                mid_prices.append((ap1 + bp1) / 2)
            
            bv_total = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            av_total = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            depths.append(bv_total + av_total)
        
        if len(mid_prices) < 5 or len(depths) < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        mid_prices = np.array(mid_prices)
        depths = np.array(depths)
        
        # 已实现波动率: 对数收益率的标准差
        returns = np.diff(np.log(mid_prices + 1e-12))
        real_vol = np.std(returns) if len(returns) > 0 else 0
        
        avg_depth = np.mean(depths)
        
        if avg_depth < 1 or real_vol < 1e-12:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # DAV = -volatility / depth → 波动小+深度大 = DAV接近0(大值)
        # 取负号: 深度不足=DAV负值小, 深度充分=DAV负值绝对值小 = 接近0
        dav = -real_vol / np.log1p(avg_depth)
        
        results.append({'date': dt, 'instrument': inst, 'factor': dav})
    
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
