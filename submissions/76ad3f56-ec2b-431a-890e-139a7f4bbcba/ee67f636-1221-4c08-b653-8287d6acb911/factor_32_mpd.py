"""
因子 32: MPD — 微观价格漂移 (Micro Price Drift)
================================================

经济逻辑:
加权中间价在最后15分钟的漂移方向和幅度反映收盘前的信息消化。
- 向上漂移 → 收盘前买盘持续推高均衡价格 → 看涨
- 向下漂移 → 收盘前卖盘持续压低均衡价格 → 看跌

方法：
- 计算每分钟的加权中间价: micro_price = (bid_px*ask_vol + ask_px*bid_vol) / (bid_vol + ask_vol)
- 对近档(1-3)和全档分别计算
- MPD = 最后15分钟 micro_price 的线性斜率

方向: MPD 越大 → 收盘前价格向上漂移 → 预期正收益

AI应用说明:
加权中间价(micro price)比成交价更及时地反映供需。
LLM提示: "盘口揭示的均衡价格漂移"。
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
        
        microprices = []
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            
            if not (pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0):
                continue
            
            # 用近3档深度做加权中间价
            bid_vol_near = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 4))
            ask_vol_near = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 4))
            
            total_vol = bid_vol_near + ask_vol_near
            if total_vol < 10:
                continue
            
            # 微价格 = (bid * ask_vol + ask * bid_vol) / total_vol
            micro = (bp1 * ask_vol_near + ap1 * bid_vol_near) / total_vol
            microprices.append(micro)
        
        if len(microprices) < 15:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        microprices = np.array(microprices)
        
        # 最后15分钟(或后15%)
        tail_len = max(min(15, len(microprices) // 4), 5)
        tail = microprices[-tail_len:]
        
        if len(tail) < 3:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 线性斜率
        x = np.arange(len(tail))
        slope = np.polyfit(x, tail, 1)[0]
        
        # 用全程价格范围归一化
        price_range = microprices.max() - microprices.min()
        if price_range < 1e-8:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        mpd = slope / price_range
        
        results.append({'date': dt, 'instrument': inst, 'factor': mpd})
    
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
