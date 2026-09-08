"""
因子 39: DPM — 深度动量 (Depth Momentum)
========================================

经济逻辑:
收盘前总深度相对于开盘后总深度的积累反映当日深度变化方向。
深度持续积累 → 做市商或大资金持续提供流动性 → 市场信心足 → 看涨。
深度持续萎缩 → 流动性撤退 → 市场脆弱 → 看跌。

方法：
- 将日内分为开盘期(前25%)和收盘期(后25%)
- DPM = 收盘期平均总深度 / 开盘期平均总深度 - 1
- 配合买卖深度差异：bid深度积累 > ask深度积累 → 更看涨

方向: DPM 越大 → 深度在积累 → 预期正收益

AI应用说明:
类比价格动量，但在流动性维度上。
LLM提示: "深度本身也有趋势吗？"。
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
        
        if len(grp) < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 计算每分钟总深度
        total_depths = []
        bid_depths = []
        ask_depths = []
        
        for _, row in grp.iterrows():
            bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            total_depths.append(bv + av)
            bid_depths.append(bv)
            ask_depths.append(av)
        
        total_depths = np.array(total_depths)
        bid_depths = np.array(bid_depths)
        ask_depths = np.array(ask_depths)
        
        n = len(total_depths)
        n_open = max(n // 4, 2)
        n_close = max(n // 4, 2)
        
        open_avg = np.mean(total_depths[:n_open])
        close_avg = np.mean(total_depths[-n_close:])
        
        if open_avg < 1:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 总深度变化率
        depth_chg = close_avg / open_avg - 1
        
        # 买卖深度差异的变化率
        bid_open = np.mean(bid_depths[:n_open])
        bid_close = np.mean(bid_depths[-n_close:])
        ask_open = np.mean(ask_depths[:n_open])
        ask_close = np.mean(ask_depths[-n_close:])
        
        bid_chg = (bid_close - bid_open) / (bid_open + 1e-8)
        ask_chg = (ask_close - ask_open) / (ask_open + 1e-8)
        
        # 总深度积累 + 买盘积累超过卖盘
        dpm = depth_chg + (bid_chg - ask_chg)
        
        results.append({'date': dt, 'instrument': inst, 'factor': dpm})
    
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
