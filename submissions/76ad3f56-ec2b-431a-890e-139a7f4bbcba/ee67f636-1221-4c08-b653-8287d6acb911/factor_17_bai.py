"""
因子 17: BAI — 买卖方主动性指数 (Bid-Ask Initiative Index)
=========================================================

经济逻辑:
比较 total_bid_volume 和 total_ask_volume 的变化速度。
买方挂单增长快 → 买方积极性上升 → 看涨
卖方挂单增长快 → 卖方积极性上升 → 看跌
用两侧总深度的变化率差来衡量。

方向: 买方深度增长 > 卖方 → 看涨 → 正因子

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
        
        # 获取总买卖深度
        if 'total_bid_volume' in grp.columns and 'total_ask_volume' in grp.columns:
            bid_totals = grp['total_bid_volume'].values
            ask_totals = grp['total_ask_volume'].values
        else:
            # 手动求和
            bid_totals = []
            ask_totals = []
            for _, row in grp.iterrows():
                bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
                av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
                bid_totals.append(bv)
                ask_totals.append(av)
            bid_totals = np.array(bid_totals)
            ask_totals = np.array(ask_totals)
        
        if len(bid_totals) < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 前后半段深度变化
        n = len(bid_totals)
        n_half = n // 2
        
        bid_first = np.mean(bid_totals[:n_half])
        bid_last = np.mean(bid_totals[n_half:])
        ask_first = np.mean(ask_totals[:n_half])
        ask_last = np.mean(ask_totals[n_half:])
        
        # 变化率
        bid_chg = (bid_last - bid_first) / (bid_first + 1e-8)
        ask_chg = (ask_last - ask_first) / (ask_first + 1e-8)
        
        # 买方深度增长 > 卖方 → 买压上升
        factor = bid_chg - ask_chg
        
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
