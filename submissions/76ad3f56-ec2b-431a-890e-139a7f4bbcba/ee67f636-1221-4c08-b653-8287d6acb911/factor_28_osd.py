"""
因子 28: OSD — 委托规模分化 (Order Size Divergence)
=====================================================

经济逻辑:
比较买卖方的平均单笔委托规模 (volume/num_orders)。
买方平均单笔大 vs 卖方平均单笔小 → 买方是机构/卖方是散户 → 看涨。
买方平均单笔小 vs 卖方平均单笔大 → 买方是散户/卖方是机构 → 看跌。

方法：
- 对bid/ask分别计算 avg_order_size = total_volume / total_num_orders
- OSD = log(bid_avg_size / ask_avg_size)
- 买方单笔更大 → OSD > 0 → 机构在买 → 看涨

方向: OSD 越大 → 买方单笔规模更大 → 预期正收益

AI应用说明:
订单拆单行为研究：机构拆多笔小单隐藏意图 vs 散户大单。
LLM提示: "买卖方委托单笔大小的系统性差异"。
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
        osd_vals = []
        
        for _, row in grp.iterrows():
            bid_total_vol = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            ask_total_vol = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            
            has_orders = any(f'bid_num_orders{i}' in row.index for i in range(1, 4))
            
            if has_orders:
                bid_total_ords = sum(row.get(f'bid_num_orders{i}', 0) or 0 for i in range(1, 11))
                ask_total_ords = sum(row.get(f'ask_num_orders{i}', 0) or 0 for i in range(1, 11))
            else:
                # 回退用 num_orders 总量按深度比例分配
                total_ords = row.get('num_orders', 0) or 0
                all_vol = bid_total_vol + ask_total_vol
                if all_vol > 0 and total_ords > 0:
                    bid_total_ords = bid_total_vol / all_vol * total_ords
                    ask_total_ords = ask_total_vol / all_vol * total_ords
                else:
                    continue
            
            if bid_total_ords < 0.5 or ask_total_ords < 0.5:
                continue
            
            bid_avg_size = bid_total_vol / bid_total_ords
            ask_avg_size = ask_total_vol / ask_total_ords
            
            if bid_avg_size > 0 and ask_avg_size > 0:
                osd_vals.append(np.log(bid_avg_size / ask_avg_size))
        
        factor = np.mean(osd_vals) if osd_vals else 0.0
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
