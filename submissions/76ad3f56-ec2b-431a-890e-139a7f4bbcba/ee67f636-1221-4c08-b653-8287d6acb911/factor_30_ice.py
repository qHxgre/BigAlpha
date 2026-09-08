"""
因子 30: ICE — 冰山订单检测 (Iceberg Order Detection)
========================================================

经济逻辑:
volume/num_orders 极大 → 单笔委托规模异常大 → 可能是冰山订单（只显示部分）。
冰山订单意味着大机构在悄悄建仓 → 有持续买/卖需求 → 信息含量高。
买方冰山 → 隐藏大买单 → 看涨。
卖方冰山 → 隐藏大卖单 → 看跌。

方法：
- 计算各档 volume/orders 比率
- 识别比率超过3倍日内的异常档位
- ICE = bid冰山程度 - ask冰山程度
- 配合价格方向确认

方向: ICE 越大 → 买方冰山更活跃 → 预期正收益

AI应用说明:
冰山订单是算法交易的经典形态。
LLM提示: "如何从盘口数据发现冰山订单？"。
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
        ice_vals = []
        
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            has_orders = any(f'bid_num_orders{i}' in row.index for i in range(1, 4))
            
            if has_orders:
                bid_ords = np.array([row.get(f'bid_num_orders{i}', 1) or 1 for i in range(1, 11)], dtype=float)
                ask_ords = np.array([row.get(f'ask_num_orders{i}', 1) or 1 for i in range(1, 11)], dtype=float)
            else:
                # 回退: 假设均匀分配, 用总num_orders/总vol比例分到各档
                total_orders = max(row.get('num_orders', 0) or 0, 1)
                bid_total = bid_vols.sum()
                ask_total = ask_vols.sum()
                all_vol = bid_total + ask_total
                if all_vol > 0:
                    bid_ords = np.maximum(bid_vols / all_vol * total_orders, 1)
                    ask_ords = np.maximum(ask_vols / all_vol * total_orders, 1)
                else:
                    continue
            
            # 每档 avg_size = volume / num_orders
            bid_avg_size = bid_vols / bid_ords
            ask_avg_size = ask_vols / ask_ords
            
            # 整体中位数作为基准
            all_avg_sizes = np.concatenate([bid_avg_size, ask_avg_size])
            baseline = np.median(all_avg_sizes[all_avg_sizes > 0]) if len(all_avg_sizes[all_avg_sizes > 0]) > 0 else 1
            
            if baseline < 0.01:
                continue
            
            # 超过3倍基准的视为冰山信号
            bid_ice = np.sum(np.maximum(bid_avg_size / baseline - 3, 0))
            ask_ice = np.sum(np.maximum(ask_avg_size / baseline - 3, 0))
            
            if bid_ice > 0 or ask_ice > 0:
                ice_vals.append(np.log1p(bid_ice) - np.log1p(ask_ice))
        
        factor = np.mean(ice_vals) if ice_vals else 0.0
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
