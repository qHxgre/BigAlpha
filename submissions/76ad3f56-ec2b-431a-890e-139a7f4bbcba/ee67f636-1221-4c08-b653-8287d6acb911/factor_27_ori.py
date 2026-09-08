"""
因子 27: ORI — 委托笔数不平衡 (Order-Request Imbalance)
========================================================

经济逻辑:
bid各档委托笔数 vs ask各档委托笔数的差异反映散户vs机构的交易意图。
num_orders 反映订单碎片化——机构高频拆单vs散户大单。
bid笔数 > ask笔数 → 更多参与者在买方 → 买压分散但持续 → 看涨。
bid笔数 < ask笔数 → 更多参与者在卖方 → 卖压分散 → 看跌。

方法：
- 对各档 num_orders 求和(若字段存在)，否则用 num_trades 替代
- ORI = (sum_bid_orders - sum_ask_orders) / total_orders
- 近档(1-5)加权 > 远档(6-10)

方向: ORI 越大 → 买方参与者更多 → 预期正收益

AI应用说明:
区别于volume-based imbalance(因子08 OBSH)，
orders-based imbalance捕捉"谁在交易"而非"交易了多少"。
LLM提示: "委托笔数的买卖方向差异"。
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    # 近档权重高于远档
    w = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1])
    w = w / w.sum()
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        ori_vals = []
        
        for _, row in grp.iterrows():
            # 尝试获取各档委托笔数
            has_orders = any(f'bid_num_orders{i}' in row.index for i in range(1, 4))
            
            if has_orders:
                bid_ords = np.array([row.get(f'bid_num_orders{i}', 0) or 0 for i in range(1, 11)], dtype=float)
                ask_ords = np.array([row.get(f'ask_num_orders{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            else:
                # 回退: 用 num_orders 总量 + 深度比例分配
                total_orders = row.get('num_orders', 0) or 0
                bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
                ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
                bv_sum = bid_vols.sum()
                av_sum = ask_vols.sum()
                total_vol = bv_sum + av_sum
                if total_vol > 0 and total_orders > 0:
                    bid_ords = bid_vols / total_vol * total_orders * w
                    ask_ords = ask_vols / total_vol * total_orders * w
                else:
                    continue
            
            bid_w = np.dot(w, bid_ords)
            ask_w = np.dot(w, ask_ords)
            total = bid_w + ask_w
            
            if total > 0:
                ori_vals.append((bid_w - ask_w) / total)
        
        factor = np.mean(ori_vals) if ori_vals else 0.0
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
