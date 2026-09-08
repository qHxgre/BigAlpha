"""
因子 29: OCP — 撤单概率代理 (Order Cancellation Proxy)
========================================================

经济逻辑:
num_orders 急剧下降但 volume 不变 → 大量委托被撤销 → 做市商撤退。
撤单行为透露交易者失去信心 → 流动性即将枯竭 → 价格承压。
买方撤单 > 卖方撤单 → 买方力量减弱 → 看跌 → 取负号。
卖方撤单 > 买方撤单 → 卖方力量减弱 → 看涨。

方法：
- 计算 num_orders 的前后半段变化率
- 比较 volume 变化率（volume变化小 + orders变化大 = 撤单）
- OCP = (ask订单变化率 - bid订单变化率)，配合volume确认

方向: OCP 越大 → 卖方撤单多于买方 → 卖压减弱 → 预期正收益

AI应用说明:
源自"幽灵流动性"研究(phantom liquidity)。
LLM提示: "订单量减少但成交量不变意味着什么？"。
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
        
        # 计算每分钟的订单量和成交量
        order_seqs = []
        volume_seqs = []
        
        for _, row in grp.iterrows():
            # 各档订单总和
            has_orders = any(f'bid_num_orders{i}' in row.index for i in range(1, 4))
            if has_orders:
                bid_ords = sum(row.get(f'bid_num_orders{i}', 0) or 0 for i in range(1, 11))
                ask_ords = sum(row.get(f'ask_num_orders{i}', 0) or 0 for i in range(1, 11))
            else:
                no = row.get('num_orders', 0) or 0
                bid_vols = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
                ask_vols = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
                total_vol = bid_vols + ask_vols
                bid_ords = bid_vols / (total_vol + 1e-8) * no
                ask_ords = ask_vols / (total_vol + 1e-8) * no
            
            bid_vols_sum = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            ask_vols_sum = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            
            order_seqs.append((bid_ords, ask_ords))
            volume_seqs.append((bid_vols_sum, ask_vols_sum))
        
        n = len(order_seqs)
        n_half = n // 2
        
        # 前后半段
        bid_ords_first = np.mean([o[0] for o in order_seqs[:n_half]])
        bid_ords_last = np.mean([o[0] for o in order_seqs[n_half:]])
        ask_ords_first = np.mean([o[1] for o in order_seqs[:n_half]])
        ask_ords_last = np.mean([o[1] for o in order_seqs[n_half:]])
        
        bid_vol_first = np.mean([v[0] for v in volume_seqs[:n_half]])
        bid_vol_last = np.mean([v[0] for v in volume_seqs[n_half:]])
        ask_vol_first = np.mean([v[1] for v in volume_seqs[:n_half]])
        ask_vol_last = np.mean([v[1] for v in volume_seqs[n_half:]])
        
        # 订单变化率 (减少=撤单)
        bid_ord_chg = (bid_ords_last - bid_ords_first) / (bid_ords_first + 1e-8)
        ask_ord_chg = (ask_ords_last - ask_ords_first) / (ask_ords_first + 1e-8)
        
        # 成交量变化率 (应该小)
        bid_vol_chg = (bid_vol_last - bid_vol_first) / (bid_vol_first + 1e-8)
        ask_vol_chg = (ask_vol_last - ask_vol_first) / (ask_vol_first + 1e-8)
        
        # 撤单信号: 订单大幅减少但成交量不变
        bid_cancel = -bid_ord_chg / (abs(bid_vol_chg) + 0.01)
        ask_cancel = -ask_ord_chg / (abs(ask_vol_chg) + 0.01)
        
        # ask撤单多 > bid撤单多 → 卖压减少 → 看涨
        factor = ask_cancel - bid_cancel
        
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
