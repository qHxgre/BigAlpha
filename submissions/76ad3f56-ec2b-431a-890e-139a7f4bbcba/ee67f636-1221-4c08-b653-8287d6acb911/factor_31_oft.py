"""
因子 31: OFT — 订单流毒性 (Order Flow Toxicity)
==================================================

经济逻辑:
VPIN变体：用 num_orders 替代 volume 做桶分割。
委托笔数在买卖方向上的不均衡反映订单流的"毒性"——即知情交易方向性。
高毒性 = 大量订单集中在一侧 = 知情交易者在行动 = 信息即将释放。

方法：
- 按固定分钟数分桶，每桶计算订单不平衡
- OFT = 日内各桶|OBI_orders|的均值 × OBI方向
- 用 orders 比 volume 更灵敏（订单是意图信号，volume是执行信号）

方向: OFT 越大 → 订单流方向性越强且偏买 → 预期正收益

AI应用说明:
源自 Easley et al. VPIN，将 volume bucketing 改为 order bucketing。
LLM提示: "委托笔数的桶不平衡"。
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    bucket_size = 10  # 每10分钟一个桶
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        
        if len(grp) < bucket_size:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        bucket_obis = []
        
        for start in range(0, len(grp), bucket_size):
            bucket = grp.iloc[start:start + bucket_size]
            bid_ords_total = 0
            ask_ords_total = 0
            bid_vols_total = 0
            ask_vols_total = 0
            
            for _, row in bucket.iterrows():
                has_orders = any(f'bid_num_orders{i}' in row.index for i in range(1, 4))
                if has_orders:
                    bo = sum(row.get(f'bid_num_orders{i}', 0) or 0 for i in range(1, 11))
                    ao = sum(row.get(f'ask_num_orders{i}', 0) or 0 for i in range(1, 11))
                else:
                    no = row.get('num_orders', 0) or 0
                    bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
                    av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
                    tv = bv + av
                    bo = bv / (tv + 1e-8) * no
                    ao = av / (tv + 1e-8) * no
                
                bid_ords_total += bo
                ask_ords_total += ao
                
                bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
                av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
                bid_vols_total += bv
                ask_vols_total += av
            
            total_orders = bid_ords_total + ask_ords_total
            total_vol = bid_vols_total + ask_vols_total
            
            if total_orders > 0 and total_vol > 0:
                obi_orders = (bid_ords_total - ask_ords_total) / total_orders
                obi_vol = (bid_vols_total - ask_vols_total) / total_vol
                
                # 订单不平衡和成交量不平衡的一致性
                bucket_obis.append((obi_orders, obi_vol))
        
        if len(bucket_obis) < 2:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        obi_orders_arr = np.array([b[0] for b in bucket_obis])
        obi_vol_arr = np.array([b[1] for b in bucket_obis])
        
        # 毒性 = 桶间OBI的波动 × 方向一致性
        toxicity = np.std(obi_orders_arr)
        direction = np.mean(obi_orders_arr)
        
        # order-OBI和vol-OBI的一致性(当二者同向且order-OBI更强=知情交易)
        alignment = np.corrcoef(obi_orders_arr, obi_vol_arr)[0, 1] if len(obi_orders_arr) > 2 else 0
        
        factor = direction * (toxicity + 0.01) * max(alignment, 0)
        
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
