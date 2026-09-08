"""
因子 40: QSR — 报价填充率 (Quote Stuffing Ratio)
==================================================

经济逻辑:
num_orders 的分钟级自相关检测高频报价填充行为。
报价填充 → 极短时间内大量委托后撤单 → 制造虚假流动性 → 微观结构噪音。
高填充率 → 噪音主导 → 价格信号不可靠 → 看跌(取负)。
低填充率 → 订单真实 → 流动性可信 → 看涨。

方法：
- 计算 num_orders 的时间序列自相关(ACF lag-1)
- 高自相关 = 订单流平滑 = 真实流动性
- 低自相关(甚至负) = 订单流突变 = 报价填充嫌疑
- QSR = ACF lag-1 → 高自相关=好

方向: QSR 越大 → 订单流平滑真实 → 预期正收益

AI应用说明:
检测市场微观结构中的噪音制造者。
LLM提示: "订单频率的持续性"。
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
        
        # 获取 num_orders 序列
        orders_seq = []
        bid_orders_seq = []
        ask_orders_seq = []
        
        for _, row in grp.iterrows():
            no = row.get('num_orders', 0) or 0
            orders_seq.append(no)
            
            has_orders = any(f'bid_num_orders{i}' in row.index for i in range(1, 4))
            if has_orders:
                bo = sum(row.get(f'bid_num_orders{i}', 0) or 0 for i in range(1, 11))
                ao = sum(row.get(f'ask_num_orders{i}', 0) or 0 for i in range(1, 11))
            else:
                bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
                av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
                tv = bv + av
                bo = bv / (tv + 1e-8) * no
                ao = av / (tv + 1e-8) * no
            
            bid_orders_seq.append(bo)
            ask_orders_seq.append(ao)
        
        orders_seq = np.array(orders_seq, dtype=float)
        bid_orders_seq = np.array(bid_orders_seq, dtype=float)
        ask_orders_seq = np.array(ask_orders_seq, dtype=float)
        
        # 计算lag-1自相关
        def acf1(x):
            if len(x) < 5 or np.std(x) < 1e-12:
                return 0.0
            x_dm = x - np.mean(x)
            num = np.dot(x_dm[1:], x_dm[:-1])
            den = np.dot(x_dm, x_dm)
            return num / den if den > 1e-12 else 0.0
        
        total_acf = acf1(orders_seq)
        bid_acf = acf1(bid_orders_seq)
        ask_acf = acf1(ask_orders_seq)
        
        # 高自相关 = 真实流动性 = 好
        # 买卖方自相关差异: bid自相关>ask = 买方订单流更稳定
        qsr = total_acf + (bid_acf - ask_acf)
        
        results.append({'date': dt, 'instrument': inst, 'factor': qsr})
    
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
