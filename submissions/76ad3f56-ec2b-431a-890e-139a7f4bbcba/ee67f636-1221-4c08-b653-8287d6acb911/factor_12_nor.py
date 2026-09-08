"""
因子 12: NOR — 委托笔数比 (Number-of-Orders Ratio)
===================================================

经济逻辑:
委托笔数(num_orders)反映订单碎片化程度。大单会被拆成多笔小单提交。
近档笔数多+远档笔数少 = 算法拆单执行中 = 机构在交易
近档笔数/远档笔数的比值变化捕捉算法交易行为。
卖方近档委托笔数 >> 买方近档 → 机构在卖出 → 看跌 → 取反。

方向: NOR取反 → 卖方拆单越活跃 → 越看跌 → 取负号后正向

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
        nor_vals = []
        
        for _, row in grp.iterrows():
            # 检查是否有委托笔数数据 (ask_num_orders / bid_num_orders)
            has_orders = any(f'bid_num_orders{i}' in row.index for i in range(1, 4))
            
            if has_orders:
                # 买盘近档(1-3)委托笔数 vs 远档(8-10)
                bid_near = sum(row.get(f'bid_num_orders{i}', 0) or 0 for i in range(1, 4))
                bid_far = sum(row.get(f'bid_num_orders{i}', 0) or 0 for i in range(8, 11))
                ask_near = sum(row.get(f'ask_num_orders{i}', 0) or 0 for i in range(1, 4))
                ask_far = sum(row.get(f'ask_num_orders{i}', 0) or 0 for i in range(8, 11))
                
                bid_ratio = bid_near / (bid_far + 1)  # 近档/远档笔数比
                ask_ratio = ask_near / (ask_far + 1)
                
                if bid_ratio > 0 and ask_ratio > 0:
                    # 买盘拆单活跃 vs 卖盘拆单活跃
                    nor_vals.append(np.log(bid_ratio + 1) - np.log(ask_ratio + 1))
            else:
                # 回退: 用成交量近似笔数 (volume/total_depth 比值)
                bid_vol_near = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 4))
                ask_vol_near = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 4))
                if bid_vol_near > 0 and ask_vol_near > 0:
                    nor_vals.append(np.log(bid_vol_near + 1) - np.log(ask_vol_near + 1))
        
        factor = np.mean(nor_vals) if nor_vals else 0.0
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
