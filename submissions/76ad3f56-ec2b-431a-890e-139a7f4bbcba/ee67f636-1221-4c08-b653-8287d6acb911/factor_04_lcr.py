"""
因子 04: LCR — 流动性级联风险 (Liquidity Cascade Risk)
=======================================================

经济逻辑:
市场深度的突然变薄(Liquidity Cascade)是流动性危机的前兆。
当各档挂单量在短时间内急剧萎缩，做市商撤退 → 未来价格波动加剧。
捕捉"深度崩塌"的信号：对比当前深度与近期平均深度。

方向: LCR 小(负值大)= 深度崩塌 → 未来反弹概率高 → 取负号使因子正向

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
        
        if len(grp) < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 每分钟总深度(bid+ask前5档)
        depths = []
        for _, row in grp.iterrows():
            depth = 0
            for i in range(1, 6):
                bv = row.get(f'bid_volume{i}', 0) or 0
                av = row.get(f'ask_volume{i}', 0) or 0
                depth += bv + av
            depths.append(depth)
        
        depths = np.array(depths)
        
        # 前50%时间的平均深度 vs 后10%时间的深度
        n = len(depths)
        first_half_avg = np.mean(depths[:n//2]) if n >= 2 else depths[0]
        last_tail_avg = np.mean(depths[-max(1, n//10):])
        
        if first_half_avg < 1:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 深度崩塌比例
        cascade = (last_tail_avg - first_half_avg) / first_half_avg
        
        # 同时考虑崩溃的突然性（深度最小值/平均值）
        min_depth = np.min(depths)
        avg_depth = np.mean(depths)
        fragility = min_depth / avg_depth if avg_depth > 0 else 1.0
        
        # 综合: 深度变化 × 脆弱性
        # cascade < 0 = 尾盘深度减少 = 风险
        # fragility < 1 = 存在脆弱时刻
        # 取负号: 深度崩塌 → 超卖 → 可能反弹 → 正因子
        factor = -(cascade * fragility)
        
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
