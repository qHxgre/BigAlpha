"""
因子 42: PSR — 盘前压力残留 (Pre-market Stress Residue)
========================================================

经济逻辑:
开盘前 bid_vol/ask_vol 的对比持续性反映隔夜信息的消化程度。
开盘初段 bid/ask 深度比偏离1且持续 → 隔夜信息未完全消化 → 方向性机会。
bid持续>ask(前段) → 隔夜利好 → 买方承接积极 → 看涨。
ask持续>bid → 隔夜利空 → 卖方压力持续 → 看跌。

方法：
- 取前15分钟的bid/ask深度比序列
- PSR = 前15分钟bid/ask的中位数 - 后段bid/ask的中位数
- 如果前段比后段高且bid>ask → 隔夜利好正在消化 → 看涨

方向: PSR 越大 → 开盘买盘压力残留转为买入 → 预期正收益

AI应用说明:
A股特有的开盘竞价和盘前信息消化。
LLM提示: "前15分钟订单簿反映什么隔夜信息？"。
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
        
        if len(grp) < 20:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 计算每分钟的bid/ask比
        ratios = []
        for _, row in grp.iterrows():
            bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            total = bv + av
            if total > 10:
                ratios.append(bv / total)  # bid占比
            else:
                ratios.append(0.5)
        
        ratios = np.array(ratios)
        
        # 前15分钟(前15%)和后段
        n_early = max(len(ratios) // 6, 2)  # 约前15%
        n_late = max(len(ratios) // 4, 3)
        
        early_ratio = np.mean(ratios[:n_early])
        late_ratio = np.mean(ratios[-n_late:])
        full_ratio = np.mean(ratios)
        
        # 开盘买盘偏强且持续(前段high后段也high) = 利好驱动
        # 开盘买盘偏强但衰减(前段high后段low) = 利好出尽
        # 开盘卖盘偏强(前段low) = 利空压力
        
        # PSR: 前段bid占比 - 0.5, 用全段方向修正
        # 前段bid强 + 全段方向 = 持续买盘
        psr = (early_ratio - 0.5) * np.sign(full_ratio - 0.5)
        
        # 也考虑前段到后段的变化方向
        trend = late_ratio - early_ratio
        
        # 前段bid强且趋势维持或增强 = 好
        factor = psr * 2 + trend
        
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
