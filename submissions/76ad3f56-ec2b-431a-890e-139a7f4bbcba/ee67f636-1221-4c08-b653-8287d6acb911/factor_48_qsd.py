"""
因子 48: QSD — 报价填充检测 (Quote Stuffing Detection)
========================================================

经济逻辑:
num_orders 极高(>95分位)的分钟数占比检测报价填充行为。
报价填充 → 高频挂单后快速撤单 → 制造虚假流动性 → 盘口噪音 → 负alpha。
高QSD比例 → 该股票的微观结构被噪音污染 → 价格信号质量差 → 看跌(取负)。

方法：
- 对每个股票，统计 num_orders > 日内95分位的分钟比例
- QSD = -填充比例 → 越高填充越看跌

方向: QSD 越大(即填充比例越低) → 盘口干净 → 预期正收益

AI应用说明:
打击市场操纵的监管关注点之一。
LLM提示: "异常高频的委托频率"。
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
        
        # 获取 num_orders 序列
        orders = []
        for _, row in grp.iterrows():
            no = row.get('num_orders', 0) or 0
            orders.append(no)
        
        orders = np.array(orders, dtype=float)
        
        if len(orders) < 5 or np.max(orders) < 1:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 日内95分位
        threshold = np.percentile(orders, 95)
        
        if threshold <= 1:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 填充比例
        stuffing_ratio = np.mean(orders > threshold)
        
        # QSD = -填充率 → 填充少=正
        qsd = -stuffing_ratio
        
        results.append({'date': dt, 'instrument': inst, 'factor': qsd})
    
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
