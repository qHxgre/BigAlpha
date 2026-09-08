"""
因子 19: VAF — 成交量异常分数 (Volume Anomaly Factor)
=====================================================

经济逻辑:
将每只股票的日内成交量模式与其自身的历史模式比较。
成交量异常偏离历史均值 = 有信息事件
异常放量+价格上涨 = 利多信息
异常放量+价格下跌 = 利空信息
用 (当日量/过去N日均量 - 1) × 日内价格方向 衡量。

方向: 异常放量+涨 → 看涨 → 正因子

AI技术: LLM生成公式 → 硬编码提交
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # 按日期和股票聚合日频数据
    daily = df.groupby(['date', 'instrument']).agg(
        total_volume=('volume', 'sum'),
        avg_price=('price', 'mean') if 'price' in df.columns else ('close', 'mean'),
        open_price=('open', 'first') if 'open' in df.columns else ('price', 'first'),
        close_price=('price', 'last') if 'price' in df.columns else ('close', 'last'),
    ).reset_index()
    
    daily = daily.sort_values(['instrument', 'date'])
    
    results = []
    lookback = 20  # 历史窗口
    
    for inst, grp in daily.groupby('instrument'):
        grp = grp.sort_values('date')
        vols = grp['total_volume'].values
        closes = grp['close_price'].values
        
        if len(grp) < lookback:
            for i in range(len(grp)):
                results.append({
                    'date': grp.iloc[i]['date'],
                    'instrument': inst,
                    'factor': 0.0
                })
            continue
        
        for i in range(len(grp)):
            if i < lookback:
                results.append({
                    'date': grp.iloc[i]['date'],
                    'instrument': inst,
                    'factor': 0.0
                })
                continue
            
            # 历史均量和标准差
            hist_vol = vols[i-lookback:i]
            vol_mean = np.mean(hist_vol)
            vol_std = np.std(hist_vol)
            
            if vol_std < 1e-8:
                results.append({
                    'date': grp.iloc[i]['date'],
                    'instrument': inst,
                    'factor': 0.0
                })
                continue
            
            # 成交量异常: z-score
            vol_z = (vols[i] - vol_mean) / vol_std
            
            # 价格方向
            if i > 0:
                price_ret = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-8)
            else:
                price_ret = 0
            
            # 异常量 × 方向 = 信息驱动方向
            # 大正z+涨 = 利多放量 → 正向
            # 大正z+跌 = 利空放量 → 负向
            factor = vol_z * np.sign(price_ret) * min(abs(vol_z), 3)  # 截断极端值
            
            results.append({
                'date': grp.iloc[i]['date'],
                'instrument': inst,
                'factor': factor
            })
    
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
