"""
因子 18: GPR — 缺口压力释放 (Gap Pressure Release)
===================================================

经济逻辑:
开盘跳空(pre_close→open)后，观察日内价格是否回补缺口。
不完全回补的缺口 = 存在持续压力(买方或卖方)
计算: (当日价格极值 - 开盘价) / abs(开盘跳空幅度)
回补率 < 0.5 → 压力持续 → 跳空方向延续至次日

方向: 向上跳空+不完全回补 → 买压持续 → 正因子
      向下跳空+不完全回补 → 卖压持续 → 负因子

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
        
        first = grp.iloc[0]
        pre_close = first.get('pre_close', np.nan)
        open_price = first.get('open', np.nan)
        
        if not (pd.notna(pre_close) and pd.notna(open_price) and pre_close > 0):
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        gap = (open_price - pre_close) / pre_close
        
        if abs(gap) < 0.001:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        high = grp['high'].max() if 'high' in grp.columns else grp['price'].max()
        low = grp['low'].min() if 'low' in grp.columns else grp['price'].min()
        close = grp['price'].iloc[-1] if 'price' in grp.columns else grp['close'].iloc[-1]
        
        if gap > 0:
            # 向上跳空: 最低价回补程度
            retrace = (open_price - low) / (open_price - pre_close + 1e-8)
            # 回补少 = 压力强 = 继续看涨
            factor = gap * (1 - min(retrace, 1.0))
        else:
            # 向下跳空: 最高价回补程度
            retrace = (high - open_price) / (pre_close - open_price + 1e-8)
            # 向下跳空+回补少 = 继续看跌 → 负因子
            factor = gap * (1 - min(retrace, 1.0))
        
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
