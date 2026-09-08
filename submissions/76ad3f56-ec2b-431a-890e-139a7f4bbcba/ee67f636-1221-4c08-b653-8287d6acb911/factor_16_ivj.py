"""
因子 16: IVJ — 日内波动跳跃 (Intraday Volatility Jump)
======================================================

经济逻辑:
分钟收益率的极端跳跃(>3倍日内波动率)代表信息冲击。
跳跃方向如果是正向且伴随放量 = 利好信息释放 → 趋势延续
跳跃方向如果是负向且缩量 = 恐慌噪声 → 即将反弹
用跳跃检测 + 量价条件判断方向。

方向: 正面跳跃+放量 → 看涨 → 正因子

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
        
        prices = grp['price'].values if 'price' in grp.columns else grp['close'].values
        volumes = grp['volume'].values if 'volume' in grp.columns else np.ones(len(prices))
        
        if len(prices) < 30:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 分钟收益率
        rets = np.diff(prices) / (prices[:-1] + 1e-8)
        
        # 日内波动率 (排除跳跃)
        ret_med = np.median(rets)
        ret_mad = np.median(np.abs(rets - ret_med)) * 1.4826
        
        # 跳跃阈值: 3倍MAD
        threshold = 3 * ret_mad if ret_mad > 1e-8 else 0.01
        
        # 检测跳跃
        jump_signals = []
        for i in range(1, len(rets)):
            if abs(rets[i]) > threshold:
                # 跳跃方向 × 量相对于前面5分钟均量的变化
                vol_ratio = volumes[i+1] / (np.mean(volumes[max(0,i-4):i+1]) + 1e-8) if i+1 < len(volumes) else 1
                jump_signals.append(rets[i] * vol_ratio)
        
        if not jump_signals:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 跳跃强度: 平均跳跃信号 × 跳跃频率
        avg_jump = np.mean(jump_signals)
        jump_freq = len(jump_signals) / len(rets)
        
        # 正向跳跃+放量 = 利好信息
        factor = avg_jump * np.sqrt(jump_freq)
        
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
