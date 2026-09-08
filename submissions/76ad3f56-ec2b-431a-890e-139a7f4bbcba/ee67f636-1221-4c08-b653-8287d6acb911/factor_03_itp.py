"""
因子 03: ITP — 知情交易概率 (Informed Trading Probability)
===========================================================

经济逻辑:
成交笔数(num_trades)异常高+买卖价差异常扩大的组合表示知情交易活跃。
知情交易者急于成交 → 大量拆单 → 笔数暴增 → 信息即将释放。
笔数高而价差正常 = 流动性好；笔数高+价差宽 = 知情交易信号。

方向: ITP 越大 → 知情交易概率越高 → 信息驱动方向(取 spread 收敛方向)

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
        
        # 计算每分钟的知情交易代理指标
        itp_vals = []
        for _, row in grp.iterrows():
            trades = row.get('num_trades', 0) or 0
            ask_px = row.get('ask_price1', np.nan)
            bid_px = row.get('bid_price1', np.nan)
            
            if not (pd.notna(ask_px) and pd.notna(bid_px) and ask_px > bid_px > 0):
                continue
            
            spread = (ask_px - bid_px) / ((ask_px + bid_px) / 2)
            
            # 笔数 × 价差 = 知情交易强度
            # 笔数取对数避免极端值主导
            itp = np.log1p(trades) * spread
            itp_vals.append(itp)
        
        if not itp_vals:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 取日内高值(知情交易往往集中在特定时段)
        # 用 90分位代替max避免噪声
        itp_90 = np.percentile(itp_vals, 90)
        itp_mean = np.mean(itp_vals)
        itp_std = np.std(itp_vals) if len(itp_vals) > 1 else 0
        
        # 高均值+高尾部 = 持续知情交易
        # 因子大 → 知情交易强 → 信息即将释放 → 看涨(取买盘方向)
        # 配合买卖压力方向修正
        obi_vals = []
        for _, row in grp.iterrows():
            bv = row.get('bid_volume1', 0) or 0
            av = row.get('ask_volume1', 0) or 0
            total = bv + av
            if total > 0:
                obi_vals.append((bv - av) / total)
        obi_mean = np.mean(obi_vals) if obi_vals else 0
        
        # ITP 强度 × OBI 方向 = 知情交易的方向性
        factor = itp_90 * np.sign(obi_mean) if obi_mean != 0 else 0
        
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
