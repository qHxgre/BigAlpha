"""
因子 15: FTR — 流动性周转率 (Flow Turnover Ratio)
==================================================

经济逻辑:
对比"总成交量"与"总挂单量"的比率来推断日内资金周转速度。
高周转率 = 交易活跃 + 挂单被快速消耗 = 信息驱动交易
成交量/挂单总量 高于均值的时段 → 知情交易概率上升
用日内周转率的变化幅度(峰度)来度量信息的集中释放。

方向: 高周转峰度 → 信息集中释放 → 方向由OBI决定 → 正因子

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
        
        turnovers = []
        obis = []
        
        for _, row in grp.iterrows():
            vol = row.get('volume', 0) or 0
            
            # 总挂单量 (前5档)
            depth = 0
            for i in range(1, 6):
                bv = row.get(f'bid_volume{i}', 0) or 0
                av = row.get(f'ask_volume{i}', 0) or 0
                depth += bv + av
            
            if depth > 0:
                turnovers.append(vol / depth)
            
            # 同时记录OBI
            bv1 = row.get('bid_volume1', 0) or 0
            av1 = row.get('ask_volume1', 0) or 0
            tot = bv1 + av1
            if tot > 0:
                obis.append((bv1 - av1) / tot)
        
        if len(turnovers) < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        turnovers = np.array(turnovers)
        
        # 周转率的集中度：高周转是否集中在少数分钟
        mean_t = np.mean(turnovers)
        max_t = np.max(turnovers)
        skew_t = np.sum(((turnovers - mean_t) / (np.std(turnovers) + 1e-8)) ** 3) / len(turnovers) if np.std(turnovers) > 1e-8 else 0
        
        # 峰度: 极高周转 = 信息爆发
        kurt_t = np.sum(((turnovers - mean_t) / (np.std(turnovers) + 1e-8)) ** 4) / len(turnovers) - 3 if np.std(turnovers) > 1e-8 else 0
        
        # 周转集中度: (max/mean - 1)
        concentration = max_t / (mean_t + 1e-8) - 1
        
        # 方向: 信息爆发 + OBI方向 = 知情交易方向
        obi_dir = np.sign(np.mean(obis)) if obis else 0
        
        # 高集中 + 高峰度 = 信息事件
        # 配合OBI方向判断涨跌
        factor = (concentration + max(0, kurt_t)) * obi_dir
        
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
