"""
因子 07: VPR — 成交量-价格共振 (Volume-Price Resonance)
========================================================

经济逻辑:
成交量与价格变化的"共振频率"反映市场参与者的协调程度。
高共振 = 一致行动 = 强趋势；低共振 = 分歧 = 均值回复。
用成交量与价格变化的动态相关性来衡量。

方向: 高共振+上涨 = 趋势延续 → 正因子

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
        
        if 'price' not in grp.columns or 'volume' not in grp.columns:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        prices = grp['price'].values
        volumes = grp['volume'].values
        
        if len(prices) < 30:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 分钟收益率
        rets = np.diff(prices) / (prices[:-1] + 1e-8)
        vols = volumes[1:]
        
        # 滑动窗口计算量价相关性(10分钟窗口)
        window = min(10, len(rets))
        cors = []
        for i in range(len(rets) - window + 1):
            r = rets[i:i+window]
            v = vols[i:i+window]
            if np.std(r) > 1e-8 and np.std(v) > 1e-8:
                cor = np.corrcoef(r, v)[0, 1]
                if not np.isnan(cor):
                    cors.append(cor)
        
        if not cors:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        avg_cor = np.mean(cors)
        cor_std = np.std(cors) if len(cors) > 1 else 0
        
        # 高平均相关性 + 低波动 = 一致的量价关系 = 强信号
        resonance = abs(avg_cor) / (1 + cor_std)
        
        # 价格方向
        price_dir = np.sign(prices[-1] - prices[0])
        
        # 正相关+涨 = 量增价涨 = 强趋势
        # 负相关+涨 = 量减价涨 = 弱趋势(背离)
        factor = resonance * price_dir * np.sign(avg_cor) if avg_cor != 0 else 0
        
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
