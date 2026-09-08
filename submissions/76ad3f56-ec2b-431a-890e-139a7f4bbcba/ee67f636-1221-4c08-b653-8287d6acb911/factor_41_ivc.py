"""
因子 41: IVC — 日内波动聚类 (Intraday Volatility Clustering)
=============================================================

经济逻辑:
30分钟滚动已实现波动的相对集中度捕捉波动聚集现象。
波动高度聚集 → 不确定性集中释放 → 信息事件 → 趋势可能即将反转。
波动均匀 → 持续的信息消化 → 趋势可信。

方法：
- 计算30分钟滚动已实现波动率
- IVC = rolling_vol_ratio = 最近30分钟波动 / 全天波动
- 取波动聚集的方向性: 波动聚集+当前OBI方向

方向: IVC 越大 → 波动聚集+买盘主导 → 信息释放充分后看涨
  注意：纯波动聚集是风险信号，需配合方向

AI应用说明:
波动聚集是金融时间序列的典型特征。
LLM提示: "波动在什么时候集中爆发？"。
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    window = 30  # 30分钟滚动窗口
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        
        if len(grp) < window:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 计算每分钟中间价和OBI
        mid_prices = []
        obi_vals = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            
            if pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0:
                mid_prices.append((ap1 + bp1) / 2)
            else:
                mid_prices.append(np.nan)
            
            bv = row.get('bid_volume1', 0) or 0
            av = row.get('ask_volume1', 0) or 0
            total = bv + av
            obi_vals.append((bv - av) / total if total > 0 else 0)
        
        mid_prices = np.array(mid_prices)
        
        # 计算每分钟收益率(用前向填充缺失)
        valid_mask = ~np.isnan(mid_prices)
        if valid_mask.sum() < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 简单用mid变化做收益率
        returns = np.diff(mid_prices[valid_mask]) / (mid_prices[valid_mask][:-1] + 1e-12)
        
        if len(returns) < window // 2:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 滚动波动率
        rolling_vols = []
        for i in range(len(returns) - min(window, len(returns)) + 1):
            rw = returns[i:i + min(window, len(returns))]
            rolling_vols.append(np.std(rw))
        
        rolling_vols = np.array(rolling_vols)
        
        if len(rolling_vols) < 2 or np.mean(rolling_vols) < 1e-12:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 波动聚类 = 最近波动 / 平均波动
        recent_vol = np.mean(rolling_vols[-min(5, len(rolling_vols)):])
        avg_vol = np.mean(rolling_vols)
        
        vol_cluster = recent_vol / (avg_vol + 1e-12)
        
        # 取log使分布更正态
        log_cluster = np.log(vol_cluster + 1e-8)
        
        # 配合OBI方向: OBI买盘强+波动聚集=买盘在吸收信息
        obi_mean = np.mean(obi_vals) if obi_vals else 0
        
        ivc = log_cluster * np.sign(obi_mean) if obi_mean != 0 else -log_cluster
        
        results.append({'date': dt, 'instrument': inst, 'factor': ivc})
    
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
