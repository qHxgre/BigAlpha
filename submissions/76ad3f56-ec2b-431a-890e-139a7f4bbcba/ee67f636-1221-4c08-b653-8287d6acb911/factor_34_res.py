"""
因子 34: RES — 实际价差 (Realized Effective Spread)
======================================================

经济逻辑:
有效价差 = 成交价与中间价的偏离 / 报价价差，反映真实交易成本。
成交价偏向ask → 买方支付的溢价高 → 买盘成本大 → 看跌(取负)。
成交价偏向bid → 卖方接受的折价大 → 卖盘压力大 → 看涨。
用 volume-weighted average price 或 last_price 替代成交价。

方法：
- RES = (mid_price - 代理成交价) / quoted_spread
- 代理成交价可用 (ask1*bid_vol1 + bid1*ask_vol1) / (bid_vol1+ask_vol1)
- RES > 0 表示成交更接近bid(买方有利) → 看涨
- RES < 0 表示成交更接近ask(卖方有利) → 取正后看跌

方向: RES 越大 → 交易成本越低(偏bid成交) → 预期正收益

AI应用说明:
实际交易成本比报价价差更能反映市场质量。
LLM提示: "真实的交易摩擦"。
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
        res_vals = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            
            if not (pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0):
                continue
            
            mid = (ap1 + bp1) / 2
            spread = ap1 - bp1
            
            if spread <= 0:
                continue
            
            # 代理成交价: 用第1档加权
            bv1 = row.get('bid_volume1', 0) or 0
            av1 = row.get('ask_volume1', 0) or 0
            total_v1 = bv1 + av1
            
            if total_v1 < 1:
                # 回退: 如果有last_price直接用
                lp = row.get('last_price', row.get('close', np.nan))
                if pd.notna(lp) and lp > 0:
                    proxy_px = lp
                else:
                    proxy_px = mid
            else:
                # 加权成交价: 深度大的一侧更可能成交
                proxy_px = (ap1 * bv1 + bp1 * av1) / total_v1
            
            # 有效价差: (mid - proxy_px) / spread
            # 正向 = proxy_px低于mid = 接近bid成交 = 买方优势
            effective = (mid - proxy_px) / spread
            
            res_vals.append(effective)
        
        factor = np.mean(res_vals) if res_vals else 0.0
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
