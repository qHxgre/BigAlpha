"""
因子 49: SPS — 亚档位跳跃 (Sub-Penny Skipping)
================================================

经济逻辑:
成交价集中在 bid_price1 和 ask_price1 之间的比例异常。
- 正常情况: 成交均匀分布在bid-ask区间
- 成交集中在最优买卖价→ 订单在报价内部成交 → 暗池/内部化 → 信息不透明。
- SPS高 → 大量交易绕过order book显示 → 隐藏交易 → 信息泄露。

方法：
- 用代理: (第1档spread - 隐含有效spread) / 第1档spread
- 或者: num_trades / total_depth × spread 作为内部成交代理
- SPS = -异常比例 → 异常交易少 = 正因子

方向: SPS 越大 → 亚档位异常交易少 → 预期正收益

AI应用说明:
检测在visible order book之外成交的异常。
LLM提示: "成交价在bid-ask内部的比例"。
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
        
        if len(grp) < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        sps_vals = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            ap2 = row.get('ask_price2', np.nan)
            bp2 = row.get('bid_price2', np.nan)
            
            if not (pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0):
                continue
            
            spread = ap1 - bp1
            
            # 有last_price时可以直接算
            lp = row.get('last_price', row.get('close', np.nan))
            if pd.notna(lp) and lp > 0:
                # 成交价在bid-ask之间的偏离度
                mid = (ap1 + bp1) / 2
                deviation = abs(lp - mid) / (spread / 2 + 1e-8)
                # deviation接近1 = 在bid或ask附近; 接近0 = 在中间
                # 在中间成交 = 可能是内部化
                sps_vals.append(1.0 - min(deviation, 1.0))
            else:
                # 用盘口不均衡代理: 成交量/stickiness
                bv1 = row.get('bid_volume1', 0) or 0
                av1 = row.get('ask_volume1', 0) or 0
                total_v1 = bv1 + av1
                
                if total_v1 < 1:
                    continue
                
                # 如果第1档深度极度不均衡，说明有隐藏的成交在发生
                # 用(第1档spread - 第2档spread)作为信息代理
                if pd.notna(ap2) and pd.notna(bp2) and ap2 > bp2 > 0:
                    spread2 = ap2 - bp2
                    # spread收窄快 + 深度大 = 可能内部成交
                    sps_vals.append((spread - spread2) / (spread + 1e-8))
        
        if not sps_vals:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # SPS = -mean(异常程度)
        # 异常少 = 市场透明 = 正因子
        sps = -np.mean(sps_vals)
        
        results.append({'date': dt, 'instrument': inst, 'factor': sps})
    
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
