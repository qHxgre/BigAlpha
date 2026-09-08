"""
因子 14: BSR — 买卖价弹性比 (Bid-Ask Elasticity Ratio)
======================================================

经济逻辑:
价格每远离1档(到2档、3档...)，挂单量如何变化？这就是"弹性"。
买方弹性 = bid_volume变化率 / bid_price变化率
卖方弹性 = ask_volume变化率 / ask_price变化率
弹性比 ≠ 1 → 供需不对称 → 价格将向弹性更低(更刚性)的方向移动。
卖方更刚性(弹性小) = 卖盘不愿降价 = 买方需要支付更高溢价 → 看涨。

方向: 卖方弹性 < 买方弹性 → 卖方更刚性 → 看涨 → 正因子

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
        elasticity_vals = []
        
        for _, row in grp.iterrows():
            # 提取价格和量
            bid_px = np.array([row.get(f'bid_price{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_px = np.array([row.get(f'ask_price{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            bid_vol = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vol = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            # 过滤无效
            valid_bid = (bid_px > 0) & (bid_vol > 0)
            valid_ask = (ask_px > 0) & (ask_vol > 0)
            
            if valid_bid.sum() < 5 or valid_ask.sum() < 5:
                continue
            
            bpx = bid_px[valid_bid]
            bvol = bid_vol[valid_bid]
            apx = ask_px[valid_ask]
            avol = ask_vol[valid_ask]
            
            # 价格变化(各档之间) - 使用绝对变化避免零除
            bpx_diff = np.diff(bpx)
            apx_diff = np.diff(apx)
            bvol_diff = np.diff(bvol)
            avol_diff = np.diff(avol)
            
            # 过滤价格变化太小的档位
            bid_valid = np.abs(bpx_diff) > 1e-6
            ask_valid = np.abs(apx_diff) > 1e-6
            
            if not bid_valid.any() or not ask_valid.any():
                continue
            
            # 弹性 = d(volume)/d(price)
            bid_elast = np.median(bvol_diff[bid_valid] / bpx_diff[bid_valid])
            ask_elast = np.median(avol_diff[ask_valid] / apx_diff[ask_valid])
            
            # 卖方更刚性(弹性低/绝对值小) = 看涨
            if abs(bid_elast) > 1e-12:
                elasticity_vals.append(np.sign(bid_elast) * np.log1p(abs(bid_elast)) - 
                                     np.sign(ask_elast) * np.log1p(abs(ask_elast)))
        
        factor = np.mean(elasticity_vals) if elasticity_vals else 0.0
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
