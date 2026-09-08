"""
因子 36: SLP — 滑点预期 (Slippage Expectation)
================================================

经济逻辑:
模拟扫单100万元的预期滑点成本，反映大单执行的真实成本。
滑点成本高 → 大资金进出困难 → 流动性溢价 → 看跌(取负)。
滑点成本低 → 流动性充裕 → 交易成本低 → 看涨。

方法：
- 从ask第1档开始累计成交量至目标金额
- 计算成交均价与最优ask的偏离百分比
- SLP = -expected_slippage_bps

方向: SLP 越大 → 滑点成本越低 → 预期正收益

AI应用说明:
机构交易员最关心的执行成本。
LLM提示: "100万扫单要花多少滑点？"。
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    target_amount = 100 * 10000  # 100万
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        slip_vals = []
        
        for _, row in grp.iterrows():
            ask_prices = np.array([row.get(f'ask_price{i}', np.nan) for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            valid = pd.notna(ask_prices) & (ask_prices > 0) & (ask_vols > 0)
            if valid.sum() < 2:
                continue
            
            ap = ask_prices[valid]
            av = ask_vols[valid]
            
            ap1 = ap[0]
            
            # 累计金额
            cum_amount = np.cumsum(ap * av)
            
            if cum_amount[-1] < target_amount * 0.1:
                # 深度不足，滑点极大
                slip_bps = 500  # 500bps 上限
            else:
                # 找到达到目标金额所需的档位
                idx = np.searchsorted(cum_amount, target_amount)
                idx = min(idx, len(ap) - 1)
                
                # 截断到那一档
                vols_to_use = av[:idx + 1].copy()
                if idx < len(cum_amount):
                    # 最后一档部分使用
                    if idx == 0:
                        partial = target_amount
                    else:
                        partial = target_amount - cum_amount[idx - 1]
                    vols_to_use[-1] = min(vols_to_use[-1], partial / ap[idx])
                
                # 加权平均成交价
                vwap = np.dot(ap[:idx + 1], vols_to_use) / (vols_to_use.sum() + 1e-8)
                
                # 滑点 = (vwap - ap1) / ap1，单位bps
                slip_bps = (vwap - ap1) / ap1 * 10000
            
            slip_vals.append(slip_bps)
        
        if not slip_vals:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 平均滑点，取负(滑点小=好)
        # 取对数避免极端值
        avg_slip = np.mean(slip_vals)
        factor = -np.log1p(avg_slip)
        
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
