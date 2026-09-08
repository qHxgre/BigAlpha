"""
因子 46: RDC — 反转×深度交互 (Reversal-Depth Concentration)
============================================================

经济逻辑:
午后反转幅度 / (1 + 深度集中度) 捕捉反转信号的可靠性。
深度集中在近档(高DFL) = 流动性脆弱 → 反转容易被操纵 → 信号打折。
深度均匀分布(低集中度) = 流动性健康 → 反转反映真实供需 → 信号可信。
午后反转发生在流动性差的环境 = 假反转 → 看跌。
午后反转发生在流动性好的环境 = 真反转 → 看涨(反转向上时)。

方法：
- 午后反转 = (上午末价格 - 下午末价格) / 上午末价格
- 深度集中度 = 前3档占比
- RDC = 反转幅度 / (1 + 深度集中度)

方向: RDC 越大 → 午后买盘反转且深度健康 → 预期正收益

AI应用说明:
反转策略需要区分"真反转"和"流动性幻觉"。
LLM提示: "下午反转的可信度"。
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
        
        if len(grp) < 20:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        mids = []
        depth_concs = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            if pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0:
                mids.append((ap1 + bp1) / 2)
            else:
                if mids:
                    mids.append(mids[-1])
                else:
                    mids.append(np.nan)
            
            # 深度集中度: 前3/总
            bv = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            av = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            total = bv.sum() + av.sum()
            front = bv[:3].sum() + av[:3].sum()
            depth_concs.append(front / total if total > 0 else 1.0)
        
        mids = np.array(mids)
        depth_concs = np.array(depth_concs)
        
        valid = ~np.isnan(mids)
        if valid.sum() < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        mp = mids[valid]
        dc = depth_concs[valid]
        
        # 分成上午(前50%)和下午(后50%)
        n = len(mp)
        n_morning = n // 2
        n_afternoon = n - n_morning
        
        if n_morning < 3 or n_afternoon < 3:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        morning_end = np.mean(mp[n_morning - 3:n_morning])  # 上午末3点平均
        afternoon_end = np.mean(mp[-3:])  # 下午末3点平均
        
        # 午后反转 = 下午末 - 上午末 / 上午末
        reversal = (afternoon_end - morning_end) / (morning_end + 1e-8)
        
        # 下午时段的平均深度集中度
        afternoon_dc = np.mean(dc[n_morning:]) if len(dc[n_morning:]) > 0 else 1.0
        
        # RDC = reversal / (1 + concentration)
        # 集中度高→分母大→reversal信号打折
        rdc = reversal / (1.0 + afternoon_dc)
        
        results.append({'date': dt, 'instrument': inst, 'factor': rdc})
    
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
