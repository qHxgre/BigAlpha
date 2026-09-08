"""
因子 38: LCD — 档位相关性衰减 (Level Correlation Decay)
=========================================================

经济逻辑:
近档(1-3)与远档(8-10)深度变化的相关性反映跨档联动的信息传递。
相关性高 → 全部档位同步变动 → 市场反应一致 → 信息有共识。
相关性低 → 近档和远档脱钩 → 知情交易者只操作近档 → 信息不对称。
买盘联动 > 卖盘联动 → 买盘信息传递更一致 → 看涨。

方法：
- 对每分钟的近3档深度变化和远3档深度变化计算Spearman秩相关
- 用滑动窗口(如20分钟)计算滚动相关
- LCD = bid滚动相关 - ask滚动相关

方向: LCD 越大 → 买盘跨档信息传递更一致 → 预期正收益

AI应用说明:
订单簿中的信息扩散速度。
LLM提示: "近档深度变化能预测远档深度变化吗？"。
"""

import pandas as pd
import numpy as np
from scipy import stats


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    window = min(20, 15)
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        
        if len(grp) < window + 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 各档深度序列
        bid_near_seq = []  # 近3档平均
        bid_far_seq = []   # 远3档平均
        ask_near_seq = []
        ask_far_seq = []
        
        for _, row in grp.iterrows():
            b_near = np.mean([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 4)])
            b_far = np.mean([row.get(f'bid_volume{i}', 0) or 0 for i in range(8, 11)])
            a_near = np.mean([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 4)])
            a_far = np.mean([row.get(f'ask_volume{i}', 0) or 0 for i in range(8, 11)])
            
            bid_near_seq.append(b_near)
            bid_far_seq.append(b_far)
            ask_near_seq.append(a_near)
            ask_far_seq.append(a_far)
        
        # 计算深度变化(一阶差分)
        dbn = np.diff(bid_near_seq)
        dbf = np.diff(bid_far_seq)
        dan = np.diff(ask_near_seq)
        daf = np.diff(ask_far_seq)
        
        if len(dbn) < window:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 滚动相关
        bid_corrs = []
        ask_corrs = []
        
        for i in range(len(dbn) - window + 1):
            w_bn = dbn[i:i + window]
            w_bf = dbf[i:i + window]
            w_an = dan[i:i + window]
            w_af = daf[i:i + window]
            
            if np.std(w_bn) > 0 and np.std(w_bf) > 0:
                bid_c, _ = stats.spearmanr(w_bn, w_bf)
                bid_corrs.append(bid_c if not np.isnan(bid_c) else 0)
            
            if np.std(w_an) > 0 and np.std(w_af) > 0:
                ask_c, _ = stats.spearmanr(w_an, w_af)
                ask_corrs.append(ask_c if not np.isnan(ask_c) else 0)
        
        bid_avg_corr = np.mean(bid_corrs) if bid_corrs else 0
        ask_avg_corr = np.mean(ask_corrs) if ask_corrs else 0
        
        # 买盘联动 > 卖盘联动 → 买盘信息传递好 → 看涨
        lcd = bid_avg_corr - ask_avg_corr
        
        results.append({'date': dt, 'instrument': inst, 'factor': lcd})
    
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
