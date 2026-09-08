"""
因子 26: HDI — 深度Herfindahl指数 (Herfindahl Depth Index)
===========================================================

经济逻辑:
10档深度的Herfindahl-Hirschman集中度指数检测单一档位的流动性垄断。
HHI 高 → 深度集中在某一档 → 可能是冰山订单/隐藏大单 → 信息含量高。
买方高集中度 → 大买单隐藏在某档 → 看涨。
卖方高集中度 → 大卖单隐藏在某档 → 看跌。

方法：
- 对bid/ask各侧: HHI = Σ(share_i²), share_i = vol_i / total_vol
- 计算每档单独贡献以定位集中档位
- HDI = bid_HHI - ask_HHI（调整后）

方向: HDI 越大 → 买盘集中度高于卖盘 → 大买单隐藏 → 预期正收益

AI应用说明:
借用产业经济学中的HHI指数度量市场集中度。
LLM提示: "哪个档位有异常集中的深度？"。
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
        hdi_vals = []
        
        for _, row in grp.iterrows():
            bid_vols = np.array([row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            ask_vols = np.array([row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11)], dtype=float)
            
            bid_hhi = _compute_hhi(bid_vols)
            ask_hhi = _compute_hhi(ask_vols)
            
            if bid_hhi is None or ask_hhi is None:
                continue
            
            # 买盘集中度高出卖盘 = 买盘有大单 → 看涨
            hdi_vals.append(bid_hhi - ask_hhi)
        
        factor = np.mean(hdi_vals) if hdi_vals else 0.0
        results.append({'date': dt, 'instrument': inst, 'factor': factor})
    
    result = pd.DataFrame(results)
    return _normalize(result)


def _compute_hhi(vols):
    """计算Herfindahl指数"""
    total = vols.sum()
    if total < 1:
        return None
    shares = vols / total
    hhi = np.sum(shares ** 2)
    return hhi


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
