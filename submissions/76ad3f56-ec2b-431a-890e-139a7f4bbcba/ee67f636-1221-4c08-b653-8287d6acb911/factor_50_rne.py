"""
因子 50: RNE — 整数效应 (Round-Number Effect)
================================================

经济逻辑:
成交价接近整数位的比例捕捉行为金融偏差。
整数位附近的挂单更密集 → 整数位成为支撑/阻力 → 行为偏差。
某股票成交价频繁接近整数位 → 散户行为主导 → 噪音交易 → 负alpha(取负)。
成交量在非整数位更多 → 机构在交易 → 信息驱动 → 正alpha。

方法：
- 用mid_price接近整数位的程度衡量
- RNE = -|mid_price - round(mid_price)| 的平均归一化值
- 离整数位越远(非整数位活跃) → 机构主导 → 看涨

方向: RNE 越大 → 远离整数位(机构交易) → 预期正收益

AI应用说明:
行为金融学的整数效应在订单簿中的体现。
LLM提示: "整数价位附近有异常吗？"。
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
        
        round_proximities = []
        
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            
            if not (pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0):
                continue
            
            mid = (ap1 + bp1) / 2
            
            # 确定整数位精度
            magnitude = np.floor(np.log10(mid + 1e-8))
            
            if magnitude >= 2:
                # 价格>100: 用整数位
                step = 1.0
            elif magnitude >= 1:
                # 价格10-100: 用0.5
                step = 0.5
            elif magnitude >= 0:
                # 价格1-10: 用0.1
                step = 0.1
            else:
                # 价格<1: 用0.01
                step = 0.01
            
            # 距离最近整数(按step取整)的距离
            nearest_round = np.round(mid / step) * step
            dist_to_round = abs(mid - nearest_round) / step
            
            round_proximities.append(dist_to_round)
        
        if not round_proximities:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 平均距离整数位的远近
        # 距离远 = 非整数位交易活跃 = 机构主导 = 好
        avg_dist = np.mean(round_proximities)
        
        # 也考虑距离的稳定性(如果总是很近=散户行为稳定)
        std_dist = np.std(round_proximities) if len(round_proximities) > 1 else 0
        
        # RNE: 远离整数位(非整数位活跃)且稳定的 = 好
        # avg_dist越大越好, std越小越好
        rne = avg_dist - std_dist * 0.5
        
        results.append({'date': dt, 'instrument': inst, 'factor': rne})
    
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
