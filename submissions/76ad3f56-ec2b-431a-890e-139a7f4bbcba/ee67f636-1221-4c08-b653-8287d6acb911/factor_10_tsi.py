"""
因子 10: TSI — 时间分段信息度 (Time-Segmented Information)
=========================================================

经济逻辑:
不同时段的交易行为反映不同类型投资者：
- 开盘(前30分钟): 零售投资者 + 隔夜信息消化
- 上午(10:00-11:30): 机构调仓
- 下午(13:00-14:30): 趋势跟随
- 尾盘(最后30分钟): 知情交易/做市商对冲

对比各时段盘口不平衡度的变化方向，捕捉机构vs散户行为差异。

方向: 尾盘买盘压力 > 开盘买盘压力 → 机构建仓 → 正因子

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
        
        time_col = 'time' if 'time' in grp.columns else None
        if 'datetime' in grp.columns:
            dt_series = pd.to_datetime(grp['datetime'])
            hours = dt_series.dt.hour.values
            minutes = dt_series.dt.minute.values
            total_minutes = hours * 60 + minutes
        elif time_col:
            # time 格式 HHMMSSmmm, 取前4位 HHMM
            total_minutes = grp[time_col].values.astype(int) // 100000
        else:
            total_minutes = np.arange(len(grp))
        
        # 定义时段（用实际小时判断）
        def segment(hour, minute):
            t = hour * 60 + minute
            if t < 600:    # 9:30-10:00 开盘
                return 'open'
            elif t < 690:  # 10:00-11:30 上午
                return 'morning'
            elif t < 840:  # 13:00-14:00 午后
                return 'afternoon'
            else:          # 14:00-15:00 尾盘
                return 'close'
        
        segments = {'open': [], 'morning': [], 'afternoon': [], 'close': []}
        
        for i, (_, row) in enumerate(grp.iterrows()):
            h = hours[i] if i < len(hours) else 9
            m = minutes[i] if i < len(minutes) else 30
            seg = segment(h, m)
            
            # 计算盘口不平衡
            bv = row.get('bid_volume1', 0) or 0
            av = row.get('ask_volume1', 0) or 0
            total = bv + av
            obi = (bv - av) / total if total > 0 else 0
            segments[seg].append(obi)
        
        # 各时段平均 OBI
        seg_means = {}
        for seg, vals in segments.items():
            seg_means[seg] = np.mean(vals) if vals else 0.0
        
        # 核心: 尾盘买压 vs 开盘买压
        close_pressure = seg_means.get('close', 0)
        open_pressure = seg_means.get('open', 0)
        
        # 机构vs散户: 机构倾向尾盘操作(减少冲击成本)
        # 尾盘买压上升 = 机构净买入 = 看涨
        factor = close_pressure - open_pressure
        
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
