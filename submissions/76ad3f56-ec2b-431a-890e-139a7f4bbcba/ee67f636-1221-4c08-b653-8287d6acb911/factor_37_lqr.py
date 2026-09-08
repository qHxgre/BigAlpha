"""
因子 37: LQR — 流动性韧性 (Liquidity Resilience)
===================================================

经济逻辑:
大成交量后的下一分钟深度恢复比例反映市场吸收冲击的能力。
- 韧性高 → 深度快速恢复 → 流动性内生 → 市场健康 → 看涨
- 韧性低 → 深度持续萎缩 → 流动性脆弱 → 后续波动 → 看跌(取负)

方法：
- 识别日内成交量突增的分钟(>75分位)
- 计算"冲击后"深度 / "冲击前"深度
- LQR = mean(恢复比例) - 1，平均韧性

方向: LQR 越大 → 流动性恢复更快 → 预期正收益

AI应用说明:
源自韧性(resilience)概念，衡量订单簿的自愈能力。
LLM提示: "大单冲击后订单簿多久恢复？"。
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
        
        if len(grp) < 10:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 计算每分钟的总成交量(代理)和总深度
        volumes = []
        total_depths = []
        
        for _, row in grp.iterrows():
            # 成交量代理: 可以用 num_trades 或 total volume change
            vol = row.get('volume', row.get('num_trades', 0)) or 0
            bv = sum(row.get(f'bid_volume{i}', 0) or 0 for i in range(1, 11))
            av = sum(row.get(f'ask_volume{i}', 0) or 0 for i in range(1, 11))
            
            volumes.append(vol)
            total_depths.append(bv + av)
        
        volumes = np.array(volumes, dtype=float)
        total_depths = np.array(total_depths, dtype=float)
        
        # 识别大成交量分钟(>75分位)
        if len(volumes) < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        threshold = np.percentile(volumes, 75)
        shock_idx = np.where((volumes > threshold) & (volumes > 0))[0]
        
        recovery_ratios = []
        for idx in shock_idx:
            if idx < 1 or idx >= len(total_depths) - 1:
                continue
            
            depth_before = total_depths[idx - 1]
            depth_shock = total_depths[idx]
            # 下一分钟深度(如有)
            if idx + 1 < len(total_depths):
                depth_after = total_depths[idx + 1]
            else:
                depth_after = depth_shock
            
            if depth_before < 1:
                continue
            
            # 恢复比例 = 冲击后深度 / 冲击前深度
            recovery = depth_after / depth_before
            recovery_ratios.append(recovery)
        
        if not recovery_ratios:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        # 平均恢复率，>1表示深度恢复甚至增加
        lqr = np.mean(recovery_ratios) - 1.0
        
        results.append({'date': dt, 'instrument': inst, 'factor': lqr})
    
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
