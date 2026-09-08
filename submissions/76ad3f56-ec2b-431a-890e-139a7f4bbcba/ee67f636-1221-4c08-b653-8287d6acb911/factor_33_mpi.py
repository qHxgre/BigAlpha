"""
因子 33: MPI — 边际价格冲击 (Marginal Price Impact)
=====================================================

经济逻辑:
从第1档吃到第N档需要多付的成本曲线斜率反映市场深度。
斜率越陡 → 每多吃一档价格抬升越快 → 流动性脆弱 → 买盘成本高 → 看跌（取负）。
斜率越平 → 深度充裕 → 流动性好 → 成本低 → 看涨。

方法：
- 计算每多扫一档的边际成本: (ask_price_k - ask_price_1) / ask_price_1
- 对边际成本拟合斜率
- MPI = -slope (斜率大=流动性差→取负号)

方向: MPI 越大 → 边际冲击成本越小 → 预期正收益

AI应用说明:
从订单簿直接计算冲击成本，而非间接代理。
LLM提示: "扫单的边际成本曲线"。
"""

import pandas as pd
import numpy as np


def main(data):
    df = data.copy()
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    levels = np.arange(1, 11, dtype=float)
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        mpi_vals = []
        
        for _, row in grp.iterrows():
            ask_prices = np.array([row.get(f'ask_price{i}', np.nan) for i in range(1, 11)], dtype=float)
            bid_prices = np.array([row.get(f'bid_price{i}', np.nan) for i in range(1, 11)], dtype=float)
            
            # 过滤无效价格
            valid_ask = pd.notna(ask_prices) & (ask_prices > 0)
            valid_bid = pd.notna(bid_prices) & (bid_prices > 0)
            
            if valid_ask.sum() < 3 or valid_bid.sum() < 3:
                continue
            
            ap1 = ask_prices[0]
            bp1 = bid_prices[0]
            
            if ap1 <= bp1:
                continue
            
            # 边际价格冲击: 每档价格偏离第1档的程度
            ask_impact = (ask_prices[valid_ask] - ap1) / ap1
            bid_impact = (bp1 - bid_prices[valid_bid]) / bp1
            
            # 对冲击曲线拟合斜率(使用实际档位)
            ask_lvls = levels[valid_ask][:len(ask_impact)] - 1
            bid_lvls = levels[valid_bid][:len(bid_impact)] - 1
            
            ask_slope = np.polyfit(ask_lvls, ask_impact, 1)[0] if len(ask_lvls) >= 3 else 0
            bid_slope = np.polyfit(bid_lvls, bid_impact, 1)[0] if len(bid_lvls) >= 3 else 0
            
            # 斜率小 = 冲击成本低 = 好; 取负使方向正确
            # bid斜率大=买入便宜(bid价格下跌快), ask斜率大=买入贵
            # 实际: bid斜率大对买方有利; ask斜率小对买方有利
            # MPI = -ask_slope + bid_slope → 买盘冲击小=正
            mpi_vals.append(-ask_slope + bid_slope)
        
        factor = np.mean(mpi_vals) if mpi_vals else 0.0
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
