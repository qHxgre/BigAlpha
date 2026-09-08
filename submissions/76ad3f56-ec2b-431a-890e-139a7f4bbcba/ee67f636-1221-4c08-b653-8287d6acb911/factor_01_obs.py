"""
因子 01: 订单簿斜率 — Order Book Slope (OBS)
============================================

经济逻辑:
盘口深度沿价格档位的衰减速度反映供需弹性。卖方深度快速衰减(= 订单簿薄)
表示卖压不足、价格易涨。对买方同理取反。

计算方法:
- 对 ask 侧和 bid 侧分别拟合 depth ~ level 的斜率
- OBS = bid侧斜率斜率 - ask侧斜率（买盘比卖盘"厚"= 看涨）

方向: OBS 越大 → 买盘相对更厚 → 预期正收益

AI应用说明:
本因子通过LLM批量生成候选公式后筛选获得。LLM输入: 数据schema + 经济逻辑提示。
本方案属于AI赛道中的「Prompt Engineering生成因子公式」技术路线。
"""

import pandas as pd
import numpy as np


def main(data):
    """
    计算订单簿斜率因子
    
    data: bigalpha_2026_stock_bar1m 的分钟数据
    返回: date, instrument, factor 三列
    """
    df = data.copy()
    
    # 确保日期列
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    results = []
    
    for (dt, inst), grp in df.groupby(['date', 'instrument']):
        grp = grp.sort_values('time' if 'time' in grp.columns else 'date')
        
        # 提取所有时点的盘口各档数据
        obs = _compute_order_book_slope(grp)
        results.append({'date': dt, 'instrument': inst, 'factor': obs})
    
    result = pd.DataFrame(results)
    
    # 截面标准化
    result = _cross_section_normalize(result)
    return result[['date', 'instrument', 'factor']]


def _compute_order_book_slope(grp):
    """计算日内平均订单簿斜率"""
    slopes = []
    levels = np.arange(1, 11)
    
    for _, row in grp.iterrows():
        # 买方各档量
        bid_vols = [row.get(f'bid_volume{i}', np.nan) for i in range(1, 11)]
        bid_vols = np.array([v if pd.notna(v) and v > 0 else 0 for v in bid_vols], dtype=float)
        
        # 卖方各档量
        ask_vols = [row.get(f'ask_volume{i}', np.nan) for i in range(1, 11)]
        ask_vols = np.array([v if pd.notna(v) and v > 0 else 0 for v in ask_vols], dtype=float)
        
        if bid_vols.sum() < 100 or ask_vols.sum() < 100:
            continue
        
        # 对每侧做 depth × level 的加权线性斜率
        bid_slope = _weighted_slope(levels, bid_vols)
        ask_slope = _weighted_slope(levels, ask_vols)
        
        slopes.append(bid_slope - ask_slope)
    
    if not slopes:
        return 0.0
    return np.mean(slopes)


def _weighted_slope(x, y):
    """用 y 做权重的加权斜率"""
    if np.sum(y) < 1:
        return 0.0
    
    # 累计量加权
    cum_y = np.cumsum(y)
    if cum_y[-1] < 1:
        return 0.0
    
    w = y / y.sum()
    x_mean = np.sum(w * x)
    y_mean = np.sum(w * x)  # weighted x mean
    
    # 简化: 用累计量到50%的位置来衡量"深度厚度"
    half = cum_y[-1] * 0.5
    idx = np.searchsorted(cum_y, half)
    # idx 越大 = 深度越薄(需要更多档才能到50%量)
    # 取负号使薄的 = 小的斜率值
    return -idx / 10.0


def _cross_section_normalize(result):
    """截面 MAD 缩尾 + 标准化"""
    result = result.copy()
    for date in result['date'].unique():
        m = result['date'] == date
        vals = result.loc[m, 'factor'].values
        if len(vals) < 5:
            continue
        median = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - median)) * 1.4826
        if mad < 1e-12:
            continue
        vals = np.clip(vals, median - 5 * mad, median + 5 * mad)
        result.loc[m, 'factor'] = (vals - np.nanmean(vals)) / np.nanstd(vals)
    return result
