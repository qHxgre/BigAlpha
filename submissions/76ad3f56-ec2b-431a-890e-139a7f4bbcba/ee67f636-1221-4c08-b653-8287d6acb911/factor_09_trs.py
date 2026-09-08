"""
因子 09: TRS — 尾盘反转信号 (Tail Reversal Signal)
==================================================

经济逻辑:
最后30分钟的价格行为包含隔夜信息预期。
尾盘急跌+放量 = 恐慌性抛售 → 次日开盘反弹概率高
尾盘急涨+缩量 = 操纵性拉升 → 次日回落概率高
用尾盘收益率的极值条件化来捕捉。

方向: 尾盘超跌 → 次日反弹 → 正因子

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
        
        if len(grp) < 30:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        n = len(grp)
        tail_n = min(30, n // 6)
        
        # 尾盘收益率
        tail_prices = grp['price'].iloc[-tail_n:].values if 'price' in grp.columns else grp['close'].iloc[-tail_n:].values
        tail_ret = (tail_prices[-1] - tail_prices[0]) / (tail_prices[0] + 1e-8)
        
        # 尾盘成交量比例
        if 'volume' in grp.columns:
            tail_vol = grp['volume'].iloc[-tail_n:].mean()
            full_vol = grp['volume'].mean()
            vol_ratio = tail_vol / (full_vol + 1e-8)
        else:
            vol_ratio = 1.0
        
        # 全天波动率
        all_prices = grp['price'].values if 'price' in grp.columns else grp['close'].values
        all_ret = np.diff(all_prices) / (all_prices[:-1] + 1e-8)
        daily_vol = np.std(all_ret) if len(all_ret) > 0 else 0
        
        # 尾盘是否"极端": 收益率偏离全天均值超过2个标准差
        if daily_vol > 1e-8 and len(all_ret) > 0:
            z_score = tail_ret / (daily_vol * np.sqrt(tail_n) + 1e-8)
        else:
            z_score = 0
        
        # 策略: 尾盘超跌(负z) + 放量(恐慌)→ 次日反弹 → 正因子
        # 尾盘超涨(正z) + 缩量(操纵)→ 次日回调 → 负因子
        is_oversold = z_score < -1.5
        is_overbought = z_score > 1.5
        is_high_vol = vol_ratio > 1.5
        is_low_vol = vol_ratio < 0.5
        
        if is_oversold and is_high_vol:
            factor = abs(z_score)  # 超跌越深，反弹越强
        elif is_overbought and is_low_vol:
            factor = -abs(z_score)  # 操纵拉升，看跌
        else:
            # 一般情况：尾盘动量延续
            factor = z_score * vol_ratio
        
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
