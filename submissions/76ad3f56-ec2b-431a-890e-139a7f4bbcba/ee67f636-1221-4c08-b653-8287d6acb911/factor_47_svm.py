"""
因子 47: SVM — 偏度×成交量 (Skewness × Volume)
================================================

经济逻辑:
日内价格分布的偏度 × log(volume) 捕捉极端收益的可信度。
正偏度 + 高成交量 → 买方推动的价格上行偏态可信 → 看涨。
负偏度 + 高成交量 → 卖方推动的价格下行偏态可信 → 看跌。
偏度绝对值大但成交量低 → 噪声、不可信。

方法：
- 计算日内mid_price收益率的偏度
- 计算成交量代理的对数
- SVM = skewness × log(volume)

方向: SVM 越大 → 正偏度+有量 → 预期正收益

AI应用说明:
偏度反映分布不对称性，成交量验证其可信度。
LLM提示: "收益分布的形状+成交验证"。
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
        
        returns_list = []
        volumes_list = []
        
        prev_mid = None
        for _, row in grp.iterrows():
            ap1 = row.get('ask_price1', np.nan)
            bp1 = row.get('bid_price1', np.nan)
            
            vol = row.get('volume', row.get('num_trades', 0)) or 0
            volumes_list.append(vol)
            
            if pd.notna(ap1) and pd.notna(bp1) and ap1 > bp1 > 0:
                mid = (ap1 + bp1) / 2
                if prev_mid is not None and prev_mid > 0:
                    ret = np.log(mid / prev_mid)
                    returns_list.append(ret)
                prev_mid = mid
            else:
                prev_mid = None
        
        if len(returns_list) < 5:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        returns = np.array(returns_list)
        volumes = np.array(volumes_list)
        
        # 偏度
        ret_mean = np.mean(returns)
        ret_std = np.std(returns)
        if ret_std < 1e-12:
            results.append({'date': dt, 'instrument': inst, 'factor': 0.0})
            continue
        
        skew = np.mean((returns - ret_mean) ** 3) / (ret_std ** 3 + 1e-12)
        
        # 总成交量
        total_vol = np.sum(volumes) if len(volumes) > 0 else 1
        avg_vol = np.mean(volumes) if len(volumes) > 0 else 1
        
        # SVM: 偏度 × log(volume)
        # 正偏度+大成交量 = 买盘推高价格 → 看涨
        # 负偏度+大成交量 = 卖盘压低价格 → 看跌
        svm = skew * np.log1p(avg_vol)
        
        results.append({'date': dt, 'instrument': inst, 'factor': svm})
    
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
