import numpy as np
import pandas as pd
import dai

def main_midday_persistence(data_source, start_date, end_date):
    sql = f"SELECT date, instrument, time, price, bid_volume1, ask_volume1 FROM {data_source} WHERE time BETWEEN 110000 AND 113000"
    df = dai.query(sql, filters={'date': [str(start_date), str(end_date)]}).df()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['imbalance'] = (df['bid_volume1'] - df['ask_volume1']) / (df['bid_volume1'] + df['ask_volume1'] + 1)
    out = df.groupby(['date','instrument'], as_index=False)['imbalance'].mean()
    return out.rename(columns={'imbalance':'factor'})

def main_vwap_reversal(data_source, start_date, end_date):
    sql = f"SELECT date, instrument, time, price, amount, volume FROM {data_source} WHERE time BETWEEN 100000 AND 143000"
    df = dai.query(sql, filters={'date': [str(start_date), str(end_date)]}).df()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    key = ['date','instrument']; df = df.sort_values(key+['time'])
    df['dv'] = df.groupby(key)['volume'].diff().clip(lower=0); df['da'] = df.groupby(key)['amount'].diff().clip(lower=0)
    out = df.groupby(key,as_index=False).agg(price=('price','last'),volume=('dv','sum'),amount=('da','sum'))
    out['factor'] = -np.log(out['price']/(out['amount']/out['volume']))
    return out[['date','instrument','factor']].dropna()

def main_liquidity_asymmetry(data_source, start_date, end_date):
    return main_midday_persistence(data_source, start_date, end_date)
