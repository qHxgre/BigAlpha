import numpy as np
import pandas as pd
import dai

def _load(s,a,b,x,y):
 q=f"SELECT date,instrument,time,price,bid_price1,ask_price1,bid_volume1,ask_volume1 FROM {s} WHERE time BETWEEN {x} AND {y}"
 d=dai.query(q,filters={'date':[str(a),str(b)]}).df();d['date']=pd.to_datetime(d.date).dt.normalize();return d.sort_values(['date','instrument','time'])
def main_open_reversal(s,a,b):
 d=_load(s,a,b,93100,100000);k=['date','instrument'];x=d.groupby(k,as_index=False).agg(f=('price','first'),l=('price','last'));x['factor']=-np.log(x.l/x.f);return x[['date','instrument','factor']]
def main_close_vwap(s,a,b):
 d=_load(s,a,b,143000,145700);k=['date','instrument'];x=d.groupby(k,as_index=False).agg(f=('price','first'),l=('price','last'));x['factor']=np.log(x.l/x.f);return x[['date','instrument','factor']]
def main_spread_convergence(s,a,b):
 d=_load(s,a,b,130000,140000);k=['date','instrument'];d['spread']=(d.ask_price1-d.bid_price1)/((d.ask_price1+d.bid_price1)/2);x=d.groupby(k,as_index=False).agg(f=('spread','first'),l=('spread','last'),i=('bid_volume1','mean'));x['factor']=np.log(x.f/x.l)*np.sign(x.i);return x[['date','instrument','factor']].replace([np.inf,-np.inf],np.nan).dropna()
