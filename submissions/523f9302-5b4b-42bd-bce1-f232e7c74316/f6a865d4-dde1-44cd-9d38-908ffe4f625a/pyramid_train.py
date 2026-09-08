"""Temporal Pyramid LSTM definitions and BigQuant cloud data pipeline."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "pyramid_model.json"
SEQ_LEN, BATCH, SEED = 240, 256, 2026
BAR_COLS = ["open", "high", "low", "close", "deal_number", "volume", "amount"]
BOOK_COLS = [f"{side}_{kind}{level}" for kind in ("price", "volume", "num_orders") for side in ("bid", "ask") for level in (1, 2, 3)]
FEATURE_COLS = BAR_COLS + BOOK_COLS
PRICE_COLS = ["open", "high", "low", "close"] + [f"{side}_price{level}" for side in ("bid", "ask") for level in (1, 2, 3)]
LOG_COLS = [c for c in FEATURE_COLS if c not in PRICE_COLS]


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dim,dim//2),nn.Tanh(),nn.Linear(dim//2,1))
    def forward(self,x): return (x*self.net(x).softmax(1)).sum(1)


class ResidualHead(nn.Module):
    def __init__(self,dim):
        super().__init__();self.skip=nn.Linear(dim,1);self.body=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*2),nn.GLU(),nn.GELU(),nn.Dropout(.15),nn.Linear(dim,1))
    def forward(self,x): return (self.skip(x)+self.body(x)).squeeze(-1)


class TemporalPyramid(nn.Module):
    """Learned 1m/5m/15m temporal branches over the raw 1m sequence."""
    def __init__(self, input_dim=len(FEATURE_COLS)):
        super().__init__()
        if input_dim != len(FEATURE_COLS):
            raise ValueError(f"expected {len(FEATURE_COLS)} fields, got {input_dim}")
        self.p = nn.Linear(input_dim, 96)
        self.r1 = nn.LSTM(96, 128, 1, batch_first=True)
        self.c5 = nn.Conv1d(96, 96, 7, 5, 3)
        self.r5 = nn.GRU(96, 96, 1, batch_first=True)
        self.c15 = nn.Conv1d(96, 64, 15, 15)
        self.r15 = nn.GRU(64, 64, 1, batch_first=True)
        self.head = ResidualHead(288)

    def forward(self, x):
        latent = self.p(x)
        one = self.r1(latent)[0][:, -1]
        five = self.r5(self.c5(latent.transpose(1, 2)).transpose(1, 2))[0][:, -1]
        fifteen = self.r15(self.c15(latent.transpose(1, 2)).transpose(1, 2))[0][:, -1]
        return self.head(torch.cat([one, five, fifteen], -1))


def load_model(model_path=MODEL_PATH,map_location="cpu"):
    payload=json.loads(Path(model_path).read_text(encoding="utf-8"));state={}
    for name,meta in payload["state_dict"].items():
        state[name]=torch.tensor(meta["data"],dtype=getattr(torch,meta["dtype"])).reshape(meta["shape"]).to(map_location)
    payload["state_dict"]=state;return payload


def pool_frame(start_date,end_date):
    import dai
    pool=dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",filters={"date":[str(start_date),str(end_date)]}).df()
    pool["date"]=pd.to_datetime(pool["date"]).dt.normalize();return pool.drop_duplicates(["date","instrument"])


def build_cloud_day(table,day,instruments,stats):
    import dai
    start=pd.Timestamp(day).strftime("%Y-%m-%d 00:00:00");end=pd.Timestamp(day).strftime("%Y-%m-%d 23:59:59")
    sql=f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    df=dai.query(sql,filters={"date":[start,end],"instrument":list(instruments)}).df();df["date"]=pd.to_datetime(df["date"])
    for col in LOG_COLS:df[col]=np.log1p(df[col].clip(lower=0)).astype(np.float32)
    df[FEATURE_COLS]=df.groupby("instrument",sort=False)[FEATURE_COLS].ffill();windows=[];keys=[]
    for instrument,sub in df.groupby("instrument",sort=False):
        if len(sub)<SEQ_LEN:continue
        values=sub[FEATURE_COLS].to_numpy(np.float32);dates=sub["date"].dt.normalize().to_numpy();window=values[-SEQ_LEN:]
        if pd.Timestamp(dates[-SEQ_LEN])!=pd.Timestamp(day) or not np.isfinite(window).all():continue
        windows.append(window);keys.append((pd.Timestamp(day),instrument))
    if not windows:raise RuntimeError(f"No valid 1m samples for {day}")
    x=np.stack(windows).astype(np.float32);mean,std=stats;x=((x-mean)/std).astype(np.float32)
    return x,pd.DataFrame(keys,columns=["date","instrument"])


def parameter_count(): return sum(p.numel() for p in TemporalPyramid().parameters() if p.requires_grad)


def save_model(model,stats,model_path=MODEL_PATH):
    tensors={}
    for name,value in model.state_dict().items():
        tensor=value.detach().cpu();tensors[name]={"dtype":str(tensor.dtype).replace("torch.",""),"shape":list(tensor.shape),"data":tensor.reshape(-1).tolist()}
    payload={"state_dict":tensors,"model_cfg":{"input_dim":len(FEATURE_COLS)},"feature_cols":FEATURE_COLS,"seq_len":SEQ_LEN,"mean":np.asarray(stats[0],np.float32).tolist(),"std":np.asarray(stats[1],np.float32).tolist(),"train_start":"2019-01-01","train_end":"2022-12-31 23:59:59","validation":"2023","epochs":6,"seed":SEED,"parameters":parameter_count()}
    Path(model_path).write_text(json.dumps(payload),encoding="utf-8");return str(model_path)


def _cloud_frame(table,start,end,instruments):
    import dai
    sql=f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    df=dai.query(sql,filters={"date":[str(start),str(end)],"instrument":list(instruments)}).df();df["date"]=pd.to_datetime(df["date"])
    for col in LOG_COLS:df[col]=np.log1p(df[col].clip(lower=0)).astype(np.float32)
    df[FEATURE_COLS]=df.groupby("instrument",sort=False)[FEATURE_COLS].ffill();return df


def _cloud_stats(table,periods):
    count=np.zeros(len(FEATURE_COLS),np.int64);total=np.zeros(len(FEATURE_COLS));square=np.zeros(len(FEATURE_COLS))
    for period in periods:
        start=period.start_time.strftime("%Y-%m-%d");end=period.end_time.strftime("%Y-%m-%d 23:59:59");pool=pool_frame(start,end);df=_cloud_frame(table,start,end,pool.instrument.unique());values=df[FEATURE_COLS].to_numpy(np.float64);finite=np.isfinite(values);count+=finite.sum(0);total+=np.where(finite,values,0).sum(0);square+=np.where(finite,values*values,0).sum(0)
    mean=total/np.maximum(count,1);var=square/np.maximum(count,1)-mean*mean;return mean.astype(np.float32),np.sqrt(np.maximum(var,1e-12)).astype(np.float32)


def _cloud_training_month(table,period,stats):
    start=period.start_time;end=period.end_time;query_end=(end+pd.Timedelta(days=7)).strftime("%Y-%m-%d 23:59:59");pool=pool_frame(start,query_end);df=_cloud_frame(table,start.strftime("%Y-%m-%d"),query_end,pool.instrument.unique());windows=[];targets=[];dates=[]
    for _,sub in df.groupby("instrument",sort=False):
        values=sub[FEATURE_COLS].to_numpy(np.float32);days=sub.date.dt.normalize().to_numpy();ends=np.flatnonzero(np.r_[days[1:]!=days[:-1],True]);closes=sub.close.to_numpy(np.float64)[ends]
        for k,pos in enumerate(ends[:-1]):
            day=pd.Timestamp(days[pos]);
            if day<start or day>end or pos+1<SEQ_LEN or pd.Timestamp(days[pos-SEQ_LEN+1])!=day or closes[k]<=0:continue
            window=values[pos-SEQ_LEN+1:pos+1];target=closes[k+1]/closes[k]-1
            if np.isfinite(window).all() and np.isfinite(target):windows.append(window);targets.append(target);dates.append(day)
    x=(np.stack(windows)-stats[0])/stats[1];return x.astype(np.float32),np.asarray(targets,np.float32),np.asarray(dates)


def _rank_loss(score,target):
    score_z=(score-score.mean())/(score.std(unbiased=False)+1e-6);target_z=(target-target.mean())/(target.std(unbiased=False)+1e-6);ic=1-(score_z*target_z).mean();point=F.smooth_l1_loss(score_z,target_z);n=len(score);pairs=min(2048,max(n*2,1));i=torch.randint(n,(pairs,),device=score.device);j=torch.randint(n,(pairs,),device=score.device);pair=F.softplus(-torch.sign(target[i]-target[j])*(score[i]-score[j])/.5).mean();return .55*ic+.25*pair+.20*point


def train_and_save(datasources,model_path=MODEL_PATH):
    """Reproduce the public model from scratch for isolated private retraining."""
    np.random.seed(SEED);torch.manual_seed(SEED);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");table=datasources["bar1m"];periods=list(pd.period_range("2019-01","2022-12",freq="M"));stats=_cloud_stats(table,periods);model=TemporalPyramid().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,6*len(periods));model.train()
    for epoch in range(6):
        order=np.random.default_rng(SEED+epoch).permutation(len(periods))
        for index in order:
            x,y,dates=_cloud_training_month(table,periods[int(index)],stats);lo,hi=np.percentile(y,[1,99]);y=np.clip(y,lo,hi)
            for day in np.unique(dates):
                selection=np.flatnonzero(dates==day);xb=torch.from_numpy(x[selection]).to(device);yb=torch.from_numpy(y[selection]).to(device);optimizer.zero_grad(set_to_none=True);loss=_rank_loss(model(xb),yb);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            scheduler.step()
    return save_model(model,stats,model_path)
