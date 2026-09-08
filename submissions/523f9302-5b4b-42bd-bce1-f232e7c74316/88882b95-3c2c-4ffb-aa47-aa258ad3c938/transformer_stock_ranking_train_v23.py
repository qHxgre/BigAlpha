"""
transformer_stock_ranking_train_v23.py
V23 = V21 + 更多股票 (MAX_TRAIN_INST 500→800).
其余完全一致: 全数据/swiglu+rope+conv/log1p/loss/EMA.
"""

import json, time, gc, base64
import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
import structlog

logger = structlog.get_logger()

TRAIN_TABLE = "bigalpha_2026_stock_bar1m"
VAL_TABLE   = "bigalpha_2026_stock_bar1m"
TRAIN_START, TRAIN_END = "2019-01-01", "2022-12-31 23:59:59"
VAL_START,   VAL_END   = "2023-01-01", "2023-12-31 23:59:59"

SEQ_LEN = 240; EPOCHS = 6; BATCH = 512; LR = 3e-4; SEED = 42
PATIENCE = 3
MAX_TRAIN_INST = 800  # ← 500→800
MAX_VAL_SAMPLES = 150000; MAX_STATS_SAMPLES = 150000
DAI_CHUNK = 100

np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("设备", device=str(device))

RAW_FEATURES = ["open", "high", "low", "close",
                "bid_price1", "ask_price1",
                "volume", "amount", "bid_volume1", "ask_volume1"]
VOL_COLS = ["volume", "amount", "bid_volume1", "ask_volume1"]
N_FEAT = len(RAW_FEATURES)

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x, offset=0):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device) + offset
        freqs = torch.einsum("i,j->ij", t.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb).unsqueeze(0), torch.sin(emb).unsqueeze(0)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, dim_ff, dropout=0.15):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_ff, bias=False)
        self.w2 = nn.Linear(d_model, dim_ff, bias=False)
        self.w3 = nn.Linear(dim_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0: return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device); random_tensor.floor_()
        return x / keep_prob * random_tensor

class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.15, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SwiGLUFFN(d_model, dim_ff, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.rope = RotaryPositionalEncoding(d_model // nhead)
    def forward(self, x, attn_mask=None):
        residual = x; x_norm = self.norm1(x); q = k = v = x_norm
        cos, sin = self.rope(x_norm)
        B,L,D = q.shape; H = self.attn.num_heads; Dh = D//H
        q_r=q.view(B,L,H,Dh).transpose(1,2); k_r=k.view(B,L,H,Dh).transpose(1,2); v_r=v.view(B,L,H,Dh).transpose(1,2)
        cos_r=cos.unsqueeze(1); sin_r=sin.unsqueeze(1); q_r,k_r=apply_rotary_pos_emb(q_r,k_r,cos_r,sin_r)
        q_h=q_r.transpose(1,2).reshape(B,L,D); k_h=k_r.transpose(1,2).reshape(B,L,D); v_h=v_r.transpose(1,2).reshape(B,L,D)
        attn_out,_ = self.attn(q_h,k_h,v_h,need_weights=False,attn_mask=attn_mask)
        x = residual + self.drop_path(attn_out); residual = x
        x = residual + self.drop_path(self.ffn(self.norm2(x))); return x

class ConvTransformer(nn.Module):
    def __init__(self, n_feat=N_FEAT, d_model=128, nhead=4, nlayers=4,
                 dim_ff=384, dropout=0.15, drop_path_rate=0.05):
        super().__init__()
        d1=42;d2=42;d3=44
        self.conv_s=nn.Sequential(nn.Conv1d(n_feat,d1,3,2,1),nn.GELU())
        self.conv_m=nn.Sequential(nn.Conv1d(n_feat,d2,5,2,2),nn.GELU())
        self.conv_l=nn.Sequential(nn.Conv1d(n_feat,d3,7,2,3),nn.GELU())
        self.conv_proj=nn.Sequential(nn.Conv1d(d_model,d_model,3,2,1),nn.GELU()); self.seq_out=60
        drop_paths=[drop_path_rate*(i/max(1,nlayers-1)) for i in range(nlayers)]
        self.layers=nn.ModuleList([RoPETransformerLayer(d_model,nhead,dim_ff,dropout,drop_paths[i]) for i in range(nlayers)])
        self.attn_pool=nn.Linear(d_model,1)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,96),nn.GELU(),nn.Dropout(dropout),nn.Linear(96,1))
    def forward(self, x):
        hs=self.conv_s(x.permute(0,2,1)); hm=self.conv_m(x.permute(0,2,1)); hl=self.conv_l(x.permute(0,2,1))
        h=torch.cat([hs,hm,hl],dim=1); h=self.conv_proj(h).permute(0,2,1)
        for layer in self.layers: h=layer(h)
        w=F.softmax(self.attn_pool(h),dim=1); hp=(h*w).sum(dim=1); return self.head(hp).squeeze(-1)

@torch.no_grad()
def ema_update(ema_model, model, decay=0.9995):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        if p.requires_grad: ema_p.mul_(decay).add_(p, alpha=1.0 - decay)

@torch.no_grad()
def predict_batched(model, X_np, bs=256):
    model.eval(); preds=[]; X_t=torch.from_numpy(X_np)
    for i in range(0, len(X_np), bs): preds.append(model(X_t[i:i+bs].to(device)).cpu().numpy())
    return np.concatenate(preds)

def pearson_loss(pred, target):
    pm=pred-pred.mean(); tm=target-target.mean()
    return 1.0-(pm*tm).sum()/(torch.sqrt((pm**2).sum()*(tm**2).sum())+1e-8)

def pairwise_margin_loss(pred, target, margin=0.01, n_pairs=2048):
    B=len(pred)
    if B<2: return torch.tensor(0.0, device=pred.device)
    n=min(n_pairs, B*(B-1)//2)
    i=torch.randint(0,B,(n,),device=pred.device); j=torch.randint(0,B,(n,),device=pred.device)
    mask=(target[i]-target[j]).abs()>1e-6; i,j=i[mask],j[mask]
    if len(i)==0: return torch.tensor(0.0, device=pred.device)
    sign=torch.sign(target[i]-target[j]); diff=sign*(pred[i]-pred[j]); return F.relu(margin-diff).mean()

def rank_ic_np(pred, target):
    c,_ = spearmanr(pred, target); return c if np.isfinite(c) else 0.0

def save_checkpoint(state_dict, stats, path, model_cfg):
    keys=list(state_dict.keys()); shapes=[list(state_dict[k].shape) for k in keys]
    flat=np.concatenate([state_dict[k].cpu().numpy().ravel() for k in keys]).astype(np.float16)
    payload={"keys":keys,"shapes":shapes,"flat":base64.b64encode(flat.tobytes()).decode("ascii"),
             "dtype":str(flat.dtype),"model_cfg":model_cfg,
             "mean":base64.b64encode(stats[0].tobytes()).decode("ascii"),
             "std":base64.b64encode(stats[1].tobytes()).decode("ascii"),
             "stats_dtype":str(stats[0].dtype)}
    with open(path,"w") as f: json.dump(payload,f)
    logger.info(f"已保存: {path} ({round(len(json.dumps(payload))/1e6,1)} MB)")

def build_features(df):
    feats=np.column_stack([df[c].to_numpy(np.float32) for c in RAW_FEATURES])
    feats=np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    for ci, c in enumerate(RAW_FEATURES):
        if c in VOL_COLS: feats[:, ci] = np.log1p(feats[:, ci].clip(min=0))
    return feats.astype(np.float32)

def build_dataset(sd, ed, instruments, table, stats=None, need_label=False, chunk_size=DAI_CHUNK):
    t0=time.time()
    buf=(pd.to_datetime(sd)-pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    sd_ts,ed_ts=pd.to_datetime(sd),pd.to_datetime(ed)
    wins,ys,keys=[],[],[]
    n_batches=(len(instruments)+chunk_size-1)//chunk_size
    for ci in range(0,len(instruments),chunk_size):
        batch_inst=instruments[ci:ci+chunk_size]
        sql=f"SELECT date, instrument, {', '.join(RAW_FEATURES)} FROM {table} ORDER BY instrument, date"
        df=dai.query(sql, filters={"date":[buf,ed],"instrument":batch_inst}).df()
        df=df[df["date"].notna()]
        logger.info(f"  分批读取 {ci//chunk_size+1}/{n_batches} inst={len(batch_inst)} rows={len(df)}")
        for ins,sub in df.groupby("instrument",sort=False):
            if len(sub)<=SEQ_LEN: continue
            feats=build_features(sub)
            day=sub["date"].dt.normalize().to_numpy()
            close_pos=np.flatnonzero(np.append(day[1:]!=day[:-1],True))
            close_px=sub["close"].to_numpy(np.float64)[close_pos]; dates=day[close_pos]
            for k,p in enumerate(close_pos):
                if k+1>=len(close_pos): continue
                d=pd.Timestamp(dates[k])
                if d<sd_ts or d>ed_ts: continue
                if p<SEQ_LEN: continue
                r=close_px[k+1]/close_px[k]-1.0
                if need_label and not np.isfinite(r): continue
                wins.append(feats[p-SEQ_LEN+1:p+1].copy()); ys.append(np.float32(r)); keys.append((d,ins))
        del df; gc.collect()
    if not keys: raise RuntimeError(f"无样本 {sd}~{ed} table={table}")
    X=np.stack(wins).astype(np.float32); y_arr=np.array(ys,np.float32)
    del wins,ys; gc.collect()
    if stats is None:
        n_stat=min(len(X),MAX_STATS_SAMPLES)
        if len(X)>n_stat:
            idx_s=np.random.RandomState(SEED).choice(len(X),n_stat,replace=False)
            flat=X[idx_s].reshape(-1,N_FEAT)
        else: flat=X.reshape(-1,N_FEAT)
        stats=(flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32)+1e-6)
    m,s=stats; X=(X-m[np.newaxis,np.newaxis,:])/s[np.newaxis,np.newaxis,:]
    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32,copy=False)
    if need_label and len(keys)>0:
        tmp=pd.DataFrame({"date":[k[0] for k in keys],"_y":y_arr})
        tmp["_y_cs"]=tmp.groupby("date")["_y"].transform(lambda g:(g-g.mean())/(g.std()+1e-8))
        y_arr=tmp["_y_cs"].to_numpy(np.float32)
    idx_df=pd.DataFrame(keys,columns=["date","instrument"])
    logger.info(f"数据 {sd[:7]}~{ed[:7]} n={len(keys)} t={round(time.time()-t0,1)}s")
    return X,y_arr,idx_df,stats

def pool(sd,ed):
    return dai.query("SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
                     filters={"date":[sd,ed]}).df()["instrument"].tolist()

def train_main():
    logger.info("训练集",start=TRAIN_START,end=TRAIN_END,
                inst=min(MAX_TRAIN_INST,len(pool(TRAIN_START,TRAIN_END))))
    tr_inst=pool(TRAIN_START,TRAIN_END)[:MAX_TRAIN_INST]
    Xtr,ytr,_,stats=build_dataset(TRAIN_START,TRAIN_END,tr_inst,table=TRAIN_TABLE,need_label=True)
    ytr=np.nan_to_num(ytr,nan=0.0,posinf=0.0,neginf=0.0)
    lo,hi=np.percentile(ytr,[1,99]); ytr=ytr.clip(lo,hi)
    logger.info("训练样本",n=len(Xtr),inst=len(tr_inst)); gc.collect()
    logger.info("验证集",start=VAL_START,end=VAL_END)
    v_inst=sorted(set(pool(VAL_START,VAL_END))&set(tr_inst))[:MAX_TRAIN_INST]
    Xval,yval,_,_=build_dataset(VAL_START,VAL_END,v_inst,table=VAL_TABLE,stats=stats,need_label=True)
    yval=np.nan_to_num(yval,nan=0.0,posinf=0.0,neginf=0.0); yval=yval.clip(lo,hi)
    if len(Xval)>MAX_VAL_SAMPLES:
        idx=np.random.RandomState(SEED).choice(len(Xval),MAX_VAL_SAMPLES,replace=False)
        Xval,yval=Xval[idx],yval[idx]
    logger.info("验证样本",n=len(Xval))
    model=ConvTransformer().to(device); ema_model=ConvTransformer().to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters(): p.requires_grad=False
    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("参数量",n=n_params)
    model_cfg={"d_model":128,"nhead":4,"nlayers":4,"dim_ff":384,"n_feat":N_FEAT}
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),
                      batch_size=BATCH,shuffle=True,pin_memory=(device.type=="cuda"),drop_last=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=5e-4)
    T_0=2; sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=T_0*len(loader),T_mult=2,eta_min=LR*0.01)
    scaler=torch.amp.GradScaler("cuda") if device.type=="cuda" else None
    best_ic,best_st,wait=-1.0,None,0
    for ep in range(EPOCHS):
        t0=time.time(); tot_loss,tot_mse,tot_pr,tot_pair,nb=0.0,0.0,0.0,0.0,0; model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    pred=model(xb); loss_mse=F.mse_loss(pred,yb); loss_pr=pearson_loss(pred,yb)
                    loss_pair=pairwise_margin_loss(pred,yb); loss=loss_mse+0.3*loss_pr+0.5*loss_pair
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            else:
                pred=model(xb); loss_mse=F.mse_loss(pred,yb); loss_pr=pearson_loss(pred,yb)
                loss_pair=pairwise_margin_loss(pred,yb); loss=loss_mse+0.3*loss_pr+0.5*loss_pair
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            sched.step()
            if torch.isnan(loss) or torch.isnan(pred).any(): raise RuntimeError("NaN detected")
            ema_update(ema_model,model,decay=0.9995)
            tot_loss+=loss.item(); tot_mse+=loss_mse.item(); tot_pr+=loss_pr.item(); tot_pair+=loss_pair.item(); nb+=1
        ema_model.eval(); val_pred=predict_batched(ema_model,Xval); val_ic=rank_ic_np(val_pred,yval)
        logger.info(f"E{ep+1:2d}/{EPOCHS} loss={tot_loss/max(nb,1):.4f} mse={tot_mse/max(nb,1):.4f} pr={tot_pr/max(nb,1):.4f} pair={tot_pair/max(nb,1):.4f} val_ic={val_ic:.6f} t={time.time()-t0:.0f}s")
        save_checkpoint({k:v.cpu().clone() for k,v in ema_model.state_dict().items()},stats,f"transformer_stock_ranking_v23_ep{ep+1:02d}.json",model_cfg)
        if val_ic>best_ic: best_ic=val_ic; best_st_epoch=ep+1; best_st={k:v.cpu().clone() for k,v in ema_model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=PATIENCE: logger.info(f"早停@{ep+1} best_ic={best_ic:.6f}"); break
    if best_st is None: raise RuntimeError("训练失败")
    ema_model.load_state_dict(best_st)
    logger.info(f"best val_ic={best_ic:.6f} @ epoch {best_st_epoch}")
    save_checkpoint(best_st,stats,"transformer_stock_ranking_v23.json",model_cfg)

if __name__=="__main__": train_main()
