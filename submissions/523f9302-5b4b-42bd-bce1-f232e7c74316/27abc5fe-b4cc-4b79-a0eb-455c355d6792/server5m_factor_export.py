"""四模型 × 两优化器 × 三采样方式，串行训练并导出 48 个因子。"""
from __future__ import annotations
import gc, json, logging, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from server5m_data import cross_sectional_rank, load_manifest
from server5m_model import (MultiTaskAttentionLOB, TCNFusionMultiTaskLOB,
                    SmallMultiTaskAttentionLOB, SmallTCNFusionMultiTaskLOB)
from server5m_train import InferenceDataset
from server5m_train_multitask import _daily_zscore, train_multitask

ROOT=Path("training_results/grid24")
FACTOR_ROOT=ROOT/"factors"
STATE=ROOT/"state.json"

MODELS={
 "lob1m_tcn": dict(cache="data_sample/local_1min_5level_relative_cs_cache",
   model_type="fusion_tcn",feature_dim=128,head_dim=64,tcn_channels=64,
   tcn_bins=1,dropout=.08,batch_size=256,lr=.0005,
   regression_weights=[0.,0.,0.],classification_weights=[1.,.3,.1]),
 "lob5m_attn": dict(cache="data_sample/local_5level_relative_cs_cache",
   model_type="attention",feature_dim=64,head_dim=32,dropout=.10,
   batch_size=1024,lr=.001,regression_weights=[.25,.15,.10],
   classification_weights=[1.,.5,.25]),
 "lob1m_tcn_small": dict(cache="data_sample/local_1min_5level_relative_cs_cache",
   model_type="fusion_tcn_small",feature_dim=64,head_dim=32,tcn_channels=32,
   tcn_bins=1,dropout=.15,batch_size=256,lr=.0005,
   regression_weights=[0.,0.,0.],classification_weights=[1.,.3,.1]),
 "lob5m_attn_small": dict(cache="data_sample/local_5level_relative_cs_cache",
   model_type="attention_small",feature_dim=32,head_dim=16,dropout=.15,
   batch_size=1024,lr=.001,regression_weights=[.25,.15,.10],
   classification_weights=[1.,.5,.25]),
}
OPTIMIZERS={
 "adamw":dict(optimizer="adamw"),
 "pvr025":dict(optimizer="pvr_adamw",optimizer_params=dict(
   sketch_rank=8,oja_lr=.01,qr_freq=50,project_ratio=.25)),
}
SAMPLERS={"default":{},"ds240":{"drift_half_life_days":240},
          "ds120":{"drift_half_life_days":120}}

def config(name, model, optimizer, sampler):
    cfg=dict(run_name=name,output_root=str(ROOT/"runs"),
      data={"cache_dir":model["cache"]},epochs=50,weight_decay=.0001,
      ordinal_sigma=.7,rank_loss_weight=0.,selection_score="cls_blend_532",
      early_stop_metric="loss",grad_clip=1.,early_stop_patience=10,
      num_workers=4,amp=True,seed=42)
    cfg.update({k:v for k,v in model.items() if k!="cache"})
    cfg.update(optimizer); cfg.update(sampler); return cfg

def _model_from_checkpoint(checkpoint):
    cls={"attention":MultiTaskAttentionLOB,"fusion_tcn":TCNFusionMultiTaskLOB,
         "attention_small":SmallMultiTaskAttentionLOB,
         "fusion_tcn_small":SmallTCNFusionMultiTaskLOB}[
        checkpoint["model_type"]]
    model=cls(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state"]); return model

@torch.no_grad()
def scores(model, X, batch_size, workers, device):
    dl=DataLoader(InferenceDataset(X),batch_size=batch_size,shuffle=False,
      num_workers=workers,pin_memory=device.type=="cuda",
      persistent_workers=workers>0)
    regression=[]; expected=[]; model.eval()
    classes=torch.arange(5,device=device,dtype=torch.float32)
    for x in dl:
      x=x.to(device,non_blocking=True)
      with torch.autocast(device_type=device.type,dtype=torch.float16,
                          enabled=device.type=="cuda"):
        out=model(x)
      regression.append(out["regression"].float().cpu().numpy())
      expected.append((torch.softmax(out["classification"].float(),dim=2)
                       @classes).cpu().numpy())
    return np.column_stack((np.concatenate(regression),
                            np.concatenate(expected))).astype(np.float32)

def export_pair(result_dir, cfg, experiment, years=(2024,)):
    factor_root = Path(result_dir) / "factors"
    cache=Path(cfg["data"]["cache_dir"]); device=torch.device(
      "cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(Path(result_dir)/"best_model.pt",map_location="cpu",
                    weights_only=False)
    model=_model_from_checkpoint(ckpt).to(device)
    raw={}; dates={}; stocks={}
    for split in ("train","val","test"):
      dates[split]=np.asarray(np.load(cache/f"{split}_date_ids.npy",mmap_mode="r"))
      stocks[split]=np.asarray(np.load(cache/f"{split}_stocks.npy",mmap_mode="r"))
      raw[split]=scores(model,np.load(cache/f"{split}_X.npy",mmap_mode="r"),
                        int(cfg["batch_size"]),int(cfg.get("num_workers",4)),device)
    cls_norm={s:_daily_zscore(raw[s][:,3:6],dates[s]) for s in raw}
    fixed={s:cls_norm[s]@np.array([.5,.3,.2]) for s in raw}
    ret=np.load(cache/"train_ret_1.npy",mmap_mode="r")
    target=cross_sectional_rank(ret,dates["train"])
    valid=np.isfinite(target)&np.isfinite(raw["train"]).all(axis=1)
    scaler=StandardScaler(); tx=scaler.fit_transform(raw["train"][valid])
    ridge=Ridge(alpha=1000.0).fit(tx,target[valid])
    ridge_score={s:ridge.predict(scaler.transform(raw[s])).astype(np.float32)
                 for s in raw}
    manifest=load_manifest(cache); frames={}
    for method,pred in (("fixed532",fixed),("ridge6",ridge_score)):
      pieces=[]
      selected_splits=("train","val","test") if any(y<2024 for y in years) else ("val","test")
      for split in selected_splits:
        pieces.append(pd.DataFrame({
          "date":pd.to_datetime([manifest["dates"][int(x)] for x in dates[split]]),
          "instrument":stocks[split].astype(str),"score":pred[split].astype(np.float32)}))
      frame=pd.concat(pieces,ignore_index=True)
      paths={}
      for year in years:
        annual=frame.loc[frame.date.dt.year==year].reset_index(drop=True)
        suffix="" if tuple(years)==(2024,) else f"__{year}"
        path=factor_root/f"{experiment}__{method}{suffix}.parquet"
        path.parent.mkdir(parents=True,exist_ok=True); annual.to_parquet(path,index=False)
        paths[str(year)]={"path":str(path.resolve()),"rows":len(annual),
                          "days":int(annual.date.nunique())}
      frames[method]={"years":paths,
        "ridge_alpha":1000. if method=="ridge6" else None}
    (Path(result_dir)/"factor_exports.json").write_text(
      json.dumps(frames,ensure_ascii=False,indent=2),encoding="utf8")
    del model,raw; gc.collect()
