import os, glob, json, time, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略 pyarrow read_feather 弃用提示

# ========== 改这几个常量即可 ==========
LOCAL_DATA_ROOT = r"D:\下载\bigalpha_2026_e2e_bar30m"             # 解压后的根目录
LOCAL_TABLE     = "bigalpha_2026_e2e_bar30m"       # 频率（目录名）
TRAIN_START, TRAIN_END = "2020-01-01", "2022-12-31 23:59:59"
SEQ_LEN, EPOCHS, BATCH = 40, 20, 512                # 30分钟每天8根bar，回看40根≈5天
MAX_INSTRUMENTS = 400                              # demo 限量控时长；全量设 None
# =====================================

MODEL_PATH   = os.path.join(LOCAL_DATA_ROOT, "transformer_model_30min3.json")
PRICE_COLS   = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
VOL_COLS     = ["volume", "amount", "bid_volume1", "ask_volume1"]
FEATURE_COLS = PRICE_COLS + VOL_COLS
N_FEAT       = len(FEATURE_COLS)
SCALE_FIELDS = ["open", "high", "low", "close", "amount", "bid_price1", "ask_price1"]
OHLC_COLS    = ["open", "high", "low", "close"]
MODEL_CFG    = dict(n_feat=N_FEAT, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN)
np.random.seed(42); torch.manual_seed(42)

# ---- 1. 设备：CUDA -> Apple MPS -> CPU ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("设备:", device)

# ---- 2. 模型 ----
class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model, nhead, nlayers, dim_ff, seq_len):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, 0.1,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
    def forward(self, x):                                  # (B, L, N_FEAT) -> (B,)
        return self.head(self.encoder(self.proj(x) + self.pos).mean(1)).squeeze(-1)

# ---- 3. 读本地 feather，还原压缩：分->元、-1->NaN、instrument_id 作分组键 ----
root = os.path.join(LOCAL_DATA_ROOT, LOCAL_TABLE)
buf  = (pd.to_datetime(TRAIN_START) - pd.Timedelta(days=20)).strftime("%Y%m")  # 缓冲凑窗口
lo, hi = int(buf), int(pd.Timestamp(TRAIN_END).strftime("%Y%m"))
need = set(FEATURE_COLS) | {"date", "instrument_id"}
parts = [pd.read_feather(fp, columns=list(need))
         for fp in sorted(glob.glob(os.path.join(root, "*.feather")))
         if os.path.basename(fp).split(".")[0].isdigit()
         and lo <= int(os.path.basename(fp).split(".")[0]) <= hi]
df = pd.concat(parts, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
for c in OHLC_COLS:      df.loc[df[c] == -1, c] = np.nan       # OHLC 缺失
for c in SCALE_FIELDS:   df[c] = df[c] / 100.0                 # 分 -> 元
for c in VOL_COLS:       df[c] = np.log1p(df[c].clip(lower=0)) # 量纲大先 log1p
df["key"] = df["instrument_id"]
if MAX_INSTRUMENTS:                                            # demo 限量
    df = df[df["key"].isin(df["key"].value_counts().index[:MAX_INSTRUMENTS])]
df = df.sort_values(["key", "date"])
for c in OHLC_COLS:      df[c] = df.groupby("key")[c].ffill()  # 停牌前向填充
print("标的数:", df["key"].nunique(), "| 行数:", len(df))

# ---- 4. 切滑动窗口，标签=每日末根 bar 的未来 1 日收益 ----
sd, ed = pd.to_datetime(TRAIN_START), pd.to_datetime(TRAIN_END)
wins, ys = [], []
for k, sub in df.groupby("key", sort=False):
    if len(sub) <= SEQ_LEN: continue
    feats = sub[FEATURE_COLS].to_numpy(np.float32)
    day   = sub["date"].dt.normalize().to_numpy()
    eod   = np.flatnonzero(np.append(day[1:] != day[:-1], True))   # 每日最后一根 bar
    cpx   = sub["close"].to_numpy(np.float64)[eod]
    for j, p in enumerate(eod):
        d = pd.Timestamp(day[eod][j])
        if p + 1 < SEQ_LEN or d < sd or d > ed: continue
        win = feats[p - SEQ_LEN + 1: p + 1]
        if j + 1 < len(eod) and cpx[j] > 0 and np.isfinite(win).all():
            r = cpx[j + 1] / cpx[j] - 1.0
            if np.isfinite(r):
                wins.append(win); ys.append(np.float32(r))
X = np.stack(wins).astype(np.float32)
y = np.array(ys, np.float32)
mean = X.reshape(-1, N_FEAT).mean(0).astype(np.float32)
std  = X.reshape(-1, N_FEAT).std(0).astype(np.float32) + 1e-6
X = ((X - mean) / std).astype(np.float32)
p1, p99 = np.percentile(y, [1, 99])
y = np.clip(y, p1, p99).astype(np.float32)   # winsorize；.astype 必留，否则 MPS 不吃 float64
print("样本数:", len(y))

# ---- 5. 训练 ----
model = StockTransformer(**MODEL_CFG).to(device)
loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                    batch_size=BATCH, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
model.train()
for ep in range(EPOCHS):
    t, tot = time.time(), 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item()
    print(f"epoch {ep+1}/{EPOCHS}  mse={tot/len(loader):.6f}  {time.time()-t:.1f}s")

# ---- 6. 存成纯文本 JSON（权重 + 标准化统计 + 结构超参） ----
sd_json = {k: {"dtype": str(v.dtype).replace("torch.", ""),
               "shape": list(v.shape), "data": v.cpu().reshape(-1).tolist()}
           for k, v in model.state_dict().items()}
with open(MODEL_PATH, "w", encoding="utf-8") as f:
    json.dump({"state_dict": sd_json, "model_cfg": MODEL_CFG, "feature_cols": FEATURE_COLS,
               "seq_len": SEQ_LEN, "mean": mean.tolist(), "std": std.tolist()}, f)
print("模型已保存:", MODEL_PATH)
