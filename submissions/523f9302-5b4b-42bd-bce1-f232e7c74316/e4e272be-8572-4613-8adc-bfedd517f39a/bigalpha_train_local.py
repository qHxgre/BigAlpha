# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端提交 —— 训练与推理共用定义（本地训练脚本）。

按官方《模型本地化训练指南》组织：本地用 e2e 压缩表训练，云端用原始 stock 表推理，
两边先经 `to_canonical` 归一到同一份表示，再共用 `build_windows` 构建输入，杜绝
train/infer 漂移。

配套文件：
  bigalpha_predict.py   推理侧，定义平台调用的 main(datasources, start_date, end_date)
  bigalpha_model.json   本地训练产物，随代码一起上传

与官方模板的三处不同，均来自离线扫描（官方标签口径下 34 次运行，此前另有 112 次）：

* 用 30 分钟 bar 而非 1 分钟。5m/15m 都试过，按官方四项指标的复合排名单调变差 ——
  patch 宽度固定为一天时，更细的粒度只是把同一天的信息摊薄。
* 损失是截面 IC 而非 MSE。四项指标里三项是截面排序性质，逐点回归会被收益肥尾主导。
* 多个种子的模型按日内排名平均。成员数受提交文件 50MB 上限约束，见 SEEDS 处说明。

标准化用 RevIN（网络内部、逐样本逐通道），而非在训练集上算一次全局 mean/std 存盘。
好处是推理端无须复用统计量，也就不存在统计量漂移；代价是每次前向多一次归约。
注意：RevIN 对每通道的常数缩放严格不变，所以"分 vs 元"的量纲差异本身不影响结果，
但 `to_canonical` 仍照官方清单执行——若日后换回全局统计量，对齐必须已经是对的。
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ========== 需要改的常量都在这里 ==========
LOCAL_DATA_ROOT = os.environ.get("BIGALPHA_LOCAL_DATA", "/data/raw")
LOCAL_TABLE = "bar30m"                       # 本地解压目录名
CLOUD_KEY = "bar30m"                         # datasources 的键
TRAIN_START, TRAIN_END = "2019-01-01", "2024-12-31 23:59:59"
CHUNK_SIZE = 20                              # 按股票分块，OOM 时调小
# ========================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_NAME = "bigalpha_model.json"
# 提交后所有文件平铺在同一路径，没有子目录；优先取模块同级，找不到再退回 CWD
MODEL_PATH = (os.path.join(_HERE, _MODEL_NAME)
              if os.path.exists(os.path.join(_HERE, _MODEL_NAME))
              else os.path.abspath(_MODEL_NAME))

BARS_PER_DAY = 8                             # 30 分钟：10:00 … 15:00
LOOKBACK_DAYS = 20
SEQ_LEN = BARS_PER_DAY * LOOKBACK_DAYS

OHLC_COLS = ["open", "high", "low", "close"]
LEVELS = (1, 2, 3)                           # 云端 4/5 档丢弃，与本地对齐
FEATURE_COLS = (OHLC_COLS + ["deal_number", "volume", "amount"]
                + ["ask_price%d" % i for i in LEVELS]
                + ["bid_price%d" % i for i in LEVELS]
                + ["ask_volume%d" % i for i in LEVELS]
                + ["bid_volume%d" % i for i in LEVELS]
                + ["ask_num_orders%d" % i for i in LEVELS]
                + ["bid_num_orders%d" % i for i in LEVELS])
N_FEAT = len(FEATURE_COLS)

PRICE_SCALE = 100.0
SCALE_FIELDS = OHLC_COLS + ["amount"] + \
    ["ask_price%d" % i for i in LEVELS] + ["bid_price%d" % i for i in LEVELS]

MODEL_CFG = dict(n_feat=N_FEAT, bars_per_day=BARS_PER_DAY, lookback_days=LOOKBACK_DAYS,
                 d_model=192, n_layers=4, n_heads=8, d_ff=768, dropout=0.1)
EPOCHS, DAYS_PER_BATCH, LR, WD = 12, 8, 3e-4, 0.01
# 提交的 json 不得超过 50MB（约 480 万参数）。单模型质量的差距远大于集成
# 从 2 个加到 5 个的收益（官方口径下 IC 仅 0.0872 -> 0.0875），所以宁可
# 保住 1.84M 的模型只集成 2 个种子，也不换成更小的模型集成 5 个。
SEEDS = [0, 1]
WEIGHT_DECIMALS = 6


# ---------------- 数据对齐 ----------------
def to_canonical(df, is_local):
    """本地 e2e 表 / 云端 stock 表 -> 同一份表示：价格金额为元(float)、缺失为 NaN、
    盘口只留 3 档、标的键统一为 key。"""
    df = df.copy()
    if is_local:
        for c in OHLC_COLS:
            if c in df.columns:
                df.loc[df[c] <= 0, c] = np.nan          # -1（及个别 0）为缺失
        for c in SCALE_FIELDS:
            if c in df.columns:
                df[c] = df[c].astype("float64") / PRICE_SCALE
        df["key"] = df["instrument_id"]
    else:
        drop = [c for c in df.columns
                if any(c.startswith(p) for p in ("ask_price", "bid_price", "ask_volume",
                                                 "bid_volume", "ask_num_orders",
                                                 "bid_num_orders"))
                and c[-1] not in "123"]
        df = df.drop(columns=drop, errors="ignore")
        for c in OHLC_COLS:
            if c in df.columns:
                df.loc[df[c] <= 0, c] = np.nan
        df["key"] = df["instrument"]
    return df


def build_windows(canon, sd, ed, mode):
    """canonical 表 -> (X, idx)。X 形如 (样本, SEQ_LEN, N_FEAT)。

    每个样本 = 某标的在某交易日收盘时刻、回看 LOOKBACK_DAYS 个交易日的全部 bar。
    标签沿用官方口径：close(T+1)/close(T) - 1，收盘决策、次日收盘兑现。
    """
    sd_ts, ed_ts = pd.to_datetime(sd).normalize(), pd.to_datetime(ed).normalize()
    canon = canon.sort_values(["key", "date"], kind="mergesort")
    wins, ys, keys = [], [], []
    for k, sub in canon.groupby("key", sort=False):
        feats = sub[FEATURE_COLS].to_numpy(np.float32)
        days = sub["date"].dt.normalize().to_numpy()
        last = np.flatnonzero(np.append(days[1:] != days[:-1], True))
        if len(last) <= LOOKBACK_DAYS:
            continue
        close_px = sub["close"].to_numpy(np.float64)[last]
        for j in range(LOOKBACK_DAYS - 1, len(last)):
            d = pd.Timestamp(days[last[j]])
            if d < sd_ts or d > ed_ts:
                continue
            p = last[j]
            start = last[j - LOOKBACK_DAYS + 1] - BARS_PER_DAY + 1
            if start < 0 or p + 1 - start != SEQ_LEN:
                continue                                 # 当日 bar 数不足则跳过
            label = np.nan
            if j + 1 < len(last) and close_px[j] > 0:
                r = close_px[j + 1] / close_px[j] - 1.0
                if np.isfinite(r):
                    label = r
            if mode == "train" and not np.isfinite(label):
                continue
            wins.append(feats[start:p + 1])
            ys.append(np.float32(label))
            keys.append((d, k))
    if not keys:
        return np.empty((0, SEQ_LEN, N_FEAT), np.float32), pd.DataFrame(
            columns=["date", "key", "label"])
    X = np.nan_to_num(np.stack(wins), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    idx = pd.DataFrame(keys, columns=["date", "key"])
    idx["label"] = ys
    return X, idx


# ---------------- 分块取数 ----------------
def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _buffer_start(sd):
    """回看窗口需要更早的历史；节假日让交易日少于自然日，留足冗余。"""
    return (pd.to_datetime(sd) - pd.Timedelta(days=LOOKBACK_DAYS * 3 + 30)).strftime("%Y-%m-%d")


def local_frames(sd, ed, chunk_size=CHUNK_SIZE):
    """逐批产出本地 e2e 数据的 canonical 表。支持 feather（官方压缩包）与 parquet。"""
    root = os.path.join(LOCAL_DATA_ROOT, LOCAL_TABLE)
    files = sorted(glob.glob(os.path.join(root, "*.feather"))) or \
        sorted(glob.glob(os.path.join(root, "*.parquet")))
    if not files:
        raise RuntimeError("本地数据目录为空: %s" % root)
    reader = pd.read_feather if files[0].endswith(".feather") else pd.read_parquet
    cols = ["date", "instrument_id"] + FEATURE_COLS
    lo, hi = pd.to_datetime(_buffer_start(sd)), pd.to_datetime(ed)

    parts = []
    for fp in files:
        df = reader(fp, columns=cols)
        df = df[(df["date"] >= lo) & (df["date"] <= hi)]
        if len(df):
            parts.append(df)
    if not parts:
        raise RuntimeError("区间内无本地数据: %s ~ %s" % (sd, ed))
    raw = pd.concat(parts, ignore_index=True)
    del parts
    ids = np.sort(raw["instrument_id"].unique())
    for chunk in _chunks(ids, chunk_size):
        yield to_canonical(raw[raw["instrument_id"].isin(chunk)], is_local=True)


# 官方数据页的字段表与其下方的读取示例并不一致：字段表写 price / num_trades，
# 示例输出写 close / deal_number。本地开发表用的是后者，而云端表到底是哪一套只有
# 在平台上跑才知道。与其赌一边，不如运行时按表的真实列名对齐——两组名字指的是同
# 一个量（成交价、成交笔数），别名不改变模型看到的东西。
# 云端表的标的列名。留成模块变量，是为了能用有权限的 e2e 开发表当替身，把
# cloud_frames 本身跑一遍——这条路径已经失败过两次，光测 SQL 片段不够。
CLOUD_INSTRUMENT_COL = "instrument"

FIELD_ALIASES = {"close": ["close", "price"],
                 "deal_number": ["deal_number", "num_trades"],
                 "amount": ["amount", "turnover"]}


def _resolve_columns(table, sd, ed):
    """探一行，把 FEATURE_COLS 映射到该表实际存在的列名。

    这是一段防御性代码，所以它自己绝不能成为故障源：探测失败一律退回直用
    FEATURE_COLS，让真正的取数去报错，而不是在这里把整次运行打掉。
    LIMIT 写在 SQL 里、并带上日期过滤——dai.query 没有 limit 参数，不带过滤的
    SELECT * 在分钟表上也不可接受。
    """
    try:
        import dai
        probe = dai.query("SELECT * FROM %s LIMIT 1" % table,
                          filters={"date": [sd, str(ed)]}).df()
        have = set(probe.columns)
    except Exception as e:                       # noqa: BLE001
        print("[_resolve_columns] 探测失败，按原字段名取数: %s" % str(e)[:160], flush=True)
        return list(FEATURE_COLS)

    sel, missing = [], []
    for c in FEATURE_COLS:
        if c in have:
            sel.append(c)
            continue
        alt = next((a for a in FIELD_ALIASES.get(c, []) if a in have), None)
        if alt:
            sel.append("%s AS %s" % (alt, c))
        else:
            missing.append(c)
    if missing:
        print("[_resolve_columns] 表 %s 缺字段 %s；实际列: %s"
              % (table, missing, sorted(have)), flush=True)
        return list(FEATURE_COLS)
    return sel


def cloud_frames(table, sd, ed, chunk_size=CHUNK_SIZE):
    """逐批产出云端表的 canonical 数据。

    按股票分块而非按日期：回看窗口要求单只标的在时间上连续，按日期切会让跨边界的
    窗口损坏。这也是官方 OOM 说明里给的做法。
    """
    import dai
    buf = _buffer_start(sd)
    sel = _resolve_columns(table, buf, ed)
    print("[cloud_frames] 字段映射: %s" % [c for c in sel if " AS " in c], flush=True)
    icol = CLOUD_INSTRUMENT_COL
    ins = dai.query("SELECT DISTINCT %s FROM %s" % (icol, table),
                    filters={"date": [buf, str(ed)]}).df()[icol].tolist()
    cols = ", ".join(sel)
    for chunk in _chunks(ins, chunk_size):
        # 标的写进 WHERE，filters 只留 date。把非分区键列放进 filters 会在
        # BdbPartitionPruning 里触发服务端断言失败（实测：Invalid Input Error /
        # 空指针解引用），而官方 OOM 说明的写法是否适用于每张表无从验证。
        # WHERE 形态两种表都成立，返回行数一致，分块的内存效果不变。
        quoted = ", ".join("'%s'" % str(i).replace("'", "''") for i in chunk)
        sql = ("SELECT date, %s AS instrument, %s FROM %s WHERE %s IN (%s) "
               "ORDER BY %s, date" % (icol, cols, table, icol, quoted, icol))
        df = dai.query(sql, filters={"date": [buf, str(ed)]}, compression=True).df()
        if len(df):
            yield to_canonical(df, is_local=False)
        del df


# ---------------- 模型 ----------------
class RevIN(nn.Module):
    """逐样本逐通道标准化 + 可学习仿射。对每通道的常数缩放严格不变，因此原始字段的
    绝对量纲不影响结果。"""

    def __init__(self, n_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True)
        return (x - mu) / (sd + self.eps) * self.weight + self.bias


class PatchEncoder(nn.Module):
    """一天一个 patch。收盘那根 bar 占全日成交量约三分之一，对 bar 等权池化会把这个
    结构摊平，所以 patch 嵌入是 (bar, field) 联合而非逐通道独立。"""

    def __init__(self, n_feat, bars_per_day, lookback_days, d_model,
                 n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.bars_per_day, self.lookback_days = bars_per_day, lookback_days
        self.revin = RevIN(n_feat)
        self.patch = nn.Linear(bars_per_day * n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback_days, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, x):
        b, l, f = x.shape
        x = self.revin(x)
        h = self.patch(x.reshape(b, self.lookback_days, self.bars_per_day * f)) + self.pos
        return self.head(self.norm(self.encoder(h)).mean(dim=1)).squeeze(-1)


def cs_ic_loss(pred, target):
    """单日截面上的 1 - Pearson 相关。target 是排名标准化后的标签：官方指标是
    rank IC，对原始收益做回归会让肥尾主导拟合方向。"""
    p, t = pred - pred.mean(), target - target.mean()
    den = p.norm() * t.norm()
    if den < 1e-12:
        return pred.sum() * 0.0
    return 1.0 - (p * t).sum() / den


# ---------------- 权重存取：一律 JSON 文本 ----------------
def _short(t, decimals):
    """先转 float64 再舍入。直接舍入 float32 会留下需要 17 位展开才能表示的值，
    json 仍按全长写出，文件根本不会变小。"""
    return np.round(t.detach().cpu().double().numpy().reshape(-1), decimals).tolist()


def save_model(payload, path=MODEL_PATH, decimals=WEIGHT_DECIMALS):
    out = {k: v for k, v in payload.items() if k != "members"}
    out["members"] = [{k: {"dtype": str(v.dtype).replace("torch.", ""),
                           "shape": list(v.shape), "data": _short(v, decimals)}
                       for k, v in sd.items()} for sd in payload["members"]]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return path


def load_model(path=MODEL_PATH, map_location="cpu"):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["members"] = [
        {k: torch.tensor(v["data"], dtype=getattr(torch, v["dtype"])
                         ).reshape(v["shape"]).to(map_location)
         for k, v in m.items()} for m in payload["members"]]
    return payload


# ---------------- 训练 ----------------
def _rank_std(v):
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype("float32")
    r -= r.mean()
    return r / (r.std() + 1e-12)


def train_members(X, idx, device, seeds=SEEDS, epochs=EPOCHS):
    """按日成批：截面 IC 损失需要一整天的标的同时前向，无法用梯度累积拆分。"""
    order = np.argsort(idx["date"].to_numpy(), kind="stable")
    X, idx = X[order], idx.iloc[order].reset_index(drop=True)
    dts = idx["date"].to_numpy()
    bounds = np.append(np.flatnonzero(np.append(True, dts[1:] != dts[:-1])), len(idx))
    slices = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    slices = [s for s in slices if s[1] - s[0] >= 50]
    targets = np.concatenate([_rank_std(idx["label"].to_numpy()[a:b]) for a, b in slices])
    offs = np.cumsum([0] + [b - a for a, b in slices])

    members = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PatchEncoder(**MODEL_CFG).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        total = epochs * max(len(slices) // DAYS_PER_BATCH, 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total)
        rng = np.random.default_rng(seed)
        model.train()
        for ep in range(1, epochs + 1):
            t0, tot, nb = time.time(), 0.0, 0
            perm = rng.permutation(len(slices))
            for i in range(0, len(perm), DAYS_PER_BATCH):
                chunk = perm[i:i + DAYS_PER_BATCH]
                xb = np.concatenate([X[slices[j][0]:slices[j][1]] for j in chunk])
                tb = np.concatenate([targets[offs[j]:offs[j + 1]] for j in chunk])
                segs, o = [], 0
                for j in chunk:
                    n = slices[j][1] - slices[j][0]
                    segs.append((o, o + n))
                    o += n
                xb = torch.from_numpy(xb).to(device)
                tb = torch.from_numpy(tb).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    pred = model(xb)
                loss = torch.stack([cs_ic_loss(pred.float()[a:b], tb[a:b])
                                    for a, b in segs]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                tot += float(loss)
                nb += 1
            print("[train] seed=%d ep=%d loss=%.4f %.0fs"
                  % (seed, ep, tot / max(nb, 1), time.time() - t0), flush=True)
        members.append(model.state_dict())
    return members


def train_and_save(model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] device=%s" % device, flush=True)
    Xs, idxs = [], []
    for canon in local_frames(TRAIN_START, TRAIN_END):
        X, idx = build_windows(canon, TRAIN_START, TRAIN_END, "train")
        if len(idx):
            Xs.append(X)
            idxs.append(idx)
        del canon, X, idx
    X = np.concatenate(Xs)
    idx = pd.concat(idxs, ignore_index=True)
    del Xs, idxs
    print("[train] samples=%d" % len(idx), flush=True)

    t0 = time.time()
    members = train_members(X, idx, device)
    save_model({"model_cfg": MODEL_CFG, "feature_cols": FEATURE_COLS,
                "bars_per_day": BARS_PER_DAY, "lookback_days": LOOKBACK_DAYS,
                "seeds": SEEDS, "train_range": [TRAIN_START, TRAIN_END],
                "members": members}, model_path)
    print("[train] %d members, %.1f min, %.1f MB"
          % (len(members), (time.time() - t0) / 60,
             os.path.getsize(model_path) / 1e6), flush=True)
    return model_path


if __name__ == "__main__":
    train_and_save()
