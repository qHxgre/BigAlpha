import os, json, time, math, random, gc, base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ##########################################################################
# ##########            ★★★  配 置 区  ★★★                        ##########
# ##########################################################################

# ---------- 平台 ----------
TABLE_KEY    = "bar5m"                              # datasources 里的键
BAR_TABLE    = "bigalpha_2026_stock_bar5m"          # 5 分钟 K 线(38 字段全)
INSTRU_TABLE = "bigalpha_2026_instruments"          # PIT 中证1000 成分股
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w3b_model.json")

# ---------- 区间(★唯一要填的东西)----------
# ⚠️ 这是**可见数据的全区间**, 训练集和验证集都从它里面切(见 split_fit):
#      TRAIN_START ─── 80% 训练 ───┤GAP├─── 20% 验证 ─── TRAIN_END
#    真正的训练集区间在 meta["split"]["train_start"/"train_end"] 里, 别和这两个常量混淆。
TRAIN_START, TRAIN_END = "2019-08-01", "2024-12-31"
VAL_FRAC  = 0.2     # 末尾这么多比例当验证集(早停)
GAP_DAYS  = 3       # 训练集与验证集之间空几个交易日 —— label 用到 t+2, 不空就泄漏

# ---------- 集成 ----------
SEEDS      = (42, 43, 44)   # K=3: 每个种子跑完整三阶段(N05→W1→W3b)
DROP_RATIO = 0.8            # 某成员的 W3b val IC < 中位数×这个值 ⇒ 判定崩溃, 剔除
# ⚠️ 0.8 的依据: 等权集成 IC ≈ mean(IC_k)·√(K/(1+(K−1)ρ̄))。K=3 时, 一个成员 IC 低 10%
#    仍然让集成优于单模型(ρ̄<0.9); 只有掉到一半才会拖垮。所以门槛防的是**崩溃**, 不是"稍差"。

# ---------- 结构 ----------
NB      = 48        # 每天 5m bar 数
D       = 30        # 回看天数(规则 A5: ≤240)
D_EMB   = 128       # 日嵌入维度
PLE_D   = 8         # PLE 每字段读出维度
PLE_T   = 32        # PLE 分箱数
D_MOD   = 48        # 调制块宽度
N_HEAD  = 2         # 调制块头数
FFN_MULT = 2
CSF_TOPK, CSF_DA = 16, 16      # 前置截面
AX_H             = 16          # 日间轴 GRU 隐维(实测 h=8 贡献为零、h=32 只剩三分之一)
CS_TOPK,  CS_DA  = 16, 16      # 后置截面
NOISE_SCRATCH    = 0.05        # 从零训的两个阶段加噪(冻结阶段不加: 主干不动, 加噪只会让 raw 与冻结的 H 失配)

# ---------- 训练 ----------
SEED        = 42
EPOCHS_MAX  = 100          # 早停的上限(实际轮数由 val 决定)
PATIENCE    = 15
LR          = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.1
GRAD_CLIP   = 1.0
EMA_DECAY   = 0.999
MB_DAYS     = 10           # 每个优化步用多少天(损失相加, 一次 backward)
MIN_STOCKS  = 30           # 当日有效票数低于此值就跳过
SR_TAU      = 0.5          # soft-rank 温度
BUF_DAYS    = 70           # 读数据时往前多取的缓冲(凑 D 天回看窗口)
USE_AMP     = True
NUM_THREADS = 4            # ⚠️ 实例只 ~1 真核却报告 64 核, 不钳会慢几十倍
GPU_LIMIT_H = 3            # 本账号的 GPU notebook 单次上限(规则 A12 写的是 6h, 实际给 3h)
MAX_JSON_MB = 50           # 提交的权重文件上限
SEG_MONTHS  = 6            # 分块读取的粒度(平台单次 query ≤200MB, 5m 全周期 ~11GB)

# ---------- 38 个原始字段(规则 A3: ≤100)----------
PRICE_F = ["pre_close", "open", "high", "low", "close"] + \
          [f"{s}_price{i}" for s in ("ask", "bid") for i in range(1, 6)]
VOL_F   = ["volume", "amount", "deal_number"] + \
          [f"{s}_volume{i}" for s in ("ask", "bid") for i in range(1, 6)]
NUM_F   = [f"{s}_num_orders{i}" for s in ("ask", "bid") for i in range(1, 6)]
FEAT    = PRICE_F + VOL_F + NUM_F
assert len(FEAT) == 38

# bycat 归一化的字段分类: 价格按"电平 + 相对结构"拆, 量类同语义填充后 log1p + 共享 z
J_PRECLOSE = FEAT.index("pre_close")
J_OPEN     = FEAT.index("open")
IDX_STRUCT = np.array([FEAT.index(c) for c in ["open", "high", "low", "close"] +
                       [f"{s}_price{i}" for s in ("ask", "bid") for i in range(1, 6)]], np.int64)
SET_OBPRICE = {FEAT.index(f"{s}_price{i}") for s in ("ask", "bid") for i in range(1, 6)}
IDX_VOL3 = np.array([FEAT.index(c) for c in ["volume", "amount", "deal_number"]], np.int64)
IDX_OBV  = np.array([FEAT.index(f"{s}_volume{i}") for s in ("ask", "bid") for i in range(1, 6)], np.int64)
IDX_OBN  = np.array([FEAT.index(c) for c in NUM_F], np.int64)
IDX_FILL = np.concatenate([IDX_VOL3, IDX_OBV, IDX_OBN])       # 23 个"零 = 没成交"的字段

# PLE 箱界分组 = bycat 的统计量分组(7 组): **同组共享一套箱界**, 保类内**跨字段**可比 ——
# 盘口档间的深度衰减、价格阶梯本身就是跨字段的信息。逐字段独立分箱会把常空深档的微小噪声
# 拉伸到与档1同样的满量程(受害最大的是盘口 10 档量 + 10 档笔数这 20 个字段)。
# ⚠️ 共享的只有**箱界**; PLE 的 W_f 与 e_mean 仍逐字段。
# 只用训练块统计 ⇒ 无未来函数; 无监督分位分箱(不碰 label)⇒ 私榜重训可复现。
PLE_GROUPS = [("pc",      [int(J_PRECLOSE)]),                 # 价格·电平 1
              ("pstruct", [int(j) for j in IDX_STRUCT]),      # 价格·结构 14 共享
              ("volume",  [int(IDX_VOL3[0])]),                # 量能 逐字段(尺度各异, 水平即信号)
              ("amount",  [int(IDX_VOL3[1])]),
              ("dealnum", [int(IDX_VOL3[2])]),
              ("obv",     [int(j) for j in IDX_OBV]),         # 盘口量 10 共享
              ("obn",     [int(j) for j in IDX_OBN])]         # 笔数 10 共享
assert sorted(j for _, idxs in PLE_GROUPS for j in idxs) == list(range(len(FEAT))), \
    "PLE 分组须不重不漏覆盖 38 个字段"

# PLE 箱界分组 = bycat 的统计量分组(7 组): **同组共享一套箱界**, 保类内**跨字段**可比 ——
# 盘口档间的深度衰减、价格阶梯本身就是跨字段的信息。逐字段独立分箱会把常空深档的微小噪声
# 拉伸到与档1同样的满量程(受害最大的是盘口 10 档量 + 10 档笔数这 20 个字段)。
# ⚠️ 共享的只有**箱界**; PLE 的 W_f 与 e_mean 仍逐字段。
# 只用训练块统计 ⇒ 无未来函数; 无监督分位分箱(不碰 label)⇒ 私榜重训可复现。
PLE_GROUPS = [("pc",      [int(J_PRECLOSE)]),                 # 价格·电平 1
              ("pstruct", [int(j) for j in IDX_STRUCT]),      # 价格·结构 14 共享
              ("volume",  [int(IDX_VOL3[0])]),                # 量能 逐字段(尺度各异, 水平即信号)
              ("amount",  [int(IDX_VOL3[1])]),
              ("dealnum", [int(IDX_VOL3[2])]),
              ("obv",     [int(j) for j in IDX_OBV]),         # 盘口量 10 共享
              ("obn",     [int(j) for j in IDX_OBN])]         # 笔数 10 共享
assert sorted(j for _, idxs in PLE_GROUPS for j in idxs) == list(range(len(FEAT))), \
    "PLE 分组须不重不漏覆盖 38 个字段"


def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _log(msg):
    print(msg, flush=True)


# ##########################################################################
# ##########                      数 据                            ##########
# ##########################################################################

def trading_days(lo, hi):
    """区间内交易日(自然日)。用 instruments 表 —— 它天然是"有成分股的日子"。"""
    import dai
    df = dai.query(f"SELECT DISTINCT date FROM {INSTRU_TABLE}", filters={"date": [lo, hi]}).df()
    d = pd.to_datetime(df["date"]).dt.normalize().unique()
    return np.sort(np.asarray(d, dtype="datetime64[ns]"))


def split_fit(cal):
    """把 [TRAIN_START, TRAIN_END] 里的交易日切成 训练 / GAP / 验证, 返回四个 "YYYY-MM-DD"。

    ⚠️ 切的是**交易日序号**而不是自然日 —— 按自然日切会因为节假日分布不均而让两段的
       实际交易日数偏离目标比例。
    ⚠️ GAP 必须在训练集**之后**、验证集之前: 训练集最后一天 t 的 label 用 t+1/t+2 的开盘价,
       空出 GAP_DAYS(≥2)才能保证那两天不落进验证集。"""
    lo, hi = np.datetime64(TRAIN_START), np.datetime64(TRAIN_END)
    days = np.asarray([d for d in cal if lo <= d <= hi])
    n = len(days)
    n_val = int(round(n * VAL_FRAC))
    i_va = n - n_val                       # 验证集第一天的下标
    i_tr = i_va - GAP_DAYS                 # 训练集最后一天的下标 + 1
    assert GAP_DAYS >= 2, f"❌ GAP_DAYS={GAP_DAYS} < 2 —— label 用到 t+2, 会泄漏进验证集"
    assert n_val > D + 5, \
        f"❌ 验证集只有 {n_val} 个交易日, 不足回看窗口 D={D} —— 区间太短或 VAL_FRAC 太小"
    assert i_tr > D + 5, \
        f"❌ 训练集只有 {i_tr} 个交易日, 不足回看窗口 D={D}"
    f = lambda x: str(x)[:10]
    return f(days[0]), f(days[i_tr - 1]), f(days[i_va]), f(days[-1])


def pit_pool(lo, hi):
    """PIT 中证1000 成分股 (date, instrument)。
    ⚠️ 规则 A21: bar 表存的是**全周期并集**(单日 ~1700 只 > 当日 1000), 训练与打分都必须
       逐日 join 这张表筛当日成员 —— 池外股票属违规数据。"""
    import dai
    df = dai.query(f"SELECT date, instrument FROM {INSTRU_TABLE}", filters={"date": [lo, hi]}).df()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.drop_duplicates(["date", "instrument"])


def _month_segs(lo, hi, months=SEG_MONTHS):
    out, cur, end = [], pd.Timestamp(lo), pd.Timestamp(hi)
    while cur <= end:
        nxt = min(cur + pd.DateOffset(months=months) - pd.Timedelta(days=1), end)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.Timedelta(days=1)
    return out


def _blocks_from_df(df, store, cal_pos):
    """df → 每票每天一个 (48, 38) 的块。只收"当天铺满 48 根 bar"的日子。
    cal_pos: {自然日 -> 交易日序号}, 用来判断窗口是否连续。"""
    for code, sub in df.groupby("instrument", sort=False, observed=True):
        sub = sub.sort_values("date")
        arr = sub[FEAT].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        last = np.flatnonzero(np.append(day[1:] != day[:-1], True))     # 每日最后一根 bar
        blocks, cps, opn = [], [], []
        for lp in last:
            st, d = lp - NB + 1, day[lp]
            if st < 0 or day[st] != d or d not in cal_pos:              # 不满 48 根 / 跨日 / 非交易日
                continue
            blocks.append(arr[st:lp + 1]); cps.append(cal_pos[d]); opn.append(arr[st, J_OPEN])
        if not blocks:
            continue
        blk = np.stack(blocks).astype(np.float32)
        prev = store.get(str(code))
        cur = dict(blocks=blk, ok=np.isfinite(blk).all(axis=(1, 2)),
                   cps=np.array(cps, np.int64), open=np.array(opn, np.float64))
        if prev is None:
            store[str(code)] = cur
        else:                                                           # 分块读取时同票会出现多段
            store[str(code)] = {k: np.concatenate([prev[k], cur[k]]) for k in cur}


def load_blocks(table, lo, hi, pool_df, cal_pos, tag=""):
    """分段读 5m bar(逐日限 PIT 池)→ 建块。

    ★右边界: bar 表的 date 带时分(09:35...15:00), 而 filters 的右端点按 00:00 比较 ——
      直接传段末 "2019-11-22" 会把那天 48 根 bar 全部排除, 而这天又落不进下一段
      (下段从 11-23 起)⇒ 每个分段的末尾交易日**永久丢失**。一天 4.8 万行 / 580 万行,
      在累计行数里看不出来, 但 build_samples 要连续 D 天 ⇒ 每丢一天作废其后 D 天的样本。
      实测漏了 6 天 ⇒ 每轮 80 步而不是 106 步, 训练量少 25%。
      改成: 右界传"段末+1 自然日", 读回后按段末截断(多读一天, 与 dai 的边界语义无关)。"""
    import dai
    sql = f"SELECT date, instrument, {', '.join(FEAT)} FROM {table} ORDER BY instrument, date"
    store, n, t0 = {}, 0, time.time()
    pool_mi = pd.MultiIndex.from_frame(pool_df[["date", "instrument"]])
    segs = _month_segs(lo, hi)
    seg_ends = {s for _, s in segs}
    for slo, shi in segs:
        rb = (pd.Timestamp(shi) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # ★排他右界
        df = dai.query(sql, filters={"date": [slo, rb]}).df()
        if len(df):
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] < pd.Timestamp(rb)]        # ★截回段末, 防右界含整天时段间重复
        if len(df):
            keep = pd.MultiIndex.from_arrays(
                [df["date"].dt.normalize(), df["instrument"]]).isin(pool_mi)
            df = df[keep].copy()
            df["instrument"] = df["instrument"].astype("category")
            _blocks_from_df(df, store, cal_pos)
            n += len(df)
        del df; gc.collect()
        _log(f"  [{tag}] {slo}~{shi}  累计 {n:,} 行  +{time.time()-t0:.0f}s")
    _log(f"  [{tag}] {len(store)} 票 / {sum(len(s['cps']) for s in store.values()):,} 块  "
         f"+{time.time()-t0:.0f}s")

    # ★日历一致性: cal 来自 instruments(日频表, 不受上面那个边界影响), 块来自 bar5m。
    #   某天有成分股却没建出块 ⇒ 所有票的 cps 在那里断裂 ⇒ 其后 D 天的样本全部作废。
    _have = {int(c) for s in store.values() for c in s["cps"]}
    _lo, _hi = np.datetime64(lo), np.datetime64(hi)
    _gap = sorted(d for d, i in cal_pos.items() if _lo <= d <= _hi and i not in _have)
    _span = sum(1 for d in cal_pos if _lo <= d <= _hi)
    if _gap:
        _g = [str(d)[:10] for d in _gap]
        _at_end = [x for x in _g if x in seg_ends]        # 段末型 = 右边界修正没生效
        _log(f"  [{tag}] ⚠️⚠️ 日历缺口 {len(_gap)}/{_span} 天 —— 每个缺口作废其后 {D} 个交易日的样本")
        _log(f"  [{tag}]    前 12 个: {_g[:12]}")
        if _at_end:
            _log(f"  [{tag}]    ❌ 其中 {len(_at_end)} 天落在分段右端点上 —— "
                 f"右边界修正没生效: {_at_end[:6]}")
        else:
            _log(f"  [{tag}]    ✅ 无段末型缺口(右边界已修正), 余下是 bar 表本身没覆盖的日子")
        _cnt = {}
        for s in store.values():
            for c in s["cps"]:
                _cnt[int(c)] = _cnt.get(int(c), 0) + 1
        _inv = {i: d for d, i in cal_pos.items()}
        _log(f"  [{tag}]    票数最少的 5 天: "
             f"{[(str(_inv[k])[:10], v) for v, k in sorted((v, k) for k, v in _cnt.items())[:5]]}")
    else:
        _log(f"  [{tag}] 日历一致 ✅ {_span} 天全部建出块")
    # ⚠️ 这里**故意不排序**: store 的键序决定 fit_bycat 的分位采样、fit_ple_edges 按行号
    #   抽到的块、以及每天票的排列 ⇒ 排序会改掉归一化常数与 PLE 箱界, 让跨轮次的结果失去锚点。
    #   (排序本身是对的 —— 能让公榜首跑/调试重跑/私榜重训的常数逐位一致 —— 但那是定版时再加的事,
    #    调试期需要的是稳定基准。见 c:/tmp/patch_order.py 与 patch_unsort.py。)
    return store


def bycat_fill(store):
    """同语义填充(23 个量类字段, in-place): 零 → 块内非零min → 该股全期非零min → 全局非零min。
    ⚠️ 价格类不填(盘口价的零在归一化时内联填 pre_close)。这是数据本地操作, 不算训练统计量。"""
    smin, gmin = {}, np.full(len(IDX_FILL), np.inf)
    for c, st in store.items():
        col = st["blocks"][:, :, IDX_FILL]
        smin[c] = (np.where(col > 0, col, np.inf).reshape(-1, len(IDX_FILL)).min(0)
                   if col.size else np.full(len(IDX_FILL), np.inf)).astype(np.float64)
        gmin = np.minimum(gmin, smin[c])
    gmin = np.where(np.isfinite(gmin), gmin, 1.0)
    for c, st in store.items():
        blk, sm = st["blocks"], smin[c]
        for k, j in enumerate(IDX_FILL):
            col = blk[:, :, int(j)]                                    # (N,48) view
            bmin = np.where(col > 0, col, np.inf).min(1)               # 块内非零 min
            fv = np.where(np.isinf(bmin), sm[k], bmin)
            fv = np.where(np.isfinite(fv), fv, gmin[k]).astype(np.float32)
            zm = col == 0
            col[zm] = np.broadcast_to(fv[:, None], col.shape)[zm]


def fit_bycat(store):
    """在(已填充的)训练块上算归一化参数。⚠️ 规则 A20: 统计量只能用训练集算。
    价格拆成"电平(log pre_close 的 z)"+"结构(相对 pre_close 的对数偏离)";
    量类 log1p 后, 成交量三项各自 z、盘口量/笔数各自**共享**一套 z(同语义不该被拆开)。"""
    total = sum(st["blocks"].shape[0] for st in store.values())
    K = max(1, total // 10000)                                          # 分位数采样 ~1 万块
    pcs, devs, bc = [], [], 0
    s1v = np.zeros(len(IDX_VOL3)); s2v = np.zeros(len(IDX_VOL3)); nv = 0
    s1o = s2o = no = s1n = s2n = nn_ = 0.0
    for st in store.values():
        blk = st["blocks"]; N = blk.shape[0]
        v3 = np.log1p(np.clip(blk[:, :, IDX_VOL3].astype(np.float64), 0, None)).reshape(-1, len(IDX_VOL3))
        s1v += v3.sum(0); s2v += (v3 * v3).sum(0); nv += v3.shape[0]
        ov = np.log1p(np.clip(blk[:, :, IDX_OBV].astype(np.float64), 0, None))
        s1o += ov.sum(); s2o += (ov * ov).sum(); no += ov.size
        on = np.log1p(np.clip(blk[:, :, IDX_OBN].astype(np.float64), 0, None))
        s1n += on.sum(); s2n += (on * on).sum(); nn_ += on.size
        sel = np.nonzero((np.arange(N) + bc) % K == 0)[0]; bc += N
        if sel.size:
            sb = blk[sel].astype(np.float64)
            pc = np.log(np.clip(sb[:, :, J_PRECLOSE], 1e-6, None)); pcs.append(pc.ravel())
            for j in IDX_STRUCT:
                x = sb[:, :, int(j)]
                if int(j) in SET_OBPRICE:
                    x = np.where(x > 0, x, sb[:, :, J_PRECLOSE])         # 无报价 → 填 pre_close
                devs.append((np.log(np.clip(x, 1e-6, None)) - pc).ravel())
    pcs, devs = np.concatenate(pcs), np.concatenate(devs)
    mu_pc, sd_pc = pcs.mean(), pcs.std() + 1e-9
    lo_pc, hi_pc = np.percentile((pcs - mu_pc) / sd_pc, [1, 99])
    wlo, whi = np.percentile(devs, [0.1, 99.9])
    mu_v3 = s1v / nv
    return dict(mu_pc=float(mu_pc), sd_pc=float(sd_pc), lo_pc=float(lo_pc), hi_pc=float(hi_pc),
                wlo=float(wlo), whi=float(whi),
                sig_struct=float(np.std(np.clip(devs, wlo, whi)) + 1e-9),
                mu_v3=mu_v3, sd_v3=np.sqrt(np.maximum(s2v / nv - mu_v3 ** 2, 1e-12)) + 1e-9,
                mu_ov=float(s1o / no), sd_ov=float(np.sqrt(max(s2o / no - (s1o / no) ** 2, 1e-12)) + 1e-9),
                mu_on=float(s1n / nn_), sd_on=float(np.sqrt(max(s2n / nn_ - (s1n / nn_) ** 2, 1e-12)) + 1e-9))


def normalize_bycat(store, S):
    """逐块施加归一化 → f16(省一半显存)。"""
    for st in store.values():
        blk = st["blocks"]; out = np.empty_like(blk, np.float32)
        pc = np.log(np.clip(blk[:, :, J_PRECLOSE], 1e-6, None)).astype(np.float32)
        out[:, :, J_PRECLOSE] = np.clip((pc - S["mu_pc"]) / S["sd_pc"], S["lo_pc"], S["hi_pc"])
        for j in IDX_STRUCT:
            jj = int(j); x = blk[:, :, jj]
            if jj in SET_OBPRICE:
                x = np.where(x > 0, x, blk[:, :, J_PRECLOSE])
            dev = np.log(np.clip(x, 1e-6, None)).astype(np.float32) - pc
            out[:, :, jj] = np.clip(dev, S["wlo"], S["whi"]) / S["sig_struct"]
        for k, j in enumerate(IDX_VOL3):
            v = np.log1p(np.clip(blk[:, :, int(j)], 0, None)).astype(np.float32)
            out[:, :, int(j)] = (v - S["mu_v3"][k]) / S["sd_v3"][k]
        for j in IDX_OBV:
            v = np.log1p(np.clip(blk[:, :, int(j)], 0, None)).astype(np.float32)
            out[:, :, int(j)] = (v - S["mu_ov"]) / S["sd_ov"]
        for j in IDX_OBN:
            v = np.log1p(np.clip(blk[:, :, int(j)], 0, None)).astype(np.float32)
            out[:, :, int(j)] = (v - S["mu_on"]) / S["sd_on"]
        st["blocks"] = out.astype(np.float16)


def build_samples(store, cal, lo, hi, need_label=True):
    """→ {日期: [(票, 块序号, label), ...]}。
    label = open[t+2] / open[t+1] − 1(**次日开盘 → 后日开盘**, 可实际交易的口径)。
    ⚠️ 窗口必须是**连续 D 个交易日**且每块都 finite —— 停牌/半天的票不参与那一天。"""
    lo, hi = np.datetime64(lo), np.datetime64(hi)
    groups = {}
    for code, s in store.items():
        cps, opn, n = s["cps"], s["open"], len(s["cps"])
        for li in range(n):
            cp = cps[li]; d = cal[cp]
            if d < lo or d > hi or li - D + 1 < 0:
                continue
            if not np.array_equal(cps[li - D + 1:li + 1], np.arange(cp - D + 1, cp + 1)):
                continue                                                # 窗口不连续
            if not s["ok"][li - D + 1:li + 1].all():
                continue
            has = li + 2 < n and cps[li + 1] == cp + 1 and cps[li + 2] == cp + 2 and opn[li + 1] > 0
            r = opn[li + 2] / opn[li + 1] - 1.0 if has else np.nan
            if need_label and not np.isfinite(r):
                continue
            groups.setdefault(d, []).append((code, li, np.float32(r)))
    return groups


def to_tensors(store, groups, device=DEVICE):
    """块表 (n,48,38) f16 + 每日的窗口索引 (n_samples, D) + label。"""
    codes = list(store.keys()); off, tot = {}, 0
    for c in codes:
        off[c] = tot; tot += store[c]["blocks"].shape[0]
    BLK = torch.empty((tot, NB, len(FEAT)), dtype=torch.float16, device=device)
    for c in codes:
        b = store[c]["blocks"]; o = off[c]
        BLK[o:o + b.shape[0]] = torch.from_numpy(b).to(device)
        store[c]["blocks"] = None
    idx, y, meta = {}, {}, {}
    for d, ents in groups.items():
        rows = np.empty((len(ents), D), np.int64); ys = np.empty(len(ents), np.float32)
        for i, (c, li, r) in enumerate(ents):
            base = off[c] + li
            rows[i] = np.arange(base - D + 1, base + 1); ys[i] = r
        idx[d] = torch.from_numpy(rows).to(device)
        y[d] = torch.from_numpy(ys).to(device)
        meta[d] = [(c, li) for c, li, _ in ents]
    return BLK, idx, y, meta


def cur_blocks(idx_d):
    """窗口末列 = 当天的块。"""
    return idx_d[:, -1]


def fit_ple_edges(BLK, T=PLE_T, n_sample=20000, seed=0, chunk=2000):
    """分位数箱界(只在训练块上算一次, 三阶段共用)。是数据的分位数, 不是学出来的参数。

    ★按 PLE_GROUPS **分组共享**箱界(与 bycat 的统计量分组一致)—— 见 PLE_GROUPS 处的说明。
    ⚠️ 分位数走 **float64 numpy** 而不是 torch.quantile: 后者是 f32, 而且**不过滤非有限值**,
       一个 NaN 就让整个字段的箱界变 NaN, 再经 inv_w 传染到全网。
    ⚠️ 零宽箱的下限是**范围的千分之一**而不是固定 1e-6: 0 质量点(如常空深档)会给出一串
       相等的分位数, 固定 1e-6 会让 inv_w 冲到 1e6。
    ⚠️ e_mean 用 f64 累加: 96 万项在 f32 上累加, 尾部会被整个吃掉。"""
    rng = np.random.default_rng(seed)
    n = BLK.shape[0]
    sel = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
    sub = BLK[torch.from_numpy(sel).to(BLK.device)].float().cpu()        # (m,48,38)
    _log(f"[ple] 采样 {len(sel):,}/{n:,} 块({sub.numel()/1e6:.1f}M 值) T={T}")

    edges = np.zeros((len(FEAT), T + 1), np.float32)
    qs = np.linspace(0.0, 1.0, T + 1)
    for gname, idxs in PLE_GROUPS:
        v = sub[:, :, idxs].reshape(-1).numpy()
        v = v[np.isfinite(v)]
        q = np.quantile(v, qs)
        floor = 1e-3 * max(q[-1] - q[0], 1e-6)
        n_fix = 0
        for i in range(1, T + 1):
            if q[i] < q[i - 1] + floor:
                q[i] = q[i - 1] + floor; n_fix += 1
        for j in idxs:
            edges[j] = q
        _log(f"  {gname:8} {len(idxs):>2}字段 共享  范围[{q[0]:+.3f},{q[-1]:+.3f}]  "
             f"中位箱宽={np.median(np.diff(q)):.4f}" + (f"  ⚠️零宽修正{n_fix}处" if n_fix else ""))

    # e_mean: 温度计编码的均值, 用来把 PLE 的 DC 分量折进 bias(显著改善优化条件数)
    lo = torch.from_numpy(edges[:, :-1]); hi = torch.from_numpy(edges[:, 1:])
    inv_w = 1.0 / (hi - lo).clamp(min=1e-6)
    cmin = torch.zeros(T); cmin[0] = float("-inf")
    cmax = torch.ones(T);  cmax[-1] = float("inf")
    acc = torch.zeros(len(FEAT), T, dtype=torch.float64); cnt = 0
    for i in range(0, sub.shape[0], chunk):
        e = torch.clamp((sub[i:i + chunk].unsqueeze(-1) - lo) * inv_w, cmin, cmax)
        acc += e.reshape(-1, len(FEAT), T).sum(0).double()
        cnt += e.shape[0] * e.shape[1]
    e_mean = (acc / max(cnt, 1)).float().numpy()
    _log(f"[ple] e_mean 完成(逐字段, {cnt:,} 样本)  DC分量均值={e_mean.mean():.3f}")
    return edges, e_mean


# ##########################################################################
# ##########                      模 型                            ##########
# ##########################################################################

def _zero(*mods):
    for m in mods:
        nn.init.zeros_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


class PLE(nn.Module):
    """分段线性编码: x(...,38) → (...,38,d)。每字段一套 (T,d), 把 T 个温度计分量压成 d 个读数。

    ⚠️ **首末箱不截断**(e₁ 可 <0, e_T 可 >1): 保尾部分辨率, 且使 Σe·箱宽 = x−b₀ 在全定义域成立
       ⇒ identity 初始化下 PLE **严格等价于恒等**, 起点不引入任何偏移。
    ⚠️ **中间箱必须截断**: 不截则 T 个分量线性相关、秩塌成 1, PLE 退化成普通线性缩放。"""

    def __init__(self, edges, e_mean, d=PLE_D):
        super().__init__()
        F, Tp1 = edges.shape; T = Tp1 - 1
        self.F, self.T, self.d = F, T, d
        lo = torch.as_tensor(np.ascontiguousarray(edges[:, :-1]), dtype=torch.float32)
        hi = torch.as_tensor(np.ascontiguousarray(edges[:, 1:]), dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("inv_w", 1.0 / (hi - lo).clamp(min=1e-6))
        self.register_buffer("width", hi - lo)
        self.register_buffer("e_mean", torch.as_tensor(np.ascontiguousarray(e_mean), dtype=torch.float32))
        cmin = torch.zeros(T); cmin[0] = float("-inf")
        cmax = torch.ones(T);  cmax[-1] = float("inf")
        self.register_buffer("cmin", cmin); self.register_buffer("cmax", cmax)
        self.W = nn.Parameter(torch.randn(F, T, d) / (T ** 0.5))
        self.b = nn.Parameter(torch.zeros(F, d))
        with torch.no_grad():                                   # identity 初始化: 第0列还原 x, 其余小随机
            self.W.mul_(0.02)
            self.W[:, :, 0] = self.width
            self.b.zero_()
            self.b[:, 0] = self.lo[:, 0] + (self.e_mean * self.width).sum(-1)

    def forward(self, x):
        e = torch.clamp((x.unsqueeze(-1) - self.lo) * self.inv_w, self.cmin, self.cmax)
        # center 折进 bias: einsum(e−ē,W)+b ≡ einsum(e,W)+(b−einsum(ē,W)), 省一份 (B,48,F,T) 分配
        b_eff = self.b - torch.einsum("ft,ftd->fd", self.e_mean, self.W)
        return torch.einsum("...ft,ftd->...fd", e, self.W) + b_eff


class DenoiseBlock(nn.Module):
    """48 根 bar 互相看 → 对 x 的加性修正。**修正作用在 38 维原始空间**, 不是 PLE 的 304 维。

    ⚠️ 为什么必须在 38 维: PLE identity 下 304 维里 38 个 d0 通道 std≈1、其余 266 个辅助读数
       std≈0.0084。在 304 维上修正会让辅助通道收到自身尺度 28 倍的扰动, transformer 会把它们
       整个用噪声覆盖。作用在 38 维则修正只落在字段值上、PLE 自己传播。
    ⚠️ 必须减去 z_in: 否则 in_proj(x)@W ≈ x 会混进修正量, 使 α 同时在**整体缩放 x** ——
       那样 α 不再是纯"修正强度", 而且 PLE 箱界固定、x 被放大就整体外移。
    ⚠️ 加法走 f32: α 起步为 0, 早期 α·Δ 在 1e-3 量级, 而 bf16 在 1.0 附近步长 0.0078 ⇒
       增量会被整个舍掉、**α 永远醒不过来**。"""

    def __init__(self, n_feat=len(FEAT), nb=NB, d=D_MOD, n_state=6, n_head=N_HEAD, ffn_mult=FFN_MULT):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, d, bias=False)          # 无 bias: 写回时绑定转置, bias 无对应
        self.pos_emb = nn.Parameter(torch.randn(nb, d) * 0.02)   # ×0.02: 不能盖过内容
        self.type_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.state_proj = nn.Linear(n_state, d)                  # 寄存器 token(state 恒零 ⇒ 用它的 bias)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Linear(d * ffn_mult, d))
        self.alpha = nn.Parameter(torch.zeros(n_feat))           # ★逐字段, init 0 ⇒ 起点 = 无调制
        self.n_state = n_state
        with torch.no_grad():                                    # 稀疏选择: 对角 +1, 其余小随机
            self.in_proj.weight.normal_(0, 0.02)                 # (d, n_feat)
            for f in range(min(n_feat, d)):
                self.in_proj.weight[f, f] += 1.0

    def forward(self, x):                                        # (B,48,38) f32
        # state 恒零(state_off): 市场状态那一路在本方案里关闭, 寄存器退化成 state_proj.bias + type_emb
        st = torch.zeros(x.shape[0], self.n_state, device=x.device, dtype=torch.float32)
        z_in = self.in_proj(x) + self.pos_emb                    # (B,48,d)
        s = (self.state_proj(st) + self.type_emb)[:, None, :]    # (B,1,d) 寄存器
        z = torch.cat([s, z_in], 1)
        h = self.ln1(z)
        z = z + self.attn(h, h, h, need_weights=False)[0]
        z = z + self.ffn(self.ln2(z))
        dz = z[:, 1:] - z_in                                     # 纯修正(丢掉寄存器那一行)
        d_x = dz @ self.in_proj.weight                           # 绑定转置写回 → (B,48,38)
        return x + self.alpha * d_x.float()


class Trunk(nn.Module):
    """主干: DenoiseBlock → PLE → 双向 GRU → raw(128)。

    ⚠️ `seed` 在建 GRU **之前**重置 —— PLE 的 W=randn 会消耗 RNG, 不重置的话不同阶段的
       GRU 初始化会错开, 阶段之间不可比。"""

    def __init__(self, edges, e_mean, seed=None):
        super().__init__()
        self.ple = PLE(edges, e_mean, d=PLE_D)                   # ★必须最先建(它消耗 RNG)
        self.block = DenoiseBlock()
        if seed is not None:
            seed_all(seed)
        self.gru = nn.GRU(len(FEAT) * PLE_D, D_EMB // 2, batch_first=True, bidirectional=True)

    def forward(self, x):                                        # (B,48,38) → (B,128)
        p = self.ple(self.block(x.float())).flatten(-2)          # (B,48,38*8)
        _, hn = self.gru(p)
        return torch.cat([hn[0], hn[1]], -1)                     # 双向末态拼接


class CSFront(nn.Module):
    """前置截面: 当日全票互看 → 修正**输入 x**。相似度基底来自一个冻结的模型(见三阶段说明)。

        b = 基底(N,128)  →  q,k → top-k16 softmax → A
        d = A@b − b                    ← 差分: A 均匀时 d_i = mean(b) − b_i, 截面上仍是变化的
        x_new = x + Wout(d)[:, None, :]   ← (N,1,38) 沿 48 根 bar 广播 = **整天电平平移**

    ⚠️ 基底先做**全局标量**标准化: 不做的话不同基底的注意力温度差两个数量级(实测 raw 的
       logits_std 0.038 ⇒ top-32 内几乎均匀)。必须是全局标量而非逐维 —— 逐维会改变基底的
       逐维加权, 引入没有依据的先验。
    ⚠️ 整段 f32: `A@b − b` 是差分, 而 top-k 挑的恰恰是最相似的邻居 ⇒ 相消损失打在要害上。
    ⚠️ top-k 是这条线唯一的正旋钮: 不截断时有效邻居 841/995(≈市场平均), 前 32 个权重只占 10%。"""

    def __init__(self, d_basis=D_EMB, n_feat=len(FEAT), d_a=CSF_DA, topk=CSF_TOPK):
        super().__init__()
        self.d_a, self.topk = d_a, topk
        self.Wq = nn.Linear(d_basis, d_a, bias=False)
        self.Wk = nn.Linear(d_basis, d_a, bias=False)
        self.Wout = nn.Linear(d_basis, n_feat, bias=False)
        nn.init.zeros_(self.Wout.weight)                         # ★零初始化 ⇒ 起点 = 无截面

    def forward(self, x, basis):                                 # x:(N,48,38)  basis:(N,128)
        with torch.autocast(x.device.type, enabled=False):
            b = basis.float()
            b = b / (b.std() + 1e-8)                             # ★全局标量, 抹平量级不碰结构
            q, k = self.Wq(b), self.Wk(b)
            lg = (q @ k.transpose(0, 1)) / (self.d_a ** 0.5)
            if self.topk < lg.shape[1]:
                kk = min(self.topk, lg.shape[1])
                lg = lg.masked_fill(lg < lg.topk(kk, -1).values[:, -1:], float("-inf"))
            A = lg.softmax(-1)
            bb = basis.float()
            return x.float() + self.Wout(A @ bb - bb)[:, None, :]


class DayAxis(nn.Module):
    """日间轴: 用过去 29 天的日嵌入修正今天的 raw。

        H (N,29,128) → 单向 GRU(h=16) → hn → raw + Wout(hn)

    ⚠️ **单向**(与主干的日内 GRU 相反): 末态对第 t 步的有效权重 ∝ ∏_{s>t} z_s ⇒ 近期权重大。
       在日内那是缺陷, 在日间正是想要的。双向实测没有增量。
    ⚠️ **不含今天**(29 天而非 30): 实测含不含读不出差别, 取点估计更高的那个。
    ⚠️ 出口零初始化, 且整段 f32 —— 训练在 autocast 下、验证不在, 不统一就会"在 bf16 的修正量上学、
       在 f32 的上验", 系统性差异且不报错。"""

    def __init__(self, d_emb=D_EMB, h=AX_H, n_days=D - 1):
        super().__init__()
        self.n_days = n_days
        self.gru = nn.GRU(d_emb, h, batch_first=True)
        self.Wout = nn.Linear(h, d_emb, bias=False)
        nn.init.zeros_(self.Wout.weight)                         # ★零初始化 ⇒ 起点 = 无日间轴

    def forward(self, raw, H):                                   # raw:(N,128)  H:(N,29,128)
        assert H.shape[1] == self.n_days, f"H 天数 {H.shape[1]} ≠ {self.n_days}"
        assert H.shape[0] == raw.shape[0], "H 与 raw 行数不符 —— 逐行对齐是硬前提"
        with torch.autocast(raw.device.type, enabled=False):
            _, hn = self.gru(H.float())
            return raw.float() + self.Wout(hn[-1])


class CSLayer(nn.Module):
    """后置截面: 在日间轴之后的 raw 上做当日全票互看(top-k16)。

        z = 逐日**截面**标准化(raw)         ← q/k 用它, 相似度不被大方差维主导
        A = softmax(top-k(Wq(z) @ Wk(z)ᵀ / √d_a))
        raw_new = raw + Wo(A @ Wv(raw))    ← v 用**原尺度**: ‖raw‖ 是"离市场多远", 正是要排序的

    ⚠️ 必须在 proj/LN **之前**: LN 是逐票跨特征的, 它把每只票的模长归一化, 而那个模长恰好是
       要排序的东西。
    ⚠️ z 走 f32: bf16 下大量票的 raw 相等 ⇒ z 也相等 ⇒ 截面上直接失去区分度。
    ⚠️ 池子逐日变(N 在 892~999 浮动)⇒ 机制必须**置换等变**: 无位置编码、无股票 id 嵌入。"""

    def __init__(self, d=D_EMB, d_a=CS_DA, topk=CS_TOPK):
        super().__init__()
        self.d_a, self.topk = d_a, topk
        self.Wq = nn.Linear(d, d_a, bias=False)
        self.Wk = nn.Linear(d, d_a, bias=False)
        self.Wv = nn.Linear(d, d_a, bias=False)
        self.Wo = nn.Linear(d_a, d, bias=False)
        nn.init.zeros_(self.Wo.weight)                           # ★零初始化 ⇒ 起点 = 无截面

    def forward(self, raw):                                      # (N,128) 当日全截面
        r = raw.float()
        z = (r - r.mean(0, keepdim=True)) / (r.std(0, keepdim=True) + 1e-6)
        q, k, v = self.Wq(z), self.Wk(z), self.Wv(r)
        lg = q @ k.transpose(0, 1) / (self.d_a ** 0.5)
        if self.topk < lg.shape[1]:
            kk = min(self.topk, lg.shape[1])
            lg = lg.masked_fill(lg < lg.topk(kk, -1).values[:, -1:], float("-inf"))
        return raw + self.Wo(lg.softmax(-1) @ v)


class Model(nn.Module):
    """整链。`use_csf` / `use_dayax` / `use_cs` 三个开关对应三个训练阶段。

    ⚠️ 建模顺序固定: trunk → 读出头 → csf → dayax → cs。前两者决定主干与读出头的初始化,
       后三个各自用 RNG 存档-还原包起来 ⇒ **不同阶段的主干逐位同初始化**, 阶段之间可比。"""

    def __init__(self, edges, e_mean, use_csf=False, use_dayax=False, use_cs=False, seed=SEED):
        super().__init__()
        self.use_csf, self.use_dayax, self.use_cs = use_csf, use_dayax, use_cs
        self.trunk = Trunk(edges, e_mean, seed=seed)
        self.proj = nn.Linear(D_EMB, D_EMB)
        self.norm = nn.LayerNorm(D_EMB)
        self.mlp = nn.Sequential(nn.Linear(D_EMB, 64), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(32, 1))
        for flag, name, ctor in ((use_csf, "csf", CSFront), (use_dayax, "dayax", DayAxis),
                                 (use_cs, "cs", CSLayer)):
            if flag:
                st = torch.get_rng_state()                       # ★存档: 新模块不许污染上面的初始化
                setattr(self, name, ctor())
                torch.set_rng_state(st)
            else:
                setattr(self, name, None)

    def encode(self, x, basis=None):
        """x(N,48,38) → raw(N,128)。有 csf 时先做跨票修正(需要当日**全截面**同时在场)。"""
        if self.csf is not None:
            assert basis is not None, "csf 需要相似度基底(冻结模型算的日嵌入)"
            x = self.csf(x, basis)
        return self.trunk(x)

    def score(self, raw, H=None):
        """raw(N,128) → 分数(N,)。日间轴与后置截面都在 proj/LN 之前。"""
        if self.dayax is not None:
            assert H is not None, "日间轴需要 H(过去 29 天的日嵌入)"
            raw = self.dayax(raw, H)
        if self.cs is not None:
            raw = self.cs(raw)
        return self.mlp(self.norm(self.proj(raw))).squeeze(-1)

    def forward(self, x, basis=None, H=None):
        return self.score(self.encode(x, basis), H)


# ##########################################################################
# ##########                      训 练                            ##########
# ##########################################################################

def soft_spearman_loss(pred, y, tau=SR_TAU):
    """1 − Spearman。pred 走软秩(带梯度), y 走硬秩(无梯度, 天然抗离群、不需要 winsorize)。"""
    x = pred.float()
    xd = (x - x.mean()) / (x.std() + 1e-4)                        # 先标准化, 让 tau 与幅度解耦
    sr = torch.sigmoid((xd[:, None] - xd[None, :]) / tau).sum(1)  # 软"有多少个比我小"
    tr = torch.argsort(torch.argsort(y.float())).float()
    sr = sr - sr.mean(); tr = tr - tr.mean()
    return 1.0 - (sr * tr).sum() / (sr.norm() * tr.norm() + 1e-12)


class EMA:
    """权重滑动平均。验证与最终权重都用 shadow —— 它比最后一步的权重稳得多。"""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def _param_groups(model, wd):
    """分组 weight decay。⚠️ 调制块的 in_proj 对角是 1.0 的大权重, 与 alpha/pos_emb/LN 一样
       **不该吃 WD** —— AdamW 的解耦衰减按相同比例作用, 会把那个恒等结构慢慢磨掉。
    ⚠️ `.attn.` 整个排除在免 WD 之外: 注意力的 QKV 应当正常吃 WD。"""
    NO_WD = (".alpha", ".pos_emb", ".type_emb", ".in_proj.", ".state_proj.", ".ln1.", ".ln2.")
    def no_wd(n):
        return (".attn." not in n) and any(k in n for k in NO_WD)
    a = [p for n, p in model.named_parameters() if p.requires_grad and no_wd(n)]
    b = [p for n, p in model.named_parameters() if p.requires_grad and not no_wd(n)]
    return [{"params": a, "weight_decay": 0.0}, {"params": b, "weight_decay": wd}]


def _add_noise(x, scale, gen=None, sigma=None):
    """加噪(0 新增参数, 实测 +0.00087)。只在训练时加 —— val 全在 no_grad 下 ⇒ 自动不加。

    ★perfeat(实验 NF05 用的就是这一档): sigma_f ∝ 该字段的截面 std、归一到 mean=1
      ⇒ **整体强度与 flat 完全相同**, 只改噪声在 38 个字段上的分配。
      (盘口 20 字段截面方差 1.05, 价格+量能 18 字段 0.58~0.65, 差 1.6 倍。)
      sigma=None 时退化为 flat。
    ⚠️ gen 必须是**独立** Generator: 用全局 RNG(randn_like)会让 mlp 两个 Dropout 的掩码
       序列从第一步就错开 ⇒ 消融的配对失效, SE 从 ~0.0002 涨到 ~0.0014。"""
    if scale <= 0 or not torch.is_grad_enabled():
        return x
    assert gen is not None, "❌ 加噪必须传独立 Generator(见 docstring)"
    e = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
    return x + scale * (e if sigma is None else e * sigma)


@torch.no_grad()
def evaluate(model, BLK, idx, y, basis=None, H_tab=None):
    """逐日真 IC / IC-IR。与训练走**同一条**前向 —— 分两份写就会出现"训练有、验证没有"。"""
    model.eval()
    ics = []
    for d in sorted(idx):
        yy = y[d]; m = torch.isfinite(yy)
        if int(m.sum()) <= MIN_STOCKS:
            continue
        ix = idx[d][m]
        rows = cur_blocks(ix)
        x = BLK[rows].float()
        b = None if basis is None else basis[rows].float()
        H = None if H_tab is None else _gather_H(H_tab, ix)
        pred = model(x, b, H).float()
        yv = yy[m]
        p = pred - pred.mean(); q = yv - yv.mean()
        ic = float((p * q).sum() / (p.norm() * q.norm() + 1e-12))
        if np.isfinite(ic):
            ics.append(ic)
    model.train()
    if not ics:
        return float("nan"), float("nan")
    a = np.array(ics)
    return float(a.mean()), float(a.mean() / (a.std() + 1e-12))


def _gather_H(H_tab, idx_rows):
    """取窗口的前 29 天(**不含今天**)→ (n, 29, 128)。"""
    sl = idx_rows[:, -D:-1]
    return H_tab[sl.reshape(-1)].float().reshape(idx_rows.shape[0], D - 1, -1)


def train_stage(tag, DATA, use_csf=False, use_dayax=False, use_cs=False,
                init_from=None, freeze=(), noise=0.0, noise_mode="perfeat",
                epochs=None, seed=SEED):
    """训一个阶段。返回 (model, ema_state, info)。

    ⚠️ freeze: 要冻住的参数前缀。**冻结的模块必须真的冻住** —— 分阶段变成联合训练是
       静默的错误(结果正常、不报错), 所以下面有硬检查。
    ⚠️ 逐日编码: csf 要"当日全票同时在场"且在主干**之前**, 不能像无截面时那样把 4000 块
       混在一起批量编码。10 天各自 forward、损失相加、**一次** backward。"""
    t0 = time.time()
    seed_all(seed)
    model = Model(DATA["edges"], DATA["e_mean"], use_csf, use_dayax, use_cs, seed=seed).to(DEVICE)
    if init_from is not None:
        miss, unexp = model.load_state_dict(init_from, strict=False)
        bad = [k for k in miss if not k.startswith(tuple(f"{m}." for m in ("csf", "dayax", "cs")))]
        assert not bad and not unexp, f"❌ {tag} 载入异常: 缺={sorted(bad)} 多={sorted(unexp)}"
    for n, p in model.named_parameters():
        if n.startswith(tuple(freeze)):
            p.requires_grad_(False)
    if freeze:                                                    # ★硬检查: 冻结真的生效了
        leak = [n for n, p in model.named_parameters() if n.startswith(tuple(freeze)) and p.requires_grad]
        assert not leak, f"❌ {tag} 的 {freeze} 没冻住: {leak[:4]}"
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

    BLK, idx_tr, y_tr = DATA["BLK_tr"], DATA["idx_tr"], DATA["y_tr"]
    basis_tr = DATA.get("basis_tr")
    H_tr = DATA.get("H_tr")
    days = sorted([d for d in idx_tr if idx_tr[d].shape[0] >= MIN_STOCKS])
    groups = [days[i:i + MB_DAYS] for i in range(0, len(days), MB_DAYS)]
    epochs = epochs or EPOCHS_MAX
    total = len(groups) * epochs
    warm = max(int(total * WARMUP_FRAC), 1)

    opt = torch.optim.AdamW(_param_groups(model, WEIGHT_DECAY), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    ema = EMA(model)
    # ★噪声流按成员错开: 集成要的是多样性, 让 K 个成员看到不同的噪声 realization。
    #   (实验里写死 12345 是为了让消融的臂之间**配对可比**, 那个诉求这里不存在。)
    gen = torch.Generator(device=DEVICE).manual_seed(12345 + int(seed) - int(SEEDS[0]))
    nsig = None
    if noise > 0 and noise_mode == "perfeat":            # ★perfeat: 标定逐字段 σ_f
        # ⚠️ W1 用 flat —— batch30-1 的 ipynb 里没有任何 noise_mode 键 ⇒ 全部臂走默认 flat;
        #    perfeat 只属于 batch26 的 NF05(即本代码的 N05 阶段)。
        with torch.no_grad():
            _g2 = torch.Generator().manual_seed(2024)    # 独立, 不碰全局 RNG(否则 dropout 序列错开)
            _ri = torch.randperm(BLK.shape[0], generator=_g2)[:min(4096, BLK.shape[0])]
            _sf = BLK[_ri].to(DEVICE).float().std(dim=(0, 1))            # (38,)
            nsig = (_sf / _sf.mean()).view(1, 1, -1)
        _log(f"    [噪声] perfeat σ_f/mean: min={float(nsig.min()):.3f} "
             f"max={float(nsig.max()):.3f}(整体强度与 flat 相同)")
    _log(f"\n{'='*84}\n【{tag}】参数 {n_all:,}(可训练 {n_tr:,})  最多 {epochs} 轮 × {len(groups)} 步  "
         f"早停 patience={PATIENCE}"
         + (f"  冻结={freeze}" if freeze else "")
         + (f"  噪声={noise}({noise_mode})" if noise else ""))

    best = dict(ic=-1e9, icir=float("nan"), ep=0, state=None)
    bad_ep, ep_secs = 0, []
    for ep in range(epochs):
        te = time.time()
        # ★★每轮打乱 minibatch 顺序(实验 probe6.py:1214 `model.train(); random.shuffle(groups)`)。
        #   不打乱的话 100 轮都按固定的时间序 2019-08→2023-06 走同一串 batch ——
        #   梯度噪声的模式逐轮重复, train_loss 压得更低而泛化更差。
        #   random.seed 已由 seed_all(SEED) 定过 ⇒ 顺序仍然逐位可复现。
        model.train(); random.shuffle(groups)
        tot, nstep = 0.0, 0
        for grp in groups:
            # ⚠️ cache_enabled=False 与实验一致: 本步是"10 天各自 forward、损失相加、**一次**
            #    backward", 缓存的 bf16 权重副本会跨这 10 次 forward 复用(且常驻显存)。
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(USE_AMP and DEVICE.type == "cuda"), cache_enabled=False):
                losses = []
                for d in grp:
                    ix = idx_tr[d]
                    rows = cur_blocks(ix)
                    x = _add_noise(BLK[rows].float(), noise, gen, nsig)
                    b = None if basis_tr is None else basis_tr[rows].float()
                    H = None if H_tr is None else _gather_H(H_tr, ix)
                    losses.append(soft_spearman_loss(model(x, b, H), y_tr[d]))
                loss = sum(losses) / len(losses)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step(); sched.step(); ema.update(model)
            tot += float(loss); nstep += 1
        t_tr = time.time() - te
        # 验证走 EMA shadow(比最后一步的权重稳)
        bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        ic, icir = evaluate(model, DATA["BLK_va"], DATA["idx_va"], DATA["y_va"],
                            DATA.get("basis_va"), DATA.get("H_va"))
        shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(bak)
        t_va = time.time() - te - t_tr
        ep_secs.append(time.time() - te)
        mark = ""
        if ic > best["ic"]:
            best.update(ic=ic, icir=icir, ep=ep + 1, state=shadow); bad_ep = 0; mark = " *best"
        else:
            bad_ep += 1
        _log(f"ep {ep+1:>3}  train_loss={tot/max(nstep,1):.6f}  val真IC={ic:.4f}  "
             f"val_ICIR={icir:.3f}  训{t_tr:.0f}s/验{t_va:.0f}s{mark}")
        if bad_ep >= PATIENCE:
            _log("early stop")
            break
    dt = time.time() - t0
    info = dict(tag=tag, val_ic=round(best["ic"], 5), val_icir=round(best["icir"], 4),
                best_ep=best["ep"], n_epoch=len(ep_secs), params=n_all, params_train=n_tr,
                seconds=round(dt, 1), sec_per_epoch=round(float(np.mean(ep_secs)), 2))
    # ★★每轮耗时要记: 私榜是**从零重训**而 GPU 单次上限 6h, 靠这个数外推
    _log(f"【{tag}】val真IC={info['val_ic']:.5f}  IC-IR={info['val_icir']:.4f}  "
         f"@ep{info['best_ep']}  共 {dt/60:.1f} min  ★★完整一轮 {info['sec_per_epoch']:.1f}s")
    return model, best["state"], info


@torch.no_grad()
def precompute_emb(state, DATA, tag="", splits=("tr", "va")):
    """用一个冻结的模型算全部块的日嵌入(N,128)。给 csf 当基底 / 给日间轴当 H。
    ⚠️ 只用 trunk(不含 csf)⇒ 可以批量算, 覆盖**全部**块;逐日算只能覆盖"当天的块",
       而日间轴取的是窗口全部 30 列, 某天不在样本集里就会留一行 0。"""
    t0 = time.time()
    m = Model(DATA["edges"], DATA["e_mean"], seed=SEED).to(DEVICE)
    sd = {k: v for k, v in state.items() if not k.startswith(("csf.", "dayax.", "cs."))}
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not [k for k in miss if k.startswith("trunk.")], f"❌ trunk 权重没载全: {miss[:4]}"
    m.eval()
    out = {}
    for sp in splits:                                   # 推理只有一份块 ⇒ splits=("tr",)
        BLK = DATA[f"BLK_{sp}"]
        n = BLK.shape[0]
        buf = torch.empty((n, D_EMB), dtype=torch.float32, device=DEVICE)
        step = 4000
        for i in range(0, n, step):
            buf[i:i + step] = m.trunk(BLK[i:i + step].float())
        out[sp] = buf
    del m; gc.collect(); torch.cuda.empty_cache()
    _log(f"[{tag}] 日嵌入就绪 " + " ".join(f"{sp}{tuple(out[sp].shape)}" for sp in splits)
         + f"  +{time.time()-t0:.0f}s")
    return tuple(out[sp] for sp in splits)


# ##########################################################################
# ##########                  权重 JSON(平台要求文本格式)          ##########
# ##########################################################################

def _enc(o):
    """★数组走 base64(而不是十进制文本): K=3 的权重共约 190 万个数,
       `tolist()` 存成文本要 ~29MB 且 json.load 要构造 190 万个 Python float(几十秒);
       base64 只要 ~10MB、解析是毫秒级, 而且直接存二进制**无精度损失**。
       文件仍是标准 json —— 只是 data 字段从数字数组变成一个字符串。"""
    if isinstance(o, torch.Tensor):
        a = np.ascontiguousarray(o.detach().cpu().numpy())
        return {"__t__": 1, "dtype": str(a.dtype), "shape": list(a.shape),
                "b64": base64.b64encode(a.tobytes()).decode("ascii")}
    if isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        return {"__a__": 1, "dtype": str(a.dtype), "shape": list(a.shape),
                "b64": base64.b64encode(a.tobytes()).decode("ascii")}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_enc(v) for v in o]
    return o


def _buf(o):
    """兼容两种载荷: base64(本版)与 data 列表(batch1 存的旧权重)。"""
    if "b64" in o:
        return np.frombuffer(base64.b64decode(o["b64"]), dtype=np.dtype(o["dtype"]))
    return np.asarray(o["data"], dtype=np.dtype(o["dtype"]))


def _dec(o, device="cpu"):
    if isinstance(o, dict):
        if o.get("__t__"):
            return torch.from_numpy(_buf(o).reshape(o["shape"]).copy()).to(device)
        if o.get("__a__"):
            return _buf(o).reshape(o["shape"]).copy()
        return {k: _dec(v, device) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(v, device) for v in o]
    return o


def save_model(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_enc(payload), f)
    mb = os.path.getsize(path) / 2**20
    # K=3 时约 197 万个数 ⇒ base64 约 10.6 MB。上限 50MB, 余量 4.7 倍。
    # (若改回十进制文本存法会涨到约 36MB, 且 json.load 要几十秒 —— 见 _enc 的注释。)
    _log(f"[save] {path}  {mb:.1f} MB"
         + (f"  ⚠️⚠️ 超过 {MAX_JSON_MB}MB 上限, 提交会被拒!" if mb > MAX_JSON_MB
            else f"  ✅ 上限 {MAX_JSON_MB}MB"))


def load_model(path, device="cpu"):
    with open(path, "r", encoding="utf-8") as f:
        return _dec(json.load(f), device)


# ##########################################################################
# ##########                  数 据 准 备                          ##########
# ##########################################################################

def prepare_train_data(datasources):
    """训练用: 训练分区 + 验证分区。返回的 DATA 直接喂给 train_stage。"""
    table = datasources[TABLE_KEY]
    t0 = time.time()
    lo_all = (pd.Timestamp(TRAIN_START) - pd.Timedelta(days=BUF_DAYS)).strftime("%Y-%m-%d")
    hi_all = (pd.Timestamp(TRAIN_END) + pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    cal = trading_days(lo_all, hi_all)
    cal_pos = {d: i for i, d in enumerate(cal)}
    pool = pit_pool(lo_all, hi_all)
    _log(f"[prep] 交易日 {len(cal)} ({str(cal[0])[:10]} ~ {str(cal[-1])[:10]})  "
         f"PIT 池 {len(pool):,} 条")

    # ★★区间自动划分: 只给 TRAIN_START/TRAIN_END, 末尾 VAL_FRAC 当验证集, 中间空 GAP_DAYS
    tr_lo, tr_hi, va_lo, va_hi = split_fit(cal)
    _n_all = sum(1 for d in cal if np.datetime64(TRAIN_START) <= d <= np.datetime64(TRAIN_END))
    _n_tr = sum(1 for d in cal if np.datetime64(tr_lo) <= d <= np.datetime64(tr_hi))
    _n_va = sum(1 for d in cal if np.datetime64(va_lo) <= d <= np.datetime64(va_hi))
    _log(f"[split] {TRAIN_START}~{TRAIN_END} 共 {_n_all} 个交易日 ⇒ "
         f"训练 {tr_lo}~{tr_hi} ({_n_tr}天, {_n_tr/_n_all:.1%})  "
         f"空 {_n_all-_n_tr-_n_va} 天  验证 {va_lo}~{va_hi} ({_n_va}天, {_n_va/_n_all:.1%})")
    assert _n_all - _n_tr - _n_va == GAP_DAYS, \
        f"❌ 空档 {_n_all-_n_tr-_n_va} 天 ≠ GAP_DAYS={GAP_DAYS} —— 划分算错了"

    st_tr = load_blocks(table, lo_all,
                        (pd.Timestamp(tr_hi) + pd.Timedelta(days=12)).strftime("%Y-%m-%d"),
                        pool, cal_pos, "train")
    bycat_fill(st_tr)
    stats = fit_bycat(st_tr)
    _log("[bycat] 常数指纹(流式量应跨环境逐位一致; 分位量允许 <1% 的采样差):")
    _log(f"  mu_pc={stats['mu_pc']:.6f} sd_pc={stats['sd_pc']:.6f} "
         f"lo_pc={stats['lo_pc']:.6f} hi_pc={stats['hi_pc']:.6f}")
    _log(f"  wlo={stats['wlo']:.6f} whi={stats['whi']:.6f} sig_struct={stats['sig_struct']:.6f}")
    _log(f"  mu_v3={np.round(np.asarray(stats['mu_v3'], np.float64), 6).tolist()} "
         f"sd_v3={np.round(np.asarray(stats['sd_v3'], np.float64), 6).tolist()}")
    _log(f"  mu_ov={stats['mu_ov']:.6f} sd_ov={stats['sd_ov']:.6f} "
         f"mu_on={stats['mu_on']:.6f} sd_on={stats['sd_on']:.6f}")                       # ★统计量只在训练集上算(规则 A20)
    normalize_bycat(st_tr, stats)
    gc.collect()

    v_lo = (pd.Timestamp(va_lo) - pd.Timedelta(days=BUF_DAYS)).strftime("%Y-%m-%d")
    v_hi = (pd.Timestamp(va_hi) + pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    st_va = load_blocks(table, v_lo, v_hi, pool, cal_pos, "val")
    bycat_fill(st_va)
    normalize_bycat(st_va, stats)                  # 填充是本地操作; z 参数用训练集的
    gc.collect()

    g_tr = build_samples(st_tr, cal, tr_lo, tr_hi, need_label=True)
    # ★need_label=False: 验证集**末尾两天**必然没有 label(要用 t+1/t+2 的开盘价, 而数据到此为止)。
    #   保留这些样本, 由 evaluate 的 isfinite 掩码把它们整天跳过 —— 与实验一致。
    #   若这里用 True, 那两天会在**建样本阶段**就被丢掉, 结果相同但天数对不上, 不便于比对。
    g_va = build_samples(st_va, cal, va_lo, va_hi, need_label=False)
    BLK_tr, idx_tr, y_tr, _ = to_tensors(st_tr, g_tr)
    BLK_va, idx_va, y_va, _ = to_tensors(st_va, g_va)
    del st_tr, st_va, g_tr, g_va; gc.collect()

    edges, e_mean = fit_ple_edges(BLK_tr)
    n_tr = sum(v.shape[0] for v in y_tr.values()); n_va = sum(v.shape[0] for v in y_va.values())
    _log(f"[prep] 样本 train {n_tr:,} ({len(idx_tr)}天) / val {n_va:,} ({len(idx_va)}天)  "
         f"总耗时 {time.time()-t0:.0f}s")
    _nl = sum(int(torch.isfinite(v).sum()) for v in y_va.values())
    _log(f"[prep] val 里有 label 的样本 {_nl:,}/{n_va:,}"
         f"(末尾 2 天无 label 属正常 —— label 要 t+1/t+2 的开盘价)")
    return dict(BLK_tr=BLK_tr, idx_tr=idx_tr, y_tr=y_tr,
                BLK_va=BLK_va, idx_va=idx_va, y_va=y_va,
                stats=stats, edges=edges, e_mean=e_mean,
                split=dict(train_start=tr_lo, train_end=tr_hi,
                           val_start=va_lo, val_end=va_hi,
                           n_train_days=_n_tr, n_val_days=_n_va))


def prepare_infer_data(datasources, start_date, end_date, stats):
    """推理用: 只建测试区间。⚠️ 往前多读 BUF_DAYS 凑 D 天回看(规则 A23: 评估集自带 warm-up)。"""
    table = datasources[TABLE_KEY]
    t0 = time.time()
    sd = pd.Timestamp(str(start_date)).strftime("%Y-%m-%d")
    ed = pd.Timestamp(str(end_date)).strftime("%Y-%m-%d")
    lo = (pd.Timestamp(sd) - pd.Timedelta(days=BUF_DAYS)).strftime("%Y-%m-%d")
    cal = trading_days(lo, ed)
    assert len(cal) > D, f"❌ 只取到 {len(cal)} 个交易日, 不足回看窗口 D={D}"
    cal_pos = {d: i for i, d in enumerate(cal)}
    pool = pit_pool(lo, ed)
    st = load_blocks(table, lo, ed, pool, cal_pos, "infer")
    bycat_fill(st)
    normalize_bycat(st, stats)                     # ★z 参数用**训练集**存下来的
    g = build_samples(st, cal, sd, ed, need_label=False)
    BLK, idx, y, meta = to_tensors(st, g)
    del st, g; gc.collect()
    _log(f"[infer] 样本 {sum(v.shape[0] for v in idx.values()):,} / {len(idx)} 天  "
         f"+{time.time()-t0:.0f}s")
    return BLK, idx, meta


# ##########################################################################
# ##########                  三 阶 段 训 练                        ##########
# ##########################################################################

@torch.no_grad()
def member_scores(state, BLK, idx, basis, H_tab, edges, e_mean, seed, y=None):
    """一个成员的**逐日截面 z-score**。→ {日期: (n,) ndarray}

    ⚠️ z-score 而不是原始分数: 损失是 soft-Spearman(排序学习), 分数的绝对尺度没有意义,
       不同种子的尺度可能差很多, 直接平均会让尺度大的成员主导整个集成。
    ⚠️ y 给定时只对**有 label 的票**打分 —— 与 evaluate 同口径, 这样集成 IC 才能和
       单成员的 val_ic 直接比。推理时不传 y(全票都要出分)。"""
    m = Model(edges, e_mean, use_csf=True, use_dayax=True, use_cs=True, seed=seed).to(DEVICE)
    miss, unexp = m.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"❌ 成员权重载入异常: 缺={sorted(miss)} 多={sorted(unexp)}"
    m.eval()
    out = {}
    for d in sorted(idx):
        ix = idx[d]
        if y is not None:
            msk = torch.isfinite(y[d])
            if int(msk.sum()) <= MIN_STOCKS:
                continue
            ix = ix[msk]
        s = m(BLK[cur_blocks(ix)].float(), basis[cur_blocks(ix)].float(),
              _gather_H(H_tab, ix)).float()
        out[d] = ((s - s.mean()) / (s.std() + 1e-8)).cpu().numpy()
    del m; gc.collect(); torch.cuda.empty_cache()
    return out


def eval_ensemble(zs, y, keep):
    """等权集成在验证集上的 IC / IC-IR / 成员间平均相关 ρ̄。

    ρ̄ 是判断集成值不值的关键量: IC_ens ≈ mean(IC_k)·√(K/(1+(K−1)ρ̄)) ——
    ρ̄ 越低去相关收益越大。ρ̄→1 时集成退化成单模型。"""
    ics, rhos = [], []
    K = len(keep)
    for d in sorted(y):
        if d not in zs[keep[0]]:
            continue                                   # 该日被 MIN_STOCKS 跳过
        msk = torch.isfinite(y[d])
        Z = np.stack([zs[k][d] for k in keep])         # (K, n)
        if K > 1:
            C = np.corrcoef(Z)
            rhos.append((C.sum() - K) / (K * (K - 1)))
        p = Z.mean(0)
        q = y[d][msk].cpu().numpy().astype(np.float64)
        p = p - p.mean(); q = q - q.mean()
        ic = float((p * q).sum() / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-12))
        if np.isfinite(ic):
            ics.append(ic)
    a = np.asarray(ics)
    return (float(a.mean()), float(a.mean() / (a.std() + 1e-12)), float(a.std()),
            float(np.mean(rhos)) if rhos else float("nan"))


def train_and_save(datasources, model_path=MODEL_PATH):
    """K 个种子各自跑完整三阶段 → 崩溃剔除 → 存 JSON。返回 meta。"""
    T0 = time.time()
    torch.set_num_threads(NUM_THREADS)
    DATA = prepare_train_data(datasources)

    members, infos, z_va = [], [], []
    for k, sd in enumerate(SEEDS):
        _log(f"\n{'#'*84}\n###  成员 {k+1}/{len(SEEDS)}  seed={sd}  "
             f"(已用时 {(time.time()-T0)/60:.0f} min)\n{'#'*84}")

        # ---- 阶段0: 主干 + 加噪, 从零训 → 它的 raw 当 csf 的相似度基底 ----
        _, s0, i0 = train_stage(f"N05-s{sd}", DATA, noise=NOISE_SCRATCH, seed=sd)
        DATA["basis_tr"], DATA["basis_va"] = precompute_emb(s0, DATA, f"N05-s{sd}")

        # ---- 阶段1: 主干 + csf, 从零训 → 它的 raw 当日间轴的 H ----
        # ★noise_mode="flat": 与实验 batch30-1 一致(那批的臂 cfg 里没有 noise_mode ⇒ 默认 flat;
        #   perfeat 只在 batch26 训基底 N05 时用过)。
        _, s1, i1 = train_stage(f"W1-s{sd}", DATA, use_csf=True, noise=NOISE_SCRATCH,
                                noise_mode="flat", seed=sd)
        DATA["H_tr"], DATA["H_va"] = precompute_emb(s1, DATA, f"W1-s{sd}")

        # ---- 阶段2: 冻住 W1(主干 + csf), 日间轴 + 后置截面**同步**训 ----
        _, s2, i2 = train_stage(f"W3b-s{sd}", DATA, use_csf=True, use_dayax=True, use_cs=True,
                                init_from=s1, freeze=("trunk.", "csf."), seed=sd)

        # ★成员的验证集 z-score 当场算好: basis_va / H_va 还在, 零额外开销
        z_va.append(member_scores(s2, DATA["BLK_va"], DATA["idx_va"],
                                  DATA["basis_va"], DATA["H_va"],
                                  DATA["edges"], DATA["e_mean"], sd, y=DATA["y_va"]))
        members.append(dict(seed=sd, n05=s0, w1=s1, w3b=s2))
        infos.append(dict(seed=sd, n05=i0, w1=i1, w3b=i2))
        _log(f"###  成员 {k+1} 完成: N05={i0['val_ic']:.5f} → W1={i1['val_ic']:.5f} "
             f"→ W3b={i2['val_ic']:.5f}")

        # ★两张表下个成员要重算 —— 不清掉白占约 1.4 GB 显存
        for _k in ("basis_tr", "basis_va", "H_tr", "H_va"):
            DATA.pop(_k, None)
        gc.collect(); torch.cuda.empty_cache()

    # ---- 崩溃剔除(只防"崩", 不做精调; 见 DROP_RATIO 处的推导)----
    ics = [i["w3b"]["val_ic"] for i in infos]
    med = float(np.median(ics))
    keep = [k for k, v in enumerate(ics) if v >= DROP_RATIO * med]
    assert keep, f"❌ 全部成员都被判崩溃, IC={[round(v,5) for v in ics]}"
    if len(keep) < len(ics):
        _log(f"\n⚠️ 剔除 {len(ics)-len(keep)} 个崩溃成员(IC < {DROP_RATIO}×中位数 {med:.5f}): "
             + ", ".join(f"seed{SEEDS[k]}={ics[k]:.5f}" for k in range(len(ics)) if k not in keep))

    # ---- 集成评估 ----
    e_ic, e_icir, e_std, rho = eval_ensemble(z_va, DATA["y_va"], keep)
    best_single = max(ics[k] for k in keep)
    _log(f"\n{'='*84}\n★集成({len(keep)} 个成员, 逐日截面 z-score 等权平均)")
    for k in keep:
        _log(f"     seed{SEEDS[k]}  IC={ics[k]:.5f}  IC-IR={infos[k]['w3b']['val_icir']:.4f}")
    _log(f"     ─────────────────────────────────────────────")
    _log(f"     集成    IC={e_ic:.5f}  IC-IR={e_icir:.4f}  日间波动={e_std:.5f}")
    _log(f"     单模型最好 IC={best_single:.5f}  ⇒ 集成增量 {e_ic-best_single:+.5f}")
    _log(f"     成员间平均相关 ρ̄={rho:.4f}  "
         f"(理论去相关增益 ×{(len(keep)/(1+(len(keep)-1)*rho))**0.5:.3f})")
    if e_ic < best_single:
        _log("     ⚠️ 集成不如最好的单成员 —— ρ̄ 太高(成员太相似)或有成员偏弱")

    meta = dict(data_range=[TRAIN_START, TRAIN_END], split=DATA["split"],
                val_frac=VAL_FRAC, gap_days=GAP_DAYS,
                seeds=[SEEDS[k] for k in keep], seeds_all=list(SEEDS),
                nb=NB, D=D, feat=FEAT,
                member_ic=[round(ics[k], 5) for k in keep],
                ens_ic=round(e_ic, 5), ens_icir=round(e_icir, 4), ens_rho=round(rho, 4),
                stage={f"{n}-s{infos[k]['seed']}": infos[k][n]
                       for k in keep for n in ("n05", "w1", "w3b")},
                minutes_total=round((time.time() - T0) / 60, 1))
    save_model(model_path, dict(
        meta=meta, stats=DATA["stats"], edges=DATA["edges"], e_mean=DATA["e_mean"],
        # ★推理要现算两张表: csf 的基底(N05 算)、日间轴的 H(W1 算) —— 存表太大(1e6×128 ≈ 500MB),
        #   所以存这两个阶段的**权重**, 推理时用它们在测试区间上重算。
        members=[members[k] for k in keep]))

    # ---- 时间账(私榜从零重训按这个外推)----
    dt_h = (time.time() - T0) / 3600
    _log(f"\n{'='*84}\n★全部完成, 总用时 {dt_h*60:.1f} min")
    _log(f"  可训练参数 {infos[0]['w3b']['params']:,}(规则 A6: 10万 ~ 1亿)")
    _log(f"\n★★完整一轮耗时(私榜从零重训按这个外推, GPU 单次上限 {GPU_LIMIT_H}h):")
    for k in range(len(SEEDS)):
        for n in ("n05", "w1", "w3b"):
            i = infos[k][n]
            _log(f"     s{SEEDS[k]} {n:<4} {i['sec_per_epoch']:>6.1f}s/轮 × {i['n_epoch']:>3} 轮"
                 f"(best@ep{i['best_ep']}) = {i['seconds']/60:>5.1f} min")
    _log(f"     ⇒ {len(SEEDS)} 个成员合计 {dt_h:.2f} h"
         + (f"  ⚠️ 超过 {GPU_LIMIT_H}h 上限!" if dt_h > GPU_LIMIT_H else f"  ✅ 在 {GPU_LIMIT_H}h 内"))
    return meta


# ##########################################################################
# ##########                      推 理                            ##########
# ##########################################################################

@torch.no_grad()
def predict(datasources, start_date, end_date, model_path=MODEL_PATH):
    """加载 JSON 权重, 在样本外测试区间打分 → ['date','instrument','score']。**不训练**。"""
    import dai
    torch.set_num_threads(NUM_THREADS)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到 {model_path};请先跑 train_and_save 并随 notebook 一起上传")
    ck = load_model(model_path)
    meta = ck["meta"]
    edges, e_mean = np.asarray(ck["edges"]), np.asarray(ck["e_mean"])
    mems = ck["members"]
    sp = meta["split"]
    _log(f"[infer] 已加载 {model_path}  训练 {sp['train_start']}~{sp['train_end']}  "
         f"验证 {sp['val_start']}~{sp['val_end']}")
    _log(f"[infer] {len(mems)} 个成员 seeds={meta['seeds']}  "
         f"集成 val IC={meta['ens_ic']} IC-IR={meta['ens_icir']} ρ̄={meta['ens_rho']}")

    BLK, idx, meta_map = prepare_infer_data(datasources, start_date, end_date, ck["stats"])

    # ---- K 个成员各自打分 → 逐日截面 z-score → 等权平均 ----
    # ⚠️ 推理**不传 y**: 全票都要出分(规则 A22 要完整面板), 而 label 在测试区间根本不存在。
    #    这也是验证集"末尾两天没 label"那个问题在推理侧不存在的原因 —— 打分只用过去 D 天的
    #    窗口, 与未来价格无关。
    acc, t0 = {}, time.time()
    for mem in mems:
        D2 = dict(BLK_tr=BLK, edges=edges, e_mean=e_mean)
        (basis,) = precompute_emb(mem["n05"], D2, f"N05-s{mem['seed']}", splits=("tr",))
        (H_tab,) = precompute_emb(mem["w1"], D2, f"W1-s{mem['seed']}", splits=("tr",))
        z = member_scores(mem["w3b"], BLK, idx, basis, H_tab, edges, e_mean, mem["seed"])
        for d, v in z.items():
            acc[d] = v if d not in acc else acc[d] + v
        del basis, H_tab, z; gc.collect(); torch.cuda.empty_cache()

    rows = []
    for d in sorted(idx):
        s = acc[d] / len(mems)
        assert len(s) == len(meta_map[d]), \
            f"❌ {d} 的分数 {len(s)} 行 ≠ 样本 {len(meta_map[d])} 行 —— 与票的对应错位了"
        for (code, _li), sc in zip(meta_map[d], s):
            rows.append((pd.Timestamp(d), code, float(sc)))
    _log(f"[infer] {len(mems)} 个成员打分完成 {len(rows):,} 行  +{time.time()-t0:.0f}s")

    # ---- 对齐 PIT 全票池 + 补全面板(规则 A22 / A16)----
    stk = dai.query(f"SELECT date, instrument FROM {INSTRU_TABLE}",
                    filters={"date": [str(start_date), str(end_date)]}).df()
    stk["date"] = pd.to_datetime(stk["date"]).dt.normalize()
    stk = stk.drop_duplicates(["date", "instrument"])
    res = stk.merge(pd.DataFrame(rows, columns=["date", "instrument", "score"]),
                    on=["date", "instrument"], how="left")
    res["score"] = res["score"].replace([np.inf, -np.inf], np.nan)
    miss_rate = res.groupby("date")["score"].apply(lambda x: x.isna().mean())
    # ★缺失填**当日截面均值** = 中性分: 不改当日排序, 但把面板补满
    #   (评分前会强制 merge 全票池, 散点 NaN 会报错;整日全 NaN 也不允许)
    res["score"] = res.groupby("date")["score"].transform(lambda x: x.fillna(x.mean()))
    res["score"] = res["score"].fillna(0.0).astype(np.float64)
    res = res[["date", "instrument", "score"]].reset_index(drop=True)
    over = miss_rate[miss_rate > 0.4]
    _log(f"[infer] 输出 {len(res):,} 行 / {res['date'].nunique()} 天 / "
         f"{res['instrument'].nunique()} 票  最大单日缺失 {miss_rate.max():.1%}  "
         f"超40%的天数 {len(over)}")
    if len(over):
        _log(f"   ⚠️ 这些交易日缺失率超 40%(规则 A16 红线): {[str(x)[:10] for x in over.index[:5]]}")
    return res
