# %% [markdown]
# # S06 · 订单流毒性冲击 `VPIN-3`（传统赛道）
#
# **【经济逻辑】** Easley, López de Prado & O'Hara (2012) 的 **VPIN**（Volume-Synchronized
# Probability of Informed Trading）：把成交量切成桶，用 **BVC（Bulk Volume Classification）**
# 把每桶成交量按标准化价格变动分成主动买/主动卖，桶内买卖失衡的平均绝对值即"订单流毒性"。
# 毒性高 ⇒ 逆向选择风险高 ⇒ 流动性提供者撤退、要求更高即时性补偿 ⇒ 价格被过度压低 ⇒ 次日修复。
#
# **为什么正交**：VPIN 是**无符号**的失衡强度，与所有"有符号"的订单流/动量/反转因子在构造上正交；
# 又因为进一步只取相对个股自身 20 日基准的**冲击部分**，规模/换手/波动率风格在时序内部已被剔除，
# 能在平台 BARRA 残差化之后存活。
#
# **【统计方法】** BVC 用 logistic 近似正态 CDF：`buy_share = 1 / (1 + exp(-1.702 · r/σ))`
# （σ 为当日分钟收益标准差，`r/σ` 截断在 ±8 防溢出）；桶 = 连续 `BUCKET_MIN` 分钟；
# 日度 `VPIN = Σ|2B − V| / ΣV`；再做 20 日 trailing z-score → 截面 rank → 平滑 → 等权。
#
# ⚠️ **三个子因子的方向均需实证确认**（默认 `+1`，依据"流动性冲击后修复"）。
# 请先跑 `research/R02_platform_eval.py`，用输出的 Rank IC 符号覆盖下面的 `MEMBERS`。

# %%
LOOKBACK_CAL_DAYS = 50
BASE_WIN = 20
SMOOTH_WIN = 5
BUCKET_MIN = 10             # 桶大小（分钟）。用时间桶近似成交量桶，1 分钟数据下的标准做法
MIN_BARS = 60
MAX_LEVELS = 1

MEMBERS = [
    ("vpin_innov",      +1.0),   # 全天毒性冲击
    ("vpin_tail_innov", +1.0),   # 尾盘 30 分钟毒性冲击
    ("vpin_trend",      +1.0),   # 下午毒性 − 上午毒性
]


def _probe(bar1m, start_date):
    import pandas as pd
    import dai

    s = pd.to_datetime(start_date)
    head = dai.query(f"SELECT * FROM {bar1m} LIMIT 5",
                     filters={"date": [str(s), str(s + pd.Timedelta(days=12))]}).df()
    cols = set(head.columns)
    px = "close" if "close" in cols else "price"
    cum = False
    try:
        inst = str(head["instrument"].iloc[0])
        day = pd.to_datetime(head["date"].iloc[0]).date()
        one = dai.query(f"SELECT date, volume FROM {bar1m} WHERE instrument='{inst}' "
                        f"AND CAST(date AS DATE)=DATE '{day}' ORDER BY date",
                        filters={"date": [f"{day} 00:00:00", f"{day} 23:59:59"]}).df()
        cum = bool(len(one) > 20 and one["volume"].is_monotonic_increasing)
    except Exception:
        pass
    print(f"[probe] 价格列={px} volume累计={cum}")
    return {"px": px, "cum": cum}


def _build_sql(bar1m, cfg):
    px, cum = cfg["px"], cfg["cum"]
    if cum:
        vol_expr = "GREATEST(volume - COALESCE(LAG(volume) OVER w0, 0), 0)"
        win0 = "\n        WINDOW w0 AS (PARTITION BY instrument, dt ORDER BY date)"
    else:
        vol_expr, win0 = "GREATEST(volume, 0)", ""

    return f"""
    WITH b0 AS (
        SELECT date, instrument, CAST(date AS DATE) AS dt, CAST(date AS TIME) AS t,
               {px} AS px, volume
        FROM {bar1m}
        WHERE CAST(date AS TIME) >= TIME '09:30:00' AND CAST(date AS TIME) <= TIME '15:00:00'
    ),
    b1 AS (SELECT * EXCLUDE (volume), {vol_expr} AS vol FROM b0{win0}),
    b2 AS (
        SELECT *,
               ROW_NUMBER() OVER w AS mn,
               LAG(px) OVER w AS px_1
        FROM b1
        WINDOW w AS (PARTITION BY instrument, dt ORDER BY date)
    ),
    b3 AS (
        SELECT instrument, dt, t, mn, vol,
               CASE WHEN px > 0 AND px_1 > 0 THEN LN(px / px_1) END AS ret
        FROM b2
    ),
    b4 AS (
        SELECT *, stddev_samp(ret) OVER (PARTITION BY instrument, dt) AS sd
        FROM b3
    ),
    -- BVC：把每分钟成交量按标准化收益分成主动买 / 主动卖
    b5 AS (
        SELECT instrument, dt, t, mn, vol,
               vol / (1.0 + EXP(-1.702 * LEAST(GREATEST(ret / NULLIF(sd, 0), -8.0), 8.0))) AS buy
        FROM b4
        WHERE ret IS NOT NULL AND sd > 0 AND vol > 0
    ),
    -- 按 BUCKET_MIN 分钟成桶，统计桶内买卖失衡
    bk AS (
        SELECT instrument, dt,
               CAST(FLOOR((mn - 1) / {BUCKET_MIN}) AS INTEGER) AS bkt,
               MIN(t)      AS t0,
               MAX(mn)     AS mn_max,
               SUM(vol)    AS v,
               SUM(buy)    AS b
        FROM b5
        GROUP BY instrument, dt, CAST(FLOOR((mn - 1) / {BUCKET_MIN}) AS INTEGER)
    ),
    bk2 AS (
        SELECT *,
               ABS(2.0 * b - v)               AS imb,
               MAX(mn_max) OVER (PARTITION BY instrument, dt) AS mn_day
        FROM bk
    )
    SELECT instrument, dt,
           SUM(v)                                                       AS dvol,
           MAX(mn_day)                                                  AS nbar,
           SUM(imb) / NULLIF(SUM(v), 0)                                 AS vpin,
           SUM(CASE WHEN t0 >= TIME '14:30:00' THEN imb END)
             / NULLIF(SUM(CASE WHEN t0 >= TIME '14:30:00' THEN v END), 0) AS vpin_tail,
           SUM(CASE WHEN t0 >= TIME '13:00:00' THEN imb END)
             / NULLIF(SUM(CASE WHEN t0 >= TIME '13:00:00' THEN v END), 0) AS vpin_pm,
           SUM(CASE WHEN t0 <  TIME '11:30:00' THEN imb END)
             / NULLIF(SUM(CASE WHEN t0 <  TIME '11:30:00' THEN v END), 0) AS vpin_am
    FROM bk2
    GROUP BY instrument, dt
    HAVING MAX(mn_day) >= {MIN_BARS}
    """


def _nanmean(frames):
    import numpy as np
    s = n = None
    for f in frames:
        v, m = f.fillna(0.0), f.notna().astype("float32")
        s = v if s is None else s.add(v, fill_value=0.0)
        n = m if n is None else n.add(m, fill_value=0.0)
    return s / n.replace(0.0, np.nan), n


def _ts_innov(piv, win):
    mu = piv.rolling(win, min_periods=max(5, win // 3)).mean().shift(1)
    sd = piv.rolling(win, min_periods=max(5, win // 3)).std().shift(1)
    return (piv - mu) / sd.replace(0.0, float("nan"))


def _finalize(df, start_date, end_date, name="factor"):
    import numpy as np
    import pandas as pd
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["factor"] = pd.to_numeric(df["factor"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "instrument", "factor"])
    df = df[(df["date"] >= pd.to_datetime(start_date).normalize())
            & (df["date"] <= pd.to_datetime(end_date).normalize())]
    df = df[["date", "instrument", "factor"]].reset_index(drop=True)
    df["instrument"] = df["instrument"].astype(str)
    c = df.groupby("date")["factor"].size()
    print(f"[{name}] 行数={len(df)} 交易日={c.size} 日均标的={c.mean():.0f} 最少={c.min()}")
    return df


# %%
def main(datasources, start_date, end_date):
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    q_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_CAL_DAYS)).strftime("%Y-%m-%d 00:00:00")

    cfg = _probe(bar1m, start_date)
    raw = dai.query(_build_sql(bar1m, cfg),
                    filters={"date": [q_start, end_date]}, compression=True).df()
    raw["dt"] = pd.to_datetime(raw["dt"]).dt.normalize()
    raw["instrument"] = raw["instrument"].astype(str)
    raw = raw.drop_duplicates(subset=["dt", "instrument"])
    print(f"[sql] {raw.shape}")

    pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",
                     filters={"date": [q_start, end_date]}).df()
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    raw = raw.merge(pool.rename(columns={"date": "dt"}), how="inner", on=["dt", "instrument"])

    P = {c: raw.pivot(index="dt", columns="instrument", values=c).sort_index()
         for c in ("vpin", "vpin_tail", "vpin_pm", "vpin_am")}
    built = {
        "vpin_innov": _ts_innov(P["vpin"], BASE_WIN),
        "vpin_tail_innov": _ts_innov(P["vpin_tail"], BASE_WIN),
        "vpin_trend": P["vpin_pm"] - P["vpin_am"],
    }

    frames = []
    for col, sgn in MEMBERS:
        rk = (sgn * built[col]).rank(axis=1, pct=True) - 0.5
        if rk.notna().sum().sum() == 0:
            continue
        frames.append(rk.rolling(SMOOTH_WIN, min_periods=1).mean())
    if not frames:
        raise RuntimeError("无可用子因子")
    comp, _ = _nanmean(frames)

    out = comp.stack().rename("factor").reset_index()
    out.columns = ["date", "instrument", "factor"]
    return _finalize(out, start_date, end_date, name="VPIN-3")


# %%
if __name__ == "__main__":
    from bigmodule import M
    import dai

    datasources = {"bar1m": "bigalpha_2026_stock_bar1m", "financial": "bigalpha_2026_financial"}
    start_date, end_date = "2024-01-01 00:00:00", "2024-12-31 23:59:59"
    factor_data = main(datasources, start_date, end_date)
    factor_pool = dai.query("SELECT * FROM bigalpha_2026_factorlib",
                            filters={"date": [start_date, end_date]}).df()
    result = M.bigalpha_eval._latest(factor_data=factor_data, factor_pool=factor_pool,
                                    process_pools=False, show=True)
