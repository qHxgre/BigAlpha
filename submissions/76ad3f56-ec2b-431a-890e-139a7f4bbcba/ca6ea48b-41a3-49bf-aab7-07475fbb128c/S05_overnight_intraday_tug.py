# %% [markdown]
# # S05 · 隔夜 / 日内收益拉锯 `TUG-2`（传统赛道）
#
# **【经济逻辑】** Lou, Polk & Skouras (2019) *A Tug of War*：同一只股票的收益可以拆成
# **隔夜段**与**日内段**，两段由结构完全不同的投资者主导——
# 隔夜段（跳空）主要由散户情绪与消息面驱动，日内连续竞价段主要由机构与套利资金驱动。
# 两段收益在后续期呈**相反**的可预测性，且两段之差与"总收益"（普通反转/动量因子）**在构造上近似正交**，
# 这正是本因子在全场因子池中能保留独立权重的原因。
#
# 因子 = 过去 20 日「日内累计收益 − 隔夜累计收益」（机构主导段强、散户主导段弱 ⇒ 后续正向）。
#
# **风控设计**：`pre_close` 在除权除息日语义可能不一致，`|隔夜收益| > 21%` 一律置空（A 股单日
# 涨跌幅上限为 10%/20%），避免复权异常污染因子。
#
# ⚠️ 方向以 A 股先验设为 `+1`（日内强 − 隔夜强 ⇒ 正向）。请用 `research/R02` 确认符号后再提交。

# %%
LOOKBACK_CAL_DAYS = 50
WIN_LONG = 20
WIN_SHORT = 5
SMOOTH_WIN = 3          # 本因子本身已是 20 日累计，平滑窗口取小
MIN_BARS = 30
JUMP_CAP = 0.21         # 隔夜收益绝对值上限（超出视为除权异常）

MEMBERS = [
    ("tug20", +1.0),
    ("tug5",  +1.0),
]


def _probe(bar1m, start_date):
    import pandas as pd
    import dai

    s = pd.to_datetime(start_date)
    head = dai.query(f"SELECT * FROM {bar1m} LIMIT 5",
                     filters={"date": [str(s), str(s + pd.Timedelta(days=12))]}).df()
    cols = set(head.columns)
    px = "close" if "close" in cols else "price"
    has_pc = "pre_close" in cols
    print(f"[probe] 价格列={px} 有 pre_close={has_pc}")
    return {"px": px, "has_pc": has_pc}


def _build_sql(bar1m, cfg):
    px, has_pc = cfg["px"], cfg["has_pc"]
    pc_sel = "FIRST(pre_close ORDER BY date) AS preclose_d," if has_pc else \
             "CAST(NULL AS DOUBLE) AS preclose_d,"
    return f"""
    WITH b0 AS (
        SELECT date, instrument, CAST(date AS DATE) AS dt, {px} AS px
               {", pre_close" if has_pc else ""}
        FROM {bar1m}
        WHERE CAST(date AS TIME) >= TIME '09:30:00' AND CAST(date AS TIME) <= TIME '15:00:00'
    )
    SELECT instrument, dt,
           COUNT(*) AS nbar,
           {pc_sel}
           FIRST(px ORDER BY date) AS open_d,
           LAST(px  ORDER BY date) AS close_d
    FROM b0
    GROUP BY instrument, dt
    HAVING COUNT(*) >= {MIN_BARS}
    """


def _nanmean(frames):
    import numpy as np
    s = n = None
    for f in frames:
        v, m = f.fillna(0.0), f.notna().astype("float32")
        s = v if s is None else s.add(v, fill_value=0.0)
        n = m if n is None else n.add(m, fill_value=0.0)
    return s / n.replace(0.0, np.nan), n


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
    import numpy as np
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

    o = raw.pivot(index="dt", columns="instrument", values="open_d").sort_index()
    c = raw.pivot(index="dt", columns="instrument", values="close_d").sort_index()
    if cfg["has_pc"] and raw["preclose_d"].notna().any():
        pc = raw.pivot(index="dt", columns="instrument", values="preclose_d").sort_index()
    else:
        pc = c.shift(1)                      # 退化方案：用前一交易日收盘（除权日会被 JUMP_CAP 过滤）

    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = np.log(o / pc)
        intraday = np.log(c / o)
    overnight = overnight.where(overnight.abs() <= JUMP_CAP)      # 剔除疑似除权
    intraday = intraday.where(intraday.abs() <= JUMP_CAP)

    tug = {
        "tug20": intraday.rolling(WIN_LONG, min_periods=WIN_LONG // 2).sum()
                 - overnight.rolling(WIN_LONG, min_periods=WIN_LONG // 2).sum(),
        "tug5": intraday.rolling(WIN_SHORT, min_periods=3).sum()
                - overnight.rolling(WIN_SHORT, min_periods=3).sum(),
    }

    # 成分股对齐（在排名之前）
    pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",
                     filters={"date": [q_start, end_date]}).df()
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    mask = (pool.assign(v=1.0).drop_duplicates(["date", "instrument"])
                .pivot(index="date", columns="instrument", values="v"))

    frames = []
    for col, sgn in MEMBERS:
        x = tug[col].reindex(index=mask.index.union(tug[col].index)).reindex(columns=mask.columns.union(tug[col].columns))
        m = mask.reindex(index=x.index, columns=x.columns)
        x = x.where(m.notna())
        rk = (sgn * x).rank(axis=1, pct=True) - 0.5
        if rk.notna().sum().sum() == 0:
            continue
        frames.append(rk.rolling(SMOOTH_WIN, min_periods=1).mean())
    if not frames:
        raise RuntimeError("无可用子因子")
    comp, _ = _nanmean(frames)

    out = comp.stack().rename("factor").reset_index()
    out.columns = ["date", "instrument", "factor"]
    return _finalize(out, start_date, end_date, name="TUG-2")


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
