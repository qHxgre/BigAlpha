from __future__ import annotations

ACTIVE = "book_depth_slope"
BUFFER_DAYS = 5


def main(datasources, start_date, end_date):
    import dai
    import numpy as np
    import pandas as pd

    bar1m = datasources["bar1m"]
    financial = datasources.get("financial", "bigalpha_2026_financial")
    q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=BUFFER_DAYS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    sql = {
        "book_depth_slope": f"""
SELECT
    date::DATE::DATETIME AS date,
    instrument,
    AVG(
        (
            (bid_volume1 + bid_volume2 + bid_volume5)
            - (ask_volume1 + ask_volume2 + ask_volume5)
        ) * 1.0
        / NULLIF(
            bid_volume1 + bid_volume2 + bid_volume5
            + ask_volume1 + ask_volume2 + ask_volume5, 0
        )
    ) AS factor
FROM {bar1m}
GROUP BY date::DATE, instrument
ORDER BY date, instrument
""",
        "rel_spread_stress": f"""
SELECT
    date::DATE::DATETIME AS date,
    instrument,
    AVG(
        (ask_price1 - bid_price1) * 1.0
        / NULLIF((ask_price1 + bid_price1) / 2.0, 0)
    ) AS factor
FROM {bar1m}
GROUP BY date::DATE, instrument
ORDER BY date, instrument
""",
        "vwap_dev_close": f"""
SELECT
    date::DATE::DATETIME AS date,
    instrument,
    LAST(close ORDER BY date)
        / NULLIF(
            LAST(amount ORDER BY date)
                / NULLIF(LAST(volume ORDER BY date), 0),
            0
        )
        - 1 AS factor
FROM {bar1m}
GROUP BY date::DATE, instrument
ORDER BY date, instrument
""",
        "volume_tail_share": f"""
SELECT
    date::DATE::DATETIME AS date,
    instrument,
    SUM(CASE WHEN vol_rank >= 0.8 THEN bar_vol ELSE 0 END) * 1.0
        / NULLIF(SUM(bar_vol), 0) AS factor
FROM (
    SELECT
        date,
        instrument,
        bar_vol,
        PERCENT_RANK() OVER (
            PARTITION BY instrument, date::DATE ORDER BY bar_vol
        ) AS vol_rank
    FROM (
        SELECT
            date,
            instrument,
            GREATEST(
                volume - COALESCE(
                    LAG(volume) OVER (
                        PARTITION BY instrument, date::DATE ORDER BY date
                    ),
                    0
                ),
                0
            ) AS bar_vol
        FROM {bar1m}
    ) x
) t
GROUP BY date::DATE, instrument
ORDER BY date, instrument
""",
        "roe_ttm_pit": f"""
SELECT
    date::DATE::DATETIME AS date,
    instrument,
    MAX(CASE WHEN category = 'ttm' THEN net_profit_to_parent_shareholders END) * 1.0
        / NULLIF(
            MAX(CASE WHEN category = 'lf' THEN total_equity_to_parent_shareholders END),
            0
        ) AS factor
FROM {financial}
GROUP BY date::DATE, instrument
ORDER BY date, instrument
""",
        "book_pressure_l1": f"""
SELECT
    date::DATE::DATETIME AS date,
    instrument,
    AVG(
        (bid_volume1 - ask_volume1) * 1.0
        / NULLIF(bid_volume1 + ask_volume1, 0)
    ) AS factor
FROM {bar1m}
GROUP BY date::DATE, instrument
ORDER BY date, instrument
""",
    }[ACTIVE]

    q_start = q0 if ACTIVE != "roe_ttm_pit" else (
        pd.to_datetime(start_date) - pd.Timedelta(days=120)
    ).strftime("%Y-%m-%d %H:%M:%S")

    df = dai.query(
        sql,
        filters={"date": [q_start, end_date]},
        compression=True,
    ).df()

    if ACTIVE == "roe_ttm_pit":
        pool = dai.query(
            "SELECT date, instrument FROM bigalpha_2026_instruments",
            filters={"date": [q_start, end_date]},
            compression=True,
        ).df()
        pool = pool.sort_values(["instrument", "date"])
        fin = df.dropna(subset=["factor"]).sort_values(["instrument", "date"])
        parts = []
        for inst, g in pool.groupby("instrument", sort=False):
            f = fin[fin["instrument"] == inst]
            if f.empty:
                continue
            m = pd.merge_asof(
                g.sort_values("date"),
                f[["date", "factor"]].sort_values("date"),
                on="date",
                direction="backward",
            )
            parts.append(m)
        df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

    df = df[
        (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ]
    stk = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    df = pd.merge(df, stk, how="inner", on=["date", "instrument"])
    df = df[["date", "instrument", "factor"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor"])
    return df


if __name__ == "__main__":
    print(
        main(
            {
                "bar1m": "bigalpha_2026_stock_bar1m",
                "financial": "bigalpha_2026_financial",
            },
            "2019-01-01 00:00:00",
            "2019-03-31 23:59:59",
        ).head()
    )
