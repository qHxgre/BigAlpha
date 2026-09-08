def main(datasources, start_date, end_date):
    """
    BigAlpha 2026 AI赛道因子：
    尾盘拥挤反转 + 流动性质量过滤因子

    核心逻辑：
    1. 尾盘买盘压力越强，说明短线资金可能越拥挤；
    2. 尾盘买盘相对全天突然增强，说明尾盘抢筹/拉升特征更明显；
    3. 收盘价越靠近日内高位、越高于日内VWAP，说明当天被推高越明显；
    4. 这些“尾盘拥挤/过热”信号在前期测试中表现为反向有效，因此取负向；
    5. 同时保留流动性好、日内波动不过分的股票，降低噪声。

    返回：
    pd.DataFrame，包含且仅包含 ['date', 'instrument', 'factor']
    """
    import pandas as pd
    import numpy as np
    import dai

    # 使用平台注入的数据源映射，不在因子SQL中硬编码分钟表名
    bar1m = datasources["bar1m"]

    sql = f"""
    WITH base AS (
        SELECT
            date AS minute_ts,
            instrument,
            date_trunc('day', date)::DATE AS trading_day,
            strftime(date, '%H:%M:%S') AS intraday_time,

            close,
            high,
            low,
            pre_close,
            volume,
            amount,

            ask_price1,
            bid_price1,

            COALESCE(bid_volume1, 0) AS bid_volume1,
            COALESCE(bid_volume2, 0) AS bid_volume2,
            COALESCE(bid_volume3, 0) AS bid_volume3,
            COALESCE(bid_volume4, 0) AS bid_volume4,
            COALESCE(bid_volume5, 0) AS bid_volume5,

            COALESCE(ask_volume1, 0) AS ask_volume1,
            COALESCE(ask_volume2, 0) AS ask_volume2,
            COALESCE(ask_volume3, 0) AS ask_volume3,
            COALESCE(ask_volume4, 0) AS ask_volume4,
            COALESCE(ask_volume5, 0) AS ask_volume5

        FROM {bar1m}
        WHERE
            ask_price1 > 0
            AND bid_price1 > 0
            AND close > 0
            AND high > 0
            AND low > 0
            AND pre_close > 0
            AND volume > 0
            AND amount > 0
    ),

    level1 AS (
        SELECT
            minute_ts,
            instrument,
            trading_day,
            intraday_time,

            close,
            high,
            low,
            pre_close,
            volume,
            amount,

            (
                bid_volume1
                + bid_volume2 * EXP(-0.3)
                + bid_volume3 * EXP(-0.6)
                + bid_volume4 * EXP(-0.9)
                + bid_volume5 * EXP(-1.2)
            ) AS weight_bid,

            (
                ask_volume1
                + ask_volume2 * EXP(-0.3)
                + ask_volume3 * EXP(-0.6)
                + ask_volume4 * EXP(-0.9)
                + ask_volume5 * EXP(-1.2)
            ) AS weight_ask,

            (ask_price1 - bid_price1)
            /
            NULLIF((ask_price1 + bid_price1) / 2, 0) AS spread_ratio

        FROM base
    ),

    minute_signal AS (
        SELECT
            minute_ts,
            instrument,
            trading_day,
            intraday_time,

            close,
            high,
            low,
            pre_close,
            volume,
            amount,
            spread_ratio,

            (weight_bid - weight_ask)
            /
            NULLIF(weight_bid + weight_ask, 0) AS weighted_imbalance

        FROM level1
        WHERE
            weight_bid + weight_ask > 0
            AND spread_ratio IS NOT NULL
    ),

    daily AS (
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,

            COUNT(*) AS n_minutes,

            AVG(weighted_imbalance) AS avg_imbalance,

            AVG(
                CASE
                    WHEN intraday_time >= '14:30:00'
                    THEN weighted_imbalance
                    ELSE NULL
                END
            ) AS tail_imbalance,

            AVG(
                CASE
                    WHEN intraday_time >= '14:30:00'
                    THEN weighted_imbalance
                    ELSE NULL
                END
            ) - AVG(weighted_imbalance) AS tail_delta,

            (
                ARG_MAX(close, minute_ts) - MIN(low)
            )
            /
            NULLIF(MAX(high) - MIN(low), 0) AS close_position,

            (
                ARG_MAX(close, minute_ts)
                /
                NULLIF(SUM(amount) / NULLIF(SUM(volume), 0), 0)
                - 1
            ) AS close_vs_vwap,

            AVG(spread_ratio) AS avg_spread,

            (
                MAX(high) - MIN(low)
            )
            /
            NULLIF(AVG(pre_close), 0) AS intraday_range

        FROM minute_signal
        GROUP BY trading_day, instrument
    ),

    ranked AS (
        SELECT
            date,
            instrument,

            c_rank(COALESCE(tail_imbalance, avg_imbalance)) AS r_tail_imbalance,
            c_rank(COALESCE(tail_delta, 0)) AS r_tail_delta,
            c_rank(close_position) AS r_close_position,
            c_rank(close_vs_vwap) AS r_close_vwap,

            c_rank(-avg_spread) AS r_liquidity,
            c_rank(-intraday_range) AS r_stability

        FROM daily
        WHERE
            n_minutes >= 120
            AND avg_imbalance IS NOT NULL
            AND close_position IS NOT NULL
            AND close_vs_vwap IS NOT NULL
            AND avg_spread IS NOT NULL
            AND intraday_range IS NOT NULL
    )

    SELECT
        date,
        instrument,

        (
            -0.35 * r_tail_imbalance
            -0.20 * r_tail_delta
            -0.20 * r_close_position
            -0.15 * r_close_vwap
            +0.06 * r_liquidity
            +0.04 * r_stability
        ) AS factor

    FROM ranked
    ORDER BY date, instrument
    """

    df = dai.query(
        sql,
        filters={"date": [start_date, end_date]},
        compression=True
    ).df()

    # 对齐中证1000股票池
    stk_pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [start_date, end_date]},
        compression=True
    ).df()

    # 保证合并字段类型稳定
    df["date"] = pd.to_datetime(df["date"])
    stk_pool["date"] = pd.to_datetime(stk_pool["date"])
    df["instrument"] = df["instrument"].astype(str)
    stk_pool["instrument"] = stk_pool["instrument"].astype(str)

    # 以股票池为基准，尽量避免覆盖率问题
    df = pd.merge(
        stk_pool,
        df[["date", "instrument", "factor"]],
        how="left",
        on=["date", "instrument"]
    )

    # 清理 factor，避免 inf / -inf / 非数值
    df["factor"] = pd.to_numeric(df["factor"], errors="coerce")
    df["factor"] = df["factor"].replace([np.inf, -np.inf], np.nan)

    # 对少量缺失值按当日中位数填充，进一步降低覆盖率风险
    df["factor"] = df.groupby("date")["factor"].transform(
        lambda x: x.fillna(x.median())
    )

    # 如果某天极端情况下全为空，则填 0，保证不返回 NaN
    df["factor"] = df["factor"].fillna(0.0)

    # 最终只返回比赛要求的三列
    df = df[["date", "instrument", "factor"]].copy()

    return df