def main(datasources, start_date, end_date):
    """
    因子构建主函数

    评测时平台会自动替换 datasources / start_date / end_date 三个入参并调用本函数

    参数:
        datasources (dict): 数据源表名映射 {逻辑名: 物理表名}。一个因子可同时用到多张表，
                            通过逻辑名取出该阶段实际的物理表名，平台会在公榜/私榜自动切换。
                            当前可用逻辑名:
                                "bar1m"     -> 分钟 K 线表
                                "financial" -> 财务数据表
        start_date (str): 开始时间
        end_date (str):   结束时间

    返回:
        pd.DataFrame: 因子数据，须包含三列 ['date', 'instrument', 'factor']，且不含 inf
    """
    import pandas as pd
    import dai

    # 从映射里取出本阶段实际的物理表名（切勿在 SQL 里硬编码表名，否则公榜/私榜无法切换）
    bar1m = datasources["bar1m"]

    # 若计算滚动/时序类因子，可以多取若干天数据作为缓冲（本示例无需滚动，仅作演示）
    LOOKBACK_DAYS = 7
    query_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)

    # ===== 编写因子 SQL =====
    # 示例因子：基于 1 分钟盘口快照计算日内订单簿压力，按交易日聚合为日频因子
    # DAI 函数文档：https://bigquant.com/wiki/doc/Rceb2JQBdS
    sql = f"""
    WITH cte_bar1m AS (
        SELECT
            date, instrument, volume,

            -- 交易日
            strftime(date, '%Y-%m-%d') as trading_day,

            -- 计算中间价格
            (ask_price1 + bid_price1) / 2 as mid_price,

            -- 计算 加权5档买方量
            (
                COALESCE(bid_volume1, 0) * 1.0 +
                COALESCE(bid_volume2, 0) * EXP(-0.3) +
                COALESCE(bid_volume3, 0) * EXP(-0.6) +
                COALESCE(bid_volume4, 0) * EXP(-0.9) +
                COALESCE(bid_volume5, 0) * EXP(-1.2)
            ) as weight_bid,

            -- 计算 加权5档卖方量
            (
                COALESCE(ask_volume1, 0) * 1.0 +
                COALESCE(ask_volume2, 0) * EXP(-0.3) +
                COALESCE(ask_volume3, 0) * EXP(-0.6) +
                COALESCE(ask_volume4, 0) * EXP(-0.9) +
                COALESCE(ask_volume5, 0) * EXP(-1.2)
            ) as weight_ask,

            -- 计算 加权订单簿不平衡度
            (weight_bid - weight_ask) / (weight_bid + weight_ask + 1e-8) as weighted_imbalance,

            -- 计算 价差因子
            (ask_price1 - bid_price1) / mid_price as relative_spread,

            -- 计算 3档买方量
            (
                COALESCE(bid_volume1, 0) +
                COALESCE(bid_volume2, 0) +
                COALESCE(bid_volume3, 0)
            ) as total_bid,

            -- 计算 3档卖方量
            (
                -- 加权卖方成交量
                COALESCE(ask_volume1, 0) +
                COALESCE(ask_volume2, 0) +
                COALESCE(ask_volume3, 0)
            ) as total_ask,

            -- 计算 深度比率
            (total_bid - total_ask) / (total_bid + total_ask + 1e-8) as depth_ratio,

            -- 计算 压力因子
            (weighted_imbalance / (sqrt(abs(relative_spread)) + 1e-8)) * weighted_imbalance as raw_pressure

        FROM {bar1m}
        -- 剔除集合竞价/停牌等无有效盘口的行（mid_price=0 会导致 log(0) 报错）
        WHERE ask_price1 > 0 AND bid_price1 > 0
    ),
    -- 计算分钟级时序量（按交易日分组的相邻收益）
    cte_rolling AS(
        SELECT
            *,
            lag(mid_price, 1) OVER (PARTITION BY instrument, trading_day ORDER BY date) as prev_mid_price,
            abs(mid_price / prev_mid_price - 1) as returns,
        FROM cte_bar1m
    ),
    -- 按交易日聚合为日频因子值
    cte_window AS (
        SELECT
            trading_day, instrument,

            -- 标准化原始压力因子
            avg(raw_pressure) as raw_pressure_mean,
            nanstd(raw_pressure) as raw_pressure_std,
            (last(raw_pressure order by date) - raw_pressure_mean) / raw_pressure_std as standardized_pressure,

            -- 计算波动率阀值
            CASE
                WHEN COUNT(*) > 10
                THEN quantile(returns, 0.8)
                ELSE 0.01
            END as volatility_threshold,

            -- 根据波动率调整因子（高波动时降低因子值）
            -- 仅在 mid_price 与 prev_mid_price 均为正时取对数，避免 log(0)/log(null)
            nanstd(
                CASE
                    WHEN mid_price > 0 AND prev_mid_price > 0
                    THEN log(mid_price / prev_mid_price)
                    ELSE NULL
                END
            ) as volatility,
            CASE
                WHEN volatility > volatility_threshold
                THEN standardized_pressure * 0.7
                ELSE standardized_pressure
            END as adjusted_pressure,

            tanh(adjusted_pressure) as factor
        FROM cte_rolling
        GROUP BY instrument, trading_day
        ORDER BY instrument, trading_day
    )
    -- 输出 date / instrument / factor 三列
    SELECT
        -- 转换为日频的 date 列（取交易日 00:00:00）
        CAST(trading_day AS DATETIME) AS date,
        instrument,
        -- 因子方向为 -1
        factor * -1 as factor
    FROM cte_window
    """

    # ===== 调用dai计算因子 =====
    # compression=True 会把 instrument 列转为 category 类型，显著降低内存占用
    df = dai.query(sql, filters={'date': [start_date, end_date]}, compression=True).df()

    # ===== 对齐股票池 =====
    # 数据源保留了 2019 年至今所有成分股的数据以便计算时序因子，
    # 因此需与中证 1000 成分股做内连接，只保留当日属于成分股的标的
    # bigalpha_2026_instruments 已经收录了2019年以来的所有数据，不用替换
    stk_pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={'date': [start_date, end_date]},
    ).df()
    df = pd.merge(df, stk_pool, how='inner', on=['date', 'instrument'])

    return df


if __name__ == '__main__':
    from bigmodule import M
    import dai
    import structlog

    logger = structlog.get_logger()

    # 本地自测时自行构造数据源映射（评测时由平台注入，逻辑名固定为 "bar1m"/"financial"）
    datasources = {'bar1m': 'bigalpha_2026_stock_bar1m'}
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-12-31 23:59:59'

    # 计算因子
    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)

    # 读取平台因子库用于回归评估，您可以换成自己的因子库
    logger.info(f"读取因子库，区间：{start_date} ~ {end_date}")
    factor_pool = dai.query(
        "SELECT * FROM bigalpha_2026_factorlib",
        filters={'date': [start_date, end_date]},
    ).df()

    # 评估系统：
    # process_pools=False 表示不对因子库再做预处理（bigalpha_2026_factorlib 已处理过）
    # show=True 表示画出评估图表
    result = M.bigalpha_eval._latest(
        factor_data=factor_data,
        factor_pool=factor_pool,
        process_pools=False,
        show=True,
    )
