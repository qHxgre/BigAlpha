## Overview

本比赛只可使用指定的数据源来构建因子。评判程序会 `import` 参赛用户提交代码里的 `main` 函数，并传入数据源名、开始日期时间、结束日期时间来调用。

## 数据源

- 股票池：沪深300指数在历史相应时间点上的成分股
- 数据频率：snapshot级别K线及盘口快照数据
- 开发数据源：[cpt_dwc_2026_stock_hs300_snapshot](https://deepwin.bigquant.com/data/datasources/cpt_dwc_2026_stock_hs300_snapshot) 包含2023年1月1日至2024年12月31日的数据
- 公榜数据源：参赛用户不可见。
- 私榜数据源：参赛用户不可见。

## 数据格式

参考：[cpt_dwc_2026_stock_hs300_snapshot](https://deepwin.bigquant.com/data/datasources/cpt_dwc_2026_stock_hs300_snapshot)

以下是根据您提供的内容转换成的 Markdown 格式表格：

| 字段名 | 数据类型 | 描述 |
| --- | --- | --- |
| date | np.datetime64 | 时间 |
| instrument | pd.StringDtype | 标的 |
| time | np.int32 | 时间(HHMMSSmmm) |
| trading_day | np.int32 | 交易日期 |
| pre_close | np.float32 | 前收盘 |
| open | np.float32 | 开盘价 |
| high | np.float32 | 最高价 |
| low | np.float32 | 最低价 |
| price | np.float32 | 成交价 |
| ask_price1 | np.float32 | 1档委卖价 |
| ask_price2 | np.float32 | 2档委卖价 |
| ask_price3 | np.float32 | 3档委卖价 |
| ask_price4 | np.float32 | 4档委卖价 |
| ask_price5 | np.float32 | 5档委卖价 |
| ask_price6 | np.float32 | 6档委卖价 |
| ask_price7 | np.float32 | 7档委卖价 |
| ask_price8 | np.float32 | 8档委卖价 |
| ask_price9 | np.float32 | 9档委卖价 |
| ask_price10 | np.float32 | 10档委卖价 |
| ask_volume1 | np.int64 | 1档委卖量 |
| ask_volume2 | np.int32 | 2档委卖量 |
| ask_volume3 | np.int32 | 3档委卖量 |
| ask_volume4 | np.int32 | 4档委卖量 |
| ask_volume5 | np.int32 | 5档委卖量 |
| ask_volume6 | np.int32 | 6档委卖量 |
| ask_volume7 | np.int32 | 7档委卖量 |
| ask_volume8 | np.int32 | 8档委卖量 |
| ask_volume9 | np.int32 | 9档委卖量 |
| ask_volume10 | np.int32 | 10档委卖量 |
| bid_price1 | np.float32 | 1档委买价 |
| bid_price2 | np.float32 | 2档委买价 |
| bid_price3 | np.float32 | 3档委买价 |
| bid_price4 | np.float32 | 4档委买价 |
| bid_price5 | np.float32 | 5档委买价 |
| bid_price6 | np.float32 | 6档委买价 |
| bid_price7 | np.float32 | 7档委买价 |
| bid_price8 | np.float32 | 8档委买价 |
| bid_price9 | np.float32 | 9档委买价 |
| bid_price10 | np.float32 | 10档委买价 |
| bid_volume1 | np.int64 | 1档委买量 |
| bid_volume2 | np.int32 | 2档委买量 |
| bid_volume3 | np.int32 | 3档委买量 |
| bid_volume4 | np.int32 | 4档委买量 |
| bid_volume5 | np.int32 | 5档委买量 |
| bid_volume6 | np.int32 | 6档委买量 |
| bid_volume7 | np.int32 | 7档委买量 |
| bid_volume8 | np.int32 | 8档委买量 |
| bid_volume9 | np.int32 | 9档委买量 |
| bid_volume10 | np.int32 | 10档委买量 |
| bid_num_orders1 | np.int32 | 卖1档委托笔数 |
| bid_num_orders2 | np.int32 | 卖2档委托笔数 |
| bid_num_orders3 | np.int32 | 卖3档委托笔数 |
| bid_num_orders4 | np.int32 | 卖4档委托笔数 |
| bid_num_orders5 | np.int32 | 卖5档委托笔数 |
| bid_num_orders6 | np.int32 | 卖6档委托笔数 |
| bid_num_orders7 | np.int32 | 卖7档委托笔数 |
| bid_num_orders8 | np.int32 | 卖8档委托笔数 |
| bid_num_orders9 | np.int32 | 卖9档委托笔数 |
| bid_num_orders10 | np.int32 | 卖10档委托笔数 |
| ask_num_orders1 | np.int32 | 买1档委托笔数 |
| ask_num_orders2 | np.int32 | 买2档委托笔数 |
| ask_num_orders3 | np.int32 | 买3档委托笔数 |
| ask_num_orders4 | np.int32 | 买4档委托笔数 |
| ask_num_orders5 | np.int32 | 买5档委托笔数 |
| ask_num_orders6 | np.int32 | 买6档委托笔数 |
| ask_num_orders7 | np.int32 | 买7档委托笔数 |
| ask_num_orders8 | np.int32 | 买8档委托笔数 |
| ask_num_orders9 | np.int32 | 买9档委托笔数 |
| ask_num_orders10 | np.int32 | 买10档委托笔数 |
| num_trades | np.int32 | 成交笔数 |
| volume | np.int64 | 当日累计成交量 |
| amount | np.float64 | 当日成交额(元) |
| total_bid_volume | np.int64 | 委买总量 |
| total_ask_volume | np.int64 | 委卖总量 |
| bid_avg_price | np.float32 | 加权平均委买价 |
| ask_avg_price | np.float32 | 加权平均委卖价 |

## 读取示例

```
import dai

dai.query("SELECT * FROM cpt_dwc_2026_stock_hs300_snapshot", filters={"date": ["2023-01-01 00:00:00", "2023-01-05 23:59:59"]}).df().head()
```

|    | date                |   instrument_id |   pre_close |   open |   high |   low |   price |   ask_price1 |   ask_price2 |   ask_price3 |   ask_price4 |   ask_price5 |   ask_price6 |   ask_price7 |   ask_price8 |   ask_price9 |   ask_price10 |   ask_volume1 |   ask_volume2 |   ask_volume3 |   ask_volume4 |   ask_volume5 |   ask_volume6 |   ask_volume7 |   ask_volume8 |   ask_volume9 |   ask_volume10 |   bid_price1 |   bid_price2 |   bid_price3 |   bid_price4 |   bid_price5 |   bid_price6 |   bid_price7 |   bid_price8 |   bid_price9 |   bid_price10 |   bid_volume1 |   bid_volume2 |   bid_volume3 |   bid_volume4 |   bid_volume5 |   bid_volume6 |   bid_volume7 |   bid_volume8 |   bid_volume9 |   bid_volume10 |   bid_num_orders1 |   bid_num_orders2 |   bid_num_orders3 |   bid_num_orders4 |   bid_num_orders5 |   bid_num_orders6 |   bid_num_orders7 |   bid_num_orders8 |   bid_num_orders9 |   bid_num_orders10 |   ask_num_orders1 |   ask_num_orders2 |   ask_num_orders3 |   ask_num_orders4 |   ask_num_orders5 |   ask_num_orders6 |   ask_num_orders7 |   ask_num_orders8 |   ask_num_orders9 |   ask_num_orders10 |   num_trades |   volume |   amount |   total_bid_volume |   total_ask_volume |   bid_avg_price |   ask_avg_price |
|---:|:--------------------|----------------:|------------:|-------:|-------:|------:|--------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|---------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------:|-------------:|---------:|---------:|-------------------:|-------------------:|----------------:|----------------:|
|  0 | 2023-01-03 09:15:00 |               1 |       13.16 |      0 |      0 |     0 |       0 |        13.15 |            0 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |          3200 |          3900 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |        13.15 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |          3200 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |            0 |        0 |        0 |                  0 |                  0 |               0 |               0 |
|  1 | 2023-01-03 09:15:00 |               2 |       18.2  |      0 |      0 |     0 |       0 |        18.2  |            0 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |         12000 |          4600 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |        18.2  |          nan |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |         12000 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |            0 |        0 |        0 |                  0 |                  0 |               0 |               0 |
|  2 | 2023-01-03 09:15:00 |              56 |       25.86 |      0 |      0 |     0 |       0 |        25.86 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |          1500 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |        25.86 |            0 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |          1500 |           400 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |            0 |        0 |        0 |                  0 |                  0 |               0 |               0 |
|  3 | 2023-01-03 09:15:00 |              60 |        5.33 |      0 |      0 |     0 |       0 |         5.36 |            0 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |          1000 |           500 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |         5.36 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |          1000 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |            0 |        0 |        0 |                  0 |                  0 |               0 |               0 |
|  4 | 2023-01-03 09:15:00 |              68 |        3.72 |      0 |      0 |     0 |       0 |         3.73 |            0 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |         19600 |       1773200 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |         3.73 |          nan |          nan |          nan |          nan |          nan |          nan |          nan |          nan |           nan |         19600 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |             0 |              0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                 0 |                  0 |            0 |        0 |        0 |                  0 |                  0 |               0 |               0 |

可以注意到 cpt_dwc_2026_stock_hs300_snapshot 的 **instrument_id** 是 BigQuant 平台的自定义的代码ID，没有特殊含义。如果要转换为我们常见的证券代码，可以通过 all_instruments 表进行映射

```
import dai

dai.query("""
SELECT
    all_instruments.instrument,
    cpt_dwc_2026_stock_hs300_snapshot.*
FROM cpt_dwc_2026_stock_hs300_snapshot
LEFT JOIN all_instruments USING (instrument_id)
""", filters={"date": ["2023-01-01 00:00:00", "2023-01-05 23:59:59"]}).df().head()
```

|    | instrument   | date                |   instrument_id |   pre_close |   open |   high |    low |   price |   ask_price1 |   ask_price2 |   ask_price3 |   ask_price4 |   ask_price5 |   ask_price6 |   ask_price7 |   ask_price8 |   ask_price9 |   ask_price10 |   ask_volume1 |   ask_volume2 |   ask_volume3 |   ask_volume4 |   ask_volume5 |   ask_volume6 |   ask_volume7 |   ask_volume8 |   ask_volume9 |   ask_volume10 |   bid_price1 |   bid_price2 |   bid_price3 |   bid_price4 |   bid_price5 |   bid_price6 |   bid_price7 |   bid_price8 |   bid_price9 |   bid_price10 |   bid_volume1 |   bid_volume2 |   bid_volume3 |   bid_volume4 |   bid_volume5 |   bid_volume6 |   bid_volume7 |   bid_volume8 |   bid_volume9 |   bid_volume10 |   bid_num_orders1 |   bid_num_orders2 |   bid_num_orders3 |   bid_num_orders4 |   bid_num_orders5 |   bid_num_orders6 |   bid_num_orders7 |   bid_num_orders8 |   bid_num_orders9 |   bid_num_orders10 |   ask_num_orders1 |   ask_num_orders2 |   ask_num_orders3 |   ask_num_orders4 |   ask_num_orders5 |   ask_num_orders6 |   ask_num_orders7 |   ask_num_orders8 |   ask_num_orders9 |   ask_num_orders10 |   num_trades |   volume |      amount |   total_bid_volume |   total_ask_volume |   bid_avg_price |   ask_avg_price |
|---:|:-------------|:--------------------|----------------:|------------:|-------:|-------:|-------:|--------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|---------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------:|-------------:|---------:|------------:|-------------------:|-------------------:|----------------:|----------------:|
|  0 | 600436.SH    | 2023-01-03 09:37:44 |            3442 |      288.46 | 288.5  | 290.51 | 285.52 |  285.81 |       286.29 |       286.3  |       286.71 |       286.72 |       286.75 |       286.96 |       287.07 |       287.93 |       287.94 |        287.98 |           200 |           100 |           100 |           600 |           100 |           600 |           100 |           100 |           100 |            100 |       285.81 |       285.8  |       285.76 |       285.61 |       285.6  |       285.59 |       285.58 |       285.57 |       285.53 |        285.52 |           200 |           600 |           100 |           100 |          1000 |           100 |           600 |          1700 |           600 |            300 |                 1 |                 3 |                 1 |                 1 |                 6 |                 1 |                 6 |                 6 |                 5 |                  3 |                 1 |                 1 |                 1 |                 2 |                 1 |                 1 |                 1 |                 1 |                 1 |                  1 |         1041 |   166800 | 4.79901e+07 |              52800 |             165970 |         280.649 |         299.794 |
|  1 | 600438.SH    | 2023-01-03 09:37:44 |            3443 |       38.58 |  38.55 |  38.64 |  37.89 |   38.31 |        38.31 |        38.32 |        38.33 |        38.34 |        38.35 |        38.36 |        38.37 |        38.38 |        38.39 |         38.4  |           100 |           600 |          2400 |          2900 |          1000 |           800 |          1400 |          9300 |           100 |           3000 |        38.3  |        38.29 |        38.28 |        38.27 |        38.26 |        38.25 |        38.23 |        38.22 |        38.21 |         38.2  |         26800 |           500 |          1100 |          4000 |          1100 |          2000 |          1100 |          3300 |          1900 |          19500 |                 1 |                 2 |                 5 |                 2 |                 4 |                 3 |                 3 |                 7 |                 6 |                 17 |                 1 |                 2 |                10 |                 5 |                 4 |                 3 |                 6 |                 8 |                 1 |                  7 |         8319 |  4436485 | 1.69521e+08 |            1610900 |            2002900 |          36.416 |          40.487 |
|  2 | 600570.SH    | 2023-01-03 09:37:44 |            3547 |       40.46 |  40.35 |  40.81 |  40.11 |   40.5  |        40.49 |        40.5  |        40.51 |        40.58 |        40.59 |        40.6  |        40.61 |        40.62 |        40.64 |         40.65 |          2000 |           294 |          3300 |           600 |           600 |          5000 |          3100 |          3000 |           100 |            400 |        40.48 |        40.45 |        40.44 |        40.43 |        40.4  |        40.38 |        40.34 |        40.32 |        40.31 |         40.3  |           100 |         14600 |            90 |           200 |          2500 |           800 |           600 |           800 |          2000 |           3600 |                 1 |                 4 |                 1 |                 1 |                 3 |                 2 |                 1 |                 1 |                 1 |                  5 |                 1 |                 1 |                 1 |                 5 |                 3 |                 4 |                 3 |                 2 |                 1 |                  2 |         2093 |   808200 | 3.27148e+07 |             176490 |            1007273 |          39.494 |          43.263 |
|  3 | 600588.SH    | 2023-01-03 09:37:44 |            3564 |       24.17 |  24.17 |  24.56 |  24    |   24.53 |        24.54 |        24.55 |        24.56 |        24.57 |        24.58 |        24.59 |        24.6  |        24.61 |        24.62 |         24.63 |          6200 |         10100 |          9800 |         22200 |         27200 |         19600 |         28900 |           500 |           100 |            700 |        24.53 |        24.5  |        24.49 |        24.48 |        24.47 |        24.46 |        24.45 |        24.44 |        24.42 |         24.41 |           800 |          9300 |          1300 |          2300 |          7900 |         14100 |          6800 |           900 |          5200 |            700 |                 2 |                 2 |                 5 |                 3 |                 2 |                 3 |                 4 |                 5 |                 3 |                  2 |                 4 |                15 |                12 |                14 |                23 |                14 |                37 |                 2 |                 1 |                  2 |         4941 |  2390201 | 5.82448e+07 |             479600 |            1544200 |          23.742 |          25.626 |
|  4 | 600600.SH    | 2023-01-03 09:37:44 |            3576 |      107.5  | 107.99 | 107.99 | 104.66 |  105.44 |       105.49 |       105.52 |       105.6  |       105.84 |       105.9  |       105.93 |       105.96 |       105.97 |       105.98 |        105.99 |           100 |           100 |           100 |           100 |           100 |           500 |           200 |           300 |           200 |            100 |       105.26 |       105.2  |       105.19 |       105.11 |       105.1  |       105.09 |       105.08 |       105.05 |       105.04 |        105.01 |          1200 |          1700 |           300 |           100 |           100 |           200 |          2900 |           400 |           200 |            100 |                 4 |                 3 |                 3 |                 1 |                 1 |                 1 |                 3 |                 4 |                 2 |                  1 |                 1 |                 1 |                 1 |                 1 |                 1 |                 1 |                 2 |                 1 |                 1 |                  1 |         2403 |   489543 | 5.17393e+07 |              75173 |             149000 |         103.917 |         110.062 |

## 因子计算示例

### DAI 数据引擎

DAI 内置多种算子/函数，参考 [DAI函数文档](https://bigquant.com/wiki/doc/Rceb2JQBdS)。DAI是专为AI/量化场景优化的超高性能计算引擎，能充分利用现代CPU/GPU能力。

```
import dai

dai.query("""
    SELECT
        date::DATE::DATETIME AS date,
        instrument_id,
        AVG(price) / LAST(price) AS factor
    FROM cpt_dwc_2026_stock_hs300_snapshot
    GROUP BY date::DATE, instrument_id
    ORDER BY date, instrument_id
""", filters={"date": ["2023-01-01 00:00:00", "2023-02-01 23:59:59"]}, compression=True).df().head()
```

|    | date                |   instrument_id |   factor |
|---:|:--------------------|----------------:|---------:|
|  0 | 2023-01-03 00:00:00 |               1 | 0.986934 |
|  1 | 2023-01-03 00:00:00 |               2 | 0.982757 |
|  2 | 2023-01-03 00:00:00 |              56 | 0.983466 |
|  3 | 2023-01-03 00:00:00 |              60 | 0.984321 |
|  4 | 2023-01-03 00:00:00 |              68 | 0.995045 |

### UDF

DAI 数据引擎支持UDF（User-Defined Function，用户定义函数），指允许用户通过编写自定义函数来扩展算子。

```
import dai

# 定义计算因子的UDF - 带类型声明（推荐）
def calculate_factor(instrument_id: str, prices: list) -> float:
    """计算因子：平均价格/最后价格"""
    if not prices:
        return None
    
    avg_price = sum(prices) / len(prices)
    last_price = prices[-1]
    
    # 避免除零错误
    if last_price == 0:
        return None
    
    return avg_price / last_price

# 计算因子
dai.query("""
    WITH grouped_data AS (
        SELECT
            date::DATE::DATETIME AS date,
            instrument_id,
            ARRAY_AGG(price ORDER BY date) AS price_list
        FROM cpt_dwc_2026_stock_hs300_snapshot
        GROUP BY date::DATE, instrument_id
    )
    SELECT
        date,
        instrument_id,
        calculate_factor(instrument_id, price_list) AS factor
    FROM grouped_data
    ORDER BY date, instrument_id
""", 
filters={"date": ["2023-01-01 00:00:00", "2023-02-01 23:59:59"]}, 
compression=True,
udf_list=[
    dai.DaiUDF(
        name="calculate_factor",
        function=calculate_factor,
    )
]).df().head()
```

|    | date                |   instrument_id |   factor |
|---:|:--------------------|----------------:|---------:|
|  0 | 2023-01-03 00:00:00 |               1 | 0.973316 |
|  1 | 2023-01-03 00:00:00 |               2 | 0.988687 |
|  2 | 2023-01-03 00:00:00 |              56 | 0.989155 |
|  3 | 2023-01-03 00:00:00 |              60 | 0.999206 |
|  4 | 2023-01-03 00:00:00 |              68 | 0.981883 |

### 第三计算库

DAI 数据引擎也支持将数据转换为 pd.DataFrame、pl.DataFrame、arrow 等格式的数据类型。

```
import dai

data = dai.query("SELECT date, instrument, price FROM cpt_dwc_2026_stock_hs300_snapshot", filters={"date": ["2023-01-01 00:00:00", "2023-02-01 23:59:59"]}, compression=True)

# 转为 pandas dataframe，然后在 pandas 继续计算
df = data.df()

# 转为 polars dataframe，然后在 polars 继续计算，polars没有内置在 aistudio 中，可以自行安装 pip3 install polars
df = data.pl()

# 转为 apache arrow，然后在 arrow 继续计算
df = data.arrow()

# 更多参考 DAI 文档
```

更多参考 DAI 文档：[DAI文档](https://bigquant.com/wiki/doc/PLSbc1SbZX)

### 混合计算

dai 与 pandas 等混合计算，可以在 dai 中高性能的使用 pandas、polars、arrow等数据（通过参数 `bind_relations` 绑定），并支持JOIN等操作。

```
import dai

data = dai.query("SELECT date, instrument, price FROM cpt_dwc_2026_stock_hs300_snapshot", filters={"date": ["2023-01-01 00:00:00", "2023-02-01 23:59:59"]})

df = data.df()

dai.query("""
    SELECT
        date::DATE::DATETIME AS date,
        instrument AS instrument,
        AVG(price) AS avg_close
    FROM df
    GROUP BY date::DATE::DATETIME, instrument
""", bind_relations={"df": df}).df()
```

## Tips

- 关于日期和时间
  - `date::DATE` 只取得 `date` 的日期部分，e.g. `2025-01-03 14:24:31`::DATE 为 `2025-01-03`。注意：`date` 的类型是 TIMESTAMP，`date::DATE` 类型是 `DATE`，可能在部分比较时出现数据类型不一致问题
  - `date::DATE::DATETIME` 可以这样再将数据类型转为 `DATETIME`
  - 更多日期/时间函数参考 [DAI函数文档](https://bigquant.com/wiki/doc/Rceb2JQBdS)：`date_trunc`, `time_bucket`
- 时序算子，DAI提供的函数，一般情况下，`m_` 前缀的是时序算子，并且默认基于 `GROUP BY instrument` 计算，e.g. `SELECT date, instrument, close / m_lag(close, 1) AS close_1 FROM cpt_dwc_2026_stock_hs300_snapshot`
- 截面算子，DAI提供的函数，一般情况下，`c_` 前缀的是时序算子，并且默认基于 `GROUP BY instrument` 计算，e.g. `SELECT date, instrument, close / m_lag(close, 1) AS close_1, c_rank(close_1) FROM cpt_dwc_2026_stock_hs300_snapshot`

## 计算资源

- 使用 `compression=True` 参数可降低内存占用：`dai.query("SELECT * FROM cpt_dwc_2026_stock_hs300_snapshot", filters={"date": ["2023-01-01 00:00:00", "2023-02-01 23:59:59"]}, compression=True).df()`。数据量很大，可以开启该参数（设为 True）后，系统会自动将 instrument 列的字符串类型转换为 category 类型，显著降低内存占用。在 dai sql 中查询和计算 pandas category 类型，可能会遇到不兼容问题，可以尝试用 `instrument::string` 转会为字符串。
- 资源规格：在 aistudio 状态栏里可以点击计算资源规格并切换，推荐 4C/16G 或者更高的资源规格。参赛后平台会赠送宽币用于升级计算资源，用户也可以通过参加比赛培训、[邀请用户](https://bigquant.com/spark)等获得更多宽币。

## DAI 文档

DAI（DATA FOR AI）是BigQuant研发的高性能分布式数据平台

* 使用简单：通过统一接口访问BigQuant各类数据。
* 数据丰富：提供PB级金融数据、另类投资数据和因子数据 (数据字典)，并支持用户自定义数据。
* 技术先进：采用现代化的分布式架构，支持大规模数据的低延迟读写和高性能计算。
* 使用文档访问链接：https://bigquant.com/wiki/doc/PLSbc1SbZX

## 其他数据

本次大赛除了提供 L2 snapshot 因子数据供参赛者挖掘因子，同时在评估系统中使用因子库数据和行业成分数据进行因子正交化，考量该因子带来的增量信息，以下为数据表链接

* 因子库数据，参考：[cpt_dwc_factorlib](https://deepwin.bigquant.com/data/datasources/cpt_dwc_factorlib)
* 行业成分数据，参考：[cpt_dwc_2026_stock_industry_component](https://deepwin.bigquant.com/data/datasources/cpt_dwc_2026_stock_industry_component)
* 代码ID映射表，参考：[all_instruments](https://deepwin.bigquant.com/data/datasources/all_instruments)
