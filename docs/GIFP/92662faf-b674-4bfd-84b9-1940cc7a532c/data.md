## 数据概览

比赛数据覆盖沪深 300（`000300.SH`）历史成分股，时间范围为 **2021 年 1 月 1 日至 2025 年 12 月 31 日**。

参赛者只能使用赛事平台明确开放并授权的数据源。各表的实际可见范围以比赛空间的数据权限为准，未授权的评估标签不得读取、推断或用于因子计算。

## 数据表

| 数据表 | 频率 | 说明 |
|---|---|---|
| `cpt_gifp_2026_stock_bar1m` | 1 分钟 | 沪深 300 历史成分股的 K 线、区间成交统计及十档盘口 |
| `cpt_gifp_2026_instruments` | 15 分钟 | 每个合法因子时点的沪深 300 历史成分股截面 |
| `cpt_gifp_2026_exposure` | 日频 | 沪深 300 历史成分股的 BARRA 风格与行业暴露 |
| `cpt_gifp_2026_return` | 15 分钟 | 下一交易日同一分钟的复权收益标签，主要用于平台评估 |

数据表链接：

* cpt_gifp_2026_stock_bar1m：https://gifp.bigquant.com/data/datasources/cpt_gifp_2026_stock_bar1m
* cpt_gifp_2026_instruments：https://gifp.bigquant.com/data/datasources/cpt_gifp_2026_instruments
* cpt_gifp_2026_exposure：https://gifp.bigquant.com/data/datasources/cpt_gifp_2026_exposure
* cpt_gifp_2026_return：https://gifp.bigquant.com/data/datasources/cpt_gifp_2026_return


合法的 15 分钟采样时点共 16 个：

```text
09:45、10:00、10:15、10:30、10:45、11:00、11:15、11:30
13:15、13:30、13:45、14:00、14:15、14:30、14:45、15:00
```

09:30 和 13:00 不纳入采样，因为此时尚未形成对应连续竞价时段内的首个完整 15 分钟信息窗口。

## 1 分钟行情表

### `cpt_gifp_2026_stock_bar1m`

每个交易日只保留当日有效的沪深 300 成分股，主键为 `date` 和 `instrument`。

#### 基础字段

| 字段名 | 类型 | 描述 |
|---|---|---|
| `date` | datetime | 1 分钟 Bar 结束时刻 |
| `instrument` | string | 股票代码，如 `000001.SZ`、`600000.SH` |
| `instrument_id` | int16 | 股票代码的整数 ID |
| `adjust_factor` | float32 | 当日累计后复权因子 |
| `pre_close` | float32 | 前一交易日收盘价 |
| `open` | float32 | 该分钟第一笔有效成交价 |
| `high` | float32 | 该分钟最高有效成交价 |
| `low` | float32 | 该分钟最低有效成交价 |
| `close` | float32 | 该分钟最后一笔有效成交价 |
| `deal_number` | int64 | 该分钟成交笔数 |
| `volume` | int64 | 该分钟成交量 |
| `amount` | float64 | 该分钟成交额 |

成交笔数、成交量和成交额由日内累计值的相邻快照差分后按分钟汇总。盘口字段取该分钟最后一条快照。

#### 十档盘口字段

下表中的 `N` 取 1—5，第 1 档为最优报价。

| 字段模式 | 类型 | 描述 |
|---|---|---|
| `ask_priceN` | float32 | 卖 N 档委托价格 |
| `bid_priceN` | float32 | 买 N 档委托价格 |
| `ask_volumeN` | int64 | 卖 N 档委托量 |
| `bid_volumeN` | int64 | 买 N 档委托量 |
| `ask_num_ordersN` | int32 | 卖 N 档委托笔数 |
| `bid_num_ordersN` | int32 | 买 N 档委托笔数 |

例如，`ask_price1` 表示卖一价，`bid_volume5` 表示买五档委托量，`ask_num_orders10` 表示卖十档委托笔数。

#### 分钟时间戳口径

- 连续竞价快照按左闭右开区间映射：`[HH:MM:00, HH:MM+1:00)` 标记为下一分钟的 Bar 结束时刻。
- 09:15—09:25 的集合竞价快照按分钟保留，09:25:00 的撮合快照归入 09:25 Bar。
- 上午和下午最后一分钟允许接纳收盘时点后一分钟内到达的快照。
- 15:00 Bar 接纳 `[14:59:00, 15:01:00)` 内的快照，并使用最后一笔有效成交价作为收盘价；15:01:00 及之后的数据不会进入当日收盘 Bar。

盘口价格或数量为 0，可能表示对应档位不存在，不应无条件用前值、均值或未来值填充。

## 15 分钟股票池表

### `cpt_gifp_2026_instruments`

该表将每个交易日实际有效的沪深 300 成分股展开到上述 16 个采样时点，用于确定每个因子截面的合法股票池。主键为 `date` 和 `instrument`。

| 字段名 | 类型 | 描述 |
|---|---|---|
| `date` | datetime | 15 分钟因子计算时点 |
| `instrument` | string | 股票代码 |
| `name` | string | 证券简称 |

参赛因子应与该表按 `date`、`instrument` 对齐，不得使用当前成分股名单回填历史股票池。

## 风险暴露表

### `cpt_gifp_2026_exposure`

该表提供沪深 300 当日成分股的日频 BARRA 风格因子、一级行业和评估辅助字段，主键为 `date` 和 `instrument`。日频暴露与分钟因子连接时，应遵守信息实际可得时间，避免把盘后或未来数据用于当日盘中因子。

#### 标识及辅助字段

| 字段名 | 类型 | 描述 |
|---|---|---|
| `date` | datetime | 交易日 |
| `instrument` | string | 股票代码 |
| `industry_level1_code` | string | 一级行业代码 |
| `float_market_cap` | float64 | 流通市值 |
| `weights` | float64 | 截面归一化市值权重 |
| `ret` | float64 | 当期收益率 |

#### 风格因子字段

| 字段名 | 描述 |
|---|---|
| `SIZE` | 市值 |
| `BETA` | 贝塔 |
| `MOMENTUM` | 传统动量 |
| `RESVOL` | 残差波动率 |
| `SIZENL` | 非线性市值 |
| `BTOP` | 账面市值比 |
| `LIQUIDTY` | 流动性 |
| `EARNYILD` | 盈利能力 |
| `GROWTH` | 成长 |
| `LEVERAGE` | 杠杆 |

以上风格因子字段类型均为 `float64`。字段名 `LIQUIDTY`、`EARNYILD` 采用数据表中的实际拼写。

#### 行业哑变量字段

行业字段类型均为 `int8`，取值用于标识股票所属一级行业。

| 字段名 | 行业 | 字段名 | 行业 |
|---|---|---|---|
| `AGRIFOREST` | 农林牧渔 | `MINING` | 采掘 |
| `CHEM` | 化工 | `IRONSTEEL` | 钢铁 |
| `NONFERMETAL` | 有色金属 | `ELECTRONICS` | 电子 |
| `AUTO` | 汽车 | `HOUSEAPP` | 家用电气 |
| `FOODBEVER` | 食品饮料 | `TEXTILE` | 纺织服饰 |
| `LIGHTINDUS` | 轻工制造 | `HEALTH` | 医药生物 |
| `UTILITIES` | 公用事业 | `TRANSPORTATION` | 交通运输 |
| `REALESTATE` | 房地产 | `COMMETRADE` | 商业贸易 |
| `LEISERVICE` | 休闲服务 | `BANK` | 银行 |
| `NONBANKFINAN` | 非银金融 | `CONGLOMERATES` | 综合 |
| `CONMAT` | 建筑材料 | `BUILDDECO` | 建筑装饰 |
| `ELECEQP` | 电气设备 | `MACHIEQUIP` | 机械设备 |
| `AERODEF` | 国防军工 | `COMPUTER` | 计算机 |
| `MEDIA` | 传媒 | `TELECOM` | 通信 |
| `COAL` | 煤炭 | `PETRO` | 石油石化 |
| `ENVP` | 环保 | `BEAUTY` | 美容护理 |

## 收益标签表

### `cpt_gifp_2026_return`

该表记录 16 个采样时点上，股票从当日该时点到下一实际交易日同一时点的复权收益。下一交易日按全市场实际交易日确定，会自然跨过周末和节假日；不会按自然日简单加一天，也不会因为单只股票停牌而跨越到其下一条可用记录。

| 字段名 | 类型 | 描述 |
|---|---|---|
| `date` | datetime | 收益起点，即当日 15 分钟采样时点 |
| `instrument` | string | 股票代码 |
| `return_rate` | float32 | 下一交易日同分钟复权收益率 |
| `close` | float32 | 起点分钟收盘价 |
| `adjust_factor` | float32 | 起点日累计后复权因子 |
| `future_date` | datetime | 下一交易日同一采样时点 |
| `future_close` | float32 | 终点分钟收盘价 |
| `future_adjust_factor` | float32 | 终点日累计后复权因子 |

预测区间为：

```text
(t 日采样时点, t+1 交易日同一时点]
```

收益公式为：

```text
return_rate =
    future_close * future_adjust_factor
    / (close * adjust_factor)
    - 1
```

累计后复权因子用于修正现金分红、送转和拆并股造成的机械价格跳变。若股票在起点或下一交易日终点没有对应分钟 Bar，或任一价格、复权因子缺失或无效，则不生成该条标签；不会使用附近分钟价格填充。

## 数据读取示例

建议只读取实际需要的字段和日期范围，并使用 `compression=True` 降低股票代码列的内存占用。

```python
import dai

bars = dai.query(
    """
    SELECT
        date,
        instrument,
        open,
        high,
        low,
        close,
        volume,
        amount,
        ask_price1,
        bid_price1,
        ask_volume1,
        bid_volume1
    FROM cpt_gifp_2026_stock_bar1m
    ORDER BY date, instrument
    """,
    filters={
        "date": ["2024-01-02 00:00:00", "2024-02-01 23:59:59"]
    },
    compression=True,
).df()
```

读取合法因子截面：

```python
sections = dai.query(
    """
    SELECT date, instrument, name
    FROM cpt_gifp_2026_instruments
    ORDER BY date, instrument
    """,
    filters={
        "date": ["2024-01-02 00:00:00", "2024-01-05 23:59:59"]
    },
    compression=True,
).df()
```

## 因子计算示例

以下示例用前三档盘口不平衡度构建演示因子，并只保留合法的 15 分钟截面。示例仅用于说明读取、聚合和输出格式，不代表有效投资策略。

```python
import dai

factor = dai.query(
    """
    WITH base AS (
        SELECT
            date,
            instrument,
            (
                bid_volume1 + bid_volume2 + bid_volume3
                - ask_volume1 - ask_volume2 - ask_volume3
            ) / NULLIF(
                bid_volume1 + bid_volume2 + bid_volume3
                + ask_volume1 + ask_volume2 + ask_volume3,
                0
            ) AS order_imbalance
        FROM cpt_gifp_2026_stock_bar1m
    ),
    aggregated AS (
        SELECT
            time_bucket(
                INTERVAL '15 minutes',
                date - INTERVAL '1 microsecond'
            ) + INTERVAL '15 minutes' AS date,
            instrument,
            AVG(order_imbalance) AS factor
        FROM base
        GROUP BY 1, 2
    )
    SELECT a.date, a.instrument, a.factor
    FROM aggregated a
    INNER JOIN cpt_gifp_2026_instruments i
        ON a.date = i.date
        AND a.instrument = i.instrument
    ORDER BY a.date, a.instrument
    """,
    filters={
        "date": ["2024-01-02 00:00:00", "2024-02-01 23:59:59"]
    },
    compression=True,
).df()
```

最终返回值必须且仅包含以下三列：

| date | instrument | factor |
|---|---|---:|
| 2024-01-02 10:00:00 | 000001.SZ | 0.0521 |
| 2024-01-02 10:00:00 | 000002.SZ | -0.1184 |

## 时间对齐与数据质量

- `date` 表示信息可用时点，而不是预测区间结束时点；时点 `t` 的因子只能使用 `t` 及以前已经产生且实际可获得的数据。
- 不得使用 `shift(-1)`、居中窗口、未来区间聚合、全样本归一化或其他会引入未来信息的处理。
- 股票池必须按历史时点匹配，不得使用当前沪深 300 成分股回填完整历史区间。
- 新股、停牌、涨跌停及低流动性股票可能缺少 Bar，盘口也可能不完整；缺失值与异常值处理必须统一、可解释、可复现。
- 不得为满足覆盖率要求而伪造因子值，也不得使用未来截面或未来时间的信息填充。
- 每个评估截面的因子缺失率不得高于 40%。

## 性能建议

- 优先在 SQL 中筛选日期和字段，避免一次加载两年的全量分钟数据。
- 可按月或按交易日分批计算，但应为滚动窗口预留足够的前置数据，保证分区边界结果连续一致。
- 对相同中间结果合理复用，避免重复扫描分钟表。
- 使用 Pandas、Polars 或其他计算库时，应关注排序、数据类型和内存峰值。
- 先用少量日期验证字段、时间对齐、股票池覆盖和无未来函数，再扩展到完整区间。

## DAI 参考

DAI 是 BigQuant 提供的数据与计算引擎，支持 SQL、时序与截面算子、UDF，以及 Pandas、Polars、Arrow 等数据格式。实际使用时以比赛空间内的数据权限、数据字典、模板 Notebook 和赛事公告为准。

DAI 使用文档链接：https://gifp.bigquant.com/wiki/doc/PLSbc1SbZX?space_id=65296e4f-e6e2-4ff1-8c5d-be1e36f80995