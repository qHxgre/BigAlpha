## 数据概览

本比赛仅允许使用赛事平台明确开放的数据源构建因子。比赛提供沪深 300 历史成分股的 **1 分钟 K 线数据（带盘口）**。参赛者应使用分钟级价格、成交和订单簿信息，生成平台规定采样时点上的 15 分钟频率因子。

平台评测程序将导入参赛代码中的入口函数，并传入数据源名称、开始时间和结束时间运行。提交内容必须是能够从指定数据源完整生成因子结果的代码，不得提交预先计算的离线因子值。

## 数据范围

- **股票池**：沪深 300 指数历史成分股，以各交易日实际成分为准。
- **时间范围**：2024 年 1 月 1 日至 2025 年 12 月 31 日。
- **数据频率**：1 分钟。
- **行情类型**：分钟 K 线及该分钟结束时点的十档盘口信息。
- **交易市场**：沪深 A 股市场。
- **因子频率**：参赛者基于 1 分钟数据构建 15 分钟频率因子。

正式数据表名称、数据链接、因子采样时点和评估标签口径，以比赛空间上线后的数据字典及赛事公告为准。下文示例使用 `cpt_gifp_2026_stock_bar1m` 作为分钟行情表名。

## 1 分钟 K 线及盘口数据

### 基础字段

| 字段名 | 建议类型 | 描述 |
|---|---|---|
| `date` | datetime | Bar 结束时刻，精确到分钟 |
| `instrument` | string | 股票代码，如 `000001.SZ`、`600000.SH` |
| `instrument_id` | integer | 股票代码的整数 ID，用于降低内存占用（如平台提供） |
| `adjust_factor` | float | 复权因子 |
| `pre_close` | float | 前一交易日收盘价 |
| `open` | float | 该分钟开盘价 |
| `high` | float | 该分钟最高价 |
| `low` | float | 该分钟最低价 |
| `close` | float | 该分钟收盘价 |
| `volume` | integer/float | 该分钟成交量 |
| `amount` | float | 该分钟成交额 |
| `deal_number` | integer | 该分钟成交笔数 |

### 十档盘口字段

盘口字段包含买卖各 10 档的委托价格、委托量和委托笔数。`N` 的取值为 1—10，其中第 1 档为最优报价。

| 字段模式 | 建议类型 | 描述 |
|---|---|---|
| `ask_priceN` | float | 卖 N 档委托价格 |
| `bid_priceN` | float | 买 N 档委托价格 |
| `ask_volumeN` | integer/float | 卖 N 档委托量 |
| `bid_volumeN` | integer/float | 买 N 档委托量 |
| `ask_num_ordersN` | integer | 卖 N 档委托笔数 |
| `bid_num_ordersN` | integer | 买 N 档委托笔数 |

例如：`ask_price1` 表示卖一价，`bid_volume5` 表示买五档委托量，`ask_num_orders10` 表示卖十档委托笔数。

> 盘口为对应 1 分钟 Bar 结束时点的快照。参赛者不得将当前 Bar 结束后才产生的信息用于当前时点因子计算。停牌、涨跌停或流动性不足时，部分盘口字段可能为空或为特殊值，应按照平台数据字典处理。

## 辅助数据

比赛空间可根据最终评估需要提供以下辅助表，具体以正式数据字典为准：

- **历史成分股表**：记录沪深 300 在各历史时点的有效成分股，用于确定合法股票池。
- **风险暴露表**：提供平台统一使用的 BARRA 风险暴露，主要用于评估阶段的风格中性化。
- **未来收益标签表**：供平台评估使用。为防止未来信息泄露，该表可能不向参赛者开放全部区间。

未经赛事公告明确授权，参赛者不得使用比赛空间之外的数据源。

## 数据读取示例

```python
import dai

bars = dai.query(
    """
    SELECT *
    FROM cpt_gifp_2026_stock_bar1m
    ORDER BY date, instrument
    """,
    filters={
        "date": ["2024-01-02 00:00:00", "2024-01-05 23:59:59"]
    },
    compression=True,
).df()

bars.head()
```

建议在研究阶段只读取实际需要的列和日期区间：

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

## 因子计算示例

以下示例使用盘口不平衡度构建演示因子，并按 15 分钟频率输出。示例仅用于说明数据读取和输出格式，不代表有效投资策略；正式采样时点以平台公告为准。

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
    )
    SELECT
        time_bucket(INTERVAL '15 minutes', date) + INTERVAL '15 minutes' AS date,
        instrument,
        AVG(order_imbalance) AS factor
    FROM base
    GROUP BY 1, 2
    ORDER BY 1, 2
    """,
    filters={
        "date": ["2024-01-02 00:00:00", "2024-02-01 23:59:59"]
    },
    compression=True,
).df()

factor.head()
```

因子函数的最终返回值必须且仅包含：

| date | instrument | factor |
|---|---|---:|
| 2024-01-02 10:00:00 | 000001.SZ | 0.0521 |
| 2024-01-02 10:00:00 | 000002.SZ | -0.1184 |

## 时间对齐要求

- `date` 表示信息可用时点，而不是预测区间结束时点。
- 时点 `t` 的因子只能使用 `t` 时刻及以前已经产生的数据。
- 不得使用 `shift(-1)`、未来窗口、全样本归一化或其他会引入未来信息的处理。
- 跨日计算时，应明确是否允许使用上一交易日数据；不得把下一交易日信息回填到前一交易日。
- 股票池必须按历史时点匹配，不得使用当前沪深 300 成分股回填整个历史区间。
- 集合竞价、午间休市、停牌、异常交易状态及非标准 Bar 的处理，以平台正式采样规则为准。

## 数据质量与缺失值

- 新股、停牌股票、涨跌停股票及低流动性股票可能存在缺失 Bar 或盘口不完整情况。
- 盘口价格为 0、委托量为 0 或字段为空时，应先判断其业务含义，不应无条件填充。
- 缺失值、异常值和极端值的处理方法必须统一、可解释且可复现。
- 不得为了通过覆盖率检查而伪造因子值；合理填充也不得使用未来截面或未来时间的信息。
- 每个评估截面的因子缺失率不得高于 40%。

## 性能建议

- 优先在 SQL 中筛选日期和字段，避免直接加载全量两年数据。
- 使用 `compression=True` 降低字符串型股票代码的内存占用。
- 可按月或按交易日分批计算，但必须保证滚动窗口在分区边界处连续、结果一致。
- 对相同中间结果合理复用，避免重复扫描分钟表。
- 使用 Pandas、Polars 或其他计算库时，应关注排序、数据类型和内存峰值。
- 推荐先用少量日期完成正确性验证，再扩展到完整数据区间。

## DAI 参考

DAI 是 BigQuant 提供的高性能数据与计算引擎，支持 SQL、时序算子、截面算子、UDF，以及 Pandas、Polars、Arrow 等数据格式。平台正式开放后，请以比赛空间中的数据字典、模板 Notebook 和 DAI 文档为准。
