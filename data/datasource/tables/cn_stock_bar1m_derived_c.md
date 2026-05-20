## 表描述

A 股各证券的**后复权**分钟行情，并附带每分钟收盘时刻的盘口快照（买一/卖一价量）。后复权价格 = 原价格 × 累计复权因子。

该表按**月份截面**（`_c` 后缀）分区存储，每个分区包含一个自然月内全市场所有股票的分钟数据，适合提取某段时间全市场截面数据，不适合长时间单股票序列查询。

## 字段

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 交易日期（含时分秒，实际值为当日 `HH:MM:SS` 形式，非 `00:00:00`） | pd.NaT |
| time | np.datetime64 | 分钟时间戳（精确到分钟，如 09:31） | pd.NaT |
| instrument | pd.StringDtype | 证券代码 | np.nan |
| open | np.double | 开盘价（后复权） | np.nan |
| close | np.double | 收盘价（后复权） | np.nan |
| high | np.double | 最高价（后复权） | np.nan |
| low | np.double | 最低价（后复权） | np.nan |
| volume | np.int64 | 成交量（股） | 0 |
| amount | np.double | 成交金额（元） | np.nan |
| deal_number | np.int32 | 成交笔数 | 0 |
| ask_price1 | np.double | 卖一价（后复权，分钟末盘口） | np.nan |
| ask_volume1 | np.int64 | 卖一量（股，分钟末盘口） | 0 |
| bid_price1 | np.double | 买一价（后复权，分钟末盘口） | np.nan |
| bid_volume1 | np.int64 | 买一量（股，分钟末盘口） | 0 |

## 分区说明

- 分区键：自然月（`YYYY-MM`）
- 每个分区文件包含该月全部交易日、全市场所有股票的分钟数据
- 推荐用法：按时间范围过滤，避免全表扫描
- 不推荐用法：仅查询单只股票的长时间序列（跨越大量分区，性能较差，建议改用按股票分区的表）

## date 字段注意事项

`date` 字段存储的是带时分秒的 datetime，而非纯日期。字符串 `'2026-05-01'` 会被解析为 `'2026-05-01 00:00:00'`，而 `00:00:00` 不是交易时间，因此以下写法**查不到任何数据**：

```sql
-- 错误：等价于 date = '2026-05-01 00:00:00'，无数据
WHERE date >= '2026-05-01' AND date <= '2026-05-01'
```

查询单日数据应使用半开区间，或用 `DATE()` 函数：

```sql
-- 正确写法一：半开区间
WHERE date >= '2026-05-01' AND date < '2026-05-02'

-- 正确写法二：DATE() 函数截断时间部分
WHERE DATE(date) = '2026-05-01'
```

## 示例

查询 2026-05-19 全市场所有股票 09:31 分钟的行情与盘口：

```python
import dai
dai.query("""
SELECT date, time, instrument, open, close, high, low, volume, amount,
       bid_price1, bid_volume1, ask_price1, ask_volume1
FROM cn_stock_bar1m_derived_c
WHERE date >= '2026-05-19' AND date < '2026-05-20'
  AND time = '2026-05-19 09:31:00'
ORDER BY instrument
""").df()
```
