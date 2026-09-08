## 表描述

A 股各证券的 **Level2 高频盘口快照**数据，包含 10 档买卖委托价、委托量、委托笔数，以及当日累计成交量额、加权平均委买/委卖价等。Level2 快照通常按交易所推送频率（约 3 秒）刷新，价格字段为**原始价格**（未做复权处理），跨除权日做长期序列分析时需结合复权因子使用。

## 字段

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 快照时间戳（含日期与时分秒，毫秒级） | pd.NaT |
| instrument | pd.StringDtype | 证券代码 | np.nan |
| trading_day | np.int32 | 交易日期（YYYYMMDD） | 0 |
| time | np.int32 | 当日时间（HHMMSSmmm，毫秒级） | 0 |
| pre_close | np.float32 | 前收盘价 | np.nan |
| open | np.float32 | 开盘价 | np.nan |
| high | np.float32 | 当日最高价 | np.nan |
| low | np.float32 | 当日最低价 | np.nan |
| price | np.float32 | 最新成交价 | np.nan |
| ask_price1 ~ ask_price10 | np.float32 | 1~10 档委卖价 | np.nan |
| ask_volume1 | np.int64 | 1 档委卖量（股） | 0 |
| ask_volume2 ~ ask_volume10 | np.int32 | 2~10 档委卖量（股） | 0 |
| bid_price1 ~ bid_price10 | np.float32 | 1~10 档委买价 | np.nan |
| bid_volume1 | np.int64 | 1 档委买量（股） | 0 |
| bid_volume2 ~ bid_volume10 | np.int32 | 2~10 档委买量（股） | 0 |
| ask_num_orders1 ~ ask_num_orders10 | np.int32 | 1~10 档委卖委托笔数 | 0 |
| bid_num_orders1 ~ bid_num_orders10 | np.int32 | 1~10 档委买委托笔数 | 0 |
| num_trades | np.int32 | 当日累计成交笔数 | 0 |
| volume | np.int64 | 当日累计成交量（股） | 0 |
| amount | np.float64 | 当日累计成交额（元） | np.nan |
| total_bid_volume | np.int64 | 委买总量（全档累计） | 0 |
| total_ask_volume | np.int64 | 委卖总量（全档累计） | 0 |
| bid_avg_price | np.float32 | 加权平均委买价 | np.nan |
| ask_avg_price | np.float32 | 加权平均委卖价 | np.nan |

> 注：`volume`、`amount`、`num_trades` 为**当日累计值**，跨快照单调非递减；要得到区间增量需用相邻快照差分。

## 分区说明

- 分区：每只股票交易日（`YYYY-MM-DD`）
- 每个分区文件包含该交易日全市场所有股票的全部快照
- 数据量较大（单日数千股 × 数千快照），推荐用法：按交易日 + 标的过滤，避免全表扫描
- 不推荐用法：跨多日全市场扫描（IO 与内存压力较大，建议分日并行处理）

## date 与 time 字段注意事项

`date` 字段存储的是带时分秒（毫秒级）的 datetime，而非纯日期。字符串 `'2026-05-01'` 会被解析为 `'2026-05-01 00:00:00'`，而 `00:00:00` 不是交易时间，因此以下写法**查不到任何数据**：

```sql
-- 错误：等价于 date = '2026-05-01 00:00:00'，无数据
WHERE date >= '2026-05-01' AND date <= '2026-05-01'
```

查询单日数据应使用半开区间，或使用 `trading_day` 整型字段：

```sql
-- 正确写法一：半开区间
WHERE date >= '2026-05-01' AND date < '2026-05-02'

-- 正确写法二：DATE() 函数截断时间部分
WHERE DATE(date) = '2026-05-01'

-- 正确写法三：直接用 trading_day（整型，最快）
WHERE trading_day = 20260501
```

`time` 字段是 `HHMMSSmmm` 格式的整型（如 09:30:01.500 → `93001500`），不能直接当字符串/时间比较，按时段过滤建议：

```sql
-- 09:30 ~ 10:00 的快照
WHERE time >= 93000000 AND time < 100000000
```

## 示例

查询 000001.SZ 在 2026-05-19 全天的 Level2 快照（含 5 档盘口）：

```python
import dai
dai.query("""
SELECT date, time, instrument, price, volume, amount,
       bid_price1, bid_price2, bid_price3, bid_price4, bid_price5,
       bid_volume1, bid_volume2, bid_volume3, bid_volume4, bid_volume5,
       ask_price1, ask_price2, ask_price3, ask_price4, ask_price5,
       ask_volume1, ask_volume2, ask_volume3, ask_volume4, ask_volume5
FROM cn_stock_level2_snapshot
WHERE trading_day = 20260519
  AND instrument = '000001.SZ'
ORDER BY time
""").df()
```

查询 2026-05-19 09:30 ~ 09:31 某只股票的一档盘口：

```python
import dai
dai.query("""
SELECT date, time, instrument, price,
       bid_price1, bid_volume1, ask_price1, ask_volume1
FROM cn_stock_level2_snapshot
WHERE trading_day = 20260519
  AND time >= 93000000 AND time < 93100000
  AND instrument = '000001.SZ'
ORDER BY instrument, time
""", filters).df()
```
