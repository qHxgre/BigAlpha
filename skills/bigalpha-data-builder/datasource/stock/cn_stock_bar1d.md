## 表描述

A 股各证券的**后复权**日行情。后复权将分红、配股等事件按复权因子折算到历史价格，使长期趋势具有可比性。后复权价格 = 原价格 × 累计复权因子。

## 字段

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 日期 | np.nan |
| instrument | pd.StringDtype | 证券代码 | np.nan |
| name | pd.StringDtype | 证券简称 | np.nan |
| adjust_factor | np.double | 累计后复权因子 | np.nan |
| pre_close | np.double | 昨收盘价（后复权） | np.nan |
| open | np.double | 开盘价（后复权） | np.nan |
| close | np.double | 收盘价（后复权） | np.nan |
| high | np.double | 最高价（后复权） | np.nan |
| low | np.double | 最低价（后复权） | np.nan |
| volume | np.int64 | 成交量 | 0 |
| deal_number | np.int32 | 成交笔数 | 0 |
| amount | np.double | 成交金额 | np.nan |
| change_ratio | np.double | 涨跌幅（后复权） | 0 |
| turn | np.double | 换手率 | 0 |
| upper_limit | np.double | 涨停价 | np.nan |
| lower_limit | np.double | 跌停价 | np.nan |

## 示例

查询 000001.SZ 在 2026-05-01 至 2026-05-20 的日行情：

```python
import dai
dai.query("""
SELECT date, instrument, open, close, high, low, volume, amount, change_ratio
FROM cn_stock_bar1d
WHERE date >= '2026-05-01'
  AND date <= '2026-05-20'
  AND instrument = '000001.SZ'
ORDER BY date
""").df()
```
