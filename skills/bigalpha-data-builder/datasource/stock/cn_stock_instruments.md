## 表描述

每日股票列表，支持 A 股。记录每个交易日的全市场证券代码、简称及证券类型，可用于获取某一日期的全部上市标的，或筛选特定类型的证券。

## 字段

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 日期 | pd.NaT |
| instrument | pd.StringDtype | 证券代码 | np.nan |
| name | pd.StringDtype | 证券简称 | np.nan |
| type | pd.StringDtype | 证券类型 | np.nan |

## 示例

查询 2026-05-20 的全市场股票列表：

```python
import dai
dai.query("""
SELECT date, instrument, name, type
FROM cn_stock_instruments
WHERE date = '2026-05-20'
ORDER BY instrument
""").df()
```
