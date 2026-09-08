## 表描述

A 股常见宽基指数的成分股快照，由指数编制机构每日发布。记录指数代码、成分股代码及对应日期，可用于筛选特定指数的股票池。

## 字段

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 日期 | pd.NaT |
| instrument | pd.StringDtype | 指数代码 | np.nan |
| name | pd.StringDtype | 指数简称 | np.nan |
| member_code | pd.StringDtype | 成分股代码 | np.nan |
| member_name | pd.StringDtype | 成分股简称 | np.nan |

## 常用指数代码

| instrument | name |
|---|---|
| 000016.SH | 上证50 |
| 000300.SH | 沪深300 |
| 000510.SH | 中证A500 |
| 000688.SH | 科创50 |
| 000852.SH | 中证1000 |
| 000903.SH | 中证A100 |
| 000905.SH | 中证500 |
| 000985.CSI | 中证全指 |
| 399006.SZ | 创业板指 |
| 399303.SZ | 国证2000 |
| 899050.BJ | 北证50 |
| 932000.CSI | 中证2000 |

> 完整指数列表可通过 `SELECT DISTINCT instrument, name FROM cn_stock_index_component` 查询。

## 示例

查询中证1000 在 2026-05-18 的成分股：

```python
import dai
dai.query("""
SELECT date, member_code, member_name
FROM cn_stock_index_component
WHERE date = '2026-05-18'
  AND instrument = '000852.SH'
""").df()
```
