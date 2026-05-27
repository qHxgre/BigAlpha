## 表描述

A 股各证券的行业分类快照，按交易日记录每只股票在某一套行业标准下的最具体行业及一/二/三级行业代码与名称。可用于行业归属查询、行业内选股、行业中性化、行业轮动等。同一证券在同一日期下可能有多条记录，对应不同的 `industry`（行业标准）。

## 字段

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 日期 | pd.NaT |
| instrument | pd.StringDtype | 证券代码 | np.nan |
| industry | pd.StringDtype | 行业标准 | np.nan |
| industry_name | pd.StringDtype | 行业简称（最具体） | np.nan |
| industry_instrument | pd.StringDtype | 行业代码（最具体） | np.nan |
| industry_level1_code | pd.StringDtype | 一级行业代码 | np.nan |
| industry_level1_name | pd.StringDtype | 一级行业名称 | np.nan |
| industry_level2_code | pd.StringDtype | 二级行业代码 | np.nan |
| industry_level2_name | pd.StringDtype | 二级行业名称 | np.nan |
| industry_level3_code | pd.StringDtype | 三级行业代码 | np.nan |
| industry_level3_name | pd.StringDtype | 三级行业名称 | np.nan |

## 字段说明

- `industry`：行业分类标准的标识（如申万 2014、申万 2021、中信等），同一证券在不同标准下会拆分为多行。
- `industry_name` / `industry_instrument`：当前证券所属的**最具体**一级（通常等于三级，缺失时回退到上一级）的行业名称与代码，便于直接取用，无需再判空。
- `industry_levelN_code` / `industry_levelN_name`：标准化的三层级树形结构。若某标准只有两级，`level3_*` 为空。

> 完整行业标准列表可通过 `SELECT DISTINCT industry FROM cn_stock_industry_component` 查询。

## 示例

查询 000001.SZ 在 2026-05-20 的全部行业分类：

```python
import dai
dai.query("""
SELECT date, instrument, industry, industry_name,
       industry_level1_name, industry_level2_name, industry_level3_name
FROM cn_stock_industry_component
WHERE date = '2026-05-20'
  AND instrument = '000001.SZ'
""").df()
```

按申万 2021 一级行业筛选 2026-05-20 的银行股：

```python
import dai
dai.query("""
SELECT date, instrument, industry_level1_code, industry_level1_name
FROM cn_stock_industry_component
WHERE date = '2026-05-20'
  AND industry = 'sw2021'
  AND industry_level1_name = '银行'
ORDER BY instrument
""").df()
```
