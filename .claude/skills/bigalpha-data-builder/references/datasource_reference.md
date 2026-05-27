# 底层数据源参考文档

当 `build()` 中需要从 BigQuant 底层表抽数时，按以下两级策略查阅。

## 文档根路径

`/Users/xiehao/Desktop/workspace/BigQuant/BigAlpha/data/datasource`

## 第一级：类别索引文件（轻量，优先读）

每个类别文件包含：表名、中文名、一句话描述、字段名列表、完整文档路径。

| 文件 | 适用场景 |
|---|---|
| `datasource_stock.md` | 股票行情、基本信息等 |
| `datasource_index.md` | 指数成分、指数行情 |
| `datasource_financial.md` | 三大报表、财务衍生指标 |
| `datasource_industry.md` | 行业分类、行业指数 |
| `datasource_factor.md` | 股票因子、风险因子 |

## 第二级：完整表文档（按需读）

位于 `tables/` 子目录，每张表一个文件，包含字段类型、默认值、示例 SQL。

| 文件 | 对应表 |
|---|---|
| `tables/cn_stock_bar1d.md` | cn_stock_bar1d |
| `tables/cn_stock_index_component.md` | cn_stock_index_component |

## 使用规则

1. **Step 1 确认数据源**：Read 对应类别文件，通过字段名列表判断是否满足需求。
2. **Step 3 编写 SQL**：若需确认字段类型或查看示例，再 Read `tables/` 下的完整文档。
3. 若用户描述模糊，先提示其参考类别文件，不要凭空构造字段名。
