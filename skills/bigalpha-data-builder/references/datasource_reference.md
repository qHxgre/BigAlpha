# 底层数据源参考

本 skill 已内置 BigAlpha 当前可用的底层数据表目录和完整表文档，路径均相对于本文件。数据源总入口为 [datasource/README.md](../datasource/README.md)。

## 使用顺序

1. 先读取 [datasource/README.md](../datasource/README.md)，按业务类别进入对应目录。
2. 再读取对应业务目录的 `README.md`，定位候选表。
3. 按需读取候选表的完整文档，确认字段、类型、默认值、日期语义、分区规则和示例 SQL。
4. 只有在完成上述核对后，才在 Step 1 中确认数据源，并在 Step 3 中编写 SQL。

不要一次性读取全部表文档。对于字段很多的财务表，先读业务目录入口，再用表文档中的字段标题或字段名搜索定位。

## 业务目录

| 目录 | 适用场景 | 入口 |
|---|---|---|
| `stock/` | 股票列表、日行情、分钟行情、Level2 | [README](../datasource/stock/README.md) |
| `index/` | 指数成分 | [README](../datasource/index/README.md) |
| `financial/` | 财务报表及财务衍生数据 | [README](../datasource/financial/README.md) |
| `industry/` | 行业分类与行业归属 | [README](../datasource/industry/README.md) |
| `factor/` | Barra 风格因子暴露 | [README](../datasource/factor/README.md) |

## 生成代码时的核对规则

* 只使用业务目录和完整表文档中明确出现的表名和字段名；字段不确定时先查文档，不要猜测。
* `date` 的含义必须写入架构确认：行情/行业/指数通常是交易日，财务表是公告日，Level2 是带时分秒的快照时间。
* 组合指数成分或行业分类时，明确连接键和过滤条件；行业表要考虑 `industry` 产生的多行记录，指数表要区分指数代码 `instrument` 与成分股 `member_code`。
* Level2 和分钟表数据量较大，必须按文档要求下推日期、交易日、时间或标的过滤，避免全表扫描。
* 财务表字段带有 `_lf`、`_ly`、`_mrq`、`_ttm` 口径后缀；先确认业务口径，再选择对应表名和字段。
* 示例 SQL 仅用于确认字段和查询习惯；实际 `build()` 仍必须传入与时间范围匹配的 `filters`。
