# 底层数据源目录

按业务域查找数据源。先读取对应业务域的 `README.md`，再按需读取具体表文档。

| 业务域 | 适用场景 | 入口 |
|---|---|---|
| 股票 | 股票列表、日行情、分钟行情、Level2 | [stock/README.md](./stock/README.md) |
| 指数 | 指数成分股 | [index/README.md](./index/README.md) |
| 行业 | 行业分类与行业归属 | [industry/README.md](./industry/README.md) |
| 财务 | 财务报表及财务衍生数据 | [financial/README.md](./financial/README.md) |
| 因子 | Barra 风格因子暴露 | [factor/README.md](./factor/README.md) |

## 使用规则

1. 先按业务场景选择一个业务域。
2. 读取该业务域的 `README.md`，定位候选表。
3. 再读取候选表的完整文档，确认字段、粒度、日期语义、分区和示例 SQL。
4. 只有完成上述核对后，才能在架构确认和 Builder SQL 中使用该表。

不要一次性读取全部表文档。字段很多的财务表，先读业务域入口，再按字段名定位具体章节。
