
## cn_stock_financial_lf_shift / ly_shift / mrq_shift / ttm_shift

四张表结构相同，仅财务口径不同，共用同一份文档。

| 表名 | 字段后缀 | 口径 | 含资产负债表 |
|---|---|---|---|
| cn_stock_financial_lf_shift | `_lf` | 最新一期（Latest Filing） | 是 |
| cn_stock_financial_ly_shift | `_ly` | 最新一期年报（Latest Year） | 是 |
| cn_stock_financial_mrq_shift | `_mrq` | 单季度（Most Recent Quarter） | 否 |
| cn_stock_financial_ttm_shift | `_ttm` | 滚动十二个月（Trailing Twelve Months） | 否 |

- **公共字段**：`date`（公告日，PIT 基准）/ `instrument` / `report_date` / `shift`（报告期偏移，0=最新期）
- **完整文档**：`data/datasource/tables/cn_stock_financial_shift.md`
