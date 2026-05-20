# 股票数据源

## cn_stock_bar1m_derived_c

- **中文名**：股票后复权分钟行情（含盘口，月截面分区）
- **描述**：A 股各证券的分钟行情数据，采用后复权价格，并附带每分钟收盘时刻的买一/卖一盘口快照。按自然月截面分区，适合提取某段时间全市场所有股票的分钟数据。
- **字段**：`date` / `time` / `instrument` / `open` / `close` / `high` / `low` / `volume` / `amount` / `deal_number` / `ask_price1` / `ask_volume1` / `bid_price1` / `bid_volume1`
- **完整文档**：`data/datasource/tables/cn_stock_bar1m_derived_c.md`

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

## cn_stock_bar1d

- **中文名**：股票后复权日行情
- **描述**：A 股各证券的日行情数据，采用后复权价格，消除分红/配股引起的价格跳变。
- **字段**：`date` / `instrument` / `name` / `adjust_factor` / `pre_close` / `open` / `close` / `high` / `low` / `volume` / `deal_number` / `amount` / `change_ratio` / `turn` / `upper_limit` / `lower_limit`
- **完整文档**：`data/datasource/tables/cn_stock_bar1d.md`
