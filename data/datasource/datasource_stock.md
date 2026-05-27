# 股票数据源


## cn_stock_instruments

- **中文名**：每日股票列表
- **描述**：A 股每日全市场证券列表，记录每个交易日的证券代码、简称及证券类型。
- **字段**：`date` / `instrument` / `name` / `type`
- **完整文档**：`data/datasource/tables/cn_stock_instruments.md`


## cn_stock_bar1d

- **中文名**：股票后复权日行情
- **描述**：A 股各证券的日行情数据，采用后复权价格，消除分红/配股引起的价格跳变。
- **字段**：`date` / `instrument` / `name` / `adjust_factor` / `pre_close` / `open` / `close` / `high` / `low` / `volume` / `deal_number` / `amount` / `change_ratio` / `turn` / `upper_limit` / `lower_limit`
- **完整文档**：`data/datasource/tables/cn_stock_bar1d.md`

## cn_stock_bar1m_derived_c

- **中文名**：股票后复权1分钟行情，包含该分钟最后一刻的截面盘口数据
- **分区规则**：按月分区
- **描述**：A 股各证券的分钟行情数据，采用后复权价格，并附带每分钟收盘时刻的买一/卖一盘口快照。按自然月截面分区，适合提取某段时间全市场所有股票的分钟数据。
- **字段**：`date` / `time` / `instrument` / `open` / `close` / `high` / `low` / `volume` / `amount` / `deal_number` / `ask_price1` / `ask_volume1` / `bid_price1` / `bid_volume1`
- **完整文档**：`data/datasource/tables/cn_stock_bar1m_derived_c.md`

## cn_stock_level2_snapshot

- **中文名**：股票L2快照数据
- **描述**：
- **字段**：
- **完整文档**：`data/datasource/tables/cn_stock_level2_snapshot.md`