# 股票数据源

| 表名 | 适用场景 | 时间粒度 | 分区方式 | 详情 |
|---|---|---|---|---|
| `cn_stock_instruments` | 股票池、上市标的筛选 | 日频 | 年 | [表文档](./cn_stock_instruments.md) |
| `cn_stock_bar1d` | 后复权日行情 | 日频 | 年 | [表文档](./cn_stock_bar1d.md) |
| `cn_stock_bar1m_derived_c` | 后复权分钟行情及分钟末盘口 | 分钟 | 月 | [表文档](./cn_stock_bar1m_derived_c.md) |
| `cn_stock_level2_snapshot` | 高频盘口快照 | 毫秒级快照 | 交易日 | [表文档](./cn_stock_level2_snapshot.md) |

## 选择注意事项

- 日行情和分钟行情使用后复权价格；Level2 使用原始价格。
- 分钟表的 `date` 带时分秒，查询单日数据使用半开区间。
- Level2 优先按 `trading_day` 和 `instrument` 过滤，避免跨多日全市场扫描。
