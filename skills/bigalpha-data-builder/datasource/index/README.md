# 指数数据源

| 表名 | 适用场景 | 粒度 | 详情 |
|---|---|---|---|
| `cn_stock_index_component` | 指数成分股筛选 | 日期 × 指数 × 成分股 | [表文档](./cn_stock_index_component.md) |

## 选择注意事项

- `instrument` 是指数代码，`member_code` 是成分股代码。
- 与行情表连接时，使用 `member_code` 连接目标股票的 `instrument`。
- 先按指数代码和日期过滤，再与行情数据连接。
