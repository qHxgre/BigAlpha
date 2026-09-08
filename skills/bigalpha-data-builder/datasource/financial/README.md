# 财务数据源

| 表名 | 口径 | 资产负债表 | 详情 |
|---|---|---|---|
| `cn_stock_financial_lf_shift` | 最新一期 | 有 | [共用表文档](./cn_stock_financial_shift.md) |
| `cn_stock_financial_ly_shift` | 最新一期年报 | 有 | [共用表文档](./cn_stock_financial_shift.md) |
| `cn_stock_financial_mrq_shift` | 单季度 | 无 | [共用表文档](./cn_stock_financial_shift.md) |
| `cn_stock_financial_ttm_shift` | 滚动十二个月 | 无 | [共用表文档](./cn_stock_financial_shift.md) |

## 选择注意事项

- `date` 是公告日，用于 Point-in-Time 查询；`report_date` 是报告期截止日。
- 同一 `(date, instrument)` 下通常有多个 `shift`，不能把它当作唯一粒度。
- 先确认财务口径，再选择对应表名和带后缀字段。
