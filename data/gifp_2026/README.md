# GIFP 2026 数据构建

本目录用于构建“智量未来”全国高校 AI 量化投资挑战赛的数据表。

## 数据口径

- 股票池：沪深 300 历史成分股（`000300.SH`）
- 时间范围：2024-01-01 至 2025-12-31
- 行情频率：1 分钟
- 行情内容：OHLC、区间成交统计及十档盘口快照
- 原始行情：`cn_stock_level2_snapshot`

## 目标数据表

| 数据表 | 构建代码 | 说明 |
|---|---|---|
| `cpt_gifp_2026_stock_bar1m` | `cpt_gifp_2026_stock_bar1m` | 沪深 300 历史成分股 1 分钟 K 线及十档盘口 |
| `cpt_gifp_2026_instruments` | `cpt_gifp_2026_instruments` | 沪深 300 历史成分股的 15 分钟截面 |
| `cpt_gifp_2026_exposure` | `cpt_gifp_2026_exposure` | 沪深 300 每日 BARRA 风格与行业暴露 |
| `cpt_gifp_2026_return` | `cpt_gifp_2026_return` | 下一交易日同一分钟的 15 分钟频率复权收益标签 |

## 收益标签口径

- 采样时点：09:45—11:30、13:15—15:00 的 15 分钟端点。
- 预测区间：`(t 日采样时点, t+1 交易日同一时点]`，周末和节假日顺延。
- 基准价格：起止时点对应 1 分钟 Bar 的收盘价；15:00 Bar 接纳
  `[14:59:00, 15:01:00)` 内到达的收盘快照。
- 收益公式：`future_close * future_adjust_factor / (close * adjust_factor) - 1`，
  使用累计后复权因子处理现金分红、送转和拆并股。
- 停牌或起止价格、复权因子缺失时不生成该条标签。

## 推荐构建顺序

1. `cpt_gifp_2026_instruments`
2. `cpt_gifp_2026_exposure`
3. `cpt_gifp_2026_stock_bar1m`
4. `cpt_gifp_2026_return`

各目录中的 `running.ipynb` 提供了默认日期范围的运行入口。
