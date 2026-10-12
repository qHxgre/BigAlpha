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
| `cpt_gifp_2026_instruments` | `cpt_gifp_2026_instruments` | 沪深 300 每日历史成分股列表 |
| `cpt_gifp_2026_exposure` | `cpt_gifp_2026_exposure` | 沪深 300 每日 BARRA 风格与行业暴露 |

## 暂未构建

收益率标签的预测窗口、基准价格、采样时点和跨午休处理尚未确定，因此当前不创建标签表及相应构建代码。待标签口径确认后，再单独增加标签构建器，并根据标签采样时点决定是否将 `instruments` 和 `exposure` 展开到日内截面。

## 推荐构建顺序

1. `cpt_gifp_2026_instruments`
2. `cpt_gifp_2026_exposure`
3. `cpt_gifp_2026_stock_bar1m`

各目录中的 `running.ipynb` 提供了默认日期范围的运行入口。
