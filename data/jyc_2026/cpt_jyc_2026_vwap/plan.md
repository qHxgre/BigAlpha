# 未来 30 分钟 VWAP 收益数据构建方案

## 数据范围

从 `cn_stock_level2_snapshot` 读取快照数据，并通过
`cn_stock_index_component` 筛选当日中证 2000（`932000.CSI`）历史成分股。
使用字段：`date`、`instrument`、`price`、`volume`、`amount`、`num_trades`。

## 构建方法

1. 按股票、交易日和快照时间排序。
2. `volume`、`amount`、`num_trades` 为日内累计值，先按股票和交易日做相邻快照差分，得到单条快照的成交增量；若差分为负，视为累计值重置，使用当前累计值作为增量。
3. 将成交增量映射至左开右闭的 30 分钟窗口，并分别汇总窗口成交量、成交额和成交笔数。
4. 以窗口成交额除以窗口成交量得到 VWAP；取窗口内最后一个大于 0 的 `price` 作为终点价格。
5. 计算收益率：

```text
vwap = window_amount / window_volume
vwap_return = end_price / vwap - 1
```

## 时间窗口

每日生成 8 个信号截面：`09:30`、`10:00`、`10:30`、`11:00`、`11:30`、
`13:30`、`14:00`、`14:30`。其中 `11:30` 信号跳过午休，对应
`(13:00, 13:30]`；其余信号对应其后的半小时交易窗口，最后一个窗口为
`(14:30, 15:00]`。

## 输出

输出 `date`、`instrument`、`vwap_return`、`vwap`、`end_price`、`volume`、
`amount`、`num_trades`，以 `date + instrument` 为唯一键。窗口成交量为 0 或
缺少有效终点价格时，不计算有效收益率。
