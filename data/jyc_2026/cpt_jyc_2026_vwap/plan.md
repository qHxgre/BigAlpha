# 未来 30 分钟 VWAP 收益数据构建方案

## 数据范围

从 `cn_stock_level2_snapshot` 读取股票快照数据，并通过
`cn_stock_index_component` 筛选当日中证 1000（`000852.SH`）历史成分股；
同时从 `cn_stock_index_snapshot` 读取 `000852.SH` 指数快照。两类数据统一使用
字段：`date`、`instrument`、`price`、`volume`、`amount`、`num_trades`。
股票和指数分别查询，再通过 Pandas 纵向合并，不在 SQL 中跨数据表合并。

## 构建方法

1. 按标的、交易日和快照时间排序，股票和指数采用相同的聚合逻辑。
2. `volume`、`amount`、`num_trades` 为日内累计值，先按股票和交易日做相邻快照差分，得到单条快照的成交增量；若差分为负，视为累计值重置，使用当前累计值作为增量。
3. 将成交增量映射至左开右闭的 30 分钟窗口，并分别汇总窗口成交量、成交额和成交笔数。
4. 股票以窗口成交额除以窗口成交量得到 VWAP。指数快照的成交额和成交量
   是成分证券汇总值，二者相除不是指数点位，因此指数使用
   `sum(price * volume_delta) / sum(volume_delta)`。取窗口内最后一个大于 0 的
   `price` 作为终点价格。
5. 计算收益率：

```text
stock_vwap = window_amount / window_volume
index_vwap = sum(index_price * volume_delta) / sum(volume_delta)
vwap_return = end_price / vwap - 1
```

## 股票与指数 VWAP 口径差异

### 问题现象

如果对 `cn_stock_index_snapshot` 直接使用股票公式
`amount / volume`，计算出的数值会与指数 `price` 相差数倍。例如验证
`000852.SH` 的 2020-01-02 快照时，指数点位约为 5,600～5,680，
但 `amount / volume` 仅约为 970～1,030，二者不在同一量纲。

### 原因

股票是直接交易的证券，其成交数据满足：

```text
amount = sum(trade_price * trade_volume)
```

因此股票的 `amount / volume` 就是成交量加权平均成交价。

指数本身并不直接成交。指数快照中的字段含义为：

- `price`：按照指数编制规则计算的指数点位；
- `volume`：指数成分证券的汇总成交量；
- `amount`：指数成分证券的汇总成交额。

因此，指数的 `amount / volume` 表示成分证券的平均成交价格，并不等于、
也不能直接换算成指数点位，不能用于计算指数 VWAP 收益率。

### 解决方案

股票保持标准 VWAP 公式不变：

```text
stock_vwap = sum(amount_delta) / sum(volume_delta)
```

指数使用“指数点位按成分证券成交量增量加权”的代理 VWAP：

```text
index_vwap = sum(index_price * volume_delta) / sum(volume_delta)
```

这样得到的 `index_vwap` 与指数 `end_price` 都是指数点位，可以计算：

```text
vwap_return = end_price / index_vwap - 1
```

需要注意，指数本身没有真实成交价格，因此不存在与股票完全相同的严格
VWAP。这里的指数 VWAP 是用于生成指数收益标签的代理指标。如果目标是衡量
真实可交易证券的成交成本，应改为计算对应 ETF 或股指期货的 VWAP。

## 时间窗口

每日生成 8 个信号截面：`09:30`、`10:00`、`10:30`、`11:00`、`11:30`、
`13:30`、`14:00`、`14:30`。其中 `11:30` 信号跳过午休，对应
`(13:00, 13:30]`；其余信号对应其后的半小时交易窗口，最后一个窗口为
`(14:30, 15:00]`。

## 输出

输出 `date`、`instrument`、`vwap_return`、`vwap`、`end_price`、`volume`、
`amount`、`num_trades`，以 `date + instrument` 为唯一键。窗口成交量为 0 或
缺少有效终点价格时，不计算有效收益率。
