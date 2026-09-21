# Snapshot 生成未来 30 分钟 VWAP 收益算法

## 1. 收益口径

每个因子信号对应信号发出后的下一可交易半小时：

```text
future_vwap = 窗口成交额 / 窗口成交量
vwap_return = window_end_price / future_vwap - 1
```

- `future_vwap` 是下一可交易半小时的成交量加权平均价；
- `window_end_price` 是该窗口结束时的最新有效成交价；
- 该收益可理解为按窗口 VWAP 执行，并在窗口结束时按终点价格估值；
- 不使用隔夜数据。

## 2. 八个信号截面

每天固定产生以下 8 个因子信号截面：

```text
[09:30, 10:00, 10:30, 11:00, 11:30, 13:30, 14:00, 14:30]
```

信号与收益窗口的对应关系为：

| 信号时间 | VWAP 执行窗口 | 窗口终点价格 |
| --- | --- | --- |
| 09:30 | `(09:30, 10:00]` | 10:00 最新成交价 |
| 10:00 | `(10:00, 10:30]` | 10:30 最新成交价 |
| 10:30 | `(10:30, 11:00]` | 11:00 最新成交价 |
| 11:00 | `(11:00, 11:30]` | 11:30 最新成交价 |
| 11:30 | `(13:00, 13:30]` | 13:30 最新成交价 |
| 13:30 | `(13:30, 14:00]` | 14:00 最新成交价 |
| 14:00 | `(14:00, 14:30]` | 14:30 最新成交价 |
| 14:30 | `(14:30, 15:00]` | 15:00 收盘价 |

`11:30` 是特殊截面：信号在上午收盘时产生，执行窗口跳过午休，使用下午
第一个可交易半小时 `(13:00, 13:30]`。

`14:30` 截面的收益为：

```text
15:00 收盘价 / VWAP(14:30, 15:00] - 1
```

## 3. 输入数据

从 `cn_stock_level2_snapshot` 读取：

| 字段 | 用途 |
| --- | --- |
| `date` | 快照时间 |
| `trading_day` | 交易日及历史成分股匹配 |
| `instrument` | 股票代码 |
| `price` | 窗口终点成交价 |
| `volume` | 当日累计成交量 |
| `amount` | 当日累计成交额 |
| `num_trades` | 当日累计成交笔数 |

通过 `cn_stock_index_component` 过滤，只保留当日属于中证 2000
（`932000.CSI`）历史成分股的数据。

## 4. 累计字段转成交增量

Snapshot 的 `volume`、`amount`、`num_trades` 是当日累计值，不能在窗口内
直接求和。对每只股票、每个交易日按快照时间排序后，计算相邻差分：

```text
delta_volume[i] = volume[i] - volume[i-1]
delta_amount[i] = amount[i] - amount[i-1]
delta_num_trades[i] = num_trades[i] - num_trades[i-1]
```

第 `i` 条快照的增量代表上一条快照到当前快照之间新发生的成交，并归入
当前快照所属的半小时窗口。

如果累计字段盘中重置而产生负差分，则将当前累计值作为重置后的有效增量：

```text
delta = current_value if difference < 0 else difference
```

## 5. 窗口边界

窗口采用左开右闭：

- `10:00:00.000` 属于 `(09:30, 10:00]`；
- `10:00:00.001` 属于 `(10:00, 10:30]`；
- 集合竞价、午休和收盘后的快照不进入成交窗口。

左开右闭可以保证窗口终点快照的累计增量和成交价都归入正确窗口。

## 6. 计算窗口 VWAP

按照 `instrument + signal_time` 分组，对快照成交增量求和：

```text
window_volume = sum(delta_volume)
window_amount = sum(delta_amount)
window_num_trades = sum(delta_num_trades)
future_vwap = window_amount / window_volume
```

只有当 `window_volume > 0` 且成交额有效时才计算 VWAP。

## 7. 确定窗口终点价格

先将 `price <= 0` 视为无效价格，然后取每个窗口内最后一个有效 `price`
作为 `window_end_price`。

例如：

- `(10:00, 10:30]` 使用不晚于 `10:30` 的最后一个有效成交价；
- `(14:30, 15:00]` 使用不晚于 `15:00` 的最后一个有效成交价，即收盘价。

## 8. 计算收益率

```text
vwap_return = window_end_price / future_vwap - 1
```

例如：

```text
窗口成交量 = 40,000
窗口成交额 = 420,000
future_vwap = 10.50
window_end_price = 10.71
vwap_return = 10.71 / 10.50 - 1 = 0.02
```

收益率为 `2%`。

## 9. 输出字段

| 字段 | 含义 |
| --- | --- |
| `date` | 因子信号时点 |
| `instrument` | 股票代码 |
| `vwap_return` | 窗口终点价格相对窗口 VWAP 的收益率 |
| `vwap` | 信号后的下一可交易半小时 VWAP |
| `end_price` | 窗口终点最新有效成交价 |
| `volume` | 窗口成交量 |
| `amount` | 窗口成交额 |
| `num_trades` | 窗口成交笔数 |

结果以 `date, instrument` 为唯一键，并按这两个字段排序。

## 10. 异常处理

- `11:30` 信号跳过午休，映射到 `(13:00, 13:30]`；
- `14:30` 信号截止 `15:00`，不跨到下一交易日；
- 零成交量：VWAP 和收益率为空；
- 无有效终点价格：收益率为空；
- 累计字段重置：负差分使用当前累计值；
- 停牌或窗口内完全没有快照：不生成该股票在该截面的标签；
- `price <= 0`：不作为窗口终点价格。

## 11. 完整流程

```text
读取 Snapshot 并按历史成分股过滤
             ↓
按股票、交易日和时间排序
             ↓
累计 volume/amount/num_trades 做快照差分
             ↓
将快照映射到 8 个信号对应的成交窗口
             ↓
汇总窗口成交量、成交额和成交笔数
             ↓
future_vwap = window_amount / window_volume
             ↓
取窗口内最后一个有效 price 作为 window_end_price
             ↓
vwap_return = window_end_price / future_vwap - 1
             ↓
标准化字段类型并写入 cpt_jyc_2026_vwap
```

对应实现位于同目录的 `builder.py`。
