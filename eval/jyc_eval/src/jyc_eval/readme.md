预测目标：采用日内每30分钟一个截面、非重叠的采样方式，对每个采样时点、每只股票产出一个预测值，衡量其对未来30分钟收益的预测能力。
收益口径：标签采用未来30分钟VWAP（成交量加权均价）收益，而非收盘价对收盘价，以贴近真实可成交价格并减少“卡收盘价”空间。
采样间隔=预测窗口=30分钟，一天约8个日内截面；仅预测日内收益，不预测隔夜收益，并跳过开盘/收盘集合竞价等特殊时段。
终得分公式：五个正向项加权求和，各项均为全体提交上的百分位排名（pct rank，取值 [0,1]），五项建议等权。
Score_final = 0.2 × Rank_IC_mean + 0.2 × Rank_IC_IR + 0.2 × Rank_SR + 0.2 × Rank_Stress + 0.2 × Rank_Turnover
Rank_IC_mean：分钟级截面 RankIC（Spearman 秩相关）均值的全场排名；每个30分钟截面计算一次，对评估区间内所有截面等权取均值。
Rank_IC_IR：分钟级 RankIC 序列的 IR（均值/标准差）排名，衡量因子跨时点的稳定性。
Rank_SR：以因子/分数分组构建多空组合，按 30 分钟频率调仓后的年化夏普比率排名，年化按约 8 期/日 × 242 日折算。
Rank_Stress：日内 regime 稳健性排名，按高/低波动截面，可辅以放量/缩量和不同日内时段分桶，考察 RankIC_IR 在各 regime 下的一致性。
Rank_Turnover：因子换手率的全场百分位排名，按换手率从小到大排序，换手越低排名越高，用于奖励低换手、可落地的分钟级信号。

你先不考虑百分位排名，根据上述内容修改 /Users/xiehao/Desktop/workspace/BigAlpha/eval/jyc_eval/src/jyc_eval/factoranalyze.py 这个单因子分析体系，返回 IC_mean, IC_IR，SR，stress, turnonver 这几个指标的值，根据下面的数据：

* 数据获取均来自于：/Users/xiehao/Desktop/workspace/BigAlpha/eval/jyc_eval/src/jyc_eval/data.py
* cpt_jyc_2026_vwap 除了包含股票的数据，也包含指数的数据，因此也能计算超额收益率，详细查看：/Users/xiehao/Desktop/workspace/BigAlpha/data/jyc_2026/cpt_jyc_2026_vwap
* 要注意现有的代码是日频单因子分析，相当于要改成30分钟频率的单因子分析，且不考虑隔夜收益

## `factoranalyze.py` 指标设计

### 1. 总体口径

`FactorAnalyze` 以 `(date, instrument)` 为唯一键，将处理后的因子数据与
`data.py` 中 `load_evaluation_data()` 返回的未来 30 分钟 VWAP 收益标签对齐。

- `date` 表示一个 30 分钟预测窗口的起点。
- 每个 `date` 是一个独立的股票横截面。
- 每个交易日约有 8 个截面。
- 所有 IC、分组收益和换手率均按 30 分钟频率计算。
- 不对收益率或因子值进行日频 `shift`。
- 不使用下一交易日的数据，也不计算隔夜收益。

评估最终返回以下五个原始指标，暂不进行全体提交间的百分位排名：

```python
{
    "ic_mean": float,
    "ic_ir": float,
    "sharpe_ratio": float,
    "stress_stability": float,
    "turnover": float,
}
```

### 2. 超额收益

`cpt_jyc_2026_vwap` 同时包含股票和中证 1000 指数 `000852.SH` 的未来
30 分钟 VWAP 收益。对于截面 `t` 中的股票 `i`，先计算：

```text
excess_return(i,t) = forward_return(i,t) - benchmark_return(t)
```

其中：

- `forward_return(i,t)` 为股票未来 30 分钟 VWAP 收益。
- `benchmark_return(t)` 为同一截面中证 1000 指数未来 30 分钟 VWAP 收益。

截面内减去相同的指数收益不会改变股票收益的横截面排序，因此不会改变
RankIC，但会使分组收益和多空组合收益明确表示相对基准的超额收益。如果
某个截面缺少指数标签，当前实现将该截面的指数收益按 `0` 处理并记录警告。

### 3. IC Mean

每个 30 分钟截面独立计算一次 Spearman RankIC：

```text
IC(t) = Spearman(factor(i,t), excess_return(i,t))
```

因子或收益全部相同、有效股票数不足 2 只的截面无法计算相关系数，将从 IC
序列中剔除。

最终 IC 均值为全部有效截面的等权平均：

```text
ic_mean = mean(IC(t))
```

这里不会先计算每日 IC，也不会按照交易日进行加权；每个有效的 30 分钟截面
权重相同。

### 4. IC IR

IC IR 用于衡量 RankIC 跨截面的稳定性：

```text
ic_ir = mean(IC(t)) / sample_std(IC(t))
```

标准差使用样本标准差，即 `ddof=1`。IC IR 当前不进行年化。如果有效 IC
少于 2 个，或者 IC 标准差为 0，则返回 `0.0`。

### 5. 多空组合与 Sharpe Ratio

每个截面按照因子值从低到高进行等数量分组，默认分为 5 组：

- 第 0 组：因子值最低组，作为空头组合。
- 第 4 组：因子值最高组，作为多头组合。
- 股票因子值相同时，使用稳定的顺序排名后再分组，尽量保证分组可用。
- 当截面股票数少于默认组数时，实际组数自动缩减。

每组内部使用股票超额收益的等权平均。截面多空收益为：

```text
long_short_return(t) = high_group_return(t) - low_group_return(t)
```

因子时点和未来 30 分钟标签已经直接对齐，因此分组收益不再执行日频分析中
的 `shift(1)`。

多空夏普按每个交易日约 8 期、每年 242 个交易日进行年化：

```text
periods_per_year = 8 * 242

sharpe_ratio = mean(long_short_return)
               / sample_std(long_short_return)
               * sqrt(periods_per_year)
```

当前采用零无风险利率。如果有效多空收益少于 2 期，或者标准差为 0，则返回
`0.0`。

### 6. Stress Stability

`stress_stability` 用于衡量因子在不同日内波动环境中的稳定性。它不是每个
截面单独返回一个 stress 值，而是使用评估期内全部截面汇总得到一个最终值。

首先，对每个 30 分钟截面计算股票超额收益的横截面标准差：

```text
section_volatility(t) = sample_std(excess_return(i,t))
```

然后使用评估期内所有有效截面波动率的中位数划分 regime：

```text
low_volatility  = section_volatility(t) <= median(section_volatility)
high_volatility = section_volatility(t) >  median(section_volatility)
```

分别收集高、低波动 regime 中的截面 RankIC，并计算各自的 IC IR：

```text
low_vol_ic_ir  = mean(IC(t) in low_volatility)
                 / sample_std(IC(t) in low_volatility)

high_vol_ic_ir = mean(IC(t) in high_volatility)
                 / sample_std(IC(t) in high_volatility)
```

最终取两个 regime 中表现较弱的一方：

```text
stress_stability = min(low_vol_ic_ir, high_vol_ic_ir)
```

该设计采用“短板”口径：因子只有在高波动和低波动环境中都具有稳定预测能力，
才会获得较高的 `stress_stability`。任一 regime 有效 IC 少于 2 个，或者 IC
标准差为 0，该 regime 的 IC IR 按 `0.0` 处理。

### 7. Turnover

换手率基于每个截面的最高因子组和最低因子组持仓计算。对于同一交易日内的
两个相邻截面，分别计算多头和空头的单边换手：

```text
long_turnover(t) = 1 - |long(t-1) ∩ long(t)| / |long(t-1)|
short_turnover(t) = 1 - |short(t-1) ∩ short(t)| / |short(t-1)|
```

该次调仓的多空平均换手为：

```text
section_turnover(t) = (long_turnover(t) + short_turnover(t)) / 2
```

最终换手率为全部有效日内调仓的等权平均：

```text
turnover = mean(section_turnover(t))
```

跨交易日的两个截面不会参与换手计算，因此不会把隔夜持仓变化计入日内换手。
当前返回的是原始换手率，数值越低表示信号越稳定。后续计算全场百分位排名时，
应按照低换手优先的方向转换为 `Rank_Turnover`。

### 8. 指标方向

当前五项均返回原始值，未进行方向翻转、绝对值处理或百分位排名：

- `ic_mean`：越高越好。
- `ic_ir`：越高越好。
- `sharpe_ratio`：越高越好。
- `stress_stability`：越高越好。
- `turnover`：越低越好。

全场最终评分阶段应分别计算五项指标的百分位排名，其中换手率需要采用与其他
指标相反的排序方向，再按照约定权重合成总分。
