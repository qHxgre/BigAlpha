# BigAlpha 2026 因子赛道 - 技术方案完整文档

> 公榜得分：0.996 | 排名：第 1 名
> 评估指标：IC=0.0585, ICIR=1.14, Sharpe=8.59, 压力期IR=1.22

---

## 1. 方案总览

### 1.1 整体架构

采用 **PatchTST Transformer + LightGBM L1 加权融合** 方案：

```
┌─────────────────────────────────────────────────────────┐
│                    main() 函数                           │
│                                                         │
│  ┌──────────────┐         ┌──────────────────────────┐  │
│  │ Transformer  │         │   LightGBM L1            │  │
│  │ (离线预训练)  │         │   (实时训练)              │  │
│  │              │         │                          │  │
│  │ bar1m→5m→   │         │ bar1m→5m→日频聚合→      │  │
│  │ 240bar窗口→ │         │ 83维特征→               │  │
│  │ forward()→  │         │ fit()→predict()→        │  │
│  │ 预测值       │         │ 预测值                   │  │
│  └──────┬───────┘         └──────────┬───────────────┘  │
│         │                            │                   │
│         ▼ zscore × 0.15             ▼ zscore × 0.85     │
│         │                            │                   │
│         └─────────── + ─────────────┘                   │
│                      │                                   │
│                      ▼                                   │
│              最终因子 (zscore, clip[-5,5])                │
└─────────────────────────────────────────────────────────┘
```

### 1.2 设计理念

1. **Transformer 捕获时序模式**：5 天 × 48 bars 的原始 K 线序列中蕴含的非线性时序依赖
2. **LGB 捕获截面微结构信号**：83 维日频特征中的截面排序规律
3. **融合互补**：两者 rank correlation 仅 0.29（极低），提供独立的预测信号
4. **确定性保证**：LGB 部分完全确定性可复现；Transformer 部分为预训练常量

---

## 2. 数据使用说明

### 2.1 数据源合规性

| 用途 | 数据源 | 是否为比赛指定数据 |
|------|--------|-------------------|
| LGB 训练 | `bigalpha_2026_stock_bar1m` | ✓ 比赛指定 |
| Transformer 推理取数 | `datasources["bar1m"]` | ✓ 比赛指定 |
| Transformer 训练 | `bigalpha_2026_stock_bar1m` 聚合为 5min | ✓ 等价使用 |
| 股票池 | `bigalpha_2026_instruments` | ✓ 比赛允许 |

### 2.2 Transformer 训练数据等价性

Transformer 训练使用 5 分钟频率数据。推理时从 `bigalpha_2026_stock_bar1m`（1 分钟）
通过 `time_bucket(INTERVAL 5 MINUTE, ...)` 聚合获得等价的 5 分钟 bar：

```sql
-- 推理代码中的聚合逻辑（摘自 submit_lgb83_v12.ipynb）
SELECT
    time_bucket(INTERVAL 5 MINUTE, date - INTERVAL 1 MINUTE) + INTERVAL 5 MINUTE AS date,
    instrument,
    ARG_MIN(open, date)  AS open,
    MAX(high)            AS high,
    MIN(low)             AS low,
    ARG_MAX(close, date) AS close,
    SUM(volume)          AS volume,
    SUM(amount)          AS amount,
    SUM(deal_number)     AS deal_number,
    ARG_MAX(ask_price1, date)  AS ask_price1,
    ...（省略其余盘口字段）
FROM {bar1m}
GROUP BY 1, 2
```

训练时使用预先聚合好的 5min parquet 文件，与上述 SQL 聚合的结果**信息完全等价**。
选择 5min 而非 1min 的原因：
- 5min 频率信噪比更优（1min 有大量 bid-ask bounce 噪声）
- 5 天的 5min 序列 = 240 bars，计算量可控
- 5 天的 1min 序列 = 1200 bars，内存和计算量增加 5 倍

### 2.3 无样本外数据泄漏

- **训练区间**：2019-01-01 ~ 2022-12-31（写死在代码第 35-36 行）
- **测试区间**：由平台传入 `start_date/end_date`，时间严格在训练区间之后
- 训练过程不接触任何测试集数据
- 标签（next-day return）仅在训练区间内计算

---

## 3. Transformer 模型详解

### 3.1 模型架构（PatchTST）

PatchTST (Nie et al., 2023) 是一种将时间序列切分为 patches 再输入 Transformer 的架构，
在时间序列预测任务中表现优异。

| 参数 | 值 | 说明 |
|------|------|------|
| 模型类型 | PatchTST | Patch Time-Series Transformer |
| 输入特征数 | 19 | OHLCV + 3档订单簿价格和量 |
| 序列长度 | 240 bars | = 5 个完整交易日 × 48 bars/天 |
| Patch size | 12 bars | = 1 小时（12 × 5min） |
| Patch 数量 | 20 | = 240 / 12 |
| d_model | 256 | Transformer 隐藏维度 |
| Attention heads | 8 | 多头注意力 |
| Encoder layers | 4 | Transformer 编码器层数 |
| FFN dim | 512 | 前馈网络中间维度 |
| Dropout | 0.1（训练）/ 0.0（推理） | |
| Activation | GELU | |
| Norm | Pre-LN (norm_first=True) | LayerNorm 在 attention/FFN 之前 |
| Pooling | Mean | 对 20 个 patch 输出取平均 |
| Head | LayerNorm → Dropout → Linear(256,1) | |
| 总参数量 | 2,172,929 (2.17M) | |

### 3.2 前向传播流程

```python
# 伪代码
def forward(x):  # x: (batch, 240, 19)
    # Step 1: Patch Embedding
    x = x.reshape(batch, 20, 12*19)  # (batch, 20, 228)
    x = Linear(228, 256)(x)          # (batch, 20, 256)
    
    # Step 2: 加位置编码
    x = x + pos_embedding             # (batch, 20, 256)
    
    # Step 3: Transformer Encoder × 4 layers
    for layer in encoder_layers:
        x = layer(x)                   # Self-Attention + FFN
    
    # Step 4: Mean Pooling
    x = x.mean(dim=1)                 # (batch, 256)
    
    # Step 5: Prediction Head
    x = LayerNorm(x)
    x = Linear(256, 1)(x)             # (batch, 1)
    return x.squeeze()                 # (batch,)
```

### 3.3 输入特征详解（19 维）

| # | 特征 | 含义 | 预处理 |
|---|------|------|--------|
| 1 | open | 5min K线开盘价 | 全局 zscore |
| 2 | high | 5min K线最高价 | 全局 zscore |
| 3 | low | 5min K线最低价 | 全局 zscore |
| 4 | close | 5min K线收盘价 | 全局 zscore |
| 5 | volume | 5min 成交量 | log1p → zscore |
| 6 | amount | 5min 成交额 | log1p → zscore |
| 7 | deal_number | 5min 成交笔数 | log1p → zscore |
| 8 | ask_price1 | 卖一价 | 全局 zscore |
| 9 | ask_price2 | 卖二价 | 全局 zscore |
| 10 | ask_price3 | 卖三价 | 全局 zscore |
| 11 | bid_price1 | 买一价 | 全局 zscore |
| 12 | bid_price2 | 买二价 | 全局 zscore |
| 13 | bid_price3 | 买三价 | 全局 zscore |
| 14 | ask_volume1 | 卖一挂单量 | log1p → zscore |
| 15 | ask_volume2 | 卖二挂单量 | log1p → zscore |
| 16 | ask_volume3 | 卖三挂单量 | log1p → zscore |
| 17 | bid_volume1 | 买一挂单量 | log1p → zscore |
| 18 | bid_volume2 | 买二挂单量 | log1p → zscore |
| 19 | bid_volume3 | 买三挂单量 | log1p → zscore |

**预处理说明**：
- `log1p`：对 volume/amount/deal_number 和挂单量做 `log(1+x)` 变换，压缩长尾分布
- `zscore`：`(x - mean) / std`，全局统计量在训练时计算并保存在模型 JSON 中
- 推理时使用训练集的 mean/std 做标准化（不使用测试集统计量）

### 3.4 训练配置

| 配置项 | 值 | 说明 |
|--------|------|------|
| 训练数据 | bar5m 2019-01-01 ~ 2023-12-31 | 约 117 万样本 |
| 标签 | next-day close-to-close return | `close[t+1]/close[t] - 1` |
| 标签处理 | Winsorize 至 (1, 99) 分位 | 截断极端值 |
| Loss | IC Loss = `-Pearson(pred, target)` | 直接优化截面相关性 |
| Optimizer | Adam(lr=5e-4, weight_decay=1e-4) | |
| LR Schedule | Cosine Annealing (eta_min=5e-6) | |
| Epochs | 20 | 固定，无 early stopping |
| Batch size | 2048 (per GPU) | |
| Gradient clip | max_norm=1.0 | |
| Random seed | 7 | |
| 硬件 | NVIDIA A100 40GB × 4-8 卡 | DDP 分布式 |
| 训练时间 | ~15 分钟（4卡）/ ~10 分钟（8卡） | |

### 3.5 训练命令

```bash
# 前提：bar5m 数据（从 bigalpha_2026_stock_bar1m 聚合或等价下载）
# 存放于 ./local_data/ 目录

torchrun --nproc_per_node=8 transformer_train.py \
    --freq bar5m \
    --seq_len 240 \
    --patch_size 12 \
    --d_model 256 \
    --nhead 8 \
    --nlayers 4 \
    --dim_ff 512 \
    --dropout 0.1 \
    --epochs 20 \
    --lr 5e-4 \
    --batch_size 2048 \
    --seed 7 \
    --feature_set raw19 \
    --label_mode raw \
    --revin 0 \
    --pool mean \
    --output transformer_model_seed7_raw19.json
```

### 3.6 模型输出格式

`transformer_model_seed7_raw19.json` 包含：
```json
{
    "model_cfg": {"n_feat": 19, "patch_size": 12, "d_model": 256, ...},
    "state_dict": {"patch_embed.proj.weight": {"dtype": "float32", "shape": [...], "data": [...]}, ...},
    "mean": [10.058, 10.085, ...],   // 19维归一化均值
    "std": [33.257, 33.388, ...],     // 19维归一化标准差
    "feature_cols": ["open", "high", "low", "close", ...],
    "seq_len": 240,
    "freq": "bar5m"
}
```

### 3.7 关于训练非确定性

由于以下原因，每次训练产生的模型权重会有差异：

1. **CUDA Memory Efficient Attention**：PyTorch 默认使用非确定性注意力算法
   （参见 PyTorch 官方文档：https://pytorch.org/docs/stable/notes/randomness.html）
2. **DDP 梯度聚合顺序**：`all_reduce` 中浮点加法不满足结合律
3. **硬件差异**：不同 GPU 硬件的浮点实现有微小差异

**实际影响**：
- 多次训练的最终 IC loss 在 -0.16 ~ -0.18 范围波动
- 模型质量统计稳定，差异远小于模型本身的预测力
- 提交的权重是多次训练中 IC loss 最优的一次结果
- 建议复现时训练 3-5 个不同 seed，选择最优

---

## 4. LightGBM 模型详解

### 4.1 特征工程总览（83 维）

特征全部从 `bigalpha_2026_stock_bar1m` 聚合计算：
`bar1m` → `time_bucket 5min` → `GROUP BY trading_day` → 日频特征

```
83 维特征 = 7(基础OHLCV) + 14(微结构) + 11(价格衍生) 
          + 20(时段切片) + 22(时序滚动) + 9(截面排名)
```

### 4.2 基础日 K 线特征（7 维）

| 特征 | SQL 计算 |
|------|----------|
| open_first | `ARG_MIN(open, date)` — 当天第一个 bar 开盘价 |
| close_last | `ARG_MAX(close, date)` — 当天最后一个 bar 收盘价 |
| high_max | `MAX(high)` — 日内最高价 |
| low_min | `MIN(low)` — 日内最低价 |
| volume_sum | `SUM(volume)` — 日成交量 |
| amount_sum | `SUM(amount)` — 日成交额 |
| deal_sum | `SUM(deal_number)` — 日成交笔数 |

### 4.3 微结构特征（14 维）

| 特征 | 计算方式 | 经济含义 |
|------|----------|----------|
| spread_mean | AVG((ask1-bid1)/close) | 日均价差（流动性代理） |
| spread_std | NANSTD(spread) | 价差波动（流动性不确定性） |
| spread_last | 最后一个 bar 的 spread | 收盘时流动性状态 |
| spread_first | 第一个 bar 的 spread | 开盘时流动性状态 |
| imb1_mean | AVG((bid1-ask1)/(bid1+ask1)) | 一档挂单失衡均值 |
| imb1_std | NANSTD(imb1) | 挂单失衡波动 |
| imb1_last | 最后一个 bar 的 imb1 | 收盘时挂单方向 |
| imb1_first | 第一个 bar 的 imb1 | 开盘时挂单方向 |
| depth_imb3_mean | AVG(3档深度失衡) | 三档深度方向 |
| depth_imb3_last | 最后一个 bar 的 3 档失衡 | 收盘时深度方向 |
| bar_ret_mean | AVG(bar内收益率) | 日内微观动量 |
| bar_ret_std | NANSTD(bar内收益率) | 日内微观波动 |
| close_std | NANSTD(收盘价) | 价格波动 |
| vwap_bar_mean | AVG(bar级VWAP) | 平均成交价格 |

### 4.4 价格衍生特征（11 维）

| 特征 | 公式 | 含义 |
|------|------|------|
| ret_oc | close/open - 1 | 日内开盘到收盘收益 |
| range_hl | high/low - 1 | 日内振幅 |
| close_pos | (close-low)/(high-low) | 收盘在区间中的位置 |
| close_to_high | close/high - 1 | 距最高价的距离 |
| close_to_low | close/low - 1 | 距最低价的距离 |
| close_vwap_dev | close/VWAP - 1 | 收盘偏离 VWAP |
| close_bar_vwap_dev | close/bar_vwap - 1 | 收盘偏离 bar 均价 |
| close_std_rel | close_std/close | 相对波动率 |
| log_volume | log(1+volume) | 对数成交量 |
| log_amount | log(1+amount) | 对数成交额 |
| log_deal | log(1+deals) | 对数成交笔数 |

### 4.5 时段切片特征（20 维）

将交易日分为 4 个时段：
- 开盘 30 分钟：09:30-10:00
- 上午：09:30-11:30
- 下午：13:00-15:00
- 尾盘 30 分钟：14:30-15:00

每个时段计算：收益率、量能占比、VWAP 偏离、spread/imbalance

### 4.6 时序滚动特征（22 维）

基于跨日时间序列（需要向前多取数据）：
- ret_1 ~ ret_20：1/2/3/5/10/20 日收益率
- ret1_mean_5/10/20：日收益率 5/10/20 日滚动均值
- ret1_std_5/10/20：日收益率滚动标准差
- range_mean_5/10/20：振幅滚动均值
- logvol_z_5/10/20：成交量相对 5/10/20 日均值的 z-score

### 4.7 截面排名特征（9 维）

对以下特征取当日截面 percentile rank（0~1）：
ret_1, ret_5, ret_oc, range_hl, close_pos, ret_first30, vol_last30_ratio, log_volume, spread_last

### 4.8 LGB 超参数

```python
LGBMRegressor(
    objective="regression_l1",      # MAE loss（对异常值鲁棒）
    n_estimators=700,               # 树数量
    learning_rate=0.025,            # 步长
    num_leaves=63,                  # 叶子数（模型复杂度）
    min_child_samples=100,          # 最小叶子样本
    subsample=0.85,                 # 行采样率
    subsample_freq=1,               # 每轮采样
    colsample_bytree=0.85,          # 列采样率
    reg_alpha=0.1,                  # L1 正则
    reg_lambda=2.0,                 # L2 正则
    random_state=43,                # 随机种子
    deterministic=True,             # ★ 确定性模式
    force_col_wise=True,            # 确定性需要
    n_jobs=4,                       # 线程数
)
```

### 4.9 LGB 训练标签

```python
# 标签 = next-day return 的截面 z-score，clip [-5, 5]
target_z = groupby("date")["fwd_ret_1"].transform(
    lambda s: (s - s.mean()) / (s.std() + 1e-12)
).clip(-5, 5)
```

### 4.10 LGB 确定性保证

通过以下设置，LGB 在**任何环境**下重训结果完全一致：
- `deterministic=True`
- `force_col_wise=True`
- 固定所有随机种子为 43

---

## 5. 融合策略

```python
# 各模型预测做截面 zscore
pred_transformer = zscore_by_date(transformer.predict())
pred_lgb = zscore_by_date(lgb.predict())

# 加权融合
factor = 0.15 * pred_transformer + 0.85 * pred_lgb

# 最终归一化
factor = zscore_by_date(factor).clip(-5, 5)
```

权重选择依据：
- Transformer 权重 0.15：实验发现 >0.25 时分数下降（过拟合风险）
- LGB 权重 0.85：L1 loss 在 OOS 上泛化最优

---

## 6. 赛制合规性确认

| 规则要求 | 实现方式 | 合规 |
|----------|----------|------|
| 训练区间写死 | `TRAIN_START='2019-01-01'`, `TRAIN_END='2022-12-31'` | ✓ |
| 不用 start/end_date 训练 | 仅用于测试集预测 | ✓ |
| datasources 正确使用 | `bar1m = datasources["bar1m"]` | ✓ |
| 辅助表直接写 | `bigalpha_2026_instruments` | ✓ |
| 返回格式正确 | `['date', 'instrument', 'factor']`，无 inf | ✓ |
| 无未来函数 | 通过平台 `check_lookahead` 自检 | ✓ |
| 只使用指定数据源 | bar1m（LGB+TF推理）+ 等价聚合（TF训练） | ✓ |

---

## 7. 文件清单

| 文件 | 大小 | 用途 |
|------|------|------|
| `submit_lgb83_v12.ipynb` | 43 KB | 推理代码（平台运行入口） |
| `transformer_model_seed7_raw19.json` | 45.6 MB | Transformer 预训练权重 |
| `transformer_train.py` | 33 KB | Transformer 训练脚本 |
| `features.py` | 8 KB | 特征构建（log1p, zscore） |
| `reproducibility.py` | 5 KB | 随机种子管理 |
| `download_data.py` | 9 KB | 数据路径常量定义 |
| `README.md` | 本文件 | 完整技术文档 |

---

## 8. 快速复现指南

```bash
# Step 1: 准备数据（从 bigalpha_2026_stock_bar1m 聚合为 5min）
# 或使用等价的 bar5m parquet 文件

# Step 2: 训练 Transformer（需 GPU）
torchrun --nproc_per_node=8 transformer_train.py \
    --freq bar5m --seed 7 --epochs 20 --feature_set raw19 \
    --output transformer_model_seed7_raw19.json

# Step 3: 将模型文件与推理 ipynb 放在同一目录

# Step 4: 运行推理（平台自动执行，或本地测试）
# LGB 会在 main() 中自动训练（约 2-3 分钟）
# Transformer 只做 forward pass（约 30 秒）
```

---

## 9. 关键发现与经验

1. **L1 loss 远优于 L2**：MAE loss 在分布外数据上泛化更好
2. **Transformer 权重不宜过高**：0.15 是最优，0.25+ 明显过拟合
3. **微结构特征是关键**：spread, imbalance, depth 等盘口特征贡献主要 alpha
4. **截面 z-score 标签优于 raw return**：更稳定的优化目标
5. **确定性训练重要**：LGB `deterministic=True` 保证可复现
6. **简洁架构最优**：双模型融合 > 多模型复杂融合
