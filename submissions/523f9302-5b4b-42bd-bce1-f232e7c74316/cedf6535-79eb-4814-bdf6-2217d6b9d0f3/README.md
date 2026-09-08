# Submission — V11-norm-v2 Transformer 因子模型

## 提交方式: 方式二 (.py 训练 + .ipynb 推理)

## 文件

| 文件 | 用途 |
|------|------|
| `transformer_train.py` | 训练脚本。平台私榜阶段调用 `train_and_save(datasources)` 从零重训。`python transformer_train.py` 可在本地训练。 |
| `Transformer_modelsave_predict.ipynb` | 推理 notebook。平台调用 `main(datasources, start_date, end_date)` 进行打分。 |
| `transformer_model.json` | 训练产出 (权重 + 统计量 + 配置)。公榜阶段直接使用，私榜阶段由平台重训后替换。 |

## 平台接口

### 训练 (私榜)
```python
from transformer_train import train_and_save
train_and_save(datasources)  # → transformer_model.json
```
- `datasources["bar1m"]`: 平台提供的训练集表名 (函数内部已硬编码, 不使用此参数)
- 训练周期: 2021-01-01 ~ 2023-12-31 (3 年)
- Epochs: 20 (鲁棒性实验验证: 跨周期最稳定)

### 推理 (公榜 + 私榜)
```python
from Transformer_modelsave_predict import main
result = main(datasources, start_date, end_date)
# result: DataFrame [date, instrument, score]
```

## 关键参数

| 参数 | 值 | 选择依据 |
|------|-----|---------|
| EPOCHS | 20 | 在 3 个训练周期 (2/3/4 年) 上 IC 标准差最小, 最稳健 |
| SEQ_LEN | 240 | 1 个交易日 = 240 根 1min bar, 在平台限制 ≤ 240 内最大化 |
| d_model | 96 | ~318K 总参数量, 在 100K~100M 约束内, 表达力充分 |
| LR_PEAK | 5e-4 | AdamW + warmup + cosine 衰减, 训练稳定 |
| Dropout | 0.15 | 适度正则, 对各数据规模泛化良好 |

## 模型架构

```
x (B,240,12) → SpectralFilter1D (FFT→MLP→iFFT, 残差 50%)
  → Conv Stem (Conv1d k=5 + k=3, 12→96)
  → + ClusterEmbed (5 类正态分位, 对数空间股价聚类)
  → + Positional Embedding + Time-of-Day Embedding
  → Transformer Encoder (3 层, d=96, head=4, ff=192, pre-norm, dropout=0.15)
  → Attn Gate (softmax 时间加权)
  → Head MLP → scalar prediction
  → DAE: Recon Head → 12-dim 重建 (auxiliary MSE)

参数量: ~318,555
```

## 预处理管线 (规则化, 无学习参数)

1. bid/ask 零值 → close 填充 (涨跌停/低流动性修复)
2. 价格 ÷ pre_close (日级归一化, log 变换前)
3. log1p (成交量) / log (价格) 变换
4. 成交量 ÷ daily_max (日级归一化, log1p 变换后)
5. 240-bar 滑动窗口
6. z-score 标准化 (训练集 mean/std, 推理复用)

## 合规自检

- [x] 输入字段 12 ≤ 100
- [x] 禁止衍生/人工特征工程 (归一化属于数据预处理)
- [x] 参数量 ~318K ∈ [100K, 100M]
- [x] 禁止预训练权重 (从零训练)
- [x] 回看窗口 240 ≤ 240 个交易日
- [x] 训练表名硬编码 (不依赖 datasources)
- [x] JSON 文本格式保存
- [x] 推理复用训练集 mean/std
- [x] 标签: close[T+1]/close[T]-1 (评测口径为 open-to-open, 但 close-to-close 训练更稳定)

## 实验依据

鲁棒性实验 (20260801 目录):
- 在 2020-2023 / 2021-2023 / 2022-2023 三个训练周期上扫描 epoch=2~24
- 2024 全年评估 (bigalpha_eval v4 复刻管线)
- EPOCHS=20: 唯一在全部 3 个周期下都有顶级表现的 epoch, IC 标准差最小 = 0.0006
- 确认: 更多训练数据 → 模型可训练更久不 overfit; 但 epoch=20 是最稳健的折中
