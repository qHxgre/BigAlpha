# BigAlpha 2026 端到端第一梯队完整候选包

## 1. 提交包概览

- 提交名称：`submit_d128bag_s7s99e18_55_w440_p205_0805_1730_1111111111111`
- 当前状态：第一梯队候选，生成完整包时尚未取得该候选平台公榜分数
- 推理入口：`predict.ipynb`
- 命名推理副本：`predict__submit_d128bag_s7s99e18_55_w440_p205_0805_1730_1111111111111.ipynb`
- 主要权重：`transformer_model_5m.json`、`transformer_model_daily_s7.json`、`transformer_model_daily_s99e18.json`

本包在轻量候选包基础上补齐官方审查需要的材料。推理 notebook 与候选权重保持原候选逻辑；新增文件仅用于说明、复现和代码审查，不改变线上推理。

## 2. 官方要求对应关系

| 官方要求 | 本包对应文件 |
| --- | --- |
| 推理脚本 | `predict.ipynb` 和 `predict__*.ipynb` |
| 训练脚本 | `train_5m_transformer.py`、`train_daily_ohlcv_cross_transformer.py` |
| 依赖声明 | `requirements.txt` |
| 超参配置 | `submission_config.json` |
| 随机种子 | 5m seed=42；daily seed=[7, 99] |
| 已训练权重 | `transformer_model_5m.json`、`transformer_model_daily_s7.json`、`transformer_model_daily_s99e18.json` |
| 辅助代码 | `download_data.py`、`features_raw.py`、`parquet_compat.py`、`numpy_compat.py`、`reproducibility.py` |
| 训练日志 | `logs/` 目录 |
| 文件哈希清单 | `MANIFEST.json` |

## 3. 模型与融合逻辑

本候选类型为：daily seed-bag。

- 5m 分支：raw 5 分钟 Transformer，使用 19 个原始量价和盘口字段。
- daily 分支：从 5m bar 聚合日频 OHLCV，使用 d128 Cross Transformer；seed-bag 候选会先在 daily 分支内部融合多个 daily 权重。
- 最终融合：5m 权重 `0.560`，daily 权重 `0.440`，softsign 参数 `2.05`，融合后按交易日截面 z-score。

## 4. 训练数据与权重来源

本包只使用 BigAlpha 端到端赛道提供的数据，不使用外部数据。

- 5m 模型训练区间：`2019-01-01` 至 `2023-12-31`，seed=42。
- daily 模型训练区间：`2019-01-01` 至 `2023-12-31`，验证区间：`2024-01-01` 至 `2024-12-31`。
- seed99_ep18：训练 18 epoch，最佳 epoch=12，`val_rank_ic=0.061784`，`val_ir=0.4064`。

## 5. 文件说明

- `predict.ipynb`：平台推理入口。
- `submission_config.json`：结构化记录模型超参、训练区间、随机种子、融合参数。
- `MANIFEST.json`：包内文件大小和 SHA256。
- `logs/`：相关 daily 模型训练日志。

## 6. 私榜选择建议

如果本候选平台公榜分数超过当前最高 `0.78158`，应把该候选作为新的私榜主候选，并保留本完整包用于代码审查。若未超过，则继续以平台实际最高分为准。
