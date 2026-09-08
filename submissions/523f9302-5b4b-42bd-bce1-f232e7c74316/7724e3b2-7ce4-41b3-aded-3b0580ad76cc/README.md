# model_dynamic1 提交包（BigAlpha 2026 · 端到端大模型赛道）

本目录是 model_dynamic1 模型的提交包（与 `model_dynamic2` 并列，两支
模型可同时提交）。推理文件为唯一 notebook `reasoning.ipynb`。

## 目录结构

```text
model_dynamic1/（全部为文件，无子目录，共 12 个）
├── README.md            # 本文件
├── requirements.txt     # 依赖
├── model.json           # 已训练权重（BigQuant 纯文本格式）
├── reasoning.ipynb      # 唯一推理 notebook（自包含，含 main()）
├── pretrain.yaml        # 预训练配置
├── finetune.yaml        # 微调配置
├── core.py              # 配置加载 + 数据 IO + 特征变换 + 指标 + 分布式工具
├── dataset.py           # 日级截面 batch + 日级特征缓存 + 覆盖率分析
├── losses.py            # RankingLoss 全套损失
├── model.py             # AlphaModel（dynamic-top2-v1）
├── pretrain.py          # 预训练主循环 + 缓存构建（--build-cache）
└── train.py             # 微调主循环 + 权重导出（--export）
```

## 权重与指标

| 项目 | 取值 |
| --- | --- |
| 架构 | `dynamic-top2-v1`，format `bigalpha_model_json_v1` |
| checkpoint | epoch 14，137 个张量（提交权重为 27 特征） |
| valid（2023H2，124 天） | RankIC 0.1222 / RankICIR 2.984 / IC 0.1204 |
| test（2024 全年，241 天） | RankIC 0.1136 / RankICIR 2.276 / IC 0.1314 |
| 损失权重（可学习） | 初始 daily_z 0.07 / rank_corr 0.80 / pairwise 0.13；epoch 14 时漂移至 0.060 / 0.538 / 0.403 |

## 数据导入（训练数据从哪来）

训练代码从配置文件里的 `data_root` 读取数据（默认 `data`，相对运行目录），
其下按表名存放 hive 分区 parquet：

```text
data/
├── bigalpha_2026_stock_bar1m/year=*/month=*/*.parquet   # 1 分钟 bar（5 档盘口）
├── bigalpha_2026_factorlib/year=*/month=*/*.parquet     # 标签：daily_return
├── bigalpha_2026_exposure/year=*/month=*/*.parquet      # BARRA 风格暴露
└── bigalpha_2026_instruments/year=*/month=*/*.parquet   # 股票池
```

- 表名必须与配置 `tables` 完全一致（`bigalpha_2026_stock_bar1m` /
  `bigalpha_2026_factorlib` / `bigalpha_2026_exposure` / `bigalpha_2026_instruments`），
  训练与推理（`reasoning.ipynb` 通过平台 DAI 读取同名表）共用同一套表名；
- 数据格式要求与 `data_io.py` 一致：`instrument` 为字符串、价格为元、5 档盘口、
  缺失分钟为 NaN（`to_canonical` 已做本地压缩表与云端原始表的对齐）；
- 若平台提供的训练集不是该 parquet 布局（例如官方 e2e feather 压缩格式：
  3 档、分、instrument_id），需先转换到上述目录结构，或相应调整
  `data_io.py` 的读取逻辑（当前按 5 档/元/字符串 instrument 读取）。

## 训练设置（model_dynamic1 实际配置）

- 输入：1 分钟 bar，5 个交易日回看，`seq_len = 1200`，37 维特征（14 价格相对
  pre_close + 13 量类 log1p + 10 个委托笔数 `bid/ask_num_orders1-5` 的 log1p，
  私榜重训配置）。随包提交的 `model.json` 为 27 特征权重（公榜推理用），
  推理 notebook 按 json 配置自适应特征数。
- 路由：`fusion_mass_bias`，top-2，patch_lens [5,15,30,60]，
  expert_prior [0.15,0.4,0.3,0.15]，target_load [0.15,0.35,0.35,0.15]，
  bias_update_speed 0.005，load_ema_decay 0.99。
- 微调：AdamW 分组 lr（head 1e-4 / router 5e-5 / experts 2e-5 / backbone 2e-5 /
  loss 1e-4），weight_decay 1e-4，cosine warm restart（T0=3, Tmult=2,
  eta_min=5e-6），grad_clip 1.0，seed 42，15 epochs。
- 数据划分：train 2019-01-02~2023-06-30 / valid 2023H2 / test 2024 全年。
- 评测口径：分数经截面 1%/99% winsorize + z-score + 风格回归取残差后，与
  raw return 计算 IC / RankIC / RankICIR。

## 推理（唯一 notebook）

`reasoning.ipynb` 是唯一推理文件，自包含模型定义、特征变换、DAI 数据加载与
打分逻辑。平台调用入口：

```python
main(datasources, start_date, end_date) -> DataFrame[date, instrument, score]
```

- 读取同目录 `model.json`（normalizer / feature_names / state_dict）。
- 输出三列 `date / instrument / score`，逐日补全交易日、缺失率 ≤ 40%。
- 公榜：平台直接用提交的权重推理。

## 训练（从零重训，完整链路）

本包为完整训练 + 推理包，只依赖第三方库（`requirements.txt`），可独立完成
数据 -> 模型 -> 预训练 -> 微调 -> 导出 -> 推理：

```bash
pip install -r requirements.txt

# 1) 可选：预构建日级特征缓存（多卡训练前建议执行一次）
python pretrain.py --build-cache --config pretrain.yaml

# 2) 预训练（重建式自监督 + 路由学习），只保存最优
#    outputs/pretrain/pretrain_best.pt
python -m pretrain --config pretrain.yaml

# 3) 微调（逐日截面排序，可学习损失权重），只保存最优
#    outputs/finetune/model_best.pt（按验证 RankIC）
python -m train --config finetune.yaml

# 4) 导出最优权重为推理 JSON
python train.py --export \
  --checkpoint outputs/finetune/model_best.pt \
  --out model.json
```

多卡（DDP，1–5 卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  -m pretrain --config pretrain.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  -m train --config finetune.yaml
```

说明：模型类由权重对应训练代码重建（与推理 notebook 内嵌模型完全一致），
微调配置来自权重内嵌配置（精确），预训练配置按训练日志还原。

## 依赖

`torch` / `numpy` / `pandas` / `pyarrow` / `PyYAML` / `tqdm`（见 requirements.txt）。
