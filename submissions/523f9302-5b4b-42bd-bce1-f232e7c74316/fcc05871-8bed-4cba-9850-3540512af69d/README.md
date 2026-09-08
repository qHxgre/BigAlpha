# v136final — 竞赛终选包:60 bars × 120d,三 seed 集成

**模型**:cs 归一化 hist_day(日内 GRU 8→256 → 跨日 GRU 256 → HIST 隐概念图 → Linear),
每天取尾盘 60 根 1 分钟 bar(14:01 起)、回看 120 个交易日(seq 7200),core8 特征
(盘口/成交 8 通道),label-rank 标签,IC 损失,日批。三个同配方 seed(0/1/2),
分数 = 三成员 raw 输出的算术平均。

## 包内文件
| 文件 | 作用 |
|---|---|
| `v136final.ipynb` | 提交笔记本;cell-1 = 完整推理库(与 inference_lib.py 逐字节一致) |
| `inference_lib.py` | 推理:`main(datasources, start, end)` → 每日横截面分数 |
| `final_model.json` | 已训权重(公榜训练集 2019-2023)+ 归一化统计 + 全部超参 |
| `train.py` | 完整重训脚本(自带服务器取数);私榜重训入口 |
| `requirements.txt` | 依赖声明 |

## 如何跑推理(公榜路径)
平台直接执行笔记本;本地等价调用:
```python
from inference_lib import main
scores = main(None, "2025-01-01", "2025-06-30")   # DataFrame[date, instrument, score]
```

## 如何跑重训(私榜路径)
```bash
python train.py
```
零参数即可。流程:
1. **数据获取**(自动二选一):目录下已有 `e2e_data/bigalpha_2026_e2e_bar1m/*.feather`
   月度文件 → 直接使用;没有 → 通过平台 `dai` 拉取云表 `bigalpha_2026_stock_bar1m`
   (与推理同源的 SQL 下推 + 10 天切片回退),**内联 `bigalpha_2026_instruments`
   逐日过滤比赛宇宙**(裸表是全市场,不过滤会训成另一个模型),价格字段换算为
   本地 fen 整数制 —— 产出与官方训练镜像逐值一致(已验证:整月 126 万行、
   25 字段 max|d|=0);
2. **顺序训练三个 seed(0/1/2)**:每 seed 固定 12 epochs、cosine 学习率、取最后
   epoch(--patience 0,无验证选择),训练区间 2019-01-01..2023-12-31;
3. **自检 + 重建**:每个新 ckpt 与 manifest 核对几何/参数量,通过后把三份新权重
   打包回 `final_model.json`(原地覆盖)。

预期时长:A6000 约 4 小时/seed(共 ~12h);数据桥首跑另需 ~0.5-1.5h。
显存 <10GB;数据装载内存峰值(宇宙过滤后)约 30-50GB。

## 如何修改训练时段
只改 `train.py` 里 `TRAIN_ARGS` 的四个时间旗标:
```python
"--train-start", "2019-01-01",
"--train-end",   "2023-12-31",
"--valid-start", "2024-01-01",
"--valid-end",   "2024-12-31",
```
数据桥的拉取窗口**自动跟随**(起点自动前推 ~232 个日历日作为 120 交易日回看的
缓冲,终点 = valid-end)。例如私榜若要求训练到 2024 年底:把 train-end 改为
2024-12-31、valid 区间改为 2025 年即可,其余零改动。epochs 可用命令行覆盖
(`python train.py --epochs N`,仅用于冒烟;正式复现请保持 12)。

## 随机性与复现精度
确定性开关已开:同硬件 + 同 torch 版本重跑逐位一致;跨硬件为统计等价
(公榜实测:同 seed 跨硬件孪生差 -0.017,新 seed 差 -0.05 量级,均为方差非偏差)。
seed 钉死于 TRAIN_ARGS 与集成驱动(0/1/2),不依赖外部状态。

## 已知边界
- torch ≥2.6 的 `weights_only` 默认变更已处理(驱动内显式 False,只加载本包自产文件);
- 训练数据只需 1 分钟 bar 表 + 成员表,无外部行业/风格数据(合规:无手工特征)。
