# BigAlpha 2026 — 中证 1000 Alpha 因子挖掘参赛工程

> 当前提交只包含部分重点代码，完整代码仓库层次结构复杂暂未整体提交

本仓库是 BigQuant BigAlpha 2026「AI 因子挖掘」赛道的完整参赛工程：
**一条从原始行情到平台可提交代码的自动化流水线**——本地复刻平台评测口径、
用强类型遗传规划（GP）在表达式空间里搜索因子、层次聚类去重后人工挑选，
再把胜出的表达式树逐字翻译成平台 SQL 并通过逐格金标准验证后提交。

> 平台分数以官方榜单为准。本仓库不记录成绩数字——B 项随全场提交池实时漂移，
> 任何本地快照都会过期。

---

## 1. 赛题与本工程的解题思路

### 1.1 评分机制决定了方法

平台评分 `final = 0.3 × A + 0.7 × B`：

| 项 | 定义 | 本地可复刻性 |
|---|---|---|
| **A** | IC 类四指标（IC 均值 / IC_IR / 多空夏普 / 压力测试 IC）在全场参赛因子中的 Rank 平均 | 指标原始值可复刻，Rank 百分位不可（需要全场因子池） |
| **B** | 该因子在**全场所有参赛者提交池**的滚动 ElasticNet 中的边际贡献 | **不可复刻** |

B 项占七成权重，且奖励增量信息、惩罚拥挤与共线。这条机制推出两个直接结论，
构成本工程的方法论主轴：

1. **不提交线性组合因子**。组合因子可被他人的纯因子线性表出，边际贡献归零。
   本项目实测过一个组合因子与它的单一成分簇，组合的 B 项惨败于成分。
   → 提交池只放**纯簇因子**。
2. **搜索目标是"多样性"而非"单点最优"**。既然 B 项按边际贡献打分，
   与已有因子高度相关的强因子不如一个中等强度但正交的因子。
   → 搜索产出必须过**去重收割**，且要主动做**禁词轮**去逼引擎离开主导因子族。

### 1.2 流水线

```
   平台数据                    本地复刻层                    搜索层
┌──────────────┐        ┌────────────────────┐      ┌────────────────────┐
│ bar1m/5m/... │  ───►  │ dai.py 兼容层      │ ───► │ 终端面板 npz       │
│ bar1d        │        │ (duckdb on parquet)│      │ 135 P + 16 AXIS    │
│ financial    │        │ bqlocal/ 评测管线  │      │ 1456 日 × 2074 股  │
│ instruments  │        │ (BARRA 残差化 +    │      └─────────┬──────────┘
│ exposure     │        │  o2o 标签 + 四指标)│                │
└──────────────┘        └────────────────────┘                ▼
                                                    ┌────────────────────┐
   交付层                     收割层                 │ gp/ 强类型 GP 引擎 │
┌──────────────┐        ┌────────────────────┐      │ 37 算子 / 类型闭包 │
│ 平台 SQL 因子│  ◄───  │ 层次聚类 @0.6      │ ◄─── │ 种群 800 × 100 代  │
│ + 金标准验证 │        │ + 全量开箱 holdout │      │ 多进程 × 多种子    │
│ + 确定性加固 │        │ + HTML 查看器      │      └────────────────────┘
└──────┬───────┘        └────────────────────┘
       │
       ▼  提交
┌──────────────┐        ┌────────────────────┐
│  平台评测    │  ───►  │ 分数监控与反推     │  ──► 回灌下一轮决策
└──────────────┘        │ (B 项漂移 / 区间)  │
                        └────────────────────┘
```

---

## 2. 技术方法

### 2.1 本地评测复刻——整条流水线的地基

搜索引擎每秒要给成千上万棵表达式树打分，**打分口径若与平台不一致，
搜索就是在优化一个错误的目标**。因此第一步是把平台评测口径实测定稿：

- **标签**：`open_{T+1} → open_{T+2}`（次日开盘执行的开盘到开盘收益），
  不是常见的 c2c。实现见 `bqlocal/barra.py::next_returns_o2o`、
  `gp/context.py` 的 `y1o`。
- **预处理**：逐日截面去极值（1%/99%）→ zscore → 缺失填 0 → **只在因子侧**做
  风格残差化（10 个 BARRA 风格 + 行业哑变量，无截距 OLS），收益侧保持原始。
  这一点很关键：收益侧残差化对风格共线因子不设防，不是等价替代。
  （平台文档口径为 31 个行业，本地 `exposure` 表实测 32 列；两端 `process_factor`
  与本地残差化的逐日 rank 相关为 0.9998，差异不影响搜索排序。）
- **宇宙**：中证 1000 历史成分股；样本过滤需约 21 个前向交易日。
- **压力测试**：平台未公开定义，本地近似为「市场等权收益最差 10% 交易日上的 RankIC 均值」。

这些结论及其实测证据全部写在 **`factors/platform_eval_notes.md`**（本工程
关于平台口径的唯一权威文档），并由 `bqlocal/evaluate.py` 与 `gp/context.py`
两处独立实现，互为交叉验证。

**`dai.py` 本地兼容层**是另一块地基：它用 duckdb 在本地 parquet 缓存上
按平台表名建视图，使 `factors/` 下的因子代码**同一份文件在本地和平台
AIStudio 两端零修改运行**。兼容层同时吸收了两端的口径差异（价格单位分↔元、
匿名 id ↔ 真实代码、缺失值编码、盘口档位数），并主动对已知陷阱告警。

### 2.2 输入层——终端面板

GP 的叶子节点称为「终端」。终端质量决定搜索天花板，因此终端扩展是本项目
投入最大的方向之一。当前面板规模：

| 类别 | 数量 | 内容 |
|---|---|---|
| **P**（常规面板，逐日截面 winsor + zscore 后参与运算） | 135 | 99 个量价主库终端 + 36 个财务比率终端 |
| **AXIS**（事件轴，原样放行不做标准化） | 16 | `days_since_{high,low,turnmax,ampmax,retmax,retmin}_{20,60,250}`、`days_since_ann` |

GP 词表合计 **151 个终端**。数据网格：**1456 个交易日 × 2074 只股票**
（2019-01-02 ~ 2024-12-31），每张面板 `[1456, 2074] float32`。

终端来自四个数据方向的系统性扩展，每个方向都有独立的候选构造脚本
（`scripts/build_*_candidates.py`）：

1. **五档盘口簿形状**——本地 API 只给 L1-3，通过平台侧导出 L4/5 差量包补齐；
2. **bar1m 高频微结构**——VPIN、tick-rule 订单流不平衡、成交笔数族、
   微价格偏离等；
3. **财务基本面**——363 个财务字段中筛出可用者，按公告日（而非报告期，
   用后者就是未来函数）展开到交易日网格；
4. **价格/量能事件轴**——把「距上次极值多少天」从标量特征升级为可被
   `event_decay` / `within` 等事件算子消费的独立类型。

**终端准入是有门槛的**：`scripts/gp_admission_score.py` 对每个候选终端跑
「回归库 + 闭包字典 + 四门检验」，输出 Tier A/B/C 分档与死刑判定。
被现有词表线性表出的候选会被字典正确击杀，不进池。这道闸门防止面板被
同质终端稀释——面板每加一列，搜索空间指数膨胀，但有效信息未必增加。

### 2.3 搜索层——强类型表达式树 GP

`gp/` 是自研的遗传规划引擎，核心设计：

**强类型语法（v3）**。单一事实源是 `gp/ops.py` 的 `OPS` 注册表
（参数类型串 / 返回型 / 槽位预设集合 / 求值函数），翻译层、闭包字典、
开箱统计一律从这里读，禁止第二份抄本。类型系统：

- `P`：常规面板（默认）
- `AXIS`：事件轴原始值，白名单准入，禁止靠后缀名嗅探
- `BOOL`：布尔面板，供 `gate` / `cond` / 逻辑算子消费
- 常量槽 `TWINDOW` / `FLOAT`：离散预设集合，**不使用连续 ERC**

类型闭包保证随机生成、交叉、变异产出的树永远类型合法——这比"生成后再校验
丢弃"高效得多，也杜绝了运行期类型错误。

**37 个算子**，分五族：算术（`add`/`div_p`/`log_p`/…）、截面
（`cs_rank`/`cs_zscore`/`cs_resid`/`mul_rank`/`cs_top_q`）、时序
（`ts_mean`/`ts_std`/`ts_corr`/`ts_beta`/`ts_resid`/`ts_argmax`/…）、
事件（`event_decay`/`within`/`since_ann_sum`/`ann_delta`/`ann_rank`）、
逻辑与门控（`gt`/`and`/`or`/`not`/`gate`/`cond`）。

**分轴常量预设**（`AXIS_SLOT_SETS`）：`within` 的窗口 d、`event_decay`
的半衰期 h 按事件轴的实际射程分档——公告轴和「距 20 日新高」轴的合理
时间尺度差一个量级，用同一套常量集合是浪费搜索预算。档位由
`scripts/gp_event_decay_calib.py` 实测 IC 衰减曲线标定。

**适应度函数**：

```
fitness = |平台口径 RankIC_IR(训练窗)| − λ × 节点数     (λ = 1e-4)
```

- 默认 `hybrid` 口径：先用廉价的收益侧残差 IC 粗筛全种群，
  只对前 20% 做昂贵的因子侧残差精评，再用两者比值的中位数 κ
  把粗筛分校准到精评量纲。等预算下比全员精评多搜三到五倍的树。
- **样本天数地板**（`min_days_frac=0.6`）：评测窗有效天数不足窗口 60% 的
  统计量一律按 0。这是被实测教训逼出来的——嵌套事件算子能造出只有个位数
  覆盖天的面板，3 个点算出的 IC_IR 会爆到 +65，然后劫持整个种群 19 代。

**进化参数**（生产标准配置）：种群 800、固定 100 代、锦标赛选择（k=5）、
精英保留 5%、交叉 70% / 子树变异 15% / 点变异 10% / hoist 变异 5%、
最大深度 6 / 最大节点 25（常量原子不计节点）。交叉按**积权重类型配对**
（类型 t 以 `count_a(t)·count_b(t)` 抽取），保证全体合法配对等概率。

**两段准入调度**是本项目最有价值的单项优化：`--admit-top-k 150
--admit-top-k-late 0 --admit-late-frac 0.334`，即前 2/3 代只让 fitness
前 150 的候选进入昂贵的验证窗评测，后 1/3 代全开。动机来自插桩实测——
基线跑中**前 3 代的准入存活为零，末 3 代贡献了 70% 的 HOF 成员**。
把验证预算后移后，总评测量降到基线的 48%，而验证窗 |IR| 只掉 0.6%。
**等预算下"分配"显著优于"均匀"**。

**HOF（名人堂）与同质筛**：候选须先过训练窗 |IC_IR| ≥ 0.2 的门槛，
再与堂内成员比相关（训练窗逐日截面 pct 秩池化 Pearson，|ρ| ≥ 0.7 判同质）——
与堂内成员同质时，只有**击败全部同质成员**才能替换它们，形成棘轮
（旧版只删一个，导致同质并存，已修）。堂容量 50，**排序键是验证窗 |IC_IR|**
（训练窗定 fitness 与准入，验证窗只定排序与簇代表挑选）。相关筛本身做了
两段式提速：先抽 1/8 交易日估 |ρ|，距阈值超过 0.08 直接判定，带内才回退全样本。

**四层缓存**（一致性语义各不相同，`gp/CLAUDE.md` 有完整规范）：

| 层 | 位置 | 内容 | 一致性风险 |
|---|---|---|---|
| `fit_cache` | 磁盘 JSON | canon 表达式串 → 训练窗 fitness | **最高**：口径变更须手动升 `_FIT_META.ver`，否则新代码吃旧数字 |
| `SubtreeCache` | 进程内 LRU | 子树面板，存储时舍入 float32 | 已根治：求值值是表达式的**纯函数**，与缓存命中/驱逐/求值顺序无关 |
| 终端/标签 npz | 磁盘 | 面板矩阵 | 由数据指纹（mtime+size）自动使 fit_cache 失效 |
| `_yrank_cache` | 进程内 | 标签侧秩 | 无（纯等价加速） |

其中 `fit_cache` 的数据指纹机制是防「静默错误」的关键：npz 重建后指纹变化，
整个 fitness 缓存自动作废，不依赖人工记得清缓存。**吃了毒缓存的代价是
整轮搜索作废且可能察觉不到**——比崩溃危险。

**逐代检查点**（`factor_out/ckpt_<tag>/gen_XXXX.json`，原子写）支持任意代续跑，
也使「全过程并集开箱」成为可能：从所有检查点取唯一树的并集，而不是只看
末代种群。这一点有实证价值——某轮实测末代 595 棵树 vs 全过程并集 1415 棵，
**末代丢掉了大量中途见顶后被侵蚀掉的优质树**。

### 2.4 收割层——聚类去重与全量开箱

搜索产出上千棵树，其中大量是同一个因子的换皮写法。收割流程（
`.claude/skills/gp-unbox/SKILL.md` 定稿）：

1. **合并累计池**——新轮 HOF 并入历史池（跨轮去重，语法版本不同须先迁移）；
2. **层次聚类**——两两相关用训练窗逐日截面 pct 秩池化 Pearson，
   **距离取 `1 − |ρ|`**（符号不变：因子与其负版是同一个信号），
   average-linkage 层次聚类、`fcluster` 按距离 0.4 切（即同簇平均 |ρ| ≥ 0.6），
   **簇代表 = 验证窗 |IC_IR| 最高者**（`scripts/harvest_gp_hof.py`）。
   相关矩阵按成员串对齐缓存，池并轮时增量复用旧块只算缺格，
   成本从 O(N²) 降到 O(N·ΔN)。选层次聚类而非贪心是为了确定性——
   旧的贪心版按验证 |IR| 排队，实测出现过假分裂；
3. **全量开箱**——全体簇代表出 holdout(2024H2) 指标 + 两两相关矩阵
   （`scripts/eval_all_clusters.py`）。口径是 **IC 方向按验证窗归一**：
   按验证窗 IC 符号翻正后才算指标，符号记入 `dir` 字段；归一后相关为正
   = 同向重复，为负 = 真正的反向信息；
4. **HTML 查看器**——`scripts/build_cluster_viewer.py` 出单文件离线页面
   （排序表 + 相关热图 + 新/老池归属徽章），由人做最终挑选。

**holdout(2024H2) 对进化引擎全程锁定**——引擎的训练窗是 2019-07~2023-06，
验证窗 2023-07~2024-06，holdout 只在开箱流程显式出数。

### 2.5 交付层——平台 SQL 翻译与三道验证关卡

挖掘端的因子是 numpy 表达式树，平台要的是一份跑 DAI SQL 的 notebook。
**这一步是整条流水线最容易出错、也最容易被低估的环节**。

每个入选簇翻译成 `factors/gp_<批次>_c<簇号>.py`，文件 docstring 固定记录：
中文语义、来源池与开箱日期、holdout 指标、`dir` 符号、
`Z[·]` 形式的完整表达式、终端 SQL 口径、时序回看深度与预热天数。

翻译完成后必须过三道关卡：

**关卡一：金标准逐格验证**（`scripts/verify_<批次>_factors.py`，每批一个）
——用 `dai` 兼容层跑**即将提交的真 SQL**，与挖掘端（npz 终端 + `gp.ops`
求值）的面板做逐日 Spearman，要求 min = median = 1.000000。
不达标的必须做对照实验锁死根因，证明是数据差异而非翻译错误
（例如本地 instruments 比 exposure 网格多一只幽灵股，会被硬分支算子放大）。

**关卡二：确定性/未来函数检测复现器**（`verify_*_truncation.py`、
`verify_swap_lookahead.py`）——平台有两套未来函数检测，都在本地做了复刻：

- *截断式*：全窗跑一次、截断后再跑一次，截断日之前必须逐格 `max|diff| = 0`。
  本项目实测过 5 只被误标的因子，根因不是未来函数而是
  **DuckDB 并行聚合的浮点合并顺序不确定**导致两跑数值不可复现。
  修法成为提交代码的确定性规范：矩类统计走 DECIMAL 精确矩量和、
  `LAST` 改 `MAX(CASE ...)`、终端 f32 量化 + 输出 round(8) + 固定行序。
- *换表式*：平台用两张历史覆盖起点不同的检测表跑同一份代码比对结果。
  长回看因子会因为检测表历史只有 ~60 个交易日而误标。
  修法：**预热段固定读正式表，只有评测段才用 datasources 传入的表**
  （无前视，正式打分逐位不变），且预热 ≥ 300 天。

**关卡三：覆盖率审计**（`scripts/audit_factor_coverage.py`）——平台 run error
的一类病因是截面覆盖不足。本项目用**全样本交叉表**（已出分因子 vs
run error 因子，`audit_coverage_vs_outcome.py`）定位判别量，推翻了
"单日缺失率 > 40% 即拒"的旧表述：破线天数完全不 discriminate，
真判据是**最差单日的覆盖深度**。杀手日是 2020-02-03（新冠开市千股跌停），
依赖成交/盘口的终端在该日整截面失效。修法是逐股 `ffill(limit=5)` 沿用
最近一次有定义的信号（只读过去，无前视）。

### 2.6 反馈层——平台分数监控

`scripts/fetch_platform_scores.py` 定时快照全部提交的 A/B/final 分数，
增量落 JSONL；`score_watch_to_csv.py` 转成「因子为列 / 评测轮次为行」的
透视表。主要观察对象是 **B 项随全场池的实时漂移**——B 是全场博弈的产物，
不是因子的固有属性。

`scripts/infer_eval_window.py` 反向工程平台的评测区间：逐日算 IC 与五分层
多空序列，用前缀和 O(1) 出任意 `(start, end)` 窗口的指标，与官方周报公布的
(IC, ICIR, Sharpe) 做误差面搜索，再与覆盖率反推出的硬约束交叉验证。

---

## 3. 项目结构

```
CLAUDE.md               面向 AI 协作者的项目约定（模块地图 / 环境 / 硬纪律）
dai.py                  本地 dai 兼容层：duckdb 读 parquet 缓存，按平台表名建视图，
                        使 factors/ 代码两端零修改可跑（切勿上传到 AIStudio）
run / run.cmd           正式评测简写（六年全窗口 + BARRA 残差化）

gp/                     表达式树因子挖掘引擎
  ops.py                  算子注册表 = 单一事实源（类型串/槽位预设/求值函数）
  engine.py               进化主循环、fitness、HOF、缓存、检查点
  context.py              评测上下文：标签、BARRA 暴露、platform_ic / label_ic
  CLAUDE.md               缓存约束与优化验证规范（改引擎前必读）

bqlocal/                平台评测的本地复刻
  config.py               路径 / 逻辑表名映射 / 数据范围
  data.py                 权限探测 / 按月下载（200MB 限额自适应二分）/ 规范化
  evaluate.py             A 项四指标 + 诊断 + 平台格式校验 + 因子相关性
  barra.py                id 映射 / 风格残差化 / o2o 标签
  factor.py               因子文件执行器

factors/                因子实现与平台提交源码（本地/AIStudio 通用）
  platform_eval_notes.md  **平台评测口径的唯一权威文档**（实测定稿）
  README.md               财务表/日线表口径与日期处理规范
  gp_<批次>_c<簇号>.py    GP 簇因子（ck / x2 / nr / af / afg / h06 六个批次）
  gp_terminals/PLAN.md    日频原子终端规格
  *_candidates.csv        四个数据方向的候选终端清单与准入判决

scripts/                62 个脚本，按职能分七组：
  aistudio_export_*.py    平台侧数据导出（辅助表 / 财务 / 日线 / L4-5 差量包）
  download_data.py        本地缓存下载（幂等续传）
  build_*_terminals.py    终端面板构建
  build_*_candidates.py   候选终端构造（五档 / bar1m / 财务 / 事件轴 / 笔数族）
  gp_admission_score.py   终端准入评分（回归库 + 闭包字典 + 四门 + Tier 分档）
  run_gp_mine.py          **挖掘主入口**
  run_factor.py           单/多因子本地评测报告
  harvest_gp_hof.py       层次聚类收割
  eval_all_clusters.py    全量开箱（簇代表出 holdout）
  build_cluster_viewer.py 开箱结果 → 单文件离线 HTML 查看器
  verify_*.py             23 个验证脚本（金标准 / 检测器复刻 / 等价性回归）
  audit_*.py              覆盖率审计与全样本对照
  fetch_platform_scores.py / score_watch_to_csv.py / infer_eval_window.py
                          平台分数监控与评测区间反推

docs/
  superpowers/specs/      方案定稿（唯一权威的决策结论）
  gp_input_expansion_handoff.md   输入扩展的实施记录与实证证据
  gp_speedup_handoff.md           引擎优化的实测数字与方法论坑

.claude/skills/gp-unbox/  开箱流程的标准作业程序

data_cache/ , factor_out/   数据缓存与挖掘产出（均已 gitignore）
```

**关于仓库内容边界**：`data_cache/`（parquet 缓存）与 `factor_out/`
（HOF 池、survivors、开箱 JSON、查看器、日志）整个不入库。因此
`docs/` 下 handoff 文档里的实测数字是那些实验唯一留存的证据记录，
写作时按「结论 + 证据」的格式保存，而不只是结论。

---

## 4. 复现步骤

环境：Python 3.11，一律用仓库内 `.venv/Scripts/python.exe`。

```bash
# 1. 下载行情缓存（幂等，中断重跑即续传）
.venv/Scripts/python.exe scripts/download_data.py --freqs bar30m,bar15m,bar5m,bar1m

# 2. 辅助表（instruments / factorlib / exposure / financial / bar1d）
#    本地 API 无权限，须经 AIStudio 中转：
#    上传 scripts/aistudio_export*.py 到平台 notebook 运行 → 下载产物 →
#    解压到 data_cache/ → python scripts/verify_aux.py 验收
.venv/Scripts/python.exe scripts/build_id_mapping.py    # 匿名 id ↔ 真实代码

# 3. 构建 GP 终端面板与评测上下文
.venv/Scripts/python.exe scripts/build_gp_terminals.py
.venv/Scripts/python.exe scripts/build_gp_context.py

# 4. 挖掘（标准配置：pop800 × 100 代 + 两段准入调度）
.venv/Scripts/python.exe scripts/run_gp_mine.py \
    --pop 800 --gens 100 --seeds 90 --tag x2_s90 \
    --admit-top-k 150 --admit-top-k-late 0 --admit-late-frac 0.334

# 5. 开箱：收割 → 全量出 holdout → 查看器（池并轮时必带 --reuse-corr）
.venv/Scripts/python.exe scripts/harvest_gp_hof.py factor_out/gp_hof_<池>.json \
    --reuse-corr factor_out/gp_hof_<上一轮池>_corr.npz \
    --out factor_out/gp_survivors_<池>.json
.venv/Scripts/python.exe scripts/eval_all_clusters.py \
    --surv factor_out/gp_survivors_<池>.json --out factor_out/gp_<池>_all_unbox.json
.venv/Scripts/python.exe scripts/build_cluster_viewer.py \
    --src factor_out/gp_<池>_all_unbox.json --out factor_out/cluster_viewer_<池>.html

# 6. 单因子本地评测（六年全窗口 + BARRA 残差化）
./run factors/gp_ck_c102.py

# 7. 提交前验证（金标准逐格 + 覆盖率审计）
.venv/Scripts/python.exe scripts/verify_ck_factors.py 102
.venv/Scripts/python.exe scripts/audit_factor_coverage.py factors/gp_ck_c102.py

# 8. 提交：把 factors/gp_ck_c102.py 的内容原样复制到平台 notebook
```

**资源需求**：挖掘进程约 13~14.4 GB 内存（pop800 × 135 面板），
引擎受 GIL 束缚约占 1 核。64 GB 机器的并行上限是 3 进程；
`--threads` 实测负收益（`platform_ic` 是小数组 numpy 调用，线程只在抢 GIL），
并行靠进程级。一轮 12 种子 × 100 代约 41 小时。

