# BigQuant 因子生成工作流

基于 RAG + LLM 的两阶段因子策略开发管线：自然语言输入 → 自动梳理策略与数据字段 → 生成可执行的 DAI SQL 代码。

## 目录结构

```
bigquant/
├── main.py                        # BigQuant 示例因子（策略模板参考）
├── structure.md                   # 架构设计文档
├── optimization.md                # 优化方案与实施记录
├── README.md                      # 本文件
├── OPERATION.md                   # 操作指南
│
├── data/                          # 数据表字段 schema（5 张表，509 个字段）
│   ├── 5分钟附盘口(...).json       #  42 字段：OHLCV + 5档盘口买卖价/量/笔数
│   ├── 日行情数据(...).json         #  13 字段：日频 OHLCV + 换手率
│   ├── 财务数据(...).json           # 368 字段：资产负债表/利润表/现金流量表/衍生指标
│   ├── 因子暴露(...).json           #  48 字段：风格因子 + 行业哑变量
│   ├── 因子库(...).json             #  38 字段：预计算技术指标/财务因子
│   └── factor_templates.json      # 27 个因子策略模板
│
├── calculate/                     # DAI SQL 函数库
│   ├── dai_functions.json         # 818 个 DAI 函数（含操作符/标量/聚合/窗口/TA-Lib/macro）
│   └── dai_functions.md           # 同上，人读 Markdown 格式
│
├── workflow/                      # 核心管线
│   ├── pipeline.py                # 主编排：Stage1 多轮对话 → Stage2 代码生成
│   ├── embedder.py                # 阿里云 text-embedding-v4（RAG 向量化）
│   ├── retriever.py               # 兼容层 → retrievers/
│   ├── llm.py                     # DeepSeek V4 API（推理 + 结构化输出）
│   ├── diverge.py                 # 发散模块：4 维度批量生成变体
│   ├── validator.py               # AST 校验 + 白名单 + 语义检查
│   ├── build_index.py             # 索引构建（一次性）
│   ├── stage1_strategy.py         # Stage 1：多轮策略编写
│   ├── stage2_formula.py          # Stage 2：DAI SQL 生成
│   └── prompts/                   # System prompts
│       ├── stage1_system.txt      # 策略研究员角色
│       └── stage2_system.txt      # DAI SQL 专家角色
│
└── chroma_db/                     # 向量索引（build_index 生成）
    ├── fields/                    # 字段 schema 索引
    └── functions/                 # DAI 函数索引
```

## 功能

### Stage 1 — 多轮策略编写

- 用户用自然语言描述策略想法    
- 智能体检索数据字段 schema，帮助梳理策略逻辑
- 支持多轮修改："加上波动率调整"、"去掉行业中性化"
- 输出结构化 JSON：策略名称、描述、所需字段列表、计算逻辑概要
- 字段白名单硬约束：输出的字段必须在 data/ 中存在

### Stage 2 — DAI SQL 生成

- 根据 Stage 1 确认的策略，自动检索相关 DAI 函数
- 生成完整可执行的 DAI SQL（CTE 风格、三列输出、除零保护）
- 参考 main.py 的代码风格和结构
- 函数白名单硬约束：只使用 calculate/ 中存在的函数
- SQL 质量检查：三列/除零/log 保护

### 检索特性

- **混合检索**：阿里云 text-embedding-v4（语义）+ BM25（关键词）+ RRF 融合
- **表级智能排序**：自动识别查询中的表名偏好（"财务因子"→财务字段排前）
- **jieba 中文分词**：词组级匹配，提升中文查询召回率
- **分类预过滤**：窗口/TA-Lib/聚合等关键词自动过滤函数分类

### 发散与批量生成

- Stage 1 确认后自动从 4 个维度发散 4~5 个变体
- **经济解释** — 同一数据的不同假说方向
- **代理变量** — 同一概念用不同字段表达
- **时间尺度** — 不同窗口周期或聚合方式
- **数据来源** — 跨表迁移（bar1m→bar1d→financial）
- 并行执行所有变体的 Stage 2，最后汇总保存到 `output/batch_xxx/`

### 质量保障

- 字段白名单校验 → 失败重试（最多 2 次）
- 函数白名单校验 → 失败重试（最多 2 次）
- SQL 三列/除零/log 保护检查
- Few-Shot 示例 + Chain-of-Thought 推理步骤

## 技术栈

| 组件 | 选型 |
|:---|:---|
| Embedding | 阿里云 text-embedding-v4（1024 维） |
| 向量库 | ChromaDB（持久化，cosine 距离） |
| 关键词索引 | BM25（rank-bm25）+ jieba 分词 |
| 推理 LLM | DeepSeek V4 Pro |
| 检索融合 | RRF（Reciprocal Rank Fusion，k=60） |

## 环境变量

```bash
DASHSCOPE_API_KEY    # 阿里云灵积
DEEPSEEK_API_KEY     # DeepSeek API
DEEPSEEK_BASE_URL    # DeepSeek endpoint（默认 https://api.deepseek.com）
```

## 数据表

| 表 | 字段数 | 主要内容 |
|:---|:---|:---|
| 5分钟附盘口 | 42 | 分钟级 OHLCV + 5 档盘口买卖价/量/委托笔数 |
| 日行情数据 | 13 | 日频 OHLCV + 换手率 + 涨跌幅 |
| 财务数据 | 368 | 资产负债表 + 利润表 + 现金流量表 + 衍生指标 |
| 因子暴露 | 48 | 10 个风格因子 + 33 个行业哑变量 |
| 因子库 | 38 | 预计算技术指标 + 财务因子 + 资金流向 |

## 因子模板库（27 个）

| 分类 | 模板 |
|:---|:---|
| 盘口因子 | 日内盘口不平衡度、VWAP 偏离 |
| 动量因子 | 日内动量、日频动量、行业中性动量 |
| 反转因子 | 日内反转、日频反转、隔夜跳空 |
| 波动率因子 | 日频波动率、残差波动率 |
| 流动性因子 | 换手率、Amihud 非流动性 |
| 估值因子 | PE、账面市值比 |
| 质量因子 | ROE、毛利率、FCF 收益率、资产增长率、盈利稳定性 |
| 规模因子 | 市值因子 |
| 风险因子 | Beta、杠杆 |
| 成长因子 | 营收/盈利增长率 |
| 行为因子 | 最大日收益（MAX）、偏度 |
| 资金流向 | 主力资金净买入 |
| 技术指标 | MACD+RSI 复合信号 |
