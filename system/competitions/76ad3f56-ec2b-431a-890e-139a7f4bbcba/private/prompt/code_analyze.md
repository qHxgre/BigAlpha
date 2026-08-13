# 任务：逐 submission 因子代码分析与统一汇总

本任务只分为两个阶段：

1. 逐一静态分析 `prepared/submissions/` 下的每个 submission，并为每个 submission 生成一份 JSON。
2. 读取全部 submission JSON，生成一份总 JSON 和一份 Markdown 分析报告。

允许读取 `prepared/metadata.json`，但只可用于获取 submission 与用户、团队、团队成员、公榜分数之间的关联。不得使用 metadata 中的分数、排名或参赛者资料判断因子质量、创新性、原创性或规则合规性。

不要生成单 submission Markdown、独立团队报告或独立成员报告。团队分组只体现在统一的 `summary.json` 和 `summary.md` 中。

## 一、固定路径

submission 根目录：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prepared/submissions`

规则摘要：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260812_180129/artifacts/coder_analysis/rules_summary.md`

参赛者信息：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/scripts/private/alphathon__user.csv`

submission、用户与团队关联信息：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prepared/metadata.json`

输出目录：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260812_180129/artifacts/coder_analysis`

JSON 模板：

- 单 submission：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prompt/submission_analysis_template.json`
- 总汇总：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prompt/summary_template.json`

开始前必须读取 `rules_summary.md`。若文件不存在、为空或无法读取，立即停止并说明原因，不得自行从其他材料补充比赛规则。

## 二、阶段 1：逐一分析 submission

遍历 `prepared/submissions/` 的所有一级子目录。一级目录名就是 `submission_id`。每份代码必须独立审查，不得根据团队、成员、公榜分数或排名改变代码评价或合规结论。metadata 仅用于身份关联和公榜分数。

每完成一个 submission，立即输出：

`.../artifacts/coder_analysis/submissions/<submission_id>.json`

不生成 `<submission_id>.md`。

### 2.1 静态审查边界

- 只能静态阅读源码、配置、依赖文件、Markdown、文本、JSON，以及 notebook 的 JSON 内容和代码单元。
- 禁止执行参赛代码、导入参赛模块、联网运行代码、查询数据，或加载/反序列化模型与数据文件。
- 不得加载 pickle、joblib、torch、parquet、feather 等文件。二进制文件只记录路径、类型和大小。
- 不得声称已经运行因子、查询数据、完成训练、加载权重、验证输出或复算得分。
- 若 submission 中存在多个 notebook、缺少 notebook 或入口不清晰，应记录材料/接口问题，但仍分析所有可读代码。

### 2.2 每个 submission 要分析的内容

1. **实现逻辑**：识别 `main(datasets, start_date, end_date)` 或实际入口，追踪数据查询、预处理、因子计算、日频聚合、截面处理和最终 `date`、`instrument`、`factor` 输出链路。
2. **数据使用**：列出数据表、逻辑表名、字段、频率、股票池、查询日期、warmup、过滤、连接、对齐、填充、复权、PIT 和聚合方式；判断是否存在外部或越权数据线索。
3. **因子构造**：还原因子公式或算法，说明输入变量、窗口、方向、经济或统计逻辑、日内到日频的聚合，以及平台处理前参赛者自行进行的去极值、标准化、中性化或截面变换。
4. **时序与泄漏**：逐项检查 rolling/shift/lead、排序方向、窗口端点、财务公告时点、训练标签、训练—输出区间隔离、全样本拟合、回填和日期映射。重点判断 t 日因子是否只使用当时可获得的信息。
5. **模型与 AI**：若使用模型、LLM、强化学习、遗传算法、神经网络、AutoML 或自动搜索，说明特征、标签、切分、目标函数、超参数、随机种子、模型或权重来源、训练和推理一致性及 AI 的实际参与环节；未使用则明确说明。
6. **赛道归属**：从提交材料中识别其声明的传统量化或 AI 智能赛道。传统赛道重点检查是否以受限的自动生成技术作为因子生成主体；AI 赛道重点检查是否由 AI 主导并提供足以定位关键环节的说明材料。不能确认声明时不得猜测。
7. **接口与可执行性**：静态检查唯一 notebook、入口签名、依赖、硬编码路径、网络调用、运行时间风险、返回类型、三列契约、列名、日期范围、交易日覆盖和每日缺失率风险。
8. **代码质量与复现性**：评价正确性、完整性、可读性、健壮性、效率、可维护性、依赖完整性、确定性、异常处理和文档说明。
9. **合规与诚信风险**：严格按 `rules_summary.md` 检查未来数据、外部数据、数据投毒、覆盖度规避、多因子拼接、重复微调、简单反向、系统规避及跨提交高度相似等线索。单份代码相似不能直接证明抄袭或串通。
10. **创新与人工复核**：说明方法特点、经济解释、优势、局限和失效场景，并提出可操作的人工复核问题。

重要判断必须给出 submission 内相对路径、函数/变量名及尽可能准确的行号，并明确区分：代码已证实、基于代码推测、材料不足无法核验。缺失材料时仍应记录已确认链路、断点和影响，不能只写“无法分析”。高分不能作为优秀、创新或违规的证据。

### 2.3 单 submission JSON

生成前读取 `submission_analysis_template.json`，严格使用模板对象结构：不得删除、重命名或新增对象字段；数组可增删示例元素；空值使用 `null`、`[]` 或 `{}`；输出必须是合法 UTF-8 JSON。

固定枚举：

- 分析状态：`完成`、`部分完成`、`无法分析`
- 置信度：`高`、`中`、`低`
- 方案类型：`传统统计因子`、`规则或公式因子`、`机器学习因子`、`深度学习因子`、`AI 自动生成或优化因子`、`混合方法`、`无法确认`
- 质量等级：`优秀`、`良好`、`一般`、`较差`、`无法评价`
- 合规状态：`符合`、`违规`、`可疑`、`无法核验`、`不适用`
- 风险等级：`高`、`中`、`低`、`无法核验`
- 人工复核优先级：`高`、`中`、`低`
- `rank_source`：有 metadata 分数时写 `metadata.public_score`；`official_rank` 仍写 `null`

“违规”必须同时具备明确规则条款和 submission 文件证据；只有线索时写“可疑”，信息不足时写“无法核验”。静态代码无法证明交易日完整性、每日覆盖度或三小时内运行完成时，应评估风险并标为“无法核验”，不得假装运行验证。

## 三、阶段 2：生成统一汇总

所有 submission JSON 完成后，读取它们并且只生成：

- `.../artifacts/coder_analysis/summary.json`
- `.../artifacts/coder_analysis/summary.md`

汇总以团队为单位分组。有 `team_id` 时按 `team_id` 合并；没有 `team_id` 时按 `user_id` 各自成组。团队内 submissions 按 `submission_id` 升序排列。

允许根据 metadata 的 `public_score` 派生公榜排名，但不得称为官方排名或私榜排名，也不得按代码质量重新排名：

- submission 分数取对应 `public_score`；
- submission 排名在全部 submissions 中按 `public_score` 从高到低计算；
- 团队分数取组内最高 submission 的 `public_score`；
- 团队排名按团队分数从高到低计算；
- 使用竞赛排名法：同分并列，后续名次留空档；
- `ranking_basis.entity` 写 `team`；
- `ranking_basis.source` 写 `prepared/metadata.json public_score`；
- 描述中明确这是按上述算法派生的公榜排名，不是官方私榜排名。

### 3.1 补充参赛者信息

读取 `prepared/metadata.json` 和 `alphathon__user.csv`。metadata 只用于 submission、用户、团队、成员与公榜分数关联；CSV 的 `data` 列为 JSON 字符串，可补充姓名、电话、学校、专业和学历。

匹配规则：先按当前 `competition_id` 过滤 CSV，再按 `user_id` 精确匹配；只接受唯一匹配。零条或多条时不得猜测，字段写 `null` 并列入无法核验事项。团队成员优先取 metadata，资料再由 CSV 补充。`email` 字段因模板存在而保留，但统一写 `null`，Markdown 不展示邮箱。个人敏感信息只写入指定私有输出，不在终端回复中展示。

### 3.2 summary.json

生成前读取 `summary_template.json` 并严格遵循其结构。每个 `rankings` 元素代表一个团队或个人组；`members` 放全部成员；`submissions` 放组内全部提交。每份 submission 的方案、质量、合规、风险和创新结论必须直接来自对应单 submission JSON，不在汇总阶段重新分析或改变结论。

### 3.3 summary.md

Markdown 报告应可独立阅读，包含：

1. 分析范围、数量、完成情况、静态审查边界和规则依据；
2. 团队总览表：团队/个人、成员资料、submission IDs、团队公榜分数及派生排名、赛道、方案类型、质量、合规和主要风险；
3. 按团队展示各 submission 的核心逻辑、数据、因子构造、时序、模型/AI、接口、质量、合规、证据、疑点、无法核验事项与人工复核建议；
4. 横向比较赛道归属、数据使用、因子类型、时序泄漏、代码质量、复现性、创新性和跨提交相似性线索；
5. 按严重度整理全局风险与人工复核清单。

不要原样粘贴全部 JSON，也不能只写一句结论。终端完成回复只报告生成文件路径和简短统计，不展示成员电话等敏感信息。
