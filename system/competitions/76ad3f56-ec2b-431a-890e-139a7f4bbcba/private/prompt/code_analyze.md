# 任务：逐 submission 因子代码分析与统一汇总

本任务只分为两个阶段：

1. 逐一静态分析 `prepared/submissions/` 下的每个 submission，并为每个 submission 生成一份 JSON。
2. 读取全部 submission JSON，生成一份总 JSON 和一份 Markdown 分析报告。

允许读取 `prepared/metadata.json`，但只可用于获取 submission 与用户、团队、团队成员之间的关联。私榜分数与排名只取指定的私榜结果文件。不得使用任何分数、排名或参赛者资料判断因子质量、创新性、原创性或规则合规性。

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

私榜结果：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260812_180129/artifacts/team_private_leaderboard.json`

输出目录：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260812_180129/artifacts/coder_analysis`

JSON 模板：

- 单 submission：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prompt/submission_analysis_template.json`
- 总汇总：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prompt/summary_template.json`

开始前必须读取 `rules_summary.md`。若文件不存在、为空或无法读取，立即停止并说明原因，不得自行从其他材料补充比赛规则。

## 二、阶段 1：逐一分析 submission

遍历 `prepared/submissions/` 的所有一级子目录。一级目录名就是 `submission_id`。每份代码必须独立审查，不得根据团队、成员、私榜分数或排名改变代码评价或合规结论。metadata 仅用于身份关联；私榜结果仅用于填写分数与排名。

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
4. **时序与泄漏**：逐项检查 rolling/shift/lead、排序方向、窗口端点、财务公告时点、训练标签、训练—输出区间隔离、全样本拟合、回填和日期映射。必须按下文“时序与前视判定口径”区分训练标签、同一交易日日内计算和跨交易日未来信息；重点判断输出交易日 `t` 的因子特征或因子值是否使用了 `t+1` 或更晚交易日的信息。
5. **模型与 AI**：若使用模型、LLM、强化学习、遗传算法、神经网络、AutoML 或自动搜索，说明特征、标签、切分、目标函数、超参数、随机种子、模型或权重来源、训练和推理一致性及 AI 的实际参与环节；未使用则明确说明。
6. **赛道归属**：从提交材料中识别其声明的传统量化或 AI 智能赛道。传统赛道重点检查是否以受限的自动生成技术作为因子生成主体；AI 赛道重点检查是否由 AI 主导并提供足以定位关键环节的说明材料。不能确认声明时不得猜测。
7. **接口与可执行性**：静态检查唯一 notebook、入口签名、依赖、硬编码路径、网络调用、运行时间风险、返回类型、三列契约、列名、日期范围、交易日覆盖和每日缺失率风险。
8. **代码质量与复现性**：评价正确性、完整性、可读性、健壮性、效率、可维护性、依赖完整性、确定性、异常处理和文档说明。
9. **合规与诚信风险**：严格按 `rules_summary.md` 检查未来数据、外部数据、数据投毒、覆盖度规避、多因子拼接、重复微调、简单反向、系统规避及跨提交高度相似等线索。单份代码相似不能直接证明抄袭或串通。
10. **创新与人工复核**：说明方法特点、经济解释、优势、局限和失效场景，并提出可操作的人工复核问题。

重要判断必须给出 submission 内相对路径、函数/变量名及尽可能准确的行号，并明确区分：代码已证实、基于代码推测、材料不足无法核验。缺失材料时仍应记录已确认链路、断点和影响，不能只写“无法分析”。高分不能作为优秀、创新或违规的证据。

### 2.3 时序与前视判定口径（强制）

本任务评审的是由高频数据构建并按交易日输出的**日频因子**。日期 `t` 表示该交易日完整日内数据聚合后形成的日频因子归属日，而不是日内某个更早的下单时刻。除非 `rules_summary.md` 另有明确规定，必须遵守以下口径；不得仅凭关键词命中就判定前视：

1. **训练标签中的未来函数不算前视**：用于构造监督学习目标、收益标签或评估标签的 `shift(-1)`、`shift(-N)`、`lead`、未来收益、下一日价格等是正常标签定义，本身不属于因子前视。必须继续追踪数据流：只有当这些未来标签、由其直接计算出的值或包含未来期信息的统计量进入推理特征、输出因子，或训练/验证/模型选择没有与输出日期隔离时，才记录标签泄漏或前视风险。
2. **同一交易日内使用完整高频数据不算前视**：允许使用交易日 `t` 内任意分钟、盘口或逐笔数据，允许使用当日最后一根记录、全日 `SUM/MAX/MIN/AVG`、日内排序、日内居中窗口、日内反向排序、收盘数据及完整日内序列，并聚合为归属于 `t` 的日频因子。即使早盘记录的中间行计算使用了当日下午数据，只要最终只输出交易日 `t` 的一个日频因子值，也不得判为前视。
3. **跨交易日使用未来数据才是前视重点**：若输出交易日 `t` 的推理特征或最终因子值使用交易日 `t+1`、`t+2` 或更晚交易日的数据，才属于明确前视线索。例如按证券跨日 `shift(-1)` 后进入特征/因子、跨日 centered rolling 包含后续交易日、从后续交易日向前回填、用未来日期键映射回 `t`、使用全期统计量计算历史日因子，或用未来股票池/非 PIT 财务信息回写历史日期。
4. **按交易日边界判断，不按原始行位置判断**：发现负向 shift、lead、反向排序、`center=True`、`bfill`、未来窗口等语法时，先确认其分组键和输出粒度。操作若严格限制在同一 `instrument + trading_date` 分组内并最终聚合到该日，不算跨交易日前视；若分组只按 `instrument`、窗口越过日期边界或日期映射把后续日信息带回先前日，则继续追踪其是否流入推理特征或因子输出。
5. **训练时序隔离单独判断**：未来收益标签可以存在，但对任一输出日期 `t`，用于生成该输出所加载权重的训练、验证、特征筛选、超参数选择和预处理拟合数据，其标签终点和可用数据不得越过 `t`。若提交使用固定离线权重，应核验权重训练区间是否早于评估输出区间；仅看到标签构造函数不能推断权重存在泄漏。
6. **证据必须形成完整链路**：判定“高风险”“可疑”或“违规”前，证据至少应说明：未来数据产生位置、所属交易日、经过的变量/函数、进入推理特征或最终因子的路径，以及它被错误对齐到哪个更早交易日。若只能看到未被推理调用的训练标签函数，结论应写“正常训练标签构造，不构成前视”；若调用关系或权重训练边界无法确认，写“无法核验”，不得写“发现明确前视线索”。

时序结论的推荐表述：

- 无跨日未来信息：`未发现跨交易日前视；训练标签中的未来收益构造和同交易日内高频聚合按本任务口径不计为前视。`
- 存在训练标签但隔离不明：`检测到未来收益标签构造；标签本身不构成前视，但权重训练区间与输出日期的隔离关系无法核验。`
- 存在明确跨日前视：`发现输出交易日 t 的特征/因子使用了交易日 t+N 的数据`，并列出完整代码链路。

### 2.4 单 submission JSON

生成前读取 `submission_analysis_template.json`，严格使用模板对象结构：不得删除、重命名或新增对象字段；数组可增删示例元素；空值使用 `null`、`[]` 或 `{}`；输出必须是合法 UTF-8 JSON。

固定枚举：

- 分析状态：`完成`、`部分完成`、`无法分析`
- 置信度：`高`、`中`、`低`
- 方案类型：`传统统计因子`、`规则或公式因子`、`机器学习因子`、`深度学习因子`、`AI 自动生成或优化因子`、`混合方法`、`无法确认`
- 质量等级：`优秀`、`良好`、`一般`、`较差`、`无法评价`
- 合规状态：`符合`、`违规`、`可疑`、`无法核验`、`不适用`
- 风险等级：`高`、`中`、`低`、`无法核验`
- 人工复核优先级：`高`、`中`、`低`
- `rank_source`：有私榜结果时写 `team_private_leaderboard.json private_score.final_score`；没有私榜结果时写 `unavailable`
- `official_rank`：有私榜结果时写该 submission 在 `team_private_leaderboard.json` 中的 `private_rank`；没有私榜结果时写 `null`

“违规”必须同时具备明确规则条款和 submission 文件证据；只有线索时写“可疑”，信息不足时写“无法核验”。静态代码无法证明交易日完整性、每日覆盖度或三小时内运行完成时，应评估风险并标为“无法核验”，不得假装运行验证。

## 三、阶段 2：生成统一汇总

所有 submission JSON 完成后，读取它们并且只生成：

- `.../artifacts/coder_analysis/summary.json`
- `.../artifacts/coder_analysis/summary.md`

汇总以团队为单位分组。有 `team_id` 时按 `team_id` 合并；没有 `team_id` 时按 `user_id` 各自成组。团队内 submissions 按 `submission_id` 升序排列。

排名以 `team_private_leaderboard.json` 的私榜结果为准，不得按代码质量重新排名：

- submission 分数取对应 `private_score.final_score`；
- submission 排名取对应 `private_rank`；
- 团队分数取对应 `best_private_score`；
- 团队排名取对应 `private_rank`；
- 未出现在私榜结果文件中的 submission 或团队/个人组保留分析，但分数、排名写 `null`；
- `ranking_basis.entity` 写 `team`；
- `ranking_basis.source` 写 `team_private_leaderboard.json private_rank/private_score`；
- 描述中明确排名和分数直接取自指定私榜结果文件。

### 3.1 补充参赛者信息

读取 `prepared/metadata.json`、`team_private_leaderboard.json` 和 `alphathon__user.csv`。metadata 只用于 submission、用户、团队与成员关联；私榜分数和排名只取私榜结果文件。CSV 的 `data` 列为 JSON 字符串，可补充姓名、电话、学校、专业和学历。

匹配规则：先按当前 `competition_id` 过滤 CSV，再按 `user_id` 精确匹配；只接受唯一匹配。零条或多条时不得猜测，字段写 `null` 并列入无法核验事项。团队成员优先取 metadata，资料再由 CSV 补充。`email` 字段因模板存在而保留，但统一写 `null`，Markdown 不展示邮箱。个人敏感信息只写入指定私有输出，不在终端回复中展示。

### 3.2 summary.json

生成前读取 `summary_template.json` 并严格遵循其结构。每个 `rankings` 元素代表一个团队或个人组；`members` 放全部成员；`submissions` 放组内全部提交。每份 submission 的方案、质量、合规、风险和创新结论必须直接来自对应单 submission JSON，不在汇总阶段重新分析或改变结论。

### 3.3 summary.md

Markdown 报告应可独立阅读，包含：

1. 分析范围、数量、完成情况、静态审查边界和规则依据；
2. 团队总览表：团队/个人、成员资料、submission IDs、团队私榜分数及私榜排名、赛道、方案类型、质量、合规和主要风险；
3. 按团队展示各 submission 的核心逻辑、数据、因子构造、时序、模型/AI、接口、质量、合规、证据、疑点、无法核验事项与人工复核建议；
4. 横向比较赛道归属、数据使用、因子类型、时序泄漏、代码质量、复现性、创新性和跨提交相似性线索；
5. 按严重度整理全局风险与人工复核清单。

不要原样粘贴全部 JSON，也不能只写一句结论。终端完成回复只报告生成文件路径和简短统计，不展示成员电话等敏感信息。
