# 任务：逐 submission 代码分析与统一汇总

本任务只分为两个阶段：

1. 逐一分析 `prepared/submissions/` 下的每个 submission，并为每个 submission 生成一份 JSON。
2. 读取全部 submission JSON，生成一份总 JSON 和一份 Markdown 分析报告。

允许读取 `prepared/metadata.json`，但只可用于获取 submission 与用户、团队、团队成员、公榜分数之间的关联。不得使用 metadata 中的分数、排名或参赛者资料判断代码质量、创新性、原创性或规则合规性。

不要生成单 submission Markdown、独立团队报告或独立成员报告。团队分组只体现在统一的 `summary.json` 和 `summary.md` 中。

## 一、固定路径

submission 根目录：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/prepared/submissions`

规则摘要：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/runs/20260811_104853/artifacts/coder_analysis/rules_summary.md`

参赛者信息：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/scripts/private/alphathon__user.csv`

submission、用户与团队关联信息：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/prepared/metadata.json`

输出目录：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/runs/20260811_104853/artifacts/coder_analysis`

JSON 模板：

- 单 submission：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/523f9302-5b4b-42bd-bce1-f232e7c74316/private/prompt/submission_analysis_template.json`
- 总汇总：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/523f9302-5b4b-42bd-bce1-f232e7c74316/private/prompt/summary_template.json`

开始前必须读取 `rules_summary.md`。若文件不存在、为空或无法读取，立即停止并说明原因，不得自行从其他材料补充比赛规则。

## 二、阶段 1：逐一分析 submission

遍历 `prepared/submissions/` 的所有一级子目录。一级目录名就是 `submission_id`；每个目录的代码分析必须独立进行，不得根据 metadata 中的团队、成员、公榜分数或排名改变代码分析、质量评价或合规结论。metadata 仅用于补充身份关联和公榜分数。

每完成一个 submission，立即输出：

`.../artifacts/coder_analysis/submissions/<submission_id>.json`

不生成 `<submission_id>.md`。

### 2.1 静态审查边界

只能静态阅读文件：

- 可以阅读源码、配置、依赖文件、Markdown、文本、JSON，以及 notebook 的 JSON 内容和代码单元。
- 禁止执行参赛代码、导入参赛模块、联网运行代码，或加载/反序列化模型与数据文件。
- 不得加载 pickle、joblib、torch、parquet、feather 等文件。二进制文件只记录路径、类型和大小。
- 不得声称已经运行模型、完成推理、加载权重或验证运行结果。

### 2.2 每个 submission 要分析的内容

分析应围绕以下核心问题展开，不必机械撰写冗长章节，但结论必须有代码证据：

1. **实现逻辑**：识别入口，追踪主要调用链，说明数据如何经过预处理、模型、推理和后处理，最终生成 `date`、`instrument`、`score`。
2. **数据与特征**：使用哪些数据表、字段、频率、窗口、过滤、对齐、填充、标准化、特征工程、样本和标签。
3. **模型与训练**：模型结构、关键超参数、张量流转、训练切分、损失、优化器、epoch、随机种子、checkpoint 和权重来源。
4. **参数量**：区分总参数量、可训练参数量和冻结参数量。能静态计算时列出公式；不能确认时写明缺失信息，不得按模型文件大小估算。
5. **代码质量**：评价正确性、完整性、可读性、健壮性、效率、可维护性、可复现性和训练—推理一致性。
6. **规则合规**：严格按 `rules_summary.md` 逐项检查数据、字段、特征工程、回看窗口、参数量、外部资源、未来数据、泄漏、作弊风险、接口和提交材料要求。
7. **创新与风险**：说明方法特点、优势、局限、可能失效场景，以及需要人工复核的具体问题。

重要判断必须给出证据，尽量包含 submission 内相对路径、函数/类/配置名和行号。明确区分：

- 代码已证实；
- 基于代码推测；
- 材料不足，无法核验。

缺失材料时仍应记录已确认的链路、断点及影响，不能只写“无法分析”。高分不能作为代码优秀、创新或违规的证据；代码相似不能直接作为抄袭结论。

### 2.3 单 submission JSON

生成前读取 `submission_analysis_template.json`，输出必须严格使用该模板的对象结构：

- 不删除、重命名或新增对象字段。
- 数组可按实际情况复制或删除示例元素；空数组写 `[]`。
- 无数据使用 `null`、`[]` 或 `{}`，不要用说明性文字冒充空值。
- JSON 必须是合法 UTF-8 JSON，不得含注释、Markdown 围栏、`NaN` 或 `Infinity`。
- `team`、`member` 和公榜分数可根据 metadata 中的明确关联填写。正式排名、私榜分数和私榜排名没有明确来源时仍写 `null`，不得猜测。

固定枚举：

- 分析状态：`完成`、`部分完成`、`无法分析`
- 置信度：`高`、`中`、`低`
- 方案类型：`单模型`、`集成模型`、`规则方法`、`混合方法`、`无法确认`
- 参数统计类型：`精确值`、`可确认下界`、`可确认区间`、`无法静态确认`
- 质量等级：`优秀`、`良好`、`一般`、`较差`、`无法评价`
- 合规状态：`符合`、`违规`、`可疑`、`无法核验`、`不适用`
- 风险等级：`高`、`中`、`低`、`无法核验`
- 人工复核优先级：`高`、`中`、`低`
- `rank_source`：单 submission JSON 中写 `metadata.public_score`，表示分数来自 metadata；`official_rank` 仍写 `null`。报告派生的公榜排名只写入汇总，不得冒充官方排名。

“违规”必须同时具备明确的规则条款和 submission 文件证据；只有风险线索时写“可疑”，信息不足时写“无法核验”。

## 三、阶段 2：生成统一汇总

所有 submission JSON 完成后，读取它们并生成且只生成：

- `.../artifacts/coder_analysis/summary.json`
- `.../artifacts/coder_analysis/summary.md`

汇总以团队为单位分组。存在 `team_id` 的 submissions 按 `team_id` 合并；没有 `team_id` 的个人参赛者按 `user_id` 各自成组。团队内 submissions 按 `submission_id` 升序排列。

允许根据 metadata 中的 `public_score` 派生公榜排名，但不得将其称为官方排名或私榜排名，也不得按代码质量评分重新排名：

- 单 submission 分数取对应的 `public_score`。
- submission 公榜派生排名在全部 submissions 中按 `public_score` 从高到低计算。
- 团队分数取该团队全部 submissions 的最高 `public_score`。
- 团队公榜派生排名按团队分数从高到低计算。
- 排名采用竞赛排名法：同分并列，后续名次留空档。
- `ranking_basis.entity` 写 `team`。
- `ranking_basis.source` 写 `prepared/metadata.json public_score`。
- `ranking_basis.description` 必须明确说明分组、团队分数和派生排名算法，并注明其不是官方私榜排名。

### 3.1 补充参赛者信息

汇总阶段读取 `prepared/metadata.json` 和 `alphathon__user.csv`。metadata 用于确定 `submission_id → user_id/team_id`、团队名称、团队成员和 `public_score`；CSV 的 `data` 列是 JSON 字符串，可提供姓名、电话、学校、专业、学历等信息。

匹配规则：

1. 使用 metadata 中的明确关系取得每个 submission 的 `user_id` 和 `team_id`，不得根据目录名、代码内容或相似姓名猜测身份。
2. 读取 CSV 时必须先按当前 `competition_id` 过滤，再使用 `user_id` 精确匹配。
3. 只接受过滤后的唯一匹配；出现零条或多条候选时不得猜测，相关字段写 `null`，并在汇总的无法核验事项中说明。
4. 团队成员列表优先来自 metadata 的团队成员关系；成员的学校、专业、学历和电话通过过滤后的 CSV 以 `user_id` 补充。
5. CSV 和 metadata 中的身份及分数信息不得用于判断代码质量、创新性、原创性或规则合规性。
6. 在 `summary.json` 和 `summary.md` 中加入姓名、学校、专业、学历和电话，不展示邮箱。由于 JSON 模板包含 `email` 字段，该字段不得删除，统一写 `null`。
7. 个人敏感信息仅写入指定的私有输出文件，不在终端回复中展示。

### 3.2 summary.json

生成前读取 `summary_template.json`，并严格遵循其结构。`rankings` 中每个元素代表一个团队或个人参赛组：

- 存在 `team_id` 时，以 `team_id` 为组键；没有 `team_id` 时，以 `user_id` 为个人组键。
- `team_id` 和 `team_name` 填写 metadata 中的团队信息；个人组的 `team_id` 写 `null`，`team_name` 可写个人姓名。
- `members` 放入团队的全部成员；个人组放入当前个人。成员 `email` 统一写 `null`。
- `submissions` 放入该团队或个人组的全部 submissions，并按 `submission_id` 升序排列。
- 每个 submission 的 `score` 填写 metadata 中的 `public_score`。
- 每个 submission 的 `submission_rank` 填写全部 submissions 范围内按公榜分数派生的排名。
- 团队条目的 `team_score` 取组内最高 submission 公榜分数。
- 团队条目的 `rank` 填写按 `team_score` 派生的团队公榜排名。
- `submission_markdown` 写 `null`。
- `submission_json` 写对应 JSON 的相对或绝对路径。
- `ranking_summary` 说明团队分数口径和派生排名性质，不得表述为官方私榜排名。

汇总中的代码逻辑、数据特征、模型、参数量、质量、合规、创新和风险结论必须直接来自单 submission JSON，不要重新分析代码，也不要在汇总阶段改变单份分析结论。团队关系、成员资料、公榜分数和派生排名按本节规定来自 metadata 和 CSV。

### 3.3 summary.md

Markdown 报告应可独立阅读，包含：

1. 分析范围、submission 数量、完成情况和规则依据。
2. 团队总览表：团队名或个人姓名、全部成员、学校、专业、学历、电话、组内 submission IDs、团队公榜分数、团队公榜派生排名、方案类型、质量评分、总体合规状态和主要风险；不展示邮箱。
3. 按团队分组展示各 submission 的简明分析：先说明团队成员、团队公榜分数和组内 submissions，再逐一说明每个 submission 的分数、submission 公榜派生排名、核心逻辑、数据与特征、模型、训练与推理、参数量、代码质量、合规结论、主要证据、问题、疑点、无法核验事项和人工复核建议。
4. 全局横向比较：方案类型、数据使用、模型规模、代码质量、可复现性、创新性和风险。
5. 按严重度整理的全局风险与人工复核清单。

不要把所有单 submission JSON 原样粘贴进 Markdown；应提炼关键信息，但不能只写一句结论。
