# 任务：逐 submission 代码分析与统一汇总

本任务只分为两个阶段：

1. 逐一分析 `prepared/submissions/` 下的每个 submission，并为每个 submission 生成一份 JSON。
2. 读取全部 submission JSON，生成一份总 JSON 和一份 Markdown 分析报告。

不要读取或使用 `prepared/metadata.json`，不要生成单 submission Markdown、团队报告或成员报告。

## 一、固定路径

submission 根目录：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/prepared/submissions`

规则摘要：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/runs/20260811_104853/artifacts/coder_analysis/rules_summary.md`

参赛者信息：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/scripts/private/alphathon__user.csv`

输出目录：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/runs/20260811_104853/artifacts/coder_analysis`

JSON 模板：

- 单 submission：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/523f9302-5b4b-42bd-bce1-f232e7c74316/private/prompt/submission_analysis_template.json`
- 总汇总：`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/523f9302-5b4b-42bd-bce1-f232e7c74316/private/prompt/summary_template.json`

开始前必须读取 `rules_summary.md`。若文件不存在、为空或无法读取，立即停止并说明原因，不得自行从其他材料补充比赛规则。

## 二、阶段 1：逐一分析 submission

遍历 `prepared/submissions/` 的所有一级子目录。一级目录名就是 `submission_id`；每个目录独立分析，不依赖 `metadata.json`。

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
- 因不读取 `metadata.json`，无法从 submission 材料确认的团队、成员、比赛分数和正式排名字段写 `null`，不得猜测。

固定枚举：

- 分析状态：`完成`、`部分完成`、`无法分析`
- 置信度：`高`、`中`、`低`
- 方案类型：`单模型`、`集成模型`、`规则方法`、`混合方法`、`无法确认`
- 参数统计类型：`精确值`、`可确认下界`、`可确认区间`、`无法静态确认`
- 质量等级：`优秀`、`良好`、`一般`、`较差`、`无法评价`
- 合规状态：`符合`、`违规`、`可疑`、`无法核验`、`不适用`
- 风险等级：`高`、`中`、`低`、`无法核验`
- 人工复核优先级：`高`、`中`、`低`
- `rank_source`：本任务不使用排名，统一为 `unavailable`

“违规”必须同时具备明确的规则条款和 submission 文件证据；只有风险线索时写“可疑”，信息不足时写“无法核验”。

## 三、阶段 2：生成统一汇总

所有 submission JSON 完成后，读取它们并生成且只生成：

- `.../artifacts/coder_analysis/summary.json`
- `.../artifacts/coder_analysis/summary.md`

汇总以 `submission_id` 升序排列，不生成或推导赛事排名，不按质量评分重新排名。`ranking_basis.entity` 写 `submission`，`ranking_basis.source` 写 `unavailable`，相关排名和赛事分数字段写 `null`。

### 3.1 补充参赛者信息

汇总阶段可以读取 `alphathon__user.csv`。该 CSV 的 `data` 列是 JSON 字符串，可提供姓名、邮箱、电话、学校、专业、学历等信息。

匹配规则：

1. 仅在 submission 文件或单 submission JSON 中存在可核验的 `user_id`、姓名、邮箱、电话等标识时进行匹配。
2. 优先使用唯一标识精确匹配：`user_id`，其次是邮箱、电话，再其次是姓名。
3. 只接受唯一匹配；出现零条或多条候选时不得猜测，相关字段写 `null`，并在汇总的无法核验事项中说明。
4. CSV 只用于补充参赛者资料，不用于判断代码质量、合规性或赛事排名。
5. 在 `summary.json` 和 `summary.md` 中加入能够匹配到的姓名、学校、专业、学历、邮箱和电话。注意这些是个人敏感信息，仅写入指定的私有输出文件，不在终端回复中展示。

### 3.2 summary.json

生成前读取 `summary_template.json`，并严格遵循其结构。`rankings` 中每个元素代表一个 submission，而不是正式名次：

- `rank`、`team_score`、`submission_rank` 写 `null`。
- `submissions` 数组只放当前 submission。
- `members` 填入 CSV 唯一匹配得到的信息；无法匹配时使用空数组或 `null` 字段。
- `submission_markdown` 写 `null`。
- `submission_json` 写对应 JSON 的相对或绝对路径。

汇总数据必须直接来自单 submission JSON，不要重新分析代码，也不要在汇总阶段改变单份分析结论。

### 3.3 summary.md

Markdown 报告应可独立阅读，包含：

1. 分析范围、submission 数量、完成情况和规则依据。
2. 总览表：submission ID、参赛者、学校、专业、学历、电话、邮箱、方案类型、参数量、质量评分、总体合规状态和主要风险。
3. 每个 submission 的简明分析：核心逻辑、数据与特征、模型、训练与推理、参数量、代码质量、合规结论、主要证据、问题、疑点、无法核验事项和人工复核建议。
4. 全局横向比较：方案类型、数据使用、模型规模、代码质量、可复现性、创新性和风险。
5. 按严重度整理的全局风险与人工复核清单。

不要把所有单 submission JSON 原样粘贴进 Markdown；应提炼关键信息，但不能只写一句结论。