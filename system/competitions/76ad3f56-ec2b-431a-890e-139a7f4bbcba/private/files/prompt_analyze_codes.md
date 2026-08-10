# 第一步：读取并整理比赛规则

比赛 ID：`76ad3f56-ec2b-431a-890e-139a7f4bbcba`

读取比赛文档：

`/Users/xiehao/Desktop/workspace/BigAlpha/docs/76ad3f56-ec2b-431a-890e-139a7f4bbcba/desc/因子挖掘_介绍_20260804.md`

总结可用于审核 submission 的比赛规则，包括但不限于：

1. 允许使用的数据、字段、特征工程方式和时间范围。
2. submission 的文件、入口、运行环境和输出要求。
3. 因子数量、模型参数、回看期等明确限制。
4. 明确禁止的行为、潜在作弊方式和评分口径。
5. 文档没有明确规定的内容不得自行补充，需标记为“规则未说明”。

将规则摘要写入：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/code_analysis/rules.md`

# 第二步：确定待分析的 submission

参赛 submission 已由私榜准备程序保存在：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prepared/`

其中：

- `metadata.json` 保存团队、成员、学校、公榜分数和 submission 元数据。
- `submissions/<submission_id>/` 保存对应 submission 的全部原始文件。

默认分析 `metadata.json` 中的全部 submission。如果本次调用时另外指定了
`team_id`、`member_id` 或 `submission_id`，则只分析指定范围。

先读取 `metadata.json`，建立 team、member、submission 之间的对应关系，并使用元数据中的
分数计算团队和成员的最高分、平均分、最低分。不得从目录名猜测这些信息。

# 第三步：逐个静态分析 submission

只能静态阅读 submission 文件，严禁执行参赛代码，严禁导入参赛模块，严禁加载或反序列化
模型、pickle、joblib、torch、parquet 等文件。二进制文件只记录文件名、类型和大小；如果无法
从可信文本元数据确认其内容，应明确写“无法静态核验”。

对每个 submission 完成以下审查：

1. 列出主要文件，识别 notebook、Python 文件、配置、模型和数据文件。
2. 按实际入口分析完整推理调用链。该比赛的提交通常以唯一的 `.ipynb` 文件为入口，私榜程序
   会提取 notebook 的代码单元执行；如果 submission 的实际结构不同，应如实说明。
3. 总结输入数据、特征处理、因子或模型构建逻辑、预测过程和输出结果。
4. 分析训练代码与训练过程，包括训练数据集、字段、时间区间、标签、数据切分、随机种子、
   超参数、依赖和模型文件生成方式，判断现有材料是否足以复现。
5. 检查是否符合第一步整理的比赛规则。参数量、特征量、因子数量等只有在代码或可信元数据
   中可以严格确认时才能给出具体数值，否则写“无法确认”，不得估算后当作事实。
6. 检查训练和推理过程中的作弊或泄漏风险，包括但不限于：未来数据、标签泄漏、违规数据源、
   联网取数、硬编码答案、读取评测区间结果、绕过评测、动态下载或执行代码。
7. 分析创新性和原创性：先基于代码说明与常见方法的异同；必要时可搜索论文、研报或公开代码，
   但必须给出可核验来源。不能因为方法常见就断言抄袭，也不能因为分数高就推断作弊。

所有判断必须使用以下级别之一：

- `符合`：有充分证据证明符合规则。
- `违规`：有明确规则和代码证据证明违反要求。
- `可疑`：存在具体风险线索，但证据不足以认定违规。
- `无法核验`：代码、模型、训练材料或规则缺失，无法作出判断。
- `不适用`：该检查项与当前 submission 无关。

每个重要结论都应附带证据，优先写明相对 submission 目录的文件路径、函数或类名、代码行号，
并解释证据如何支持结论。正则关键词命中只能作为线索，必须阅读上下文后判断。

将每个 submission 的详细分析写入：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/code_analysis/submissions/<submission_id>.md`

每份报告至少包含：

1. submission 基本信息。
2. 文件与入口。
3. 推理流程和模型/因子逻辑。
4. 训练过程与可复现性。
5. 规则合规检查。
6. 作弊与数据泄漏风险。
7. 创新性与公开方法对比。
8. 已确认问题、疑点、无法核验事项。
9. 建议人工复核的文件和问题。

# 第四步：生成成员分析报告

按 `member_id` 汇总该成员的全部 submission，输出到：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/code_analysis/teams/<team_id>/members/<member_id>.md`

成员报告应包含：

1. 姓名、学校、所属团队、submission 数量。
2. 最高分、平均分、最低分，以及最高分对应的 submission ID。
3. 每个 submission 的核心方法、得分和审查结论。
4. 不同 submission 之间的演进、重复点和差异。
5. 已确认问题、疑点和无法核验事项。
6. 具体、可操作的下一步建议。

# 第五步：生成团队分析报告

按 `team_id` 汇总团队报告，输出到：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/competitions/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/code_analysis/teams/<team_id>/<team_id>.md`

团队报告应包含：

1. 团队名称、成员姓名和学校、submission 数量。
2. 团队最高分、平均分、最低分，以及最高分对应的 member 和 submission ID。
3. 重点详细总结团队最高分 submission 的模型/因子逻辑、合规性、可复现性、作弊风险和创新性。
4. 概括团队其他 submission 的共性方法、主要差异和演进过程。
5. 分开列出已确认问题、疑点和无法核验事项，不得混为确定事实。
6. 给出下一步指导建议，并列出优先需要人工复核的文件和问题。

# 第六步：完成前检查

完成后检查：

1. `metadata.json` 中本次范围内的每个 submission 都有对应的详细报告。
2. 每个成员和团队都有汇总报告，报告中的 ID、成员关系和学校与 metadata 一致。
3. 最高分、平均分、最低分均由 metadata 中可用的数值分数计算；无分数时明确说明，不得填 0。
4. 团队最高分 submission 的选择与分数一致。
5. “违规”结论必须同时有明确比赛规则和文件/行号证据。
6. 材料缺失只能判为“可疑”或“无法核验”，不能直接判定违规。
7. 报告不得声称运行过代码或成功加载过模型。
8. 所有引用的论文、研报和公开项目均可核验，不得虚构来源。

最后输出一段简短执行摘要，说明分析范围、生成的文件数量、主要风险和仍需人工复核的事项。
