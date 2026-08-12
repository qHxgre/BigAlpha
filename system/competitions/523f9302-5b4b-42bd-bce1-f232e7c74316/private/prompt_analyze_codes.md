# 使用方式

本提示词分两次独立输入给大模型，不得在一次调用中连续执行两个阶段：

1. **第一次输入**：只执行“第一阶段：生成规则摘要”。规则摘要写入指定文件后立即停止，不读取或分析任何 submission。
2. **第二次输入**：只执行“第二阶段：分析 submission 并生成报告”。必须先读取第一次生成的 `rules_summary.md`，以该摘要作为合规审查依据，不再重新总结比赛文档。

# 第一阶段（第一次输入）：生成规则摘要

Competition_ID: `76ad3f56-ec2b-431a-890e-139a7f4bbcba`
Batch_ID: `20260811_102014`

本次分析生成的全部文件统一保存到：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260811_102014/artifacts/coder_analysis`

读取比赛文档：

`/Users/xiehao/Desktop/workspace/BigAlpha/docs/76ad3f56-ec2b-431a-890e-139a7f4bbcba/desc/因子挖掘_介绍_20260804.md`

总结可用于审核 submission 的比赛规则，包括但不限于：

1. 允许使用的数据、字段、特征工程方式和时间范围。
2. submission 的文件、入口、运行环境和输出要求。
3. 因子数量、模型参数、回看期等明确限制。
4. 明确禁止的行为、潜在作弊方式和评分口径。
5. 文档没有明确规定的内容不得自行补充，需标记为“规则未说明”。

将规则摘要写入：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260811_102014/artifacts/coder_analysis/rules_summary.md`

写入完成后，输出规则摘要文件路径和一段简短说明，然后立即停止。此阶段严禁读取 `metadata.json`、submission 目录或参赛代码，也不得生成 submission 或团队分析报告。

# 第二阶段（第二次输入）：分析 submission 并生成报告

开始分析前，必须读取第一阶段生成的规则摘要：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260811_102014/artifacts/coder_analysis/rules_summary.md`

后续所有合规判断均以该文件为依据。若文件不存在、为空或无法读取，应停止分析并明确报告原因，不得自行重新读取比赛文档或凭常识补充规则。

## 第一步：确定待分析的 submission

参赛 submission 已由私榜准备程序保存在：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/prepared/submissions`

其中：

- `metadata.json` 保存团队、成员、学校、公榜分数和 submission 元数据。
- `submissions/<submission_id>/` 保存对应 submission 的全部原始文件。

默认分析 `metadata.json` 中的全部 submission。如果本次调用时另外指定了`team_id`、`member_id` 或 `submission_id`，则只分析指定范围。

先读取 `metadata.json`，建立 team、member、submission 之间的对应关系，并使用元数据中的分数计算团队和成员的最高分、平均分、最低分。不得从目录名猜测这些信息。

## 第二步：逐个静态分析 submission

只能静态阅读 submission 文件，严禁执行参赛代码，严禁导入参赛模块，严禁加载或反序列化模型、pickle、joblib、torch、parquet 等文件。二进制文件只记录文件名、类型和大小；如果无法从可信文本元数据确认其内容，应明确写“无法静态核验”。

对每个 submission 完成以下审查：

1. 列出主要文件，识别 notebook、Python 文件、配置、模型和数据文件。
2. 按实际入口分析完整推理调用链。该比赛的提交通常以唯一的 `judge_runner.py` 文件中的`judge_runner_main()`函数为入口，私榜程序会提取 notebook 的代码复制到这个py文件中。
3. 总结输入数据、特征处理、因子或模型构建逻辑、预测过程和输出结果。
4. 分析训练代码与训练过程，包括训练数据集、字段、时间区间、标签、数据切分、随机种子、超参数、依赖和模型文件生成方式，判断现有材料是否足以复现。
5. 检查是否符合第一阶段生成的 `rules_summary.md`。参数量、特征量、因子数量等只有在代码或可信元数据中可以严格确认时才能给出具体数值，否则写“无法确认”，不得估算后当作事实。
6. 检查训练和推理过程中的作弊或泄漏风险，包括但不限于：未来数据、标签泄漏、违规数据源、联网取数、硬编码答案、读取评测区间结果、绕过评测、动态下载或执行代码。
7. 分析创新性和原创性：先基于代码说明与常见方法的异同；必要时可搜索论文、研报或公开代码，但必须给出可核验来源。不能因为方法常见就断言抄袭，也不能因为分数高就推断作弊。

所有判断必须使用以下级别之一：

- `符合`：有充分证据证明符合规则。
- `违规`：有明确规则和代码证据证明违反要求。
- `可疑`：存在具体风险线索，但证据不足以认定违规。
- `无法核验`：代码、模型、训练材料或规则缺失，无法作出判断。
- `不适用`：该检查项与当前 submission 无关。

每个重要结论都应附带证据，优先写明相对 submission 目录的文件路径、函数或类名、代码行号，并解释证据如何支持结论。正则关键词命中只能作为线索，必须阅读上下文后判断。

将每个 submission 的详细分析写入：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260811_102014/artifacts/coder_analysis/submissions/<submission_id>.md`

每份报告至少包含：

1. submission 基本信息。
2. 推理流程和模型/因子逻辑。
3. 训练过程与可复现性。
4. 规则合规检查。
5. 作弊与数据泄漏风险。
6. 创新性与公开方法对比。
7. 已确认问题、疑点、无法核验事项。
8. 建议人工复核的文件和问题。

## 第三步：按成员汇总并生成团队分析报告

私榜阶段每位参赛者最多只有 2 个 submission，因此不再生成单独的成员分析报告。按 `team_id` 汇总团队内全部成员及其 submission，并将成员维度的分析直接写入团队报告。

按 `team_id` 汇总团队报告，输出到：

`/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/20260811_102014/artifacts/coder_analysis/teams/<team_id>/<team_id>.md`

团队报告应包含：

1. 团队名称、成员姓名和学校、submission 数量。
2. 团队最高分、平均分、最低分，以及最高分对应的 member 和 submission ID。
3. 重点详细总结团队最高分 submission 的模型/因子逻辑、合规性、可复现性、作弊风险和创新性。
4. 按成员分节汇总：列出成员姓名、学校、submission 数量和分数；逐个概括该成员每个 submission 的核心方法与审查结论；如有 2 个 submission，说明两者的演进、重复点和差异。无有效分数时明确说明，不得填 0。
5. 概括团队全部 submission 的共性方法、主要差异和演进过程，并说明成员之间是否存在明显的代码、模型或思路复用；仅凭相似不得断言抄袭。
6. 分开列出已确认问题、疑点和无法核验事项，不得混为确定事实。
7. 给出具体、可操作的下一步建议，并列出优先需要人工复核的文件和问题。

## 第四步：完成前检查

完成后检查：

1. `metadata.json` 中本次范围内的每个 submission 都有对应的详细报告。
2. 每个团队都有一份汇总报告；每位成员均在所属团队报告中有独立小节，报告中的 ID、成员关系和学校与 metadata 一致。不生成单独的成员报告。
3. 最高分、平均分、最低分均由 metadata 中可用的数值分数计算；无分数时明确说明，不得填 0。
4. 团队最高分 submission 的选择与分数一致。
5. “违规”结论必须同时有明确比赛规则和文件/行号证据。
6. 材料缺失只能判为“可疑”或“无法核验”，不能直接判定违规。
7. 报告不得声称运行过代码或成功加载过模型。
8. 所有引用的论文、研报和公开项目均可核验，不得虚构来源。

最后输出一段简短执行摘要，说明分析范围、生成的文件数量、主要风险和仍需人工复核的事项。
