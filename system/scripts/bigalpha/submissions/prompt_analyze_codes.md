## 第一步 读取比赛规则

读取 /Users/xiehao/Desktop/workspace/BigAlpha/docs/523f9302-5b4b-42bd-bce1-f232e7c74316/desc/端到端模型_介绍_20260804.md 这个比赛文档，总结比赛规则，包括但不限于：

1. 该比赛对参赛者提交的submission的要求，比如模型参数等

把规则文档落盘到 /Users/xiehao/Desktop/workspace/BigAlpha/system/files/submissions/rules_523f9302-5b4b-42bd-bce1-f232e7c74316.md


## 第二步 分析各个参赛者的 submission

读取 /Users/xiehao/Desktop/workspace/BigAlpha/system/files/submissions/rules_523f9302-5b4b-42bd-bce1-f232e7c74316.md 这个文档，理解“端到端模型”比赛的规则和要求。

参赛者的submission都保存在 文件路径：/Users/xiehao/Desktop/workspace/BigAlpha/system/files/submissions/523f9302-5b4b-42bd-bce1-f232e7c74316/<team_id>/<member_id>/<submission_id>

本次待分析的目标是：team_id = '20_1黑Q黑Q_1a6ef4c5-1a04-4f25-8bc0-ece11f0d784a'

以这个团队下面每个 member 进行分组，对每个submission进行分析，规则和要求如下：

1. 该文件目录下是按照 ”团队>>成员>>submission“ 组织文件的，参考文件：.json。代码都无法运行，因此仅能通过查阅他们的代码和模型来进行分析。
2. 所有 submission 下面的入口是 judge_runner.py 的 judge_runner_main() 函数。
3. 分析模型构建代码，总结其模型构建的逻辑。
4. 分析总结模型，检查其是否符合比赛规则要求，包括但限于：模型的参数量、特征量等关键信息。
5. 检查其训练代码，后续我是否能复现，比如：部分参赛者并没有说明他用于训练的数据集和时间段。
6. 检查在模型训练和预测推理过程中是否有作弊行为。
7. 分析说明该模型是否有足够的创新点，可以搜索网上的论文研报相关资料，指出该模型的原创性如何。


最后把分析结果输出到文件路径：/Users/xiehao/Desktop/workspace/BigAlpha/system/files/submissions/analyze_outputs/523f9302-5b4b-42bd-bce1-f232e7c74316/<team_id> 中，输出多个json文件：

1. 以 team_id 作为json文件名：重点分析最高得分的submission的情况
    * 首先说明该团队的基本信息：最高分、平均得分、最低分、团队成员信息（含学校）。
    * 只总结最高分的submission的分析情况。
    * 总结问题并给出下一步的指导建议。
2. 以 member_id 作为json文件名：详细分析每个成员的提交情况
    * 总结该成员的基本信息：最高分、平均得分、最低分、学校。
    * 汇总其提交的每个submission的分析结果。
    * 总结问题并给出下一步的指导建议。

输出的json文件模版参考：

* /Users/xiehao/Desktop/workspace/BigAlpha/system/files/submissions/analyze_outputs/member_analysis_template.json
* /Users/xiehao/Desktop/workspace/BigAlpha/system/files/submissions/analyze_outputs/team_analysis_template.json