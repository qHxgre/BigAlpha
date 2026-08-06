# 私榜手工评测

本目录是独立于 `public/` 的私榜评测实现，只复用
`system/alphathonapiserver` 提供的通用评测框架。

当前是最小版本：保留单次评估、A/B/最终分、产物落盘和人工发布；未来函数检测、
样本外验证、并行执行与断点恢复暂不包含。

代码按职责拆分：`prepare_submissions.py` 固化私榜输入，`private.py` 是评测入口，
`private_judge.py` 编排批次流程，
`templates.py` 生成隔离 runner，`scoring.py` 负责纯评分计算，`regression.py`
负责因子池回归，`fileio.py` 管理批次文件。评估结束日由入口在运行时通过
`datetime.now()` 取当前日期。

## 准备私榜输入

```bash
python prepare_submissions.py --batch-id final_20260806
```

脚本会查询 `selected_for_private=True` 的提交，并生成：

```text
private/prepared/final_20260806/
├── metadata.json
└── submissions/
    └── <submission_id>/
        ├── <原始 notebook 和附件>
        └── submission_code.py
```

`metadata.json` 包含团队、团队成员姓名、学校、公榜分数、重新按纯公榜分数计算的
公榜排名、每队私榜 submission 数，以及全部入围 submission 的原始 API 快照和落盘路径。
没有团队的个人参赛者也会单独记录。`.parquet` 文件不会转移。

## 运行评估

评测必须显式指定上一步生成的输入包：

```bash
python private.py --input /path/to/private/prepared/final_20260806
```

也可以同时指定评测批次 ID：

```bash
PRIVATE_BATCH_ID=final_private python private.py --input /path/to/private/prepared/final_20260806
```

`private.py` 不再下载代码，也不再统计公榜信息；但每次启动评测时仍会通过 API 查询
当前 `selected_for_private=True` 的 submission，并与 `metadata.json` 中的固化 ID 集合
做严格比对。数量和 ID 必须完全一致才会继续评测；如果参赛者在准备后重新选择，程序
会列出线上新增及已取消选择的 submission，抛错并要求重新运行
`prepare_submissions.py`。验证通过后，实际代码只从对应的 `submission_code.py` 读取。

程序会在
`/home/aiuser/work/workspace/BigAlpha/system/files/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/<batch_id>/`
下保存批次产物：

- `manifest.json`：批次状态、数据区间、数据表和提交数量；
- `submissions.json`：本次入围提交快照；
- `submissions/`：从输入包复制的源文件、运行日志及逐提交产物；
- `artifacts/`：单因子、回归、最终榜单和汇总文件；
- `pending_publish.jsonl`：待人工确认的后台分数更新；
- `logs/judge_private.log`：本批次评测日志。

评估过程不会更新后台 submission 分数。批次成功结束后，`manifest.json` 状态为
`review_pending`。

## 审查与发布

先预览：

```bash
python publish.py --run runs/<batch_id> --dry-run
```

人工核验全部文件后发布：

```bash
python publish.py --run runs/<batch_id>
```

交互输入 `PUBLISH` 后才会更新后台。发布结果写入批次目录，成功后 manifest 状态改为
`published`，再次发布同一批次会被拒绝。
