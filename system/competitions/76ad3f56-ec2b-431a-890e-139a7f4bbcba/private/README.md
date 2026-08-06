# 私榜手工评测

本目录是独立于 `public/` 的私榜评测实现，只复用
`system/alphathonapiserver` 提供的通用评测框架。

当前是最小版本：保留单次评估、A/B/最终分、产物落盘和人工发布；未来函数检测、
样本外验证、并行执行与断点恢复暂不包含。

代码按职责拆分：`private.py` 是兼容入口，`private_judge.py` 编排批次流程，
`templates.py` 生成隔离 runner，`scoring.py` 负责纯评分计算，`regression.py`
负责因子池回归，`fileio.py` 管理批次文件。评估结束日由入口在运行时通过
`datetime.now()` 取当前日期。

## 运行评估

```bash
python private.py
```

也可以显式指定批次 ID，便于记录本次运行用途：

```bash
PRIVATE_BATCH_ID=final_20260806 python private.py
```

程序会在
`/home/aiuser/work/workspace/BigAlpha/system/files/76ad3f56-ec2b-431a-890e-139a7f4bbcba/private/runs/<batch_id>/`
下保存批次产物：

- `manifest.json`：批次状态、数据区间、数据表和提交数量；
- `submissions.json`：本次入围提交快照；
- `submissions/`：提交源文件、运行日志及逐提交产物；
- `artifacts/`：单因子、回归、最终榜单和汇总文件；
- `pending_publish.jsonl`：待人工确认的后台分数更新；
- `logs/judge_private.log`：本批次评测日志。

入围提交的原始上传文件单独保存在同一 `private` 根目录下：

```text
private/selected_submissions/<submission_id>/
```

该目录跳过 `.parquet`，不包含 runner、日志、结果 JSON、缓存等评测中间产物；
文件已存在时不会重复下载，因此不同批次可以共享同一份原始提交归档。

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
