# 私榜手工评测

本目录是独立于 `public/` 的私榜评测实现，只复用
`system/alphathonapiserver` 提供的通用评测框架。

当前是最小版本：保留单次评估、A/B/最终分、产物落盘、断点续跑和人工发布；
未来函数检测与样本外验证暂不包含。

代码按职责拆分：`prepare_submissions.py` 固化私榜输入，`private.py` 是评测入口，
`private_judge.py` 编排批次流程，
`templates.py` 生成隔离 runner，`scoring.py` 负责纯评分计算，`regression.py`
负责因子池回归，`fileio.py` 管理批次文件。评估结束日由入口在运行时通过
`datetime.now()` 取当前日期。

## 准备私榜输入

```bash
python prepare_submissions.py
```

脚本会查询 `selected_for_private=True` 的提交，并生成：

```text
private/prepared/
├── metadata.json
└── submissions/
    └── <submission_id>/
        └── <API 返回的全部原始文件>
```

`metadata.json` 包含团队、团队成员姓名、学校、公榜分数、重新按纯公榜分数计算的
公榜排名、每队私榜 submission 数，以及全部入围 submission 的原始 API 快照和落盘路径。
没有团队的个人参赛者也会单独记录。脚本不判断文件类型或内容，通过 submission API
返回的文件清单逐个原样下载，包括无扩展名文件、notebook、Python 文件和 parquet。
只有 API 下载本身失败时才会记录到 `preparation_errors.json`。

输出目录固定为 `private/prepared/`，每次成功运行会替换其中唯一的 `submissions/`
目录和 `metadata.json`，不会再按时间创建多份 submission 目录。`--batch-id` 仅作为
审计字段写入 metadata，不参与目录命名。

## 运行评估

评测必须显式指定上一步生成的输入包：

```bash
python private.py --input /path/to/private/prepared
```

所有运行配置都在 `private.py` 顶部显式设置，不读取环境变量：

```python
BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESUME = False
RERUN_SUBMISSION_IDS = []
MAX_WORKERS = 5
```

默认最多并行评测 5 个 submission。需要调整时，直接修改 `MAX_WORKERS`：

```python
MAX_WORKERS = 3
```

如果运行中途被终止，将 `BATCH_ID` 改为原批次目录名，并开启续跑：

```python
BATCH_ID = "20260806_172301"
RESUME = True
```

然后直接运行 `python private.py`。

续跑会读取 `submissions/<submission_id>/result.json`：已有完整成功或失败结果的提交
直接跳过，只执行尚未生成完整结果的提交。首次运行确定的 `DATE_END` 会从 manifest
恢复，跨日续跑也不会改变评估口径。原始文件、stdout 和已有评测产物均不会删除。

如果要在原批次中只强制重跑几个 submission，保持原 `BATCH_ID`、开启续跑并填写 ID：

```python
BATCH_ID = "20260806_172301"
RESUME = True
RERUN_SUBMISSION_IDS = [
    "submission-id-1",
    "submission-id-2",
]
```

程序会校验这些 ID 必须属于原批次，然后删除对应的
`submissions/<submission_id>/` 旧运行目录，从固化输入包重新复制源文件并评测。
其他已有完整 `result.json` 的 submission 会继续跳过。评分榜、汇总文件和
`pending_publish.jsonl` 会用全批次的最新结果重新生成。已发布批次不允许原地重跑，
应创建新批次。

`private.py` 不再下载代码，也不再统计公榜信息；但每次启动评测时仍会通过 API 查询
当前 `selected_for_private=True` 的 submission，并与 `metadata.json` 中的固化 ID 集合
做严格比对。数量和 ID 必须完全一致才会继续评测；如果参赛者在准备后重新选择，程序
会列出线上新增及已取消选择的 submission，抛错并要求重新运行
`prepare_submissions.py`。

评测阶段按公榜相同规则从 submission 元数据中查找文件名以 `.ipynb` 结尾的文件，
要求恰好一个，再从固化输入目录读取该 notebook、拼接其中的代码单元并注入 runner。
其他原始文件只作为运行附件保留；私榜不会重新通过 API 下载文件，也不依赖预先生成的
`submission_code.py`。

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
