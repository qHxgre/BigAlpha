# alphathonapiserver

BigQuant Alphathon 量化比赛**评测系统**的后端 API。职责被刻意限制为：

- **查询**：比赛信息、选手提交记录、提交文件
- **回写**：judge runner 把跑分结果写回 submission

报名 / 团队 / 代码广场 / 排行榜 / 通知等业务都不在这个服务里。

服务基于 `bigshared2.bigapi.BigAPIApp` 包装的 FastAPI 实例运行，ORM 使用 Tortoise + Aerich 迁移，认证/鉴权统一走 `bigshared2.auth`。

## 目录结构

```
alphathonapiserver/
├── __init__.py        # 空，仅作为包标识
├── main.py            # 服务入口：创建 BigAPIApp、挂两个路由
├── settings.py        # 环境变量与 Tortoise 配置
├── constants.py       # BigAuth 权限定义
├── models.py          # Tortoise ORM 数据模型（Competition / Submission）
├── schemas.py         # Pydantic 输入/输出 Schema
├── helpers.py         # API 通用 helper（404、权限、分页）
├── competitions.py    # 比赛只读接口
├── submissions.py     # 提交查询、文件下载、分数回写
└── judgebase.py       # 通用比赛评测框架（独立运行的 judge 脚本）
```

## 系统由两部分组成

1. **Web API（`main.py`）**：常驻 FastAPI 服务，对外提供查询和分数回写端点。
2. **Judge runner（继承 `judgebase.JudgeBase`）**：每场比赛一个进程，长期运行；自动拉取新提交、子进程跑选手代码、把分数 POST 回 Web API。

两者通过 HTTP 通信，可以分别部署。

---

## 快速开始

### 1. 启动 Web 服务

```bash
python -m alphathonapiserver.main
# 等价于 uvicorn alphathonapiserver.main:app --host 0.0.0.0 --port 8000 --reload
```

`BigAPIApp` 已经内置 `bigshared2` 提供的统一异常、CORS、日志、Tortoise 生命周期。

### 2. 给 judge runner 准备一个 token

judge runner 调本服务回写分数时需要 `competition_manage` 权限。两种方式提供 token：

```bash
# 方式 A：写到文件（优先级高）
mkdir -p /home/aiuser/work/data/alphathon
echo "<your-jwt>" > /home/aiuser/work/data/alphathon/cptjudge.jwt

# 方式 B：环境变量
export ALPHATHON_API_TOKEN="<your-jwt>"
```

### 3. 写一个 Judge 子类，长期运行

新建 `csi1000_judge.py`：

```python
from alphathonapiserver.judgebase import JudgeBase

class Csi1000Judge(JudgeBase):
    competition_id = "5c3f7783-4158-4196-97ab-171b27218c7c"
    score_kind = "public"           # public / private
    tick_interval = 60              # 主循环间隔（秒）
    max_workers = 5                 # 并发评测线程数
    completed_ids_file = "/home/aiuser/work/data/alphathon/csi1000_completed.json"

    runner_code = '''__USER_CODE__

def judge_runner_main():
    data = main("cpt_jyc_2025_stock_csi1000_bar1m_test", "2025-01-01", "2025-07-31 23:59:59")
    from bigmodule import M
    result = M.factorlens._latest(data=data, m_cached=False)
    with open("output.data", "w") as writer:
        writer.write(result._result.id)
'''

    def score(self, df):
        # 入参是各提交 raw_result 拼成的 DataFrame，至少含 `id` 列
        return (
            df["rank_ic"].rank(pct=True) * 0.4
            + df["rank_ir"].rank(pct=True) * 0.3
            + df["sharp_ratio"].rank(pct=True) * 0.2
            + df["turnover"].rank(pct=True, ascending=False) * 0.1
        )

if __name__ == "__main__":
    Csi1000Judge().run()   # 阻塞，直到进程被杀
```

```bash
# 比赛期间一直运行
python csi1000_judge.py
```

进程行为：每 `tick_interval` 秒拉一次 `/submissions?competition_id=...`，把没跑过的提交丢线程池，写完分数后调用 `score()` 重排，循环。

---

## HTTP API

### 入口与路由

| Prefix          | Tag    | 文件              |
| --------------- | ------ | ----------------- |
| `/competitions` | 比赛   | `competitions.py` |
| `/submissions`  | 提交   | `submissions.py`  |

### 端点清单

#### `competitions.py`

| 方法 | 路径                              | 鉴权        | 说明     |
| ---- | --------------------------------- | ----------- | -------- |
| GET  | `/competitions`                   | 登录 / 匿名 | 比赛列表 |
| GET  | `/competitions/{competition_id}`  | 登录 / 匿名 | 比赛详情 |

#### `submissions.py`

| 方法 | 路径                                            | 鉴权                      | 说明                                 |
| ---- | ----------------------------------------------- | ------------------------- | ------------------------------------ |
| GET  | `/submissions`                                  | 登录                      | 提交列表（非管理员仅看自己的）       |
| POST | `/submissions/{submission_id}`                  | `competition_manage`      | 仅回写 4 个分数字段                  |
| GET  | `/submissions/files/{submission_id}/{file_id}`  | 提交本人 / 创建者 / 管理员 | 下载提交文件                         |

`POST /submissions/{submission_id}` 入参是 `SubmissionScoreUpdate`，仅允许 `public_score / public_score_data / private_score / private_score_data` 四个字段，其它字段会被忽略。

### 调用示例

注意：`bigshared2` 框架在所有路由前加了 `/bigapis/alphathon/v1` 前缀，下面示例中用 `$BASE` 表示这段。

```bash
BASE="http://localhost:8000/bigapis/alphathon/v1"
TOKEN="<your-jwt>"

# 拉单个比赛
curl "$BASE/competitions/5c3f7783-4158-4196-97ab-171b27218c7c"

# 拉某场比赛的提交（带翻页）
curl "$BASE/submissions?competition_id=5c3f...&page=1&size=100&order_by=-created_at" \
  -H "Cookie: bigjwt=$TOKEN"

# 下载某次提交的文件
curl -o code.ipynb "$BASE/submissions/files/<sub_id>/<file_id>" \
  -H "Cookie: bigjwt=$TOKEN"

# 回写分数（需要 competition_manage 权限的 token）
curl -X POST "$BASE/submissions/<sub_id>" \
  -H "Cookie: bigjwt=$TOKEN" \
  -H "content-type: application/json" \
  -d '{"public_score": 0.823, "public_score_data": {"raw_result": {"rank_ic": 0.05}}}'
```

---

## 数据模型（`models.py`）

仅保留两张表：

| 表名（DB）               | 模型类        | 说明                                                                                |
| ------------------------ | ------------- | ----------------------------------------------------------------------------------- |
| `alphathon__competition` | `Competition` | 比赛主体。`summary` 列表元信息，`data` 详情扩展                                     |
| `alphathon__submission`  | `Submission`  | 作品提交。public/private 两套分数 + 详细数据；`selected_for_private` 控制私榜入选 |

报名 / 团队 / 代码广场 等表不在此服务管理范围内。

## 权限模型

`constants.Privileges` 只声明了一个权限：

```
competition_manage = /alphathon/competition/manage
  → roles: competition_admin, super_admin, operation_manager
```

关键策略：

- judge runner 必须用具备 `competition_manage` 角色的 token 才能回写分数。
- 普通登录用户调 `/submissions` 时只能看到自己的提交记录，管理员可以看全部。
- 文件下载允许：提交本人 / 比赛创建者 / 管理员。

## 配置

### `settings.py`

| 环境变量            | 默认值                  | 用途                       |
| ------------------- | ----------------------- | -------------------------- |
| `FILE_UPLOAD_PATH`  | `/var/app/data/uploads` | 提交文件落盘根路径（只读） |

数据库连接复用 `bigshared2.db.sql.settings.BASE_TORTOISE_ORM`。

### judgebase 端

| 环境变量                  | 默认值                                                                           | 用途                                         |
| ------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------- |
| `RUNNER_BASE_DIR`         | `/home/aiuser/work/data/alphathon`                                               | 选手代码工作目录根                           |
| `ALPHATHON_API_BASE_URL`  | `http://alphathonapiserver.bigquant.svc.cluster.local:8000/bigapis/alphathon/v1` | judge runner 调本服务的入口                  |
| `ALPHATHON_API_TIMEOUT`   | `15.0`                                                                           | HTTP 超时（秒）                              |
| `ALPHATHON_API_TOKEN`     | -                                                                                | 当 `RUNNER_BASE_DIR/cptjudge.jwt` 缺失时使用 |

---

## judgebase 详解

### `JudgeBase` 类属性

在子类中按需覆盖：

| 名字                 | 必填 | 默认       | 含义                                                                         |
| -------------------- | ---- | ---------- | ---------------------------------------------------------------------------- |
| `competition_id`     | 是   | -          | 比赛 UUID                                                                    |
| `runner_code`        | 是   | -          | judge runner 模板，须包含 `__USER_CODE__` 占位符且定义 `judge_runner_main()` |
| `score_kind`         | 否   | `"public"` | 写到 `public_score` 还是 `private_score`                                     |
| `tick_interval`      | 否   | `60`       | 主循环 sleep 间隔（秒）                                                      |
| `max_workers`        | 否   | `5`        | 并发评测线程数                                                               |
| `max_pages`          | 否   | `10000`    | 拉取提交分页上限                                                             |
| `completed_ids_file` | 否   | `None`     | 持久化已完成 submission_id 的文件路径，断点续跑用                            |

### 可覆盖钩子

- `query_constraints() -> dict`：拉取提交时的过滤条件，默认空 dict（拉全量）。比如只评测 `selected_for_private=True` 的提交：`return {"selected_for_private": True}`。
- `patch_user_code(submission, code) -> str`：对个别选手代码做字符串替换补丁，默认原样返回。
- `score(df) -> pd.Series | None`：按 `df` 算每行 score；返回 `None` 时不重排。返回 Series 时框架会在每个 tick 末尾把每行分数写回 `{score_kind}_score`。

### 内部组件

- **`AlphathonAPI`**：HTTP 客户端。认证 token 优先读 `RUNNER_BASE_DIR/cptjudge.jwt`，否则用环境变量 `ALPHATHON_API_TOKEN`。提供 4 个方法：拉比赛、拉提交列表（自动翻页）、拉提交文件（支持 ipynb→py）、回写分数。子类可通过 `self.api` 直接调用。
- **`LocalProcessUserRunner`**：把选手代码 + runner 模板组装到 `RUNNER_BASE_DIR/{submission_id}/judge_runner.py`，子进程执行，stdout 落盘到同目录 `stdout` 文件，3 小时超时。
- **`JudgeBase.run()`**：无限循环 = 拉新提交 → 线程池跑 → 回写 raw_result + 单分 → `recompute_ranks` 用 `score()` 重排 → 持久化已完成 ID → sleep。子进程失败兜底分数为 -2 并附错误提示 `"run error: check your code / get code templates in [code] tab"`。

### 判分协议

- 选手 runner 必须定义 `judge_runner_main()`，并把"raw 结果"对应的 DataSource ID 写到工作目录的 `output.data` 文件。
- judgebase 读到 ID 后用 `dai.DataSource(...).read()` 拿到第一行作为该次提交的 `raw_result`，回写到 `{score_kind}_score_data.raw_result`。
- 若子类实现了 `score()`，每个 tick 末尾会按所有提交的 `raw_result` 拼成 DataFrame 重算一次全场分数（rank-based 综合打分常用）。

### 长期运行的常见做法

- 用 systemd / supervisord / k8s Deployment 把 `python csi1000_judge.py` 拉起来，崩溃自动重启。
- `completed_ids_file` 指向持久化卷，重启后能跳过已评分的提交。
- 一场比赛一个进程，互不影响；可以横向再开一份 judge 实例并发，但要注意 `completed_ids_file` 不能共享（否则两个进程会抢同一份提交）。
