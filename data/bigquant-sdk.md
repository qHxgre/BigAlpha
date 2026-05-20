# BigQuant SDK 使用文档

BigQuant SDK 是为专业量化研究员打造的本地开发工具。它让您在保留本地 IDE 开发自由度的同时，无缝调用 BigQuant 云端的海量数据与分布式算力。


---

## 目录


1. [快速安装](#%E5%BF%AB%E9%80%9F%E5%AE%89%E8%A3%85)
2. [登录认证](#%E7%99%BB%E5%BD%95%E8%AE%A4%E8%AF%81)
3. [Hello BigQuant](#hello-bigquant)
4. [数据查询（DAI）](#%E6%95%B0%E6%8D%AE%E6%9F%A5%E8%AF%A2dai)
5. [本地回测（BigTrader）](#%E6%9C%AC%E5%9C%B0%E5%9B%9E%E6%B5%8Bbigtrader)
6. [模拟交易（PaperTrading）](#%E6%A8%A1%E6%8B%9F%E4%BA%A4%E6%98%93papertrading)
7. [云端计算（AIStudio）](#%E4%BA%91%E7%AB%AF%E8%AE%A1%E7%AE%97aistudio)（暂未开放）
8. [分布式计算（FAI）](#%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%A1%E7%AE%97fai)（暂未开放）
9. [常见问题](#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)


---

## 快速安装

BigQuant SDK 支持 Windows、Linux 和 macOS，要求 Python 3.11。

**推荐工具**：[Python 3.11](https://www.python.org/downloads/) + [Visual Studio Code](https://code.visualstudio.com/)

```bash
# 安装全功能版（包含数据查询、本地回测与分布式算力模块）
pip install 'bigquant[all]' -i https://pypi.bigquant.com/simple/
```

**按需安装**：

```bash
# 仅数据查询
pip install bigquant -i https://pypi.bigquant.com/simple/

# 数据查询 + 本地回测
pip install 'bigquant[bigtrader]' -i https://pypi.bigquant.com/simple/
```

安装完成后，验证安装：

```bash
bq --version
```


---

## 登录认证

### 方式一：API Key 认证（推荐）


1. 登录 BigQuant 平台，进入 [我的 > API Keys > 创建](https://bigquant.com/aiuser/access_key)，获取 `AK.SK` 格式的凭证。
2. 在终端执行（将 `<AK.SK>` 替换为您的凭证）：

```bash
bq auth --apikey <AK.SK>
```

或者交互式输入（输入时不回显，更安全）：

```bash
bq auth configure
```


3. 验证登录状态：

```bash
bq auth status
```

### 方式二：用户名密码登录

```bash
bq auth login -u <用户名> -p <密码>
```

### 方式三：在代码中初始化

```python
import bigquant

# 从配置文件初始化（推荐，配置文件由 bq auth 命令自动生成）
bigquant.init_from_config()  # 默认读取 ~/.bigquant/config.json

# 或者直接传入 AK/SK
bigquant.init(ak="YOUR_AK", sk="YOUR_SK")

# 或者使用 Token（仅支持 HTTP，不支持 Arrow Flight 高速传输）
bigquant.init_from_token("YOUR_TOKEN")

# 或者使用用户名密码
bigquant.init_from_password("username", "password")

# 查看当前登录用户
print(bigquant.whoami())
# 输出示例：{'id': 12345, 'username': 'your_name', 'email': 'you@example.com'}
```

> **配置文件位置**：`~/.bigquant/config.json`（由 `bq auth` 命令自动创建和维护）

> **自动初始化**：如果您已通过 `bq auth` 配置了凭证，代码中无需显式调用 `init()`，SDK 会在首次使用时自动从配置文件加载。


---

## Hello BigQuant

先 执行命令 `bq pkg list` 确保安装 `bigtradercpp`，创建 `start.py`，体验 SDK 的核心功能：

```python
from bigquant import dai
from bigquant import bigtrader

# 1. 提取数据（本地不存数据，通过 SQL 即取即用）
df = dai.query(
    "SELECT date, instrument, close FROM cn_stock_bar1d WHERE date >= '2024-01-01' LIMIT 10",
    filters={"date": ["2024-01-01", "2024-12-31"]},
).df()
print("✓ 数据获取成功：\n", df.head())

# 2. 极简回测（策略逻辑 100% 留在本地）
def initialize(context):
    context.set_commission(bigtrader.PerOrder(buy_cost=0.00005))
    context.asset = "000001.SZ"

def handle_data(context, data):
    if context.get_account_position(context.asset).amount == 0:
        context.order_target_percent(context.asset, 1.0)

performance = bigtrader.run(
    start_date="2024-01-01",
    end_date="2024-01-31",
    initialize=initialize,
    handle_data=handle_data,
)
print("✓ 回测完成，夏普比率:", performance.summary["sharp_ratio"])
```

运行：

```bash
❯ python3 start.py
✓ 数据获取成功：
         date instrument        close
0 2024-01-02  000001.SZ  1074.924107
1 2024-01-02  000002.SZ  1844.293378
2 2024-01-02  000004.SZ    65.590730
3 2024-01-02  000005.SZ    10.379715
4 2024-01-02  000006.SZ   177.636341
2026-04-20 11:49:39 [info     ] bigtrader init ..
2026-04-20 11:49:39 [info     ] bigtrader.run start: market=cn_stock, frequency=1d, mode=backtest, account_type=STOCK, start_date=2024-01-01, end_date=2024-01-31
2026-04-20 11:49:39 [info     ] bigtrader<backtest> init ..
2026-04-20 11:49:39 [info     ] prepare data ..
2026-04-20 11:50:23 [info     ] bar1d_df: (340778, 16)
2026-04-20 11:50:23 [info     ] bigtrader use dividend data: (8, 10)
2026-04-20 11:50:23 [info     ] bigtrader run ..
## BigTrader 回测报告

### 绩效摘要

| 指标    | 值 |
|-------|------|
| 累计收益率 | 2.92% |
| 年化收益率 | 39.12% |
| 基准收益率 | -5.05% |
| 阿尔法   | 1.14 |
| 贝塔    | 0.73 |
| 夏普比率  | 1.81 |
| 胜率    | 100.0% |
| 盈亏比   | 0.0 |
| 收益波动率 | 17.39% |
| 信息比率  | 0.44 |
| 最大回撤  | 2.56% |

### 收益走势

| 日期 | 策略收益率 | 基准收益率 | 相对收益率 |
|------|----------|----------|----------|
| 2024-01-02 | 0.0% | 0.0% | 0.0% |
| 2024-01-05 | 0.86% | -1.69% | 2.58% |
| 2024-01-12 | -0.0% | -3.02% | 3.1% |
| 2024-01-19 | -0.22% | -3.44% | 3.28% |
| 2024-01-26 | 4.66% | -1.55% | 6.24% |
| 2024-01-31 | 2.92% | -5.05% | 8.31% |

### 交易统计

- 总平仓次数: 0, 盈利: 0, 亏损: 0
- 总买入金额: 997,115.00, 总卖出金额: 0.00
- 总手续费: 49.86
- 最终组合价值: 1,029,245.14


[bigtrader_report_output=/code/python/test/logs/0057c7a4.bigtrader.report.json.gz]
2026-04-20 11:50:24 [info     ] bigtrader run done.
✓ 回测完成，夏普比率: 1.81
```


---

## 数据查询（DAI）

DAI（Data AI）是 BigQuant SDK 的核心数据模块，通过 SQL 查询云端海量金融数据，支持 Arrow Flight 高性能传输。

### 基础查询

```python
from bigquant import dai

# 查询股票日线数据
df = dai.query("""
    SELECT date, instrument, open, high, low, close, volume
    FROM cn_stock_bar1d
    WHERE date >= '2024-01-01' AND date <= '2024-03-31'
""", filters={"date": ["2024-01-01", "2024-03-31"]}).df()

print(df.head())
```

> **重要**：查询有日期范围时，务必同时传 `filters` 参数，避免全表扫描。

### 结果格式转换

```python
result = dai.query("SELECT * FROM cn_stock_bar1d LIMIT 100", full_db_scan=True)

df = result.df()          # pandas DataFrame
table = result.arrow()    # PyArrow Table
pl_df = result.pl()       # Polars DataFrame
rows = result.fetchall()  # Python 列表（每行为 dict）

# 大数据流式读取（节省内存）
reader = result.fetch_arrow_reader(batch_size=10000)
for batch in reader:
    print(batch.num_rows)
```

### 过滤条件（filters）

`filters` 用于分区过滤，大幅提升查询性能：

```python
# 日期范围过滤
df = dai.query(
    "SELECT * FROM cn_stock_bar1d WHERE close > 10",
    filters={"date": ["2024-01-01", "2024-12-31"]},
).df()

# 多字段过滤
df = dai.query(
    "SELECT * FROM cn_stock_bar1d",
    filters={
        "date": ["2024-01-01", "2024-12-31"],
        "instrument": ["000001.SZ", "600519.SH"],
    },
).df()
```

### DataSource 读写

DataSource 是 BigQuant 的数据存储单元，支持读写自定义数据：

```python
import pandas as pd
from bigquant import dai

# 写入数据（需要开通写入权限才能执行写入操作）
df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02"],
    "instrument": ["000001.SZ", "000001.SZ"],
    "my_signal": [1.0, -1.0],
})
ds = dai.DataSource.write_bdb(
    data=df,
    id="my_signal_data",          # 指定 ID，不填则创建临时数据源
    partitioning=["date"],         # 分区列
    overwrite=True,                # 覆盖已有数据
)
print("写入成功，DataSource ID:", ds.id)

# 读取数据
ds = dai.DataSource("my_signal_data")    # 注意更换为你自己的表名
df = ds.read_bdb(
    as_type=pd.DataFrame,
    partition_filter={"date": ("2024-01-01", "2024-01-31")},  # 范围过滤
    columns=["date", "instrument", "my_signal"],               # 指定列
)
```

### 搜索和浏览数据表

```python
from bigquant import dai

# 搜索数据表（支持中文、英文、表名、描述）
results = dai.search_datasources("股票日线")
for item in results:
    docs = item.get("docs", {})
    print(item.get("id", ""), docs.get("cn_name", ""))

# 查看数据表 schema
schema = dai.get_datasource_schema("cn_stock_bar1d")
print("表名:", schema["cn_name"])
for field in schema["fields"]:
    print(f"{field['name']:20} {field['type']:15} {field['description']}")

# 查看数据表更新记录
updates = dai.get_datasource_updates("cn_stock_bar1d")
for u in updates[:5]:
    print(u["created_at"], u.get("data", {}).get("rows"), "行")

# 列出自己创建的 DataSource
my_ds = dai.list_datasources()
for ds in my_ds:
    print(ds.get("id"), ds.get("docs", {}).get("cn_name", ""))
```


---

## 本地回测（BigTrader）

BigTrader 是 BigQuant SDK 的本地回测引擎，策略逻辑完全在本地运行，保护隐私且响应极快。

### 基础回测

```python
from bigquant import bigtrader

def initialize(context):
    # 设置手续费
    context.set_commission(bigtrader.PerOrder(buy_cost=0.0003, sell_cost=0.0003))
    # 初始化变量
    context.asset = "000001.SZ"

def handle_data(context, data):
    # 获取当前持仓
    position = context.get_account_position(context.asset)
    # 如果没有持仓，买入
    if position.amount == 0:
        context.order_target_percent(context.asset, 1.0)

performance = bigtrader.run(
    start_date="2024-01-01",
    end_date="2024-01-31",
    initialize=initialize,
    handle_data=handle_data,
    capital_base=1_000_000,  # 初始资金 100 万
    benchmark="000300.SH",   # 基准指数
)

# 查看绩效摘要
print(performance.summary)
```

### 查看回测结果

```python
# 绩效摘要（字典）
summary = performance.summary
print("夏普比率:", summary["sharp_ratio"])
print("最大回撤:", summary["max_drawdown"])
print("年化收益:", summary["annual_return_ratio"])
print("累计收益:", summary["return_ratio"])
print("胜率:", summary["win_ratio"])

# 原始每日绩效数据（DataFrame）
raw_perf = performance.raw_perf
```

### 多资产策略

```python
from bigquant import bigtrader, dai

def initialize(context):
    context.set_commission(bigtrader.PerOrder(buy_cost=0.0003, sell_cost=0.0003))
    context.stocks = ["000001.SZ", "000002.SZ", "600519.SH"]

def handle_data(context, data):
    for stock in context.stocks:
        # 均等权重持仓
        context.order_target_percent(stock, 1.0 / len(context.stocks))

performance = bigtrader.run(
    start_date="2024-01-01",
    end_date="2024-01-31",
    initialize=initialize,
    handle_data=handle_data,
)
```


---

## 模拟交易（PaperTrading）

PaperTrading 模块提供云端模拟交易策略的管理与查询接口，可获取策略列表、持仓、订单、信号和绩效数据。

> **说明**：PaperTrading 管理的是在 BigQuant 平台上运行的模拟交易策略，需要先在平台上创建并运行策略。

### 获取策略列表

```python
from bigquant import papertrading

# 获取我的策略列表
result = papertrading.list(page=1, size=20)
print(f"共 {result['count']} 个策略")
for item in result["items"]:
    status = "运行中" if item["status"] == 1 else "已停止"
    print(f"{item['id']}  {item['strategy_name']}  {status}")
```

### 获取策略详情

```python
from bigquant import papertrading

strategy = papertrading.get("your_strategy_id")
print(strategy.strategy_info)
```

### 查询持仓

```python
from bigquant import papertrading

strategy = papertrading.get("your_strategy_id")

# 获取最新交易日持仓
latest = strategy.get_positions(order_by=["-trading_day"], page=1, size=1)
latest_day = latest["items"][0]["trading_day"]

positions = strategy.get_positions(
    constraints={"trading_day": latest_day},
    page=1,
    size=100,
)
for pos in positions["items"]:
    print(f"{pos['instrument']}  持仓: {pos['current_qty']}  盈亏: {pos['position_pnl']:.2f}")
```

### 查询已成交订单

```python
from bigquant import papertrading

strategy = papertrading.get("your_strategy_id")

# 获取最新交易日订单
latest = strategy.get_orders(order_by=["-trading_day"], page=1, size=1)
latest_day = latest["items"][0]["trading_day"]

orders = strategy.get_orders(
    constraints={"trading_day": latest_day},
    page=1,
    size=100,
)
for order in orders["items"]:
    direction = "买入" if str(order["direction"]) == "1" else "卖出"
    print(f"{order['instrument']}  {direction}  成交量: {order['filled_qty']}  均价: {order['average_price']:.2f}")
```

### 查询待交易信号

```python
from bigquant import papertrading

strategy = papertrading.get("your_strategy_id")

# 获取最新交易日计划订单（信号）
latest = strategy.get_planned_orders(order_by=["-trading_day"], page=1, size=1)
latest_day = latest["items"][0]["trading_day"]

signals = strategy.get_planned_orders(
    constraints={"trading_day": latest_day},
    page=1,
    size=100,
)
for sig in signals["items"]:
    if sig.get("instrument") and sig.get("order_qty", 0) > 0:
        direction = "买入" if str(sig["direction"]) == "1" else "卖出"
        print(f"{sig['instrument']}  {direction}  委托量: {sig['order_qty']}  委托价: {sig['order_price']:.2f}")
```

### 查询绩效

```python
from bigquant import papertrading

strategy = papertrading.get("your_strategy_id")
performances = strategy.get_performances()

if performances:
    latest = performances[-1]
    risk = latest[4] if isinstance(latest[4], dict) else {}
    print(f"交易日: {latest[0]}")
    print(f"累计收益: {latest[6]:.2%}")
    print(f"年化收益: {latest[3]:.2%}")
    print(f"最大回撤: {latest[2]:.2%}")
    print(f"夏普比率: {risk.get('sharpe', 'N/A')}")
```

### 获取分享策略列表

```python
from bigquant import papertrading

# 获取社区分享的策略
result = papertrading.list_shared(
    benchmark="000300.SH",
    page=1,
    size=20,
)
for item in result["items"]:
    print(f"{item.get('strategy_id')}  {item.get('strategy_name')}")
```

### 删除策略

```python
from bigquant import papertrading

# 如果策略已分享，需先取消分享
papertrading.unshare("your_strategy_id")

# 删除策略（同时取消数据订阅、删除任务）
result = papertrading.delete("your_strategy_id")
print(f"已删除 {result.get('deleted', 0)} 个策略")
```


---

## 云端计算（AIStudio）

> **暂未开放**：AIStudio SDK 接口正在开发中，敬请期待。


---

## 分布式计算（FAI）

> **暂未开放**：FAI 分布式计算 SDK 接口正在开发中，敬请期待。


---

## 常见问题

### Q: 安装后 `bq` 命令找不到？

确认 Python Scripts 目录在 PATH 中：

```bash
# macOS/Linux
export PATH="$HOME/.local/bin:$PATH"

# Windows（在 PowerShell 中）
$env:PATH += ";$env:APPDATA\Python\Python311\Scripts"
```

### Q: 查询报错 "full_db_scan=False but no filters provided"？

需要提供 `filters` 参数，或者设置 `full_db_scan=True`：

```python
# 方式一：提供 filters（推荐，性能更好）
df = dai.query(sql, filters={"date": ["2024-01-01", "2024-12-31"]}).df()

# 方式二：允许全表扫描（数据量大时较慢）
df = dai.query(sql, full_db_scan=True).df()
```

### Q: Token 认证和 AK/SK 认证有什么区别？

* **AK/SK 认证**：支持全部功能，包括 Arrow Flight 高性能数据传输和 AIStudio
* **Token 认证**：仅支持 HTTP 接口，不支持 Arrow Flight（即 `dai.query` 直连模式）

### Q: 代码在 BigQuant AIStudio 网页端和本地 SDK 有什么区别？

几乎完全一致。主要区别：

* 网页端 `import dai`，本地 `from bigquant import dai`
* 网页端无需初始化，本地需要 `bq auth` 配置凭证
* 本地 `dai.query` 默认走 Arrow Flight 直连（更快），网页端走云端引擎

### Q: 如何退出登录？

```bash
bq auth logout
```

### Q: AIStudio 启动失败怎么办？

AIStudio 启动最多等待 60 秒。如果超时，可能原因：


1. 网络连接问题，检查网络
2. 资源规格不可用，尝试使用默认免费规格（不传 `resource_spec_id`）
3. 账户余额不足，登录平台查看账户状态

### Q: FAI 集群连接失败怎么办？

确认已安装 FAI 依赖：

```bash
pip install 'bigquant[fai]' -i https://pypi.bigquant.com/simple/
```

FAI 需要 Python 3.11，不支持其他版本。