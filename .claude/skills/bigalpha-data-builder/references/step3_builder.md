# Step 3：生成 builder.py

## 文件位置

`<table_name>/builder.py`，与 `schema.py` 同目录。

## 类命名规范

* 类名：表名大驼峰 + `Builder` 后缀，例如 `BigalphaStockBar1mZz1000Builder`
* 继承自 `BaseBuilder`（来自 `base.py`）

## 必须定义的类变量

| 变量 | 类型 | 说明 |
|---|---|---|
| `datasource_id` | `str` | 即小写下划线表名，与目录名一致 |
| `unique_together` | `list[str]` | 主键字段，与 Schema 中 `primary=True` 字段对齐 |
| `sort_by` | `list[tuple]` | 排序规则，如 `[("date", "ascending"), ("instrument", "ascending")]` |
| `indexes` | `list[str]` | 分区索引，通常为 `["date"]` |
| `schema` | Schema 类引用 | 指向 `schema.py` 中的类 |

## 分区策略（dai_write 中赋值）

* **日频/周频/月频/年频**（粗粒度）→ 按**年**分区
  ```python
  df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y").astype("int64")
  ```
* **分钟频/Tick**（高频）→ 按**月**或**日**分区
  ```python
  df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y%m").astype("int64")
  ```

## 必须实现的方法

1. `__init__(self, start_date, end_date)` — 接收起止日期。
2. `normalize(self, df)` — 严格按 schema 重排列、转类型、填默认值。
3. `dai_write(self, df)` — 加分区列后调用 `dai.DataSource.write_bdb(...)` 落库。
4. `build(self)` — 主流程：抽取 → 计算/清洗 → `normalize` → `dai_write`。

## 数据抽取来源

`build()` 中通常使用 `dai.query(sql, filters={...}).df()` 拉取底层表。具体表名/字段引导用户参考 [datasource_reference.md](./datasource_reference.md)；`dai.query` / `dai.DataSource.write_bdb` / 截面 SQL 函数 / 分区字段等完整用法参考 [dai.md](./dai.md)。

> **进入本步骤前必须先 Read [dai.md](./dai.md)**，确认 `filters` 写法、分区列类型、`c_avg` / `c_std` / `c_normalize` 的 `pb:=` 用法等关键点。

## 示例代码

```python
from bigquant import dai
import pandas as pd
from datetime import datetime
from base import BaseBuilder
from bigalpha_stock_bar_1m_zz1000.schema import BigalphaStockBar1mZz1000Schema


class BigalphaStockBar1mZz1000Builder(BaseBuilder):
    datasource_id = "bigalpha_stock_bar_1m_zz1000"
    unique_together = ["date", "instrument"]
    sort_by = [("date", "ascending"), ("instrument", "ascending")]
    indexes = ["date"]
    schema = BigalphaStockBar1mZz1000Schema

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        print(f"初始化构建器: {self.datasource_id} | 周期: {self.start_date} 至 {self.end_date}")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=self.schema.columns())
        df = df.astype(self.schema.field_type_mapping())
        df = df.fillna(self.schema.field_default_mapping())
        return df

    def dai_write(self, df: pd.DataFrame):
        default_docs = self.schema.default_docs()
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.strftime("%Y%m").astype("int64")

        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )
        print(f"数据成功写入 BDB 数据源: {self.datasource_id}")

    def build(self) -> pd.DataFrame:
        print("开始构建数据...")
        t0 = datetime.now()

        # 1. 从底层数据源抽取 —— 由 AI 根据 Step 1 确认的数据源生成
        # sql = f"SELECT * FROM cn_stock_bar1m WHERE date >= '{self.start_date}' AND date <= '{self.end_date}'"
        # df = dai.query(sql, filters={"date": [self.start_date, self.end_date]}).df()

        # [此处填充核心计算/清洗逻辑]

        t1 = datetime.now()
        print(f"获取与处理数据耗时: {round((t1-t0).total_seconds(), 4)} 秒")

        df_normalized = self.normalize(df)
        self.dai_write(df_normalized)

        t2 = datetime.now()
        print(f"数据落库存储耗时: {round((t2-t1).total_seconds(), 4)} 秒")
        return df_normalized
```

## 完成本步骤后

向用户展示生成的 builder.py，并询问：

> "builder.py 已就绪，数据抽取与清洗逻辑是否符合预期？确认后我会进入 Step 4 生成 running.ipynb。"
