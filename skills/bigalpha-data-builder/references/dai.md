# dai 数据引擎核心用法

BigQuant `dai` 是平台的统一数据访问层

## 1. 引入方式

```python
import dai
```

## 2. 读数据：dai.query

```python
df = dai.query(sql, filters={...}, bind_relations={...}, compression=False).df()
```

| 参数 | 作用 |
|---|---|
| `sql` | DuckDB 风格 SQL，可直接 `SELECT ... FROM <底层表名>` |
| `filters` | 分区下推过滤。**必须传**，否则可能全表扫描。常见 key：`date`（`[start, end]`）、`instrument`（list） |
| `bind_relations` | 把本地 `pd.DataFrame` 注册为 SQL 中的虚拟表名，例如 `bind_relations={"factor_data": df}` 后即可 `FROM factor_data` |
| `compression` | 高频分钟表传 `True` 可减小内存 |

`.df()` 把结果物化成 `pandas.DataFrame`。

`filters` 中的 date 边界格式：

* 日频：`"YYYY-MM-DD"`
* 分钟频：`"YYYY-MM-DD HH:MM:SS"`，区间一般写 `"start 00:00:00"` 到 `"end 23:59:59"`

## 3. 写数据：write_bdb

`base.py` 已经在 `Base.dai_write` 里封装好，子类一般只需重写"分区列怎么算"那一行：

```python
def dai_write(self, df: pd.DataFrame):
    default_docs = self.schema.default_docs()
    df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.year.astype("int64")  # 按年分区
    dai.DataSource.write_bdb(
        df,
        id=self.datasource_id,
        unique_together=self.unique_together,
        sort_by=self.sort_by,
        indexes=self.indexes,
        docs=default_docs,
    )
```

要点：

* `dai.DEFAULT_PARTITION_FIELD` 是 dai 约定的分区列名常量，**不要硬编码字符串**。
* 分区粒度选择（与 [step3_builder.md](./step3_builder.md) 一致）：
  * 日/周/月/年频 → `df["date"].dt.year.astype("int64")`
  * 分钟/Tick 频 → `df["date"].dt.strftime("%Y%m").astype("int64")`
* `unique_together` 决定主键去重，写入会覆盖同主键旧记录。
* `docs` 必须由 `schema.default_docs()` 提供，前端展示依赖它。

## 4. 常见坑

1. **`filters` 不传** → 触发全表扫描，分钟级数据可能直接 OOM。
2. **分钟表 date 边界缺时分秒** → 边界日数据可能漏。
3. **写库前没跑 `normalize`** → 列顺序、类型、缺失值不符合 schema，前端展示或下游查询会报错。
4. **分区列类型不是 int64** → write_bdb 会拒绝；`.astype("int64")` 不要省。
