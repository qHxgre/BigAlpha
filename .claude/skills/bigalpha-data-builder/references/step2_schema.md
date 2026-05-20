# Step 2：生成 schema.py

## 文件位置

`<table_name>/schema.py`，其中 `<table_name>` 与表名（小写下划线）一致。

## 类命名规范

* 文件名：`schema.py`
* 类名：表名转**大驼峰** + `Schema` 后缀
  * 例如表名 `bigalpha_stock_bar_1m_zz1000` → 类名 `BigalphaStockBar1mZz1000Schema`
* 继承自 `BaseSchema`（来自 `base.py`，环境已提供，无需创建）

## 字段定义规则

* 必须使用 `pydantic.Field` 并显式标注 `description`
* 主键字段（Step 1 中确认的 unique_together）需 `primary=True`，默认值通常为 `np.nan`
* 类型映射约定：
  | 业务含义 | Python 类型 |
  |---|---|
  | 日期 | `np.datetime64` |
  | 标的代码 / 字符串 | `pd.StringDtype` |
  | 浮点数（价格、因子值等） | `np.float32` |
  | 整数（数量、计数） | `np.int32` |
* 必须包含 `class Config: arbitrary_types_allowed = True`

## 示例代码

```python
import numpy as np
import pandas as pd
from pydantic import Field
from base import BaseSchema


class BigalphaStockBar1mZz1000Schema(BaseSchema):
    date: np.datetime64 = Field(description="日期", default=np.nan, primary=True)
    instrument: pd.StringDtype = Field(description="股票代码", default=np.nan, primary=True)
    high: np.float32 = Field(description="最高价", default=np.nan)
    open: np.float32 = Field(description="开盘价", default=np.nan)
    low: np.float32 = Field(description="最低价", default=np.nan)
    close: np.float32 = Field(description="收盘价", default=np.nan)

    class Config:
        arbitrary_types_allowed = True
```

## 完成本步骤后

向用户展示生成的 schema.py，并询问：

> "schema.py 已就绪，字段定义是否符合预期？确认后我会进入 Step 3 生成 builder.py。"
