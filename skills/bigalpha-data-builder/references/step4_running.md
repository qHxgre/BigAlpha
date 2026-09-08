# Step 4：生成 running.ipynb

## 文件位置

`<table_name>/running.ipynb`，与 schema.py / builder.py 同目录。

## 内容要求

* 必须包含动态向 `sys.path` 添加项目根目录的逻辑，避免多层级 import 报错。
* 实例化 Builder 时，时间参数格式为 `YYYY-MM-DD`。
* 通常一个 cell 即可，方便用户本地或线上调试。

## 示例代码

```python
import sys
import os

# 动态获取并添加项目根目录（自适应本地路径）
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from bigalpha_stock_bar1m_zz1000.builder import BigalphaStockBar1mZz1000Builder

# 执行数据构建
builder = BigalphaStockBar1mZz1000Builder(start_date='2026-01-01', end_date='2026-02-01')
df = builder.build()
```

## 完成本步骤后

向用户展示 running.ipynb 内容，告知：

> "三件套已完整生成。如需调整任何文件或追加测试 cell，请告诉我。"
