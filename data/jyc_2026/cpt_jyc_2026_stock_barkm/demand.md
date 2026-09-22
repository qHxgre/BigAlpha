修改分钟数据的构建代码，要求如下：

1. 数据源用：cn_stock_level2_snapshot，数据schema参考：/Users/xiehao/Desktop/workspace/bigdatawarehouse/warehouse/builder/cn_stock_level2_snapshot/schema.py
2. 示例数据参考：/Users/xiehao/Desktop/workspace/BigAlpha/data/jyc_2026/cpt_jyc_2026_stock_barkm/cn_stock_level2_snapshot.parquet
3. 从快照数据改为分钟数据时，采用 左闭右开 的方式，比如 [10:00:00, 10:01:00) 归为 10:01:00，但是对于收盘时间点，可以向后取1分钟，比如：[14:59:00, 15:01:00) 归为 15:00:00.
4. 对于集合竞价期间，优先使用 price 作为价格，但当price为0或者nan时，用买卖一价作为价格price，高开低收入
3. 集合竞价统一归纳为 09:25:00，开盘第一分钟的数据 [09:30:00, 09:31:00] 归纳为 09:31:00，盘中正常分钟采用左开右闭的方式，开盘、收盘、集合竞价时间点需要特殊处理
4. 只需要构建 2020年至以来的中证1000的股票池
5. 构建了一分钟数据后，再把instrument_id和复权因子合并进来