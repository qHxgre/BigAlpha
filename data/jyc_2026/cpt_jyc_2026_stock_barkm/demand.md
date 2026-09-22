/Users/xiehao/Desktop/workspace/BigAlpha/data/jyc_2026/cpt_jyc_2026_stock_barkm/builder.py 修改这个builder，分为两个数据构建的 builder 类：

1. 从snapshot 中构建1分钟数据
2. 从1分钟数据中聚合 K 分钟数据，这个K由外部传入