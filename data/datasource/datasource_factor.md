# 因子数据源

## bq_exposure

- **中文名**：Barra 风格因子暴露表
- **描述**：A 股各证券的 Barra 风格因子暴露，包含 10 个风格因子（SIZE / BETA / MOMENTUM / RESVOL / SIZENL / BTOP / LIQUIDTY / EARNYILD / GROWTH / LEVERAGE）、一级行业、流通市值与权重、当期收益率，以及 31 个行业哑变量（覆盖 sw2014、sw2021 两套申万行业分类），可直接用于因子收益率回归、风格归因与行业中性化。扩展自 `jq_style_factor`。
- **字段**：`date` / `instrument` / 10 个风格因子 / `industry_level1_code` / `float_market_cap` / `weights` / `ret` / 31 个行业哑变量
- **完整文档**：`data/datasource/tables/bq_exposure.md`
