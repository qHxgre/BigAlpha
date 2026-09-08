## 财务报表科目系列表（_shift）

本文档覆盖以下四张表，它们结构完全一致，仅财务口径不同：

| 表名 | 字段后缀 | 口径 | 含义 |
|---|---|---|---|
| cn_stock_financial_lf_shift | `_lf` | Latest Filing，最新一期 | 截至公告日已披露的最新修订版本，含资产负债表 + 利润表 + 现金流量表 |
| cn_stock_financial_ly_shift | `_ly` | Latest Year，最新一期年报 | 仅取年报（12-31 报告期）的最新修订版本，含资产负债表 + 利润表 + 现金流量表 |
| cn_stock_financial_mrq_shift | `_mrq` | Most Recent Quarter，单季度 | 当季新增量（季报减去上季末累计值），**不含资产负债表** |
| cn_stock_financial_ttm_shift | `_ttm` | Trailing Twelve Months，滚动十二个月 | 过去四个季度加总，**不含资产负债表** |

> **字段命名规则**：将下表中的 `<suffix>` 替换为对应表的后缀即可。例如净利润字段在四张表中分别为 `net_profit_lf`、`net_profit_ly`、`net_profit_mrq`、`net_profit_ttm`。

### 为什么 MRQ / TTM 不含资产负债表

资产负债表反映的是**某一时点的存量**（如期末总资产），对存量做单季度差分或滚动加总在会计上没有意义——总资产的"单季度增量"不等于任何标准财务指标。需要分析资产变化时，应直接用 lf/ly 表的相邻 shift 做差。

---

## 核心概念

**PIT（Point-In-Time，信息点时间）**
每行数据的 `date` 字段代表该财务数据**实际可被市场获知的日期**（即公告日），而非报告期截止日。回测时以策略运行日过滤 `date`，可严格避免使用未来财务数据，消除前视偏差（Look-Ahead Bias）。

**Shift（报告期偏移）**
`shift` 字段表示相对于当前最新报告期的**向前偏移期数**（非负整数）：
- `shift = 0`：最新一期
- `shift = 1`：上一期
- `shift = 4`：去年同期（季报口径下，4 期前 = 去年同季）

同一 `(date, instrument)` 下有多行，每行对应不同 shift，方便直接计算同比/环比，无需自行做时间对齐。

**report_date（报告期）**
财务报告所覆盖的会计期间截止日，如 `2025-09-30` 表示 2025 年三季报。与 `date`（公告日）不同，`report_date` 是会计意义上的期末日期。

### 数据结构示意

```
date(公告日)   instrument  shift  report_date   net_profit_lf  ...
2025-10-31     000001.SZ    0     2025-09-30    xxxxxxx
2025-10-31     000001.SZ    1     2025-06-30    xxxxxxx
2025-10-31     000001.SZ    2     2025-03-31    xxxxxxx
2025-10-31     000001.SZ    3     2024-12-31    xxxxxxx
2025-10-31     000001.SZ    4     2024-09-30    xxxxxxx   ← 同比对比期
```

---

## 字段

### 索引字段（四表通用）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 公告日（PIT 基准日，回测时以此字段过滤） | pd.NaT |
| instrument | pd.StringDtype | 证券代码 | np.nan |
| report_date | np.datetime64 | 报告期截止日（如 2025-09-30 表示三季报） | pd.NaT |
| shift | np.int8 | 报告期偏移量（0=最新期，1=上一期，依此类推） | np.nan |

### 资产负债表——流动资产（仅 lf / ly）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| moneytary_assets_`<suffix>` | np.double | 货币资金 | np.nan |
| settlment_reserves_`<suffix>` | np.double | 结算备付金 | np.nan |
| loans_to_banks_and_fin_institutions_`<suffix>` | np.double | 拆出资金 | np.nan |
| tradable_fin_assets_`<suffix>` | np.double | 交易性金融资产 | np.nan |
| derivatives_fin_assets_`<suffix>` | np.double | 衍生金融资产 | np.nan |
| notes_receivable_`<suffix>` | np.double | 应收票据 | np.nan |
| accounts_receivable_`<suffix>` | np.double | 应收账款 | np.nan |
| notes_and_accounts_receivable_`<suffix>` | np.double | 应收票据及应收账款 | np.nan |
| receivables_financing_`<suffix>` | np.double | 应收款项融资 | np.nan |
| prepayments_`<suffix>` | np.double | 预付款项 | np.nan |
| premiums_receivable_`<suffix>` | np.double | 应收保费 | np.nan |
| reinsurance_receivables_`<suffix>` | np.double | 应收分保账款 | np.nan |
| receivable_reinsurance_contract_reserve_`<suffix>` | np.double | 应收分保合同准备金 | np.nan |
| interest_receivable_`<suffix>` | np.double | 应收利息 | np.nan |
| dividends_receivable_`<suffix>` | np.double | 应收股利 | np.nan |
| other_receivables_`<suffix>` | np.double | 其他应收款 | np.nan |
| other_receivables_sum_`<suffix>` | np.double | 其他应收款合计 | np.nan |
| fin_assets_purchased_under_resale_`<suffix>` | np.double | 买入返售金融资产 | np.nan |
| inventories_`<suffix>` | np.double | 存货 | np.nan |
| contract_assets_`<suffix>` | np.double | 合同资产 | np.nan |
| assets_held_for_sale_`<suffix>` | np.double | 持有待售资产 | np.nan |
| noncurr_assets_due_within_1y_`<suffix>` | np.double | 一年内到期的非流动资产 | np.nan |
| other_current_assets_`<suffix>` | np.double | 其他流动资产 | np.nan |
| spec_diff_of_current_assets_`<suffix>` | np.double | 流动资产差额（特殊报表科目） | np.nan |
| totbal_diff_of_current_assets_`<suffix>` | np.double | 流动资产差额（合计平衡科目） | np.nan |
| total_current_assets_`<suffix>` | np.double | 流动资产合计 | np.nan |

### 资产负债表——非流动资产（仅 lf / ly）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| loans_and_advances_`<suffix>` | np.double | 发放贷款及垫款 | np.nan |
| fin_assets_by_amortized_cost_`<suffix>` | np.double | 以摊余成本计量的金融资产 | np.nan |
| fin_assets_by_fair_value_`<suffix>` | np.double | 以公允价值计量且其变动计入其他综合收益的金融资产 | np.nan |
| available_for_sale_fin_assets_`<suffix>` | np.double | 可供出售金融资产 | np.nan |
| held_to_maturity_invesments_`<suffix>` | np.double | 持有至到期投资 | np.nan |
| debt_investments_`<suffix>` | np.double | 债权投资 | np.nan |
| other_debt_investments_`<suffix>` | np.double | 其他债权投资 | np.nan |
| longterm_receivables_`<suffix>` | np.double | 长期应收款 | np.nan |
| longterm_equity_investments_`<suffix>` | np.double | 长期股权投资 | np.nan |
| other_equity_investments_`<suffix>` | np.double | 其他权益工具投资 | np.nan |
| other_noncurr_fin_assets_`<suffix>` | np.double | 其他非流动金融资产 | np.nan |
| investment_property_`<suffix>` | np.double | 投资性房地产 | np.nan |
| fixed_assets_`<suffix>` | np.double | 固定资产 | np.nan |
| fixed_assets_sum_`<suffix>` | np.double | 固定资产合计 | np.nan |
| construction_in_progress_`<suffix>` | np.double | 在建工程 | np.nan |
| construction_in_progress_sum_`<suffix>` | np.double | 在建工程合计 | np.nan |
| project_materials_`<suffix>` | np.double | 工程物资 | np.nan |
| fixed_assets_disposal_`<suffix>` | np.double | 固定资产清理 | np.nan |
| productive_biological_assets_`<suffix>` | np.double | 生产性生物资产 | np.nan |
| oil_and_gas_assets_`<suffix>` | np.double | 油气资产 | np.nan |
| right_of_use_assets_`<suffix>` | np.double | 使用权资产 | np.nan |
| intangible_assets_`<suffix>` | np.double | 无形资产 | np.nan |
| development_costs_`<suffix>` | np.double | 开发支出 | np.nan |
| goodwill_`<suffix>` | np.double | 商誉 | np.nan |
| longterm_prepaid_expense_`<suffix>` | np.double | 长期待摊费用 | np.nan |
| deferred_tax_assets_`<suffix>` | np.double | 递延所得税资产 | np.nan |
| other_noncurr_assets_`<suffix>` | np.double | 其他非流动资产 | np.nan |
| spec_diff_of_noncurr_assets_`<suffix>` | np.double | 非流动资产差额（特殊报表科目） | np.nan |
| totbal_diff_of_noncurr_assets_`<suffix>` | np.double | 非流动资产差额（合计平衡科目） | np.nan |
| total_noncurr_assets_`<suffix>` | np.double | 非流动资产合计 | np.nan |
| spec_diff_of_total_assets_`<suffix>` | np.double | 资产差额（特殊报表科目） | np.nan |
| totbal_diff_of_total_assets_`<suffix>` | np.double | 资产差额（合计平衡科目） | np.nan |
| total_assets_`<suffix>` | np.double | 资产总计 | np.nan |

### 资产负债表——流动负债（仅 lf / ly）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| shortterm_borrowings_`<suffix>` | np.double | 短期借款 | np.nan |
| borrowing_from_central_bank_`<suffix>` | np.double | 向中央银行借款 | np.nan |
| deposits_from_banks_and_fin_instiutions_`<suffix>` | np.double | 吸收存款及同业存放 | np.nan |
| loans_from_banks_and_fin_institutions_`<suffix>` | np.double | 拆入资金 | np.nan |
| tradable_fin_liabilities_`<suffix>` | np.double | 交易性金融负债 | np.nan |
| derivatives_fin_liabilities_`<suffix>` | np.double | 衍生金融负债 | np.nan |
| notes_payable_`<suffix>` | np.double | 应付票据 | np.nan |
| accounts_payable_`<suffix>` | np.double | 应付账款 | np.nan |
| notes_and_accounts_payable_`<suffix>` | np.double | 应付票据及应付账款 | np.nan |
| advances_`<suffix>` | np.double | 预收款项 | np.nan |
| contract_liabilities_`<suffix>` | np.double | 合同负债 | np.nan |
| fin_assets_sold_under_resale_`<suffix>` | np.double | 卖出回购金融资产款 | np.nan |
| fees_and_commissions_payable_`<suffix>` | np.double | 应付手续费及佣金 | np.nan |
| employee_benefits_payable_`<suffix>` | np.double | 应付职工薪酬 | np.nan |
| taxes_and_levies_payable_`<suffix>` | np.double | 应交税费 | np.nan |
| interest_payable_`<suffix>` | np.double | 应付利息 | np.nan |
| dividends_payable_`<suffix>` | np.double | 应付股利 | np.nan |
| other_payables_`<suffix>` | np.double | 其他应付款 | np.nan |
| other_payables_sum_`<suffix>` | np.double | 其他应付款合计 | np.nan |
| reinsurance_payables_`<suffix>` | np.double | 应付分保账款 | np.nan |
| insurance_contract_reserves_`<suffix>` | np.double | 保险合同准备金 | np.nan |
| acting_trading_payables_`<suffix>` | np.double | 代理买卖证券款 | np.nan |
| underwriting_payables_`<suffix>` | np.double | 代理承销证券款 | np.nan |
| liabilities_held_for_sale_`<suffix>` | np.double | 持有待售负债 | np.nan |
| noncurr_liabilities_due_within_1y_`<suffix>` | np.double | 一年内到期的非流动负债 | np.nan |
| deferred_income_current_liabilities_`<suffix>` | np.double | 递延收益（流动负债） | np.nan |
| shortterm_bonds_payable_`<suffix>` | np.double | 应付短期债券 | np.nan |
| other_current_liabilities_`<suffix>` | np.double | 其他流动负债 | np.nan |
| spec_diff_of_current_liabilities_`<suffix>` | np.double | 流动负债差额（特殊报表科目） | np.nan |
| totbal_diff_of_current_liabilities_`<suffix>` | np.double | 流动负债差额（合计平衡科目） | np.nan |
| total_current_liabilities_`<suffix>` | np.double | 流动负债合计 | np.nan |

### 资产负债表——非流动负债（仅 lf / ly）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| longterm_borrowings_`<suffix>` | np.double | 长期借款 | np.nan |
| bonds_payable_`<suffix>` | np.double | 应付债券 | np.nan |
| perpetual_bonds_`<suffix>` | np.double | 永续债 | np.nan |
| preference_shares_`<suffix>` | np.double | 优先股 | np.nan |
| lease_liabilities_`<suffix>` | np.double | 租赁负债 | np.nan |
| longterm_payables_`<suffix>` | np.double | 长期应付款 | np.nan |
| longterm_payables_sum_`<suffix>` | np.double | 长期应付款合计 | np.nan |
| longterm_employee_benefits_`<suffix>` | np.double | 长期应付职工薪酬 | np.nan |
| specific_payables_`<suffix>` | np.double | 专项应付款 | np.nan |
| provisions_`<suffix>` | np.double | 预计负债 | np.nan |
| deferred_tax_liabilities_`<suffix>` | np.double | 递延所得税负债 | np.nan |
| deferred_income_noncurr_liabilities_`<suffix>` | np.double | 递延收益（非流动负债） | np.nan |
| other_noncurr_liabilities_`<suffix>` | np.double | 其他非流动负债 | np.nan |
| spec_diff_of_noncurr_liabilities_`<suffix>` | np.double | 非流动负债差额（特殊报表科目） | np.nan |
| totbal_diff_of_noncurr_liabilities_`<suffix>` | np.double | 非流动负债差额（合计平衡科目） | np.nan |
| total_noncurr_liabilities_`<suffix>` | np.double | 非流动负债合计 | np.nan |
| spec_diff_of_total_liabilities_`<suffix>` | np.double | 负债差额（特殊报表科目） | np.nan |
| totbal_diff_of_total_liabilities_`<suffix>` | np.double | 负债差额（合计平衡科目） | np.nan |
| total_liabilities_`<suffix>` | np.double | 负债合计 | np.nan |

### 资产负债表——所有者权益（仅 lf / ly）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| share_capital_`<suffix>` | np.double | 实收资本（或股本） | np.nan |
| capital_reserves_`<suffix>` | np.double | 资本公积 | np.nan |
| treasury_shares_`<suffix>` | np.double | 库存股 | np.nan |
| balance_othcom_income_`<suffix>` | np.double | 其他综合收益 | np.nan |
| other_equity_instruments_`<suffix>` | np.double | 其他权益工具 | np.nan |
| preference_of_other_equity_instruments_`<suffix>` | np.double | 其中：优先股 | np.nan |
| specific_reserve_`<suffix>` | np.double | 专项储备 | np.nan |
| surplus_reserve_`<suffix>` | np.double | 盈余公积 | np.nan |
| general_reserve_`<suffix>` | np.double | 一般风险准备 | np.nan |
| undistributed_profit_`<suffix>` | np.double | 未分配利润 | np.nan |
| balance_translation_diff_of_foreign_currency_`<suffix>` | np.double | 外币报表折算差额 | np.nan |
| total_equity_to_parent_shareholders_`<suffix>` | np.double | 归属于母公司所有者权益合计 | np.nan |
| minority_interests_`<suffix>` | np.double | 少数股东权益 | np.nan |
| spec_diff_of_shareholders_equity_`<suffix>` | np.double | 股东权益差额（特殊报表科目） | np.nan |
| totbal_diff_of_shareholders_equity_`<suffix>` | np.double | 股东权益差额（合计平衡科目） | np.nan |
| total_owner_equity_`<suffix>` | np.double | 所有者权益合计 | np.nan |
| spec_diff_of_liabilities_and_shareholder_equity_`<suffix>` | np.double | 负债及股东权益差额（特殊报表科目） | np.nan |
| totbal_diff_of_liabilities_and_shareholder_equity_`<suffix>` | np.double | 负债及股东权益差额（合计平衡科目） | np.nan |
| total_liabilities_and_owner_equity_`<suffix>` | np.double | 负债和所有者权益总计 | np.nan |

### 利润表（四表通用）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| total_operating_revenue_`<suffix>` | np.float64 | 营业总收入 | np.nan |
| operating_revenue_`<suffix>` | np.float64 | 营业收入 | np.nan |
| interest_income_`<suffix>` | np.float64 | 利息收入 | np.nan |
| insurance_premium_income_`<suffix>` | np.float64 | 已赚保费 | np.nan |
| fee_and_commission_income_`<suffix>` | np.float64 | 手续费及佣金收入 | np.nan |
| spec_diff_of_operating_revenue_`<suffix>` | np.float64 | 营业总收入差额（特殊报表科目） | np.nan |
| totbal_diff_of_operating_revenue_`<suffix>` | np.float64 | 营业总收入差额（合计平衡科目） | np.nan |
| total_operating_costs_`<suffix>` | np.float64 | 营业总成本 | np.nan |
| operating_costs_`<suffix>` | np.float64 | 营业成本 | np.nan |
| interest_costs_`<suffix>` | np.float64 | 利息支出 | np.nan |
| fee_and_commission_costs_`<suffix>` | np.float64 | 手续费及佣金支出 | np.nan |
| surrenders_`<suffix>` | np.float64 | 退保金 | np.nan |
| net_insurance_claims_paid_`<suffix>` | np.float64 | 赔付支出净额 | np.nan |
| net_amount_of_insurance_reserve_`<suffix>` | np.float64 | 提取保险合同准备金净额 | np.nan |
| expense_on_policy_dividends_`<suffix>` | np.float64 | 保单红利支出 | np.nan |
| reinsurance_premium_expense_`<suffix>` | np.float64 | 分保费用 | np.nan |
| taxes_and_levies_`<suffix>` | np.float64 | 税金及附加 | np.nan |
| selling_epense_`<suffix>` | np.float64 | 销售费用 | np.nan |
| administrative_expense_`<suffix>` | np.float64 | 管理费用 | np.nan |
| research_and_development_expense_`<suffix>` | np.float64 | 研发费用 | np.nan |
| finance_expense_`<suffix>` | np.float64 | 财务费用 | np.nan |
| fin_interest_expense_`<suffix>` | np.float64 | 财务费用：利息费用 | np.nan |
| fin_interest_income_`<suffix>` | np.float64 | 财务费用：利息收入 | np.nan |
| asset_impairment_loss_`<suffix>` | np.float64 | 资产减值损失 | np.nan |
| credit_impairment_loss_`<suffix>` | np.float64 | 信用减值损失 | np.nan |
| spec_diff_of_operating_costs_`<suffix>` | np.float64 | 营业总成本差额（特殊报表科目） | np.nan |
| totbal_diff_of_operating_costs_`<suffix>` | np.float64 | 营业总成本差额（合计平衡科目） | np.nan |
| fair_value_chg_gain_`<suffix>` | np.float64 | 公允价值变动收益 | np.nan |
| invest_income_`<suffix>` | np.float64 | 投资收益 | np.nan |
| invest_income_of_jv_and_associates_`<suffix>` | np.float64 | 对联营企业和合营企业的投资收益 | np.nan |
| income_derecognition_of_fin_assets_at_amortized_cost_`<suffix>` | np.float64 | 以摊余成本计量的金融资产终止确认收益 | np.nan |
| net_income_of_open_hedge_`<suffix>` | np.float64 | 净敞口套期收益 | np.nan |
| exchange_gain_`<suffix>` | np.float64 | 汇兑收益 | np.nan |
| asset_disposal_income_`<suffix>` | np.float64 | 资产处置收益 | np.nan |
| other_income_`<suffix>` | np.float64 | 其他收益 | np.nan |
| spec_diff_of_operating_profit_`<suffix>` | np.float64 | 营业利润差额（特殊报表科目） | np.nan |
| totbal_diff_of_operating_profit_`<suffix>` | np.float64 | 营业利润差额（合计平衡科目） | np.nan |
| operating_profit_`<suffix>` | np.float64 | 营业利润 | np.nan |
| nonoperating_income_`<suffix>` | np.float64 | 营业外收入 | np.nan |
| noncurr_assets_dispose_gain_`<suffix>` | np.float64 | 非流动资产处置利得 | np.nan |
| nonoperating_costs_`<suffix>` | np.float64 | 营业外支出 | np.nan |
| noncurr_assets_dispose_loss_`<suffix>` | np.float64 | 非流动资产处置损失 | np.nan |
| spec_diff_of_total_profit_`<suffix>` | np.float64 | 利润总额差额（特殊报表科目） | np.nan |
| totbal_diff_of_total_profit_`<suffix>` | np.float64 | 利润总额差额（合计平衡科目） | np.nan |
| total_profit_`<suffix>` | np.float64 | 利润总额 | np.nan |
| income_tax_expense_`<suffix>` | np.float64 | 所得税费用 | np.nan |
| spec_diff_of_net_profit_`<suffix>` | np.float64 | 净利润差额（特殊报表科目） | np.nan |
| totbal_diff_of_net_profit_`<suffix>` | np.float64 | 净利润差额（合计平衡科目） | np.nan |
| net_profit_`<suffix>` | np.float64 | 净利润 | np.nan |
| continuing_operation_net_profit_`<suffix>` | np.float64 | 持续经营净利润 | np.nan |
| discontinued_operation_net_profit_`<suffix>` | np.float64 | 终止经营净利润 | np.nan |
| net_profit_to_parent_shareholders_`<suffix>` | np.float64 | 归属于母公司所有者的净利润 | np.nan |
| net_profit_to_minority_`<suffix>` | np.float64 | 少数股东损益 | np.nan |
| eps_basic_`<suffix>` | np.float64 | 基本每股收益 | np.nan |
| eps_diluted_`<suffix>` | np.float64 | 稀释每股收益 | np.nan |
| income_othcom_income_`<suffix>` | np.float64 | 其他综合收益 | np.nan |
| othcom_income_to_parent_shareholders_`<suffix>` | np.float64 | 归属母公司所有者的其他综合收益 | np.nan |
| othcom_income_cannt_reclass_`<suffix>` | np.float64 | 以后不能重分类进损益的其他综合收益 | np.nan |
| chg_by_remeasurements_`<suffix>` | np.float64 | 重新计量设定受益计划净负债或净资产的变动 | np.nan |
| othcom_income_cannt_reclass_under_equity_method_`<suffix>` | np.float64 | 权益法下在被投资单位不能重分类进损益的其他综合收益中享有的份额 | np.nan |
| other_cannt_reclass_`<suffix>` | np.float64 | 其他以后不能重分类进损益 | np.nan |
| other_equity_instruments_fair_value_chg_`<suffix>` | np.float64 | 其他权益工具投资公允价值变动 | np.nan |
| own_credit_risk_fair_value_chg_`<suffix>` | np.float64 | 企业自身信用风险公允价值变动 | np.nan |
| othcom_income_reclass_`<suffix>` | np.float64 | 以后将重分类进损益的其他综合收益 | np.nan |
| othcom_income_reclass_under_equity_method_`<suffix>` | np.float64 | 权益法下在被投资单位以后将重分类进损益的其他综合收益中享有的份额 | np.nan |
| available_for_sale_fin_assets_fair_value_chg_`<suffix>` | np.float64 | 可供出售金融资产公允价值变动损益 | np.nan |
| gains_or_losses_from_htm_to_afs_`<suffix>` | np.float64 | 持有至到期投资重分类为可供出售金融资产损益 | np.nan |
| effective_of_gains_or_losses_on_cashflow_hedge_`<suffix>` | np.float64 | 现金流量套期损益的有效部分 | np.nan |
| income_translation_diff_of_foreign_currency_`<suffix>` | np.float64 | 外币财务报表折算差额 | np.nan |
| other_reclass_`<suffix>` | np.float64 | 其他以后将重分类进损益 | np.nan |
| other_debt_investments_fair_value_chg_`<suffix>` | np.float64 | 其他债权投资公允价值变动 | np.nan |
| othcom_income_from_reclass_of_fin_assets_`<suffix>` | np.float64 | 金融资产重分类计入其他综合收益的金额 | np.nan |
| credit_impairment_of_other_debt_investments_`<suffix>` | np.float64 | 其他债权投资信用减值准备 | np.nan |
| cashflow_hedge_reserve_`<suffix>` | np.float64 | 现金流量套期储备 | np.nan |
| othcom_income_to_minority_`<suffix>` | np.float64 | 归属于少数股东的其他综合收益 | np.nan |
| total_comprehensive_income_`<suffix>` | np.float64 | 综合收益总额 | np.nan |
| total_comprehensive_income_to_parent_shareholders_`<suffix>` | np.float64 | 归属于母公司股东的综合收益总额 | np.nan |

### 现金流量表——经营活动（四表通用）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| cash_received_from_sales_and_services_`<suffix>` | np.double | 销售商品、提供劳务收到的现金 | np.nan |
| netinc_in_deposits_`<suffix>` | np.double | 客户存款和同业存放款项净增加额 | np.nan |
| netinc_in_borrowings_from_central_bank_`<suffix>` | np.double | 向中央银行借款净增加额 | np.nan |
| netinc_in_loans_from_other_fin_institutions_`<suffix>` | np.double | 向其他金融机构拆入资金净增加额 | np.nan |
| cash_received_from_premiums_`<suffix>` | np.double | 收到原保险合同保费取得的现金 | np.nan |
| net_cash_received_from_reinsurance_`<suffix>` | np.double | 收到再保业务现金净额 | np.nan |
| netinc_in_insurance_deposits_and_invest_`<suffix>` | np.double | 保户储金及投资款净增加额 | np.nan |
| netinc_in_disposal_fin_assets_`<suffix>` | np.double | 处置以公允价值计量且其变动计入当期损益的金融资产净增加额 | np.nan |
| cash_received_from_interests_fess_and_commissions_`<suffix>` | np.double | 收取利息、手续费及佣金的现金 | np.nan |
| netinc_in_loans_from_banks_and_fin_institutions_`<suffix>` | np.double | 拆入资金净增加额 | np.nan |
| netinc_in_repurchase_transactions_`<suffix>` | np.double | 回购业务资金净增加额 | np.nan |
| taxes_and_levies_rebates_`<suffix>` | np.double | 收到的税费返还 | np.nan |
| cash_received_from_other_operating_`<suffix>` | np.double | 收到其他与经营活动有关的现金 | np.nan |
| spec_diff_of_cifoa_`<suffix>` | np.double | 经营活动现金流入差额（特殊报表科目） | np.nan |
| totbal_diff_of_cifoa_`<suffix>` | np.double | 经营活动现金流入差额（合计平衡科目） | np.nan |
| subtotal_cifoa_`<suffix>` | np.double | 经营活动现金流入小计 | np.nan |
| cash_paid_for_goods_and_services_`<suffix>` | np.double | 购买商品、接受劳务支付的现金 | np.nan |
| netinc_in_loans_and_advances_`<suffix>` | np.double | 客户贷款及垫款净增加额 | np.nan |
| netinc_deposits_central_bank_interbank_`<suffix>` | np.double | 存放中央银行和同业款项净增加额 | np.nan |
| cash_paid_for_claims_`<suffix>` | np.double | 支付原保险合同赔付款项的现金 | np.nan |
| cash_paid_for_interests_fees_and_commissions_`<suffix>` | np.double | 支付利息、手续费及佣金的现金 | np.nan |
| cash_paid_for_policy_dividends_`<suffix>` | np.double | 支付保单红利的现金 | np.nan |
| cash_paid_for_employees_`<suffix>` | np.double | 支付给职工以及为职工支付的现金 | np.nan |
| cash_paid_for_taxes_and_levies_`<suffix>` | np.double | 支付的各项税费 | np.nan |
| other_cofoa_`<suffix>` | np.double | 支付其他与经营活动有关的现金 | np.nan |
| spec_diff_of_cofoa_`<suffix>` | np.double | 经营活动现金流出差额（特殊报表科目） | np.nan |
| totbal_diff_of_cofoa_`<suffix>` | np.double | 经营活动现金流出差额（合计平衡科目） | np.nan |
| subtotal_cofoa_`<suffix>` | np.double | 经营活动现金流出小计 | np.nan |
| spec_diff_of_net_cffoa_`<suffix>` | np.double | 经营活动现金流量净额差额（特殊报表科目） | np.nan |
| totbal_diff_of_net_cffoa_`<suffix>` | np.double | 经营活动现金流量净额差额（合计平衡科目） | np.nan |
| net_cffoa_`<suffix>` | np.double | 经营活动产生的现金流量净额 | np.nan |

### 现金流量表——投资活动（四表通用）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| cash_received_from_disposal_investments_`<suffix>` | np.double | 收回投资收到的现金 | np.nan |
| return_on_investment_`<suffix>` | np.double | 取得投资收益收到的现金 | np.nan |
| net_cash_received_from_disposal_filt_assets_`<suffix>` | np.double | 处置固定资产、无形资产和其他长期资产收回的现金净额 | np.nan |
| net_cash_received_from_disposal_subsidiaries_`<suffix>` | np.double | 处置子公司及其他营业单位收到的现金净额 | np.nan |
| cash_received_from_other_investing_`<suffix>` | np.double | 收到其他与投资活动有关的现金 | np.nan |
| spec_diff_of_cifia_`<suffix>` | np.double | 投资活动现金流入差额（特殊报表科目） | np.nan |
| totbal_diff_of_cifia_`<suffix>` | np.double | 投资活动现金流入差额（合计平衡科目） | np.nan |
| subtotal_cifia_`<suffix>` | np.double | 投资活动现金流入小计 | np.nan |
| cash_paid_for_filt_assets_`<suffix>` | np.double | 购建固定资产、无形资产和其他长期资产支付的现金 | np.nan |
| cash_paid_for_investments_`<suffix>` | np.double | 投资支付的现金 | np.nan |
| netinc_in_pledge_loans_`<suffix>` | np.double | 质押贷款净增加额 | np.nan |
| cash_paid_by_acquiring_subsidiaries_`<suffix>` | np.double | 取得子公司及其他营业单位支付的现金净额 | np.nan |
| cash_paid_for_other_investing_`<suffix>` | np.double | 支付其他与投资活动有关的现金 | np.nan |
| spec_diff_of_cofia_`<suffix>` | np.double | 投资活动现金流出差额（特殊报表科目） | np.nan |
| totbal_diff_of_cofia_`<suffix>` | np.double | 投资活动现金流出差额（合计平衡科目） | np.nan |
| subtotal_of_cofia_`<suffix>` | np.double | 投资活动现金流出小计 | np.nan |
| spec_diff_of_net_cffia_`<suffix>` | np.double | 投资活动现金流量净额差额（特殊报表科目） | np.nan |
| totbal_diff_of_net_cffia_`<suffix>` | np.double | 投资活动现金流量净额差额（合计平衡科目） | np.nan |
| net_cffia_`<suffix>` | np.double | 投资活动产生的现金流量净额 | np.nan |

### 现金流量表——筹资活动（四表通用）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| capital_contributions_received_`<suffix>` | np.double | 吸收投资收到的现金 | np.nan |
| cash_received_by_subsidiaries_from_minority_`<suffix>` | np.double | 子公司吸收少数股东投资收到的现金 | np.nan |
| cash_received_from_borrowings_`<suffix>` | np.double | 取得借款收到的现金 | np.nan |
| cash_received_from_bond_issue_`<suffix>` | np.double | 发行债券收到的现金 | np.nan |
| cash_received_from_other_financing_`<suffix>` | np.double | 收到其他与筹资活动有关的现金 | np.nan |
| spec_diff_of_ciffa_`<suffix>` | np.double | 筹资活动现金流入差额（特殊报表科目） | np.nan |
| totbal_diff_of_ciffa_`<suffix>` | np.double | 筹资活动现金流入差额（合计平衡科目） | np.nan |
| subtotal_ciffa_`<suffix>` | np.double | 筹资活动现金流入小计 | np.nan |
| cash_paid_for_debt_repayment_`<suffix>` | np.double | 偿还债务支付的现金 | np.nan |
| cash_paid_for_dividends_profits_interests_`<suffix>` | np.double | 分配股利、利润或偿付利息支付的现金 | np.nan |
| cash_paid_by_subsidiaries_to_minority_`<suffix>` | np.double | 子公司支付给少数股东的股利、利润 | np.nan |
| cash_paid_for_other_financing_`<suffix>` | np.double | 支付其他与筹资活动有关的现金 | np.nan |
| spec_diff_of_coffa_`<suffix>` | np.double | 筹资活动现金流出差额（特殊报表科目） | np.nan |
| totbal_diff_of_coffa_`<suffix>` | np.double | 筹资活动现金流出差额（合计平衡科目） | np.nan |
| subtotal_of_coffa_`<suffix>` | np.double | 筹资活动现金流出小计 | np.nan |
| spec_diff_of_net_cfffa_`<suffix>` | np.double | 筹资活动现金流量净额差额（特殊报表科目） | np.nan |
| totbal_diff_of_net_cfffa_`<suffix>` | np.double | 筹资活动现金流量净额差额（合计平衡科目） | np.nan |
| net_cfffa_`<suffix>` | np.double | 筹资活动产生的现金流量净额 | np.nan |
| effect_of_exchange_chg_on_cce_`<suffix>` | np.double | 汇率变动对现金及现金等价物的影响 | np.nan |
| spec_diff_of_netinc_in_cce_`<suffix>` | np.double | 直接法-现金及现金等价物净增加额差额（特殊报表科目） | np.nan |
| totbal_diff_of_netinc_in_cce_`<suffix>` | np.double | 直接法-现金及现金等价物净增加额差额（合计平衡科目） | np.nan |
| netinc_in_cce_`<suffix>` | np.double | 现金及现金等价物净增加额 | np.nan |
| cce_beginning_`<suffix>` | np.double | 期初现金及现金等价物余额 | np.nan |
| cce_ending_`<suffix>` | np.double | 期末现金及现金等价物余额 | np.nan |

### 现金流量表——间接法补充资料（四表通用）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| net_profit_in_cashflow_sheet_`<suffix>` | np.double | 现金流量表-净利润 | np.nan |
| asset_impairment_reserve_`<suffix>` | np.double | 资产减值准备 | np.nan |
| depreciation_of_fa_oga_pba_`<suffix>` | np.double | 固定资产折旧、油气资产折耗、生产性生物资产折旧 | np.nan |
| amorization_of_intangible_assets_`<suffix>` | np.double | 无形资产摊销 | np.nan |
| amortization_of_longterm_deferred_expenses_`<suffix>` | np.double | 长期待摊费用摊销 | np.nan |
| loss_from_disposal_of_fa_ia_lta_`<suffix>` | np.double | 处置固定资产、无形资产和其他长期资产的损失 | np.nan |
| loss_from_scraping_of_fixed_assets_`<suffix>` | np.double | 固定资产报废损失 | np.nan |
| loss_from_fair_value_chg_`<suffix>` | np.double | 公允价值变动损失 | np.nan |
| finance_expenses_in_cashflow_sheet_`<suffix>` | np.double | 现金流量表-财务费用 | np.nan |
| invest_loss_`<suffix>` | np.double | 投资损失 | np.nan |
| decrease_in_deferred_tax_assets_`<suffix>` | np.double | 递延所得税资产减少 | np.nan |
| increase_in_deferred_tax_liabilities_`<suffix>` | np.double | 递延所得税负债增加 | np.nan |
| decrease_in_inventories_`<suffix>` | np.double | 存货的减少 | np.nan |
| decrease_in_operating_receivables_`<suffix>` | np.double | 经营性应收项目的减少 | np.nan |
| increase_in_operating_payables_`<suffix>` | np.double | 经营性应付项目的增加 | np.nan |
| others_in_cashflow_sheet_`<suffix>` | np.double | 其他 | np.nan |
| spec_diff_of_net_cffoa_indirect_`<suffix>` | np.double | 间接法-经营活动现金流量净额差额（特殊报表科目） | np.nan |
| totbal_diff_of_net_cffoa_indirect_`<suffix>` | np.double | 间接法-经营活动现金流量净额差额（合计平衡科目） | np.nan |
| net_cffoa_indirect_`<suffix>` | np.double | 间接法-经营活动产生的现金流量净额 | np.nan |
| debt_transfer_to_capital_`<suffix>` | np.double | 债务转为资本 | np.nan |
| conv_corp_bonds_within_1y_`<suffix>` | np.double | 一年内到期的可转换公司债券 | np.nan |
| fin_lease_fixed_assets_`<suffix>` | np.double | 融资租入固定资产 | np.nan |
| cash_balance_ending_`<suffix>` | np.double | 现金的期末余额 | np.nan |
| cash_balance_beginning_`<suffix>` | np.double | 现金的期初余额 | np.nan |
| cce_balance_ending_`<suffix>` | np.double | 现金等价物的期末余额 | np.nan |
| cce_balance_beginning_`<suffix>` | np.double | 现金等价物的期初余额 | np.nan |
| spec_diff_of_netinc_in_cce_indirect_`<suffix>` | np.double | 间接法-现金及现金等价物净增加额差额（特殊报表科目） | np.nan |
| totbal_diff_of_netinc_in_cce_indirect_`<suffix>` | np.double | 间接法-现金及现金等价物净增加额差额（合计平衡科目） | np.nan |
| netinc_in_cce_indirect_`<suffix>` | np.double | 间接法-现金及现金等价物净增加额 | np.nan |
| credit_impairment_loss_in_cashflow_sheet_`<suffix>` | np.double | 信用减值损失（现金流量表） | np.nan |

---

## 示例

### lf：查询最新一期核心科目（shift=0）

```python
import dai
dai.query("""
SELECT date, instrument, report_date, shift,
       total_assets_lf, total_liabilities_lf, total_owner_equity_lf,
       total_operating_revenue_lf, net_profit_lf
FROM cn_stock_financial_lf_shift
WHERE date = '2025-10-31'
  AND instrument = '000001.SZ'
  AND shift = 0
""").df()
```

### lf：用 shift 计算同比增长率（shift=0 vs shift=4）

```python
import dai
dai.query("""
SELECT
    a.date, a.instrument,
    a.report_date AS cur_period, b.report_date AS yoy_period,
    a.net_profit_lf AS net_profit_cur,
    b.net_profit_lf AS net_profit_yoy,
    (a.net_profit_lf - b.net_profit_lf) / ABS(b.net_profit_lf) AS yoy_growth
FROM cn_stock_financial_lf_shift a
JOIN cn_stock_financial_lf_shift b
  ON a.date = b.date AND a.instrument = b.instrument
WHERE a.shift = 0 AND b.shift = 4
  AND a.date >= '2025-10-01' AND a.date <= '2025-10-31'
ORDER BY a.date, a.instrument
""").df()
```

### mrq：查询单季度净利润

```python
import dai
dai.query("""
SELECT date, instrument, report_date, shift, net_profit_mrq
FROM cn_stock_financial_mrq_shift
WHERE date = '2025-10-31'
  AND shift = 0
ORDER BY instrument
""").df()
```

### ttm：查询滚动十二个月营收与净利润

```python
import dai
dai.query("""
SELECT date, instrument, report_date,
       total_operating_revenue_ttm, net_profit_ttm
FROM cn_stock_financial_ttm_shift
WHERE date = '2025-10-31'
  AND shift = 0
ORDER BY instrument
""").df()
```

### ly：查询最新年报资产负债表

```python
import dai
dai.query("""
SELECT date, instrument, report_date,
       total_assets_ly, total_liabilities_ly, total_owner_equity_ly
FROM cn_stock_financial_ly_shift
WHERE date >= '2025-04-01' AND date <= '2025-04-30'
  AND shift = 0
ORDER BY date, instrument
""").df()
```
