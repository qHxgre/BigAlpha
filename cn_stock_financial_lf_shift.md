import numpy as np
import pandas as pd
from pydantic import Field
from warehouse.builder.base import BaseSchema


class CNStockFinancialDerivativeLFShiftSchema(BaseSchema):
    """财务衍生数据 (最新一期, 偏移, PIT)"""

    date: np.datetime64 = Field(description="公告日", default=pd.NaT)
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan)
    report_date: np.datetime64 = Field(description="报告期", default=pd.NaT)
    shift: np.int8 = Field(description="偏移报告期", default=np.nan)
    moneytary_assets_lf: np.double = Field(description="货币资金(最新一期)", default=np.nan) 
    settlment_reserves_lf: np.double = Field(description="结算备付金(最新一期)", default=np.nan) 
    loans_to_banks_and_fin_institutions_lf: np.double = Field(description="拆出资金(最新一期)", default=np.nan) 
    tradable_fin_assets_lf: np.double = Field(description="交易性金融资产(最新一期)", default=np.nan) 
    derivatives_fin_assets_lf: np.double = Field(description="衍生金融资产(最新一期)", default=np.nan) 
    notes_receivable_lf: np.double = Field(description="应收票据(最新一期)", default=np.nan) 
    accounts_receivable_lf: np.double = Field(description="应收账款(最新一期)", default=np.nan) 
    notes_and_accounts_receivable_lf: np.double = Field(description="应收票据及应收账款(最新一期)", default=np.nan) 
    receivables_financing_lf: np.double = Field(description="应收款项融资(最新一期)", default=np.nan) 
    prepayments_lf: np.double = Field(description="预付款项(最新一期)", default=np.nan) 
    premiums_receivable_lf: np.double = Field(description="应收保费(最新一期)", default=np.nan) 
    reinsurance_receivables_lf: np.double = Field(description="应收分保账款(最新一期)", default=np.nan) 
    receivable_reinsurance_contract_reserve_lf: np.double = Field(description="应收分保合同准备金(最新一期)", default=np.nan) 
    interest_receivable_lf: np.double = Field(description="应收利息(最新一期)", default=np.nan) 
    dividends_receivable_lf: np.double = Field(description="应收股利(最新一期)", default=np.nan) 
    other_receivables_lf: np.double = Field(description="其他应收款(最新一期)", default=np.nan) 
    other_receivables_sum_lf: np.double = Field(description="其他应收款合计(最新一期)", default=np.nan) 
    fin_assets_purchased_under_resale_lf: np.double = Field(description="买入返售金融资产(最新一期)", default=np.nan) 
    inventories_lf: np.double = Field(description="存货(最新一期)", default=np.nan) 
    contract_assets_lf: np.double = Field(description="合同资产(最新一期)", default=np.nan) 
    assets_held_for_sale_lf: np.double = Field(description="持有待售资产(最新一期)", default=np.nan) 
    noncurr_assets_due_within_1y_lf: np.double = Field(description="一年内到期的非流动资产(最新一期)", default=np.nan) 
    other_current_assets_lf: np.double = Field(description="其他流动资产(最新一期)", default=np.nan) 
    spec_diff_of_current_assets_lf: np.double = Field(description="流动资产差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_current_assets_lf: np.double = Field(description="流动资产差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_current_assets_lf: np.double = Field(description="流动资产合计(最新一期)", default=np.nan) 
    loans_and_advances_lf: np.double = Field(description="发放贷款及垫款(最新一期)", default=np.nan) 
    fin_assets_by_amortized_cost_lf: np.double = Field(description="以摊余成本计量的金融资产(最新一期)", default=np.nan) 
    fin_assets_by_fair_value_lf: np.double = Field(description="以公允价值计量且其变动计入其他综合收益的金融资产(最新一期)", default=np.nan) 
    available_for_sale_fin_assets_lf: np.double = Field(description="可供出售金融资产(最新一期)", default=np.nan) 
    held_to_maturity_invesments_lf: np.double = Field(description="持有至到期投资(最新一期)", default=np.nan) 
    debt_investments_lf: np.double = Field(description="债权投资(最新一期)", default=np.nan) 
    other_debt_investments_lf: np.double = Field(description="其他债权投资(最新一期)", default=np.nan) 
    longterm_receivables_lf: np.double = Field(description="长期应收款(最新一期)", default=np.nan) 
    longterm_equity_investments_lf: np.double = Field(description="长期股权投资(最新一期)", default=np.nan) 
    other_equity_investments_lf: np.double = Field(description="其他权益工具投资(最新一期)", default=np.nan) 
    other_noncurr_fin_assets_lf: np.double = Field(description="其他非流动金融资产(最新一期)", default=np.nan) 
    investment_property_lf: np.double = Field(description="投资性房地产(最新一期)", default=np.nan) 
    fixed_assets_lf: np.double = Field(description="固定资产(最新一期)", default=np.nan) 
    fixed_assets_sum_lf: np.double = Field(description="固定资产合计(最新一期)", default=np.nan) 
    construction_in_progress_lf: np.double = Field(description="在建工程(最新一期)", default=np.nan) 
    construction_in_progress_sum_lf: np.double = Field(description="在建工程合计(最新一期)", default=np.nan) 
    project_materials_lf: np.double = Field(description="工程物资(最新一期)", default=np.nan) 
    fixed_assets_disposal_lf: np.double = Field(description="固定资产清理(最新一期)", default=np.nan) 
    productive_biological_assets_lf: np.double = Field(description="生产性生物资产(最新一期)", default=np.nan) 
    oil_and_gas_assets_lf: np.double = Field(description="油气资产(最新一期)", default=np.nan) 
    right_of_use_assets_lf: np.double = Field(description="使用权资产(最新一期)", default=np.nan) 
    intangible_assets_lf: np.double = Field(description="无形资产(最新一期)", default=np.nan) 
    development_costs_lf: np.double = Field(description="开发支出(最新一期)", default=np.nan) 
    goodwill_lf: np.double = Field(description="商誉(最新一期)", default=np.nan) 
    longterm_prepaid_expense_lf: np.double = Field(description="长期待摊费用(最新一期)", default=np.nan) 
    deferred_tax_assets_lf: np.double = Field(description="递延所得税资产(最新一期)", default=np.nan) 
    other_noncurr_assets_lf: np.double = Field(description="其他非流动资产(最新一期)", default=np.nan) 
    spec_diff_of_noncurr_assets_lf: np.double = Field(description="非流动资产差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_noncurr_assets_lf: np.double = Field(description="非流动资产差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_noncurr_assets_lf: np.double = Field(description="非流动资产合计(最新一期)", default=np.nan) 
    spec_diff_of_total_assets_lf: np.double = Field(description="资产差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_total_assets_lf: np.double = Field(description="资产差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_assets_lf: np.double = Field(description="资产总计(最新一期)", default=np.nan) 
    shortterm_borrowings_lf: np.double = Field(description="短期借款(最新一期)", default=np.nan) 
    borrowing_from_central_bank_lf: np.double = Field(description="向中央银行借款(最新一期)", default=np.nan) 
    deposits_from_banks_and_fin_instiutions_lf: np.double = Field(description="吸收存款及同业存放(最新一期)", default=np.nan) 
    loans_from_banks_and_fin_institutions_lf: np.double = Field(description="拆入资金(最新一期)", default=np.nan) 
    tradable_fin_liabilities_lf: np.double = Field(description="交易性金融负债(最新一期)", default=np.nan) 
    derivatives_fin_liabilities_lf: np.double = Field(description="衍生金融负债(最新一期)", default=np.nan) 
    notes_payable_lf: np.double = Field(description="应付票据(最新一期)", default=np.nan) 
    accounts_payable_lf: np.double = Field(description="应付账款(最新一期)", default=np.nan) 
    notes_and_accounts_payable_lf: np.double = Field(description="应付票据及应付账款(最新一期)", default=np.nan) 
    advances_lf: np.double = Field(description="预收款项(最新一期)", default=np.nan) 
    contract_liabilities_lf: np.double = Field(description="合同负债(最新一期)", default=np.nan) 
    fin_assets_sold_under_resale_lf: np.double = Field(description="卖出回购金融资产款(最新一期)", default=np.nan) 
    fees_and_commissions_payable_lf: np.double = Field(description="应付手续费及佣金(最新一期)", default=np.nan) 
    employee_benefits_payable_lf: np.double = Field(description="应付职工薪酬(最新一期)", default=np.nan) 
    taxes_and_levies_payable_lf: np.double = Field(description="应交税费(最新一期)", default=np.nan) 
    interest_payable_lf: np.double = Field(description="应付利息(最新一期)", default=np.nan) 
    dividends_payable_lf: np.double = Field(description="应付股利(最新一期)", default=np.nan) 
    other_payables_lf: np.double = Field(description="其他应付款(最新一期)", default=np.nan) 
    other_payables_sum_lf: np.double = Field(description="其他应付款合计(最新一期)", default=np.nan) 
    reinsurance_payables_lf: np.double = Field(description="应付分保账款(最新一期)", default=np.nan) 
    insurance_contract_reserves_lf: np.double = Field(description="保险合同准备金(最新一期)", default=np.nan) 
    acting_trading_payables_lf: np.double = Field(description="代理买卖证券款(最新一期)", default=np.nan) 
    underwriting_payables_lf: np.double = Field(description="代理承销证券款(最新一期)", default=np.nan) 
    liabilities_held_for_sale_lf: np.double = Field(description="持有待售负债(最新一期)", default=np.nan) 
    noncurr_liabilities_due_within_1y_lf: np.double = Field(description="一年内到期的非流动负债(最新一期)", default=np.nan) 
    deferred_income_current_liabilities_lf: np.double = Field(description="递延收益-流动负债(最新一期)", default=np.nan) 
    shortterm_bonds_payable_lf: np.double = Field(description="应付短期债券(最新一期)", default=np.nan) 
    other_current_liabilities_lf: np.double = Field(description="其他流动负债(最新一期)", default=np.nan) 
    spec_diff_of_current_liabilities_lf: np.double = Field(description="流动负债差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_current_liabilities_lf: np.double = Field(description="流动负债差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_current_liabilities_lf: np.double = Field(description="流动负债合计(最新一期)", default=np.nan) 
    longterm_borrowings_lf: np.double = Field(description="长期借款(最新一期)", default=np.nan) 
    bonds_payable_lf: np.double = Field(description="应付债券(最新一期)", default=np.nan) 
    perpetual_bonds_lf: np.double = Field(description="永续债(最新一期)", default=np.nan) 
    preference_shares_lf: np.double = Field(description="优先股(最新一期)", default=np.nan) 
    lease_liabilities_lf: np.double = Field(description="租赁负债(最新一期)", default=np.nan) 
    longterm_payables_lf: np.double = Field(description="长期应付款(最新一期)", default=np.nan) 
    longterm_payables_sum_lf: np.double = Field(description="长期应付款合计(最新一期)", default=np.nan) 
    longterm_employee_benefits_lf: np.double = Field(description="长期应付职工薪酬(最新一期)", default=np.nan) 
    specific_payables_lf: np.double = Field(description="专项应付款(最新一期)", default=np.nan) 
    provisions_lf: np.double = Field(description="预计负债(最新一期)", default=np.nan) 
    deferred_tax_liabilities_lf: np.double = Field(description="递延所得税负债(最新一期)", default=np.nan) 
    deferred_income_noncurr_liabilities_lf: np.double = Field(description="递延收益-非流动负债(最新一期)", default=np.nan) 
    other_noncurr_liabilities_lf: np.double = Field(description="其他非流动负债(最新一期)", default=np.nan) 
    spec_diff_of_noncurr_liabilities_lf: np.double = Field(description="非流动负债差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_noncurr_liabilities_lf: np.double = Field(description="非流动负债差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_noncurr_liabilities_lf: np.double = Field(description="非流动负债合计(最新一期)", default=np.nan) 
    spec_diff_of_total_liabilities_lf: np.double = Field(description="负债差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_total_liabilities_lf: np.double = Field(description="负债差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_liabilities_lf: np.double = Field(description="负债合计(最新一期)", default=np.nan) 
    share_capital_lf: np.double = Field(description="实收资本(或股本)(最新一期)", default=np.nan) 
    capital_reserves_lf: np.double = Field(description="资本公积(最新一期)", default=np.nan) 
    treasury_shares_lf: np.double = Field(description="库存股(最新一期)", default=np.nan) 
    balance_othcom_income_lf: np.double = Field(description="其他综合收益(最新一期)", default=np.nan) 
    other_equity_instruments_lf: np.double = Field(description="其他权益工具(最新一期)", default=np.nan) 
    preference_of_other_equity_instruments_lf: np.double = Field(description="其中:优先股(最新一期)", default=np.nan) 
    specific_reserve_lf: np.double = Field(description="专项储备(最新一期)", default=np.nan) 
    surplus_reserve_lf: np.double = Field(description="盈余公积(最新一期)", default=np.nan) 
    general_reserve_lf: np.double = Field(description="一般风险准备(最新一期)", default=np.nan) 
    undistributed_profit_lf: np.double = Field(description="未分配利润(最新一期)", default=np.nan) 
    balance_translation_diff_of_foreign_currency_lf: np.double = Field(description="外币报表折算差额(最新一期)", default=np.nan) 
    total_equity_to_parent_shareholders_lf: np.double = Field(description="归属于母公司所有者权益合计(最新一期)", default=np.nan) 
    minority_interests_lf: np.double = Field(description="少数股东权益(最新一期)", default=np.nan) 
    spec_diff_of_shareholders_equity_lf: np.double = Field(description="股东权益差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_shareholders_equity_lf: np.double = Field(description="股权权益差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_owner_equity_lf: np.double = Field(description="所有者权益合计(最新一期)", default=np.nan) 
    spec_diff_of_liabilities_and_shareholder_equity_lf: np.double = Field(description="负债及股东权益差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_liabilities_and_shareholder_equity_lf: np.double = Field(description="负债及股东权益差额(合计平衡科目)(最新一期)", default=np.nan) 
    total_liabilities_and_owner_equity_lf: np.double = Field(description="负债和所有者权益总计(最新一期)", default=np.nan) 
    total_operating_revenue_lf: np.float64 = Field(description="营业总收入(最新一期)", default=np.nan)
    operating_revenue_lf: np.float64 = Field(description="营业收入(最新一期)", default=np.nan)
    interest_income_lf: np.float64 = Field(description="利息收入(最新一期)", default=np.nan)
    insurance_premium_income_lf: np.float64 = Field(description="已赚保费(最新一期)", default=np.nan)
    fee_and_commission_income_lf: np.float64 = Field(description="手续费及佣金收入(最新一期)", default=np.nan)
    spec_diff_of_operating_revenue_lf: np.float64 = Field(description="营业总收入差额(特殊报表科目)(最新一期)", default=np.nan)
    totbal_diff_of_operating_revenue_lf: np.float64 = Field(description="营业总收入差额(合计平衡科目)(最新一期)", default=np.nan)
    total_operating_costs_lf: np.float64 = Field(description="营业总成本(最新一期)", default=np.nan)
    operating_costs_lf: np.float64 = Field(description="营业成本(最新一期)", default=np.nan)
    interest_costs_lf: np.float64 = Field(description="利息支出(最新一期)", default=np.nan)
    fee_and_commission_costs_lf: np.float64 = Field(description="手续费及佣金支出(最新一期)", default=np.nan)
    surrenders_lf: np.float64 = Field(description="退保金(最新一期)", default=np.nan)
    net_insurance_claims_paid_lf: np.float64 = Field(description="赔付支出净额(最新一期)", default=np.nan)
    net_amount_of_insurance_reserve_lf: np.float64 = Field(description="提取保险合同准备金净额(最新一期)", default=np.nan)
    expense_on_policy_dividends_lf: np.float64 = Field(description="保单红利支出(最新一期)", default=np.nan)
    reinsurance_premium_expense_lf: np.float64 = Field(description="分保费用(最新一期)", default=np.nan)
    taxes_and_levies_lf: np.float64 = Field(description="税金及附加(最新一期)", default=np.nan)
    selling_epense_lf: np.float64 = Field(description="销售费用(最新一期)", default=np.nan)
    administrative_expense_lf: np.float64 = Field(description="管理费用(最新一期)", default=np.nan)
    research_and_development_expense_lf: np.float64 = Field(description="研发费用(最新一期)", default=np.nan)
    finance_expense_lf: np.float64 = Field(description="财务费用(最新一期)", default=np.nan)
    fin_interest_expense_lf: np.float64 = Field(description="财务费用：利息费用(最新一期)", default=np.nan)
    fin_interest_income_lf: np.float64 = Field(description="财务费用：利息收入(最新一期)", default=np.nan)
    asset_impairment_loss_lf: np.float64 = Field(description="资产减值损失(最新一期)", default=np.nan)
    credit_impairment_loss_lf: np.float64 = Field(description="信用减值损失(最新一期)", default=np.nan)
    spec_diff_of_operating_costs_lf: np.float64 = Field(description="营业总成本差额(特殊报表科目)(最新一期)", default=np.nan)
    totbal_diff_of_operating_costs_lf: np.float64 = Field(description="营业总成本差额(合计平衡科目)(最新一期)", default=np.nan)
    fair_value_chg_gain_lf: np.float64 = Field(description="公允价值变动收益(最新一期)", default=np.nan)
    invest_income_lf: np.float64 = Field(description="投资收益(最新一期)", default=np.nan)
    invest_income_of_jv_and_associates_lf: np.float64 = Field(description="对联营企业和合营企业的投资收益(最新一期)", default=np.nan)
    income_derecognition_of_fin_assets_at_amortized_cost_lf: np.float64 = Field(description="以摊余成本计量的金融资产终止确认收益(最新一期)", default=np.nan)
    net_income_of_open_hedge_lf: np.float64 = Field(description="净敞口套期收益(最新一期)", default=np.nan)
    exchange_gain_lf: np.float64 = Field(description="汇兑收益(最新一期)", default=np.nan)
    asset_disposal_income_lf: np.float64 = Field(description="资产处置收益(最新一期)", default=np.nan)
    other_income_lf: np.float64 = Field(description="其他收益(最新一期)", default=np.nan)
    spec_diff_of_operating_profit_lf: np.float64 = Field(description="营业利润差额(特殊报表科目)(最新一期)", default=np.nan)
    totbal_diff_of_operating_profit_lf: np.float64 = Field(description="营业利润差额(合计平衡科目)(最新一期)", default=np.nan)
    operating_profit_lf: np.float64 = Field(description="营业利润(最新一期)", default=np.nan)
    nonoperating_income_lf: np.float64 = Field(description="营业外收入(最新一期)", default=np.nan)
    noncurr_assets_dispose_gain_lf: np.float64 = Field(description="非流动资产处置利得(最新一期)", default=np.nan)
    nonoperating_costs_lf: np.float64 = Field(description="营业外支出(最新一期)", default=np.nan)
    noncurr_assets_dispose_loss_lf: np.float64 = Field(description="非流动资产处置损失(最新一期)", default=np.nan)
    spec_diff_of_total_profit_lf: np.float64 = Field(description="利润总额差额(特殊报表科目)(最新一期)", default=np.nan)
    spec_diff_of_total_profit_lf: np.float64 = Field(description="利润总额差额(特殊报表科目)(最新一期)", default=np.nan)
    totbal_diff_of_total_profit_lf: np.float64 = Field(description="利润总额差额(合计平衡科目)(最新一期)", default=np.nan)
    total_profit_lf: np.float64 = Field(description="利润总额(最新一期)", default=np.nan)
    income_tax_expense_lf: np.float64 = Field(description="所得税费用(最新一期)", default=np.nan)
    spec_diff_of_net_profit_lf: np.float64 = Field(description="净利润差(特殊报表科目)(最新一期)", default=np.nan)
    totbal_diff_of_net_profit_lf: np.float64 = Field(description="净利润差额(合计平衡科目)(最新一期)", default=np.nan)
    net_profit_lf: np.float64 = Field(description="净利润(最新一期)", default=np.nan)
    continuing_operation_net_profit_lf: np.float64 = Field(description="(一)持续经营净利润(最新一期)", default=np.nan)
    discontinued_operation_net_profit_lf: np.float64 = Field(description="(二)终止经营净利润(最新一期)", default=np.nan)
    net_profit_to_parent_shareholders_lf: np.float64 = Field(description="归属于母公司所有者的净利润(最新一期)", default=np.nan)
    net_profit_to_minority_lf: np.float64 = Field(description="少数股东损益(最新一期)", default=np.nan)
    eps_basic_lf: np.float64 = Field(description="基本每股收益(最新一期)", default=np.nan)
    eps_diluted_lf: np.float64 = Field(description="稀释每股收益(最新一期)", default=np.nan)
    income_othcom_income_lf: np.float64 = Field(description="其他综合收益(最新一期)", default=np.nan)
    othcom_income_to_parent_shareholders_lf: np.float64 = Field(description="归属母公司所有者的其他综合收益(最新一期)", default=np.nan)
    othcom_income_cannt_reclass_lf: np.float64 = Field(description="以后不能重分类进损益的其他综合收益(最新一期)", default=np.nan)
    chg_by_remeasurements_lf: np.float64 = Field(description="重新计量设定受益计划净负债或净资产的变动(最新一期)", default=np.nan)
    othcom_income_cannt_reclass_under_equity_method_lf: np.float64 = Field(description="权益法下在被投资单位不能重分类进损益的其他综合收益中享有的份额(最新一期)", default=np.nan)
    other_cannt_reclass_lf: np.float64 = Field(description="其他以后不能重分类进损益(最新一期)", default=np.nan)
    other_equity_instruments_fair_value_chg_lf: np.float64 = Field(description="其他权益工具投资公允价值变动(最新一期)", default=np.nan)
    own_credit_risk_fair_value_chg_lf: np.float64 = Field(description="企业自身信用风险公允价值变动(最新一期)", default=np.nan)
    othcom_income_reclass_lf: np.float64 = Field(description="以后将重分类进损益的其他综合收益(最新一期)", default=np.nan)
    othcom_income_reclass_under_equity_method_lf: np.float64 = Field(description="权益法下在被投资单位以后将重分类进损益的其他综合收益中享有的份额(最新一期)", default=np.nan)
    available_for_sale_fin_assets_fair_value_chg_lf: np.float64 = Field(description="可供出售金融资产公允价值变动损益(最新一期)", default=np.nan)
    gains_or_losses_from_htm_to_afs_lf: np.float64 = Field(description="持有至到期投资重分类为可供出售金融资产损益(最新一期)", default=np.nan)
    effective_of_gains_or_losses_on_cashflow_hedge_lf: np.float64 = Field(description="现金流量套期损益的有效部分(最新一期)", default=np.nan)
    income_translation_diff_of_foreign_currency_lf: np.float64 = Field(description="外币财务报表折算差额(最新一期)", default=np.nan)
    other_reclass_lf: np.float64 = Field(description="其他以后将重分类进损益(最新一期)", default=np.nan)
    other_debt_investments_fair_value_chg_lf: np.float64 = Field(description="其他债权投资公允价值变动(最新一期)", default=np.nan)
    othcom_income_from_reclass_of_fin_assets_lf: np.float64 = Field(description="金融资产重分类计入其他综合收益的金额(最新一期)", default=np.nan)
    credit_impairment_of_other_debt_investments_lf: np.float64 = Field(description="其他债权投资信用减值准备(最新一期)", default=np.nan)
    cashflow_hedge_reserve_lf: np.float64 = Field(description="现金流量套期储备(最新一期)", default=np.nan)
    othcom_income_to_minority_lf: np.float64 = Field(description="归属于少数股东的其他综合收益(最新一期)", default=np.nan)
    total_comprehensive_income_lf: np.float64 = Field(description="综合收益总额(最新一期)", default=np.nan)
    total_comprehensive_income_to_parent_shareholders_lf: np.float64 = Field(description="归属于母公司股东的综合收益总额(最新一期)", default=np.nan)
    cash_received_from_sales_and_services_lf: np.double = Field(description="销售商品、提供劳务收到的现金(最新一期)", default=np.nan) 
    netinc_in_deposits_lf: np.double = Field(description="客户存款和同业存放款项净增加额(最新一期)", default=np.nan) 
    netinc_in_borrowings_from_central_bank_lf: np.double = Field(description="向中央银行借款净增加额(最新一期)", default=np.nan) 
    netinc_in_loans_from_other_fin_institutions_lf: np.double = Field(description="向其他金融机构拆入资金净增加额(最新一期)", default=np.nan) 
    cash_received_from_premiums_lf: np.double = Field(description="收到原保险合同保费取得的现金(最新一期)", default=np.nan) 
    net_cash_received_from_reinsurance_lf: np.double = Field(description="收到再保业务现金净额(最新一期)", default=np.nan) 
    netinc_in_insurance_deposits_and_invest_lf: np.double = Field(description="保户储金及投资款净增加额(最新一期)", default=np.nan) 
    netinc_in_disposal_fin_assets_lf: np.double = Field(description="处置以公允价值计量且其变动计入当期损益的金融资产净增加额(最新一期)", default=np.nan) 
    cash_received_from_interests_fess_and_commissions_lf: np.double = Field(description="收取利息、手续费及佣金的现金(最新一期)", default=np.nan) 
    netinc_in_loans_from_banks_and_fin_institutions_lf: np.double = Field(description="拆入资金净增加额(最新一期)", default=np.nan) 
    netinc_in_repurchase_transactions_lf: np.double = Field(description="回购业务资金净增加额(最新一期)", default=np.nan) 
    taxes_and_levies_rebates_lf: np.double = Field(description="收到的税费返还(最新一期)", default=np.nan) 
    cash_received_from_other_operating_lf: np.double = Field(description="收到其他与经营活动有关的现金(最新一期)", default=np.nan) 
    spec_diff_of_cifoa_lf: np.double = Field(description="经营活动现金流入差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_cifoa_lf: np.double = Field(description="经营活动现金流入差额(合计平衡科目)(最新一期)", default=np.nan) 
    subtotal_cifoa_lf: np.double = Field(description="经营活动现金流入小计(最新一期)", default=np.nan) 
    cash_paid_for_goods_and_services_lf: np.double = Field(description="购买商品、接受劳务支付的现金(最新一期)", default=np.nan) 
    netinc_in_loans_and_advances_lf: np.double = Field(description="客户贷款及垫款净增加额(最新一期)", default=np.nan) 
    netinc_deposits_central_bank_interbank_lf: np.double = Field(description="存放中央银行和同业款项净增加额(最新一期)", default=np.nan) 
    cash_paid_for_claims_lf: np.double = Field(description="支付原保险合同赔付款项的现金(最新一期)", default=np.nan) 
    cash_paid_for_interests_fees_and_commissions_lf: np.double = Field(description="支付利息、手续费及佣金的现金(最新一期)", default=np.nan) 
    cash_paid_for_policy_dividends_lf: np.double = Field(description="支付保单红利的现金(最新一期)", default=np.nan) 
    cash_paid_for_employees_lf: np.double = Field(description="支付给职工以及为职工支付的现金(最新一期)", default=np.nan) 
    cash_paid_for_taxes_and_levies_lf: np.double = Field(description="支付的各项税费(最新一期)", default=np.nan) 
    other_cofoa_lf: np.double = Field(description="支付其他与经营活动有关的现金(最新一期)", default=np.nan) 
    spec_diff_of_cofoa_lf: np.double = Field(description="经营活动现金流出差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_cofoa_lf: np.double = Field(description="经营活动现金流出差额(合计平衡科目)(最新一期)", default=np.nan) 
    subtotal_cofoa_lf: np.double = Field(description="经营活动现金流出小计(最新一期)", default=np.nan) 
    spec_diff_of_net_cffoa_lf: np.double = Field(description="经营活动产生的现金流量净额差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_net_cffoa_lf: np.double = Field(description="经营活动产生的现金流量净额差额(合计平衡科目)(最新一期)", default=np.nan) 
    net_cffoa_lf: np.double = Field(description="经营活动产生的现金流量净额(最新一期)", default=np.nan) 
    cash_received_from_disposal_investments_lf: np.double = Field(description="收回投资收到的现金(最新一期)", default=np.nan) 
    return_on_investment_lf: np.double = Field(description="取得投资收益收到的现金(最新一期)", default=np.nan) 
    net_cash_received_from_disposal_filt_assets_lf: np.double = Field(description="处置固定资产、无形资产和其他长期资产收回的现金净额(最新一期)", default=np.nan) 
    net_cash_received_from_disposal_subsidiaries_lf: np.double = Field(description="处置子公司及其他营业单位收到的现金净额(最新一期)", default=np.nan) 
    cash_received_from_other_investing_lf: np.double = Field(description="收到其他与投资活动有关的现金(最新一期)", default=np.nan) 
    spec_diff_of_cifia_lf: np.double = Field(description="投资活动现金流入差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_cifia_lf: np.double = Field(description="投资活动现金流入差额(合计平衡科目)(最新一期)", default=np.nan) 
    subtotal_cifia_lf: np.double = Field(description="投资活动现金流入小计(最新一期)", default=np.nan) 
    cash_paid_for_filt_assets_lf: np.double = Field(description="购建固定资产、无形资产和其他长期资产支付的现金(最新一期)", default=np.nan) 
    cash_paid_for_investments_lf: np.double = Field(description="投资支付的现金(最新一期)", default=np.nan) 
    netinc_in_pledge_loans_lf: np.double = Field(description="质押贷款净增加额(最新一期)", default=np.nan) 
    cash_paid_by_acquiring_subsidiaries_lf: np.double = Field(description="取得子公司及其他营业单位支付的现金净额(最新一期)", default=np.nan) 
    cash_paid_for_other_investing_lf: np.double = Field(description="支付其他与投资活动有关的现金(最新一期)", default=np.nan) 
    spec_diff_of_cofia_lf: np.double = Field(description="投资活动现金流出差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_cofia_lf: np.double = Field(description="投资活动现金流出差额(合计平衡科目)(最新一期)", default=np.nan) 
    subtotal_of_cofia_lf: np.double = Field(description="投资活动现金流出小计(最新一期)", default=np.nan) 
    spec_diff_of_net_cffia_lf: np.double = Field(description="投资活动产生的现金流量净额差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_net_cffia_lf: np.double = Field(description="投资活动产生的现金流量净额差额(合计平衡科目)(最新一期)", default=np.nan) 
    net_cffia_lf: np.double = Field(description="投资活动产生的现金流量净额(最新一期)", default=np.nan) 
    capital_contributions_received_lf: np.double = Field(description="吸收投资收到的现金(最新一期)", default=np.nan) 
    cash_received_by_subsidiaries_from_minority_lf: np.double = Field(description="子公司吸收少数股东投资收到的现金(最新一期)", default=np.nan) 
    cash_received_from_borrowings_lf: np.double = Field(description="取得借款收到的现金(最新一期)", default=np.nan) 
    cash_received_from_bond_issue_lf: np.double = Field(description="发行债券收到的现金(最新一期)", default=np.nan) 
    cash_received_from_other_financing_lf: np.double = Field(description="收到其他与筹资活动有关的现金(最新一期)", default=np.nan) 
    spec_diff_of_ciffa_lf: np.double = Field(description="筹资活动现金流入差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_ciffa_lf: np.double = Field(description="筹资活动现金流入差额(合计平衡科目)(最新一期)", default=np.nan) 
    subtotal_ciffa_lf: np.double = Field(description="筹资活动现金流入小计(最新一期)", default=np.nan) 
    cash_paid_for_debt_repayment_lf: np.double = Field(description="偿还债务支付的现金(最新一期)", default=np.nan) 
    cash_paid_for_dividends_profits_interests_lf: np.double = Field(description="分配股利、利润或偿付利息支付的现金(最新一期)", default=np.nan) 
    cash_paid_by_subsidiaries_to_minority_lf: np.double = Field(description="子公司支付给少数股东的股利、利润(最新一期)", default=np.nan) 
    cash_paid_for_other_financing_lf: np.double = Field(description="支付其他与筹资活动有关的现金(最新一期)", default=np.nan) 
    spec_diff_of_coffa_lf: np.double = Field(description="筹资活动现金流出差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_coffa_lf: np.double = Field(description="筹资活动现金流出差额(合计平衡科目)(最新一期)", default=np.nan) 
    subtotal_of_coffa_lf: np.double = Field(description="筹资活动现金流出小计(最新一期)", default=np.nan) 
    spec_diff_of_net_cfffa_lf: np.double = Field(description="筹资活动产生的现金流量净额差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_net_cfffa_lf: np.double = Field(description="筹资活动产生的现金流量净额差额(合计平衡科目)(最新一期)", default=np.nan) 
    net_cfffa_lf: np.double = Field(description="筹资活动产生的现金流量净额(最新一期)", default=np.nan) 
    effect_of_exchange_chg_on_cce_lf: np.double = Field(description="汇率变动对现金及现金等价物的影响(最新一期)", default=np.nan) 
    spec_diff_of_netinc_in_cce_lf: np.double = Field(description="直接法-现金及现金等价物净增加额差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_netinc_in_cce_lf: np.double = Field(description="直接法-现金及现金等价物净增加额差额(合计平衡科目)(最新一期)", default=np.nan) 
    netinc_in_cce_lf: np.double = Field(description="现金及现金等价物净增加额(最新一期)", default=np.nan) 
    cce_beginning_lf: np.double = Field(description="期初现金及现金等价物余额(最新一期)", default=np.nan) 
    cce_ending_lf: np.double = Field(description="期末现金及现金等价物余额(最新一期)", default=np.nan) 
    net_profit_in_cashflow_sheet_lf: np.double = Field(description="现金流量表-净利润(最新一期)", default=np.nan) 
    asset_impairment_reserve_lf: np.double = Field(description="资产减值准备(最新一期)", default=np.nan) 
    depreciation_of_fa_oga_pba_lf: np.double = Field(description="固定资产折旧、油气资产折耗、生产性生物资产折旧(最新一期)", default=np.nan) 
    amorization_of_intangible_assets_lf: np.double = Field(description="无形资产摊销(最新一期)", default=np.nan) 
    amortization_of_longterm_deferred_expenses_lf: np.double = Field(description="长期待摊费用摊销(最新一期)", default=np.nan) 
    loss_from_disposal_of_fa_ia_lta_lf: np.double = Field(description="处置固定资产、无形资产和其他长期资产的损失(最新一期)", default=np.nan) 
    loss_from_scraping_of_fixed_assets_lf: np.double = Field(description="固定资产报废损失(最新一期)", default=np.nan) 
    loss_from_fair_value_chg_lf: np.double = Field(description="公允价值变动损失(最新一期)", default=np.nan) 
    finance_expenses_in_cashflow_sheet_lf: np.double = Field(description="现金流量表-财务费用(最新一期)", default=np.nan) 
    invest_loss_lf: np.double = Field(description="投资损失(最新一期)", default=np.nan) 
    decrease_in_deferred_tax_assets_lf: np.double = Field(description="递延所得税资产减少(最新一期)", default=np.nan) 
    increase_in_deferred_tax_liabilities_lf: np.double = Field(description="递延所得税负债增加(最新一期)", default=np.nan) 
    decrease_in_inventories_lf: np.double = Field(description="存货的减少(最新一期)", default=np.nan) 
    decrease_in_operating_receivables_lf: np.double = Field(description="经营性应收项目的减少(最新一期)", default=np.nan) 
    increase_in_operating_payables_lf: np.double = Field(description="经营性应付项目的增加(最新一期)", default=np.nan) 
    others_in_cashflow_sheet_lf: np.double = Field(description="其他(最新一期)", default=np.nan) 
    spec_diff_of_net_cffoa_indirect_lf: np.double = Field(description="间接法-经营活动现金流量净额差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_net_cffoa_indirect_lf: np.double = Field(description="间接法-经营活动现金流量净额差额(合计平衡科目)(最新一期)", default=np.nan) 
    net_cffoa_indirect_lf: np.double = Field(description="间接法-经营活动产生的现金流量净额(最新一期)", default=np.nan) 
    debt_transfer_to_capital_lf: np.double = Field(description="债务转为资本(最新一期)", default=np.nan) 
    conv_corp_bonds_within_1y_lf: np.double = Field(description="一年内到期的可转换公司债券(最新一期)", default=np.nan) 
    fin_lease_fixed_assets_lf: np.double = Field(description="融资租入固定资产(最新一期)", default=np.nan) 
    cash_balance_ending_lf: np.double = Field(description="现金的期末余额(最新一期)", default=np.nan) 
    cash_balance_beginning_lf: np.double = Field(description="现金的期初余额(最新一期)", default=np.nan) 
    cce_balance_ending_lf: np.double = Field(description="现金等价物的期末余额(最新一期)", default=np.nan) 
    cce_balance_beginning_lf: np.double = Field(description="现金等价物的期初余额(最新一期)", default=np.nan) 
    spec_diff_of_netinc_in_cce_indirect_lf: np.double = Field(description="间接法-现金及现金等价物净增加额差额(特殊报表科目)(最新一期)", default=np.nan) 
    totbal_diff_of_netinc_in_cce_indirect_lf: np.double = Field(description="间接法-现金及现金等价物净增加额差额(合计平衡科目)(最新一期)", default=np.nan) 
    netinc_in_cce_indirect_lf: np.double = Field(description="间接法-现金及现金等价物净增加额(最新一期)", default=np.nan) 
    credit_impairment_loss_in_cashflow_sheet_lf: np.double = Field(description="信用减值损失(最新一期)", default=np.nan) 

    class Config:
        arbitrary_types_allowed = True




**请注意：该表不是日频数据，不能直接获取作为因子数据。该数据适合对财务数据有深入研究的用户使用，请详细阅读后续文档！**

**若想获取日频财务因子，请使用 cn_stock_factors_financial_items 和 cn_stock_factors_financial_indicators 数据表**

* cn_stock_factors_financial_items: https://bigquant.com/data/datasources/cn_stock_factors_financial_items
* cn_stock_factors_financial_indicators: https://bigquant.com/data/datasources/cn_stock_factors_financial_indicators

# 一、数据简介

LF (last file) 指最新一期的财务数据。

* 数据起始时间：2005-01-01
* 数据更新频率：不定期，当上市公司发布相关公告时更新
* 表主键如下：

| 关键字 | 释意 |
| --- | --- |
| date | 指该财报的公布日期或者变更日期，与cn_stock_financial_changedate中的changedate对应 |
| instrument | 股票代码 |
| report_date | 财务报告期，规则见后文“财务通用知识” |
| shift | 偏移报告期，即站在历史节点t上，可以向前查看的偏移n期的财务数据 |


# 二、举例说明

### 1. 举例说明 LF 和 SHIFT 的概念

我们以 华润微（688396.SH）为例，该股票于 2020年2月27日正式在科创板上市，2020年04月23日首次披露财报

```python
import dai
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)

changedate_data = dai.query("""
SELECT *
FROM cn_stock_financial_changedate
WHERE instrument = '688396.SH'
AND report_date = '2020-03-31'
""",).df().sort_values(["changedate", "report_date"])

raw_data = dai.query("""
SELECT date, instrument, report_date, change_type, fs_quarter_index, net_profit
FROM cn_stock_financial_income_general_pit
WHERE instrument = '688396.SH'
""", filters={"date": ["2020-01-01", "2021-05-30"]}).df().sort_values(["date", "report_date"])

lf_data1 = dai.query("""
SELECT date, instrument, report_date, shift, net_profit_lf
FROM cn_stock_financial_lf_shift
WHERE instrument = '688396.SH'
AND date='2020-04-23'
""").df()

lf_data2 = dai.query("""
SELECT date, instrument, report_date, shift, net_profit_lf
FROM cn_stock_financial_lf_shift
WHERE instrument = '688396.SH'
AND date='2021-04-30'
""").df()
```

当你平台调用上述代码，会得到以数据：

* 下表展示了财务变更日期（changedate_data）：从下表数据看出，2020年一季报（report_date=2020-03-31）的财务数据最早于2020-04-23公布，2020-04-24对其进行更更正，同时在2021-04-30这一天公布2020年年报数据时，附带了2020年一季报的利润表和现金流量表数据.

{{cn_stock_financial_lf_shift_changedate}}

* 下表展示了原始财务数据（raw_data）：2020年一季报的数据在对应的changedate上都有一条记录，且只有第一条的change_type=1；在2021年4月30日时，上市公司同时公布了2020年年报和2021年一季报的财务报告。

{{cn_stock_financial_lf_shift_raw}}

* 下表展示了 2020-04-23 这一天的 LF 数据（lf_data1）：LF数据对原始数据进行了处理，比如下表中的这条数据只用公告日即date在2020-04-23之前的财务数据进行构建，由于这是该公司的第一份财报，所以只有一条数据。

{{cn_stock_financial_lf_shift_lf1}}

* 下表展示了 2021-04-30 这一天的 LF 数据（lf_data2）：当时间来到2021年4月30日时，上市公司已经公布了2021年一季报（report_date=2021-03-31），因此，我们在当日能获取从第一期财报（2020年一季报）到最新一期财报（2021年一季报）的所有相关财务数据。shift指偏移期，shift=0指当期，shift=1指上一期，shift=4为上年同期，以此类推。

{{cn_stock_financial_lf_shift_lf2}}

### **2. 如何利用衍生数据加工财务指标**

**利用这种shift数据，我们可以在历史时间节点 T 往前拿到所有财报期的财务数据，因此计算很多财务指标都是可行的了，比如：计算N年复合增长率，具体代码实现可以参考此链接的文档：https://bigquant.com/wiki/doc/n-KfcuAUffwy**



# 【通用知识——财务衍生】一、财务衍生基础

因为原始财务数据是记录性质的数据，只会在上市公司公布财报时有一定对应报告期的记录，记录对应财务数据的值，所以这类非规整的数据很难直接用于量化投资中。**因此我们基于原始财务数据进行了加工，这样对财务报表有深入研究的投资可以基于这些加工后财务衍生数据计算更加丰富的财务指标和因子。** 财务衍生是对原始财务数据的加工，加工成 LF、LY、MRQ、TTM 四类数据，因此财务衍生数据的字段都是在原始财务字段上加上对应的后缀。若要查询具体的字段，可以先去原始财报的文档中查询具体的财务字段，然后加上对应的后缀即可，原始财务数据表的相关知识可以查询下面各表的文档：

* 财报变更日期（cn_stock_financial_changedate）：[数据表文档链接点击这里](https://bigquant.com/data/datasources/cn_stock_financial_changedate "点击访问")
* 利润表（cn_stock_financial_income_general_pit）：[数据表文档链接点击这里](https://bigquant.com/data/datasources/cn_stock_financial_income_general_pit "点击访问")
* 资产负债表（cn_stock_financial_balance_general_pit）：[数据表文档链接点击这里](https://bigquant.com/data/datasources/cn_stock_financial_balance_general_pit "点击访问")
* 现金流量表（cn_stock_financial_cashflow_general_pit）：[数据表文档链接点击这里](https://bigquant.com/data/datasources/cn_stock_financial_cashflow_general_pit "点击访问")

在 BigQuant 平台上，我们通常用date 或者 changedate 表示财报变更日期，即上市公司披露财报的公告日；用report_date 表示报告期，即这一条数据来自来个报告期，分别有以下几个值：

* xxxx-03-31 表示xxxx年的一季报，比如：2024-03-31 指 2024年一季报。
* xxxx-06-30 表示xxxx年的半年报，比如：2024-06-30 指 2024年半年报。
* xxxx-09-30 表示xxxx年的三季报，比如：2024-09-30 指 2024年三季报。
* xxxx-12-31 表示xxxx年的年报，比如：2024-12-31 指 2024年年报。

接下来，着重介绍 **财务衍生数据表** 中会涉及的专业名词

# 【通用知识——财务衍生】二、专业术语

* **lf (last file)**: 指该表里面的数据取自最新一期财务数据，具体例子说明见 cn_stock_financial_lf_shift 表文档。

* **ly (last year)**: 指该表里面的数据取自最新一年报财务数据，具体例子说明见 cn_stock_financial_ly_shift 表文档。

* **mrq (most recent quarter)**: 指该表里面的数据计算的是最新一个单季度的财务数据，具体例子说明见 cn_stock_financial_mrq_shift 表文档。

* **ttm (trailing twelve months)**: 指该表里面的数据取自最新滚动十二月的财务数据，具体例子说明见 cn_stock_financial_ttm_shift 表文档。

* **shift (偏移期)**: 偏移期的概念会在每张表的文档中举例说明。

# 【通用知识——财务衍生】三、PIT (point in time)

在平台加工的“衍生数据”和“财务分析”两大类的财务数据中（这两大类数据的用法参见后文），我们同样进行PIT处理，我们以 002473.SZ 这只股票的流动资产（total_current_assets）数据举例：

```python
import dai
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)

dai.query("""
SELECT date, instrument, report_date, shift, total_current_assets_lf
FROM cn_stock_financial_lf_shift
WHERE instrument='002473.SZ'
AND shift < 3
AND ((date='2020-04-30')
OR (date='2020-08-27'))
""").df().sort_values("date")
```

通过上述代码，我们得到以下数据，可以看出，站在2020年4月30日这一天，我们能获取到 002473.SZ 的2019年年报的流动资产为 223449880.95，但站在2020-08-27这一天时，我们能获取到的2019年年报的流动资产为 228470428.90,其主要原因是该公司于2020年4月30日首次公布了2019年年报的财务数据，其披露的流动资产（total_current_assets）的值为223449880.95，但该公司又于2020年8月27日2019年年报的流动资产数据进行修正为228470428.90。

|    | date       | instrument   | report_date |   shift |   total_current_assets_lf |
|---:|:-----------|:-------------|:------------|--------:|--------------------------:|
|  0 | 2020-04-30 | 002473.SZ    | 2019-09-30  |       2 |              268546184.69 |
|  1 | 2020-04-30 | 002473.SZ    | 2019-12-31  |       1 |          **223449880.95** |
|  2 | 2020-04-30 | 002473.SZ    | 2020-03-31  |       0 |              213964081.16 |
|  3 | 2020-08-27 | 002473.SZ    | 2019-12-31  |       2 |          **228470428.90** |
|  4 | 2020-08-27 | 002473.SZ    | 2020-03-31  |       1 |              233220236.75 |
|  5 | 2020-08-27 | 002473.SZ    | 2020-06-30  |       0 |              222315509.19 |
