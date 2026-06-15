import numpy as np
import pandas as pd
from pydantic import Field
from base import BaseSchema

class Bigalpha2026FactorlibSchema(BaseSchema):
    """精选因子库 Schema

    从 预计算因子全集 (~6700 字段) 中精选的 37 个核心因子, 覆盖
    量价、估值、技术指标、财务、资金流、风险与基本信息 6 个维度,
    用于构建 20-50 规模的轻量因子库。字段定义 (类型/描述/默认值) 与
    源 schema.py 保持一致。
    """

    date: np.datetime64 = Field(description="日期", default=np.nan, group="")
    instrument: pd.StringDtype = Field(description="证券代码", default=np.nan, group="")

    # ---- 量价因子 (9) ----
    close: np.double = Field(description='收盘价（后复权）, SQL 算子: cn_stock_bar1d.close', default=0, group='量价因子', free=True)
    volume: np.int32 = Field(description='成交量, SQL 算子: cn_stock_bar1d.volume', default=0, group='量价因子', free=True)
    amount: np.double = Field(description='成交金额, SQL 算子: cn_stock_bar1d.amount', default=0, group='量价因子', free=True)
    turn: np.double = Field(description='换手率, SQL 算子: cn_stock_bar1d.turn', default=0, group='量价因子', free=True)
    change_ratio: np.double = Field(description='涨跌幅（后复权）, SQL 算子: cn_stock_bar1d.change_ratio', default=0, group='量价因子', free=True)
    daily_return: np.double = Field(description='日收益率, SQL 算子: cn_stock_bar1d.close / m_lag(cn_stock_bar1d.close, 1) - 1', default=0, group='量价因子', free=True)
    momentum_5: np.double = Field(description='5日动量, 涉及窗口函数，建议向前取5日, SQL 算子: cn_stock_bar1d.close / m_lag(cn_stock_bar1d.close, 5) - 1，其他周期的因子只需将 m_lag(close, N) 中的N进行替换', default=0, group='量价因子', free=True)
    reversal_5: np.double = Field(description='5日反转, 涉及窗口函数，建议向前取N日, SQL 算子: momentum_5 * -1，其他周期的因子只需对对应周期的动量因子取反', default=0, group='量价因子', free=True)
    volatility_5: np.double = Field(description='5日波动率, 涉及窗口函数，建议向前取N日, SQL 算子: m_nanstd(daily_return, 5)，其他周期的因子只需将 m_nanstd(daily_return, 5) 中的N进行替换', default=0, group='量价因子', free=True)

    # ---- 估值因子 (5) ----
    total_market_cap: np.double = Field(description='总市值, 公式=当日收盘价*当日总股本', default=0, group='估值因子', free=True)
    float_market_cap: np.double = Field(description='流通市值, 公式=当日收盘价*当日总股本', default=0, group='估值因子', free=True)
    pe_ttm: np.double = Field(description='市盈率TTM, 公式=当日总市值/归母净利润TTM', default=0, group='估值因子', visible=True, free=True)
    pb: np.double = Field(description='市净率, 公式=当日总市值/最新一期所有者权益', default=0, group='估值因子', visible=True, free=True)
    ps_ttm: np.double = Field(description='市销率TTM, 公式=当日总市值/营业总收入TTM', default=0, group='估值因子', visible=True, free=True)

    # ---- 技术指标 (11) ----
    sma_20: np.double = Field(description='简单移动平均线。指标解释：计算n1个时间周期内收盘价的平均值。SQL 算子：m_ta_sma(cn_stock_bar1d.close, 20)', default=np.nan, group='技术指标', visible=True, free=True)
    ema_20: np.double = Field(description='指数移动平均线。指标解释：对n1个时间周期内收盘价通过指数加权平均的方式计算的均价，使其对近期价格变化更敏感。SQL 算子：m_ta_ema(cn_stock_bar1d.close, 20)', default=np.nan, group='技术指标', visible=True, free=True)
    macd_diff_12_26_9: np.double = Field(description='MACD的DIF线。指标解释：MACD指标分为三个子指标。DIFF是收盘价快周期(n1)均线减去慢周期(n2)均线，DEA线是DIFF线的n3日的指数平滑移动平均线, HIST值是DIFF线与DEA线的差值。SQL 算子：m_ta_macd_dif(cn_stock_bar1d.close, fastperiod:=12, slowperiod:=26, signalperiod:=9)', default=np.nan, group='技术指标', visible=True, free=True)
    macd_dea_12_26_9: np.double = Field(description='MACD的DEA线。指标解释：MACD指标分为三个子指标。DIFF是收盘价快周期(n1)均线减去慢周期(n2)均线，DEA线是DIFF线的n3日的指数平滑移动平均线, HIST值是DIFF线与DEA线的差值。SQL 算子：m_ta_macd_dea(cn_stock_bar1d.close, fastperiod:=12, slowperiod:=26, signalperiod:=9)', default=np.nan, group='技术指标', visible=True, free=True)
    macd_hist_12_26_9: np.double = Field(description='MACD的HIST值。指标解释：MACD指标分为三个子指标。DIFF是收盘价快周期(n1)均线减去慢周期(n2)均线，DEA线是DIFF线的n3日的指数平滑移动平均线, HIST值是DIFF线与DEA线的差值。SQL 算子：m_ta_macd_hist(cn_stock_bar1d.close, fastperiod:=12, slowperiod:=26, signalperiod:=9)', default=np.nan, group='技术指标', visible=True, free=True)
    rsi_12: np.double = Field(description='相对强弱指数。指标解释：通过比较n1周期内价格上涨的天数与价格下跌的天数来衡量资产的内在强度，数值范围在0～100。。SQL 算子：m_ta_rsi(cn_stock_bar1d.close, 12)', default=np.nan, group='技术指标', visible=True, free=True)
    kdj_k_9_3_3: np.double = Field(description='KDJ的K值。指标解释：KDJ分为四个子指标。RSV指未成熟随机值，公式=(收盘价-n1日内最低价)/(n1日内最高价-n1日内最低价)；K值是RSV的n2日移动平均值；D值是K值的n3日的移动平均值；J值是对K和D的进一步派生，公式=3*K-2*D。SQL 算子：m_ta_kdj_k(cn_stock_bar1d.high, cn_stock_bar1d.low, cn_stock_bar1d.close, fastk_period:=9, slowk_period:=3, slowd_period:=3)', default=np.nan, group='技术指标', visible=True, free=True)
    kdj_d_9_3_3: np.double = Field(description='KDJ的D值。指标解释：KDJ分为四个子指标。RSV指未成熟随机值，公式=(收盘价-n1日内最低价)/(n1日内最高价-n1日内最低价)；K值是RSV的n2日移动平均值；D值是K值的n3日的移动平均值；J值是对K和D的进一步派生，公式=3*K-2*D。SQL 算子：m_ta_kdj_d(cn_stock_bar1d.high, cn_stock_bar1d.low, cn_stock_bar1d.close, fastk_period:=9, slowk_period:=3, slowd_period:=3)', default=np.nan, group='技术指标', visible=True, free=True)
    bias_20: np.double = Field(description='乖离率。指标解释：衡量当前价格与某一平均价格间偏离程度的技术指标，当前价格与最近n1期的平均价格差异。SQL 算子：m_ta_bias(cn_stock_bar1d.close, 20)', default=np.nan, group='技术指标', visible=True, free=True)
    cci_14: np.double = Field(description='商品通道指数。指标解释：计算方法: TP=(最高价+最低价+收盘价)÷3，CCI=(TP-TP的n1日均值)/(0.0015*TP的n1日平均误差)。SQL 算子：m_ta_cci(cn_stock_bar1d.high, cn_stock_bar1d.low, cn_stock_bar1d.close, 14)', default=np.nan, group='技术指标', visible=True, free=True)
    atr_14: np.double = Field(description='平均真实波动率。指标解释：首先计算每一根K线的真实波动幅度，它是以下三者中的最大值：当日最高价与最低价的差值；当日最高价与前一日收盘价的绝对值差；当日最低价与前一日收盘价的绝对值差。再计算 n1 周期内TR值的平均值，即为ATR。。SQL 算子：m_ta_atr(cn_stock_bar1d.high, cn_stock_bar1d.low, cn_stock_bar1d.close, 14)', default=np.nan, group='技术指标', visible=True, free=True)

    # ---- 财务因子 (6) ----
    roe_avg_ttm: np.double = Field(description='净资产收益率(平均)(滚动十二期)', default=np.nan, group='财务因子', visible=True, free=True)
    roa_avg_ttm: np.double = Field(description='总资产净利率(平均)(滚动十二期)', default=np.nan, group='财务因子', visible=True, free=True)
    gross_profit_rate_ttm: np.double = Field(description='销售毛利率(滚动十二期)', default=np.nan, group='财务因子', visible=True, free=True)
    net_profit_rate_ttm: np.double = Field(description='销售净利率(滚动十二期)', default=np.nan, group='财务因子', visible=True, free=True)
    debt_to_asset_lf: np.double = Field(description='资产负债率(最新一期)', default=np.nan, group='财务因子', visible=True, free=False)
    current_ratio_lf: np.double = Field(description='流动比率(最新一期)', default=np.nan, group='财务因子', visible=True, free=False)

    # ---- 资金流 (3) ----
    netflow_amount_main: np.double = Field(description='净流入额(主力)=流入额(主力)-流出额(主力)', default=0, group='资金流', visible=True, free=False)
    netflow_amount_rate_main: np.double = Field(description='资金净流入成交额率(主力) = 资金净流入额(主力) / 当日总成交额', default=0, group='资金流', visible=True, free=False)
    net_active_buy_amount_main: np.double = Field(description='净主动买入额(主力)=主动买入额(主力)-主动卖出额(主力)', default=0, group='资金流', visible=True, free=False)

    # ---- 风险/基本信息 (2) ----
    beta_000300SH_22: np.double = Field(description='沪深300指数的22日BETA系数, SQL 算子: m_regr_slope(个股收益率, 指数收益率, N), 因为该算子涉及窗口函数，所以前N天无法算出该因子', default=0, group='指数相关', free=True)
    list_days: np.int64 = Field(description='已上市天数 (按自然日), SQL 算子: day(date - cn_stock_basic_info.list_date)', default=0, group='基本信息', free=True)

    class Config:
        arbitrary_types_allowed = True
