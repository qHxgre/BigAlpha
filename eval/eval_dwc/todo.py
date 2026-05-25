import numpy as np
import pandas as pd
import empyrical as ep
import structlog
from bigmodule import I
from pandas.api.types import is_integer_dtype

logger = structlog.get_logger()


def _annual_return(ret: pd.Series) -> float:
    """年化收益率"""
    ret = pd.Series(ret).dropna()
    if len(ret) < 2:
        return np.nan
    return float(ep.annual_return(ret, period="daily"))

def _sharpe(ret: pd.Series) -> float:
    """夏普比例"""
    ret = pd.Series(ret).dropna()
    if len(ret) < 3:
        return np.nan
    return float(ep.sharpe_ratio(ret.values, risk_free=0.035/242))

def _max_drawdown(ret: pd.Series) -> float:
    """最大回测"""
    ret = pd.Series(ret).dropna()
    if len(ret) < 3:
        return np.nan
    return float(ep.max_drawdown(ret))


def run(
    data: I.port("因子数据: 包含列 (date, instrument, factor) 的DataFrame/DataSource"),
    check_data: I.port("因子检查数据: 包含列 (date, instrument, factor) 的DataFrame/DataSource") = None,
    factor_field: I.str("因子列名,如果不指定则优先使用'factor'列,否则使用date和instrument外的列") = None,
    show: I.bool("画出绩效图") = True,
):
    import pandas as pd
    import dai

    # 检查date列
    if is_integer_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

    start_date = data['date'].min().strftime("%Y-%m-%d")
    end_date = data['date'].max().strftime("%Y-%m-%d")

    data = data.rename(columns={factor_field: 'factor'})
    if check_data is not None:
        check_data = check_data.rename(columns={factor_field: 'factor'})

    from .datachecker import DataCheck
    logger.info('========== 数据检查 ==========')
    dc = DataCheck(start_date, end_date, check_data=check_data)
    dc.validate(data)

    from .dataprocess import DataProcess
    logger.info('========== 数据预处理 ==========')
    dp = DataProcess()
    processed_data = dp.validate(data, factor_field)

    from .factoranalyze import FactorAnalyze
    logger.info('========== 单因子分析 ==========')
    sfa = FactorAnalyze(start_date, end_date)
    sfa_sharp = sfa.validate(processed_data, 'factor')
    sfa_sharp = round(sfa_sharp, 4)

    from .factorbacktest import FactorBacktest
    logger.info('========== 单因子回测 ==========')
    result = {}
    processed_data['cutoff_time'] = processed_data['date'].dt.strftime('%H%M')
    group_items = list(processed_data.groupby('cutoff_time'))
    total_groups = len(group_items)
    for idx, (cutoff_time, group_df) in enumerate(group_items, 1):
        logger.info(f'===>>> 正在处理第 {idx}/{total_groups} 个时间点: {cutoff_time}')
        backtest = FactorBacktest(start_date, end_date, group_df)
        backtest.run()

        # 投资组合的数据
        daily_portfolio = backtest.daily_portfolio.copy()
        daily_portfolio['cutoff_time'] = cutoff_time
        portfolio_trades = pd.DataFrame(backtest.trades)

        result[cutoff_time] = {
            'portfolio': daily_portfolio,
            'trades': portfolio_trades
        }
    
    # 统计分截断时间的数据
    df_list = []
    for cutoff_time, cutoff_data in result.items():
        df_list.append(cutoff_data['portfolio'])

    # 求16个投资组合的平均值
    portfolio_sum = pd.concat(df_list)
    avg_portfolio = portfolio_sum.groupby('trading_day')[['portfolio_ret', 'benchmark_ret', 'excess_ret']].mean()
    avg_portfolio['portfolio_cumret'] = (1+avg_portfolio['portfolio_ret']).cumprod()
    avg_portfolio['benchmark_cumret'] = (1+avg_portfolio['benchmark_ret']).cumprod()
    avg_portfolio['excess_cumret'] = (1+avg_portfolio['excess_ret']).cumprod()

    # 统计平均指标
    avg_excess_return = round(avg_portfolio['excess_cumret'].values[-1] - 1, 4)
    avg_annual_return = round(_annual_return(avg_portfolio['excess_ret']), 4)
    avg_sharp_ratio = round(_sharpe(avg_portfolio['excess_ret']), 4)
    avg_max_drawdown = round(_max_drawdown(avg_portfolio['excess_ret']), 4)

    if show is True:
        from .render import plot_sfa_group_ret, plot_avg_backtest
        from IPython.display import HTML

        c1 = plot_sfa_group_ret(sfa.groupcumret_pivotdata, sfa.group_num)
        c2 = plot_avg_backtest(avg_portfolio)

        # 创建HTML文件
        html_content = f"""
        <div>
            <h1>单因子分析 - 统计指标</h1>
            <h2> 多空组合的夏普比例 = {sfa_sharp} </h1>
            <br>

            <img src="data:image/png;base64,{c1}" alt="Plot 1">
            <br>


            <h1>单因子回测 - 统计指标</h1>
            <h2>16个组合平均超额收益 = {avg_excess_return}</h2>
            <h2>16个组合平均超额年化收益 = {avg_annual_return}</h2>
            <h2>16个组合平均超额夏普 = {avg_sharp_ratio}</h2>
            <h2>16个组合平均超额收益的最大回撤 = {avg_max_drawdown}</h2>

            <img src="data:image/png;base64,{c2}" alt="Plot 1">
            <br>
        </div>
        """
        render_html = HTML(html_content)
        from IPython.display import display
        display(render_html)


    score_data = pd.DataFrame({
        'sfa_sharp': [sfa_sharp],
        'backtest_excess': [avg_excess_return],
        'bacttest_sharp': [avg_sharp_ratio],
    })
    
    return dict(result=dai.DataSource.write_pickle(score_data), details=dai.DataSource.write_pickle(result))


def post_run(outputs):
    """后置运行函数"""
    return outputs
