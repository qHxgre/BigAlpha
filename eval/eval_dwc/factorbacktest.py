import dai
import logging
import structlog
import pandas as pd
import matplotlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
)
logger = structlog.get_logger()

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
warnings.filterwarnings('ignore')


class Position:
    """持仓类 - 优化为支持动态价格更新"""
    def __init__(self, symbol: str, shares: int, price: float):
        # 证券代码
        self.symbol = symbol

        # 持仓数量
        self.shares = shares

        # 买入成本（VWAP）
        self.cost_price = price

        # 当前价格
        self.current_price = price

        # 入场时间
        self.entry_time = None

    def update_price(self, current_price: float):
        """更新当前价格和PNL"""
        if current_price and current_price > 0:
            self.current_price = current_price


class Portfolio:
    """投资组合类"""
    def __init__(self, initial_capital: float = 10000000):
        # 静态信息
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.initial_capital = initial_capital
        self.total_value = initial_capital

        # 持仓历史记录
        self.positions_history = []
    
    def add_position_history(self, date: datetime, positions_count: int):
        """记录持仓历史"""
        self.positions_history.append({
            'date': date,
            'positions_count': positions_count,
            'total_value': self.total_value,
            'cash': self.cash
        })


class FactorBacktest:
    """15分钟频率因子回测引擎"""
    
    def __init__(self, start_date: str, end_date: str, signal_df: pd.DataFrame):
        self.start_date = start_date
        self.end_date = end_date
        self.signal_df = signal_df

        # 截断时间
        self.cutoff_time = self.signal_df['cutoff_time'].unique().tolist()[0]
        
        # 验证和预处理信号数据
        self._validate_and_preprocess_signals()
        
        # 回测参数
        self.initial_capital = 10000000
        self.transaction_cost = 0.0005  # 双边手续费0.05%
        self.top_n = 60  # 持仓数量
        self.min_trade_value = 10000  # 最小交易金额
        self.max_position_weight = 0.1  # 单票最大权重
        
        # 数据容器优化
        self.callback_points = []
        self.vwap_data = pd.DataFrame()
        self.close_data = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        self.daily_portfolio = pd.DataFrame()
        
        # 性能优化缓存
        self._price_cache = {}
        
        # 回测结果
        self.portfolio = Portfolio(self.initial_capital)
        self.results = []
        self.trades = []
        self.daily_summary = []
        
        # 初始化
        self._initialize()

    def _validate_and_preprocess_signals(self):
        """验证和预处理信号数据"""
        # 确保date列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.signal_df['date']):
            self.signal_df['date'] = pd.to_datetime(self.signal_df['date'])
        
        # 按时间排序
        self.signal_df = self.signal_df.sort_values(['date', 'instrument']).reset_index(drop=True)

    def _initialize(self):
        """初始化数据加载"""
        # 获取交易时点列表
        self.callback_points = self._get_callback_points()
        
        # 批量加载价格数据
        self._load_price_data_batch(self.signal_df['instrument'].unique().tolist())
        
        # 加载基准数据
        self._load_benchmark_data()
        
        # 投资组合
        self.results.append({
            'date': datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=1),
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'num_positions': 0,
        })

        logger.debug(f"[单因子回测] 初始化数据加载完成")

    def _get_callback_points(self) -> List[datetime]:
        """获取交易时点列表"""
        sql = "SELECT date FROM all_trading_days WHERE market_code='CN'"
        df = dai.query(sql, filters={'date': [self.start_date, self.end_date]}).df()
        # 扩展到15分钟时间点
        # time_segment = ['0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130', 
        #                 '1315', '1330', '1345', '1400', '1415', '1430', '1445', '1500']
        time_segment = [self.cutoff_time]
        all_timestamps = [
            pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {ts[:2]}:{ts[2:]}:00")
            for date in df['date']
            for ts in time_segment
        ]
        expanded_df = pd.DataFrame({'date': sorted(all_timestamps)})
        return sorted(pd.to_datetime(expanded_df['date']).dt.to_pydatetime())
    
    def _load_price_data_batch(self, stock_list: List[str]):
        """批量加载VWAP和close价格数据 - 优化查询性能"""
        sql = 'SELECT date, instrument, vwap, close FROM cpt_dwc_2026_stock_bar15m'
        df_all = dai.query(sql, filters={
            'date': [self.start_date, self.end_date],
            'instrument': stock_list
        }).df()
        
        # 转换为宽表格式以提高查询效率
        if not df_all.empty:
            # VWAP数据
            self.vwap_data = df_all.pivot_table(
                index='date', 
                columns='instrument', 
                values='vwap'
            )
            self.vwap_data = self.vwap_data.fillna(method='ffill')
            
            # Close数据
            self.close_data = df_all.pivot_table(
                index='date', 
                columns='instrument', 
                values='close'
            )
            self.close_data = self.close_data.fillna(method='ffill')
            
            # 确保索引为datetime类型
            self.vwap_data.index = pd.to_datetime(self.vwap_data.index)
            self.close_data.index = pd.to_datetime(self.close_data.index)
    
    def _load_benchmark_data(self):
        """加载基准指数数据 - 添加异常处理"""
        try:
            before_start_date = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')

            sql = """
            SELECT date, close / m_lag(close, 1) - 1 as benchmark_ret
            FROM cn_stock_index_bar1d 
            WHERE instrument = '000300.SH'
            """
            self.benchmark_data = dai.query(sql, filters={'date': [before_start_date, self.end_date]}).df()
            self.benchmark_data = self.benchmark_data[(self.benchmark_data['date']>=self.start_date)&((self.benchmark_data['date']<=self.end_date))]
        except Exception as e:
            error_msg = f"加载基准数据时出错: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _get_price(self, date: datetime, symbol: str, price_type: str = 'vwap') -> Optional[float]:
        """获取价格"""
        cache_key = (date, symbol, price_type)
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        try:
            if price_type == 'vwap':
                price_data = self.vwap_data
            else:
                price_data = self.close_data
            
            # 查找价格
            if date in price_data.index and symbol in price_data.columns:
                price = price_data.at[date, symbol]
                # 处理NaN和无效价格
                if pd.isna(price) or price <= 0:
                    price = None
                else:
                    price = float(price)
            else:
                price = None
            
            self._price_cache[cache_key] = price
            return price
            
        except Exception as e:
            logger.error(f"获取价格时出错 {date} {symbol}: {e}")
            return None
    
    def _get_valid_stocks(self, date: datetime) -> List[str]:
        """获取有价格数据的股票"""
        try:
            daily_signals = self.signal_df[self.signal_df['date'] == date]
            
            if daily_signals.empty:
                return []
            
            valid_stocks = []
            for symbol in daily_signals['instrument'].unique():
                vwap_price = self._get_price(date, symbol, 'vwap')
                close_price = self._get_price(date, symbol, 'close')
                
                if (vwap_price and close_price and 
                    vwap_price > 0 and close_price > 0 and
                    not pd.isna(vwap_price) and not pd.isna(close_price)
                ):
                    valid_stocks.append(symbol)
            
            return valid_stocks
            
        except Exception as e:
            logger.error(f"获取有效股票时出错 {date}: {e}")
            return []
    
    def _get_target_stocks(self, date: datetime) -> List[str]:
        """获取目标持仓股票 - 添加更多过滤条件"""
        try:
            daily_signals = self.signal_df[self.signal_df['date'] == date]
            
            if daily_signals.empty:
                return []
            
            # 过滤有效股票
            valid_stocks = self._get_valid_stocks(date)
            if not valid_stocks:
                return []
            
            valid_signals = daily_signals[daily_signals['instrument'].isin(valid_stocks)].copy()
            
            # 按因子值排序
            valid_signals = valid_signals.sort_values('factor', ascending=False)
            
            # 限制持仓数量
            top_stocks = valid_signals.head(min(self.top_n, len(valid_signals)))
            
            return top_stocks['instrument'].tolist()
            
        except Exception as e:
            logger.error(f"获取目标股票时出错 {date}: {e}")
            return []
    
    def _calculate_portfolio_value(self, date: datetime) -> float:
        """计算投资组合当前市值 - 添加缓存"""
        # 更新持仓市值
        positions_value = 0.0
        for symbol, position in self.portfolio.positions.items():
            close_price = self._get_price(date, symbol, 'close')
            if close_price:
                position.update_price(close_price)
                positions_value += position.shares * close_price
            else:
                # 如果无法获取价格，使用成本价
                positions_value += position.shares * position.cost_price
        
        total_value = self.portfolio.cash + positions_value
        self.portfolio.total_value = total_value
        return total_value
    
    def _execute_trade(self, date: datetime, symbol: str, target_shares: int):
        """执行交易"""
        vwap_price = self._get_price(date, symbol, 'vwap')
        if not vwap_price or vwap_price <= 0:
            logger.warnings(f"警告: {date} {symbol} 价格无效: {vwap_price}")
            return
        
        current_position = self.portfolio.positions.get(symbol)
        current_shares = current_position.shares if current_position else 0
        
        # 计算需要交易的股数
        trade_shares = target_shares - current_shares

        if trade_shares == 0:
            return
        
        # 计算交易金额
        trade_amount = abs(trade_shares) * vwap_price
        
        # 检查最小交易金额
        if trade_amount < self.min_trade_value:
            return
        
        # 计算交易成本
        transaction_cost = trade_amount * self.transaction_cost
        
        if trade_shares > 0:  # 买入
            # 检查现金是否足够
            total_cost = trade_amount + transaction_cost
            
            # 检查单票权重限制
            portfolio_value = self._calculate_portfolio_value(date)
            max_position_value = portfolio_value * self.max_position_weight
            
            if total_cost > self.portfolio.cash:
                # 调整买入数量
                max_buy_amount = min(self.portfolio.cash, max_position_value) / (1 + self.transaction_cost)
                trade_shares = int(max_buy_amount / vwap_price)
                trade_amount = trade_shares * vwap_price
                transaction_cost = trade_amount * self.transaction_cost
                total_cost = trade_amount + transaction_cost
            
            if trade_shares > 0:
                # 更新持仓
                if symbol in self.portfolio.positions:
                    old_position = self.portfolio.positions[symbol]
                    total_shares = old_position.shares + trade_shares
                    total_cost_value = (old_position.shares * old_position.cost_price + 
                                       trade_shares * vwap_price)
                    avg_cost = total_cost_value / total_shares
                    
                    self.portfolio.positions[symbol].shares = total_shares
                    self.portfolio.positions[symbol].cost_price = avg_cost
                    self.portfolio.positions[symbol].entry_time = date
                else:
                    self.portfolio.positions[symbol] = Position(symbol, trade_shares, vwap_price)
                    self.portfolio.positions[symbol].entry_time = date
                
                # 更新现金
                self.portfolio.cash -= total_cost
                
                # 记录交易
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'type': 'BUY',
                    'shares': trade_shares,
                    'price': vwap_price,
                    'amount': trade_amount,
                    'cost': transaction_cost,
                    'cash_after': self.portfolio.cash
                })
        
        else:  # 卖出
            trade_shares = abs(trade_shares)
            
            # 确保不会卖出超过持仓的数量
            trade_shares = min(trade_shares, current_shares)
            
            sell_amount = trade_shares * vwap_price
            transaction_cost = sell_amount * self.transaction_cost
            net_proceeds = sell_amount - transaction_cost
            
            # 更新持仓
            if symbol in self.portfolio.positions:
                self.portfolio.positions[symbol].shares -= trade_shares
                
                # 如果持仓为0，移除
                if self.portfolio.positions[symbol].shares <= 0:
                    del self.portfolio.positions[symbol]
            
            # 更新现金
            self.portfolio.cash += net_proceeds
            
            # 记录交易
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'type': 'SELL',
                'shares': trade_shares,
                'price': vwap_price,
                'amount': sell_amount,
                'cost': transaction_cost,
                'cash_after': self.portfolio.cash,
                'pnl': trade_shares * (vwap_price - current_position.cost_price) if current_position else 0
            })
    
    def _rebalance_portfolio(self, date: datetime):
        """调仓逻辑"""
        # 获取目标持仓股票
        target_stocks = self._get_target_stocks(date)
        
        if not target_stocks:
            return
        
        # 计算当前组合市值
        portfolio_value = self._calculate_portfolio_value(date)
        
        # 等权重配置
        target_value_per_stock = portfolio_value / len(target_stocks)
        
        # 计算目标持仓
        target_positions = {}
        for symbol in target_stocks:
            vwap_price = self._get_price(date, symbol, 'vwap')
            if vwap_price and vwap_price > 0:
                target_shares = int(target_value_per_stock / vwap_price)
                if target_shares > 0:
                    target_positions[symbol] = target_shares
        
        # 执行交易
        # 先卖出不在目标持仓中的股票
        current_symbols = list(self.portfolio.positions.keys())
        for symbol in current_symbols:
            if symbol not in target_positions:
                self._execute_trade(date, symbol, 0)
        
        # 再调整目标持仓中的股票
        for symbol, target_shares in target_positions.items():
            self._execute_trade(date, symbol, target_shares)
    
    def run(self) -> Dict:
        """运行回测"""
        sd = datetime.now()

        for i, date in enumerate(self.callback_points):
            # 显示进度
            if i % 100 == 0 or i == len(self.callback_points) - 1:
                logger.debug(f"进度: {i+1}/{len(self.callback_points)} ({date})")
            
            # 调仓
            self._rebalance_portfolio(date)
            
            # 计算当前投资组合的总资产
            portfolio_value = self._calculate_portfolio_value(date)

            # 记录持仓历史
            self.portfolio.add_position_history(date, len(self.portfolio.positions))
            
            # 记录结果
            self.results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'num_positions': len(self.portfolio.positions),
            })
        
        # 投资组合统计数据
        self.daily_portfolio = self.calc_portfolio_ret()

        ed = datetime.now()
        logger.debug(f"[单因子回测] 完成回测: {self.start_date} 到 {self.end_date}, 总共 {len(self.callback_points)} 个调仓时点, 耗时: {round((ed-sd).total_seconds(), 4)} 秒")


    def calc_portfolio_ret(self) -> pd.DataFrame:
        """计算投资组合的收益"""
        # 投资组合收益率
        results = pd.DataFrame(self.results).sort_values('date')
        results['trading_day'] = pd.to_datetime(results['date'].dt.strftime('%Y-%m-%d'))
        daily_df = results.groupby('trading_day').agg('last').reset_index()
        daily_df['portfolio_ret'] = daily_df['portfolio_value'].pct_change()

        # 指数收益率
        benchmark_df = self.benchmark_data.rename(columns={'date': 'trading_day'})

        # 超额收益率
        daily_df = pd.merge(daily_df, benchmark_df, how='left', on=['trading_day'])
        df = daily_df[['trading_day', 'portfolio_ret', 'benchmark_ret']].fillna(0)
        df['excess_ret'] = df['portfolio_ret'] - df['benchmark_ret']

        # 累计收益率
        df['portfolio_cumret'] = (1+df['portfolio_ret']).cumprod()
        df['benchmark_cumret'] = (1+df['benchmark_ret']).cumprod()
        df['excess_cumret'] = (1+df['excess_ret']).cumprod()

        return df
