"""
Strategy Tester for Trading Algorithm Research.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from tqdm import tqdm

# Add the current directory to Python path to ensure imports work
# This works for script execution, Jupyter notebooks, and imported modules
try:
    # Try to get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # If __file__ is not available (some Jupyter environments), use current working directory
    current_dir = os.getcwd()

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from strategies.base_strategy import BaseStrategy, PositionType
import config


class StrategyTester:
    """
    Strategy testing framework for backtesting trading strategies.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the strategy tester.
        
        Args:
            initial_capital: Starting capital for backtests
            commission_rate: Commission rate as decimal
            slippage: Slippage rate as decimal
        """
        self.initial_capital = initial_capital
        self.test_results = {}
        

    
    def list_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return ['SimpleMAStrategy', 'FibonacciChannelStrategy']
    
    def load_strategy(self, strategy_name: str, **params) -> BaseStrategy:
        """Load a strategy instance with given parameters."""
        # Ensure the current directory is in the path for imports
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Set default config values if not provided
        default_params = {
            'initial_capital': getattr(config, 'INITIAL_CAPITAL', 10000.0),
            'maker_fee': getattr(config, 'MAKER_FEE', 0.0002),
            'taker_fee': getattr(config, 'TAKER_FEE', 0.00055),
            'commission_rate': getattr(config, 'COMMISSION_RATE', 0.001),  # Legacy fallback
            'slippage': getattr(config, 'SLIPPAGE', 0.0005)
        }
        
        # Merge user params with defaults (user params override defaults)
        final_params = {**default_params, **params}
            
        if strategy_name == 'SimpleMAStrategy':
            from strategies.simple_ma_strategy import SimpleMAStrategy
            return SimpleMAStrategy(**final_params)
        elif strategy_name == 'FibonacciChannelStrategy':
            from strategies.fibnacci_stratergy import FibonacciChannelStrategy
            return FibonacciChannelStrategy(**final_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def backtest_strategy(self, strategy: BaseStrategy, data: pd.DataFrame,
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Run a complete backtest of a strategy with pandas DataFrame.
        
        Args:
            strategy: Strategy instance to test
            data: Pandas DataFrame with OHLCV data. Must have columns: 
                  ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing backtest results and metrics
            
        Example:
            >>> import pandas as pd
            >>> # Load your data into a DataFrame
            >>> data = pd.read_csv('your_data.csv')
            >>> strategy = tester.load_strategy('SimpleMAStrategy', fast_period=10, slow_period=20)
            >>> results = tester.backtest_strategy(strategy, data, verbose=True)
        """
        if verbose:
            print(f"Starting backtest for {strategy.name}")
            print(f"Data period: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"Data points: {len(data)}")
        
        # Reset strategy state
        strategy.reset()
        
        # Generate signals
        if verbose:
            print("Generating trading signals...")
        
        signals_data = strategy.generate_signals(data)
        if verbose:
            print("Signals — long:", int(signals_data["long_signal"].sum()),
                "short:", int(signals_data["short_signal"].sum()))
            # show first few bars that try to enter
            first_longs  = signals_data.index[signals_data["long_signal"]].tolist()[:3]
            first_shorts = signals_data.index[signals_data["short_signal"]].tolist()[:3]
            print("First long idx:", first_longs, "First short idx:", first_shorts)
        # Run backtest
        if verbose:
            print("Running backtest simulation...")
        
        # Create progress bar for the backtest loop
        progress_bar = tqdm(signals_data.iterrows(), 
                           total=len(signals_data),
                           desc="Backtesting",
                           unit="bars",
                           disable=not verbose)
        
        for i, (timestamp, row) in enumerate(progress_bar):
            current_price = row['close']
            
            # Get the actual timestamp from the row data, not the index
            actual_timestamp = row['timestamp'] if 'timestamp' in row else pd.Timestamp.now()
            # Ensure timestamp is a pandas Timestamp object
            actual_timestamp = pd.to_datetime(actual_timestamp)
            
            # Stop-loss & TPs (row, not price)
            if hasattr(strategy, 'check_stop_loss') and strategy.position != PositionType.FLAT:
                if strategy.check_stop_loss(row, actual_timestamp):
                    continue

            if hasattr(strategy, 'check_take_profits') and strategy.position != PositionType.FLAT:
                strategy.check_take_profits(row, actual_timestamp)

            # Entries -> size with the same row (so fib SL is correct)
            if strategy.position == PositionType.FLAT:
                if strategy.should_enter_long(row, signals_data):
                    size = (strategy.calculate_position_size_for_direction(current_price, True,  signals_row=row)
                            if hasattr(strategy, 'calculate_position_size_for_direction')
                            else strategy.calculate_position_size(current_price))
                    if size > 0:
                        strategy.enter_long(actual_timestamp, current_price, size)

                elif strategy.should_enter_short(row, signals_data):
                    size = (strategy.calculate_position_size_for_direction(current_price, False, signals_row=row)
                            if hasattr(strategy, 'calculate_position_size_for_direction')
                            else strategy.calculate_position_size(current_price))
                    if size > 0:
                        strategy.enter_short(actual_timestamp, current_price, size)
            
            elif strategy.position == PositionType.LONG:
                # Check for exit signals
                if strategy.should_exit_long(row, signals_data):
                    strategy.exit_long(actual_timestamp, current_price)
            
            elif strategy.position == PositionType.SHORT:
                # Check for exit signals
                if strategy.should_exit_short(row, signals_data):
                    strategy.exit_short(actual_timestamp, current_price)
            
            # Update progress bar with current info every 100 iterations
            if verbose and i % 100 == 0:
                progress_bar.set_postfix({
                    'Trades': len(strategy.trades),
                    'Capital': f"${strategy.current_capital:,.0f}",
                    'Position': strategy.position.name
                })
            # Track per-bar equity (capital + unrealized PnL)
            if strategy.position == PositionType.LONG:
                unreal = (row['close'] - strategy.entry_price) * strategy.current_size
            elif strategy.position == PositionType.SHORT:
                unreal = (strategy.entry_price - row['close']) * strategy.current_size
            else:
                unreal = 0.0
            strategy_bar_equity = strategy.current_capital + unreal
            # store it somewhere accessible, e.g. on the strategy:
            if not hasattr(strategy, 'bar_equity'):
                strategy.bar_equity = []
            strategy.bar_equity.append((actual_timestamp, strategy_bar_equity))
        
        # Close any open positions at the end
        if strategy.position != PositionType.FLAT:
            final_price = data.iloc[-1]['close']
            final_timestamp = pd.to_datetime(data.iloc[-1]['timestamp'])
            if strategy.position == PositionType.LONG:
                strategy.exit_long(final_timestamp, final_price)
            else:
                strategy.exit_short(final_timestamp, final_price)
            
            if verbose:
                print(f"Closed final position at {final_price:.2f}")
        
        # Close the progress bar and add final update
        if verbose:
            progress_bar.set_postfix({
                'Trades': len(strategy.trades),
                'Capital': f"${strategy.current_capital:,.0f}",
                'Status': 'Complete'
            })
        progress_bar.close()
        
        if verbose:
            print("Calculating performance metrics...")
        
        metrics = strategy.get_performance_metrics()
        
        # Store results
        result = {
            'strategy_name': strategy.name,
            'metrics': metrics,
            'trades': strategy.trades,
            'equity_curve': strategy.equity_curve,
            'original_data': data.copy(),  # Store original data for plotting
            'data_period': {
                'start': data['timestamp'].min(),
                'end': data['timestamp'].max(),
                'points': len(data)
            }
        }
        
        self.test_results[strategy.name] = result
        
        if verbose:
            print(f"\n🎯 BACKTEST COMPLETED SUCCESSFULLY!")
            print(f"   Strategy: {strategy.name}")
            print(f"   Total Trades: {metrics['total_trades']}")
            print(f"   Final Return: {metrics['total_return']:.2%}")
            print(f"   Final Capital: ${metrics['final_capital']:,.2f}")
            print(f"   Win Rate: {metrics['win_rate']:.2%}")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Risk:Reward: {metrics['risk_reward_ratio']:.2f}")
            print(f"   Market Exposure: {metrics['exposure_percentage']:.2f}%")
            print(f"   Total Fees: ${metrics['total_fees']:,.2f}")
            print(f"\n💡 Use display_comprehensive_metrics('{strategy.name}') for detailed analysis")
        
        return result
    
    def test_strategy_from_data(self, strategy_name: str, data: pd.DataFrame,
                               **strategy_params) -> Dict[str, Any]:
        """
        Test a strategy by name with given pandas DataFrame.
        
        Args:
            strategy_name: Name of the strategy to test ('SimpleMAStrategy' or 'FibonacciChannelStrategy')
            data: Pandas DataFrame with OHLCV data. Must have columns: 
                  ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            **strategy_params: Parameters to pass to the strategy constructor
            
        Returns:
            Dictionary containing backtest results and metrics
            
        Example:
            >>> import pandas as pd
            >>> # Load your data into a DataFrame
            >>> data = pd.read_csv('your_data.csv')
            >>> tester = StrategyTester(initial_capital=10000)
            >>> results = tester.test_strategy_from_data('SimpleMAStrategy', data, 
            ...                                         fast_period=10, slow_period=20)
        """
        strategy = self.load_strategy(strategy_name, **strategy_params)
        return self.backtest_strategy(strategy, data)
    
    def run_backtest(self, strategy_name: str, data: pd.DataFrame, 
                    verbose: bool = True, **strategy_params) -> Dict[str, Any]:
        """
        Convenient method to run a backtest with a pandas DataFrame.
        
        This is an alias for test_strategy_from_data() with verbose option.
        
        Args:
            strategy_name: Name of the strategy to test ('SimpleMAStrategy' or 'FibonacciChannelStrategy')
            data: Pandas DataFrame with OHLCV data. Must have columns: 
                  ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            verbose: Whether to print progress and results
            **strategy_params: Parameters to pass to the strategy constructor
            
        Returns:
            Dictionary containing backtest results and metrics
            
        Example:
            >>> import pandas as pd
            >>> data = pd.read_csv('your_data.csv')  # Your OHLCV data
            >>> tester = StrategyTester(initial_capital=10000)
            >>> results = tester.run_backtest('SimpleMAStrategy', data, 
            ...                              fast_period=10, slow_period=20, verbose=True)
        """
        strategy = self.load_strategy(strategy_name, **strategy_params)
        return self.backtest_strategy(strategy, data, verbose=verbose)
    
    def plot_results(self, strategy_name: str, data: pd.DataFrame = None, 
                    save_path: str = None, show_plot: bool = True) -> None:
        """
        Generate comprehensive plots for strategy results.
        
        Args:
            strategy_name: Name of the strategy to plot
            data: Original OHLCV DataFrame used for backtesting. If None, will try to 
                  extract from stored results or provide helpful error message.
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        if strategy_name not in self.test_results:
            print(f"No results found for strategy: {strategy_name}")
            return
        
        result = self.test_results[strategy_name]
        trades = result['trades']
        equity_curve = result['equity_curve']
        
        # Handle missing data parameter
        if data is None:
            # Try to get data from stored results if available
            if 'original_data' in result:
                data = result['original_data']
            else:
                print(f"❌ Error: No price data provided for plotting.")
                print(f"")
                print(f"The plot_results method needs the original OHLCV data to plot price charts.")
                print(f"")
                print(f"💡 Solutions:")
                print(f"1. If you used test_fibonacci_strategy() or similar helper functions:")
                print(f"   tester.plot_results('{strategy_name}', results['data'])")
                print(f"")
                print(f"2. If you have the original data in a variable 'data':")
                print(f"   tester.plot_results('{strategy_name}', data)")
                print(f"")
                print(f"3. If you used run_backtest() with a DataFrame 'df':")
                print(f"   tester.plot_results('{strategy_name}', df)")
                return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Strategy Results: {strategy_name}', fontsize=16, fontweight='bold')
        
        # 1. Price chart with buy/sell signals
        # Convert timestamps to datetime for matplotlib compatibility
        plot_timestamps = pd.to_datetime(data['timestamp']).dt.to_pydatetime()
        ax1.plot(plot_timestamps, data['close'], label='Price', alpha=0.7, linewidth=1)
        
        # Plot buy/sell signals with timestamp conversion
        buy_signals = [(t.entry_time, t.entry_price) for t in trades if t.position == PositionType.LONG]
        sell_signals = [(t.exit_time, t.exit_price) for t in trades if t.position == PositionType.LONG and t.exit_time]
        short_signals = [(t.entry_time, t.entry_price) for t in trades if t.position == PositionType.SHORT]
        cover_signals = [(t.exit_time, t.exit_price) for t in trades if t.position == PositionType.SHORT and t.exit_time]
        
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            # Convert timestamps to datetime
            buy_times = [pd.to_datetime(t).to_pydatetime() for t in buy_times]
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy', alpha=0.8)
        
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            # Convert timestamps to datetime
            sell_times = [pd.to_datetime(t).to_pydatetime() for t in sell_times]
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell', alpha=0.8)
        
        if short_signals:
            short_times, short_prices = zip(*short_signals)
            # Convert timestamps to datetime
            short_times = [pd.to_datetime(t).to_pydatetime() for t in short_times]
            ax1.scatter(short_times, short_prices, color='orange', marker='v', s=100, label='Short', alpha=0.8)
        
        if cover_signals:
            cover_times, cover_prices = zip(*cover_signals)
            # Convert timestamps to datetime
            cover_times = [pd.to_datetime(t).to_pydatetime() for t in cover_times]
            ax1.scatter(cover_times, cover_prices, color='blue', marker='^', s=100, label='Cover', alpha=0.8)
        
        ax1.set_title('Price Chart with Trading Signals')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Equity curve
        timestamps = data['timestamp'].iloc[:len(equity_curve)]
        # Convert timestamps to datetime for matplotlib compatibility
        plot_timestamps_equity = pd.to_datetime(timestamps).dt.to_pydatetime()
        ax2.plot(plot_timestamps_equity, equity_curve, label='Equity Curve', linewidth=2, color='blue')
        ax2.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Capital (USDT)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Drawdown chart
        peak = self.initial_capital
        drawdowns = []
        for capital in equity_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            drawdowns.append(drawdown)
        
        ax3.fill_between(plot_timestamps_equity, drawdowns, 0, alpha=0.3, color='red', label='Drawdown')
        ax3.plot(plot_timestamps_equity, drawdowns, color='red', linewidth=1)
        ax3.set_title('Drawdown Analysis')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_ylim(bottom=0)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Trade distribution
        if trades:
            trade_returns = [t.pnl / self.initial_capital * 100 for t in trades]
            ax4.hist(trade_returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Break-even')
            ax4.set_title('Trade Return Distribution')
            ax4.set_xlabel('Trade Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No trades executed', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Trade Return Distribution')
        
        # Adjust layout and save/show
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def display_comprehensive_metrics(self, strategy_name: str) -> None:
        """Display comprehensive metrics in a formatted table."""
        if strategy_name not in self.test_results:
            print(f"No results found for strategy: {strategy_name}")
            return
        
        result = self.test_results[strategy_name]
        metrics = result['metrics']
        
        print(f"\n📊 COMPREHENSIVE METRICS TABLE FOR {strategy_name.upper()}")
        print("=" * 100)
        
        # Create a comprehensive metrics display
        metric_groups = {
            "PERFORMANCE": {
                "Total Return": f"{metrics['total_return']:.2%}",
                "Final Capital": f"${metrics['final_capital']:,.2f}",
                "Total Trades": str(metrics['total_trades']),
                "Win Rate": f"{metrics['win_rate']:.2%}",
                "Profit Factor": f"{metrics['profit_factor']:.2f}",
                "Risk:Reward": f"{metrics['risk_reward_ratio']:.2f}"
            },
            "RISK METRICS": {
                "Max Drawdown": f"{metrics['max_drawdown']:.2%}",
                "Sharpe Ratio": f"{metrics['sharpe_ratio']:.3f}",
                "Sortino Ratio": f"{metrics['sortino_ratio']:.3f}",
                "Calmar Ratio": f"{metrics['calmar_ratio']:.3f}",
                "Volatility": f"{metrics['volatility']:.2%}",
                "VaR (95%)": f"{metrics['var_95']:.2%}"
            },
            "TRADE ANALYSIS": {
                "Avg Win": f"${metrics['avg_win']:,.2f}",
                "Avg Loss": f"${metrics['avg_loss']:,.2f}",
                "Largest Win": f"${metrics['largest_win']:,.2f}",
                "Largest Loss": f"${metrics['largest_loss']:,.2f}",
                "Max Consec Losses": str(metrics['max_consecutive_losses']),
                "Max Consec Wins": str(metrics['max_consecutive_wins'])
            },
            "STRATEGY BEHAVIOR": {
                "Trades/Day": f"{metrics['trades_per_day']:.2f}",
                "Avg Hold Time": f"{metrics['avg_holding_time']:.2f}h",
                "Market Exposure": f"{metrics['exposure_percentage']:.2f}%",
                "Break-even": f"{metrics['break_even_point']:.2%}",
                "Total Fees": f"${metrics['total_fees']:,.2f}",
                "Fees %": f"{metrics['total_fees']/metrics['final_capital']:.2%}"
            }
        }
        
        # Display metrics in columns
        for group_name, group_metrics in metric_groups.items():
            print(f"\n{group_name}:")
            print("-" * 50)
            for metric_name, metric_value in group_metrics.items():
                print(f"  {metric_name:<20}: {metric_value}")
        
        print("\n" + "=" * 100)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all test results."""
        if not self.test_results:
            return pd.DataFrame()
        
        summary_data = []
        for strategy_name, result in self.test_results.items():
            metrics = result['metrics']
            summary_data.append({
                'strategy_name': strategy_name,
                'total_return': metrics['total_return'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown': metrics['max_drawdown'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_trades': metrics['total_trades'],
                'final_capital': metrics['final_capital']
            })
        
        return pd.DataFrame(summary_data)
