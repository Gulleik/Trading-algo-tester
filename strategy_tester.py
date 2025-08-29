"""
Strategy Tester for Trading Algorithm Research.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from strategies.base_strategy import BaseStrategy, Position


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
        return ['SimpleMAStrategy']
    
    def load_strategy(self, strategy_name: str, **params) -> BaseStrategy:
        """Load a strategy instance with given parameters."""
        if strategy_name == 'SimpleMAStrategy':
            from strategies.simple_ma_strategy import SimpleMAStrategy
            return SimpleMAStrategy(**params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def backtest_strategy(self, strategy: BaseStrategy, data: pd.DataFrame,
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Run a complete backtest of a strategy.
        
        Args:
            strategy: Strategy instance to test
            data: OHLCV data for backtesting
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing backtest results and metrics
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
        
        # Run backtest
        if verbose:
            print("Running backtest simulation...")
        
        for i, (timestamp, row) in enumerate(signals_data.iterrows()):
            current_price = row['close']
            
            # Get the actual timestamp from the row data, not the index
            actual_timestamp = row['timestamp'] if 'timestamp' in row else pd.Timestamp.now()
            
            # Check for entry/exit signals
            if strategy.position == strategy.position.FLAT:
                # Check for entry signals
                if strategy.should_enter_long(row, signals_data):
                    size = strategy.calculate_position_size(current_price)
                    strategy.enter_long(actual_timestamp, current_price, size)
                
                elif strategy.should_enter_short(row, signals_data):
                    size = strategy.calculate_position_size(current_price)
                    strategy.enter_short(actual_timestamp, current_price, size)
            
            elif strategy.position == strategy.position.LONG:
                # Check for exit signals
                if strategy.should_exit_long(row, signals_data):
                    strategy.exit_long(actual_timestamp, current_price)
            
            elif strategy.position == strategy.position.SHORT:
                # Check for exit signals
                if strategy.should_exit_short(row, signals_data):
                    strategy.exit_short(actual_timestamp, current_price)
        
        # Close any open positions at the end
        if strategy.position != strategy.position.FLAT:
            final_price = data.iloc[-1]['close']
            if strategy.position == strategy.position.LONG:
                strategy.exit_long(data.iloc[-1]['timestamp'], final_price)
            else:
                strategy.exit_short(data.iloc[-1]['timestamp'], final_price)
            
            if verbose:
                print(f"Closed final position at {final_price:.2f}")
        
        # Calculate final metrics
        if verbose:
            print("Calculating performance metrics...")
        
        metrics = strategy.get_performance_metrics()
        
        # Store results
        result = {
            'strategy_name': strategy.name,
            'metrics': metrics,
            'trades': strategy.trades,
            'equity_curve': strategy.equity_curve,
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
        """Test a strategy by name with given data."""
        strategy = self.load_strategy(strategy_name, **strategy_params)
        return self.backtest_strategy(strategy, data)
    
    def plot_results(self, strategy_name: str, data: pd.DataFrame, 
                    save_path: str = None, show_plot: bool = True) -> None:
        """Generate comprehensive plots for strategy results."""
        if strategy_name not in self.test_results:
            print(f"No results found for strategy: {strategy_name}")
            return
        
        result = self.test_results[strategy_name]
        trades = result['trades']
        equity_curve = result['equity_curve']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Strategy Results: {strategy_name}', fontsize=16, fontweight='bold')
        
        # 1. Price chart with buy/sell signals
        ax1.plot(data['timestamp'], data['close'], label='Price', alpha=0.7, linewidth=1)
        
        # Plot buy/sell signals
        buy_signals = [(t.entry_time, t.entry_price) for t in trades if t.position == Position.LONG]
        sell_signals = [(t.exit_time, t.exit_price) for t in trades if t.position == Position.LONG and t.exit_time]
        short_signals = [(t.entry_time, t.entry_price) for t in trades if t.position == Position.SHORT]
        cover_signals = [(t.exit_time, t.exit_price) for t in trades if t.position == Position.SHORT and t.exit_time]
        
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy', alpha=0.8)
        
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell', alpha=0.8)
        
        if short_signals:
            short_times, short_prices = zip(*short_signals)
            ax1.scatter(short_times, short_prices, color='orange', marker='v', s=100, label='Short', alpha=0.8)
        
        if cover_signals:
            cover_times, cover_prices = zip(*cover_signals)
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
        ax2.plot(timestamps, equity_curve, label='Equity Curve', linewidth=2, color='blue')
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
        
        ax3.fill_between(timestamps, drawdowns, 0, alpha=0.3, color='red', label='Drawdown')
        ax3.plot(timestamps, drawdowns, color='red', linewidth=1)
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
                "Fees %": f"{metrics['total_fees']/metrics['final_capital']*100:.2%}"
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
