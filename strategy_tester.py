"""
Strategy Tester for Trading Algorithm Research.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Type
import importlib
import inspect
from pathlib import Path

from strategies.base_strategy import BaseStrategy
from config import INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE


class StrategyTester:
    """
    Strategy testing framework for backtesting trading strategies.
    """
    
    def __init__(self, initial_capital: float = None, commission_rate: float = None,
                 slippage: float = None):
        """
        Initialize the strategy tester.
        
        Args:
            initial_capital: Starting capital for backtests
            commission_rate: Commission rate as decimal
            slippage: Slippage rate as decimal
        """
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.commission_rate = commission_rate or COMMISSION_RATE
        self.slippage = slippage or SLIPPAGE
        
        # Available strategies
        self.available_strategies = self._discover_strategies()
        
        # Test results storage
        self.test_results: Dict[str, Dict[str, Any]] = {}
        
    def _discover_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """Discover available strategies in the strategies folder."""
        strategies = {}
        strategies_dir = Path("strategies")
        
        if not strategies_dir.exists():
            return strategies
        
        # Look for Python files in strategies directory
        for strategy_file in strategies_dir.glob("*.py"):
            if strategy_file.name.startswith("__"):
                continue
                
            try:
                # Import the module
                module_name = f"strategies.{strategy_file.stem}"
                module = importlib.import_module(module_name)
                
                # Find strategy classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        strategies[name] = obj
                        
            except Exception as e:
                print(f"Warning: Could not load strategy from {strategy_file}: {e}")
        
        return strategies
    
    def list_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self.available_strategies.keys())
    
    def load_strategy(self, strategy_name: str, **params) -> BaseStrategy:
        """Load a strategy instance with given parameters."""
        if strategy_name not in self.available_strategies:
            available = ", ".join(self.available_strategies.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")
        
        strategy_class = self.available_strategies[strategy_name]
        
        # Set default parameters
        default_params = {
            'initial_capital': self.initial_capital,
            'commission_rate': self.commission_rate,
            'slippage': self.slippage
        }
        
        # Override with provided parameters
        default_params.update(params)
        
        return strategy_class(**default_params)
    
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
            
            # Check for entry/exit signals
            if strategy.position == strategy.position.FLAT:
                # Check for entry signals
                if strategy.should_enter_long(row, signals_data):
                    size = strategy.calculate_position_size(current_price)
                    strategy.enter_long(timestamp, current_price, size)
                
                elif strategy.should_enter_short(row, signals_data):
                    size = strategy.calculate_position_size(current_price)
                    strategy.enter_short(timestamp, current_price, size)
            
            elif strategy.position == strategy.position.LONG:
                # Check for exit signals
                if strategy.should_exit_long(row, signals_data):
                    strategy.exit_long(timestamp, current_price)
            
            elif strategy.position == strategy.position.SHORT:
                # Check for exit signals
                if strategy.should_exit_short(row, signals_data):
                    strategy.exit_short(timestamp, current_price)
        
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
            print(f"Backtest completed. Total trades: {metrics['total_trades']}")
            print(f"Final return: {metrics['total_return']:.2%}")
            print(f"Win rate: {metrics['win_rate']:.2%}")
            print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        
        return result
    
    def test_strategy_from_data(self, strategy_name: str, data: pd.DataFrame,
                               **strategy_params) -> Dict[str, Any]:
        """Test a strategy by name with given data."""
        strategy = self.load_strategy(strategy_name, **strategy_params)
        return self.backtest_strategy(strategy, data)
    
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
