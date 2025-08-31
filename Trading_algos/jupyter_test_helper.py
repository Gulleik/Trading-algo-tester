"""
Helper functions for testing trading strategies in Jupyter notebooks.
This addresses common issues that prevent trades from executing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.simple_ma_strategy import SimpleMAStrategy
from strategies.fibnacci_stratergy import FibonacciChannelStrategy
from strategy_tester import StrategyTester


def create_test_data(n_points=100, start_price=100.0, volatility=0.02, trend=0.1):
    """
    Create realistic test data that will generate trading signals.
    
    Args:
        n_points: Number of data points
        start_price: Starting price
        volatility: Daily volatility (0.02 = 2%)
        trend: Total trend over period (0.1 = 10% move)
    
    Returns:
        DataFrame with OHLCV data that should generate signals
    """
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Create timestamps (5-minute intervals)
    start_time = datetime.now() - timedelta(hours=n_points * 5 / 60)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_points)]
    
    # Generate price series with trend and volatility
    trend_component = np.linspace(0, trend, n_points)
    noise = np.random.normal(0, volatility/np.sqrt(252*24*12), n_points)  # Scale for 5-min intervals
    
    # Create price series
    prices = [start_price]
    for i in range(1, n_points):
        change = trend_component[i]/n_points + noise[i]
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    # Create OHLCV data
    data = []
    for i, timestamp in enumerate(timestamps):
        price = prices[i]
        # Create realistic OHLC spreads
        spread = price * 0.001  # 0.1% spread
        high = price + np.random.uniform(0, spread)
        low = price - np.random.uniform(0, spread)
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(price, high, open_price),
            'low': min(price, low, open_price),
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_simple_ma_strategy(data=None, verbose=True):
    """
    Test Simple MA strategy with optimal parameters for signal generation.
    
    Args:
        data: Optional data DataFrame. If None, creates test data.
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with strategy, signals, and results
    """
    if data is None:
        data = create_test_data(n_points=80, trend=0.15, volatility=0.025)
    
    # Create strategy with parameters optimized for signal generation
    strategy = SimpleMAStrategy(
        name="JupyterTest_SimpleMA",
        fast_period=5,      # Short period for quick signals
        slow_period=15,     # Medium period for clear crossovers
        initial_capital=10000.0,
        use_take_profits=True,
        stop_loss_pct=1.0,  # Wider stop loss to avoid premature exits
        risk_per_trade=0.01,  # Conservative sizing
        verbose_trading=verbose
    )
    
    # Run backtest
    tester = StrategyTester(initial_capital=10000.0)
    results = tester.backtest_strategy(strategy, data, verbose=verbose)
    
    # Generate signals for analysis
    signals = strategy.generate_signals(data)
    
    if verbose:
        print(f"\n📊 SIMPLE MA STRATEGY TEST RESULTS")
        print(f"=" * 50)
        print(f"Data points: {len(data)}")
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"Signals generated: {signals['long_signal'].sum() + signals['short_signal'].sum()}")
        print(f"Long signals: {signals['long_signal'].sum()}")
        print(f"Short signals: {signals['short_signal'].sum()}")
        print(f"Trades executed: {results['metrics']['total_trades']}")
        print(f"Final return: {results['metrics']['total_return']:.2%}")
        print(f"Positions created: {len(strategy.positions)}")
        print(f"Orders executed: {len(strategy.orders)}")
    
    return {
        'strategy': strategy,
        'signals': signals,
        'results': results,
        'data': data
    }


def test_fibonacci_strategy(data=None, verbose=True):
    """
    Test Fibonacci strategy with optimal parameters for signal generation.
    
    Args:
        data: Optional data DataFrame. If None, creates test data.
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with strategy, signals, and results
    """
    if data is None:
        data = create_test_data(n_points=100, trend=0.2, volatility=0.03)
    
    # Create strategy with parameters optimized for signal generation
    strategy = FibonacciChannelStrategy(
        name="JupyterTest_Fibonacci",
        sensitivity=3,      # More sensitive for quicker signals
        initial_capital=10000.0,
        sl_percent=1.5,     # Wider stop loss
        use_take_profits=True,
        tp1_pct=0.5, tp1_close=25.0,
        tp2_pct=1.0, tp2_close=25.0,
        tp3_pct=1.5, tp3_close=50.0,
        risk_per_trade=0.01  # Conservative sizing
    )
    
    # Run backtest
    tester = StrategyTester(initial_capital=10000.0)
    results = tester.backtest_strategy(strategy, data, verbose=verbose)
    
    # Generate signals for analysis
    signals = strategy.generate_signals(data)
    
    if verbose:
        print(f"\n📊 FIBONACCI STRATEGY TEST RESULTS")
        print(f"=" * 50)
        print(f"Data points: {len(data)}")
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"Signals generated: {signals['long_signal'].sum() + signals['short_signal'].sum()}")
        print(f"Long signals: {signals['long_signal'].sum()}")
        print(f"Short signals: {signals['short_signal'].sum()}")
        print(f"Trades executed: {results['metrics']['total_trades']}")
        print(f"Final return: {results['metrics']['total_return']:.2%}")
        print(f"Positions created: {len(strategy.positions)}")
        print(f"Orders executed: {len(strategy.orders)}")
        
        # Show Fibonacci-specific info
        fib_signals = signals[signals['long_signal'] | signals['short_signal']]
        if len(fib_signals) > 0:
            print(f"\nFibonacci Signal Details:")
            for i, (idx, row) in enumerate(fib_signals.iterrows()):
                signal_type = "LONG" if row['long_signal'] else "SHORT"
                print(f"  Signal {i+1}: {signal_type} at ${row['close']:.2f}")
                print(f"    Fib 236: ${row['fib_236']:.2f}, Fib 500: ${row['fib_500']:.2f}, Fib 786: ${row['fib_786']:.2f}")
    
    return {
        'strategy': strategy,
        'signals': signals,
        'results': results,
        'data': data
    }


def quick_test_both_strategies():
    """
    Quick test of both strategies with guaranteed signal generation.
    Perfect for Jupyter notebook testing.
    """
    print("🚀 QUICK STRATEGY TEST FOR JUPYTER")
    print("=" * 60)
    
    # Test Simple MA strategy
    ma_results = test_simple_ma_strategy(verbose=True)
    
    # Test Fibonacci strategy 
    fib_results = test_fibonacci_strategy(verbose=True)
    
    print(f"\n💡 SUMMARY")
    print(f"=" * 30)
    print(f"Simple MA - Trades: {ma_results['results']['metrics']['total_trades']}, Return: {ma_results['results']['metrics']['total_return']:.2%}")
    print(f"Fibonacci - Trades: {fib_results['results']['metrics']['total_trades']}, Return: {fib_results['results']['metrics']['total_return']:.2%}")
    
    return ma_results, fib_results


def analyze_signals(signals_df, strategy_name="Strategy"):
    """
    Analyze signal generation in detail.
    
    Args:
        signals_df: DataFrame with signals from generate_signals()
        strategy_name: Name for display
    """
    print(f"\n🔍 SIGNAL ANALYSIS: {strategy_name}")
    print(f"=" * 40)
    
    if 'long_signal' in signals_df.columns:
        long_signals = signals_df[signals_df['long_signal'] == True]
        short_signals = signals_df[signals_df['short_signal'] == True]
        
        print(f"Long signals: {len(long_signals)}")
        print(f"Short signals: {len(short_signals)}")
        
        if len(long_signals) > 0:
            print(f"Long signal prices: {long_signals['close'].tolist()}")
        if len(short_signals) > 0:
            print(f"Short signal prices: {short_signals['close'].tolist()}")
    
    # Show first few rows with key columns
    display_cols = ['close']
    if 'fast_ma' in signals_df.columns:
        display_cols.extend(['fast_ma', 'slow_ma', 'long_signal', 'short_signal'])
    elif 'fib_500' in signals_df.columns:
        display_cols.extend(['fib_236', 'fib_500', 'fib_786', 'long_signal', 'short_signal'])
    
    print(f"\nFirst 10 signal rows:")
    print(signals_df[display_cols].head(10).to_string())


# Convenient function for Jupyter cells
def run_strategy_test():
    """One-line function to run complete strategy test."""
    return quick_test_both_strategies()


# Example usage for Jupyter:
"""
# In Jupyter cell:
from jupyter_test_helper import run_strategy_test, test_simple_ma_strategy, create_test_data

# Quick test both strategies
ma_results, fib_results = run_strategy_test()

# Or test individual strategies with custom data
data = create_test_data(n_points=150, trend=0.1, volatility=0.02)
ma_test = test_simple_ma_strategy(data)
"""
