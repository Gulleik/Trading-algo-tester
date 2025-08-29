"""
Example usage of the Strategy Tester.
"""

import pandas as pd
import numpy as np
from strategy_tester import StrategyTester
from data_loader import get_exchange_instance, fetch_ohlcv_all, fetch_ohlcv_with_date_range


def test_with_real_data():
    """Test strategies using real data from exchange."""
    print("\n" + "=" * 60)
    print("TESTING STRATEGIES WITH REAL EXCHANGE DATA")
    print("=" * 60)
    
    # Initialize strategy tester
    tester = StrategyTester(initial_capital=10000.0)
    
    # Check available strategies
    strategies = tester.list_available_strategies()
    if not strategies:
        print("No strategies available for testing.")
        return
    
    # Test with real data (BTC/USDT:USDT on Bybit)
    symbol = "BTC/USDT:USDT"
    timeframe = "5m"
    data_length = 2000  # Reduced for faster testing
    
    print(f"\nFetching real data for {symbol}...")
    try:
        # Fetch data
        exchange = get_exchange_instance()
        data = fetch_ohlcv_all(exchange, symbol, timeframe, data_length)
        
        if data.empty:
            print("No data fetched")
            return
            
        print(f"Fetched {len(data)} candles")
        
        # Test SimpleMA strategy with real data
        result = tester.test_strategy_from_data(
            'SimpleMAStrategy',
            data,
            fast_period=10,
            slow_period=20
        )
        
        metrics = result['metrics']
        print(f"\nReal data test results for {symbol}:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Total Trades: {metrics['total_trades']}")
        
    except Exception as e:
        print(f"Error testing with real data: {e}")
        print("This might be due to API rate limits or network issues.")


def main():
    """Main function to run all examples."""
    print("TRADING STRATEGY TESTER - EXAMPLE USAGE")
    print("=" * 60)
    
    try:
        # Test with real data (if available)
        test_with_real_data()
        
        print("\n" + "=" * 60)
        print("EXAMPLE USAGE COMPLETED")
        print("=" * 60)
        print("\nYou can now:")
        print("1. Fetch large amounts of data (>1000 candles) using pagination")
        print("2. Fetch data within specific date ranges")
        print("3. Create your own strategies by inheriting from BaseStrategy")
        print("4. Test strategies with real exchange data")
        print("5. Optimize strategy parameters")
        print("6. Compare multiple strategies")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed and the strategies folder is set up correctly.")


if __name__ == "__main__":
    main()
