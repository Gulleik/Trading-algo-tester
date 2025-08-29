"""
Main entry point for trading strategy research.
"""

from data_loader import get_exchange_instance, fetch_ohlcv_all
from strategy_tester import StrategyTester


def main():
    """Main function to test trading strategies."""
    try:
        # Initialize exchange and fetch data
        exchange = get_exchange_instance()
        symbol = "BTC/USDT:USDT"
        timeframe = "5m"
        
        print(f"Fetching {timeframe} data for {symbol}")
        data = fetch_ohlcv_all(exchange, symbol, timeframe, 1000)
        
        if data.empty:
            print("No data fetched")
            return
            
        print(f"Fetched {len(data)} candles")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        # Test strategy
        tester = StrategyTester()
        result = tester.test_strategy_from_data('SimpleMAStrategy', data, fast_period=10, slow_period=20)
        
        # Print results
        metrics = result['metrics']
        print(f"\nResults:")
        print(f"Total return: {metrics['total_return']:.2%}")
        print(f"Win rate: {metrics['win_rate']:.2%}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total trades: {metrics['total_trades']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
