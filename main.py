"""
Main entry point for trading strategy research.
"""

from data_loader import get_exchange_instance, fetch_ohlcv_all
from strategy_tester import StrategyTester


def display_results(symbol, data, result):
    """Display strategy test results and data information."""
    print("TRADING STRATEGY TESTER")
    print("=" * 60)
    
    # Display data info
    print(f"Fetched {len(data)} candles")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Display strategy results
    metrics = result['metrics']
    print(f"\nStrategy Test Results for {symbol}:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Average Win: {metrics['avg_win']:.2%}")
    print(f"  Average Loss: {metrics['avg_loss']:.2%}")
    print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")
    
    print(f"\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)


def run_strategy_test(symbol, timeframe, data_length, initial_capital):
    """Run a complete strategy test with the given parameters."""
    try:
        # Initialize strategy tester
        tester = StrategyTester(initial_capital=initial_capital)
        
        # Check available strategies
        strategies = tester.list_available_strategies()
        if not strategies:
            print("No strategies available for testing.")
            return None
            
        print(f"Available strategies: {', '.join(strategies)}")
        
        # Initialize exchange and fetch data
        exchange = get_exchange_instance()
        print(f"\nFetching {timeframe} data for {symbol}...")
        data = fetch_ohlcv_all(exchange, symbol, timeframe, data_length)
        
        if data.empty:
            print("No data fetched")
            return None
            
        # Test strategy
        print(f"\nTesting Fibonacci Channel strategy...")
        result = tester.test_strategy_from_data('FibonacciChannelStrategy', data, 
                                              sensitivity=5, 
                                              stop_loss_pct=0.75,
                                              risk_per_trade=0.02,
                                              use_take_profits=True)
        
        return tester, data, result
        
    except Exception as e:
        print(f"Error: {e}")
        print("This might be due to API rate limits, network issues, or configuration problems.")
        return None


def main():
    """Main function to execute the program."""
    # Configuration parameters - easy to modify here
    symbol = "BTC/USDT:USDT"
    timeframe = "5m"
    data_length = 2000
    initial_capital = 10000.0
    
    # Plotting options
    show_plots = True
    save_plots = True
    plot_filename = f"strategy_results_{symbol.replace('/', '_').replace(':', '_')}_{timeframe}.png"
    
    # Run the test
    result = run_strategy_test(symbol, timeframe, data_length, initial_capital)
    
    # Display results and generate plots if successful
    if result:
        tester, data, strategy_result = result
        display_results(symbol, data, strategy_result)
        
        # Display comprehensive metrics table
        print("\n" + "="*80)
        tester.display_comprehensive_metrics('FibonacciChannelStrategy')
        
        # Generate plots
        if show_plots or save_plots:
            print(f"\nGenerating plots...")
            tester.plot_results(
                'FibonacciChannelStrategy', 
                data, 
                save_path=plot_filename if save_plots else None,
                show_plot=show_plots
            )


if __name__ == "__main__":
    main()
