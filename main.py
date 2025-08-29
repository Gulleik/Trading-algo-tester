"""
Main entry point for the trading strategy research application.

This file serves as the CLI entry point and should contain minimal logic,
importing functionality from dedicated modules.
"""

from data_loader import get_exchange_instance, fetch_ohlcv_all, fetch_funding_rates


def main():
    """Main function to demonstrate data fetching functionality."""
    try:
        # Initialize exchange
        exchange = get_exchange_instance()
        print(f"Connected to exchange")
        
        # Example: Fetch data for BTC/USDT:USDT
        symbol = "BTC/USDT:USDT"
        timeframe = "5m"
        
        print(f"\nFetching {timeframe} data for {symbol}")
        df = fetch_ohlcv_all(exchange, symbol, timeframe)
        
        if not df.empty:
            print(f"Fetched {len(df)} candles")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print("\nFirst few rows:")
            print(df.head())
            print("\nLast few rows:")
            print(df.tail())
        else:
            print("No data fetched")
            
        # Example: Fetch funding rates
        print(f"\nFetching funding rates for {symbol}")
        funding_df = fetch_funding_rates(exchange, symbol)
        
        if not funding_df.empty:
            print(f"Fetched {len(funding_df)} funding rate records")
            print("\nFirst few funding rates:")
            print(funding_df.head())
        else:
            print("No funding rate data available")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
