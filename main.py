# pip install ccxt pandas pyarrow
import ccxt
import pandas as pd
import time
from config import (
    EXCHANGE_NAME, DEFAULT_TIMEFRAME, DEFAULT_LIMIT, MAX_API_CALLS, 
    RATE_LIMIT_DELAY, SYMBOLS_CONFIG, get_symbol_by_name, get_timeframes_for_symbol,
    get_data_length_for_symbol
)

def get_exchange_instance():
    """Get exchange instance based on configuration."""
    exchange_map = {
        "bybit": ccxt.bybit,
        "binance": ccxt.binance,
        "okx": ccxt.okx,
        "kucoin": ccxt.kucoin,
        "coinbase": ccxt.coinbase
    }
    
    if EXCHANGE_NAME not in exchange_map:
        raise ValueError(f"Unsupported exchange: {EXCHANGE_NAME}")
    
    return exchange_map[EXCHANGE_NAME]()

def fetch_ohlcv_all(exchange, symbol, timeframe, data_length=None, since=None):
    """
    Fetch all OHLCV data for a symbol and timeframe.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol
        timeframe: Data timeframe
        data_length: Number of candles to fetch (overrides symbol config)
        since: Start timestamp for fetching
        
    Returns:
        DataFrame with OHLCV data
    """
    # Get symbol configuration
    symbol_config = get_symbol_by_name(symbol)
    if symbol_config:
        # Use symbol-specific data length if not specified
        if data_length is None:
            data_length = symbol_config["data_length"]
        
        # Validate timeframe
        if timeframe not in symbol_config["timeframes"]:
            print(f"Warning: {timeframe} not in allowed timeframes for {symbol}")
            print(f"Allowed timeframes: {symbol_config['timeframes']}")
    
    # Use default if no symbol config found
    if data_length is None:
        data_length = DEFAULT_LIMIT
    
    all_rows = []
    api_calls = 0
    
    while api_calls < MAX_API_CALLS:
        try:
            chunk = exchange.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                since=since, 
                limit=min(data_length, DEFAULT_LIMIT)
            )
            
            if not chunk:
                break
                
            all_rows.extend(chunk)
            api_calls += 1
            
            # Check if we have enough data
            if len(all_rows) >= data_length:
                all_rows = all_rows[:data_length]
                break
            
            # Next since = last timestamp + 1 ms
            since = chunk[-1][0] + 1
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            # Check if we got less data than requested (end of available data)
            if len(chunk) < DEFAULT_LIMIT:
                break
                
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            break
    
    if not all_rows:
        return pd.DataFrame()
    
    # Create DataFrame
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(all_rows, columns=cols)
    
    # Clean and process data
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    return df.reset_index(drop=True)

def main():
    """Main function to demonstrate configuration usage."""
    try:
        # Initialize exchange
        exchange = get_exchange_instance()
        print(f"Connected to {EXCHANGE_NAME}")
        
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
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
