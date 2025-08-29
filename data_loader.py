"""
Data loading module for fetching OHLCV data from Bybit.
"""

import ccxt
import pandas as pd
import time
from typing import Optional


def get_exchange_instance() -> ccxt.Exchange:
    """Get Bybit exchange instance."""
    return ccxt.bybit()


def fetch_ohlcv_all(
    exchange: ccxt.Exchange, 
    symbol: str, 
    timeframe: str, 
    data_length: int = 1000, 
    since: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol and timeframe with pagination support.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol (e.g., "BTC/USDT:USDT")
        timeframe: Data timeframe (e.g., "5m", "1h")
        data_length: Number of candles to fetch (can be > 1000)
        since: Start timestamp for fetching (milliseconds)
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data columns: 
                     timestamp, open, high, low, close, volume
    """
    all_rows = []
    api_calls = 0
    max_api_calls = 200  # Increased limit for larger data requests
    
    # Bybit limit per request
    max_limit_per_request = 1000
    
    print(f"Starting to fetch {data_length} candles for {symbol} at {timeframe}")
    
    # If no since timestamp provided, start from a reasonable past time
    if since is None:
        # Calculate timeframe in milliseconds
        if timeframe == '1m':
            time_ms = 60 * 1000
        elif timeframe == '5m':
            time_ms = 5 * 60 * 1000
        elif timeframe == '15m':
            time_ms = 15 * 60 * 1000
        elif timeframe == '30m':
            time_ms = 30 * 60 * 1000
        elif timeframe == '1h':
            time_ms = 60 * 60 * 1000
        elif timeframe == '4h':
            time_ms = 4 * 60 * 60 * 1000
        elif timeframe == '1d':
            time_ms = 24 * 60 * 60 * 1000
        else:
            # Default to 1 hour if timeframe not recognized
            time_ms = 60 * 60 * 1000
        
        # Start from a time that's data_length * timeframe in the past
        # This ensures we have enough historical data to work with
        current_time = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        since = current_time - (data_length * time_ms)
        print(f"Starting from calculated past time: {pd.to_datetime(since, unit='ms', utc=True)}")
    
    while api_calls < max_api_calls and len(all_rows) < data_length:
        try:
            # Calculate how many candles to request in this batch
            remaining_candles = data_length - len(all_rows)
            current_limit = min(remaining_candles, max_limit_per_request)
            
            print(f"API call {api_calls + 1}: Requesting {current_limit} candles, {remaining_candles} remaining...")
            
            chunk = exchange.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                since=since, 
                limit=current_limit
            )
            
            if not chunk:
                print(f"No more data available for {symbol} at {timeframe}")
                break
                
            all_rows.extend(chunk)
            api_calls += 1
            
            print(f"Fetched batch {api_calls}: {len(chunk)} candles. Total: {len(all_rows)}/{data_length}")
            
            # Check if we have enough data
            if len(all_rows) >= data_length:
                all_rows = all_rows[:data_length]
                print(f"Reached target data length: {len(all_rows)} candles")
                break
            
            # For backwards fetching, we need to calculate the previous timestamp
            # based on the timeframe
            if len(chunk) > 0:
                # Calculate the timestamp for the previous period
                if timeframe == '1m':
                    time_ms = 60 * 1000
                elif timeframe == '5m':
                    time_ms = 5 * 60 * 1000
                elif timeframe == '15m':
                    time_ms = 15 * 60 * 1000
                elif timeframe == '30m':
                    time_ms = 30 * 60 * 1000
                elif timeframe == '1h':
                    time_ms = 60 * 60 * 1000
                elif timeframe == '4h':
                    time_ms = 4 * 60 * 60 * 1000
                elif timeframe == '1d':
                    time_ms = 24 * 60 * 60 * 1000
                else:
                    # Default to 1 hour if timeframe not recognized
                    time_ms = 60 * 60 * 1000
                
                # Go backwards in time
                old_since = since
                since = since - (len(chunk) * time_ms)
                print(f"Updated since timestamp: {old_since} -> {since} ({pd.to_datetime(since, unit='ms', utc=True)})")
            
            # Rate limiting - be more conservative with larger requests
            if data_length > 5000:
                time.sleep(0.2)  # Slower for very large requests
            else:
                time.sleep(0.1)
            
            # Check if we got less data than requested (end of available data)
            # Only break if we got significantly less than requested
            if len(chunk) < current_limit * 0.9:  # Allow some variance
                print(f"Reached end of available data. Got {len(chunk)} instead of {current_limit}")
                break
                
        except Exception as e:
            print(f"Error fetching data for {symbol} (batch {api_calls}): {e}")
            # Wait a bit longer on error before retrying
            time.sleep(1)
            continue
    
    if not all_rows:
        print(f"No data fetched for {symbol}")
        return pd.DataFrame()
    
    print(f"Successfully fetched {len(all_rows)} candles in {api_calls} API calls")
    
    # Create DataFrame
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(all_rows, columns=cols)
    
    # Clean and process data
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    return df.reset_index(drop=True)


def fetch_ohlcv_with_date_range(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_candles: int = 10000
) -> pd.DataFrame:
    """
    Fetch OHLCV data within a specific date range with pagination.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol (e.g., "BTC/USDT:USDT")
        timeframe: Data timeframe (e.g., "5m", "1h")
        start_date: Start date string (e.g., "2024-01-01")
        end_date: End date string (e.g., "2024-12-31")
        max_candles: Maximum number of candles to fetch
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    # Convert dates to timestamps if provided
    since = None
    if start_date:
        since = pd.to_datetime(start_date).tz_localize('UTC').timestamp() * 1000
    
    # Fetch data with pagination
    df = fetch_ohlcv_all(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        data_length=max_candles,
        since=since
    )
    
    # Filter by end date if specified
    if end_date and not df.empty:
        end_timestamp = pd.to_datetime(end_date).tz_localize('UTC')
        df = df[df['timestamp'] <= end_timestamp]
    
    return df
