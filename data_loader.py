"""
Data loading module for fetching OHLCV and funding data from cryptocurrency exchanges.

This module handles all data fetching operations using the ccxt library,
including rate limiting, error handling, and data processing.
"""

import ccxt
import pandas as pd
import time
from typing import Optional, Dict, List, Any
from config import (
    EXCHANGE_NAME, DEFAULT_TIMEFRAME, DEFAULT_CANDLE_LIMIT, MAX_API_CALLS, 
    RATE_LIMIT_DELAY, SYMBOLS_CONFIG, get_symbol_by_name, get_timeframes_for_symbol,
    get_data_length_for_symbol
)


def get_exchange_instance() -> ccxt.Exchange:
    """
    Get exchange instance based on configuration.
    
    Returns:
        ccxt.Exchange: Configured exchange instance
        
    Raises:
        ValueError: If the exchange is not supported
    """
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


def fetch_ohlcv_all(
    exchange: ccxt.Exchange, 
    symbol: str, 
    timeframe: str, 
    data_length: Optional[int] = None, 
    since: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch all OHLCV data for a symbol and timeframe.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol (e.g., "BTC/USDT:USDT")
        timeframe: Data timeframe (e.g., "5m", "1h")
        data_length: Number of candles to fetch (overrides symbol config)
        since: Start timestamp for fetching (milliseconds)
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data columns: 
                     timestamp, open, high, low, close, volume
        
    Note:
        - Timestamps are converted to UTC datetime
        - Data is deduplicated and sorted by timestamp
        - Rate limiting is applied between API calls
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
        data_length = DEFAULT_CANDLE_LIMIT
    
    all_rows = []
    api_calls = 0
    
    while api_calls < MAX_API_CALLS:
        try:
            chunk = exchange.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                since=since, 
                limit=min(data_length, DEFAULT_CANDLE_LIMIT)
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
            if len(chunk) < DEFAULT_CANDLE_LIMIT:
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


def fetch_funding_rates(
    exchange: ccxt.Exchange, 
    symbol: str, 
    since: Optional[int] = None, 
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch funding rates for perpetual contracts.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol (e.g., "BTC/USDT:USDT")
        since: Start timestamp for fetching (milliseconds)
        limit: Maximum number of funding rate records to fetch
        
    Returns:
        pd.DataFrame: DataFrame with funding rate data columns:
                     timestamp, fundingRate
                     
    Note:
        - Only works for exchanges that support funding rate fetching
        - Returns empty DataFrame if funding rates not available
    """
    try:
        # Check if exchange supports funding rates
        if not hasattr(exchange, 'fetch_funding_rate_history'):
            print(f"Warning: {exchange.id} does not support funding rate fetching")
            return pd.DataFrame()
        
        funding_data = exchange.fetch_funding_rate_history(
            symbol, 
            since=since, 
            limit=limit
        )
        
        if not funding_data:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(funding_data)
        
        # Standardize column names
        if 'fundingRate' in df.columns:
            df = df[['timestamp', 'fundingRate']]
        elif 'funding_rate' in df.columns:
            df = df[['timestamp', 'funding_rate']]
            df = df.rename(columns={'funding_rate': 'fundingRate'})
        else:
            print(f"Warning: Unexpected funding rate column names: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Clean and process data
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error fetching funding rates for {symbol}: {e}")
        return pd.DataFrame()


def fetch_ticker(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetch current ticker information for a symbol.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol
        
    Returns:
        Dict containing ticker information (last price, bid, ask, volume, etc.)
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker
    except Exception as e:
        print(f"Error fetching ticker for {symbol}: {e}")
        return {}


def get_available_symbols(exchange: ccxt.Exchange) -> List[str]:
    """
    Get list of available trading symbols from the exchange.
    
    Args:
        exchange: CCXT exchange instance
        
    Returns:
        List of available symbol strings
    """
    try:
        markets = exchange.load_markets()
        return list(markets.keys())
    except Exception as e:
        print(f"Error loading markets: {e}")
        return []


def validate_symbol(exchange: ccxt.Exchange, symbol: str) -> bool:
    """
    Check if a symbol is valid and available on the exchange.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol to validate
        
    Returns:
        True if symbol is valid and available, False otherwise
    """
    try:
        markets = exchange.load_markets()
        return symbol in markets
    except Exception as e:
        print(f"Error validating symbol {symbol}: {e}")
        return False
