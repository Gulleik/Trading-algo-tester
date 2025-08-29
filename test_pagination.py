"""
Test script to verify pagination functionality works correctly.
"""

from data_loader import get_exchange_instance, fetch_ohlcv_all
import time
import pandas as pd

def test_pagination():
    """Test that we can actually fetch more than 1000 candles."""
    print("Testing pagination functionality...")
    print("=" * 50)
    
    exchange = get_exchange_instance()
    symbol = "BTC/USDT:USDT"
    timeframe = "1h"
    
    # Test 1: Fetch exactly 1000 candles (should work as before)
    print("\n1. Testing fetch of 1000 candles:")
    start_time = time.time()
    data_1000 = fetch_ohlcv_all(exchange, symbol, timeframe, data_length=1000)
    end_time = time.time()
    
    print(f"   Result: {len(data_1000)} candles fetched in {end_time - start_time:.2f} seconds")
    
    # Test 2: Fetch 1500 candles (should require pagination)
    print("\n2. Testing fetch of 1500 candles (requires pagination):")
    start_time = time.time()
    data_1500 = fetch_ohlcv_all(exchange, symbol, timeframe, data_length=1500)
    end_time = time.time()
    
    print(f"   Result: {len(data_1500)} candles fetched in {end_time - start_time:.2f} seconds")
    
    # Test 3: Fetch 2000 candles (should definitely require pagination)
    print("\n3. Testing fetch of 2000 candles (definitely requires pagination):")
    start_time = time.time()
    data_2000 = fetch_ohlcv_all(exchange, symbol, timeframe, data_length=2000)
    end_time = time.time()
    
    print(f"   Result: {len(data_2000)} candles fetched in {end_time - start_time:.2f} seconds")
    
    # Summary
    print("\n" + "=" * 50)
    print("PAGINATION TEST SUMMARY:")
    print(f"   1000 candles: {len(data_1000)} (expected: 1000)")
    print(f"   1500 candles: {len(data_1500)} (expected: 1500)")
    print(f"   2000 candles: {len(data_2000)} (expected: 2000)")
    
    # Verify data integrity
    if len(data_2000) > 0:
        print(f"\nData spans from {data_2000['timestamp'].min()} to {data_2000['timestamp'].max()}")
        print(f"Timeframe: {timeframe}")
        print(f"Total rows: {len(data_2000)}")
        
        # Check for duplicates
        duplicates = data_2000.duplicated(subset=['timestamp']).sum()
        print(f"Duplicate timestamps: {duplicates}")
        
        # Check data continuity
        timestamps = data_2000['timestamp'].sort_values()
        time_diffs = timestamps.diff().dropna()
        expected_diff = pd.Timedelta(timeframe)
        print(f"Expected time difference: {expected_diff}")
        print(f"Actual time differences range: {time_diffs.min()} to {time_diffs.max()}")

if __name__ == "__main__":
    test_pagination()
