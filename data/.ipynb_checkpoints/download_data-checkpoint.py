"""
Data download utility for fetching and storing large chunks of trading data from Bybit.
Downloads OHLCV and funding data for multiple symbols and timeframes, storing as CSV files.
"""

import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Downloads and stores large chunks of trading data from Bybit for backtesting.
    Supports multiple symbols, timeframes, and data types (OHLCV + funding).
    """
    
    def __init__(self, data_dir: str = "data", max_api_calls: int = 500):
        """
        Initialize the data downloader.
        
        Args:
            data_dir: Directory to store downloaded data
            max_api_calls: Maximum API calls per download session
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_api_calls = max_api_calls
        self.exchange = ccxt.bybit()
        
        # Create subdirectories
        (self.data_dir / "ohlcv").mkdir(exist_ok=True)
        (self.data_dir / "funding").mkdir(exist_ok=True)
        (self.data_dir / "metadata").mkdir(exist_ok=True)
        
        # Supported timeframes and their millisecond equivalents
        self.timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
    
    def download_symbol_data(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: str,
        end_date: str,
        include_funding: bool = True
    ) -> Dict[str, Dict[str, str]]:
        """
        Download data for a specific symbol across multiple timeframes.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT:USDT")
            timeframes: List of timeframes to download
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_funding: Whether to download funding rate data
            
        Returns:
            Dict containing file paths for downloaded data
        """
        logger.info(f"Starting download for {symbol} from {start_date} to {end_date}")
        
        results = {
            'symbol': symbol,
            'timeframes': {},
            'funding': None,
            'metadata': None
        }
        
        # Convert dates to timestamps
        start_ts = int(pd.to_datetime(start_date).tz_localize('UTC').timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).tz_localize('UTC').timestamp() * 1000)
        
        # Download OHLCV data for each timeframe
        for timeframe in timeframes:
            if timeframe not in self.timeframe_ms:
                logger.warning(f"Unsupported timeframe: {timeframe}, skipping...")
                continue
                
            logger.info(f"Downloading {timeframe} data for {symbol}")
            file_path = self._download_ohlcv_data(symbol, timeframe, start_ts, end_ts)
            if file_path:
                results['timeframes'][timeframe] = file_path
        
        # Download funding data if requested
        if include_funding:
            logger.info(f"Downloading funding data for {symbol}")
            funding_path = self._download_funding_data(symbol, start_ts, end_ts)
            if funding_path:
                results['funding'] = funding_path
        
        # Save metadata
        metadata_path = self._save_metadata(symbol, timeframes, start_date, end_date, results)
        results['metadata'] = metadata_path
        
        logger.info(f"Download completed for {symbol}")
        return results
    
    def _download_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int
    ) -> Optional[str]:
        """
        Download OHLCV data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            
        Returns:
            Path to saved CSV file or None if failed
        """
        all_data = []
        current_ts = start_ts
        api_calls = 0
        
        # Clean symbol for filename
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        filename = f"{clean_symbol}_{timeframe}_{pd.to_datetime(start_ts, unit='ms').strftime('%Y%m%d')}_{pd.to_datetime(end_ts, unit='ms').strftime('%Y%m%d')}.csv"
        file_path = self.data_dir / "ohlcv" / filename
        
        logger.info(f"Downloading {timeframe} data from {pd.to_datetime(start_ts, unit='ms')} to {pd.to_datetime(end_ts, unit='ms')}")
        
        while current_ts < end_ts and api_calls < self.max_api_calls:
            try:
                # Fetch chunk of data
                chunk = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not chunk:
                    logger.info("No more data available")
                    break
                
                all_data.extend(chunk)
                api_calls += 1
                
                # Update timestamp for next request
                if chunk:
                    current_ts = chunk[-1][0] + self.timeframe_ms[timeframe]
                
                # Rate limiting
                time.sleep(0.1)
                
                logger.info(f"Downloaded {len(chunk)} candles, total: {len(all_data)}, API calls: {api_calls}")
                
            except Exception as e:
                logger.error(f"Error downloading {timeframe} data: {e}")
                time.sleep(1)
                continue
        
        if not all_data:
            logger.warning(f"No OHLCV data downloaded for {symbol} {timeframe}")
            return None
        
        # Process and save data
        df = self._process_ohlcv_data(all_data)
        
        # Filter to date range and remove duplicates
        df = df[(df['timestamp'] >= pd.to_datetime(start_ts, unit='ms')) & 
                (df['timestamp'] <= pd.to_datetime(end_ts, unit='ms'))]
        df = df.drop_duplicates('timestamp').sort_values('timestamp')
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} OHLCV records to {file_path}")
        
        return str(file_path)
    
    def _download_funding_data(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int
    ) -> Optional[str]:
        """
        Download funding rate data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            
        Returns:
            Path to saved CSV file or None if failed
        """
        try:
            # Bybit funding rate endpoint
            funding_data = self.exchange.fetch_funding_rate_history(
                symbol,
                since=start_ts,
                limit=1000
            )
            
            if not funding_data:
                logger.warning(f"No funding data available for {symbol}")
                return None
            
            # Process funding data
            df = pd.DataFrame(funding_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df[['timestamp', 'fundingRate']].copy()
            
            # Filter to date range
            df = df[(df['timestamp'] >= pd.to_datetime(start_ts, unit='ms')) & 
                    (df['timestamp'] <= pd.to_datetime(end_ts, unit='ms'))]
            df = df.drop_duplicates('timestamp').sort_values('timestamp')
            
            # Save to CSV
            clean_symbol = symbol.replace('/', '_').replace(':', '_')
            filename = f"{clean_symbol}_funding_{pd.to_datetime(start_ts, unit='ms').strftime('%Y%m%d')}_{pd.to_datetime(end_ts, unit='ms').strftime('%Y%m%d')}.csv"
            file_path = self.data_dir / "funding" / filename
            
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(df)} funding records to {file_path}")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading funding data for {symbol}: {e}")
            return None
    
    def _process_ohlcv_data(self, raw_data: List) -> pd.DataFrame:
        """
        Process raw OHLCV data into a clean DataFrame.
        
        Args:
            raw_data: List of OHLCV data from exchange
            
        Returns:
            Cleaned DataFrame with OHLCV columns
        """
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(raw_data, columns=cols)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _save_metadata(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: str,
        end_date: str,
        results: Dict
    ) -> str:
        """
        Save metadata about the download session.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            results: Download results
            
        Returns:
            Path to metadata file
        """
        metadata = {
            'download_timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframes': timeframes,
            'start_date': start_date,
            'end_date': end_date,
            'results': results,
            'exchange': 'bybit',
            'data_types': ['ohlcv', 'funding'] if results['funding'] else ['ohlcv']
        }
        
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        filename = f"{clean_symbol}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = self.data_dir / "metadata" / filename
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(file_path)
    
    def batch_download(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: str,
        end_date: str,
        include_funding: bool = True
    ) -> Dict[str, Dict]:
        """
        Download data for multiple symbols in batch.
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_funding: Whether to download funding data
            
        Returns:
            Dict containing results for all symbols
        """
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"Processing symbol: {symbol}")
            try:
                result = self.download_symbol_data(
                    symbol, timeframes, start_date, end_date, include_funding
                )
                all_results[symbol] = result
                
                # Rate limiting between symbols
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to download data for {symbol}: {e}")
                all_results[symbol] = {'error': str(e)}
        
        return all_results
    
    def download_candles(
        self,
        symbol: str,
        timeframe: str,
        num_candles: int,
        include_funding: bool = False
    ) -> Dict[str, str]:
        """
        Download a specific number of candles for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT:USDT")
            timeframe: Data timeframe (e.g., "5m", "1h")
            num_candles: Number of candles to download
            include_funding: Whether to download funding data
            
        Returns:
            Dict containing file paths for downloaded data
        """
        logger.info(f"Downloading {num_candles} {timeframe} candles for {symbol}")
        
        # Calculate start time based on number of candles and timeframe
        if timeframe not in self.timeframe_ms:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Calculate how far back to go based on timeframe and number of candles
        timeframe_ms = self.timeframe_ms[timeframe]
        total_time_ms = num_candles * timeframe_ms
        
        # Get current time and calculate start time
        end_ts = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        start_ts = end_ts - total_time_ms
        
        # Convert to dates for filename
        start_date = pd.to_datetime(start_ts, unit='ms').strftime('%Y%m%d')
        end_date = pd.to_datetime(end_ts, unit='ms').strftime('%Y%m%d')
        
        # Download OHLCV data
        ohlcv_path = self._download_ohlcv_data(symbol, timeframe, start_ts, end_ts)
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'ohlcv': ohlcv_path,
            'candles_downloaded': num_candles
        }
        
        # Download funding data if requested
        if include_funding:
            funding_path = self._download_funding_data(symbol, start_ts, end_ts)
            if funding_path:
                results['funding'] = funding_path
        
        logger.info(f"Download completed: {num_candles} {timeframe} candles for {symbol}")
        return results

    def get_available_data(self) -> Dict[str, List[str]]:
        """
        Get list of available data files.
        
        Returns:
            Dict mapping data types to list of available files
        """
        available = {
            'ohlcv': [],
            'funding': [],
            'metadata': []
        }
        
        for data_type in available.keys():
            data_path = self.data_dir / data_type
            if data_path.exists():
                available[data_type] = [f.name for f in data_path.glob('*.csv') if f.is_file()]
                if data_type == 'metadata':
                    available[data_type] = [f.name for f in data_path.glob('*.json') if f.is_file()]
        
        return available


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download trading data from Bybit')
    parser.add_argument('--mode', choices=['candles', 'date-range'], default='candles', 
                       help='Download mode: candles (specify number) or date-range (specify dates)')
    
    # Candles mode arguments
    parser.add_argument('--symbol', help='Trading symbol (for candles mode)')
    parser.add_argument('--timeframe', help='Data timeframe (for candles mode)')
    parser.add_argument('--candles', type=int, help='Number of candles to download (for candles mode)')
    
    # Date range mode arguments
    parser.add_argument('--symbols', nargs='+', help='Trading symbols to download (for date-range mode)')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '1h'], help='Timeframes to download (for date-range mode)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD) (for date-range mode)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD) (for date-range mode)')
    
    # Common arguments
    parser.add_argument('--no-funding', action='store_true', help='Skip funding data download')
    parser.add_argument('--data-dir', default='data', help='Directory to store data')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DataDownloader(data_dir=args.data_dir)
    
    if args.mode == 'candles':
        # Simple candles mode
        if not all([args.symbol, args.timeframe, args.candles]):
            print("Error: For candles mode, you must specify --symbol, --timeframe, and --candles")
            return
        
        result = downloader.download_candles(
            symbol=args.symbol,
            timeframe=args.timeframe,
            num_candles=args.candles,
            include_funding=not args.no_funding
        )
        
        print("\n" + "="*50)
        print("CANDLES DOWNLOAD SUMMARY")
        print("="*50)
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Candles downloaded: {result['candles_downloaded']}")
        print(f"OHLCV file: {result['ohlcv']}")
        if 'funding' in result:
            print(f"Funding file: {result['funding']}")
    
    else:
        # Date range mode (original functionality)
        if not all([args.symbols, args.start_date, args.end_date]):
            print("Error: For date-range mode, you must specify --symbols, --start-date, and --end-date")
            return
        
        results = downloader.batch_download(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            include_funding=not args.no_funding
        )
        
        print("\n" + "="*50)
        print("DATE RANGE DOWNLOAD SUMMARY")
        print("="*50)
        
        for symbol, result in results.items():
            if 'error' in result:
                print(f"{symbol}: ERROR - {result['error']}")
            else:
                print(f"{symbol}:")
                for timeframe, path in result['timeframes'].items():
                    print(f"  {timeframe}: {path}")
                if result['funding']:
                    print(f"  funding: {result['funding']}")
    
    print(f"\nData stored in: {downloader.data_dir}")
    
    # Show available data
    available = downloader.get_available_data()
    print(f"\nAvailable data files:")
    for data_type, files in available.items():
        if files:
            print(f"  {data_type}: {len(files)} files")


if __name__ == "__main__":
    main()
