"""
Minimalistic data download utility for fetching trading data from Bybit.
Downloads a specific number of candles for a symbol and timeframe.
"""

import ccxt
import pandas as pd
import time
from typing import Dict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Downloads and stores trading data from Bybit for backtesting.
    Minimalistic design - just download the candles you need.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data downloader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.exchange = ccxt.bybit()
        
        # Create subdirectories
        (self.data_dir / "ohlcv").mkdir(exist_ok=True)
        (self.data_dir / "funding").mkdir(exist_ok=True)
        
        # Supported timeframes and their millisecond equivalents
        self.timeframe_ms = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
    
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
            include_funding: Whether to download funding rate data
            
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
    
    def _download_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int
    ) -> str:
        """
        Download OHLCV data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            
        Returns:
            Path to saved CSV file
        """
        all_data = []
        current_ts = start_ts
        
        # Clean symbol for filename
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        filename = f"{clean_symbol}_{timeframe}_{pd.to_datetime(start_ts, unit='ms').strftime('%Y%m%d')}_{pd.to_datetime(end_ts, unit='ms').strftime('%Y%m%d')}.csv"
        file_path = self.data_dir / "ohlcv" / filename
        
        logger.info(f"Downloading {timeframe} data from {pd.to_datetime(start_ts, unit='ms')} to {pd.to_datetime(end_ts, unit='ms')}")
        
        while current_ts < end_ts:
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
                
                # Update timestamp for next request
                if chunk:
                    current_ts = chunk[-1][0] + self.timeframe_ms[timeframe]
                
                # Rate limiting
                time.sleep(0.1)
                
                logger.info(f"Downloaded {len(chunk)} candles, total: {len(all_data)}")
                
            except Exception as e:
                logger.error(f"Error downloading {timeframe} data: {e}")
                time.sleep(1)
                continue
        
        if not all_data:
            raise ValueError(f"No OHLCV data downloaded for {symbol} {timeframe}")
        
        # Process and save data
        df = self._process_ohlcv_data(all_data)
        
        # Filter to date range and remove duplicates
        start_dt = pd.to_datetime(start_ts, unit='ms', utc=True)
        end_dt = pd.to_datetime(end_ts, unit='ms', utc=True)
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
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
    ) -> str:
        """
        Download funding rate data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            
        Returns:
            Path to saved CSV file
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
            start_dt = pd.to_datetime(start_ts, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_ts, unit='ms', utc=True)
            df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
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
    
    def _process_ohlcv_data(self, raw_data: list) -> pd.DataFrame:
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


def main():
    """Main function - just change the parameters below and run the file."""
    
    print("Starting data download...")
    
    # ========================================
    # CONFIGURE YOUR DOWNLOAD PARAMETERS HERE
    # ========================================
    SYMBOL = "SOLUSDT"      # Trading symbol
    TIMEFRAME = "15m"              # Data timeframe (1m, 3m 5m, 15m, 30m, 1h, 4h, 1d)
    NUM_CANDLES = 100000            # Number of candles to download
    INCLUDE_FUNDING = True       # Set to True to download funding data
    DATA_DIR = "data"             # Directory to store data
    # ========================================
    
    print(f"Downloading {NUM_CANDLES} {TIMEFRAME} candles for {SYMBOL}")
    
    # Initialize downloader
    downloader = DataDownloader(data_dir=DATA_DIR)
    
    # Download data
    result = downloader.download_candles(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        num_candles=NUM_CANDLES,
        include_funding=INCLUDE_FUNDING
    )
    
    # Print summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Symbol: {result['symbol']}")
    print(f"Timeframe: {result['timeframe']}")
    print(f"Candles downloaded: {result['candles_downloaded']}")
    print(f"OHLCV file: {result['ohlcv']}")
    if 'funding' in result:
        print(f"Funding file: {result['funding']}")
    
    print(f"\nData stored in: {downloader.data_dir}")
    print("Download completed successfully!")


if __name__ == "__main__":
    main()
