# Trading Strategy Research Project

A Python-based trading strategy research and backtesting framework focused on cryptocurrency markets.

## Features

- **Flexible Configuration**: Easy-to-modify Python configuration file
- **Multi-Exchange Support**: Bybit, Binance, OKX, KuCoin, Coinbase
- **Multiple Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Symbol Management**: Organized by exchange and contract type
- **Data Storage**: Parquet/CSV output with configurable naming
- **Rate Limiting**: Built-in API rate limiting and error handling

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses `config.py` for all configuration settings. You can easily modify:

### Exchange Settings
```python
EXCHANGE_NAME = "bybit"  # Change to binance, okx, etc.
EXCHANGE_TESTNET = False  # Set to True for testing
```

### Symbols and Timeframes
Add or modify symbols in the `SYMBOLS_CONFIG` dictionary:

```python
SYMBOLS_CONFIG = {
    "bybit_perp": [
        {
            "symbol": "BTC/USDT:USDT",
            "name": "Bitcoin Perpetual",
            "timeframes": ["5m", "15m", "1h", "4h"],
            "data_length": 2000
        }
    ]
}
```

### Data Storage
```python
OUTPUT_DIR = "data"           # Output directory
FILE_FORMAT = "parquet"       # parquet or csv
COMPRESSION = "snappy"        # Compression for parquet files
```

## Usage

### Basic Data Fetching
```python
from config import EXCHANGE_NAME, get_symbol_by_name
from main import get_exchange_instance, fetch_ohlcv_all

# Get exchange instance
exchange = get_exchange_instance()

# Fetch data for a specific symbol
symbol = "BTC/USDT:USDT"
timeframe = "5m"
df = fetch_ohlcv_all(exchange, symbol, timeframe)
```

### Configuration Helper Functions
```python
from config import (
    get_all_symbols,
    get_symbols_by_category,
    get_symbol_by_name,
    get_timeframes_for_symbol,
    get_data_length_for_symbol
)

# Get all symbols
all_symbols = get_all_symbols()

# Get symbols by category
bybit_symbols = get_symbols_by_category("bybit_perp")

# Get symbol configuration
btc_config = get_symbol_by_name("BTC/USDT:USDT")

# Get available timeframes for a symbol
timeframes = get_timeframes_for_symbol("BTC/USDT:USDT")

# Get data length for a symbol
data_length = get_data_length_for_symbol("BTC/USDT:USDT")
```

## Project Structure

```
Trading-algos/
├── config.py          # Configuration file
├── main.py            # Main data fetching logic
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Adding New Symbols

1. Open `config.py`
2. Add your symbol to the appropriate category in `SYMBOLS_CONFIG`
3. Specify the symbol format, timeframes, and data length
4. The system will automatically validate and use your configuration

Example:
```python
{
    "symbol": "ADA/USDT:USDT",
    "name": "Cardano Perpetual",
    "base_currency": "ADA",
    "quote_currency": "USDT",
    "contract_type": "perpetual",
    "timeframes": ["5m", "15m", "1h"],
    "data_length": 1500
}
```

## Supported Exchanges

- **Bybit**: Perpetual futures and spot
- **Binance**: Spot and futures
- **OKX**: Spot and futures
- **KuCoin**: Spot and futures
- **Coinbase**: Spot trading

## Timeframes

- 1m (1 minute)
- 5m (5 minutes)
- 15m (15 minutes)
- 30m (30 minutes)
- 1h (1 hour)
- 4h (4 hours)
- 1d (1 day)

## Data Format

All data is returned as pandas DataFrames with columns:
- `timestamp`: UTC timestamp (index)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## Rate Limiting

The system includes built-in rate limiting to respect exchange API limits:
- Configurable delay between requests
- Maximum API call limits
- Error handling for rate limit violations

## Output Files

Data is saved with descriptive filenames:
```
{symbol}_{timeframe}_{start_date}_{end_date}.{ext}
```

Example: `BTC_USDT_USDT_5m_2023-01-01_2023-12-31.parquet`

## Contributing

1. Follow PEP8 style guidelines
2. Add type hints and docstrings
3. Keep functions small and testable
4. Update configuration as needed

## License

This project is for educational and research purposes only.
