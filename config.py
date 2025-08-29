# =========================================================
# Trading Strategy Research Configuration
# =========================================================

# Exchange configuration for data fetching only
EXCHANGE_NAME = "bybit"  # Options: bybit, binance, okx, kucoin, etc.
EXCHANGE_TESTNET = False  # Set to true for testing with testnet
EXCHANGE_SANDBOX = False  # Set to true for sandbox environment

# Data fetching configuration
DEFAULT_TIMEFRAME = "5m"
AVAILABLE_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_CANDLE_LIMIT = 1000
MAX_API_CALLS = 100
RATE_LIMIT_DELAY = 0.1  # seconds

# Data storage configuration
OUTPUT_DIR = "data"
FILE_FORMAT = "parquet"  # Options: parquet, csv
COMPRESSION = "snappy"
INCLUDE_METADATA = True
FILENAME_PATTERN = "{symbol}_{timeframe}_{start_date}_{end_date}.{ext}"

# Backtesting configuration
DEFAULT_START_DATE = "2023-01-01"  # ISO format: YYYY-MM-DD
DEFAULT_END_DATE = ""  # Leave empty for current date
INITIAL_CAPITAL = 10000.0
COMMISSION_RATE = 0.1  # percentage
SLIPPAGE = 0.05  # percentage

# Logging configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "trading_logs.log"
LOG_CONSOLE = True

# Symbols configuration
SYMBOLS_CONFIG = {
    # Bybit perpetual futures
    "bybit_perp": [
        {
            "symbol": "BTC/USDT:USDT",
            "name": "Bitcoin Perpetual",
            "base_currency": "BTC",
            "quote_currency": "USDT",
            "contract_type": "perpetual",
            "timeframes": ["5m", "15m", "1h", "4h"],
            "data_length": 2000
        },
        {
            "symbol": "ETH/USDT:USDT",
            "name": "Ethereum Perpetual",
            "base_currency": "ETH",
            "quote_currency": "USDT",
            "contract_type": "perpetual",
            "timeframes": ["5m", "15m", "1h", "4h"],
            "data_length": 2000
        },
        {
            "symbol": "SOL/USDT:USDT",
            "name": "Solana Perpetual",
            "base_currency": "SOL",
            "quote_currency": "USDT",
            "contract_type": "perpetual",
            "timeframes": ["5m", "15m", "1h"],
            "data_length": 1500
        }
    ],
    
    # Binance spot (example)
    "binance_spot": [
        {
            "symbol": "BTC/USDT",
            "name": "Bitcoin Spot",
            "base_currency": "BTC",
            "quote_currency": "USDT",
            "contract_type": "spot",
            "timeframes": ["5m", "15m", "1h", "4h", "1d"],
            "data_length": 3000
        },
        {
            "symbol": "ETH/USDT",
            "name": "Ethereum Spot",
            "base_currency": "ETH",
            "quote_currency": "USDT",
            "contract_type": "spot",
            "timeframes": ["5m", "15m", "1h", "4h", "1d"],
            "data_length": 3000
        }
    ]
}

# Helper functions for easy access to configuration
def get_all_symbols():
    """Get all symbols from all categories."""
    all_symbols = []
    for category_symbols in SYMBOLS_CONFIG.values():
        all_symbols.extend(category_symbols)
    return all_symbols

def get_symbols_by_category(category: str):
    """Get all symbols for a specific category."""
    return SYMBOLS_CONFIG.get(category, [])

def get_symbol_by_name(symbol_name: str):
    """Find a symbol configuration by symbol name."""
    for category_symbols in SYMBOLS_CONFIG.values():
        for symbol_config in category_symbols:
            if symbol_config["symbol"] == symbol_name:
                return symbol_config
    return None

def get_timeframes_for_symbol(symbol_name: str):
    """Get available timeframes for a specific symbol."""
    symbol_config = get_symbol_by_name(symbol_name)
    if symbol_config:
        return symbol_config["timeframes"]
    return []

def get_data_length_for_symbol(symbol_name: str):
    """Get data length for a specific symbol."""
    symbol_config = get_symbol_by_name(symbol_name)
    if symbol_config:
        return symbol_config["data_length"]
    return DEFAULT_CANDLE_LIMIT

def validate_timeframe(timeframe: str) -> bool:
    """Validate if a timeframe is supported."""
    return timeframe in AVAILABLE_TIMEFRAMES

def get_filename(symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
    """Generate filename based on configuration pattern."""
    ext = FILE_FORMAT
    filename = FILENAME_PATTERN.format(
        symbol=symbol.replace('/', '_').replace(':', '_'),
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        ext=ext
    )
    return filename

