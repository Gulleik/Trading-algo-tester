# =========================================================
# Trading Strategy Research Configuration
# =========================================================

# Exchange configuration
EXCHANGE_NAME = "bybit"

# Data fetching configuration
DEFAULT_TIMEFRAME = "5m"
DEFAULT_CANDLE_LIMIT = 50000

# Backtesting configuration
INITIAL_CAPITAL = 10000.0
MAKER_FEE = 0.0002  # 0.02% - for placing positions/partial positions
TAKER_FEE = 0.00055  # 0.055% - for stop loss and breakeven hits
COMMISSION_RATE = 0.001  # 0.1% - legacy fallback
SLIPPAGE = 0.0005  # 0.05%

