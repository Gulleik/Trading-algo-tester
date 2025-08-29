# Trading Strategy Testing Framework

This document explains how to use the comprehensive strategy testing framework for trading algorithm research.

## Overview

The strategy testing framework consists of:

1. **Base Strategy Class** (`strategies/base_strategy.py`) - Abstract base class that all strategies must inherit from
2. **Example Strategy** (`strategies/simple_ma_strategy.py`) - Simple moving average crossover strategy
3. **Strategy Tester** (`strategy_tester.py`) - Main testing framework for backtesting strategies
4. **Performance Metrics** (`metrics.py`) - Advanced performance analysis tools
5. **Example Usage** (`example_usage.py`) - Demonstrates how to use the framework

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Example

```bash
python example_usage.py
```

This will:
- Test the SimpleMA strategy with sample data
- Show available strategies
- Demonstrate plotting and export capabilities

## Creating Your Own Strategy

### Step 1: Inherit from BaseStrategy

Create a new file in the `strategies/` folder:

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name="MyStrategy", **params):
        super().__init__(name, **params)
        # Your strategy parameters here
        
    def generate_signals(self, data):
        # Generate your trading signals
        # Return DataFrame with additional signal columns
        pass
        
    def should_enter_long(self, row, signals):
        # Return True if should enter long position
        pass
        
    def should_exit_long(self, row, signals):
        # Return True if should exit long position
        pass
        
    def should_enter_short(self, row, signals):
        # Return True if should enter short position
        pass
        
    def should_exit_short(self, row, signals):
        # Return True if should exit short position
        pass
```

### Step 2: Implement Required Methods

You must implement these abstract methods:

- `generate_signals()` - Creates trading signals from OHLCV data
- `should_enter_long()` - Determines long entry conditions
- `should_exit_long()` - Determines long exit conditions
- `should_enter_short()` - Determines short entry conditions
- `should_exit_short()` - Determines short exit conditions

### Step 3: Test Your Strategy

```python
from strategy_tester import StrategyTester

# Initialize tester
tester = StrategyTester()

# Test your strategy
result = tester.test_strategy_from_data('MyStrategy', data, **params)

# View results
print(result['metrics'])
```

## Using the Strategy Tester

### Basic Usage

```python
from strategy_tester import StrategyTester

# Initialize with custom parameters
tester = StrategyTester(
    initial_capital=10000.0,
    commission_rate=0.001,  # 0.1%
    slippage=0.0005         # 0.05%
)

# List available strategies
strategies = tester.list_available_strategies()
print(f"Available strategies: {strategies}")

# Test a strategy with sample data
result = tester.test_strategy_from_data('SimpleMAStrategy', data, 
                                      fast_period=10, slow_period=20)

# Test with real exchange data
result = tester.test_strategy_from_exchange('SimpleMAStrategy', 'BTC/USDT:USDT', 
                                          '5m', 1000, fast_period=10, slow_period=20)
```

### Advanced Features

#### Compare Multiple Strategies

```python
# Compare strategies on the same data
comparison = tester.compare_strategies(['SimpleMAStrategy', 'OtherStrategy'], data)
print(comparison)
```

#### Plot Results

```python
# Generate performance plots
tester.plot_results('SimpleMAStrategy', 'my_results.png')
```

#### Export Results

```python
# Export to CSV
tester.export_results('SimpleMAStrategy', 'results.csv')
# Creates: results.csv (trades) and results_metrics.csv (metrics)
```

#### Get Summary

```python
# Get summary of all test results
summary = tester.get_summary()
print(summary)
```

## Performance Metrics

The framework tracks comprehensive performance metrics:

### Core Metrics
- **Total Return** - Overall percentage return
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Ratio of total wins to total losses
- **Sharpe Ratio** - Risk-adjusted return measure
- **Maximum Drawdown** - Largest peak-to-trough decline

### Advanced Metrics
- **Sortino Ratio** - Downside risk-adjusted return
- **Calmar Ratio** - Return vs maximum drawdown
- **Value at Risk (VaR)** - Potential loss at 95% confidence
- **Ulcer Index** - Drawdown severity measure

## Data Requirements

### OHLCV Data Format

Your data must have these columns:
- `timestamp` - Pandas datetime (UTC)
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume

### Example Data Structure

```python
import pandas as pd

data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='5min', tz='UTC'),
    'open': [100.0, 101.0, ...],
    'high': [102.0, 103.0, ...],
    'low': [99.0, 100.0, ...],
    'close': [101.0, 102.0, ...],
    'volume': [1000, 1500, ...]
})
```

## Configuration

### Strategy Parameters

Configure strategy behavior through parameters:

```python
# Test with different parameters
result = tester.test_strategy_from_data('SimpleMAStrategy', data,
                                      fast_period=5,      # Fast MA period
                                      slow_period=20,     # Slow MA period
                                      ma_type='ema',      # MA type: 'sma', 'ema', 'wma'
                                      leverage=2.0,       # Leverage multiplier
                                      commission_rate=0.001,  # Commission rate
                                      slippage=0.0005)    # Slippage rate
```

### Risk Management

The framework includes built-in risk management:

- **Position Sizing** - Automatic calculation based on risk per trade
- **Leverage Support** - Configurable leverage with margin considerations
- **Fee Calculation** - Includes commission and slippage costs
- **Drawdown Tracking** - Real-time drawdown monitoring

## Best Practices

### 1. Strategy Design
- Keep strategies simple and focused
- Use clear entry/exit conditions
- Implement proper risk management
- Test with sufficient historical data

### 2. Backtesting
- Use realistic commission and slippage rates
- Test across different market conditions
- Validate results with out-of-sample data
- Consider transaction costs impact

### 3. Performance Analysis
- Look beyond total returns
- Consider risk-adjusted metrics
- Analyze drawdown characteristics
- Examine trade distribution patterns

## Troubleshooting

### Common Issues

1. **No strategies found**
   - Ensure strategies folder exists
   - Check that strategy files inherit from BaseStrategy
   - Verify Python import paths

2. **Import errors**
   - Install required dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

3. **Data format errors**
   - Verify OHLCV column names
   - Ensure timestamp is UTC datetime
   - Check for missing or invalid data

4. **Performance issues**
   - Reduce data size for testing
   - Use sample data for development
   - Optimize strategy logic

### Getting Help

1. Check the example usage script
2. Review strategy base class implementation
3. Verify data format requirements
4. Check console error messages

## Extending the Framework

### Adding New Metrics

Extend the `metrics.py` module:

```python
def calculate_custom_metric(trades, equity_curve):
    # Your custom calculation
    return metric_value
```

### Custom Plotting

Modify the plotting functions in `strategy_tester.py`:

```python
def plot_custom_analysis(self, strategy_name):
    # Your custom plotting logic
    pass
```

### Integration with Other Tools

The framework can be integrated with:
- **Optuna** - Parameter optimization
- **Streamlit** - Web-based interface
- **Jupyter** - Interactive analysis
- **External databases** - Historical data storage

## Example Strategies

### Simple Moving Average Crossover

The included `SimpleMAStrategy` demonstrates:
- Signal generation from technical indicators
- Parameter configuration
- Entry/exit logic implementation
- Performance optimization capabilities

### Creating Your Own

Use the SimpleMA strategy as a template:
1. Copy the file structure
2. Modify the signal generation logic
3. Adjust entry/exit conditions
4. Add your own parameters
5. Test with the framework

## Conclusion

This framework provides a solid foundation for trading strategy research. Key benefits:

- **Modular Design** - Easy to add new strategies
- **Comprehensive Testing** - Full backtesting capabilities
- **Rich Metrics** - Detailed performance analysis
- **Flexible Configuration** - Customizable parameters
- **Real Data Support** - Exchange integration ready

Start with the example usage script to understand the framework, then create your own strategies following the established patterns.
