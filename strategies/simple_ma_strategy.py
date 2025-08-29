"""
Simple Moving Average Crossover Strategy.

This is an example implementation of a basic trading strategy that uses
two moving averages to generate buy/sell signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from pandas import DataFrame
from .base_strategy import BaseStrategy, Position


class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    This strategy generates signals based on the crossover of two moving averages:
    - Fast MA (shorter period) crossing above Slow MA (longer period) = BUY
    - Fast MA crossing below Slow MA = SELL
    
    Parameters:
        fast_period: Period for fast moving average
        slow_period: Period for slow moving average
        ma_type: Type of moving average ('sma', 'ema', 'wma')
    """
    
    def __init__(self, name: str = "SimpleMAStrategy", fast_period: int = 10, 
                 slow_period: int = 20, ma_type: str = 'sma',
                 initial_capital: float = 10000.0, leverage: float = 1.0,
                 commission_rate: float = 0.001, slippage: float = 0.0005):
        """
        Initialize the Simple MA Strategy.
        
        Args:
            name: Strategy name
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            initial_capital: Starting capital
            leverage: Leverage multiplier
            commission_rate: Commission rate as decimal
            slippage: Slippage rate as decimal
        """
        super().__init__(name, initial_capital, leverage, commission_rate, slippage)
        
        # Strategy parameters
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        
        # Validate parameters
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if ma_type not in ['sma', 'ema', 'wma']:
            raise ValueError("MA type must be 'sma', 'ema', or 'wma'")
    
    def _calculate_ma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate moving average based on specified type.
        
        Args:
            data: Price series
            period: Moving average period
            
        Returns:
            Moving average series
        """
        if self.ma_type == 'sma':
            return data.rolling(window=period).mean()
        elif self.ma_type == 'ema':
            return data.ewm(span=period).mean()
        elif self.ma_type == 'wma':
            weights = np.arange(1, period + 1)
            return data.rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        else:
            raise ValueError(f"Unsupported MA type: {self.ma_type}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with additional signal columns
        """
        if len(data) < self.slow_period:
            return data
        
        # Calculate moving averages
        close_prices = data['close']
        fast_ma = self._calculate_ma(close_prices, self.fast_period)
        slow_ma = self._calculate_ma(close_prices, self.slow_period)
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['fast_ma'] = fast_ma
        signals['slow_ma'] = slow_ma
        
        # Crossover signals
        signals['ma_cross'] = np.where(fast_ma > slow_ma, 1, -1)
        signals['ma_cross_prev'] = signals['ma_cross'].shift(1)
        
        # Entry signals
        signals['long_signal'] = (signals['ma_cross'] == 1) & (signals['ma_cross_prev'] == -1)
        signals['short_signal'] = (signals['ma_cross'] == -1) & (signals['ma_cross_prev'] == 1)
        
        # Exit signals (opposite of entry)
        signals['exit_long_signal'] = (signals['ma_cross'] == -1) & (signals['ma_cross_prev'] == 1)
        signals['exit_short_signal'] = (signals['ma_cross'] == 1) & (signals['ma_cross_prev'] == -1)
        
        # Combine with original data
        result = pd.concat([data, signals], axis=1)
        
        return result
    
    def should_enter_long(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """
        Determine if we should enter a long position.
        
        Args:
            row: Current data row
            signals: DataFrame with generated signals
            
        Returns:
            True if should enter long position
        """
        return row.get('long_signal', False)
    
    def should_exit_long(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """
        Determine if we should exit a long position.
        
        Args:
            row: Current data row
            signals: DataFrame with generated signals
            
        Returns:
            True if should exit long position
        """
        return row.get('exit_long_signal', False)
    
    def should_enter_short(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """
        Determine if we should enter a short position.
        
        Args:
            row: Current data row
            signals: DataFrame with generated signals
            
        Returns:
            True if should enter short position
        """
        return row.get('short_signal', False)
    
    def should_exit_short(self, row: pd.Series, signals: DataFrame) -> bool:
        """
        Determine if we should exit a short position.
        
        Args:
            row: Current data row
            signals: DataFrame with generated signals
            
        Returns:
            True if should exit short position
        """
        return row.get('exit_short_signal', False)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters for optimization.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'ma_type': self.ma_type,
            'leverage': self.leverage,
            'commission_rate': self.commission_rate,
            'slippage': self.slippage
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            params: Dictionary of parameters to set
        """
        if 'fast_period' in params:
            self.fast_period = params['fast_period']
        if 'slow_period' in params:
            self.slow_period = params['slow_period']
        if 'ma_type' in params:
            self.ma_type = params['ma_type']
        if 'leverage' in params:
            self.leverage = params['leverage']
        if 'commission_rate' in params:
            self.commission_rate = params['commission_rate']
        if 'slippage' in params:
            self.slippage = params['slippage']
        
        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if self.ma_type not in ['sma', 'ema', 'wma']:
            raise ValueError("MA type must be 'sma', 'ema', or 'wma'")
