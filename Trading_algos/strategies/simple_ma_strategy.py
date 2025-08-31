"""
Advanced Simple Moving Average Crossover Strategy with Take Profit Levels and Fixed Stop Loss.

This strategy uses moving average crossovers for entry signals and implements
multiple take profit levels for partial position exits:
- TP1: 0.5% - Exit 40% of position
- TP2: 1.0% - Exit 30% of position  
- TP3: 1.5% - Exit 20% of position
- TP4: 2.0% - Exit 10% of position

Risk Management:
- Fixed stop loss at 0.5% from entry price
- Position sizing to risk exactly 2% of total equity per trade
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pandas import DataFrame
from .base_strategy import BaseStrategy, PositionType, OrderType


class SimpleMAStrategy(BaseStrategy):
    """
    Advanced Simple Moving Average Crossover Strategy with Take Profit Levels and Fixed Stop Loss.
    
    This strategy generates signals based on the crossover of two moving averages
    and implements multiple partial take profit levels for better profit management.
    
    Parameters:
        fast_period: Period for fast moving average
        slow_period: Period for slow moving average
        ma_type: Type of moving average ('sma', 'ema', 'wma')
        use_take_profits: Whether to use take profit levels
        stop_loss_pct: Fixed stop loss percentage (default: 0.5%)
        risk_per_trade: Risk per trade as percentage of equity (default: 2.0%)
    """
    
    def __init__(self, name: str = "SimpleMAStrategy", fast_period: int = 10, 
                 slow_period: int = 20, ma_type: str = 'sma',
                 initial_capital: float = 10000.0, leverage: float = 1.0,
                 commission_rate: float = 0.001, slippage: float = 0.0005,
                 use_take_profits: bool = True, stop_loss_pct: float = 0.5,
                 risk_per_trade: float = 0.02, verbose_trading: bool = False):
        """
        Initialize the Advanced Simple MA Strategy.
        
        Args:
            name: Strategy name
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            initial_capital: Starting capital
            leverage: Leverage multiplier
            commission_rate: Commission rate as decimal
            slippage: Slippage rate as decimal
            use_take_profits: Whether to use take profit levels
            stop_loss_pct: Fixed stop loss percentage (default: 0.5%)
            risk_per_trade: Risk per trade as percentage of equity (default: 2.0%)
            verbose_trading: Whether to print detailed trade entry/exit information
        """
        super().__init__(name, initial_capital, leverage, commission_rate, slippage)
        
        # Strategy parameters
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        self.use_take_profits = use_take_profits
        self.stop_loss_pct = stop_loss_pct
        self.risk_per_trade = risk_per_trade
        self.verbose_trading = verbose_trading
        
        # Take profit levels: (percentage, exit_percentage)
        self.take_profit_levels = [
            (0.5, 0.4),  # TP1: 0.5% price change, exit 40%
            (1.0, 0.3),  # TP2: 1.0% price change, exit 30%
            (1.5, 0.2),  # TP3: 1.5% price change, exit 20%
            (2.0, 0.1),  # TP4: 2.0% price change, exit 10%
        ]
        
        # Track which TP levels have been hit
        self.tp_levels_hit = []
        
        # Validate parameters
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if ma_type not in ['sma', 'ema', 'wma']:
            raise ValueError("MA type must be 'sma', 'ema', or 'wma'")
        if stop_loss_pct <= 0:
            raise ValueError("Stop loss percentage must be positive")
        if risk_per_trade <= 0 or risk_per_trade > 1:
            raise ValueError("Risk per trade must be between 0 and 1")
    
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
    
    def check_take_profits(self, current_price: float, timestamp) -> None:
        """
        Check and execute take profit levels.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        if not self.use_take_profits or self.position == PositionType.FLAT:
            return
        
        entry_price = self.entry_price
        
        for i, (tp_percentage, exit_percentage) in enumerate(self.take_profit_levels):
            # Skip if this TP level was already hit
            if i in self.tp_levels_hit:
                continue
            
            # Check if TP level is hit
            tp_hit = False
            target_price = 0
            
            if self.current_position_type == PositionType.LONG:
                target_price = entry_price * (1 + tp_percentage / 100)
                tp_hit = current_price >= target_price
            elif self.current_position_type == PositionType.SHORT:
                target_price = entry_price * (1 - tp_percentage / 100)
                tp_hit = current_price <= target_price
            
            if tp_hit:
                # Calculate exit size based on original position size
                exit_size = self.original_position_size * exit_percentage
                
                # Ensure we don't exit more than remaining position
                exit_size = min(exit_size, self.current_size)
                
                if exit_size > 0:
                    # Execute partial exit using the new order system
                    if self.position == PositionType.LONG:
                        order_id = self.exit_long(timestamp, current_price, exit_size)
                    else:
                        order_id = self.exit_short(timestamp, current_price, exit_size)
                    
                    # Mark this TP level as hit
                    self.tp_levels_hit.append(i)
                    
                    if self.verbose_trading:
                        print(f"   🎯 TP{i+1} ({tp_percentage}%) exit: {exit_size:.4f} units at {current_price:.2f}, Order: {order_id}")
    

    
    def calculate_position_size(self, price: float) -> float:
        """
        Calculate position size to risk exactly 2% of total equity per trade.
        
        Args:
            price: Entry price for the position
            
        Returns:
            Position size in units
        """
        # Calculate the dollar amount to risk (2% of current equity)
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Calculate position size based on stop loss distance
        # For a 0.5% stop loss, we need to size so that a 0.5% move equals our risk amount
        stop_loss_distance = self.stop_loss_pct / 100.0
        
        # Position size = Risk amount / Stop loss distance
        position_value = risk_amount / stop_loss_distance
        
        # Convert to units
        position_size = position_value / price
        
        return position_size
    
    def check_stop_loss(self, current_price: float, timestamp) -> bool:
        """
        Check if stop loss has been hit.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            True if stop loss was hit and position was closed
        """
        if self.position == PositionType.FLAT:
            return False
        
        entry_price = self.entry_price
        stop_loss_hit = False
        
        if self.position == PositionType.LONG:
            # For long positions, stop loss is below entry price
            stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100.0)
            stop_loss_hit = current_price <= stop_loss_price
        elif self.position == PositionType.SHORT:
            # For short positions, stop loss is above entry price
            stop_loss_price = entry_price * (1 + self.stop_loss_pct / 100.0)
            stop_loss_hit = current_price >= stop_loss_price
        
        if stop_loss_hit:
            if self.verbose_trading:
                print(f"   🛑 STOP LOSS HIT at {current_price:.2f} (entry: {entry_price:.2f})")
            
            # Close the entire position at stop loss price
            if self.position == PositionType.LONG:
                self.exit_long(timestamp, current_price)
            else:
                self.exit_short(timestamp, current_price)
            
            # Reset TP tracking
            self.tp_levels_hit = []
            return True
        
        return False
    
    def get_stop_loss_distance(self, current_price: float) -> float:
        """
        Get the current distance to stop loss as a percentage.
        
        Args:
            current_price: Current market price
            
        Returns:
            Distance to stop loss as percentage (positive = safe, negative = stop loss hit)
        """
        if self.position == PositionType.FLAT:
            return 0.0
        
        entry_price = self.entry_price
        
        if self.position == PositionType.LONG:
            # For long positions, calculate how far above stop loss we are
            stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100.0)
            distance_pct = ((current_price - stop_loss_price) / entry_price) * 100
        else:  # SHORT
            # For short positions, calculate how far below stop loss we are
            stop_loss_price = entry_price * (1 + self.stop_loss_pct / 100.0)
            distance_pct = ((stop_loss_price - current_price) / entry_price) * 100
        
        return distance_pct
    
    def should_enter_long(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """
        Determine if we should enter a long position.
        
        Args:
            row: Current data row
            signals: DataFrame with generated signals
            
        Returns:
            True if should enter long position
        """
        return row.get('long_signal', False) and self.position == PositionType.FLAT
    
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
        return row.get('short_signal', False) and self.position == PositionType.FLAT
    
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
    
    def enter_long(self, timestamp, price: float, size: float) -> str:
        """Enter a long position with take profit initialization."""
        position_id = super().enter_long(timestamp, price, size)
        # Reset TP tracking for new position
        self.tp_levels_hit = []
        if self.verbose_trading:
            print(f"   📈 LONG entry at {price:.2f}, size: {size:.4f}, Position: {position_id}")
        return position_id
    
    def enter_short(self, timestamp, price: float, size: float) -> str:
        """Enter a short position with take profit initialization."""
        position_id = super().enter_short(timestamp, price, size)
        # Reset TP tracking for new position
        self.tp_levels_hit = []
        if self.verbose_trading:
            print(f"   📉 SHORT entry at {price:.2f}, size: {size:.4f}, Position: {position_id}")
        return position_id
    
    def exit_long(self, timestamp, price: float, size=None) -> str:
        """Exit remaining long position."""
        if self.position == PositionType.LONG:
            if self.verbose_trading:
                print(f"   🔚 LONG exit at {price:.2f}, remaining: {self.current_size:.4f}")
            order_id = super().exit_long(timestamp, price, size)
            if self.position == PositionType.FLAT:  # Position fully closed
                self.tp_levels_hit = []
            return order_id
        return ""
    
    def exit_short(self, timestamp, price: float, size=None) -> str:
        """Exit remaining short position."""
        if self.position == PositionType.SHORT:
            if self.verbose_trading:
                print(f"   🔚 SHORT exit at {price:.2f}, remaining: {self.current_size:.4f}")
            order_id = super().exit_short(timestamp, price, size)
            if self.position == PositionType.FLAT:  # Position fully closed
                self.tp_levels_hit = []
            return order_id
        return ""
    
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
            'slippage': self.slippage,
            'use_take_profits': self.use_take_profits,
            'stop_loss_pct': self.stop_loss_pct,
            'risk_per_trade': self.risk_per_trade,
            'verbose_trading': self.verbose_trading
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
        if 'use_take_profits' in params:
            self.use_take_profits = params['use_take_profits']
        if 'stop_loss_pct' in params:
            self.stop_loss_pct = params['stop_loss_pct']
        if 'risk_per_trade' in params:
            self.risk_per_trade = params['risk_per_trade']
        if 'verbose_trading' in params:
            self.verbose_trading = params['verbose_trading']
        
        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if self.ma_type not in ['sma', 'ema', 'wma']:
            raise ValueError("MA type must be 'sma', 'ema', or 'wma'")
        if self.stop_loss_pct <= 0:
            raise ValueError("Stop loss percentage must be positive")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 1:
            raise ValueError("Risk per trade must be between 0 and 1")
