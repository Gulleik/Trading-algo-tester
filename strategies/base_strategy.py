"""
Base strategy class for trading algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class Position(Enum):
    """Position types for trading."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    position: Position = Position.FLAT
    size: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, initial_capital: float = 10000.0, 
                 leverage: float = 1.0, commission_rate: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name for identification
            initial_capital: Starting capital in quote currency
            leverage: Leverage multiplier for position sizing
            commission_rate: Commission rate as decimal (0.001 = 0.1%)
            slippage: Slippage rate as decimal (0.0005 = 0.05%)
        """
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # Strategy state
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.current_size = 0.0
        
        # Trade history
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
        # Performance tracking
        self.equity_curve: List[float] = [initial_capital]
        self.peak_capital = initial_capital
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns (timestamp, open, high, low, close, volume)
            
        Returns:
            DataFrame with additional signal columns (e.g., 'signal', 'position')
        """
        pass
    
    @abstractmethod
    def should_enter_long(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """Determine if we should enter a long position."""
        pass
    
    @abstractmethod
    def should_exit_long(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """Determine if we should exit a long position."""
        pass
    
    @abstractmethod
    def should_enter_short(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """Determine if we should enter a short position."""
        pass
    
    @abstractmethod
    def should_exit_short(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        """Determine if we should exit a short position."""
        pass
    
    def calculate_position_size(self, price: float, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management rules."""
        risk_amount = self.current_capital * risk_per_trade
        position_value = risk_amount * self.leverage
        return position_value / price
    
    def enter_long(self, timestamp: pd.Timestamp, price: float, size: float) -> None:
        """Enter a long position."""
        if self.position != Position.FLAT:
            return  # Already in a position
            
        # Calculate fees and slippage
        entry_value = price * size
        fees = entry_value * self.commission_rate
        slippage_cost = entry_value * self.slippage
        
        # Update state
        self.position = Position.LONG
        self.entry_price = price
        self.entry_time = timestamp
        self.current_size = size
        
        # Create trade record
        self.current_trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            position=Position.LONG,
            size=size,
            fees=fees
        )
        
        # Update capital
        self.current_capital -= fees + slippage_cost
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
    
    def exit_long(self, timestamp: pd.Timestamp, price: float) -> None:
        """Exit a long position."""
        if self.position != Position.LONG or not self.current_trade:
            return
            
        # Calculate PnL and costs
        exit_value = price * self.current_size
        entry_value = self.entry_price * self.current_size
        pnl = exit_value - entry_value
        
        fees = exit_value * self.commission_rate
        slippage_cost = exit_value * self.slippage
        
        # Update trade record
        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price = price
        self.current_trade.pnl = pnl
        self.current_trade.fees += fees
        
        # Update state
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.current_size = 0.0
        
        # Update capital
        self.current_capital += pnl - fees - slippage_cost
        
        # Store completed trade
        self.trades.append(self.current_trade)
        self.current_trade = None
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
    
    def enter_short(self, timestamp: pd.Timestamp, price: float, size: float) -> None:
        """Enter a short position."""
        if self.position != Position.FLAT:
            return  # Already in a position
            
        # Calculate fees and slippage
        entry_value = price * size
        fees = entry_value * self.commission_rate
        slippage_cost = entry_value * self.slippage
        
        # Update state
        self.position = Position.SHORT
        self.entry_price = price
        self.entry_time = timestamp
        self.current_size = size
        
        # Create trade record
        self.current_trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            position=Position.SHORT,
            size=size,
            fees=fees
        )
        
        # Update capital
        self.current_capital -= fees + slippage_cost
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
    
    def exit_short(self, timestamp: pd.Timestamp, price: float) -> None:
        """Exit a short position."""
        if self.position != Position.SHORT or not self.current_trade:
            return
            
        # Calculate PnL and costs
        exit_value = price * self.current_size
        entry_value = self.entry_price * self.current_size
        pnl = entry_value - exit_value  # Short: sell high, buy low
        
        fees = exit_value * self.commission_rate
        slippage_cost = exit_value * self.slippage
        
        # Update trade record
        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price = price
        self.current_trade.pnl = pnl
        self.current_trade.fees += fees
        
        # Update state
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.current_size = 0.0
        
        # Update capital
        self.current_capital += pnl - fees - slippage_cost
        
        # Store completed trade
        self.trades.append(self.current_trade)
        self.current_trade = None
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return performance metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_consecutive_losses': 0
            }
        
        # Basic metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average win/loss
        avg_win = total_wins / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0
        
        # Max drawdown
        drawdowns = []
        peak = self.initial_capital
        for capital in self.equity_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Sharpe ratio (simplified - assuming 0 risk-free rate)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Max consecutive losses
        max_consecutive_losses = 0
        current_consecutive = 0
        for trade in self.trades:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_consecutive_losses': max_consecutive_losses,
            'final_capital': self.current_capital
        }
    
    def reset(self) -> None:
        """Reset strategy to initial state."""
        self.current_capital = self.initial_capital
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.current_size = 0.0
        self.trades = []
        self.current_trade = None
        self.equity_curve = [self.initial_capital]
        self.peak_capital = self.initial_capital
