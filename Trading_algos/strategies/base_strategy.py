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
        
        # Partial position management
        self.original_position_size = 0.0
        self.remaining_position_size = 0.0
        
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
        """
        Calculate position size based on risk management rules.
        
        Args:
            price: Entry price for the position
            risk_per_trade: Risk per trade as percentage of capital (default: 2%)
            
        Returns:
            Position size in units
        """
        # Default base implementation
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
        self.original_position_size = size
        self.remaining_position_size = size
        
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
        self.original_position_size = size
        self.remaining_position_size = size
        
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
        """Calculate and return comprehensive performance metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'total_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_consecutive_losses': 0,
                'final_capital': self.current_capital,
                'risk_reward_ratio': 0.0,
                'exposure_percentage': 0.0,
                'break_even_point': 0.0,
                'trades_per_day': 0.0,
                'avg_holding_time': 0.0,
                'total_fees': 0.0,
                'total_slippage': 0.0,
                'margin_utilization': 0.0,
                'liquidation_risk': 0.0,
                'calmar_ratio': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'volatility': 0.0,
                'var_95': 0.0
            }
        
        # Basic metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Profit factor and R:R ratio
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average win/loss and R:R ratio
        avg_win = total_wins / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Max drawdown and peak analysis
        drawdowns = []
        peak = self.initial_capital
        for capital in self.equity_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Risk-adjusted returns
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Sortino ratio (downside deviation only)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0
            sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
            
            # Volatility and VaR
            volatility = std_return * np.sqrt(252 * 24 * 12) if self.trades else 0  # Annualized for 5m data
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = volatility = var_95 = 0
        
        # Consecutive wins/losses
        max_consecutive_losses = max_consecutive_wins = current_consecutive_losses = current_consecutive_wins = 0
        for trade in self.trades:
            if trade.pnl < 0:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
        
        # Trade timing and behavior metrics
        if self.trades:
            # Calculate average holding time
            holding_times = []
            for trade in self.trades:
                if trade.exit_time and trade.entry_time:
                    # Convert timestamps to pandas Timestamp objects if they aren't already
                    exit_time = pd.to_datetime(trade.exit_time) if not isinstance(trade.exit_time, pd.Timestamp) else trade.exit_time
                    entry_time = pd.to_datetime(trade.entry_time) if not isinstance(trade.entry_time, pd.Timestamp) else trade.entry_time
                    holding_time = (exit_time - entry_time).total_seconds() / 3600  # hours
                    holding_times.append(holding_time)
            
            avg_holding_time = np.mean(holding_times) if holding_times else 0
            
            # Calculate trades per day (assuming 5-minute data)
            if len(self.trades) > 1:
                first_trade = min(pd.to_datetime(t.entry_time) if not isinstance(t.entry_time, pd.Timestamp) else t.entry_time for t in self.trades)
                last_trade = max(pd.to_datetime(t.exit_time) if not isinstance(t.exit_time, pd.Timestamp) else t.exit_time for t in self.trades if t.exit_time)
                if first_trade and last_trade:
                    days_trading = (last_trade - first_trade).total_seconds() / (24 * 3600)
                    trades_per_day = len(self.trades) / days_trading if days_trading > 0 else 0
                else:
                    trades_per_day = 0
            else:
                trades_per_day = 0
        else:
            avg_holding_time = trades_per_day = 0
        
        # Fee and slippage analysis
        total_fees = sum(t.fees for t in self.trades)
        total_slippage = 0  # Would need to track this separately in a real implementation
        
        # Leverage and margin metrics
        if self.leverage > 1.0:
            # Calculate average position size as percentage of capital
            avg_position_sizes = [t.size * t.entry_price / self.initial_capital for t in self.trades if t.entry_price > 0]
            avg_position_size_pct = np.mean(avg_position_sizes) if avg_position_sizes else 0
            margin_utilization = avg_position_size_pct * self.leverage
            liquidation_risk = 1.0 / self.leverage  # Simplified liquidation risk
        else:
            margin_utilization = liquidation_risk = 0
        
        # Exposure percentage (time in market)
        if self.trades:
            total_time = 0
            for trade in self.trades:
                if trade.exit_time and trade.entry_time:
                    try:
                        # Try to calculate time difference safely
                        if hasattr(trade.exit_time, 'total_seconds') and hasattr(trade.entry_time, 'total_seconds'):
                            total_time += (trade.exit_time - trade.entry_time).total_seconds()
                        else:
                            # Default estimate: 2 hours per trade
                            total_time += 2 * 3600  # 2 hours in seconds
                    except (AttributeError, TypeError):
                        # Default estimate: 2 hours per trade
                        total_time += 2 * 3600  # 2 hours in seconds
            
            # Assuming 5-minute data, calculate total available time
            if len(self.equity_curve) > 1:
                total_available_time = (len(self.equity_curve) - 1) * 5 * 60  # seconds
                exposure_percentage = (total_time / total_available_time) * 100 if total_available_time > 0 else 0
            else:
                exposure_percentage = 0
        else:
            exposure_percentage = 0
        
        # Break-even point (including fees)
        if total_fees > 0:
            break_even_point = total_fees / self.initial_capital
        else:
            break_even_point = 0
        
        # Largest win/loss
        largest_win = max((t.pnl for t in winning_trades), default=0)
        largest_loss = min((t.pnl for t in losing_trades), default=0)
        
        # Calmar ratio (annual return / max drawdown)
        if max_drawdown > 0:
            # Annualize return (assuming 5-minute data)
            annual_return = total_return * (252 * 24 * 12) / len(self.equity_curve) if len(self.equity_curve) > 1 else 0
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_consecutive_losses': max_consecutive_losses,
            'max_consecutive_wins': max_consecutive_wins,
            'final_capital': self.current_capital,
            'risk_reward_ratio': risk_reward_ratio,
            'exposure_percentage': exposure_percentage,
            'break_even_point': break_even_point,
            'trades_per_day': trades_per_day,
            'avg_holding_time': avg_holding_time,
            'total_fees': total_fees,
            'total_slippage': total_slippage,
            'margin_utilization': margin_utilization,
            'liquidation_risk': liquidation_risk,
            'calmar_ratio': calmar_ratio,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'volatility': volatility,
            'var_95': var_95
        }
    
    def reset(self) -> None:
        """Reset strategy to initial state."""
        self.current_capital = self.initial_capital
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.current_size = 0.0
        self.original_position_size = 0.0
        self.remaining_position_size = 0.0
        self.trades = []
        self.current_trade = None
        self.equity_curve = [self.initial_capital]
        self.peak_capital = self.initial_capital
