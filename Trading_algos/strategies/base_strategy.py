"""
Base strategy class for trading algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import uuid


class PositionType(Enum):
    """Position types for trading."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(Enum):
    """Order types for position management."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PARTIAL_CLOSE = "partial_close"
    FULL_CLOSE = "full_close"


@dataclass
class Order:
    """Represents a single order (buy/sell action)."""
    order_id: str
    timestamp: pd.Timestamp
    order_type: OrderType
    price: float
    size: float
    position_id: Optional[str] = None  # Links order to position
    fees: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0  # PnL for this specific order
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())[:8]


@dataclass
class Position:
    """Represents a trading position (can have multiple orders)."""
    position_id: str
    position_type: PositionType
    entry_time: pd.Timestamp
    entry_price: float
    initial_size: float
    current_size: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    average_exit_price: Optional[float] = None
    total_pnl: float = 0.0
    total_fees: float = 0.0
    orders: List[Order] = field(default_factory=list)
    is_closed: bool = False
    
    def __post_init__(self):
        if not self.position_id:
            self.position_id = str(uuid.uuid4())[:8]
        if self.current_size == 0.0:
            self.current_size = self.initial_size
    
    def add_order(self, order: Order) -> None:
        """Add an order to this position."""
        order.position_id = self.position_id
        self.orders.append(order)
        self.total_fees += order.fees
        
        # Update position based on order type
        if order.order_type in [OrderType.MARKET_SELL, OrderType.PARTIAL_CLOSE, 
                               OrderType.FULL_CLOSE, OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            self.current_size -= order.size
            self.total_pnl += order.pnl
            
            # Check if position is fully closed
            if self.current_size <= 0.001:  # Small threshold for floating point errors
                self.is_closed = True
                self.exit_time = order.timestamp
                self._calculate_average_exit_price()
    
    def _calculate_average_exit_price(self) -> None:
        """Calculate the average exit price from all exit orders."""
        exit_orders = [o for o in self.orders if o.order_type in [
            OrderType.MARKET_SELL, OrderType.PARTIAL_CLOSE, OrderType.FULL_CLOSE,
            OrderType.STOP_LOSS, OrderType.TAKE_PROFIT
        ]]
        
        if exit_orders:
            total_value = sum(o.price * o.size for o in exit_orders)
            total_size = sum(o.size for o in exit_orders)
            self.average_exit_price = total_value / total_size if total_size > 0 else None
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for remaining position size."""
        if self.is_closed or self.current_size <= 0:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (current_price - self.entry_price) * self.current_size
        else:  # SHORT
            return (self.entry_price - current_price) * self.current_size


# Legacy Trade class for backward compatibility
@dataclass
class Trade:
    """Legacy trade class - kept for backward compatibility."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    position: PositionType = PositionType.FLAT
    size: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, initial_capital: float = 10000.0, 
                 leverage: float = 1.0, commission_rate: float = 0.001,
                 slippage: float = 0.0005, timeframe: str = "5m"):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name for identification
            initial_capital: Starting capital in quote currency
            leverage: Leverage multiplier for position sizing
            commission_rate: Commission rate as decimal (0.001 = 0.1%)
            slippage: Slippage rate as decimal (0.0005 = 0.05%)
            timeframe: Data timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
        """
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.timeframe = timeframe
        
                # Strategy state - maintaining legacy compatibility
        self.position = PositionType.FLAT  # Legacy compatibility
        self.entry_price = 0.0
        self.entry_time = None
        self.current_size = 0.0
        
        # Partial position management (legacy compatibility)
        self.original_position_size = 0.0
        self.remaining_position_size = 0.0
        
        # Enhanced position/order tracking (new system)
        self.current_position_type = PositionType.FLAT
        self.current_position: Optional[Position] = None
        self.positions: List[Position] = []
        self.orders: List[Order] = []
        
        # Legacy trade history
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
        # Performance tracking
        self.equity_curve: List[float] = [initial_capital]
        self.peak_capital = initial_capital
    
    def _timeframe_to_minutes(self) -> int:
        """Convert timeframe string to minutes for exposure calculations."""
        timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        return timeframe_map.get(self.timeframe, 5)  # Default to 5 minutes if unknown
        
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
    
    def enter_long(self, timestamp: pd.Timestamp, price: float, size: float) -> str:
        """Enter a long position. Returns position_id."""
        if self.position != PositionType.FLAT:
            return ""  # Already in a position
            
        # Calculate fees and slippage
        entry_value = price * size
        fees = entry_value * self.commission_rate
        slippage_cost = entry_value * self.slippage
        
        # Update legacy state (for compatibility)
        self.position = PositionType.LONG
        self.entry_price = price
        self.entry_time = timestamp
        self.current_size = size
        self.original_position_size = size
        self.remaining_position_size = size
        
        # Create legacy trade record
        self.current_trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            position=PositionType.LONG,
            size=size,
            fees=fees
        )
        
        # Create new position for enhanced tracking
        position = Position(
            position_id=str(uuid.uuid4())[:8],
            position_type=PositionType.LONG,
            entry_time=timestamp,
            entry_price=price,
            initial_size=size,
            current_size=size
        )
        
        # Create entry order
        entry_order = Order(
            order_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            order_type=OrderType.MARKET_BUY,
            price=price,
            size=size,
            fees=fees,
            slippage=slippage_cost
        )
        
        # Link order to position
        position.add_order(entry_order)
        
        # Update enhanced state
        self.current_position_type = PositionType.LONG
        self.current_position = position
        self.positions.append(position)
        self.orders.append(entry_order)
        
        # Update capital
        self.current_capital -= fees + slippage_cost
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        return position.position_id
    
    def exit_long(self, timestamp: pd.Timestamp, price: float, size: Optional[float] = None) -> str:
        """Exit a long position (fully or partially). Returns order_id."""
        if self.position != PositionType.LONG or not self.current_trade:
            return ""
        
        # Default to closing entire remaining position
        if size is None:
            size = self.current_size
        
        # Ensure we don't close more than available
        size = min(size, self.current_size)
        
        if size <= 0:
            return ""
            
        # Calculate PnL and costs
        exit_value = price * size
        entry_value = self.entry_price * size
        pnl = exit_value - entry_value
        
        fees = exit_value * self.commission_rate
        slippage_cost = exit_value * self.slippage
        self.current_trade.pnl  += pnl
        self.current_trade.fees += fees
        # Determine if this is a full or partial close
        is_full_close = size >= self.current_size - 0.001
        order_type = OrderType.FULL_CLOSE if is_full_close else OrderType.PARTIAL_CLOSE
        
        # Create exit order for enhanced tracking
        exit_order = Order(
            order_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            order_type=order_type,
            price=price,
            size=size,
            fees=fees,
            slippage=slippage_cost,
            pnl=pnl
        )
        
        # Update enhanced tracking
        if self.current_position:
            self.current_position.add_order(exit_order)
        self.orders.append(exit_order)
        
        # Update legacy state
        self.current_size -= size
        
        # Update capital
        self.current_capital += pnl - fees - slippage_cost
        
        # Check if position is fully closed
        if is_full_close or self.current_size <= 0.001:
            # Update legacy trade record
            self.current_trade.exit_time  = timestamp
            self.current_trade.exit_price = price
            
            # Reset legacy state
            self.position = PositionType.FLAT
            self.entry_price = 0.0
            self.entry_time = None
            self.current_size = 0.0
            
            # Store completed trade
            self.trades.append(self.current_trade)
            self.current_trade = None
            
            # Reset enhanced state
            self.current_position_type = PositionType.FLAT
            if self.current_position:
                self.current_position.is_closed = True
                self.current_position.exit_time = timestamp
            self.current_position = None
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        return exit_order.order_id
    
    def enter_short(self, timestamp: pd.Timestamp, price: float, size: float) -> str:
        """Enter a short position. Returns position_id."""
        if self.position != PositionType.FLAT:
            return ""  # Already in a position
            
        # Calculate fees and slippage
        entry_value = price * size
        fees = entry_value * self.commission_rate
        slippage_cost = entry_value * self.slippage
        
        # Update legacy state (for compatibility)
        self.position = PositionType.SHORT
        self.entry_price = price
        self.entry_time = timestamp
        self.current_size = size
        self.original_position_size = size
        self.remaining_position_size = size
        
        # Create legacy trade record
        self.current_trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            position=PositionType.SHORT,
            size=size,
            fees=fees
        )
        
        # Create new position for enhanced tracking
        position = Position(
            position_id=str(uuid.uuid4())[:8],
            position_type=PositionType.SHORT,
            entry_time=timestamp,
            entry_price=price,
            initial_size=size,
            current_size=size
        )
        
        # Create entry order
        entry_order = Order(
            order_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            order_type=OrderType.MARKET_SELL,
            price=price,
            size=size,
            fees=fees,
            slippage=slippage_cost
        )
        
        # Link order to position
        position.add_order(entry_order)
        
        # Update enhanced state
        self.current_position_type = PositionType.SHORT
        self.current_position = position
        self.positions.append(position)
        self.orders.append(entry_order)
        
        # Update capital
        self.current_capital -= fees + slippage_cost
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        return position.position_id
    
    def exit_short(self, timestamp: pd.Timestamp, price: float, size: Optional[float] = None) -> str:
        """Exit a short position (fully or partially). Returns order_id."""
        if self.position != PositionType.SHORT or not self.current_trade:
            return ""
        
        # Default to closing entire remaining position
        if size is None:
            size = self.current_size
        
        # Ensure we don't close more than available
        size = min(size, self.current_size)
        
        if size <= 0:
            return ""
            
        # Calculate PnL and costs
        exit_value = price * size
        entry_value = self.entry_price * size
        pnl = entry_value - exit_value  # Short: sell high, buy low
        
        fees = exit_value * self.commission_rate
        slippage_cost = exit_value * self.slippage
        self.current_trade.pnl  += pnl
        self.current_trade.fees += fees
        # Determine if this is a full or partial close
        is_full_close = size >= self.current_size - 0.001
        order_type = OrderType.FULL_CLOSE if is_full_close else OrderType.PARTIAL_CLOSE
        
        # Create exit order for enhanced tracking
        exit_order = Order(
            order_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            order_type=order_type,
            price=price,
            size=size,
            fees=fees,
            slippage=slippage_cost,
            pnl=pnl
        )
        
        # Update enhanced tracking
        if self.current_position:
            self.current_position.add_order(exit_order)
        self.orders.append(exit_order)
        
        # Update legacy state
        self.current_size -= size
        
        # Update capital
        self.current_capital += pnl - fees - slippage_cost
        
        # Check if position is fully closed
        if is_full_close or self.current_size <= 0.001:
            # Update legacy trade record
            self.current_trade.exit_time  = timestamp
            self.current_trade.exit_price = price
            
            # Reset legacy state
            self.position = PositionType.FLAT
            self.entry_price = 0.0
            self.entry_time = None
            self.current_size = 0.0
            
            # Store completed trade
            self.trades.append(self.current_trade)
            self.current_trade = None
            
            # Reset enhanced state
            self.current_position_type = PositionType.FLAT
            if self.current_position:
                self.current_position.is_closed = True
                self.current_position.exit_time = timestamp
            self.current_position = None
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        return exit_order.order_id
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return comprehensive performance metrics (per-bar aware)."""
        # Empty result when no trades at all
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
                'max_consecutive_wins': 0,
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
                'var_95': 0.0,
                'position_metrics': {'total_positions': 0, 'positions_with_partial_exits': 0,
                                    'avg_orders_per_position': 0.0, 'avg_position_duration': 0.0,
                                    'partial_exit_efficiency': 0.0},
                'order_metrics': {'total_orders': 0, 'entry_orders': 0, 'exit_orders': 0,
                                'partial_close_orders': 0, 'stop_loss_orders': 0,
                                'take_profit_orders': 0, 'avg_order_size': 0.0,
                                'order_type_distribution': {}}
            }

        # ---------- Basic P/L tallies ----------
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades  = [t for t in self.trades if t.pnl < 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0

        total_wins   = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        avg_win = (total_wins / len(winning_trades)) if winning_trades else 0.0
        avg_loss = (total_losses / len(losing_trades)) if losing_trades else 0.0
        risk_reward_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # ---------- Risk-adjusted returns (prefer per-bar equity if present) ----------
        PERIODS_PER_YEAR = 252 * 24 * 12  # ~5m bars/year; adjust if your data frequency differs
        use_bar_equity = hasattr(self, 'bar_equity') and self.bar_equity

        if use_bar_equity:
            _, eq_series = zip(*self.bar_equity)
            eq = np.asarray(eq_series, dtype=float)
        else:
            eq = np.asarray(self.equity_curve, dtype=float)

        # Drawdown from equity
        if eq.size > 0:
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / np.where(peak == 0, 1.0, peak)
            max_drawdown = float(np.max(dd)) if dd.size else 0.0
        else:
            max_drawdown = 0.0

        # Per-period returns (guard for length)
        if eq.size > 1:
            returns = np.diff(eq) / eq[:-1]
            mean_ret = float(np.mean(returns))
            std_ret  = float(np.std(returns))
            volatility = std_ret * np.sqrt(PERIODS_PER_YEAR)

            sharpe_ratio = (np.sqrt(PERIODS_PER_YEAR) * mean_ret / std_ret) if std_ret > 0 else 0.0

            downside = returns[returns < 0]
            downside_std = float(np.std(downside)) if downside.size > 0 else 0.0
            sortino_ratio = (np.sqrt(PERIODS_PER_YEAR) * mean_ret / downside_std) if downside_std > 0 else 0.0

            var_95 = float(np.percentile(returns, 5))
            annual_return = mean_ret * PERIODS_PER_YEAR  # linearized; fine for sanity check
        else:
            returns = np.array([])
            volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            var_95 = 0.0
            annual_return = 0.0

        calmar_ratio = (annual_return / max_drawdown) if max_drawdown > 0 else 0.0

        # ---------- Streaks ----------
        max_consecutive_losses = 0
        max_consecutive_wins = 0
        cur_losses = cur_wins = 0
        for t in self.trades:
            if t.pnl < 0:
                cur_losses += 1
                cur_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, cur_losses)
            else:
                cur_wins += 1
                cur_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, cur_wins)

        # ---------- Time & frequency metrics ----------
        # Average holding time
        holding_hours = []
        for t in self.trades:
            if t.exit_time and t.entry_time:
                exit_ts = pd.to_datetime(t.exit_time)
                entry_ts = pd.to_datetime(t.entry_time)
                holding_hours.append((exit_ts - entry_ts).total_seconds() / 3600.0)
        avg_holding_time = float(np.mean(holding_hours)) if holding_hours else 0.0

        # Trades per day + exposure use wall-clock window from first to last trade
        if len(self.trades) > 1:
            first_ts = min(pd.to_datetime(t.entry_time) for t in self.trades if t.entry_time)
            last_ts  = max(pd.to_datetime(t.exit_time)  for t in self.trades if t.exit_time) or first_ts
            days_span = max((last_ts - first_ts).total_seconds() / (24 * 3600), 1e-9)
            trades_per_day = len(self.trades) / days_span

            total_time_in_market = sum(
                max((pd.to_datetime(t.exit_time) - pd.to_datetime(t.entry_time)).total_seconds(), 0.0)
                for t in self.trades if t.exit_time and t.entry_time
            )
            exposure_percentage = (total_time_in_market / max((last_ts - first_ts).total_seconds(), 1.0)) * 100.0
        else:
            trades_per_day = 0.0
            exposure_percentage = 0.0

        # ---------- Costs ----------
        total_fees = sum(t.fees for t in self.trades)
        total_slippage = 0.0  # track separately if you store it per order

        # ---------- Leverage/margin (rough) ----------
        if self.leverage > 1.0:
            avg_pos_val = [t.size * t.entry_price for t in self.trades if t.entry_price > 0]
            margin_utilization = float(np.mean([v / self.initial_capital for v in avg_pos_val])) * self.leverage if avg_pos_val else 0.0
            liquidation_risk = 1.0 / self.leverage
        else:
            margin_utilization = 0.0
            liquidation_risk = 0.0

        # ---------- Largest win/loss ----------
        largest_win  = max((t.pnl for t in winning_trades), default=0.0)
        largest_loss = min((t.pnl for t in losing_trades), default=0.0)

        # ---------- Position & order metrics ----------
        position_metrics = self._calculate_position_metrics()
        order_metrics = self._calculate_order_metrics()

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
            'break_even_point': (total_fees / self.initial_capital) if self.initial_capital > 0 else 0.0,
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
            'var_95': var_95,
            'position_metrics': position_metrics,
            'order_metrics': order_metrics
        }

    
    def _calculate_position_metrics(self) -> Dict[str, Any]:
        """Calculate metrics specific to positions."""
        if not self.positions:
            return {
                'total_positions': 0,
                'positions_with_partial_exits': 0,
                'avg_orders_per_position': 0.0,
                'avg_position_duration': 0.0,
                'partial_exit_efficiency': 0.0
            }
        
        closed_positions = [p for p in self.positions if p.is_closed]
        
        # Count positions with partial exits (more than 2 orders: entry + multiple exits)
        positions_with_partial_exits = sum(1 for p in closed_positions if len(p.orders) > 2)
        
        # Average orders per position
        avg_orders_per_position = sum(len(p.orders) for p in self.positions) / len(self.positions)
        
        # Average position duration
        if closed_positions:
            durations = []
            for pos in closed_positions:
                if pos.exit_time and pos.entry_time:
                    duration = (pos.exit_time - pos.entry_time).total_seconds() / 3600  # hours
                    durations.append(duration)
            avg_position_duration = np.mean(durations) if durations else 0.0
        else:
            avg_position_duration = 0.0
        
        # Partial exit efficiency (PnL from partial exits vs full position PnL)
        partial_exit_pnl = 0.0
        total_position_pnl = 0.0
        for pos in closed_positions:
            partial_orders = [o for o in pos.orders if o.order_type == OrderType.PARTIAL_CLOSE]
            partial_exit_pnl += sum(o.pnl for o in partial_orders)
            total_position_pnl += pos.total_pnl
        
        partial_exit_efficiency = (partial_exit_pnl / total_position_pnl * 100) if total_position_pnl != 0 else 0.0
        
        return {
            'total_positions': len(self.positions),
            'positions_with_partial_exits': positions_with_partial_exits,
            'avg_orders_per_position': avg_orders_per_position,
            'avg_position_duration': avg_position_duration,
            'partial_exit_efficiency': partial_exit_efficiency
        }
    
    def _calculate_order_metrics(self) -> Dict[str, Any]:
        """Calculate metrics specific to orders."""
        if not self.orders:
            return {
                'total_orders': 0,
                'entry_orders': 0,
                'exit_orders': 0,
                'partial_close_orders': 0,
                'stop_loss_orders': 0,
                'take_profit_orders': 0,
                'avg_order_size': 0.0,
                'order_type_distribution': {}
            }
        
        # Count orders by type
        order_counts = {}
        for order_type in OrderType:
            order_counts[order_type.value] = sum(1 for o in self.orders if o.order_type == order_type)
        
        entry_orders = order_counts.get('market_buy', 0) + order_counts.get('market_sell', 0)
        exit_orders = len(self.orders) - entry_orders
        
        # Average order size
        avg_order_size = sum(o.size for o in self.orders) / len(self.orders)
        
        return {
            'total_orders': len(self.orders),
            'entry_orders': entry_orders,
            'exit_orders': exit_orders,
            'partial_close_orders': order_counts.get('partial_close', 0),
            'stop_loss_orders': order_counts.get('stop_loss', 0),
            'take_profit_orders': order_counts.get('take_profit', 0),
            'avg_order_size': avg_order_size,
            'order_type_distribution': order_counts
        }
    
    def get_current_position_info(self) -> Dict[str, Any]:
        """Get information about the current open position."""
        if not self.current_position or self.current_position_type == PositionType.FLAT:
            return {'status': 'flat'}
        
        return {
            'status': 'open',
            'position_id': self.current_position.position_id,
            'position_type': self.current_position.position_type.value,
            'entry_price': self.current_position.entry_price,
            'entry_time': self.current_position.entry_time,
            'initial_size': self.current_position.initial_size,
            'current_size': self.current_position.current_size,
            'total_orders': len(self.current_position.orders),
            'total_fees': self.current_position.total_fees,
            'realized_pnl': self.current_position.total_pnl
        }
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """Get detailed history of all positions."""
        history = []
        for pos in self.positions:
            pos_info = {
                'position_id': pos.position_id,
                'position_type': pos.position_type.value,
                'entry_time': pos.entry_time,
                'entry_price': pos.entry_price,
                'initial_size': pos.initial_size,
                'current_size': pos.current_size,
                'exit_time': pos.exit_time,
                'average_exit_price': pos.average_exit_price,
                'total_pnl': pos.total_pnl,
                'total_fees': pos.total_fees,
                'is_closed': pos.is_closed,
                'order_count': len(pos.orders),
                'orders': [{
                    'order_id': o.order_id,
                    'timestamp': o.timestamp,
                    'order_type': o.order_type.value,
                    'price': o.price,
                    'size': o.size,
                    'fees': o.fees,
                    'pnl': o.pnl
                } for o in pos.orders]
            }
            history.append(pos_info)
        return history
    
    def reset(self) -> None:
        """Reset strategy to initial state."""
        self.current_capital = self.initial_capital
        
        # Reset legacy state
        self.position = PositionType.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.current_size = 0.0
        self.original_position_size = 0.0
        self.remaining_position_size = 0.0
        self.current_trade = None
        
        # Reset enhanced state
        self.current_position_type = PositionType.FLAT
        self.current_position = None
        self.positions = []
        self.orders = []
        
        # Reset shared state
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.peak_capital = self.initial_capital
