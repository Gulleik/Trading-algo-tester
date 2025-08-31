"""
Fibonacci Channel Strategy (Bybit perps, leveraged compatible)

Logic:
- Lookback = sensitivity * 10
- Channel: highest(high, lookback), lowest(low, lookback)
- Fib levels: 0.236, 0.382, 0.5, 0.618, 0.786; imba_trend_line = 0.5
- Long trend when close >= fib_0.5 and >= fib_0.236
- Short trend when close <= fib_0.5 and <= fib_0.786
- Entry on trend flip (cross into new trend from the opposite side)
Risk:
- Fixed stop loss % from entry (required, e.g. 0.75 for 0.75%)
- Optional breakeven move after price moves `break_even_target_pct` in favor
- Up to 4 partial TPs with per-target close percentages
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, Position, Trade


class FibonacciChannelStrategy(BaseStrategy):
    def __init__(self, 
                 name: str = "FibonacciChannelStrategy",
                 sensitivity: int = 5,
                 initial_capital: float = 10_000.0,
                 leverage: float = 1.0,
                 commission_rate: float = 0.0006,
                 slippage: float = 0.0002,
                 sl_percent: float = 0.75,
                 fixed_stop: bool = True,
                 break_even_after_tp: int = 0,
                 use_take_profits: bool = True,
                 tp1_pct: float = 0.50, tp1_close: float = 25.0,
                 tp2_pct: float = 1.00, tp2_close: float = 25.0,
                 tp3_pct: float = 1.50, tp3_close: float = 25.0,
                 tp4_pct: float = 2.00, tp4_close: float = 25.0,
                 risk_per_trade: float = 0.02):
        super().__init__(name, initial_capital, leverage, commission_rate, slippage)

        self.sensitivity = int(sensitivity)
        if fixed_stop and sl_percent <= 0:
            raise ValueError("sl_percent must be > 0 when using fixed_stop=True")
        self.sl_percent = float(sl_percent)
        self.fixed_stop = bool(fixed_stop)

        self.break_even_after_tp = max(0, min(4, int(break_even_after_tp)))
        self.use_take_profits = bool(use_take_profits)
        self.risk_per_trade = float(risk_per_trade)

        # TP levels [(move %, fraction of size)]
        self.tp_levels: list[tuple[float, float]] = []
        for pct, close in [
            (tp1_pct, tp1_close),
            (tp2_pct, tp2_close),
            (tp3_pct, tp3_close),
            (tp4_pct, tp4_close),
        ]:
            if pct > 0 and close > 0:
                self.tp_levels.append((float(pct), float(close) / 100.0))

        self._tp_hit_idx: set[int] = set()
        self._breakeven_triggered: bool = False

    # ---------- Signals ----------
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        lookback = max(1, int(self.sensitivity * 10))
        high_lv = df["high"].rolling(lookback, min_periods=lookback).max()
        low_lv = df["low"].rolling(lookback, min_periods=lookback).min()
        rng = (high_lv - low_lv).clip(lower=1e-12)

        df["fib_236"] = high_lv - rng * 0.236
        df["fib_382"] = high_lv - rng * 0.382
        df["fib_500"] = high_lv - rng * 0.5
        df["fib_618"] = high_lv - rng * 0.618
        df["fib_786"] = high_lv - rng * 0.786
        df["imba_trend_line"] = df["fib_500"]

        # Stateful regime detection (matches Pine's regime flip behavior)
        df["long_signal"] = False
        df["short_signal"] = False
        df["trend_long"] = False
        df["trend_short"] = False

        in_long = False
        in_short = False

        for i in range(len(df)):
            c = df["close"].iat[i]
            f236 = df["fib_236"].iat[i]
            f500 = df["fib_500"].iat[i]
            f786 = df["fib_786"].iat[i]

            # Only arm a new long/short when we are NOT already in that regime
            can_long  = (c >= f500) and (c >= f236) and (not in_long)
            can_short = (c <= f500) and (c <= f786) and (not in_short)

            if can_long:
                df.at[df.index[i], "long_signal"] = True
                in_long, in_short = True, False
            elif can_short:
                df.at[df.index[i], "short_signal"] = True
                in_long, in_short = False, True

            df.at[df.index[i], "trend_long"] = in_long
            df.at[df.index[i], "trend_short"] = in_short

        # Exit signals fire on opposite regime flip (mirrors your original intent)
        df["exit_long_signal"] = df["short_signal"]
        df["exit_short_signal"] = df["long_signal"]

        return df


    # ---------- Decision hooks ----------
    def should_enter_long(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        return bool(row.get("long_signal", False) and self.position == Position.FLAT)

    def should_exit_long(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        return bool(row.get("exit_long_signal", False))

    def should_enter_short(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        return bool(row.get("short_signal", False) and self.position == Position.FLAT)

    def should_exit_short(self, row: pd.Series, signals: pd.DataFrame) -> bool:
        return bool(row.get("exit_short_signal", False))

    # ---------- Position sizing ---------- 
    def calculate_position_size_for_direction(self, entry_price: float, is_long: bool,
                                            risk_per_trade: Optional[float] = None,
                                            signals_row: Optional[pd.Series] = None) -> float:
        """
        Calculate position size based on actual stop loss price for the specific trade direction.
        This ensures we risk exactly the specified amount if stop loss is hit.
        """
        rpt = self.risk_per_trade if risk_per_trade is None else float(risk_per_trade)
        risk_amount = self.current_capital * rpt
        
        # Calculate the exact stop loss price for this direction
        sl_price = self._calculate_stop_loss_price(entry_price, is_long, signals_row)
        
        # Calculate stop loss distance
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return 0.0
            
        # Calculate position size so that sl_distance * position_size = risk_amount
        # position_size = risk_amount / sl_distance
        # But we need to account for leverage: position_value = position_size * entry_price
        # So: position_size = (risk_amount / sl_distance) 
        position_size = risk_amount / sl_distance
        
        return max(0.0, position_size)
    
    def _calculate_stop_loss_price(self, entry_price: float, is_long: bool, 
                                  signals_row: Optional[pd.Series] = None) -> float:
        """Calculate the exact stop loss price for a given direction."""
        if self.fixed_stop:
            # Fixed stop: percentage from entry
            sl_frac = self.sl_percent / 100.0
            if is_long:
                return entry_price * (1.0 - sl_frac)
            else:
                return entry_price * (1.0 + sl_frac)
        else:
            # Fibonacci stop: use fib levels
            if signals_row is None:
                # Fallback to fixed calculation
                sl_frac = abs(self.sl_percent) / 100.0 if self.sl_percent != 0 else 0.01
                if is_long:
                    return entry_price * (1.0 - sl_frac)
                else:
                    return entry_price * (1.0 + sl_frac)
            else:
                sl_frac_adj = self.sl_percent / 100.0
                if is_long:
                    fib_786 = signals_row.get('fib_786', entry_price)
                    return fib_786 * (1.0 - sl_frac_adj)
                else:
                    fib_236 = signals_row.get('fib_236', entry_price)  
                    return fib_236 * (1.0 + sl_frac_adj)
                    
    def calculate_position_size(self, price: float, risk_per_trade: Optional[float] = None, 
                               signals_row: Optional[pd.Series] = None) -> float:
        """
        Legacy method for backward compatibility. 
        Uses long direction as default for sizing estimation.
        """
        return self.calculate_position_size_for_direction(price, True, risk_per_trade, signals_row)

    # ---------- Order lifecycle overrides (reset TP flags) ----------
    def enter_long(self, timestamp: pd.Timestamp, price: float, size: float) -> None:
        super().enter_long(timestamp, price, size)
        self._tp_hit_idx.clear()
        self._breakeven_triggered = False

    def enter_short(self, timestamp: pd.Timestamp, price: float, size: float) -> None:
        super().enter_short(timestamp, price, size)
        self._tp_hit_idx.clear()
        self._breakeven_triggered = False

    def exit_long(self, timestamp: pd.Timestamp, price: float) -> None:
        super().exit_long(timestamp, price)
        self._tp_hit_idx.clear()
        self._breakeven_triggered = False

    def exit_short(self, timestamp: pd.Timestamp, price: float) -> None:
        super().exit_short(timestamp, price)
        self._tp_hit_idx.clear()
        self._breakeven_triggered = False

    # ---------- Core lifecycle ----------
    def check_stop_loss(self, row: pd.Series, timestamp: pd.Timestamp) -> bool:
        if self.position == Position.FLAT:
            return False

        ts = pd.to_datetime(timestamp)
        same_bar = (self.entry_time is not None and pd.to_datetime(self.entry_time) == ts)

        entry = float(self.entry_price)
        low = float(row["low"])
        high = float(row["high"])
        open_ = float(row.get("open", np.nan))
        close_ = float(row.get("close", np.nan))

        # --- Breakeven logic (Pine parity): no BE exit on entry bar; require candle body direction ---
        if self._breakeven_triggered:
            if self.position == Position.LONG and not same_bar:
                # For LONG: BE hit only if price revisits entry on a red/flat candle
                if (low <= entry) and not (close_ >= open_):
                    self.exit_long(ts, entry)
                    self._tp_hit_idx.clear()
                    return True
            elif self.position == Position.SHORT and not same_bar:
                # For SHORT: BE hit only if price revisits entry on a green/flat candle
                if (high >= entry) and not (close_ <= open_):
                    self.exit_short(ts, entry)
                    self._tp_hit_idx.clear()
                    return True
            return False

        # --- Regular SL (block same-bar stops like Pine) ---
        sl_frac = self.sl_percent / 100.0
        if self.position == Position.LONG:
            sl_price = entry * (1.0 - sl_frac) if self.fixed_stop else float(row.get("fib_786", entry)) * (1.0 - sl_frac)
            if (low <= sl_price) and not same_bar:
                self.exit_long(ts, sl_price)
                self._tp_hit_idx.clear()
                return True
        else:  # SHORT
            sl_price = entry * (1.0 + sl_frac) if self.fixed_stop else float(row.get("fib_236", entry)) * (1.0 + sl_frac)
            if (high >= sl_price) and not same_bar:
                self.exit_short(ts, sl_price)
                self._tp_hit_idx.clear()
                return True

        return False


    def check_breakeven(self, current_price: float) -> None:
        """Move stop to breakeven after specified TP level is hit.
        This method is now handled within check_take_profits().
        """
        # This method is now deprecated - breakeven logic moved to check_take_profits()
        pass

    def check_take_profits(self, row: pd.Series, timestamp: pd.Timestamp) -> None:
        """Intrabar TP checks (use high/low)."""
        if not self.use_take_profits or self.position == Position.FLAT or self.current_size <= 0:
            return

        entry, low, high = self.entry_price, row["low"], row["high"]

        for idx, (tp_pct, close_frac) in enumerate(self.tp_levels):
            if idx in self._tp_hit_idx:
                continue

            target_hit = False
            target_price = None
            if self.position == Position.LONG:
                target_price = entry * (1.0 + tp_pct / 100.0)
                target_hit = high >= target_price
            else:
                target_price = entry * (1.0 - tp_pct / 100.0)
                target_hit = low <= target_price

            if not target_hit:
                continue

            # Partial close
            base = self.original_position_size or self.current_size
            size_to_exit = min(base * close_frac, self.current_size)
            if size_to_exit <= 0:
                self._tp_hit_idx.add(idx)
                continue

            exit_value = target_price * size_to_exit
            entry_value = entry * size_to_exit
            pnl = (exit_value - entry_value) if self.position == Position.LONG else (entry_value - exit_value)
            fees = exit_value * self.commission_rate
            slip = exit_value * self.slippage

            partial = Trade(
                entry_time=self.entry_time,
                exit_time=timestamp,
                entry_price=entry,
                exit_price=target_price,
                position=self.position,
                size=size_to_exit,
                pnl=pnl,
                fees=fees,
            )
            self.current_size -= size_to_exit
            self.current_capital += pnl - fees - slip
            self.trades.append(partial)
            self.equity_curve.append(self.current_capital)
            self._tp_hit_idx.add(idx)

            # Trigger breakeven if configured
            if (self.break_even_after_tp > 0 and 
                (idx + 1) == self.break_even_after_tp and 
                not self._breakeven_triggered):
                self._breakeven_triggered = True

            # If flat after TP, exit fully
            if self.current_size <= 1e-9:
                if self.position == Position.LONG:
                    self.exit_long(timestamp, target_price)
                else:
                    self.exit_short(timestamp, target_price)
                self._tp_hit_idx.clear()
                break

    # ---------- Risk-Reward helper ----------
    @staticmethod
    def calc_rr(entry: float, stop: float, target: float) -> float:
        """Risk-reward ratio identical to Pine calc_rr."""
        if entry > stop:
            return (target - entry) / (entry - stop)
        else:
            return (entry - target) / (stop - entry)                

    # ---------- Param IO ----------
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "sensitivity": self.sensitivity,
            "leverage": self.leverage,
            "commission_rate": self.commission_rate,
            "slippage": self.slippage,
            "sl_percent": self.sl_percent,
            "fixed_stop": self.fixed_stop,
            "break_even_after_tp": self.break_even_after_tp,
            "use_take_profits": self.use_take_profits,
            "tp_levels": [(pct, frac * 100.0) for (pct, frac) in self.tp_levels],  # as (% move, % close)
            "risk_per_trade": self.risk_per_trade,
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        if "sensitivity" in params:
            self.sensitivity = max(1, int(params["sensitivity"]))
        if "leverage" in params:
            self.leverage = float(params["leverage"])
        if "commission_rate" in params:
            self.commission_rate = float(params["commission_rate"])
        if "slippage" in params:
            self.slippage = float(params["slippage"])
        if "sl_percent" in params:
            v = float(params["sl_percent"])
            if self.fixed_stop and v <= 0:
                raise ValueError("sl_percent must be > 0 when using fixed_stop=True")
            self.sl_percent = v
        if "fixed_stop" in params:
            self.fixed_stop = bool(params["fixed_stop"])
        if "break_even_after_tp" in params:
            self.break_even_after_tp = max(0, min(4, int(params["break_even_after_tp"])))
        if "use_take_profits" in params:
            self.use_take_profits = bool(params["use_take_profits"])
        if "tp_levels" in params:
            # Expect list of (move %, close %) tuples
            self.tp_levels = []
            for pct, close_pct in params["tp_levels"]:
                if pct > 0 and close_pct > 0:
                    self.tp_levels.append((float(pct), float(close_pct) / 100.0))
        if "risk_per_trade" in params:
            v = float(params["risk_per_trade"])
            if not (0 < v <= 1):
                raise ValueError("risk_per_trade must be in (0,1]")
            self.risk_per_trade = v
