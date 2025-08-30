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
    def __init__(
        self,
        name: str = "FibonacciChannelStrategy",
        sensitivity: int = 5,                 # same meaning as your Pine; internally x10
        initial_capital: float = 10_000.0,
        leverage: float = 1.0,
        commission_rate: float = 0.0006,      # 0.06% typical taker
        slippage: float = 0.0002,             # 0.02% assumed
        # Risk & exits
        stop_loss_pct: float = 0.75,          # % from entry (e.g., 0.75 = 0.75%)
        break_even_target_pct: float = 0.0,   # % move to move SL to BE (0 disables)
        use_take_profits: bool = True,
        # TP targets (% move) and how much of original position to exit at each
        tp1_pct: float = 0.50, tp1_close: float = 25.0,
        tp2_pct: float = 1.00, tp2_close: float = 25.0,
        tp3_pct: float = 1.50, tp3_close: float = 25.0,
        tp4_pct: float = 2.00, tp4_close: float = 25.0,
        # Sizing
        risk_per_trade: float = 0.02          # 2% of equity risked via SL distance
    ):
        super().__init__(name, initial_capital, leverage, commission_rate, slippage)
        # Core params
        if sensitivity < 1:
            raise ValueError("sensitivity must be >= 1")
        self.sensitivity = int(sensitivity)

        if stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be > 0")
        self.stop_loss_pct = float(stop_loss_pct)

        self.break_even_target_pct = max(0.0, float(break_even_target_pct))
        self.use_take_profits = bool(use_take_profits)
        self.risk_per_trade = float(risk_per_trade)

        # Prepare TP configuration (percent as %)
        self.tp_levels: List[tuple[float, float]] = []
        for pct, close in [
            (tp1_pct, tp1_close),
            (tp2_pct, tp2_close),
            (tp3_pct, tp3_close),
            (tp4_pct, tp4_close),
        ]:
            if pct > 0 and close > 0:
                self.tp_levels.append((float(pct), float(close) / 100.0))  # (move %, close fraction)

        # Runtime tracking
        self._tp_hit_idx: set[int] = set()

    # ---------- Signals ----------
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds:
          - fib_* columns and imba_trend_line
          - trend_long, trend_short (bool)
          - long_signal, short_signal (flip into trend)
          - exit_long_signal, exit_short_signal (flip out)
        """
        df = data.copy()

        lookback = max(1, int(self.sensitivity * 10))
        high_lv = df["high"].rolling(lookback, min_periods=lookback).max()
        low_lv = df["low"].rolling(lookback, min_periods=lookback).min()
        rng = (high_lv - low_lv).clip(lower=1e-12)

        fib_236 = high_lv - rng * 0.236
        fib_382 = high_lv - rng * 0.382
        fib_500 = high_lv - rng * 0.5
        fib_618 = high_lv - rng * 0.618
        fib_786 = high_lv - rng * 0.786

        df["fib_236"] = fib_236
        df["fib_382"] = fib_382
        df["fib_500"] = fib_500
        df["fib_618"] = fib_618
        df["fib_786"] = fib_786
        df["imba_trend_line"] = fib_500

        c = df["close"]
        trend_long = (c >= fib_500) & (c >= fib_236)
        trend_short = (c <= fib_500) & (c <= fib_786)

        # Trend flip detection
        trend_long_prev = trend_long.shift(1, fill_value=False)
        trend_short_prev = trend_short.shift(1, fill_value=False)

        df["trend_long"] = trend_long
        df["trend_short"] = trend_short

        df["long_signal"] = trend_long & (~trend_long_prev)  # start of long trend
        df["short_signal"] = trend_short & (~trend_short_prev)  # start of short trend

        # Exits: flip to opposite trend
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
    def calculate_position_size(self, price: float, risk_per_trade: Optional[float] = None) -> float:
        """
        Risk-based sizing using fixed stop distance.
        """
        rpt = self.risk_per_trade if risk_per_trade is None else float(risk_per_trade)
        risk_amount = self.current_capital * rpt
        sl_frac = self.stop_loss_pct / 100.0
        if sl_frac <= 0:
            return 0.0
        position_value = risk_amount / sl_frac
        units = (position_value * self.leverage) / max(price, 1e-12)
        return max(0.0, units)

    # ---------- Order lifecycle overrides (reset TP flags) ----------
    def enter_long(self, timestamp: pd.Timestamp, price: float, size: float) -> None:
        super().enter_long(timestamp, price, size)
        self._tp_hit_idx.clear()

    def enter_short(self, timestamp: pd.Timestamp, price: float, size: float) -> None:
        super().enter_short(timestamp, price, size)
        self._tp_hit_idx.clear()

    def exit_long(self, timestamp: pd.Timestamp, price: float) -> None:
        super().exit_long(timestamp, price)
        self._tp_hit_idx.clear()

    def exit_short(self, timestamp: pd.Timestamp, price: float) -> None:
        super().exit_short(timestamp, price)
        self._tp_hit_idx.clear()

    # ---------- Risk management helpers (to be called per-bar by your runner) ----------
    def check_stop_loss(self, current_price: float, timestamp: pd.Timestamp) -> bool:
        if self.position == Position.FLAT:
            return False

        entry = self.entry_price
        sl_frac = self.stop_loss_pct / 100.0
        if self.position == Position.LONG:
            sl_price = entry * (1.0 - sl_frac)
            hit = current_price <= sl_price
        else:
            sl_price = entry * (1.0 + sl_frac)
            hit = current_price >= sl_price

        if hit:
            if self.position == Position.LONG:
                self.exit_long(timestamp, current_price)
            else:
                self.exit_short(timestamp, current_price)
            self._tp_hit_idx.clear()
            return True
        return False

    def check_breakeven(self, current_price: float) -> None:
        """Move stop to breakeven (conceptual). Here we simulate by shrinking risk:
        if BE is hit we reduce effective stop distance to entry, so further partial exits are safer.
        In this minimal version, we don't store an explicit SL price; we just no-op if disabled.
        """
        if self.break_even_target_pct <= 0 or self.position == Position.FLAT:
            return

        move_frac = self.break_even_target_pct / 100.0
        if self.position == Position.LONG and current_price >= self.entry_price * (1 + move_frac):
            # Conceptually move SL to entry (handled by your runner if you implement price-based SL storage)
            pass
        elif self.position == Position.SHORT and current_price <= self.entry_price * (1 - move_frac):
            pass

    def check_take_profits(self, current_price: float, timestamp: pd.Timestamp) -> None:
        """Execute partial exits on configured TP levels relative to original size."""
        if not self.use_take_profits or self.position == Position.FLAT or self.current_size <= 0:
            return

        entry = self.entry_price
        for idx, (tp_pct, close_frac) in enumerate(self.tp_levels):
            if idx in self._tp_hit_idx:
                continue

            target_hit = False
            if self.position == Position.LONG:
                target_hit = current_price >= entry * (1.0 + tp_pct / 100.0)
            else:
                target_hit = current_price <= entry * (1.0 - tp_pct / 100.0)

            if not target_hit:
                continue

            # Exit portion based on original size
            base = self.original_position_size if self.original_position_size > 0 else self.current_size
            size_to_exit = min(base * close_frac, self.current_size)
            if size_to_exit <= 0:
                self._tp_hit_idx.add(idx)
                continue

            # Realize PnL on the exited slice
            exit_value = current_price * size_to_exit
            entry_value = entry * size_to_exit
            pnl = (exit_value - entry_value) if self.position == Position.LONG else (entry_value - exit_value)

            fees = exit_value * self.commission_rate
            slip = exit_value * self.slippage

            partial = Trade(
                entry_time=self.entry_time,
                exit_time=timestamp,
                entry_price=entry,
                exit_price=current_price,
                position=self.position,
                size=size_to_exit,
                pnl=pnl,
                fees=fees,
            )
            self.current_size -= size_to_exit
            if hasattr(self, "remaining_position_size"):
                self.remaining_position_size = max(0.0, self.remaining_position_size - size_to_exit)

            self.current_capital += pnl - fees - slip
            self.trades.append(partial)
            self.equity_curve.append(self.current_capital)
            self._tp_hit_idx.add(idx)

            # If fully closed, reset state
            if self.current_size <= 1e-9:
                if self.position == Position.LONG:
                    self.exit_long(timestamp, current_price)
                else:
                    self.exit_short(timestamp, current_price)
                self._tp_hit_idx.clear()
                break

    # ---------- Param IO ----------
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "sensitivity": self.sensitivity,
            "leverage": self.leverage,
            "commission_rate": self.commission_rate,
            "slippage": self.slippage,
            "stop_loss_pct": self.stop_loss_pct,
            "break_even_target_pct": self.break_even_target_pct,
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
        if "stop_loss_pct" in params:
            v = float(params["stop_loss_pct"])
            if v <= 0:
                raise ValueError("stop_loss_pct must be > 0")
            self.stop_loss_pct = v
        if "break_even_target_pct" in params:
            self.break_even_target_pct = max(0.0, float(params["break_even_target_pct"]))
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
