import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, PositionType

class DonchianATRTrail(BaseStrategy):
    def __init__(self, name="DonchianATRTrail", 
                 lookback=20, atr_n=14, atr_k=2.5,
                 persistence=2, buffer_bps=5,
                 risk_per_trade=0.02, **kw):
        super().__init__(name, **kw)
        self.lookback = int(lookback)
        self.atr_n = int(atr_n)
        self.atr_k = float(atr_k)
        self.persistence = int(persistence)
        self.buffer = buffer_bps/10000.0
        self.risk_per_trade = float(risk_per_trade)
        self._trail = None  # current trailing stop

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        highN = df["high"].rolling(self.lookback, min_periods=self.lookback).max()
        lowN  = df["low"] .rolling(self.lookback, min_periods=self.lookback).min()

        tr1 = (df['high'] - df['low']).abs()
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low']  - df['close'].shift()).abs()
        df["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(self.atr_n, min_periods=self.atr_n).mean()

        up  = highN * (1 + self.buffer)
        dn  = lowN  * (1 - self.buffer)
        cond_long  = (df["close"] > up)
        cond_short = (df["close"] < dn)

        df["long_signal"]  = cond_long.rolling(self.persistence).sum() == self.persistence
        df["short_signal"] = cond_short.rolling(self.persistence).sum() == self.persistence

        # exit on opposite breakout
        df["exit_long_signal"]  = df["short_signal"]
        df["exit_short_signal"] = df["long_signal"]
        return df

    def _position_size(self, entry, atr):
        # risk = k*ATR trail distance initially
        stop_dist = max(atr * self.atr_k, entry * 0.0005)  # small floor
        return (self.current_capital * self.risk_per_trade) / stop_dist

    def should_enter_long(self, row, signals):  return bool(row.get("long_signal", False)  and self.position == PositionType.FLAT)
    def should_exit_long (self, row, signals):  return bool(row.get("exit_long_signal", False))
    def should_enter_short(self, row, signals): return bool(row.get("short_signal", False) and self.position == PositionType.FLAT)
    def should_exit_short(self, row, signals):  return bool(row.get("exit_short_signal", False))

    # Use next-bar open fills in your tester; manage trail on current bar's H/L
    def check_stop_loss(self, row: pd.Series, timestamp: pd.Timestamp) -> bool:
        if self.position == PositionType.FLAT or self.current_size <= 0:
            return False
        high, low = float(row["high"]), float(row["low"])
        atr = float(row.get("atr", np.nan))
        if np.isnan(atr): 
            return False

        # update trail
        if self.position == PositionType.LONG:
            # move up with price
            new_trail = float(row["close"]) - self.atr_k * atr
            self._trail = new_trail if self._trail is None else max(self._trail, new_trail)
            if low <= self._trail:  # hit
                self.exit_long(timestamp, self._trail)
                self._trail = None
                return True

        elif self.position == PositionType.SHORT:
            new_trail = float(row["close"]) + self.atr_k * atr
            self._trail = new_trail if self._trail is None else min(self._trail, new_trail)
            if high >= self._trail:
                self.exit_short(timestamp, self._trail)
                self._trail = None
                return True
        return False

    # size at entry
    def calculate_position_size(self, price: float, risk_per_trade: float=None, signals_row: pd.Series=None) -> float:
        atr = float(signals_row.get("atr", np.nan)) if signals_row is not None else np.nan
        if np.isnan(atr):
            return 0.0
        return max(0.0, self._position_size(price, atr))
