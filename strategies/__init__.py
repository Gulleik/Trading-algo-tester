"""
Strategies package for trading algorithm research.

This package contains base strategy classes and concrete strategy implementations
for backtesting and optimization.
"""

from .base_strategy import BaseStrategy
from .simple_ma_strategy import SimpleMAStrategy

__all__ = ['BaseStrategy', 'SimpleMAStrategy']
