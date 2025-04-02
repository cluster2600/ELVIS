"""
Strategies package for the ELVIS project.
"""

from trading.strategies.base_strategy import BaseStrategy
from trading.strategies.technical_strategy import TechnicalStrategy
from trading.strategies.mean_reversion_strategy import MeanReversionStrategy
from trading.strategies.trend_following_strategy import TrendFollowingStrategy
from trading.strategies.ema_rsi_strategy import EmaRsiStrategy
from trading.strategies.ensemble_strategy import EnsembleStrategy

__all__ = [
    "BaseStrategy",
    "TechnicalStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
    "EmaRsiStrategy",
    "EnsembleStrategy"
]