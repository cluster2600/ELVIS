from trading.strategies.base_strategy import BaseStrategy
from trading.market_regime_detector import MarketRegimeDetector

class StrategyManager:
    """
    Manages the active trading strategy based on the current market regime.
    """

    def __init__(self, market_regime_detector: MarketRegimeDetector, strategies: dict):
        """
        Initialize the StrategyManager.

        Args:
            market_regime_detector (MarketRegimeDetector): The market regime detector.
            strategies (dict): A dictionary of available strategies, keyed by regime.
        """
        self.market_regime_detector = market_regime_detector
        self.strategies = strategies
        self.active_strategy = None

    def get_active_strategy(self, data) -> BaseStrategy:
        """
        Get the active strategy based on the current market regime.
        """
        regime = self.market_regime_detector.get_regime(data)
        
        if regime in self.strategies:
            self.active_strategy = self.strategies[regime]
        else:
            # Default to a base strategy if no specific strategy for the regime
            self.active_strategy = self.strategies.get('default')
            
        return self.active_strategy
