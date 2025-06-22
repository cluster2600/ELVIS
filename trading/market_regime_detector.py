import pandas as pd
import ta

class MarketRegimeDetector:
    """
    Detects the current market regime (e.g., trending, mean-reverting).
    """

    def __init__(self, adx_period: int = 14, adx_threshold: int = 25):
        """
        Initialize the MarketRegimeDetector.

        Args:
            adx_period (int): The period for the ADX indicator.
            adx_threshold (int): The threshold for the ADX indicator to determine a trend.
        """
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def get_regime(self, data: pd.DataFrame) -> str:
        """
        Get the current market regime.

        Args:
            data (pd.DataFrame): The market data.

        Returns:
            str: The current market regime ('trending' or 'mean-reverting').
        """
        adx = ta.trend.ADXIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.adx_period
        ).adx()

        if adx.iloc[-1] > self.adx_threshold:
            return 'trending'
        else:
            return 'mean-reverting'
