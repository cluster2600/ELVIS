import pandas as pd

class OrderFlowAnalyzer:
    """
    Analyzes order book data to identify order flow patterns.
    """

    def __init__(self, order_book_depth: int = 20):
        """
        Initialize the OrderFlowAnalyzer.

        Args:
            order_book_depth (int): The depth of the order book to analyze.
        """
        self.order_book_depth = order_book_depth

    def get_order_flow_imbalance(self, bids: pd.DataFrame, asks: pd.DataFrame) -> float:
        """
        Calculate the order flow imbalance.

        A positive value indicates buying pressure, while a negative value indicates selling pressure.
        """
        if bids.empty or asks.empty:
            return 0.0

        bid_volume = bids.head(self.order_book_depth)['qty'].sum()
        ask_volume = asks.head(self.order_book_depth)['qty'].sum()

        return bid_volume - ask_volume
