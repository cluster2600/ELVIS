import numpy as np
import pandas as pd
from trading.strategies.base_strategy import BaseStrategy
import logging

class GridStrategy(BaseStrategy):
    """
    A grid trading strategy that places a series of buy and sell orders at predefined intervals.
    """

    def __init__(self, logger: logging.Logger, grid_levels: int = 5, grid_spacing: float = 0.01, **kwargs):
        """
        Initialize the grid strategy.

        Args:
            logger (logging.Logger): The logger to use.
            grid_levels (int): The number of grid levels.
            grid_spacing (float): The spacing between grid levels.
        """
        super().__init__(logger, **kwargs)
        self.grid_levels = grid_levels
        self.grid_spacing = grid_spacing

    def generate_signals(self, data: pd.DataFrame):
        """
        Grid trading does not generate signals in the traditional sense.
        The grid levels are the signals.
        """
        pass

    def calculate_position_size(self, data: pd.DataFrame, current_price: float, available_capital: float) -> float:
        """
        Calculate the position size for each grid level.
        """
        return (available_capital / self.grid_levels) / current_price

    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the stop loss for the entire grid.
        """
        return entry_price * (1 - (self.grid_levels + 1) * self.grid_spacing)

    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        The take profit for a grid is typically the next grid level.
        """
        return entry_price * (1 + self.grid_spacing)

    def get_grid_levels(self, current_price: float):
        """
        Get the buy and sell grid levels.
        """
        buy_levels = [current_price * (1 - i * self.grid_spacing) for i in range(1, self.grid_levels + 1)]
        sell_levels = [current_price * (1 + i * self.grid_spacing) for i in range(1, self.grid_levels + 1)]
        return buy_levels, sell_levels
