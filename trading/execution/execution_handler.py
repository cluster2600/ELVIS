### execution_handler.py
"""
Execution Handler for BTC_BOT

Handles order execution for both real trading and paper (simulation) trading.
Automatically selects the appropriate executor based on the configured mode.
"""

import logging

# Import the real executor
from trading.execution.binance_executor import BinanceExecutor

# Later we can create a proper MockExecutor, for now we simulate


class MockExecutor:
    """A simple paper trading (mock) executor."""

    def __init__(self, logger):
        self.logger = logger

    def buy(self, symbol: str, quantity: float, leverage: int = 1):
        self.logger.info(
            f"[MOCK BUY] {symbol} - Quantity: {quantity}, Leverage: {leverage}"
        )
        return {
            "status": "mock_buy_executed",
            "symbol": symbol,
            "quantity": quantity,
            "leverage": leverage,
        }

    def sell(self, symbol: str, quantity: float, leverage: int = 1):
        self.logger.info(
            f"[MOCK SELL] {symbol} - Quantity: {quantity}, Leverage: {leverage}"
        )
        return {
            "status": "mock_sell_executed",
            "symbol": symbol,
            "quantity": quantity,
            "leverage": leverage,
        }

    def close(self, symbol: str):
        self.logger.info(f"[MOCK CLOSE] {symbol}")
        return {"status": "mock_close_executed", "symbol": symbol}


class ExecutionHandler:
    """
    Unified Execution Handler.
    Switches between real and paper trading based on mode.
    """

    def __init__(
        self,
        mode: str,
        logger: logging.Logger,
        api_key: str = None,
        api_secret: str = None,
    ):
        """
        Initialize the execution handler.

        Args:
            mode (str): Trading mode ('real' or 'paper').
            logger (logging.Logger): Logger instance.
            api_key (str, optional): API key for Binance. Required for real mode.
            api_secret (str, optional): API secret for Binance. Required for real mode.
        """
        self.logger = logger
        self.mode = mode

        if self.mode == "real":
            self.logger.info("ExecutionHandler running in REAL mode.")
            self.executor = BinanceExecutor(
                logger=self.logger, api_key=api_key, api_secret=api_secret
            )
        elif self.mode == "paper":
            self.logger.info("ExecutionHandler running in PAPER (mock) mode.")
            self.executor = MockExecutor(logger=self.logger)
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'real' or 'paper'.")

    def buy(self, symbol: str, quantity: float, leverage: int = 1):
        """Place a buy order."""
        return self.executor.buy(symbol, quantity, leverage)

    def sell(self, symbol: str, quantity: float, leverage: int = 1):
        """Place a sell order."""
        return self.executor.sell(symbol, quantity, leverage)

    def close(self, symbol: str):
        """Close an open position."""
        return self.executor.close(symbol)


# End of file
