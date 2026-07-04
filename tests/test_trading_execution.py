#!/usr/bin/env python3
"""
Simple test script to verify trading execution without the full bot framework.
"""

import logging
import time

import numpy as np
import pandas as pd

from trading.execution.binance_executor import BinanceExecutor
from trading.strategies.ensemble_strategy import EnsembleStrategy

# Setup simple logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_trading_execution():
    """Test the trading execution directly."""

    # Initialize executor in paper trading mode
    logger.info("Initializing BinanceExecutor in paper trading mode...")
    executor = BinanceExecutor(logger=logger, is_testnet=True)
    executor.initialize()

    # Initialize ensemble strategy
    logger.info("Initializing EnsembleStrategy...")
    strategy = EnsembleStrategy(logger=logger)

    # Create mock market data
    logger.info("Creating mock market data...")
    np.random.seed(42)
    mock_data = {
        "open": np.random.normal(97000, 200, 50),
        "high": np.random.normal(97200, 200, 50),
        "low": np.random.normal(96800, 200, 50),
        "close": np.random.normal(97000, 200, 50),
        "volume": np.random.normal(1000, 100, 50),
    }
    data = pd.DataFrame(mock_data)

    # Add basic indicators manually
    data["rsi"] = 65  # Overbought - should trigger SELL for testing
    data["macd"] = -3.0
    data["signal_line"] = 2.0  # MACD < signal - should trigger SELL
    data["sma_20"] = 97100  # Price below SMA - should trigger SELL
    data["adx"] = 40
    data["atr"] = 100

    logger.info(
        f"Mock data created: RSI={data['rsi'].iloc[-1]}, MACD={data['macd'].iloc[-1]}, Price={data['close'].iloc[-1]:.2f}"
    )

    # Test signal generation
    logger.info("Testing signal generation...")
    signals = strategy.generate_signals({"BTCUSDT": data})
    logger.info(f"Generated signals: {signals}")

    # Test trade execution
    signal_info = signals.get("BTCUSDT", {})
    signal = signal_info.get("signal", "HOLD")
    confidence = signal_info.get("confidence", 0.0)

    logger.info(f"Signal: {signal}, Confidence: {confidence}")

    if signal in ["BUY", "SELL"] and confidence > 0.1:
        current_price = data["close"].iloc[-1]
        available_balance = executor.get_account_balance()

        # Calculate position size (simple)
        position_size = min(
            0.001, available_balance * 0.01 / current_price
        )  # 1% of balance

        logger.info(
            f"Executing {signal} order - Price: ${current_price:.2f}, Size: {position_size:.6f}"
        )

        # Execute trade
        if signal == "BUY":
            result = executor.execute_buy("BTCUSDT", position_size, current_price)
            logger.info(f"BUY result: {result}")
        elif signal == "SELL":
            result = executor.execute_sell("BTCUSDT", position_size, current_price)
            logger.info(f"SELL result: {result}")

        return True
    else:
        logger.info("No trade executed - conditions not met")
        return False


if __name__ == "__main__":
    logger.info("=== TRADING EXECUTION TEST ===")
    success = test_trading_execution()
    logger.info(f"Test completed. Trade executed: {success}")
