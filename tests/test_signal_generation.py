#!/usr/bin/env python3
"""Test the improved signal generation to verify both BUY and SELL signals work."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from trading.strategies.ensemble_strategy import EnsembleStrategy

# Setup logger
logger = logging.getLogger("test")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# Test signal generation with different market conditions
strategy = EnsembleStrategy(logger)

print("🧪 TESTING IMPROVED SIGNAL GENERATION")
print("=====================================")

# Test bearish market conditions
print("\n📉 TEST 1: BEARISH MARKET CONDITIONS")
bearish_data = {
    "close": 100,
    "price": 100,
    "rsi": 75,  # Overbought - should trigger SELL
    "macd": -2,  # Negative
    "macd_signal": -1,  # MACD < signal (bearish)
    "macd_histogram": -1,
    "sma_20": 105,  # Price below SMA20
    "sma_50": 110,  # SMA20 < SMA50 (bearish alignment)
    "bb_upper": 115,
    "bb_lower": 85,
    "bb_middle": 100,
}

signal, confidence = strategy.generate_signal("BTCUSDT", bearish_data)
print(f"🔴 Bearish market result: {signal} with confidence: {confidence:.3f}")

# Test bullish market conditions
print("\n📈 TEST 2: BULLISH MARKET CONDITIONS")
bullish_data = {
    "close": 110,
    "price": 110,
    "rsi": 25,  # Oversold - should trigger BUY
    "macd": 2,  # Positive
    "macd_signal": 1,  # MACD > signal (bullish)
    "macd_histogram": 1,
    "sma_20": 105,  # Price above SMA20
    "sma_50": 100,  # SMA20 > SMA50 (bullish alignment)
    "bb_upper": 115,
    "bb_lower": 85,
    "bb_middle": 100,
}

signal, confidence = strategy.generate_signal("BTCUSDT", bullish_data)
print(f"🟢 Bullish market result: {signal} with confidence: {confidence:.3f}")

# Test neutral/mixed conditions
print("\n➡️  TEST 3: NEUTRAL MARKET CONDITIONS")
neutral_data = {
    "close": 102,
    "price": 102,
    "rsi": 50,  # Neutral
    "macd": 0.1,
    "macd_signal": 0.1,
    "macd_histogram": 0,
    "sma_20": 101,
    "sma_50": 100,
    "bb_upper": 115,
    "bb_lower": 85,
    "bb_middle": 100,
}

signal, confidence = strategy.generate_signal("BTCUSDT", neutral_data)
print(f"⚪ Neutral market result: {signal} with confidence: {confidence:.3f}")

print("\n✅ Signal generation test complete!")
