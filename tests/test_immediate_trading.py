#!/usr/bin/env python3
"""
Test script to verify trading signal generation and execution.
This bypasses the complex main loop to directly test trading functionality.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ta

from core.bootstrap import bootstrap_application
from core.di import container


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the price data."""
    try:
        if len(data) < 50:
            return data

        # Ensure numeric types for calculations
        for col in ["close", "high", "low"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        # Add Simple Moving Averages
        data["sma_20"] = ta.trend.sma_indicator(data["close"], window=20)
        data["sma_50"] = ta.trend.sma_indicator(data["close"], window=50)

        # Add ADX
        data["adx"] = ta.trend.adx(data["high"], data["low"], data["close"], window=14)

        # Add RSI
        data["rsi"] = ta.momentum.rsi(data["close"], window=14)

        # Add MACD
        macd = ta.trend.MACD(data["close"])
        data["macd"] = macd.macd()
        data["signal_line"] = macd.macd_signal()

        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data["close"])
        data["lower_bb"] = bollinger.bollinger_lband()
        data["sma_bb"] = bollinger.bollinger_mavg()
        data["upper_bb"] = bollinger.bollinger_hband()

        # Add ATR
        data["atr"] = ta.volatility.average_true_range(
            data["high"], data["low"], data["close"]
        )

        return data
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return data


def create_mock_data():
    """Create mock market data for testing."""
    # Create trending mock data to trigger signals
    base_price = 97000
    prices = []

    # Create a strong uptrend to trigger BUY signals
    for i in range(100):
        trend = i * 20  # Strong upward trend
        noise = np.random.normal(0, 50)  # Some volatility
        price = base_price + trend + noise
        prices.append(price)

    # Create OHLCV data
    mock_data = {
        "open": [p + np.random.normal(0, 10) for p in prices],
        "high": [p + abs(np.random.normal(20, 10)) for p in prices],
        "low": [p - abs(np.random.normal(20, 10)) for p in prices],
        "close": prices,
        "volume": [np.random.normal(1000, 100) for _ in prices],
    }

    return pd.DataFrame(mock_data)


def main():
    """Test trading signal generation and execution."""
    print("🚀 ELVIS Trading Bot - Immediate Trading Test")
    print("=" * 60)

    # Bootstrap with paper trading mode
    bootstrapper = bootstrap_application("paper", "INFO")
    logger = container.get("logger")

    try:
        # Get strategy and executor
        strategy = container.get("strategy")
        executor = container.get("executor")

        print(f"✅ Strategy loaded: {type(strategy).__name__}")
        print(f"✅ Executor loaded: {type(executor).__name__}")

        # Create test market data
        print("\n📊 Creating mock market data...")
        data = create_mock_data()
        data = add_technical_indicators(data)

        print(f"✅ Mock data created: {data.shape}")
        print(f"✅ Latest close price: ${data.iloc[-1]['close']:.2f}")
        print(f"✅ RSI: {data.iloc[-1]['rsi']:.2f}")
        print(f"✅ MACD: {data.iloc[-1]['macd']:.4f}")

        # Test signal generation multiple times
        print("\n🎯 Testing signal generation (5 iterations)...")

        for i in range(5):
            print(f"\n--- Iteration {i+1} ---")

            # Slightly modify data to get different signals
            test_data = data.copy()
            if i > 0:
                # Add some variation to trigger different signals
                variation = np.random.normal(0, 100, len(test_data))
                test_data["close"] = test_data["close"] + variation
                test_data = add_technical_indicators(test_data)

            # Generate signals
            if hasattr(strategy, "symbols"):
                # Ensemble strategy
                signal_data = {"BTCUSDT": test_data}
                signals = strategy.generate_signals(signal_data)

                for symbol, signal_info in signals.items():
                    signal = signal_info.get("signal", "HOLD")
                    confidence = signal_info.get("confidence", 0.0)

                    print(f"📈 {symbol}: {signal} (confidence: {confidence:.3f})")

                    # Test with VERY low threshold (0.01 = 1%)
                    if signal in ["BUY", "SELL"] and confidence > 0.01:
                        current_price = test_data.iloc[-1]["close"]
                        available_balance = executor.get_account_balance()

                        # Calculate position size
                        leverage = getattr(executor, "default_leverage", 10)
                        position_size = strategy.calculate_position_size(
                            test_data,
                            current_price,
                            available_balance,
                            leverage,
                            confidence,
                        )

                        print(f"💰 Balance: ${available_balance:.2f}")
                        print(f"🎯 Position size: {position_size:.6f} BTC")
                        print(f"⚡ Leverage: {leverage}x")

                        # Execute the trade
                        if signal == "BUY":
                            order_result = executor.place_order(
                                symbol, "buy", position_size, current_price
                            )
                            if order_result:
                                print(
                                    f"✅ BUY ORDER EXECUTED: {position_size:.6f} {symbol} at ${current_price:.2f}"
                                )
                            else:
                                print(f"❌ BUY order failed")
                        elif signal == "SELL":
                            order_result = executor.place_order(
                                symbol, "sell", position_size, current_price
                            )
                            if order_result:
                                print(
                                    f"✅ SELL ORDER EXECUTED: {position_size:.6f} {symbol} at ${current_price:.2f}"
                                )
                            else:
                                print(f"❌ SELL order failed")
                    else:
                        print(
                            f"⏸️  No trade: signal={signal}, confidence={confidence:.3f} (threshold=0.01)"
                        )

        # Check recent trades
        print("\n📊 Checking recent trades...")
        try:
            from utils.paper_trade_db import get_all_trades

            recent_trades = get_all_trades(limit=5, exclude_test=True)
            if recent_trades:
                print(f"✅ Found {len(recent_trades)} recent trades:")
                for trade in recent_trades:
                    if len(trade) >= 7:
                        print(
                            f"   - {trade[1]} | {trade[2]} | {trade[3]} | ${trade[4]:.2f} | {trade[5]:.6f} | PnL: ${trade[6]:.2f}"
                        )
            else:
                print("⚠️  No recent trades found")
        except Exception as e:
            print(f"❌ Error checking trades: {e}")

        print("\n🎉 Test completed!")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        bootstrapper.cleanup()


if __name__ == "__main__":
    main()
