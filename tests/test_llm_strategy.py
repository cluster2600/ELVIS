#!/usr/bin/env python3
"""
Test script to verify LLM-enhanced strategy functionality.
"""

import numpy as np
import pandas as pd

from trading.strategies.llm_enhanced_strategy import LLMEnhancedStrategy
from utils.logging_utils import setup_logger


def create_sample_data(n_points=100):
    """Create sample OHLCV data for testing"""
    np.random.seed(42)

    # Simulate Bitcoin price around $118,000
    base_price = 118000
    prices = [base_price]

    for _ in range(n_points - 1):
        change = np.random.normal(0, 0.02)  # 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(50000, min(200000, new_price)))  # Keep in reasonable range

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n_points, freq="5min"),
            "open": [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            "close": prices,
            "volume": [1000 + abs(np.random.normal(0, 500)) for _ in prices],
        }
    )

    return data


def main():
    """Test LLM-enhanced strategy functionality"""
    print("🧠 Testing LLM-Enhanced Trading Strategy")
    print("=" * 50)

    # Setup logger
    logger = setup_logger("llm_strategy_test", log_level="INFO")

    # Create strategy instance
    strategy = LLMEnhancedStrategy(logger=logger, model_base_path="models")

    # Create sample data
    test_data = create_sample_data(100)
    current_price = test_data["close"].iloc[-1]

    print(f"\n📊 Test Data Summary:")
    print(f"   Data points: {len(test_data)}")
    print(f"   Current price: ${current_price:,.2f}")
    print(
        f"   Price range: ${test_data['close'].min():,.2f} - ${test_data['close'].max():,.2f}"
    )

    # Test signal generation
    print(f"\n🎯 Testing Signal Generation:")
    try:
        buy_signal, sell_signal = strategy.generate_signals(test_data)
        print(f"   Buy signal: {buy_signal}")
        print(f"   Sell signal: {sell_signal}")

        # Test individual should_buy/should_sell methods
        should_buy = strategy.should_buy(test_data, current_price)
        should_sell = strategy.should_sell(test_data, current_price)
        print(f"   Should buy: {should_buy}")
        print(f"   Should sell: {should_sell}")

    except Exception as e:
        print(f"   ❌ Signal generation failed: {e}")

    # Test position sizing
    print(f"\n💰 Testing Position Sizing:")
    try:
        available_capital = 1000.0
        position_size = strategy.calculate_position_size(
            test_data, current_price, available_capital
        )
        position_value = position_size * current_price
        print(f"   Available capital: ${available_capital:.2f}")
        print(f"   Position size: {position_size:.6f} BTC")
        print(f"   Position value: ${position_value:.2f}")
        print(f"   Capital allocation: {position_value / available_capital * 100:.1f}%")

    except Exception as e:
        print(f"   ❌ Position sizing failed: {e}")

    # Test risk management
    print(f"\n🛡️  Testing Risk Management:")
    try:
        entry_price = current_price
        stop_loss = strategy.calculate_stop_loss(test_data, entry_price)
        take_profit = strategy.calculate_take_profit(test_data, entry_price)

        stop_loss_pct = ((entry_price - stop_loss) / entry_price) * 100
        take_profit_pct = ((take_profit - entry_price) / entry_price) * 100

        print(f"   Entry price: ${entry_price:,.2f}")
        print(f"   Stop loss: ${stop_loss:,.2f} (-{stop_loss_pct:.1f}%)")
        print(f"   Take profit: ${take_profit:,.2f} (+{take_profit_pct:.1f}%)")
        print(f"   Risk/Reward: 1:{take_profit_pct/stop_loss_pct:.1f}")

    except Exception as e:
        print(f"   ❌ Risk management failed: {e}")

    # Test strategy info
    print(f"\n📋 Strategy Information:")
    try:
        info = strategy.get_strategy_info()
        for key, value in info.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

    except Exception as e:
        print(f"   ❌ Strategy info failed: {e}")

    print(f"\n✅ LLM Strategy Test Complete!")


if __name__ == "__main__":
    main()
