"""
Balanced Starter Strategy - Opens both LONG and SHORT positions initially,
then adapts based on market conditions and performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from utils.paper_trade_db import (
    add_open_position,
    close_position,
    get_open_positions,
    record_trade,
)


class BalancedStarterStrategy:
    """
    Strategy that opens balanced LONG/SHORT positions at start,
    then adapts based on market performance and conditions.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.initialized = False
        self.initial_positions_opened = False
        self.last_adaptation_time = None
        self.adaptation_interval = 900  # 15 minutes between adaptations (responsive)
        self.last_trade_time = None  # Track last trade time for cooldown

        # OPTIMIZED: Better position management for profitability
        self.target_long_positions = 3  # Increased for better opportunity capture
        self.target_short_positions = 3
        # Position limits removed per user request

        # OPTIMIZED: Profit targets that beat fees consistently
        self.target_profit_per_trade = 5.00  # $5.00 profit target (5x fee cost)
        self.daily_trade_target = 200  # 200 trades per day (8-9 per hour)
        self.trades_per_hour = 8  # Active trading frequency
        self.min_position_hold_time = 0  # No cooldown - maximum trading speed

        # Market bias tracking
        self.market_bias = 0.0  # -1.0 (bearish) to +1.0 (bullish)
        self.performance_tracker = {
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "long_count": 0,
            "short_count": 0,
        }

    def get_current_price(self) -> float:
        """Get current BTC price from API"""
        try:
            response = requests.get(
                "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT",
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                return float(data["price"])
        except Exception as e:
            self.logger.error(f"Error getting BTC price: {e}")
        return 0.0

    def analyze_current_positions(self) -> Dict[str, int]:
        """Analyze current position distribution"""
        positions = get_open_positions()
        long_count = sum(1 for pos in positions if pos[2].upper() == "BUY")
        short_count = sum(1 for pos in positions if pos[2].upper() == "SELL")

        return {
            "long_count": long_count,
            "short_count": short_count,
            "total_count": len(positions),
        }

    def calculate_position_performance(self, current_price: float) -> Dict[str, float]:
        """Calculate performance of current positions"""
        positions = get_open_positions()
        long_pnl = 0.0
        short_pnl = 0.0

        for pos in positions:
            pos_id, symbol, side, entry_price, quantity, leverage, entry_time = pos

            if side.upper() == "BUY":  # LONG position
                pnl = (current_price - entry_price) * quantity
                long_pnl += pnl
            else:  # SHORT position
                pnl = (entry_price - current_price) * quantity
                short_pnl += pnl

        return {
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "total_pnl": long_pnl + short_pnl,
        }

    def open_initial_balanced_positions(
        self, current_price: float, available_capital: float
    ):
        """Open initial balanced LONG and SHORT positions"""
        if self.initial_positions_opened:
            return

        self.logger.info(
            f"🎯 OPENING INITIAL BALANCED POSITIONS at ${current_price:,.2f}"
        )

        # OPTIMIZED POSITION SIZE - Target $5.00 profit per trade (5x fee cost)
        # With $5.00 target and ~$50-100 price movements, need proper position sizing
        leverage = 100.0

        # Calculate position size for $5.00 profit target
        # PROFITABLE SCALPING: Larger moves, bigger positions, profitable frequency
        expected_price_move = 25.0  # $25 average movement for $5.00 profit (realistic)

        # Calculate proper position size in USD first, then convert to BTC
        # PROFITABLE: Larger positions for meaningful profits that beat fees
        target_notional_usd = (
            1000.0  # $1000 notional per position (enough for real profits)
        )
        target_quantity = target_notional_usd / current_price  # Convert USD to BTC

        # Scale with leverage and ensure profitable size
        position_size = max(
            target_quantity, 0.0008
        )  # Minimum 0.0008 BTC (~$95 at current prices)

        self.logger.info(
            f"💰 PROFITABLE SCALPING SETUP: Targeting ${self.target_profit_per_trade} profit per trade"
        )
        self.logger.info(
            f"📊 Position size: {position_size:.6f} BTC (${position_size * current_price:.2f} notional)"
        )
        self.logger.info(
            f"⚡ Expected price move: ${expected_price_move} for target profit (PROFITABLE!)"
        )

        try:
            # Open LONG positions
            for i in range(self.target_long_positions):
                add_open_position(
                    symbol="BTCUSDT",
                    side="BUY",
                    entry_price=current_price,
                    quantity=position_size,
                    leverage=leverage,
                )
                record_trade(
                    symbol="BTCUSDT",
                    side="BUY",
                    price=current_price,
                    quantity=position_size,
                    pnl=0.0,
                    fee=current_price * position_size * 0.0004,
                )
                # Reduced logging - only log first position
                if i == 0:
                    self.logger.info(
                        f"✅ Opening {self.target_long_positions} LONG positions @ ${current_price:,.2f}"
                    )

            # Open SHORT positions
            for i in range(self.target_short_positions):
                add_open_position(
                    symbol="BTCUSDT",
                    side="SELL",
                    entry_price=current_price,
                    quantity=position_size,
                    leverage=leverage,
                )
                record_trade(
                    symbol="BTCUSDT",
                    side="SELL",
                    price=current_price,
                    quantity=position_size,
                    pnl=0.0,
                    fee=current_price * position_size * 0.0004,
                )
                # Reduced logging - only log first position
                if i == 0:
                    self.logger.info(
                        f"✅ Opening {self.target_short_positions} SHORT positions @ ${current_price:,.2f}"
                    )

            self.initial_positions_opened = True
            self.logger.info(
                f"🎯 BALANCED STARTUP COMPLETE: {self.target_long_positions} LONG + {self.target_short_positions} SHORT positions"
            )

        except Exception as e:
            self.logger.error(f"Error opening initial positions: {e}")

    def calculate_market_bias(self, data: pd.DataFrame, current_price: float) -> float:
        """Calculate current market bias based on multiple factors"""
        try:
            # Initialize all bias components
            momentum_bias = 0.0
            volume_bias = 0.0
            rsi_bias = 0.0
            perf_bias = 0.0
            level_bias = 0.0
            bias_score = 0.0

            # Allow calculation with less data but warn about it
            if len(data) < 5:
                self.logger.warning(
                    f"Insufficient data for bias calculation: {len(data)} rows"
                )
                # At least calculate level bias which doesn't need historical data
            else:
                self.logger.debug(
                    f"Calculating market bias with {len(data)} data points"
                )

            # 1. Price momentum (20%) - only if we have enough data
            if len(data) >= 5:
                try:
                    # Clean close prices for momentum calculation
                    clean_closes = pd.to_numeric(
                        data["close"], errors="coerce"
                    ).dropna()
                    if len(clean_closes) >= 5:
                        price_5m_ago = float(clean_closes.iloc[-5])
                        price_change_5m = (current_price - price_5m_ago) / price_5m_ago

                        price_change_15m = 0
                        if len(clean_closes) >= 15:
                            price_15m_ago = float(clean_closes.iloc[-15])
                            price_change_15m = (
                                current_price - price_15m_ago
                            ) / price_15m_ago

                        momentum_bias = (
                            price_change_5m * 0.7 + price_change_15m * 0.3
                        ) * 10  # Scale to -1 to +1
                        bias_score += momentum_bias * 0.2
                    else:
                        self.logger.warning(
                            f"Insufficient clean data for momentum: {len(clean_closes)}"
                        )
                except Exception as momentum_error:
                    self.logger.error(f"Error calculating momentum: {momentum_error}")
                    # Continue without momentum bias

            # 2. Volume analysis (15%) - with error handling
            if "volume" in data.columns and len(data) >= 10:
                try:
                    # Clean volume data
                    clean_volume = pd.to_numeric(
                        data["volume"], errors="coerce"
                    ).dropna()
                    if len(clean_volume) >= 10:
                        recent_volume = clean_volume.iloc[-5:].mean()
                        avg_volume = clean_volume.iloc[
                            -min(20, len(clean_volume)) :
                        ].mean()

                        if avg_volume > 0:
                            volume_bias = (recent_volume - avg_volume) / avg_volume
                            bias_score += np.clip(volume_bias, -0.5, 0.5) * 0.15
                        else:
                            self.logger.warning(
                                "Average volume is zero, skipping volume bias"
                            )
                    else:
                        self.logger.warning(
                            f"Insufficient clean volume data: {len(clean_volume)}"
                        )
                except Exception as volume_error:
                    self.logger.error(f"Error calculating volume bias: {volume_error}")
                    # Continue without volume bias

            # 3. RSI analysis (20%) - with error handling
            if len(data) >= 14:
                try:
                    # Clean and convert close prices with better error handling
                    close_prices = pd.to_numeric(data["close"], errors="coerce")
                    close_prices = close_prices.dropna()

                    if len(close_prices) >= 14:
                        price_diff = close_prices.diff()
                        gains = price_diff.where(price_diff > 0, 0.0)
                        losses = -price_diff.where(price_diff < 0, 0.0)
                        avg_gains = gains.rolling(window=14).mean()
                        avg_losses = losses.rolling(window=14).mean()
                        rs = avg_gains / (avg_losses + 1e-10)
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]

                        if pd.notna(current_rsi):
                            if current_rsi > 70:
                                rsi_bias = -0.3  # Overbought, bearish bias
                            elif current_rsi < 30:
                                rsi_bias = 0.3  # Oversold, bullish bias
                            else:
                                rsi_bias = (50 - current_rsi) / 100  # Neutral scaling

                            bias_score += rsi_bias * 0.2
                        else:
                            self.logger.warning("RSI calculation resulted in NaN")
                    else:
                        self.logger.warning(
                            f"Insufficient clean close price data for RSI: {len(close_prices)}"
                        )
                except Exception as rsi_error:
                    self.logger.error(f"Error calculating RSI: {rsi_error}")
                    # Continue without RSI bias

            # 4. Position performance bias (10% - REDUCED WEIGHT to prevent feedback loops)
            # Don't let current losing positions bias future decisions too much
            performance = self.calculate_position_performance(current_price)
            total_performance = performance["long_pnl"] + performance["short_pnl"]

            # Only use performance bias if losses aren't extreme (avoid feedback loops)
            if abs(total_performance) < 5.0:  # Only if total loss < $5
                if performance["long_pnl"] > performance["short_pnl"]:
                    perf_bias = 0.1  # Reduced from 0.3
                elif performance["short_pnl"] > performance["long_pnl"]:
                    perf_bias = -0.1  # Reduced from -0.3
                else:
                    perf_bias = 0.0
            else:
                perf_bias = 0.0  # Ignore performance when losses are extreme
                self.logger.warning(
                    f"🚨 Ignoring performance bias due to extreme losses: ${total_performance:.2f}"
                )

            bias_score += perf_bias * 0.1  # Reduced weight from 0.3 to 0.1

            # 5. ATH/Support levels (25% - DYNAMIC ATH DETECTION)
            # CRITICAL FIX: Detect ATH breakouts and avoid shorting strong momentum
            if len(data) >= 5:
                recent_high = (
                    data["high"].tail(5).max()
                    if "high" in data.columns
                    else current_price
                )
                price_momentum = (
                    (current_price - data["close"].iloc[-5]) / data["close"].iloc[-5]
                    if len(data) >= 5
                    else 0
                )

                # Strong upward momentum at new highs = AVOID SHORTS
                if (
                    current_price > recent_high * 1.002 and price_momentum > 0.01
                ):  # Breaking to new highs with momentum
                    level_bias = 0.5  # BULLISH bias during breakouts
                    self.logger.info(
                        f"🚀 ATH BREAKOUT: ${current_price:,.2f} breaking higher with {price_momentum*100:.2f}% momentum - AVOID SHORTS"
                    )
                elif (
                    current_price > 125000 and price_momentum > 0.005
                ):  # High price with momentum - NEW PARADIGM
                    level_bias = 0.2  # Slight bullish bias
                    self.logger.info(
                        f"📈 HIGH MOMENTUM: ${current_price:,.2f} with momentum - REDUCED SHORT BIAS"
                    )
                elif current_price > 123000:
                    level_bias = -0.3  # Reduced bearish bias (was -0.8) - NEW PARADIGM
                    self.logger.info(
                        f"⚠️ HIGH LEVEL: ${current_price:,.2f} > $123,000 - CAUTIOUS BEARISH"
                    )
                elif current_price > 120000:
                    level_bias = -0.2  # Slight bearish bias - NEW PARADIGM
                else:
                    level_bias = 0.0
            else:
                level_bias = 0.0

            bias_score += level_bias * 0.25  # Increased from 0.15 to 0.25

            # Clip to -1.0 to +1.0 range
            market_bias = np.clip(bias_score, -1.0, 1.0)

            # ALWAYS log bias components for debugging during ATH
            if (
                current_price > 125000
            ):  # Log detailed bias during high prices - NEW PARADIGM
                self.logger.info(f"📊 BIAS COMPONENTS:")
                self.logger.info(f"   Momentum: {momentum_bias:.3f} (weight: 0.2)")
                self.logger.info(f"   Volume: {volume_bias:.3f} (weight: 0.15)")
                self.logger.info(f"   RSI: {rsi_bias:.3f} (weight: 0.2)")
                self.logger.info(
                    f"   Performance: {perf_bias:.3f} (weight: 0.1 - REDUCED)"
                )
                self.logger.info(f"   Level: {level_bias:.3f} (weight: 0.25)")
                self.logger.info(f"   FINAL BIAS: {market_bias:.3f}")
            # Only log bias when it's significant or changes
            elif abs(market_bias) > 0.2 or (
                hasattr(self, "last_logged_bias")
                and abs(market_bias - self.last_logged_bias) > 0.1
            ):
                self.logger.info(
                    f"📊 Market Bias: {market_bias:.3f} (Level: {level_bias:.3f})"
                )
                self.last_logged_bias = market_bias

            return market_bias

        except Exception as e:
            self.logger.error(f"Error calculating market bias: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0

    def adapt_positions(
        self, current_price: float, available_capital: float, market_bias: float
    ):
        """Adapt position distribution based on market bias"""
        try:
            pos_analysis = self.analyze_current_positions()
            current_long = pos_analysis["long_count"]
            current_short = pos_analysis["short_count"]

            # EMERGENCY FIX: NEVER SHORT DURING OBVIOUS UPTRENDS TO PREVENT ATH DISASTERS
            # Check for obvious uptrend conditions (ATH scenarios)
            price_above_125k = current_price > 125000  # NEW PARADIGM: $120k+ is normal
            strong_upward_momentum = market_bias > 0.02

            # ATH PROTECTION: Focus on trend-following during strong moves
            if price_above_125k and strong_upward_momentum:
                # ATH PROTECTION: Only LONG positions during potential ATH runs
                target_long = self.target_long_positions
                target_short = 0  # ZERO SHORT positions during ATH
                self.logger.error(
                    f"🚨 ATH PROTECTION: BTC ${current_price:,.0f} - ONLY LONG POSITIONS"
                )
            elif market_bias > 0.2:  # Strong bullish
                target_long = self.target_long_positions
                target_short = 0  # No shorts during strong bull
                self.logger.info("📈 STRONG BULLISH BIAS - LONG only")
            elif market_bias > 0.05:  # Moderate bullish
                target_long = self.target_long_positions
                target_short = 1  # Minimal shorts
                self.logger.info("📈 MODERATE BULLISH BIAS - Favor LONG")
            elif market_bias < -0.2:  # Strong bearish
                target_long = 0
                target_short = (
                    self.target_short_positions
                )  # SHORT only during strong bear
                self.logger.info("📉 STRONG BEARISH BIAS - SHORT only")
            elif market_bias < -0.05:  # Moderate bearish
                target_long = 1
                target_short = self.target_short_positions  # Favor shorts
                self.logger.info("📉 MODERATE BEARISH BIAS - Favor SHORT")
            else:  # Neutral
                target_long = 2  # More aggressive neutral positioning
                target_short = 2
                self.logger.info("⚖️ NEUTRAL BIAS - Balanced positions")

            # Now log adaptation if significant change needed
            total_to_add = max(0, target_long - current_long) + max(
                0, target_short - current_short
            )
            if total_to_add > 0:
                self.logger.info(
                    f"🔄 ADAPTING: {current_long}→{target_long} LONG, {current_short}→{target_short} SHORT"
                )

            # OPTIMIZED POSITION SIZE - PROFITABLE SCALPING
            leverage = 100.0
            expected_price_move = 25.0  # $25 average movement for $5.00 profit

            # PROFITABLE: Larger positions for meaningful profits
            target_notional_usd = (
                1000.0  # $1000 notional per position (optimized for $5.00 profits)
            )
            target_quantity = target_notional_usd / current_price  # Convert USD to BTC

            # Smart sizing based on portfolio health
            if available_capital < 500.0:  # Portfolio damaged
                # Conservative scaling for recovery
                size_reduction = max(
                    0.3, available_capital / 1000.0
                )  # Minimum 70% reduction
                target_quantity *= size_reduction
                self.logger.warning(
                    f"🔧 RECOVERY MODE: Portfolio ${available_capital:.2f} - Scaling positions to {size_reduction:.2f}x"
                )
            elif available_capital < 800.0:  # Portfolio partially damaged
                # Moderate scaling for safety
                size_reduction = max(
                    0.6, available_capital / 1000.0
                )  # Minimum 40% reduction
                target_quantity *= size_reduction
                self.logger.info(
                    f"⚠️ CAUTIOUS MODE: Portfolio ${available_capital:.2f} - Scaling positions to {size_reduction:.2f}x"
                )

            position_size = max(
                target_quantity, 0.0008
            )  # Minimum 0.0008 BTC (~$95 at current prices)

            self.logger.info(
                f"💰 Profitable position size: {position_size:.6f} BTC (${position_size * current_price:.2f} notional)"
            )

            # Add LONG positions if needed
            long_to_add = max(0, target_long - current_long)
            for i in range(long_to_add):
                add_open_position(
                    symbol="BTCUSDT",
                    side="BUY",
                    entry_price=current_price,
                    quantity=position_size,
                    leverage=leverage,
                )
                record_trade(
                    symbol="BTCUSDT",
                    side="BUY",
                    price=current_price,
                    quantity=position_size,
                    pnl=0.0,
                    fee=current_price * position_size * 0.0004,
                )
                # Track trade timing
                from datetime import datetime

                self.last_trade_time = datetime.now()
                # Reduced logging for position additions
                pass

            # Add SHORT positions if needed
            short_to_add = max(0, target_short - current_short)
            for i in range(short_to_add):
                add_open_position(
                    symbol="BTCUSDT",
                    side="SELL",
                    entry_price=current_price,
                    quantity=position_size,
                    leverage=leverage,
                )
                record_trade(
                    symbol="BTCUSDT",
                    side="SELL",
                    price=current_price,
                    quantity=position_size,
                    pnl=0.0,
                    fee=current_price * position_size * 0.0004,
                )
                # Track trade timing
                from datetime import datetime

                self.last_trade_time = datetime.now()
                # Reduced logging for position additions
                pass

            self.logger.info(
                f"🎯 ADAPTATION COMPLETE: {target_long} LONG, {target_short} SHORT target"
            )

        except Exception as e:
            self.logger.error(f"Error adapting positions: {e}")

    def close_conflicting_positions(self, current_price: float, target_bias: str):
        """Close positions that conflict with target bias at critical levels"""
        try:
            positions = get_open_positions()

            closed_count = 0
            for pos in positions:
                pos_id, symbol, side, entry_price, quantity, leverage, entry_time = pos

                # At ATH with SHORT bias, close LONG positions
                if target_bias == "SHORT" and side.upper() == "BUY":
                    # Calculate P&L for this position
                    pnl = (current_price - entry_price) * quantity

                    # Close the position
                    close_position(pos_id, current_price, pnl)
                    closed_count += 1

                    self.logger.info(
                        f"🚨 ATH CLOSURE: Closed LONG position {pos_id} - Entry: ${entry_price:,.2f}, Exit: ${current_price:,.2f}, P&L: ${pnl:.2f}"
                    )

                # At strong bullish bias, close SHORT positions
                elif target_bias == "LONG" and side.upper() == "SELL":
                    pnl = (entry_price - current_price) * quantity

                    close_position(pos_id, current_price, pnl)
                    closed_count += 1

                    self.logger.info(
                        f"📈 BULLISH CLOSURE: Closed SHORT position {pos_id} - Entry: ${entry_price:,.2f}, Exit: ${current_price:,.2f}, P&L: ${pnl:.2f}"
                    )

            if closed_count > 0:
                self.logger.info(
                    f"🎯 EMERGENCY CLOSURE COMPLETE: Closed {closed_count} conflicting positions"
                )

        except Exception as e:
            self.logger.error(f"Error closing conflicting positions: {e}")

    def take_profits_on_scalping_positions(self, current_price: float):
        """Close positions that have reached $5.00 profit target"""
        try:
            # Get current timestamp for trade execution
            from datetime import datetime

            current_time = datetime.now()

            positions = get_open_positions()
            closed_count = 0

            for pos in positions:
                pos_id, symbol, side, entry_price, quantity, leverage, entry_time = pos

                # Calculate current P&L
                if side.upper() == "BUY":  # LONG position
                    pnl = (current_price - entry_price) * quantity
                else:  # SHORT position
                    pnl = (entry_price - current_price) * quantity

                # Close position if profit target reached
                # OPTIMIZED: Profitable targets that beat fees consistently
                realistic_profit_target = 5.00  # $5.00 profit target (5x fee cost)
                realistic_stop_loss = -1.00  # $1.00 stop loss (5:1 reward:risk ratio)

                if pnl >= realistic_profit_target:
                    close_position(pos_id, current_price, pnl)
                    closed_count += 1
                    self.last_trade_time = current_time  # Track last trade time

                    # Log all profitable closes for scalping
                    self.logger.info(
                        f"💰 PROFIT: {side} ${pnl:.4f} (target: ${realistic_profit_target})"
                    )

                # EMERGENCY: Close ANY position losing more than stop loss
                elif pnl <= realistic_stop_loss:
                    close_position(pos_id, current_price, pnl)
                    closed_count += 1
                    self.last_trade_time = current_time  # Track last trade time
                    self.logger.error(
                        f"🛑 STOP LOSS: {side} ${pnl:.4f} (target: ${realistic_stop_loss})"
                    )

                # OPTIMIZED: Force close positions with reasonable losses (safety net)
                elif pnl <= -2.00:  # Emergency stop if loss exceeds -$2.00
                    close_position(pos_id, current_price, pnl)
                    closed_count += 1
                    self.last_trade_time = current_time  # Track last trade time
                    self.logger.error(
                        f"🚨 EMERGENCY STOP: {side} ${pnl:.4f} - FORCE CLOSING RUNAWAY LOSS"
                    )

            # Only log summary if many positions closed
            if closed_count > 5:
                self.logger.info(f"⚡ SCALPING: Closed {closed_count} positions")

        except Exception as e:
            self.logger.error(f"Error taking profits on scalping positions: {e}")

    def emergency_close_losing_positions(self, current_price: float):
        """Emergency function to close all losing positions when portfolio is severely damaged"""
        try:
            positions = get_open_positions()
            closed_count = 0

            for pos in positions:
                pos_id, symbol, side, entry_price, quantity, leverage, entry_time = pos

                # Calculate P&L
                if side.upper() == "BUY":
                    pnl = (current_price - entry_price) * quantity
                else:
                    pnl = (entry_price - current_price) * quantity

                # Close ALL positions that are losing money
                if pnl < -0.05:  # Close anything losing more than 5 cents
                    close_position(pos_id, current_price, pnl)
                    closed_count += 1
                    self.logger.error(
                        f"🚨 EMERGENCY CLOSE: {side} position {pos_id} - Loss: ${pnl:.2f}"
                    )

            if closed_count > 0:
                self.logger.error(
                    f"🚨 EMERGENCY COMPLETE: Closed {closed_count} losing positions to protect portfolio"
                )

        except Exception as e:
            self.logger.error(f"Error in emergency close: {e}")

    def should_adapt(self) -> bool:
        """Check if it's time to adapt positions"""
        if self.last_adaptation_time is None:
            return True

        time_since_adaptation = datetime.now() - self.last_adaptation_time
        return time_since_adaptation.total_seconds() > self.adaptation_interval

    def execute_strategy(
        self, data: pd.DataFrame, available_capital: float
    ) -> Dict[str, Any]:
        """Main strategy execution"""
        try:
            current_price = self.get_current_price()
            if current_price <= 0:
                self.logger.error("Could not get current price")
                return {"action": "HOLD", "reason": "No price data"}

            # EMERGENCY PORTFOLIO PROTECTION - Check for major losses FIRST.
            # Threshold is relative to the CONFIGURED deposit (the old
            # hardcoded $200 assumed the $1000 paper world and fired every
            # cycle once the deposit became $100).
            import os as _os

            from config.config import PAPER_TRADING_CONFIG as _PTC

            _emergency_floor = float(_PTC.get("INITIAL_USDT_BALANCE", 100.0)) * float(
                _os.getenv("ELVIS_EMERGENCY_CLOSE_PCT", "0.2")
            )
            if available_capital < _emergency_floor:
                self.logger.error(
                    f"🚨 EMERGENCY: Portfolio at ${available_capital:.2f} "
                    f"(< ${_emergency_floor:.2f}) - CLOSING ALL LOSING POSITIONS"
                )
                self.emergency_close_losing_positions(current_price)
                return {
                    "action": "EMERGENCY_PROTECTION",
                    "reason": "Portfolio protection activated",
                }

            # SCALPING PROFIT TAKING - Check for profit targets
            self.take_profits_on_scalping_positions(current_price)

            # Calculate market bias first to determine action
            self.market_bias = self.calculate_market_bias(data, current_price)

            # NEW ATH LOGIC: Follow momentum instead of fighting it
            if current_price > 123000:
                self.logger.info(
                    f"🚀 ATH LEVEL: Price ${current_price:,.2f} > $123,000 - NEW PARADIGM"
                )
                self.logger.info(f"📊 Market bias: {self.market_bias:.3f}")

                # If momentum is BULLISH at ATH, follow it (don't fight the trend)
                if self.market_bias > 0.0:
                    self.logger.info(f"🚀 BULLISH MOMENTUM AT ATH - Following uptrend")
                    # Close SHORT positions that fight the trend
                    self.close_conflicting_positions(current_price, target_bias="LONG")
                else:
                    self.logger.info(f"📉 BEARISH BIAS AT ATH - Proceeding carefully")

                # Adapt based on actual market bias (not forced SHORT)
                self.adapt_positions(current_price, available_capital, self.market_bias)
                self.last_adaptation_time = datetime.now()

                return {
                    "action": "ATH_ADAPTED",
                    "price": current_price,
                    "market_bias": self.market_bias,
                    "positions": self.analyze_current_positions(),
                }

            # Step 1: Open initial balanced positions if not done
            if not self.initial_positions_opened:
                self.open_initial_balanced_positions(current_price, available_capital)
                self.last_adaptation_time = datetime.now()
                return {"action": "INITIALIZED", "price": current_price}

            # Step 2: Check if adaptation is needed (but always adapt at extreme bias)
            if not self.should_adapt() and abs(self.market_bias) < 0.4:
                return {
                    "action": "MONITORING",
                    "price": current_price,
                    "market_bias": self.market_bias,
                }

            # Step 3: Adapt positions based on bias
            self.adapt_positions(current_price, available_capital, self.market_bias)

            # Step 4: Update timing
            self.last_adaptation_time = datetime.now()

            return {
                "action": "ADAPTED",
                "price": current_price,
                "market_bias": self.market_bias,
                "positions": self.analyze_current_positions(),
            }

        except Exception as e:
            self.logger.error(f"Error in balanced starter strategy: {e}")
            return {"action": "ERROR", "reason": str(e)}

    def generate_signal(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Generate trading signal for main loop compatibility"""
        try:
            # Use the market data to create a simple DataFrame
            import pandas as pd

            # Create a minimal DataFrame from market data
            data = pd.DataFrame([market_data])

            # Get current price and capital
            current_price = market_data.get("price", market_data.get("close", 0))
            available_capital = 1000.0  # Default, should be passed from executor

            # Execute the strategy
            result = self.execute_strategy(data, available_capital)

            # Convert strategy result to signal format
            action = result.get("action", "HOLD")
            market_bias = result.get("market_bias", 0.0)

            # Map actions to signals
            if action in ["INITIALIZED", "ADAPTED", "ATH_ADAPTED"]:
                # For balanced strategy, we don't generate individual signals
                # The strategy manages positions internally
                return "HOLD", 0.5
            elif action == "EMERGENCY_PROTECTION":
                return "HOLD", 0.1
            elif action == "MONITORING":
                return "HOLD", 0.5
            else:
                # Convert market bias to signal - MORE AGGRESSIVE
                if market_bias > 0.15:  # Lower threshold for buy signals
                    return "BUY", min(0.9, 0.6 + abs(market_bias))
                elif market_bias < -0.15:  # Lower threshold for sell signals
                    return "SELL", min(0.9, 0.6 + abs(market_bias))
                else:
                    return "HOLD", 0.5

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return "HOLD", 0.1
