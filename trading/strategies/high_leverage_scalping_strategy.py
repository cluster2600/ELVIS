"""
High Leverage Scalping Strategy - Optimized for 100x Leverage Profits

This strategy focuses on:
1. High-frequency scalping with 100x leverage
2. Quick profit taking on small price movements
3. Tight stop losses to minimize risk
4. Momentum-based entries for higher win rate
5. Volume analysis for better timing

Target: Make consistent small profits that compound with 100x leverage
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import ta


class HighLeverageScalpingStrategy:
    """
    Optimized scalping strategy for 100x leverage profit maximization
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

        # SCALPING PARAMETERS - Optimized for 100x leverage
        self.leverage = 100  # Full 100x leverage
        self.min_confidence = 0.65  # Lower threshold for more opportunities
        self.scalp_profit_target = 0.05  # 0.05% target = 5% profit with 100x leverage
        self.stop_loss_pct = 0.03  # 0.03% stop = 3% loss with 100x leverage
        self.position_hold_time = 300  # 5 minutes max hold time

        # WIN RATE OPTIMIZATION
        self.momentum_threshold = 0.02  # 0.02% momentum required
        self.volume_multiplier = 1.5  # Volume must be 1.5x average
        self.rsi_oversold = 30  # RSI oversold level
        self.rsi_overbought = 70  # RSI overbought level

        # TRACKING
        self.last_signal_time = None
        self.position_entry_time = None
        self.recent_signals = []  # Track recent signal performance

        self.logger.info("🚀 HIGH LEVERAGE SCALPING STRATEGY INITIALIZED")
        self.logger.info(
            f"⚡ Target: {self.scalp_profit_target}% moves = {self.scalp_profit_target * 100}% profit @ {self.leverage}x"
        )
        self.logger.info(
            f"🛑 Stop Loss: {self.stop_loss_pct}% = {self.stop_loss_pct * 100}% loss @ {self.leverage}x"
        )

    def analyze_scalping_opportunity(
        self, data: pd.DataFrame
    ) -> Tuple[str, float, Dict]:
        """
        Analyze market for high-probability scalping opportunities
        """
        if len(data) < 50:
            return "HOLD", 0.0, {}

        try:
            # Get latest data
            current = data.iloc[-1]
            previous = data.iloc[-2]
            recent_data = data.tail(20)

            # Calculate key metrics for scalping
            price = float(current["close"])
            volume = float(current.get("volume", 1000))
            avg_volume = (
                float(recent_data["volume"].mean()) if "volume" in recent_data else 1000
            )

            # Technical indicators
            rsi = float(current.get("rsi", 50))
            macd = float(current.get("macd", 0))
            macd_signal = float(current.get("signal_line", 0))
            sma_5 = float(data["close"].rolling(5).mean().iloc[-1])
            sma_20 = float(data["close"].rolling(20).mean().iloc[-1])

            # Price momentum analysis
            price_change_1min = (
                (price - float(previous["close"])) / float(previous["close"])
            ) * 100
            price_change_5min = (
                (
                    (price - float(data.iloc[-6]["close"]))
                    / float(data.iloc[-6]["close"])
                )
                * 100
                if len(data) > 6
                else 0
            )

            # Volume confirmation
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1

            analysis = {
                "price": price,
                "price_change_1min": price_change_1min,
                "price_change_5min": price_change_5min,
                "volume_ratio": volume_ratio,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "sma_5": sma_5,
                "sma_20": sma_20,
                "momentum_strength": abs(price_change_1min) + abs(price_change_5min),
            }

            # SCALPING SIGNAL LOGIC
            signal, confidence = self._generate_scalping_signal(analysis)

            return signal, confidence, analysis

        except Exception as e:
            self.logger.error(f"Error in scalping analysis: {e}")
            return "HOLD", 0.0, {}

    def _generate_scalping_signal(self, analysis: Dict) -> Tuple[str, float]:
        """
        Generate scalping signals optimized for 100x leverage profits
        """
        try:
            # Extract analysis data
            price = analysis["price"]
            price_change_1min = analysis["price_change_1min"]
            price_change_5min = analysis["price_change_5min"]
            volume_ratio = analysis["volume_ratio"]
            rsi = analysis["rsi"]
            macd = analysis["macd"]
            macd_signal = analysis["macd_signal"]
            sma_5 = analysis["sma_5"]
            sma_20 = analysis["sma_20"]
            momentum_strength = analysis["momentum_strength"]

            # Initialize signal scoring
            buy_score = 0.0
            sell_score = 0.0
            confidence_factors = []

            # 1. MOMENTUM ANALYSIS (40% weight)
            if abs(price_change_1min) > self.momentum_threshold:
                if price_change_1min > 0:
                    buy_score += 0.4
                    confidence_factors.append(
                        f"Positive 1min momentum: {price_change_1min:.3f}%"
                    )
                else:
                    sell_score += 0.4
                    confidence_factors.append(
                        f"Negative 1min momentum: {price_change_1min:.3f}%"
                    )

            # 2. VOLUME CONFIRMATION (20% weight)
            if volume_ratio > self.volume_multiplier:
                momentum_boost = 0.2
                buy_score += momentum_boost if price_change_1min > 0 else 0
                sell_score += momentum_boost if price_change_1min < 0 else 0
                confidence_factors.append(
                    f"Volume confirmation: {volume_ratio:.1f}x average"
                )

            # 3. RSI EXTREMES FOR REVERSAL SCALPING (20% weight)
            if rsi < self.rsi_oversold:
                buy_score += 0.2
                confidence_factors.append(f"RSI oversold: {rsi:.1f}")
            elif rsi > self.rsi_overbought:
                sell_score += 0.2
                confidence_factors.append(f"RSI overbought: {rsi:.1f}")

            # 4. MACD MOMENTUM (15% weight)
            if macd > macd_signal:
                buy_score += 0.15
                confidence_factors.append("MACD bullish")
            elif macd < macd_signal:
                sell_score += 0.15
                confidence_factors.append("MACD bearish")

            # 5. MOVING AVERAGE TREND (5% weight)
            if sma_5 > sma_20:
                buy_score += 0.05
            elif sma_5 < sma_20:
                sell_score += 0.05

            # SIGNAL DECISION
            max_score = max(buy_score, sell_score)

            if buy_score > sell_score and buy_score >= self.min_confidence:
                signal = "BUY"
                confidence = buy_score
            elif sell_score > buy_score and sell_score >= self.min_confidence:
                signal = "SELL"
                confidence = sell_score
            else:
                signal = "HOLD"
                confidence = max_score

            # Log signal reasoning
            if signal != "HOLD":
                self.logger.info(f"💹 SCALPING SIGNAL: {signal} @ ${price:.2f}")
                self.logger.info(
                    f"🎯 Confidence: {confidence:.1%} | Factors: {', '.join(confidence_factors)}"
                )
                self.logger.info(
                    f"📊 Momentum: {momentum_strength:.3f}% | Volume: {volume_ratio:.1f}x | RSI: {rsi:.1f}"
                )

            return signal, confidence

        except Exception as e:
            self.logger.error(f"Error generating scalping signal: {e}")
            return "HOLD", 0.0

    def calculate_scalping_position_size(
        self, current_price: float, available_capital: float, confidence: float
    ) -> float:
        """
        Calculate optimal position size for 100x leverage scalping
        """
        try:
            # Base position sizing for scalping
            base_risk_pct = 0.02  # 2% of capital at risk per trade

            # Adjust risk based on confidence
            confidence_multiplier = min(
                2.0, max(0.5, confidence * 2)
            )  # 0.5x to 2x based on confidence

            # Calculate position size
            risk_amount = available_capital * base_risk_pct * confidence_multiplier

            # With 100x leverage, position size is risk amount / (stop loss % * price)
            stop_loss_dollars = (
                current_price * (self.stop_loss_pct / 100) * self.leverage
            )

            if stop_loss_dollars > 0:
                position_size_btc = risk_amount / stop_loss_dollars
            else:
                position_size_btc = risk_amount / current_price * 0.01  # Fallback

            # Ensure reasonable bounds
            min_size = 0.001  # 0.001 BTC minimum
            max_size = available_capital / current_price * 0.1  # 10% of capital max

            position_size_btc = max(min_size, min(position_size_btc, max_size))

            self.logger.info(
                f"💰 SCALP POSITION: {position_size_btc:.6f} BTC (${position_size_btc * current_price:.2f})"
            )
            self.logger.info(
                f"⚡ Leverage: {self.leverage}x | Risk: {base_risk_pct * confidence_multiplier:.1%} | Confidence: {confidence:.1%}"
            )

            return position_size_btc

        except Exception as e:
            self.logger.error(f"Error calculating scalping position size: {e}")
            return 0.001

    def generate_signal(self, symbol: str, market_data: dict) -> Tuple[str, float]:
        """
        Main entry point for signal generation
        """
        try:
            # Convert market data to DataFrame if needed
            if "data" in market_data and isinstance(market_data["data"], list):
                df = pd.DataFrame(market_data["data"])
            else:
                # Create DataFrame from current market data
                current_price = market_data.get("close", market_data.get("price", 0))
                df = pd.DataFrame(
                    {
                        "close": [current_price] * 50,
                        "volume": [market_data.get("volume", 1000)] * 50,
                        "high": [market_data.get("high", current_price * 1.001)] * 50,
                        "low": [market_data.get("low", current_price * 0.999)] * 50,
                    }
                )

                # Add provided indicators
                for key in ["rsi", "macd", "signal_line"]:
                    if key in market_data:
                        df[key] = [market_data[key]] * 50

            # Ensure we have required columns
            if "close" not in df.columns:
                return "HOLD", 0.0

            # Analyze scalping opportunity
            signal, confidence, analysis = self.analyze_scalping_opportunity(df)

            # Record signal for performance tracking
            if signal in ["BUY", "SELL"]:
                self.last_signal_time = datetime.now()
                self.recent_signals.append(
                    {
                        "signal": signal,
                        "confidence": confidence,
                        "price": analysis.get("price", 0),
                        "time": self.last_signal_time,
                    }
                )

                # Keep only last 20 signals
                if len(self.recent_signals) > 20:
                    self.recent_signals.pop(0)

            return signal, confidence

        except Exception as e:
            self.logger.error(f"Error in signal generation: {e}")
            return "HOLD", 0.0

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        current_price: float,
        available_capital: float,
        leverage: float = 100.0,
        signal_confidence: float = 0.5,
    ) -> float:
        """
        Calculate position size for the strategy
        """
        return self.calculate_scalping_position_size(
            current_price, available_capital, signal_confidence
        )

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and performance
        """
        recent_performance = {}
        if self.recent_signals:
            recent_count = len(self.recent_signals)
            avg_confidence = (
                sum(s["confidence"] for s in self.recent_signals) / recent_count
            )
            recent_performance = {
                "recent_signals": recent_count,
                "avg_confidence": avg_confidence,
                "last_signal_time": self.last_signal_time,
            }

        return {
            "strategy_name": "HighLeverageScalpingStrategy",
            "leverage": self.leverage,
            "profit_target": f"{self.scalp_profit_target}% ({self.scalp_profit_target * self.leverage}% with leverage)",
            "stop_loss": f"{self.stop_loss_pct}% ({self.stop_loss_pct * self.leverage}% with leverage)",
            "min_confidence": self.min_confidence,
            **recent_performance,
        }
