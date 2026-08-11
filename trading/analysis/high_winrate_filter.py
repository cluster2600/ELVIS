"""
High Win Rate Signal Filter - 60%+ Win Rate Optimization
Advanced filtering system for trade quality improvement
"""

import logging
import os
from datetime import datetime, timedelta
from numbers import Real
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta


def classify_take_profit_regime(regime: object, volatility: object) -> Optional[str]:
    """Translate a validated topology observation into a take-profit profile."""
    if (
        not isinstance(regime, str)
        or not isinstance(volatility, Real)
        or isinstance(volatility, (bool, np.bool_))
    ):
        return None
    try:
        volatility_value = float(volatility)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(volatility_value) or volatility_value < 0.0:
        return None
    if volatility_value >= 0.05:
        return "CHOPPY"
    if regime in {"trending_strong", "trending_moderate"}:
        return "TRENDING"
    if regime in {"trending_weak", "ranging"}:
        return "RANGING"
    return None


class HighWinRateFilter:
    """
    Advanced signal filtering system designed to achieve 60%+ win rate
    through multi-indicator confluence and market regime detection
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Fewer, bigger trades parameters
        self.min_confidence_threshold = float(
            os.getenv("ELVIS_WINRATE_MIN_CONF", "0.85")
        )  # min confidence for trades (env-tunable)
        self.confluence_required = 4  # Minimum 4 out of 5 indicators must agree
        self.volatility_threshold = 0.015  # 1.5% volatility threshold (stricter)

        # Market regime parameters
        self.trend_strength_min = 0.6  # Minimum trend strength
        self.volume_spike_threshold = 1.5  # 1.5x average volume required

        self.logger.info(
            "🎯 Fewer, Bigger Trades Filter initialized - Target: Quality over Quantity"
        )
        self.logger.info(
            f"   - Min Confidence: {self.min_confidence_threshold*100:.0f}%"
        )
        self.logger.info(
            f"   - Confluence Required: {self.confluence_required}/5 indicators"
        )
        self.logger.info(f"   - Strategy: FEWER trades, BIGGER positions")

    def analyze_signal_quality(
        self, symbol: str, market_data: Dict, historical_data: pd.DataFrame
    ) -> Dict:
        """
        Comprehensive signal quality analysis for high win rate trading

        Args:
            symbol: Trading symbol
            market_data: Current market data
            historical_data: Historical price data

        Returns:
            Dict with signal analysis and recommendations
        """
        try:
            if historical_data.empty or len(historical_data) < 50:
                return self._reject_signal("Insufficient data")

            # 1. TECHNICAL CONFLUENCE ANALYSIS
            confluence_score = self._calculate_technical_confluence(
                market_data, historical_data
            )

            # 2. MARKET REGIME DETECTION
            market_regime = self._detect_market_regime(historical_data)

            # 3. VOLATILITY ANALYSIS
            volatility_analysis = self._analyze_volatility(historical_data)

            # 4. VOLUME CONFIRMATION
            volume_confirmation = self._check_volume_confirmation(historical_data)

            # 5. TIMING ANALYSIS
            timing_score = self._analyze_market_timing()

            # 6. AGGREGATE SIGNAL QUALITY
            overall_quality = self._calculate_overall_quality(
                confluence_score,
                market_regime,
                volatility_analysis,
                volume_confirmation,
                timing_score,
            )

            return {
                "symbol": symbol,
                "overall_quality": overall_quality,
                "confidence": overall_quality["confidence"],
                "recommendation": overall_quality["recommendation"],
                "confluence_score": confluence_score,
                "market_regime": market_regime,
                "volatility_analysis": volatility_analysis,
                "volume_confirmation": volume_confirmation,
                "timing_score": timing_score,
                "trade_approved": overall_quality["confidence"]
                >= self.min_confidence_threshold,
                "rejection_reason": overall_quality.get("rejection_reason"),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error in signal quality analysis: {e}")
            return self._reject_signal(f"Analysis error: {e}")

    def _calculate_technical_confluence(
        self, market_data: Dict, historical_data: pd.DataFrame
    ) -> Dict:
        """Calculate technical indicator confluence score"""
        try:
            signals = []
            reasons = []

            # RSI Analysis (Avoid extremes)
            rsi = market_data.get("rsi", 50)
            if 30 < rsi < 70:  # Sweet spot for entries
                signals.append(1)
                reasons.append(f"RSI in optimal range: {rsi:.1f}")
            elif 25 < rsi < 75:  # Acceptable range
                signals.append(0.5)
                reasons.append(f"RSI acceptable: {rsi:.1f}")
            else:
                signals.append(0)
                reasons.append(f"RSI extreme: {rsi:.1f}")

            # MACD Momentum
            macd = market_data.get("macd", 0)
            macd_signal = market_data.get("macd_signal", 0)
            if abs(macd - macd_signal) > 0.001:  # Strong divergence
                signals.append(1)
                reasons.append("Strong MACD momentum")
            else:
                signals.append(0.3)
                reasons.append("Weak MACD momentum")

            # Moving Average Alignment
            sma_20 = market_data.get("sma_20", 0)
            sma_50 = market_data.get("sma_50", 0)
            current_price = market_data.get("price", 0)

            if sma_20 > sma_50 and current_price > sma_20:  # Bullish alignment
                signals.append(1)
                reasons.append("Bullish MA alignment")
            elif sma_20 < sma_50 and current_price < sma_20:  # Bearish alignment
                signals.append(1)
                reasons.append("Bearish MA alignment")
            else:
                signals.append(0.2)
                reasons.append("Neutral MA alignment")

            # Bollinger Bands Analysis
            bb_upper = market_data.get("bb_upper", 0)
            bb_lower = market_data.get("bb_lower", 0)
            bb_middle = market_data.get("bb_middle", 0)

            if bb_lower < current_price < bb_upper:  # Within bands
                signals.append(1)
                reasons.append("Price within BB bands")
            else:
                signals.append(0.4)
                reasons.append("Price outside BB bands")

            # ADX Trend Strength
            adx = market_data.get("adx", 25)
            if adx > 25:  # Strong trend
                signals.append(1)
                reasons.append(f"Strong trend: ADX {adx:.1f}")
            elif adx > 20:
                signals.append(0.6)
                reasons.append(f"Moderate trend: ADX {adx:.1f}")
            else:
                signals.append(0.2)
                reasons.append(f"Weak trend: ADX {adx:.1f}")

            confluence_score = np.mean(signals)
            active_signals = sum(1 for s in signals if s >= 0.6)

            return {
                "score": confluence_score,
                "active_signals": active_signals,
                "total_signals": len(signals),
                "confluence_met": active_signals >= self.confluence_required,
                "individual_scores": signals,
                "reasons": reasons,
            }

        except Exception as e:
            self.logger.error(f"Technical confluence calculation error: {e}")
            return {"score": 0, "active_signals": 0, "confluence_met": False}

    def _detect_market_regime(self, historical_data: pd.DataFrame) -> Dict:
        """Detect current market regime for optimal trading conditions"""
        try:
            if len(historical_data) < 20:
                return {
                    "regime": "unknown",
                    "strength": 0,
                    "favorable": False,
                    "take_profit_regime": None,
                }

            # Calculate trend using linear regression
            closes = historical_data["close"].tail(20)
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)

            # Normalize slope to PERCENT MOVE OVER THE WHOLE WINDOW.
            # (Was per-bar percent: on 1m data that demanded >1%/minute
            # sustained for 20 minutes — mathematically unreachable, so the
            # filter vetoed 100% of signals forever.)
            avg_price = closes.mean()
            trend_strength = abs(slope) * len(closes) / avg_price * 100

            # Determine regime
            if trend_strength > 2:  # Strong trend
                regime = "trending_strong"
                favorable = True
            elif trend_strength > 1:  # Moderate trend
                regime = "trending_moderate"
                favorable = True
            elif trend_strength > 0.5:  # Weak trend
                regime = "trending_weak"
                favorable = False
            else:  # Ranging
                regime = "ranging"
                favorable = False

            # Calculate volatility
            returns = closes.pct_change().dropna()
            volatility = returns.std() * np.sqrt(1440)  # Dailyized from 1-min data

            return {
                "regime": regime,
                "trend_direction": "bullish" if slope > 0 else "bearish",
                "strength": trend_strength,
                "volatility": volatility,
                "favorable": favorable and volatility < 0.05,  # Not too volatile
                "take_profit_regime": classify_take_profit_regime(regime, volatility),
            }

        except Exception as e:
            self.logger.error(f"Market regime detection error: {e}")
            return {
                "regime": "unknown",
                "strength": 0,
                "favorable": False,
                "take_profit_regime": None,
            }

    def _analyze_volatility(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze volatility for trade timing"""
        try:
            if len(historical_data) < 10:
                return {"level": "unknown", "favorable": False}

            # Calculate recent volatility
            recent_closes = historical_data["close"].tail(10)
            returns = recent_closes.pct_change().dropna()
            volatility = returns.std()

            # Classify volatility level
            if volatility < 0.01:  # Low volatility
                level = "low"
                favorable = True
            elif volatility < 0.03:  # Normal volatility
                level = "normal"
                favorable = True
            elif volatility < 0.05:  # High volatility
                level = "high"
                favorable = False
            else:  # Extreme volatility
                level = "extreme"
                favorable = False

            return {
                "level": level,
                "value": volatility,
                "favorable": favorable,
                "within_threshold": volatility <= self.volatility_threshold,
            }

        except Exception as e:
            self.logger.error(f"Volatility analysis error: {e}")
            return {"level": "unknown", "favorable": False}

    def _check_volume_confirmation(self, historical_data: pd.DataFrame) -> Dict:
        """Check volume confirmation for signal strength"""
        try:
            if len(historical_data) < 5:
                return {"confirmed": False, "ratio": 0}

            # Compare recent volume to average
            recent_volume = historical_data["volume"].tail(3).mean()
            avg_volume = historical_data["volume"].tail(20).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            confirmed = volume_ratio >= self.volume_spike_threshold

            return {
                "confirmed": confirmed,
                "ratio": volume_ratio,
                "recent_volume": recent_volume,
                "average_volume": avg_volume,
                "threshold_met": volume_ratio >= self.volume_spike_threshold,
            }

        except Exception as e:
            self.logger.error(f"Volume confirmation error: {e}")
            return {"confirmed": False, "ratio": 0}

    def _analyze_market_timing(self) -> Dict:
        """Analyze optimal market timing"""
        try:
            now = datetime.now()
            hour = now.hour

            # Define optimal trading hours (UTC)
            optimal_hours = [7, 8, 9, 13, 14, 15, 16, 17]  # High liquidity periods
            avoid_hours = [22, 23, 0, 1, 2, 3, 4, 5]  # Low liquidity periods

            if hour in optimal_hours:
                timing_score = 1.0
                session = "optimal"
            elif hour in avoid_hours:
                timing_score = 0.2
                session = "avoid"
            else:
                timing_score = 0.6
                session = "acceptable"

            return {
                "score": timing_score,
                "session": session,
                "hour": hour,
                "favorable": timing_score >= 0.6,
            }

        except Exception as e:
            self.logger.error(f"Market timing analysis error: {e}")
            return {"score": 0.5, "session": "unknown", "favorable": False}

    def _calculate_overall_quality(
        self,
        confluence_score: Dict,
        market_regime: Dict,
        volatility_analysis: Dict,
        volume_confirmation: Dict,
        timing_score: Dict,
    ) -> Dict:
        """Calculate overall signal quality and recommendation"""
        try:
            # Weight different factors
            weights = {
                "confluence": 0.40,  # 40% - Most important
                "regime": 0.25,  # 25% - Market conditions
                "volatility": 0.15,  # 15% - Risk management
                "volume": 0.10,  # 10% - Confirmation
                "timing": 0.10,  # 10% - Market timing
            }

            # Calculate weighted score
            confluence_component = (
                confluence_score.get("score", 0) * weights["confluence"]
            )
            regime_component = (
                (market_regime.get("strength", 0) / 3) * weights["regime"]
                if market_regime.get("favorable")
                else 0
            )
            volatility_component = (
                1 if volatility_analysis.get("favorable") else 0.3
            ) * weights["volatility"]
            volume_component = (
                1 if volume_confirmation.get("confirmed") else 0.5
            ) * weights["volume"]
            timing_component = timing_score.get("score", 0.5) * weights["timing"]

            overall_confidence = (
                confluence_component
                + regime_component
                + volatility_component
                + volume_component
                + timing_component
            )

            # Determine recommendation
            if overall_confidence >= 0.75:
                recommendation = "STRONG_BUY" if overall_confidence >= 0.85 else "BUY"
                approved = True
                rejection_reason = None
            elif overall_confidence >= 0.60:
                recommendation = "WEAK_BUY"
                approved = False
                rejection_reason = "Confidence below threshold"
            else:
                recommendation = "REJECT"
                approved = False
                rejection_reason = f"Low quality signal: {overall_confidence:.2f}"

            # Additional rejection criteria
            if not confluence_score.get("confluence_met", False):
                recommendation = "REJECT"
                approved = False
                rejection_reason = "Insufficient confluence"

            if not market_regime.get("favorable", False):
                recommendation = "REJECT"
                approved = False
                rejection_reason = "Unfavorable market regime"

            return {
                "confidence": overall_confidence,
                "recommendation": recommendation,
                "approved": approved,
                "rejection_reason": rejection_reason,
                "components": {
                    "confluence": confluence_component,
                    "regime": regime_component,
                    "volatility": volatility_component,
                    "volume": volume_component,
                    "timing": timing_component,
                },
                "weights": weights,
            }

        except Exception as e:
            self.logger.error(f"Overall quality calculation error: {e}")
            return {
                "confidence": 0,
                "recommendation": "REJECT",
                "approved": False,
                "rejection_reason": f"Calculation error: {e}",
            }

    def _reject_signal(self, reason: str) -> Dict:
        """Return a rejected signal with reason"""
        return {
            "overall_quality": {
                "confidence": 0,
                "recommendation": "REJECT",
                "approved": False,
                "rejection_reason": reason,
            },
            "trade_approved": False,
            "rejection_reason": reason,
            "timestamp": datetime.now(),
        }

    def get_filter_stats(self) -> Dict:
        """Get filter performance statistics"""
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "confluence_required": self.confluence_required,
            "volatility_threshold": self.volatility_threshold,
            "trend_strength_min": self.trend_strength_min,
            "volume_spike_threshold": self.volume_spike_threshold,
        }


if __name__ == "__main__":
    # Test the filter
    logger = logging.getLogger("HighWinRateFilter")
    logging.basicConfig(level=logging.INFO)

    filter_system = HighWinRateFilter(logger)

    # Mock data for testing
    mock_market_data = {
        "price": 97000,
        "rsi": 45,
        "macd": 0.002,
        "macd_signal": 0.001,
        "sma_20": 96500,
        "sma_50": 96000,
        "adx": 28,
        "bb_upper": 97500,
        "bb_lower": 96500,
        "bb_middle": 97000,
    }

    # Mock historical data
    mock_historical = pd.DataFrame(
        {
            "close": np.random.normal(97000, 500, 50),
            "volume": np.random.normal(1000, 200, 50),
        }
    )

    result = filter_system.analyze_signal_quality(
        "BTCUSDT", mock_market_data, mock_historical
    )

    print("=== HIGH WIN RATE FILTER TEST ===")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Recommendation: {result['overall_quality']['recommendation']}")
    print(f"Trade Approved: {result['trade_approved']}")
    if result.get("rejection_reason"):
        print(f"Rejection Reason: {result['rejection_reason']}")
