"""
Market Regime Detection for High Win Rate Trading
Identifies optimal market conditions for trade execution
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta
from scipy import stats
from sklearn.cluster import KMeans


class MarketRegimeDetector:
    """
    Advanced market regime detection system to identify
    optimal trading conditions for 60%+ win rate
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Regime classification parameters
        self.lookback_period = 50
        self.trend_threshold = 0.02  # 2% minimum for strong trend
        self.volatility_periods = [10, 20, 50]

        # Trading session definitions (UTC hours)
        self.trading_sessions = {
            "asian": {"start": 0, "end": 8, "liquidity": "medium"},
            "london": {"start": 8, "end": 16, "liquidity": "high"},
            "new_york": {"start": 13, "end": 21, "liquidity": "high"},
            "overlap": {"start": 13, "end": 16, "liquidity": "highest"},
        }

        self.logger.info("📊 Market Regime Detector initialized")
        self.logger.info(f"   - Lookback period: {self.lookback_period}")
        self.logger.info(f"   - Trend threshold: {self.trend_threshold*100:.1f}%")

    def detect_current_regime(self, historical_data: pd.DataFrame) -> Dict:
        """
        Detect current market regime based on multiple factors

        Returns comprehensive regime analysis for trading decisions
        """
        try:
            if len(historical_data) < self.lookback_period:
                return self._get_unknown_regime("Insufficient historical data")

            # 1. TREND ANALYSIS
            trend_analysis = self._analyze_trend(historical_data)

            # 2. VOLATILITY REGIME
            volatility_regime = self._classify_volatility_regime(historical_data)

            # 3. MOMENTUM ANALYSIS
            momentum_analysis = self._analyze_momentum(historical_data)

            # 4. MARKET MICROSTRUCTURE
            microstructure = self._analyze_market_microstructure(historical_data)

            # 5. TRADING SESSION ANALYSIS
            session_analysis = self._analyze_trading_session()

            # 6. REGIME CLASSIFICATION
            regime_classification = self._classify_regime(
                trend_analysis,
                volatility_regime,
                momentum_analysis,
                microstructure,
                session_analysis,
            )

            return {
                "regime": regime_classification,
                "trend_analysis": trend_analysis,
                "volatility_regime": volatility_regime,
                "momentum_analysis": momentum_analysis,
                "microstructure": microstructure,
                "session_analysis": session_analysis,
                "trading_recommendation": self._get_trading_recommendation(
                    regime_classification
                ),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return self._get_unknown_regime(f"Analysis error: {e}")

    def _analyze_trend(self, historical_data: pd.DataFrame) -> Dict:
        """Comprehensive trend analysis"""
        try:
            closes = historical_data["close"].tail(self.lookback_period)

            # Linear regression trend
            x = np.arange(len(closes))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, closes)

            # Normalize slope to percentage
            avg_price = closes.mean()
            trend_slope_pct = (slope / avg_price) * 100

            # Moving average trend
            sma_short = closes.rolling(10).mean().iloc[-1]
            sma_long = closes.rolling(20).mean().iloc[-1]
            ma_trend = (sma_short - sma_long) / sma_long * 100

            # ADX calculation for trend strength
            high = historical_data["high"].tail(self.lookback_period)
            low = historical_data["low"].tail(self.lookback_period)

            try:
                adx = ta.trend.ADXIndicator(high, low, closes).adx().iloc[-1]
            except:
                adx = 25  # Default neutral value

            # Trend classification
            if abs(trend_slope_pct) > self.trend_threshold:
                if trend_slope_pct > 0:
                    trend_type = "strong_uptrend"
                    strength = min(abs(trend_slope_pct) / self.trend_threshold, 3.0)
                else:
                    trend_type = "strong_downtrend"
                    strength = min(abs(trend_slope_pct) / self.trend_threshold, 3.0)
            elif abs(trend_slope_pct) > self.trend_threshold / 2:
                trend_type = "weak_uptrend" if trend_slope_pct > 0 else "weak_downtrend"
                strength = abs(trend_slope_pct) / self.trend_threshold
            else:
                trend_type = "sideways"
                strength = 0.5

            return {
                "type": trend_type,
                "strength": strength,
                "slope_pct": trend_slope_pct,
                "r_squared": r_value**2,
                "p_value": p_value,
                "ma_trend_pct": ma_trend,
                "adx": adx,
                "favorable_for_trading": strength > 1.0 and adx > 25,
            }

        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {"type": "unknown", "strength": 0, "favorable_for_trading": False}

    def _classify_volatility_regime(self, historical_data: pd.DataFrame) -> Dict:
        """Classify volatility regime"""
        try:
            closes = historical_data["close"].tail(self.lookback_period)

            # Calculate returns and volatilities for different periods
            returns = closes.pct_change().dropna()

            volatilities = {}
            for period in self.volatility_periods:
                if len(returns) >= period:
                    vol = returns.tail(period).std() * np.sqrt(1440)  # Annualized
                    volatilities[f"vol_{period}"] = vol

            # Current volatility (short-term)
            current_vol = volatilities.get("vol_10", 0.02)

            # Long-term average volatility
            long_term_vol = volatilities.get("vol_50", 0.02)

            # Volatility ratio
            vol_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0

            # Classify volatility regime
            if current_vol < 0.015:  # Very low volatility
                vol_regime = "low"
                trading_favorable = True
            elif current_vol < 0.03:  # Normal volatility
                vol_regime = "normal"
                trading_favorable = True
            elif current_vol < 0.05:  # High volatility
                vol_regime = "high"
                trading_favorable = False
            else:  # Extreme volatility
                vol_regime = "extreme"
                trading_favorable = False

            return {
                "regime": vol_regime,
                "current_vol": current_vol,
                "long_term_vol": long_term_vol,
                "vol_ratio": vol_ratio,
                "volatilities": volatilities,
                "trading_favorable": trading_favorable,
                "vol_expanding": vol_ratio > 1.2,
                "vol_contracting": vol_ratio < 0.8,
            }

        except Exception as e:
            self.logger.error(f"Volatility regime error: {e}")
            return {"regime": "unknown", "trading_favorable": False}

    def _analyze_momentum(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        try:
            closes = historical_data["close"].tail(self.lookback_period)

            # RSI
            try:
                rsi = ta.momentum.RSIIndicator(closes).rsi().iloc[-1]
            except:
                rsi = 50

            # MACD
            try:
                macd_indicator = ta.trend.MACD(closes)
                macd = macd_indicator.macd().iloc[-1]
                macd_signal = macd_indicator.macd_signal().iloc[-1]
                macd_histogram = macd_indicator.macd_diff().iloc[-1]
            except:
                macd = macd_signal = macd_histogram = 0

            # Momentum classification
            momentum_signals = []

            # RSI momentum
            if 40 < rsi < 60:
                momentum_signals.append("neutral")
            elif 30 < rsi < 70:
                momentum_signals.append("healthy")
            else:
                momentum_signals.append("extreme")

            # MACD momentum
            if abs(macd_histogram) > 0.001:
                momentum_signals.append("strong")
            else:
                momentum_signals.append("weak")

            # Overall momentum assessment
            if "healthy" in momentum_signals and "strong" in momentum_signals:
                momentum_quality = "excellent"
                momentum_score = 1.0
            elif "healthy" in momentum_signals or "strong" in momentum_signals:
                momentum_quality = "good"
                momentum_score = 0.7
            elif "neutral" in momentum_signals:
                momentum_quality = "neutral"
                momentum_score = 0.5
            else:
                momentum_quality = "poor"
                momentum_score = 0.2

            return {
                "quality": momentum_quality,
                "score": momentum_score,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
                "signals": momentum_signals,
                "trading_favorable": momentum_score >= 0.7,
            }

        except Exception as e:
            self.logger.error(f"Momentum analysis error: {e}")
            return {"quality": "unknown", "score": 0.5, "trading_favorable": False}

    def _analyze_market_microstructure(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze market microstructure indicators"""
        try:
            if "volume" not in historical_data.columns:
                return {"liquidity": "unknown", "trading_favorable": False}

            volumes = historical_data["volume"].tail(20)
            closes = historical_data["close"].tail(20)

            # Volume analysis
            avg_volume = volumes.mean()
            recent_volume = volumes.tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Price-volume relationship
            price_changes = closes.pct_change().tail(10)
            volume_changes = volumes.pct_change().tail(10)

            # Correlation between price and volume changes
            try:
                pv_correlation = np.corrcoef(
                    price_changes.dropna(), volume_changes.dropna()
                )[0, 1]
                if np.isnan(pv_correlation):
                    pv_correlation = 0
            except:
                pv_correlation = 0

            # Liquidity assessment
            if volume_ratio > 1.5:
                liquidity = "high"
                liquidity_score = 1.0
            elif volume_ratio > 1.0:
                liquidity = "normal"
                liquidity_score = 0.7
            elif volume_ratio > 0.5:
                liquidity = "low"
                liquidity_score = 0.4
            else:
                liquidity = "very_low"
                liquidity_score = 0.2

            return {
                "liquidity": liquidity,
                "liquidity_score": liquidity_score,
                "volume_ratio": volume_ratio,
                "pv_correlation": pv_correlation,
                "avg_volume": avg_volume,
                "recent_volume": recent_volume,
                "trading_favorable": liquidity_score >= 0.7,
            }

        except Exception as e:
            self.logger.error(f"Microstructure analysis error: {e}")
            return {"liquidity": "unknown", "trading_favorable": False}

    def _analyze_trading_session(self) -> Dict:
        """Analyze current trading session"""
        try:
            current_hour = datetime.utcnow().hour

            # Determine active sessions
            active_sessions = []
            for session_name, session_info in self.trading_sessions.items():
                if session_name == "overlap":
                    continue  # Handle overlap separately

                start_hour = session_info["start"]
                end_hour = session_info["end"]

                if start_hour <= current_hour < end_hour:
                    active_sessions.append(session_name)

            # Check for overlap session
            if (
                self.trading_sessions["overlap"]["start"]
                <= current_hour
                < self.trading_sessions["overlap"]["end"]
            ):
                active_sessions.append("overlap")

            # Determine session quality
            if "overlap" in active_sessions:
                session_quality = "excellent"
                liquidity_level = "highest"
                session_score = 1.0
            elif any(session in active_sessions for session in ["london", "new_york"]):
                session_quality = "good"
                liquidity_level = "high"
                session_score = 0.8
            elif "asian" in active_sessions:
                session_quality = "moderate"
                liquidity_level = "medium"
                session_score = 0.6
            else:
                session_quality = "poor"
                liquidity_level = "low"
                session_score = 0.3

            return {
                "active_sessions": active_sessions,
                "quality": session_quality,
                "liquidity_level": liquidity_level,
                "score": session_score,
                "current_hour_utc": current_hour,
                "trading_favorable": session_score >= 0.6,
            }

        except Exception as e:
            self.logger.error(f"Trading session analysis error: {e}")
            return {"quality": "unknown", "trading_favorable": False}

    def _classify_regime(
        self,
        trend_analysis: Dict,
        volatility_regime: Dict,
        momentum_analysis: Dict,
        microstructure: Dict,
        session_analysis: Dict,
    ) -> Dict:
        """Classify overall market regime"""
        try:
            # Calculate regime score
            factors = [
                trend_analysis.get("favorable_for_trading", False),
                volatility_regime.get("trading_favorable", False),
                momentum_analysis.get("trading_favorable", False),
                microstructure.get("trading_favorable", False),
                session_analysis.get("trading_favorable", False),
            ]

            favorable_factors = sum(factors)
            regime_score = favorable_factors / len(factors)

            # Classify regime
            if regime_score >= 0.8:
                regime_class = "optimal"
                trade_recommendation = "aggressive"
            elif regime_score >= 0.6:
                regime_class = "favorable"
                trade_recommendation = "normal"
            elif regime_score >= 0.4:
                regime_class = "neutral"
                trade_recommendation = "conservative"
            else:
                regime_class = "unfavorable"
                trade_recommendation = "avoid"

            # Special conditions
            if volatility_regime.get("regime") == "extreme":
                regime_class = "unfavorable"
                trade_recommendation = "avoid"

            if trend_analysis.get("type") == "sideways" and regime_score < 0.7:
                trade_recommendation = "avoid"

            return {
                "class": regime_class,
                "score": regime_score,
                "favorable_factors": favorable_factors,
                "total_factors": len(factors),
                "trade_recommendation": trade_recommendation,
                "should_trade": regime_score >= 0.6,
                "confidence": regime_score,
            }

        except Exception as e:
            self.logger.error(f"Regime classification error: {e}")
            return {
                "class": "unknown",
                "score": 0,
                "trade_recommendation": "avoid",
                "should_trade": False,
            }

    def _get_trading_recommendation(self, regime_classification: Dict) -> Dict:
        """Get specific trading recommendations based on regime"""
        recommendation = regime_classification.get("trade_recommendation", "avoid")

        if recommendation == "aggressive":
            return {
                "action": "trade_aggressively",
                "position_size_multiplier": 1.5,
                "max_concurrent_trades": 3,
                "confidence_threshold": 0.70,
                "message": "Optimal conditions for aggressive trading",
            }
        elif recommendation == "normal":
            return {
                "action": "trade_normally",
                "position_size_multiplier": 1.0,
                "max_concurrent_trades": 2,
                "confidence_threshold": 0.75,
                "message": "Favorable conditions for normal trading",
            }
        elif recommendation == "conservative":
            return {
                "action": "trade_conservatively",
                "position_size_multiplier": 0.5,
                "max_concurrent_trades": 1,
                "confidence_threshold": 0.85,
                "message": "Neutral conditions - trade conservatively",
            }
        else:
            return {
                "action": "avoid_trading",
                "position_size_multiplier": 0,
                "max_concurrent_trades": 0,
                "confidence_threshold": 0.95,
                "message": "Unfavorable conditions - avoid trading",
            }

    def _get_unknown_regime(self, reason: str) -> Dict:
        """Return unknown regime with safe defaults"""
        return {
            "regime": {
                "class": "unknown",
                "score": 0,
                "trade_recommendation": "avoid",
                "should_trade": False,
            },
            "trading_recommendation": {
                "action": "avoid_trading",
                "message": f"Unknown regime: {reason}",
            },
            "timestamp": datetime.now(),
        }


if __name__ == "__main__":
    # Test the regime detector
    logger = logging.getLogger("MarketRegimeDetector")
    logging.basicConfig(level=logging.INFO)

    detector = MarketRegimeDetector(logger)

    # Mock historical data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1min")
    mock_data = pd.DataFrame(
        {
            "close": np.cumsum(np.random.randn(100) * 0.01) + 97000,
            "high": np.cumsum(np.random.randn(100) * 0.01) + 97050,
            "low": np.cumsum(np.random.randn(100) * 0.01) + 96950,
            "volume": np.random.lognormal(10, 0.5, 100),
        },
        index=dates,
    )

    result = detector.detect_current_regime(mock_data)

    print("=== MARKET REGIME DETECTION TEST ===")
    print(f"Regime Class: {result['regime']['class']}")
    print(f"Regime Score: {result['regime']['score']:.2%}")
    print(f"Should Trade: {result['regime']['should_trade']}")
    print(f"Recommendation: {result['trading_recommendation']['action']}")
    print(f"Message: {result['trading_recommendation']['message']}")
