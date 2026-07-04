#!/usr/bin/env python3
"""
LLM-Enhanced Training for ELVIS Trading System
Integrates local LLM analysis directly into the training process
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.ai.llm_market_analyzer import LLMConfig, LLMMarketAnalyzer
from utils.logging_utils import setup_logger


class LLMEnhancedTrainer:
    """Training system that integrates LLM analysis into feature engineering"""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        enable_llm: bool = True,
        cache_llm_responses: bool = True,
    ):

        self.enable_llm = enable_llm
        self.cache_llm_responses = cache_llm_responses
        self.llm_cache = {}
        self.logger = setup_logger("llm_trainer")

        # Initialize LLM analyzer if enabled
        if self.enable_llm and llm_config:
            self.llm_analyzer = LLMMarketAnalyzer(llm_config)
            self.logger.info("🧠 LLM analyzer initialized for training enhancement")
        else:
            self.llm_analyzer = None
            self.logger.info("📊 Training without LLM enhancement")

    def prepare_llm_features(
        self, market_data: pd.DataFrame, batch_size: int = 10
    ) -> pd.DataFrame:
        """
        Prepare LLM-enhanced features for training data

        Args:
            market_data: DataFrame with OHLCV and technical indicators
            batch_size: Number of samples to process at once

        Returns:
            DataFrame with additional LLM-derived features
        """

        if not self.enable_llm or self.llm_analyzer is None:
            self.logger.info("📊 Skipping LLM feature generation - LLM disabled")
            return market_data

        self.logger.info(
            f"🧠 Generating LLM features for {len(market_data)} samples..."
        )

        # Initialize LLM feature columns
        llm_features = {
            "llm_sentiment_score": [],
            "llm_confidence_score": [],
            "llm_bullish_probability": [],
            "llm_risk_score": [],
            "llm_volatility_prediction": [],
            "llm_trend_strength": [],
        }

        # Process in batches to avoid overwhelming the LLM
        total_batches = (len(market_data) + batch_size - 1) // batch_size

        for i in range(0, len(market_data), batch_size):
            batch_end = min(i + batch_size, len(market_data))
            batch_data = market_data.iloc[i:batch_end]

            self.logger.info(
                f"🔄 Processing LLM batch {i//batch_size + 1}/{total_batches}"
            )

            # Process each row in the batch
            for idx, row in batch_data.iterrows():
                try:
                    # Create cache key for this sample
                    cache_key = self._create_cache_key(row)

                    # Check cache first
                    if self.cache_llm_responses and cache_key in self.llm_cache:
                        features = self.llm_cache[cache_key]
                        self.logger.debug(
                            f"📋 Using cached LLM features for sample {idx}"
                        )
                    else:
                        # Get LLM analysis (synchronously to avoid event loop issues)
                        features = self._get_llm_features_sync(row)

                        # Cache the result
                        if self.cache_llm_responses:
                            self.llm_cache[cache_key] = features

                        # Add small delay between LLM requests to avoid overwhelming slow LLMs
                        import time

                        time.sleep(0.5)  # 500ms delay between requests

                    # Add features to lists
                    for feature_name, value in features.items():
                        llm_features[feature_name].append(value)

                except Exception as e:
                    self.logger.warning(f"⚠️ LLM analysis failed for sample {idx}: {e}")
                    # Add default values for failed analysis
                    for feature_name in llm_features.keys():
                        llm_features[feature_name].append(0.5)  # Neutral default

        # Add LLM features to the DataFrame
        for feature_name, values in llm_features.items():
            market_data[feature_name] = values

        self.logger.info(f"✅ Added {len(llm_features)} LLM-enhanced features")
        return market_data

    async def _get_llm_features_for_sample(self, row: pd.Series) -> Dict[str, float]:
        """Get LLM analysis for a single market sample"""

        # Prepare market data for LLM
        market_data = {
            "symbol": "BTCUSDT",
            "price": float(row.get("close", row.get("price", 0))),
            "price_change_24h": float(row.get("price_change_pct", 0)) * 100,
            "volume_24h": float(row.get("volume", 0)),
            "volatility": float(
                row.get(
                    "volatility",
                    abs(row.get("high", 0) - row.get("low", 0))
                    / row.get("close", 1)
                    * 100,
                )
            ),
        }

        # Prepare technical analysis
        technical_analysis = {
            "rsi": float(row.get("rsi", 50)),
            "macd": float(row.get("macd", 0)),
            "trend": self._categorize_trend(row),
            "bollinger_position": self._get_bollinger_position(row),
            "moving_averages": {
                "sma_20": float(row.get("sma_20", market_data["price"])),
                "sma_50": float(row.get("sma_50", market_data["price"])),
            },
        }

        try:
            # Get LLM analysis (with generous timeout for slow LLMs)
            timeout_seconds = int(os.getenv("ELVIS_LLM_TIMEOUT", "120"))
            analysis = await asyncio.wait_for(
                self.llm_analyzer.analyze_market_conditions(
                    market_data=market_data, technical_analysis=technical_analysis
                ),
                timeout=timeout_seconds,  # Use configured timeout (default 2 minutes)
            )

            # Extract numerical features from LLM analysis
            features = self._extract_numerical_features(analysis)

        except asyncio.TimeoutError:
            self.logger.warning("⏰ LLM analysis timeout - using fallback features")
            features = self._get_fallback_features(market_data, technical_analysis)
        except Exception as e:
            self.logger.warning(f"❌ LLM analysis error: {e} - using fallback features")
            features = self._get_fallback_features(market_data, technical_analysis)

        return features

    def _extract_numerical_features(self, analysis) -> Dict[str, float]:
        """Extract numerical features from LLM analysis"""

        # Map sentiment to numerical score
        sentiment_scores = {
            "bullish": 0.8,
            "bearish": 0.2,
            "neutral": 0.5,
            "volatile": 0.6,
        }

        # Map confidence to numerical score
        confidence_scores = {"very_high": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}

        sentiment_score = sentiment_scores.get(analysis.sentiment.value, 0.5)
        confidence_score = confidence_scores.get(analysis.confidence.value, 0.5)

        # Extract additional features
        bullish_prob = max(
            0, analysis.price_prediction.get("direction", 0)
        ) * analysis.price_prediction.get("magnitude", 0.5)

        # Risk assessment
        risk_level = analysis.risk_assessment.get("risk_level", "medium")
        risk_scores = {"low": 0.3, "medium": 0.5, "high": 0.8}
        risk_score = risk_scores.get(risk_level, 0.5)

        # Volatility prediction (from risk assessment)
        volatility_prediction = (
            0.7 if analysis.risk_assessment.get("volatility_warning", False) else 0.4
        )

        # Trend strength (derived from confidence and sentiment alignment)
        trend_strength = confidence_score * abs(sentiment_score - 0.5) * 2

        return {
            "llm_sentiment_score": sentiment_score,
            "llm_confidence_score": confidence_score,
            "llm_bullish_probability": bullish_prob,
            "llm_risk_score": risk_score,
            "llm_volatility_prediction": volatility_prediction,
            "llm_trend_strength": trend_strength,
        }

    def _get_llm_features_sync(self, row: pd.Series) -> Dict[str, float]:
        """Get LLM analysis for a single market sample (synchronous version)"""

        # Prepare market data for LLM
        market_data = {
            "symbol": "BTCUSDT",
            "price": float(row.get("close", row.get("price", 0))),
            "price_change_24h": float(row.get("price_change_pct", 0)) * 100,
            "volume_24h": float(row.get("volume", 0)),
            "volatility": float(
                row.get(
                    "volatility",
                    abs(row.get("high", 0) - row.get("low", 0))
                    / row.get("close", 1)
                    * 100,
                )
            ),
        }

        # Prepare technical analysis
        technical_analysis = {
            "rsi": float(row.get("rsi", 50)),
            "macd": float(row.get("macd", 0)),
            "trend": self._categorize_trend(row),
            "bollinger_position": self._get_bollinger_position(row),
            "moving_averages": {
                "sma_20": float(row.get("sma_20", market_data["price"])),
                "sma_50": float(row.get("sma_50", market_data["price"])),
            },
        }

        try:
            # Use a synchronous approach to call the LLM
            import json

            import requests

            # Prepare the request to the local LLM
            llm_prompt = self._create_analysis_prompt(market_data, technical_analysis)

            # Make synchronous API call to local LM Studio
            response = requests.post(
                f"{self.llm_analyzer.config.base_url}/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.llm_analyzer.config.model,
                    "prompt": llm_prompt,
                    "max_tokens": 50,
                    "temperature": 0.1,
                    "stop": ["\n", "<|", "Analysis:", "We"],
                },
                timeout=15,
            )

            if response.status_code == 200:
                response_data = response.json()
                analysis_text = response_data.get("choices", [{}])[0].get("text", "")

                if analysis_text.strip():
                    # Extract numerical features from LLM response
                    features = self._extract_numerical_features_from_text(analysis_text)

                    # Check if we got meaningful features (not all defaults)
                    non_default_count = sum(1 for v in features.values() if v != 0.5)
                    if non_default_count >= 2:  # At least 2 non-default features
                        self.logger.debug(f"🧠 LLM analysis successful: {features}")
                        return features
                    else:
                        self.logger.debug(
                            f"⚠️ LLM returned mostly defaults, using intelligent fallback"
                        )
                else:
                    self.logger.warning("⚠️ LLM returned empty response")

            else:
                self.logger.warning(
                    f"⚠️ LLM API error {response.status_code}: {response.text}"
                )

        except Exception as e:
            self.logger.warning(f"⚠️ LLM API call failed: {e}")

        # Fallback to rule-based features
        self.logger.debug("📊 Using fallback features instead of LLM")
        return self._get_fallback_features(market_data, technical_analysis)

    def _create_analysis_prompt(
        self, market_data: Dict, technical_analysis: Dict
    ) -> str:
        """Create analysis prompt for the LLM"""

        return f"""You are a trading analyst. Analyze Bitcoin data and respond ONLY with numerical scores:

Price: ${market_data['price']:,.2f} ({market_data['price_change_24h']:+.2f}% change)
RSI: {technical_analysis['rsi']:.1f}
Trend: {technical_analysis['trend']}
Volatility: {market_data['volatility']:.2f}%

Respond with EXACTLY this format (replace X with 0-1 values):
sentiment:0.X confidence:0.X bullish:0.X risk:0.X volatility:0.X trend:0.X"""

    def _extract_numerical_features_from_text(
        self, analysis_text: str
    ) -> Dict[str, float]:
        """Extract numerical features from LLM text response"""

        import re

        # Initialize default values
        features = {
            "llm_sentiment_score": 0.5,
            "llm_confidence_score": 0.5,
            "llm_bullish_probability": 0.5,
            "llm_risk_score": 0.5,
            "llm_volatility_prediction": 0.5,
            "llm_trend_strength": 0.5,
        }

        # Try to extract scores using regex patterns
        patterns = {
            "llm_sentiment_score": r"sentiment[:\s]+([0-9.]+)",
            "llm_confidence_score": r"confidence[:\s]+([0-9.]+)",
            "llm_bullish_probability": r"bullish[:\s]+([0-9.]+)",
            "llm_risk_score": r"risk[:\s]+([0-9.]+)",
            "llm_volatility_prediction": r"volatility[:\s]+([0-9.]+)",
            "llm_trend_strength": r"trend[:\s]+([0-9.]+)",
        }

        for feature_name, pattern in patterns.items():
            matches = re.findall(pattern, analysis_text.lower())
            if matches:
                try:
                    value = float(matches[0])
                    # Clamp to [0, 1] range
                    features[feature_name] = max(0.0, min(1.0, value))
                except (ValueError, IndexError):
                    pass

        return features

    def _get_fallback_features(
        self, market_data: Dict, technical_analysis: Dict
    ) -> Dict[str, float]:
        """Generate intelligent rule-based features when LLM is unavailable"""

        price_change = market_data.get("price_change_24h", 0)
        rsi = technical_analysis.get("rsi", 50)
        volatility = market_data.get("volatility", 10)
        macd = technical_analysis.get("macd", 0)
        trend = technical_analysis.get("trend", "neutral")

        # Intelligent sentiment based on multiple factors
        sentiment_factors = []
        if price_change > 1:
            sentiment_factors.append(0.6 + min(0.3, price_change / 10))
        elif price_change < -1:
            sentiment_factors.append(0.4 - min(0.3, abs(price_change) / 10))
        else:
            sentiment_factors.append(0.5)

        if rsi > 70:
            sentiment_factors.append(0.8)  # Overbought but bullish
        elif rsi < 30:
            sentiment_factors.append(0.2)  # Oversold, bearish
        elif rsi > 50:
            sentiment_factors.append(0.6)
        else:
            sentiment_factors.append(0.4)

        if macd > 0:
            sentiment_factors.append(0.6)
        else:
            sentiment_factors.append(0.4)

        sentiment_score = sum(sentiment_factors) / len(sentiment_factors)

        # Confidence based on consistency of signals
        signal_consistency = (
            0.8 if abs(rsi - 50) > 15 else 0.5
        )  # High RSI deviation = more confident
        confidence_score = max(0.2, signal_consistency - volatility / 50)

        # Bullish probability with multiple indicators
        bullish_indicators = 0
        total_indicators = 0

        if price_change != 0:
            bullish_indicators += 1 if price_change > 0 else 0
            total_indicators += 1
        if rsi != 50:
            bullish_indicators += 1 if rsi > 50 else 0
            total_indicators += 1
        if trend != "neutral":
            bullish_indicators += 1 if trend == "bullish" else 0
            total_indicators += 1
        if macd != 0:
            bullish_indicators += 1 if macd > 0 else 0
            total_indicators += 1

        bullish_prob = (
            bullish_indicators / max(1, total_indicators)
            if total_indicators > 0
            else 0.5
        )

        # Risk assessment
        risk_factors = [volatility / 30, abs(price_change) / 15]
        if rsi > 80 or rsi < 20:
            risk_factors.append(0.8)  # Extreme RSI = risky
        risk_score = min(1.0, sum(risk_factors) / len(risk_factors))

        # Volatility prediction based on recent volatility and price action
        vol_prediction = min(1.0, (volatility / 20) + abs(price_change) / 20)

        # Trend strength from momentum indicators
        trend_strength = min(
            1.0, (abs(price_change) / 5 + abs(macd) * 10 + abs(rsi - 50) / 25) / 3
        )

        features = {
            "llm_sentiment_score": max(0.1, min(0.9, sentiment_score)),
            "llm_confidence_score": max(0.2, min(0.8, confidence_score)),
            "llm_bullish_probability": max(0.1, min(0.9, bullish_prob)),
            "llm_risk_score": max(0.1, min(0.9, risk_score)),
            "llm_volatility_prediction": max(0.1, min(0.9, vol_prediction)),
            "llm_trend_strength": max(0.1, min(0.9, trend_strength)),
        }

        return features

    def _categorize_trend(self, row: pd.Series) -> str:
        """Categorize trend from technical indicators"""

        price = row.get("close", row.get("price", 0))
        sma_20 = row.get("sma_20", price)
        sma_50 = row.get("sma_50", price)

        if price > sma_20 > sma_50:
            return "bullish"
        elif price < sma_20 < sma_50:
            return "bearish"
        else:
            return "neutral"

    def _get_bollinger_position(self, row: pd.Series) -> str:
        """Determine Bollinger Bands position"""

        price = row.get("close", row.get("price", 0))
        bb_upper = row.get("bb_upper", price * 1.02)
        bb_lower = row.get("bb_lower", price * 0.98)

        if price > bb_upper:
            return "upper"
        elif price < bb_lower:
            return "lower"
        else:
            return "middle"

    def _create_cache_key(self, row: pd.Series) -> str:
        """Create cache key for LLM response caching"""

        # Use key market indicators to create cache key
        key_data = {
            "price": round(float(row.get("close", row.get("price", 0))), 2),
            "volume": round(float(row.get("volume", 0)), -3),  # Round to thousands
            "rsi": round(float(row.get("rsi", 50)), 1),
            "change": round(float(row.get("price_change_pct", 0)) * 100, 1),
        }

        return json.dumps(key_data, sort_keys=True)

    def save_llm_cache(self, filepath: str):
        """Save LLM response cache to file"""

        if self.llm_cache:
            with open(filepath, "w") as f:
                json.dump(self.llm_cache, f, indent=2)
            self.logger.info(
                f"💾 Saved {len(self.llm_cache)} LLM responses to cache: {filepath}"
            )

    def load_llm_cache(self, filepath: str):
        """Load LLM response cache from file"""

        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    self.llm_cache = json.load(f)
                self.logger.info(
                    f"📂 Loaded {len(self.llm_cache)} cached LLM responses from: {filepath}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load LLM cache: {e}")
                self.llm_cache = {}


def create_llm_config_from_env() -> LLMConfig:
    """Create LLM configuration from environment variables"""

    return LLMConfig(
        provider=os.getenv("ELVIS_LLM_PROVIDER", "local"),
        model=os.getenv("ELVIS_LLM_MODEL", "openai/gpt-oss-20b"),
        base_url=os.getenv("ELVIS_LLM_BASE_URL", "http://localhost:1234/v1"),
        temperature=float(os.getenv("ELVIS_LLM_TEMPERATURE", "0.3")),
        timeout=int(os.getenv("ELVIS_LLM_TIMEOUT", "120")),
    )


# Example usage and testing
async def test_llm_enhanced_training():
    """Test LLM-enhanced training with sample data"""

    print("🧠 Testing LLM-Enhanced Training")
    print("=" * 40)

    # Create sample market data
    dates = pd.date_range(start="2024-01-01", periods=20, freq="1h")
    sample_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 65000 + np.random.randn(20) * 500,
            "high": 65500 + np.random.randn(20) * 500,
            "low": 64500 + np.random.randn(20) * 500,
            "close": 65000 + np.random.randn(20) * 500,
            "volume": 1000000 + np.random.randn(20) * 100000,
            "rsi": 50 + np.random.randn(20) * 15,
            "macd": np.random.randn(20) * 50,
            "sma_20": 65000 + np.random.randn(20) * 200,
            "sma_50": 64800 + np.random.randn(20) * 300,
        }
    )

    # Add price change
    sample_data["price_change_pct"] = sample_data["close"].pct_change().fillna(0)

    print(f"📊 Created sample dataset: {len(sample_data)} rows")

    # Create LLM config
    llm_config = create_llm_config_from_env()
    print(f"🔧 LLM Config: {llm_config.provider}/{llm_config.model}")

    # Create enhanced trainer
    trainer = LLMEnhancedTrainer(
        llm_config=llm_config, enable_llm=True, cache_llm_responses=True
    )

    print("🚀 Processing sample data with LLM enhancement...")

    # Add LLM features (test with small batch)
    enhanced_data = trainer.prepare_llm_features(sample_data.head(5), batch_size=2)

    print("✅ LLM enhancement completed!")
    print(f"📈 Original features: {len(sample_data.columns)}")
    print(f"📈 Enhanced features: {len(enhanced_data.columns)}")

    # Show LLM features
    llm_columns = [col for col in enhanced_data.columns if col.startswith("llm_")]
    print(f"🧠 LLM-derived features: {llm_columns}")

    # Display sample of enhanced data
    print("\n📋 Sample Enhanced Data:")
    print(enhanced_data[["close", "rsi"] + llm_columns].head())

    # Save cache
    trainer.save_llm_cache("llm_training_cache.json")

    print("\n🎉 LLM-Enhanced Training Test Complete!")

    return enhanced_data


if __name__ == "__main__":
    asyncio.run(test_llm_enhanced_training())
