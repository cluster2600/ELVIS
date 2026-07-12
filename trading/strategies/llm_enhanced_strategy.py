import glob
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from trading.strategies.base_strategy import BaseStrategy


class LLMEnhancedStrategy(BaseStrategy):
    """
    LLM-Enhanced Trading Strategy that integrates local LLM analysis
    with traditional technical indicators for improved prediction accuracy.

    This strategy loads pre-trained LLM-enhanced models and uses them
    for real-time trading decisions, combining:
    - Traditional technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - LLM-derived features (sentiment, confidence, risk assessment, etc.)
    - Advanced machine learning models (Random Forest with 58+ features)
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_base_path: str = "models",
        fallback_to_traditional: bool = True,
    ):
        """
        Initialize the LLM-enhanced strategy.

        Args:
            logger: Logger instance
            model_base_path: Path where trained models are stored
            fallback_to_traditional: Whether to fall back to traditional indicators if LLM models fail
        """
        super().__init__(logger)
        self.logger = logger
        self.model_base_path = model_base_path
        self.fallback_to_traditional = fallback_to_traditional

        # Model components
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.model_metadata = None
        self.llm_features_expected = 0

        # Load the latest trained models
        self._load_latest_models()

        # Strategy parameters
        self.min_confidence_threshold = 0.6  # Minimum prediction confidence for trading
        self.max_position_size = 0.1  # Maximum 10% of capital per trade
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.03  # 3% take profit

        self.logger.info(f"🧠 LLM-Enhanced Strategy initialized")
        self.logger.info(f"   Model loaded: {'✅' if self.classifier else '❌'}")
        self.logger.info(f"   LLM features: {self.llm_features_expected}")
        self.logger.info(f"   Fallback enabled: {self.fallback_to_traditional}")

    def _load_latest_models(self):
        """Load the most recently trained LLM-enhanced models"""

        try:
            # Find the latest LLM-enhanced model files
            classifier_pattern = os.path.join(
                self.model_base_path, "llm_enhanced_classifier_*.joblib"
            )
            classifier_files = glob.glob(classifier_pattern)

            if not classifier_files:
                self.logger.warning("⚠️ No LLM-enhanced classifier models found")
                return

            # Sort by modification time and get the latest
            latest_classifier = max(classifier_files, key=os.path.getmtime)
            timestamp = latest_classifier.split("_")[-1].replace(".joblib", "")

            # Load corresponding regressor, scaler, and metadata
            regressor_file = os.path.join(
                self.model_base_path, f"llm_enhanced_regressor_{timestamp}.joblib"
            )
            scaler_file = os.path.join(
                self.model_base_path, f"llm_enhanced_scaler_{timestamp}.joblib"
            )
            metadata_file = os.path.join(
                self.model_base_path, f"training_metadata_{timestamp}.json"
            )

            # Load all components
            self.classifier = joblib.load(latest_classifier)
            self.regressor = (
                joblib.load(regressor_file) if os.path.exists(regressor_file) else None
            )
            self.scaler = (
                joblib.load(scaler_file) if os.path.exists(scaler_file) else None
            )

            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    self.model_metadata = json.load(f)
                self.llm_features_expected = self.model_metadata.get(
                    "data_info", {}
                ).get("llm_features", 0)

            self.logger.info(f"✅ Loaded LLM-enhanced models from {timestamp}")
            if self.model_metadata:
                accuracy = self.model_metadata.get("model_performance", {}).get(
                    "classification_accuracy", "unknown"
                )
                mse = self.model_metadata.get("model_performance", {}).get(
                    "regression_mse", "unknown"
                )
                if isinstance(accuracy, (int, float)):
                    self.logger.info(f"   Classification accuracy: {accuracy:.3f}")
                else:
                    self.logger.info(f"   Classification accuracy: {accuracy}")
                if isinstance(mse, (int, float)):
                    self.logger.info(f"   Regression MSE: {mse:.6f}")
                else:
                    self.logger.info(f"   Regression MSE: {mse}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load LLM-enhanced models: {e}")
            self.classifier = None
            self.regressor = None
            self.scaler = None
            self.model_metadata = None

    def _prepare_features(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare features for prediction, including both traditional and LLM features.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (feature_array, feature_info)
        """

        if len(data) < 50:  # Need enough data for technical indicators
            raise ValueError("Insufficient data for feature preparation")

        # Create a working copy
        df = data.copy()

        # Add basic price features
        df["price_change_pct"] = df["close"].pct_change().fillna(0)
        df["volatility"] = df["price_change_pct"].rolling(20).std() * 100

        # Add technical indicators
        self._add_technical_indicators(df)

        # Add LLM-style features (intelligent rule-based for now)
        self._add_llm_style_features(df)

        # Select the same features as in training
        feature_columns = self._get_expected_feature_columns(df)

        # Clean and prepare features
        X_df = df[feature_columns].copy()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median())
        X_df = X_df.fillna(0)
        X_df = X_df.clip(-1e10, 1e10)

        feature_info = {
            "total_features": len(feature_columns),
            "llm_features": len(
                [col for col in feature_columns if col.startswith("llm_")]
            ),
            "latest_price": df["close"].iloc[-1],
            "price_change": df["price_change_pct"].iloc[-1],
            "volatility": df["volatility"].iloc[-1],
        }

        return X_df.values[-1:], feature_info  # Return only the latest sample

    def _add_technical_indicators(self, df: pd.DataFrame):
        """Add technical indicators to the dataframe"""

        # Moving averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        bb_middle = df["close"].rolling(bb_window).mean()
        bb_std_dev = df["close"].rolling(bb_window).std()
        df["bb_upper"] = bb_middle + (bb_std_dev * bb_std)
        df["bb_lower"] = bb_middle - (bb_std_dev * bb_std)
        df["bb_middle"] = bb_middle

        # Volume indicators
        if "volume" in df.columns:
            df["volume_sma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
        else:
            df["volume_sma"] = 1000000
            df["volume_ratio"] = 1.0

        # Fill NaN values
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)

    def _add_llm_style_features(self, df: pd.DataFrame):
        """Add LLM-style features using intelligent rule-based analysis"""

        # Get the latest data
        price_change = df["price_change_pct"].iloc[-1]
        rsi = df["rsi"].iloc[-1] if not pd.isna(df["rsi"].iloc[-1]) else 50
        volatility = (
            df["volatility"].iloc[-1] if not pd.isna(df["volatility"].iloc[-1]) else 10
        )
        macd = df["macd"].iloc[-1] if not pd.isna(df["macd"].iloc[-1]) else 0

        # Sentiment analysis
        sentiment_factors = []
        if price_change > 1:
            sentiment_factors.append(0.6 + min(0.3, price_change / 10))
        elif price_change < -1:
            sentiment_factors.append(0.4 - min(0.3, abs(price_change) / 10))
        else:
            sentiment_factors.append(0.5)

        if rsi > 70:
            sentiment_factors.append(0.8)
        elif rsi < 30:
            sentiment_factors.append(0.2)
        elif rsi > 50:
            sentiment_factors.append(0.6)
        else:
            sentiment_factors.append(0.4)

        if macd > 0:
            sentiment_factors.append(0.6)
        else:
            sentiment_factors.append(0.4)

        df["llm_sentiment_score"] = sum(sentiment_factors) / len(sentiment_factors)

        # Confidence based on signal consistency
        signal_consistency = 0.8 if abs(rsi - 50) > 15 else 0.5
        df["llm_confidence_score"] = max(0.2, signal_consistency - volatility / 50)

        # Bullish probability
        bullish_indicators = 0
        total_indicators = 0

        if price_change != 0:
            bullish_indicators += 1 if price_change > 0 else 0
            total_indicators += 1
        if rsi != 50:
            bullish_indicators += 1 if rsi > 50 else 0
            total_indicators += 1
        if macd != 0:
            bullish_indicators += 1 if macd > 0 else 0
            total_indicators += 1

        df["llm_bullish_probability"] = (
            bullish_indicators / max(1, total_indicators)
            if total_indicators > 0
            else 0.5
        )

        # Risk assessment
        risk_factors = [volatility / 30, abs(price_change) / 15]
        if rsi > 80 or rsi < 20:
            risk_factors.append(0.8)
        df["llm_risk_score"] = min(1.0, sum(risk_factors) / len(risk_factors))

        # Volatility prediction
        df["llm_volatility_prediction"] = min(
            1.0, (volatility / 20) + abs(price_change) / 20
        )

        # Trend strength
        df["llm_trend_strength"] = min(
            1.0, (abs(price_change) / 5 + abs(macd) * 10 + abs(rsi - 50) / 25) / 3
        )

        # Ensure values are in [0.1, 0.9] range
        llm_columns = [
            "llm_sentiment_score",
            "llm_confidence_score",
            "llm_bullish_probability",
            "llm_risk_score",
            "llm_volatility_prediction",
            "llm_trend_strength",
        ]

        for col in llm_columns:
            df[col] = df[col].clip(0.1, 0.9)

    def _get_expected_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get the feature columns expected by the trained model"""

        # These should match the columns used during training
        exclude_columns = ["timestamp", "future_price", "price_change_future", "side"]

        feature_columns = []
        for col in df.columns:
            if (
                col not in exclude_columns
                and not col.startswith("target_")
                and not col.startswith("future_")
            ):
                try:
                    pd.to_numeric(df[col], errors="raise")
                    feature_columns.append(col)
                except (ValueError, TypeError):
                    continue

        return feature_columns

    def should_buy(self, data: pd.DataFrame, current_price: float, **kwargs) -> bool:
        """Determine if we should buy based on LLM-enhanced model prediction"""

        if not self.classifier or not self.scaler:
            if self.fallback_to_traditional:
                return self._traditional_buy_signal(data, current_price)
            return False

        try:
            # Prepare features
            X, feature_info = self._prepare_features(data)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get predictions
            buy_probability = self.classifier.predict_proba(X_scaled)[0]
            buy_confidence = max(buy_probability)
            will_buy = (
                buy_probability[1] > buy_probability[0]
            )  # Index 1 is typically the "positive" class

            # Additional confidence check using regressor if available
            if self.regressor:
                price_change_prediction = self.regressor.predict(X_scaled)[0]
                # Increase confidence if regressor agrees with classifier
                if (will_buy and price_change_prediction > 0) or (
                    not will_buy and price_change_prediction < 0
                ):
                    buy_confidence *= 1.1

            # Log prediction details
            self.logger.info(
                f"🧠 LLM Prediction: Buy={will_buy}, Confidence={buy_confidence:.3f}"
            )
            self.logger.info(
                f"   Features: {feature_info['total_features']} total, {feature_info['llm_features']} LLM"
            )
            self.logger.info(
                f"   Market: Price=${feature_info['latest_price']:.2f}, Change={feature_info['price_change']:.2f}%, Vol={feature_info['volatility']:.2f}%"
            )

            # Make decision based on confidence threshold
            should_buy = will_buy and buy_confidence >= self.min_confidence_threshold

            if should_buy:
                self.logger.info(f"✅ BUY signal with {buy_confidence:.1%} confidence")
            else:
                self.logger.info(
                    f"❌ No buy signal (confidence {buy_confidence:.1%} < {self.min_confidence_threshold:.1%})"
                )

            return should_buy

        except Exception as e:
            self.logger.error(f"❌ LLM prediction failed: {e}")
            if self.fallback_to_traditional:
                self.logger.info("🔄 Falling back to traditional analysis")
                return self._traditional_buy_signal(data, current_price)
            return False

    def should_sell(self, data: pd.DataFrame, current_price: float, **kwargs) -> bool:
        """Determine if we should sell based on LLM-enhanced model prediction"""

        if not self.classifier or not self.scaler:
            if self.fallback_to_traditional:
                return self._traditional_sell_signal(data, current_price)
            return False

        try:
            # Prepare features
            X, feature_info = self._prepare_features(data)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get predictions
            sell_probability = self.classifier.predict_proba(X_scaled)[0]
            sell_confidence = max(sell_probability)
            will_sell = (
                sell_probability[0] > sell_probability[1]
            )  # Index 0 is typically the "negative" class

            # Additional confidence check using regressor if available
            if self.regressor:
                price_change_prediction = self.regressor.predict(X_scaled)[0]
                # Increase confidence if regressor agrees with classifier
                if (will_sell and price_change_prediction < 0) or (
                    not will_sell and price_change_prediction > 0
                ):
                    sell_confidence *= 1.1

            # Log prediction details
            self.logger.info(
                f"🧠 LLM Prediction: Sell={will_sell}, Confidence={sell_confidence:.3f}"
            )

            # Make decision based on confidence threshold
            should_sell = will_sell and sell_confidence >= self.min_confidence_threshold

            if should_sell:
                self.logger.info(
                    f"✅ SELL signal with {sell_confidence:.1%} confidence"
                )
            else:
                self.logger.info(
                    f"❌ No sell signal (confidence {sell_confidence:.1%} < {self.min_confidence_threshold:.1%})"
                )

            return should_sell

        except Exception as e:
            self.logger.error(f"❌ LLM prediction failed: {e}")
            if self.fallback_to_traditional:
                self.logger.info("🔄 Falling back to traditional analysis")
                return self._traditional_sell_signal(data, current_price)
            return False

    def _traditional_buy_signal(self, data: pd.DataFrame, current_price: float) -> bool:
        """Traditional technical analysis buy signal as fallback"""

        if len(data) < 20:
            return False

        close = data["close"].iloc[-1]
        rsi = (
            data["close"]
            .diff()
            .rolling(14)
            .apply(lambda x: 100 - (100 / (1 + x[x > 0].mean() / abs(x[x < 0]).mean())))
        )
        sma_20 = data["close"].rolling(20).mean()

        # Simple conditions: oversold RSI and price above SMA
        is_oversold = rsi.iloc[-1] < 35 if not pd.isna(rsi.iloc[-1]) else False
        above_sma = close > sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else False

        return is_oversold and above_sma

    def _traditional_sell_signal(
        self, data: pd.DataFrame, current_price: float
    ) -> bool:
        """Traditional technical analysis sell signal as fallback"""

        if len(data) < 20:
            return False

        close = data["close"].iloc[-1]
        rsi = (
            data["close"]
            .diff()
            .rolling(14)
            .apply(lambda x: 100 - (100 / (1 + x[x > 0].mean() / abs(x[x < 0]).mean())))
        )
        sma_20 = data["close"].rolling(20).mean()

        # Simple conditions: overbought RSI and price below SMA
        is_overbought = rsi.iloc[-1] > 70 if not pd.isna(rsi.iloc[-1]) else False
        below_sma = close < sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else False

        return is_overbought and below_sma

    def get_position_size(
        self, available_capital: float, current_price: float, **kwargs
    ) -> float:
        """Calculate position size based on model confidence and risk management"""

        # Base position size
        base_size = available_capital * self.max_position_size

        # Adjust based on model confidence if available
        if self.classifier and hasattr(self, "_last_confidence"):
            # Scale position size by confidence (higher confidence = larger position)
            confidence_multiplier = min(
                2.0, self._last_confidence / self.min_confidence_threshold
            )
            base_size *= confidence_multiplier

        # Ensure minimum and maximum limits
        min_trade_value = 10.0  # Minimum $10 trade
        max_trade_value = available_capital * 0.2  # Maximum 20% of capital

        position_value = max(min_trade_value, min(base_size, max_trade_value))
        position_size = position_value / current_price

        self.logger.info(
            f"💰 Position size: ${position_value:.2f} ({position_size:.6f} BTC)"
        )

        return position_size

    def get_stop_loss_price(self, entry_price: float, is_long: bool) -> Optional[float]:
        """Calculate stop loss price"""
        if is_long:
            return entry_price * (1 - self.stop_loss_percentage)
        else:
            return entry_price * (1 + self.stop_loss_percentage)

    def get_take_profit_price(
        self, entry_price: float, is_long: bool
    ) -> Optional[float]:
        """Calculate take profit price"""
        if is_long:
            return entry_price * (1 + self.take_profit_percentage)
        else:
            return entry_price * (1 - self.take_profit_percentage)

    def train_model(self, data: pd.DataFrame):
        """
        Retrain the model with new data (placeholder for future implementation)
        For now, this just logs that retraining is not yet implemented.
        """
        self.logger.info("🔄 Model retraining requested but not yet implemented")
        self.logger.info("   Current models will continue to be used")
        self.logger.info(
            "   Consider running scripts/train_with_llm.py manually for retraining"
        )

    def generate_signals(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Generate buy/sell signals based on the data.

        Args:
            data: The data to generate signals from

        Returns:
            Tuple[bool, bool]: A tuple of (buy_signal, sell_signal)
        """
        if data.empty:
            return False, False

        current_price = data["close"].iloc[-1]

        buy_signal = self.should_buy(data, current_price)
        sell_signal = self.should_sell(data, current_price)

        return buy_signal, sell_signal

    def calculate_position_size(
        self, data: pd.DataFrame, current_price: float, available_capital: float
    ) -> float:
        """
        Calculate the position size based on the data and available capital.

        Args:
            data: The data to calculate position size from
            current_price: The current price
            available_capital: The available capital

        Returns:
            float: The position size
        """
        return self.get_position_size(available_capital, current_price)

    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the stop loss price based on the data and entry price.

        Args:
            data: The data to calculate stop loss from
            entry_price: The entry price

        Returns:
            float: The stop loss price
        """
        return self.get_stop_loss_price(entry_price, is_long=True) or entry_price * 0.98

    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the take profit price based on the data and entry price.

        Args:
            data: The data to calculate take profit from
            entry_price: The entry price

        Returns:
            float: The take profit price
        """
        return (
            self.get_take_profit_price(entry_price, is_long=True) or entry_price * 1.03
        )

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current strategy configuration"""

        info = {
            "strategy_name": "LLM-Enhanced Trading Strategy",
            "model_loaded": self.classifier is not None,
            "llm_features": self.llm_features_expected,
            "confidence_threshold": self.min_confidence_threshold,
            "max_position_size": self.max_position_size,
            "stop_loss_pct": self.stop_loss_percentage * 100,
            "take_profit_pct": self.take_profit_percentage * 100,
            "fallback_enabled": self.fallback_to_traditional,
        }

        if self.model_metadata:
            info.update(
                {
                    "model_accuracy": self.model_metadata.get(
                        "model_performance", {}
                    ).get("classification_accuracy"),
                    "model_mse": self.model_metadata.get("model_performance", {}).get(
                        "regression_mse"
                    ),
                    "training_samples": self.model_metadata.get("data_info", {}).get(
                        "total_samples"
                    ),
                    "total_features": self.model_metadata.get("data_info", {}).get(
                        "total_features"
                    ),
                    "training_timestamp": self.model_metadata.get("timestamp"),
                }
            )

        return info
