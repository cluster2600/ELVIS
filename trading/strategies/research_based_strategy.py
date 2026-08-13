import logging
import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from trading.models.artifact_manifest import (
    ArtifactDescriptor,
    validate_classifier_classes,
    validate_feature_manifest,
    write_feature_manifest,
)
from trading.models.feature_schema import FeatureContractError
from trading.models.feature_schemas import (
    RESEARCH_FINANCIAL_9_V1,
    RESEARCH_SOCIAL_11_V1,
)
from trading.strategies.base_strategy import BaseStrategy

RESEARCH_ARTIFACT_DESCRIPTOR = ArtifactDescriptor(
    model_kind="random-forest-classifier",
    library="scikit-learn",
    library_version=sklearn.__version__,
)


def _validate_research_components(model: object, scaler: object) -> None:
    if not isinstance(model, RandomForestClassifier):
        raise FeatureContractError("research model must be a RandomForestClassifier")
    if not isinstance(scaler, StandardScaler):
        raise FeatureContractError("research scaler must be a StandardScaler")


# Optional social-data dependencies are not part of the canonical ELVIS
# dependency sets, so their imports are guarded: the collectors below fall back
# to neutral constants (with a debug log) whenever a library or its API
# credentials are absent. See docs/research_strategy_guide.md.
try:  # pragma: no cover - exercised only when tweepy is installed
    import tweepy  # type: ignore
except ImportError:  # pragma: no cover
    tweepy = None

try:  # pragma: no cover - exercised only when pytrends is installed
    from pytrends.request import TrendReq  # type: ignore
except ImportError:  # pragma: no cover
    TrendReq = None


class ResearchBasedStrategy(BaseStrategy):
    """
    Research-Based Strategy implementing the exact methodology from:
    "High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features"
    by Annelotte Bonenkamp (2021)

    Key Features:
    - Pure Random Forest classifier (600 trees, 10-fold CV)
    - Exactly 9 financial indicators as specified in research
    - 2 social features (Twitter sentiment + Google Trends)
    - 5-minute trading frequency
    - Binary classification (BUY/SELL only, no HOLD)
    - Rolling 1-week training windows
    - Paper-only evaluation; historical paper metrics are not a target
    """

    def __init__(
        self,
        logger: logging.Logger,
        social_data_enabled: bool = True,
        enable_rolling_training: bool = True,
        model_save_path: str = "models/research_based",
    ):
        """
        Initialize the research-based strategy.

        Args:
            logger: Logger instance
            social_data_enabled: Whether to use social features (Twitter + Google Trends)
            enable_rolling_training: Whether to use rolling 1-week training windows
            model_save_path: Path to save/load trained models
        """
        super().__init__(logger)
        self.logger = logger
        self.social_data_enabled = social_data_enabled
        self.enable_rolling_training = enable_rolling_training
        self.model_save_path = model_save_path
        self.feature_schema = (
            RESEARCH_SOCIAL_11_V1 if social_data_enabled else RESEARCH_FINANCIAL_9_V1
        )

        # Research-specific parameters (exact from paper)
        self.n_estimators = 600  # Exactly as specified in research
        self.cv_folds = 10  # 10-fold cross-validation
        self.trading_frequency_minutes = 5  # 5-minute intervals
        self.training_window_days = 7  # 1-week training windows

        # Model components
        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
        )
        self.feature_scaler = StandardScaler()

        # Training state
        self.is_trained = False
        self.last_retrain_time = None
        self.training_data_buffer = []
        self.feature_names = list(self.feature_schema.names)

        # Social data collectors
        self.twitter_collector = (
            TwitterSentimentCollector(logger) if social_data_enabled else None
        )
        self.google_trends_collector = (
            GoogleTrendsCollector(logger) if social_data_enabled else None
        )

        # Create model directory
        os.makedirs(model_save_path, exist_ok=True)

        self.symbols = ["BTCUSDT"]

        # Attempt to load a pre-trained model on initialization
        self.load_model()

        self.logger.info("🔬 Research-Based Strategy initialized")
        self.logger.info(
            f"📊 Settings: Social={social_data_enabled}, Rolling={enable_rolling_training}"
        )
        self.logger.info("🎯 Paper evaluation only; no return target is promised")

    def calculate_financial_indicators(self, data: pd.DataFrame) -> dict:
        """
        Calculate exactly the 9 financial indicators from the research paper.

        Research indicators (Appendix A):
        1. RSI - Relative Strength Index
        2. STOCH - Stochastic Oscillator
        3. ROC - Rate of Change
        4. EMA - Exponential Moving Average
        5. MACD - Moving Average Convergence-Divergence
        6. CCI - Commodity Channel Index
        7. OBV - On Balance Volume
        8. ATR - Average True Range
        9. WILLR - Williams %R
        """
        try:
            if len(data) < 50:
                self.logger.warning("Insufficient data for indicators, using defaults")
                return self._get_default_indicators()

            # Ensure required columns exist
            required_cols = ["high", "low", "close", "volume"]
            for col in required_cols:
                if col not in data.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return self._get_default_indicators()

            latest = data.iloc[-1]
            indicators = {}

            # 1. RSI - Relative Strength Index
            # RSI = 100 - (100 / (1 + RS)) where RS = Ave(Gains) / Ave(Losses)
            close_prices = data["close"].astype(float)
            price_diff = close_prices.diff()
            gains = price_diff.where(price_diff > 0, 0.0)
            losses = -price_diff.where(price_diff < 0, 0.0)
            avg_gains = gains.rolling(window=14, min_periods=1).mean()
            avg_losses = losses.rolling(window=14, min_periods=1).mean()
            rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            indicators["RSI"] = float(rsi.iloc[-1])

            # 2. STOCH - Stochastic Oscillator
            # STOCH = 100 * (Ck - Lp) / (Hp - Lp)
            high_14 = data["high"].astype(float).rolling(window=14, min_periods=1).max()
            low_14 = data["low"].astype(float).rolling(window=14, min_periods=1).min()
            stoch = 100 * (close_prices - low_14) / (high_14 - low_14 + 1e-10)
            indicators["STOCH"] = float(stoch.iloc[-1])

            # 3. ROC - Rate of Change
            # ROC(p) = 100 * (Ck - Ck-p) / Ck-p
            period = min(14, len(data) - 1)
            if period > 0:
                current_price = close_prices.iloc[-1]
                past_price = (
                    close_prices.iloc[-period - 1]
                    if len(close_prices) > period
                    else close_prices.iloc[0]
                )
                roc = 100 * (current_price - past_price) / (past_price + 1e-10)
                indicators["ROC"] = float(roc)
            else:
                indicators["ROC"] = 0.0

            # 4. EMA - Exponential Moving Average
            # EMAk(p) = EMAk-1(p) + (2/(p+1)) * (Ck - EMAk-1(p))
            ema_period = 12
            alpha = 2 / (ema_period + 1)
            ema = close_prices.ewm(alpha=alpha, adjust=False).mean()
            indicators["EMA"] = float(ema.iloc[-1])

            # 5. MACD - Moving Average Convergence-Divergence
            # MACD = EMA12 - EMA26, Signal = EMA9(MACD)
            ema_12 = close_prices.ewm(span=12, adjust=False).mean()
            ema_26 = close_prices.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators["MACD"] = float(macd_line.iloc[-1])
            indicators["MACD_SIGNAL"] = float(signal_line.iloc[-1])

            # 6. CCI - Commodity Channel Index
            # CCI(n) = (1/0.015) * (TPk - SMAn(TPk)) / σn(TPk)
            # where TPk = (Hk + Lk + Ck) / 3
            typical_price = (
                data["high"].astype(float) + data["low"].astype(float) + close_prices
            ) / 3
            sma_tp = typical_price.rolling(window=14, min_periods=1).mean()
            mad = typical_price.rolling(window=14, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            )
            cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
            indicators["CCI"] = float(cci.iloc[-1])

            # 7. OBV - On Balance Volume
            # OBV calculation based on price direction and volume
            volume = data["volume"].astype(float)
            obv = np.zeros(len(data))
            obv[0] = volume.iloc[0]
            for i in range(1, len(data)):
                if close_prices.iloc[i] > close_prices.iloc[i - 1]:
                    obv[i] = obv[i - 1] + volume.iloc[i]
                elif close_prices.iloc[i] < close_prices.iloc[i - 1]:
                    obv[i] = obv[i - 1] - volume.iloc[i]
                else:
                    obv[i] = obv[i - 1]
            indicators["OBV"] = float(obv[-1])

            # 8. ATR - Average True Range
            # ATR = max(Hk-Lk, |Hk-Ck-1|, |Lk-Ck-1|)
            high = data["high"].astype(float)
            low = data["low"].astype(float)
            prev_close = close_prices.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14, min_periods=1).mean()
            indicators["ATR"] = float(atr.iloc[-1])

            # 9. WILLR - Williams %R
            # WILLR = -100 * (Hp - Ck) / (Hp - Lp)
            willr = -100 * (high_14 - close_prices) / (high_14 - low_14 + 1e-10)
            indicators["WILLR"] = float(willr.iloc[-1])

            self.logger.debug(
                f"📊 Calculated 9 research indicators: {list(indicators.keys())}"
            )
            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating financial indicators: {e}")
            return self._get_default_indicators()

    def _get_default_indicators(self) -> dict:
        """Return default indicator values when calculation fails."""
        return {
            "RSI": 50.0,
            "STOCH": 50.0,
            "ROC": 0.0,
            "EMA": 50000.0,  # Reasonable default for Bitcoin price
            "MACD": 0.0,
            "MACD_SIGNAL": 0.0,
            "CCI": 0.0,
            "OBV": 1000.0,
            "ATR": 1000.0,  # Reasonable default for Bitcoin volatility
            "WILLR": -50.0,
        }

    def collect_social_features(self) -> dict:
        """
        Collect social features as specified in research:
        1. Twitter sentiment analysis (lagged 'Price' sentiment)
        2. Google Trends data for 'Bitcoin'
        """
        social_features = {}

        if not self.social_data_enabled:
            # Return neutral values when social data is disabled
            social_features["TWITTER_PRICE_SENTIMENT"] = 0.0
            social_features["GOOGLE_TRENDS_BITCOIN"] = 50.0
            return social_features

        try:
            # 1. Twitter 'Price' sentiment (lagged by 5 minutes as per research)
            if self.twitter_collector:
                price_sentiment = self.twitter_collector.get_lagged_price_sentiment()
                social_features["TWITTER_PRICE_SENTIMENT"] = price_sentiment
            else:
                social_features["TWITTER_PRICE_SENTIMENT"] = 0.0

            # 2. Google Trends for 'Bitcoin' (interpolated to 5-minute frequency)
            if self.google_trends_collector:
                bitcoin_trends = (
                    self.google_trends_collector.get_bitcoin_trends_interpolated()
                )
                social_features["GOOGLE_TRENDS_BITCOIN"] = bitcoin_trends
            else:
                social_features["GOOGLE_TRENDS_BITCOIN"] = 50.0

            self.logger.debug(f"📱 Collected social features: {social_features}")
            return social_features

        except Exception as e:
            self.logger.warning(f"Error collecting social features: {e}")
            # Return neutral values on error
            return {"TWITTER_PRICE_SENTIMENT": 0.0, "GOOGLE_TRENDS_BITCOIN": 50.0}

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature vector exactly as specified in research:
        - 9 financial indicators
        - 2 social features (if enabled)
        - Feature standardization: (x - μ) / σ
        """
        try:
            # Calculate financial indicators
            financial_indicators = self.calculate_financial_indicators(data)

            # Collect social features
            social_features = self.collect_social_features()

            feature_values = dict(financial_indicators)
            if self.social_data_enabled:
                feature_values.update(social_features)

            features = np.asarray(
                [self.feature_schema.vectorize(feature_values)], dtype=np.float64
            )

            # A trained strategy must never bypass preprocessing.  An unfitted
            # strategy may still expose the raw vector to diagnostics/tests.
            if self.is_trained or hasattr(self.feature_scaler, "mean_"):
                self.feature_schema.validate_fitted_component(
                    self.feature_scaler, "research scaler"
                )
                features = self.feature_scaler.transform(features)

            self.logger.debug(
                f"🔢 Prepared {features.shape[1]} features for prediction"
            )
            return features

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise FeatureContractError(
                f"cannot produce schema {self.feature_schema.identity}"
            ) from e

    def train_model(self, training_data: pd.DataFrame) -> float:
        """
        Train the Random Forest model using research methodology:
        - 600 trees (n_estimators=600)
        - 10-fold cross-validation
        - Binary classification (BUY=1, SELL=0)
        """
        try:
            if len(training_data) < 100:
                self.logger.warning(
                    f"Insufficient training data: {len(training_data)} samples"
                )
                return 0.0

            self.logger.info(
                f"🎓 Training Random Forest model with {len(training_data)} samples"
            )

            # Prepare features for all training samples
            X = []
            y = []

            for i in range(len(training_data) - 1):
                # Use data up to point i for feature calculation
                data_slice = training_data.iloc[: i + 1]
                if len(data_slice) < 20:  # Need minimum data for indicators
                    continue

                # Calculate features
                financial_indicators = self.calculate_financial_indicators(data_slice)
                social_features = self.collect_social_features()

                feature_values = dict(financial_indicators)
                if self.social_data_enabled:
                    feature_values.update(social_features)

                X.append(self.feature_schema.vectorize(feature_values))

                # Create binary label: 1 if next price is higher (BUY), 0 if lower (SELL)
                current_price = training_data.iloc[i]["close"]
                next_price = training_data.iloc[i + 1]["close"]
                label = 1 if next_price > current_price else 0
                y.append(label)

            if len(X) < 50:
                self.logger.warning("Insufficient valid training samples")
                return 0.0

            X = np.array(X)
            y = np.array(y)

            self.logger.info(
                f"📊 Training data prepared: {X.shape[0]} samples, {X.shape[1]} features"
            )
            self.logger.info(
                f"📈 Label distribution: BUY={np.sum(y)}, SELL={len(y)-np.sum(y)}"
            )

            candidate_scaler = clone(self.feature_scaler)
            candidate_model = clone(self.rf_model)
            _validate_research_components(candidate_model, candidate_scaler)

            candidate_scaler.fit(X)
            X_scaled = candidate_scaler.transform(X)

            # 10-fold cross-validation as specified in research.
            # Financial data is ordered in time, so a time-series-aware split is
            # used (TimeSeriesSplit, no shuffle) instead of the default
            # (Stratified)KFold that cross_val_score picks for an int cv value.
            cv_splitter = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = cross_val_score(
                candidate_model,
                X_scaled,
                y,
                cv=cv_splitter,
                scoring="f1",
                n_jobs=-1,
            )

            candidate_model.fit(X_scaled, y)
            self.feature_schema.validate_fitted_component(
                candidate_model, "research model"
            )
            self.feature_schema.validate_fitted_component(
                candidate_scaler, "research scaler"
            )
            validate_classifier_classes(candidate_model, (0, 1), "research model")

            mean_cv_score = cv_scores.mean()
            if not np.isfinite(mean_cv_score):
                raise FeatureContractError("research cross-validation score is invalid")
            self.logger.info(
                f"🎯 Model training complete: F1-score={mean_cv_score:.3f} (target: 0.576)"
            )
            self.logger.info(f"📝 CV scores: {cv_scores}")

            self.save_model(candidate_model, candidate_scaler)

            self.rf_model = candidate_model
            self.feature_scaler = candidate_scaler
            self.is_trained = True
            self.last_retrain_time = datetime.now()

            return mean_cv_score

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return 0.0

    def generate_signals(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for all symbols based on the research model.
        This method aligns with the main trading loop's expectation of a dictionary-based signal.
        """
        signals = {}
        for symbol in self.symbols:
            if symbol not in data or data[symbol].empty:
                # Binary strategy (no HOLD): with no market data there is no
                # signal to act on, so default to the conservative SELL side.
                signals[symbol] = {"signal": "SELL", "confidence": 0.0}
                continue

            df = data[symbol]

            try:
                self.logger.info(f"🔬 Generating research-based signal for {symbol}")

                # ATH DETECTION - Override all signals at extreme prices
                current_price = df["close"].iloc[-1] if not df.empty else 50000

                # UPDATED ATH THRESHOLDS FOR NEW PARADIGM
                if current_price > 125000:  # High ATH levels - FORCE SELL
                    if current_price > 130000:  # Extreme ATH
                        signal, confidence = "SELL", 0.95
                        self.logger.warning(
                            f"🚨 EXTREME ATH OVERRIDE: BTC at ${current_price:,.0f} - FORCING SELL"
                        )
                        signals[symbol] = {"signal": signal, "confidence": confidence}
                        continue
                    else:
                        signal, confidence = "SELL", 0.85
                        self.logger.warning(
                            f"🔴 HIGH ATH OVERRIDE: BTC at ${current_price:,.0f} - FORCING SELL"
                        )
                        signals[symbol] = {"signal": signal, "confidence": confidence}
                        continue

                if not self.is_trained:
                    self.logger.warning(
                        "Model not trained, using default signal logic."
                    )
                    # Fallback to a simple RSI-based logic if model is not ready.
                    # Binary strategy: always emit BUY or SELL, never HOLD.
                    indicators = self.calculate_financial_indicators(df)
                    rsi = indicators.get("RSI", 50)
                    if rsi < 35:
                        signal, confidence = "BUY", 0.55
                    elif rsi > 65:
                        signal, confidence = "SELL", 0.55
                    else:
                        # Neutral RSI: pick a side instead of holding.
                        signal, confidence = ("SELL", 0.5) if rsi < 50 else ("BUY", 0.5)
                else:
                    try:
                        features = self.prepare_features(df)
                        # Ensure features have the shape used at training time.
                        # This respects the social flag: 11 features when social
                        # data is enabled (9 financial + 2 social), else 9.
                        if features.shape != (1, self.feature_schema.size):
                            raise FeatureContractError(
                                "research inference vector does not match "
                                f"{self.feature_schema.identity}"
                            )

                        probabilities = self.rf_model.predict_proba(features)[0]
                        sell_prob, buy_prob = probabilities[0], probabilities[1]

                        if buy_prob > 0.5:
                            signal = "BUY"
                            confidence = buy_prob
                        else:
                            signal = "SELL"
                            confidence = sell_prob
                    except Exception as pred_error:
                        self.logger.error(f"Model prediction failed: {pred_error}")
                        self.logger.info("Falling back to RSI-based logic")
                        # Fallback to RSI-based logic if model prediction fails.
                        # Binary strategy: always emit BUY or SELL, never HOLD.
                        indicators = self.calculate_financial_indicators(df)
                        rsi = indicators.get("RSI", 50)
                        if rsi < 35:
                            signal, confidence = "BUY", 0.55
                        elif rsi > 65:
                            signal, confidence = "SELL", 0.55
                        else:
                            # Neutral RSI: pick a side instead of holding.
                            signal, confidence = (
                                ("SELL", 0.5) if rsi < 50 else ("BUY", 0.5)
                            )

                self.logger.info(
                    f"🎯 Research signal for {symbol}: {signal} with {confidence:.3f} confidence"
                )
                signals[symbol] = {"signal": signal, "confidence": confidence}

            except Exception as e:
                self.logger.error(f"Error generating research signal for {symbol}: {e}")
                # Binary strategy (no HOLD): default to the conservative SELL
                # side when signal generation fails entirely.
                signals[symbol] = {"signal": "SELL", "confidence": 0.0}

        return signals

    def _market_data_to_dataframe(self, market_data: dict) -> pd.DataFrame:
        """Convert market data dict to DataFrame format for indicator calculation."""
        try:
            # Create a simple DataFrame with the required columns
            data = pd.DataFrame(
                {
                    "high": [market_data.get("high", market_data.get("price", 50000))],
                    "low": [market_data.get("low", market_data.get("price", 50000))],
                    "close": [
                        market_data.get("close", market_data.get("price", 50000))
                    ],
                    "volume": [market_data.get("volume", 1000)],
                }
            )

            # Create historical data simulation for indicators (simplified)
            # In a real implementation, you'd fetch actual historical data
            current_price = float(data["close"].iloc[0])

            # Generate 50 data points with some realistic variation
            np.random.seed(int(time.time()) % 1000)
            price_history = []
            base_price = current_price * 0.98  # Start slightly lower

            for i in range(50):
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
                base_price *= 1 + change
                price_history.append(base_price)

            # Set the last price to current price
            price_history[-1] = current_price

            # Create full historical DataFrame
            historical_data = pd.DataFrame(
                {
                    "high": [
                        p * (1 + abs(np.random.normal(0, 0.01))) for p in price_history
                    ],
                    "low": [
                        p * (1 - abs(np.random.normal(0, 0.01))) for p in price_history
                    ],
                    "close": price_history,
                    "volume": [
                        market_data.get("volume", 1000) * (1 + np.random.normal(0, 0.3))
                        for _ in price_history
                    ],
                }
            )

            return historical_data

        except Exception as e:
            self.logger.error(f"Error converting market data: {e}")
            return pd.DataFrame()

    def should_retrain(self) -> bool:
        """
        Check if model should be retrained based on research methodology:
        - Rolling 1-week training windows
        - Retrain daily
        """
        if not self.enable_rolling_training:
            return False

        if self.last_retrain_time is None:
            return True

        # Retrain every 24 hours (daily updates for 1-week window)
        time_since_retrain = datetime.now() - self.last_retrain_time
        return time_since_retrain.total_seconds() > (24 * 3600)

    def save_model(self, model=None, scaler=None):
        """Save the trained model and scaler."""
        target_model = self.rf_model if model is None else model
        target_scaler = self.feature_scaler if scaler is None else scaler
        _validate_research_components(target_model, target_scaler)
        self.feature_schema.validate_fitted_component(target_model, "research model")
        self.feature_schema.validate_fitted_component(target_scaler, "research scaler")
        validate_classifier_classes(target_model, (0, 1), "research model")

        model_path = os.path.join(self.model_save_path, "research_rf_model.pkl")
        scaler_path = os.path.join(self.model_save_path, "research_scaler.pkl")
        manifest_path = os.path.join(self.model_save_path, "feature_manifest.json")
        joblib.dump(target_model, model_path)
        joblib.dump(target_scaler, scaler_path)
        write_feature_manifest(
            manifest_path,
            self.feature_schema,
            RESEARCH_ARTIFACT_DESCRIPTOR,
            {"model": model_path, "scaler": scaler_path},
        )

        self.logger.info(f"💾 Model saved to {model_path}")

    def load_model(self) -> bool:
        """Load the trained model and scaler."""
        try:
            model_path = os.path.join(self.model_save_path, "research_rf_model.pkl")
            scaler_path = os.path.join(self.model_save_path, "research_scaler.pkl")
            manifest_path = os.path.join(self.model_save_path, "feature_manifest.json")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                components = {"model": model_path, "scaler": scaler_path}
                validate_feature_manifest(
                    manifest_path,
                    self.feature_schema,
                    RESEARCH_ARTIFACT_DESCRIPTOR,
                    components,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    loaded_model = joblib.load(model_path)
                    loaded_scaler = joblib.load(scaler_path)
                _validate_research_components(loaded_model, loaded_scaler)
                self.feature_schema.validate_fitted_component(
                    loaded_model, "research model"
                )
                self.feature_schema.validate_fitted_component(
                    loaded_scaler, "research scaler"
                )
                validate_classifier_classes(loaded_model, (0, 1), "research model")
                self.rf_model = loaded_model
                self.feature_scaler = loaded_scaler
                self.is_trained = True
                self.logger.info(f"📚 Model loaded from {model_path}")
                return True
            else:
                self.logger.info("No saved model found, will train from scratch")
                return False

        except FeatureContractError as e:
            self.logger.warning(f"Incompatible research model ignored: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        current_price: float,
        available_capital: float,
        **kwargs,
    ) -> float:
        """
        Calculate a realistic and safe position size based on available capital and risk.
        This method now considers the actual available balance to prevent "insufficient balance" errors.
        """
        if current_price <= 0 or available_capital <= 0:
            return 0.0

        # Get signal direction from kwargs, default to 'buy' if not provided
        side = kwargs.get("side", "buy")

        # Get available balances from kwargs, passed from the executor
        balance_usdt = kwargs.get("balance_usdt", available_capital)
        balance_btc = kwargs.get("balance_btc", 0.0)

        # Define risk parameters
        risk_per_trade = (
            0.02  # Risk 2% of capital per trade for more meaningful positions
        )
        leverage = kwargs.get("leverage", 100.0)  # Use a default leverage of 100x

        if side.lower() == "sell":
            # For a SELL order, the position size cannot exceed the available BTC balance
            if balance_btc > 0:
                # Sell a fraction of the available BTC, e.g., 50%
                position_size_btc = balance_btc * 0.5
                self.logger.info(
                    f"SELL order: Sizing based on available BTC. Selling 50% of {balance_btc:.6f} BTC."
                )
            else:
                self.logger.warning("SELL signal received, but no BTC balance to sell.")
                return 0.0
        else:  # BUY
            # For a BUY order, calculate size based on USDT balance and risk
            position_value_usd = balance_usdt * risk_per_trade * leverage
            position_size_btc = position_value_usd / current_price

        # Ensure the calculated size is within reasonable limits
        min_trade_size = 0.0001  # A more realistic minimum trade size

        if position_size_btc < min_trade_size:
            self.logger.warning(
                f"Calculated position size {position_size_btc:.6f} is below minimum {min_trade_size}. Adjusting up."
            )
            position_size_btc = min_trade_size

        # Final check to ensure we can afford the trade
        if side.lower() == "buy":
            required_margin = (position_size_btc * current_price) / leverage
            if required_margin > balance_usdt:
                self.logger.warning(
                    f"Cannot afford BUY trade. Required margin ${required_margin:.2f} > Available USDT ${balance_usdt:.2f}. Reducing size."
                )
                # Reduce size to what is affordable
                position_size_btc = (
                    balance_usdt * leverage * 0.95
                ) / current_price  # Use 95% of available capital

        self.logger.info(f"Final calculated position size: {position_size_btc:.6f} BTC")
        return position_size_btc

    def calculate_stop_loss(
        self, data: pd.DataFrame, entry_price: float, **kwargs
    ) -> float:
        """
        Calculate stop loss using ATR.
        The research paper does not specify a method, so a standard one is used.
        """
        if data.empty:
            return entry_price * 0.98  # Default 2% stop loss

        indicators = self.calculate_financial_indicators(data)
        atr = indicators.get("ATR", entry_price * 0.02)
        stop_loss = entry_price - (2 * atr)

        self.logger.info(f"Calculated stop loss: ${stop_loss:.2f} (ATR: {atr:.2f})")
        return stop_loss

    def calculate_take_profit(
        self, data: pd.DataFrame, entry_price: float, **kwargs
    ) -> float:
        """
        Calculate take profit using ATR.
        The research paper does not specify a method, so a standard one is used.
        """
        if data.empty:
            return entry_price * 1.04  # Default 4% take profit

        indicators = self.calculate_financial_indicators(data)
        atr = indicators.get("ATR", entry_price * 0.02)
        take_profit = entry_price + (4 * atr)  # Risk:Reward of 1:2

        self.logger.info(f"Calculated take profit: ${take_profit:.2f} (ATR: {atr:.2f})")
        return take_profit


class TwitterSentimentCollector:
    """Collect Twitter sentiment data as specified in research."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        # In a real implementation, you'd use Twitter API v2
        # For now, return simulated data

    def get_lagged_price_sentiment(self) -> float:
        """
        Get lagged 'Price' sentiment from Twitter data.
        Research specifies 5-minute lag.

        Real data requires the optional ``tweepy`` dependency together with a
        Twitter/X API bearer token (``TWITTER_BEARER_TOKEN``). When either is
        absent, a neutral constant is returned and the reason is logged at
        debug level.

        Returns:
            float: Sentiment score (normalized to [-1, 1])
        """
        neutral = 0.0
        try:
            bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
            if tweepy is None:
                self.logger.debug(
                    "tweepy not installed; returning neutral Twitter sentiment "
                    f"({neutral})"
                )
                return neutral
            if not bearer_token:
                self.logger.debug(
                    "TWITTER_BEARER_TOKEN not set; returning neutral Twitter "
                    f"sentiment ({neutral})"
                )
                return neutral

            # Real path: query recent 'Bitcoin price' tweets and average a
            # simple sentiment score normalized to [-1, 1].
            client = tweepy.Client(bearer_token=bearer_token)
            response = client.search_recent_tweets(
                query="Bitcoin price -is:retweet lang:en", max_results=100
            )
            tweets = getattr(response, "data", None) or []
            if not tweets:
                self.logger.debug(
                    f"No tweets returned; returning neutral sentiment ({neutral})"
                )
                return neutral

            scores = [self._score_text(t.text) for t in tweets]
            return float(np.clip(sum(scores) / len(scores), -1.0, 1.0))

        except Exception as e:
            self.logger.warning(f"Error fetching Twitter sentiment: {e}")
            return neutral

    @staticmethod
    def _score_text(text: str) -> float:
        """Lightweight lexicon sentiment score in [-1, 1] for a single tweet."""
        positive = ("bull", "moon", "up", "gain", "buy", "surge", "rally")
        negative = ("bear", "down", "loss", "sell", "crash", "dump", "drop")
        lowered = text.lower()
        score = sum(w in lowered for w in positive) - sum(
            w in lowered for w in negative
        )
        return float(np.clip(score, -1.0, 1.0))


class GoogleTrendsCollector:
    """Collect Google Trends data as specified in research."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        # In a real implementation, you'd use Google Trends API

    def get_bitcoin_trends_interpolated(self) -> float:
        """
        Get Google Trends data for 'Bitcoin' interpolated to 5-minute frequency.
        Research uses linear interpolation from daily to 5-minute data.

        Real data requires the optional ``pytrends`` dependency. Google Trends
        needs no API key, but ``pytrends`` may need ``GOOGLE_TRENDS_HL`` /
        ``GOOGLE_TRENDS_TZ`` overrides for locale/timezone. When ``pytrends`` is
        absent, a neutral constant is returned and the reason is logged at debug
        level.

        Returns:
            float: Trends score (0-100 scale)
        """
        neutral = 50.0
        try:
            if TrendReq is None:
                self.logger.debug(
                    "pytrends not installed; returning neutral Google Trends "
                    f"score ({neutral})"
                )
                return neutral

            # Real path: pull the latest 'Bitcoin' interest-over-time value.
            hl = os.environ.get("GOOGLE_TRENDS_HL", "en-US")
            tz = int(os.environ.get("GOOGLE_TRENDS_TZ", "0"))
            pytrends = TrendReq(hl=hl, tz=tz)
            pytrends.build_payload(["Bitcoin"], timeframe="now 1-d")
            interest = pytrends.interest_over_time()
            if interest is None or interest.empty:
                self.logger.debug(
                    f"Empty Google Trends response; returning neutral ({neutral})"
                )
                return neutral

            return float(interest["Bitcoin"].iloc[-1])

        except Exception as e:
            self.logger.warning(f"Error fetching Google Trends: {e}")
            return neutral
