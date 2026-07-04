"""
Enhanced Trading Feature Pipeline for ELVIS Random Forest
Comprehensive feature engineering for cryptocurrency trading signals.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import ta

    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


class TradingFeaturePipeline:
    """
    Comprehensive feature engineering pipeline for cryptocurrency trading.

    Generates 50+ features covering:
    - Price action and momentum
    - Technical indicators
    - Volatility measures
    - Volume analysis
    - Market structure
    - Time-based features
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.feature_groups = {
            "price": [],
            "volume": [],
            "technical": [],
            "volatility": [],
            "momentum": [],
            "market_structure": [],
            "time": [],
        }

        if not TA_AVAILABLE:
            self.logger.warning("TA-Lib not available - using basic indicators only")

        self.logger.info("🔧 Trading Feature Pipeline initialized")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive trading features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']

        Returns:
            DataFrame with engineered features
        """

        if len(df) < 50:
            self.logger.warning(
                f"Insufficient data for feature generation: {len(df)} rows"
            )
            return self._generate_minimal_features(df)

        self.logger.info(f"🚀 Generating features for {len(df)} data points")

        # Start with a copy of the original data
        features_df = df.copy()

        # Ensure timestamp is datetime
        if "timestamp" in features_df.columns:
            features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])

        try:
            # Core feature groups
            features_df = self._add_price_features(features_df)
            features_df = self._add_volume_features(features_df)
            features_df = self._add_volatility_features(features_df)
            features_df = self._add_momentum_features(features_df)

            if TA_AVAILABLE:
                features_df = self._add_technical_indicators(features_df)
            else:
                features_df = self._add_basic_technical_indicators(features_df)

            features_df = self._add_market_structure_features(features_df)
            features_df = self._add_time_features(features_df)
            features_df = self._add_feature_interactions(features_df)

            # Clean and finalize
            features_df = self._clean_features(features_df)

            self.logger.info(
                f"✅ Generated {len([c for c in features_df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} features"
            )

        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            return self._generate_minimal_features(df)

        return features_df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""

        # Basic price relationships
        df["price_change"] = df["close"].pct_change()
        df["price_change_abs"] = df["price_change"].abs()
        df["high_low_ratio"] = df["high"] / df["low"].replace(0, np.nan)
        df["close_to_high"] = df["close"] / df["high"]
        df["close_to_low"] = df["close"] / df["low"]
        df["open_to_close"] = df["close"] / df["open"].replace(0, np.nan)

        # Candlestick patterns
        df["body_size"] = (df["close"] - df["open"]).abs() / df["close"]
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df[
            "close"
        ]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df[
            "close"
        ]
        df["is_hammer"] = (
            (df["lower_shadow"] > 2 * df["body_size"])
            & (df["upper_shadow"] < df["body_size"])
        ).astype(int)
        df["is_doji"] = (df["body_size"] < 0.001).astype(int)

        # Moving averages (multiple timeframes)
        for window in [5, 10, 20, 50]:
            # Simple moving averages
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"price_to_sma_{window}"] = df["close"] / df[f"sma_{window}"]
            df[f"sma_{window}_slope"] = df[f"sma_{window}"].pct_change()

            # Exponential moving averages
            df[f"ema_{window}"] = df["close"].ewm(span=window).mean()
            df[f"price_to_ema_{window}"] = df["close"] / df[f"ema_{window}"]
            df[f"ema_{window}_slope"] = df[f"ema_{window}"].pct_change()

        # Price position in range
        for window in [14, 28]:
            rolling_high = df["high"].rolling(window).max()
            rolling_low = df["low"].rolling(window).min()
            df[f"price_position_{window}"] = (df["close"] - rolling_low) / (
                rolling_high - rolling_low
            ).replace(0, np.nan)

        # Gap analysis
        df["gap_up"] = (df["open"] > df["close"].shift(1)).astype(int)
        df["gap_down"] = (df["open"] < df["close"].shift(1)).astype(int)
        df["gap_size"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        self.feature_groups["price"] = [
            col
            for col in df.columns
            if any(
                x in col
                for x in [
                    "price",
                    "sma",
                    "ema",
                    "body",
                    "shadow",
                    "hammer",
                    "doji",
                    "gap",
                    "position",
                ]
            )
        ]

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""

        # Basic volume features
        df["volume_change"] = df["volume"].pct_change()
        df["volume_sma_10"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"]

        # Volume-price relationship
        df["price_volume_trend"] = (df["close"] - df["close"].shift(1)) * df["volume"]
        df["volume_price_correlation"] = df["close"].rolling(20).corr(df["volume"])

        # On-Balance Volume (OBV)
        df["obv"] = (df["volume"] * np.sign(df["close"].diff())).cumsum()
        df["obv_sma_10"] = df["obv"].rolling(10).mean()
        df["obv_ratio"] = df["obv"] / df["obv_sma_10"]

        # Volume moving averages
        for window in [5, 20]:
            df[f"volume_sma_{window}"] = df["volume"].rolling(window).mean()
            df[f"relative_volume_{window}"] = df["volume"] / df[f"volume_sma_{window}"]

        # Volume-weighted average price
        df["vwap_numerator"] = (df["close"] * df["volume"]).rolling(20).sum()
        df["vwap_denominator"] = df["volume"].rolling(20).sum()
        df["vwap"] = df["vwap_numerator"] / df["vwap_denominator"].replace(0, np.nan)
        df["price_to_vwap"] = df["close"] / df["vwap"]

        self.feature_groups["volume"] = [
            col
            for col in df.columns
            if any(x in col for x in ["volume", "obv", "vwap"])
        ]

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""

        # True Range and Average True Range
        df["high_low"] = df["high"] - df["low"]
        df["high_close_prev"] = (df["high"] - df["close"].shift(1)).abs()
        df["low_close_prev"] = (df["low"] - df["close"].shift(1)).abs()
        df["true_range"] = df[["high_low", "high_close_prev", "low_close_prev"]].max(
            axis=1
        )

        for window in [14, 28]:
            df[f"atr_{window}"] = df["true_range"].rolling(window).mean()
            df[f"volatility_{window}"] = df["close"].rolling(window).std()
            df[f"normalized_volatility_{window}"] = (
                df[f"volatility_{window}"] / df["close"]
            )

        # Garman-Klass volatility estimator
        df["gk_volatility"] = np.sqrt(
            0.5 * (np.log(df["high"] / df["low"])) ** 2
            - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"])) ** 2
        )

        # Rolling volatility measures
        df["volatility_ratio"] = df["volatility_14"] / df["volatility_28"]
        df["volatility_trend"] = df["volatility_14"].pct_change()

        # Volatility breakouts
        df["volatility_breakout"] = (df["true_range"] > df["atr_14"] * 2).astype(int)

        self.feature_groups["volatility"] = [
            col
            for col in df.columns
            if any(x in col for x in ["volatility", "atr", "true_range", "gk_"])
        ]

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""

        # Rate of Change (ROC)
        for window in [5, 10, 20]:
            df[f"roc_{window}"] = df["close"].pct_change(window)
            df[f"roc_{window}_ma"] = df[f"roc_{window}"].rolling(5).mean()

        # Momentum indicators
        df["momentum_10"] = df["close"] / df["close"].shift(10)
        df["momentum_20"] = df["close"] / df["close"].shift(20)

        # Acceleration
        df["price_acceleration"] = df["price_change"].diff()
        df["volume_acceleration"] = df["volume_change"].diff()

        # Momentum oscillators (basic versions)
        for window in [14, 21]:
            # Williams %R
            rolling_high = df["high"].rolling(window).max()
            rolling_low = df["low"].rolling(window).min()
            df[f"williams_r_{window}"] = (
                -100 * (rolling_high - df["close"]) / (rolling_high - rolling_low)
            )

            # Stochastic %K
            df[f"stoch_k_{window}"] = (
                100 * (df["close"] - rolling_low) / (rolling_high - rolling_low)
            )
            df[f"stoch_d_{window}"] = df[f"stoch_k_{window}"].rolling(3).mean()

        self.feature_groups["momentum"] = [
            col
            for col in df.columns
            if any(
                x in col
                for x in ["roc", "momentum", "acceleration", "williams", "stoch"]
            )
        ]

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators using TA-Lib."""

        try:
            # RSI variations
            for window in [14, 21]:
                df[f"rsi_{window}"] = ta.momentum.RSIIndicator(
                    df["close"], window=window
                ).rsi()

            # MACD
            macd_indicator = ta.trend.MACD(df["close"])
            df["macd"] = macd_indicator.macd()
            df["macd_signal"] = macd_indicator.macd_signal()
            df["macd_histogram"] = macd_indicator.macd_diff()

            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df["close"])
            df["bb_upper"] = bb_indicator.bollinger_hband()
            df["bb_lower"] = bb_indicator.bollinger_lband()
            df["bb_middle"] = bb_indicator.bollinger_mavg()
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

            # ADX and trend indicators
            adx_indicator = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
            df["adx"] = adx_indicator.adx()
            df["di_plus"] = adx_indicator.adx_pos()
            df["di_minus"] = adx_indicator.adx_neg()

            # Commodity Channel Index
            df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci()

            # Money Flow Index
            df["mfi"] = ta.volume.MFIIndicator(
                df["high"], df["low"], df["close"], df["volume"]
            ).money_flow_index()

            # Parabolic SAR
            df["psar"] = ta.trend.PSARIndicator(
                df["high"], df["low"], df["close"]
            ).psar()
            df["psar_signal"] = (df["close"] > df["psar"]).astype(int)

        except Exception as e:
            self.logger.warning(f"Advanced technical indicators failed: {e}")
            return self._add_basic_technical_indicators(df)

        self.feature_groups["technical"] = [
            col
            for col in df.columns
            if any(
                x in col
                for x in ["rsi", "macd", "bb_", "adx", "di_", "cci", "mfi", "psar"]
            )
        ]

        return df

    def _add_basic_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators without TA-Lib."""

        # Basic RSI implementation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Basic MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Basic Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (2 * std_20)
        df["bb_lower"] = sma_20 - (2 * std_20)
        df["bb_middle"] = sma_20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        return df

    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""

        # Support and resistance levels
        for window in [20, 50]:
            df[f"resistance_{window}"] = df["high"].rolling(window).max()
            df[f"support_{window}"] = df["low"].rolling(window).min()
            df[f"distance_to_resistance_{window}"] = (
                df[f"resistance_{window}"] - df["close"]
            ) / df["close"]
            df[f"distance_to_support_{window}"] = (
                df["close"] - df[f"support_{window}"]
            ) / df["close"]

        # Trend detection
        df["higher_highs"] = (df["high"] > df["high"].shift(1)).rolling(5).sum()
        df["lower_lows"] = (df["low"] < df["low"].shift(1)).rolling(5).sum()
        df["trend_strength"] = (df["higher_highs"] - df["lower_lows"]) / 5

        # Fractal patterns
        df["local_high"] = (
            (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
        ).astype(int)
        df["local_low"] = (
            (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))
        ).astype(int)

        # Market regime detection
        df["regime_volatility"] = (
            df["volatility_14"] > df["volatility_14"].rolling(50).mean()
        ).astype(int)
        df["regime_trend"] = (df["close"] > df["sma_50"]).astype(int)

        self.feature_groups["market_structure"] = [
            col
            for col in df.columns
            if any(
                x in col for x in ["resistance", "support", "trend", "local_", "regime"]
            )
        ]

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""

        if "timestamp" not in df.columns:
            return df

        try:
            # Extract time components
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["day_of_month"] = df["timestamp"].dt.day
            df["month"] = df["timestamp"].dt.month

            # Cyclical encoding for time features
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

            # Market session indicators
            df["us_session"] = ((df["hour"] >= 9) & (df["hour"] <= 16)).astype(int)
            df["europe_session"] = ((df["hour"] >= 2) & (df["hour"] <= 11)).astype(int)
            df["asia_session"] = ((df["hour"] >= 17) | (df["hour"] <= 2)).astype(int)

            # Weekend indicator
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        except Exception as e:
            self.logger.warning(f"Time feature generation failed: {e}")

        self.feature_groups["time"] = [
            col
            for col in df.columns
            if any(x in col for x in ["hour", "day", "month", "session", "weekend"])
        ]

        return df

    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add important feature interactions."""

        try:
            # Price-volume interactions
            df["price_volume_momentum"] = df["price_change"] * df["volume_ratio"]
            df["volatility_volume"] = df["volatility_14"] * df["relative_volume_5"]

            # Technical indicator combinations
            df["rsi_bb_combo"] = df["rsi_14"] * df["bb_position"]
            df["macd_rsi_combo"] = df["macd_histogram"] * (df["rsi_14"] - 50) / 50

            # Momentum-volatility interactions
            df["momentum_volatility_ratio"] = df["momentum_10"] / (
                df["volatility_14"] + 1e-8
            )

            # Trend confirmation signals
            df["trend_confirmation"] = (
                (df["price_to_sma_20"] > 1) & (df["rsi_14"] > 50) & (df["macd"] > 0)
            ).astype(int)

        except Exception as e:
            self.logger.warning(f"Feature interaction generation failed: {e}")

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features."""

        # Drop timestamp column as it's not suitable for ML
        if "timestamp" in df.columns:
            df = df.drop("timestamp", axis=1)

        # Convert all columns to numeric, dropping non-numeric columns
        numeric_df = pd.DataFrame()
        for col in df.columns:
            try:
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                if not numeric_series.isna().all():  # Keep if not all NaN
                    numeric_df[col] = numeric_series
            except Exception:
                continue

        df = numeric_df

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with forward fill, then backward fill, then 0
        df = df.ffill().bfill().fillna(0)

        # Drop columns with too many NaN values (>90%)
        threshold = len(df) * 0.1  # Keep columns with at least 10% valid data
        df = df.dropna(axis=1, thresh=threshold)

        # Remove features with zero variance
        for col in df.columns:
            if df[col].std() == 0:
                df = df.drop(col, axis=1)

        # Ensure all data is float64
        df = df.astype(np.float64)

        return df

    def _generate_minimal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate minimal features when full pipeline fails."""

        features_df = df.copy()

        try:
            # Basic price features
            features_df["price_change"] = features_df["close"].pct_change()
            features_df["high_low_ratio"] = features_df["high"] / features_df["low"]
            features_df["volume_change"] = (
                features_df["volume"].pct_change()
                if "volume" in features_df.columns
                else 0
            )

            # Simple moving averages
            features_df["sma_5"] = features_df["close"].rolling(5).mean()
            features_df["sma_20"] = features_df["close"].rolling(20).mean()
            features_df["price_to_sma_20"] = (
                features_df["close"] / features_df["sma_20"]
            )

            # Basic volatility
            features_df["volatility_10"] = features_df["close"].rolling(10).std()

            # Clean minimal features
            features_df = features_df.ffill().fillna(0)

        except Exception as e:
            self.logger.error(f"Minimal feature generation failed: {e}")

        return features_df

    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names by category."""
        return self.feature_groups.copy()

    def get_feature_count(self) -> int:
        """Get total feature count."""
        return sum(len(features) for features in self.feature_groups.values())

    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate generated features."""

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        validation = {
            "total_features": len(df.columns),
            "numeric_features": len(numeric_cols),
            "null_counts": df.isnull().sum().to_dict(),
            "infinite_counts": np.isinf(df.select_dtypes(include=[np.number]))
            .sum()
            .to_dict(),
            "zero_variance_features": [
                col for col in numeric_cols if df[col].std() == 0
            ],
            "feature_ranges": {
                col: {"min": df[col].min(), "max": df[col].max()}
                for col in numeric_cols[:10]
            },  # First 10 features only
            "data_quality_score": 1.0
            - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
        }

        return validation
