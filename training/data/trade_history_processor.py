#!/usr/bin/env python3
"""
Trade History Data Processor for Training Pipeline
Extracts and processes trade history from the database for model training.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.paper_trade_db import get_all_trades, get_trade_count


class TradeHistoryProcessor:
    """
    Processes trade history from the database for training purposes.
    Extracts features from trading patterns, market conditions, and outcomes.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.trade_data = None
        self.processed_features = None

    def load_trade_history(
        self, limit: Optional[int] = None, exclude_test: bool = True
    ) -> pd.DataFrame:
        """
        Load trade history from the database.

        Args:
            limit: Maximum number of trades to load (None for all)
            exclude_test: Whether to exclude test trades

        Returns:
            DataFrame with trade history
        """
        try:
            # Get trade count for logging
            total_trades = get_trade_count(exclude_test=exclude_test)
            self.logger.info(
                f"Loading trade history: {total_trades} total trades available"
            )

            # Load trades from database
            trades = get_all_trades(
                limit=limit or total_trades, exclude_test=exclude_test
            )

            if not trades:
                self.logger.warning("No trades found in database")
                return pd.DataFrame()

            # Convert to DataFrame
            columns = [
                "id",
                "timestamp",
                "symbol",
                "side",
                "price",
                "quantity",
                "pnl",
                "fee",
            ]
            df = pd.DataFrame(trades, columns=columns)

            # Data type conversions
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
            df["fee"] = pd.to_numeric(df["fee"], errors="coerce")

            # Remove invalid entries
            df = df.dropna(subset=["price", "quantity", "pnl", "fee"])

            self.logger.info(f"Loaded {len(df)} valid trades for training")
            self.trade_data = df
            return df

        except Exception as e:
            self.logger.error(f"Error loading trade history: {e}")
            raise

    def extract_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract market-related features from trade data.

        Args:
            df: Trade data DataFrame

        Returns:
            DataFrame with market features
        """
        try:
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Calculate time-based features
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["minute"] = df["timestamp"].dt.minute

            # Calculate price movements
            df["price_change"] = df["price"].pct_change()
            df["price_volatility"] = df["price"].rolling(window=10, min_periods=1).std()
            df["price_sma_5"] = df["price"].rolling(window=5).mean()
            df["price_sma_20"] = df["price"].rolling(window=20).mean()

            # Calculate volume features
            df["volume_change"] = df["quantity"].pct_change()
            df["volume_sma"] = df["quantity"].rolling(window=10).mean()

            # Calculate momentum indicators
            df["price_momentum_5"] = (df["price"] - df["price"].shift(5)) / df[
                "price"
            ].shift(5)
            df["price_momentum_20"] = (df["price"] - df["price"].shift(20)) / df[
                "price"
            ].shift(20)

            # High/Low price levels
            df["price_high_10"] = df["price"].rolling(window=10).max()
            df["price_low_10"] = df["price"].rolling(window=10).min()
            df["price_position"] = (df["price"] - df["price_low_10"]) / (
                df["price_high_10"] - df["price_low_10"]
            )

            return df

        except Exception as e:
            self.logger.error(f"Error extracting market features: {e}")
            raise

    def extract_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract trading-specific features from trade data.

        Args:
            df: Trade data DataFrame

        Returns:
            DataFrame with trading features
        """
        try:
            # Position side encoding
            df["is_long"] = (df["side"] == "BUY").astype(int)
            df["is_short"] = (df["side"] == "SELL").astype(int)

            # Calculate notional value
            df["notional_value"] = df["price"] * df["quantity"]

            # Calculate net result
            df["net_result"] = df["pnl"] - df["fee"]

            # Profit/Loss categorization
            df["is_profitable"] = (df["net_result"] > 0).astype(int)
            df["is_loss"] = (df["net_result"] < 0).astype(int)
            df["is_breakeven"] = (df["net_result"] == 0).astype(int)

            # Fee impact analysis
            df["fee_ratio"] = df["fee"] / df["notional_value"]
            df["fee_to_pnl_ratio"] = df["fee"] / df["pnl"].abs()

            # Rolling performance metrics
            df["rolling_pnl_5"] = df["pnl"].rolling(window=5).sum()
            df["rolling_pnl_20"] = df["pnl"].rolling(window=20).sum()
            df["rolling_winrate_10"] = df["is_profitable"].rolling(window=10).mean()
            df["rolling_winrate_50"] = df["is_profitable"].rolling(window=50).mean()

            # Consecutive wins/losses
            df["win_streak"] = self._calculate_streaks(df["is_profitable"])
            df["loss_streak"] = self._calculate_streaks(df["is_loss"])

            # Position sizing analysis
            df["position_size_zscore"] = (
                df["notional_value"]
                - df["notional_value"].rolling(window=50, min_periods=1).mean()
            ) / df["notional_value"].rolling(window=50, min_periods=1).std().replace(
                0, np.nan
            )

            return df

        except Exception as e:
            self.logger.error(f"Error extracting trading features: {e}")
            raise

    def _calculate_streaks(self, series: pd.Series) -> pd.Series:
        """
        Calculate consecutive streaks of 1s in a binary series.

        Args:
            series: Binary series (0s and 1s)

        Returns:
            Series with streak counts
        """
        streaks = []
        current_streak = 0

        for value in series:
            if value == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)

        return pd.Series(streaks, index=series.index)

    def create_target_labels(
        self, df: pd.DataFrame, prediction_horizon: int = 5
    ) -> pd.DataFrame:
        """
        Create target labels for supervised learning.

        Args:
            df: Trade data DataFrame
            prediction_horizon: Number of trades to look ahead for targets

        Returns:
            DataFrame with target labels
        """
        try:
            # Future profitability (main target)
            df["future_profitable"] = df["is_profitable"].shift(-prediction_horizon)

            # Future price direction
            df["future_price_up"] = (
                df["price"].shift(-prediction_horizon) > df["price"]
            ).astype(int)

            # Future market bias (based on side distribution)
            df["future_long_bias"] = (
                df["is_long"]
                .rolling(window=prediction_horizon)
                .mean()
                .shift(-prediction_horizon)
            )

            # Future volatility
            df["future_volatility"] = (
                df["price"]
                .rolling(window=prediction_horizon, min_periods=1)
                .std()
                .shift(-prediction_horizon)
            )

            # Future performance
            df["future_pnl"] = df["pnl"].shift(-prediction_horizon)
            df["future_net_result"] = df["net_result"].shift(-prediction_horizon)

            # Remove rows with NaN targets
            df = df.dropna(subset=["future_profitable", "future_price_up"])

            return df

        except Exception as e:
            self.logger.error(f"Error creating target labels: {e}")
            raise

    def process_for_training(
        self, limit: Optional[int] = None, prediction_horizon: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full processing pipeline for training data.

        Args:
            limit: Maximum number of trades to process
            prediction_horizon: Number of trades to look ahead for targets

        Returns:
            Tuple of (features_df, targets_df)
        """
        try:
            # Load trade history
            df = self.load_trade_history(limit=limit)

            if df.empty:
                self.logger.warning("No trade data available for processing")
                return pd.DataFrame(), pd.DataFrame()

            # Extract features
            df = self.extract_market_features(df)
            df = self.extract_trading_features(df)

            # Create targets
            df = self.create_target_labels(df, prediction_horizon)

            # Separate features and targets
            feature_columns = [
                col
                for col in df.columns
                if not col.startswith("future_")
                and col not in ["id", "timestamp", "symbol"]
            ]
            target_columns = [col for col in df.columns if col.startswith("future_")]

            features_df = df[feature_columns].copy()
            targets_df = df[target_columns].copy()

            # Handle missing values - only fill numeric columns
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df[numeric_columns] = features_df[numeric_columns].fillna(
                features_df[numeric_columns].mean()
            )

            # Fill non-numeric columns with forward fill and backward fill
            non_numeric_columns = features_df.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_columns:
                features_df[col] = features_df[col].ffill().bfill()

            targets_df = targets_df.fillna(0)

            self.logger.info(
                f"Processed {len(features_df)} samples with {len(feature_columns)} features"
            )
            self.logger.info(f"Feature columns: {feature_columns}")
            self.logger.info(f"Target columns: {target_columns}")

            self.processed_features = features_df
            return features_df, targets_df

        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {e}")
            raise

    def get_feature_importance_data(self) -> Dict:
        """
        Get data for feature importance analysis.

        Returns:
            Dictionary with feature importance data
        """
        if self.processed_features is None:
            self.logger.warning(
                "No processed features available. Run process_for_training first."
            )
            return {}

        try:
            # Only analyze numeric columns for correlations and stats
            numeric_features = self.processed_features.select_dtypes(
                include=[np.number]
            )

            # Basic statistics
            feature_stats = {
                "feature_count": len(self.processed_features.columns),
                "sample_count": len(self.processed_features),
                "feature_names": list(self.processed_features.columns),
                "numeric_feature_count": len(numeric_features.columns),
                "numeric_features": list(numeric_features.columns),
                "feature_correlations": (
                    numeric_features.corr().to_dict()
                    if len(numeric_features.columns) > 0
                    else {}
                ),
                "feature_means": (
                    numeric_features.mean().to_dict()
                    if len(numeric_features.columns) > 0
                    else {}
                ),
                "feature_stds": (
                    numeric_features.std().to_dict()
                    if len(numeric_features.columns) > 0
                    else {}
                ),
            }

            return feature_stats

        except Exception as e:
            self.logger.error(f"Error getting feature importance data: {e}")
            return {}

    def save_processed_data(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame, output_dir: str
    ):
        """
        Save processed data for training.

        Args:
            features_df: Features DataFrame
            targets_df: Targets DataFrame
            output_dir: Output directory path
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save features and targets
            features_df.to_csv(output_path / "trade_features.csv", index=False)
            targets_df.to_csv(output_path / "trade_targets.csv", index=False)

            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "feature_count": len(features_df.columns),
                "target_count": len(targets_df.columns),
                "sample_count": len(features_df),
                "feature_names": list(features_df.columns),
                "target_names": list(targets_df.columns),
            }

            import json

            with open(output_path / "trade_data_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Processed trade data saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise


def main():
    """Test the trade history processor."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    processor = TradeHistoryProcessor(logger)

    # Process trade history
    features_df, targets_df = processor.process_for_training(limit=1000)

    if not features_df.empty:
        print(f"Features shape: {features_df.shape}")
        print(f"Targets shape: {targets_df.shape}")

        # Save processed data
        processor.save_processed_data(features_df, targets_df, "data/processed")

        # Get feature importance data
        feature_info = processor.get_feature_importance_data()
        print(f"Feature importance data: {len(feature_info)} items")
    else:
        print("No data processed")


if __name__ == "__main__":
    main()
