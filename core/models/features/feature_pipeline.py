import pandas as pd
import numpy as np
from typing import Dict
import hashlib

class FeaturePipeline:
    """
    Modular feature engineering pipeline.
    Handles versioning and integration of market data features.
    """

    def __init__(self):
        self.version = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transformations.
        """
        df = df.copy()

        # Example: OHLCV features
        df['price_diff'] = df['close'] - df['open']
        df['high_low_ratio'] = df['high'] / df['low'].replace(0, np.nan)

        # Example: Rolling features
        df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
        df['rolling_std_5'] = df['close'].rolling(window=5).std()

        # Save feature version
        self.version = self._hash_features(df)

        return df.dropna()

    def _hash_features(self, df: pd.DataFrame) -> str:
        """
        Create a version hash of the feature columns.
        """
        feature_cols = sorted(df.columns)
        feature_str = ''.join(feature_cols).encode('utf-8')
        return hashlib.md5(feature_str).hexdigest()

    def get_version(self) -> str:
        return self.version