"""Canonical feature schemas used by ELVIS strategies and model artefacts."""

from trading.models.feature_schema import FeatureSchema, FeatureSpec


def _features(*names: str, dtype: str = "float64") -> tuple[FeatureSpec, ...]:
    return tuple(FeatureSpec(name, dtype) for name in names)


_FINANCIAL_FEATURES = _features(
    "RSI", "STOCH", "ROC", "EMA", "MACD", "CCI", "OBV", "ATR", "WILLR"
)

RESEARCH_FINANCIAL_9_V1 = FeatureSchema(
    "elvis.research.financial",
    1,
    _FINANCIAL_FEATURES,
    "standard-scaler.v1",
)

BONENKAMP_FINANCIAL_9_V1 = FeatureSchema(
    "elvis.bonenkamp.financial",
    1,
    _FINANCIAL_FEATURES,
    "standard-scaler.v1",
)

RESEARCH_SOCIAL_11_V1 = FeatureSchema(
    "elvis.research.social",
    1,
    RESEARCH_FINANCIAL_9_V1.features
    + _features("TWITTER_PRICE_SENTIMENT", "GOOGLE_TRENDS_BITCOIN"),
    "standard-scaler.v1",
)

BONENKAMP_SOCIAL_11_V1 = FeatureSchema(
    "elvis.bonenkamp.social",
    1,
    BONENKAMP_FINANCIAL_9_V1.features + _features("TWITTER_PRICE_LAG", "GOOGLE_TRENDS"),
    "standard-scaler.v1",
)

_ENSEMBLE_COREML_FEATURE_NAMES = (
    "price",
    "Order_Amount",
    "sma",
    "Filled",
    "Total",
    "future_price",
    "atr",
    "vol_adjusted_price",
    "volume_ma",
    "macd",
    "signal_line",
    "lower_bb",
    "sma_bb",
    "upper_bb",
    "news_sentiment",
    "social_feature",
    "adx",
    "rsi",
    "order_book_depth",
    "volume",
)

ENSEMBLE_YDF_20_V1 = FeatureSchema(
    "elvis.ensemble.ydf",
    1,
    _features(
        "price",
        "volume",
        "rsi",
        "macd",
        "sma",
        "atr",
        "adx",
        "Order_Amount",
        "Filled",
        "Total",
        "future_price",
        "vol_adjusted_price",
        "volume_ma",
        "signal_line",
        "lower_bb",
        "sma_bb",
        "upper_bb",
        "news_sentiment",
        "social_feature",
        "order_book_depth",
    ),
    "identity.v1",
)

ENSEMBLE_COREML_20_V1 = FeatureSchema(
    "elvis.ensemble.coreml",
    1,
    _features(*_ENSEMBLE_COREML_FEATURE_NAMES, dtype="float32"),
    "legacy-amplitude.v1",
)

COREML_SCHEMA_METADATA_KEY = "com.elvis.feature-schema"
