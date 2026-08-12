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
