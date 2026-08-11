"""Regression tests for the historical Research 9/11 feature mismatch."""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trading.models.feature_schemas import (
    RESEARCH_FINANCIAL_9_V1,
    RESEARCH_SOCIAL_11_V1,
)
from trading.strategies.research_based_strategy import ResearchBasedStrategy


@pytest.mark.parametrize("social", [False, True])
def test_research_feature_vector_matches_selected_schema(
    tmp_path: Path, social: bool
) -> None:
    with patch.object(ResearchBasedStrategy, "load_model", return_value=False):
        strategy = ResearchBasedStrategy(
            logger=logging.getLogger("test-feature-fix"),
            social_data_enabled=social,
            enable_rolling_training=False,
            model_save_path=str(tmp_path),
        )

    financial = {
        name: float(index + 1)
        for index, name in enumerate(RESEARCH_FINANCIAL_9_V1.names)
    }
    social_values = {
        "TWITTER_PRICE_SENTIMENT": 10.0,
        "GOOGLE_TRENDS_BITCOIN": 11.0,
    }
    with (
        patch.object(
            strategy, "calculate_financial_indicators", return_value=financial
        ),
        patch.object(strategy, "collect_social_features", return_value=social_values),
    ):
        features = strategy.prepare_features(np.empty((0, 0)))

    expected = RESEARCH_SOCIAL_11_V1 if social else RESEARCH_FINANCIAL_9_V1
    assert strategy.feature_schema is expected
    assert features.shape == (1, expected.size)
    assert tuple(features[0]) == expected.vectorize({**financial, **social_values})
