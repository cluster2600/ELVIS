from dataclasses import FrozenInstanceError

import pytest

from trading.models.feature_schema import (
    FeatureContractError,
    FeatureSchema,
    FeatureSpec,
)
from trading.models.feature_schemas import (
    BONENKAMP_FINANCIAL_9_V1,
    BONENKAMP_SOCIAL_11_V1,
    RESEARCH_FINANCIAL_9_V1,
    RESEARCH_SOCIAL_11_V1,
)


@pytest.mark.parametrize(
    ("schema", "size"),
    [
        (RESEARCH_FINANCIAL_9_V1, 9),
        (BONENKAMP_FINANCIAL_9_V1, 9),
        (RESEARCH_SOCIAL_11_V1, 11),
        (BONENKAMP_SOCIAL_11_V1, 11),
    ],
)
def test_registered_schemas_have_stable_sizes(schema: FeatureSchema, size: int) -> None:
    assert schema.size == size
    assert len(schema.names) == size
    assert len(schema.dtypes) == size
    assert len(set(schema.names)) == size


def test_optional_social_variants_have_distinct_identities() -> None:
    assert RESEARCH_FINANCIAL_9_V1.identity != BONENKAMP_FINANCIAL_9_V1.identity
    assert RESEARCH_FINANCIAL_9_V1.identity != RESEARCH_SOCIAL_11_V1.identity
    assert BONENKAMP_FINANCIAL_9_V1.identity != BONENKAMP_SOCIAL_11_V1.identity
    assert RESEARCH_SOCIAL_11_V1.identity != BONENKAMP_SOCIAL_11_V1.identity
    assert RESEARCH_SOCIAL_11_V1.names[-2:] == (
        "TWITTER_PRICE_SENTIMENT",
        "GOOGLE_TRENDS_BITCOIN",
    )
    assert BONENKAMP_SOCIAL_11_V1.names[-2:] == (
        "TWITTER_PRICE_LAG",
        "GOOGLE_TRENDS",
    )


def test_vectorize_uses_schema_order_and_ignores_unrelated_context() -> None:
    values = {
        name: index + 0.5 for index, name in enumerate(RESEARCH_FINANCIAL_9_V1.names)
    }
    values["symbol"] = "BTCUSDT"

    vector = RESEARCH_FINANCIAL_9_V1.vectorize(values)

    assert vector == tuple(index + 0.5 for index in range(9))


@pytest.mark.parametrize("bad_value", [True, "1.0", float("nan"), float("inf")])
def test_vectorize_rejects_non_numeric_or_non_finite_values(bad_value: object) -> None:
    values = {name: 1.0 for name in RESEARCH_FINANCIAL_9_V1.names}
    values[RESEARCH_FINANCIAL_9_V1.names[3]] = bad_value

    with pytest.raises(FeatureContractError, match=RESEARCH_FINANCIAL_9_V1.names[3]):
        RESEARCH_FINANCIAL_9_V1.vectorize(values)


def test_vectorize_rejects_a_missing_feature_instead_of_padding() -> None:
    values = {name: 1.0 for name in RESEARCH_FINANCIAL_9_V1.names[:-1]}

    with pytest.raises(FeatureContractError, match=RESEARCH_FINANCIAL_9_V1.names[-1]):
        RESEARCH_FINANCIAL_9_V1.vectorize(values)


def test_schema_rejects_duplicate_features() -> None:
    feature = FeatureSpec("price", "float64")

    with pytest.raises(ValueError, match="unique"):
        FeatureSchema("elvis.test", 1, (feature, feature))


def test_schema_is_frozen_and_hashable() -> None:
    assert {RESEARCH_FINANCIAL_9_V1} == {RESEARCH_FINANCIAL_9_V1}

    with pytest.raises(FrozenInstanceError):
        RESEARCH_FINANCIAL_9_V1.version = 2  # type: ignore[misc]


class FittedComponent:
    n_features_in_ = 9


@pytest.mark.parametrize("count", [8, 10, None, True])
def test_fitted_component_dimension_must_match_schema(count: object) -> None:
    component = FittedComponent()
    component.n_features_in_ = count

    with pytest.raises(FeatureContractError, match="scaler"):
        RESEARCH_FINANCIAL_9_V1.validate_fitted_component(component, "scaler")


def test_fitted_component_with_exact_dimension_is_accepted() -> None:
    RESEARCH_FINANCIAL_9_V1.validate_fitted_component(FittedComponent(), "model")
