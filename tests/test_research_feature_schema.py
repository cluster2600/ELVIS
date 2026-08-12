import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from trading.models.artifact_manifest import ArtifactDescriptor, write_feature_manifest
from trading.models.feature_schema import FeatureContractError
from trading.models.feature_schemas import (
    RESEARCH_FINANCIAL_9_V1,
    RESEARCH_SOCIAL_11_V1,
)

DESCRIPTOR = ArtifactDescriptor(
    model_kind="random-forest-classifier",
    library="scikit-learn",
    library_version=sklearn.__version__,
)


def make_strategy(tmp_path: Path, *, social: bool = False):
    from trading.strategies.research_based_strategy import ResearchBasedStrategy

    with patch.object(ResearchBasedStrategy, "load_model", return_value=False):
        return ResearchBasedStrategy(
            logging.getLogger("research-feature-schema-test"),
            social_data_enabled=social,
            enable_rolling_training=False,
            model_save_path=str(tmp_path),
        )


def financial_feature_values() -> dict[str, float]:
    return {
        name: float(index + 1)
        for index, name in enumerate(RESEARCH_FINANCIAL_9_V1.names)
    }


@pytest.mark.parametrize("social", [False, True])
def test_research_strategy_selects_explicit_schema(
    tmp_path: Path, social: bool
) -> None:
    strategy = make_strategy(tmp_path, social=social)

    expected = RESEARCH_SOCIAL_11_V1 if social else RESEARCH_FINANCIAL_9_V1
    assert strategy.feature_schema is expected
    assert tuple(strategy.feature_names) == expected.names


def test_research_loader_rejects_incompatible_manifest_before_joblib(
    tmp_path: Path,
) -> None:
    from trading.strategies.research_based_strategy import ResearchBasedStrategy

    model_path = tmp_path / "research_rf_model.pkl"
    scaler_path = tmp_path / "research_scaler.pkl"
    model_path.write_bytes(b"not-a-pickle")
    scaler_path.write_bytes(b"not-a-pickle")
    write_feature_manifest(
        tmp_path / "feature_manifest.json",
        RESEARCH_SOCIAL_11_V1,
        DESCRIPTOR,
        {"model": model_path, "scaler": scaler_path},
    )

    with patch("trading.strategies.research_based_strategy.joblib.load") as load:
        strategy = ResearchBasedStrategy(
            logging.getLogger("research-feature-schema-test"),
            social_data_enabled=False,
            enable_rolling_training=False,
            model_save_path=str(tmp_path),
        )

    load.assert_not_called()
    assert strategy.is_trained is False


def test_research_loader_rejects_missing_manifest_before_joblib(
    tmp_path: Path,
) -> None:
    from trading.strategies.research_based_strategy import ResearchBasedStrategy

    (tmp_path / "research_rf_model.pkl").write_bytes(b"not-a-pickle")
    (tmp_path / "research_scaler.pkl").write_bytes(b"not-a-pickle")

    with patch("trading.strategies.research_based_strategy.joblib.load") as load:
        strategy = ResearchBasedStrategy(
            logging.getLogger("research-feature-schema-test"),
            social_data_enabled=False,
            enable_rolling_training=False,
            model_save_path=str(tmp_path),
        )

    load.assert_not_called()
    assert strategy.is_trained is False


def test_scaler_mismatch_is_not_used_as_unscaled_inference(tmp_path: Path) -> None:
    strategy = make_strategy(tmp_path)
    strategy.feature_scaler = StandardScaler().fit(np.ones((2, 11)))

    with patch.object(
        strategy,
        "calculate_financial_indicators",
        return_value=financial_feature_values(),
    ):
        with pytest.raises(FeatureContractError, match="cannot produce"):
            strategy.prepare_features(np.empty((0, 0)))


def test_trained_strategy_never_predicts_with_unfitted_scaler(tmp_path: Path) -> None:
    strategy = make_strategy(tmp_path)
    model = MagicMock()
    strategy.rf_model = model
    strategy.is_trained = True
    market_data = pd.DataFrame({"close": [100.0]})

    with patch.object(
        strategy,
        "calculate_financial_indicators",
        return_value=financial_feature_values(),
    ):
        signal = strategy.generate_signals({"BTCUSDT": market_data})["BTCUSDT"]

    model.predict_proba.assert_not_called()
    assert signal["signal"] in ("BUY", "SELL")


def test_research_strategy_artifact_round_trip(tmp_path: Path) -> None:
    from trading.strategies.research_based_strategy import ResearchBasedStrategy

    source = make_strategy(tmp_path)
    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    source.feature_scaler = StandardScaler().fit(training)
    source.rf_model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        source.feature_scaler.transform(training), labels
    )
    source.is_trained = True
    source.save_model()

    restored = ResearchBasedStrategy(
        logging.getLogger("research-feature-schema-test"),
        social_data_enabled=False,
        enable_rolling_training=False,
        model_save_path=str(tmp_path),
    )

    assert restored.is_trained is True
    assert restored.feature_schema is RESEARCH_FINANCIAL_9_V1
    prediction = restored.rf_model.predict(
        restored.feature_scaler.transform(training[[0]])
    )
    assert prediction.shape == (1,)


def test_research_loader_does_not_partially_assign_invalid_scaler(
    tmp_path: Path,
) -> None:
    from trading.strategies.research_based_strategy import ResearchBasedStrategy

    source = make_strategy(tmp_path)
    training = np.arange(180, dtype=float).reshape(20, 9)
    source.rf_model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        training, np.array([0, 1] * 10)
    )
    source.feature_scaler = StandardScaler().fit(np.ones((2, 11)))
    model_path = tmp_path / "research_rf_model.pkl"
    scaler_path = tmp_path / "research_scaler.pkl"
    joblib.dump(source.rf_model, model_path)
    joblib.dump(source.feature_scaler, scaler_path)
    write_feature_manifest(
        tmp_path / "feature_manifest.json",
        RESEARCH_FINANCIAL_9_V1,
        DESCRIPTOR,
        {"model": model_path, "scaler": scaler_path},
    )

    restored = ResearchBasedStrategy(
        logging.getLogger("research-feature-schema-test"),
        social_data_enabled=False,
        enable_rolling_training=False,
        model_save_path=str(tmp_path),
    )

    assert restored.is_trained is False
    assert not hasattr(restored.rf_model, "n_features_in_")
    assert not hasattr(restored.feature_scaler, "n_features_in_")


@pytest.mark.parametrize("wrong_component", ["model", "scaler"])
def test_research_loader_rejects_descriptor_with_wrong_implementation(
    tmp_path: Path, wrong_component: str
) -> None:
    from trading.strategies.research_based_strategy import ResearchBasedStrategy

    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    model = RandomForestClassifier(n_estimators=2, random_state=3).fit(training, labels)
    scaler = StandardScaler().fit(training)
    if wrong_component == "model":
        model = LogisticRegression().fit(training, labels)
    else:
        scaler = MinMaxScaler().fit(training)
    model_path = tmp_path / "research_rf_model.pkl"
    scaler_path = tmp_path / "research_scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    write_feature_manifest(
        tmp_path / "feature_manifest.json",
        RESEARCH_FINANCIAL_9_V1,
        DESCRIPTOR,
        {"model": model_path, "scaler": scaler_path},
    )

    restored = ResearchBasedStrategy(
        logging.getLogger("research-feature-schema-test"),
        social_data_enabled=False,
        enable_rolling_training=False,
        model_save_path=str(tmp_path),
    )

    assert restored.is_trained is False
    assert isinstance(restored.rf_model, RandomForestClassifier)
    assert isinstance(restored.feature_scaler, StandardScaler)
    assert not hasattr(restored.rf_model, "n_features_in_")
    assert not hasattr(restored.feature_scaler, "n_features_in_")


def test_failed_retraining_preserves_active_model_and_scaler(tmp_path: Path) -> None:
    import trading.strategies.research_based_strategy as strategy_module

    strategy = make_strategy(tmp_path)
    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    active_scaler = StandardScaler().fit(training)
    active_model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        active_scaler.transform(training), labels
    )
    last_retrain_time = datetime(2025, 1, 2, 3, 4, 5)
    original_mean = active_scaler.mean_.copy()
    strategy.feature_scaler = active_scaler
    strategy.rf_model = active_model
    strategy.is_trained = True
    strategy.last_retrain_time = last_retrain_time
    market_history = pd.DataFrame({"close": np.arange(120, dtype=float)})

    with (
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value=financial_feature_values(),
        ),
        patch.object(
            strategy_module, "cross_val_score", side_effect=RuntimeError("cv failed")
        ),
    ):
        score = strategy.train_model(market_history)

    assert score == 0.0
    assert strategy.rf_model is active_model
    assert strategy.feature_scaler is active_scaler
    np.testing.assert_array_equal(strategy.feature_scaler.mean_, original_mean)
    assert strategy.is_trained is True
    assert strategy.last_retrain_time == last_retrain_time
    assert not (tmp_path / "feature_manifest.json").exists()


def test_single_class_retraining_never_activates_or_persists_model(
    tmp_path: Path,
) -> None:
    import trading.strategies.research_based_strategy as strategy_module

    strategy = make_strategy(tmp_path)
    original_model = strategy.rf_model
    original_scaler = strategy.feature_scaler
    market_history = pd.DataFrame({"close": np.arange(120, dtype=float)})

    with (
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value=financial_feature_values(),
        ),
        patch.object(strategy_module, "cross_val_score", return_value=np.full(10, 0.5)),
    ):
        score = strategy.train_model(market_history)

    assert score == 0.0
    assert strategy.rf_model is original_model
    assert strategy.feature_scaler is original_scaler
    assert strategy.is_trained is False
    assert strategy.last_retrain_time is None
    assert not (tmp_path / "feature_manifest.json").exists()
