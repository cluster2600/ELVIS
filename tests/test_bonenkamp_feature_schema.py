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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from trading.models.artifact_manifest import ArtifactDescriptor, write_feature_manifest
from trading.models.feature_schema import FeatureContractError
from trading.models.feature_schemas import (
    BONENKAMP_FINANCIAL_9_V1,
    BONENKAMP_SOCIAL_11_V1,
)

DESCRIPTOR = ArtifactDescriptor(
    model_kind="random-forest-classifier",
    library="scikit-learn",
    library_version=sklearn.__version__,
)


def make_strategy(tmp_path: Path, *, social: bool = False):
    from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy

    with patch.object(BonenkampHFTStrategy, "load_model", return_value=False):
        return BonenkampHFTStrategy(
            logging.getLogger("bonenkamp-feature-schema-test"),
            use_social_features=social,
            model_save_path=str(tmp_path),
        )


def financial_feature_values() -> dict[str, float]:
    return {
        name: float(index + 1)
        for index, name in enumerate(BONENKAMP_FINANCIAL_9_V1.names)
    }


def social_feature_values() -> dict[str, float]:
    return {"TWITTER_PRICE_LAG": 10.0, "GOOGLE_TRENDS": 11.0}


def alternating_history(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame({"close": np.resize(np.array([100.0, 101.0, 100.0]), rows)})


@pytest.mark.parametrize("social", [False, True])
def test_bonenkamp_strategy_selects_and_vectorizes_explicit_schema(
    tmp_path: Path, social: bool
) -> None:
    strategy = make_strategy(tmp_path, social=social)
    expected = BONENKAMP_SOCIAL_11_V1 if social else BONENKAMP_FINANCIAL_9_V1

    with (
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value=financial_feature_values(),
        ),
        patch.object(
            strategy, "collect_social_features", return_value=social_feature_values()
        ),
    ):
        features = strategy.prepare_feature_vector(pd.DataFrame())

    assert strategy.feature_schema is expected
    assert tuple(strategy._get_feature_names()) == expected.names
    assert features.shape == (1, expected.size)
    assert tuple(features[0]) == expected.vectorize(
        {**financial_feature_values(), **social_feature_values()}
    )


def test_bonenkamp_loader_rejects_incompatible_manifest_before_joblib(
    tmp_path: Path,
) -> None:
    from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy

    model_path = tmp_path / "bonenkamp_rf_model.pkl"
    scaler_path = tmp_path / "bonenkamp_scaler.pkl"
    model_path.write_bytes(b"not-a-pickle")
    scaler_path.write_bytes(b"not-a-pickle")
    write_feature_manifest(
        tmp_path / "feature_manifest.json",
        BONENKAMP_SOCIAL_11_V1,
        DESCRIPTOR,
        {"model": model_path, "scaler": scaler_path},
    )

    with patch("trading.strategies.bonenkamp_hft_strategy.joblib.load") as load:
        strategy = BonenkampHFTStrategy(
            logging.getLogger("bonenkamp-feature-schema-test"),
            use_social_features=False,
            model_save_path=str(tmp_path),
        )

    load.assert_not_called()
    assert strategy.is_trained is False


def test_bonenkamp_loader_rejects_missing_manifest_before_joblib(
    tmp_path: Path,
) -> None:
    from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy

    (tmp_path / "bonenkamp_rf_model.pkl").write_bytes(b"not-a-pickle")
    (tmp_path / "bonenkamp_scaler.pkl").write_bytes(b"not-a-pickle")

    with patch("trading.strategies.bonenkamp_hft_strategy.joblib.load") as load:
        strategy = BonenkampHFTStrategy(
            logging.getLogger("bonenkamp-feature-schema-test"),
            use_social_features=False,
            model_save_path=str(tmp_path),
        )

    load.assert_not_called()
    assert strategy.is_trained is False


def test_bonenkamp_scaler_mismatch_never_returns_unscaled_features(
    tmp_path: Path,
) -> None:
    strategy = make_strategy(tmp_path)
    strategy.feature_scaler = StandardScaler().fit(np.ones((2, 11)))

    with patch.object(
        strategy,
        "calculate_financial_indicators",
        return_value=financial_feature_values(),
    ):
        with pytest.raises(FeatureContractError, match="cannot produce"):
            strategy.prepare_feature_vector(pd.DataFrame())


def test_trained_bonenkamp_never_predicts_with_unfitted_scaler(
    tmp_path: Path,
) -> None:
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
    assert signal["signal"] == "HOLD"


def test_untrained_bonenkamp_does_not_prepare_model_features(tmp_path: Path) -> None:
    strategy = make_strategy(tmp_path)
    market_data = pd.DataFrame({"close": [100.0]})

    with (
        patch.object(strategy, "prepare_feature_vector") as prepare,
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value={"RSI": 50.0},
        ),
    ):
        signal = strategy.generate_signals({"BTCUSDT": market_data})["BTCUSDT"]

    prepare.assert_not_called()
    assert signal["signal"] == "HOLD"


def test_bonenkamp_artifact_round_trip_loads_during_construction(
    tmp_path: Path,
) -> None:
    from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy

    source = make_strategy(tmp_path)
    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    source.feature_scaler = StandardScaler().fit(training)
    source.rf_model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        source.feature_scaler.transform(training), labels
    )
    source.is_trained = True
    source._save_model()

    restored = BonenkampHFTStrategy(
        logging.getLogger("bonenkamp-feature-schema-test"),
        use_social_features=False,
        model_save_path=str(tmp_path),
    )

    assert restored.is_trained is True
    assert restored.feature_schema is BONENKAMP_FINANCIAL_9_V1
    prediction = restored.rf_model.predict(
        restored.feature_scaler.transform(training[[0]])
    )
    assert prediction.shape == (1,)


def test_bonenkamp_loader_does_not_partially_assign_invalid_scaler(
    tmp_path: Path,
) -> None:
    from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy

    training = np.arange(180, dtype=float).reshape(20, 9)
    model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        training, np.array([0, 1] * 10)
    )
    scaler = StandardScaler().fit(np.ones((2, 11)))
    model_path = tmp_path / "bonenkamp_rf_model.pkl"
    scaler_path = tmp_path / "bonenkamp_scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    write_feature_manifest(
        tmp_path / "feature_manifest.json",
        BONENKAMP_FINANCIAL_9_V1,
        DESCRIPTOR,
        {"model": model_path, "scaler": scaler_path},
    )

    restored = BonenkampHFTStrategy(
        logging.getLogger("bonenkamp-feature-schema-test"),
        use_social_features=False,
        model_save_path=str(tmp_path),
    )

    assert restored.is_trained is False
    assert not hasattr(restored.rf_model, "n_features_in_")
    assert not hasattr(restored.feature_scaler, "n_features_in_")


def test_bonenkamp_loader_rejects_single_class_model(tmp_path: Path) -> None:
    from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy

    training = np.arange(180, dtype=float).reshape(20, 9)
    scaler = StandardScaler().fit(training)
    model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        scaler.transform(training), np.ones(20, dtype=int)
    )
    model_path = tmp_path / "bonenkamp_rf_model.pkl"
    scaler_path = tmp_path / "bonenkamp_scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    write_feature_manifest(
        tmp_path / "feature_manifest.json",
        BONENKAMP_FINANCIAL_9_V1,
        DESCRIPTOR,
        {"model": model_path, "scaler": scaler_path},
    )

    restored = BonenkampHFTStrategy(
        logging.getLogger("bonenkamp-feature-schema-test"),
        use_social_features=False,
        model_save_path=str(tmp_path),
    )

    assert restored.is_trained is False
    assert not hasattr(restored.rf_model, "n_features_in_")
    assert not hasattr(restored.feature_scaler, "n_features_in_")


@pytest.mark.parametrize("wrong_component", ["model", "scaler"])
def test_bonenkamp_loader_rejects_wrong_implementation(
    tmp_path: Path, wrong_component: str
) -> None:
    from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy

    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    model = RandomForestClassifier(n_estimators=2, random_state=3).fit(training, labels)
    scaler = StandardScaler().fit(training)
    if wrong_component == "model":
        model = LogisticRegression().fit(training, labels)
    else:
        scaler = MinMaxScaler().fit(training)
    model_path = tmp_path / "bonenkamp_rf_model.pkl"
    scaler_path = tmp_path / "bonenkamp_scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    write_feature_manifest(
        tmp_path / "feature_manifest.json",
        BONENKAMP_FINANCIAL_9_V1,
        DESCRIPTOR,
        {"model": model_path, "scaler": scaler_path},
    )

    restored = BonenkampHFTStrategy(
        logging.getLogger("bonenkamp-feature-schema-test"),
        use_social_features=False,
        model_save_path=str(tmp_path),
    )

    assert restored.is_trained is False
    assert isinstance(restored.rf_model, RandomForestClassifier)
    assert isinstance(restored.feature_scaler, StandardScaler)


def test_failed_bonenkamp_retraining_preserves_active_pair(tmp_path: Path) -> None:
    import trading.strategies.bonenkamp_hft_strategy as strategy_module

    strategy = make_strategy(tmp_path)
    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    active_scaler = StandardScaler().fit(training)
    active_model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        active_scaler.transform(training), labels
    )
    last_training_time = datetime(2025, 1, 2, 3, 4, 5)
    original_mean = active_scaler.mean_.copy()
    strategy.feature_scaler = active_scaler
    strategy.rf_model = active_model
    strategy.is_trained = True
    strategy.last_training_time = last_training_time

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
        score = strategy.train_model(alternating_history())

    assert score == 0.0
    assert strategy.rf_model is active_model
    assert strategy.feature_scaler is active_scaler
    np.testing.assert_array_equal(strategy.feature_scaler.mean_, original_mean)
    assert strategy.is_trained is True
    assert strategy.last_training_time == last_training_time
    assert not (tmp_path / "feature_manifest.json").exists()


def test_failed_bonenkamp_persistence_preserves_all_active_state(
    tmp_path: Path,
) -> None:
    import trading.strategies.bonenkamp_hft_strategy as strategy_module

    strategy = make_strategy(tmp_path)
    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    active_scaler = StandardScaler().fit(training)
    active_model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        active_scaler.transform(training), labels
    )
    last_training_time = datetime(2025, 1, 2, 3, 4, 5)
    strategy.feature_scaler = active_scaler
    strategy.rf_model = active_model
    strategy.is_trained = True
    strategy.last_training_time = last_training_time
    strategy.f1_scores = [0.25]

    with (
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value=financial_feature_values(),
        ),
        patch.object(strategy_module, "cross_val_score", return_value=np.full(10, 0.5)),
        patch.object(strategy, "_save_model", side_effect=OSError("disk full")),
    ):
        score = strategy.train_model(alternating_history())

    assert score == 0.0
    assert strategy.rf_model is active_model
    assert strategy.feature_scaler is active_scaler
    assert strategy.is_trained is True
    assert strategy.last_training_time == last_training_time
    assert strategy.f1_scores == [0.25]


def test_single_class_bonenkamp_retraining_never_activates_or_persists(
    tmp_path: Path,
) -> None:
    import trading.strategies.bonenkamp_hft_strategy as strategy_module

    strategy = make_strategy(tmp_path)
    original_model = strategy.rf_model
    original_scaler = strategy.feature_scaler
    monotonic_history = pd.DataFrame({"close": np.arange(120, dtype=float)})

    with (
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value=financial_feature_values(),
        ),
        patch.object(strategy_module, "cross_val_score", return_value=np.full(10, 0.5)),
    ):
        score = strategy.train_model(monotonic_history)

    assert score == 0.0
    assert strategy.rf_model is original_model
    assert strategy.feature_scaler is original_scaler
    assert strategy.is_trained is False
    assert strategy.last_training_time is None
    assert strategy.f1_scores == []
    assert not (tmp_path / "feature_manifest.json").exists()


def test_bonenkamp_training_uses_time_series_split(tmp_path: Path) -> None:
    import trading.strategies.bonenkamp_hft_strategy as strategy_module

    strategy = make_strategy(tmp_path)
    strategy.rf_model = RandomForestClassifier(n_estimators=2, random_state=3)
    captured = {}

    def fake_cross_val_score(estimator, X, y, cv=None, **kwargs):
        captured["cv"] = cv
        return np.full(getattr(cv, "n_splits", 1), 0.5)

    with (
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value=financial_feature_values(),
        ),
        patch.object(
            strategy_module, "cross_val_score", side_effect=fake_cross_val_score
        ),
        patch.object(strategy, "_save_model"),
    ):
        score = strategy.train_model(alternating_history())

    assert score == 0.5
    assert isinstance(captured["cv"], TimeSeriesSplit)
    assert captured["cv"].n_splits == strategy.cv_folds
