"""Regression tests for retired synthetic YDF/CoreML ensemble members."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from trading.data.data_processor import DataProcessor
from trading.strategies.ensemble_strategy import EnsembleStrategy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PLACEHOLDER_FEATURES = {
    "Order_Amount",
    "Filled",
    "Total",
    "future_price",
    "vol_adjusted_price",
    "news_sentiment",
    "social_feature",
    "order_book_depth",
}


def test_ensemble_runtime_exposes_no_retired_model_hooks() -> None:
    source_path = REPOSITORY_ROOT / "trading/strategies/ensemble_strategy.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert {"ydf", "coremltools"}.isdisjoint(imported_roots)
    assert "ydf_model_path" not in inspect.signature(EnsembleStrategy).parameters
    assert "coreml_model_path" not in inspect.signature(EnsembleStrategy).parameters
    for retired_hook in (
        "_load_ydf_model",
        "_load_coreml_model",
        "self.ydf_model",
        "self.nn_model",
        'preds["ydf"]',
        'preds["nn"]',
        "REQUIRED_FEATURES",
    ):
        assert retired_hook not in source


def test_injected_retired_models_are_never_invoked() -> None:
    class RetiredModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("retired model was invoked")

    strategy = object.__new__(EnsembleStrategy)
    strategy.logger = MagicMock()
    strategy.CLASSES = ["BUY", "HOLD", "SELL"]
    strategy.trade_learned_model = None
    strategy.mlx_available = False
    strategy.drl_agent = None
    strategy.enable_research_strategy = False
    strategy.research_strategy = None
    strategy.enable_rl_strategy = False
    strategy.rl_strategy = None
    strategy.enable_bonenkamp_hft = False
    strategy.bonenkamp_strategy = None
    retired_ydf = RetiredModel()
    retired_coreml = RetiredModel()
    strategy.ydf_model = retired_ydf
    strategy.nn_model = retired_coreml

    predictions = strategy._get_model_predictions(
        {
            "price": 101.0,
            "volume": 12.0,
            "rsi": 50.0,
            "macd": 0.0,
            "signal_line": 0.0,
            "sma": 100.0,
        }
    )

    assert retired_ydf.calls == 0
    assert retired_coreml.calls == 0
    assert set(predictions) == {"technical"}


def test_dataframe_feature_adapter_does_not_invent_model_inputs() -> None:
    strategy = object.__new__(EnsembleStrategy)
    frame = pd.DataFrame(
        [
            {
                "close": 101.0,
                "volume": 12.0,
                "rsi": 48.0,
                "macd": 1.5,
                "macd_signal": 1.25,
                "sma_20": 99.0,
                "volume_ma": 10.5,
                "bb_low": 90.0,
                "bb_mid": 100.0,
                "bb_high": 110.0,
            }
        ]
    )

    features = strategy._create_features_from_data(frame)

    assert features["price"] == 101.0
    assert features["signal_line"] == 1.25
    assert features["lower_bb"] == 90.0
    assert features["sma_bb"] == 100.0
    assert features["upper_bb"] == 110.0
    assert features["volume_ma"] == 10.5
    assert LEGACY_PLACEHOLDER_FEATURES.isdisjoint(features)

    without_rolling_mean = strategy._create_features_from_data(
        frame.drop(columns="volume_ma")
    )
    assert "volume_ma" not in without_rolling_mean


def test_data_processor_does_not_backfill_retired_model_columns() -> None:
    class Exchange:
        @staticmethod
        def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
            del symbol, timeframe, limit
            return [[1_700_000_000_000, 100.0, 102.0, 99.0, 101.0, 12.0]]

    processor = DataProcessor(
        Exchange(),
        SimpleNamespace(
            market_regime_features=False,
            orderbook_features=False,
            funding_features=False,
        ),
        SimpleNamespace(handle_missing_data=False),
        SimpleNamespace(
            debug=lambda *_args, **_kwargs: None, error=lambda *_args: None
        ),
    )
    processor.add_technical_indicators = lambda frame: frame

    frame = processor.get_latest_data("BTCUSDT", limit=1)

    assert frame.iloc[-1]["close"] == 101.0
    assert LEGACY_PLACEHOLDER_FEATURES.isdisjoint(frame.columns)
    assert "volume_ma" not in frame.columns


def test_synthetic_assets_and_dependencies_are_not_shipped() -> None:
    from trading.models import feature_schemas

    assert not (REPOSITORY_ROOT / "models/model_rf_tf").exists()
    assert not (REPOSITORY_ROOT / "scripts/create_coreml_model.py").exists()
    assert not (REPOSITORY_ROOT / "requirements/requirements_ydf.txt").exists()
    assert not (REPOSITORY_ROOT / "requirements/requirements_coreml.txt").exists()
    assert not (REPOSITORY_ROOT / "requirements/requirements_tensorflow.txt").exists()
    training_requirements = REPOSITORY_ROOT / "requirements/requirements_ml310.txt"
    assert training_requirements.is_file()
    requirement_lines = {
        line.strip()
        for line in training_requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    requirement_names = {
        line.split(";", 1)[0].split("==", 1)[0].strip().lower().replace("_", "-")
        for line in requirement_lines
    }

    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    config = (REPOSITORY_ROOT / "config/__init__.py").read_text(encoding="utf-8")
    trainer = (REPOSITORY_ROOT / "docker/Dockerfile.ml310").read_text(encoding="utf-8")

    assert "coremltools" not in pyproject
    assert '"RF_MODEL"' not in config
    assert '"COREML_MODEL"' not in config
    assert "ydf" not in trainer.lower()
    assert "coreml" not in trainer.lower()
    assert "requirements_ml310.txt" in trainer
    assert {"ydf", "coremltools"}.isdisjoint(requirement_names)
    assert "torch==2.10.0+cpu" in requirement_lines
    assert "seaborn==0.13.2" in requirement_lines
    assert "tensorflow==2.16.2" in requirement_lines
    assert not hasattr(feature_schemas, "ENSEMBLE_YDF_20_V1")
    assert not hasattr(feature_schemas, "ENSEMBLE_COREML_20_V1")
