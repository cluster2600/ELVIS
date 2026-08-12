"""Per-symbol market-frame isolation and main-loop wiring tests."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import trading.data.market_frames as market_frames
from trading.risk.position_sizing import volume_multiplier

MAIN_PATH = Path(__file__).parents[1] / "main.py"
REQUIRED_INDICATORS = {
    "sma_20",
    "sma_50",
    "adx",
    "rsi",
    "macd",
    "signal_line",
    "macd_histogram",
    "lower_bb",
    "sma_bb",
    "upper_bb",
    "atr",
}


def _ohlcv(start: float, stop: float, last_volume: float) -> pd.DataFrame:
    close = np.linspace(start, stop, 60)
    volume = np.full(60, 10.0)
    volume[-1] = last_volume
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        }
    )


def _primary_symbol_loop(tree: ast.AST) -> ast.For:
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "symbol"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "symbols_to_trade"
    ]
    assert len(loops) == 1
    return loops[0]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_enrich_symbol_frames_isolates_symbols_and_preserves_inputs() -> None:
    raw = {
        "BTCUSDT": _ohlcv(100.0, 160.0, last_volume=100.0),
        "BNBUSDT": _ohlcv(600.0, 540.0, last_volume=1.0),
    }
    before = {symbol: frame.copy(deep=True) for symbol, frame in raw.items()}

    enriched = market_frames.enrich_symbol_frames(raw)

    assert tuple(enriched) == ("BTCUSDT", "BNBUSDT")
    for symbol in raw:
        assert enriched[symbol] is not raw[symbol]
        assert_frame_equal(raw[symbol], before[symbol])
        assert REQUIRED_INDICATORS.issubset(enriched[symbol].columns)
    assert enriched["BTCUSDT"].iloc[-1]["rsi"] > 70.0
    assert enriched["BNBUSDT"].iloc[-1]["rsi"] < 30.0
    assert volume_multiplier(enriched["BTCUSDT"]) == 2.0
    assert volume_multiplier(enriched["BNBUSDT"]) == 0.5

    enriched["BTCUSDT"].loc[0, "close"] = -1.0
    assert raw["BTCUSDT"].loc[0, "close"] == before["BTCUSDT"].loc[0, "close"]
    assert enriched["BNBUSDT"].loc[0, "close"] != -1.0


def test_enrichment_is_independent_of_symbol_iteration_order() -> None:
    raw = {
        "BTCUSDT": _ohlcv(100.0, 160.0, last_volume=100.0),
        "BNBUSDT": _ohlcv(600.0, 540.0, last_volume=1.0),
    }

    forward = market_frames.enrich_symbol_frames(raw)
    reverse = market_frames.enrich_symbol_frames(dict(reversed(tuple(raw.items()))))

    for symbol in raw:
        assert_frame_equal(forward[symbol], reverse[symbol])


@pytest.mark.parametrize("failure", ["raise", "partial"])
def test_one_bad_symbol_cannot_leak_a_partial_frame(monkeypatch, failure: str) -> None:
    raw = {
        "BTCUSDT": _ohlcv(100.0, 160.0, last_volume=100.0),
        "BNBUSDT": _ohlcv(600.0, 540.0, last_volume=1.0),
    }
    before = {symbol: frame.copy(deep=True) for symbol, frame in raw.items()}
    real_enricher = market_frames.add_technical_indicators

    def controlled_enricher(frame, logger=None):
        if frame.iloc[0]["close"] > 500.0:
            if failure == "raise":
                raise RuntimeError("synthetic enrichment failure")
            frame["rsi"] = 50.0
            return frame
        return real_enricher(frame, logger)

    monkeypatch.setattr(market_frames, "add_technical_indicators", controlled_enricher)

    enriched = market_frames.enrich_symbol_frames(raw)

    assert tuple(enriched) == ("BTCUSDT",)
    assert_frame_equal(raw["BTCUSDT"], before["BTCUSDT"])
    assert_frame_equal(raw["BNBUSDT"], before["BNBUSDT"])


def test_short_or_incomplete_frames_are_not_tradeable() -> None:
    frames = {
        "SHORT": _ohlcv(100.0, 120.0, last_volume=10.0).head(20),
        "MISSING": pd.DataFrame({"close": np.linspace(100.0, 120.0, 60)}),
    }

    assert market_frames.enrich_symbol_frames(frames) == {}


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("close", np.nan),
        ("high", np.inf),
        ("low", None),
        ("volume", np.nan),
        ("volume", np.inf),
    ],
)
def test_non_finite_latest_market_observation_omits_only_that_symbol(
    column: str, invalid_value: object
) -> None:
    healthy = _ohlcv(100.0, 160.0, last_volume=100.0)
    invalid = _ohlcv(600.0, 540.0, last_volume=1.0)
    invalid.loc[invalid.index[-1], column] = invalid_value

    enriched = market_frames.enrich_symbol_frames(
        {"BTCUSDT": healthy, "BNBUSDT": invalid}
    )

    assert tuple(enriched) == ("BTCUSDT",)


def test_missing_volume_column_is_not_tradeable() -> None:
    frame = _ohlcv(100.0, 160.0, last_volume=100.0).drop(columns="volume")

    assert market_frames.enrich_symbol_frames({"BTCUSDT": frame}) == {}


@pytest.mark.parametrize("column", ["close", "macd_histogram"])
def test_non_finite_penultimate_divergence_input_omits_only_that_symbol(
    monkeypatch: pytest.MonkeyPatch, column: str
) -> None:
    raw = {
        "BTCUSDT": _ohlcv(100.0, 160.0, last_volume=100.0),
        "BNBUSDT": _ohlcv(600.0, 540.0, last_volume=1.0),
    }
    real_enricher = market_frames.add_technical_indicators

    def invalidate_penultimate_observation(frame, logger=None):
        candidate = real_enricher(frame, logger)
        if candidate.iloc[0]["close"] > 500.0:
            candidate.loc[candidate.index[-2], column] = np.nan
        return candidate

    monkeypatch.setattr(
        market_frames,
        "add_technical_indicators",
        invalidate_penultimate_observation,
    )

    enriched = market_frames.enrich_symbol_frames(raw)

    assert tuple(enriched) == ("BTCUSDT",)


def test_main_symbol_loop_uses_only_its_symbol_history() -> None:
    tree = ast.parse(MAIN_PATH.read_text())
    symbol_loop = _primary_symbol_loop(tree)

    global_data_reads = [
        node
        for node in ast.walk(symbol_loop)
        if isinstance(node, ast.Name)
        and node.id == "data"
        and isinstance(node.ctx, ast.Load)
    ]
    assert global_data_reads == []

    symbol_data_assignments = [
        node
        for node in ast.walk(symbol_loop)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "symbol_data"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "all_data"
        and node.value.func.attr == "get"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "symbol"
    ]
    assert len(symbol_data_assignments) == 1

    history_assignments = [
        node
        for node in ast.walk(symbol_loop)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "symbol_history"
            for target in node.targets
        )
    ]
    assert len(history_assignments) == 1
    history_call = history_assignments[0].value
    assert isinstance(history_call, ast.Call)
    assert isinstance(history_call.func, ast.Attribute)
    assert history_call.func.attr == "copy"
    tail_call = history_call.func.value
    assert isinstance(tail_call, ast.Call)
    assert isinstance(tail_call.func, ast.Attribute)
    assert isinstance(tail_call.func.value, ast.Name)
    assert tail_call.func.value.id == "symbol_data"
    assert tail_call.func.attr == "tail"
    assert len(tail_call.args) == 1
    assert isinstance(tail_call.args[0], ast.Constant)
    assert tail_call.args[0].value == 100

    calls = [node for node in ast.walk(symbol_loop) if isinstance(node, ast.Call)]
    for consumer in (
        "analyze_signal_quality",
        "detect_current_regime",
        "apply_signal_filters",
        "volume_multiplier",
    ):
        matching = [call for call in calls if _call_name(call) == consumer]
        assert len(matching) == 1
        assert any(
            isinstance(argument, ast.Name) and argument.id == "symbol_history"
            for argument in matching[0].args
        )


def test_main_resets_regime_context_before_per_symbol_consumers() -> None:
    tree = ast.parse(MAIN_PATH.read_text())
    symbol_loop = _primary_symbol_loop(tree)
    calls = [node for node in ast.walk(symbol_loop) if isinstance(node, ast.Call)]
    consumer_lines = [
        call.lineno
        for call in calls
        if _call_name(call) in {"analyze_signal_quality", "detect_current_regime"}
    ]

    assert len(symbol_loop.body) == 1
    symbol_try = symbol_loop.body[0]
    assert isinstance(symbol_try, ast.Try)

    reset_lines: dict[str, int] = {}
    for node in symbol_try.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "filter_result",
                    "regime_result",
                }:
                    reset_lines[target.id] = node.lineno

    assert set(reset_lines) == {"filter_result", "regime_result"}
    assert max(reset_lines.values()) < min(consumer_lines)
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "locals" for call in calls
    )

    regime_guards = [
        node
        for node in ast.walk(symbol_loop)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and {
            comparison.left.id
            for comparison in node.test.values
            if isinstance(comparison, ast.Compare)
            and isinstance(comparison.left, ast.Name)
            and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.IsNot)
            and len(comparison.comparators) == 1
            and isinstance(comparison.comparators[0], ast.Constant)
            and comparison.comparators[0].value is None
        }
        == {"filter_result", "regime_result"}
    ]
    assert len(regime_guards) == 1


def test_main_enriches_the_complete_symbol_mapping_before_processing() -> None:
    tree = ast.parse(MAIN_PATH.read_text())
    symbol_loop = _primary_symbol_loop(tree)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "all_data"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "enrich_symbol_frames"
    ]

    assert len(assignments) == 1
    assert assignments[0].lineno < symbol_loop.lineno
    assert [
        argument.id
        for argument in assignments[0].value.args
        if isinstance(argument, ast.Name)
    ] == ["all_data", "logger"]
