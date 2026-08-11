import pandas as pd
import pytest

from trading.analysis.technical_indicators import add_technical_indicators


def test_add_technical_indicators_with_missing_columns_returns_input():
    data = pd.DataFrame({"close": [1.0] * 60})
    result = add_technical_indicators(data.copy())
    assert "sma_20" not in result.columns
    assert list(result.columns) == ["close"]


def test_add_technical_indicators_adds_expected_columns():
    rows = 80
    data = pd.DataFrame(
        {
            "close": [100 + i * 0.1 for i in range(rows)],
            "high": [101 + i * 0.1 for i in range(rows)],
            "low": [99 + i * 0.1 for i in range(rows)],
        }
    )

    result = add_technical_indicators(data.copy())

    expected_columns = {
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
    assert expected_columns.issubset(set(result.columns))
    latest = result.dropna(subset=["macd", "signal_line", "macd_histogram"]).iloc[-1]
    assert latest["macd_histogram"] == pytest.approx(
        latest["macd"] - latest["signal_line"]
    )
