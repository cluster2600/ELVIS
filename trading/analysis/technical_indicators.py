import logging

import pandas as pd
import ta


def add_technical_indicators(
    data: pd.DataFrame, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """
    Add technical indicators to price data.
    """
    try:
        if len(data) < 50:
            return data

        required_columns = ["close", "high", "low"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            if logger:
                logger.error(
                    f"Missing required columns for technical indicators: {missing_columns}"
                )
                logger.info(f"Available columns: {list(data.columns)}")
            return data

        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data["sma_20"] = ta.trend.sma_indicator(data["close"], window=20)
        data["sma_50"] = ta.trend.sma_indicator(data["close"], window=50)
        data["adx"] = ta.trend.adx(data["high"], data["low"], data["close"], window=14)
        data["rsi"] = ta.momentum.rsi(data["close"], window=14)

        macd = ta.trend.MACD(data["close"])
        data["macd"] = macd.macd()
        data["signal_line"] = macd.macd_signal()

        bollinger = ta.volatility.BollingerBands(data["close"])
        data["lower_bb"] = bollinger.bollinger_lband()
        data["sma_bb"] = bollinger.bollinger_mavg()
        data["upper_bb"] = bollinger.bollinger_hband()
        data["atr"] = ta.volatility.average_true_range(
            data["high"], data["low"], data["close"]
        )

        return data
    except Exception as exc:
        if logger:
            logger.error(f"Error calculating technical indicators: {exc}")
        return data
