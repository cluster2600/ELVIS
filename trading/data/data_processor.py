import pandas as pd
import ta


class DataProcessor:
    def __init__(self, exchange, feature_config, quality_config, logger):
        self.exchange = exchange
        self.feature_config = feature_config
        self.quality_config = quality_config
        self.logger = logger

    def get_latest_data(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch the latest market data incrementally for real-time trading.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for data (e.g., '1m')
            limit: Number of recent candles to fetch

        Returns:
            DataFrame with enhanced market data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Add market regime features
            if self.feature_config.market_regime_features:
                df = self.add_market_regime_features(df)

            # Add order book features
            if self.feature_config.orderbook_features:
                orderbook = self.exchange.fetch_order_book(symbol)
                df["bid"] = (
                    orderbook["bids"][0][0]
                    if orderbook["bids"]
                    else df["close"].iloc[-1]
                )
                df["ask"] = (
                    orderbook["asks"][0][0]
                    if orderbook["asks"]
                    else df["close"].iloc[-1]
                )
                df["spread"] = df["ask"] - df["bid"]

            # Add funding rate
            if self.feature_config.funding_features:
                funding = self.exchange.fetch_funding_rate(symbol)
                df["funding_rate"] = funding["fundingRate"]

            # Handle missing data
            if self.quality_config.handle_missing_data:
                df = self.handle_missing_data(df)

            self.logger.debug(f"Latest data fetched and processed: {df.tail(1)}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching latest data: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on failure

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the price data."""
        try:
            if len(data) < 50:  # Need enough data for indicators
                return data

            # Add Simple Moving Averages
            data["sma_20"] = ta.trend.sma_indicator(data["close"], window=20)
            data["sma_50"] = ta.trend.sma_indicator(data["close"], window=50)

            # Add ADX
            data["adx"] = ta.trend.adx(
                data["high"], data["low"], data["close"], window=14
            )

            # Add RSI
            data["rsi"] = ta.momentum.rsi(data["close"], window=14)

            # Add MACD
            macd = ta.trend.MACD(data["close"])
            data["macd"] = macd.macd()
            data["macd_signal"] = macd.macd_signal()

            # Add Bollinger Bands (with both naming conventions)
            bollinger = ta.volatility.BollingerBands(data["close"])
            data["bb_low"] = bollinger.bollinger_lband()
            data["bb_mid"] = bollinger.bollinger_mavg()
            data["bb_high"] = bollinger.bollinger_hband()

            # Add mean reversion strategy compatible names
            data["lowerband"] = data["bb_low"]
            data["middleband"] = data["bb_mid"]
            data["upperband"] = data["bb_high"]

            # Add other indicators that might be needed
            data["atr"] = ta.volatility.average_true_range(
                data["high"], data["low"], data["close"]
            )

            return data
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return data

    def add_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification features."""
        try:
            # Add volume moving average
            data["volume_ma"] = data["volume"].rolling(window=20).mean()

            # Add volatility measure
            data["volatility"] = data["close"].pct_change().rolling(window=20).std()

            return data
        except Exception as e:
            self.logger.error(f"Error adding market regime features: {e}")
            return data

    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in the DataFrame."""
        try:
            # Forward fill missing values (updated to avoid deprecation warning)
            data = data.ffill()

            # If still missing, fill with 0
            data = data.fillna(0)

            return data
        except Exception as e:
            self.logger.error(f"Error handling missing data: {e}")
            return data
