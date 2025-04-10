# trading/data/data_processor.py
# Add this method to your existing DataProcessor class

def get_latest_data(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
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
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Add market regime features
        if self.feature_config.market_regime_features:
            df = self.add_market_regime_features(df)
        
        # Add order book features
        if self.feature_config.orderbook_features:
            orderbook = self.exchange.fetch_order_book(symbol)
            df['bid'] = orderbook['bids'][0][0] if orderbook['bids'] else df['close'].iloc[-1]
            df['ask'] = orderbook['asks'][0][0] if orderbook['asks'] else df['close'].iloc[-1]
            df['spread'] = df['ask'] - df['bid']
        
        # Add funding rate
        if self.feature_config.funding_features:
            funding = self.exchange.fetch_funding_rate(symbol)
            df['funding_rate'] = funding['fundingRate']
        
        # Fill missing EnsembleStrategy features with defaults
        required_features = [
            "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
            "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb", "sma_bb",
            "upper_bb", "news_sentiment", "social_feature", "adx", "rsi", "order_book_depth", "volume"
        ]
        for feature in required_features:
            if feature not in df:
                if feature == "price":
                    df[feature] = df['close']
                elif feature == "sma":
                    df[feature] = df['sma_20']  # From market regime
                elif feature == "atr":
                    df[feature] = df['atr']  # From technical indicators
                elif feature == "macd":
                    df[feature] = df['macd']  # From technical indicators
                elif feature == "signal_line":
                    df[feature] = df['macd_signal']  # From technical indicators
                elif feature == "lower_bb":
                    df[feature] = df['bb_low']  # From Bollinger Bands
                elif feature == "sma_bb":
                    df[feature] = df['bb_mid']  # From Bollinger Bands
                elif feature == "upper_bb":
                    df[feature] = df['bb_high']  # From Bollinger Bands
                elif feature == "rsi":
                    df[feature] = df['rsi']  # From technical indicators
                elif feature == "volume":
                    df[feature] = df['volume']  # From OHLCV
                else:
                    df[feature] = 0.0  # Default for missing features
        
        # Handle missing data
        if self.quality_config.handle_missing_data:
            df = self.handle_missing_data(df)
        
        self.logger.debug(f"Latest data fetched and processed: {df.tail(1)}")
        return df
    
    except Exception as e:
        self.logger.error(f"Error fetching latest data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure