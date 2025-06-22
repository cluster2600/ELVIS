import numpy as np
import pandas as pd
import os
import requests
import coremltools as ct
# import ydf
import logging
from typing import Dict, Any, List
from trading.strategies.base_strategy import BaseStrategy
import ta

class EnsembleStrategy(BaseStrategy):
    """
    EnsembleStrategy combines predictions from multiple models:
    - YDF Random Forest model
    - CoreML Neural Network model
    - (optional) MLX Large Language Model for additional decision support
    
    This strategy averages model outputs to determine a consensus BUY, SELL, or HOLD signal.
    """

    def __init__(self, logger: logging.Logger, 
                 symbols: List[str] = ['BTCUSDT'],
                 ydf_model_path: str = "/Users/maxime/BTC_BOT/BTC_BOT/model_rf.ydf",
                 coreml_model_path: str = "/Users/maxime/BTC_BOT/BTC_BOT/NNModel.mlpackage",
                 mlx_url: str = None,
                 risk_per_trade: float = 0.01,
                 min_position_size: float = 0.001,
                 max_position_size: float = 0.1,
                 order_flow_analyzer=None,
                 price_fetcher=None):
        """
        Initialize the ensemble strategy, loading models and setting parameters.

        Args:
            logger (logging.Logger): The logger for debugging/info output.
            symbols (List[str]): The trading pairs to manage.
            ydf_model_path (str): Path to the YDF Random Forest model.
            coreml_model_path (str): Path to the CoreML Neural Network model.
            mlx_url (str, optional): URL to MLX server for LLM support.
            risk_per_trade (float): The percentage of the portfolio to risk on a single trade.
            min_position_size (float): The minimum position size in BTC.
            max_position_size (float): The maximum position size in BTC.
            order_flow_analyzer: The order flow analyzer instance.
            price_fetcher: The price fetcher instance.
        """
        super().__init__(logger)
        self.logger = logger
        self.symbols = symbols
        self.order_flow_analyzer = order_flow_analyzer
        self.price_fetcher = price_fetcher
        self.REQUIRED_FEATURES = [
            "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
            "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb",
            "sma_bb", "upper_bb", "news_sentiment", "social_feature", "adx", "rsi",
            "order_book_depth", "volume", "high", "low", "close"
        ]
        self.CLASSES = ["BUY", "HOLD", "SELL"]

        self.risk_per_trade = risk_per_trade
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size

        self.mlx_url = mlx_url or os.getenv('MLX_URL', '')
        self.mlx_available = False

        # Load models
        # self.ydf_model = self._load_ydf_model(ydf_model_path)
        self.ydf_model = None
        self.nn_model = self._load_coreml_model(coreml_model_path)
        self._check_mlx_connectivity()

    # def _load_ydf_model(self, model_path: str):
    #     """Load the YDF model from disk."""
    #     try:
    #         if not os.path.exists(model_path):
    #             raise FileNotFoundError(f"YDF model file not found at {model_path}")
    #         model = ydf.from_tensorflow_decision_forests(model_path)
    #         self.logger.info(f"YDF model loaded from {model_path}")
    #         return model
    #     except Exception as e:
    #         self.logger.error(f"Failed to load YDF model: {e}")
    #         return None

    def _load_coreml_model(self, model_path: str):
        """Load the CoreML model from disk."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"CoreML model file not found at {model_path}")
            model = ct.models.MLModel(model_path)
            self.logger.info(f"CoreML model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load CoreML model: {e}")
            return None

    def _check_mlx_connectivity(self):
        """Check if MLX server is available."""
        if not self.mlx_url:
            return
        try:
            resp = requests.get(f"{self.mlx_url.split('/v1/')[0]}/v1/models", timeout=5)
            resp.raise_for_status()
            self.mlx_available = True
            self.logger.info("MLX server available.")
        except Exception as e:
            self.logger.warning(f"MLX server not available: {e}")

    def _mlx_generate(self, prompt: str) -> str:
        """Generate a decision using MLX server."""
        if not self.mlx_available:
            return "HOLD"
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "llama-3.2-3b-instruct",
                "prompt": prompt,
                "max_tokens": 10
            }
            resp = requests.post(self.mlx_url, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            decision = resp.json()["choices"][0]["text"].strip().upper()
            return decision
        except Exception as e:
            self.logger.warning(f"MLX generation error: {e}")
            return "HOLD"

    def _parse_mlx_decision(self, text: str) -> str:
        """Parse the MLX model text output."""
        for word in text.split():
            if word in self.CLASSES:
                return word
        return "HOLD"

    def _get_model_predictions(self, features: dict) -> Dict[str, np.ndarray]:
        """Predict using YDF, CoreML, and optionally MLX."""
        import subprocess
        import json

        def predict_with_ydf(features: dict) -> dict:
            ydf_env_path = "/path/to/env-ydf/bin/python"  # Update this path to your ydf environment python
            ydf_script_path = "/Users/maxime/BTC_BOT/BTC_BOT/predict_with_ydf.py"
            result = subprocess.run(
                [ydf_env_path, ydf_script_path],
                input=json.dumps(features).encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return json.loads(result.stdout)

        preds = {}

        try:
            output = predict_with_ydf(features)
            if "probabilities" in output:
                preds['ydf'] = np.array(output["probabilities"])
            else:
                self.logger.warning(f"YDF prediction error: {output.get('error', 'Unknown error')}")
                preds['ydf'] = np.array([0.0, 1.0, 0.0])
        except Exception as e:
            self.logger.warning(f"YDF prediction subprocess failed: {e}")
            preds['ydf'] = np.array([0.0, 1.0, 0.0])

        try:
            nn_input = {'features': np.array([[features[col] for col in self.REQUIRED_FEATURES]], dtype=np.float32)}
            nn_pred = self.nn_model.predict(nn_input)
            probs = nn_pred.get('classLabel_probs') or nn_pred.get('classProbability', {})
            preds['nn'] = np.array([probs.get(cls, 0.0) for cls in self.CLASSES])
        except Exception as e:
            self.logger.warning(f"CoreML prediction failed: {e}")
            preds['nn'] = np.array([0.0, 1.0, 0.0])

        if self.mlx_available:
            try:
                mlx_decision = self._parse_mlx_decision(
                    self._mlx_generate(
                        f"Predict market move for features: {features} -> BUY, SELL, or HOLD."
                    )
                )
                preds['mlx'] = np.array([1.0 if c == mlx_decision else 0.0 for c in self.CLASSES])
            except Exception as e:
                self.logger.warning(f"MLX prediction error: {e}")

        return preds

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for all symbols based on ensemble voting.

        Args:
            data (Dict[str, pd.DataFrame]): A dictionary of market data for each symbol.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of signals for each symbol.
        """
        signals = {}
        for symbol in self.symbols:
            if symbol not in data or data[symbol].empty:
                signals[symbol] = {"signal": "HOLD", "confidence": 0.0}
                continue

            features = data[symbol].iloc[-1].to_dict()
            features = {k: features.get(k, 0.0) for k in self.REQUIRED_FEATURES}

            preds = self._get_model_predictions(features)

            pred_array = np.mean([p for p in preds.values() if p is not None], axis=0)
            best_idx = np.argmax(pred_array)

            decision = self.CLASSES[best_idx]
            confidence = float(pred_array[best_idx])

            self.logger.info(f"Ensemble decision for {symbol}: {decision} ({confidence:.4f})")

            signals[symbol] = {"signal": decision, "confidence": confidence}

        return signals

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the Average True Range (ATR) as a measure of volatility.

        Args:
            data (pd.DataFrame): DataFrame with high, low, and close prices.
            period (int): The period over which to calculate the ATR.

        Returns:
            float: The latest ATR value.
        """
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            self.logger.warning("ATR calculation requires 'high', 'low', 'close' columns.")
            return 0.01 # Return a default small volatility
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        tr = np.max([high_low, high_close, low_close], axis=0)
        atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
        return atr if pd.notna(atr) else 0.01

    def calculate_position_size(self, data: pd.DataFrame, current_price: float, available_capital: float) -> float:
        """
        Calculate position size based on volatility (ATR) and risk per trade.
        
        Args:
            data (pd.DataFrame): The data to calculate position size from.
            current_price (float): The current price.
            available_capital (float): The available capital.
            
        Returns:
            float: The position size in BTC.
        """
        atr = self._calculate_atr(data)
        if atr == 0:
            return self.min_position_size

        trend_strength = self._calculate_trend_strength(data)
        
        # Adjust position size based on trend strength
        size_multiplier = 1.0 + (trend_strength - 0.5) # Scale from 0.5 to 1.5

        # Adjust position size based on order flow imbalance
        if self.order_flow_analyzer and self.price_fetcher:
            symbol = getattr(data, 'name', self.symbols[0]) # Fallback to first symbol
            order_book = self.price_fetcher.get_order_book(symbol)
            if order_book:
                bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
                asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
                imbalance = self.order_flow_analyzer.get_order_flow_imbalance(bids, asks)
                size_multiplier += imbalance / 1000 # Small adjustment based on imbalance

        # Dynamic stop loss based on ATR
        stop_loss_distance = atr * 2  # Example: 2 * ATR
        
        # Amount to risk
        risk_amount = available_capital * self.risk_per_trade
        
        # Calculate position size
        position_size = (risk_amount / stop_loss_distance) * size_multiplier
        
        # Clamp the position size to the min/max limits
        position_size = np.clip(position_size, self.min_position_size, self.max_position_size)
        
        # Ensure we don't exceed available capital
        if position_size * current_price > available_capital:
            position_size = available_capital / current_price
            
        return max(position_size, self.min_position_size)

    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the stop loss price based on ATR.
        
        Args:
            data (pd.DataFrame): The data to calculate stop loss from.
            entry_price (float): The entry price.
            
        Returns:
            float: The stop loss price.
        """
        atr = self._calculate_atr(data)
        return entry_price - (atr * 2) # Example: 2 * ATR below entry

    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the take profit price based on ATR.
        
        Args:
            data (pd.DataFrame): The data to calculate take profit from.
            entry_price (float): The entry price.
            
        Returns:
            float: The take profit price.
        """
        atr = self._calculate_atr(data)
        return entry_price + (atr * 3) # Example: 3 * ATR above entry

    def _calculate_trend_strength(self, data: pd.DataFrame, adx_period: int = 14, rsi_period: int = 14) -> float:
        """
        Calculate the trend strength using ADX and RSI.
        
        Returns:
            float: A value between 0 and 1 representing the trend strength.
        """
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=adx_period).adx()
        rsi = ta.momentum.RSIIndicator(data['close'], window=rsi_period).rsi()
        
        # Normalize ADX and RSI to a 0-1 scale
        adx_strength = min(adx.iloc[-1] / 50, 1.0) # ADX > 50 is a strong trend
        rsi_strength = abs(rsi.iloc[-1] - 50) / 50 # RSI further from 50 is a stronger trend
        
        return (adx_strength + rsi_strength) / 2

    def calculate_cross_pair_correlation(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the close prices of all symbols.
        """
        close_prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
        return close_prices.corr()
