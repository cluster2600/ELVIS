import numpy as np
import pandas as pd
import os
import requests
import coremltools as ct
# import ydf
import logging
from typing import Dict, Any
from trading.strategies.base_strategy import BaseStrategy

class EnsembleStrategy(BaseStrategy):
    """
    EnsembleStrategy combines predictions from multiple models:
    - YDF Random Forest model
    - CoreML Neural Network model
    - (optional) MLX Large Language Model for additional decision support
    
    This strategy averages model outputs to determine a consensus BUY, SELL, or HOLD signal.
    """

    def __init__(self, logger: logging.Logger, 
                 ydf_model_path: str = "/Users/maxime/BTC_BOT/BTC_BOT/model_rf.ydf",
                 coreml_model_path: str = "/Users/maxime/BTC_BOT/BTC_BOT/NNModel.mlpackage",
                 mlx_url: str = None):
        """
        Initialize the ensemble strategy, loading models and setting parameters.

        Args:
            logger (logging.Logger): The logger for debugging/info output.
            ydf_model_path (str): Path to the YDF Random Forest model.
            coreml_model_path (str): Path to the CoreML Neural Network model.
            mlx_url (str, optional): URL to MLX server for LLM support.
        """
        super().__init__(logger)
        self.logger = logger
        self.REQUIRED_FEATURES = [
            "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
            "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb",
            "sma_bb", "upper_bb", "news_sentiment", "social_feature", "adx", "rsi",
            "order_book_depth", "volume"
        ]
        self.CLASSES = ["BUY", "HOLD", "SELL"]

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

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a trading signal based on ensemble voting.

        Args:
            data (pd.DataFrame): The latest market data.

        Returns:
            Dict[str, Any]: {'signal': 'BUY'/'SELL'/'HOLD', 'confidence': float}
        """
        if data.empty:
            return {"signal": "HOLD", "confidence": 0.0}

        features = data.iloc[-1].to_dict()
        features = {k: features.get(k, 0.0) for k in self.REQUIRED_FEATURES}

        preds = self._get_model_predictions(features)

        pred_array = np.mean([p for p in preds.values() if p is not None], axis=0)
        best_idx = np.argmax(pred_array)

        decision = self.CLASSES[best_idx]
        confidence = float(pred_array[best_idx])

        self.logger.info(f"Ensemble decision: {decision} ({confidence:.4f})")

        return {"signal": decision, "confidence": confidence}

    def calculate_position_size(self, portfolio_value: float, price: float, volatility: float) -> float:
        """
        Basic risk-based position sizing: 1% of portfolio at risk.

        Args:
            portfolio_value (float): Total portfolio value.
            price (float): Current asset price.
            volatility (float): Estimated volatility (for size adjustment).

        Returns:
            float: Size of the position.
        """
        risk_amount = portfolio_value * 0.01
        position_size = risk_amount / (price * volatility)
        return max(position_size, 0.001)  # minimum threshold
