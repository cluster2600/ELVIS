import numpy as np
import pandas as pd
import tensorflow as tf
import ydf
import requests
import coremltools as ct
from trading.strategies.base_strategy import BaseStrategy

class EnsembleStrategy(BaseStrategy):
    def __init__(self, logger, ydf_model_path="/Users/maxime/BTC_BOT/model_rf.ydf", 
                 coreml_model_path="/Users/maxime/BTC_BOT/models/trading_model.mlmodel", 
                 mlx_url="http://192.168.1.165:1234/v1/completions"):
        super().__init__(logger)
        self.logger = logger
        self.REQUIRED_FEATURES = [
            "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
            "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb", "sma_bb",
            "upper_bb", "news_sentiment", "social_feature", "adx", "rsi", "order_book_depth", "volume"
        ]
        self.CLASSES = ["BUY", "HOLD", "SELL"]
        self.mlx_url = mlx_url
        
        # Load models during initialization
        self.ydf_model = self._load_ydf_model(ydf_model_path)
        self.nn_model = self._load_coreml_model(coreml_model_path)
        self._check_mlx_connectivity()

    def _load_ydf_model(self, model_path):
        """Load the YDF model."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YDF model file not found at {model_path}")
            ydf_model = ydf.from_tensorflow_decision_forests(model_path)
            self.logger.info(f"YDF model loaded from {model_path}")
            return ydf_model
        except Exception as e:
            self.logger.error(f"Failed to load YDF model: {e}")
            raise

    def _load_coreml_model(self, model_path):
        """Load the Core ML model."""
        try:
            nn_model = ct.models.MLModel(model_path)
            self.logger.info(f"Core ML model loaded from {model_path}")
            return nn_model
        except Exception as e:
            self.logger.error(f"Failed to load Core ML model: {e}")
            raise

    def _check_mlx_connectivity(self):
        """Check connectivity to MLX model."""
        try:
            response = requests.get(f"{self.mlx_url.split('/v1/')[0]}/v1/models", timeout=5)
            response.raise_for_status()
            self.logger.info(f"MLX model connected at {self.mlx_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to MLX server: {e}")
            raise

    def _ensure_required_features(self, features: dict) -> dict:
        """Ensure all required features are present."""
        return {key: features.get(key, 0.0) for key in self.REQUIRED_FEATURES}

    def _mlx_generate(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate trading decision using MLX model."""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
            response = requests.post(self.mlx_url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            self.logger.warning(f"MLX generation failed: {e}")
            return "HOLD"

    def _parse_mlx_decision(self, output: str) -> str:
        """Extract trading decision from MLX output."""
        output = output.upper()
        for word in output.split():
            if word in self.CLASSES:
                return word
        return "HOLD"

    def generate_signals(self, data):
        """Generate trading signals using the ensemble model."""
        features = self._ensure_required_features(data)

        # YDF prediction
        try:
            ydf_input = pd.DataFrame([features])
            ydf_pred = self.ydf_model.predict(ydf_input)
            ydf_probs = ydf_pred[0] if isinstance(ydf_pred[0], np.ndarray) else np.array([0.0, 1.0, 0.0])
        except Exception as e:
            self.logger.warning(f"YDF prediction failed: {e}")
            ydf_probs = np.array([0.0, 1.0, 0.0])

        # Core ML NN prediction
        try:
            nn_input = np.array([[features[col] for col in self.REQUIRED_FEATURES]], dtype=np.float32)
            nn_pred = self.nn_model.predict({"input": nn_input})
            probs_dict = nn_pred.get('classLabel_probs', nn_pred.get('classProbability', {}))
            nn_probs = np.array([probs_dict.get(cls, 0.0) for cls in self.CLASSES], dtype=np.float32)
            if nn_probs.sum() == 0:
                nn_probs = np.array([0.0, 1.0, 0.0])
        except Exception as e:
            self.logger.warning(f"Core ML prediction failed: {e}")
            nn_probs = np.array([0.0, 1.0, 0.0])

        # MLX prediction
        prompt = f"Market features: {', '.join(f'{k}: {v}' for k, v in features.items())}. Recommend: BUY, SELL, or HOLD."
        mlx_output = self._mlx_generate(prompt)
        mlx_decision = self._parse_mlx_decision(mlx_output)
        mlx_probs = np.array([1.0 if cls == mlx_decision else 0.0 for cls in self.CLASSES], dtype=np.float32)

        # Ensemble
        avg_probs = np.mean([ydf_probs, nn_probs, mlx_probs], axis=0)
        decision_idx = np.argmax(avg_probs)
        decision = self.CLASSES[decision_idx]
        confidence = avg_probs[decision_idx]

        self.logger.debug(f"YDF: {ydf_probs}, NN: {nn_probs}, MLX: {mlx_probs}, Avg: {avg_probs}")
        self.logger.info(f"Ensemble decision: {decision}, Confidence: {confidence:.4f}")
        
        return {"signal": decision, "confidence": float(confidence)}