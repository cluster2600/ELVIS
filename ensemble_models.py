import numpy as np
import pandas as pd
import tensorflow as tf
import ydf
import requests
import coremltools as ct
import os

# List of required features for the models
REQUIRED_FEATURES = [
    "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
    "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb", "sma_bb",
    "upper_bb", "news_sentiment", "social_feature", "adx", "rsi", "order_book_depth", "volume"
]

CLASSES = ["BUY", "HOLD", "SELL"]

def ensure_required_features(features: dict) -> dict:
    """Ensure all required features are present, filling missing ones with 0.0."""
    return {key: features.get(key, 0.0) for key in REQUIRED_FEATURES}

def load_ydf_model(model_path: str = "model_rf.ydf"):
    """Load the YDF model from the specified path."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YDF model file not found at {model_path}")
        
        ydf_model = ydf.from_tensorflow_decision_forests(model_path)
        print(f"YDF model loaded from {model_path}")
        return ydf_model
    except Exception as e:
        raise RuntimeError(f"Failed to load YDF model: {e}")

def load_mlx_model():
    """Check connectivity to MLX model via LM Studio server."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        response.raise_for_status()
        print("MLX model connected via LM Studio at http://localhost:1234")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to MLX server: {e}")

def mlx_generate(prompt: str, url: str = "http://localhost:1234/v1/completions", max_tokens: int = 10) -> str:
    """Generate trading decision using MLX model via LM Studio API."""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"MLX generation failed: {e}")
        return "HOLD"

def parse_mlx_decision(output: str) -> str:
    """Extract trading decision from MLX output."""
    output = output.upper()
    for word in output.split():
        if word in CLASSES:
            return word
    return "HOLD"

def ensemble_predict(ydf_model, nn_model, features: dict, mlx_url: str = "http://localhost:1234/v1/completions"):
    """Predict trading decision using ensemble of models."""
    features = ensure_required_features(features)

    # YDF prediction
    try:
        ydf_input = pd.DataFrame([features])
        ydf_pred = ydf_model.predict(ydf_input)
        ydf_probs = ydf_pred[0] if isinstance(ydf_pred[0], np.ndarray) else np.array([0.0, 1.0, 0.0])
    except Exception as e:
        print(f"YDF prediction failed: {e}")
        ydf_probs = np.array([0.0, 1.0, 0.0])

    # Core ML NN prediction
    try:
        nn_input = np.array([[features[col] for col in REQUIRED_FEATURES]], dtype=np.float32)
        nn_pred = nn_model.predict({"input": nn_input})
        if 'classLabel_probs' in nn_pred:
            probs_dict = nn_pred['classLabel_probs']
        elif 'classProbability' in nn_pred:
            probs_dict = nn_pred['classProbability']
        else:
            probs_dict = {}
        nn_probs = np.array([probs_dict.get(cls, 0.0) for cls in CLASSES], dtype=np.float32)
        if nn_probs.sum() == 0:
            nn_probs = np.array([0.0, 1.0, 0.0])
    except Exception as e:
        print(f"Core ML prediction failed: {e}")
        nn_probs = np.array([0.0, 1.0, 0.0])

    # MLX prediction
    prompt = f"Market features: {', '.join(f'{k}: {v}' for k, v in features.items())}. Recommend: BUY, SELL, or HOLD."
    mlx_output = mlx_generate(prompt, mlx_url)
    mlx_decision = parse_mlx_decision(mlx_output)
    mlx_probs = np.array([1.0 if cls == mlx_decision else 0.0 for cls in CLASSES], dtype=np.float32)

    # Ensemble
    avg_probs = np.mean([ydf_probs, nn_probs, mlx_probs], axis=0)
    decision_idx = np.argmax(avg_probs)
    decision = CLASSES[decision_idx]
    confidence = avg_probs[decision_idx]

    print(f"YDF: {ydf_probs}, NN: {nn_probs}, MLX: {mlx_probs}, Avg: {avg_probs}")
    return decision, confidence

def get_ensemble_decision(features: dict, ydf_model, nn_model, device: str = "cpu", mlx_url: str = "http://localhost:1234/v1/completions") -> tuple:
    """Get ensemble decision with error handling."""
    try:
        decision, confidence = ensemble_predict(ydf_model, nn_model, features, mlx_url)
        return decision.upper(), float(confidence)
    except Exception as e:
        print(f"Ensemble prediction error: {e}")
        return "HOLD", 0.0

if __name__ == "__main__":
    # Load models
    try:
        ydf_model = load_ydf_model()
        nn_model = ct.models.MLModel("/Users/maxime/BTC_BOT/models/trading_model.mlmodel")
        load_mlx_model()
        
        # Test features
        test_features = {"price": 100.0, "volume": 1000.0}
        decision, confidence = get_ensemble_decision(test_features, ydf_model, nn_model)
        print(f"Final Decision: {decision}, Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Setup failed: {e}")