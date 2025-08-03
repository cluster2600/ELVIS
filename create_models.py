#!/usr/bin/env python3
"""
Script to create the missing models for the ensemble strategy.
This creates simple placeholder models that work with the existing code.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ydf_model():
    """Create a simple YDF model."""
    try:
        import ydf
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        
        # Create features similar to what the strategy expects
        data = pd.DataFrame({
            'price': np.random.normal(100000, 5000, n_samples),
            'volume': np.random.exponential(1000, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 100, n_samples),
            'sma': np.random.normal(100000, 4000, n_samples),
            'atr': np.random.exponential(500, n_samples),
            'adx': np.random.uniform(20, 70, n_samples),
            'Order_Amount': np.random.exponential(50, n_samples),
            'Filled': np.random.uniform(0, 100, n_samples),
            'Total': np.random.uniform(50, 200, n_samples),
            'future_price': np.random.normal(100000, 5000, n_samples),
            'vol_adjusted_price': np.random.normal(100000, 4500, n_samples),
            'volume_ma': np.random.exponential(800, n_samples),
            'signal_line': np.random.normal(0, 80, n_samples),
            'lower_bb': np.random.normal(95000, 4000, n_samples),
            'sma_bb': np.random.normal(100000, 4000, n_samples),
            'upper_bb': np.random.normal(105000, 4000, n_samples),
            'news_sentiment': np.random.uniform(-1, 1, n_samples),
            'social_feature': np.random.uniform(-1, 1, n_samples),
            'order_book_depth': np.random.exponential(100, n_samples)
        })
        
        # Create target variable (BUY=0, HOLD=1, SELL=2)
        target = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
        data['target'] = target
        
        # Train YDF model
        model = ydf.RandomForestLearner(label="target").train(data)
        
        # Save model in TensorFlow format (for from_tensorflow_decision_forests)
        model_path = "models/model_rf_tf"
        os.makedirs("models", exist_ok=True)
        model.save(model_path, advanced_options=ydf.ModelIOOptions(file_prefix="saved_model"))
        
        logger.info(f"✓ YDF model created and saved to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create YDF model: {e}")
        return False

def create_coreml_model():
    """Create a simple CoreML model."""
    try:
        import coremltools as ct
        import torch
        import torch.nn as nn
        
        # Define a simple neural network
        class SimpleNN(nn.Module):
            def __init__(self, input_size=20, hidden_size=64, output_size=3):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                    nn.Softmax(dim=1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Create and train model
        model = SimpleNN()
        model.eval()
        
        # Create sample input for tracing
        sample_input = torch.randn(1, 20)
        
        # Trace the model
        traced_model = torch.jit.trace(model, sample_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, 20))],
            minimum_deployment_target=ct.target.macOS13
        )
        
        # Save model
        model_path = "models/NNModel.mlpackage"
        os.makedirs("models", exist_ok=True)
        coreml_model.save(model_path)
        
        logger.info(f"✓ CoreML model created and saved to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create CoreML model: {e}")
        return False

def create_trade_learned_model():
    """Create a simple trade-learned model."""
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        
        # Features similar to what trade_learned model expects
        X = np.random.randn(n_samples, 10)
        y = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save model
        model_path = "training/models/trade_learned_model.pkl"
        os.makedirs("training/models", exist_ok=True)
        joblib.dump(model, model_path)
        
        logger.info(f"✓ Trade-learned model created and saved to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create trade-learned model: {e}")
        return False

def main():
    """Create all missing models."""
    logger.info("Creating missing models for ensemble strategy...")
    
    ydf_success = create_ydf_model()
    coreml_success = create_coreml_model() 
    trade_success = create_trade_learned_model()
    
    if ydf_success and coreml_success and trade_success:
        logger.info("✓ All models created successfully!")
        return True
    else:
        logger.error("✗ Some models failed to create")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)