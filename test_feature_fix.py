#!/usr/bin/env python3
"""
Test the RandomForestClassifier feature mismatch fix
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment
os.environ['VAULT_ENABLED'] = 'false'

def test_research_strategy_features():
    """Test that research strategy works with correct feature count"""
    logger.info("🧪 Testing Research Strategy Feature Fix")
    logger.info("=" * 50)
    
    try:
        from trading.strategies.research_based_strategy import ResearchBasedStrategy
        
        logger.info("1. Creating research strategy...")
        strategy = ResearchBasedStrategy(
            logger=logger,
            social_data_enabled=False,  # Use 9 features only
            enable_rolling_training=False
        )
        
        logger.info("2. Creating test market data...")
        # Create realistic test data
        test_data = pd.DataFrame({
            'high': [97500, 97800, 98200, 98000, 97900],
            'low': [96800, 97200, 97500, 97300, 97100],
            'close': [97200, 97600, 97800, 97500, 97400],
            'volume': [15000, 18000, 16500, 14200, 17300]
        })
        
        logger.info("3. Testing feature preparation...")
        features = strategy.prepare_features(test_data)
        logger.info(f"   Prepared features shape: {features.shape}")
        
        if features.shape[1] == 9:
            logger.info("   ✅ Correct feature count (9 features)")
        else:
            logger.error(f"   ❌ Wrong feature count: got {features.shape[1]}, expected 9")
            return False
        
        logger.info("4. Testing signal generation...")
        data_dict = {'BTCUSDT': test_data}
        signals = strategy.generate_signals(data_dict)
        
        if 'BTCUSDT' in signals:
            signal = signals['BTCUSDT']
            logger.info(f"   Generated signal: {signal['signal']} (confidence: {signal['confidence']:.3f})")
            logger.info("   ✅ Signal generation successful!")
            return True
        else:
            logger.error("   ❌ No signal generated")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_loading():
    """Test that the retrained model loads correctly"""
    logger.info("\n📚 Testing Model Loading")
    logger.info("-" * 50)
    
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("1. Loading retrained model...")
        model_path = "models/research_based/research_rf_model.pkl"
        scaler_path = "models/research_based/research_scaler.pkl"
        
        if not os.path.exists(model_path):
            logger.error(f"   ❌ Model file not found: {model_path}")
            return False
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logger.info("2. Checking model properties...")
        logger.info(f"   Model type: {type(model)}")
        logger.info(f"   Expected features: {model.n_features_in_}")
        logger.info(f"   Scaler features: {scaler.n_features_in_}")
        
        if model.n_features_in_ == 9:
            logger.info("   ✅ Model expects 9 features (correct)")
        else:
            logger.error(f"   ❌ Model expects {model.n_features_in_} features, should be 9")
            return False
        
        logger.info("3. Testing prediction with 9 features...")
        test_features = np.random.random((1, 9))
        test_features_scaled = scaler.transform(test_features)
        
        prediction = model.predict(test_features_scaled)
        probabilities = model.predict_proba(test_features_scaled)
        
        logger.info(f"   Prediction: {prediction[0]}")
        logger.info(f"   Probabilities: {probabilities[0]}")
        logger.info("   ✅ Model prediction successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main test function"""
    logger.info("🔧 Feature Mismatch Fix Test")
    logger.info("=" * 60)
    
    # Test 1: Research strategy features
    test1_success = test_research_strategy_features()
    
    # Test 2: Model loading
    test2_success = test_model_loading()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📋 FEATURE FIX TEST SUMMARY:")
    logger.info(f"   Research Strategy Features: {'✅' if test1_success else '❌'}")
    logger.info(f"   Model Loading and Prediction: {'✅' if test2_success else '❌'}")
    
    all_tests_passed = all([test1_success, test2_success])
    
    if all_tests_passed:
        logger.info("\n🎉 FEATURE MISMATCH FIXED!")
        logger.info("✅ Model now expects exactly 9 features")
        logger.info("✅ Research strategy provides exactly 9 features")
        logger.info("✅ Prediction should work without errors")
        logger.info("\n📋 Fixed Issues:")
        logger.info("   • RandomForestClassifier retrained with 9 features")
        logger.info("   • Feature preparation ensures 9 features")
        logger.info("   • Added safety checks for feature shape")
        logger.info("   • Model prediction errors should be resolved")
    else:
        logger.warning("\n⚠️ Some tests failed - check logs above")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()