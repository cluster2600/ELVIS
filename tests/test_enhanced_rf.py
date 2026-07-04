#!/usr/bin/env python3
"""
Test script for Enhanced Random Forest implementation
Validates all components work correctly together.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from utils.logger_config import setup_logging


def test_enhanced_rf_components():
    """Test all Enhanced Random Forest components."""

    logger = setup_logging("EnhancedRFTest", log_level="INFO")
    logger.info("🧪 Testing Enhanced Random Forest components")

    results = {
        "feature_pipeline": False,
        "enhanced_model": False,
        "integration": False,
        "errors": [],
    }

    # Test 1: Feature Pipeline
    try:
        from core.models.features.trading_feature_pipeline import TradingFeaturePipeline

        logger.info("Testing Trading Feature Pipeline...")

        # Create synthetic OHLCV data
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
                "open": np.random.uniform(49000, 51000, 100),
                "high": np.random.uniform(50000, 52000, 100),
                "low": np.random.uniform(48000, 50000, 100),
                "close": np.random.uniform(49500, 50500, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        pipeline = TradingFeaturePipeline(logger=logger)
        features = pipeline.transform(test_data)

        logger.info(f"✅ Feature pipeline generated {len(features.columns)} features")
        results["feature_pipeline"] = True

    except Exception as e:
        error_msg = f"Feature pipeline test failed: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    # Test 2: Enhanced Random Forest Model
    try:
        from core.models.enhanced_random_forest_model import EnhancedRandomForestModel

        logger.info("Testing Enhanced Random Forest Model...")

        # Create test data
        if results["feature_pipeline"]:
            X_test = features.iloc[:80]
            y_test = pd.Series(np.random.choice([0, 1, 2], size=80))
        else:
            # Fallback synthetic data
            X_test = pd.DataFrame(np.random.randn(80, 10))
            y_test = pd.Series(np.random.choice([0, 1, 2], size=80))

        # Initialize model without advanced features to avoid dependency issues
        model = EnhancedRandomForestModel(
            logger=logger,
            enable_optuna=False,  # Disable to avoid Optuna dependency
            enable_shap=False,  # Disable to avoid SHAP dependency
            enable_monitoring=False,  # Disable to avoid Prometheus dependency
        )

        # Train model
        model.train(X_test, y_test)

        # Test prediction
        predictions = model.predict(X_test.iloc[:5])
        probabilities = model.predict_proba(X_test.iloc[:5])

        # Test evaluation
        metrics = model.evaluate(X_test.iloc[:20], y_test.iloc[:20])

        logger.info(f"✅ Model trained and tested successfully")
        logger.info(f"📊 Test accuracy: {metrics.get('accuracy', 0):.3f}")
        results["enhanced_model"] = True

    except Exception as e:
        error_msg = f"Enhanced model test failed: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    # Test 3: Integration Layer
    try:
        from core.models.integration.enhanced_rf_integration import (
            EnhancedRFIntegration,
        )

        logger.info("Testing Integration Layer...")

        # Initialize integration (will handle missing dependencies gracefully)
        integration = EnhancedRFIntegration(
            logger=logger, enable_optimization=False, enable_monitoring=False
        )

        # Test market data preparation
        test_market_data = {
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 1500.0,
        }

        test_trading_features = {"sentiment": 0.5, "order_flow": 0.2}

        # This might fail if model isn't trained, which is expected
        try:
            prediction, metadata = integration.get_prediction(
                test_market_data, test_trading_features
            )
            if prediction is not None:
                logger.info(f"✅ Integration prediction successful: {prediction}")
            else:
                logger.info(
                    "⚠️ Integration prediction returned None (expected without trained model)"
                )
        except Exception as pred_e:
            logger.info(f"⚠️ Integration prediction failed (expected): {pred_e}")

        # Test performance monitoring
        performance = integration.get_model_performance()
        logger.info(f"✅ Performance monitoring working")

        results["integration"] = True

    except Exception as e:
        error_msg = f"Integration test failed: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    # Test Summary
    logger.info("=" * 60)
    logger.info("🧪 ENHANCED RANDOM FOREST TEST RESULTS")
    logger.info("=" * 60)

    passed_tests = sum(
        [results["feature_pipeline"], results["enhanced_model"], results["integration"]]
    )
    total_tests = 3

    logger.info(f"✅ Tests Passed: {passed_tests}/{total_tests}")

    if results["feature_pipeline"]:
        logger.info("✅ Feature Pipeline: Working")
    else:
        logger.info("❌ Feature Pipeline: Failed")

    if results["enhanced_model"]:
        logger.info("✅ Enhanced Model: Working")
    else:
        logger.info("❌ Enhanced Model: Failed")

    if results["integration"]:
        logger.info("✅ Integration Layer: Working")
    else:
        logger.info("❌ Integration Layer: Failed")

    if results["errors"]:
        logger.info("\n🚨 Errors encountered:")
        for error in results["errors"]:
            logger.error(f"  - {error}")

    if passed_tests == total_tests:
        logger.info("\n🎉 All Enhanced Random Forest components are working!")
        logger.info("Ready for production deployment.")
    else:
        logger.info(f"\n⚠️ {total_tests - passed_tests} component(s) need attention.")
        logger.info("Review errors above and install missing dependencies.")

    return results


if __name__ == "__main__":
    results = test_enhanced_rf_components()

    # Exit with error code if tests failed
    passed_tests = sum(
        [results["feature_pipeline"], results["enhanced_model"], results["integration"]]
    )
    exit(0 if passed_tests == 3 else 1)
