#!/usr/bin/env python3
"""
Test that both warnings are fixed:
1. Scikit-learn version mismatch warnings
2. PyTorch tensor creation performance warning
"""

import logging
import os
import sys
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging to capture warnings
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set environment
os.environ["VAULT_ENABLED"] = "false"


def test_sklearn_warnings():
    """Test that sklearn version warnings are fixed"""
    logger.info("🧪 Testing Scikit-Learn Warnings")
    logger.info("=" * 50)

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            logger.info("1. Loading research-based strategy...")
            from trading.strategies.research_based_strategy import ResearchBasedStrategy

            # Initialize strategy (this loads the models)
            strategy = ResearchBasedStrategy(
                logger=logger, social_data_enabled=False, enable_rolling_training=False
            )

            logger.info("2. Strategy loaded successfully")

            # Check for sklearn warnings
            sklearn_warnings = [
                warning
                for warning in w
                if "InconsistentVersionWarning" in str(warning.category)
            ]

            if sklearn_warnings:
                logger.error(f"   ❌ Found {len(sklearn_warnings)} sklearn warnings:")
                for warning in sklearn_warnings:
                    logger.error(f"      {warning.message}")
                return False
            else:
                logger.info("   ✅ No sklearn version warnings found!")
                return True

        except Exception as e:
            logger.error(f"❌ Error testing sklearn warnings: {e}")
            return False


def test_pytorch_warnings():
    """Test that PyTorch tensor creation warnings are fixed"""
    logger.info("\n🚀 Testing PyTorch Tensor Warnings")
    logger.info("-" * 50)

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            logger.info("1. Creating RL model and training...")
            from core.models.trading_rl_model import TradingRLModel

            # Create model and trigger training (which uses tensor creation)
            model_path = "models/test_warnings_rl_model.pth"
            if os.path.exists(model_path):
                os.remove(model_path)

            rl_model = TradingRLModel(logger, model_path)

            # Trigger synthetic training (this will create tensors)
            success = rl_model.train_on_historical_data(limit=50)

            logger.info("2. RL training completed")

            # Check for tensor creation warnings
            tensor_warnings = [
                warning
                for warning in w
                if "Creating a tensor from a list of numpy.ndarrays is extremely slow"
                in str(warning.message)
            ]

            if tensor_warnings:
                logger.error(f"   ❌ Found {len(tensor_warnings)} tensor warnings:")
                for warning in tensor_warnings:
                    logger.error(f"      {warning.message}")
                return False
            else:
                logger.info("   ✅ No PyTorch tensor warnings found!")
                return True

        except Exception as e:
            logger.error(f"❌ Error testing PyTorch warnings: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False


def main():
    """Main test function"""
    logger.info("🔧 Testing Warning Fixes")
    logger.info("=" * 60)

    # Test 1: Sklearn warnings
    test1_success = test_sklearn_warnings()

    # Test 2: PyTorch warnings
    test2_success = test_pytorch_warnings()

    # Summary
    logger.info("=" * 60)
    logger.info("📋 WARNING FIXES TEST SUMMARY:")
    logger.info(f"   Scikit-Learn Version Warnings: {'✅' if test1_success else '❌'}")
    logger.info(
        f"   PyTorch Tensor Creation Warnings: {'✅' if test2_success else '❌'}"
    )

    all_tests_passed = all([test1_success, test2_success])

    if all_tests_passed:
        logger.info("\n🎉 ALL WARNINGS FIXED!")
        logger.info("✅ Scikit-learn models retrained with correct version")
        logger.info("✅ PyTorch tensor creation optimized")
        logger.info("\n📋 Fixed Issues:")
        logger.info("   • RandomForestClassifier and StandardScaler version mismatch")
        logger.info("   • Inefficient tensor creation from list of numpy arrays")
        logger.info("   • Both warnings should no longer appear in bot output")
    else:
        logger.warning("\n⚠️ Some warnings still present - check logs above")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
