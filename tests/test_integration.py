"""
Hermetic integration tests for the supported model and monitoring components.
"""

import logging
import unittest

import numpy as np
import pandas as pd

from core.metrics.performance_monitor import PerformanceMonitor
from core.models.ensemble_model import EnsembleModel
from core.models.random_forest_model import RandomForestModel


class TestIntegration(unittest.TestCase):
    """
    Integration tests for the ELVIS project.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)

    def test_model_training_and_prediction(self):
        """
        Test model training and prediction integration.
        """
        # Create test data
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(
            {
                "feature1": rng.random(100),
                "feature2": rng.random(100),
                "feature3": rng.random(100),
            }
        )
        y_train = pd.Series(rng.integers(0, 2, 100))

        X_test = pd.DataFrame(
            {
                "feature1": rng.random(20),
                "feature2": rng.random(20),
                "feature3": rng.random(20),
            }
        )

        # Set up models
        rf_model = RandomForestModel(logger=self.logger)
        # Train Random Forest model
        rf_model.train(X_train, y_train)

        # Make predictions with Random Forest model
        rf_predictions = rf_model.predict(X_test)

        # Check predictions
        self.assertIsInstance(rf_predictions, np.ndarray)
        self.assertEqual(len(rf_predictions), len(X_test))

        # Set up ensemble model
        ensemble_model = EnsembleModel(
            logger=self.logger, models=[("random_forest", rf_model)]
        )

        # Make predictions with ensemble model
        ensemble_predictions = ensemble_model.predict(X_test)

        # Check predictions
        self.assertIsInstance(ensemble_predictions, np.ndarray)
        self.assertEqual(len(ensemble_predictions), len(X_test))

    def test_performance_monitoring(self):
        """
        Test performance monitoring integration.
        """
        # Set up performance monitor
        performance_monitor = PerformanceMonitor()

        # Add trades
        for i in range(10):
            pnl = 100.0 if i % 3 == 0 else -50.0
            performance_monitor.add_return(pnl)

        # Calculate metrics
        sharpe = performance_monitor.calculate_rolling_sharpe()
        drawdown = performance_monitor.calculate_rolling_drawdown()

        # Check metrics
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(drawdown, float)


if __name__ == "__main__":
    unittest.main()
