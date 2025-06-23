import unittest
import pandas as pd
import numpy as np
from core.models.random_forest_model import RandomForestModel
import logging
import matplotlib  # Prevent plot from blocking tests
matplotlib.use('Agg')  # Use a non-interactive backend for tests

class TestRandomForestModel(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestRandomForestModel")
        self.model = RandomForestModel(logger=self.logger)

        # Dummy binary classification dataset
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.y_train = pd.Series(np.random.randint(0, 2, size=100))

        self.X_test = pd.DataFrame({
            'feature1': np.random.rand(20),
            'feature2': np.random.rand(20),
            'feature3': np.random.rand(20)
        })
        self.y_test = pd.Series(np.random.randint(0, 2, size=20))

    def test_train_and_predict(self):
        """Test training and prediction shape + finiteness"""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_evaluate(self):
        """Test evaluation returns expected metrics and types"""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)

        for key in ['accuracy', 'loss', 'precision', 'recall', 'f1']:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

    def test_feature_importance(self):
        """Test that feature importance is a non-empty DataFrame"""
        self.model.train(self.X_train, self.y_train)
        importance_df = self.model.get_feature_importance()

        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)

    def test_cross_validate(self):
        """Test k-fold cross-validation with return of all metrics"""
        metrics = self.model.cross_validate(self.X_train, self.y_train, n_splits=3)

        for key in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], np.ndarray)

if __name__ == '__main__':
    unittest.main()
