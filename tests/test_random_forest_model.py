"""
Unit tests for the RandomForestModel class.
"""

import unittest
import logging
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock, call

from core.models.random_forest_model import RandomForestModel

class TestRandomForestModel(unittest.TestCase):
    """
    Test cases for the RandomForestModel class.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Set up model
        self.model_path = 'test_model.ydf'
        self.model = RandomForestModel(
            logger=self.logger,
            model_path=self.model_path,
            num_trees=50,
            max_depth=10,
            min_examples=3
        )
        
        # Set up test data
        self.X_train = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100)
        })
        self.y_train = pd.Series(np.random.randint(0, 2, 100))
        
        self.X_test = pd.DataFrame({
            'feature1': np.random.random(20),
            'feature2': np.random.random(20),
            'feature3': np.random.random(20)
        })
        self.y_test = pd.Series(np.random.randint(0, 2, 20))
    
    def tearDown(self):
        """
        Clean up after the test case.
        """
        # Remove test model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
    
    def test_init(self):
        """
        Test the initialization of the RandomForestModel.
        """
        # Check attributes
        self.assertEqual(self.model.name, 'random_forest')
        self.assertEqual(self.model.logger, self.logger)
        self.assertEqual(self.model.num_trees, 50)
        self.assertEqual(self.model.max_depth, 10)
        self.assertEqual(self.model.min_examples, 3)
        self.assertEqual(self.model.model_path, self.model_path)
        self.assertIsNone(self.model.model)
    
    @patch('tensorflow_decision_forests.keras.RandomForestModel.load')
    def test_load_model(self, mock_load):
        """
        Test the load_model method.
        """
        # Set up mock
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Mock os.path.exists
        with patch('os.path.exists', return_value=True):
            # Call method
            self.model.load_model()
            
            # Check model
            self.assertEqual(self.model.model, mock_model)
            
            # Check load call
            mock_load.assert_called_once_with(self.model_path)
    
    @patch('os.makedirs')
    def test_save_model(self, mock_makedirs):
        """
        Test the save_model method.
        """
        # Set up mock
        mock_model = MagicMock()
        self.model.model = mock_model
        
        # Call method
        self.model.save_model()
        
        # Check save call
        mock_model.save.assert_called_once_with(self.model_path)
        
        # Check makedirs call
        mock_makedirs.assert_called_once_with(os.path.dirname(self.model_path), exist_ok=True)
    
    @patch('tensorflow_decision_forests.keras.RandomForestModel')
    @patch('tensorflow_decision_forests.keras.pd_dataframe_to_tf_dataset')
    def test_train(self, mock_to_dataset, mock_rf_model):
        """
        Test the train method.
        """
        # Set up mocks
        mock_model = MagicMock()
        mock_rf_model.return_value = mock_model
        mock_dataset = MagicMock()
        mock_to_dataset.return_value = mock_dataset
        
        # Call method
        self.model.train(self.X_train, self.y_train)
        
        # Check model creation
        mock_rf_model.assert_called_once_with(
            num_trees=50,
            max_depth=10,
            min_examples=3,
            verbose=2
        )
        
        # Check dataset conversion
        mock_to_dataset.assert_called_once_with(self.X_train, label=self.y_train)
        
        # Check fit call
        mock_model.fit.assert_called_once_with(mock_dataset)
        
        # Check model assignment
        self.assertEqual(self.model.model, mock_model)
    
    @patch('tensorflow_decision_forests.keras.pd_dataframe_to_tf_dataset')
    def test_predict(self, mock_to_dataset):
        """
        Test the predict method.
        """
        # Set up mocks
        mock_model = MagicMock()
        self.model.model = mock_model
        mock_dataset = MagicMock()
        mock_to_dataset.return_value = mock_dataset
        mock_predictions = MagicMock()
        mock_predictions.numpy.return_value = np.array([[0.2], [0.8], [0.3]])
        mock_model.predict.return_value = mock_predictions
        
        # Call method
        result = self.model.predict(self.X_test)
        
        # Check dataset conversion
        mock_to_dataset.assert_called_once_with(self.X_test, label=None)
        
        # Check predict call
        mock_model.predict.assert_called_once_with(mock_dataset)
        
        # Check result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)
    
    @patch('tensorflow_decision_forests.keras.pd_dataframe_to_tf_dataset')
    def test_evaluate(self, mock_to_dataset):
        """
        Test the evaluate method.
        """
        # Set up mocks
        mock_model = MagicMock()
        self.model.model = mock_model
        mock_dataset = MagicMock()
        mock_to_dataset.return_value = mock_dataset
        mock_model.evaluate.return_value = {'accuracy': 0.85, 'loss': 0.3}
        
        # Call method
        result = self.model.evaluate(self.X_test, self.y_test)
        
        # Check dataset conversion
        mock_to_dataset.assert_called_once_with(self.X_test, label=self.y_test)
        
        # Check evaluate call
        mock_model.evaluate.assert_called_once_with(mock_dataset, return_dict=True)
        
        # Check result
        self.assertEqual(result['accuracy'], 0.85)
        self.assertEqual(result['loss'], 0.3)
    
    def test_get_feature_importance(self):
        """
        Test the get_feature_importance method.
        """
        # Set up mocks
        mock_model = MagicMock()
        self.model.model = mock_model
        mock_inspector = MagicMock()
        mock_model.make_inspector.return_value = mock_inspector
        mock_importance = {
            'MEAN_DECREASE_IN_ACCURACY': [
                ('feature1', 0.5),
                ('feature2', 0.3),
                ('feature3', 0.2)
            ]
        }
        mock_inspector.variable_importances.return_value = mock_importance
        
        # Call method
        result = self.model.get_feature_importance()
        
        # Check inspector call
        mock_model.make_inspector.assert_called_once()
        mock_inspector.variable_importances.assert_called_once()
        
        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ['feature', 'importance'])

if __name__ == '__main__':
    unittest.main()
