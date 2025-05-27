"""
Tests for Ensemble Model
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime
from core.models.ensemble_model import EnsembleModel


class TestEnsembleModel:
    """Test EnsembleModel functionality"""
    
    def test_ensemble_model_initialization(self, mock_logger):
        """Test EnsembleModel initialization"""
        models = {
            'rf': MagicMock(),
            'nn': MagicMock(),
            'transformer': MagicMock()
        }
        
        ensemble = EnsembleModel(
            logger=mock_logger,
            models=models,
            weights={'rf': 0.4, 'nn': 0.3, 'transformer': 0.3}
        )
        
        assert ensemble.models == models
        assert ensemble.weights == {'rf': 0.4, 'nn': 0.3, 'transformer': 0.3}
        assert sum(ensemble.weights.values()) == 1.0
    
    def test_ensemble_model_default_weights(self, mock_logger):
        """Test EnsembleModel with default equal weights"""
        models = {
            'model1': MagicMock(),
            'model2': MagicMock(),
            'model3': MagicMock()
        }
        
        ensemble = EnsembleModel(logger=mock_logger, models=models)
        
        # Should have equal weights
        expected_weight = 1.0 / len(models)
        for model_name, weight in ensemble.weights.items():
            assert weight == pytest.approx(expected_weight)
    
    def test_predict_with_multiple_models(self, mock_logger, sample_price_data):
        """Test ensemble prediction with multiple models"""
        # Create mock models with different predictions
        models = {
            'model1': MagicMock(),
            'model2': MagicMock(),
            'model3': MagicMock()
        }
        
        models['model1'].predict.return_value = np.array([0.7])  # Bullish
        models['model2'].predict.return_value = np.array([0.4])  # Bearish
        models['model3'].predict.return_value = np.array([0.6])  # Slightly bullish
        
        weights = {'model1': 0.4, 'model2': 0.3, 'model3': 0.3}
        
        ensemble = EnsembleModel(logger=mock_logger, models=models, weights=weights)
        
        # Create features
        features = sample_price_data[['close', 'volume']].values[-1:, :]
        
        prediction = ensemble.predict(features)
        
        # Calculate expected weighted average
        expected = (0.7 * 0.4 + 0.4 * 0.3 + 0.6 * 0.3)
        
        assert prediction == pytest.approx(expected, rel=1e-5)
        
        # Verify all models were called
        for model in models.values():
            model.predict.assert_called_once()
    
    def test_predict_with_model_failure(self, mock_logger):
        """Test ensemble prediction when one model fails"""
        models = {
            'model1': MagicMock(),
            'model2': MagicMock(),
            'model3': MagicMock()
        }
        
        models['model1'].predict.return_value = np.array([0.7])
        models['model2'].predict.side_effect = Exception("Model failed")
        models['model3'].predict.return_value = np.array([0.6])
        
        weights = {'model1': 0.4, 'model2': 0.3, 'model3': 0.3}
        
        ensemble = EnsembleModel(logger=mock_logger, models=models, weights=weights)
        
        features = np.array([[50000, 100]])
        prediction = ensemble.predict(features)
        
        # Should handle failure gracefully and use remaining models
        # With model2 failed, we renormalize weights: model1=0.4/0.7, model3=0.3/0.7
        expected = (0.7 * (0.4/0.7) + 0.6 * (0.3/0.7))
        
        assert prediction == pytest.approx(expected, rel=1e-5)
        mock_logger.error.assert_called()
    
    def test_predict_all_models_fail(self, mock_logger):
        """Test ensemble prediction when all models fail"""
        models = {
            'model1': MagicMock(),
            'model2': MagicMock()
        }
        
        for model in models.values():
            model.predict.side_effect = Exception("Model failed")
        
        ensemble = EnsembleModel(logger=mock_logger, models=models)
        
        features = np.array([[50000, 100]])
        prediction = ensemble.predict(features)
        
        # Should return neutral prediction (0.5)
        assert prediction == 0.5
        assert mock_logger.error.call_count >= 2
    
    def test_get_prediction_confidence(self, mock_logger):
        """Test getting prediction confidence"""
        models = {
            'model1': MagicMock(),
            'model2': MagicMock(),
            'model3': MagicMock()
        }
        
        # Set up predictions with varying agreement
        models['model1'].predict.return_value = np.array([0.8])
        models['model2'].predict.return_value = np.array([0.85])
        models['model3'].predict.return_value = np.array([0.2])  # Disagrees
        
        ensemble = EnsembleModel(logger=mock_logger, models=models)
        
        features = np.array([[50000, 100]])
        _ = ensemble.predict(features)
        confidence = ensemble.get_prediction_confidence()
        
        # With one model strongly disagreeing, confidence should be lower
        assert 0 <= confidence <= 1
        assert confidence < 0.5  # Low confidence due to disagreement
    
    def test_get_model_contributions(self, mock_logger):
        """Test getting individual model contributions"""
        models = {
            'model1': MagicMock(),
            'model2': MagicMock()
        }
        
        models['model1'].predict.return_value = np.array([0.7])
        models['model2'].predict.return_value = np.array([0.4])
        
        weights = {'model1': 0.6, 'model2': 0.4}
        
        ensemble = EnsembleModel(logger=mock_logger, models=models, weights=weights)
        
        features = np.array([[50000, 100]])
        _ = ensemble.predict(features)
        
        contributions = ensemble.get_model_contributions()
        
        assert 'model1' in contributions
        assert 'model2' in contributions
        assert contributions['model1']['prediction'] == 0.7
        assert contributions['model1']['weight'] == 0.6
        assert contributions['model1']['contribution'] == pytest.approx(0.7 * 0.6)
    
    def test_update_weights(self, mock_logger):
        """Test updating model weights"""
        models = {'model1': MagicMock(), 'model2': MagicMock()}
        
        ensemble = EnsembleModel(logger=mock_logger, models=models)
        
        new_weights = {'model1': 0.7, 'model2': 0.3}
        ensemble.update_weights(new_weights)
        
        assert ensemble.weights == new_weights
    
    def test_update_weights_normalization(self, mock_logger):
        """Test weight normalization when updating"""
        models = {'model1': MagicMock(), 'model2': MagicMock()}
        
        ensemble = EnsembleModel(logger=mock_logger, models=models)
        
        # Provide weights that don't sum to 1
        new_weights = {'model1': 2, 'model2': 3}
        ensemble.update_weights(new_weights)
        
        # Weights should be normalized
        assert ensemble.weights['model1'] == pytest.approx(0.4)
        assert ensemble.weights['model2'] == pytest.approx(0.6)
        assert sum(ensemble.weights.values()) == pytest.approx(1.0)
    
    def test_save_and_load_ensemble_config(self, mock_logger, temp_dir):
        """Test saving and loading ensemble configuration"""
        models = {
            'rf': MagicMock(),
            'nn': MagicMock()
        }
        weights = {'rf': 0.6, 'nn': 0.4}
        
        ensemble = EnsembleModel(logger=mock_logger, models=models, weights=weights)
        
        # Save configuration
        config_path = temp_dir / "ensemble_config.json"
        ensemble.save_config(str(config_path))
        
        assert config_path.exists()
        
        # Load configuration
        loaded_config = ensemble.load_config(str(config_path))
        
        assert loaded_config['weights'] == weights
        assert 'model_names' in loaded_config
        assert set(loaded_config['model_names']) == set(models.keys())
    
    @patch('core.models.ensemble_model.datetime')
    def test_track_prediction_history(self, mock_datetime, mock_logger):
        """Test tracking prediction history"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 0, 0)
        
        models = {'model1': MagicMock()}
        models['model1'].predict.return_value = np.array([0.7])
        
        ensemble = EnsembleModel(logger=mock_logger, models=models)
        
        # Make several predictions
        for i in range(5):
            features = np.array([[50000 + i * 100, 100 + i * 10]])
            ensemble.predict(features)
        
        history = ensemble.get_prediction_history(limit=3)
        
        assert len(history) == 3  # Should return last 3
        assert all('timestamp' in h for h in history)
        assert all('prediction' in h for h in history)
        assert all('model_predictions' in h for h in history)
