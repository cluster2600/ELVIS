"""
Enhanced Random Forest Integration for ELVIS Ensemble Strategy
Provides seamless integration of the enhanced RF model with existing ensemble.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import os

from core.models.enhanced_random_forest_model import EnhancedRandomForestModel
from core.models.features.trading_feature_pipeline import TradingFeaturePipeline
from core.models.optimization.hyperparameter_optimizer import HyperparameterOptimizer

class EnhancedRFIntegration:
    """
    Integration layer for Enhanced Random Forest in ELVIS ensemble strategy.
    
    Features:
    - Seamless model loading and prediction
    - Feature engineering pipeline
    - Performance monitoring and alerts
    - Automatic retraining triggers
    - Explainability integration
    """
    
    def __init__(self, 
                 logger: logging.Logger,
                 model_path: str = "models/enhanced_rf",
                 enable_optimization: bool = True,
                 enable_monitoring: bool = True,
                 auto_retrain: bool = True):
        
        self.logger = logger
        self.model_path = model_path
        self.enable_optimization = enable_optimization
        self.enable_monitoring = enable_monitoring
        self.auto_retrain = auto_retrain
        
        # Initialize components
        self.model = None
        self.feature_pipeline = TradingFeaturePipeline(logger=logger)
        self.optimizer = None
        
        # Performance tracking
        self.prediction_history = []
        self.last_retrain_check = datetime.now()
        self.retrain_check_interval = timedelta(hours=6)
        
        # Integration settings
        self.confidence_threshold = 0.6
        self.ensemble_weight = 0.3  # Weight in ensemble voting
        
        # Initialize model and optimizer
        self._initialize_components()
        
        self.logger.info("✅ Enhanced RF Integration initialized")
    
    def _initialize_components(self):
        """Initialize model and optimization components."""
        
        try:
            # Initialize enhanced Random Forest model
            self.model = EnhancedRandomForestModel(
                logger=self.logger,
                model_path=self.model_path,
                enable_optuna=self.enable_optimization,
                enable_shap=True,
                enable_monitoring=self.enable_monitoring
            )
            
            # Try to load existing model
            if os.path.exists(os.path.join(self.model_path, 'rf_model.pkl')):
                self.model = EnhancedRandomForestModel.load(self.model_path, self.logger)
                self.logger.info("📂 Loaded existing enhanced RF model")
            else:
                self.logger.info("🆕 Initialized new enhanced RF model")
            
            # Initialize optimizer if needed
            if self.enable_optimization:
                self.optimizer = HyperparameterOptimizer(
                    model_class=EnhancedRandomForestModel,
                    logger=self.logger,
                    study_name="enhanced_rf_optimization"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced RF integration: {e}")
            # Create minimal fallback
            self.model = None
    
    def get_prediction(self, market_data: Dict[str, Any], 
                      trading_features: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get enhanced Random Forest prediction with metadata.
        
        Args:
            market_data: Raw market data (OHLCV)
            trading_features: Additional trading features
            
        Returns:
            Tuple of (prediction_probabilities, metadata)
        """
        
        if self.model is None or self.model.model is None:
            return None, {'error': 'Model not available'}
        
        try:
            # Prepare data for feature engineering
            ohlcv_data = self._prepare_ohlcv_data(market_data)
            
            # Generate comprehensive features
            features_df = self.feature_pipeline.transform(ohlcv_data)
            
            if len(features_df) == 0:
                return None, {'error': 'Feature generation failed'}
            
            # Get the latest feature vector
            latest_features = features_df.iloc[-1:].copy()
            
            # Add additional trading features
            for key, value in trading_features.items():
                if key not in latest_features.columns:
                    latest_features[key] = value
            
            # Make prediction
            prediction_proba = self.model.predict_proba(latest_features)
            prediction = self.model.predict(latest_features)
            
            # Get feature importance and explanation
            feature_importance = self.model.get_feature_importance()
            explanation = self.model.explain_prediction(latest_features) if self.model.enable_shap else None
            
            # Calculate confidence and ensemble weights
            confidence = float(np.max(prediction_proba))
            ensemble_weights = self._calculate_ensemble_weights(confidence, prediction_proba[0])
            
            # Prepare metadata
            metadata = {
                'confidence': confidence,
                'prediction_class': int(prediction[0]),
                'class_probabilities': {
                    'BUY': float(prediction_proba[0][2]),    # Assuming class 2 = BUY
                    'HOLD': float(prediction_proba[0][1]),   # Assuming class 1 = HOLD  
                    'SELL': float(prediction_proba[0][0])    # Assuming class 0 = SELL
                },
                'ensemble_weights': ensemble_weights,
                'feature_count': len(latest_features.columns),
                'top_features': feature_importance.head(5).to_dict('records') if not feature_importance.empty else [],
                'model_stats': self.model.get_model_stats(),
                'prediction_time': datetime.now().isoformat()
            }
            
            # Add SHAP explanation if available
            if explanation is not None:
                metadata['shap_explanation'] = {
                    'values': explanation.tolist() if hasattr(explanation, 'tolist') else [],
                    'feature_names': list(latest_features.columns)
                }
            
            # Log prediction for monitoring
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'confidence': confidence,
                'prediction': int(prediction[0]),
                'probabilities': prediction_proba[0].tolist()
            })
            
            # Check if retraining is needed
            if self.auto_retrain and datetime.now() - self.last_retrain_check > self.retrain_check_interval:
                self._check_retrain_trigger()
            
            # Convert to ensemble format [BUY, HOLD, SELL]
            ensemble_prediction = np.array([
                float(prediction_proba[0][0]),  # SELL
                float(prediction_proba[0][1]),  # HOLD
                float(prediction_proba[0][2])   # BUY
            ])
            
            self.logger.debug(f"Enhanced RF prediction: {ensemble_prediction} (confidence: {confidence:.3f})")
            
            return ensemble_prediction, metadata
            
        except Exception as e:
            self.logger.error(f"Enhanced RF prediction failed: {e}")
            return None, {'error': str(e)}
    
    def _prepare_ohlcv_data(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare OHLCV data for feature engineering."""
        
        # If market_data contains a DataFrame or list, use it directly
        if isinstance(market_data, pd.DataFrame):
            return market_data
        
        # If it's a single data point, create a minimal DataFrame
        if isinstance(market_data, dict):
            # Create a single-row DataFrame
            df = pd.DataFrame([market_data])
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 1000.0  # Default volume
                    else:
                        df[col] = market_data.get('price', market_data.get('close', 50000.0))
            
            # Add timestamp if missing
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now()
            
            return df
        
        # Fallback: create minimal data
        default_price = 50000.0
        return pd.DataFrame([{
            'open': default_price,
            'high': default_price * 1.01,
            'low': default_price * 0.99,
            'close': default_price,
            'volume': 1000.0,
            'timestamp': datetime.now()
        }])
    
    def _calculate_ensemble_weights(self, confidence: float, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate dynamic weights for ensemble integration."""
        
        # Base weight from configuration
        base_weight = self.ensemble_weight
        
        # Adjust weight based on confidence
        confidence_multiplier = 1.0
        if confidence > 0.8:
            confidence_multiplier = 1.2  # High confidence gets more weight
        elif confidence < 0.5:
            confidence_multiplier = 0.7  # Low confidence gets less weight
        
        # Adjust based on prediction certainty (entropy)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        max_entropy = np.log(len(probabilities))
        certainty = 1.0 - (entropy / max_entropy)
        
        # Final ensemble weight
        final_weight = base_weight * confidence_multiplier * (0.5 + 0.5 * certainty)
        final_weight = np.clip(final_weight, 0.1, 0.5)  # Keep within reasonable bounds
        
        return {
            'base_weight': base_weight,
            'confidence_multiplier': confidence_multiplier,
            'certainty_score': certainty,
            'final_weight': final_weight
        }
    
    def _check_retrain_trigger(self):
        """Check if model should be retrained based on performance."""
        
        self.last_retrain_check = datetime.now()
        
        if len(self.prediction_history) < 100:
            return  # Not enough data
        
        try:
            # Check recent performance
            recent_predictions = self.prediction_history[-50:]
            recent_confidence = np.mean([p['confidence'] for p in recent_predictions])
            
            # Check for confidence degradation
            if recent_confidence < 0.5:
                self.logger.warning("⚠️ Model confidence degraded, retraining recommended")
                self._trigger_retrain_alert()
            
            # Check model-specific retrain conditions
            if hasattr(self.model, '_should_retrain') and self.model._should_retrain():
                self.logger.warning("⚠️ Model performance degraded, retraining recommended")
                self._trigger_retrain_alert()
                
        except Exception as e:
            self.logger.error(f"Retrain check failed: {e}")
    
    def _trigger_retrain_alert(self):
        """Trigger alert for model retraining."""
        
        # Log alert
        self.logger.warning("🔄 Enhanced RF model retraining triggered")
        
        # Could integrate with external alerting system here
        # For now, just log the recommendation
        
        # Clear old prediction history to prevent repeated alerts
        self.prediction_history = self.prediction_history[-20:]  # Keep recent history only
    
    def train_model(self, training_data: pd.DataFrame, labels: pd.Series, 
                   optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the enhanced Random Forest model.
        
        Args:
            training_data: Training features
            labels: Target labels
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        
        if self.model is None:
            self._initialize_components()
        
        self.logger.info(f"🚀 Training enhanced RF model with {len(training_data)} samples")
        
        try:
            # Generate features if raw OHLCV data is provided
            if 'close' in training_data.columns and len(self.feature_pipeline.get_feature_names()) == 0:
                self.logger.info("🔧 Generating features from OHLCV data")
                training_data = self.feature_pipeline.transform(training_data)
            
            # Split data for validation
            split_idx = int(len(training_data) * 0.8)
            X_train = training_data.iloc[:split_idx]
            y_train = labels.iloc[:split_idx]
            X_val = training_data.iloc[split_idx:]
            y_val = labels.iloc[split_idx:]
            
            # Optimize hyperparameters if requested
            if optimize_hyperparameters and self.optimizer is not None:
                self.logger.info("🎯 Optimizing hyperparameters...")
                
                optimization_results = self.optimizer.optimize(
                    X_train, y_train, X_val, y_val,
                    n_trials=50,
                    scoring_metric='f1_weighted'
                )
                
                # Train with best parameters
                if 'best_params' in optimization_results:
                    self.model.model = self.model.model.__class__(
                        **optimization_results['best_params'], 
                        random_state=42
                    )
            
            # Train the model
            validation_metrics = self.model.train(
                X_train, y_train, 
                validation_data=(X_val, y_val)
            )
            
            # Save the trained model
            self.model.save()
            
            # Evaluate on full dataset
            full_metrics = self.model.evaluate(training_data, labels)
            
            results = {
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'features_generated': len(training_data.columns),
                'validation_metrics': validation_metrics,
                'full_dataset_metrics': full_metrics,
                'hyperparameter_optimization': optimize_hyperparameters,
                'training_time': datetime.now().isoformat(),
                'model_path': self.model_path
            }
            
            self.logger.info(f"✅ Enhanced RF training completed!")
            self.logger.info(f"📊 Validation accuracy: {validation_metrics.get('accuracy', 0):.3f}")
            self.logger.info(f"📊 Full dataset accuracy: {full_metrics.get('accuracy', 0):.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced RF training failed: {e}")
            return {'error': str(e)}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics."""
        
        if self.model is None:
            return {'error': 'Model not available'}
        
        try:
            # Get model statistics
            model_stats = self.model.get_model_stats()
            
            # Calculate recent prediction statistics
            recent_predictions = self.prediction_history[-100:] if len(self.prediction_history) > 0 else []
            
            performance = {
                'model_stats': model_stats,
                'recent_performance': {
                    'predictions_count': len(recent_predictions),
                    'avg_confidence': np.mean([p['confidence'] for p in recent_predictions]) if recent_predictions else 0,
                    'confidence_trend': self._calculate_confidence_trend(recent_predictions),
                    'class_distribution': self._calculate_class_distribution(recent_predictions)
                },
                'feature_pipeline': {
                    'feature_groups': self.feature_pipeline.get_feature_names(),
                    'total_features': self.feature_pipeline.get_feature_count()
                },
                'integration_settings': {
                    'ensemble_weight': self.ensemble_weight,
                    'confidence_threshold': self.confidence_threshold,
                    'auto_retrain': self.auto_retrain,
                    'monitoring_enabled': self.enable_monitoring
                }
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to get model performance: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_trend(self, predictions: List[Dict]) -> str:
        """Calculate confidence trend from recent predictions."""
        
        if len(predictions) < 10:
            return 'insufficient_data'
        
        # Compare first half vs second half
        mid_point = len(predictions) // 2
        first_half_conf = np.mean([p['confidence'] for p in predictions[:mid_point]])
        second_half_conf = np.mean([p['confidence'] for p in predictions[mid_point:]])
        
        diff = second_half_conf - first_half_conf
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_class_distribution(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate class distribution from recent predictions."""
        
        if not predictions:
            return {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        
        class_counts = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        class_names = ['SELL', 'HOLD', 'BUY']  # Matches model output order
        
        for pred in predictions:
            pred_class = pred.get('prediction', 1)  # Default to HOLD
            if 0 <= pred_class < len(class_names):
                class_counts[class_names[pred_class]] += 1
        
        total = len(predictions)
        return {k: v/total for k, v in class_counts.items()}
    
    def export_model_insights(self, output_path: str = "model_insights") -> Dict[str, str]:
        """Export model insights and explanations."""
        
        if self.model is None:
            return {'error': 'Model not available'}
        
        try:
            os.makedirs(output_path, exist_ok=True)
            
            exported_files = {}
            
            # Export feature importance
            feature_importance = self.model.get_feature_importance()
            if not feature_importance.empty:
                importance_file = os.path.join(output_path, 'feature_importance.csv')
                feature_importance.to_csv(importance_file, index=False)
                exported_files['feature_importance'] = importance_file
            
            # Export model performance
            performance = self.get_model_performance()
            performance_file = os.path.join(output_path, 'model_performance.json')
            import json
            with open(performance_file, 'w') as f:
                json.dump(performance, f, indent=2, default=str)
            exported_files['performance'] = performance_file
            
            # Export prediction history
            if self.prediction_history:
                history_df = pd.DataFrame(self.prediction_history)
                history_file = os.path.join(output_path, 'prediction_history.csv')
                history_df.to_csv(history_file, index=False)
                exported_files['prediction_history'] = history_file
            
            self.logger.info(f"📊 Model insights exported to {output_path}")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Failed to export model insights: {e}")
            return {'error': str(e)}
    
    def update_configuration(self, config: Dict[str, Any]):
        """Update integration configuration."""
        
        if 'ensemble_weight' in config:
            self.ensemble_weight = float(config['ensemble_weight'])
        
        if 'confidence_threshold' in config:
            self.confidence_threshold = float(config['confidence_threshold'])
        
        if 'auto_retrain' in config:
            self.auto_retrain = bool(config['auto_retrain'])
        
        if 'retrain_check_interval_hours' in config:
            hours = int(config['retrain_check_interval_hours'])
            self.retrain_check_interval = timedelta(hours=hours)
        
        self.logger.info(f"✅ Integration configuration updated: {config}")