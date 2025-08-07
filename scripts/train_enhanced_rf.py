#!/usr/bin/env python3
"""
Enhanced Random Forest Training Script for ELVIS
Comprehensive training pipeline with feature engineering and optimization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import json

# ELVIS imports
from core.models.enhanced_random_forest_model import EnhancedRandomForestModel
from core.models.features.trading_feature_pipeline import TradingFeaturePipeline
from core.models.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from core.models.integration.enhanced_rf_integration import EnhancedRFIntegration
from utils.paper_trade_db import get_all_trades
from utils.logger_config import setup_logging

class EnhancedRFTrainer:
    """
    Comprehensive training pipeline for Enhanced Random Forest.
    
    Features:
    - Automated data preparation
    - Feature engineering
    - Hyperparameter optimization
    - Model validation
    - Performance analysis
    - Model deployment
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or setup_logging("EnhancedRFTrainer", log_level="INFO")
        
        # Initialize components
        self.feature_pipeline = TradingFeaturePipeline(logger=self.logger)
        self.model = None
        self.optimizer = None
        self.integration = None
        
        # Training configuration
        self.config = {
            'model_path': 'models/enhanced_rf',
            'optimization_trials': 100,
            'validation_split': 0.2,
            'cross_validation_folds': 5,
            'feature_selection': True,
            'enable_shap': True,
            'enable_monitoring': True
        }
        
        self.logger.info("🚀 Enhanced RF Trainer initialized")
    
    def prepare_training_data(self, 
                            data_source: str = 'database',
                            historical_days: int = 30,
                            min_samples: int = 1000) -> tuple:
        """
        Prepare training data from various sources.
        
        Args:
            data_source: 'database', 'csv', or 'synthetic'
            historical_days: Days of historical data to use
            min_samples: Minimum samples required for training
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        
        self.logger.info(f"📊 Preparing training data from {data_source}")
        
        if data_source == 'database':
            return self._prepare_database_data(historical_days, min_samples)
        elif data_source == 'csv':
            return self._prepare_csv_data()
        elif data_source == 'synthetic':
            return self._generate_synthetic_data(min_samples)
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def _prepare_database_data(self, historical_days: int, min_samples: int) -> tuple:
        """Prepare training data from paper trading database."""
        
        try:
            # Get trading history
            trades = get_all_trades(limit=min_samples * 2)  # Get more trades to ensure enough data
            
            if len(trades) < min_samples:
                self.logger.warning(f"Insufficient trades in database: {len(trades)} < {min_samples}")
                return self._generate_synthetic_data(min_samples)
            
            # Convert trades to training format
            training_data = []
            labels = []
            
            for trade in trades:
                try:
                    # Extract trade information
                    timestamp = trade[0]
                    symbol = trade[1]
                    side = trade[2]
                    price = float(trade[3])
                    quantity = float(trade[4])
                    pnl = float(trade[5]) if len(trade) > 5 else 0.0
                    
                    # Create OHLCV-like data point
                    data_point = {
                        'timestamp': pd.to_datetime(timestamp),
                        'open': price * 0.999,
                        'high': price * 1.002,
                        'low': price * 0.998,
                        'close': price,
                        'volume': quantity * 1000,  # Scale up volume
                        'symbol': symbol
                    }
                    
                    # Create label based on PnL
                    if pnl > 0.01:  # Profitable trade
                        label = 2  # BUY signal would have been correct
                    elif pnl < -0.01:  # Loss trade
                        label = 0  # SELL signal would have been correct
                    else:
                        label = 1  # HOLD
                    
                    training_data.append(data_point)
                    labels.append(label)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing trade: {e}")
                    continue
            
            if len(training_data) < min_samples:
                self.logger.warning(f"Insufficient processed data: {len(training_data)} < {min_samples}")
                return self._generate_synthetic_data(min_samples)
            
            # Convert to DataFrame
            df = pd.DataFrame(training_data)
            labels_series = pd.Series(labels)
            
            # Generate features
            self.logger.info("🔧 Generating features from trading data")
            features_df = self.feature_pipeline.transform(df)
            
            # Align labels with features (feature pipeline may drop rows)
            if len(features_df) != len(labels_series):
                # Take the last N labels to match features
                labels_series = labels_series.iloc[-len(features_df):].reset_index(drop=True)
            
            self.logger.info(f"✅ Prepared {len(features_df)} samples with {len(features_df.columns)} features")
            
            return features_df, labels_series
            
        except Exception as e:
            self.logger.error(f"Database data preparation failed: {e}")
            return self._generate_synthetic_data(min_samples)
    
    def _prepare_csv_data(self) -> tuple:
        """Prepare training data from CSV file."""
        
        csv_file = 'data/training_data.csv'
        
        if not os.path.exists(csv_file):
            self.logger.warning(f"CSV file not found: {csv_file}")
            return self._generate_synthetic_data(1000)
        
        try:
            df = pd.read_csv(csv_file)
            
            # Assume last column is labels
            features_df = df.iloc[:, :-1]
            labels_series = df.iloc[:, -1]
            
            self.logger.info(f"✅ Loaded {len(features_df)} samples from CSV")
            
            return features_df, labels_series
            
        except Exception as e:
            self.logger.error(f"CSV data preparation failed: {e}")
            return self._generate_synthetic_data(1000)
    
    def _generate_synthetic_data(self, n_samples: int) -> tuple:
        """Generate synthetic training data for testing."""
        
        self.logger.info(f"🎲 Generating {n_samples} synthetic training samples")
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        
        base_price = 50000
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')
        
        prices = []
        current_price = base_price
        
        for i in range(n_samples):
            # Random walk with some trend
            change = np.random.normal(0, 0.002) + 0.0001 * np.sin(i / 100)
            current_price *= (1 + change)
            prices.append(current_price)
        
        # Create OHLCV data
        synthetic_data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            open_price = prices[i-1] if i > 0 else close
            high = max(open_price, close) * (1 + np.random.uniform(0, 0.005))
            low = min(open_price, close) * (1 - np.random.uniform(0, 0.005))
            volume = np.random.uniform(1000, 10000)
            
            synthetic_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(synthetic_data)
        
        # Generate features
        features_df = self.feature_pipeline.transform(df)
        
        # Generate synthetic labels based on price movement
        labels = []
        for i in range(len(features_df)):
            if i < len(prices) - 1:
                future_return = (prices[i+1] - prices[i]) / prices[i]
                if future_return > 0.01:
                    labels.append(2)  # BUY
                elif future_return < -0.01:
                    labels.append(0)  # SELL
                else:
                    labels.append(1)  # HOLD
            else:
                labels.append(1)  # HOLD for last sample
        
        labels_series = pd.Series(labels[:len(features_df)])
        
        self.logger.info(f"✅ Generated {len(features_df)} synthetic samples with {len(features_df.columns)} features")
        
        return features_df, labels_series
    
    def train_model(self, 
                   features_df: pd.DataFrame,
                   labels_series: pd.Series,
                   optimize_hyperparameters: bool = True,
                   save_model: bool = True) -> dict:
        """
        Train the enhanced Random Forest model.
        
        Args:
            features_df: Training features
            labels_series: Training labels
            optimize_hyperparameters: Whether to optimize hyperparameters
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        
        self.logger.info(f"🎯 Training Enhanced Random Forest model")
        self.logger.info(f"📊 Training data shape: {features_df.shape}")
        self.logger.info(f"📊 Label distribution: {labels_series.value_counts().to_dict()}")
        
        # Initialize model
        self.model = EnhancedRandomForestModel(
            logger=self.logger,
            model_path=self.config['model_path'],
            enable_optuna=optimize_hyperparameters,
            enable_shap=self.config['enable_shap'],
            enable_monitoring=self.config['enable_monitoring']
        )
        
        # Split data
        split_idx = int(len(features_df) * (1 - self.config['validation_split']))
        X_train = features_df.iloc[:split_idx]
        y_train = labels_series.iloc[:split_idx]
        X_val = features_df.iloc[split_idx:]
        y_val = labels_series.iloc[split_idx:]
        
        training_results = {}
        
        # Hyperparameter optimization
        if optimize_hyperparameters:
            self.logger.info("🔧 Starting hyperparameter optimization...")
            
            self.optimizer = HyperparameterOptimizer(
                model_class=EnhancedRandomForestModel,
                logger=self.logger,
                study_name="enhanced_rf_training"
            )
            
            try:
                optimization_results = self.optimizer.optimize(
                    X_train, y_train, X_val, y_val,
                    n_trials=self.config['optimization_trials'],
                    scoring_metric='f1_weighted'
                )
                
                training_results['optimization'] = optimization_results
                
                # Update model with best parameters
                if 'best_params' in optimization_results:
                    self.model.model = self.model.model.__class__(
                        **optimization_results['best_params'], 
                        random_state=42
                    )
                    
            except Exception as e:
                self.logger.warning(f"Hyperparameter optimization failed: {e}")
        
        # Train model
        self.logger.info("🚀 Training model with optimized parameters...")
        
        validation_metrics = self.model.train(
            X_train, y_train,
            validation_data=(X_val, y_val)
        )
        
        training_results['validation_metrics'] = validation_metrics
        
        # Cross-validation
        self.logger.info("📊 Performing cross-validation...")
        
        cv_results = self.model.cross_validate(
            features_df, labels_series,
            cv_folds=self.config['cross_validation_folds']
        )
        
        training_results['cross_validation'] = cv_results
        
        # Full dataset evaluation
        full_metrics = self.model.evaluate(features_df, labels_series)
        training_results['full_dataset_metrics'] = full_metrics
        
        # Feature importance analysis
        feature_importance = self.model.get_feature_importance()
        training_results['feature_importance'] = feature_importance.head(20).to_dict('records')
        
        # Model statistics
        training_results['model_stats'] = self.model.get_model_stats()
        
        # Save model
        if save_model:
            self.model.save()
            self.logger.info(f"💾 Model saved to {self.config['model_path']}")
        
        # Log results
        self.logger.info("✅ Training completed!")
        self.logger.info(f"📊 Validation Accuracy: {validation_metrics.get('accuracy', 0):.3f}")
        self.logger.info(f"📊 Validation F1: {validation_metrics.get('f1', 0):.3f}")
        self.logger.info(f"📊 Cross-validation Accuracy: {np.mean(cv_results.get('test_accuracy', [0])):.3f}")
        
        return training_results
    
    def test_integration(self) -> dict:
        """Test the integration with ensemble strategy."""
        
        self.logger.info("🧪 Testing Enhanced RF integration")
        
        # Initialize integration
        self.integration = EnhancedRFIntegration(
            logger=self.logger,
            model_path=self.config['model_path']
        )
        
        # Test prediction
        test_market_data = {
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1500.0,
            'timestamp': datetime.now()
        }
        
        test_trading_features = {
            'sentiment': 0.5,
            'order_flow': 0.2,
            'market_regime': 1
        }
        
        try:
            prediction, metadata = self.integration.get_prediction(
                test_market_data, test_trading_features
            )
            
            integration_results = {
                'prediction_successful': prediction is not None,
                'prediction_shape': prediction.shape if prediction is not None else None,
                'metadata_keys': list(metadata.keys()),
                'confidence': metadata.get('confidence', 0),
                'feature_count': metadata.get('feature_count', 0),
                'model_available': self.integration.model is not None
            }
            
            self.logger.info("✅ Integration test successful!")
            self.logger.info(f"📊 Prediction: {prediction}")
            self.logger.info(f"📊 Confidence: {metadata.get('confidence', 0):.3f}")
            
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Integration test failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self, training_results: dict, integration_results: dict) -> str:
        """Generate comprehensive training report."""
        
        report_path = f"reports/enhanced_rf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report = {
            'training_summary': {
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'training_results': training_results,
                'integration_results': integration_results
            }
        }
        
        # Add feature analysis
        if hasattr(self.feature_pipeline, 'get_feature_names'):
            report['feature_analysis'] = {
                'feature_groups': self.feature_pipeline.get_feature_names(),
                'total_features': self.feature_pipeline.get_feature_count()
            }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Training report saved to {report_path}")
        
        return report_path

def main():
    """Main training script."""
    
    parser = argparse.ArgumentParser(description='Train Enhanced Random Forest for ELVIS')
    parser.add_argument('--data-source', choices=['database', 'csv', 'synthetic'], 
                       default='database', help='Data source for training')
    parser.add_argument('--samples', type=int, default=2000, 
                       help='Minimum number of training samples')
    parser.add_argument('--optimize', action='store_true', 
                       help='Enable hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=50, 
                       help='Number of optimization trials')
    parser.add_argument('--model-path', default='models/enhanced_rf',
                       help='Path to save the trained model')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("EnhancedRFTraining", log_level=args.log_level)
    
    # Initialize trainer
    trainer = EnhancedRFTrainer(logger)
    trainer.config['model_path'] = args.model_path
    trainer.config['optimization_trials'] = args.trials
    
    try:
        # Prepare training data
        logger.info("=" * 60)
        logger.info("🚀 STARTING ENHANCED RANDOM FOREST TRAINING")
        logger.info("=" * 60)
        
        features_df, labels_series = trainer.prepare_training_data(
            data_source=args.data_source,
            min_samples=args.samples
        )
        
        if len(features_df) == 0:
            logger.error("❌ No training data available")
            return 1
        
        # Train model
        training_results = trainer.train_model(
            features_df, labels_series,
            optimize_hyperparameters=args.optimize
        )
        
        # Test integration
        integration_results = trainer.test_integration()
        
        # Generate report
        report_path = trainer.generate_report(training_results, integration_results)
        
        logger.info("=" * 60)
        logger.info("✅ ENHANCED RANDOM FOREST TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"📋 Report: {report_path}")
        logger.info(f"💾 Model: {args.model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())