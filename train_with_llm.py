#!/usr/bin/env python3
"""
LLM-Enhanced Training Script for ELVIS Trading System
Integrates local LLM analysis directly into the training pipeline
"""

import os
import sys
import logging
import argparse
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from training.llm_enhanced_trainer import LLMEnhancedTrainer, create_llm_config_from_env
from utils.logging_utils import setup_logger

def setup_llm_environment():
    """Setup environment variables for LLM integration"""
    
    # LLM configuration
    if not os.getenv('ELVIS_LLM_PROVIDER'):
        os.environ['ELVIS_LLM_PROVIDER'] = 'local'
    if not os.getenv('ELVIS_LLM_MODEL'):
        os.environ['ELVIS_LLM_MODEL'] = 'openai/gpt-oss-20b'
    if not os.getenv('ELVIS_LLM_BASE_URL'):
        os.environ['ELVIS_LLM_BASE_URL'] = 'http://localhost:1234/v1'
    if not os.getenv('ELVIS_LLM_TEMPERATURE'):
        os.environ['ELVIS_LLM_TEMPERATURE'] = '0.3'
    if not os.getenv('ELVIS_LLM_TIMEOUT'):
        os.environ['ELVIS_LLM_TIMEOUT'] = '30'
    
    # Training configuration
    os.environ['VAULT_ENABLED'] = 'false'
    os.environ['USE_VAULT'] = 'false'
    os.environ['VAULT_AVAILABLE'] = 'false'
    
    # Safe training credentials
    os.environ['BINANCE_API_KEY'] = 'training_with_llm_key'
    os.environ['BINANCE_API_SECRET'] = 'training_with_llm_secret'

def load_training_data(limit: int = 1000) -> pd.DataFrame:
    """Load and prepare training data"""
    
    logger = setup_logger('data_loader')
    logger.info(f"📊 Loading training data (limit: {limit})...")
    
    try:
        # Try to load from multiple sources, prioritizing real trading data
        data_sources = [
            'data/processed/trade_history/trade_features.csv',  # Real trading data
            'data/processed/trade_features.csv',
            'data/processed/trade_data.csv',
            'data/trade_data.csv'
        ]
        
        df = None
        for source in data_sources:
            if os.path.exists(source):
                logger.info(f"📂 Loading data from: {source}")
                df = pd.read_csv(source)
                break
        
        if df is None:
            # Generate synthetic data for testing
            logger.warning("⚠️ No data files found - generating synthetic data for testing")
            df = generate_synthetic_data(limit)
        else:
            # We have real data - convert to OHLCV format if needed
            df = prepare_real_trading_data(df, logger)
            
            # Try to load corresponding targets if available
            targets_file = source.replace('trade_features.csv', 'trade_targets.csv')
            if os.path.exists(targets_file):
                logger.info(f"📊 Loading targets from: {targets_file}")
                targets_df = pd.read_csv(targets_file)
                if len(targets_df) == len(df):
                    # Merge targets with features
                    for col in targets_df.columns:
                        df[col] = targets_df[col].values
                    logger.info(f"✅ Merged {len(targets_df.columns)} target columns")
                else:
                    logger.warning(f"⚠️ Target file length mismatch: {len(targets_df)} vs {len(df)}")
            else:
                logger.info("📊 No target file found, will create targets from price data")
        
        # Limit data if requested
        if limit > 0 and len(df) > limit:
            df = df.tail(limit)  # Use most recent data
            logger.info(f"📏 Limited data to most recent {limit} samples")
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Some OHLCV columns missing: {missing_columns}")
            # Add missing columns with reasonable defaults
            for col in missing_columns:
                if col == 'volume':
                    df[col] = df.get('quantity', 1000000)  # Use quantity or default
                else:
                    df[col] = df.get('price', df.get('close', 65000))  # Use price as fallback
        
        # Add technical indicators if missing
        df = add_technical_indicators(df)
        
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} features")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        raise

def prepare_real_trading_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Convert real trading data to OHLCV format for analysis"""
    
    logger.info("🔄 Converting real trading data to OHLCV format...")
    
    # Create OHLCV columns from trading data
    if 'price' in df.columns:
        df['open'] = df['price'].shift(1).fillna(df['price'])
        df['high'] = df[['open', 'price']].max(axis=1) * 1.002  # Small spread
        df['low'] = df[['open', 'price']].min(axis=1) * 0.998
        df['close'] = df['price']
    
    if 'quantity' in df.columns:
        df['volume'] = df['quantity'] * df.get('price', 65000)
    else:
        df['volume'] = 1000000  # Default volume
    
    # Ensure we have basic price movement data
    if 'price_change' not in df.columns and 'close' in df.columns:
        df['price_change_pct'] = df['close'].pct_change().fillna(0)
    elif 'price_change' in df.columns:
        df['price_change_pct'] = df['price_change'] / df['close'] 
    
    # Add volatility if missing
    if 'price_volatility' not in df.columns:
        df['volatility'] = df['price_change_pct'].rolling(20).std() * 100
    else:
        df['volatility'] = df['price_volatility']
    
    logger.info(f"✅ Converted to OHLCV format with {len(df.columns)} features")
    return df

def generate_synthetic_data(num_samples: int) -> pd.DataFrame:
    """Generate synthetic market data for testing"""
    
    # Generate realistic-looking OHLCV data
    base_price = 65000
    timestamps = pd.date_range(start='2024-01-01', periods=num_samples, freq='1H')
    
    # Random walk for price
    returns = np.random.normal(0, 0.002, num_samples)  # 0.2% hourly volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Add some realistic intraday movement
        high_factor = 1 + abs(np.random.normal(0, 0.005))
        low_factor = 1 - abs(np.random.normal(0, 0.005))
        
        open_price = prices[i-1] if i > 0 else close
        high_price = max(open_price, close) * high_factor
        low_price = min(open_price, close) * low_factor
        volume = np.random.uniform(500000, 2000000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataset"""
    
    logger = setup_logger('tech_indicators')
    logger.info("📈 Computing technical indicators...")
    
    try:
        # Price-based indicators
        df['price_change_pct'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['price_change_pct'].rolling(24).std() * 100  # 24-hour rolling volatility
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        bb_middle = df['close'].rolling(bb_window).mean()
        bb_std_dev = df['close'].rolling(bb_window).std()
        df['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
        df['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
        df['bb_middle'] = bb_middle
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Added technical indicators: {len(df.columns)} total features")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to compute technical indicators: {e}")
        raise

def create_training_targets(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create training targets for prediction"""
    
    logger = setup_logger('targets')
    logger.info(f"🎯 Creating training targets (horizon: {horizon} periods)...")
    
    # Check if we have existing targets from real trading data
    if 'future_profitable' in df.columns:
        logger.info("📊 Using existing real trading targets")
        df['target_classification'] = df['future_profitable'].astype(int)
        if 'future_pnl' in df.columns:
            df['target_regression'] = df['future_pnl'] / 1000.0  # Normalize PnL
        else:
            df['target_regression'] = df['future_price_up'].astype(float) - 0.5  # Convert to [-0.5, 0.5]
    else:
        logger.info("🔄 Creating new prediction targets from price data")
        # Future price movement prediction
        df['future_price'] = df['close'].shift(-horizon)
        df['price_direction'] = (df['future_price'] > df['close']).astype(int)
        df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
        
        # Classification target: 1 if price goes up more than 0.5%, 0 otherwise
        df['target_classification'] = ((df['price_change_future'] > 0.005).astype(int))
        
        # Regression target: normalized price change
        df['target_regression'] = df['price_change_future']
        
        # Remove rows without future data
        df = df[:-horizon]
    
    # Ensure targets are clean
    df = df.dropna(subset=['target_classification', 'target_regression'])
    
    logger.info(f"✅ Created targets for {len(df)} samples")
    logger.info(f"   Classification distribution: {df['target_classification'].value_counts().to_dict()}")
    logger.info(f"   Regression target range: [{df['target_regression'].min():.6f}, {df['target_regression'].max():.6f}]")
    
    return df

async def run_llm_enhanced_training(limit: int, epochs: int, horizon: int, 
                                  enable_llm: bool, debug: bool):
    """Run the complete LLM-enhanced training pipeline"""
    
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    logger = setup_logger('llm_training', log_level=log_level)
    
    logger.info("🚀 Starting LLM-Enhanced Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"📊 Configuration:")
    logger.info(f"   Data Limit: {limit}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Prediction Horizon: {horizon}")
    logger.info(f"   LLM Enabled: {enable_llm}")
    logger.info(f"   Debug Mode: {debug}")
    
    try:
        # 1. Load and prepare data
        logger.info("\n📂 Step 1: Loading training data...")
        df = load_training_data(limit)
        
        # 2. Create targets
        logger.info("\n🎯 Step 2: Creating training targets...")
        df = create_training_targets(df, horizon)
        
        # 3. LLM enhancement
        if enable_llm:
            logger.info("\n🧠 Step 3: LLM Feature Enhancement...")
            llm_config = create_llm_config_from_env()
            logger.info(f"   LLM Provider: {llm_config.provider}")
            logger.info(f"   LLM Model: {llm_config.model}")
            logger.info(f"   LLM URL: {llm_config.base_url}")
            
            # Create LLM-enhanced trainer
            llm_trainer = LLMEnhancedTrainer(
                llm_config=llm_config,
                enable_llm=True,
                cache_llm_responses=True
            )
            
            # Load existing cache if available
            cache_file = 'llm_training_cache.json'
            if os.path.exists(cache_file):
                llm_trainer.load_llm_cache(cache_file)
            
            # Enhance data with LLM features (process in small batches)
            batch_size = 5 if len(df) > 100 else 2
            logger.info(f"   Processing in batches of {batch_size}")
            
            df = llm_trainer.prepare_llm_features(df, batch_size=batch_size)
            
            # Save cache for future runs
            llm_trainer.save_llm_cache(cache_file)
            
        else:
            logger.info("\n📊 Step 3: Skipping LLM enhancement (disabled)")
        
        # 4. Prepare features and targets
        logger.info("\n🔧 Step 4: Preparing final dataset...")
        
        # Select feature columns (exclude metadata columns and string columns)
        exclude_columns = ['timestamp', 'future_price', 'price_change_future', 'side']
        
        # Get only numeric columns for training
        feature_columns = []
        for col in df.columns:
            if (col not in exclude_columns 
                and not col.startswith('target_') 
                and not col.startswith('future_')):
                try:
                    # Test if column is numeric
                    pd.to_numeric(df[col], errors='raise')
                    feature_columns.append(col)
                except (ValueError, TypeError):
                    logger.debug(f"Skipping non-numeric column: {col}")
        
        logger.info(f"   Selected {len(feature_columns)} numeric feature columns")
        
        # Clean the feature data
        X_df = df[feature_columns].copy()
        
        # Replace infinity values
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column medians
        X_df = X_df.fillna(X_df.median())
        
        # Final check for any remaining NaN/inf values
        if X_df.isnull().any().any():
            logger.warning("⚠️ Still have NaN values, filling with 0")
            X_df = X_df.fillna(0)
        
        if not np.isfinite(X_df.values).all():
            logger.warning("⚠️ Still have infinite values, clipping to reasonable range")
            X_df = X_df.clip(-1e10, 1e10)
        
        X = X_df.values
        y_classification = df['target_classification'].values
        y_regression = df['target_regression'].values
        
        logger.info(f"   Features shape: {X.shape}")
        logger.info(f"   Classification targets: {len(y_classification)}")
        logger.info(f"   Regression targets: {len(y_regression)}")
        
        if enable_llm:
            llm_features = [col for col in feature_columns if col.startswith('llm_')]
            logger.info(f"   LLM-derived features: {len(llm_features)}")
            if debug:
                logger.debug(f"   LLM features: {llm_features}")
        
        # 5. Train models
        logger.info("\n🏗️ Step 5: Training models...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
        from sklearn.preprocessing import StandardScaler
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_classification, y_regression, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classification model
        logger.info("   Training classification model...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_class_train)
        
        # Evaluate classification
        y_pred_class = clf.predict(X_test_scaled)
        class_accuracy = accuracy_score(y_class_test, y_pred_class)
        logger.info(f"   Classification Accuracy: {class_accuracy:.4f}")
        
        # Train regression model
        logger.info("   Training regression model...")
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train_scaled, y_reg_train)
        
        # Evaluate regression
        y_pred_reg = regressor.predict(X_test_scaled)
        reg_mse = mean_squared_error(y_reg_test, y_pred_reg)
        logger.info(f"   Regression MSE: {reg_mse:.6f}")
        
        # Feature importance (if debug mode)
        if debug and enable_llm:
            logger.debug("\n🔍 Feature Importance Analysis:")
            feature_importance = clf.feature_importances_
            feature_names = feature_columns
            
            # Sort by importance
            importance_data = list(zip(feature_names, feature_importance))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug("   Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(importance_data[:10]):
                marker = "🧠" if feature.startswith('llm_') else "📊"
                logger.debug(f"   {i+1:2d}. {marker} {feature}: {importance:.4f}")
        
        # 6. Save results
        logger.info("\n💾 Step 6: Saving results...")
        
        # Create models directory
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Save models
        import joblib
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_files = {
            'classifier': f'models/llm_enhanced_classifier_{timestamp}.joblib',
            'regressor': f'models/llm_enhanced_regressor_{timestamp}.joblib',
            'scaler': f'models/llm_enhanced_scaler_{timestamp}.joblib'
        }
        
        joblib.dump(clf, model_files['classifier'])
        joblib.dump(regressor, model_files['regressor'])
        joblib.dump(scaler, model_files['scaler'])
        
        logger.info(f"   Models saved:")
        for model_type, filepath in model_files.items():
            logger.info(f"     {model_type}: {filepath}")
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'training_config': {
                'limit': limit,
                'epochs': epochs,
                'horizon': horizon,
                'llm_enabled': enable_llm
            },
            'data_info': {
                'total_samples': len(df),
                'total_features': len(feature_columns),
                'llm_features': len([col for col in feature_columns if col.startswith('llm_')]) if enable_llm else 0
            },
            'model_performance': {
                'classification_accuracy': float(class_accuracy),
                'regression_mse': float(reg_mse)
            },
            'model_files': model_files
        }
        
        if enable_llm:
            metadata['llm_config'] = {
                'provider': llm_config.provider,
                'model': llm_config.model,
                'base_url': llm_config.base_url
            }
        
        metadata_file = f'models/training_metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"   Metadata saved: {metadata_file}")
        
        # 7. Summary
        logger.info("\n🎉 Training Complete!")
        logger.info("=" * 60)
        logger.info(f"✅ Classification Accuracy: {class_accuracy:.4f}")
        logger.info(f"✅ Regression MSE: {reg_mse:.6f}")
        logger.info(f"✅ Total Features Used: {len(feature_columns)}")
        if enable_llm:
            llm_count = len([col for col in feature_columns if col.startswith('llm_')])
            logger.info(f"🧠 LLM Features Used: {llm_count}")
        
        logger.info(f"📁 Results saved in: models/")
        logger.info(f"📊 Training metadata: {metadata_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if debug:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='LLM-Enhanced ELVIS Training')
    parser.add_argument('--limit', type=int, default=1000, 
                       help='Maximum number of samples to use')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                       help='Prediction horizon in periods')
    parser.add_argument('--enable-llm', action='store_true', default=True,
                       help='Enable LLM feature enhancement')
    parser.add_argument('--disable-llm', action='store_true',
                       help='Disable LLM feature enhancement')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Handle LLM enable/disable flags
    enable_llm = args.enable_llm and not args.disable_llm
    
    # Setup environment
    setup_llm_environment()
    
    print("🧠 LLM-Enhanced ELVIS Training")
    print("=" * 40)
    print(f"📊 Samples: {args.limit}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"🎯 Horizon: {args.prediction_horizon}")
    print(f"🧠 LLM Enabled: {enable_llm}")
    print(f"🐛 Debug: {args.debug}")
    
    if enable_llm:
        print(f"🤖 LLM Model: {os.getenv('ELVIS_LLM_MODEL')}")
        print(f"🔗 LLM URL: {os.getenv('ELVIS_LLM_BASE_URL')}")
    
    print()
    
    # Run training
    try:
        success = asyncio.run(run_llm_enhanced_training(
            limit=args.limit,
            epochs=args.epochs,
            horizon=args.prediction_horizon,
            enable_llm=enable_llm,
            debug=args.debug
        ))
        
        if success:
            print("\n🎉 Training completed successfully!")
            if enable_llm:
                print("🧠 Your local LLM has been integrated into the model!")
            print("📁 Check the models/ directory for results")
        else:
            print("\n❌ Training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()