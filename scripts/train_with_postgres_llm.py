#!/usr/bin/env python3
"""
Train LLM-enhanced models using all available PostgreSQL paper trading data.
"""

# Make the repo root importable no matter where this script is run from
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from load_postgres_training_data import (
    enhance_postgres_training_data,
    load_all_postgres_trades,
    postgres_trades_to_ohlcv,
)

from training.llm_enhanced_trainer import LLMEnhancedTrainer, create_llm_config_from_env
from utils.logging_utils import setup_logger


def setup_llm_environment():
    """Setup environment variables for LLM integration"""

    # LLM configuration
    if not os.getenv("ELVIS_LLM_PROVIDER"):
        os.environ["ELVIS_LLM_PROVIDER"] = "local"
    if not os.getenv("ELVIS_LLM_MODEL"):
        os.environ["ELVIS_LLM_MODEL"] = "openai/gpt-oss-20b"
    if not os.getenv("ELVIS_LLM_BASE_URL"):
        os.environ["ELVIS_LLM_BASE_URL"] = "http://localhost:1234/v1"
    if not os.getenv("ELVIS_LLM_TEMPERATURE"):
        os.environ["ELVIS_LLM_TEMPERATURE"] = "0.3"
    if not os.getenv("ELVIS_LLM_TIMEOUT"):
        os.environ["ELVIS_LLM_TIMEOUT"] = "30"

    # Training configuration
    os.environ["VAULT_ENABLED"] = "false"
    os.environ["USE_VAULT"] = "false"
    os.environ["VAULT_AVAILABLE"] = "false"

    # Safe training credentials
    os.environ["BINANCE_API_KEY"] = "postgres_training_key"
    os.environ["BINANCE_API_SECRET"] = "postgres_training_secret"


def load_postgres_training_data(
    include_test_trades: bool = True, max_trades: int = 0
) -> pd.DataFrame:
    """Load and prepare all PostgreSQL training data"""

    logger = setup_logger("postgres_data_loader")
    logger.info("🐘 Loading all paper trading data from PostgreSQL...")

    try:
        # Load all trades from PostgreSQL
        trades_df = load_all_postgres_trades(include_test_trades=include_test_trades)

        # Apply trades limit if specified
        if max_trades > 0 and len(trades_df) > max_trades:
            logger.info(
                f"📏 Limiting to most recent {max_trades} trades (from {len(trades_df)} available)"
            )
            trades_df = trades_df.tail(max_trades)

        if trades_df.empty:
            logger.error("❌ No trades found in PostgreSQL!")
            return pd.DataFrame()

        # Convert to OHLCV format
        ohlcv_df = postgres_trades_to_ohlcv(trades_df)

        if ohlcv_df.empty:
            logger.error("❌ No OHLCV data generated!")
            return pd.DataFrame()

        # Enhance with features and targets
        enhanced_df = enhance_postgres_training_data(ohlcv_df)

        logger.info(f"✅ Prepared PostgreSQL training data:")
        logger.info(f"   📊 Original trades: {len(trades_df)}")
        logger.info(f"   📈 OHLCV periods: {len(ohlcv_df)}")
        logger.info(f"   🎯 Training samples: {len(enhanced_df)}")
        logger.info(
            f"   📅 Date range: {enhanced_df['timestamp'].min()} to {enhanced_df['timestamp'].max()}"
        )

        return enhanced_df

    except Exception as e:
        logger.error(f"❌ Failed to load PostgreSQL training data: {e}")
        raise


def create_training_targets(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create training targets based on prediction horizon"""

    logger = setup_logger("target_creator")
    logger.info(f"🎯 Creating training targets (horizon: {horizon} periods)...")

    if len(df) < horizon + 1:
        logger.warning(f"⚠️ Not enough data for horizon {horizon}")
        return df

    # We already have basic targets, but let's enhance them with the specified horizon
    df = df.copy()

    # Future price at specified horizon
    df[f"future_price_{horizon}"] = df["close"].shift(-horizon)
    df[f"future_return_{horizon}"] = (df[f"future_price_{horizon}"] / df["close"]) - 1

    # Classification targets
    df[f"target_up_{horizon}"] = (df[f"future_return_{horizon}"] > 0).astype(int)
    df[f"target_profitable_{horizon}"] = (
        df[f"future_return_{horizon}"] > 0.001
    ).astype(
        int
    )  # >0.1% profit

    # Regression targets
    df[f"target_return_{horizon}"] = df[f"future_return_{horizon}"]

    # Remove rows without future data
    df = df[:-horizon].copy()

    # Use the horizon-specific targets as main targets
    df["target_classification"] = df[f"target_up_{horizon}"]
    df["target_regression"] = df[f"target_return_{horizon}"]

    logger.info(f"✅ Created targets for {len(df)} samples")
    logger.info(
        f"   Classification distribution: {dict(df['target_classification'].value_counts())}"
    )
    logger.info(
        f"   Regression target range: [{df['target_regression'].min():.6f}, {df['target_regression'].max():.6f}]"
    )

    return df


def train_llm_enhanced_models(
    df: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 10,
    debug: bool = False,
    args=None,
):
    """Train LLM-enhanced models on PostgreSQL data"""

    logger = setup_logger("llm_model_trainer")
    logger.info(f"🧠 Training LLM-enhanced models on {len(df)} PostgreSQL samples...")

    # Setup LLM trainer
    llm_config = create_llm_config_from_env()
    trainer = LLMEnhancedTrainer(llm_config)

    # Load existing LLM cache if available
    cache_file = "postgres_llm_cache.json"
    if os.path.exists(cache_file):
        trainer.load_llm_cache(cache_file)
        logger.info(f"📂 Loaded LLM cache from: {cache_file}")

    # Generate LLM features
    logger.info("🧠 Generating LLM features...")
    enhanced_df = trainer.prepare_llm_features(df, batch_size=batch_size)

    # Save LLM cache
    trainer.save_llm_cache(cache_file)
    logger.info(f"💾 Saved LLM cache to: {cache_file}")

    # Prepare features and targets
    feature_columns = [
        col
        for col in enhanced_df.columns
        if col
        not in ["timestamp", "symbol", "target_classification", "target_regression"]
        and not col.startswith("future_")
        and not col.startswith("target_")
    ]

    X = enhanced_df[feature_columns].fillna(0)
    y_classification = enhanced_df["target_classification"]
    y_regression = enhanced_df["target_regression"].fillna(0)

    # Remove infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_regression = y_regression.replace([np.inf, -np.inf], 0)

    logger.info(f"📊 Training data prepared:")
    logger.info(
        f"   Features: {len(feature_columns)} ({len([c for c in feature_columns if c.startswith('llm_')])} LLM features)"
    )
    logger.info(f"   Samples: {len(X)}")
    logger.info(f"   Classification targets: {len(y_classification)}")
    logger.info(f"   Regression targets: {len(y_regression)}")

    if debug:
        logger.debug(f"   Feature columns: {feature_columns}")
        logger.debug(
            f"   LLM features: {[c for c in feature_columns if c.startswith('llm_')]}"
        )

    # Train models
    logger.info("🏗️ Training models...")

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        mean_squared_error,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Split data (80/20)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
        train_test_split(
            X, y_classification, y_regression, test_size=0.2, random_state=42
        )
    )

    logger.info(
        f"📊 Data split: {len(X_train)} training, {len(X_test)} testing samples"
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classification model
    logger.info("   Training classification model...")
    # Scale n_estimators based on epochs (more epochs = more trees)
    n_estimators = min(500, max(100, epochs * 50))

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,  # Deeper trees for more complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,  # Out-of-bag scoring
    )
    classifier.fit(X_train_scaled, y_class_train)

    # Evaluate classification
    class_pred = classifier.predict(X_test_scaled)
    class_accuracy = accuracy_score(y_class_test, class_pred)
    logger.info(f"   Classification Accuracy: {class_accuracy:.4f}")

    # Train regression model
    logger.info("   Training regression model...")
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,  # Deeper trees for more complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,  # Out-of-bag scoring
    )
    regressor.fit(X_train_scaled, y_reg_train)

    # Evaluate regression
    reg_pred = regressor.predict(X_test_scaled)
    reg_mse = mean_squared_error(y_reg_test, reg_pred)
    logger.info(f"   Regression MSE: {reg_mse:.6f}")

    if debug:
        logger.debug("\n🔍 Feature Importance Analysis:")
        logger.debug("   Top 10 Most Important Features:")

        feature_importance = list(zip(feature_columns, classifier.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance[:10]):
            emoji = "🧠" if feature.startswith("llm_") else "📊"
            logger.debug(f"    {i+1:2d}. {emoji} {feature}: {importance:.4f}")

    # Save models
    from datetime import datetime

    import joblib

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_files = {
        "classifier": f"models/postgres_llm_classifier_{timestamp}.joblib",
        "regressor": f"models/postgres_llm_regressor_{timestamp}.joblib",
        "scaler": f"models/postgres_llm_scaler_{timestamp}.joblib",
    }

    os.makedirs("models", exist_ok=True)

    joblib.dump(classifier, model_files["classifier"])
    joblib.dump(regressor, model_files["regressor"])
    joblib.dump(scaler, model_files["scaler"])

    logger.info(f"💾 Models saved:")
    for model_type, filepath in model_files.items():
        logger.info(f"     {model_type}: {filepath}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_source": "postgresql_paper_trades",
        "training_config": {
            "method": "auto",
            "total_trades_requested": (
                "infinite"
                if (args and args.trades == 0)
                else (args.trades if args else "unknown")
            ),
            "total_trades_loaded": len(enhanced_df),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "epochs": epochs,
            "batch_size": batch_size,
            "n_estimators": n_estimators,
            "vault_enabled": args and hasattr(args, "vault") and args.vault,
        },
        "data_info": {
            "total_samples": len(enhanced_df),
            "total_features": len(feature_columns),
            "llm_features": len([c for c in feature_columns if c.startswith("llm_")]),
        },
        "model_performance": {
            "classification_accuracy": float(class_accuracy),
            "regression_mse": float(reg_mse),
        },
        "model_files": model_files,
        "llm_config": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "base_url": llm_config.base_url,
        },
    }

    metadata_file = f"models/postgres_training_metadata_{timestamp}.json"
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"   Metadata saved: {metadata_file}")

    return classifier, regressor, scaler, metadata


def main():
    """Main training function"""

    parser = argparse.ArgumentParser(
        description="Train LLM-enhanced models on PostgreSQL paper trading data"
    )
    parser.add_argument(
        "--method", type=str, default="auto", help="Training method (auto, manual)"
    )
    parser.add_argument(
        "--trades",
        type=int,
        default=0,
        help="Max number of trades (0 = infinite/all available)",
    )
    # Support for legacy parameter names
    parser.add_argument(
        "--limit", type=int, default=0, help="Legacy alias for --trades"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--horizon", type=int, default=5, help="Prediction horizon in periods"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="LLM processing batch size"
    )
    parser.add_argument(
        "--include-test", action="store_true", help="Include TEST trades in training"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--vault", action="store_true", help="Enable Vault for secrets management"
    )

    args = parser.parse_args()

    # Handle legacy parameter mapping
    if args.limit > 0 and args.trades == 0:
        args.trades = args.limit

    # Setup environment based on vault setting
    if args.vault:
        # Enable Vault for secrets management
        os.environ.pop("VAULT_ENABLED", None)
        os.environ.pop("USE_VAULT", None)
        os.environ.pop("VAULT_AVAILABLE", None)
        logger_level = "DEBUG" if args.debug else "INFO"
        logger = setup_logger("postgres_llm_training", logger_level)
        logger.info("🔐 Vault enabled for secrets management")
    else:
        setup_llm_environment()
        # Setup logging
        logger = setup_logger(
            "postgres_llm_training", "DEBUG" if args.debug else "INFO"
        )
        logger.info("🔓 Using environment variables for secrets")

    # Determine trades limit
    trades_text = "infinite (all available)" if args.trades == 0 else str(args.trades)

    logger.info("🚀 Starting PostgreSQL LLM-Enhanced Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"📊 Training Configuration:")
    logger.info(f"   Method: {args.method}")
    logger.info(f"   Trades: {trades_text}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Horizon: {args.horizon}")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Include TEST trades: {args.include_test}")
    logger.info(f"   Debug: {args.debug}")
    logger.info(f"   Vault: {args.vault}")

    try:
        # Step 1: Load PostgreSQL data
        logger.info("\n📂 Step 1: Loading PostgreSQL paper trading data...")
        df = load_postgres_training_data(
            include_test_trades=args.include_test, max_trades=args.trades
        )

        if df.empty:
            logger.error("❌ No data loaded from PostgreSQL!")
            return 1

        # Step 2: Create training targets
        logger.info(
            f"\n🎯 Step 2: Creating training targets (horizon: {args.horizon})..."
        )
        df_with_targets = create_training_targets(df, horizon=args.horizon)

        # Step 3: Train LLM-enhanced models
        logger.info(f"\n🧠 Step 3: Training LLM-enhanced models...")
        classifier, regressor, scaler, metadata = train_llm_enhanced_models(
            df_with_targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            debug=args.debug,
            args=args,
        )

        # Success
        logger.info(f"\n🎉 Training Complete!")
        logger.info("=" * 60)
        logger.info(
            f"✅ Classification Accuracy: {metadata['model_performance']['classification_accuracy']:.4f}"
        )
        logger.info(
            f"✅ Regression MSE: {metadata['model_performance']['regression_mse']:.6f}"
        )
        logger.info(
            f"✅ Total Features Used: {metadata['data_info']['total_features']}"
        )
        logger.info(f"🧠 LLM Features Used: {metadata['data_info']['llm_features']}")
        logger.info(f"📁 Results saved in: models/")
        logger.info(
            f"📊 Training metadata: {[f for f in metadata['model_files'].values() if 'metadata' in f]}"
        )

        print(f"\n🧠 PostgreSQL LLM-Enhanced ELVIS Training")
        print("=" * 50)
        print(f"📊 Training Configuration:")
        print(f"   Method: {metadata['training_config']['method']}")
        print(f"   Trades: {metadata['training_config']['total_trades_requested']}")
        print(f"   Epochs: {metadata['training_config']['epochs']}")
        print(f"   Horizon: {args.horizon}")
        print(f"   Debug: {args.debug}")
        print(f"   Vault: {metadata['training_config']['vault_enabled']}")
        print(f"\n📊 Training Results:")
        print(
            f"   PostgreSQL Trades Loaded: {metadata['training_config']['total_trades_loaded']}"
        )
        print(f"   Training Samples: {metadata['training_config']['training_samples']}")
        print(f"   Test Samples: {metadata['training_config']['test_samples']}")
        print(
            f"   Model Trees (n_estimators): {metadata['training_config']['n_estimators']}"
        )
        print(f"   LLM Features: {metadata['data_info']['llm_features']}")
        print(f"   Total Features: {metadata['data_info']['total_features']}")
        print(f"\n🎯 Performance:")
        print(
            f"   Classification Accuracy: {metadata['model_performance']['classification_accuracy']:.4f}"
        )
        print(
            f"   Regression MSE: {metadata['model_performance']['regression_mse']:.2e}"
        )
        print(f"\n🤖 LLM Configuration:")
        print(f"   Model: {metadata['llm_config']['model']}")
        print(f"   URL: {metadata['llm_config']['base_url']}")
        print(f"\n🎉 Training completed successfully!")
        print(f"🧠 Your local LLM has been integrated with PostgreSQL paper trades!")
        print(f"📁 Check the models/ directory for results")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
