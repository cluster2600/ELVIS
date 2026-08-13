#!/usr/bin/env python3.14
"""
Train LLM-enhanced models using ALL individual paper trades with patient LLM handling.
This version is specifically designed for slow LLMs that need more time.
"""

# Make the repo root importable no matter where this script is run from
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import os
import sys
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import time

from training.llm_enhanced_trainer import LLMEnhancedTrainer, create_llm_config_from_env
from utils.logging_utils import setup_logger
from utils.paper_trade_db import get_conn


def setup_patient_llm_environment():
    """Setup environment variables for patient LLM integration"""

    # Patient LLM configuration - much longer timeouts
    if not os.getenv("ELVIS_LLM_PROVIDER"):
        os.environ["ELVIS_LLM_PROVIDER"] = "local"
    if not os.getenv("ELVIS_LLM_MODEL"):
        os.environ["ELVIS_LLM_MODEL"] = "openai/gpt-oss-20b"
    if not os.getenv("ELVIS_LLM_BASE_URL"):
        os.environ["ELVIS_LLM_BASE_URL"] = "http://localhost:1234/v1"
    if not os.getenv("ELVIS_LLM_TEMPERATURE"):
        os.environ["ELVIS_LLM_TEMPERATURE"] = "0.3"
    if not os.getenv("ELVIS_LLM_TIMEOUT"):
        os.environ["ELVIS_LLM_TIMEOUT"] = "300"  # 5 minutes for very slow LLMs

    # Training configuration
    os.environ["VAULT_ENABLED"] = "false"
    os.environ["USE_VAULT"] = "false"
    os.environ["VAULT_AVAILABLE"] = "false"

    # Safe training credentials
    os.environ["BINANCE_API_KEY"] = "patient_training_key"
    os.environ["BINANCE_API_SECRET"] = "patient_training_secret"


def load_all_individual_trades(include_test_trades=False, max_trades=0):
    """Load all individual trades from PostgreSQL without aggregation"""

    logger = setup_logger("patient_trade_loader")
    logger.info(
        "🐘 Loading ALL individual paper trades from PostgreSQL (Patient Mode)..."
    )

    try:
        conn = get_conn()

        # Build query for all trades
        if include_test_trades:
            query = """
            SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
            FROM trades
            ORDER BY timestamp ASC
            """
            logger.info("📊 Including TEST trades in training data")
        else:
            query = """
            SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
            FROM trades
            WHERE side != 'TEST'
            ORDER BY timestamp ASC
            """
            logger.info("📊 Excluding TEST trades, using only BUY/SELL trades")

        # Apply limit if specified
        if max_trades > 0:
            query += f" LIMIT {max_trades}"
            logger.info(f"📏 Limiting to first {max_trades} trades")

        # Load data
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("⚠️ No trading data found in PostgreSQL!")
            return pd.DataFrame()

        logger.info(f"✅ Loaded {len(df)} individual trades from PostgreSQL")
        logger.info(
            f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )
        logger.info(f"   Symbols: {df['symbol'].unique().tolist()}")
        logger.info(f"   Sides: {df['side'].value_counts().to_dict()}")

        return df

    except Exception as e:
        logger.error(f"❌ Failed to load PostgreSQL trades: {e}")
        raise


def create_trade_level_features(df):
    """Create features for individual trades instead of OHLCV aggregation"""

    logger = setup_logger("patient_feature_engineer")
    logger.info(f"🔧 Creating trade-level features for {len(df)} trades...")

    if df.empty:
        return df

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"])

    # Price-based features (per symbol)
    for symbol in df["symbol"].unique():
        symbol_mask = df["symbol"] == symbol
        symbol_df = df[symbol_mask].copy()

        # Price momentum features
        df.loc[symbol_mask, "price_change"] = symbol_df["price"].pct_change()
        df.loc[symbol_mask, "price_change_abs"] = df.loc[
            symbol_mask, "price_change"
        ].abs()

        # Moving averages (last N trades)
        df.loc[symbol_mask, "price_sma_5"] = (
            symbol_df["price"].rolling(5, min_periods=1).mean()
        )
        df.loc[symbol_mask, "price_sma_10"] = (
            symbol_df["price"].rolling(10, min_periods=1).mean()
        )
        df.loc[symbol_mask, "price_sma_20"] = (
            symbol_df["price"].rolling(20, min_periods=1).mean()
        )

        # Price volatility (rolling standard deviation)
        df.loc[symbol_mask, "price_volatility_5"] = (
            symbol_df["price"].rolling(5, min_periods=1).std()
        )
        df.loc[symbol_mask, "price_volatility_10"] = (
            symbol_df["price"].rolling(10, min_periods=1).std()
        )

        # Position relative to moving averages
        df.loc[symbol_mask, "price_vs_sma5"] = (
            df.loc[symbol_mask, "price"] / df.loc[symbol_mask, "price_sma_5"]
        ) - 1
        df.loc[symbol_mask, "price_vs_sma10"] = (
            df.loc[symbol_mask, "price"] / df.loc[symbol_mask, "price_sma_10"]
        ) - 1

        # RSI-like momentum indicator
        price_changes = df.loc[symbol_mask, "price_change"]
        gains = (
            price_changes.where(price_changes > 0, 0).rolling(14, min_periods=1).mean()
        )
        losses = (
            (-price_changes.where(price_changes < 0, 0))
            .rolling(14, min_periods=1)
            .mean()
        )
        rs = gains / (losses + 1e-10)
        df.loc[symbol_mask, "rsi"] = 100 - (100 / (1 + rs))

        # Volume features
        df.loc[symbol_mask, "volume_sma"] = (
            symbol_df["quantity"].rolling(10, min_periods=1).mean()
        )
        df.loc[symbol_mask, "volume_ratio"] = df.loc[symbol_mask, "quantity"] / (
            df.loc[symbol_mask, "volume_sma"] + 1e-10
        )

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Trade sequence features
    df["trade_sequence"] = df.groupby("symbol").cumcount() + 1
    df["time_since_last"] = (
        df.groupby("symbol")["timestamp"].diff().dt.total_seconds() / 60
    )  # minutes

    # PnL and fee features
    df["pnl_per_quantity"] = df["pnl"] / (df["quantity"] + 1e-10)
    df["fee_per_quantity"] = df["fee"] / (df["quantity"] + 1e-10)
    df["net_pnl"] = df["pnl"] - df["fee"]

    # Cumulative features (per symbol)
    for symbol in df["symbol"].unique():
        symbol_mask = df["symbol"] == symbol
        df.loc[symbol_mask, "cumulative_pnl"] = df.loc[symbol_mask, "pnl"].cumsum()
        df.loc[symbol_mask, "cumulative_volume"] = df.loc[
            symbol_mask, "quantity"
        ].cumsum()
        df.loc[symbol_mask, "running_avg_price"] = (
            df.loc[symbol_mask, "price"].expanding().mean()
        )

    # Side encoding (buy=1, sell=-1, test=0)
    side_mapping = {"BUY": 1, "SELL": -1, "TEST": 0}
    df["side_encoded"] = df["side"].map(side_mapping)

    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(method="ffill").fillna(0)

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"✅ Created {len(df.columns)} features for trade-level analysis")
    logger.info(
        f"   Features per trade: {len([col for col in df.columns if col not in ['id', 'timestamp', 'symbol', 'side']])}"
    )

    return df


def create_trade_level_targets(df, horizon=5):
    """Create prediction targets for individual trades"""

    logger = setup_logger("patient_target_creator")
    logger.info(f"🎯 Creating trade-level targets (horizon: {horizon} trades ahead)...")

    if df.empty:
        return df

    # Create targets per symbol
    for symbol in df["symbol"].unique():
        symbol_mask = df["symbol"] == symbol
        symbol_data = df[symbol_mask].copy()

        if len(symbol_data) <= horizon:
            continue

        # Future price N trades ahead
        df.loc[symbol_mask, f"future_price_{horizon}"] = symbol_data["price"].shift(
            -horizon
        )

        # Future return N trades ahead
        future_return = (
            df.loc[symbol_mask, f"future_price_{horizon}"]
            / df.loc[symbol_mask, "price"]
        ) - 1
        df.loc[symbol_mask, f"future_return_{horizon}"] = future_return

        # Classification targets
        df.loc[symbol_mask, "target_up"] = (future_return > 0).astype(int)
        df.loc[symbol_mask, "target_profitable"] = (future_return > 0.001).astype(
            int
        )  # >0.1% profit

        # Regression target
        df.loc[symbol_mask, "target_return"] = future_return

    # Remove rows without future data
    valid_mask = ~df["target_return"].isna()
    df = df[valid_mask].copy()

    # Fill any remaining NaN targets
    df["target_up"] = df["target_up"].fillna(0).astype(int)
    df["target_profitable"] = df["target_profitable"].fillna(0).astype(int)
    df["target_return"] = df["target_return"].fillna(0)

    logger.info(f"✅ Created targets for {len(df)} trades")
    logger.info(f"   Target UP distribution: {dict(df['target_up'].value_counts())}")
    logger.info(
        f"   Target PROFITABLE distribution: {dict(df['target_profitable'].value_counts())}"
    )
    logger.info(
        f"   Return range: [{df['target_return'].min():.6f}, {df['target_return'].max():.6f}]"
    )

    return df


def train_patient_llm_models(df, epochs=20, batch_size=5, debug=False, args=None):
    """Train LLM-enhanced models with patient handling for slow LLMs"""

    logger = setup_logger("patient_llm_trainer")
    logger.info(
        f"🐌 Training Patient LLM-enhanced models on {len(df)} individual trades..."
    )

    # Setup LLM trainer with patient configuration
    llm_config = create_llm_config_from_env()
    trainer = LLMEnhancedTrainer(llm_config)

    # Load existing LLM cache if available
    cache_file = "patient_llm_cache.json"
    if os.path.exists(cache_file):
        trainer.load_llm_cache(cache_file)
        logger.info(f"📂 Loaded LLM cache from: {cache_file}")

    # For large datasets, be very conservative with LLM sampling
    sample_size = min(200, len(df) // 10)  # Use only 10% of data or 200 samples max
    if len(df) > sample_size:
        logger.info(
            f"📊 Large dataset detected ({len(df)} trades). Using patient sampling ({sample_size} samples)..."
        )
        llm_sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
        llm_sample_df = df.iloc[llm_sample_indices].copy()

        # Generate LLM features with very small batches for patient processing
        logger.info("🐌 Generating LLM features with patient processing...")
        logger.info(
            f"⏰ Using {batch_size} batch size and 5-minute timeout per request..."
        )

        start_time = time.time()
        enhanced_sample = trainer.prepare_llm_features(
            llm_sample_df, batch_size=batch_size
        )
        end_time = time.time()

        llm_processing_time = end_time - start_time
        logger.info(
            f"⏱️ LLM processing took {llm_processing_time/60:.1f} minutes for {sample_size} samples"
        )

        # Apply LLM features to full dataset using interpolation/mapping
        llm_columns = [col for col in enhanced_sample.columns if col.startswith("llm_")]
        logger.info(f"🔗 Applying {len(llm_columns)} LLM features to full dataset...")

        # Create default LLM feature values based on sample statistics
        for col in llm_columns:
            df[col] = enhanced_sample[col].mean()  # Use mean as default

        # Apply more sophisticated mapping based on price similarity
        for i, row in enhanced_sample.iterrows():
            price = row["price"]
            # Find similar priced trades in full dataset
            price_range = abs(df["price"] - price) / price < 0.01  # Within 1% of price
            if price_range.any():
                for col in llm_columns:
                    df.loc[price_range, col] = row[col]

        enhanced_df = df
    else:
        # Small dataset - process all trades with LLM but be patient
        logger.info(f"🐌 Generating patient LLM features for all {len(df)} trades...")
        logger.info(
            f"⏰ This may take {len(df) * batch_size * 5 / 60:.1f} minutes with patient processing..."
        )
        enhanced_df = trainer.prepare_llm_features(df, batch_size=batch_size)

    # Save LLM cache
    trainer.save_llm_cache(cache_file)
    logger.info(f"💾 Saved LLM cache to: {cache_file}")

    # Prepare features and targets
    exclude_columns = [
        "id",
        "timestamp",
        "symbol",
        "side",
        "target_up",
        "target_profitable",
        "target_return",
    ]
    exclude_columns.extend(
        [col for col in enhanced_df.columns if col.startswith("future_")]
    )
    exclude_columns.extend(
        [col for col in enhanced_df.columns if col.startswith("target_")]
    )

    feature_columns = [col for col in enhanced_df.columns if col not in exclude_columns]

    X = enhanced_df[feature_columns].fillna(0)
    y_classification = enhanced_df["target_up"]
    y_regression = enhanced_df["target_return"].fillna(0)

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
        logger.debug(f"   Feature columns: {feature_columns[:20]}...")  # Show first 20
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
            X,
            y_classification,
            y_regression,
            test_size=0.2,
            random_state=42,
            stratify=y_classification,
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
    # Scale n_estimators based on data size and epochs
    n_estimators = min(1000, max(100, epochs * 25, len(X_train) // 100))

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=25,  # Deeper for individual trades
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
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
        max_depth=25,  # Deeper for individual trades
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    regressor.fit(X_train_scaled, y_reg_train)

    # Evaluate regression
    reg_pred = regressor.predict(X_test_scaled)
    reg_mse = mean_squared_error(y_reg_test, reg_pred)
    logger.info(f"   Regression MSE: {reg_mse:.8f}")

    if debug:
        logger.debug("\n🔍 Feature Importance Analysis:")
        logger.debug("   Top 15 Most Important Features:")

        feature_importance = list(zip(feature_columns, classifier.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance[:15]):
            emoji = "🧠" if feature.startswith("llm_") else "📊"
            logger.debug(f"    {i+1:2d}. {emoji} {feature}: {importance:.4f}")

    # Save models
    from datetime import datetime

    import joblib

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_files = {
        "classifier": f"models/patient_llm_classifier_{timestamp}.joblib",
        "regressor": f"models/patient_llm_regressor_{timestamp}.joblib",
        "scaler": f"models/patient_llm_scaler_{timestamp}.joblib",
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
        "data_source": "all_individual_paper_trades_patient_llm",
        "training_config": {
            "method": "patient_llm",
            "total_trades_requested": (
                "all_available"
                if (args and args.trades == 0)
                else (args.trades if args else "unknown")
            ),
            "total_trades_loaded": len(enhanced_df),
            "llm_sample_size": sample_size if len(df) > sample_size else len(df),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "epochs": epochs,
            "batch_size": batch_size,
            "n_estimators": n_estimators,
            "vault_enabled": args and hasattr(args, "vault") and args.vault,
            "horizon": args.horizon if args else 5,
            "llm_timeout": int(os.getenv("ELVIS_LLM_TIMEOUT", "300")),
            "patient_mode": True,
        },
        "data_info": {
            "total_samples": len(enhanced_df),
            "total_features": len(feature_columns),
            "llm_features": len([c for c in feature_columns if c.startswith("llm_")]),
            "trade_level_processing": True,
            "aggregation_method": "individual_trades",
        },
        "model_performance": {
            "classification_accuracy": float(class_accuracy),
            "regression_mse": float(reg_mse),
            "oob_score_classification": float(classifier.oob_score_),
            "oob_score_regression": float(regressor.oob_score_),
        },
        "model_files": model_files,
        "llm_config": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "timeout": llm_config.timeout,
        },
    }

    metadata_file = f"models/patient_llm_metadata_{timestamp}.json"
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"   Metadata saved: {metadata_file}")

    return classifier, regressor, scaler, metadata


def main():
    """Main training function for patient LLM processing"""

    parser = argparse.ArgumentParser(
        description="Train Patient LLM-enhanced models on ALL individual paper trades"
    )
    parser.add_argument(
        "--trades",
        type=int,
        default=0,
        help="Max number of trades (0 = infinite/all available)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--horizon", type=int, default=5, help="Prediction horizon in trades ahead"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="LLM batch size (very small for patient processing)",
    )
    parser.add_argument(
        "--include-test", action="store_true", help="Include TEST trades in training"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--vault",
        action="store_true",
        default=True,
        help="Enable Vault for secrets management",
    )

    args = parser.parse_args()

    # Setup environment based on vault setting
    if args.vault:
        # Enable Vault for secrets management
        os.environ.pop("VAULT_ENABLED", None)
        os.environ.pop("USE_VAULT", None)
        os.environ.pop("VAULT_AVAILABLE", None)
        logger_level = "DEBUG" if args.debug else "INFO"
        logger = setup_logger("patient_llm_training", logger_level)
        logger.info("🔐 Vault enabled for secrets management")
    else:
        setup_patient_llm_environment()
        # Setup logging
        logger = setup_logger("patient_llm_training", "DEBUG" if args.debug else "INFO")
        logger.info("🔓 Using environment variables for secrets")

    # Determine trades limit
    trades_text = "ALL AVAILABLE" if args.trades == 0 else str(args.trades)

    logger.info("🐌 Starting PATIENT LLM Training Pipeline")
    logger.info("=" * 50)
    logger.info(f"📊 Training Configuration:")
    logger.info(f"   Mode: Patient LLM Processing")
    logger.info(f"   Trades: {trades_text}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Horizon: {args.horizon} trades ahead")
    logger.info(f"   Batch Size: {args.batch_size} (very small for patience)")
    logger.info(f"   Include TEST trades: {args.include_test}")
    logger.info(f"   Debug: {args.debug}")
    logger.info(f"   Vault: {args.vault}")
    logger.info(f"   LLM Timeout: {os.getenv('ELVIS_LLM_TIMEOUT', '300')}s (5 minutes)")

    try:
        # Step 1: Load all individual trades
        logger.info("\n📂 Step 1: Loading ALL individual paper trades...")
        trades_df = load_all_individual_trades(
            include_test_trades=args.include_test, max_trades=args.trades
        )

        if trades_df.empty:
            logger.error("❌ No trades loaded from PostgreSQL!")
            return 1

        # Step 2: Create trade-level features
        logger.info(f"\n🔧 Step 2: Creating trade-level features...")
        features_df = create_trade_level_features(trades_df)

        # Step 3: Create prediction targets
        logger.info(
            f"\n🎯 Step 3: Creating prediction targets (horizon: {args.horizon})..."
        )
        targets_df = create_trade_level_targets(features_df, horizon=args.horizon)

        if targets_df.empty:
            logger.error("❌ No training samples created!")
            return 1

        # Step 4: Train Patient LLM-enhanced models
        logger.info(f"\n🐌 Step 4: Training Patient LLM-enhanced models...")
        classifier, regressor, scaler, metadata = train_patient_llm_models(
            targets_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            debug=args.debug,
            args=args,
        )

        # Success
        logger.info(f"\n🎉 Patient LLM Training Complete!")
        logger.info("=" * 50)
        logger.info(
            f"✅ Classification Accuracy: {metadata['model_performance']['classification_accuracy']:.4f}"
        )
        logger.info(
            f"✅ Regression MSE: {metadata['model_performance']['regression_mse']:.8f}"
        )
        logger.info(
            f"✅ Total Features Used: {metadata['data_info']['total_features']}"
        )
        logger.info(f"🧠 LLM Features Used: {metadata['data_info']['llm_features']}")
        logger.info(f"📁 Results saved in: models/")

        print(f"\n🐌 PATIENT LLM-Enhanced ELVIS Training")
        print("=" * 45)
        print(f"📊 Training Configuration:")
        print(f"   Mode: Patient LLM Processing")
        print(
            f"   Trades Requested: {metadata['training_config']['total_trades_requested']}"
        )
        print(
            f"   Trades Loaded: {metadata['training_config']['total_trades_loaded']:,}"
        )
        print(f"   LLM Sample Size: {metadata['training_config']['llm_sample_size']:,}")
        print(
            f"   Training Samples: {metadata['training_config']['training_samples']:,}"
        )
        print(f"   Test Samples: {metadata['training_config']['test_samples']:,}")
        print(f"   Epochs: {metadata['training_config']['epochs']}")
        print(f"   Horizon: {metadata['training_config']['horizon']} trades ahead")
        print(f"   Batch Size: {metadata['training_config']['batch_size']} (patient)")
        print(f"   LLM Timeout: {metadata['training_config']['llm_timeout']}s")
        print(f"   Debug: {args.debug}")
        print(f"   Vault: {metadata['training_config']['vault_enabled']}")
        print(f"\n📊 Model Architecture:")
        print(
            f"   Model Trees (n_estimators): {metadata['training_config']['n_estimators']}"
        )
        print(f"   LLM Features: {metadata['data_info']['llm_features']}")
        print(f"   Total Features: {metadata['data_info']['total_features']}")
        print(f"   Processing Method: Individual Trade Level")
        print(f"\n🎯 Performance:")
        print(
            f"   Classification Accuracy: {metadata['model_performance']['classification_accuracy']:.4f}"
        )
        print(
            f"   Regression MSE: {metadata['model_performance']['regression_mse']:.2e}"
        )
        print(
            f"   OOB Score (Classification): {metadata['model_performance']['oob_score_classification']:.4f}"
        )
        print(
            f"   OOB Score (Regression): {metadata['model_performance']['oob_score_regression']:.4f}"
        )
        print(f"\n🤖 Patient LLM Configuration:")
        print(f"   Model: {metadata['llm_config']['model']}")
        print(f"   URL: {metadata['llm_config']['base_url']}")
        print(f"   Timeout: {metadata['llm_config']['timeout']}s (5 minutes)")
        print(f"\n🎉 Patient training completed successfully!")
        print(
            f"🐌 Your slow LLM was given plenty of time to analyze {metadata['training_config']['llm_sample_size']:,} trades!"
        )
        print(f"📁 Check the models/ directory for results")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback

        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
