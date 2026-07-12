#!/usr/bin/env python3
"""
Enhanced Training Script with PostgreSQL Trade History Integration
Trains ELVIS models using both market data and actual trade history from the database.
"""

# Make the repo root importable no matter where this script is run from
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_vault_environment():
    """Setup Vault environment variables if not already set"""
    if not os.getenv("VAULT_ADDR"):
        os.environ["VAULT_ADDR"] = "http://127.0.0.1:8200"
        print("🔐 Set VAULT_ADDR to default: http://127.0.0.1:8200")

    if not os.getenv("VAULT_TOKEN"):
        os.environ["VAULT_TOKEN"] = "trading-bot-token"
        print("🔐 Set VAULT_TOKEN to default: trading-bot-token")

    print(f"🔐 Using Vault URL: {os.getenv('VAULT_ADDR')}")


def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import numpy as np
        import pandas as pd
        import torch

        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train ELVIS models with integrated trade history"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of trades to include (default: 1000)",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=5,
        help="Number of trades to look ahead for targets (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--output", type=str, default="models", help="Output directory for models"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument(
        "--no-vault", action="store_true", help="Skip Vault setup (for testing)"
    )

    args = parser.parse_args()

    print("🚀 ELVIS Enhanced Training with Trade History Integration")
    print("=" * 60)
    print(f"📊 Trade limit: {args.limit}")
    print(f"🎯 Prediction horizon: {args.prediction_horizon} trades")
    print(f"⏱️  Training epochs: {args.epochs}")
    print(f"📁 Output directory: {args.output}")
    print(f"🐛 Debug mode: {args.debug}")
    print("=" * 60)

    # Check dependencies first
    if not check_dependencies():
        print("❌ Missing required dependencies. Please install requirements.txt")
        sys.exit(1)

    # Setup Vault environment unless disabled
    if not args.no_vault:
        setup_vault_environment()
    else:
        print("⚠️  Vault setup skipped - using environment variables only")

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Build command for main training script
    cmd_parts = [
        sys.executable,
        "training/train_models.py",
        "--include-trade-history",  # Enable trade history integration
        "--config",
        args.config,
        "--output",
        args.output,
    ]

    if args.debug:
        cmd_parts.append("--debug")

    if args.distributed:
        cmd_parts.append("--distributed")

    # Set environment variables for trade processing
    os.environ["TRADE_LIMIT"] = str(args.limit)
    os.environ["PREDICTION_HORIZON"] = str(args.prediction_horizon)
    os.environ["TRAINING_EPOCHS"] = str(args.epochs)

    # Set logging level based on debug mode
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        os.environ["LOG_LEVEL"] = "INFO"
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    print("🔄 Starting training with trade history integration...")
    print(f"📋 Command: {' '.join(cmd_parts)}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Execute training
    import subprocess

    try:
        # Run with environment variables and capture output
        env = os.environ.copy()
        result = subprocess.run(
            cmd_parts,
            check=True,
            env=env,
            capture_output=False,  # Show output in real-time
            text=True,
        )

        print(
            f"\n✅ Training completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!"
        )
        print("📈 Models have been trained with integrated trade history")
        print(f"💾 Check {args.output}/ for trained models and logs")
        print(
            f"📊 Training used {args.limit} trades with {args.prediction_horizon} prediction horizon"
        )

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("💡 Try running with --debug for more information")
        print("💡 Check Vault server is running: vault status")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💡 Training progress may have been saved to checkpoints")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ Training script not found: training/train_models.py")
        print("💡 Make sure you're running from the correct directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
