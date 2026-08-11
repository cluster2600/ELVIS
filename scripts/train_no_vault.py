#!/usr/bin/env python3
"""
Training script that completely bypasses Vault authentication
Uses environment variables only for training purposes
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def setup_training_environment():
    """Setup environment variables for training without Vault"""
    print("🔧 Setting up training environment (Vault-free)...")

    # Disable Vault completely
    os.environ["VAULT_ENABLED"] = "false"
    os.environ["USE_VAULT"] = "false"
    os.environ["VAULT_AVAILABLE"] = "false"

    # Set training credentials (these are safe for paper trading)
    os.environ["BINANCE_API_KEY"] = "training_mode_key"
    os.environ["BINANCE_API_SECRET"] = "training_mode_secret"

    # Database settings for training
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5432")
    os.environ.setdefault("POSTGRES_USER", "postgres")
    os.environ.setdefault("POSTGRES_PASSWORD", "training_password")
    os.environ.setdefault("POSTGRES_DBNAME", "trading_db")

    # Redis settings
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")

    print("✅ Training environment configured")


def patch_imports():
    """Patch Python path to avoid Vault imports"""
    # Add current directory to path
    current_dir = Path(__file__).resolve().parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Create a mock vault client if needed
    mock_vault = """
class VaultClient:
    def __init__(self, *args, **kwargs):
        pass
    def authenticate(self):
        return False
    def get_secret(self, *args, **kwargs):
        return None
"""

    with open("vault_mock.py", "w") as f:
        f.write(mock_vault)


def run_training(args):
    """Run training with vault-free environment"""
    print("🚀 Starting Vault-free training...")
    print("=" * 50)

    # Setup environment
    setup_training_environment()
    patch_imports()

    # Build training command
    cmd_parts = [
        sys.executable,
        "training/train_models.py",
        "--include-trade-history",
        "--config",
        args.config,
        "--output",
        args.output,
    ]

    if args.debug:
        cmd_parts.append("--debug")
        os.environ["LOG_LEVEL"] = "DEBUG"

    # Set training parameters
    os.environ["TRADE_LIMIT"] = str(args.limit)
    os.environ["PREDICTION_HORIZON"] = str(args.prediction_horizon)
    os.environ["TRAINING_EPOCHS"] = str(args.epochs)

    print(f"📊 Training Parameters:")
    print(f"   Limit: {args.limit} trades")
    print(f"   Epochs: {args.epochs}")
    print(f"   Horizon: {args.prediction_horizon}")
    print(f"   Debug: {args.debug}")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print()

    try:
        # Run training with clean environment
        repository_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            cmd_parts, env=os.environ.copy(), cwd=str(repository_root)
        )

        if result.returncode == 0:
            print(f"\n✅ Training completed successfully!")
            print(f"🕒 Finished: {datetime.now().strftime('%H:%M:%S')}")
            print(f"📂 Check {args.output}/ for results")
        else:
            print(f"\n❌ Training failed with code {result.returncode}")
        return result.returncode

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return 1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train ELVIS models without Vault")
    parser.add_argument("--limit", type=int, default=500, help="Max trades to use")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--prediction-horizon", type=int, default=5, help="Prediction horizon"
    )
    parser.add_argument(
        "--config", default="training/config/model_config.yaml", help="Config file"
    )
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    print("🎯 ELVIS Training - Vault-Free Mode")
    print("=" * 40)
    print("🔓 Vault authentication: DISABLED")
    print("📝 Using environment variables only")
    print("🧪 Safe for training and testing")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run training
    status = run_training(args)

    if status == 0:
        print("\n💡 Next steps:")
        print("   - Check model outputs in models/")
        print("   - Review logs for training progress")
        print("   - Test models with paper trading")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
