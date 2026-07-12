#!/usr/bin/env python3
"""
Debug helper script for ELVIS training system
Diagnoses common issues and provides troubleshooting information
"""

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path


def check_python_environment():
    """Check Python environment and virtual environment"""
    print("🐍 Python Environment Check")
    print("-" * 40)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Check if in virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")

    print()


def check_required_packages():
    """Check if required Python packages are installed"""
    print("📦 Package Dependencies Check")
    print("-" * 40)

    required_packages = [
        "pandas",
        "numpy",
        "torch",
        "tensorflow",
        "sklearn",
        "hvac",
        "redis",
        "psycopg2",
        "binance",
        "ta",
    ]

    missing_packages = []
    available_packages = []

    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"❌ {package}: Not installed")
            missing_packages.append(package)
        else:
            print(f"✅ {package}: Available")
            available_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
    else:
        print("\n✅ All required packages are available")

    print()
    return len(missing_packages) == 0


def check_vault_connection():
    """Check Vault server connection and authentication"""
    print("🔐 Vault Connection Check")
    print("-" * 40)

    vault_addr = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
    vault_token = os.getenv("VAULT_TOKEN")

    print(f"VAULT_ADDR: {vault_addr}")
    print(f"VAULT_TOKEN: {'Set' if vault_token else 'Not set'}")

    try:
        result = subprocess.run(
            ["vault", "status"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("✅ Vault server is running")

            # Check authentication
            try:
                auth_result = subprocess.run(
                    ["vault", "auth", "-method=token"],
                    input=vault_token,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if auth_result.returncode == 0:
                    print("✅ Vault authentication successful")
                else:
                    print("❌ Vault authentication failed")
                    print(f"Error: {auth_result.stderr}")
            except Exception as e:
                print(f"❌ Vault auth check failed: {e}")

        else:
            print("❌ Vault server is not running")
            print(
                "💡 Start with: vault server -dev -dev-root-token-id=<choose-a-local-dev-token>"
            )
    except FileNotFoundError:
        print("❌ Vault CLI not found")
        print("💡 Install Vault: https://www.vaultproject.io/downloads")
    except subprocess.TimeoutExpired:
        print("⏰ Vault connection timeout")
    except Exception as e:
        print(f"❌ Vault check failed: {e}")

    print()


def check_database_connection():
    """Check database connectivity"""
    print("🗄️  Database Connection Check")
    print("-" * 40)

    # Check if paper_trades.db exists
    db_path = Path("paper_trades.db")
    if db_path.exists():
        print(f"✅ SQLite database found: {db_path}")
        print(f"Database size: {db_path.stat().st_size / 1024:.1f} KB")
    else:
        print("❌ SQLite database not found: paper_trades.db")
        print("💡 Database will be created on first run")

    print()


def check_training_files():
    """Check training-related files and directories"""
    print("📁 Training Files Check")
    print("-" * 40)

    required_files = [
        "training/train_models.py",
        "training/config/model_config.yaml",
        "main.py",
        "config/config.py",
    ]

    required_dirs = ["models", "training/data", "utils", "core/models"]

    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: Missing")

    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/: Missing")

    print()


def check_model_files():
    """Check existing model files"""
    print("🤖 Model Files Check")
    print("-" * 40)

    models_dir = Path("models")
    if models_dir.exists():
        model_files = (
            list(models_dir.glob("*.pth"))
            + list(models_dir.glob("*.pkl"))
            + list(models_dir.glob("*.json"))
        )
        if model_files:
            print(f"Found {len(model_files)} model files:")
            for model_file in sorted(model_files)[:10]:  # Show first 10
                print(f"  📄 {model_file.name}")
            if len(model_files) > 10:
                print(f"  ... and {len(model_files) - 10} more")
        else:
            print("No trained models found")
    else:
        print("Models directory does not exist")

    print()


def run_quick_test():
    """Run a quick test of core functionality"""
    print("🧪 Quick Functionality Test")
    print("-" * 40)

    try:
        # Test imports
        import numpy as np
        import pandas as pd

        print("✅ Core data libraries import OK")

        # Test basic data processing
        test_data = pd.DataFrame(
            {"price": np.random.randn(100), "volume": np.random.randn(100)}
        )
        test_data["sma"] = test_data["price"].rolling(10).mean()
        print("✅ Basic data processing OK")

        # Test configuration loading
        sys.path.append(".")
        from config.config import TRADING_CONFIG

        print("✅ Configuration loading OK")

        print("✅ All quick tests passed")

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback

        traceback.print_exc()

    print()


def provide_recommendations():
    """Provide troubleshooting recommendations"""
    print("💡 Troubleshooting Recommendations")
    print("-" * 40)

    print("1. 🔐 For Vault issues:")
    print(
        "   - Start Vault: vault server -dev -dev-root-token-id=<choose-a-local-dev-token>"
    )
    print("   - Set environment: export VAULT_ADDR=http://127.0.0.1:8200")
    print("   - Set token: export VAULT_TOKEN=<choose-a-local-dev-token>")
    print()

    print("2. 📦 For missing dependencies:")
    print("   - Install requirements: pip install -r requirements.txt")
    print(
        "   - Use virtual environment: python -m venv venv && source venv/bin/activate"
    )
    print()

    print("3. 🗄️  For database issues:")
    print("   - Check PostgreSQL is running: pg_ctl status")
    print("   - Reset paper trading: python reset_paper_trading.py")
    print()

    print("4. 🚀 For training issues:")
    print("   - Start small: ./train_with_trades.py --limit 100 --epochs 5 --debug")
    print("   - Skip Vault: ./train_with_trades.py --no-vault --debug")
    print("   - Check logs in models/ directory")
    print()


def main():
    """Main diagnostic function"""
    print("🔧 ELVIS Training System Diagnostic Tool")
    print("=" * 50)
    print()

    # Run all checks
    check_python_environment()
    packages_ok = check_required_packages()
    check_vault_connection()
    check_database_connection()
    check_training_files()
    check_model_files()

    if packages_ok:
        run_quick_test()

    provide_recommendations()

    print("🏁 Diagnostic complete!")
    print("💡 Run './train_with_trades.py --debug --limit 100' to test training")


if __name__ == "__main__":
    main()
