#!/usr/bin/env python3
"""
Test vault connection for the bot
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def _configure_local_vault_test():
    """Configure the explicit local Vault target for manual execution only."""
    os.environ["VAULT_ADDR"] = "http://127.0.0.1:8200"
    os.environ["VAULT_TOKEN"] = "trading-bot-token"
    os.environ["VAULT_ENABLED"] = "true"


def check_vault():
    """Test vault connection"""
    print("🔒 Testing Vault Connection")
    print("=" * 50)

    try:
        import logging

        from utils.vault_client import get_vault_client

        # Create logger
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        print("1. Getting vault client...")
        vault_client = get_vault_client(logger)

        if not vault_client:
            print("❌ Failed to get vault client")
            return False

        print("2. Testing health check...")
        health = vault_client.health_check()
        print(f"   Health check result: {health}")

        if not health:
            print("❌ Vault health check failed")
            return False

        print("3. Testing secret operations...")
        # Try to write a secret
        result = vault_client.set_secret("trading/test", "api_key", "test_value_123")
        print(f"   Write result: {result}")

        # Try to read the secret
        secret = vault_client.get_secret("trading/test", "api_key")
        print(f"   Read result: {secret}")

        if secret == "test_value_123":
            print("✅ Vault connection test PASSED!")
            return True
        else:
            print("❌ Secret read/write test failed")
            return False

    except Exception as e:
        print(f"❌ Vault test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_secrets_manager():
    """Test secrets manager with vault"""
    print("\n🔐 Testing Secrets Manager")
    print("-" * 50)

    try:
        from utils.secrets_manager import get_enhanced_secrets_manager

        print("1. Getting secrets manager...")
        secrets_manager = get_enhanced_secrets_manager()

        print("2. Testing vault status...")
        vault_status = secrets_manager.get_vault_status()
        print(f"   Vault status: {vault_status}")

        print("3. Testing secret storage...")
        # Test storing a secret
        success = secrets_manager.set_secret("test_api_key", "test_secret_value_456")
        print(f"   Store result: {success}")

        # Test retrieving the secret
        retrieved = secrets_manager.get_secret("test_api_key")
        print(f"   Retrieved secret: {retrieved}")

        if retrieved == "test_secret_value_456":
            print("✅ Secrets manager test PASSED!")
            return True
        else:
            print("❌ Secrets manager test failed")
            return False

    except Exception as e:
        print(f"❌ Secrets manager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    _configure_local_vault_test()
    vault_ok = check_vault()
    secrets_ok = check_secrets_manager()

    print("\n" + "=" * 60)
    print("📋 VAULT TEST SUMMARY:")
    print(f"   Vault Client: {'✅' if vault_ok else '❌'}")
    print(f"   Secrets Manager: {'✅' if secrets_ok else '❌'}")

    if vault_ok and secrets_ok:
        print("\n🎉 All vault tests PASSED! Vault should show green in dashboard.")
    else:
        print("\n⚠️ Some vault tests failed. Check configuration.")
    print("=" * 60)
