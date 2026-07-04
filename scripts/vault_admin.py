#!/usr/bin/env python3
"""
Vault Administration Script for ELVIS Trading Bot
Manage secrets in HashiCorp Vault
"""

import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.secrets_manager import get_enhanced_secrets_manager
from utils.vault_client import get_vault_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def list_all_secrets():
    """List all secrets in Vault"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)
        vault_status = secrets_manager.get_vault_status()

        if not vault_status["healthy"]:
            logger.error(f"❌ Vault not available: {vault_status}")
            return False

        print("🔐 Secrets in Vault:")
        print("=" * 40)

        secrets = secrets_manager.list_secrets()
        if not secrets:
            print("No secrets found")
            return True

        for category, secret_names in secrets.items():
            print(f"\n📁 {category}:")
            for name in secret_names:
                print(f"  • {name}")

        return True

    except Exception as e:
        logger.error(f"Failed to list secrets: {e}")
        return False


def add_secret():
    """Interactively add a secret to Vault"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)

        print("📝 Add Secret to Vault")
        print("=" * 30)

        # Get secret details
        categories = ["api_keys", "database", "webhooks", "default"]
        print(f"Available categories: {', '.join(categories)}")
        category = input("Category: ").strip() or "default"

        name = input("Secret name (e.g., BINANCE_API_KEY): ").strip()
        if not name:
            logger.error("Secret name is required")
            return False

        # Get secret value securely
        is_sensitive = input("Is this a sensitive value? (Y/n): ").lower() != "n"
        if is_sensitive:
            value = getpass.getpass("Secret value: ")
        else:
            value = input("Secret value: ").strip()

        if not value:
            logger.error("Secret value is required")
            return False

        # Store the secret
        secrets_manager.set_secret(name, value, category)
        logger.info(f"✅ Secret '{name}' added to category '{category}'")

        return True

    except Exception as e:
        logger.error(f"Failed to add secret: {e}")
        return False


def update_secret():
    """Update an existing secret"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)

        print("✏️ Update Secret in Vault")
        print("=" * 30)

        # List existing secrets first
        list_all_secrets()

        category = input("\nCategory of secret to update: ").strip() or "default"
        name = input("Secret name to update: ").strip()

        if not name:
            logger.error("Secret name is required")
            return False

        # Show current value (masked)
        current_value = secrets_manager.get_secret(name, category)
        if current_value:
            masked_value = (
                current_value[:4] + "*" * (len(current_value) - 4)
                if len(current_value) > 4
                else "****"
            )
            print(f"Current value: {masked_value}")
        else:
            print("Secret not found - will create new")

        # Get new value
        is_sensitive = input("Is this a sensitive value? (Y/n): ").lower() != "n"
        if is_sensitive:
            new_value = getpass.getpass("New secret value: ")
        else:
            new_value = input("New secret value: ").strip()

        if not new_value:
            logger.error("Secret value is required")
            return False

        # Update the secret
        secrets_manager.set_secret(name, new_value, category)
        logger.info(f"✅ Secret '{name}' updated in category '{category}'")

        return True

    except Exception as e:
        logger.error(f"Failed to update secret: {e}")
        return False


def delete_secret():
    """Delete a secret from Vault"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)

        print("🗑️ Delete Secret from Vault")
        print("=" * 30)

        # List existing secrets first
        list_all_secrets()

        category = input("\nCategory of secret to delete: ").strip() or "default"
        name = input("Secret name to delete: ").strip()

        if not name:
            logger.error("Secret name is required")
            return False

        # Confirm deletion
        confirm = input(
            f"Are you sure you want to delete '{name}' from '{category}'? (y/N): "
        )
        if confirm.lower() != "y":
            print("Deletion cancelled")
            return True

        # Delete the secret
        if secrets_manager.delete_secret(name, category):
            logger.info(f"✅ Secret '{name}' deleted from category '{category}'")
        else:
            logger.error(f"❌ Failed to delete secret '{name}'")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to delete secret: {e}")
        return False


def check_vault_status():
    """Check Vault connection and status"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)
        vault_status = secrets_manager.get_vault_status()

        print("🔍 Vault Status")
        print("=" * 20)
        print(f"Enabled: {'✅' if vault_status['enabled'] else '❌'}")
        print(f"Connected: {'✅' if vault_status['connected'] else '❌'}")
        print(f"Healthy: {'✅' if vault_status['healthy'] else '❌'}")

        if "url" in vault_status:
            print(f"URL: {vault_status['url']}")

        if "error" in vault_status:
            print(f"Error: {vault_status['error']}")

        return vault_status["healthy"]

    except Exception as e:
        logger.error(f"Failed to check Vault status: {e}")
        return False


def backup_vault_secrets():
    """Create a backup of all Vault secrets (encrypted)"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)

        # This would create an encrypted backup
        # For now, just list what would be backed up
        print("💾 Vault Backup")
        print("=" * 20)

        secrets = secrets_manager.list_secrets()
        total_secrets = sum(len(names) for names in secrets.values())

        print(f"Would backup {total_secrets} secrets from {len(secrets)} categories")
        for category, names in secrets.items():
            print(f"  {category}: {len(names)} secrets")

        # In production, this would create an encrypted backup file
        print(
            "\n⚠️ Note: Actual backup implementation would encrypt and store secrets safely"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to backup secrets: {e}")
        return False


def test_credentials():
    """Test that credentials can be retrieved successfully"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)

        print("🧪 Testing Credential Retrieval")
        print("=" * 35)

        # Test Binance credentials
        print("Testing Binance credentials...")
        binance_creds = secrets_manager.get_binance_credentials()
        binance_ok = bool(
            binance_creds.get("api_key") and binance_creds.get("api_secret")
        )
        print(f"  Binance: {'✅' if binance_ok else '❌'}")

        # Test database credentials
        print("Testing database credentials...")
        db_creds = secrets_manager.get_database_credentials()
        db_ok = bool(db_creds.get("host") and db_creds.get("user"))
        print(f"  Database: {'✅' if db_ok else '❌'}")

        # Test Redis credentials
        print("Testing Redis credentials...")
        redis_creds = secrets_manager.get_redis_credentials()
        redis_ok = bool(redis_creds.get("host"))
        print(f"  Redis: {'✅' if redis_ok else '❌'}")

        # Test Telegram credentials
        print("Testing Telegram credentials...")
        telegram_creds = secrets_manager.get_telegram_credentials()
        telegram_ok = bool(telegram_creds.get("bot_token"))
        print(f"  Telegram: {'✅' if telegram_ok else '❌'}")

        overall_ok = binance_ok and db_ok and redis_ok
        print(
            f"\nOverall status: {'✅ All critical credentials available' if overall_ok else '⚠️ Some credentials missing'}"
        )

        return overall_ok

    except Exception as e:
        logger.error(f"Failed to test credentials: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Vault Administration for ELVIS Trading Bot"
    )
    parser.add_argument("--list", action="store_true", help="List all secrets")
    parser.add_argument("--add", action="store_true", help="Add a new secret")
    parser.add_argument(
        "--update", action="store_true", help="Update an existing secret"
    )
    parser.add_argument("--delete", action="store_true", help="Delete a secret")
    parser.add_argument("--status", action="store_true", help="Check Vault status")
    parser.add_argument("--backup", action="store_true", help="Backup Vault secrets")
    parser.add_argument("--test", action="store_true", help="Test credential retrieval")

    args = parser.parse_args()

    if args.status:
        check_vault_status()
    elif args.list:
        list_all_secrets()
    elif args.add:
        add_secret()
    elif args.update:
        update_secret()
    elif args.delete:
        delete_secret()
    elif args.backup:
        backup_vault_secrets()
    elif args.test:
        test_credentials()
    else:
        parser.print_help()
        print("\n🔐 Vault Administration Commands:")
        print("  --status   Check Vault connection")
        print("  --list     List all secrets")
        print("  --add      Add a new secret")
        print("  --update   Update existing secret")
        print("  --delete   Delete a secret")
        print("  --test     Test credential retrieval")


if __name__ == "__main__":
    main()
