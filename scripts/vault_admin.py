#!/usr/bin/env python3
"""
Vault Administration Script for ELVIS Trading Bot
Manage secrets in HashiCorp Vault
"""

import argparse
import getpass
import json
import logging
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.secrets_manager import _VAULT_KEY_MAP, get_enhanced_secrets_manager
from utils.vault_client import get_vault_client

# Default output path for `--backup`. Matches the `.gitignore` entry so the
# encrypted dump never lands in version control.
DEFAULT_BACKUP_PATH = ".vault-backup.enc"

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


def _backup_paths():
    """Return the ordered, de-duplicated set of Vault KV paths to back up.

    Always includes the two service paths (`binance`, `binance_testnet`) and
    every distinct path referenced by ``_VAULT_KEY_MAP`` so the backup stays in
    sync with the secrets ELVIS actually reads.
    """
    paths = ["binance", "binance_testnet"]
    for vault_path, _field in _VAULT_KEY_MAP.values():
        if vault_path not in paths:
            paths.append(vault_path)
    return paths


def _get_backup_cipher():
    """Return a Fernet cipher for the backup, sourcing the key from BACKUP_KEY.

    If ``BACKUP_KEY`` is unset a fresh key is generated and printed exactly
    once so the operator can persist it; without that key the backup cannot be
    decrypted, so it must be captured now.
    """
    key = os.getenv("BACKUP_KEY")
    if key:
        return Fernet(key.encode() if isinstance(key, str) else key)

    key = Fernet.generate_key()
    print("🔑 No BACKUP_KEY set - generated a new one. SAVE THIS, it is the")
    print("   only way to decrypt the backup and it will NOT be shown again:")
    print(f"\n   BACKUP_KEY={key.decode()}\n")
    return Fernet(key)


def backup_vault_secrets(out_path: str = DEFAULT_BACKUP_PATH):
    """Create an encrypted backup of the known Vault secret paths.

    Reads each service path via the enhanced secrets manager's Vault client,
    serialises the collected secrets to JSON, encrypts the whole blob with
    Fernet, and writes the ciphertext to ``out_path``. Plaintext secrets are
    never written to disk.
    """
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)
        vault_client = secrets_manager.vault_client or get_vault_client(logger=logger)

        print("💾 Vault Backup")
        print("=" * 20)

        if vault_client is None or not vault_client.health_check():
            logger.error("❌ Vault not available - cannot back up secrets")
            return False

        collected = {}
        for path in _backup_paths():
            try:
                data = vault_client.get_secret(path)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Failed to read Vault path '{path}': {e}")
                data = None
            if data:
                collected[path] = data
                print(f"  • {path}: {len(data)} field(s)")
            else:
                print(f"  • {path}: (empty / not found)")

        if not collected:
            logger.error("No secrets found to back up")
            return False

        # Encrypt before touching the filesystem: only ciphertext is written.
        cipher = _get_backup_cipher()
        plaintext = json.dumps(collected, sort_keys=True).encode()
        ciphertext = cipher.encrypt(plaintext)

        out = Path(out_path)
        with open(out, "wb") as f:
            f.write(ciphertext)
        if hasattr(os, "chmod"):
            os.chmod(out, 0o600)

        total = sum(len(fields) for fields in collected.values())
        logger.info(
            f"✅ Backed up {total} secret field(s) from {len(collected)} path(s) "
            f"to encrypted file {out}"
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
    parser.add_argument(
        "--out",
        default=DEFAULT_BACKUP_PATH,
        help=f"Encrypted backup output path (default: {DEFAULT_BACKUP_PATH})",
    )
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
        backup_vault_secrets(args.out)
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
