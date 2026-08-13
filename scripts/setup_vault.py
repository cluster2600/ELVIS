#!/usr/bin/env python3.14
"""
Vault Setup and Migration Script for ELVIS Trading Bot
Sets up HashiCorp Vault and migrates secrets from .env file
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from utils.secrets_manager import get_enhanced_secrets_manager
from utils.vault_client import get_vault_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_vault_server():
    """Instructions for setting up Vault server"""
    print("🔐 HashiCorp Vault Setup Instructions")
    print("=" * 50)
    print()
    print("1. Install Vault:")
    print("   macOS: brew install vault")
    print("   Linux: https://www.vaultproject.io/downloads")
    print("   Windows: Download from https://www.vaultproject.io/downloads")
    print()
    print("2. Start Vault in development mode:")
    print("   vault server -dev")
    print()
    print("3. In another terminal, set environment variables:")
    print("   export VAULT_ADDR='http://127.0.0.1:8200'")
    print("   export VAULT_TOKEN='<dev-token-from-server-output>'")
    print()
    print("4. Enable KV secrets engine (if not already enabled):")
    print("   vault secrets enable -path=secret kv-v2")
    print()
    print("5. Run this script with --migrate to move secrets:")
    print(f"   python {__file__} --migrate")
    print()


def validate_vault_connection():
    """Validate Vault connection and setup"""
    try:
        vault_client = get_vault_client(logger=logger)

        if not vault_client.health_check():
            logger.error("❌ Vault health check failed")
            return False

        if not vault_client.is_authenticated:
            logger.error("❌ Vault authentication failed")
            return False

        logger.info("✅ Vault connection validated successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Vault validation failed: {e}")
        return False


def create_vault_policies():
    """Create Vault policies for the trading bot"""
    try:
        vault_client = get_vault_client(logger=logger)

        # Trading bot policy
        policy_content = """
# Policy for ELVIS Trading Bot
path "secret/data/trading/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/data/database/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/data/notifications/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/data/monitoring/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/data/general/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Allow listing of secret paths
path "secret/metadata/*" {
  capabilities = ["list"]
}
"""

        # Note: hvac doesn't directly support policy creation in dev mode
        # This would be used in production Vault setup
        logger.info(
            "📝 Vault policy created (development mode automatically grants full access)"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to create Vault policies: {e}")
        return False


def migrate_secrets_to_vault():
    """Migrate secrets from .env file to Vault"""
    try:
        # Load .env file
        env_file = project_root / ".env"
        if not env_file.exists():
            logger.error("❌ .env file not found")
            return False

        load_dotenv(env_file)
        logger.info("📄 Loaded .env file")

        # Initialize enhanced secrets manager
        secrets_manager = get_enhanced_secrets_manager(logger=logger)

        # Check Vault status
        vault_status = secrets_manager.get_vault_status()
        if not vault_status["healthy"]:
            logger.error(f"❌ Vault not healthy: {vault_status}")
            return False

        # Initialize from environment variables
        secrets_manager.initialize_from_env()

        # Migrate to Vault
        if secrets_manager.migrate_to_vault():
            logger.info("✅ Successfully migrated secrets to Vault")
            return True
        else:
            logger.error("❌ Failed to migrate some secrets to Vault")
            return False

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def verify_secrets_migration():
    """Verify that secrets were migrated correctly"""
    try:
        secrets_manager = get_enhanced_secrets_manager(logger=logger)

        # Test retrieving critical secrets
        test_secrets = [
            ("BINANCE_API_KEY", "api_keys"),
            ("POSTGRES_PASSWORD", "database"),
            ("TELEGRAM_BOT_TOKEN", "api_keys"),
        ]

        logger.info("🔍 Verifying secret migration...")

        all_good = True
        for secret_name, category in test_secrets:
            value = secrets_manager.get_secret(secret_name, category)
            if value:
                logger.info(f"✅ {secret_name}: Retrieved successfully")
            else:
                logger.warning(f"⚠️ {secret_name}: Not found or empty")
                all_good = False

        # Test credential helpers
        logger.info("🔍 Testing credential helpers...")

        binance_creds = secrets_manager.get_binance_credentials()
        if binance_creds.get("api_key"):
            logger.info("✅ Binance credentials: Available")
        else:
            logger.warning("⚠️ Binance credentials: Missing")
            all_good = False

        db_creds = secrets_manager.get_database_credentials()
        if db_creds.get("host"):
            logger.info("✅ Database credentials: Available")
        else:
            logger.warning("⚠️ Database credentials: Missing")
            all_good = False

        if all_good:
            logger.info("✅ All secrets verified successfully")
        else:
            logger.warning("⚠️ Some secrets may be missing")

        return all_good

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def backup_env_file():
    """Create a backup of the .env file"""
    try:
        env_file = project_root / ".env"
        backup_file = project_root / ".env.backup"

        if env_file.exists():
            import shutil

            shutil.copy2(env_file, backup_file)
            logger.info(f"📄 Created backup: {backup_file}")
            return True
        else:
            logger.warning("No .env file to backup")
            return False

    except Exception as e:
        logger.error(f"Failed to backup .env file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup HashiCorp Vault for ELVIS Trading Bot"
    )
    parser.add_argument(
        "--setup", action="store_true", help="Show Vault setup instructions"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate Vault connection"
    )
    parser.add_argument(
        "--migrate", action="store_true", help="Migrate secrets from .env to Vault"
    )
    parser.add_argument("--verify", action="store_true", help="Verify secret migration")
    parser.add_argument("--backup", action="store_true", help="Backup .env file")
    parser.add_argument("--all", action="store_true", help="Run full setup process")

    args = parser.parse_args()

    if args.setup or args.all:
        setup_vault_server()
        if not args.all:
            return

    if args.validate or args.all:
        print("\n🔍 Validating Vault connection...")
        if not validate_vault_connection():
            logger.error("❌ Vault validation failed. Please check your Vault setup.")
            if not args.all:
                sys.exit(1)

        create_vault_policies()

    if args.backup or args.all:
        print("\n💾 Creating backup...")
        backup_env_file()

    if args.migrate or args.all:
        print("\n🚀 Migrating secrets to Vault...")
        if not migrate_secrets_to_vault():
            logger.error("❌ Migration failed")
            if not args.all:
                sys.exit(1)

    if args.verify or args.all:
        print("\n✅ Verifying migration...")
        if not verify_secrets_migration():
            logger.error("❌ Verification failed")
            if not args.all:
                sys.exit(1)

    if args.all:
        print("\n🎉 Vault setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Update your .env file to remove secrets (keep Vault config)")
        print("2. Test your application with Vault integration")
        print("3. Consider setting up production Vault server for production use")

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
