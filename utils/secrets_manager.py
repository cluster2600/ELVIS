"""
Enhanced Secrets Manager for ELVIS Trading Bot
Provides secure handling with HashiCorp Vault integration and fallback support
"""

import base64
import getpass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import keyring
from cryptography.fernet import Fernet

# Import Vault client if available
try:
    from .vault_client import VaultClient, get_vault_client

    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    VaultClient = None

logger = logging.getLogger(__name__)

# ELVIS secret name -> (KV path under the `secrets` mount, field). Matches the
# native OpenBao convention (secrets/binance, secrets/binance_testnet) rather
# than ELVIS's legacy trading/api-keys/binance-api-key scheme.
_VAULT_KEY_MAP = {
    "BINANCE_API_KEY": ("binance", "api_key"),
    "BINANCE_API_SECRET": ("binance", "secret_key"),
    "BINANCE_FUTURES_TESTNET_API_KEY": ("binance_testnet", "api_key"),
    "BINANCE_FUTURES_TESTNET_API_SECRET": ("binance_testnet", "secret_key"),
    # X-API-Key shared by the trade-history API and its dashboards
    # (store: bao kv put -mount=secrets dashboard api_key=<value>)
    "DASHBOARD_API_KEY": ("dashboard", "api_key"),
}


class EnhancedSecretsManager:
    """Enhanced secrets manager with Vault integration and fallback support"""

    def __init__(self, app_name: str = "ELVIS_TRADING_BOT", use_vault: bool = True):
        self.app_name = app_name
        self.secrets_file = Path.home() / ".elvis" / "secrets.enc"
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = None
        self._secrets_cache = {}

        # Vault integration
        self.use_vault = (
            use_vault
            and VAULT_AVAILABLE
            and os.getenv("VAULT_ENABLED", "true").lower() == "true"
        )
        self.vault_client = None

        if self.use_vault:
            try:
                self.vault_client = get_vault_client(logger=logger)
                if not self.vault_client.health_check():
                    logger.warning("Vault health check failed, using local storage")
                    self.vault_client = None
                else:
                    logger.info("Using HashiCorp Vault for secrets management")
            except Exception as e:
                logger.warning(f"Failed to initialize Vault, using local storage: {e}")
                self.vault_client = None

        if not self.vault_client:
            logger.info("Using local encrypted storage for secrets management")

    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key from OS keyring"""
        key_name = f"{self.app_name}_MASTER_KEY"

        # macOS Keychain blocks forever on an interactive authorization
        # prompt when this python binary isn't approved for the item yet
        # (fresh venv, headless run). Read on a daemon thread with a timeout
        # so we degrade to an ephemeral key instead of hanging the process.
        import threading

        result = []

        def _read():
            try:
                result.append(keyring.get_password(self.app_name, key_name))
            except Exception as e:
                logger.warning(f"Keyring read failed: {e}")
                result.append(None)

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=10)

        if reader.is_alive():
            logger.warning(
                "Keychain access timed out (python binary not authorized yet?) - "
                "using ephemeral key; local encrypted store is unavailable this run"
            )
            return Fernet.generate_key()

        if result and result[0]:
            return base64.b64decode(result[0].encode())

        # Generate new key
        logger.info("Generating new master encryption key...")
        key = Fernet.generate_key()

        # Store in keyring
        try:
            keyring.set_password(
                self.app_name, key_name, base64.b64encode(key).decode()
            )
        except Exception as e:
            logger.warning(f"Could not persist master key to keyring: {e}")

        return key

    def _get_cipher(self) -> Fernet:
        """Get or create cipher for encryption/decryption"""
        if not self._cipher:
            key = self._get_or_create_key()
            self._cipher = Fernet(key)
        return self._cipher

    def get_secret(
        self,
        name: str,
        category: str = "default",
        default: Optional[str] = None,
        warn_if_missing: bool = True,
    ) -> Optional[str]:
        """
        Retrieve a secret with Vault and local fallback

        Args:
            name: Secret name
            category: Category name
            default: Default value if secret not found
            warn_if_missing: Whether to log warning if secret is missing

        Returns:
            Secret value or default
        """
        cache_key = f"{category}.{name}"

        # Check cache first
        if cache_key in self._secrets_cache:
            return self._secrets_cache[cache_key]

        # Try Vault first if available
        if self.vault_client:
            try:
                # Known secrets live under this OpenBao's native layout
                # (mount `secrets`, one path per service). Fall back to the
                # generic category/field derivation for anything unmapped.
                if name in _VAULT_KEY_MAP:
                    vault_path, field_name = _VAULT_KEY_MAP[name]
                else:
                    vault_path = self._category_to_vault_path(category)
                    field_name = name.lower().replace(
                        "_", "-"
                    )  # Convert to vault format
                value = self.vault_client.get_secret(vault_path, field_name)
                if value:
                    self._secrets_cache[cache_key] = str(value)
                    logger.debug(f"Retrieved {name} from Vault")
                    return str(value)
            except Exception as e:
                logger.warning(f"Failed to get {name} from Vault: {e}")

        # Check environment variables before the encrypted file: reading the
        # local store touches the macOS Keychain, which blocks on an
        # interactive prompt for any not-yet-authorized python binary.
        env_value = os.getenv(name)
        if env_value:
            logger.debug(f"Retrieved {name} from environment")
            self._secrets_cache[cache_key] = env_value
            return env_value

        # Load from encrypted file
        secrets = self._load_secrets()

        if category in secrets and name in secrets[category]:
            value = secrets[category][name]
            self._secrets_cache[cache_key] = value
            return value

        if warn_if_missing:
            logger.warning(f"Secret {name} not found in any source")
        return default

    def set_secret(self, name: str, value: str, category: str = "default") -> None:
        """
        Store a secret securely in Vault and/or local storage

        Args:
            name: Secret name (e.g., 'BINANCE_API_KEY')
            value: Secret value
            category: Category for organization (e.g., 'api_keys', 'passwords')
        """
        # Try to store in Vault first
        vault_stored = False
        if self.vault_client:
            try:
                vault_path = self._category_to_vault_path(category)
                field_name = name.lower().replace("_", "-")

                # Get existing secrets at this path
                existing_secrets = self.vault_client.get_secret(vault_path) or {}
                existing_secrets[field_name] = value

                if self.vault_client.put_secret(vault_path, existing_secrets):
                    vault_stored = True
                    logger.info(f"Secret '{name}' stored in Vault at {vault_path}")
            except Exception as e:
                logger.error(f"Failed to store {name} in Vault: {e}")

        # Always store locally as backup
        secrets = self._load_secrets()
        if category not in secrets:
            secrets[category] = {}
        secrets[category][name] = value
        self._save_secrets(secrets)

        # Update cache
        self._secrets_cache[f"{category}.{name}"] = value

        storage_info = "Vault and local" if vault_stored else "local"
        logger.info(f"Secret '{name}' stored in {storage_info} storage")

    def delete_secret(self, name: str, category: str = "default") -> bool:
        """Delete a secret from both Vault and local storage"""
        deleted = False

        # Delete from Vault if available
        if self.vault_client:
            try:
                vault_path = self._category_to_vault_path(category)
                existing_secrets = self.vault_client.get_secret(vault_path) or {}
                field_name = name.lower().replace("_", "-")

                if field_name in existing_secrets:
                    del existing_secrets[field_name]
                    if self.vault_client.put_secret(vault_path, existing_secrets):
                        logger.info(f"Secret '{name}' deleted from Vault")
                        deleted = True
            except Exception as e:
                logger.error(f"Failed to delete {name} from Vault: {e}")

        # Delete from local storage
        secrets = self._load_secrets()
        if category in secrets and name in secrets[category]:
            del secrets[category][name]
            if not secrets[category]:
                del secrets[category]

            self._save_secrets(secrets)
            deleted = True

            # Remove from cache
            cache_key = f"{category}.{name}"
            self._secrets_cache.pop(cache_key, None)

            logger.info(f"Secret '{name}' deleted from local storage")

        return deleted

    def list_secrets(self, category: Optional[str] = None) -> Dict[str, list]:
        """List all secret names (not values) by category"""
        all_secrets = {}

        # Get from Vault if available
        if self.vault_client:
            try:
                # List common vault paths
                vault_paths = [
                    "trading/api-keys",
                    "database/credentials",
                    "notifications/webhooks",
                    "general/secrets",
                ]
                # Also probe the flat per-service paths the bot actually
                # reads (secrets/binance, secrets/binance_testnet, ...) per
                # _VAULT_KEY_MAP — field NAMES only, never values.
                flat_paths = sorted({path for path, _field in _VAULT_KEY_MAP.values()})
                for vault_path in vault_paths + flat_paths:
                    try:
                        vault_secrets = self.vault_client.get_secret(vault_path)
                        if vault_secrets:
                            category_name = vault_path.split("/")[-1]
                            all_secrets[f"vault_{category_name}"] = list(
                                vault_secrets.keys()
                            )
                    except:
                        continue
            except Exception as e:
                logger.warning(f"Failed to list Vault secrets: {e}")

        # Get from local storage
        local_secrets = self._load_secrets()
        for cat, names in local_secrets.items():
            key = f"local_{cat}" if self.vault_client else cat
            all_secrets[key] = list(names.keys())

        if category:
            return {k: v for k, v in all_secrets.items() if category in k}

        return all_secrets

    def _category_to_vault_path(self, category: str) -> str:
        """Convert category to Vault path"""
        category_mapping = {
            "api_keys": "trading/api-keys",
            "database": "database/credentials",
            "database/credentials": "database/credentials",
            "trading/api-keys": "trading/api-keys",
            "notifications/telegram": "notifications/telegram",
            "webhooks": "notifications/webhooks",
            "default": "general/secrets",
        }
        return category_mapping.get(category, f"general/{category}")

    def migrate_to_vault(self) -> bool:
        """
        Migrate all local secrets to Vault

        Returns:
            True if successful, False otherwise
        """
        if not self.vault_client:
            logger.error("Cannot migrate: Vault not available")
            return False

        try:
            secrets = self._load_secrets()
            migrated_count = 0
            total_count = 0

            for category, category_secrets in secrets.items():
                vault_path = self._category_to_vault_path(category)

                # Convert keys to Vault format
                vault_secrets = {}
                for name, value in category_secrets.items():
                    field_name = name.lower().replace("_", "-")
                    vault_secrets[field_name] = value
                    total_count += 1

                if vault_secrets:
                    if self.vault_client.put_secret(vault_path, vault_secrets):
                        migrated_count += len(vault_secrets)
                        logger.info(
                            f"Migrated {len(vault_secrets)} secrets to {vault_path}"
                        )

            if migrated_count == total_count:
                logger.info(
                    f"Successfully migrated all {migrated_count} secrets to Vault"
                )
                return True
            else:
                logger.warning(f"Migrated {migrated_count}/{total_count} secrets")
                return False

        except Exception as e:
            logger.error(f"Failed to migrate secrets to Vault: {e}")
            return False

    def get_binance_credentials(self) -> Dict[str, str]:
        """Get Binance API credentials"""
        return {
            "api_key": self.get_secret("BINANCE_API_KEY", "api_keys"),
            "api_secret": self.get_secret("BINANCE_API_SECRET", "api_keys"),
            "futures_testnet_api_key": self.get_secret(
                "BINANCE_FUTURES_TESTNET_API_KEY", "api_keys"
            ),
            "futures_testnet_api_secret": self.get_secret(
                "BINANCE_FUTURES_TESTNET_API_SECRET", "api_keys"
            ),
        }

    def get_database_credentials(self) -> Dict[str, str]:
        """Get database connection credentials"""
        return {
            "host": self.get_secret("POSTGRES_HOST", "database", "localhost"),
            "port": self.get_secret("POSTGRES_PORT", "database", "5432"),
            "user": self.get_secret("POSTGRES_USER", "database", "postgres"),
            "password": self.get_secret("POSTGRES_PASSWORD", "database"),
            "dbname": self.get_secret("POSTGRES_DBNAME", "database", "trading_bot"),
        }

    def get_redis_credentials(self) -> Dict[str, str]:
        """Get Redis connection credentials"""
        # Check if Redis password is actually configured before looking for it
        redis_password = None
        try:
            # Only attempt to get password if it exists, don't warn if missing
            redis_password = self.get_secret(
                "REDIS_PASSWORD", "database", warn_if_missing=False
            )
        except:
            pass  # Password is optional for Redis

        return {
            "host": self.get_secret("REDIS_HOST", "database", "localhost"),
            "port": self.get_secret("REDIS_PORT", "database", "6379"),
            "db": self.get_secret("REDIS_DB", "database", "0"),
            "password": redis_password,  # None if not configured (which is fine)
        }

    def get_telegram_credentials(self) -> Dict[str, str]:
        """Get Telegram bot credentials"""
        return {
            "bot_token": self.get_secret("TELEGRAM_BOT_TOKEN", "api_keys"),
            "chat_id": self.get_secret("TELEGRAM_CHAT_ID", "api_keys"),
        }

    def get_vault_status(self) -> Dict[str, Any]:
        """Get Vault connection status"""
        if not self.vault_client:
            return {
                "enabled": False,
                "connected": False,
                "healthy": False,
                "reason": "Vault client not initialized",
            }

        try:
            healthy = self.vault_client.health_check()
            authenticated = self.vault_client.is_authenticated

            return {
                "enabled": True,
                "connected": authenticated,
                "healthy": healthy,
                "url": self.vault_client.vault_url,
            }
        except Exception as e:
            return {
                "enabled": True,
                "connected": False,
                "healthy": False,
                "error": str(e),
            }

    def _load_secrets(self) -> Dict[str, Dict[str, str]]:
        """Load and decrypt secrets from file"""
        if not self.secrets_file.exists():
            return {}

        try:
            cipher = self._get_cipher()

            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            return {}

    def _save_secrets(self, secrets: Dict[str, Dict[str, str]]) -> None:
        """Encrypt and save secrets to file"""
        try:
            cipher = self._get_cipher()

            json_data = json.dumps(secrets).encode()
            encrypted_data = cipher.encrypt(json_data)

            with open(self.secrets_file, "wb") as f:
                f.write(encrypted_data)

            # Set restrictive permissions (Unix-like systems)
            if hasattr(os, "chmod"):
                os.chmod(self.secrets_file, 0o600)

        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise

    def initialize_from_env(self) -> None:
        """Initialize secrets from environment variables"""
        env_mapping = {
            "api_keys": [
                "BINANCE_API_KEY",
                "BINANCE_API_SECRET",
                "BINANCE_FUTURES_TESTNET_API_KEY",
                "BINANCE_FUTURES_TESTNET_API_SECRET",
                "TELEGRAM_BOT_TOKEN",
                "TELEGRAM_CHAT_ID",
                "GRAFANA_API_KEY",
            ],
            "database": [
                "POSTGRES_HOST",
                "POSTGRES_PORT",
                "POSTGRES_USER",
                "POSTGRES_PASSWORD",
                "POSTGRES_DBNAME",
                "REDIS_HOST",
                "REDIS_PORT",
                "REDIS_DB",
                "REDIS_PASSWORD",
            ],
            "webhooks": ["DISCORD_WEBHOOK_URL", "SLACK_WEBHOOK_URL"],
        }

        initialized = 0
        for category, keys in env_mapping.items():
            for key in keys:
                value = os.getenv(key)
                if value:
                    self.set_secret(key, value, category)
                    initialized += 1

        logger.info(f"Initialized {initialized} secrets from environment variables")


# Backward compatibility aliases
SecretsManager = EnhancedSecretsManager

# Global enhanced secrets manager instance
_enhanced_secrets_manager = None


def get_enhanced_secrets_manager(
    logger: logging.Logger = None,
) -> EnhancedSecretsManager:
    """
    Get or create the global enhanced secrets manager instance.

    Args:
        logger: Logger instance

    Returns:
        EnhancedSecretsManager instance
    """
    global _enhanced_secrets_manager
    if _enhanced_secrets_manager is None:
        _enhanced_secrets_manager = EnhancedSecretsManager()
    return _enhanced_secrets_manager


# Backward compatibility function
def get_secrets_manager(logger: logging.Logger = None) -> EnhancedSecretsManager:
    """Backward compatibility wrapper"""
    return get_enhanced_secrets_manager(logger)


def main(argv: Optional[list] = None) -> int:
    """
    Interactive command-line entry point for managing ELVIS secrets.

    Usage:
        python utils/secrets_manager.py --set NAME [--category CAT]
        python utils/secrets_manager.py --get NAME [--category CAT]
        python utils/secrets_manager.py --list

    ``--set`` prompts for the value with a hidden (``getpass``) input so the
    secret never lands in shell history. ``--get`` only reports presence and
    never prints the stored value. ``--list`` shows secret names grouped by
    source/category (never values).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="secrets_manager",
        description="Manage ELVIS trading bot secrets (Vault + local fallback).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--set",
        metavar="NAME",
        help="Store a secret named NAME (value is prompted, hidden).",
    )
    group.add_argument(
        "--get",
        metavar="NAME",
        help="Report whether the secret NAME is present (never prints value).",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List secret names by category (never prints values).",
    )
    parser.add_argument(
        "--category",
        default="default",
        help="Category for --set/--get (default: 'default').",
    )

    args = parser.parse_args(argv)

    manager = get_enhanced_secrets_manager()

    if args.set:
        value = getpass.getpass(f"Enter value for {args.set}: ")
        if not value:
            print(f"No value entered for {args.set}; nothing stored.")
            return 1
        manager.set_secret(args.set, value, args.category)
        print(f"Secret '{args.set}' stored in category '{args.category}'.")
        return 0

    if args.get:
        value = manager.get_secret(args.get, args.category, warn_if_missing=False)
        if value is not None:
            print(f"Secret '{args.get}' is PRESENT in category '{args.category}'.")
            return 0
        print(f"Secret '{args.get}' is MISSING in category '{args.category}'.")
        return 1

    # --list
    all_secrets = manager.list_secrets()
    if not all_secrets:
        print("No secrets found.")
        return 0
    for category, names in all_secrets.items():
        print(f"{category}:")
        for name in names:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
