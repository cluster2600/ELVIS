"""
Secrets Manager for ELVIS Trading Bot
Provides secure handling of sensitive configuration data
"""

import os
import json
import base64
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import getpass

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manage secrets securely using encryption and OS keyring"""
    
    def __init__(self, app_name: str = "ELVIS_TRADING_BOT"):
        self.app_name = app_name
        self.secrets_file = Path.home() / ".elvis" / "secrets.enc"
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = None
        self._secrets_cache = {}
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key from OS keyring"""
        key_name = f"{self.app_name}_MASTER_KEY"
        
        # Try to get existing key from keyring
        stored_key = keyring.get_password(self.app_name, key_name)
        
        if stored_key:
            return base64.b64decode(stored_key.encode())
        
        # Generate new key
        logger.info("Generating new master encryption key...")
        key = Fernet.generate_key()
        
        # Store in keyring
        keyring.set_password(
            self.app_name,
            key_name,
            base64.b64encode(key).decode()
        )
        
        return key
    
    def _get_cipher(self) -> Fernet:
        """Get or create cipher for encryption/decryption"""
        if not self._cipher:
            key = self._get_or_create_key()
            self._cipher = Fernet(key)
        return self._cipher
    
    def set_secret(self, name: str, value: str, category: str = "default") -> None:
        """
        Store a secret securely
        
        Args:
            name: Secret name (e.g., 'BINANCE_API_KEY')
            value: Secret value
            category: Category for organization (e.g., 'api_keys', 'passwords')
        """
        # Load existing secrets
        secrets = self._load_secrets()
        
        # Update secret
        if category not in secrets:
            secrets[category] = {}
        secrets[category][name] = value
        
        # Save encrypted
        self._save_secrets(secrets)
        
        # Update cache
        self._secrets_cache[f"{category}.{name}"] = value
        
        logger.info(f"Secret '{name}' stored in category '{category}'")
    
    def get_secret(self, name: str, category: str = "default", 
                   default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a secret
        
        Args:
            name: Secret name
            category: Category name
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        cache_key = f"{category}.{name}"
        
        # Check cache first
        if cache_key in self._secrets_cache:
            return self._secrets_cache[cache_key]
        
        # Load from encrypted file
        secrets = self._load_secrets()
        
        if category in secrets and name in secrets[category]:
            value = secrets[category][name]
            self._secrets_cache[cache_key] = value
            return value
        
        # Check environment variable as fallback
        env_value = os.getenv(name)
        if env_value:
            return env_value
        
        return default
    
    def delete_secret(self, name: str, category: str = "default") -> bool:
        """Delete a secret"""
        secrets = self._load_secrets()
        
        if category in secrets and name in secrets[category]:
            del secrets[category][name]
            if not secrets[category]:
                del secrets[category]
            
            self._save_secrets(secrets)
            
            # Remove from cache
            cache_key = f"{category}.{name}"
            self._secrets_cache.pop(cache_key, None)
            
            logger.info(f"Secret '{name}' deleted from category '{category}'")
            return True
        
        return False
    
    def list_secrets(self, category: Optional[str] = None) -> Dict[str, list]:
        """List all secret names (not values) by category"""
        secrets = self._load_secrets()
        
        if category:
            return {category: list(secrets.get(category, {}).keys())}
        
        return {cat: list(names.keys()) for cat, names in secrets.items()}
    
    def _load_secrets(self) -> Dict[str, Dict[str, str]]:
        """Load and decrypt secrets from file"""
        if not self.secrets_file.exists():
            return {}
        
        try:
            cipher = self._get_cipher()
            
            with open(self.secrets_file, 'rb') as f:
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
            
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                os.chmod(self.secrets_file, 0o600)
                
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise
    
    def initialize_from_env(self) -> None:
        """Initialize secrets from environment variables"""
        env_mapping = {
            'api_keys': [
                'BINANCE_API_KEY',
                'BINANCE_API_SECRET',
                'TELEGRAM_BOT_TOKEN',
                'GRAFANA_API_KEY'
            ],
            'database': [
                'POSTGRES_PASSWORD',
                'REDIS_PASSWORD'
            ],
            'webhooks': [
                'DISCORD_WEBHOOK_URL',
                'SLACK_WEBHOOK_URL'
            ]
        }
        
        initialized = 0
        for category, keys in env_mapping.items():
            for key in keys:
                value = os.getenv(key)
                if value:
                    self.set_secret(key, value, category)
                    initialized += 1
        
        logger.info(f"Initialized {initialized} secrets from environment variables")


class ConfigLoader:
    """Load configuration with secrets integration"""
    
    def __init__(self, secrets_manager: Optional[SecretsManager] = None):
        self.secrets_manager = secrets_manager or SecretsManager()
    
    def load_config(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration with secrets replacement
        
        Config file can contain placeholders like:
        {
            "api_key": "${SECRET:api_keys.BINANCE_API_KEY}",
            "password": "${SECRET:database.POSTGRES_PASSWORD}"
        }
        """
        if not config_file:
            config_file = Path("config/config.json")
        
        if not config_file.exists():
            return {}
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Replace secret placeholders
        return self._replace_secrets(config)
    
    def _replace_secrets(self, obj: Any) -> Any:
        """Recursively replace secret placeholders in configuration"""
        if isinstance(obj, dict):
            return {k: self._replace_secrets(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._replace_secrets(item) for item in obj]
        
        elif isinstance(obj, str) and obj.startswith("${SECRET:"):
            # Extract secret reference
            secret_ref = obj[9:-1]  # Remove ${SECRET: and }
            
            if '.' in secret_ref:
                category, name = secret_ref.split('.', 1)
                return self.secrets_manager.get_secret(name, category) or obj
            else:
                return self.secrets_manager.get_secret(secret_ref) or obj
        
        return obj


def interactive_setup():
    """Interactive setup for secrets"""
    print("🔐 ELVIS Trading Bot - Secrets Setup")
    print("=" * 40)
    
    manager = SecretsManager()
    
    # Check if already initialized
    existing = manager.list_secrets()
    if existing:
        print("\n📋 Existing secrets found:")
        for category, names in existing.items():
            print(f"  {category}: {', '.join(names)}")
        
        if input("\nReinitialize secrets? (y/N): ").lower() != 'y':
            return
    
    print("\n🔑 Enter your API credentials (leave blank to skip):")
    
    # Binance API
    binance_key = getpass.getpass("Binance API Key: ")
    if binance_key:
        binance_secret = getpass.getpass("Binance API Secret: ")
        manager.set_secret("BINANCE_API_KEY", binance_key, "api_keys")
        manager.set_secret("BINANCE_API_SECRET", binance_secret, "api_keys")
    
    # Telegram Bot
    telegram_token = getpass.getpass("Telegram Bot Token: ")
    if telegram_token:
        telegram_chat_id = input("Telegram Chat ID: ")
        manager.set_secret("TELEGRAM_BOT_TOKEN", telegram_token, "api_keys")
        manager.set_secret("TELEGRAM_CHAT_ID", telegram_chat_id, "api_keys")
    
    print("\n✅ Secrets stored securely!")
    print("\n📝 Usage example:")
    print("  from utils.secrets_manager import SecretsManager")
    print("  secrets = SecretsManager()")
    print("  api_key = secrets.get_secret('BINANCE_API_KEY', 'api_keys')")


if __name__ == "__main__":
    interactive_setup()
