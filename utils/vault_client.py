"""
HashiCorp Vault client for secure secret management.
"""

import os
import logging
import hvac
from typing import Dict, Any, Optional
from functools import lru_cache
import time
from cryptography.fernet import Fernet
import base64

class VaultClient:
    """
    HashiCorp Vault client for managing trading bot secrets.
    
    This client provides secure access to secrets stored in Vault,
    with caching and fallback mechanisms for reliability.
    """
    
    def __init__(self, 
                 vault_url: str = None,
                 vault_token: str = None,
                 mount_point: str = 'secrets',
                 logger: logging.Logger = None):
        """
        Initialize Vault client.
        
        Args:
            vault_url: Vault server URL (defaults to env VAULT_ADDR or VAULT_URL)
            vault_token: Vault token (defaults to env VAULT_TOKEN)
            mount_point: KV secrets engine mount point
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.vault_url = vault_url or os.getenv('VAULT_ADDR', os.getenv('VAULT_URL', 'http://localhost:8200'))
        self.vault_token = vault_token or os.getenv('VAULT_TOKEN')
        self.mount_point = mount_point
        
        # Initialize client
        self.client = None
        self.is_authenticated = False
        self._secret_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._last_cache_update = {}
        
        # Initialize encryption for local cache
        self._init_cache_encryption()
        
        # Initialize connection
        self._initialize_client()
    
    def _init_cache_encryption(self):
        """Initialize encryption for local secret cache."""
        try:
            # Try to get encryption key from environment or generate one
            key = os.getenv('VAULT_CACHE_KEY')
            if not key:
                # Generate a new key and warn about persistence
                key = Fernet.generate_key().decode()
                self.logger.warning("Generated new cache encryption key. Set VAULT_CACHE_KEY env var to persist across restarts.")
            
            self._cipher = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception as e:
            self.logger.error(f"Failed to initialize cache encryption: {e}")
            self._cipher = None
    
    def _initialize_client(self):
        """Initialize and authenticate Vault client."""
        try:
            if not self.vault_token:
                self.logger.error("Vault token not provided. Set VAULT_TOKEN environment variable.")
                return False
            
            # Create client
            self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
            
            # Test authentication
            if self.client.is_authenticated():
                self.is_authenticated = True
                self.logger.info(f"Successfully authenticated with Vault at {self.vault_url}")
                return True
            else:
                self.logger.error("Failed to authenticate with Vault")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Vault client: {e}")
            return False
    
    def _is_cache_valid(self, secret_path: str) -> bool:
        """Check if cached secret is still valid."""
        if secret_path not in self._last_cache_update:
            return False
        
        last_update = self._last_cache_update[secret_path]
        return (time.time() - last_update) < self._cache_ttl
    
    def _encrypt_for_cache(self, data: str) -> str:
        """Encrypt data for local cache storage."""
        if not self._cipher:
            return data  # Fallback to unencrypted if encryption failed
        
        try:
            return self._cipher.encrypt(data.encode()).decode()
        except Exception as e:
            self.logger.warning(f"Failed to encrypt cache data: {e}")
            return data
    
    def _decrypt_from_cache(self, encrypted_data: str) -> str:
        """Decrypt data from local cache."""
        if not self._cipher:
            return encrypted_data  # Fallback if encryption not available
        
        try:
            return self._cipher.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            self.logger.warning(f"Failed to decrypt cache data: {e}")
            return encrypted_data
    
    def get_secret(self, secret_path: str, key: str = None) -> Optional[Any]:
        """
        Retrieve a secret from Vault.
        
        Args:
            secret_path: Path to the secret in Vault (e.g., 'trading/binance')
            key: Specific key within the secret (optional)
            
        Returns:
            Secret value or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{secret_path}:{key}" if key else secret_path
            if self._is_cache_valid(cache_key) and cache_key in self._secret_cache:
                cached_data = self._decrypt_from_cache(self._secret_cache[cache_key])
                self.logger.debug(f"Retrieved secret from cache: {secret_path}")
                return cached_data
            
            # Ensure client is authenticated
            if not self.is_authenticated:
                if not self._initialize_client():
                    self.logger.error("Cannot retrieve secret: Vault not authenticated")
                    return None
            
            # Retrieve from Vault
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_path,
                mount_point=self.mount_point
            )
            
            if response and 'data' in response and 'data' in response['data']:
                secret_data = response['data']['data']
                
                # Cache the entire secret
                encrypted_data = self._encrypt_for_cache(str(secret_data))
                self._secret_cache[secret_path] = encrypted_data
                self._last_cache_update[secret_path] = time.time()
                
                # Return specific key or entire secret
                if key:
                    value = secret_data.get(key)
                    if value:
                        # Cache individual key as well
                        encrypted_value = self._encrypt_for_cache(str(value))
                        self._secret_cache[cache_key] = encrypted_value
                        self._last_cache_update[cache_key] = time.time()
                    return value
                else:
                    return secret_data
                    
            else:
                self.logger.warning(f"Secret not found: {secret_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret {secret_path}: {e}")
            return None
    
    def put_secret(self, secret_path: str, secret_data: Dict[str, Any]) -> bool:
        """
        Store a secret in Vault.
        
        Args:
            secret_path: Path where to store the secret
            secret_data: Dictionary of key-value pairs to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure client is authenticated
            if not self.is_authenticated:
                if not self._initialize_client():
                    self.logger.error("Cannot store secret: Vault not authenticated")
                    return False
            
            # Store in Vault
            response = self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_path,
                secret=secret_data,
                mount_point=self.mount_point
            )
            
            # Invalidate cache for this path
            self._invalidate_cache(secret_path)
            
            self.logger.info(f"Successfully stored secret: {secret_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret {secret_path}: {e}")
            return False
    
    def _invalidate_cache(self, secret_path: str):
        """Invalidate cache entries for a secret path."""
        keys_to_remove = [key for key in self._secret_cache.keys() if key.startswith(secret_path)]
        for key in keys_to_remove:
            del self._secret_cache[key]
            del self._last_cache_update[key]
    
    def list_secrets(self, path: str = '') -> Optional[list]:
        """
        List secrets at a given path.
        
        Args:
            path: Path to list secrets from
            
        Returns:
            List of secret names or None if failed
        """
        try:
            if not self.is_authenticated:
                if not self._initialize_client():
                    return None
            
            response = self.client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=self.mount_point
            )
            
            if response and 'data' in response and 'keys' in response['data']:
                return response['data']['keys']
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to list secrets at {path}: {e}")
            return None
    
    def delete_secret(self, secret_path: str) -> bool:
        """
        Delete a secret from Vault.
        
        Args:
            secret_path: Path of the secret to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_authenticated:
                if not self._initialize_client():
                    return False
            
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=secret_path,
                mount_point=self.mount_point
            )
            
            # Invalidate cache
            self._invalidate_cache(secret_path)
            
            self.logger.info(f"Successfully deleted secret: {secret_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret {secret_path}: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if Vault is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Use a simple authenticated operation instead of health status
            # which may not return the expected format
            try:
                # Try to list auth methods - this requires authentication
                self.client.sys.list_auth_methods()
                return True
            except:
                # Fallback: try a simple sys operation
                self.client.sys.read_health_status()
                return True
            
        except Exception as e:
            self.logger.error(f"Vault health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear the local secret cache."""
        self._secret_cache.clear()
        self._last_cache_update.clear()
        self.logger.info("Cleared secret cache")


# Global Vault client instance
_vault_client = None

def get_vault_client(logger: logging.Logger = None) -> VaultClient:
    """
    Get or create the global Vault client instance.
    
    Args:
        logger: Logger instance
        
    Returns:
        VaultClient instance
    """
    global _vault_client
    if _vault_client is None:
        _vault_client = VaultClient(logger=logger)
    return _vault_client