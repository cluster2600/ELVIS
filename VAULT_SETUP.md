# HashiCorp Vault Integration for ELVIS Trading Bot

This document describes how to set up and use HashiCorp Vault for secure secrets management in the ELVIS Trading Bot.

## Features

- **Secure Secret Storage**: All API keys, passwords, and sensitive data stored in Vault
- **Automatic Fallback**: Falls back to environment variables if Vault is unavailable
- **Encrypted Local Cache**: Secrets cached locally with encryption for performance
- **Easy Migration**: Migrate existing `.env` secrets to Vault with one command
- **Admin Tools**: Command-line tools for managing secrets

## Quick Start

### 1. Install Dependencies

```bash
pip install hvac cryptography
```

### 2. Setup Vault Server (Development)

```bash
# Install Vault
brew install vault  # macOS
# or download from https://www.vaultproject.io/downloads

# Start Vault in development mode
vault server -dev
```

### 3. Configure Environment

```bash
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='<dev-token-from-server-output>'
```

### 4. Migrate Secrets

```bash
# Run the complete setup process
python scripts/setup_vault.py --all

# Or step by step:
python scripts/setup_vault.py --validate  # Check Vault connection
python scripts/setup_vault.py --backup    # Backup .env file
python scripts/setup_vault.py --migrate   # Migrate secrets
python scripts/setup_vault.py --verify    # Verify migration
```

### 5. Update Environment File

```bash
# Backup your current .env
cp .env .env.backup

# Use the Vault-ready configuration
cp .env.vault-ready .env

# Edit .env to set your Vault token
VAULT_TOKEN=your_actual_vault_token_here
```

## Managing Secrets

### List All Secrets

```bash
python scripts/vault_admin.py --list
```

Lists field **names only** (never values), covering both the legacy category
paths (`trading/api-keys`, `database/credentials`, …) and the flat per-service
paths the bot actually reads via `_VAULT_KEY_MAP` — `secrets/binance` and
`secrets/binance_testnet` (`api_key` / `secret_key`).

### Add a New Secret

```bash
python scripts/vault_admin.py --add
```

### Update Existing Secret

```bash
python scripts/vault_admin.py --update
```

### Delete a Secret

```bash
python scripts/vault_admin.py --delete
```

### Test Credential Retrieval

```bash
python scripts/vault_admin.py --test
```

### Check Vault Status

```bash
python scripts/vault_admin.py --status
```

## Secret Organization

Secrets are organized in the following structure:

```
secret/
├── trading/
│   ├── api-keys/          # Binance API credentials
│   └── binance-testnet/   # Testnet credentials
├── database/
│   └── credentials/       # PostgreSQL credentials
├── notifications/
│   └── webhooks/          # Telegram, Discord webhooks
├── monitoring/
│   ├── prometheus/        # Monitoring credentials
│   └── grafana/          # Grafana API keys
└── general/
    └── secrets/          # Other secrets
```

## Code Usage

### Using Enhanced Secrets Manager

```python
from utils.secrets_manager_enhanced import get_enhanced_secrets_manager

# Initialize
secrets = get_enhanced_secrets_manager()

# Get individual secret
api_key = secrets.get_secret('BINANCE_API_KEY', 'api_keys')

# Get credential groups
binance_creds = secrets.get_binance_credentials()
db_creds = secrets.get_database_credentials()
```

### Automatic Fallback

The system automatically falls back to environment variables if:
- Vault is disabled (`VAULT_ENABLED=false`)
- Vault is unreachable
- Secret not found in Vault

## Production Deployment

### 1. Production Vault Server

```bash
# Install Vault on production server
# Configure with proper TLS, authentication, and policies

# Example production config
vault server -config=/etc/vault/vault.hcl
```

### 2. Environment Variables

```bash
export VAULT_ENABLED=true
export VAULT_URL=https://vault.yourcompany.com:8200
export VAULT_TOKEN=your_production_token
export VAULT_CACHE_KEY=your_production_cache_key
```

### 3. Security Considerations

- Use TLS for Vault server communication
- Implement proper Vault policies and authentication
- Rotate Vault tokens regularly
- Monitor Vault access logs
- Use dedicated Vault tokens per environment

## Troubleshooting

### Vault Connection Issues

```bash
# Check Vault status
python scripts/vault_admin.py --status

# Validate connection
python scripts/setup_vault.py --validate
```

### Missing Secrets

```bash
# List all secrets
python scripts/vault_admin.py --list

# Test credential retrieval
python scripts/vault_admin.py --test
```

### Fallback to Environment

If Vault is unavailable, the system automatically falls back to environment variables. Check logs for warnings:

```
WARNING: Failed to get BINANCE_API_KEY from Vault, falling back to environment
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VAULT_ENABLED` | Enable/disable Vault | `true` |
| `VAULT_URL` | Vault server URL | `http://localhost:8200` |
| `VAULT_TOKEN` | Vault authentication token | None |
| `VAULT_CACHE_KEY` | Cache encryption key | Auto-generated |

### Vault Client Settings

- **Cache TTL**: 300 seconds (5 minutes)
- **Connection Timeout**: 5 seconds
- **Mount Point**: `secret` (KV v2)
- **Retry Policy**: Automatic fallback to environment

## Migration from .env

The migration process:

1. **Backup**: Creates `.env.backup`
2. **Parse**: Reads current environment variables
3. **Categorize**: Groups secrets by type
4. **Store**: Saves to appropriate Vault paths
5. **Verify**: Tests retrieval of migrated secrets

### Manual Migration

If automatic migration fails, manually add secrets:

```bash
# Add Binance API credentials
python scripts/vault_admin.py --add
# Category: api_keys
# Name: BINANCE_API_KEY
# Value: your_api_key

python scripts/vault_admin.py --add
# Category: api_keys  
# Name: BINANCE_API_SECRET
# Value: your_api_secret
```

## Backups

`scripts/vault_admin.py --backup` writes an **encrypted** dump of the known
Vault secret paths so credentials can be restored after a Vault rebuild or
disaster. Plaintext secrets are never written to disk.

### How it works

1. **Collect** — reads each known KV path through the secrets manager's Vault
   client: `secrets/binance`, `secrets/binance_testnet`, plus every distinct
   path referenced by `_VAULT_KEY_MAP` in `utils/secrets_manager.py`. The set
   stays in sync with the secrets ELVIS actually reads.
2. **Encrypt** — the collected secrets are serialised to JSON and encrypted
   with [Fernet](https://cryptography.io/en/latest/fernet/) (AES-128-CBC +
   HMAC). Only the ciphertext is ever written; the JSON never touches disk.
3. **Write** — the ciphertext is saved to the `--out` path (default
   `.vault-backup.enc`, which is gitignored) with `0600` permissions.

The Fernet key comes from the `BACKUP_KEY` environment variable. If it is
unset, a fresh key is generated and **printed once** — save it, because it is
the only way to decrypt the backup and it will not be shown again.

### How to use

```bash
# Reuse a persistent key (recommended for scripted/scheduled backups)
export BACKUP_KEY='<your-fernet-key>'
python scripts/vault_admin.py --backup                 # -> .vault-backup.enc
python scripts/vault_admin.py --backup --out /secure/vault-2026.enc

# First run without BACKUP_KEY: a key is generated and printed once.
python scripts/vault_admin.py --backup
# 🔑 ...  BACKUP_KEY=<copy this and store it in a password manager>
```

Restore is a decrypt: load the file, `Fernet(BACKUP_KEY).decrypt(...)`, then
`json.loads(...)` to recover a `{path: {field: value}}` mapping and re-`--add`
the secrets. Store `BACKUP_KEY` **separately** from the `.enc` file — the
backup is only as safe as the key.

## Monitoring and Logging

The Vault integration logs all operations:

- Secret retrieval attempts
- Fallback to environment variables
- Cache hits/misses
- Connection issues
- Migration progress

Example log output:

```
INFO: Using HashiCorp Vault for secrets management
DEBUG: Retrieved BINANCE_API_KEY from Vault
WARNING: Failed to get TELEGRAM_TOKEN from Vault, falling back to environment
INFO: Secret 'POSTGRES_PASSWORD' stored in Vault and local storage
```

## Support

For issues with Vault integration:

1. Check Vault server status
2. Verify network connectivity
3. Test authentication token
4. Review application logs
5. Use fallback mode if needed

The system is designed to be resilient and will continue operating even if Vault becomes unavailable by falling back to environment variables.