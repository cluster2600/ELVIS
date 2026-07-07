# HashiCorp Vault Setup Guide - ELVIS Trading Bot

## Overview
This guide provides complete setup instructions for HashiCorp Vault integration with the ELVIS Trading Bot for secure secrets management.

## 🚀 Quick Start (Development)

### 1. Install HashiCorp Vault
```bash
# macOS
brew install vault

# Linux
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install vault

# Windows
choco install vault
```

### 2. Start Development Server
```bash
# Start Vault in development mode (NOT for production)
vault server -dev -dev-root-token-id=trading-bot-token

# Set environment variables
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN=trading-bot-token
```

### 3. Initialize Trading Secrets
The KV v2 secrets engine is mounted at `secrets` (plural), and ELVIS reads the
Binance credentials from one path per service (`secrets/binance`,
`secrets/binance_testnet`) with the fields `api_key` and `secret_key`. This
mapping is defined in `utils/secrets_manager.py` (`_VAULT_KEY_MAP`); the paths
below match it exactly. `vault` and `bao` (OpenBao) are interchangeable CLIs.

```bash
# Store Binance API credentials (the paths ELVIS actually reads)
bao kv put -mount=secrets binance \
    api_key=your-binance-api-key \
    secret_key=your-binance-api-secret

# Store Binance Futures testnet credentials
bao kv put -mount=secrets binance_testnet \
    api_key=your-testnet-api-key \
    secret_key=your-testnet-secret-key
```

Optional legacy category paths (Telegram, database, webhooks) map to
`secrets/notifications/telegram`, `secrets/database/credentials`, and
`secrets/notifications/webhooks` via `_category_to_vault_path` in
`utils/secrets_manager.py`. The Binance credentials are never read from these
paths. Store them only if you use those integrations:

```bash
# Optional: notification / database secrets (category-derived paths)
bao kv put -mount=secrets notifications/telegram \
    telegram-bot-token=your-telegram-bot-token
bao kv put -mount=secrets database/credentials \
    postgres-host=localhost \
    postgres-user=elvis_user \
    postgres-password=your-secure-password \
    redis-password=your-redis-password
```

### 4. Verify Setup
```bash
# Test secret retrieval
bao kv get -mount=secrets binance
bao kv get -mount=secrets binance_testnet

# Check Vault status
bao status
```

## 🏢 Production Setup

### 1. Infrastructure Requirements
```yaml
# Minimum Production Setup
Servers: 3 (HA cluster)
Storage: Consul/Raft backend
TLS: Required for all communications
Authentication: LDAP/AWS/GCP integration
Policies: Role-based access control
Monitoring: Vault audit logs + metrics
Backup: Automated snapshot strategy
```

### 2. Production Configuration
```hcl
# /etc/vault.d/vault.hcl
storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
}

listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "/opt/vault/tls/tls.crt"
  tls_key_file  = "/opt/vault/tls/tls.key"
}

api_addr = "https://vault.company.com:8200"
cluster_addr = "https://vault.company.com:8201"
ui = true
```

### 3. Initialize Production Vault
```bash
# Initialize Vault
vault operator init

# Unseal Vault (3 of 5 key shares needed)
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# Enable KV v2 secrets engine at the `secrets` mount ELVIS expects
vault secrets enable -path=secrets kv-v2
```

### 4. Authentication Setup
```bash
# Enable LDAP authentication
vault auth enable ldap

# Configure LDAP
vault write auth/ldap/config \
    url="ldap://ldap.company.com" \
    userdn="ou=users,dc=company,dc=com" \
    groupdn="ou=groups,dc=company,dc=com" \
    binddn="cn=vault,ou=users,dc=company,dc=com" \
    bindpass="secure-password"
```

### 5. Policy Configuration
```hcl
# trading-bot-policy.hcl
# KV v2 data lives under <mount>/data/... — here the mount is `secrets`.
# ELVIS only needs the two Binance service paths to trade.
path "secrets/data/binance" {
  capabilities = ["read"]
}

path "secrets/data/binance_testnet" {
  capabilities = ["read"]
}

# Optional: grant these only if you use the corresponding integrations.
path "secrets/data/notifications/*" {
  capabilities = ["read"]
}

path "secrets/data/database/*" {
  capabilities = ["read"]
}

# Apply policy
vault policy write trading-bot trading-bot-policy.hcl
```

## 🔧 ELVIS Bot Configuration

### 1. Environment Variables
```bash
# Production environment
export VAULT_ADDR=https://vault.company.com:8200
export VAULT_TOKEN=your-production-token
export VAULT_ENABLED=true

# Development environment  
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN=trading-bot-token
export VAULT_ENABLED=true
```

### 2. Bot Configuration
The ELVIS bot automatically:
- Detects Vault availability
- Falls back to secure local storage if Vault unavailable
- Encrypts local cache with Fernet encryption
- Provides real-time connection monitoring

### 3. Secret Structure
```
Vault KV v2 Paths Used by ELVIS (mount: secrets):
├── secrets/binance/                 # Primary Binance credentials (read by _VAULT_KEY_MAP)
│   ├── api_key                      # Binance API key
│   └── secret_key                   # Binance API secret
└── secrets/binance_testnet/         # Binance Futures testnet credentials
    ├── api_key                      # Testnet API key
    └── secret_key                   # Testnet API secret

Optional legacy category paths (NOT read for Binance; only populated/read
if you use the ELVIS CLIs with the matching category):
├── secrets/notifications/telegram/  # telegram-bot-token, ...
├── secrets/database/credentials/    # postgres-host, postgres-user, ...
└── secrets/notifications/webhooks/  # discord-webhook, slack-webhook
```

## 🛡️ Security Best Practices

### 1. Access Control
```bash
# Create dedicated service account
vault auth enable approle
vault write auth/approle/role/trading-bot \
    token_policies="trading-bot" \
    token_ttl=1h \
    token_max_ttl=4h

# Get role credentials
vault read auth/approle/role/trading-bot/role-id
vault write -f auth/approle/role/trading-bot/secret-id
```

### 2. Token Management
```bash
# Use limited-time tokens
vault write auth/token/create \
    policies="trading-bot" \
    ttl=1h \
    renewable=true

# Enable token renewal in bot
# (automatically handled by ELVIS bot)
```

### 3. Audit Logging
```bash
# Enable audit logging
vault audit enable file file_path=/var/log/vault_audit.log

# Monitor audit logs
tail -f /var/log/vault_audit.log | jq '.'
```

## 📊 Monitoring & Health Checks

### 1. Vault Health Monitoring
The ELVIS bot provides real-time Vault monitoring:
- ✅ **Connected**: Vault accessible and authenticated
- ⚠️ **Warning**: Vault accessible but degraded
- ❌ **Error**: Vault unreachable or authentication failed

### 2. Dashboard Integration
```python
# Real-time status in console dashboard
vault_status = {
    'enabled': True,
    'connected': True, 
    'healthy': True,
    'response_time': 0.003,  # 3ms
    'url': 'https://vault.company.com:8200'
}
```

### 3. Alerting
```bash
# Set up monitoring alerts for:
- Vault seal status
- Authentication failures  
- High response times
- Secret access patterns
- Token expiration warnings
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Connection Refused
```bash
# Check if Vault is running
vault status

# Check network connectivity
curl -k $VAULT_ADDR/v1/sys/health

# Verify environment variables
echo $VAULT_ADDR
echo $VAULT_TOKEN
```

#### 2. Authentication Failed
```bash
# Verify token validity
vault token lookup

# Check token permissions
vault token capabilities secrets/data/binance

# Renew token if needed
vault token renew
```

#### 3. Secrets Not Found
```bash
# List available secrets
vault kv list -mount=secrets /

# Check specific path
vault kv get -mount=secrets binance

# Verify KV version
vault secrets list -detailed
```

### Fallback Behavior
If Vault is unavailable, ELVIS automatically:
1. Uses OS keyring for secret storage
2. Falls back to encrypted local files
3. Displays warning in dashboard
4. Continues trading with cached secrets
5. Automatically reconnects when Vault available

## 📚 Additional Resources

### Documentation
- [Vault Getting Started](https://developer.hashicorp.com/vault/tutorials/getting-started)
- [Production Hardening](https://developer.hashicorp.com/vault/tutorials/security)
- [API Documentation](https://developer.hashicorp.com/vault/api-docs)

### Security References
- [Vault Security Model](https://developer.hashicorp.com/vault/docs/internals/security)
- [Encryption Details](https://developer.hashicorp.com/vault/docs/internals/token)
- [Audit Device Configuration](https://developer.hashicorp.com/vault/docs/audit)

---

**Next Steps**: After setup, verify integration with `python main.py --mode paper` and check the API status widget shows ✅ Vault connected.