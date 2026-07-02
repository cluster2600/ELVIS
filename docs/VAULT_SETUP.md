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
```bash
# Store Binance API credentials
vault kv put secret/trading/api-keys \
    binance-api-key=your-binance-api-key \
    binance-api-secret=your-binance-api-secret \
    telegram-bot-token=your-telegram-bot-token

# Store database credentials
vault kv put secret/database/credentials \
    postgres-host=localhost \
    postgres-user=elvis_user \
    postgres-password=your-secure-password \
    redis-password=your-redis-password

# Store notification webhooks
vault kv put secret/notifications/webhooks \
    discord-webhook=https://discord.com/api/webhooks/... \
    slack-webhook=https://hooks.slack.com/services/...
```

### 4. Verify Setup
```bash
# Test secret retrieval
vault kv get secret/trading/api-keys
vault kv get secret/database/credentials

# Check Vault status
vault status
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

# Enable KV v2 secrets engine
vault secrets enable -path=secret kv-v2
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
path "secret/data/trading/*" {
  capabilities = ["read"]
}

path "secret/data/database/*" {
  capabilities = ["read"]
}

path "secret/data/notifications/*" {
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
Vault KV v2 Paths Used by ELVIS:
├── secret/trading/api-keys/
│   ├── binance-api-key         # Binance API key
│   ├── binance-api-secret      # Binance API secret
│   └── telegram-bot-token      # Telegram notifications
├── secret/database/credentials/
│   ├── postgres-host           # Database host
│   ├── postgres-user           # Database username
│   ├── postgres-password       # Database password
│   └── redis-password          # Redis password (optional)
└── secret/notifications/webhooks/
    ├── discord-webhook         # Discord webhook URL
    └── slack-webhook           # Slack webhook URL
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
vault token capabilities secret/trading/api-keys

# Renew token if needed
vault token renew
```

#### 3. Secrets Not Found
```bash
# List available secrets
vault kv list secret/

# Check specific path
vault kv get secret/trading/api-keys

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