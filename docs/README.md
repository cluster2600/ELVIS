# ELVIS Trading Bot Documentation

## 📚 Documentation Index

### 🔐 Security & Configuration
- **[SECURITY.md](../SECURITY.md)** - Complete security implementation with HashiCorp Vault
- **[VAULT_SETUP.md](VAULT_SETUP.md)** - Step-by-step Vault configuration guide

### 📊 System Architecture
- **[API Monitoring](../utils/api_connection_tester.py)** - Real-time API health monitoring
- **[Console Dashboard](../utils/console_dashboard.py)** - Live trading dashboard with visual indicators
- **[Secrets Management](../utils/secrets_manager.py)** - Multi-layer security with Vault integration

### 🚀 Trading Features
- **Maximum Speed Trading** - Zero cooldowns for ultra-fast execution
- **Real-time Monitoring** - Live API status with response time tracking
- **Enterprise Security** - Vault-based secrets management with encryption

### 🛡️ Security Features

#### HashiCorp Vault Integration
- **Centralized Secret Management**: All API keys stored in Vault KV v2 engine
- **AES-256-GCM Encryption**: Strong encryption for secrets
- **Multi-Layer Fallback**: Vault → OS Keyring → Encrypted Files → Environment
- **Audit Trail**: Complete logging of all secret access
- **Real-time Monitoring**: Live health checks and status indicators

#### Security Compliance
- **OWASP Top 10**: Follows industry security standards
- **SOC 2 Type II**: Vault provides enterprise compliance
- **FIPS 140-2**: Cryptographic module compliance
- **No Production Secrets in Source**: Credentials must come from Vault/environment

### 📈 Performance Optimizations
- **Zero Cooldowns**: Maximum trading speed with no artificial delays
- **Low-Latency Vault Checks**: Vault connectivity and latency are monitored
- **Health Visibility**: Critical services are continuously monitored
- **Error-Free Operation**: Comprehensive NoneType protection

### 🔎 Security Assumptions
- Development may use `vault server -dev` with a local, explicit token.
- Production must not use dev mode, shared/fixed tokens, or committed credentials.
- API keys and database credentials must be injected via Vault or secure environment variables.

### 🔧 Quick Setup

#### 1. Start Vault (Development)
```bash
export VAULT_DEV_ROOT_TOKEN_ID="<choose-a-local-dev-token>"
vault server -dev -dev-root-token-id="$VAULT_DEV_ROOT_TOKEN_ID"
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN="$VAULT_DEV_ROOT_TOKEN_ID"
```

#### 2. Initialize Secrets
```bash
vault kv put secret/trading/api-keys \
    binance-api-key=your-api-key \
    binance-api-secret=your-api-secret
```

#### 3. Start Trading Bot
```bash
python main.py --mode dashboard
```

#### 4. Verify Security
- Check dashboard shows ✅ Vault connected
- Verify API health at 88%+
- Confirm zero NoneType errors

### 📊 Dashboard Features

#### Real-time API Status
```
--- API Status ---
✅ Overall: 88%
✅ Vault        3ms
✅ Binance Spot 45ms  
✅ Postgres     12ms
✅ Redis        32ms
Other: BIN✅ RED✅ TEL❌ PRO✅
Updated: 18:15:42
```

#### Visual Indicators
- ✅ **Connected** - Service operational and healthy
- ⚠️ **Warning** - Connected but degraded performance  
- ❌ **Error** - Service unavailable or failed
- ⏳ **Testing** - Currently running health check
- ❓ **Unknown** - Status not yet determined

### 🔍 Monitoring & Alerts

#### Security Monitoring
- **Vault Health**: Real-time authentication and connectivity
- **Secret Access**: Comprehensive audit logging
- **Token Status**: Automatic renewal and expiration tracking
- **Encryption Status**: Cache and storage encryption verification

#### Performance Monitoring
- **API Response Times**: Sub-millisecond precision tracking
- **Health Percentages**: Real-time system health calculation
- **Error Rates**: Zero-tolerance error monitoring
- **Uptime Tracking**: Continuous availability monitoring

### 🚨 Emergency Procedures

#### Security Incidents
1. **Secret Compromise**: Immediate token revocation and rotation
2. **Vault Unavailable**: Automatic fallback to secure local storage
3. **Authentication Failure**: Alert and graceful degradation
4. **Audit Trail**: Complete forensic logging for investigation

#### System Recovery
1. **Service Restart**: Automatic reconnection to all APIs
2. **Fallback Mode**: Continue trading with cached credentials
3. **Health Recovery**: Real-time monitoring of service restoration
4. **Alert Resolution**: Automatic clearing of resolved issues

---

**For detailed setup instructions, see [VAULT_SETUP.md](VAULT_SETUP.md)**  
**For security details, see [SECURITY.md](../SECURITY.md)**  
**For support, check the main [README.md](../README.md)**
