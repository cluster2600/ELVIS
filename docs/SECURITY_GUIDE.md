# ELVIS Security Guide

> Extracted from the original README during the 2026-07 root cleanup; content preserved verbatim.

See also: [SECURITY.md](../SECURITY.md) (policy) and [VAULT_INTEGRATION.md](VAULT_INTEGRATION.md).

## Security Documentation

### 🔐 **Enterprise-Grade Security Implementation**

ELVIS Trading Bot implements comprehensive security with HashiCorp Vault integration:

#### **Security Architecture**
- **[SECURITY.md](../SECURITY.md)** - Complete security architecture and implementation details
- **[docs/VAULT_SETUP.md](VAULT_SETUP.md)** - Step-by-step Vault configuration guide
- **[docs/README.md](README.md)** - Documentation index and quick start

#### **Key Security Features**
```
🛡️ HashiCorp Vault / OpenBao Integration
├── KV v2 secrets engine (mount `secrets`) with versioning
├── Fernet-encrypted local fallback (AES-128-CBC + HMAC-SHA256),
│   master key in the OS keyring
├── Lookup order: Vault → environment → encrypted local file
├── Encrypted local cache with 5-minute TTL
└── Live health monitoring in the console dashboard

🔒 Zero Hardcoded Secrets
├── All API keys read at runtime (Vault or env), never committed
└── Role-based access control on the control API (JWT `role` claim)

📊 Security Monitoring
├── Live dashboard with visual indicators (✅❌⏳)
├── Response time tracking per service
└── Overall health percentage
```

> **Honest scope**: this is a personal/experimental bot. It has **no** SOC 2,
> FIPS 140-2, or OWASP certification, and does not use AES-256-GCM. Any formal
> compliance would come from the Vault/OpenBao deployment you run, not from
> this repository. See [SECURITY.md](../SECURITY.md) for the authoritative posture.

#### **Quick Security Setup**
```bash
# 1. Start Vault (Development)
vault server -dev -dev-root-token-id=<choose-a-local-dev-token>

# 2. Configure environment
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN=<choose-a-local-dev-token>

# 3. Store secrets securely (KV v2 mount `secrets`, one path per service)
vault kv put secrets/binance \
    api_key=your-api-key \
    secret_key=your-api-secret

# 4. Verify security status
python main.py --mode paper
# Dashboard shows: ✅ Vault 3ms (connected)
```

#### **Security Monitoring Dashboard**
The console dashboard provides real-time security monitoring:
```
--- API Status ---
✅ Overall: 100%   (connected services ÷ total; example)
✅ Vault        3ms
✅ Binance Spot 45ms  
✅ Postgres     12ms
Other: BIN✅ RED✅ TEL❌ PRO✅
Updated: 18:15:42
```

#### **Emergency Procedures**
- **Secret Compromise**: Immediate token revocation and rotation
- **Vault Unavailable**: Automatic fallback to secure local storage  
- **Authentication Failure**: Alert and graceful degradation
- **Incident Response**: Complete forensic audit trail

For complete security documentation, see **[SECURITY.md](../SECURITY.md)** and **[docs/VAULT_SETUP.md](VAULT_SETUP.md)**.

---

