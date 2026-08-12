# Security Documentation - ELVIS Trading Bot

## Overview
ELVIS Trading Bot integrates with HashiCorp Vault / OpenBao for secrets
management, with env-var and encrypted-local-file fallback, and protects the
Flask trade-history API with an `X-API-Key` header. This document describes the
security behaviour that is actually implemented in the code.

## Sprint-1 Hardening Updates (2026-02-18)

These security and exposure changes were shipped in the Sprint-1 hardening pass:

- Removed hardcoded Vault token behavior.
  - `VAULT_TOKEN` must now be provided externally when Vault-backed secret loading is required.
- Removed hardcoded Postgres password fallback.
  - `POSTGRES_PASSWORD` has no default and must be supplied via environment/secrets manager.
- Changed Trade History API exposure to local-only by default.
  - `TRADE_HISTORY_API_HOST` default: `127.0.0.1`
  - `TRADE_HISTORY_API_PORT` default: `5050`
- Added API key authentication to the Flask trade-history API (`X-API-Key` header).
  - Implemented as a `@app.before_request` hook (`require_api_key`) in
    `trading/utils/trade_history_api.py`.
  - The expected key is read from the `API_KEY` environment variable.
  - Both `/health` and `/metrics` are exempt so Docker/load-balancer health
    checks and the Prometheus scraper (which does not send `X-API-Key`) still
    work.
  - Fails closed: if `API_KEY` is not set on the server, every non-exempt
    request is rejected with `503` ("API authentication not configured").
    A request with a wrong/missing key gets `401`.

Operational implications:
- Prometheus can scrape `/metrics` without any credentials because that path is
  already exempt from the `X-API-Key` check — no extra Prometheus config or
  middleware change is required.
- For remote/API access from outside localhost (for example in containerized
  deployments), set `TRADE_HISTORY_API_HOST=0.0.0.0` intentionally and provide
  `API_KEY` so authenticated endpoints are reachable.

## Security Hardening Follow-up (2026-07-05)

- Removed residual hardcoded Vault token from the health-check path.
  - `utils/api_connection_tester.py` no longer injects
    `os.environ["VAULT_TOKEN"] = "trading-bot-token"` nor falls back to that
    literal when constructing the `hvac` client.
  - `VAULT_TOKEN` must now come from the environment. When it is unset, the
    Vault connectivity check logs a warning and is skipped gracefully
    (reported as `DISCONNECTED` with `error_message="VAULT_TOKEN not set"`),
    keeping the "Zero Hardcoded Secrets" guarantee intact.
  - Regression coverage: `tests/test_api_connection_tester_vault_token.py`
    asserts the `trading-bot-token` literal is absent and that a missing
    `VAULT_TOKEN` is not silently defaulted.

## 🧑‍⚖️ Role-Based Access Control (control API)

**How it works:** the control API (`trading/api/app.py`) issues JWTs with a
`role` claim at `/api/auth/login` (`API_ROLE` env var, default `admin` for the
single env-configured operator). A `require_role("admin")` decorator protects
the mutating endpoints — `/api/bot/start`, `/api/bot/stop`, `PUT /api/config`,
`/api/dashboard/broadcast/alert` — returning **403** for other roles. Read
endpoints require any valid token. Tokens **without** a role claim (issued
before RBAC) are treated as read-only `viewer`, so old tokens keep working for
reads but can no longer mutate.

**How to use:** set `API_ROLE=viewer` before issuing tokens for dashboards or
monitors that must not control the bot; leave unset (admin) for operator
tokens. Coverage: `tests/test_api_rbac.py`.

**Network binding:** both Flask apps bind `127.0.0.1` by default; set
`API_HOST=0.0.0.0` / `TRADE_API_HOST=0.0.0.0` (docker-compose does) to expose
them, and `API_DEBUG=true` explicitly for debug mode.

## 🔐 HashiCorp Vault Integration

The bot targets HashiCorp Vault / OpenBao's KV v2 secrets engine for API keys
and credentials, with graceful fallback to environment variables and an
encrypted local file when Vault is unavailable.

### Security Architecture
- **Centralized Secret Management**: API keys and credentials are read from
  Vault when it is reachable, using the `hvac` client.
- **Encryption at Rest**: Vault encrypts stored secrets on its backend; the
  local secret cache and fallback file are encrypted with Fernet.
- **Encryption in Transit**: Communication with Vault uses whatever scheme the
  `VAULT_ADDR` URL specifies (use `https://` + TLS in production; the dev
  default is `http://localhost:8200`).
- **Token-Based Authentication**: The client authenticates with the token from
  `VAULT_TOKEN`; no token is hardcoded.

### Implementation Details

#### Vault Client (`utils/vault_client.py`)
- **Purpose**: Secure interface to the Vault KV v2 secrets engine.
- **KV mount point**: `secrets` (the `VaultClient` default; the runtime never
  overrides it). Secrets are read from `secrets/<path>`.
- **Configuration**: `VAULT_ADDR` (or `VAULT_URL`, default
  `http://localhost:8200`) and `VAULT_TOKEN`. If `VAULT_TOKEN` is unset the
  client logs an error and stays unauthenticated.
- **Features**:
  - In-memory secret cache, each entry encrypted with Fernet before storage.
  - TTL-based cache invalidation (`_cache_ttl = 300`, i.e. 5 minutes).
  - The Fernet cache key comes from `VAULT_CACHE_KEY`; if unset a random key is
    generated per process (cache does not survive restarts) and a warning is
    logged.
  - On demand re-authentication: `get_secret`/`put_secret` call
    `_initialize_client()` again if the client is not authenticated. There is
    no background token-renewal loop.

#### Enhanced Secrets Manager (`utils/secrets_manager.py`)
`EnhancedSecretsManager.get_secret(name, category, default, warn_if_missing)`
resolves a secret in this order:
  1. **Process cache** — values already fetched this run.
  2. **HashiCorp Vault** — only if enabled (`VAULT_ENABLED` != `false`, Vault
     importable, and the startup health check passed).
  3. **Environment variable** named `name` (checked before the local file
     because reading the file can touch the macOS Keychain and block).
  4. **Encrypted local file** `~/.elvis/secrets.enc` — a Fernet-encrypted JSON
     store, written with `0o600` permissions.
  5. **`default`** if nothing is found.

The **OS keyring** is not a secret-storage tier; it only holds the Fernet
*master key* used to encrypt/decrypt `~/.elvis/secrets.enc` (read on a daemon
thread with a 10s timeout, degrading to an ephemeral key if the Keychain
prompt blocks).

#### Secret Storage Hierarchy
Known Binance secrets use OpenBao's native one-path-per-service layout under
the `secrets` mount (see `_VAULT_KEY_MAP` in `utils/secrets_manager.py`), not
the legacy `trading/api-keys/binance-api-key` scheme:
```
Vault KV v2 mount: secrets
├── secrets/binance
│   ├── api_key      ← BINANCE_API_KEY
│   └── secret_key   ← BINANCE_API_SECRET
└── secrets/binance_testnet
    ├── api_key      ← BINANCE_FUTURES_TESTNET_API_KEY
    └── secret_key   ← BINANCE_FUTURES_TESTNET_API_SECRET
```
Secret names not in `_VAULT_KEY_MAP` (e.g. Postgres/Telegram credentials) fall
back to a generic derivation: the category maps to a path via
`_category_to_vault_path()` (e.g. `database` → `database/credentials`) and the
field name is `name.lower().replace("_", "-")`.

### Security Benefits

#### 1. **Zero Hardcoded Secrets**
- No API keys, passwords, or tokens in source code
- All sensitive data retrieved dynamically from secure storage
- Reduces risk of accidental exposure in logs or code repositories

#### 2. **Principle of Least Privilege**
- Each component requests only the secrets it needs, via
  `get_secret(name, category)`.
- Role-based access control and time-limited tokens are Vault-side features you
  configure on the server; the bot does not manage Vault policies at runtime.
  `scripts/setup_vault.py --validate` prints a sample policy but notes that
  dev-mode Vault grants full access.

#### 3. **Audit Trail (Vault-side)**
- Secret access auditing is a HashiCorp Vault / OpenBao server feature (enable
  an audit device on the server). The bot itself does not implement a
  secret-access audit log; it logs its own retrieval attempts at DEBUG level
  without secret values.

#### 4. **Encryption**
- **At Rest**: Vault encrypts secrets on its backend; the local cache and the
  `~/.elvis/secrets.enc` fallback file are Fernet-encrypted.
- **In Transit**: TLS applies when `VAULT_ADDR` uses `https://` (production).
  The development default is plain `http://`.

#### 5. **Graceful Degradation**
- The system keeps operating if Vault is unavailable: the startup health check
  drops the Vault client and the secrets manager falls back to environment
  variables and the encrypted local file.
- `utils/api_connection_tester.py` monitors Vault connectivity and reports it
  through the dashboard/health summary.

## 🔧 Configuration & Setup

### Development Environment
```bash
# Start Vault dev server (pick any dev token; it is not baked into the code)
vault server -dev -dev-root-token-id=my-dev-token

# In another terminal, point the bot at it (both vars must be set externally)
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN=my-dev-token

# The bot reads from the `secrets` KV v2 mount. Vault dev mode enables a KV v2
# engine at `secret/`, so create the `secrets` mount the client expects:
vault secrets enable -path=secrets kv-v2

# Store Binance credentials at the paths the code actually reads
# (see _VAULT_KEY_MAP in utils/secrets_manager.py)
vault kv put secrets/binance \
    api_key=your-api-key \
    secret_key=your-api-secret
```

Alternatively, run `python scripts/setup_vault.py --setup` for interactive
setup instructions, or `--migrate` to push secrets from a local `.env` into
Vault via the secrets manager.

### Production Environment
```bash
# Use proper Vault cluster with:
# - TLS certificates
# - Authentication backends (LDAP/AWS/GCP)
# - Comprehensive policies
# - High availability setup
# - Backup and disaster recovery
```

### Security Best Practices

#### 1. **Token Management**
- The Vault token is supplied via the `VAULT_TOKEN` environment variable and is
  never hardcoded. In production, prefer short-lived tokens issued by a Vault
  auth backend and rotate them out-of-band — the bot does not renew tokens
  automatically; it re-authenticates on demand with whatever token is present.

#### 2. **Error Handling**
- Retrieval failures are caught and logged; the manager falls back through
  env vars and the encrypted local file rather than crashing.
- Secret *values* are not printed. The `secrets_manager.py` CLI reports only
  presence/absence (`--get`) and never echoes values (`--set` uses `getpass`).

#### 3. **Cache Security**
- Cached secrets are Fernet-encrypted in memory and expire after the 5-minute
  TTL (`_cache_ttl = 300`).
- Set `VAULT_CACHE_KEY` to a persistent Fernet key so the cache key is stable
  across restarts; otherwise a fresh key is generated each run.

## 🚨 Security Monitoring

### Real-Time Health Checks
- **API Connection Tester** (`utils/api_connection_tester.py`): `test_vault()`
  checks Vault connectivity/authentication alongside Binance, Postgres, Redis,
  Telegram and Prometheus. It requires `VAULT_TOKEN`; when unset it logs a
  warning and reports `DISCONNECTED` (`error_message="VAULT_TOKEN not set"`)
  rather than injecting a default.
- **Health summary**: `get_overall_health()` aggregates the per-API
  `ConnectionStatus` values into `healthy`/`warning`/`critical`.

### Vault Health-Check States (`ConnectionStatus`)
- ✅ **CONNECTED**: authenticated and a simple `sys.list_auth_methods()`
  operation succeeded.
- ❌ **ERROR**: reachable but not authenticated, or an operation failed.
- ⚪ **DISCONNECTED**: `VAULT_TOKEN` not set, or Vault disabled/unreachable.

### Vault Status Shape (`EnhancedSecretsManager.get_vault_status()`)
```python
# When a Vault client is active:
vault_status = {
    "enabled": True,
    "connected": True,   # client.is_authenticated
    "healthy": True,     # vault_client.health_check()
    "url": "http://127.0.0.1:8200",
}
# When no client is initialized:
# {"enabled": False, "connected": False, "healthy": False,
#  "reason": "Vault client not initialized"}
```

## 🛡️ Security Posture

> This is a personal/experimental trading bot, not a certified product. It has
> no SOC 2, FIPS 140-2, or Common Criteria evaluation. The notes below describe
> what the code actually does; any formal compliance would come from the
> underlying Vault/OpenBao deployment you run, not from this repository.

### Data Protection
- **No hardcoded secrets**: API keys, tokens and passwords are read at runtime
  from Vault, environment variables, or the encrypted local file — never
  committed literals. Regression test:
  `tests/test_api_connection_tester_vault_token.py`.
- **Encryption library**: secret caching and the local fallback file use
  Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` package.
- **No PII**: the bot stores trade/position data, not personal user data.

### Risk Mitigation
- **Secret rotation**: rotate credentials in Vault/OpenBao out-of-band; the bot
  picks up new values on the next fetch (subject to the 5-minute cache TTL).
  There is no automated rotation loop in the code.
- **Kill-switch / emergency stop**: `trading/utils/trade_history_api.py`
  exposes `POST /emergency_stop` (and `/emergency-stop`) to halt trading, with
  the state persisted in Redis under `ELVIS_KILL_SWITCH` so it survives
  restarts; `DELETE` clears it and `GET /emergency_stop/status` reports it.
- **Fail-closed API**: the trade-history API rejects all authenticated requests
  with `503` when `API_KEY` is unset.

## 🔍 Security Testing

### Automated Security Checks
- `tests/test_api_connection_tester_vault_token.py` — asserts the old
  hardcoded `trading-bot-token` literal is gone and that a missing
  `VAULT_TOKEN` is handled gracefully (warn + `DISCONNECTED`) instead of being
  defaulted.

Run it with:
```bash
pytest tests/test_api_connection_tester_vault_token.py
```

`tests/test_vault_connection.py` and `tests/test_vault_integration.py` are
manual diagnostics, not pytest tests. They can contact a local Vault and the
former can write test secrets, so run them only against an explicitly prepared
disposable Vault.

## 📚 Additional Resources

### Documentation
- [HashiCorp Vault Documentation](https://developer.hashicorp.com/vault/docs)
- [Vault Security Model](https://developer.hashicorp.com/vault/docs/internals/security)
- [Best Practices Guide](https://developer.hashicorp.com/vault/tutorials/security)

### Emergency Procedures
- **Vault / token compromise**: revoke and reissue the `VAULT_TOKEN` on the
  Vault server, then update the environment the bot runs with. Rotation and
  revocation are manual/server-side; the code has no automated re-keying.
- **Secret exposure**: rotate the affected secret in Vault; the bot picks up
  the new value on the next fetch (after the 5-minute cache TTL).
- **Halt trading now**: activate the kill-switch — `POST /emergency_stop`
  (persisted in Redis, survives restarts). Clear it with `DELETE`.
- **Vault outage**: the secrets manager falls back to environment variables and
  the encrypted local file automatically.

---

**Last Updated**: July 5, 2026

> **Note**: This document describes the secret-handling and API-authentication
> behaviour actually implemented in the code (Vault/OpenBao integration with
> env-var and encrypted-local-file fallback, `X-API-Key` on the trade-history
> API, and the Redis-backed kill-switch). It is a personal project, not a
> certified or independently audited product.
