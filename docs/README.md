# ELVIS documentation

ELVIS documentation is now organised around the in-progress V2 architecture
programme. The compatibility paper runtime is still authoritative; V2 durable
owners and activation capabilities are not composed into deployment, and
`ACTIVE` remains a **NO-GO**.

## Start here

1. [V2 architecture](V2_ARCHITECTURE.md) — concise explanation of the new
   approach, data flow, implementation state, migration phases, and operator
   safety contract.
2. [Architecture migration](architecture_migration/README.md) — evidence,
   detailed target contracts, and the authoritative execution ledger.
3. [V2 PostgreSQL rehearsal](V2_POSTGRES_REHEARSAL.md) — isolated fresh-cluster
   SCRAM/HBA proof, volume guard, rollback boundary, and Mermaid flows.
4. [V2 fresh-target cut-over preflight](V2_FRESH_TARGET_CUTOVER.md) — read-only
   source-clone/target admission, canonical evidence, rollback phases, and the
   explicit no-copy boundary.
5. [Compatibility-runtime architecture](architecture.md) — verified topology
   of the runtime that remains active during migration.

## Documentation authority matrix

| Document class | Purpose | Authority |
|---|---|---|
| [V2 overview](V2_ARCHITECTURE.md) | V2 intent, boundaries, and reader map | Canonical overview |
| [Target architecture](architecture_migration/03-target-architecture.md) | Detailed V2 component and data contracts | Canonical design |
| [Migration roadmap](architecture_migration/04-migration-roadmap.md) | Slice status, verification, rollback, and cut-over gates | Canonical status |
| [PostgreSQL bootstrap](V2_POSTGRES_BOOTSTRAP.md), [fresh rehearsal](V2_POSTGRES_REHEARSAL.md), and [fresh-target preflight](V2_FRESH_TARGET_CUTOVER.md) | Dormant operator contracts, disposable proof, and read-only migration admission | V2 operator runbooks; not deployment authority |
| [Repository analysis](architecture_migration/01-elvis-repository-analysis.md) | Immutable audited baseline | Historical evidence |
| [Runtime architecture](architecture.md), [components](COMPONENTS.md), and [system topology](ELVIS_SYSTEM_ARCHITECTURE.md) | Current compatibility-runtime references | Legacy operational context |
| [Deployment guide](DEPLOYMENT.md) | Existing Compose/Ansible workflow | Legacy deployment only; not V2 cut-over |
| `docs/archive/` | Superseded reports and completion claims | Historical only |

Source code and the roadmap's latest committed record win if a legacy guide
disagrees. A successful test, generated artefact, or healthy compatibility
container is not proof of a V2 deployment or activation.

## Compatibility-runtime documentation

The material below describes current features and operations while V2 is being
built. It does not grant V2 runtime authority.

### 🔐 Security & Configuration
- **[SECURITY.md](../SECURITY.md)** - Secrets handling with HashiCorp Vault / OpenBao and local fallback
- **[VAULT_SETUP.md](VAULT_SETUP.md)** - Step-by-step Vault configuration guide

### 🧰 Operations Runbooks
- **[2026-02-18 Container Observability Runbook](ops/2026-02-18_container_observability_runbook.md)** - Container startup, Grafana/Prometheus "No data" troubleshooting, and post-hardening env requirements
- **[2026-02-10 ELVIS No Data Debug](ops/2026-02-10_elvis_no_data_debug.md)** - Prior investigation notes for dashboard data issues

### 📊 System Architecture
- **[Architecture migration](architecture_migration/README.md)** - Measured repository audit, reference comparison, target architecture, and incremental migration ledger
- **[API Monitoring](../utils/api_connection_tester.py)** - Connectivity/health checks for Binance, Postgres, Redis, Vault, Telegram and the Prometheus Pushgateway
- **[Console Dashboard](../utils/console_dashboard.py)** - Live curses trading dashboard with API status widget
- **[Secrets Management](../utils/secrets_manager.py)** - Vault-backed secrets with environment and encrypted-file fallback

### 🚀 Trading Features
- **Fast trading loop** - no per-iteration sleep, with a configurable inter-trade cooldown (default 15 min) enforced by `TradeCooldownManager` (see `trading/cooldown/trade_cooldown_manager.py`)
- **Runtime monitoring** - Per-service connectivity checks with response-time measurement (`utils/api_connection_tester.py`)
- **Vault-backed secrets** - API keys read from Vault at runtime, never hardcoded

### 🛡️ Security Features

#### HashiCorp Vault / OpenBao Integration
- **Centralized secret storage**: Binance credentials live in a Vault KV v2 mount named `secrets`, one path per service (`secrets/binance`, `secrets/binance_testnet`). The mapping is defined in `utils/secrets_manager.py` (`_VAULT_KEY_MAP`).
- **Layered fallback**: `SecretsManager.get_secret()` tries Vault first, then process environment variables, then the encrypted local file at `~/.elvis/secrets.enc`. (Environment is checked before the encrypted file so an unauthorized Python binary does not block on the macOS Keychain prompt.)
- **Local encryption**: the cache/fallback file uses **Fernet** (AES-128-CBC + HMAC-SHA256) from the `cryptography` package. The Fernet key itself is stored in the OS keyring.
- **No hardcoded secrets**: keys and tokens are read at runtime, never committed as literals.

> This is a personal/experimental trading bot, not a certified product. It has
> **no** SOC 2, FIPS 140-2, or Common Criteria evaluation, and does not implement
> AES-256-GCM. Any formal compliance would come from the Vault/OpenBao deployment
> you run, not from this repository. See [SECURITY.md](../SECURITY.md) for the
> authoritative security posture.

### 🔧 Quick Setup

#### 1. Start Vault / OpenBao (Development)
```bash
# dev-mode server; pick any local token (scripts/start_bot_with_vault.sh requires
# VAULT_DEV_ROOT_TOKEN_ID to be set in the environment)
vault server -dev -dev-root-token-id=<choose-a-local-dev-token>
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN=<choose-a-local-dev-token>
```

#### 2. Initialize Secrets
```bash
# ELVIS reads Binance creds from the `secrets` KV v2 mount, path `binance`,
# fields api_key / secret_key (see utils/secrets_manager.py _VAULT_KEY_MAP).
bao kv put -mount=secrets binance \
    api_key=your-api-key \
    secret_key=your-api-secret
# Testnet: bao kv put -mount=secrets binance_testnet api_key=... secret_key=...
```

#### 3. Start Trading Bot
```bash
# paper is the only executable mode; live is rejected before bootstrap
python main.py --mode paper
```

#### 4. Verify
- Console dashboard shows ✅ for Vault under the API Status widget
- Overall health percentage is reported (see below)

### 📊 Dashboard Features

The curses dashboard (`utils/console_dashboard.py`) renders an API Status widget.
It shows an overall health percentage, then the four critical services
individually, then the remaining services in a compact `Other:` line.

```
--- API Status ---
✅ Overall: 100%
✅ Binance Spot    45ms
✅ Binance Futu    52ms
✅ Postgres        12ms
✅ Vault            3ms
Other: BIN✅ RED✅ TEL❌ PRO✅
Updated: 18:15:42
```

- **Critical services** (shown with response time): `binance_spot`, `binance_futures`, `postgres`, `vault`.
- **`Other:` line** (secondary services, first 3 letters uppercased): `BIN` = `binance_testnet`, `RED` = `redis`, `TEL` = `telegram`, `PRO` = `prometheus` (Pushgateway).
- **Overall %** = connected services ÷ total services × 100 (`APIConnectionTester.get_overall_health()`); status is `healthy` / `warning` / `critical` based on that percentage.

#### Visual Indicators
- ✅ **Connected** - Service reachable and responding
- ⏳ **Testing** - Health check currently running
- ❌ **Error** - Service unreachable or check failed
- ❓ **Unknown** - Status not yet determined

### 🔍 Monitoring

`utils/api_connection_tester.py` runs per-service checks and records, for each:
- Reachability (connected / testing / error)
- Response time (measured with `time.time()` around each check; surfaced in ms)

The overall health summary aggregates these into a percentage and status. There
is no automatic token renewal, alerting, or audit pipeline in this repository —
those would be provided by the Vault/OpenBao deployment and any external
monitoring you attach.

### 🚨 Failure Handling

- **Vault unavailable**: `SecretsManager.get_secret()` falls through to environment
  variables and then the encrypted local file, so the bot can still read
  credentials it has cached locally.
- **Missing secret**: if no source yields a value, `get_secret()` returns the
  supplied default and (optionally) logs a warning.
- **Service check failures**: surfaced as ❌ in the dashboard API Status widget
  and reflected in the overall health percentage.

---

**For Vault setup instructions, see [VAULT_SETUP.md](VAULT_SETUP.md)**  
**For the authoritative security posture, see [SECURITY.md](../SECURITY.md)**  
**For the project overview, see the main [README.md](../README.md)**
