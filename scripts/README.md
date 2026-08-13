# scripts/

Operational shell scripts and admin tooling for ELVIS. They were moved here from
the repo root to keep the top level clean.

> **Run them from the repository root**, e.g. `./scripts/run_tests.sh`. Each
> shell script resolves the repo root itself (via `SCRIPT_DIR/..` or
> `dirname "$0"/..`), so it also works if invoked from inside `scripts/`.

## Run / operate

| Script | Purpose |
|---|---|
| `run_elvis.sh` | Create/activate `venv314` and start the bot (paper mode). |
| `run_api.sh` | Start the trading REST API. |
| `run_console_dashboard.sh` | Launch the curses console dashboard. |
| `run_research_strategy.sh` | Run the bot with the research-based strategy (`STRATEGY_MODE=research`). |
| `run_tests.sh` | Run the test suite via `venv314`'s pytest. |
| `run_training.sh` | Unified training entry point (see `../docs/UNIFIED_TRAINING_GUIDE.md`). |
| `start_bot_with_vault.sh` | Start the bot with Vault/OpenBao auth (requires `VAULT_DEV_ROOT_TOKEN_ID`). |

## Setup

| Script | Purpose |
|---|---|
| `setup_secure_config.sh` | Set up encrypted/secure config loading. |

## Offline administration (Python)

| Module | Purpose |
|---|---|
| `python -m scripts.postgres_bootstrap` | Reconcile the dormant V2 PostgreSQL roles/catalog from a strict non-secret JSON manifest and external libpq services. It is one-shot, operator-confirmed, and never runs at application startup. |
| `python -m scripts.postgres_cutover_preflight` | Inspect one stopped V1 clone and one separately bootstrapped, empty V2 target. It is read-only, emits stale evidence, and never copies data or authorises cut-over. |
| `python -m scripts.postgres_legacy_snapshot_import` | Bind a strict secret-free c3c2 `READY` receipt as stale expected evidence, revalidate both databases, then copy only the seven raw V1 relations with bounded batches, atomic row commit, and post-commit sequence recovery. It never synthesizes V2 history or authorises activation. |
| `python -m scripts.postgres_legacy_snapshot_reconciliation` | Canonically bind a c3c3a import document, sequentially revalidate the target read-only, and compare the complete imported opening candidate with a deterministic but explicitly non-runtime operator hypothesis. It authenticates no source provenance, has no match outcome, and never opens, provisions, or activates an account. |

See the [V2 PostgreSQL bootstrap runbook](../docs/V2_POSTGRES_BOOTSTRAP.md) for
the exact flags, version-1 configuration schema, external `PGSERVICEFILE` and
`PGPASSFILE` contract, receipts, exit codes, and commit-unknown recovery. A
`COMPLETE` receipt does not deploy or activate V2.

See the [fresh-target cut-over preflight](../docs/V2_FRESH_TARGET_CUTOVER.md)
for its three mandatory confirmations, closed version-1 intent document,
`READY_FOR_FRESH_TARGET`/`BLOCKED` receipts, and rollback boundary. Even a ready
receipt is non-authoritative and does not permit import, deployment, or
activation.

See the [bounded legacy snapshot import
runbook](../docs/V2_LEGACY_SNAPSHOT_IMPORT.md) for the six mandatory CLI
options, strict version-1 configuration and receipt binding,
`IMPORTED`/`REPLAYED` receipts, exact commit-unknown resume, sequence boundary,
and rollback. External libpq files remain the only connection/secret input;
`ACTIVE` remains a **NO-GO**.

See the [legacy snapshot reconciliation
runbook](../docs/V2_LEGACY_SNAPSHOT_RECONCILIATION.md) for the mandatory
reviewed-window and disposable-target assertions, canonical document hashes,
point-in-time and cross-snapshot limits, imported and operator-hypothesis
candidates, separate hypothesis fee folds, `DECISION_REQUIRED`/`BLOCKED`
outcomes, exits `10`/`21`, strict version-1 configuration, external
admin/readiness libpq services, and explicit no-source-authentication,
no-opening, no-provisioning, and no-activation boundary.

## Vault / secrets (Python)

| Script | Purpose |
|---|---|
| `setup_vault.py` | Vault/OpenBao setup + `.env` migration. |
| `vault_admin.py` | Manage secrets (`--list`/`--add`/`--update`/`--delete`/`--backup`/`--test`). |

## ML / models (Python — need the older-Python ML stack)

| Script | Purpose |
|---|---|
| `train_enhanced_rf.py` | Train the enhanced Random Forest model. |

The former random-data CoreML generator was removed. New model producers must
train on causal data and emit the validated feature manifest used at inference.

## Apple-container workflow (macOS)

A self-contained group for running ELVIS in Apple's native `container` CLI. Start
with `apple_container_native.sh` (native CLI) or `apple_container_elvis.sh`
(compose-style); they generate helper scripts (`start_elvis_apple.sh`,
`stop_elvis.sh`, `docker-compose.apple.yml`) into the repo root at setup time.

| Script | Purpose |
|---|---|
| `apple_container_native.sh` | Native Apple `container` CLI runner (`setup`/`start`/`stop`/`status`/`logs`). |
| `apple_container_elvis.sh` | Compose-style Apple-container runner. |
| `setup_apple_containers.sh` | One-time setup (generates the compose + start/stop helpers). |
| `fix_apple_container_build.sh` | Quick fix for Apple-container build issues. |
| `test_apple_container.sh` | Smoke-test the native Apple-container integration. |
| `test_container_setup.sh` | Verify container setup prerequisites. |

See `../APPLE_NATIVE_CONTAINER_GUIDE.md` and `../README_APPLE_CONTAINERS.md`.
