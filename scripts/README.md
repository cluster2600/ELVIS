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
| `run_training.sh` | Unified training entry point (see `../UNIFIED_TRAINING_GUIDE.md`). |
| `start_bot_with_vault.sh` | Start the bot with Vault/OpenBao auth (requires `VAULT_DEV_ROOT_TOKEN_ID`). |

## Setup

| Script | Purpose |
|---|---|
| `setup_secure_config.sh` | Set up encrypted/secure config loading. |

## Vault / secrets (Python)

| Script | Purpose |
|---|---|
| `setup_vault.py` | Vault/OpenBao setup + `.env` migration. |
| `vault_admin.py` | Manage secrets (`--list`/`--add`/`--update`/`--delete`/`--backup`/`--test`). |

## ML / models (Python — need the older-Python ML stack)

| Script | Purpose |
|---|---|
| `train_enhanced_rf.py` | Train the enhanced Random Forest model. |
| `create_coreml_model.py` | Build a CoreML model (needs `coremltools`/`tensorflow`; no Python 3.14 wheels). |

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
