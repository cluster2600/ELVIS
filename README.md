# ELVIS — Enhanced Leveraged Virtual Investment System

![Project Image](./images/elvis.png)

[![Release](https://img.shields.io/github/v/release/cluster2600/ELVIS)](https://github.com/cluster2600/ELVIS/releases)
[![CI](https://github.com/cluster2600/ELVIS/actions/workflows/ci.yml/badge.svg)](https://github.com/cluster2600/ELVIS/actions)
[![Python](https://img.shields.io/badge/python-3.14%20%7C%203.10%20(ML)-blue)](docs/DEPLOYMENT.md)
[![License](https://img.shields.io/badge/license-BTC__BOT-lightgrey)](LICENSE)

ELVIS is a BTC futures trading bot undergoing a V2 architecture migration. V2
keeps a small Python modular monolith while replacing coupled state changes
with typed decisions, durable journals, atomic state owners, generation-bound
activation, and least-authority PostgreSQL roles.

> **Paper trading is the only executable mode.** `--mode live` is retained as a
> compatibility value but is rejected before application bootstrap; the current
> executor has no validated live-submission capability. Nothing here is
> financial advice, and no profitability figures are guaranteed.

## ELVIS V2 programme

This branch contains the incremental build of ELVIS V2. It is a new
architecture approach, not a released or deployed V2 runtime:

- pure, fail-closed signal and risk decisions precede external effects;
- immutable journal facts and deterministic replay replace inferred state;
- one atomic owner commits each related order, fill, position, and account
  transition;
- a database fence and runtime generation prevent simultaneous legacy and V2
  authorities; and
- an offline bootstrap prepares narrowly scoped database identities without
  granting the running bot migration authority.

The compatibility paper runtime remains authoritative. The new persistence,
fence, activation, and bootstrap capabilities are implemented but dormant;
dedicated deployment identities, fail-closed runtime composition, shadow and
reconciliation evidence, rollback rehearsal, soak, and explicit operator
approval are still required. `ACTIVE` remains a **NO-GO**.

Start with the [V2 architecture overview](docs/V2_ARCHITECTURE.md). The
[migration roadmap](docs/architecture_migration/04-migration-roadmap.md) is the
authoritative implementation ledger.

## Features

- **Ensemble strategy** — technical, research, RL, Bonenkamp, and optional MLX
  signals vote; incompatible Research/Bonenkamp artefacts stay disabled
- **Signal-quality gates** — market regime, RSI, momentum persistence,
  BB squeeze, trading hours, MACD divergence, order flow, multi-timeframe
  alignment ([roadmap implementation](docs/profitability_roadmap_implementation.md))
- **Risk management** — dynamic position sizing (volume + optional Kelly),
  trailing stops, regime-aware take-profit, fee-viability gate, cooldowns
- **Secure secrets** — Vault/OpenBao KV v2, no keys on disk
  ([integration guide](docs/VAULT_INTEGRATION.md))
- **Observability** — Prometheus metrics, Grafana dashboards, Loki logs,
  trade-history API with console dashboard
- **Reproducible training** — checkpointed pipeline with resume
  (`--resume latest|best`), walk-forward optimization, RL agents

## Compatibility-runtime quick start

These commands run the current paper runtime. They do not activate the dormant
V2 authority path.

```bash
git clone https://github.com/cluster2600/ELVIS.git && cd ELVIS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Secrets (Vault/OpenBao dev server, KV v2 mount `secrets`)
vault server -dev &
export VAULT_ADDR='http://127.0.0.1:8200' VAULT_TOKEN='<dev-token>'
vault kv put secrets/binance api_key='YOUR_KEY' secret_key='YOUR_SECRET'

# Paper trading
python main.py --mode paper
# Trade API + dashboard: http://localhost:5050/health
```

### Docker

```bash
docker compose up -d                                  # bot + database
docker compose --profile observability up -d          # + Prometheus/Grafana/Loki
docker compose --profile ml run --rm elvis-ml-trainer # py3.10 ML training
```

The current Compose/Ansible path is not the V2 cut-over procedure. Do not infer
V2 database authority or readiness from a healthy legacy container.

The bot image runs Python 3.14; the unified PyTorch trainer and its optional
TensorFlow path run in a separate Python 3.10 container sharing the `models/`
volume — see
[docs/UNIFIED_TRAINING_GUIDE.md](docs/UNIFIED_TRAINING_GUIDE.md).

## Project layout

```
main.py            # trading loop entrypoint
config/            # typed config + trading_config.yaml loader
core/              # bootstrap, DI container, models
trading/           # strategies, execution, risk, signals, fees, api
training/          # training pipeline, RL agents, checkpoints
utils/             # secrets manager, price fetcher, dashboards, DB
scripts/           # operational entry scripts (training, diagnostics)
observability/     # prometheus / grafana / loki / promtail configs
docker/            # Dockerfile variants (ml310, minimal, simple)
ansible/           # server provisioning + deployment
docs/              # all documentation (index below)
tests/             # pytest suite (CI: python 3.10 + 3.14, heavy deps absent)
```

## Documentation

| Topic | Where |
|---|---|
| V2 overview and status | [docs/V2_ARCHITECTURE.md](docs/V2_ARCHITECTURE.md) · [migration ledger](docs/architecture_migration/04-migration-roadmap.md) |
| V2 PostgreSQL operator path | [offline bootstrap](docs/V2_POSTGRES_BOOTSTRAP.md) · [isolated fresh-cluster rehearsal](docs/V2_POSTGRES_REHEARSAL.md) |
| V2 detailed architecture | [target contracts and diagrams](docs/architecture_migration/03-target-architecture.md) |
| Compatibility runtime | [verified topology](docs/architecture.md) · [components](docs/COMPONENTS.md) |
| Architecture evidence | [audit and reference comparison](docs/architecture_migration/README.md) |
| Current deployment — not V2 cut-over | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Security | [SECURITY.md](SECURITY.md) · [docs/SECURITY_GUIDE.md](docs/SECURITY_GUIDE.md) |
| Vault / secrets | [docs/VAULT_INTEGRATION.md](docs/VAULT_INTEGRATION.md) · [docs/VAULT_SETUP.md](docs/VAULT_SETUP.md) |
| Training | [docs/training.md](docs/training.md) · [docs/UNIFIED_TRAINING_GUIDE.md](docs/UNIFIED_TRAINING_GUIDE.md) |
| Profitability roadmap | [docs/profitability_roadmap.md](docs/profitability_roadmap.md) · [status](docs/profitability_roadmap_implementation.md) |
| Paper trading setup | [docs/PAPER_TRADING_SETUP.md](docs/PAPER_TRADING_SETUP.md) |
| Apple containers | [docs/README_APPLE_CONTAINERS.md](docs/README_APPLE_CONTAINERS.md) |
| Release notes | [CHANGELOG.md](CHANGELOG.md) · [historical v0.2 notes](docs/archive/v1/RELEASE_NOTES.md) |

## Testing

```bash
.venv/bin/python -m pytest tests/ -q        # full suite
.venv/bin/python -m pytest -m perf          # latency tripwires
```

CI uses pinned Black/isort/flake8 versions, runs the non-PostgreSQL and
PostgreSQL 15 suites on Python 3.10 and 3.14, scans the repository with Trivy,
and builds the container on branch pushes. Pushes to `main` publish the image
to `ghcr.io/cluster2600/elvis`.

## License

[BTC_BOT License](LICENSE)
