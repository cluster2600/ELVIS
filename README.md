# ELVIS — Enhanced Leveraged Virtual Investment System

![Project Image](./images/elvis.png)

[![Release](https://img.shields.io/github/v/release/cluster2600/ELVIS)](https://github.com/cluster2600/ELVIS/releases)
[![CI](https://github.com/cluster2600/ELVIS/actions/workflows/ci.yml/badge.svg)](https://github.com/cluster2600/ELVIS/actions)
[![Python](https://img.shields.io/badge/python-3.14%20%7C%203.10%20(ML)-blue)](docs/DEPLOYMENT.md)
[![License](https://img.shields.io/badge/license-BTC__BOT-lightgrey)](LICENSE)

ELVIS is a BTC futures trading bot that combines an ML ensemble (RandomForest,
neural nets, optional YDF/CoreML members), market-regime detection, and a
layered signal-quality pipeline with strict risk management. Secrets live in
HashiCorp Vault/OpenBao; monitoring runs on Prometheus + Grafana + Loki.

> **Paper trading is the only executable mode.** `--mode live` is retained as a
> compatibility value but is rejected before application bootstrap; the current
> executor has no validated live-submission capability. Nothing here is
> financial advice, and no profitability figures are guaranteed.

## Features

- **Ensemble strategy** — multiple models vote; graceful degradation when
  optional ML deps (ydf, tensorflow, coremltools) are absent
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

## Quick start

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
docker compose --profile ml run --rm elvis-ml-trainer # py3.10 ML-legacy training
```

The bot image runs Python 3.14; TensorFlow/YDF/CoreML training runs in a
separate Python 3.10 container sharing the `models/` volume — see
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

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
tests/             # pytest suite (CI: python 3.14, heavy deps absent)
```

## Documentation

| Topic | Where |
|---|---|
| Architecture & diagrams | [docs/architecture.md](docs/architecture.md) · [docs/ELVIS_SYSTEM_ARCHITECTURE.md](docs/ELVIS_SYSTEM_ARCHITECTURE.md) |
| Architecture migration | [audit, reference comparison, target and roadmap](docs/architecture_migration/README.md) |
| Components reference | [docs/COMPONENTS.md](docs/COMPONENTS.md) |
| Deployment (Ansible + Docker) | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Security | [SECURITY.md](SECURITY.md) · [docs/SECURITY_GUIDE.md](docs/SECURITY_GUIDE.md) |
| Vault / secrets | [docs/VAULT_INTEGRATION.md](docs/VAULT_INTEGRATION.md) · [docs/VAULT_SETUP.md](docs/VAULT_SETUP.md) |
| Training | [docs/training.md](docs/training.md) · [docs/UNIFIED_TRAINING_GUIDE.md](docs/UNIFIED_TRAINING_GUIDE.md) |
| Profitability roadmap | [docs/profitability_roadmap.md](docs/profitability_roadmap.md) · [status](docs/profitability_roadmap_implementation.md) |
| Paper trading setup | [docs/PAPER_TRADING_SETUP.md](docs/PAPER_TRADING_SETUP.md) |
| Apple containers | [docs/README_APPLE_CONTAINERS.md](docs/README_APPLE_CONTAINERS.md) |
| Release notes | [CHANGELOG.md](CHANGELOG.md) · [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) |

## Testing

```bash
venv/bin/python -m pytest tests/ -q        # full suite
venv/bin/python -m pytest -m perf          # latency tripwires
```

CI runs lint (black/isort/flake8), the test suite on Python 3.14 without the
heavy ML deps, security scans (bandit, Trivy), and a Docker build on `v*` tags
publishing to `ghcr.io/cluster2600/elvis`.

## License

[BTC_BOT License](LICENSE)
